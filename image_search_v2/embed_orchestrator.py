#!/usr/bin/env python3
"""
Distributed Orchestrator

Manages the entire pipeline:
1. QPS management - checks API limits every 5 mins, adjusts sync_data delay
2. 6-hour backups - stops sync_data/image_downloader, backs up DB/FAISS/images
3. Distributed work - sends batches to workers (iMac/MBP/Sleight)

Workers do: bg_removal + all 5 embeddings
Orchestrator does: coordinate, verify, import results, update DB

Stop with: touch KILL_ORCH
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from image_db import get_connection, commit_with_retry

# Paths
BASE_DIR = Path(__file__).parent.parent
EXPORTS_DIR = BASE_DIR / 'embed_exports'
EMBEDDINGS_DIR = BASE_DIR / 'embeddings'
IMAGES_DIR = BASE_DIR / 'images'
BACKUPS_DIR = BASE_DIR / 'backups'
DB_FILE = BASE_DIR / 'etsy_data.db'
IMAGE_INDEX_FILE = EMBEDDINGS_DIR / 'image_index.json'

# Config files
QPS_CONFIG_FILE = BASE_DIR / 'qps_config.json'
ORCH_STATE_FILE = BASE_DIR / 'orchestrator_state.json'
PID_FILE = BASE_DIR / 'orchestrator.pid'
KILL_FILE = BASE_DIR / 'KILL_ORCH'

# API config
ETSY_API_KEY = os.getenv("ETSY_API_KEY")
API_BASE_URL = "https://openapi.etsy.com/v3"

# Timing
QPS_CHECK_INTERVAL = 300  # 5 minutes
BACKUP_INTERVAL = 6 * 3600  # 6 hours
BATCH_SIZE = 1000  # Images per worker batch
COMPLETED_THRESHOLD = 50000  # Import results after this many complete

# QPS limits
MIN_DELAY = 0.2    # 5 QPS max
MAX_DELAY = 2.0    # 0.5 QPS min
RATE_THRESHOLD = 10000  # Remaining quota threshold

# All models
ALL_MODELS = ['clip_vitb32', 'clip_vitl14', 'dinov2_base', 'dinov2_large', 'dinov3_base']


@dataclass
class Worker:
    name: str
    host: str
    user: str
    work_dir: str
    python_path: str
    speed_factor: float


WORKERS = {
    'imac': Worker(
        name='imac',
        host='localhost',
        user='embed',
        work_dir='/Users/embed/imageselector',
        python_path='/Users/embed/imageselector/venv/bin/python',
        speed_factor=1.0,
    ),
    'mbp': Worker(
        name='mbp',
        host='mbp.local',
        user='embed',
        work_dir='/Users/embed/imageselector',
        python_path='/Users/embed/imageselector/venv/bin/python',
        speed_factor=2.1,
    ),
    'sleight': Worker(
        name='sleight',
        host='192.168.68.117',
        user='embed',
        work_dir='E:/embed_work/imageselector',
        python_path='E:/embed_work/imageselector/venv/Scripts/python.exe',
        speed_factor=2.4,
    ),
}


# ============================================================
# UTILITIES
# ============================================================

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{ts()}] {msg}", flush=True)


def acquire_lock() -> bool:
    """Acquire PID lock. Returns False if another instance is running."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    PID_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def check_kill_file() -> bool:
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        log("Kill file detected. Shutting down...")
        return True
    return False


def load_state() -> dict:
    if ORCH_STATE_FILE.exists():
        try:
            return json.loads(ORCH_STATE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "last_qps_check": 0,
        "last_backup": 0,
        "pending_images": [],
        "completed_images": [],
    }


def save_state(state: dict):
    ORCH_STATE_FILE.write_text(json.dumps(state, indent=2))


# ============================================================
# QPS MANAGEMENT
# ============================================================

def check_rate_limit() -> dict | None:
    """Check API rate limits, returns limits dict or None on error."""
    try:
        import httpx
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{API_BASE_URL}/application/openapi-ping",
                headers={"x-api-key": ETSY_API_KEY}
            )
            return {
                "remaining": int(response.headers.get("x-remaining-today", 0)),
                "limit": int(response.headers.get("x-limit-per-day", 0)),
            }
    except Exception as e:
        log(f"Error checking rate limit: {e}")
        return None


def update_qps_config(limits: dict):
    """Update QPS config based on rate limits.

    If remaining > RATE_THRESHOLD: speed up (more headroom)
    If remaining <= RATE_THRESHOLD: slow down (approaching limit)
    """
    remaining = limits["remaining"]
    limit = limits["limit"]

    # Load current config
    config = {}
    if QPS_CONFIG_FILE.exists():
        try:
            config = json.loads(QPS_CONFIG_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass

    current_delay = config.get("api_delay", MIN_DELAY)

    # Adjust based on remaining quota (same logic as pipeline_monitor)
    if remaining > RATE_THRESHOLD:
        new_delay = max(current_delay / 1.1, MIN_DELAY)  # Speed up
        action = "INCREASE"
    else:
        new_delay = min(current_delay / 0.9, MAX_DELAY)  # Slow down
        action = "DECREASE"

    log(f"API: {remaining:,}/{limit:,} remaining | {action} QPS: {1/current_delay:.2f} -> {1/new_delay:.2f}")

    config["api_delay"] = new_delay
    config["last_updated"] = datetime.now().isoformat()
    config["remaining"] = remaining
    config["limit"] = limit

    QPS_CONFIG_FILE.write_text(json.dumps(config, indent=2))


# ============================================================
# PROCESS MANAGEMENT
# ============================================================

def soft_kill_script(kill_file: Path, name: str, timeout: int = 30):
    """Create kill file and wait for script to stop."""
    log(f"Stopping {name}...")
    kill_file.touch()

    # Wait for process to exit (check PID file)
    pid_file = BASE_DIR / f"{name}.pid"
    start = time.time()
    while time.time() - start < timeout:
        if not pid_file.exists():
            log(f"{name} stopped")
            return True
        time.sleep(1)

    log(f"{name} did not stop in {timeout}s")
    return False


def start_script(script_path: Path, log_file: Path, name: str):
    """Start a script with unbuffered output."""
    log(f"Starting {name}...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        env=env,
        start_new_session=True
    )
    log(f"{name} started")


def stop_data_scripts():
    """Stop sync_data and image_downloader for backup."""
    soft_kill_script(BASE_DIR / "KILL", "sync_data")
    soft_kill_script(BASE_DIR / "KILL_DL", "image_downloader")


def start_data_scripts():
    """Restart sync_data and image_downloader after backup."""
    start_script(BASE_DIR / "sync_data.py", BASE_DIR / "sync_data.log", "sync_data")
    start_script(BASE_DIR / "scripts/image_downloader.py", BASE_DIR / "image_downloader.log", "image_downloader")


# ============================================================
# BACKUP
# ============================================================

def do_backup():
    """Backup DB, FAISS, and image_index.json to backups/ folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_subdir = BACKUPS_DIR / f"backup_{timestamp}"
    backup_subdir.mkdir(parents=True, exist_ok=True)

    log(f"Backing up to {backup_subdir}")

    # Backup DB
    if DB_FILE.exists():
        shutil.copy2(DB_FILE, backup_subdir / "etsy_data.db")
        log("  Backed up etsy_data.db")

    # Backup FAISS files and image_index.json
    if EMBEDDINGS_DIR.exists():
        emb_backup = backup_subdir / "embeddings"
        emb_backup.mkdir()
        for f in EMBEDDINGS_DIR.glob("*"):
            if f.is_file():
                shutil.copy2(f, emb_backup / f.name)
        log("  Backed up embeddings/")

    log("Backup complete")


# ============================================================
# WORKER COORDINATION
# ============================================================

def get_pending_images(limit: int = 100000) -> List[Tuple[int, int]]:
    """Get images needing processing (download_done=2, bg_removed=0)."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT listing_id, image_id
        FROM image_status
        WHERE download_done = 2 AND bg_removed = 0
        LIMIT ?
    """, (limit,))
    results = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    return results


def get_listing_texts(listing_ids: List[int]) -> dict:
    """Get title + materials for CLIP text embeddings."""
    if not listing_ids:
        return {}

    conn = get_connection()
    placeholders = ",".join(["?"] * len(listing_ids))
    cursor = conn.execute(f"""
        SELECT listing_id, title, materials
        FROM listings_static
        WHERE listing_id IN ({placeholders})
    """, listing_ids)

    texts = {}
    for row in cursor.fetchall():
        lid, title, materials = row
        parts = [title] if title else []
        if materials:
            try:
                mats = json.loads(materials)
                if mats:
                    parts.append(", ".join(mats))
            except (json.JSONDecodeError, TypeError):
                pass
        texts[str(lid)] = ". ".join(parts)

    conn.close()
    return texts


def export_batch(batch_name: str, images: List[Tuple[int, int]]) -> Path:
    """Export batch of images for worker processing."""
    batch_dir = EXPORTS_DIR / batch_name
    if batch_dir.exists():
        shutil.rmtree(batch_dir)

    batch_dir.mkdir(parents=True)
    images_out = batch_dir / "images"
    images_out.mkdir()

    # Copy images
    manifest = []
    for lid, iid in images:
        src = IMAGES_DIR / f"{lid}_{iid}.jpg"
        if src.exists():
            shutil.copy2(src, images_out / f"{lid}_{iid}.jpg")
            manifest.append([lid, iid])

    # Write manifest
    (batch_dir / "manifest.json").write_text(json.dumps(manifest))

    # Write listings for CLIP text
    listing_ids = list(set(lid for lid, _ in manifest))
    texts = get_listing_texts(listing_ids)
    (batch_dir / "listings.json").write_text(json.dumps(texts))

    log(f"Exported {len(manifest)} images to {batch_name}")
    return batch_dir


def get_worker_batch_path(worker: Worker, batch_name: str) -> str:
    """Get remote path for batch on worker."""
    if worker.host == 'localhost':
        return f"/Users/embed/imageselector/embed_exports/{batch_name}"
    elif worker.host == '192.168.68.117':
        return f"/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}"
    else:
        return f"{worker.work_dir}/embed_exports/{batch_name}"


def rsync_to_worker(worker: Worker, batch_dir: Path) -> bool:
    """Rsync batch to worker."""
    batch_name = batch_dir.name

    if worker.host == 'localhost':
        target = f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}"
    elif worker.host == '192.168.68.117':
        target = f"{worker.user}@{worker.host}:/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}"
    else:
        target = f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}"

    cmd = ['rsync', '-av', '--delete', f"{batch_dir}/", target]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def start_worker(worker: Worker, batch_name: str) -> bool:
    """SSH to start worker process."""
    batch_path = get_worker_batch_path(worker, batch_name)
    worker_script = f"{worker.work_dir}/image_search_v2/embed_worker.py"

    if worker.host == '192.168.68.117':
        remote_cmd = f'"{worker.python_path}" "{worker_script}" --input "{batch_path}"'
    else:
        remote_cmd = f'{worker.python_path} {worker_script} --input {batch_path}'

    ssh_host = 'localhost' if worker.host == 'localhost' else worker.host
    ssh_cmd = ['ssh', f'{worker.user}@{ssh_host}', remote_cmd]

    log(f"Starting worker on {worker.name}")
    subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def rsync_poll_npy(worker: Worker, batch_name: str) -> Dict[str, bool]:
    """Rsync only .npy files back, check which models are complete."""
    batch_dir = EXPORTS_DIR / batch_name

    if worker.host == 'localhost':
        src = f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}/"
    elif worker.host == '192.168.68.117':
        src = f"{worker.user}@{worker.host}:/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}/"
    else:
        src = f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}/"

    cmd = ['rsync', '-av', '--include=*.npy', '--exclude=*', src, f"{batch_dir}/"]
    subprocess.run(cmd, capture_output=True, text=True)

    # Check which models completed
    return {model: (batch_dir / f"{model}.npy").exists() for model in ALL_MODELS}


def rsync_images_back(worker: Worker, batch_name: str) -> bool:
    """Rsync bg-removed images back from worker."""
    batch_dir = EXPORTS_DIR / batch_name

    if worker.host == 'localhost':
        src = f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}/images/"
    elif worker.host == '192.168.68.117':
        src = f"{worker.user}@{worker.host}:/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}/images/"
    else:
        src = f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}/images/"

    images_dir = batch_dir / "images"
    cmd = ['rsync', '-av', src, f"{images_dir}/"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def reset_worker_batch(worker: Worker, batch_name: str):
    """Reset batch folder on worker after completion."""
    if worker.host == 'localhost':
        target = f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}"
    elif worker.host == '192.168.68.117':
        target = f"{worker.user}@{worker.host}:/cygdrive/e/embed_work/imageselector/embed_exports/{batch_name}"
    else:
        target = f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}"

    # Delete remote directory
    ssh_host = 'localhost' if worker.host == 'localhost' else worker.host
    if worker.host == '192.168.68.117':
        rm_cmd = ['ssh', f'{worker.user}@{ssh_host}', f'cmd /c rmdir /s /q E:\\embed_work\\imageselector\\embed_exports\\{batch_name}']
    else:
        rm_cmd = ['ssh', f'{worker.user}@{ssh_host}', f'rm -rf {get_worker_batch_path(worker, batch_name)}']

    subprocess.run(rm_cmd, capture_output=True)


# ============================================================
# IMPORT RESULTS
# ============================================================

def verify_batch(batch_dir: Path) -> Tuple[bool, str]:
    """Verify batch has all required files."""
    manifest_file = batch_dir / "manifest.json"
    if not manifest_file.exists():
        return False, "Missing manifest.json"

    manifest = json.loads(manifest_file.read_text())

    # Check all .npy files exist
    for model in ALL_MODELS:
        npy_file = batch_dir / f"{model}.npy"
        if not npy_file.exists():
            return False, f"Missing {model}.npy"

    # Check all images exist
    images_dir = batch_dir / "images"
    for lid, iid in manifest:
        img_file = images_dir / f"{lid}_{iid}.jpg"
        if not img_file.exists():
            return False, f"Missing image {lid}_{iid}.jpg"

    return True, "OK"


def import_batch(batch_dir: Path) -> int:
    """Import verified batch: copy images, append FAISS, update DB."""
    import numpy as np
    import faiss

    manifest = json.loads((batch_dir / "manifest.json").read_text())
    images_src = batch_dir / "images"

    # 1. Copy bg-removed images to images/ folder
    log(f"Copying {len(manifest)} images to images/")
    for lid, iid in manifest:
        src = images_src / f"{lid}_{iid}.jpg"
        dst = IMAGES_DIR / f"{lid}_{iid}.jpg"
        shutil.copy2(src, dst)

    # 2. Load/update image_index.json
    if IMAGE_INDEX_FILE.exists():
        image_index = json.loads(IMAGE_INDEX_FILE.read_text())
    else:
        image_index = []

    existing_set = set(tuple(x) for x in image_index)
    new_indices = [i for i, img in enumerate(manifest) if tuple(img) not in existing_set]

    if not new_indices:
        log("All images already in index")
        return 0

    # 3. Append to FAISS indexes
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    for model in ALL_MODELS:
        npy_file = batch_dir / f"{model}.npy"
        embeddings = np.load(npy_file)
        new_emb = embeddings[new_indices].astype("float32")

        faiss_file = EMBEDDINGS_DIR / f"{model}.faiss"
        if faiss_file.exists():
            index = faiss.read_index(str(faiss_file))
        else:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)

        index.add(new_emb)
        faiss.write_index(index, str(faiss_file))

        # Handle text embeddings for CLIP
        text_npy = batch_dir / f"{model}_text.npy"
        if text_npy.exists():
            text_emb = np.load(text_npy)[new_indices].astype("float32")
            text_faiss = EMBEDDINGS_DIR / f"{model}_text.faiss"
            if text_faiss.exists():
                text_index = faiss.read_index(str(text_faiss))
            else:
                text_index = faiss.IndexFlatIP(text_emb.shape[1])
            text_index.add(text_emb)
            faiss.write_index(text_index, str(text_faiss))

    # 4. Update image_index.json
    for i in new_indices:
        image_index.append(manifest[i])
    IMAGE_INDEX_FILE.write_text(json.dumps(image_index))

    # 5. Batch update DB
    log(f"Updating DB for {len(new_indices)} images")
    conn = get_connection()

    for i in new_indices:
        lid, iid = manifest[i]
        conn.execute("""
            UPDATE image_status
            SET bg_removed = 1,
                embed_clip_vitb32 = 1,
                embed_clip_vitl14 = 1,
                embed_dinov2_base = 1,
                embed_dinov2_large = 1,
                embed_dinov3_base = 1
            WHERE listing_id = ? AND image_id = ?
        """, (lid, iid))

    commit_with_retry(conn)
    conn.close()

    log(f"Imported {len(new_indices)} images")
    return len(new_indices)


# ============================================================
# MAIN LOOP
# ============================================================

def distribute_and_process():
    """Main work distribution loop. Returns number of images processed."""
    # Get pending images
    pending = get_pending_images(limit=BATCH_SIZE * len(WORKERS))
    if not pending:
        return 0

    log(f"Found {len(pending)} images to process")

    # Distribute to workers proportionally by speed
    total_speed = sum(w.speed_factor for w in WORKERS.values())
    worker_batches = {}
    start_idx = 0
    batch_id = f"batch_{int(time.time())}"

    for name, worker in WORKERS.items():
        proportion = worker.speed_factor / total_speed
        count = int(len(pending) * proportion)
        if name == list(WORKERS.keys())[-1]:  # Last worker gets remainder
            worker_images = pending[start_idx:]
        else:
            worker_images = pending[start_idx:start_idx + count]
            start_idx += count

        if not worker_images:
            continue

        batch_name = f"{batch_id}_{name}"
        batch_dir = export_batch(batch_name, worker_images)

        if rsync_to_worker(worker, batch_dir):
            start_worker(worker, batch_name)
            worker_batches[name] = batch_name
        else:
            log(f"Failed to sync to {name}")

    if not worker_batches:
        return 0

    # Poll for completion
    log("Waiting for workers to complete...")
    poll_interval = 30
    timeout = 3600  # 1 hour max
    start_time = time.time()

    while time.time() - start_time < timeout:
        if check_kill_file():
            return 0

        all_done = True
        for name, batch_name in worker_batches.items():
            worker = WORKERS[name]
            status = rsync_poll_npy(worker, batch_name)

            if not all(status.values()):
                all_done = False
            else:
                # All .npy files present - rsync images back
                batch_dir = EXPORTS_DIR / batch_name
                done_marker = batch_dir / ".images_synced"
                if not done_marker.exists():
                    log(f"{name}: all models complete, syncing images back")
                    rsync_images_back(worker, batch_name)
                    done_marker.touch()

        if all_done:
            log("All workers complete!")
            break

        time.sleep(poll_interval)
    else:
        log("Timeout waiting for workers")
        return 0

    # Verify and import each batch
    total_imported = 0
    for name, batch_name in worker_batches.items():
        batch_dir = EXPORTS_DIR / batch_name
        ok, msg = verify_batch(batch_dir)
        if ok:
            imported = import_batch(batch_dir)
            total_imported += imported
            # Clean up local and remote
            reset_worker_batch(WORKERS[name], batch_name)
            shutil.rmtree(batch_dir)
        else:
            log(f"Verification failed for {name}: {msg}")

    return total_imported


def main():
    if not acquire_lock():
        print("Error: Another orchestrator instance is already running")
        return

    EXPORTS_DIR.mkdir(exist_ok=True)
    BACKUPS_DIR.mkdir(exist_ok=True)

    log("=" * 60)
    log("ORCHESTRATOR STARTED")
    log("=" * 60)
    log(f"Workers: {list(WORKERS.keys())}")
    log("Stop with: touch KILL_ORCH")

    state = load_state()

    try:
        while not check_kill_file():
            now = time.time()

            # QPS check every 5 minutes
            if now - state["last_qps_check"] > QPS_CHECK_INTERVAL:
                limits = check_rate_limit()
                if limits:
                    update_qps_config(limits)
                state["last_qps_check"] = now
                save_state(state)

            # Backup every 6 hours
            if now - state["last_backup"] > BACKUP_INTERVAL:
                log("Starting 6-hour backup...")
                stop_data_scripts()
                time.sleep(5)  # Let them finish
                do_backup()
                start_data_scripts()
                state["last_backup"] = now
                save_state(state)

            # Distributed work
            processed = distribute_and_process()
            if processed == 0:
                # No work available, wait before checking again
                log("No pending images, waiting...")
                time.sleep(300)  # 5 minutes

    except KeyboardInterrupt:
        log("Interrupted")

    finally:
        release_lock()
        log("Orchestrator stopped")


if __name__ == "__main__":
    main()
