#!/usr/bin/env python3
"""
Distributed Embedding Orchestrator

Single-threaded polling loop that:
1. Keeps 3 workers busy (iMac, MBP, Sleight)
2. Collects results as they complete
3. Imports to FAISS + updates DB faiss_row
4. Manages QPS for sync_data

Stop with: touch KILL_ORCH
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional
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
EMBEDDED_DIR = Path('/Volumes/SSD_120/embeddedimages')

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
POLL_INTERVAL = 10  # seconds between worker checks
DB_CHUNK_SIZE = 100  # commit every N images

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
    batch_size: int  # Images per batch for this worker
    rsync_path: str = 'rsync'  # For Windows: 'C:/cwrsync/bin/rsync.exe'


MAX_CHECK_RETRIES = 3  # Freeze worker after this many consecutive timeouts


@dataclass
class WorkerState:
    worker: Worker
    current_batch: Optional[str] = None
    batch_images: List[Tuple[int, int]] = field(default_factory=list)
    started_at: float = 0
    status: str = 'idle'  # idle, working, done, frozen
    check_failures: int = 0


WORKERS = {
    'imac': Worker(
        name='imac',
        host='localhost',
        user='embed',
        work_dir='/Users/embed/imageselector',
        python_path='/Users/embed/imageselector/venv/bin/python',
        batch_size=3000,
    ),
    'mbp': Worker(
        name='mbp',
        host='mbp.local',
        user='embed',
        work_dir='/Users/embed/imageselector',
        python_path='/Users/embed/imageselector/venv/bin/python',
        batch_size=3000,
    ),
    'sleight': Worker(
        name='sleight',
        host='192.168.68.117',
        user='embed',
        work_dir='C:/Users/embed/imageselector',
        python_path='C:/Users/embed/imageselector/venv/Scripts/python.exe',
        batch_size=10000,
        rsync_path='C:/cwrsync/bin/rsync.exe',
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
    """Check for kill file. Does NOT auto-remove - user must manually delete."""
    if KILL_FILE.exists():
        log("Kill file detected. Shutting down...")
        return True
    return False


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

    Keep usage under 90K: if used > 90K reduce QPS 10%, else increase 10% (max 5 QPS).
    """
    remaining = limits["remaining"]
    limit = limits["limit"]
    used = limit - remaining

    config = {}
    if QPS_CONFIG_FILE.exists():
        try:
            config = json.loads(QPS_CONFIG_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass

    current_delay = config.get("api_delay", MIN_DELAY)

    if used >= 90000:
        new_delay = current_delay * 1.1  # reduce QPS by 10%
        action = "DECREASE"
    else:
        new_delay = max(current_delay / 1.1, MIN_DELAY)  # increase QPS by 10%, max 5 QPS
        action = "INCREASE"

    log(f"API: {used:,}/{limit:,} | {action} QPS: {1/current_delay:.2f} -> {1/new_delay:.2f}")

    config["api_delay"] = new_delay
    config["last_updated"] = datetime.now().isoformat()
    config["remaining"] = remaining
    config["limit"] = limit

    QPS_CONFIG_FILE.write_text(json.dumps(config, indent=2))


# ============================================================
# BACKUP & PROCESS MANAGEMENT
# ============================================================

def soft_kill_script(kill_file: Path, name: str, timeout: int = 60):
    """Create kill file and wait for script to stop."""
    log(f"Stopping {name}...")
    kill_file.touch()

    pid_file = BASE_DIR / f"{name}.pid"
    start = time.time()
    while time.time() - start < timeout:
        if not pid_file.exists():
            log(f"  {name} stopped")
            return True
        time.sleep(2)

    log(f"  {name} did not stop in {timeout}s")
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
    log(f"  {name} started")


def stop_data_scripts():
    """Stop sync_data and image_downloader for backup."""
    soft_kill_script(BASE_DIR / "KILL", "sync_data")
    soft_kill_script(BASE_DIR / "KILL_DL", "image_downloader")


def start_data_scripts():
    """Restart sync_data and image_downloader after backup."""
    start_script(BASE_DIR / "sync_data.py", BASE_DIR / "sync_data.log", "sync_data")
    start_script(BASE_DIR / "scripts/image_downloader.py", BASE_DIR / "image_downloader.log", "image_downloader")


def check_db_faiss_consistency() -> tuple[bool, str]:
    """Check if DB faiss_row count matches FAISS vector counts.

    Returns (is_consistent, message).
    """
    import sqlite3
    import faiss

    # Get DB count
    conn = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True)
    cursor = conn.execute('SELECT COUNT(*) FROM image_status WHERE faiss_row IS NOT NULL')
    db_count = cursor.fetchone()[0]
    conn.close()

    # Get FAISS counts (all should match)
    faiss_counts = {}
    if EMBEDDINGS_DIR.exists():
        for f in EMBEDDINGS_DIR.glob("*.faiss"):
            if '_text' not in f.name:  # Skip text embeddings
                index = faiss.read_index(str(f))
                faiss_counts[f.name] = index.ntotal

    if not faiss_counts:
        # No FAISS files yet - consistent if DB also has 0
        if db_count == 0:
            return True, "Both empty"
        return False, f"DB has {db_count} but no FAISS files"

    # All FAISS files should have same count
    counts = list(faiss_counts.values())
    if len(set(counts)) > 1:
        return False, f"FAISS files have different counts: {faiss_counts}"

    faiss_count = counts[0]
    if db_count != faiss_count:
        return False, f"DB has {db_count}, FAISS has {faiss_count}"

    return True, f"Consistent: {db_count} entries"


def do_backup():
    """Backup DB and FAISS folder after checking consistency.

    Raises RuntimeError if inconsistent - orchestrator must stop.
    """
    import sqlite3

    # Check consistency first - FATAL if inconsistent
    is_consistent, msg = check_db_faiss_consistency()
    if not is_consistent:
        raise RuntimeError(f"FATAL: DB/FAISS inconsistent - {msg}. Fix before continuing.")

    log(f"Consistency check passed: {msg}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

    # Backup DB using SQLite backup API (safe for concurrent access)
    backup_db = BACKUPS_DIR / f"etsy_data_{timestamp}.db"
    log(f"Backing up database to {backup_db.name}...")
    src = sqlite3.connect(f'file:{DB_FILE}?mode=ro', uri=True)
    dst = sqlite3.connect(str(backup_db))
    src.backup(dst)
    src.close()
    dst.close()
    os.chmod(backup_db, 0o444)  # Read-only
    log(f"  Database backed up")

    # Backup entire embeddings folder
    if EMBEDDINGS_DIR.exists() and any(EMBEDDINGS_DIR.glob("*.faiss")):
        backup_emb_dir = BACKUPS_DIR / f"embeddings_{timestamp}"
        shutil.copytree(EMBEDDINGS_DIR, backup_emb_dir)
        # Make all files read-only
        for f in backup_emb_dir.glob("*"):
            os.chmod(f, 0o444)
        os.chmod(backup_emb_dir, 0o555)
        log(f"  FAISS folder backed up to {backup_emb_dir.name}")

    log("Backup complete")


# ============================================================
# DATABASE
# ============================================================

def get_next_faiss_row(conn) -> int:
    """Get the next available faiss_row index."""
    cursor = conn.execute("SELECT MAX(faiss_row) FROM image_status")
    result = cursor.fetchone()[0]
    return 0 if result is None else result + 1


def get_pending_images(limit: int) -> List[Tuple[int, int]]:
    """Get images needing embedding (download_done=2, faiss_row IS NULL)."""
    conn = get_connection()
    cursor = conn.execute("""
        SELECT listing_id, image_id
        FROM image_status
        WHERE download_done = 2 AND faiss_row IS NULL
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


# ============================================================
# BATCH EXPORT/IMPORT
# ============================================================

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


def import_batch(batch_dir: Path) -> int:
    """Import batch: copy bg-removed images, append FAISS, update DB faiss_row.

    IMPORTANT: This function is atomic - once started, it must complete.
    Do NOT add kill file checks inside this function.
    If interrupted, consistency check will catch the mismatch on next startup.
    """
    import numpy as np
    import faiss

    manifest_file = batch_dir / "manifest.json"
    if not manifest_file.exists():
        log(f"No manifest in {batch_dir}")
        return 0

    manifest = json.loads(manifest_file.read_text())
    if not manifest:
        return 0

    images_src = batch_dir / "images"

    # Verify all files exist
    for model in ALL_MODELS:
        if not (batch_dir / f"{model}.npy").exists():
            log(f"Missing {model}.npy in {batch_dir}")
            return 0

    # 1. Get starting faiss_row
    conn = get_connection()
    start_row = get_next_faiss_row(conn)
    conn.close()

    # 2. Move all bg-removed images to embeddedimages, remove from images/
    EMBEDDED_DIR.mkdir(parents=True, exist_ok=True)
    moved_count = 0

    for lid, iid in manifest:
        src = images_src / f"{lid}_{iid}.jpg"
        if src.exists():
            # Copy to embeddedimages
            dst = EMBEDDED_DIR / f"{lid}_{iid}.jpg"
            shutil.copy2(src, dst)
            moved_count += 1

            # Remove from images/
            orig = IMAGES_DIR / f"{lid}_{iid}.jpg"
            if orig.exists():
                orig.unlink()

    log(f"Moved {moved_count} images to embeddedimages, removed from images/")

    # 3. Append to FAISS indexes
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    for model in ALL_MODELS:
        npy_file = batch_dir / f"{model}.npy"
        embeddings = np.load(npy_file).astype("float32")

        faiss_file = EMBEDDINGS_DIR / f"{model}.faiss"
        if faiss_file.exists():
            index = faiss.read_index(str(faiss_file))
        else:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)

        index.add(embeddings)
        faiss.write_index(index, str(faiss_file))

        # Handle CLIP text embeddings
        text_npy = batch_dir / f"{model}_text.npy"
        if text_npy.exists():
            text_emb = np.load(text_npy).astype("float32")
            text_faiss = EMBEDDINGS_DIR / f"{model}_text.faiss"
            if text_faiss.exists():
                text_index = faiss.read_index(str(text_faiss))
            else:
                text_index = faiss.IndexFlatIP(text_emb.shape[1])
            text_index.add(text_emb)
            faiss.write_index(text_index, str(text_faiss))

    # 4. Update DB with faiss_row (chunked commits)
    log(f"Updating DB faiss_row for {len(manifest)} images...")
    conn = get_connection()

    for i, (lid, iid) in enumerate(manifest):
        faiss_row = start_row + i
        conn.execute("""
            UPDATE image_status SET faiss_row = ?
            WHERE listing_id = ? AND image_id = ?
        """, (faiss_row, lid, iid))

        # Commit every chunk
        if (i + 1) % DB_CHUNK_SIZE == 0:
            commit_with_retry(conn)

    commit_with_retry(conn)
    conn.close()

    log(f"Imported {len(manifest)} images (faiss_row {start_row}-{start_row + len(manifest) - 1})")
    return len(manifest)


# ============================================================
# WORKER COMMUNICATION
# ============================================================

def get_worker_batch_path(worker: Worker, batch_name: str) -> str:
    """Get remote path for batch on worker."""
    if worker.host == 'localhost':
        return f"/Users/embed/imageselector/embed_exports/{batch_name}"
    elif worker.host == '192.168.68.117':
        return f"C:/Users/embed/imageselector/embed_exports/{batch_name}"
    else:
        return f"{worker.work_dir}/embed_exports/{batch_name}"


def get_rsync_target(worker: Worker, batch_name: str) -> str:
    """Get rsync target path for worker."""
    if worker.host == 'localhost':
        return f"embed@localhost:/Users/embed/imageselector/embed_exports/{batch_name}"
    elif worker.host == '192.168.68.117':
        return f"{worker.user}@{worker.host}:/cygdrive/c/Users/embed/imageselector/embed_exports/{batch_name}"
    else:
        return f"{worker.user}@{worker.host}:{worker.work_dir}/embed_exports/{batch_name}"


def rsync_to_worker(worker: Worker, batch_dir: Path) -> bool:
    """Rsync batch to worker."""
    target = get_rsync_target(worker, batch_dir.name)

    cmd = ['rsync', '-av', '--delete', f"{batch_dir}/", target]
    if worker.host == '192.168.68.117':
        cmd = ['rsync', '-av', '--delete', f'--rsync-path={worker.rsync_path}', f"{batch_dir}/", target]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  {worker.name}: rsync to worker timed out")
        return False


def start_worker_job(worker: Worker, batch_name: str) -> bool:
    """SSH to start worker process (non-blocking)."""
    batch_path = get_worker_batch_path(worker, batch_name)
    worker_script = f"{worker.work_dir}/image_search_v2/embed_worker.py"

    if worker.host == '192.168.68.117':
        # Windows - run Python directly (SSH session stays open in local background via Popen)
        # start /B via SSH is unreliable; running directly works
        remote_cmd = f'cd /d "{worker.work_dir}\\image_search_v2" && "{worker.python_path}" embed_worker.py --input "{batch_path}"'
    else:
        # Mac - use nohup to keep process running after SSH exits
        remote_cmd = f'nohup {worker.python_path} {worker_script} --input {batch_path} > /dev/null 2>&1 &'

    ssh_host = 'localhost' if worker.host == 'localhost' else worker.host
    ssh_cmd = ['ssh', f'{worker.user}@{ssh_host}', remote_cmd]

    # Start non-blocking (SSH runs in background, stays connected until command completes)
    subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def check_worker_done(worker: Worker, batch_name: str) -> bool:
    """Check if worker completed by looking for last model's .npy file."""
    batch_dir = EXPORTS_DIR / batch_name

    # Quick rsync to check for .npy files
    src = get_rsync_target(worker, batch_name) + "/"

    cmd = ['rsync', '-av', '--include=*.npy', '--exclude=*', src, f"{batch_dir}/"]
    if worker.host == '192.168.68.117':
        cmd = ['rsync', '-av', f'--rsync-path={worker.rsync_path}', '--include=*.npy', '--exclude=*', src, f"{batch_dir}/"]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None  # Signal timeout (distinct from False = not done yet)

    # Check if all models complete
    return all((batch_dir / f"{model}.npy").exists() for model in ALL_MODELS)


def rsync_images_back(worker: Worker, batch_name: str) -> bool:
    """Rsync bg-removed images back from worker."""
    batch_dir = EXPORTS_DIR / batch_name
    src = get_rsync_target(worker, batch_name) + "/images/"

    cmd = ['rsync', '-av', src, f"{batch_dir}/images/"]
    if worker.host == '192.168.68.117':
        cmd = ['rsync', '-av', f'--rsync-path={worker.rsync_path}', src, f"{batch_dir}/images/"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  {worker.name}: rsync images back timed out")
        return False


def cleanup_worker_batch(worker: Worker, batch_name: str):
    """Clean up batch on worker after import."""
    ssh_host = 'localhost' if worker.host == 'localhost' else worker.host

    if worker.host == '192.168.68.117':
        # Windows
        win_path = f"C:\\Users\\embed\\imageselector\\embed_exports\\{batch_name}"
        cmd = ['ssh', f'{worker.user}@{ssh_host}', f'cmd /c rmdir /s /q "{win_path}"']
    else:
        remote_path = get_worker_batch_path(worker, batch_name)
        cmd = ['ssh', f'{worker.user}@{ssh_host}', f'rm -rf {remote_path}']

    subprocess.run(cmd, capture_output=True, timeout=60)


def kill_worker_processes(worker: Worker) -> bool:
    """Kill any running embed_worker processes on a worker."""
    ssh_host = 'localhost' if worker.host == 'localhost' else worker.host

    try:
        if worker.host == '192.168.68.117':
            # Windows - kill Python processes running embed_worker
            # wmic is flaky via SSH, use taskkill with image name
            # This will kill ALL python.exe processes for the user
            cmd = ['ssh', f'{worker.user}@{ssh_host}',
                   'taskkill /F /IM python.exe 2>nul || echo no_process']
        else:
            # Mac - kill only embed_worker processes for this user
            cmd = ['ssh', f'{worker.user}@{ssh_host}',
                   'pkill -f embed_worker.py 2>/dev/null || true']

        subprocess.run(cmd, capture_output=True, timeout=30)
        return True
    except Exception as e:
        log(f"  Warning: Could not kill processes on {worker.name}: {e}")
        return False


def clear_worker_exports(worker: Worker) -> bool:
    """Clear all batch directories on a worker."""
    ssh_host = 'localhost' if worker.host == 'localhost' else worker.host

    try:
        if worker.host == '192.168.68.117':
            # Windows - delete all batch_* dirs
            cmd = ['ssh', f'{worker.user}@{ssh_host}',
                   'for /d %i in (C:\\Users\\embed\\imageselector\\embed_exports\\batch_*) do rmdir /s /q "%i"']
        else:
            # Mac
            cmd = ['ssh', f'{worker.user}@{ssh_host}',
                   f'rm -rf {worker.work_dir}/embed_exports/batch_*']

        subprocess.run(cmd, capture_output=True, timeout=30)
        return True
    except Exception as e:
        log(f"  Warning: Could not clear exports on {worker.name}: {e}")
        return False


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    if KILL_FILE.exists():
        print(f"Error: Kill file exists ({KILL_FILE}). Remove it to start.")
        return

    if not acquire_lock():
        print("Error: Another orchestrator instance is already running")
        return

    # Fresh slate: clear any leftover export batches locally
    if EXPORTS_DIR.exists():
        for item in EXPORTS_DIR.iterdir():
            if item.is_dir() and item.name.startswith('batch_'):
                log(f"Cleaning up local batch: {item.name}")
                shutil.rmtree(item)

    EXPORTS_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # Kill any running worker processes and clear batch directories
    log("Cleaning up workers (killing processes + clearing batches)...")
    for name, worker in WORKERS.items():
        kill_worker_processes(worker)
        clear_worker_exports(worker)
        log(f"  Cleaned {name}")

    log("=" * 60)
    log("ORCHESTRATOR STARTED")
    log("=" * 60)

    # CRITICAL: Check DB/FAISS consistency before doing any work
    is_consistent, msg = check_db_faiss_consistency()
    if not is_consistent:
        log(f"FATAL: DB/FAISS inconsistent - {msg}")
        log("Fix manually: either reset faiss_row to NULL and delete FAISS files,")
        log("or restore from last consistent backup.")
        release_lock()
        return

    log(f"Consistency check: {msg}")
    log(f"Workers: {list(WORKERS.keys())}")
    log("Stop with: touch KILL_ORCH")

    # Initialize worker states
    worker_states: Dict[str, WorkerState] = {
        name: WorkerState(worker=worker)
        for name, worker in WORKERS.items()
    }

    # Import queue: completed batches waiting to be imported
    import_queue: List[Tuple[str, Path]] = []  # (worker_name, batch_dir)

    last_qps_check = 0
    batch_counter = 0
    total_imported = 0

    # Find most recent backup to avoid immediate backup on restart
    # Check both local backups/ and SSD
    last_backup = 0
    backup_dirs = [BACKUPS_DIR, Path('/Volumes/SSD_120/backups')]
    all_backups = []
    for bdir in backup_dirs:
        if bdir.exists():
            all_backups.extend(bdir.glob("etsy_data_*.db"))
    if all_backups:
        most_recent = max(all_backups, key=lambda f: f.stat().st_mtime)
        last_backup = most_recent.stat().st_mtime
        log(f"Last backup: {most_recent.name} ({most_recent.parent})")

    try:
        while not check_kill_file():
            now = time.time()

            # QPS check every 5 minutes
            if now - last_qps_check > QPS_CHECK_INTERVAL:
                limits = check_rate_limit()
                if limits:
                    update_qps_config(limits)
                last_qps_check = now

            # Backup every 6 hours (uses SQLite backup API - safe for concurrent access)
            if now - last_backup > BACKUP_INTERVAL:
                log("Starting 6-hour backup...")
                do_backup()
                last_backup = now

            # Check each worker
            should_exit = False
            for name, state in worker_states.items():
                if check_kill_file():
                    should_exit = True
                    break

                worker = state.worker

                if state.status == 'frozen':
                    continue

                if state.status == 'working':
                    # Check if done
                    result = check_worker_done(worker, state.current_batch)

                    if result is None:
                        # Timeout
                        state.check_failures += 1
                        log(f"  {name}: rsync timeout ({state.check_failures}/{MAX_CHECK_RETRIES})")
                        if state.check_failures >= MAX_CHECK_RETRIES:
                            log(f"  {name}: FROZEN after {MAX_CHECK_RETRIES} timeouts")
                            state.status = 'frozen'
                    elif result:
                        # Done
                        state.check_failures = 0
                        log(f"{name}: batch complete, syncing images back")
                        rsync_images_back(worker, state.current_batch)

                        batch_dir = EXPORTS_DIR / state.current_batch
                        import_queue.append((name, batch_dir))
                        state.status = 'done'
                    # else: not done yet, keep waiting

                elif state.status in ('idle', 'done'):
                    # Get new batch for this worker
                    images = get_pending_images(worker.batch_size)

                    if images:
                        batch_counter += 1
                        batch_name = f"batch_{batch_counter:05d}_{name}"

                        log(f"{name}: starting batch {batch_name} ({len(images)} images)")

                        batch_dir = export_batch(batch_name, images)

                        if rsync_to_worker(worker, batch_dir):
                            start_worker_job(worker, batch_name)
                            state.current_batch = batch_name
                            state.batch_images = images
                            state.started_at = now
                            state.status = 'working'
                        else:
                            log(f"{name}: rsync failed, will retry")
                            shutil.rmtree(batch_dir)
                    else:
                        if state.status != 'idle':
                            log(f"{name}: no pending images, going idle")
                        state.status = 'idle'

            if should_exit:
                break

            # Process import queue (one batch per loop iteration)
            if import_queue:
                worker_name, batch_dir = import_queue.pop(0)
                imported = import_batch(batch_dir)
                total_imported += imported

                # Cleanup
                cleanup_worker_batch(WORKERS[worker_name], batch_dir.name)
                shutil.rmtree(batch_dir)

                log(f"Total imported: {total_imported}")

            # Status summary
            working = sum(1 for s in worker_states.values() if s.status == 'working')
            idle = sum(1 for s in worker_states.values() if s.status == 'idle')

            if working == 0 and idle == len(worker_states) and not import_queue:
                log("All workers idle, no pending work. Waiting...")
                time.sleep(60)
            else:
                time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        log("Interrupted")

    finally:
        release_lock()
        log(f"Orchestrator stopped. Total imported: {total_imported}")


if __name__ == "__main__":
    main()
