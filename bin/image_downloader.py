#!/usr/bin/env python3
"""Image downloader with manager/worker architecture (continuous loop).

download_done values:
  -1 = dead URL (404 from CDN, image no longer exists)
   0 = needs download
   1 = marker (.dl) created, waiting for worker to download
   2 = download complete (jpg > 1kb on disk)
   3 = tarred (set by tar_images after each 10K batch)
   4 = shard finalized (set by tar_images when all 50 tars for 500K shard exist)

Flow (each pass):
1. manager_scan() — create .dl markers for all download_done=0, reconcile
2. Start 8 worker threads — each processes its partition of .dl markers, exits when none remain
3. Wait for all workers to finish
4. manager_scan() — final reconciliation (mark completed downloads as download_done=2)
5. Sleep 2 min, then loop back to 1

Kill file: touch KILL_DL to stop gracefully.
"""
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import urllib.request
import urllib.error

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images" / "imagedownload"
KILL_FILE = BASE_DIR / "KILL_DL"
PID_FILE = BASE_DIR / "image_downloader.pid"
PAUSE_FILE = BASE_DIR / "PAUSE_DL"
PAUSED_FILE = BASE_DIR / "PAUSED_DL"
NUM_WORKERS = 8

# Import from shared image_db module
sys.path.insert(0, str(BASE_DIR / "bin"))
from image_db import get_connection, commit_with_retry


def ts():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


_lock_acquired = False

def acquire_lock() -> bool:
    """Acquire PID lock. Returns False if another instance is running."""
    global _lock_acquired
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    PID_FILE.write_text(str(os.getpid()))
    _lock_acquired = True
    return True


def release_lock():
    """Release PID lock."""
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def get_worker_kill_file(worker_id: int) -> Path:
    return BASE_DIR / f"KILL_DL_{worker_id}"


def wait_for_backup_lock():
    while PAUSE_FILE.exists():
        PAUSED_FILE.touch()
        print(f"[{ts()}] Backup in progress, paused...")
        time.sleep(10)
    PAUSED_FILE.unlink(missing_ok=True)


def check_manager_kill_file() -> bool:
    """Check for kill file. If found, create worker kill files and return True."""
    if KILL_FILE.exists():
        print(f"\n[{ts()}] Kill file detected. Notifying workers...")
        for i in range(NUM_WORKERS):
            get_worker_kill_file(i).touch()
        return True
    return False


def check_worker_kill_file(worker_id: int) -> bool:
    """Check for worker-specific kill file. Delete if found and return True."""
    kill_file = get_worker_kill_file(worker_id)
    if kill_file.exists():
        kill_file.unlink()
        return True
    return False


# ============================================================
# MANAGER - Reads URLs from DB, creates markers, updates flags
# ============================================================

def get_marker_path(listing_id: int, image_id: int) -> Path:
    return IMAGES_DIR / f"{listing_id}_{image_id}.dl"


def get_image_path(listing_id: int, image_id: int) -> Path:
    return IMAGES_DIR / f"{listing_id}_{image_id}.jpg"


def manager_scan() -> dict:
    """Full reconciliation cycle: snapshot DB, check filesystem, batch update.

    1. Grab all download_done=0 and download_done=1 from DB, release DB
    2. Scan filesystem: find jpgs >= 1KB, find .dl markers
    3. For dl=1 with jpg >= 1KB: promote to 2, delete .dl marker
    4. For dl=0 with .dl on disk: promote to 1 (orphan reconciliation)
    5. For dl=0 without .dl: create .dl marker file, promote to 1
    6. Hold DB briefly, batch UPDATE all changes, release
    """
    stats = {"completed": 0, "markers_created": 0, "reconciled": 0}

    # --- Step 1: Snapshot DB (brief hold) ---
    conn = get_connection()
    try:
        rows_0 = conn.execute("""
            SELECT listing_id, image_id, url
            FROM image_status
            WHERE download_done = 0
        """).fetchall()
        rows_1 = conn.execute("""
            SELECT listing_id, image_id
            FROM image_status
            WHERE download_done = 1
        """).fetchall()
    finally:
        conn.close()

    print(f"\n[{ts()}] Reconcile: dl=0: {len(rows_0):,}, dl=1: {len(rows_1):,}")

    # Build lookup dicts
    needs_download = {}  # (lid, iid) -> url  for download_done=0
    for row in rows_0:
        lid, iid, url = row[0], row[1], row[2]
        if url:
            needs_download[(lid, iid)] = url

    in_progress = set()  # (lid, iid) for download_done=1
    for row in rows_1:
        in_progress.add((row[0], row[1]))

    # --- Step 2: Check filesystem (no DB hold) ---
    promote_to_2 = []   # (lid, iid) - have jpg, move 1 -> 2
    promote_to_1 = []   # (lid, iid) - have .dl or new .dl, move 0 -> 1
    markers_to_delete = []  # paths to clean up

    # Check dl=1 entries for completed jpgs
    for lid, iid in in_progress:
        jpg = get_image_path(lid, iid)
        dl = get_marker_path(lid, iid)
        if jpg.exists() and jpg.stat().st_size > 1000:
            promote_to_2.append((lid, iid))
            if dl.exists():
                markers_to_delete.append(dl)

    # Check dl=0 entries
    for (lid, iid), url in needs_download.items():
        dl = get_marker_path(lid, iid)
        jpg = get_image_path(lid, iid)
        if jpg.exists() and jpg.stat().st_size > 1000:
            # Already downloaded but DB stuck at 0 - skip to 2
            promote_to_2.append((lid, iid))
            if dl.exists():
                markers_to_delete.append(dl)
        elif dl.exists():
            # Orphan: .dl exists but DB says 0 - reconcile to 1
            promote_to_1.append((lid, iid))
            stats["reconciled"] += 1
        else:
            # No .dl, no jpg - create marker
            dl.write_text(url)
            promote_to_1.append((lid, iid))
            stats["markers_created"] += 1

    # --- Step 3: Clean up marker files ---
    for path in markers_to_delete:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # --- Step 4: Batch UPDATE DB (brief hold) ---
    stats["completed"] = len(promote_to_2)

    for attempt in range(10):
        try:
            conn = get_connection()
            try:
                if promote_to_2:
                    conn.executemany(
                        "UPDATE image_status SET download_done = 2 WHERE listing_id = ? AND image_id = ?",
                        promote_to_2
                    )
                if promote_to_1:
                    conn.executemany(
                        "UPDATE image_status SET download_done = 1 WHERE listing_id = ? AND image_id = ?",
                        promote_to_1
                    )
                commit_with_retry(conn)
            finally:
                conn.close()
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < 9:
                time.sleep(2 * (attempt + 1))
            else:
                raise

    return stats


# ============================================================
# WORKER - Downloads from marker files
# ============================================================

def get_file_partition(filename: str) -> int:
    """Get partition number from filename using first 5 digits."""
    digits = ''.join(c for c in filename if c.isdigit())
    if len(digits) < 5:
        digits = digits.zfill(5)
    return int(digits[:5]) % NUM_WORKERS


def download_one(marker_path: Path) -> bool:
    """Download image from marker file.

    1. Read URL from marker
    2. Download image
    3. Save as .jpg
    4. Delete marker

    Returns True on success.
    """
    try:
        # Read URL from marker
        url = marker_path.read_text().strip()
        if not url:
            return False

        # Parse listing_id and image_id from filename
        stem = marker_path.stem  # e.g., "123456_789012"
        parts = stem.split("_")
        if len(parts) != 2:
            return False

        listing_id, image_id = int(parts[0]), int(parts[1])
        img_path = IMAGES_DIR / f"{listing_id}_{image_id}.jpg"

        # Download
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

        # Save image
        img_path.write_bytes(data)

        # Delete marker only after successful save
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass

        return True

    except FileNotFoundError:
        return False
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Image no longer exists on CDN - mark as dead
            try:
                stem = marker_path.stem
                parts = stem.split("_")
                if len(parts) == 2:
                    listing_id, image_id = int(parts[0]), int(parts[1])
                    conn = get_connection()
                    conn.execute(
                        "UPDATE image_status SET download_done = -1 WHERE listing_id = ? AND image_id = ?",
                        (listing_id, image_id)
                    )
                    conn.commit()
                    conn.close()
                marker_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False
    except Exception:
        # Download failed - leave marker for retry
        return False


import shutil

workers_paused = False
MIN_DISK_GB = 5  # Pause workers when free space drops below this


def check_disk_space():
    """Check free disk space on the images partition. Returns free GB."""
    usage = shutil.disk_usage(IMAGES_DIR)
    return usage.free / (1024 ** 3)


def worker_loop(worker_id: int):
    """Worker loop - find and process marker files for this partition, exit when done."""
    print(f"[{ts()}] Worker {worker_id} started")

    while not check_worker_kill_file(worker_id):
        if workers_paused:
            time.sleep(5)
            continue

        # Find marker files for this worker's partition
        try:
            markers = [m for m in IMAGES_DIR.glob("*.dl")
                       if get_file_partition(m.name) == worker_id]
        except Exception:
            markers = []

        if not markers:
            break  # All markers processed, done

        for marker_path in markers:
            if check_worker_kill_file(worker_id) or workers_paused:
                break

            success = download_one(marker_path)
            if success:
                print(".", end="", flush=True)

            # Small delay between downloads
            time.sleep(0.2)

    print(f"\n[{ts()}] Worker {worker_id} stopped")


# ============================================================
# MAIN
# ============================================================

def main():
    if KILL_FILE.exists():
        print(f"Error: Kill file exists ({KILL_FILE}). Remove it to start.")
        return

    if not acquire_lock():
        print("Error: Another image_downloader.py instance is already running")
        return

    IMAGES_DIR.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"IMAGE DOWNLOADER (continuous)")
    print(f"{'='*60}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Stop with: touch KILL_DL")
    print(f"{'='*60}")

    pass_num = 0
    while not KILL_FILE.exists():
        pass_num += 1
        print(f"\n[{ts()}] === Pass {pass_num} ===")

        # Step 1: Scan — create markers, reconcile
        wait_for_backup_lock()
        stats = manager_scan()
        print(f"[{ts()}] Scan: completed={stats['completed']:,} "
              f"markers={stats['markers_created']:,} "
              f"reconciled={stats['reconciled']:,}")

        if check_manager_kill_file():
            break

        # Check disk space before starting workers
        free_gb = check_disk_space()
        # Check if any .dl markers exist on disk (workers process these)
        has_markers = next((True for f in IMAGES_DIR.iterdir() if f.suffix == '.dl'), False)
        if free_gb < MIN_DISK_GB:
            print(f"[{ts()}] DISK LOW: {free_gb:.1f}GB free — skipping downloads")
        elif not has_markers:
            print(f"[{ts()}] No markers on disk, nothing to download")
        else:
            # Step 2: Start worker threads
            threads = []
            for i in range(NUM_WORKERS):
                t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
                t.start()
                threads.append(t)

            # Step 3: Wait for all workers to finish (with periodic disk/kill/reconciliation checks)
            global workers_paused
            last_reconcile = time.time()
            while any(t.is_alive() for t in threads):
                if check_manager_kill_file():
                    break
                wait_for_backup_lock()

                free_gb = check_disk_space()
                if free_gb < MIN_DISK_GB and not workers_paused:
                    workers_paused = True
                    print(f"\n[{ts()}] DISK LOW: {free_gb:.1f}GB free — workers paused")
                elif free_gb >= MIN_DISK_GB and workers_paused:
                    workers_paused = False
                    print(f"\n[{ts()}] DISK OK: {free_gb:.1f}GB free — workers resumed")

                # Periodic reconciliation so orchestrator can see new dl=2 images
                if time.time() - last_reconcile >= 120:
                    mid_stats = manager_scan()
                    print(f"[{ts()}] Mid-pass reconcile: completed={mid_stats['completed']:,} "
                          f"markers={mid_stats['markers_created']:,}")
                    last_reconcile = time.time()

                time.sleep(1)

            for t in threads:
                t.join(timeout=5)

            # Final reconciliation for this pass
            stats = manager_scan()
            print(f"[{ts()}] Pass {pass_num} final: completed={stats['completed']:,} "
                  f"markers={stats['markers_created']:,} "
                  f"reconciled={stats['reconciled']:,}")

        if KILL_FILE.exists():
            break

        # Sleep 2 min between passes (12 × 10s), checking kill file
        print(f"[{ts()}] Pass {pass_num} done. Sleeping 2 min...")
        for _ in range(12):
            if KILL_FILE.exists():
                break
            time.sleep(10)

    release_lock()
    print(f"\n[{ts()}] Downloader stopped")


if __name__ == "__main__":
    try:
        main()
    finally:
        if _lock_acquired:
            release_lock()
