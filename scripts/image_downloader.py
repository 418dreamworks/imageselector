#!/usr/bin/env python3
"""Image downloader with manager/worker architecture.

Manager (main thread):
- Scans for images where download_done=0, reads URL from DB
- Creates .dl marker files with URLs, marks download_done=1
- Checks for completed jpgs (>1kb), marks download_done=2, cleans up .dl files

Workers (4 threads, partitioned):
- Each worker handles files where first_5_digits % 4 == worker_id
- Reads URL from marker, downloads image, saves jpg, deletes marker

Kill file: touch KILL_DL to stop gracefully.
"""
import os
import sys
import threading
import time
from pathlib import Path

import urllib.request
import urllib.error

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
KILL_FILE = BASE_DIR / "KILL_DL"
PID_FILE = BASE_DIR / "image_downloader.pid"
NUM_WORKERS = 1

# Import from shared image_db module
sys.path.insert(0, str(BASE_DIR))
from image_db import get_connection, commit_with_retry


def ts():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


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
    """Release PID lock."""
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def get_worker_kill_file(worker_id: int) -> Path:
    return BASE_DIR / f"KILL_DL_{worker_id}"


def check_manager_kill_file() -> bool:
    """Check for kill file. If found, create worker kill files and return True."""
    if KILL_FILE.exists():
        print(f"\n[{ts()}] Kill file detected. Notifying workers...")
        for i in range(NUM_WORKERS):
            get_worker_kill_file(i).touch()
        KILL_FILE.unlink()
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
    """
    1. Check in-progress images (download_done=1) - if jpg exists, mark as 2
    2. Create markers for new images (download_done=0) - read URL from DB
    """
    stats = {"completed": 0, "markers_created": 0, "skipped": 0}

    conn = get_connection()

    # --- Phase 1: Check in-progress images for completion ---
    cursor = conn.execute("""
        SELECT listing_id, image_id
        FROM image_status
        WHERE download_done = 1
        LIMIT 5000
    """)
    in_progress = cursor.fetchall()

    for row in in_progress:
        lid, iid = row[0], row[1]
        img_path = get_image_path(lid, iid)
        marker_path = get_marker_path(lid, iid)

        if img_path.exists() and img_path.stat().st_size > 1000:
            # Image complete - mark as done
            conn.execute("""
                UPDATE image_status SET download_done = 2
                WHERE listing_id = ? AND image_id = ? AND download_done = 1
            """, (lid, iid))
            stats["completed"] += 1

            # Clean up marker if it exists
            if marker_path.exists():
                try:
                    marker_path.unlink()
                except FileNotFoundError:
                    pass

    if stats["completed"] > 0:
        commit_with_retry(conn)

    # --- Phase 2: Create markers for new images ---
    cursor = conn.execute("""
        SELECT listing_id, image_id, url
        FROM image_status
        WHERE download_done = 0 AND url IS NOT NULL AND url != ''
        LIMIT 1000
    """)
    new_images = cursor.fetchall()

    # Step 1: Write ALL marker files first
    markers_written = []
    for row in new_images:
        lid, iid, url = row[0], row[1], row[2]
        marker_path = get_marker_path(lid, iid)

        if not marker_path.exists():
            marker_path.write_text(url)
            markers_written.append((lid, iid))
            stats["markers_created"] += 1
        else:
            stats["skipped"] += 1

    # Step 2: Batch update DB flags AFTER all markers written
    for lid, iid in markers_written:
        conn.execute("""
            UPDATE image_status SET download_done = 1
            WHERE listing_id = ? AND image_id = ? AND download_done = 0
        """, (lid, iid))

    if markers_written:
        commit_with_retry(conn)

    conn.close()
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
    except Exception:
        # Download failed - leave marker for retry
        return False


def worker_loop(worker_id: int):
    """Worker loop - find and process marker files for this partition."""
    print(f"[{ts()}] Worker {worker_id} started")

    while not check_worker_kill_file(worker_id):
        # Find marker files for this worker's partition
        try:
            markers = [m for m in IMAGES_DIR.glob("*.dl")
                       if get_file_partition(m.name) == worker_id]
        except Exception:
            markers = []

        if not markers:
            time.sleep(1)
            continue

        for marker_path in markers:
            if check_worker_kill_file(worker_id):
                return

            success = download_one(marker_path)
            if success:
                print(".", end="", flush=True)

            # Small delay between downloads
            time.sleep(0.05)

    print(f"\n[{ts()}] Worker {worker_id} stopped")


# ============================================================
# MAIN
# ============================================================

def main():
    if not acquire_lock():
        print("Error: Another image_downloader.py instance is already running")
        return

    IMAGES_DIR.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"IMAGE DOWNLOADER")
    print(f"{'='*60}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Stop with: touch KILL_DL")
    print(f"{'='*60}")

    # Start worker threads
    threads = []
    for i in range(NUM_WORKERS):
        t = threading.Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Manager loop
    scan_interval = 30
    last_scan = 0

    while not check_manager_kill_file():
        now = time.time()

        if now - last_scan >= scan_interval:
            stats = manager_scan()
            last_scan = now

            if stats["markers_created"] > 0 or stats["completed"] > 0:
                print(f"\n[{ts()}] Scan: markers={stats['markers_created']} "
                      f"completed={stats['completed']} skipped={stats['skipped']}")

        time.sleep(1)

    # Wait for workers to finish
    print(f"[{ts()}] Waiting for workers to finish...")
    for t in threads:
        t.join(timeout=5)

    release_lock()
    print(f"\n[{ts()}] Downloader stopped")


if __name__ == "__main__":
    try:
        main()
    finally:
        release_lock()
