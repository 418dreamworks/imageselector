#!/usr/bin/env python3
"""Image downloader with manager/worker architecture.

Manager:
- Scans SQL for images where download_done=0
- Checks if file exists on disk
- If missing, creates marker file: {listing_id}_{image_id}.dl with URL inside
- When real file detected (non-empty jpg), sets download_done=1

Worker:
- Finds .dl marker files
- Downloads the image
- Saves as .jpg, removes .dl file
- If killed mid-download, no problem - manager recreates marker

Completely decoupled from sync_data.py.
"""
import argparse
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

# Import from shared image_db module
sys.path.insert(0, str(BASE_DIR))
from image_db import (
    get_connection, get_images_to_download, mark_download_done,
    build_cdn_url, commit_with_retry
)


def ts():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


def check_kill_file() -> bool:
    """Check for kill file. Delete if found and return True."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        print(f"\n[{ts()}] Kill file detected. Stopping...")
        return True
    return False


# ============================================================
# MANAGER - Creates marker files, updates SQL when done
# ============================================================

def get_marker_path(listing_id: int, image_id: int) -> Path:
    """Get path for download marker file."""
    return IMAGES_DIR / f"{listing_id}_{image_id}.dl"


def get_image_path(listing_id: int, image_id: int) -> Path:
    """Get path for actual image file."""
    return IMAGES_DIR / f"{listing_id}_{image_id}.jpg"


def manager_scan(batch_size: int = 1000) -> dict:
    """Scan for images needing download, create markers, update completed.

    Returns stats dict.
    """
    stats = {"markers_created": 0, "completed": 0, "already_done": 0}

    conn = get_connection()
    images = get_images_to_download(conn, limit=batch_size)

    for img in images:
        lid = img["listing_id"]
        iid = img["image_id"]
        hex_val = img["hex"]
        suffix = img["suffix"]

        img_path = get_image_path(lid, iid)
        marker_path = get_marker_path(lid, iid)

        # Check if real image exists and has content
        if img_path.exists() and img_path.stat().st_size > 1000:
            # Image is complete - mark as done in SQL
            mark_download_done(conn, lid, iid)
            stats["completed"] += 1
            # Remove any stale marker
            if marker_path.exists():
                marker_path.unlink()
        elif not marker_path.exists():
            # No image, no marker - create marker with URL
            url = build_cdn_url(hex_val, iid, suffix)
            marker_path.write_text(url)
            stats["markers_created"] += 1
        else:
            stats["already_done"] += 1

    commit_with_retry(conn)
    conn.close()

    return stats


# ============================================================
# WORKER - Downloads from marker files
# ============================================================

def download_one(marker_path: Path) -> bool:
    """Download image from marker file.

    1. Read URL from marker
    2. Delete marker (claim the job)
    3. Download image
    4. Save as .jpg

    Returns True on success.
    """
    try:
        # Read URL from marker
        url = marker_path.read_text().strip()
        if not url:
            return False

        # Parse listing_id and image_id from filename
        # Format: {listing_id}_{image_id}.dl
        stem = marker_path.stem  # e.g., "123456_789012"
        parts = stem.split("_")
        if len(parts) != 2:
            return False

        listing_id, image_id = parts
        img_path = IMAGES_DIR / f"{listing_id}_{image_id}.jpg"

        # Delete marker (claim the job)
        marker_path.unlink()

        # Download
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()

        # Save image
        img_path.write_bytes(data)
        return True

    except FileNotFoundError:
        # Another worker claimed it
        return False
    except Exception as e:
        # Download failed - recreate marker so manager can retry
        try:
            if not marker_path.exists():
                marker_path.write_text(url)
        except:
            pass
        return False


def worker_loop(rate_limit: float = 10.0):
    """Worker loop - find and process marker files."""
    min_interval = 1.0 / rate_limit
    last_download = 0.0

    while not check_kill_file():
        # Find marker files
        markers = list(IMAGES_DIR.glob("*.dl"))

        if not markers:
            time.sleep(1)
            continue

        for marker_path in markers:
            if check_kill_file():
                return

            # Rate limit
            elapsed = time.time() - last_download
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            success = download_one(marker_path)
            last_download = time.time()

            if success:
                print(".", end="", flush=True)


def run_workers(num_workers: int = 4, rate_limit: float = 10.0):
    """Run multiple worker threads."""
    print(f"[{ts()}] Starting {num_workers} workers (rate limit: {rate_limit}/s)")
    print("Stop with: touch KILL_DL")

    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker_loop, args=(rate_limit / num_workers,))
        t.start()
        threads.append(t)

    # Wait for all workers
    for t in threads:
        t.join()

    print(f"\n[{ts()}] Workers stopped")


# ============================================================
# MAIN
# ============================================================

def run_once(batch_size: int = 1000, num_workers: int = 4, rate_limit: float = 10.0):
    """Run one cycle: manager scan + worker downloads."""
    IMAGES_DIR.mkdir(exist_ok=True)

    # Manager: create markers, update completed
    print(f"[{ts()}] Manager scanning...")
    stats = manager_scan(batch_size)
    print(f"  Markers created: {stats['markers_created']}")
    print(f"  Completed (file exists): {stats['completed']}")

    if stats["markers_created"] == 0:
        print("Nothing to download")
        return

    # Workers: download from markers
    run_workers(num_workers, rate_limit)

    # Final manager scan to mark completed
    print(f"\n[{ts()}] Final scan...")
    stats = manager_scan(batch_size)
    print(f"  Newly completed: {stats['completed']}")


def watch_mode(batch_size: int = 1000, num_workers: int = 4, rate_limit: float = 10.0, interval: int = 30):
    """Watch mode - continuously scan and download."""
    IMAGES_DIR.mkdir(exist_ok=True)

    print(f"[{ts()}] Watch mode: batch={batch_size}, workers={num_workers}, rate={rate_limit}/s")
    print("Stop with: touch KILL_DL")

    while not check_kill_file():
        # Manager scan
        stats = manager_scan(batch_size)

        if stats["markers_created"] > 0:
            print(f"\n[{ts()}] Created {stats['markers_created']} markers, completed {stats['completed']}")

            # Run workers until all markers processed
            while True:
                markers = list(IMAGES_DIR.glob("*.dl"))
                if not markers or check_kill_file():
                    break

                # Process one batch
                for marker_path in markers[:100]:
                    if check_kill_file():
                        break
                    download_one(marker_path)
                    print(".", end="", flush=True)
                    time.sleep(1.0 / rate_limit)

            # Mark completed
            stats = manager_scan(batch_size)
            print(f"\n  Completed: {stats['completed']}")
        else:
            print(".", end="", flush=True)
            time.sleep(interval)


def show_stats():
    """Show current download stats."""
    conn = get_connection()

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM image_status WHERE to_download = 1")
    to_dl = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 1")
    done = cursor.fetchone()[0]

    conn.close()

    # Count marker files
    IMAGES_DIR.mkdir(exist_ok=True)
    markers = len(list(IMAGES_DIR.glob("*.dl")))

    print(f"To download:    {to_dl:,}")
    print(f"Downloaded:     {done:,}")
    print(f"Pending:        {to_dl - done:,}")
    print(f"Marker files:   {markers:,}")


def main():
    parser = argparse.ArgumentParser(description="Download images with manager/worker architecture")
    parser.add_argument("--watch", action="store_true", help="Watch mode - run continuously")
    parser.add_argument("--batch-size", type=int, default=1000, help="Images per manager scan")
    parser.add_argument("--workers", type=int, default=4, help="Number of download workers")
    parser.add_argument("--rate", type=float, default=10.0, help="Downloads per second")
    parser.add_argument("--interval", type=int, default=30, help="Watch mode scan interval")
    parser.add_argument("--stats", action="store_true", help="Show current stats")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.watch:
        watch_mode(args.batch_size, args.workers, args.rate, args.interval)
    else:
        run_once(args.batch_size, args.workers, args.rate)


if __name__ == "__main__":
    main()
