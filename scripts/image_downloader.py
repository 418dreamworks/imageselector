#!/usr/bin/env python3
"""Image downloader with manager/worker architecture.

Manager (main thread):
- Fetches images where download_done=0, groups by listing_id
- Calls /application/listings/batch?includes=Images to get URLs
- Writes .dl marker files with URLs, then marks download_done=1
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
from collections import defaultdict

import urllib.request
import urllib.error
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
KILL_FILE = BASE_DIR / "KILL_DL"
NUM_WORKERS = 4

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Import from shared image_db module
sys.path.insert(0, str(BASE_DIR))
from image_db import get_connection, commit_with_retry


def ts():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")


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
# MANAGER - Fetches URLs via API, creates markers, updates SQL
# ============================================================

def get_marker_path(listing_id: int, image_id: int) -> Path:
    return IMAGES_DIR / f"{listing_id}_{image_id}.dl"


def get_image_path(listing_id: int, image_id: int) -> Path:
    return IMAGES_DIR / f"{listing_id}_{image_id}.jpg"


def get_images_needing_download(conn, max_listings: int = 100) -> list[dict]:
    """Get images where download_done=0, limited to max_listings unique listings."""
    cursor = conn.execute("""
        SELECT listing_id, image_id
        FROM image_status
        WHERE download_done = 0
        AND listing_id IN (
            SELECT DISTINCT listing_id FROM image_status
            WHERE download_done = 0
            LIMIT ?
        )
    """, (max_listings,))
    return [{"listing_id": row[0], "image_id": row[1]} for row in cursor.fetchall()]


def get_images_in_progress(conn, limit: int = 5000) -> list[dict]:
    """Get images where download_done=1 (marker created, download in progress)."""
    cursor = conn.execute("""
        SELECT listing_id, image_id
        FROM image_status
        WHERE download_done = 1
        LIMIT ?
    """, (limit,))
    return [{"listing_id": row[0], "image_id": row[1]} for row in cursor.fetchall()]


def fetch_listing_images(listing_ids: list[int]) -> dict[int, dict[int, str]]:
    """Fetch image URLs for multiple listings via /application/listings/batch.

    Args:
        listing_ids: List of listing IDs to fetch

    Returns:
        Dict mapping listing_id -> {image_id: url_570xN, ...}
    """
    if not listing_ids:
        return {}

    result = {}
    headers = {"x-api-key": ETSY_API_KEY}

    # Etsy batch endpoint accepts up to 100 listing IDs
    for i in range(0, len(listing_ids), 100):
        batch = listing_ids[i:i + 100]
        ids_param = ",".join(str(lid) for lid in batch)

        url = f"{BASE_URL}/application/listings/batch?listing_ids={ids_param}&includes=Images"

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            print(f"[{ts()}] API error: {e}")
            continue

        # Parse response
        for listing in data.get("results", []):
            lid = listing.get("listing_id")
            images = listing.get("images", [])

            if lid and images:
                result[lid] = {}
                for img in images:
                    iid = img.get("listing_image_id")
                    img_url = img.get("url_570xN")
                    if iid and img_url:
                        result[lid][iid] = img_url

        # Rate limit
        time.sleep(0.2)

    return result


def manager_scan() -> dict:
    """
    1. Check in-progress images (download_done=1) - if jpg exists, mark as 2
    2. Fetch new images (download_done=0) - get URLs, write markers, mark as 1
    """
    stats = {"completed": 0, "markers_created": 0, "api_fetched": 0, "skipped": 0}

    conn = get_connection()

    # --- Phase 1: Check in-progress images for completion ---
    in_progress = get_images_in_progress(conn)

    for img in in_progress:
        lid, iid = img["listing_id"], img["image_id"]
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

    # --- Phase 2: Fetch URLs for new images, create markers ---
    new_images = get_images_needing_download(conn, max_listings=100)

    if new_images:
        # Group by listing_id
        by_listing = defaultdict(list)
        for img in new_images:
            by_listing[img["listing_id"]].append(img["image_id"])

        # Fetch URLs from API
        listing_ids = list(by_listing.keys())
        url_data = fetch_listing_images(listing_ids)
        stats["api_fetched"] = len(url_data)

        # Step 1: Write ALL marker files first
        markers_written = []
        for lid, image_ids in by_listing.items():
            listing_urls = url_data.get(lid, {})

            for iid in image_ids:
                url = listing_urls.get(iid)
                marker_path = get_marker_path(lid, iid)

                if url:
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
    IMAGES_DIR.mkdir(exist_ok=True)

    if not ETSY_API_KEY:
        print("ERROR: ETSY_API_KEY not set in environment")
        sys.exit(1)

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
                      f"completed={stats['completed']} api={stats['api_fetched']} "
                      f"skipped={stats['skipped']}")

        time.sleep(1)

    # Wait for workers to finish
    print(f"[{ts()}] Waiting for workers to finish...")
    for t in threads:
        t.join(timeout=5)

    print(f"\n[{ts()}] Downloader stopped")


if __name__ == "__main__":
    main()
