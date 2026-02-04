#!/usr/bin/env python3
"""Remove backgrounds from images in-place.

Uses SQLite image_status table for atomic flag updates.
Can run continuously alongside other scripts.

Usage:
    python bg_remover.py              # Process all unprocessed images
    python bg_remover.py --watch      # Continuously watch for new images
    python bg_remover.py --gpu        # Use GPU acceleration (CoreML on Mac)
    python bg_remover.py --status     # Show processing status
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import time
import sys
from pathlib import Path

# Paths - adjust for iMac vs local dev
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
KILL_FILE = BASE_DIR / "KILL_BG"

# If images dir doesn't exist, try dev/images (local dev)
if not IMAGES_DIR.exists():
    IMAGES_DIR = BASE_DIR / "dev" / "images"

# Import from shared image_db module
sys.path.insert(0, str(BASE_DIR))
from image_db import (
    get_connection, get_images_for_bg_removal,
    mark_bg_removed as db_mark_bg_removed, get_stats,
    commit_with_retry
)

# Globals for rembg session
_session = None
_remove_func = None


def check_kill_file() -> bool:
    """Check if kill file exists. If so, remove it and return True."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        print("\nKill file detected. Stopping gracefully...")
        return True
    return False


def get_unprocessed_images(limit: int = 10000) -> list[tuple[int, int, Path]]:
    """Get list of (listing_id, image_id, path) for images not yet bg_removed."""
    conn = get_connection()
    images = get_images_for_bg_removal(conn, limit=limit)
    conn.close()

    result = []
    for img in images:
        lid = img["listing_id"]
        iid = img["image_id"]
        path = IMAGES_DIR / f"{lid}_{iid}.jpg"
        # Only include if file exists and has content (not placeholder)
        if path.exists() and path.stat().st_size > 1000:
            result.append((lid, iid, path))

    return result


def mark_bg_removed(listing_id: int, image_id: int, success: bool):
    """Mark an image as bg_removed in database."""
    if success:
        conn = get_connection()
        db_mark_bg_removed(conn, listing_id, image_id)
        commit_with_retry(conn)
        conn.close()


def init_rembg(use_gpu: bool = False):
    """Initialize rembg with optional GPU acceleration."""
    global _session, _remove_func
    from rembg import remove, new_session

    if use_gpu:
        try:
            _session = new_session("u2net", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
            print("Using CoreML GPU acceleration")
        except Exception as e:
            print(f"CoreML not available ({e}), falling back to CPU")
            _session = new_session("u2net")
    else:
        _session = new_session("u2net")

    _remove_func = remove


def remove_background(img_path: Path) -> bool:
    """Remove background from image and overwrite in place."""
    global _session, _remove_func
    from PIL import Image

    try:
        img = Image.open(img_path).convert("RGB")

        # Remove background (returns RGBA with transparent bg)
        img_nobg = _remove_func(img, session=_session)

        # Convert to RGB with white background
        rgb_img = Image.new("RGB", img_nobg.size, (255, 255, 255))
        rgb_img.paste(img_nobg, mask=img_nobg.split()[3])

        # Overwrite in place
        rgb_img.save(img_path, "JPEG", quality=90)
        return True

    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")
        return False


def process_batch(items: list[tuple[int, int, Path]], use_gpu: bool = False):
    """Process a batch of images."""
    from tqdm import tqdm

    if not items:
        print("No images to process")
        return

    print(f"Processing {len(items)} images...")
    init_rembg(use_gpu=use_gpu)

    success = 0
    failed = 0

    for lid, image_id, img_path in tqdm(items, desc="Removing backgrounds"):
        # Check for kill file
        if check_kill_file():
            print(f"Stopped: {success} success, {failed} failed")
            return

        ok = remove_background(img_path)
        mark_bg_removed(lid, image_id, ok)

        if ok:
            success += 1
        else:
            failed += 1

    print(f"Done: {success} success, {failed} failed")


def watch_mode(use_gpu: bool = False, interval: int = 30, batch_size: int = 10000):
    """Continuously watch for new images and process them."""
    from tqdm import tqdm

    print(f"Watch mode: checking every {interval} seconds (batch size: {batch_size})...")
    print("Stop with: touch KILL_BG")

    init_rembg(use_gpu=use_gpu)

    while True:
        # Check for kill file
        if check_kill_file():
            return

        items = get_unprocessed_images(limit=batch_size)
        if items:
            print(f"\nProcessing batch of {len(items)} images")
            for lid, image_id, img_path in tqdm(items, desc="Processing"):
                if check_kill_file():
                    return
                ok = remove_background(img_path)
                mark_bg_removed(lid, image_id, ok)
        else:
            print(".", end="", flush=True)

        time.sleep(interval)


def show_status():
    """Show processing status from SQL."""
    conn = get_connection()
    stats = get_stats(conn)
    conn.close()

    print(f"Images directory: {IMAGES_DIR}")
    print(f"")
    print(f"Total images:          {stats['total']:,}")
    print(f"Downloaded:            {stats['download_done']:,}")
    print(f"Background removed:    {stats['bg_removed']:,}")
    print(f"")
    pending = stats['download_done'] - stats['bg_removed']
    print(f"Pending BG removal:    {pending:,}")
    if stats['download_done'] > 0:
        print(f"Progress: {stats['bg_removed']/stats['download_done']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Remove backgrounds from images in-place")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration (CoreML)")
    parser.add_argument("--watch", action="store_true", help="Watch mode: continuously process new images")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds (default: 30)")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.watch:
        watch_mode(use_gpu=args.gpu, interval=args.interval)
    else:
        items = get_unprocessed_images()
        if args.limit:
            items = items[:args.limit]
        process_batch(items, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
