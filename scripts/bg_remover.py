#!/usr/bin/env python3
"""Remove backgrounds from images in-place.

Uses image_metadata.json as index (adds bg_removed flag to each image).
Can run continuously alongside sync_data.py.

Usage:
    python bg_remover.py              # Process all unprocessed images
    python bg_remover.py --watch      # Continuously watch for new images
    python bg_remover.py --gpu        # Use GPU acceleration (CoreML on Mac)
    python bg_remover.py --status     # Show processing status
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Paths - adjust for iMac vs local dev
BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
METADATA_FILE = BASE_DIR / "image_metadata.json"
KILL_FILE = BASE_DIR / "KILL_BG"

# If images dir doesn't exist, try dev/images (local dev)
if not IMAGES_DIR.exists():
    IMAGES_DIR = BASE_DIR / "dev" / "images"
    METADATA_FILE = BASE_DIR / "dev" / "image_metadata.json"

# Globals for rembg session
_session = None
_remove_func = None

# Metadata cache
_metadata = None
_metadata_dirty = False


def check_kill_file() -> bool:
    """Check if kill file exists. If so, remove it and return True."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        print("\nKill file detected. Stopping gracefully...")
        return True
    return False


def load_metadata(force_reload: bool = False, max_retries: int = 5) -> dict:
    """Load image metadata from JSON file.

    Retries on JSON decode errors (race condition with sync_data.py).
    """
    global _metadata
    if _metadata is None or force_reload:
        if METADATA_FILE.exists():
            for attempt in range(max_retries):
                try:
                    with open(METADATA_FILE) as f:
                        _metadata = json.load(f)
                    break
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        print(f"JSON decode error (attempt {attempt + 1}), retrying in 2s...")
                        time.sleep(2)
                    else:
                        print(f"Failed to load metadata after {max_retries} attempts: {e}")
                        _metadata = {}
        else:
            _metadata = {}
    return _metadata


def save_metadata():
    """Save metadata if dirty (atomic write to avoid corruption)."""
    global _metadata, _metadata_dirty
    if _metadata_dirty and _metadata:
        # Write to temp file, then rename (atomic on POSIX)
        tmp_file = METADATA_FILE.with_suffix('.json.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(_metadata, f)
        tmp_file.rename(METADATA_FILE)
        _metadata_dirty = False


def get_unprocessed_images(limit: int = 10000) -> list[tuple[str, int, Path]]:
    """Get list of (listing_id, image_id, path) for images not yet bg_removed.

    Args:
        limit: Maximum number of images to return (default 10000)
    """
    metadata = load_metadata()
    unprocessed = []

    for lid, entry in metadata.items():
        if not isinstance(entry, dict):
            continue
        # Skip listings already fully processed
        if entry.get("background_removed"):
            continue
        images = entry.get("images", [])
        for img in images:
            if img.get("bg_removed"):
                continue
            image_id = img.get("image_id")
            if image_id:
                path = IMAGES_DIR / f"{lid}_{image_id}.jpg"
                if path.exists() and path.stat().st_size > 1000:  # Skip placeholders
                    unprocessed.append((lid, image_id, path))
                    if len(unprocessed) >= limit:
                        return unprocessed

    return unprocessed


def mark_bg_removed(listing_id: str, image_id: int, success: bool):
    """Mark an image as bg_removed in metadata.

    Also sets listing-level 'background_removed: True' when all images are done.
    """
    global _metadata, _metadata_dirty
    metadata = load_metadata()

    if listing_id in metadata and isinstance(metadata[listing_id], dict):
        entry = metadata[listing_id]
        images = entry.get("images", [])
        for img in images:
            if img.get("image_id") == image_id:
                img["bg_removed"] = success
                _metadata_dirty = True
                break

        # Check if all images now have bg_removed
        if images and all(img.get("bg_removed") for img in images):
            entry["background_removed"] = True
            _metadata_dirty = True


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


def process_batch(items: list[tuple[str, int, Path]], use_gpu: bool = False, save_interval: int = 100):
    """Process a batch of images."""
    if not items:
        print("No images to process")
        return

    print(f"Processing {len(items)} images...")
    init_rembg(use_gpu=use_gpu)

    success = 0
    failed = 0

    for i, (lid, image_id, img_path) in enumerate(tqdm(items, desc="Removing backgrounds")):
        # Check for kill file
        if check_kill_file():
            save_metadata()
            print(f"Stopped: {success} success, {failed} failed")
            return

        ok = remove_background(img_path)
        mark_bg_removed(lid, image_id, ok)

        if ok:
            success += 1
        else:
            failed += 1

        # Save metadata periodically
        if (i + 1) % save_interval == 0:
            save_metadata()

    # Final save
    save_metadata()
    print(f"Done: {success} success, {failed} failed")


def watch_mode(use_gpu: bool = False, interval: int = 30, batch_size: int = 10000):
    """Continuously watch for new images and process them."""
    print(f"Watch mode: checking every {interval} seconds (batch size: {batch_size})...")
    print("Stop with: touch KILL_BG")

    init_rembg(use_gpu=use_gpu)

    while True:
        # Check for kill file
        if check_kill_file():
            save_metadata()
            return

        # Reload metadata to pick up new images from sync_data.py
        load_metadata(force_reload=True)

        items = get_unprocessed_images(limit=batch_size)
        if items:
            print(f"\nProcessing batch of {len(items)} images")
            for lid, image_id, img_path in tqdm(items, desc="Processing"):
                if check_kill_file():
                    save_metadata()
                    return
                ok = remove_background(img_path)
                mark_bg_removed(lid, image_id, ok)
            save_metadata()
        else:
            print(".", end="", flush=True)

        time.sleep(interval)


def show_status():
    """Show processing status."""
    metadata = load_metadata()

    total = 0
    processed = 0
    unprocessed = 0

    for lid, entry in metadata.items():
        if not isinstance(entry, dict):
            continue
        images = entry.get("images", [])
        for img in images:
            total += 1
            if img.get("bg_removed"):
                processed += 1
            else:
                unprocessed += 1

    print(f"Metadata file: {METADATA_FILE}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"")
    print(f"Total images in metadata:  {total:,}")
    print(f"Background removed:        {processed:,}")
    print(f"Not yet processed:         {unprocessed:,}")
    print(f"")
    if total > 0:
        print(f"Progress: {processed/total*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Remove backgrounds from images in-place")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration (CoreML)")
    parser.add_argument("--watch", action="store_true", help="Watch mode: continuously process new images")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    parser.add_argument("--interval", type=int, default=30, help="Watch interval in seconds (default: 30)")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    args = parser.parse_args()

    if not METADATA_FILE.exists():
        print(f"Metadata file not found: {METADATA_FILE}")
        return

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
