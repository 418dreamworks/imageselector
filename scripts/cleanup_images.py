#!/usr/bin/env python3
"""Cleanup non-primary images after embedding is complete.

For each listing where all images have all 5 embedded_* flags:
- Keep the primary image (is_primary: true)
- Move all non-primary images to backup location (SSD or HDD)
- Update metadata to remove moved images
"""
import argparse
import json
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
IMAGES_NOBG_DIR = BASE_DIR / "images_nobg"
METADATA_FILE = BASE_DIR / "image_metadata.json"

# Backup location for embedded images (change to HDD once verified)
# SSD: /Volumes/SSD_120/embedded_backup
# HDD: /Volumes/HDD_1000/embedded_backup (after disk check passes)
BACKUP_DIR = Path("/Volumes/SSD_120/embedded_backup")

# All embedding models that must be complete before cleanup
EMBED_MODELS = [
    "clip_vitb32",
    "clip_vitl14",
    "dinov2_base",
    "dinov2_large",
    "dinov3_base",
]


def load_metadata() -> dict:
    """Load metadata from file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict):
    """Save metadata to file."""
    tmp_file = METADATA_FILE.with_suffix('.json.tmp')
    with open(tmp_file, 'w') as f:
        json.dump(metadata, f)
    tmp_file.rename(METADATA_FILE)


def is_fully_embedded(img: dict) -> bool:
    """Check if an image has all embedding flags set."""
    return all(img.get(f"embedded_{model}") for model in EMBED_MODELS)


def find_cleanable_listings(metadata: dict) -> list[str]:
    """Find listings where all images are fully embedded."""
    cleanable = []

    for lid, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        images = entry.get("images", [])
        if not images:
            continue

        # All images must be fully embedded
        if all(is_fully_embedded(img) for img in images):
            # Must have at least one non-primary image to clean
            non_primary = [img for img in images if not img.get("is_primary")]
            if non_primary:
                cleanable.append(lid)

    return cleanable


def cleanup_listing(lid: str, entry: dict, dry_run: bool = True) -> int:
    """Move non-primary images to backup location.

    Returns count of moved files.
    """
    images = entry.get("images", [])
    moved = 0

    # Find primary and non-primary images
    primary_img = None
    to_move = []

    for img in images:
        if img.get("is_primary"):
            primary_img = img
        else:
            to_move.append(img)

    if not primary_img:
        print(f"  Warning: No primary image for {lid}")
        return 0

    # Create backup subdirs
    backup_images = BACKUP_DIR / "images"
    backup_nobg = BACKUP_DIR / "images_nobg"

    if not dry_run:
        backup_images.mkdir(parents=True, exist_ok=True)
        backup_nobg.mkdir(parents=True, exist_ok=True)

    # Move non-primary image files
    for img in to_move:
        iid = img.get("image_id")
        if not iid:
            continue

        # Original image
        orig_path = IMAGES_DIR / f"{lid}_{iid}.jpg"
        if orig_path.exists():
            dest_path = backup_images / f"{lid}_{iid}.jpg"
            if dry_run:
                print(f"  Would move: {orig_path.name} -> {dest_path}")
            else:
                shutil.move(str(orig_path), str(dest_path))
                print(f"  Moved: {orig_path.name}")
            moved += 1

        # Nobg image
        nobg_path = IMAGES_NOBG_DIR / f"{lid}_{iid}.png"
        if nobg_path.exists():
            dest_path = backup_nobg / f"{lid}_{iid}.png"
            if dry_run:
                print(f"  Would move: {nobg_path.name} -> {dest_path}")
            else:
                shutil.move(str(nobg_path), str(dest_path))
                print(f"  Moved: {nobg_path.name}")
            moved += 1

    # Update metadata to keep only primary
    if not dry_run:
        entry["images"] = [primary_img]

    return moved


def main():
    parser = argparse.ArgumentParser(description="Cleanup non-primary images")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of listings to process (for testing)",
    )
    args = parser.parse_args()

    print("Loading metadata...")
    metadata = load_metadata()
    print(f"Total listings: {len(metadata)}")

    print("\nFinding cleanable listings...")
    cleanable = find_cleanable_listings(metadata)
    print(f"Listings ready for cleanup: {len(cleanable)}")

    if not cleanable:
        print("Nothing to clean up.")
        return

    if args.limit > 0:
        cleanable = cleanable[:args.limit]
        print(f"Limited to {len(cleanable)} listings")

    total_deleted = 0
    for lid in cleanable:
        print(f"\nProcessing listing {lid}:")
        entry = metadata[lid]
        deleted = cleanup_listing(lid, entry, dry_run=args.dry_run)
        total_deleted += deleted

    if not args.dry_run and total_deleted > 0:
        print("\nSaving updated metadata...")
        save_metadata(metadata)

    print(f"\n{'Would move' if args.dry_run else 'Moved'}: {total_deleted} files")
    print(f"Backup location: {BACKUP_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
