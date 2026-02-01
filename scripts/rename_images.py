#!/usr/bin/env python3
"""Rename images from old format to new format.

Old format: {listing_id}.jpg
New format: {listing_id}_{image_id}.jpg

Uses image_id from metadata's images[0] (primary image).
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
METADATA_FILE = BASE_DIR / "image_metadata.json"
IMAGES_DIR = BASE_DIR / "images"


def rename_images():
    if not METADATA_FILE.exists():
        print("No metadata file found")
        return

    if not IMAGES_DIR.exists():
        print("No images directory found")
        return

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    renamed = 0
    already_new = 0
    no_images = 0
    missing = 0
    errors = 0

    for lid_str, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        images = entry.get("images", [])
        if not images:
            no_images += 1
            continue

        # Get primary image_id
        image_id = images[0].get("image_id")
        if not image_id:
            no_images += 1
            continue

        old_path = IMAGES_DIR / f"{lid_str}.jpg"
        new_path = IMAGES_DIR / f"{lid_str}_{image_id}.jpg"

        # Check if already in new format
        if new_path.exists():
            already_new += 1
            continue

        # Check if old format exists
        if not old_path.exists():
            missing += 1
            continue

        # Rename
        try:
            old_path.rename(new_path)
            renamed += 1
        except Exception as e:
            print(f"Error renaming {old_path}: {e}")
            errors += 1

        if renamed % 10000 == 0 and renamed > 0:
            print(f"Progress: {renamed} renamed...")

    print(f"\nRenamed: {renamed}")
    print(f"Already new format: {already_new}")
    print(f"No images in metadata: {no_images}")
    print(f"Missing files: {missing}")
    print(f"Errors: {errors}")
    print(f"Total metadata entries: {len(metadata)}")


if __name__ == "__main__":
    rename_images()
