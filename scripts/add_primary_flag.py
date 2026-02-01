#!/usr/bin/env python3
"""Add is_primary flag to first image in each listing's images array.

This migration adds is_primary: True to the first image in each listing,
making it explicit which image is the main/primary image for the listing.
"""
import json
from pathlib import Path

METADATA_FILE = Path(__file__).parent.parent / "image_metadata.json"


def migrate():
    if not METADATA_FILE.exists():
        print("No metadata file found")
        return

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    updated = 0
    already_has_flag = 0
    no_images = 0

    for lid_str, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        images = entry.get("images", [])
        if not images:
            no_images += 1
            continue

        # Check if first image already has is_primary
        if images[0].get("is_primary"):
            already_has_flag += 1
            continue

        # Add is_primary to first image
        images[0]["is_primary"] = True
        updated += 1

    print(f"Updated: {updated}")
    print(f"Already had flag: {already_has_flag}")
    print(f"No images: {no_images}")
    print(f"Total entries: {len(metadata)}")

    if updated > 0:
        print(f"\nSaving to {METADATA_FILE}...")
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f)
        print("Done!")
    else:
        print("\nNo changes needed.")


if __name__ == "__main__":
    migrate()
