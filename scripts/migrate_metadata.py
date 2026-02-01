#!/usr/bin/env python3
"""Migrate image_metadata.json from old format to new format.

Old format:
  {"listing_id": {"image_id": X, "shop_id": Y, "hex": "...", "suffix": "...", "when_made": "..."}}

New format:
  {"listing_id": {"shop_id": Y, "when_made": "...", "images": [{"image_id": X, "hex": "...", "suffix": "..."}]}}
"""
import json
from pathlib import Path

METADATA_FILE = Path(__file__).parent / "image_metadata.json"


def migrate():
    if not METADATA_FILE.exists():
        print("No metadata file found")
        return

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    converted = 0
    already_new = 0

    for lid_str, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        # Check if already new format (has "images" key)
        if "images" in entry:
            already_new += 1
            continue

        # Old format - convert it
        if "image_id" in entry:
            old_image_id = entry.pop("image_id")
            old_hex = entry.pop("hex", None)
            old_suffix = entry.pop("suffix", None)

            # Build images array (single image for old format)
            if old_image_id:
                entry["images"] = [{
                    "image_id": old_image_id,
                    "hex": old_hex,
                    "suffix": old_suffix,
                }]
            else:
                entry["images"] = []

            converted += 1
        else:
            # No image_id key - just add empty images array
            entry["images"] = []
            converted += 1

    print(f"Converted: {converted}")
    print(f"Already new format: {already_new}")
    print(f"Total entries: {len(metadata)}")

    # Save
    print(f"\nSaving to {METADATA_FILE}...")
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)
    print("Done!")


if __name__ == "__main__":
    migrate()
