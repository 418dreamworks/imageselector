#!/usr/bin/env python3
"""Rebuild image_status flags from images folder and image_index.json.

This script:
1. Sets download_done=1 for all images that exist on disk
2. Sets bg_removed=1 for images in embeddings/image_index.json
3. Sets embed_* flags for images in image_index.json
"""

import json
import os
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images" / "imagedownload"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
IMAGE_INDEX_FILE = EMBEDDINGS_DIR / "image_index.json"
DB_FILE = BASE_DIR / "data" / "db" / "etsy_data_recovered.db"

def main():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    # Step 1: Get all image files on disk
    print("Scanning images folder...")
    disk_images = set()
    count = 0
    for f in IMAGES_DIR.iterdir():
        if f.suffix == '.jpg':
            parts = f.stem.split('_')
            if len(parts) == 2:
                try:
                    lid, iid = int(parts[0]), int(parts[1])
                    disk_images.add((lid, iid))
                    count += 1
                    if count % 100000 == 0:
                        print(f"  Scanned {count} files...")
                except ValueError:
                    pass

    print(f"Found {len(disk_images)} valid image files on disk")

    # Step 2: Load image_index.json
    print("\nLoading image_index.json...")
    if IMAGE_INDEX_FILE.exists():
        with open(IMAGE_INDEX_FILE) as f:
            index = json.load(f)
        embedded_images = set(tuple(x) for x in index)
        print(f"Found {len(embedded_images)} embedded images")
    else:
        embedded_images = set()
        print("image_index.json not found")

    # Step 3: Get current image_status entries
    print("\nChecking current image_status...")
    cursor.execute("SELECT listing_id, image_id FROM image_status")
    db_images = set(cursor.fetchall())
    print(f"Found {len(db_images)} entries in image_status")

    # Step 4: Find images on disk not in DB
    missing_from_db = disk_images - db_images
    if missing_from_db:
        print(f"\n{len(missing_from_db)} images on disk but not in DB - adding them...")
        cursor.executemany(
            "INSERT OR IGNORE INTO image_status (listing_id, image_id) VALUES (?, ?)",
            list(missing_from_db)
        )
        conn.commit()

    # Step 5: Update download_done for all images on disk
    print("\nUpdating download_done flags...")
    # First reset all to 0
    cursor.execute("UPDATE image_status SET download_done = 0")

    # Then set to 1 for images that exist
    batch_size = 10000
    disk_list = list(disk_images)
    for i in range(0, len(disk_list), batch_size):
        batch = disk_list[i:i+batch_size]
        placeholders = ",".join(["(?,?)"] * len(batch))
        flat_batch = [x for pair in batch for x in pair]
        cursor.execute(f"""
            UPDATE image_status SET download_done = 1
            WHERE (listing_id, image_id) IN (VALUES {placeholders})
        """, flat_batch)
        if (i + batch_size) % 100000 == 0:
            print(f"  Updated {min(i + batch_size, len(disk_list))} images...")

    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 1")
    print(f"download_done=1: {cursor.fetchone()[0]}")

    # Step 6: Update bg_removed and embed flags for embedded images
    print("\nUpdating bg_removed and embed flags...")
    # Reset all embed flags to 0
    cursor.execute("""
        UPDATE image_status SET
            bg_removed = 0,
            embed_clip_vitb32 = 0,
            embed_clip_vitl14 = 0,
            embed_dinov2_base = 0,
            embed_dinov2_large = 0,
            embed_dinov3_base = 0
    """)

    # Set flags for embedded images
    embedded_list = list(embedded_images)
    for i in range(0, len(embedded_list), batch_size):
        batch = embedded_list[i:i+batch_size]
        placeholders = ",".join(["(?,?)"] * len(batch))
        flat_batch = [x for pair in batch for x in pair]
        cursor.execute(f"""
            UPDATE image_status SET
                bg_removed = 1,
                embed_clip_vitb32 = 1,
                embed_clip_vitl14 = 1,
                embed_dinov2_base = 1,
                embed_dinov2_large = 1,
                embed_dinov3_base = 1
            WHERE (listing_id, image_id) IN (VALUES {placeholders})
        """, flat_batch)
        if (i + batch_size) % 50000 == 0:
            print(f"  Updated {min(i + batch_size, len(embedded_list))} images...")

    conn.commit()

    # Step 7: Print summary
    print("\n=== Summary ===")
    cursor.execute("SELECT COUNT(*) FROM image_status")
    print(f"Total image_status rows: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 1")
    print(f"download_done=1: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE bg_removed = 1")
    print(f"bg_removed=1: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE embed_clip_vitb32 = 1")
    print(f"embed_clip_vitb32=1: {cursor.fetchone()[0]}")

    conn.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
