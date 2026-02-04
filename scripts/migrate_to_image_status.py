#!/usr/bin/env python3
"""Migrate image_metadata.json to image_status SQL table.

Run this ONCE on iMac before deploying the new code.
It reads the existing JSON and populates the SQL table with appropriate flags.
"""
import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_FILE = BASE_DIR / "etsy_data.db"
METADATA_FILE = BASE_DIR / "image_metadata.json"
IMAGES_DIR = BASE_DIR / "images"


def create_table(conn):
    """Create image_status table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_status (
            listing_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL,
            shop_id INTEGER,
            hex TEXT,
            suffix TEXT,
            is_primary INTEGER DEFAULT 0,
            when_made TEXT,
            to_download INTEGER DEFAULT 0,
            download_done INTEGER DEFAULT 0,
            bg_removed INTEGER DEFAULT 0,
            embed_clip_vitb32 INTEGER DEFAULT 0,
            embed_clip_vitl14 INTEGER DEFAULT 0,
            embed_dinov2_base INTEGER DEFAULT 0,
            embed_dinov2_large INTEGER DEFAULT 0,
            embed_dinov3_base INTEGER DEFAULT 0,
            PRIMARY KEY (listing_id, image_id)
        )
    """)
    conn.commit()
    print("Created image_status table")


def migrate(conn, metadata: dict):
    """Migrate metadata to SQL."""
    inserted = 0
    skipped = 0
    download_done = 0
    bg_removed_count = 0

    for lid_str, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        try:
            lid = int(lid_str)
        except ValueError:
            continue

        shop_id = entry.get("shop_id")
        when_made = entry.get("when_made", "")
        images = entry.get("images", [])
        was_bg_removed = entry.get("background_removed", False)

        for idx, img in enumerate(images):
            image_id = img.get("image_id")
            if not image_id:
                continue

            hex_val = img.get("hex", "")
            suffix = img.get("suffix", "")
            is_primary = 1 if img.get("is_primary") or idx == 0 else 0

            # Check if file exists on disk
            img_path = IMAGES_DIR / f"{lid}_{image_id}.jpg"
            file_exists = img_path.exists() and img_path.stat().st_size > 1000

            # Set flags based on current state
            to_download = 1
            done = 1 if file_exists else 0
            bg_removed = 1 if (file_exists and was_bg_removed) else 0

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO image_status (
                        listing_id, image_id, shop_id, hex, suffix,
                        is_primary, when_made, to_download, download_done, bg_removed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (lid, image_id, shop_id, hex_val, suffix,
                      is_primary, when_made, to_download, done, bg_removed))
                inserted += 1
                if done:
                    download_done += 1
                if bg_removed:
                    bg_removed_count += 1
            except sqlite3.IntegrityError:
                skipped += 1

        if inserted % 10000 == 0 and inserted > 0:
            conn.commit()
            print(f"  Progress: {inserted:,} inserted...")

    conn.commit()
    return inserted, skipped, download_done, bg_removed_count


def main():
    if not METADATA_FILE.exists():
        print(f"Error: {METADATA_FILE} not found")
        return

    print(f"Loading {METADATA_FILE}...")
    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata):,} entries")

    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL")

    # Check if table already has data
    try:
        count = conn.execute("SELECT COUNT(*) FROM image_status").fetchone()[0]
        if count > 0:
            print(f"image_status already has {count:,} rows")
            resp = input("Continue anyway? (y/N): ")
            if resp.lower() != 'y':
                print("Aborted")
                return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet

    create_table(conn)

    print("Migrating...")
    inserted, skipped, download_done, bg_removed = migrate(conn, metadata)

    print(f"\nMigration complete:")
    print(f"  Inserted: {inserted:,}")
    print(f"  Skipped (duplicate): {skipped:,}")
    print(f"  download_done=1: {download_done:,}")
    print(f"  bg_removed=1: {bg_removed:,}")

    conn.close()
    print("\nDone! You can now deploy the new code.")


if __name__ == "__main__":
    main()
