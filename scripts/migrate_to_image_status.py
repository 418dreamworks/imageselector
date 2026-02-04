#!/usr/bin/env python3
"""Migrate image_metadata.json to image_status SQL table.

Run this ONCE on iMac before deploying the new code.

Data model:
- Listing-level (fails filter): listing_id, shop_id, when_made, price, to_download=0, image_id=0
- Image-level (passes filter): full image data with to_download=1

This way we track all listings, and can audit why each was filtered.
"""
import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_FILE = BASE_DIR / "etsy_data.db"
METADATA_FILE = BASE_DIR / "image_metadata.json"
IMAGES_DIR = BASE_DIR / "images"

# Filter criteria (must match sync_data.py)
ALLOWED_WHEN_MADE = {
    "made_to_order", "2020_2026", "2010_2019",
    "2000_2009", "2000_2006", "2007_2009",
}
MIN_PRICE = 50


def create_table(conn):
    """Create image_status table (drop if exists for clean migration)."""
    conn.execute("DROP TABLE IF EXISTS image_status")
    conn.execute("""
        CREATE TABLE image_status (
            listing_id INTEGER NOT NULL,
            image_id INTEGER NOT NULL,
            shop_id INTEGER,
            hex TEXT,
            suffix TEXT,
            is_primary INTEGER,
            when_made TEXT,
            price REAL,
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
    print("Created image_status table (dropped old if existed)")


def load_listing_prices(conn) -> dict:
    """Load prices from listings table."""
    prices = {}
    cursor = conn.execute("""
        SELECT listing_id, price_amount, price_divisor
        FROM listings
    """)
    for row in cursor.fetchall():
        lid = row[0]
        amount = row[1] or 0
        divisor = row[2] or 100
        prices[lid] = amount / divisor if divisor else 0
    print(f"Loaded {len(prices):,} listing prices from database")
    return prices


def migrate(conn, metadata: dict, listing_prices: dict):
    """Migrate metadata to SQL."""
    stats = {
        "images_inserted": 0,      # image-level rows (to_download=1)
        "listings_filtered": 0,    # listing-level rows (to_download=0)
        "skipped_no_url": 0,       # images without hex/suffix
        "skipped_empty": 0,        # listings with empty images array
        "download_done": 0,
        "bg_removed": 0,
    }

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

        # Get price from DB (authoritative), fallback to JSON
        if lid in listing_prices:
            price = listing_prices[lid]
        else:
            price = entry.get("price")

        # Listing-level bg_removed flag (fallback)
        listing_bg_removed = entry.get("background_removed", False)

        # Check filter
        passes_filter = (when_made in ALLOWED_WHEN_MADE and
                        price is not None and price >= MIN_PRICE)

        if not passes_filter:
            # Insert listing-level row with to_download=0, image_id=0
            conn.execute("""
                INSERT INTO image_status (
                    listing_id, image_id, shop_id, when_made, price, to_download
                ) VALUES (?, 0, ?, ?, ?, 0)
            """, (lid, shop_id, when_made, price))
            stats["listings_filtered"] += 1
            continue

        if not images:
            stats["skipped_empty"] += 1
            continue

        # Process images for listings that pass filter
        for idx, img in enumerate(images):
            image_id = img.get("image_id")
            if not image_id:
                continue

            hex_val = img.get("hex")
            suffix = img.get("suffix")

            # Skip if no URL data
            if not hex_val or not suffix:
                stats["skipped_no_url"] += 1
                continue

            is_primary = 1 if img.get("is_primary") or idx == 0 else 0

            # Check if file exists on disk
            download_done = 0
            img_path = IMAGES_DIR / f"{lid}_{image_id}.jpg"
            if img_path.exists() and img_path.stat().st_size > 1000:
                download_done = 1

            # bg_removed: per-image flag first, then listing-level fallback
            per_image_bg = img.get("bg_removed", None)
            if per_image_bg is not None:
                bg_removed = 1 if (download_done and per_image_bg) else 0
            else:
                bg_removed = 1 if (download_done and listing_bg_removed) else 0

            # Insert image-level row with to_download=1
            conn.execute("""
                INSERT INTO image_status (
                    listing_id, image_id, shop_id, hex, suffix,
                    is_primary, when_made, price, to_download, download_done, bg_removed,
                    embed_clip_vitb32, embed_clip_vitl14, embed_dinov2_base,
                    embed_dinov2_large, embed_dinov3_base
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 0, 0, 0, 0, 0)
            """, (lid, image_id, shop_id, hex_val, suffix,
                  is_primary, when_made, price, download_done, bg_removed))

            stats["images_inserted"] += 1
            if download_done:
                stats["download_done"] += 1
            if bg_removed:
                stats["bg_removed"] += 1

        total = stats["images_inserted"] + stats["listings_filtered"]
        if total % 50000 == 0 and total > 0:
            conn.commit()
            print(f"  Progress: {stats['images_inserted']:,} images, {stats['listings_filtered']:,} filtered listings...")

    conn.commit()
    return stats


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

    # Load prices from listings table
    listing_prices = load_listing_prices(conn)

    # Warning about dropping table
    try:
        count = conn.execute("SELECT COUNT(*) FROM image_status").fetchone()[0]
        if count > 0:
            print(f"\nWARNING: image_status has {count:,} rows - will be DROPPED")
            resp = input("Continue? (y/N): ")
            if resp.lower() != 'y':
                print("Aborted")
                return
    except sqlite3.OperationalError:
        pass  # Table doesn't exist yet

    create_table(conn)

    print("Migrating...")
    stats = migrate(conn, metadata, listing_prices)

    print(f"\nMigration complete:")
    print(f"  Images (to_download=1):       {stats['images_inserted']:,}")
    print(f"  Listings filtered (to_dl=0):  {stats['listings_filtered']:,}")
    print(f"  Skipped (no hex/suffix):      {stats['skipped_no_url']:,}")
    print(f"  Skipped (empty images):       {stats['skipped_empty']:,}")
    print(f"  download_done=1:              {stats['download_done']:,}")
    print(f"  bg_removed=1:                 {stats['bg_removed']:,}")
    print(f"\nFilter: when_made in {ALLOWED_WHEN_MADE}")
    print(f"        price >= {MIN_PRICE}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
