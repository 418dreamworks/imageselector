"""
Sync Etsy shop, listing, and review data to SQLite.

Collects metadata from shops/listings discovered by sync_images.py.
Uses static/dynamic table split to minimize storage:
- Static tables: written once with INSERT OR IGNORE
- Dynamic tables: only store changed fields (NULL for unchanged), forward-fill when querying

Usage:
    python sync_data.py                    # Sync all data
    python sync_data.py --top N            # Only sync top N shops by listing count
    python sync_data.py --test             # Use test database
"""

import os
import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import httpx

load_dotenv()

# Config
ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Paths
BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "etsy_data.db"
DB_FILE_TEST = BASE_DIR / "etsy_data_test.db"

# Rate limiting
API_DELAY = 0.011  # ~90 QPS


def ts():
    """Return timestamp string for logging."""
    return datetime.now().strftime("%H:%M:%S")


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with static/dynamic tables."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        -- Shops static table (written once per shop)
        CREATE TABLE IF NOT EXISTS shops (
            shop_id INTEGER PRIMARY KEY,
            create_date INTEGER,
            url TEXT,
            is_shop_us_based INTEGER,
            shipping_from_country_iso TEXT,
            shop_location_country_iso TEXT
        );

        -- Shops dynamic table (only changed fields per snapshot)
        CREATE TABLE IF NOT EXISTS shops_dynamic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shop_id INTEGER NOT NULL,
            snapshot_timestamp INTEGER NOT NULL,
            update_date INTEGER,
            listing_active_count INTEGER,
            accepts_custom_requests INTEGER,
            num_favorers INTEGER,
            transaction_sold_count INTEGER,
            review_average REAL,
            review_count INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_shops_dynamic_shop_id ON shops_dynamic(shop_id);
        CREATE INDEX IF NOT EXISTS idx_shops_dynamic_timestamp ON shops_dynamic(snapshot_timestamp);

        -- Listings static table (written once per listing)
        CREATE TABLE IF NOT EXISTS listings (
            listing_id INTEGER PRIMARY KEY,
            shop_id INTEGER NOT NULL,
            title TEXT,
            creation_timestamp INTEGER,
            url TEXT,
            is_customizable INTEGER,
            is_personalizable INTEGER,
            listing_type TEXT,
            tags TEXT,  -- JSON array
            materials TEXT,  -- JSON array
            processing_min INTEGER,
            processing_max INTEGER,
            who_made TEXT,
            when_made TEXT,
            item_weight REAL,
            item_weight_unit TEXT,
            item_length REAL,
            item_width REAL,
            item_height REAL,
            item_dimensions_unit TEXT,
            should_auto_renew INTEGER,
            language TEXT,
            price_amount INTEGER,
            price_divisor INTEGER,
            price_currency TEXT,
            taxonomy_id INTEGER,
            production_partners TEXT  -- JSON array
        );
        CREATE INDEX IF NOT EXISTS idx_listings_shop_id ON listings(shop_id);

        -- Listings dynamic table (only changed fields per snapshot)
        CREATE TABLE IF NOT EXISTS listings_dynamic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL,
            snapshot_timestamp INTEGER NOT NULL,
            state TEXT,
            ending_timestamp INTEGER,
            quantity INTEGER,
            num_favorers INTEGER,
            views INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_listings_dynamic_listing_id ON listings_dynamic(listing_id);
        CREATE INDEX IF NOT EXISTS idx_listings_dynamic_timestamp ON listings_dynamic(snapshot_timestamp);

        -- Reviews table (append-only, no snapshots needed)
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            shop_id INTEGER NOT NULL,
            listing_id INTEGER NOT NULL,
            buyer_user_id INTEGER,
            rating INTEGER,
            review TEXT,
            language TEXT,
            create_timestamp INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reviews_shop_id ON reviews(shop_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_listing_id ON reviews(listing_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_create_timestamp ON reviews(create_timestamp);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_unique ON reviews(shop_id, listing_id, create_timestamp);

        -- Sync state table
        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn


def get_sync_state(conn: sqlite3.Connection, key: str, default=None):
    """Get sync state value."""
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    return default


def set_sync_state(conn: sqlite3.Connection, key: str, value):
    """Set sync state value."""
    conn.execute(
        "INSERT OR REPLACE INTO sync_state (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )
    conn.commit()


def get_shop_ids_from_metadata() -> dict[int, int]:
    """Get shop_ids and their listing counts from image_metadata.json."""
    metadata_file = BASE_DIR / "image_metadata.json"
    if not metadata_file.exists():
        return {}

    with open(metadata_file) as f:
        metadata = json.load(f)

    shop_counts = {}
    for entry in metadata.values():
        if isinstance(entry, dict) and entry.get("shop_id"):
            shop_id = entry["shop_id"]
            shop_counts[shop_id] = shop_counts.get(shop_id, 0) + 1

    return shop_counts


def fetch_shop(client: httpx.Client, shop_id: int) -> dict | None:
    """Fetch shop data from API."""
    time.sleep(API_DELAY)
    try:
        response = client.get(
            f"{BASE_URL}/application/shops/{shop_id}",
            headers={"x-api-key": ETSY_API_KEY},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error fetching shop {shop_id}: {e}")
        return None


def fetch_shop_listings(client: httpx.Client, shop_id: int) -> list[dict]:
    """Fetch all active listings for a shop."""
    listings = []
    offset = 0
    batch_size = 100

    while True:
        time.sleep(API_DELAY)
        try:
            response = client.get(
                f"{BASE_URL}/application/shops/{shop_id}/listings/active",
                headers={"x-api-key": ETSY_API_KEY},
                params={"limit": batch_size, "offset": offset},
            )
            if response.status_code == 404:
                break
            response.raise_for_status()

            results = response.json().get("results", [])
            if not results:
                break

            listings.extend(results)
            offset += batch_size

            if offset >= 10000:  # API limit
                break
        except Exception as e:
            print(f"  Error fetching listings for shop {shop_id}: {e}")
            break

    return listings


def fetch_shop_reviews(client: httpx.Client, shop_id: int, last_timestamp: int = 0) -> list[dict]:
    """Fetch reviews for a shop, stopping at last_timestamp."""
    reviews = []
    offset = 0
    batch_size = 100

    while True:
        time.sleep(API_DELAY)
        try:
            response = client.get(
                f"{BASE_URL}/application/shops/{shop_id}/reviews",
                headers={"x-api-key": ETSY_API_KEY},
                params={"limit": batch_size, "offset": offset},
            )
            if response.status_code == 404:
                break
            response.raise_for_status()

            results = response.json().get("results", [])
            if not results:
                break

            found_old = False
            for review in results:
                if review.get("create_timestamp", 0) <= last_timestamp:
                    found_old = True
                    break
                reviews.append(review)

            if found_old:
                break

            offset += batch_size
            if offset >= 10000:
                break
        except Exception as e:
            print(f"  Error fetching reviews for shop {shop_id}: {e}")
            break

    return reviews


def get_last_shop_dynamic(conn: sqlite3.Connection, shop_id: int) -> dict | None:
    """Get the most recent dynamic data for a shop."""
    row = conn.execute("""
        SELECT update_date, listing_active_count, accepts_custom_requests,
               num_favorers, transaction_sold_count, review_average, review_count
        FROM shops_dynamic
        WHERE shop_id = ?
        ORDER BY snapshot_timestamp DESC
        LIMIT 1
    """, (shop_id,)).fetchone()
    if row:
        return dict(row)
    return None


def get_last_listing_dynamic(conn: sqlite3.Connection, listing_id: int) -> dict | None:
    """Get the most recent dynamic data for a listing."""
    row = conn.execute("""
        SELECT state, ending_timestamp, quantity, num_favorers, views
        FROM listings_dynamic
        WHERE listing_id = ?
        ORDER BY snapshot_timestamp DESC
        LIMIT 1
    """, (listing_id,)).fetchone()
    if row:
        return dict(row)
    return None


def insert_shop_static(conn: sqlite3.Connection, shop_data: dict):
    """Insert shop static data (ignore if exists)."""
    conn.execute("""
        INSERT OR IGNORE INTO shops (
            shop_id, create_date, url, is_shop_us_based,
            shipping_from_country_iso, shop_location_country_iso
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        shop_data.get("shop_id"),
        shop_data.get("create_date"),
        shop_data.get("url"),
        1 if shop_data.get("is_shop_us_based") else 0,
        shop_data.get("shipping_from_country_iso"),
        shop_data.get("shop_location_country_iso"),
    ))


def insert_shop_dynamic(conn: sqlite3.Connection, shop_data: dict, snapshot_timestamp: int):
    """Insert shop dynamic data, only storing changed fields."""
    shop_id = shop_data.get("shop_id")
    last = get_last_shop_dynamic(conn, shop_id)

    # Current values
    current = {
        "update_date": shop_data.get("update_date"),
        "listing_active_count": shop_data.get("listing_active_count"),
        "accepts_custom_requests": 1 if shop_data.get("accepts_custom_requests") else 0,
        "num_favorers": shop_data.get("num_favorers"),
        "transaction_sold_count": shop_data.get("transaction_sold_count"),
        "review_average": shop_data.get("review_average"),
        "review_count": shop_data.get("review_count"),
    }

    # If no previous data, write all; otherwise only write changed fields
    if last is None:
        values = current
    else:
        values = {}
        for key, val in current.items():
            if last.get(key) != val:
                values[key] = val
            else:
                values[key] = None  # unchanged, store NULL

    # Only insert if at least one field changed (or first time)
    if last is None or any(v is not None for v in values.values()):
        conn.execute("""
            INSERT INTO shops_dynamic (
                shop_id, snapshot_timestamp, update_date, listing_active_count,
                accepts_custom_requests, num_favorers, transaction_sold_count,
                review_average, review_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            shop_id,
            snapshot_timestamp,
            values.get("update_date"),
            values.get("listing_active_count"),
            values.get("accepts_custom_requests"),
            values.get("num_favorers"),
            values.get("transaction_sold_count"),
            values.get("review_average"),
            values.get("review_count"),
        ))


def insert_listing_static(conn: sqlite3.Connection, listing: dict):
    """Insert listing static data (ignore if exists)."""
    price = listing.get("price", {})
    conn.execute("""
        INSERT OR IGNORE INTO listings (
            listing_id, shop_id, title, creation_timestamp, url,
            is_customizable, is_personalizable, listing_type, tags, materials,
            processing_min, processing_max, who_made, when_made,
            item_weight, item_weight_unit, item_length, item_width,
            item_height, item_dimensions_unit, should_auto_renew, language,
            price_amount, price_divisor, price_currency, taxonomy_id,
            production_partners
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        listing.get("listing_id"),
        listing.get("shop_id"),
        listing.get("title"),
        listing.get("creation_timestamp"),
        listing.get("url"),
        1 if listing.get("is_customizable") else 0,
        1 if listing.get("is_personalizable") else 0,
        listing.get("listing_type"),
        json.dumps(listing.get("tags", [])),
        json.dumps(listing.get("materials", [])),
        listing.get("processing_min"),
        listing.get("processing_max"),
        listing.get("who_made"),
        listing.get("when_made"),
        listing.get("item_weight"),
        listing.get("item_weight_unit"),
        listing.get("item_length"),
        listing.get("item_width"),
        listing.get("item_height"),
        listing.get("item_dimensions_unit"),
        1 if listing.get("should_auto_renew") else 0,
        listing.get("language"),
        price.get("amount"),
        price.get("divisor"),
        price.get("currency_code"),
        listing.get("taxonomy_id"),
        json.dumps(listing.get("production_partners", [])),
    ))


def insert_listing_dynamic(conn: sqlite3.Connection, listing: dict, snapshot_timestamp: int):
    """Insert listing dynamic data, only storing changed fields."""
    listing_id = listing.get("listing_id")
    last = get_last_listing_dynamic(conn, listing_id)

    # Current values
    current = {
        "state": listing.get("state"),
        "ending_timestamp": listing.get("ending_timestamp"),
        "quantity": listing.get("quantity"),
        "num_favorers": listing.get("num_favorers"),
        "views": listing.get("views"),
    }

    # If no previous data, write all; otherwise only write changed fields
    if last is None:
        values = current
    else:
        values = {}
        for key, val in current.items():
            if last.get(key) != val:
                values[key] = val
            else:
                values[key] = None  # unchanged, store NULL

    # Only insert if at least one field changed (or first time)
    if last is None or any(v is not None for v in values.values()):
        conn.execute("""
            INSERT INTO listings_dynamic (
                listing_id, snapshot_timestamp, state, ending_timestamp,
                quantity, num_favorers, views
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            listing_id,
            snapshot_timestamp,
            values.get("state"),
            values.get("ending_timestamp"),
            values.get("quantity"),
            values.get("num_favorers"),
            values.get("views"),
        ))


def insert_review(conn: sqlite3.Connection, review: dict):
    """Insert review (skip if duplicate)."""
    try:
        conn.execute("""
            INSERT OR IGNORE INTO reviews (
                shop_id, listing_id, buyer_user_id, rating, review,
                language, create_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            review.get("shop_id"),
            review.get("listing_id"),
            review.get("buyer_user_id"),
            review.get("rating"),
            review.get("review"),
            review.get("language"),
            review.get("create_timestamp"),
        ))
    except sqlite3.IntegrityError:
        pass  # Duplicate, skip


def sync_data(top_n: int = 0, db_path: Path = DB_FILE):
    """Main sync function."""
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    snapshot_timestamp = int(time.time())

    # Get shop IDs from metadata
    shop_counts = get_shop_ids_from_metadata()
    if not shop_counts:
        print("No shops found in image_metadata.json")
        return

    # Sort by listing count and optionally limit
    sorted_shops = sorted(shop_counts.items(), key=lambda x: x[1], reverse=True)
    if top_n > 0:
        sorted_shops = sorted_shops[:top_n]
        print(f"Syncing top {top_n} shops by listing count")

    shop_ids = [shop_id for shop_id, _ in sorted_shops]
    print(f"Found {len(shop_ids)} shops to sync")

    # Initialize database
    conn = init_db(db_path)

    # Get last review timestamps
    last_review_timestamps = get_sync_state(conn, "last_review_timestamps", {})

    stats = {
        "shops_static": 0,
        "shops_dynamic": 0,
        "listings_static": 0,
        "listings_dynamic": 0,
        "reviews": 0,
        "api_calls": 0,
    }

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for i, shop_id in enumerate(shop_ids):
            print(f"\n[{ts()}] Shop {shop_id} ({i+1}/{len(shop_ids)})...")

            # Fetch shop data
            shop_data = fetch_shop(client, shop_id)
            stats["api_calls"] += 1

            if shop_data:
                # Check if this is a new shop (for static insert)
                existing = conn.execute(
                    "SELECT 1 FROM shops WHERE shop_id = ?", (shop_id,)
                ).fetchone()
                if not existing:
                    insert_shop_static(conn, shop_data)
                    stats["shops_static"] += 1

                insert_shop_dynamic(conn, shop_data, snapshot_timestamp)
                stats["shops_dynamic"] += 1

            # Fetch listings
            listings = fetch_shop_listings(client, shop_id)
            stats["api_calls"] += (len(listings) // 100) + 1

            for listing in listings:
                listing_id = listing.get("listing_id")
                # Check if this is a new listing (for static insert)
                existing = conn.execute(
                    "SELECT 1 FROM listings WHERE listing_id = ?", (listing_id,)
                ).fetchone()
                if not existing:
                    insert_listing_static(conn, listing)
                    stats["listings_static"] += 1

                insert_listing_dynamic(conn, listing, snapshot_timestamp)
                stats["listings_dynamic"] += 1

            # Fetch reviews (incremental)
            shop_id_str = str(shop_id)
            last_ts = last_review_timestamps.get(shop_id_str, 0)
            reviews = fetch_shop_reviews(client, shop_id, last_ts)
            stats["api_calls"] += 1

            if reviews:
                newest_ts = max(r.get("create_timestamp", 0) for r in reviews)
                last_review_timestamps[shop_id_str] = newest_ts

                for review in reviews:
                    insert_review(conn, review)
                    stats["reviews"] += 1

            print(f"  Listings: {len(listings)}, New reviews: {len(reviews)}")

            # Commit periodically
            if (i + 1) % 10 == 0:
                conn.commit()
                set_sync_state(conn, "last_review_timestamps", last_review_timestamps)

    # Final commit
    conn.commit()
    set_sync_state(conn, "last_review_timestamps", last_review_timestamps)
    set_sync_state(conn, "last_sync", snapshot_timestamp)
    conn.close()

    print("\n" + "=" * 50)
    print("Sync complete!")
    print(f"  New shops (static): {stats['shops_static']}")
    print(f"  Shop snapshots (dynamic): {stats['shops_dynamic']}")
    print(f"  New listings (static): {stats['listings_static']}")
    print(f"  Listing snapshots (dynamic): {stats['listings_dynamic']}")
    print(f"  Reviews: {stats['reviews']}")
    print(f"  API calls: {stats['api_calls']}")
    print(f"\nDatabase: {db_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync Etsy shop/listing/review data")
    parser.add_argument("--top", type=int, default=0, help="Only sync top N shops by listing count")
    parser.add_argument("--test", action="store_true", help="Use test database")
    args = parser.parse_args()

    db_path = DB_FILE_TEST if args.test else DB_FILE
    if args.test:
        print("*** TEST MODE ***\n")

    sync_data(top_n=args.top, db_path=db_path)
