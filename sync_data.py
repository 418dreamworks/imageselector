"""
sync_data.py — Etsy furniture data sync (API only).

Crawls taxonomy categories and populates SQLite database.
Does NOT download images - that's handled by image_downloader.py.

Usage:
    python sync_data.py
"""

import os
import sys
import json
import time
import signal
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
import httpx

load_dotenv()

# Import from shared image_db module
from image_db import get_connection, insert_image, _retry_on_lock, commit_with_retry

# ─── Config ─────────────────────────────────────────────────────────────────

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

API_DELAY_DEFAULT = 0.2  # 5 QPS (default, overridden by qps_config.json)
MAX_OFFSET = 10000     # Etsy API offset limit
ONE_WEEK = 7 * 24 * 3600
ONE_MONTH = 30 * 24 * 3600

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
DB_FILE = BASE_DIR / "etsy_data.db"
TAXONOMY_CONFIG_FILE = BASE_DIR / "furniture_taxonomy_config.json"
KILL_FILE = BASE_DIR / "KILL"
QPS_CONFIG_FILE = BASE_DIR / "qps_config.json"
PID_FILE = BASE_DIR / "sync_data.pid"


def get_api_delay() -> float:
    """Get API delay from config file (set by qps_monitor.py)."""
    if QPS_CONFIG_FILE.exists():
        try:
            with open(QPS_CONFIG_FILE) as f:
                config = json.load(f)
                return config.get("api_delay", API_DELAY_DEFAULT)
        except (json.JSONDecodeError, IOError):
            pass
    return API_DELAY_DEFAULT

FURNITURE_TAXONOMY_IDS = {
    967, 968, 969, 12455, 12456, 970, 972, 971, 12470, 973, 974, 975, 976, 977,
    11837, 978, 979, 12403, 12405, 12406, 980, 981, 982, 983, 985, 986, 987, 988,
    989, 990, 12369, 12370, 991, 992, 993, 12371, 12372, 994, 11355, 11356, 998,
    996, 12468, 12216, 997, 999, 1000, 1001, 12408
}


# ─── Utilities ───────────────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%H:%M:%S")


def check_kill_file():
    """Check for kill file. Returns True if should exit. User must manually delete KILL file."""
    try:
        if KILL_FILE.exists():
            print(f"[{ts()}] Kill file detected at {KILL_FILE}. Shutting down...")
            return True
    except Exception as e:
        print(f"[{ts()}] Error checking kill file: {e}")
    return False


def acquire_lock() -> bool:
    """Acquire PID lock. Returns False if another instance is running."""
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            # Check if process is still running
            os.kill(old_pid, 0)
            # Process exists - another instance is running
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            # PID file is stale or process doesn't exist
            pass
    PID_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    """Release PID lock."""
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


# ─── Taxonomy Config ─────────────────────────────────────────────────────────

def load_taxonomy_config():
    with open(TAXONOMY_CONFIG_FILE) as f:
        config = json.load(f)

    crawl_units = []
    for entry in config["taxonomies"]:
        tax_id = entry["id"]
        name = entry["name"]
        breaks = entry["price_breaks"]

        for i in range(len(breaks) - 1):
            min_price = breaks[i] if i == 0 else breaks[i] + 0.01
            max_price = breaks[i + 1]

            # No price filtering - crawl all price ranges
            # (MIN_PRICE filter only applies to image downloads, not DB)

            if len(breaks) == 2:
                label = name
            else:
                label = f"{name} (${breaks[i]}-${breaks[i+1]})"

            crawl_units.append({
                "taxonomy_id": tax_id,
                "name": label,
                "min_price": min_price,
                "max_price": max_price if max_price < 1000000 else None,
            })

    return crawl_units


CRAWL_UNITS = load_taxonomy_config()


# ─── Database ────────────────────────────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS shops_static (
            shop_id INTEGER PRIMARY KEY,
            snapshot_timestamp INTEGER,
            create_date INTEGER,
            url TEXT,
            is_shop_us_based INTEGER,
            shipping_from_country_iso TEXT,
            shop_location_country_iso TEXT
        );

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

        CREATE TABLE IF NOT EXISTS listings_static (
            listing_id INTEGER PRIMARY KEY,
            snapshot_timestamp INTEGER,
            shop_id INTEGER NOT NULL,
            title TEXT,
            description TEXT,
            creation_timestamp INTEGER,
            url TEXT,
            is_customizable INTEGER,
            is_personalizable INTEGER,
            listing_type TEXT,
            tags TEXT,
            materials TEXT,
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
            taxonomy_id INTEGER,
            production_partners TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_listings_static_shop_id ON listings_static(shop_id);

        CREATE TABLE IF NOT EXISTS listings_dynamic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id INTEGER NOT NULL,
            snapshot_timestamp INTEGER NOT NULL,
            state TEXT,
            ending_timestamp INTEGER,
            quantity INTEGER,
            num_favorers INTEGER,
            views INTEGER,
            price_amount INTEGER,
            price_divisor INTEGER,
            price_currency TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_listings_dynamic_listing_id ON listings_dynamic(listing_id);
        CREATE INDEX IF NOT EXISTS idx_listings_dynamic_timestamp ON listings_dynamic(snapshot_timestamp);

        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_timestamp INTEGER,
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

        CREATE TABLE IF NOT EXISTS image_status (
            listing_id INTEGER,
            image_id INTEGER,
            is_primary INTEGER,
            url TEXT,
            download_done INTEGER DEFAULT 0,
            faiss_row INTEGER,
            PRIMARY KEY (listing_id, image_id)
        );

        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    commit_with_retry(conn)
    return conn


def get_sync_state(conn, key, default=None):
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    if row:
        return json.loads(row[0])
    return default


def set_sync_state(conn, key, value):
    conn.execute(
        "INSERT OR REPLACE INTO sync_state (key, value) VALUES (?, ?)",
        (key, json.dumps(value))
    )
    commit_with_retry(conn)


# ─── DB Insert Functions ─────────────────────────────────────────────────────

@_retry_on_lock
def insert_shop_static(conn, shop_data, snapshot_ts):
    conn.execute("""
        INSERT OR IGNORE INTO shops_static (
            shop_id, snapshot_timestamp, create_date, url, is_shop_us_based,
            shipping_from_country_iso, shop_location_country_iso
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        shop_data.get("shop_id"),
        snapshot_ts,
        shop_data.get("create_date"),
        shop_data.get("url"),
        1 if shop_data.get("is_shop_us_based") else 0,
        shop_data.get("shipping_from_country_iso"),
        shop_data.get("shop_location_country_iso"),
    ))


@_retry_on_lock
def insert_shop_dynamic(conn, shop_data, snapshot_ts):
    conn.execute("""
        INSERT INTO shops_dynamic (
            shop_id, snapshot_timestamp, update_date, listing_active_count,
            accepts_custom_requests, num_favorers, transaction_sold_count,
            review_average, review_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        shop_data.get("shop_id"),
        snapshot_ts,
        shop_data.get("update_date"),
        shop_data.get("listing_active_count"),
        1 if shop_data.get("accepts_custom_requests") else 0,
        shop_data.get("num_favorers"),
        shop_data.get("transaction_sold_count"),
        shop_data.get("review_average"),
        shop_data.get("review_count"),
    ))


@_retry_on_lock
def insert_listing_static(conn, listing, snapshot_ts):
    conn.execute("""
        INSERT OR IGNORE INTO listings_static (
            listing_id, snapshot_timestamp, shop_id, title, description,
            creation_timestamp, url,
            is_customizable, is_personalizable, listing_type, tags, materials,
            processing_min, processing_max, who_made, when_made,
            item_weight, item_weight_unit, item_length, item_width,
            item_height, item_dimensions_unit, should_auto_renew, language,
            taxonomy_id, production_partners
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        listing.get("listing_id"),
        snapshot_ts,
        listing.get("shop_id"),
        listing.get("title"),
        listing.get("description"),
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
        listing.get("taxonomy_id"),
        json.dumps(listing.get("production_partners", [])),
    ))


@_retry_on_lock
def insert_listing_dynamic(conn, listing, snapshot_ts):
    price = listing.get("price", {})
    conn.execute("""
        INSERT INTO listings_dynamic (
            listing_id, snapshot_timestamp, state, ending_timestamp,
            quantity, num_favorers, views, price_amount, price_divisor, price_currency
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        listing.get("listing_id"),
        snapshot_ts,
        listing.get("state"),
        listing.get("ending_timestamp"),
        listing.get("quantity"),
        listing.get("num_favorers"),
        listing.get("views"),
        price.get("amount"),
        price.get("divisor"),
        price.get("currency_code"),
    ))


@_retry_on_lock
def insert_review(conn, review, snapshot_ts):
    try:
        conn.execute("""
            INSERT OR IGNORE INTO reviews (
                snapshot_timestamp, shop_id, listing_id, buyer_user_id, rating, review,
                language, create_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot_ts,
            review.get("shop_id"),
            review.get("listing_id"),
            review.get("buyer_user_id"),
            review.get("rating"),
            review.get("review"),
            review.get("language"),
            review.get("create_timestamp"),
        ))
    except sqlite3.IntegrityError:
        pass


# ─── API Functions ───────────────────────────────────────────────────────────

_api_stats = {"used": 0, "limit": 10000, "remaining": 10000}


def update_api_usage(response):
    try:
        remaining = int(response.headers.get("x-remaining-today", 0))
        limit = int(response.headers.get("x-limit-per-day", 10000))
        _api_stats["remaining"] = remaining
        _api_stats["limit"] = limit
        _api_stats["used"] = limit - remaining
    except (ValueError, TypeError):
        pass


def fetch_active_listings(client, taxonomy_id, offset, min_price=None, max_price=None):
    """Fetch a batch of active listings for a taxonomy + price range, including images."""
    params = {"taxonomy_id": taxonomy_id, "limit": 100, "offset": offset, "includes": "Images"}
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price

    max_retries = 3
    for attempt in range(max_retries):
        time.sleep(get_api_delay())
        try:
            response = client.get(
                f"{BASE_URL}/application/listings/active",
                headers={"x-api-key": ETSY_API_KEY},
                params=params,
            )
            update_api_usage(response)

            if response.status_code == 429:
                print(f"[{ts()}] 429 Rate Limited!")
                return None, "rate_limited"
            if response.status_code == 400:
                return None, "bad_request"
            response.raise_for_status()

            data = response.json()
            return data.get("results", []), data.get("count", 0)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[{ts()}] Timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[{ts()}] Timeout after {max_retries} attempts, skipping batch")
                return [], 0  # Return empty to skip this batch


def fetch_shop(client, shop_id):
    """Fetch full shop data from API."""
    max_retries = 3
    for attempt in range(max_retries):
        time.sleep(get_api_delay())
        try:
            response = client.get(
                f"{BASE_URL}/application/shops/{shop_id}",
                headers={"x-api-key": ETSY_API_KEY},
            )
            update_api_usage(response)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[{ts()}] Shop {shop_id} timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[{ts()}] Shop {shop_id} timeout after {max_retries} attempts, skipping")
                return None
        except Exception as e:
            print(f"  Error fetching shop {shop_id}: {e}")
            return None


def fetch_listings_batch_with_images(client, listing_ids):
    """Fetch listings with images via batch endpoint.

    Returns dict mapping listing_id -> list of image dicts with url_570xN.
    """
    if not listing_ids:
        return {}

    max_retries = 3
    for attempt in range(max_retries):
        time.sleep(get_api_delay())
        try:
            response = client.get(
                f"{BASE_URL}/application/listings/batch",
                headers={"x-api-key": ETSY_API_KEY},
                params={
                    "listing_ids": ",".join(str(lid) for lid in listing_ids),
                    "includes": "Images"
                },
            )
            update_api_usage(response)
            response.raise_for_status()

            result = {}
            for listing in response.json().get("results", []):
                lid = listing.get("listing_id")
                images = listing.get("images", [])
                if lid and images:
                    result[lid] = images
            return result
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[{ts()}] Timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[{ts()}] Timeout after {max_retries} attempts, skipping batch")
                return {}
        except Exception as e:
            print(f"  Error fetching batch images: {e}")
            return {}


def fetch_shop_reviews(client, shop_id, last_timestamp=0):
    """Fetch reviews for a shop newer than last_timestamp."""
    reviews = []
    offset = 0
    max_retries = 3

    while True:
        success = False
        for attempt in range(max_retries):
            time.sleep(get_api_delay())
            try:
                response = client.get(
                    f"{BASE_URL}/application/shops/{shop_id}/reviews",
                    headers={"x-api-key": ETSY_API_KEY},
                    params={"limit": 100, "offset": offset},
                )
                update_api_usage(response)
                if response.status_code == 404:
                    return reviews
                response.raise_for_status()

                results = response.json().get("results", [])
                if not results:
                    return reviews

                found_old = False
                for review in results:
                    if review.get("create_timestamp", 0) <= last_timestamp:
                        found_old = True
                        break
                    reviews.append(review)

                if found_old:
                    return reviews

                offset += 100
                if offset >= 10000:
                    return reviews
                success = True
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"[{ts()}] Reviews {shop_id} timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[{ts()}] Reviews {shop_id} timeout after {max_retries} attempts, returning partial")
                    return reviews
            except Exception as e:
                print(f"  Error fetching reviews for shop {shop_id}: {e}")
                return reviews

        if not success:
            break

    return reviews


# ─── Image SQL Insert ────────────────────────────────────────────────────────

def add_images_to_sql(listing_id: int, images: list, conn=None):
    """Upsert images to SQL image_status table.

    Args:
        listing_id: The listing ID
        images: List of image dicts from API (with listing_image_id, url_570xN, rank)
        conn: Optional existing database connection (to avoid lock conflicts)

    For existing rows: updates is_primary always, url only if missing.
    For new rows: inserts with download_done=0.
    Preserves download_done and faiss_row for existing rows.
    """
    if not images:
        return

    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    for img in images:
        image_id = img.get("listing_image_id")
        url = img.get("url_570xN", "")
        rank = img.get("rank", 1)
        is_primary = 1 if rank == 1 else 0

        if not image_id:
            continue

        # Check if exists
        cursor = conn.execute("""
            SELECT url FROM image_status WHERE listing_id = ? AND image_id = ?
        """, (listing_id, image_id))
        row = cursor.fetchone()

        if row:
            # Exists - update is_primary always, url only if missing
            existing_url = row[0]
            if not existing_url:
                conn.execute("""
                    UPDATE image_status SET is_primary = ?, url = ?
                    WHERE listing_id = ? AND image_id = ?
                """, (is_primary, url, listing_id, image_id))
            else:
                conn.execute("""
                    UPDATE image_status SET is_primary = ?
                    WHERE listing_id = ? AND image_id = ?
                """, (is_primary, listing_id, image_id))
        else:
            # New - insert
            conn.execute("""
                INSERT INTO image_status (listing_id, image_id, is_primary, url, download_done)
                VALUES (?, ?, ?, ?, 0)
            """, (listing_id, image_id, is_primary, url))

    if own_conn:
        commit_with_retry(conn)
        conn.close()


# ─── Progress ────────────────────────────────────────────────────────────────

def _ensure_progress_table(conn):
    """Create sync_progress table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_progress (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            data TEXT NOT NULL
        )
    """)
    conn.commit()


def load_progress() -> dict:
    conn = get_connection()
    _ensure_progress_table(conn)
    cursor = conn.execute("SELECT data FROM sync_progress WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {
        "crawl_unit_index": 0,
        "offset": 0,
        "exhausted": [],
    }


def save_progress(progress, conn=None):
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    _ensure_progress_table(conn)
    conn.execute("""
        INSERT OR REPLACE INTO sync_progress (id, data) VALUES (1, ?)
    """, (json.dumps(progress),))
    commit_with_retry(conn)
    if own_conn:
        conn.close()


# ─── Signal Handling ─────────────────────────────────────────────────────────

def _signal_handler(signum, frame):
    print("\n\nInterrupted! Shutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)




# ─── Phase 1-3: Taxonomy Crawl ──────────────────────────────────────────────

def phase_crawl(client, conn, progress, existing_listings, listing_last_ts,
                snapshot_ts):

    crawl_unit_index = progress.get("crawl_unit_index", 0)
    exhausted = set(progress.get("exhausted", []))
    offset = progress.get("offset", 0)

    # Skip already exhausted units
    while crawl_unit_index in exhausted and crawl_unit_index < len(CRAWL_UNITS):
        crawl_unit_index += 1

    if crawl_unit_index >= len(CRAWL_UNITS):
        print(f"[{ts()}] All crawl units already exhausted.")
        return

    stats = {"new_with_images": 0, "new_no_images": 0, "existing_updated": 0, "skipped": 0}

    while crawl_unit_index < len(CRAWL_UNITS):
        # Check for kill file
        if check_kill_file():
            progress["crawl_unit_index"] = crawl_unit_index
            progress["offset"] = offset
            progress["exhausted"] = list(exhausted)
            save_progress(progress, conn)
            commit_with_retry(conn)
            return

        unit = CRAWL_UNITS[crawl_unit_index]

        if offset == 0:
            print(f"\n{'='*60}")
            print(f"CRAWL: {unit['name']} [{crawl_unit_index+1}/{len(CRAWL_UNITS)}]")
            print(f"API: {_api_stats['used']}/{_api_stats['limit']}")
            print(f"{'='*60}")

        # Fetch batch (includes images)
        result, error = fetch_active_listings(
            client, unit["taxonomy_id"], offset,
            unit["min_price"], unit["max_price"]
        )

        if error == "rate_limited":
            print("Rate limited. Saving and stopping.")
            progress["crawl_unit_index"] = crawl_unit_index
            progress["offset"] = offset
            progress["exhausted"] = list(exhausted)
            save_progress(progress, conn)
            commit_with_retry(conn)
            return

        if error == "bad_request" or result is None:
            exhausted.add(crawl_unit_index)
            crawl_unit_index += 1
            offset = 0
            while crawl_unit_index in exhausted and crawl_unit_index < len(CRAWL_UNITS):
                crawl_unit_index += 1
            progress["crawl_unit_index"] = crawl_unit_index
            progress["offset"] = 0
            progress["exhausted"] = list(exhausted)
            save_progress(progress, conn)
            continue

        results = result
        total_count = error  # second return value is count when successful

        if offset >= MAX_OFFSET:
            # Price categories should prevent this - error if it happens
            raise RuntimeError(
                f"FATAL: Offset {offset} hit 10k limit for {unit['name']}. "
                f"Price breaks need to be narrower for this taxonomy."
            )

        if not results or offset + 100 >= total_count:
            if results:
                # Process this last batch before marking exhausted
                _process_crawl_batch(
                    client, results, conn, existing_listings, listing_last_ts,
                    snapshot_ts, stats
                )
            exhausted.add(crawl_unit_index)
            print(f"  Exhausted: {unit['name']}")
            crawl_unit_index += 1
            offset = 0
            while crawl_unit_index in exhausted and crawl_unit_index < len(CRAWL_UNITS):
                crawl_unit_index += 1
            progress["crawl_unit_index"] = crawl_unit_index
            progress["offset"] = 0
            progress["exhausted"] = list(exhausted)
            save_progress(progress, conn)
            commit_with_retry(conn)
            continue

        # Process batch
        _process_crawl_batch(
            client, results, conn, existing_listings, listing_last_ts,
            snapshot_ts, stats
        )

        offset += 100
        progress["crawl_unit_index"] = crawl_unit_index
        progress["offset"] = offset
        progress["exhausted"] = list(exhausted)
        save_progress(progress, conn)
        commit_with_retry(conn)

        print(f"  offset={offset} | new+img={stats['new_with_images']} new-img={stats['new_no_images']} "
              f"dyn_upd={stats['existing_updated']} dyn_fresh={stats['skipped']} "
              f"API={_api_stats['used']}/{_api_stats['limit']}")

    print(f"\n[{ts()}] Crawl complete: {stats['new_with_images']} new+img, "
          f"{stats['new_no_images']} new-img, {stats['existing_updated']} dyn_upd, "
          f"{stats['skipped']} dyn_fresh")


def _process_crawl_batch(client, results, conn, existing_listings, listing_last_ts,
                         snapshot_ts, stats):
    """Process a batch of listings from the taxonomy crawl.

    For each listing:
    - Insert listing data into listings_static/listings_dynamic
    - Fetch images via batch API for ALL listings (updates is_primary, fills missing URLs)
    """
    all_listing_ids = []

    for listing in results:
        lid = listing["listing_id"]
        all_listing_ids.append(lid)

        if lid in existing_listings:
            # Already known listing — update dynamic if > 1 week old
            last_ts = listing_last_ts.get(lid, 0)
            if snapshot_ts - last_ts > ONE_WEEK:
                insert_listing_dynamic(conn, listing, snapshot_ts)
                listing_last_ts[lid] = snapshot_ts
                stats["existing_updated"] += 1
            else:
                stats["skipped"] += 1
        else:
            # New listing — insert static + dynamic
            insert_listing_static(conn, listing, snapshot_ts)
            insert_listing_dynamic(conn, listing, snapshot_ts)
            existing_listings.add(lid)
            listing_last_ts[lid] = snapshot_ts
            stats["new_with_images"] += 1

    # Fetch images for ALL listings in one batch API call
    if all_listing_ids:
        images_by_listing = fetch_listings_batch_with_images(client, all_listing_ids)

        for lid in all_listing_ids:
            images = images_by_listing.get(lid, [])
            if images:
                add_images_to_sql(lid, images, conn=conn)


# ─── Phase: Shop Data ───────────────────────────────────────────────────────

def phase_shops(client, conn, snapshot_ts):
    """Fetch shop data for shops that need updating.

    1. Get shop_ids from listings where MAX(listings_dynamic.snapshot_timestamp) within 7 days
    2. Remove shop_ids where MAX(shops_dynamic.snapshot_timestamp) within 7 days
    3. Fetch remaining shops one at a time
    4. Insert: if not in shops_static -> static + dynamic; else -> dynamic only
    """
    cutoff_ts = snapshot_ts - ONE_WEEK

    # Get shop_ids from listings with recent dynamic entries
    cursor = conn.execute("""
        SELECT DISTINCT ls.shop_id
        FROM listings_static ls
        JOIN (
            SELECT listing_id, MAX(snapshot_timestamp) as max_ts
            FROM listings_dynamic
            GROUP BY listing_id
        ) ld ON ls.listing_id = ld.listing_id
        WHERE ld.max_ts > ?
    """, (cutoff_ts,))
    listing_shop_ids = set(row[0] for row in cursor.fetchall())
    print(f"[{ts()}] Shops: {len(listing_shop_ids)} shops from recent listings")

    # Get shop_ids that already have recent dynamic entries
    cursor = conn.execute("""
        SELECT shop_id, MAX(snapshot_timestamp) as max_ts
        FROM shops_dynamic
        GROUP BY shop_id
        HAVING max_ts > ?
    """, (cutoff_ts,))
    recent_shop_ids = set(row[0] for row in cursor.fetchall())
    print(f"[{ts()}] Shops: {len(recent_shop_ids)} shops already have recent data")

    # Shops that need fetching
    shops_to_fetch = listing_shop_ids - recent_shop_ids
    print(f"[{ts()}] Shops: {len(shops_to_fetch)} shops need fetching")

    if not shops_to_fetch:
        return

    # Get existing shop_ids in shops_static
    cursor = conn.execute("SELECT shop_id FROM shops_static")
    existing_static = set(row[0] for row in cursor.fetchall())

    # Fetch each shop
    fetched = 0
    new_shops = 0
    updated_shops = 0

    for shop_id in shops_to_fetch:
        if check_kill_file():
            commit_with_retry(conn)
            return

        shop_data = fetch_shop(client, shop_id)
        if shop_data:
            if shop_id not in existing_static:
                insert_shop_static(conn, shop_data, snapshot_ts)
                insert_shop_dynamic(conn, shop_data, snapshot_ts)
                existing_static.add(shop_id)
                new_shops += 1
            else:
                insert_shop_dynamic(conn, shop_data, snapshot_ts)
                updated_shops += 1

        fetched += 1
        if fetched % 100 == 0:
            commit_with_retry(conn)
            print(f"  Shops: {fetched}/{len(shops_to_fetch)} (new={new_shops}, updated={updated_shops}) "
                  f"API={_api_stats['used']}/{_api_stats['limit']}")

    commit_with_retry(conn)
    print(f"[{ts()}] Shops complete: {new_shops} new, {updated_shops} updated")


# ─── Phase 5: Sync Check ────────────────────────────────────────────────────

# ─── Phase: Reviews ───────────────────────────────────────────────────────

def phase_reviews(client, conn, snapshot_ts):
    """Fetch reviews for shops with recent dynamic entries."""
    cutoff_ts = snapshot_ts - ONE_MONTH

    # Get shop_ids with recent shops_dynamic entries
    cursor = conn.execute("""
        SELECT shop_id
        FROM shops_dynamic
        GROUP BY shop_id
        HAVING MAX(snapshot_timestamp) > ?
    """, (cutoff_ts,))
    shops_with_recent_dynamic = set(row[0] for row in cursor.fetchall())
    print(f"[{ts()}] Reviews: {len(shops_with_recent_dynamic)} shops with recent dynamic data")

    # Get shop_ids that already have recent reviews
    cursor = conn.execute("""
        SELECT shop_id
        FROM reviews
        GROUP BY shop_id
        HAVING MAX(create_timestamp) > ?
    """, (cutoff_ts,))
    shops_with_recent_reviews = set(row[0] for row in cursor.fetchall())
    print(f"[{ts()}] Reviews: {len(shops_with_recent_reviews)} shops already have recent reviews")

    # Only fetch for shops without recent reviews
    shop_ids = list(shops_with_recent_dynamic - shops_with_recent_reviews)
    print(f"[{ts()}] Reviews: {len(shop_ids)} shops need review fetching")

    if not shop_ids:
        # Reset progress for next cycle
        save_progress({"crawl_unit_index": 0, "offset": 0, "exhausted": []}, conn)
        return

    last_review_ts = get_sync_state(conn, "last_review_timestamps", {})
    # Jan 1, 2000 as default
    default_ts = 946684800

    total_reviews = 0
    print(f"\n[{ts()}] Reviews: syncing {len(shop_ids)} shops with recent data")

    for i, shop_id in enumerate(shop_ids):
        if check_kill_file():
            commit_with_retry(conn)
            set_sync_state(conn, "last_review_timestamps", last_review_ts)
            return

        sid_str = str(shop_id)
        last_ts = last_review_ts.get(sid_str, default_ts)

        reviews = fetch_shop_reviews(client, shop_id, last_ts)
        if reviews:
            newest_ts = max(r.get("create_timestamp", 0) for r in reviews)
            last_review_ts[sid_str] = newest_ts
            # Insert ALL reviews (no listing_id filter — needed for sales ratio analysis)
            for review in reviews:
                insert_review(conn, review, snapshot_ts)
            total_reviews += len(reviews)

        if (i + 1) % 10 == 0:
            commit_with_retry(conn)
            set_sync_state(conn, "last_review_timestamps", last_review_ts)
            print(f"  Reviews: {i+1}/{len(shop_ids)} shops, {total_reviews} new reviews "
                  f"| API={_api_stats['used']}/{_api_stats['limit']}")

    commit_with_retry(conn)
    set_sync_state(conn, "last_review_timestamps", last_review_ts)
    print(f"[{ts()}] Reviews complete: {total_reviews} new reviews from {len(shop_ids)} shops")

    # Reset progress for next cycle
    save_progress({"crawl_unit_index": 0, "offset": 0, "exhausted": []}, conn)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    if not acquire_lock():
        print("Error: Another sync_data.py instance is already running")
        return

    IMAGES_DIR.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"ETSY FURNITURE DATA SYNC")
    print(f"{'='*60}")

    # Load state
    progress = load_progress()
    conn = init_db(DB_FILE)

    print(f"Crawl units: {len(CRAWL_UNITS)}")

    # Pre-load DB state for O(1) lookups
    print("Loading DB state...")
    existing_listings = set(r[0] for r in conn.execute("SELECT listing_id FROM listings_static").fetchall())

    listing_last_ts = {}
    for r in conn.execute("SELECT listing_id, MAX(snapshot_timestamp) FROM listings_dynamic GROUP BY listing_id").fetchall():
        listing_last_ts[r[0]] = r[1]

    print(f"DB: {len(existing_listings)} listings")

    snapshot_ts = int(time.time())

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        # Phase 1: Taxonomy crawl (listings + images)
        print(f"\n--- Phase 1: Taxonomy Crawl ---")
        phase_crawl(
            client, conn, progress, existing_listings, listing_last_ts,
            snapshot_ts
        )

        if check_kill_file():
            commit_with_retry(conn)
            conn.close()
            release_lock()
            return

        # Phase 2: Shops (stateless - skips shops with recent data)
        print(f"\n--- Phase 2: Shops ---")
        phase_shops(client, conn, snapshot_ts)

        if check_kill_file():
            commit_with_retry(conn)
            conn.close()
            release_lock()
            return

        # Phase 3: Reviews (stateless - skips shops with recent reviews)
        print(f"\n--- Phase 3: Reviews ---")
        phase_reviews(client, conn, snapshot_ts)

    # Final save
    commit_with_retry(conn)
    conn.close()
    release_lock()

    print(f"\n{'='*60}")
    print(f"SYNC COMPLETE")
    print(f"{'='*60}")
    print(f"DB: {len(existing_listings)} listings")
    print(f"API: {_api_stats['used']}/{_api_stats['limit']}")
    print(f"Run image_downloader.py to download queued images")


if __name__ == "__main__":
    try:
        main()
    finally:
        release_lock()
