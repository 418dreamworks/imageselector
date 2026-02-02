"""
sync_data.py — Merged Etsy furniture data sync.

Crawls taxonomy categories, downloads images, and populates SQLite database.
Replaces both sync_images.py and the old sync_data.py.

Usage:
    python sync_data_new.py
"""

import os
import sys
import json
import time
import signal
import re
import sqlite3
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
import httpx

load_dotenv()

# ─── Config ─────────────────────────────────────────────────────────────────

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

API_DELAY_DEFAULT = 0.2  # 5 QPS (default, overridden by qps_config.json)
CDN_RATE_LIMIT = 5       # CDN downloads per second
NUM_WORKERS = 1        # 1 download worker
MAX_OFFSET = 10000     # Etsy API offset limit
ONE_WEEK = 7 * 24 * 3600
FAILED_IMAGE_ID = 9999999999
MIN_PRICE = 50             # Only for image downloads (not DB inserts)

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
METADATA_FILE = BASE_DIR / "image_metadata.json"
PROGRESS_FILE = BASE_DIR / "sync_progress.json"
DB_FILE = BASE_DIR / "etsy_data.db"
TAXONOMY_CONFIG_FILE = BASE_DIR / "furniture_taxonomy_config.json"
KILL_FILE = BASE_DIR / "KILL"
QPS_CONFIG_FILE = BASE_DIR / "qps_config.json"


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

ALLOWED_WHEN_MADE = {
    "made_to_order", "2020_2026", "2010_2019",
    "2000_2009", "2000_2006", "2007_2009",
}


# ─── Utilities ───────────────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%H:%M:%S")


def check_kill_file():
    """Check for kill file and delete if found. Returns True if should exit."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        print(f"[{ts()}] Kill file detected. Shutting down gracefully...")
        return True
    return False


def extract_hex_suffix(url: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r'/il/([a-f0-9]+)/(\d+)/il_[^.]+\.\d+_([a-z0-9]+)\.jpg', url)
    if match:
        return match.group(1), match.group(3)
    return None, None


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
        CREATE TABLE IF NOT EXISTS shops (
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

        CREATE TABLE IF NOT EXISTS listings (
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
            price_amount INTEGER,
            price_divisor INTEGER,
            price_currency TEXT,
            taxonomy_id INTEGER,
            production_partners TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_listings_shop_id ON listings(shop_id);

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

        CREATE TABLE IF NOT EXISTS sync_state (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
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
    conn.commit()


# ─── DB Insert Functions ─────────────────────────────────────────────────────

def insert_shop_static(conn, shop_data, snapshot_ts):
    conn.execute("""
        INSERT OR IGNORE INTO shops (
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


def insert_listing_static(conn, listing, snapshot_ts):
    price = listing.get("price", {})
    conn.execute("""
        INSERT OR IGNORE INTO listings (
            listing_id, snapshot_timestamp, shop_id, title, description,
            creation_timestamp, url,
            is_customizable, is_personalizable, listing_type, tags, materials,
            processing_min, processing_max, who_made, when_made,
            item_weight, item_weight_unit, item_length, item_width,
            item_height, item_dimensions_unit, should_auto_renew, language,
            price_amount, price_divisor, price_currency, taxonomy_id,
            production_partners
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        price.get("amount"),
        price.get("divisor"),
        price.get("currency_code"),
        listing.get("taxonomy_id"),
        json.dumps(listing.get("production_partners", [])),
    ))


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
    """Fetch a batch of active listings for a taxonomy + price range."""
    time.sleep(get_api_delay())
    params = {"taxonomy_id": taxonomy_id, "limit": 100, "offset": offset}
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price

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


def fetch_shop(client, shop_id):
    """Fetch full shop data from API."""
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
    except Exception as e:
        print(f"  Error fetching shop {shop_id}: {e}")
        return None


def fetch_listings_batch(client, listing_ids):
    """Fetch up to 100 listings via batch endpoint."""
    time.sleep(get_api_delay())
    try:
        response = client.get(
            f"{BASE_URL}/application/listings/batch",
            headers={"x-api-key": ETSY_API_KEY},
            params={"listing_ids": ",".join(str(lid) for lid in listing_ids)},
        )
        update_api_usage(response)
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"  Error fetching batch of {len(listing_ids)} listings: {e}")
        return []


def get_batch_image_info(client, listing_ids):
    """Get ALL image info for up to 100 listings in one API call.

    Returns: {listing_id: [(image_id, url), ...], ...}
    """
    if not listing_ids:
        return {}

    time.sleep(get_api_delay())
    try:
        response = client.get(
            f"{BASE_URL}/application/listings/batch",
            headers={"x-api-key": ETSY_API_KEY},
            params={
                "listing_ids": ",".join(str(lid) for lid in listing_ids),
                "includes": "Images",
            },
        )
        update_api_usage(response)
        response.raise_for_status()

        results = {}
        for listing in response.json().get("results", []):
            lid = listing.get("listing_id")
            images = listing.get("images", [])
            if images:
                # Return ALL images, not just first
                results[lid] = [(img["listing_image_id"], img["url_570xN"]) for img in images]
        return results
    except Exception as e:
        print(f"  Error in batch image fetch: {e}")
        return {}


def fetch_shop_reviews(client, shop_id, last_timestamp=0):
    """Fetch reviews for a shop newer than last_timestamp."""
    reviews = []
    offset = 0

    while True:
        time.sleep(get_api_delay())
        try:
            response = client.get(
                f"{BASE_URL}/application/shops/{shop_id}/reviews",
                headers={"x-api-key": ETSY_API_KEY},
                params={"limit": 100, "offset": offset},
            )
            update_api_usage(response)
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

            offset += 100
            if offset >= 10000:
                break
        except Exception as e:
            print(f"  Error fetching reviews for shop {shop_id}: {e}")
            break

    return reviews


# ─── Metadata ────────────────────────────────────────────────────────────────

def load_metadata() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {}


def save_metadata(metadata, lock=None):
    """Save metadata to JSON file."""
    if lock:
        with lock:
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f)
    else:
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f)


# ─── Image Files ─────────────────────────────────────────────────────────────

_existing_images: Dict[str, int] = {}  # Key: "listing_id_image_id" or "listing_id"


def load_existing_images() -> Dict[str, int]:
    """Load existing images, handling both old and new naming formats.

    Old format: {listing_id}.jpg
    New format: {listing_id}_{image_id}.jpg

    Returns dict with keys as "{listing_id}_{image_id}" for new format.
    """
    existing = {}
    for f in IMAGES_DIR.glob("*.jpg"):
        stem = f.stem
        # New format: listing_id_image_id
        if "_" in stem:
            existing[stem] = f.stat().st_size
        else:
            # Old format: just listing_id - still track it
            try:
                int(stem)  # Validate it's a number
                existing[stem] = f.stat().st_size
            except ValueError:
                pass
    return existing


def download_image(client, url, listing_id, image_id) -> bool:
    """Download image with new naming: {listing_id}_{image_id}.jpg"""
    try:
        response = client.get(url)
        response.raise_for_status()
        filepath = IMAGES_DIR / f"{listing_id}_{image_id}.jpg"
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  Error downloading image {listing_id}_{image_id}: {e}")
        return False


def create_white_placeholder(filepath: Path) -> int:
    white_jpeg = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x4F, 0xFF,
        0xD9
    ])
    with open(filepath, "wb") as f:
        f.write(white_jpeg)
    return len(white_jpeg)


# ─── Download Queue ──────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, rate):
        self.min_interval = 1.0 / rate
        self.lock = threading.Lock()
        self.last_time = 0.0

    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_time = time.time()


class ImageDownloadQueue:
    """Queue for downloading images with new naming: {listing_id}_{image_id}.jpg"""

    def __init__(self, metadata, metadata_lock):
        self.queue = queue.Queue()
        self.metadata = metadata
        self.metadata_lock = metadata_lock
        self.stats = {"downloaded": 0, "errors": 0}
        self.stats_lock = threading.Lock()
        self.running = True
        self.rate_limiter = RateLimiter(CDN_RATE_LIMIT)
        self.workers = []
        self.num_workers = 8

    def start(self):
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._run_worker, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _run_worker(self):
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            while self.running:
                try:
                    job = self.queue.get(timeout=5.0)
                    if job is None:
                        break

                    listing_id, image_id, image_url = job
                    img_key = f"{listing_id}_{image_id}"
                    self.rate_limiter.acquire()
                    success = download_image(client, image_url, listing_id, image_id)

                    with self.stats_lock:
                        if success:
                            self.stats["downloaded"] += 1
                            filepath = IMAGES_DIR / f"{img_key}.jpg"
                            _existing_images[img_key] = filepath.stat().st_size
                        else:
                            self.stats["errors"] += 1
                            filepath = IMAGES_DIR / f"{img_key}.jpg"
                            _existing_images[img_key] = create_white_placeholder(filepath)

                    self.queue.task_done()
                except queue.Empty:
                    continue

    def add(self, listing_id, images_list, shop_id, when_made):
        """Add all images for a listing to download queue.

        Args:
            listing_id: The listing ID
            images_list: List of (image_id, url) tuples
            shop_id: Shop ID
            when_made: When made category
        """
        lid_str = str(listing_id)

        # Build images metadata
        images_meta = []
        for idx, (image_id, image_url) in enumerate(images_list):
            img_key = f"{listing_id}_{image_id}"
            filepath = IMAGES_DIR / f"{img_key}.jpg"
            filepath.touch()
            _existing_images[img_key] = 0

            hex_val, suffix = extract_hex_suffix(image_url)
            img_entry = {
                "image_id": image_id,
                "hex": hex_val,
                "suffix": suffix,
            }
            if idx == 0:
                img_entry["is_primary"] = True
            images_meta.append(img_entry)
            self.queue.put((listing_id, image_id, image_url))

        # Update metadata with all images
        with self.metadata_lock:
            self.metadata[lid_str] = {
                "shop_id": shop_id,
                "when_made": when_made,
                "images": images_meta,
            }

    def pending(self):
        return self.queue.qsize()

    def get_stats(self):
        with self.stats_lock:
            return dict(self.stats)

    def wait_for_completion(self):
        self.queue.join()

    def shutdown(self):
        self.running = False
        self.queue.put(None)
        if self.worker:
            self.worker.join(timeout=10.0)


# ─── Progress ────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "phase": "crawl",
        "crawl_unit_index": 0,
        "offset": 0,
        "exhausted": [],
    }


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


# ─── Signal Handling ─────────────────────────────────────────────────────────

_download_queue = None


def _signal_handler(signum, frame):
    print("\n\nInterrupted! Shutting down...")
    if _download_queue is not None:
        _download_queue.shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── Shop Handler ────────────────────────────────────────────────────────────

def handle_shop(client, shop_id, conn, existing_shops, shop_last_ts, snapshot_ts):
    """Fetch shop if new or stale. Returns True if an API call was made."""
    if shop_id not in existing_shops:
        shop_data = fetch_shop(client, shop_id)
        if shop_data:
            insert_shop_static(conn, shop_data, snapshot_ts)
            insert_shop_dynamic(conn, shop_data, snapshot_ts)
            existing_shops.add(shop_id)
            shop_last_ts[shop_id] = snapshot_ts
        return True
    else:
        last_ts = shop_last_ts.get(shop_id, 0)
        if snapshot_ts - last_ts > ONE_WEEK:
            shop_data = fetch_shop(client, shop_id)
            if shop_data:
                insert_shop_dynamic(conn, shop_data, snapshot_ts)
                shop_last_ts[shop_id] = snapshot_ts
            return True
    return False


# ─── Phase 1-3: Taxonomy Crawl ──────────────────────────────────────────────

def phase_crawl(client, metadata, metadata_lock, conn, download_queue, progress,
                existing_listings, existing_shops, listing_last_ts, shop_last_ts,
                snapshot_ts, seen_in_crawl):

    crawl_unit_index = progress.get("crawl_unit_index", 0)
    exhausted = set(progress.get("exhausted", []))
    offset = progress.get("offset", 0)

    # Skip already exhausted units
    while crawl_unit_index in exhausted and crawl_unit_index < len(CRAWL_UNITS):
        crawl_unit_index += 1

    if crawl_unit_index >= len(CRAWL_UNITS):
        print(f"[{ts()}] All crawl units already exhausted.")
        return

    stats = {"new_2000plus": 0, "new_no_img": 0, "existing_updated": 0,
             "skipped": 0, "shops_fetched": 0}

    while crawl_unit_index < len(CRAWL_UNITS):
        # Check for kill file
        if check_kill_file():
            progress["crawl_unit_index"] = crawl_unit_index
            progress["offset"] = offset
            progress["exhausted"] = list(exhausted)
            save_progress(progress)
            save_metadata(metadata, metadata_lock)
            conn.commit()
            return

        unit = CRAWL_UNITS[crawl_unit_index]

        if offset == 0:
            print(f"\n{'='*60}")
            print(f"CRAWL: {unit['name']} [{crawl_unit_index+1}/{len(CRAWL_UNITS)}]")
            print(f"API: {_api_stats['used']}/{_api_stats['limit']}")
            print(f"{'='*60}")

        # Fetch batch
        result, error = fetch_active_listings(
            client, unit["taxonomy_id"], offset,
            unit["min_price"], unit["max_price"]
        )

        if error == "rate_limited":
            print("Rate limited. Saving and stopping.")
            progress["crawl_unit_index"] = crawl_unit_index
            progress["offset"] = offset
            progress["exhausted"] = list(exhausted)
            save_progress(progress)
            save_metadata(metadata, metadata_lock)
            conn.commit()
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
            save_progress(progress)
            continue

        results = result
        total_count = error  # second return value is count when successful

        if not results or offset >= MAX_OFFSET or offset + 100 >= total_count:
            if results:
                # Process this last batch before marking exhausted
                _process_crawl_batch(
                    client, results, metadata, metadata_lock, conn, download_queue,
                    existing_listings, existing_shops, listing_last_ts, shop_last_ts,
                    snapshot_ts, seen_in_crawl, stats
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
            save_progress(progress)
            conn.commit()
            save_metadata(metadata, metadata_lock)
            continue

        # Process batch
        _process_crawl_batch(
            client, results, metadata, metadata_lock, conn, download_queue,
            existing_listings, existing_shops, listing_last_ts, shop_last_ts,
            snapshot_ts, seen_in_crawl, stats
        )

        offset += 100
        progress["crawl_unit_index"] = crawl_unit_index
        progress["offset"] = offset
        progress["exhausted"] = list(exhausted)
        save_progress(progress)
        conn.commit()
        # Save metadata every 5000 listings instead of every 100
        if offset % 5000 == 0:
            save_metadata(metadata, metadata_lock)

        dl_stats = download_queue.get_stats()
        print(f"  offset={offset} | new+img={stats['new_2000plus']} new-img={stats['new_no_img']} "
              f"dyn_upd={stats['existing_updated']} dyn_fresh={stats['skipped']} "
              f"shops={stats['shops_fetched']} img_dl={dl_stats['downloaded']} "
              f"API={_api_stats['used']}/{_api_stats['limit']}")

    progress["phase"] = "mopup"
    save_progress(progress)
    print(f"\n[{ts()}] Crawl complete: {stats['new_2000plus']} new+img, "
          f"{stats['new_no_img']} new-img, {stats['existing_updated']} dyn_upd, "
          f"{stats['skipped']} dyn_fresh, {stats['shops_fetched']} shops")


def _process_crawl_batch(client, results, metadata, metadata_lock, conn, download_queue,
                         existing_listings, existing_shops, listing_last_ts, shop_last_ts,
                         snapshot_ts, seen_in_crawl, stats):
    """Process a batch of listings from the taxonomy crawl."""

    new_listings = []  # listings not yet in metadata (need DB insert + possibly image)

    for listing in results:
        lid = listing["listing_id"]
        lid_str = str(lid)
        when_made = listing.get("when_made", "")
        shop_id = listing.get("shop_id")
        is_2000_plus = when_made in ALLOWED_WHEN_MADE

        # Get price for image download decision
        price_data = listing.get("price", {})
        price_amount = price_data.get("amount", 0)
        price_divisor = price_data.get("divisor", 100)
        price = price_amount / price_divisor if price_divisor else 0

        seen_in_crawl.add(lid)

        if lid_str in metadata:
            # Already known listing — update DB dynamic if needed
            if lid in existing_listings:
                last_ts = listing_last_ts.get(lid, 0)
                if snapshot_ts - last_ts > ONE_WEEK:
                    insert_listing_dynamic(conn, listing, snapshot_ts)
                    listing_last_ts[lid] = snapshot_ts
                    stats["existing_updated"] += 1
                else:
                    stats["skipped"] += 1
            else:
                # In metadata but not DB — insert (was pre-2000 before filter removal)
                insert_listing_static(conn, listing, snapshot_ts)
                insert_listing_dynamic(conn, listing, snapshot_ts)
                existing_listings.add(lid)
                listing_last_ts[lid] = snapshot_ts
                stats["existing_updated"] += 1

            # Handle shop for all listings
            if handle_shop(client, shop_id, conn, existing_shops, shop_last_ts, snapshot_ts):
                stats["shops_fetched"] += 1
            continue

        # New listing (not in metadata) — collect for processing
        new_listings.append(listing)

    # Process new listings: all go to DB, only some get images
    if new_listings:
        # Separate into those needing images vs not
        need_images = []
        no_images = []
        for listing in new_listings:
            wm = listing.get("when_made", "")
            price_data = listing.get("price", {})
            price_amount = price_data.get("amount", 0)
            price_divisor = price_data.get("divisor", 100)
            price = price_amount / price_divisor if price_divisor else 0

            if wm in ALLOWED_WHEN_MADE and price >= MIN_PRICE:
                need_images.append(listing)
            else:
                no_images.append(listing)

        # Batch fetch images for listings that need them
        image_info = {}
        if need_images:
            batch_ids = [l["listing_id"] for l in need_images]
            image_info = get_batch_image_info(client, batch_ids)

        # Process listings that need images
        for listing in need_images:
            lid = listing["listing_id"]
            lid_str = str(lid)
            shop_id = listing["shop_id"]
            wm = listing.get("when_made", "")

            if lid in image_info:
                # image_info[lid] is now a list of (image_id, url) tuples
                images_list = image_info[lid]
                download_queue.add(lid, images_list, shop_id, wm)
            else:
                # No images found — still add to metadata
                metadata[lid_str] = {
                    "shop_id": shop_id,
                    "when_made": wm,
                    "images": [],
                }

            # Insert listing into DB
            insert_listing_static(conn, listing, snapshot_ts)
            insert_listing_dynamic(conn, listing, snapshot_ts)
            existing_listings.add(lid)
            listing_last_ts[lid] = snapshot_ts

            # Handle shop
            if handle_shop(client, shop_id, conn, existing_shops, shop_last_ts, snapshot_ts):
                stats["shops_fetched"] += 1

            stats["new_2000plus"] += 1

        # Process listings that don't need images (pre-2000 or under $50)
        for listing in no_images:
            lid = listing["listing_id"]
            lid_str = str(lid)
            shop_id = listing["shop_id"]
            wm = listing.get("when_made", "")

            # Add to metadata (no images - pre-2000 or under $50)
            metadata[lid_str] = {"shop_id": shop_id, "when_made": wm, "images": []}

            # Insert listing into DB (ALL listings go to DB now)
            insert_listing_static(conn, listing, snapshot_ts)
            insert_listing_dynamic(conn, listing, snapshot_ts)
            existing_listings.add(lid)
            listing_last_ts[lid] = snapshot_ts

            # Handle shop
            if handle_shop(client, shop_id, conn, existing_shops, shop_last_ts, snapshot_ts):
                stats["shops_fetched"] += 1

            stats["new_no_img"] += 1


# ─── Phase 4: Mop-Up Pass ───────────────────────────────────────────────────

def phase_mopup(client, metadata, conn, existing_listings, existing_shops,
                listing_last_ts, shop_last_ts, snapshot_ts, seen_in_crawl):

    # Find ALL listings in metadata that were not seen in this crawl cycle
    # (may have changed taxonomy/price since last crawl)
    mopup_ids = []
    for lid_str, entry in metadata.items():
        if not isinstance(entry, dict):
            continue
        try:
            lid = int(lid_str)
        except ValueError:
            continue
        if lid not in seen_in_crawl:
            mopup_ids.append(lid)

    if not mopup_ids:
        print(f"[{ts()}] Mop-up: nothing to do (all listings seen in crawl)")
        return

    print(f"[{ts()}] Mop-up: {len(mopup_ids)} listings not seen in crawl")

    processed = 0
    for batch_start in range(0, len(mopup_ids), 100):
        if check_kill_file():
            conn.commit()
            return

        batch = mopup_ids[batch_start:batch_start + 100]
        results = fetch_listings_batch(client, batch)

        for listing in results:
            lid = listing["listing_id"]
            shop_id = listing["shop_id"]

            if lid not in existing_listings:
                insert_listing_static(conn, listing, snapshot_ts)
                insert_listing_dynamic(conn, listing, snapshot_ts)
                existing_listings.add(lid)
                listing_last_ts[lid] = snapshot_ts
            elif snapshot_ts - listing_last_ts.get(lid, 0) > ONE_WEEK:
                insert_listing_dynamic(conn, listing, snapshot_ts)
                listing_last_ts[lid] = snapshot_ts

            handle_shop(client, shop_id, conn, existing_shops, shop_last_ts, snapshot_ts)

        conn.commit()
        processed += len(batch)
        print(f"  Mop-up: {processed}/{len(mopup_ids)}")

    print(f"[{ts()}] Mop-up complete: {processed} listings processed")


# ─── Phase 5: Sync Check ────────────────────────────────────────────────────

def phase_sync_check(metadata, conn, existing_listings, existing_shops):
    issues = []

    # Check 1: Every metadata entry with images should have those images on disk
    for lid_str, entry in metadata.items():
        if not isinstance(entry, dict):
            continue

        try:
            lid = int(lid_str)
        except ValueError:
            continue

        # Handle both old format (image_id) and new format (images array)
        images = entry.get("images", [])
        if not images and "image_id" in entry and entry["image_id"]:
            # Old format: single image_id
            img_path = IMAGES_DIR / f"{lid}.jpg"
            if not img_path.exists() or img_path.stat().st_size == 0:
                issues.append(f"Missing/empty image: {lid}")
        else:
            # New format: images array with {listing_id}_{image_id}.jpg
            for img_meta in images:
                image_id = img_meta.get("image_id")
                if image_id:
                    img_path = IMAGES_DIR / f"{lid}_{image_id}.jpg"
                    if not img_path.exists() or img_path.stat().st_size == 0:
                        issues.append(f"Missing/empty image: {lid}_{image_id}")

    # Check 2: Every DB listing has metadata entry
    for lid in existing_listings:
        if str(lid) not in metadata:
            issues.append(f"DB listing {lid} has no metadata entry")

    # Check 3: Every metadata entry has a listing in DB
    for lid_str in metadata:
        try:
            lid = int(lid_str)
        except ValueError:
            continue
        if lid not in existing_listings:
            issues.append(f"Metadata listing {lid} not in DB")

    if issues:
        print(f"\n[{ts()}] Sync check: {len(issues)} issues found")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print(f"\n[{ts()}] Sync check: all data consistent")

    return issues


# ─── Phase 6: Reviews ───────────────────────────────────────────────────────

def phase_reviews(client, conn, existing_shops, snapshot_ts):
    shop_ids = sorted(existing_shops)
    last_review_ts = get_sync_state(conn, "last_review_timestamps", {})
    # Jan 1, 2000 as default
    default_ts = 946684800

    total_reviews = 0
    print(f"\n[{ts()}] Reviews: syncing {len(shop_ids)} shops")

    for i, shop_id in enumerate(shop_ids):
        if check_kill_file():
            conn.commit()
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
            conn.commit()
            set_sync_state(conn, "last_review_timestamps", last_review_ts)
            print(f"  Reviews: {i+1}/{len(shop_ids)} shops, {total_reviews} new reviews "
                  f"| API={_api_stats['used']}/{_api_stats['limit']}")

    conn.commit()
    set_sync_state(conn, "last_review_timestamps", last_review_ts)
    print(f"[{ts()}] Reviews complete: {total_reviews} new reviews from {len(shop_ids)} shops")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    IMAGES_DIR.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"ETSY FURNITURE DATA SYNC")
    print(f"{'='*60}")

    # Load state
    metadata = load_metadata()
    progress = load_progress()
    conn = init_db(DB_FILE)

    print(f"Metadata: {len(metadata)} entries")
    print(f"Crawl units: {len(CRAWL_UNITS)}")

    # Pre-load DB state for O(1) lookups
    print("Loading DB state...")
    existing_listings = set(r[0] for r in conn.execute("SELECT listing_id FROM listings").fetchall())
    existing_shops = set(r[0] for r in conn.execute("SELECT shop_id FROM shops").fetchall())

    listing_last_ts = {}
    for r in conn.execute("SELECT listing_id, MAX(snapshot_timestamp) FROM listings_dynamic GROUP BY listing_id").fetchall():
        listing_last_ts[r[0]] = r[1]

    shop_last_ts = {}
    for r in conn.execute("SELECT shop_id, MAX(snapshot_timestamp) FROM shops_dynamic GROUP BY shop_id").fetchall():
        shop_last_ts[r[0]] = r[1]

    print(f"DB: {len(existing_listings)} listings, {len(existing_shops)} shops")

    # Load image files
    global _existing_images
    _existing_images = load_existing_images()
    complete = sum(1 for s in _existing_images.values() if s > 0)
    pending = sum(1 for s in _existing_images.values() if s == 0)
    print(f"Images: {complete} complete, {pending} pending")

    # Start download queue
    metadata_lock = threading.Lock()
    download_queue = ImageDownloadQueue(metadata, metadata_lock)
    download_queue.start()
    global _download_queue
    _download_queue = download_queue

    snapshot_ts = int(time.time())
    seen_in_crawl = set()

    current_phase = progress.get("phase", "crawl")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        # Phase 1-3: Taxonomy crawl
        if current_phase == "crawl":
            print(f"\n--- Phase 1-3: Taxonomy Crawl ---")
            phase_crawl(
                client, metadata, metadata_lock, conn, download_queue, progress,
                existing_listings, existing_shops, listing_last_ts, shop_last_ts,
                snapshot_ts, seen_in_crawl
            )
            current_phase = progress.get("phase", "crawl")

        # Phase 4: Mop-up
        if current_phase == "mopup":
            print(f"\n--- Phase 4: Mop-Up ---")
            phase_mopup(
                client, metadata, conn, existing_listings, existing_shops,
                listing_last_ts, shop_last_ts, snapshot_ts, seen_in_crawl
            )
            progress["phase"] = "sync_check"
            save_progress(progress)
            current_phase = "sync_check"

        # Wait for all image downloads before sync check
        pending_dl = download_queue.pending()
        if pending_dl > 0:
            print(f"\nWaiting for {pending_dl} pending image downloads...")
            download_queue.wait_for_completion()

        # Phase 5: Sync check
        if current_phase == "sync_check":
            print(f"\n--- Phase 5: Sync Check ---")
            phase_sync_check(metadata, conn, existing_listings, existing_shops)
            progress["phase"] = "reviews"
            save_progress(progress)
            current_phase = "reviews"

        # Phase 6: Reviews
        if current_phase == "reviews":
            print(f"\n--- Phase 6: Reviews ---")
            phase_reviews(client, conn, existing_shops, snapshot_ts)
            progress["phase"] = "complete"
            save_progress(progress)

    # Wait for pending downloads
    pending_dl = download_queue.pending()
    if pending_dl > 0:
        print(f"\nWaiting for {pending_dl} pending downloads...")
        download_queue.wait_for_completion()

    dl_stats = download_queue.get_stats()
    download_queue.shutdown()

    # Final save
    conn.commit()
    conn.close()
    save_metadata(metadata, metadata_lock)

    # Reset progress for next cycle
    progress = {"phase": "crawl", "crawl_unit_index": 0, "offset": 0, "exhausted": []}
    save_progress(progress)

    print(f"\n{'='*60}")
    print(f"SYNC COMPLETE")
    print(f"{'='*60}")
    print(f"Images downloaded: {dl_stats['downloaded']}")
    print(f"Download errors: {dl_stats['errors']}")
    print(f"DB: {len(existing_listings)} listings, {len(existing_shops)} shops")
    print(f"Metadata: {len(metadata)} entries")
    print(f"API: {_api_stats['used']}/{_api_stats['limit']}")


if __name__ == "__main__":
    main()
