"""
Sync Etsy furniture listing images.

Downloads the first image (url_170x135) for each listing.
Can run in two modes:
1. Full scan: Download all furniture listings (taxonomy 967)
2. Incremental: Download specific listing IDs from a file

Usage:
    python sync_images.py                    # Full furniture scan
    python sync_images.py listing_ids.txt   # Sync specific IDs from file

Rate limits (your account):
- API: 150 requests/second, 100,000 requests/day
- Script uses 90,000/day max to leave room for development
"""

import os
import sys
import json
import time
import signal
import httpx
import threading
import queue
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Config
ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Load taxonomy config from JSON file
# Each entry has: id, name, price_breaks (list of prices defining intervals)
# Price breaks [0, 500, 1000000] creates intervals: [0, 500] and (500, 1000000]
TAXONOMY_CONFIG_FILE = Path(__file__).parent / "furniture_taxonomy_config.json"


def load_taxonomy_config():
    """Load taxonomy configuration from JSON file.

    Returns list of crawl units, where each unit is a dict with:
    taxonomy_id, name, min_price, max_price
    """
    with open(TAXONOMY_CONFIG_FILE) as f:
        config = json.load(f)

    crawl_units = []
    for entry in config["taxonomies"]:
        tax_id = entry["id"]
        name = entry["name"]
        breaks = entry["price_breaks"]

        # Create intervals from price breaks
        # [0, 500, 1000000] -> [(0, 500), (500.01, 1000000)]
        for i in range(len(breaks) - 1):
            min_price = breaks[i] if i == 0 else breaks[i] + 0.01
            max_price = breaks[i + 1]

            # Create label for this interval
            if len(breaks) == 2:
                # Single interval, no price label needed
                label = name
            else:
                label = f"{name} (${breaks[i]}-${breaks[i+1]})"

            crawl_units.append({
                "taxonomy_id": tax_id,
                "name": label,
                "min_price": min_price if i > 0 else None,  # None for first interval (API default)
                "max_price": max_price if max_price < 1000000 else None,  # None for last interval
            })

    return crawl_units


# Load crawl units at module level
CRAWL_UNITS = load_taxonomy_config()

# Paths
BASE_DIR = Path(__file__).parent
# Production folders
IMAGES_DIR = BASE_DIR / "images"
METADATA_FILE = BASE_DIR / "image_metadata.json"
PROGRESS_FILE = BASE_DIR / "sync_progress.json"
# Test folders (use --test flag)
IMAGES_DIR_TEST = BASE_DIR / "images_test"
METADATA_FILE_TEST = BASE_DIR / "image_metadata_test.json"
PROGRESS_FILE_TEST = BASE_DIR / "sync_progress_test.json"

IMAGES_DIR.mkdir(exist_ok=True)

# Rate limiting
DAILY_LIMIT = 90000  # Leave 10k buffer for development
API_DELAY = 0.011    # ~95 requests/second for sync (leave room for dev at 5 QPS)
CDN_DELAY = 0.05     # Throttle image downloads
MAX_OFFSET = 10000   # Etsy API doesn't allow offset > 10000

# All furniture taxonomy IDs (parent + leaf) for filtering shop listings
FURNITURE_TAXONOMY_IDS = {
    967, 968, 969, 12455, 12456, 970, 972, 971, 12470, 973, 974, 975, 976, 977,
    11837, 978, 979, 12403, 12405, 12406, 980, 981, 982, 983, 985, 986, 987, 988,
    989, 990, 12369, 12370, 991, 992, 993, 12371, 12372, 994, 11355, 11356, 998,
    996, 12468, 12216, 997, 999, 1000, 1001, 12408
}

# Global reference for signal handler cleanup
_active_download_queue = None


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully - shutdown download workers."""
    print("\n\nInterrupted! Shutting down download workers...")
    if _active_download_queue is not None:
        _active_download_queue.shutdown()
    sys.exit(0)


# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_metadata() -> dict:
    """Load existing metadata mapping listing_id -> metadata dict or error string.

    Values are either:
    - dict: {"image_id": int, "shop_id": int|None} for successful download
    - int: legacy format (just image_id) - migrated to {"image_id": int, "shop_id": None}
    - str: error reason (e.g., "404", "no_images", "cdn_error")
    """
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            data = json.load(f)
        # Migrate legacy int format to dict with shop_id: None
        migrated = False
        for listing_id, value in data.items():
            if isinstance(value, int):
                data[listing_id] = {"image_id": value, "shop_id": None}
                migrated = True
        if migrated:
            with open(METADATA_FILE, "w") as f:
                json.dump(data, f)
            print(f"Migrated legacy metadata entries to include shop_id: None")
        return data
    return {}


def is_success(value) -> bool:
    """Check if metadata value represents a successful download."""
    return isinstance(value, dict)


def needs_shop_id(value) -> bool:
    """Check if metadata entry needs shop_id to be filled in."""
    return isinstance(value, dict) and value.get("shop_id") is None


def save_metadata(metadata: dict):
    """Save metadata to disk."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)


def load_progress() -> dict:
    """Load progress from previous run."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "offset": 0,
        "api_calls_today": 0,
        "last_reset": time.time(),
        "crawl_unit_index": 0,  # Index into CRAWL_UNITS
        "exhausted": [],  # List of crawl unit indices that are fully crawled
        "synced_shops": [],  # Shop IDs that have been fully synced
        "last_shop_refresh": 0,  # Timestamp of last shop refresh (every 30 days)
    }


def save_progress(progress: dict):
    """Save progress for resume."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def get_remaining_calls(client: httpx.Client) -> int:
    """Check remaining API calls from response headers."""
    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params={"limit": 1},
    )
    remaining = int(response.headers.get("x-remaining-today", 0))
    return remaining


def get_first_image_info(client: httpx.Client, listing_id: int) -> tuple[int, str, None] | tuple[None, None, str]:
    """Get the first image's ID and URL for a listing.

    Returns (image_id, url, None) on success, or (None, None, error_reason) on failure.
    """
    try:
        response = client.get(
            f"{BASE_URL}/application/listings/{listing_id}/images",
            headers={"x-api-key": ETSY_API_KEY},
        )
        if response.status_code == 404:
            return None, None, "404"
        response.raise_for_status()
        images = response.json().get("results", [])

        if images:
            first = images[0]
            return first["listing_image_id"], first["url_170x135"], None
        return None, None, "no_images"
    except httpx.HTTPStatusError as e:
        print(f"  Error fetching images for {listing_id}: {e.response.status_code}")
        return None, None, f"http_{e.response.status_code}"


def download_image(client: httpx.Client, url: str, listing_id: int) -> bool:
    """Download image from CDN and save to disk."""
    try:
        response = client.get(url)
        response.raise_for_status()

        filepath = IMAGES_DIR / f"{listing_id}.jpg"
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  Error downloading image for {listing_id}: {e}")
        return False


# =============================================================================
# ASYNC IMAGE DOWNLOAD QUEUE
# =============================================================================
# This allows API calls to continue while images download in background threads

class ImageDownloadQueue:
    """Background queue for downloading images without blocking API calls.

    When add() is called, metadata is immediately set to a placeholder dict with
    image_id and shop_id. The worker then downloads the image. If download fails,
    metadata is updated to "cdn_error". This means skip checks can just look at
    metadata - no need for separate queue tracking.
    """

    def __init__(self, num_workers: int = 4, metadata: dict = None, metadata_lock: threading.Lock = None):
        self.queue = queue.Queue()
        self.workers = []
        self.num_workers = num_workers
        self.metadata = metadata
        self.metadata_lock = metadata_lock
        self.stats = {"downloaded": 0, "errors": 0}
        self.stats_lock = threading.Lock()
        self.running = True

    def start(self):
        """Start worker threads."""
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.workers.append(t)

    def _worker(self):
        """Worker thread that processes download jobs."""
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            was_empty = True  # Start assuming empty, so first check uses long timeout
            while self.running:
                try:
                    # If queue was empty last time, wait longer before checking again
                    timeout = 10.0 if was_empty else 0.1
                    job = self.queue.get(timeout=timeout)
                    was_empty = False  # Got a job, so queue wasn't empty

                    if job is None:  # Shutdown signal
                        break

                    listing_id, image_url = job
                    listing_id_str = str(listing_id)

                    time.sleep(CDN_DELAY)  # Rate limit CDN
                    success = download_image(client, image_url, listing_id)

                    with self.stats_lock:
                        if success:
                            self.stats["downloaded"] += 1
                            # Print progress every 1000 downloads
                            if self.stats["downloaded"] % 1000 == 0:
                                pending = self.queue.qsize()
                                print(f"    [Download workers] {self.stats['downloaded']:,} downloaded, {pending:,} pending")
                        else:
                            self.stats["errors"] += 1
                            # Update metadata to error on failure
                            if self.metadata is not None and self.metadata_lock is not None:
                                with self.metadata_lock:
                                    self.metadata[listing_id_str] = "cdn_error"

                    self.queue.task_done()
                except queue.Empty:
                    was_empty = True  # Queue is empty, wait longer next time
                    continue

    def add(self, listing_id: int, image_url: str, image_id: int, shop_id: int):
        """Add a download job to the queue.

        Immediately sets metadata to the final value (placeholder with image_id/shop_id).
        Worker just downloads the image file. On failure, metadata is updated to error.
        """
        listing_id_str = str(listing_id)
        # Set metadata immediately - this is the "placeholder" that makes the listing
        # appear as already processed to skip checks
        if self.metadata is not None and self.metadata_lock is not None:
            with self.metadata_lock:
                self.metadata[listing_id_str] = {"image_id": image_id, "shop_id": shop_id}
        self.queue.put((listing_id, image_url))

    def pending(self) -> int:
        """Return number of pending downloads."""
        return self.queue.qsize()

    def get_stats(self) -> dict:
        """Get current download stats."""
        with self.stats_lock:
            return dict(self.stats)

    def wait_for_completion(self):
        """Wait for all pending downloads to complete."""
        self.queue.join()

    def shutdown(self):
        """Shutdown all worker threads."""
        self.running = False
        # Send shutdown signals
        for _ in self.workers:
            self.queue.put(None)
        # Wait for workers to finish
        for t in self.workers:
            t.join(timeout=5.0)


def sync_shop_listings(client: httpx.Client, shop_id: int, metadata: dict, progress: dict,
                       download_queue: ImageDownloadQueue = None, metadata_lock: threading.Lock = None) -> dict:
    """Fetch all furniture listings from a shop and download missing images.

    Args:
        download_queue: Optional async download queue. If provided, downloads are queued
                       instead of being done synchronously.
        metadata_lock: Required if download_queue is provided.

    Returns stats dict with downloaded/skipped/errors counts.
    """
    stats = {"downloaded": 0, "skipped": 0, "errors": 0, "api_calls": 0}

    # First pass: collect all furniture listings from this shop
    all_furniture_listings = []
    offset = 0
    batch_size = 100

    while True:
        if progress["api_calls_today"] >= DAILY_LIMIT:
            break

        time.sleep(API_DELAY)
        response = client.get(
            f"{BASE_URL}/application/shops/{shop_id}/listings/active",
            headers={"x-api-key": ETSY_API_KEY},
            params={"limit": batch_size, "offset": offset},
        )
        progress["api_calls_today"] += 1
        stats["api_calls"] += 1

        if response.status_code == 404:
            break
        if response.status_code == 429:
            time.sleep(60)
            continue

        response.raise_for_status()
        listings = response.json().get("results", [])

        if not listings:
            break

        for listing in listings:
            taxonomy_id = listing.get("taxonomy_id")
            if taxonomy_id in FURNITURE_TAXONOMY_IDS:
                all_furniture_listings.append(listing)

        offset += batch_size
        if offset >= 10000:
            break

    # Count how many we already have vs need (include queued as "have")
    already_have = 0
    need_download = 0
    for listing in all_furniture_listings:
        listing_id_str = str(listing["listing_id"])
        existing = metadata.get(listing_id_str)
        if existing is not None and is_success(existing):
            already_have += 1
        else:
            need_download += 1

    if all_furniture_listings:
        print(f"      Shop {shop_id}: {len(all_furniture_listings)} furniture listings "
              f"({already_have} have, {need_download} need)")

    # Second pass: download missing images
    downloaded_this_shop = 0
    for listing in all_furniture_listings:
        if progress["api_calls_today"] >= DAILY_LIMIT:
            break

        listing_id = listing["listing_id"]
        listing_id_str = str(listing_id)

        existing = metadata.get(listing_id_str)

        if existing is not None and is_success(existing):
            # Already have this one, but update shop_id if missing
            if needs_shop_id(existing):
                existing["shop_id"] = shop_id
            stats["skipped"] += 1
            continue

        # Need to download this listing's image
        time.sleep(API_DELAY)
        image_id, image_url, error = get_first_image_info(client, listing_id)
        progress["api_calls_today"] += 1
        stats["api_calls"] += 1

        if error:
            metadata[listing_id_str] = error
            stats["errors"] += 1
            continue

        if download_queue is not None:
            # Async: queue the download
            download_queue.add(listing_id, image_url, image_id, shop_id)
            stats["downloaded"] += 1
            downloaded_this_shop += 1
        else:
            # Sync: download immediately
            time.sleep(CDN_DELAY)
            if download_image(client, image_url, listing_id):
                metadata[listing_id_str] = {"image_id": image_id, "shop_id": shop_id}
                stats["downloaded"] += 1
                downloaded_this_shop += 1
                # Show progress and save every 50 downloads
                if downloaded_this_shop % 10 == 0:
                    remaining = need_download - downloaded_this_shop
                    print(f"        [{downloaded_this_shop}/{need_download}] downloaded, {remaining} remaining")
                if downloaded_this_shop % 50 == 0:
                    save_metadata(metadata)
                    save_progress(progress)
            else:
                metadata[listing_id_str] = "cdn_error"
                stats["errors"] += 1

    # Mark whether this shop was fully synced
    stats["complete"] = (downloaded_this_shop + already_have >= len(all_furniture_listings))
    return stats


def sync_from_list(listing_ids: list[int]):
    """Sync images for a specific list of listing IDs."""
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    metadata = load_metadata()
    print(f"Syncing {len(listing_ids)} listings...")
    print(f"Existing images in corpus: {len(metadata)}")

    # Filter to only new listings (not already in metadata)
    new_ids = [lid for lid in listing_ids if str(lid) not in metadata]
    print(f"New listings to download: {len(new_ids)}")

    if not new_ids:
        print("Nothing to do.")
        return

    stats = {"downloaded": 0, "errors": 0}

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        # Check remaining API calls
        remaining = get_remaining_calls(client)
        print(f"API calls remaining today: {remaining}")

        if len(new_ids) > remaining:
            print(f"Warning: Only processing first {remaining} listings due to API limit.")
            new_ids = new_ids[:remaining]

        for i, listing_id in enumerate(new_ids):
            time.sleep(API_DELAY)
            image_info = get_first_image_info(client, listing_id)

            if not image_info:
                stats["errors"] += 1
                continue

            image_id, image_url = image_info

            time.sleep(CDN_DELAY)
            if download_image(client, image_url, listing_id):
                metadata[str(listing_id)] = image_id
                stats["downloaded"] += 1
            else:
                stats["errors"] += 1

            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(new_ids)} | Downloaded: {stats['downloaded']}")
                save_metadata(metadata)

    save_metadata(metadata)
    print(f"\nDone! Downloaded: {stats['downloaded']}, Errors: {stats['errors']}")


def sync_full_taxonomy(limit: int = 0):
    """Main sync loop using price-based crawl units.

    Algorithm:
    (A) Every 30 days, clear synced_shops to re-check for new listings

    AT STARTUP (before crawling):
      (B)/(C) Fix existing data first:
        - For any listing without shop_id, get the shop
        - For any shop not in synced_shops, sync all its furniture listings

    For each crawl unit (taxonomy + price range):
      Crawl listings until exhausted (no more results)
      (B)/(C) After unit exhausted:
        - Fix any new listings missing shop_id
        - Sync any new unsynced shops
      Move to next crawl unit

    When syncing a shop, only include furniture listings (taxonomy in FURNITURE_TAXONOMY_IDS).

    Args:
        limit: Max listings to process (0 = no limit)
    """
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    metadata = load_metadata()
    progress = load_progress()

    # Check if 24h has passed since last reset
    if time.time() - progress.get("last_reset", 0) > 86400:
        progress["api_calls_today"] = 0
        progress["last_reset"] = time.time()

    # (A) Every 30 days, re-check synced shops for new listings
    THIRTY_DAYS = 30 * 24 * 60 * 60
    last_shop_refresh = progress.get("last_shop_refresh", 0)

    if time.time() - last_shop_refresh > THIRTY_DAYS:
        print("(A) 30 days passed - clearing synced_shops to re-check for updates...")
        progress["synced_shops"] = []
        progress["last_shop_refresh"] = time.time()
        save_progress(progress)

    # Count successes vs errors in metadata
    success_count = sum(1 for v in metadata.values() if is_success(v))
    error_count = len(metadata) - success_count

    print(f"\n{'='*60}")
    print(f"ETSY FURNITURE IMAGE SYNC")
    print(f"{'='*60}")
    print(f"Images in corpus: {success_count} | Errors: {error_count}")
    print(f"Crawl units: {len(CRAWL_UNITS)}")
    print(f"API calls today: {progress['api_calls_today']}")

    # Create thread-safe metadata access and download queue early
    # so it can be used by both startup fix and main crawl
    global _active_download_queue
    metadata_lock = threading.Lock()
    download_queue = ImageDownloadQueue(num_workers=4, metadata=metadata, metadata_lock=metadata_lock)
    download_queue.start()
    _active_download_queue = download_queue  # For signal handler cleanup
    print(f"Started {download_queue.num_workers} background download workers")

    def run_fix_existing_data(client, metadata, progress):
        """(B) and (C): Fix existing data - get shop_id for all listings, sync all shops.

        Returns True if there's still work to do (hit daily limit), False if complete.
        """
        synced_shops = set(progress.get("synced_shops", []))

        # (B) Check if we have listings needing shop_id
        listings_needing_shop = [
            (lid, v) for lid, v in metadata.items()
            if isinstance(v, dict) and v.get("shop_id") is None
        ]

        # (C) Check for shops that have shop_id but aren't synced yet
        shops_needing_sync = set()
        for lid, v in metadata.items():
            if isinstance(v, dict) and v.get("shop_id") is not None:
                shop_id = v["shop_id"]
                if shop_id not in synced_shops:
                    shops_needing_sync.add(shop_id)

        if not listings_needing_shop and not shops_needing_sync:
            return False  # Nothing to fix

        if listings_needing_shop:
            print(f"  (B) {len(listings_needing_shop)} listings need shop_id")

        fixed_count = 0
        shops_synced = 0

        # (B) Fix listings missing shop_id
        for listing_id_str, entry in listings_needing_shop:
            if progress["api_calls_today"] >= DAILY_LIMIT:
                print(f"\nReached daily limit. {len(listings_needing_shop) - fixed_count} listings still need fixing.")
                save_progress(progress)
                save_metadata(metadata)
                return True  # Still work to do

            listing_id = int(listing_id_str)

            # Get listing details to find shop_id
            time.sleep(API_DELAY)
            try:
                response = client.get(
                    f"{BASE_URL}/application/listings/{listing_id}",
                    headers={"x-api-key": ETSY_API_KEY},
                )
                progress["api_calls_today"] += 1

                if response.status_code == 404:
                    # Listing no longer exists, mark as error
                    metadata[listing_id_str] = "404"
                    fixed_count += 1
                    continue
                if response.status_code == 429:
                    print("Rate limited. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                response.raise_for_status()

                listing_data = response.json()
                shop_id = listing_data.get("shop_id")

                if shop_id:
                    # Update the metadata with shop_id
                    entry["shop_id"] = shop_id
                    fixed_count += 1

                    # (C) Sync shop if not already done
                    if shop_id not in synced_shops:
                        # Count remaining before sync
                        remaining_need_shop = sum(
                            1 for lid, v in metadata.items()
                            if isinstance(v, dict) and v.get("shop_id") is None
                        )
                        print(f"  Syncing shop {shop_id}... ({remaining_need_shop} listings still need shop_id)")
                        shop_stats = sync_shop_listings(client, shop_id, metadata, progress, download_queue, metadata_lock)
                        # Note: "complete" is based on queued, not actually downloaded yet
                        # We mark synced optimistically - metadata updates happen async
                        if shop_stats.get("complete", False):
                            synced_shops.add(shop_id)
                            shops_synced += 1
                            progress["synced_shops"] = list(synced_shops)
                        if shop_stats["downloaded"] > 0:
                            print(f"      +{shop_stats['downloaded']} queued from shop")
                            if not shop_stats.get("complete", False):
                                print(f"      (incomplete - will resume next run)")
                        save_progress(progress)
                        with metadata_lock:
                            save_metadata(metadata)

            except Exception as e:
                print(f"    Error getting listing {listing_id}: {e}")

        # Sync any remaining shops that have shop_id but aren't synced yet
        for shop_id in shops_needing_sync:
            if progress["api_calls_today"] >= DAILY_LIMIT:
                print(f"\nReached daily limit during shop sync.")
                save_progress(progress)
                save_metadata(metadata)
                return True  # Still work to do

            if shop_id in synced_shops:
                continue  # Already synced during (B)

            remaining_need_shop = sum(
                1 for lid, v in metadata.items()
                if isinstance(v, dict) and v.get("shop_id") is None
            )
            print(f"  Syncing shop {shop_id}... ({remaining_need_shop} listings still need shop_id)")
            shop_stats = sync_shop_listings(client, shop_id, metadata, progress, download_queue, metadata_lock)
            if shop_stats.get("complete", False):
                synced_shops.add(shop_id)
                shops_synced += 1
                progress["synced_shops"] = list(synced_shops)
            if shop_stats["downloaded"] > 0:
                print(f"      +{shop_stats['downloaded']} queued from shop")
                if not shop_stats.get("complete", False):
                    print(f"      (incomplete - will resume next run)")
            save_progress(progress)
            with metadata_lock:
                save_metadata(metadata)

        print(f"\n(B)/(C) Complete: {fixed_count} listings fixed, {shops_synced} shops synced")
        save_progress(progress)
        with metadata_lock:
            save_metadata(metadata)
        return False  # All done

    # =========================================================================
    # Crawl using price-based units
    # =========================================================================
    crawl_unit_index = progress.get("crawl_unit_index", 0)
    exhausted = set(progress.get("exhausted", []))

    # Find first non-exhausted unit
    while crawl_unit_index in exhausted and crawl_unit_index < len(CRAWL_UNITS):
        crawl_unit_index += 1

    if crawl_unit_index >= len(CRAWL_UNITS):
        print("All crawl units have been exhausted!")
        print(f"Total images in corpus: {success_count}")
        download_queue.shutdown()
        return

    stats = {
        "skipped": 0,
        "downloaded": 0,
        "errors": 0,
        "processed": 0,
    }

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        offset = progress.get("offset", 0)
        batch_size = 100

        # =========================================================
        # (B)/(C) AT STARTUP: Fix existing data before crawling
        # =========================================================
        print("\n(B)/(C) Startup - fixing existing data...")
        still_fixing = run_fix_existing_data(client, metadata, progress)
        if still_fixing:
            print("Daily limit hit during startup fix. Run again tomorrow.")
            # Wait for pending downloads before exiting
            pending = download_queue.pending()
            if pending > 0:
                print(f"Waiting for {pending} pending downloads...")
                download_queue.wait_for_completion()
            download_queue.shutdown()
            return

        while crawl_unit_index < len(CRAWL_UNITS):
            # Check limits
            if limit > 0 and stats["processed"] >= limit:
                print(f"\nReached listing limit ({limit}).")
                break
            if progress["api_calls_today"] >= DAILY_LIMIT:
                print(f"\nReached daily limit ({progress['api_calls_today']} calls).")
                print("Run again after 24 hours to continue.")
                break

            current_unit = CRAWL_UNITS[crawl_unit_index]

            # Display status when starting a new unit
            if offset == 0:
                success_count = sum(1 for v in metadata.values() if is_success(v))
                error_count = len(metadata) - success_count

                print(f"\n{'='*60}")
                print(f"CRAWL UNIT: {current_unit['name']}")
                print(f"            [{crawl_unit_index + 1}/{len(CRAWL_UNITS)} units]")
                price_range = ""
                if current_unit['min_price'] is not None or current_unit['max_price'] is not None:
                    min_p = current_unit['min_price'] or 0
                    max_p = current_unit['max_price'] or "∞"
                    price_range = f" | Price: ${min_p}-${max_p}"
                print(f"            Taxonomy ID: {current_unit['taxonomy_id']}{price_range}")
                print(f"{'='*60}")
                print(f"Images in corpus: {success_count} | Errors: {error_count}")
                print(f"Exhausted units: {len(exhausted)}/{len(CRAWL_UNITS)}")
                print(f"API calls today: {progress['api_calls_today']}")

            # Build API params
            params = {
                "taxonomy_id": current_unit["taxonomy_id"],
                "limit": batch_size,
                "offset": offset,
            }
            if current_unit["min_price"] is not None:
                params["min_price"] = current_unit["min_price"]
            if current_unit["max_price"] is not None:
                params["max_price"] = current_unit["max_price"]

            # Fetch batch of listings
            time.sleep(API_DELAY)
            response = client.get(
                f"{BASE_URL}/application/listings/active",
                headers={"x-api-key": ETSY_API_KEY},
                params=params,
            )
            progress["api_calls_today"] += 1

            if response.status_code == 429:
                print("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue

            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            if not results or offset >= MAX_OFFSET:
                # No more results or hit offset limit - mark this unit as exhausted
                exhausted.add(crawl_unit_index)
                if not results:
                    print(f"\n  Exhausted: {current_unit['name']} (no more results)")
                else:
                    print(f"\n  Exhausted: {current_unit['name']} (offset limit)")

                # Run (B)/(C) after exhausting a unit
                print(f"\n(B)/(C) Unit complete - fixing existing data...")
                still_fixing = run_fix_existing_data(client, metadata, progress)
                if still_fixing:
                    progress["offset"] = 0
                    progress["crawl_unit_index"] = crawl_unit_index + 1
                    progress["exhausted"] = list(exhausted)
                    save_progress(progress)
                    return

                # Move to next unit
                crawl_unit_index += 1
                offset = 0
                progress["offset"] = 0
                progress["crawl_unit_index"] = crawl_unit_index
                progress["exhausted"] = list(exhausted)
                save_progress(progress)

                # Skip already exhausted units
                while crawl_unit_index in exhausted and crawl_unit_index < len(CRAWL_UNITS):
                    crawl_unit_index += 1
                continue

            print(f"\nBatch at offset {offset} ({len(results)} listings)...")

            # Track shops we encounter in this batch
            synced_shops = set(progress.get("synced_shops", []))
            shops_to_sync = set()

            for listing in results:
                if progress["api_calls_today"] >= DAILY_LIMIT:
                    break
                if limit > 0 and stats["processed"] >= limit:
                    break

                listing_id = listing["listing_id"]
                listing_id_str = str(listing_id)
                shop_id = listing.get("shop_id")

                # Track this shop for later syncing
                if shop_id and shop_id not in synced_shops:
                    shops_to_sync.add(shop_id)

                # Skip if already in metadata (downloaded or queued - both set metadata immediately)
                existing = metadata.get(listing_id_str)
                if existing is not None and is_success(existing):
                    if needs_shop_id(existing) and shop_id:
                        existing["shop_id"] = shop_id
                    stats["skipped"] += 1
                    stats["processed"] += 1
                    continue

                # Get image info
                time.sleep(API_DELAY)
                image_id, image_url, error = get_first_image_info(client, listing_id)
                progress["api_calls_today"] += 1

                if error:
                    metadata[listing_id_str] = error
                    stats["errors"] += 1
                    stats["processed"] += 1
                    continue

                # Queue image for background download (non-blocking)
                download_queue.add(listing_id, image_url, image_id, shop_id)
                stats["downloaded"] += 1  # Optimistic - queue will update metadata on completion
                stats["processed"] += 1

            # Sync any new shops from this batch
            for shop_id in shops_to_sync:
                if progress["api_calls_today"] >= DAILY_LIMIT:
                    break
                if shop_id in synced_shops:
                    continue
                print(f"  (C) Syncing shop {shop_id}...")
                shop_stats = sync_shop_listings(client, shop_id, metadata, progress, download_queue, metadata_lock)
                if shop_stats.get("complete", False):
                    synced_shops.add(shop_id)
                if shop_stats["downloaded"] > 0 or shop_stats["errors"] > 0:
                    print(f"    Shop {shop_id}: +{shop_stats['downloaded']} queued, {shop_stats['errors']} errors")
                    if not shop_stats.get("complete", False):
                        print(f"    (incomplete - will resume next run)")
                stats["downloaded"] += shop_stats["downloaded"]
                stats["errors"] += shop_stats["errors"]
                progress["synced_shops"] = list(synced_shops)

            # Save progress after each batch
            offset += batch_size
            progress["offset"] = offset
            save_progress(progress)
            with metadata_lock:
                save_metadata(metadata)

            queue_stats = download_queue.get_stats()
            print(f"  Queued: {stats['downloaded']} | "
                  f"Downloaded: {queue_stats['downloaded']} | "
                  f"Pending: {download_queue.pending()} | "
                  f"Skipped: {stats['skipped']} | "
                  f"Shops: {len(synced_shops)} | "
                  f"API: {progress['api_calls_today']}")

    # Wait for all pending downloads to complete
    pending = download_queue.pending()
    if pending > 0:
        print(f"\nWaiting for {pending} pending downloads to complete...")
        download_queue.wait_for_completion()

    # Get final stats from queue
    queue_stats = download_queue.get_stats()

    # Shutdown download workers
    download_queue.shutdown()

    save_progress(progress)
    with metadata_lock:
        save_metadata(metadata)

    print("\n" + "=" * 50)
    print("Session complete!")
    print(f"  Images downloaded: {queue_stats['downloaded']}")
    print(f"  Download errors: {queue_stats['errors']}")
    print(f"  Skipped (unchanged): {stats['skipped']}")
    print(f"  API calls used: {progress['api_calls_today']}")
    print(f"  Next offset: {progress.get('offset', 0)}")
    print(f"  Images stored in: {IMAGES_DIR}")


def get_image_path(listing_id: int) -> Path | None:
    """Get the local image path for a listing ID. Returns None if not downloaded."""
    path = IMAGES_DIR / f"{listing_id}.jpg"
    return path if path.exists() else None


def get_unique_shop_ids() -> set[int]:
    """Extract unique shop IDs from metadata for phase 2 crawling."""
    metadata = load_metadata()
    shop_ids = set()
    for value in metadata.values():
        if isinstance(value, dict) and "shop_id" in value:
            shop_ids.add(value["shop_id"])
    return shop_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sync Etsy furniture images")
    parser.add_argument("input_file", nargs="?", help="File with listing IDs (one per line)")
    parser.add_argument("--test", action="store_true", help="Use test folders instead of production")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of listings to process (0 = no limit)")
    args = parser.parse_args()

    # Switch to test folders if --test flag
    if args.test:
        IMAGES_DIR = IMAGES_DIR_TEST
        METADATA_FILE = METADATA_FILE_TEST
        PROGRESS_FILE = PROGRESS_FILE_TEST
        IMAGES_DIR.mkdir(exist_ok=True)
        print("*** TEST MODE - using test folders ***\n")

    if args.input_file:
        # Incremental mode: sync specific listing IDs from file
        input_file = Path(args.input_file)
        if not input_file.exists():
            print(f"Error: File not found: {input_file}")
            sys.exit(1)

        with open(input_file) as f:
            listing_ids = [int(line.strip()) for line in f if line.strip().isdigit()]

        print(f"Loaded {len(listing_ids)} listing IDs from {input_file}")
        sync_from_list(listing_ids)
    else:
        # Full mode: scan entire furniture taxonomy
        sync_full_taxonomy(limit=args.limit)
