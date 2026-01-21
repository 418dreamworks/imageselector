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
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Config
ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Leaf taxonomy IDs for furniture (deepest level categories)
# This gives us 38 categories × 10,000 offset limit = 380,000 potential unique listings
FURNITURE_TAXONOMIES = [
    12455, 12456,  # Bed Frames, Headboards
    970,   # Dressers & Armoires
    972,   # Nightstands
    971,   # Steps & Stools (bedroom)
    12470, # Vanity Tables
    974,   # Buffets & China Cabinets
    975,   # Dining Chairs
    976,   # Dining Sets
    977,   # Kitchen & Dining Tables
    11837, # Kitchen Islands
    978,   # Stools & Banquettes
    12403, # Hall Trees
    12405, # Standing Coat Racks
    12406, # Umbrella Stands
    981,   # Bean Bag Chairs
    982,   # Benches & Toy Boxes
    983,   # Bookcases (kids)
    985,   # Desks, Tables & Chairs (kids)
    986,   # Dressers & Drawers (kids)
    987,   # Steps & Stools (kids)
    988,   # Toddler Beds
    12369, # Benches
    12370, # Trunks
    991,   # Bookshelves
    992,   # Chairs
    12371, # Coffee Tables
    12372, # End Tables
    11355, # Console & Sofa Tables
    11356, # TV Stands & Media Centers
    998,   # Couches & Loveseats
    996,   # Floor Pillows
    12468, # Ottomans & Poufs
    12216, # Room Dividers
    997,   # Slipcovers
    1000,  # Desk Chairs
    1001,  # Desks
    12408, # Filing Cabinets
]

# Human-readable names for progress display
TAXONOMY_NAMES = {
    12455: "Bed Frames", 12456: "Headboards", 970: "Dressers & Armoires",
    972: "Nightstands", 971: "Steps & Stools (bedroom)", 12470: "Vanity Tables",
    974: "Buffets & China Cabinets", 975: "Dining Chairs", 976: "Dining Sets",
    977: "Kitchen & Dining Tables", 11837: "Kitchen Islands", 978: "Stools & Banquettes",
    12403: "Hall Trees", 12405: "Standing Coat Racks", 12406: "Umbrella Stands",
    981: "Bean Bag Chairs", 982: "Benches & Toy Boxes", 983: "Bookcases (kids)",
    985: "Desks, Tables & Chairs (kids)", 986: "Dressers & Drawers (kids)",
    987: "Steps & Stools (kids)", 988: "Toddler Beds", 12369: "Benches",
    12370: "Trunks", 991: "Bookshelves", 992: "Chairs", 12371: "Coffee Tables",
    12372: "End Tables", 11355: "Console & Sofa Tables", 11356: "TV Stands & Media Centers",
    998: "Couches & Loveseats", 996: "Floor Pillows", 12468: "Ottomans & Poufs",
    12216: "Room Dividers", 997: "Slipcovers", 1000: "Desk Chairs", 1001: "Desks",
    12408: "Filing Cabinets",
}

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

# Sort strategies - relevance first, then others for large categories
SORT_STRATEGIES = [
    {"sort_on": "score", "sort_order": "desc"},    # Relevance first
    {"sort_on": "created", "sort_order": "desc"},  # Newest
    {"sort_on": "created", "sort_order": "asc"},   # Oldest
    {"sort_on": "price", "sort_order": "desc"},    # Most expensive
    {"sort_on": "price", "sort_order": "asc"},     # Cheapest
]

# All furniture taxonomy IDs (parent + leaf) for filtering shop listings
FURNITURE_TAXONOMY_IDS = {
    967, 968, 969, 12455, 12456, 970, 972, 971, 12470, 973, 974, 975, 976, 977,
    11837, 978, 979, 12403, 12405, 12406, 980, 981, 982, 983, 985, 986, 987, 988,
    989, 990, 12369, 12370, 991, 992, 993, 12371, 12372, 994, 11355, 11356, 998,
    996, 12468, 12216, 997, 999, 1000, 1001, 12408
}


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
        "taxonomy_index": 0,
        "sort_index": 0,
        "exhausted": [],  # List of "taxonomy_index:sort_index" combos fully crawled
        "small_taxonomies": [],  # Taxonomy indices with < 10000 items (skip extra sorts)
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


def sync_shop_listings(client: httpx.Client, shop_id: int, metadata: dict, progress: dict) -> dict:
    """Fetch all furniture listings from a shop and download missing images.

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

    # Count how many we already have vs need
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


def ensure_complete(client: httpx.Client, metadata: dict, progress: dict) -> dict:
    """(B) and (C): Ensure all listings have images and all shops are complete.

    (B) For any listing without an image, download it.
    (C) For any listing without a shop, get the shop, then get all furniture
        listings for that shop.

    Returns stats dict.
    """
    stats = {"downloaded": 0, "shops_synced": 0, "errors": 0}
    synced_shops = set(progress.get("synced_shops", []))

    # Find listings needing shop_id
    listings_needing_shop = [
        (lid, v) for lid, v in metadata.items()
        if isinstance(v, dict) and v.get("shop_id") is None
    ]

    if not listings_needing_shop:
        return stats

    print(f"  Ensuring completeness: {len(listings_needing_shop)} listings need shop info...")

    for listing_id_str, entry in listings_needing_shop:
        if progress["api_calls_today"] >= DAILY_LIMIT:
            break

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
                continue
            response.raise_for_status()

            listing_data = response.json()
            shop_id = listing_data.get("shop_id")

            if shop_id:
                # Update the metadata with shop_id
                entry["shop_id"] = shop_id

                # (C) Sync shop if not already done
                if shop_id not in synced_shops:
                    shop_stats = sync_shop_listings(client, shop_id, metadata, progress)
                    stats["downloaded"] += shop_stats["downloaded"]
                    stats["errors"] += shop_stats["errors"]
                    if shop_stats.get("complete", False):
                        synced_shops.add(shop_id)
                        stats["shops_synced"] += 1
                        progress["synced_shops"] = list(synced_shops)

        except Exception as e:
            print(f"    Error getting listing {listing_id}: {e}")
            stats["errors"] += 1

    return stats


def sync_full_taxonomy(limit: int = 0):
    """Main sync loop following the specified algorithm.

    Algorithm:
    (A) Every 30 days, clear synced_shops to re-check for new listings

    AT STARTUP (before crawling):
      (B)/(C) Fix existing data first:
        - For any listing without shop_id, get the shop
        - For any shop not in synced_shops, sync all its furniture listings

    For each leaf taxonomy:
      (D1) Crawl listings with sort=relevance until 10k offset or exhausted
      (D2) Reset offset, crawl with sort=created_desc until 10k or exhausted
      (D3) Reset offset, crawl with sort=created_asc until 10k or exhausted
      (D4) Reset offset, crawl with sort=price_desc until 10k or exhausted
      (D5) Reset offset, crawl with sort=price_asc until 10k or exhausted
      (B)/(C) After all sorts exhausted for this leaf:
        - Fix any new listings missing shop_id
        - Sync any new unsynced shops
      Move to next leaf

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
    print(f"API calls today: {progress['api_calls_today']}")

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
        if shops_needing_sync:
            print(f"  (C) {len(shops_needing_sync)} shops need syncing")

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
                        print(f"  (C) Syncing shop {shop_id}...")
                        shop_stats = sync_shop_listings(client, shop_id, metadata, progress)
                        if shop_stats.get("complete", False):
                            synced_shops.add(shop_id)
                            shops_synced += 1
                            progress["synced_shops"] = list(synced_shops)
                        if shop_stats["downloaded"] > 0:
                            print(f"      +{shop_stats['downloaded']} images from shop")
                            if not shop_stats.get("complete", False):
                                print(f"      (incomplete - will resume next run)")

                    # Save periodically
                    if fixed_count % 100 == 0:
                        print(f"  Progress: {fixed_count}/{len(listings_needing_shop)} listings fixed, {shops_synced} shops synced")
                        save_progress(progress)
                        save_metadata(metadata)

            except Exception as e:
                print(f"    Error getting listing {listing_id}: {e}")

        # (C) Sync any remaining shops that have shop_id but aren't synced yet
        # (Some may have been synced during (B), so recheck)
        for shop_id in shops_needing_sync:
            if progress["api_calls_today"] >= DAILY_LIMIT:
                print(f"\nReached daily limit during shop sync.")
                save_progress(progress)
                save_metadata(metadata)
                return True  # Still work to do

            if shop_id in synced_shops:
                continue  # Already synced during (B)

            print(f"  (C) Syncing shop {shop_id}...")
            shop_stats = sync_shop_listings(client, shop_id, metadata, progress)
            if shop_stats.get("complete", False):
                synced_shops.add(shop_id)
                shops_synced += 1
                progress["synced_shops"] = list(synced_shops)
            if shop_stats["downloaded"] > 0:
                print(f"      +{shop_stats['downloaded']} images from shop")
                if not shop_stats.get("complete", False):
                    print(f"      (incomplete - will resume next run)")

        print(f"\n(B)/(C) Complete: {fixed_count} listings fixed, {shops_synced} shops synced")
        save_progress(progress)
        save_metadata(metadata)
        return False  # All done

    # =========================================================================
    # (D) Now crawl new taxonomies
    # =========================================================================
    # Get current state
    taxonomy_index = progress.get("taxonomy_index", 0)
    sort_index = progress.get("sort_index", 0)
    exhausted = set(progress.get("exhausted", []))
    small_taxonomies = set(progress.get("small_taxonomies", []))

    def combo_key(tax_idx, sort_idx):
        return f"{tax_idx}:{sort_idx}"

    def find_next_combo(start_tax, start_sort):
        """Find next taxonomy/sort combo that isn't exhausted.

        Logic:
        - For small taxonomies (< 10k items), only use sort_index=0 (relevance)
        - For large taxonomies, try all sorts
        """
        tax_idx = start_tax
        sort_idx = start_sort
        checked = 0
        total_combos = len(FURNITURE_TAXONOMIES) * len(SORT_STRATEGIES)

        while checked < total_combos:
            # Skip non-relevance sorts for small taxonomies
            if tax_idx in small_taxonomies and sort_idx > 0:
                sort_idx = 0
                tax_idx = (tax_idx + 1) % len(FURNITURE_TAXONOMIES)
                checked += len(SORT_STRATEGIES) - 1  # Skip all other sorts
                continue

            if combo_key(tax_idx, sort_idx) not in exhausted:
                return tax_idx, sort_idx

            # Move to next combo
            sort_idx = (sort_idx + 1) % len(SORT_STRATEGIES)
            if sort_idx == 0:
                tax_idx = (tax_idx + 1) % len(FURNITURE_TAXONOMIES)
            checked += 1

        return None, None  # All exhausted

    # Find first unexhausted combo
    taxonomy_index, sort_index = find_next_combo(taxonomy_index, sort_index)
    if taxonomy_index is None:
        print("All taxonomy/sort combinations have been exhausted!")
        print(f"Total images in corpus: {success_count}")
        return

    current_taxonomy = FURNITURE_TAXONOMIES[taxonomy_index]
    sort_strategy = SORT_STRATEGIES[sort_index]

    stats = {
        "skipped": 0,
        "downloaded": 0,
        "updated": 0,
        "errors": 0,
        "processed": 0,
    }

    def is_leaf_exhausted(tax_idx):
        """Check if all sort strategies for a taxonomy have been exhausted."""
        for s_idx in range(len(SORT_STRATEGIES)):
            if combo_key(tax_idx, s_idx) not in exhausted:
                return False
        return True

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        offset = progress["offset"]
        batch_size = 100

        # Track the last taxonomy we completed (B)/(C) for
        last_fixed_taxonomy = progress.get("last_fixed_taxonomy", -1)

        # =========================================================
        # (B)/(C) AT STARTUP: Fix existing data before crawling
        # =========================================================
        print("\n(B)/(C) Startup - fixing existing data...")
        still_fixing = run_fix_existing_data(client, metadata, progress)
        if still_fixing:
            # Hit daily limit while fixing - stop and resume tomorrow
            print("Daily limit hit during startup fix. Run again tomorrow.")
            return

        while True:
            # Check listing limit
            if limit > 0 and stats["processed"] >= limit:
                print(f"\nReached listing limit ({limit}).")
                break
            # Check daily limit
            if progress["api_calls_today"] >= DAILY_LIMIT:
                print(f"\nReached daily limit ({progress['api_calls_today']} calls).")
                print("Run again after 24 hours to continue.")
                break

            # =========================================================
            # Display status when starting a new taxonomy
            # =========================================================
            if sort_index == 0 and offset == 0:
                success_count = sum(1 for v in metadata.values() if is_success(v))
                error_count = len(metadata) - success_count
                effective_exhausted = len(exhausted)
                total_combos = len(FURNITURE_TAXONOMIES) * len(SORT_STRATEGIES)
                skipped_from_small = len(small_taxonomies) * (len(SORT_STRATEGIES) - 1)

                taxonomy_name = TAXONOMY_NAMES.get(current_taxonomy, str(current_taxonomy))
                print(f"\n{'='*60}")
                print(f"CURRENT LEAF: {taxonomy_name} (ID: {current_taxonomy})")
                print(f"             [{taxonomy_index + 1}/{len(FURNITURE_TAXONOMIES)} taxonomies]")
                print(f"{'='*60}")
                print(f"Images in corpus: {success_count} | Errors: {error_count}")
                print(f"Small taxonomies (< 10k): {len(small_taxonomies)}")
                print(f"Exhausted combos: {effective_exhausted}/{total_combos - skipped_from_small}")
                print(f"API calls today: {progress['api_calls_today']}")
                print(f"Starting sort: {sort_strategy['sort_on']} ({sort_strategy['sort_order']})")

            # Check offset limit BEFORE making request
            if offset >= MAX_OFFSET:
                # Hit 10k limit - mark this sort as exhausted
                old_taxonomy_index = taxonomy_index
                exhausted.add(combo_key(taxonomy_index, sort_index))
                taxonomy_name = TAXONOMY_NAMES.get(current_taxonomy, str(current_taxonomy))
                print(f"\n  Exhausted (offset limit): {taxonomy_name}, sort {sort_strategy['sort_on']}")

                # Find next combo
                taxonomy_index, sort_index = find_next_combo(taxonomy_index, sort_index + 1)
                if taxonomy_index is None:
                    print("\nAll taxonomy/sort combinations exhausted!")
                    break

                # Check if we finished all sorts for the old leaf
                if is_leaf_exhausted(old_taxonomy_index) and old_taxonomy_index != last_fixed_taxonomy:
                    print(f"\n(B)/(C) Leaf complete - fixing existing data...")
                    still_fixing = run_fix_existing_data(client, metadata, progress)
                    if still_fixing:
                        # Save state and exit - will resume tomorrow
                        progress["offset"] = 0
                        progress["sort_index"] = sort_index
                        progress["taxonomy_index"] = taxonomy_index
                        progress["exhausted"] = list(exhausted)
                        progress["small_taxonomies"] = list(small_taxonomies)
                        save_progress(progress)
                        return
                    last_fixed_taxonomy = old_taxonomy_index
                    progress["last_fixed_taxonomy"] = old_taxonomy_index

                current_taxonomy = FURNITURE_TAXONOMIES[taxonomy_index]
                sort_strategy = SORT_STRATEGIES[sort_index]
                taxonomy_name = TAXONOMY_NAMES.get(current_taxonomy, str(current_taxonomy))
                print(f"\nMOVING TO: {taxonomy_name}, sort: {sort_strategy['sort_on']} ({sort_strategy['sort_order']})")
                offset = 0
                progress["offset"] = 0
                progress["sort_index"] = sort_index
                progress["taxonomy_index"] = taxonomy_index
                progress["exhausted"] = list(exhausted)
                progress["small_taxonomies"] = list(small_taxonomies)
                save_progress(progress)
                continue

            # Fetch batch of listings
            time.sleep(API_DELAY)
            response = client.get(
                f"{BASE_URL}/application/listings/active",
                headers={"x-api-key": ETSY_API_KEY},
                params={
                    "taxonomy_id": current_taxonomy,
                    "limit": batch_size,
                    "offset": offset,
                    **sort_strategy,
                },
            )
            progress["api_calls_today"] += 1

            if response.status_code == 429:
                print("Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                continue

            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            if not results:
                # No more results - mark this sort as exhausted
                old_taxonomy_index = taxonomy_index
                exhausted.add(combo_key(taxonomy_index, sort_index))
                taxonomy_name = TAXONOMY_NAMES.get(current_taxonomy, str(current_taxonomy))

                # If this was relevance sort (index 0), mark as small taxonomy
                if sort_index == 0:
                    small_taxonomies.add(taxonomy_index)
                    print(f"\n  Exhausted (small, < 10k): {taxonomy_name}")
                else:
                    print(f"\n  Exhausted (no more): {taxonomy_name}, sort {sort_strategy['sort_on']}")

                # Find next combo
                taxonomy_index, sort_index = find_next_combo(taxonomy_index, sort_index + 1)
                if taxonomy_index is None:
                    print("\nAll taxonomy/sort combinations exhausted!")
                    break

                # Check if we finished all sorts for the old leaf
                if is_leaf_exhausted(old_taxonomy_index) and old_taxonomy_index != last_fixed_taxonomy:
                    print(f"\n(B)/(C) Leaf complete - fixing existing data...")
                    still_fixing = run_fix_existing_data(client, metadata, progress)
                    if still_fixing:
                        # Save state and exit - will resume tomorrow
                        progress["offset"] = 0
                        progress["sort_index"] = sort_index
                        progress["taxonomy_index"] = taxonomy_index
                        progress["exhausted"] = list(exhausted)
                        progress["small_taxonomies"] = list(small_taxonomies)
                        save_progress(progress)
                        return
                    last_fixed_taxonomy = old_taxonomy_index
                    progress["last_fixed_taxonomy"] = old_taxonomy_index

                current_taxonomy = FURNITURE_TAXONOMIES[taxonomy_index]
                sort_strategy = SORT_STRATEGIES[sort_index]
                taxonomy_name = TAXONOMY_NAMES.get(current_taxonomy, str(current_taxonomy))
                print(f"\nMOVING TO: {taxonomy_name}, sort: {sort_strategy['sort_on']} ({sort_strategy['sort_order']})")
                offset = 0
                progress["offset"] = 0
                progress["sort_index"] = sort_index
                progress["taxonomy_index"] = taxonomy_index
                progress["exhausted"] = list(exhausted)
                progress["small_taxonomies"] = list(small_taxonomies)
                save_progress(progress)
                continue

            print(f"\nBatch at offset {offset} ({len(results)} listings)...")

            # Track shops we encounter in this batch that need syncing
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

                # Skip if already downloaded (don't waste API call)
                # But retry if it was an error (string value, not dict/int)
                existing = metadata.get(listing_id_str)
                if existing is not None and is_success(existing):
                    # Update shop_id if it was missing
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

                # Download image
                time.sleep(CDN_DELAY)
                if download_image(client, image_url, listing_id):
                    metadata[listing_id_str] = {"image_id": image_id, "shop_id": shop_id}
                    stats["downloaded"] += 1
                else:
                    metadata[listing_id_str] = "cdn_error"
                    stats["errors"] += 1

                stats["processed"] += 1

            # Sync any new shops from this batch that we haven't synced yet
            for shop_id in shops_to_sync:
                if progress["api_calls_today"] >= DAILY_LIMIT:
                    break
                if shop_id in synced_shops:
                    continue
                print(f"  (C) Syncing shop {shop_id}...")
                shop_stats = sync_shop_listings(client, shop_id, metadata, progress)
                if shop_stats.get("complete", False):
                    synced_shops.add(shop_id)
                if shop_stats["downloaded"] > 0 or shop_stats["errors"] > 0:
                    print(f"    Shop {shop_id}: +{shop_stats['downloaded']} images, {shop_stats['errors']} errors")
                    if not shop_stats.get("complete", False):
                        print(f"    (incomplete - will resume next run)")
                stats["downloaded"] += shop_stats["downloaded"]
                stats["errors"] += shop_stats["errors"]
                progress["synced_shops"] = list(synced_shops)

            # Save progress after each batch
            offset += batch_size
            progress["offset"] = offset
            save_progress(progress)
            save_metadata(metadata)

            print(f"  Downloaded: {stats['downloaded']} | "
                  f"Skipped: {stats['skipped']} | "
                  f"Errors: {stats['errors']} | "
                  f"Shops: {len(synced_shops)} | "
                  f"API: {progress['api_calls_today']}")

    save_progress(progress)
    save_metadata(metadata)

    print("\n" + "=" * 50)
    print("Session complete!")
    print(f"  New downloads: {stats['downloaded']}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Skipped (unchanged): {stats['skipped']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  API calls used: {progress['api_calls_today']}")
    print(f"  Next offset: {progress['offset']}")
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
