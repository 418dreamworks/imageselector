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
TAXONOMY_ID = 967  # Furniture

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

# Different sort strategies to get broader coverage of listings
SORT_STRATEGIES = [
    {"sort_on": "created", "sort_order": "desc"},  # Newest first
    {"sort_on": "created", "sort_order": "asc"},   # Oldest first
    {"sort_on": "price", "sort_order": "asc"},     # Cheapest first
    {"sort_on": "price", "sort_order": "desc"},    # Most expensive first
    {"sort_on": "score", "sort_order": "desc"},    # Default relevance
]


def load_metadata() -> dict:
    """Load existing metadata mapping listing_id -> listing_image_id or error string.

    Values are either:
    - int: listing_image_id (successful download)
    - str starting with "error:": error reason (e.g., "error:404", "error:no_images")
    """
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            return json.load(f)
    return {}


def save_metadata(metadata: dict):
    """Save metadata to disk."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)


def load_progress() -> dict:
    """Load progress from previous run."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"offset": 0, "api_calls_today": 0, "last_reset": time.time(), "sort_index": 0}


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
    """Sync all images from furniture taxonomy. Resumes across multiple runs.

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

    # Count successes vs errors in metadata
    success_count = sum(1 for v in metadata.values() if isinstance(v, int))
    error_count = len(metadata) - success_count

    # Get current sort strategy
    sort_index = progress.get("sort_index", 0) % len(SORT_STRATEGIES)
    sort_strategy = SORT_STRATEGIES[sort_index]

    print(f"Existing images in corpus: {success_count}")
    print(f"Existing errors logged: {error_count}")
    print(f"API calls used today: {progress['api_calls_today']}")
    print(f"Starting from offset: {progress['offset']}")
    print(f"Sort strategy: {sort_strategy['sort_on']} ({sort_strategy['sort_order']})")

    stats = {
        "skipped": 0,
        "downloaded": 0,
        "updated": 0,
        "errors": 0,
        "processed": 0,
    }

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        offset = progress["offset"]
        batch_size = 100

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

            # Fetch batch of listings
            time.sleep(API_DELAY)
            response = client.get(
                f"{BASE_URL}/application/listings/active",
                headers={"x-api-key": ETSY_API_KEY},
                params={
                    "taxonomy_id": TAXONOMY_ID,
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

            if response.status_code == 400 and offset >= MAX_OFFSET:
                # Etsy API doesn't allow offset > 10000, rotate to next sort strategy
                sort_index = (sort_index + 1) % len(SORT_STRATEGIES)
                sort_strategy = SORT_STRATEGIES[sort_index]
                print(f"\nReached Etsy's offset limit ({MAX_OFFSET}).")
                print(f"Switching to sort strategy: {sort_strategy['sort_on']} ({sort_strategy['sort_order']})")
                offset = 0
                progress["offset"] = 0
                progress["sort_index"] = sort_index
                save_progress(progress)
                continue

            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            if not results:
                print("\nCompleted full sync!")
                progress["offset"] = 0  # Reset for next full run
                break

            print(f"\nBatch at offset {offset} ({len(results)} listings)...")

            for listing in results:
                if progress["api_calls_today"] >= DAILY_LIMIT:
                    break
                if limit > 0 and stats["processed"] >= limit:
                    break

                listing_id = listing["listing_id"]
                listing_id_str = str(listing_id)

                # Skip if already downloaded (don't waste API call)
                # But retry if it was an error (string value, not int image_id)
                existing = metadata.get(listing_id_str)
                if existing is not None and isinstance(existing, int):
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
                    metadata[listing_id_str] = image_id
                    stats["downloaded"] += 1
                else:
                    metadata[listing_id_str] = "cdn_error"
                    stats["errors"] += 1

                stats["processed"] += 1

            # Save progress after each batch
            offset += batch_size
            progress["offset"] = offset
            save_progress(progress)
            save_metadata(metadata)

            print(f"  Downloaded: {stats['downloaded']} | "
                  f"Skipped: {stats['skipped']} | "
                  f"Errors: {stats['errors']} | "
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

    # Note: Etsy API limits offset to 10000, so we cycle through
    # the same ~10000 listings but metadata prevents re-downloads.
    # Over time, as listings change, we'll capture different ones.


def get_image_path(listing_id: int) -> Path | None:
    """Get the local image path for a listing ID. Returns None if not downloaded."""
    path = IMAGES_DIR / f"{listing_id}.jpg"
    return path if path.exists() else None


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
