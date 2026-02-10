#!/usr/bin/env python3
"""
Rebuild metadata for orphaned images (files on disk with no metadata entry).

For each orphaned image:
1. Extract listing_id from filename
2. Call Etsy API to get listing info (shop_id, images)
3. Add metadata entry with shop_id, image_id, hex, suffix

Rate limit: 10 QPS
At 10 QPS, ~105k orphans will take ~3 hours

Usage:
    python fix_metadata.py           # Default: 5 QPS, up to 5000/day
    python fix_metadata.py --limit N # Process only N orphans
    python fix_metadata.py --test    # Use test folders
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
import httpx

# Load .env file
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
METADATA_FILE = BASE_DIR / "image_metadata.json"
PROGRESS_FILE = BASE_DIR / "fix_metadata_progress.json"

# Test paths
TEST_IMAGES_DIR = BASE_DIR / "images_test"
TEST_METADATA_FILE = BASE_DIR / "image_metadata_test.json"
TEST_PROGRESS_FILE = BASE_DIR / "fix_metadata_progress_test.json"

# API
API_KEY = os.environ.get("ETSY_API_KEY", "")
BASE_URL = "https://openapi.etsy.com/v3/application"

# Rate limit: 5 QPS
API_DELAY = 0.5  # 5 QPS


def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)


def extract_hex_suffix(url: str) -> tuple:
    """Extract hex and suffix from Etsy CDN URL."""
    # URL format: https://i.etsystatic.com/{shop}/r/il/{hex}/{image_id}/il_{size}.{image_id}_{suffix}.jpg
    match = re.search(r"/il/([a-f0-9]+)/\d+/il_\w+\.\d+_(\w+)\.jpg", url)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_listing(client: httpx.Client, listing_id: int) -> dict:
    """Fetch listing info from Etsy API."""
    url = f"{BASE_URL}/listings/{listing_id}"
    params = {"includes": "images"}
    headers = {"x-api-key": API_KEY}

    try:
        response = client.get(url, params=params, headers=headers)
        if response.status_code == 404:
            return {"error": "404"}
        if response.status_code == 429:
            return {"error": "rate_limited"}
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"http_{e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def find_orphaned_images(images_dir: Path, metadata: dict) -> list[int]:
    """Find listing IDs for images on disk with no metadata entry."""
    orphaned = []
    for f in images_dir.glob("*.jpg"):
        try:
            listing_id = int(f.stem)
            if str(listing_id) not in metadata:
                orphaned.append(listing_id)
        except ValueError:
            pass
    return orphaned


def main():
    parser = argparse.ArgumentParser(description="Rebuild metadata for orphaned images")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N orphans")
    parser.add_argument("--test", action="store_true", help="Use test folders")
    args = parser.parse_args()

    # Set paths
    if args.test:
        images_dir = TEST_IMAGES_DIR
        metadata_file = TEST_METADATA_FILE
        progress_file = TEST_PROGRESS_FILE
    else:
        images_dir = IMAGES_DIR
        metadata_file = METADATA_FILE
        progress_file = PROGRESS_FILE

    # Check API key
    if not API_KEY:
        print("Error: ETSY_API_KEY environment variable not set")
        return

    # Load data
    metadata = load_json(metadata_file)
    progress = load_json(progress_file)
    processed = set(progress.get("processed", []))

    print(f"Loaded {len(metadata)} metadata entries")
    print(f"Already processed {len(processed)} orphans in previous runs")

    # Find orphaned images
    orphaned = find_orphaned_images(images_dir, metadata)
    orphaned = [lid for lid in orphaned if lid not in processed]

    print(f"Found {len(orphaned)} orphaned images to process")

    if args.limit > 0:
        orphaned = orphaned[:args.limit]
        print(f"Limited to {len(orphaned)} orphans")

    if not orphaned:
        print("Nothing to do")
        return

    print(f"Will process {len(orphaned)} orphans")
    print(f"Rate: {1/API_DELAY:.1f} QPS")

    # Process orphans
    stats = {"found": 0, "not_found": 0, "errors": 0, "api_calls": 0}

    with httpx.Client(timeout=30) as client:
        for i, listing_id in enumerate(orphaned):
            listing_id_str = str(listing_id)

            # Get listing from API
            result = get_listing(client, listing_id)
            stats["api_calls"] += 1

            if result is None:
                stats["errors"] += 1
                processed.add(listing_id)
            elif "error" in result:
                if result["error"] == "404":
                    # Listing unavailable - mark in metadata
                    metadata[listing_id_str] = "unavailable"
                    stats["not_found"] += 1
                elif result["error"] == "rate_limited":
                    print(f"\nRate limited at {i + 1}/{len(orphaned)}. Saving progress...")
                    break
                else:
                    metadata[listing_id_str] = result["error"]
                    stats["errors"] += 1
                processed.add(listing_id)
            else:
                # Extract info
                shop_id = result.get("shop_id")
                images = result.get("images", [])

                if images:
                    image = images[0]  # First image
                    image_id = image.get("listing_image_id")
                    image_url = image.get("url_570xN", "")
                    hex_val, suffix = extract_hex_suffix(image_url)

                    if hex_val and suffix:
                        metadata[listing_id_str] = {
                            "image_id": image_id,
                            "shop_id": shop_id,
                            "hex": hex_val,
                            "suffix": suffix
                        }
                        stats["found"] += 1
                    else:
                        metadata[listing_id_str] = "no_url_parts"
                        stats["errors"] += 1
                else:
                    metadata[listing_id_str] = "no_images"
                    stats["errors"] += 1

                processed.add(listing_id)

            # Progress
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(orphaned)} | Found: {stats['found']} | 404: {stats['not_found']} | Errors: {stats['errors']}")

            # Save periodically
            if (i + 1) % 1000 == 0:
                save_json(metadata_file, metadata)
                progress["processed"] = list(processed)
                save_json(progress_file, progress)
                print(f"  Saved checkpoint at {i + 1}")

            time.sleep(API_DELAY)

    # Final save
    save_json(metadata_file, metadata)
    progress["processed"] = list(processed)
    save_json(progress_file, progress)

    print(f"\nDone!")
    print(f"  API calls: {stats['api_calls']}")
    print(f"  Found: {stats['found']}")
    print(f"  Unavailable: {stats['not_found']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Total metadata entries: {len(metadata)}")


if __name__ == "__main__":
    main()
