"""
Fix metadata by adding hex and suffix fields for entries missing them.
Batch fetches from API (100 listings per call) to get image URLs,
then extracts hex and suffix from the URLs.
"""

import os
import json
import re
import time
import urllib.request
from pathlib import Path

# Read API key from .env file directly
def load_api_key():
    env_file = Path(__file__).parent / ".env"
    with open(env_file) as f:
        for line in f:
            if line.startswith("ETSY_API_KEY"):
                return line.split("=", 1)[1].strip()
    raise ValueError("ETSY_API_KEY not found in .env")

API_KEY = load_api_key()
BASE_URL = "https://openapi.etsy.com/v3"
METADATA_FILE = Path(__file__).parent / "image_metadata.json"
BATCH_SIZE = 100
API_DELAY = 0.1  # 10 QPS to be safe


def extract_hex_suffix(url: str) -> tuple[str, str]:
    """Extract hex and suffix from Etsy image URL."""
    # URL format: https://i.etsystatic.com/{shop}/r/il/{hex}/{image_id}/il_{size}.{image_id}_{suffix}.jpg
    match = re.search(r'/il/([a-f0-9]+)/(\d+)/il_[^.]+\.\d+_([a-z0-9]+)\.jpg', url)
    if match:
        return match.group(1), match.group(3)
    return None, None


def get_batch_image_urls(listing_ids: list[int]) -> dict:
    """Fetch image URLs for up to 100 listings in one API call."""
    if not listing_ids:
        return {}

    ids_param = ",".join(str(lid) for lid in listing_ids)
    url = f"{BASE_URL}/application/listings/batch?listing_ids={ids_param}&includes=Images"

    req = urllib.request.Request(url, headers={"x-api-key": API_KEY})
    try:
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())

        results = {}
        for listing in data.get("results", []):
            lid = listing.get("listing_id")
            images = listing.get("images", [])
            if images:
                results[lid] = images[0].get("url_570xN")
        return results
    except Exception as e:
        print(f"  Error fetching batch: {e}")
        return {}


def main():
    # Load metadata
    print("Loading metadata...")
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    # Find entries missing hex/suffix
    needs_fix = []
    already_good = 0
    extracted_from_url = 0

    for lid_str, entry in metadata.items():
        if isinstance(entry, dict):
            if entry.get("hex") and entry.get("suffix"):
                already_good += 1
            elif entry.get("url"):
                # Has URL, can extract hex/suffix from it
                hex_val, suffix = extract_hex_suffix(entry["url"])
                if hex_val and suffix:
                    entry["hex"] = hex_val
                    entry["suffix"] = suffix
                    extracted_from_url += 1
                else:
                    needs_fix.append(int(lid_str))
            else:
                # No URL, needs API call
                needs_fix.append(int(lid_str))

    print(f"Already have hex/suffix: {already_good}")
    print(f"Extracted from existing URL: {extracted_from_url}")
    print(f"Need to fetch from API: {len(needs_fix)}")

    # Save if we extracted from URLs
    if extracted_from_url > 0:
        print("Saving extracted hex/suffix...")
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f)

    if not needs_fix:
        print("Nothing more to fix!")
        return

    # Batch fetch URLs
    fixed = 0
    errors = 0
    for i in range(0, len(needs_fix), BATCH_SIZE):
        batch = needs_fix[i:i + BATCH_SIZE]

        time.sleep(API_DELAY)
        urls = get_batch_image_urls(batch)

        for lid in batch:
            lid_str = str(lid)
            if lid in urls and urls[lid]:
                hex_val, suffix = extract_hex_suffix(urls[lid])
                if hex_val and suffix:
                    if isinstance(metadata[lid_str], dict):
                        metadata[lid_str]["hex"] = hex_val
                        metadata[lid_str]["suffix"] = suffix
                    fixed += 1
                else:
                    errors += 1
            else:
                errors += 1

        # Progress
        done = i + len(batch)
        print(f"  Progress: {done}/{len(needs_fix)} ({fixed} fixed, {errors} errors)")

        # Save periodically
        if done % 1000 == 0:
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f)
            print("  Saved checkpoint")

    # Final save
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f)

    print(f"\nDone! Fixed {fixed}, errors {errors}")


if __name__ == "__main__":
    main()
