"""Calculate price breaks for each taxonomy to ensure all intervals are under 10k listings."""
import os
import time
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

_key = os.getenv("ETSY_API_KEY", "")
_secret = os.getenv("ETSY_SHARED_SECRET", "")
ETSY_API_KEY = f"{_key}:{_secret}" if _secret else _key
BASE_URL = "https://openapi.etsy.com/v3"

# Crawl root taxonomy — returns ALL furniture regardless of seller tag level
TAXONOMIES = [
    (967, "Furniture"),
]

TARGET_MAX = 5000  # Target max per interval


def get_count(client: httpx.Client, taxonomy_id: int, min_price: float = None, max_price: float = None) -> int:
    """Get listing count with optional price filters."""
    params = {"taxonomy_id": taxonomy_id, "limit": 1}
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price

    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params=params,
    )
    response.raise_for_status()
    return response.json().get("count", 0)


def get_max_price(client: httpx.Client, taxonomy_id: int) -> float:
    """Get the highest listing price for a taxonomy."""
    params = {"taxonomy_id": taxonomy_id, "limit": 1,
              "sort_on": "price", "sort_order": "desc"}
    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params=params,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if results:
        return results[0]["price"]["amount"] / results[0]["price"]["divisor"]
    return 1000000


def find_price_breaks(client: httpx.Client, taxonomy_id: int, name: str) -> list[float]:
    """Find price breaks for a category. Each interval targets <5K listings.

    Start with upper=$5. Grow by 10% until >= 5K, shrink by 1% until < 5K.
    That upper becomes the next lower, next upper = lower * 1.1, repeat.

    Returns list of price points: [0, p1, p2, ..., max_price]
    """
    time.sleep(0.2)
    total = get_count(client, taxonomy_id)
    print(f"\n{name} (ID: {taxonomy_id}): {total:,} listings")

    if total <= TARGET_MAX:
        print(f"  -> No splits needed")
        return [0, 1000000]

    time.sleep(0.2)
    max_price = get_max_price(client, taxonomy_id)
    print(f"  Max price: ${max_price:,.2f}")

    price_breaks = [0]
    lower = 0
    upper = 5

    while True:
        # Grow upper by 10% until >= 5K (cap at max_price)
        while True:
            time.sleep(0.2)
            count = get_count(client, taxonomy_id,
                              min_price=lower if lower > 0 else None,
                              max_price=upper)
            print(f"    Grow: ${lower:.2f}-${upper:.2f} = {count:,}")
            if count >= TARGET_MAX or upper >= max_price:
                break
            upper *= 1.10

        # If we hit the ceiling without reaching 5K, this is the last interval
        if upper >= max_price:
            break

        # Shrink upper by 1% until < 5K
        while count >= TARGET_MAX:
            upper *= 0.99
            time.sleep(0.2)
            count = get_count(client, taxonomy_id,
                              min_price=lower if lower > 0 else None,
                              max_price=upper)
            print(f"    Shrink: ${lower:.2f}-${upper:.2f} = {count:,}")

        # Record this break
        split = round(upper, 2)
        price_breaks.append(split)
        print(f"  -> Split at ${split:.2f} ({count:,} items)")

        lower = split + 0.01
        upper = lower * 1.10

        # Next starting upper
        upper = lower * 1.10

        # Next starting upper = lower * 1.1
        upper = lower * 1.10

    price_breaks.append(1000000)
    return price_breaks


def main():
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    results = []

    with httpx.Client(timeout=30.0) as client:
        for tax_id, name in TAXONOMIES:
            breaks = find_price_breaks(client, tax_id, name)
            results.append({
                "id": tax_id,
                "name": name,
                "price_breaks": breaks
            })

    # Output as JSON config
    config = {
        "description": "Furniture taxonomy crawl configuration. Each entry has taxonomy_id, name, and price_breaks. Price breaks define intervals: [0, p1], (p1, p2], ..., (pN, 1000000]. For small categories, use [0, 1000000].",
        "taxonomies": results
    }

    output_file = "furniture_taxonomy_config.json"
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved configuration to {output_file}")
    print(f"{'='*60}")

    # Summary
    total_intervals = sum(len(r["price_breaks"]) - 1 for r in results)
    print(f"Total crawl units: {total_intervals}")


if __name__ == "__main__":
    main()
