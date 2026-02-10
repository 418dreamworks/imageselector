"""Calculate price breaks for each taxonomy to ensure all intervals are under 10k listings."""
import os
import time
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Categories to calculate - includes parent replacements for broken IDs
TAXONOMIES = [
    (12455, "Bed Frames"),
    (12456, "Headboards"),
    (970, "Dressers & Armoires"),
    (972, "Nightstands"),
    (971, "Steps & Stools (bedroom)"),
    (12470, "Vanity Tables"),
    (974, "Buffets & China Cabinets"),
    (975, "Dining Chairs"),
    (976, "Dining Sets"),
    (977, "Kitchen & Dining Tables"),
    (11837, "Kitchen Islands"),
    (978, "Stools & Banquettes"),
    (12403, "Hall Trees"),
    (979, "Entryway & Mudroom Furniture"),  # Parent - replaces broken 12405
    (12406, "Umbrella Stands"),
    (981, "Bean Bag Chairs"),
    (982, "Benches & Toy Boxes"),
    (983, "Bookcases (kids)"),
    (985, "Desks, Tables & Chairs (kids)"),
    (986, "Dressers & Drawers (kids)"),
    (987, "Steps & Stools (kids)"),
    (988, "Toddler Beds"),
    (990, "Benches & Trunks"),  # Parent - replaces broken 12369, 12370
    (991, "Bookshelves"),
    (992, "Chairs"),
    (12371, "Coffee Tables"),
    (12372, "End Tables"),
    (11355, "Console & Sofa Tables"),
    (11356, "TV Stands & Media Centers"),
    (998, "Couches & Loveseats"),
    (996, "Floor Pillows"),
    (12468, "Ottomans & Poufs"),
    (12216, "Room Dividers"),
    (997, "Slipcovers"),
    (1000, "Desk Chairs"),
    (1001, "Desks"),
    (12408, "Filing Cabinets"),
]

TARGET_MAX = 8000  # Target max per interval


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


def find_price_breaks(client: httpx.Client, taxonomy_id: int, name: str) -> list[float]:
    """Find price breaks to split a category into intervals under TARGET_MAX each.

    Returns list of price points: [0, p1, p2, ..., 1000000]
    """
    time.sleep(0.2)
    total = get_count(client, taxonomy_id)

    print(f"\n{name} (ID: {taxonomy_id}): {total:,} listings")

    if total <= TARGET_MAX:
        print(f"  -> No splits needed")
        return [0, 1000000]

    # Need to split - use binary search to find price points
    price_breaks = [0]
    current_min = 0

    while True:
        # How many left from current_min to infinity?
        time.sleep(0.2)
        remaining = get_count(client, taxonomy_id, min_price=current_min if current_min > 0 else None)

        if remaining <= TARGET_MAX:
            # Done - this last segment fits
            break

        # Binary search for a price that gives us ~TARGET_MAX items
        low = current_min
        high = 100000  # Start with reasonable high

        # First find a high that's actually high enough
        time.sleep(0.2)
        count_at_high = get_count(client, taxonomy_id,
                                   min_price=current_min if current_min > 0 else None,
                                   max_price=high)
        while count_at_high >= remaining:
            high *= 2
            time.sleep(0.2)
            count_at_high = get_count(client, taxonomy_id,
                                       min_price=current_min if current_min > 0 else None,
                                       max_price=high)
            if high > 10000000:
                break

        # Binary search
        for _ in range(25):  # Max iterations
            mid = (low + high) / 2
            time.sleep(0.2)
            count_below_mid = get_count(client, taxonomy_id,
                                         min_price=current_min if current_min > 0 else None,
                                         max_price=mid)

            print(f"    Searching: ${current_min:.0f}-${mid:.0f} = {count_below_mid:,}")

            if abs(count_below_mid - TARGET_MAX) < 500:
                # Close enough
                break
            elif count_below_mid < TARGET_MAX:
                low = mid
            else:
                high = mid

        # Round to nice number
        split_price = round(mid, -1)  # Round to nearest 10
        if split_price < 10:
            split_price = round(mid, 0)

        price_breaks.append(split_price)
        print(f"  -> Split at ${split_price:.0f} ({count_below_mid:,} items)")
        current_min = split_price + 0.01

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
