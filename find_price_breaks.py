"""Find price breaks by recursively halving until each segment is under 8k."""
import os
import time
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

TARGET_MAX = 8000


def get_count(client: httpx.Client, taxonomy_id: int, min_price: float = None, max_price: float = None) -> int:
    """Get listing count with optional price filters."""
    params = {"taxonomy_id": taxonomy_id, "limit": 1}
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price

    time.sleep(0.15)
    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params=params,
    )
    response.raise_for_status()
    return response.json().get("count", 0)


def find_half_price(client: httpx.Client, taxonomy_id: int, min_p: float, max_p: float, total: int) -> float:
    """Binary search to find price that splits the range roughly in half."""
    target = total // 2
    low, high = min_p, max_p

    for _ in range(15):  # Max iterations
        mid = (low + high) / 2
        count_below = get_count(client, taxonomy_id,
                                 min_price=min_p if min_p > 0 else None,
                                 max_price=mid)

        print(f"      ${min_p:.0f}-${mid:.0f}: {count_below:,} (target: ~{target:,})")

        if abs(count_below - target) < 200:  # Close enough
            break
        elif count_below < target:
            low = mid
        else:
            high = mid

    # Round to nice number
    return round(mid, 0)


def split_interval(client: httpx.Client, taxonomy_id: int, min_p: float, max_p: float, depth: int = 0) -> list[float]:
    """Recursively split interval until each piece is under TARGET_MAX.

    Returns list of internal breakpoints (not including min_p and max_p).
    """
    indent = "  " * depth
    count = get_count(client, taxonomy_id,
                      min_price=min_p if min_p > 0 else None,
                      max_price=max_p if max_p < 1000000 else None)

    print(f"{indent}[${min_p:.0f} - ${max_p:.0f}]: {count:,} listings")

    if count <= TARGET_MAX:
        print(f"{indent}  -> OK (under {TARGET_MAX:,})")
        return []

    # Need to split - find the half point
    print(f"{indent}  -> Splitting...")
    half = find_half_price(client, taxonomy_id, min_p, max_p, count)

    # Recursively split each half
    left_breaks = split_interval(client, taxonomy_id, min_p, half, depth + 1)
    right_breaks = split_interval(client, taxonomy_id, half, max_p, depth + 1)

    return left_breaks + [half] + right_breaks


def find_price_breaks(client: httpx.Client, taxonomy_id: int, name: str) -> list[float]:
    """Find all price breaks for a taxonomy."""
    print(f"\n{'='*60}")
    print(f"{name} (ID: {taxonomy_id})")
    print("=" * 60)

    # Get total
    total = get_count(client, taxonomy_id)
    print(f"Total: {total:,} listings")

    if total <= TARGET_MAX:
        print("No splits needed!")
        return [0, 1000000]

    internal_breaks = split_interval(client, taxonomy_id, 0, 1000000)

    # Construct full break list
    breaks = [0] + sorted(internal_breaks) + [1000000]

    print(f"\nFinal price breaks: {breaks}")
    return breaks


# All categories - with parent replacements for broken IDs
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
                "price_breaks": [int(b) if b == int(b) else b for b in breaks]
            })

    # Save to JSON
    config = {
        "description": "Furniture taxonomy crawl config. price_breaks defines intervals for crawling.",
        "taxonomies": results
    }

    with open("furniture_taxonomy_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("Saved to furniture_taxonomy_config.json")

    # Summary
    total_intervals = sum(len(r["price_breaks"]) - 1 for r in results)
    print(f"Total crawl units: {total_intervals}")


if __name__ == "__main__":
    main()
