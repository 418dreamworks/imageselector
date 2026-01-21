"""Count total listings in each furniture leaf taxonomy."""
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Leaf taxonomy IDs and names (from sync_images.py)
FURNITURE_TAXONOMIES = [
    12455, 12456, 970, 972, 971, 12470, 974, 975, 976, 977,
    11837, 978, 12403, 12405, 12406, 981, 982, 983, 985, 986,
    987, 988, 12369, 12370, 991, 992, 12371, 12372, 11355, 11356,
    998, 996, 12468, 12216, 997, 1000, 1001, 12408,
]

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


def get_listing_count(client: httpx.Client, taxonomy_id: int) -> int:
    """Get total listing count for a taxonomy."""
    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params={"taxonomy_id": taxonomy_id, "limit": 1},
    )
    response.raise_for_status()
    return response.json().get("count", 0)


def main():
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    print(f"{'Taxonomy':<35} {'ID':>6}  {'Count':>8}  {'Exceeds 10k?'}")
    print("-" * 70)

    total = 0
    over_10k = 0
    over_50k = 0

    with httpx.Client(timeout=30.0) as client:
        for tax_id in FURNITURE_TAXONOMIES:
            time.sleep(0.2)  # Be gentle on API
            count = get_listing_count(client, tax_id)
            name = TAXONOMY_NAMES.get(tax_id, str(tax_id))
            exceeds = "YES" if count > 10000 else ""
            if count > 50000:
                exceeds = "YES (>50k)"
            print(f"{name:<35} {tax_id:>6}  {count:>8,}  {exceeds}")
            total += count
            if count > 10000:
                over_10k += 1
            if count > 50000:
                over_50k += 1

    print("-" * 70)
    print(f"{'TOTAL':<35} {'':>6}  {total:>8,}")
    print(f"\nCategories exceeding 10k offset limit: {over_10k}")
    print(f"Categories exceeding 50k (5 sorts × 10k): {over_50k}")
    print(f"\nWith 5 sort strategies, max reachable per category: 50,000")
    print(f"Theoretical max reachable: {len(FURNITURE_TAXONOMIES) * 50000:,}")


if __name__ == "__main__":
    main()
