"""Test the 3 anomalous categories."""
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

WEIRD_CATEGORIES = [
    (12405, "Standing Coat Racks"),
    (12369, "Benches"),
    (12370, "Trunks"),
]


def get_count(client: httpx.Client, taxonomy_id: int) -> int:
    """Get listing count for a taxonomy."""
    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params={"taxonomy_id": taxonomy_id, "limit": 1},
    )
    response.raise_for_status()
    return response.json().get("count", 0)


def get_listings(client: httpx.Client, taxonomy_id: int, limit: int = 5) -> list:
    """Get sample listings."""
    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params={"taxonomy_id": taxonomy_id, "limit": limit},
    )
    response.raise_for_status()
    return response.json().get("results", [])


def main():
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    with httpx.Client(timeout=30.0) as client:
        for tax_id, name in WEIRD_CATEGORIES:
            print(f"\n{'='*60}")
            print(f"{name} (ID: {tax_id})")
            print("=" * 60)

            time.sleep(0.2)
            count = get_count(client, tax_id)
            print(f"Count: {count:,}")

            time.sleep(0.2)
            listings = get_listings(client, tax_id, limit=5)
            print(f"Sample listings ({len(listings)}):")
            for l in listings:
                title = l.get("title", "")[:50]
                actual_tax = l.get("taxonomy_id")
                print(f"  - {title}... (taxonomy_id: {actual_tax})")


if __name__ == "__main__":
    main()
