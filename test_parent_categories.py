"""Test parent categories for the broken ones."""
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Broken leaf nodes and their parents
CATEGORIES_TO_TEST = [
    # Entryway & Mudroom Furniture (parent of Standing Coat Racks)
    (979, "Entryway & Mudroom Furniture (parent)"),
    (12403, "Hall Trees"),
    (12405, "Standing Coat Racks (BROKEN)"),
    (12406, "Umbrella Stands"),

    # Benches & Trunks (parent of Benches, Trunks)
    (990, "Benches & Trunks (parent)"),
    (12369, "Benches (BROKEN)"),
    (12370, "Trunks (BROKEN)"),
]


def get_count_and_sample(client: httpx.Client, taxonomy_id: int) -> tuple[int, list]:
    """Get listing count and sample for a taxonomy."""
    response = client.get(
        f"{BASE_URL}/application/listings/active",
        headers={"x-api-key": ETSY_API_KEY},
        params={"taxonomy_id": taxonomy_id, "limit": 3},
    )
    response.raise_for_status()
    data = response.json()
    return data.get("count", 0), data.get("results", [])


def main():
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    with httpx.Client(timeout=30.0) as client:
        print(f"{'Category':<40} {'ID':>6}  {'Count':>12}  Sample taxonomy_ids")
        print("-" * 90)

        for tax_id, name in CATEGORIES_TO_TEST:
            time.sleep(0.2)
            count, listings = get_count_and_sample(client, tax_id)
            sample_taxs = [str(l.get("taxonomy_id")) for l in listings[:3]]
            sample_str = ", ".join(sample_taxs) if sample_taxs else "none"
            print(f"{name:<40} {tax_id:>6}  {count:>12,}  {sample_str}")


if __name__ == "__main__":
    main()
