"""Test price-based splitting for Dining Chairs category."""
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"
DINING_CHAIRS_ID = 975  # 15,871 listings


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


def get_listings(client: httpx.Client, taxonomy_id: int, min_price: float = None, max_price: float = None,
                 offset: int = 0, limit: int = 10) -> list:
    """Get listings with optional price filters."""
    params = {"taxonomy_id": taxonomy_id, "limit": limit, "offset": offset}
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
    return response.json().get("results", [])


def find_split_price(client: httpx.Client, taxonomy_id: int, target_per_half: int = 7500) -> float:
    """Binary search to find price that splits listings roughly in half."""
    total = get_count(client, taxonomy_id)
    print(f"Total listings: {total}")
    print(f"Target per half: ~{target_per_half}")

    # Start with a reasonable range
    low_price = 0
    high_price = 10000  # $10,000 max

    # Binary search for the split point
    for i in range(20):  # Max 20 iterations
        mid_price = (low_price + high_price) / 2
        time.sleep(0.2)
        count_below = get_count(client, taxonomy_id, max_price=mid_price)
        count_above = total - count_below

        print(f"  ${mid_price:.2f}: below={count_below}, above={count_above}")

        if abs(count_below - target_per_half) < 500:  # Close enough
            return mid_price

        if count_below < target_per_half:
            low_price = mid_price
        else:
            high_price = mid_price

    return mid_price


def main():
    if not ETSY_API_KEY:
        print("Error: ETSY_API_KEY not set in .env")
        return

    with httpx.Client(timeout=30.0) as client:
        print("=" * 60)
        print("Finding split price for Dining Chairs...")
        print("=" * 60)

        split_price = find_split_price(client, DINING_CHAIRS_ID, target_per_half=7500)
        print(f"\nSplit price: ${split_price:.2f}")

        # Verify counts
        time.sleep(0.2)
        count_low = get_count(client, DINING_CHAIRS_ID, max_price=split_price)
        time.sleep(0.2)
        count_high = get_count(client, DINING_CHAIRS_ID, min_price=split_price + 0.01)

        print(f"\nLower range ($0 - ${split_price:.2f}): {count_low} listings")
        print(f"Upper range (${split_price + 0.01:.2f}+): {count_high} listings")

        # Test 4 requests: first 10 and last 10 of each range
        print("\n" + "=" * 60)
        print("Testing 4 requests (first 10 & last 10 of each range)...")
        print("=" * 60)

        # Request 1: First 10 of lower range
        time.sleep(0.2)
        lower_first = get_listings(client, DINING_CHAIRS_ID, max_price=split_price, offset=0, limit=10)
        print(f"\n1. Lower range, first 10 (offset=0): {len(lower_first)} results")
        if lower_first:
            print(f"   Prices: ${lower_first[0].get('price', {}).get('amount', 0)/100:.2f} - ${lower_first[-1].get('price', {}).get('amount', 0)/100:.2f}")

        # Request 2: Last 10 of lower range (offset near count_low)
        last_offset_low = max(0, count_low - 10)
        time.sleep(0.2)
        lower_last = get_listings(client, DINING_CHAIRS_ID, max_price=split_price, offset=last_offset_low, limit=10)
        print(f"\n2. Lower range, last 10 (offset={last_offset_low}): {len(lower_last)} results")
        if lower_last:
            print(f"   Prices: ${lower_last[0].get('price', {}).get('amount', 0)/100:.2f} - ${lower_last[-1].get('price', {}).get('amount', 0)/100:.2f}")

        # Request 3: First 10 of upper range
        time.sleep(0.2)
        upper_first = get_listings(client, DINING_CHAIRS_ID, min_price=split_price + 0.01, offset=0, limit=10)
        print(f"\n3. Upper range, first 10 (offset=0): {len(upper_first)} results")
        if upper_first:
            print(f"   Prices: ${upper_first[0].get('price', {}).get('amount', 0)/100:.2f} - ${upper_first[-1].get('price', {}).get('amount', 0)/100:.2f}")

        # Request 4: Last 10 of upper range (offset near count_high)
        last_offset_high = max(0, count_high - 10)
        time.sleep(0.2)
        upper_last = get_listings(client, DINING_CHAIRS_ID, min_price=split_price + 0.01, offset=last_offset_high, limit=10)
        print(f"\n4. Upper range, last 10 (offset={last_offset_high}): {len(upper_last)} results")
        if upper_last:
            print(f"   Prices: ${upper_last[0].get('price', {}).get('amount', 0)/100:.2f} - ${upper_last[-1].get('price', {}).get('amount', 0)/100:.2f}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        all_success = all([lower_first, lower_last, upper_first, upper_last])
        print(f"All 4 requests returned results: {'YES' if all_success else 'NO'}")

        if last_offset_low < 10000 and last_offset_high < 10000:
            print(f"Both ranges stay under 10k offset limit: YES")
        else:
            print(f"WARNING: One or both ranges exceed 10k offset limit")


if __name__ == "__main__":
    main()
