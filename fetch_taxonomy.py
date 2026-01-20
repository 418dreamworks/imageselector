"""Fetch and save Etsy taxonomy for furniture category."""
import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

def fetch_taxonomy():
    """Fetch full seller taxonomy from Etsy."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{BASE_URL}/application/seller-taxonomy/nodes",
            headers={"x-api-key": ETSY_API_KEY},
        )
        response.raise_for_status()
        return response.json()

def find_furniture_subtaxonomies(taxonomy_data):
    """Find furniture (967) and all its descendants."""
    results = taxonomy_data.get("results", [])

    # Build a map of id -> node
    node_map = {}
    def build_map(nodes):
        for node in nodes:
            node_map[node["id"]] = node
            if node.get("children"):
                build_map(node["children"])
    build_map(results)

    # Find furniture node (967)
    furniture_node = node_map.get(967)
    if not furniture_node:
        print("Furniture taxonomy (967) not found!")
        return []

    # Collect all descendant IDs
    def collect_descendants(node, depth=0):
        ids = [(node["id"], node["name"], depth)]
        for child in node.get("children", []):
            ids.extend(collect_descendants(child, depth + 1))
        return ids

    return collect_descendants(furniture_node)

if __name__ == "__main__":
    print("Fetching Etsy taxonomy...")
    taxonomy = fetch_taxonomy()

    print("Finding furniture subcategories...")
    furniture_ids = find_furniture_subtaxonomies(taxonomy)

    print(f"\nFound {len(furniture_ids)} furniture taxonomy IDs:\n")
    for tax_id, name, depth in furniture_ids:
        indent = "  " * depth
        print(f"{indent}{tax_id}: {name}")

    # Save just the IDs for use in sync script
    leaf_ids = [tax_id for tax_id, name, depth in furniture_ids]
    with open("furniture_taxonomy_ids.json", "w") as f:
        json.dump(leaf_ids, f)
    print(f"\nSaved {len(leaf_ids)} taxonomy IDs to furniture_taxonomy_ids.json")
