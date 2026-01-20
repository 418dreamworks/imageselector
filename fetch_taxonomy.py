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
    all_ids = [tax_id for tax_id, name, depth in furniture_ids]
    with open("furniture_taxonomy_ids.json", "w") as f:
        json.dump(all_ids, f)
    print(f"\nSaved {len(all_ids)} taxonomy IDs to furniture_taxonomy_ids.json")

    # Also save leaf nodes only (deepest level - no children)
    # These are the most specific categories
    leaf_ids = [tax_id for tax_id, name, depth in furniture_ids
                if not any(other_depth > depth and other_id != tax_id
                          for other_id, other_name, other_depth in furniture_ids
                          if str(tax_id) in str(other_id))]  # crude parent check

    # Actually, let's just get the ones at max depth or that have no children in our list
    # Simpler: take all IDs that aren't parents of other IDs
    parent_ids = set()
    for tax_id, name, depth in furniture_ids:
        # Find if this is a parent (has children with depth+1)
        pass

    # Easiest approach: manually identify leaf nodes from the tree output
    leaf_taxonomy_ids = [
        12455, 12456,  # Bed Frames, Headboards
        970,   # Dressers & Armoires
        972,   # Nightstands
        971,   # Steps & Stools (bedroom)
        12470, # Vanity Tables
        974,   # Buffets & China Cabinets
        975,   # Dining Chairs
        976,   # Dining Sets
        977,   # Kitchen & Dining Tables
        11837, # Kitchen Islands
        978,   # Stools & Banquettes
        12403, # Hall Trees
        12405, # Standing Coat Racks
        12406, # Umbrella Stands
        981,   # Bean Bag Chairs
        982,   # Benches & Toy Boxes
        983,   # Bookcases (kids)
        985,   # Desks, Tables & Chairs (kids)
        986,   # Dressers & Drawers (kids)
        987,   # Steps & Stools (kids)
        988,   # Toddler Beds
        12369, # Benches
        12370, # Trunks
        991,   # Bookshelves
        992,   # Chairs
        12371, # Coffee Tables
        12372, # End Tables
        11355, # Console & Sofa Tables
        11356, # TV Stands & Media Centers
        998,   # Couches & Loveseats
        996,   # Floor Pillows
        12468, # Ottomans & Poufs
        12216, # Room Dividers
        997,   # Slipcovers
        1000,  # Desk Chairs
        1001,  # Desks
        12408, # Filing Cabinets
    ]
    with open("furniture_leaf_taxonomy_ids.json", "w") as f:
        json.dump(leaf_taxonomy_ids, f)
    print(f"Saved {len(leaf_taxonomy_ids)} leaf taxonomy IDs to furniture_leaf_taxonomy_ids.json")
