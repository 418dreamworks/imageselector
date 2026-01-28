"""Test runner: 2 smallest taxonomy categories with fresh data."""
import sync_data_new as sd
from pathlib import Path

# Use test paths (fresh/empty)
BASE = Path(__file__).parent
sd.IMAGES_DIR = BASE / "images_test"
sd.METADATA_FILE = BASE / "image_metadata_test.json"
sd.PROGRESS_FILE = BASE / "sync_progress_test.json"
sd.DB_FILE = BASE / "etsy_data_test.db"

sd.IMAGES_DIR.mkdir(exist_ok=True)

# Limit to 2 smallest categories: Filing Cabinets (12408), Hall Trees (12403)
sd.CRAWL_UNITS = [u for u in sd.CRAWL_UNITS if u["taxonomy_id"] in (12408, 12403)]

print(f"Test config: {len(sd.CRAWL_UNITS)} crawl units")
print(f"  Images: {sd.IMAGES_DIR}")
print(f"  Metadata: {sd.METADATA_FILE}")
print(f"  DB: {sd.DB_FILE}")
print()

sd.main()
