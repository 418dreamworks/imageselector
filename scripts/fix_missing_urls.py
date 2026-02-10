#!/usr/bin/env python3
"""One-shot script: repair images with download_done=0 and no URL."""
import os, sys, time, sqlite3
from pathlib import Path
from dotenv import load_dotenv
import httpx

load_dotenv()

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from image_db import get_connection, commit_with_retry

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

conn = get_connection()
rows = conn.execute("""
    SELECT DISTINCT listing_id FROM image_status
    WHERE download_done = 0 AND (url IS NULL OR url = '')
""").fetchall()
listing_ids = [r[0] for r in rows]
print(f"Found {len(listing_ids)} listings with missing URLs")

if not listing_ids:
    conn.close()
    sys.exit(0)

repaired = 0
with httpx.Client(timeout=30) as client:
    for i in range(0, len(listing_ids), 100):
        batch = listing_ids[i:i+100]
        time.sleep(0.2)
        try:
            resp = client.get(
                f"{BASE_URL}/application/listings/batch",
                headers={"x-api-key": ETSY_API_KEY},
                params={"listing_ids": ",".join(str(lid) for lid in batch), "includes": "Images"},
            )
            resp.raise_for_status()
            for listing in resp.json().get("results", []):
                lid = listing.get("listing_id")
                for img in listing.get("images", []):
                    iid = img.get("listing_image_id")
                    url = img.get("url_570xN")
                    if iid and url:
                        cur = conn.execute("""
                            UPDATE image_status SET url = ?
                            WHERE listing_id = ? AND image_id = ? AND (url IS NULL OR url = '')
                        """, (url, lid, iid))
                        repaired += cur.rowcount
            commit_with_retry(conn)
            print(f"  Batch {i//100 + 1}: {len(batch)} listings, {repaired} URLs repaired so far")
        except Exception as e:
            print(f"  Error on batch {i//100 + 1}: {e}")

conn.close()
print(f"Done. Repaired {repaired} URLs.")
