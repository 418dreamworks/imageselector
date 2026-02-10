#!/usr/bin/env python3
"""Maintain imageprimary/ — one bg-removed primary image per listing.

Sources images only from tar archives (embedded + archived = trustworthy).
Run manually, roughly monthly.

Usage:
    venv/bin/python3 scratch/update_primary.py
"""

import json
import os
import sqlite3
import tarfile
import time
from pathlib import Path

BASE_DIR = str(Path(__file__).parent.parent)
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "image_index.json")
LOCK_PATH = INDEX_PATH + ".lock"
TAR_DIR = os.path.join(BASE_DIR, "images", "imageall_tars")
PRIMARY_DIR = os.path.join(BASE_DIR, "images", "imageprimary")
DB_PATH = os.path.join(BASE_DIR, "data", "db", "etsy_data.db")
BATCH_SIZE = 10000


def load_index_safe():
    """Load image_index.json, waiting if another process holds the lock."""
    while True:
        if os.path.exists(LOCK_PATH):
            print("image_index.json is locked, waiting...")
            time.sleep(1)
            continue
        try:
            with open(INDEX_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            print("image_index.json partial read, retrying...")
            time.sleep(1)


def main():
    # Step 1: Load index
    print("Loading image_index.json...")
    idx = load_index_safe()
    print(f"  Total index rows: {len(idx):,}")

    # Step 2: Count tars, limit to tarred rows only
    tar_files = [f for f in os.listdir(TAR_DIR) if f.endswith(".tar")]
    num_tars = len(tar_files)
    max_row = num_tars * BATCH_SIZE
    print(f"  Tar files: {num_tars} → considering rows 0 to {max_row - 1:,}")

    tarred_idx = idx[:max_row]

    # Build set of all tarred (listing_id, image_id) and map listing→row
    tarred_pairs = set()
    row_for_pair = {}  # (lid, iid) → row number (for tar lookup)
    for row, entry in enumerate(tarred_idx):
        lid, iid = entry[0], entry[1]
        tarred_pairs.add((lid, iid))
        row_for_pair[(lid, iid)] = row

    print(f"  Tarred images: {len(tarred_pairs):,}")

    # Step 3: Query DB for primary images among tarred set
    print("Querying DB for primary images...")
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")

    # Get all primary images
    cursor = conn.execute(
        "SELECT listing_id, image_id FROM image_status WHERE is_primary = 1"
    )
    all_primaries = cursor.fetchall()
    conn.close()

    # Filter to only tarred primaries
    tarred_primaries = {}  # listing_id → image_id
    for lid, iid in all_primaries:
        if (lid, iid) in tarred_pairs:
            tarred_primaries[lid] = iid

    print(f"  Listings with tarred primary image: {len(tarred_primaries):,}")

    # Step 4: Scan existing imageprimary/ files
    print("Scanning imageprimary/...")
    existing = {}  # listing_id → image_id (from filename)
    for fn in os.listdir(PRIMARY_DIR):
        if not fn.endswith(".jpg"):
            continue
        parts = fn[:-4].split("_")  # strip .jpg, split on _
        if len(parts) == 2:
            try:
                existing[int(parts[0])] = int(parts[1])
            except ValueError:
                continue

    print(f"  Existing primary files: {len(existing):,}")

    # Step 5: Determine what to add, update, and remove
    to_extract = {}  # (listing_id, image_id) → tar_number
    to_remove = []   # filenames to remove

    # Check each tarred primary
    for lid, iid in tarred_primaries.items():
        if lid in existing:
            if existing[lid] == iid:
                # Already correct, skip
                continue
            else:
                # Wrong image_id — remove old, extract new
                old_fn = f"{lid}_{existing[lid]}.jpg"
                to_remove.append(old_fn)
                row = row_for_pair[(lid, iid)]
                to_extract[(lid, iid)] = row // BATCH_SIZE
        else:
            # Missing — extract
            row = row_for_pair[(lid, iid)]
            to_extract[(lid, iid)] = row // BATCH_SIZE

    # Files in imageprimary/ whose listing has no tarred primary → remove
    for lid, iid in existing.items():
        if lid not in tarred_primaries:
            to_remove.append(f"{lid}_{iid}.jpg")

    print(f"\n  To extract: {len(to_extract):,}")
    print(f"  To remove:  {len(to_remove):,}")
    print(f"  Already OK: {len(tarred_primaries) - len(to_extract):,}")

    # Step 6: Remove stale files
    if to_remove:
        print(f"\nRemoving {len(to_remove)} stale files...")
        for fn in to_remove:
            os.remove(os.path.join(PRIMARY_DIR, fn))
        print(f"  Removed {len(to_remove)} files")

    # Step 7: Extract needed images, grouped by tar number
    if to_extract:
        # Group by tar number
        by_tar = {}  # tar_number → list of (listing_id, image_id)
        for (lid, iid), tar_num in to_extract.items():
            by_tar.setdefault(tar_num, []).append((lid, iid))

        print(f"\nExtracting from {len(by_tar)} tar files...")
        extracted_total = 0

        for tar_num in sorted(by_tar.keys()):
            pairs = by_tar[tar_num]
            tar_path = os.path.join(TAR_DIR, f"imageall_{tar_num:05d}.tar")

            if not os.path.exists(tar_path):
                print(f"  WARNING: {tar_path} not found, skipping {len(pairs)} images")
                continue

            # Build set of filenames to extract
            needed = {f"{lid}_{iid}.jpg": (lid, iid) for lid, iid in pairs}

            extracted = 0
            with tarfile.open(tar_path, "r") as tf:
                for fn in needed:
                    try:
                        member = tf.getmember(fn)
                        with tf.extractfile(member) as src:
                            data = src.read()
                        dst_path = os.path.join(PRIMARY_DIR, fn)
                        with open(dst_path, "wb") as dst:
                            dst.write(data)
                        extracted += 1
                    except KeyError:
                        lid, iid = needed[fn]
                        print(f"  WARNING: {fn} not found in tar {tar_num:05d}")

            extracted_total += extracted
            print(f"  imageall_{tar_num:05d}.tar: extracted {extracted}/{len(pairs)}")

        print(f"\nTotal extracted: {extracted_total:,}")

    # Summary
    final_count = len([f for f in os.listdir(PRIMARY_DIR) if f.endswith(".jpg")])
    print(f"\n=== Summary ===")
    print(f"  imageprimary/ files: {final_count:,}")
    print(f"  Expected (tarred primaries): {len(tarred_primaries):,}")
    if final_count == len(tarred_primaries):
        print("  ✓ Counts match")
    else:
        print(f"  ✗ Mismatch: {len(tarred_primaries) - final_count:,} missing")


if __name__ == "__main__":
    main()
