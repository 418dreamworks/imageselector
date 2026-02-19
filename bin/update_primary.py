#!/usr/bin/env python3
"""Weekly primary image rebuild.

Clears imageprimary/, extracts all primary images from imagetarred/,
then tars everything in 10K batches (no loose files remain).

Usage:
    venv/bin/python3 bin/update_primary.py
"""

import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'embedding'))
from shard_utils import get_shard_dirs, load_shard_index, SHARD_SIZE

BASE_DIR = str(Path(__file__).parent.parent)
TAR_DIR = os.path.join(BASE_DIR, "images", "imagetarred")
PRIMARY_DIR = os.path.join(BASE_DIR, "images", "imageprimary")
DB_PATH = os.path.join(BASE_DIR, "data", "db", "etsy_data.db")
BATCH_SIZE = 10000


def clear_primary_dir():
    """Remove all files (jpgs and tars) from imageprimary/."""
    print("=== Clearing imageprimary/ ===")
    removed = 0
    for fn in os.listdir(PRIMARY_DIR):
        fp = os.path.join(PRIMARY_DIR, fn)
        if os.path.isfile(fp):
            os.remove(fp)
            removed += 1
    print(f"  Removed {removed:,} files")


def extract_primaries():
    """Extract all primary images from imagetarred/ into imageprimary/."""
    print("\n=== Extracting primary images ===")

    # Count imageall tars to determine which rows are tarred
    tar_files = [f for f in os.listdir(TAR_DIR) if f.endswith(".tar")]
    num_tars = len(tar_files)
    max_row = num_tars * BATCH_SIZE
    print(f"  imageall tars: {num_tars} → rows 0 to {max_row - 1:,}")

    # Build (listing_id, image_id) → global_row mapping from all shards
    print("Loading image_index from shards...")
    row_for_pair = {}
    total_rows = 0
    for shard_dir in get_shard_dirs():
        shard_num = int(shard_dir.name.split('_')[1])
        idx = load_shard_index(shard_dir)
        for local_row, entry in enumerate(idx):
            global_row = shard_num * SHARD_SIZE + local_row
            if global_row < max_row:
                lid, iid = entry[0], entry[1]
                row_for_pair[(lid, iid)] = global_row
        total_rows += len(idx)
        del idx
    print(f"  Total index rows: {total_rows:,}, tarred rows mapped: {len(row_for_pair):,}")

    # Query DB for primary images
    print("Querying DB for primary images...")
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.execute(
        "SELECT listing_id, image_id FROM image_status WHERE is_primary = 1"
    )
    all_primaries = cursor.fetchall()
    conn.close()

    # Filter to only primaries that exist in tars
    to_extract = {}  # (lid, iid) → tar_number
    for lid, iid in all_primaries:
        if (lid, iid) in row_for_pair:
            row = row_for_pair[(lid, iid)]
            to_extract[(lid, iid)] = row // BATCH_SIZE

    print(f"  Primary images to extract: {len(to_extract):,}")

    # Group by tar number
    by_tar = {}
    for (lid, iid), tar_num in to_extract.items():
        by_tar.setdefault(tar_num, []).append((lid, iid))

    print(f"  Extracting from {len(by_tar)} tar files...")
    extracted_total = 0

    for tar_num in sorted(by_tar.keys()):
        pairs = by_tar[tar_num]
        tar_path = os.path.join(TAR_DIR, f"imageall_{tar_num:05d}.tar")

        if not os.path.exists(tar_path):
            print(f"  WARNING: {tar_path} not found, skipping {len(pairs)} images")
            continue

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
                    print(f"  WARNING: {fn} not found in tar {tar_num:05d}")

        extracted_total += extracted
        print(f"  imageall_{tar_num:05d}.tar: extracted {extracted}/{len(pairs)}")

    print(f"\nTotal extracted: {extracted_total:,}")
    return extracted_total


def tar_primary_images():
    """Tar ALL loose .jpg files in imageprimary/ in 10K batches. No loose files remain.
    After all tars are created, builds primary_index.json with byte offsets + reverse lookup."""
    print("\n=== Tarring primary images ===")
    tarnum = -1
    forward_index = {}  # tar_name -> {filename: offset}

    while True:
        loose = sorted(f for f in os.listdir(PRIMARY_DIR) if f.endswith(".jpg"))
        if len(loose) == 0:
            print("  No loose files, done.")
            break

        tarnum += 1
        batch = loose[:BATCH_SIZE]
        tar_name = f"imageprimary_{tarnum:05d}.tar"
        tar_path = os.path.join(PRIMARY_DIR, tar_name)
        list_path = f"/tmp/tar_primary_{tarnum:05d}.txt"

        with open(list_path, "w") as f:
            for fn in batch:
                f.write(fn + "\n")

        t0 = time.time()
        print(f"  Creating {tar_name} ({len(batch):,} files)...", end=" ", flush=True)
        result = subprocess.run(
            ["tar", "--no-mac-metadata", "-cf", tar_path, "-T", list_path],
            cwd=PRIMARY_DIR, capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"ERROR: {result.stderr.strip()}")
            os.unlink(list_path)
            sys.exit(1)

        tar_size = os.path.getsize(tar_path) / (1024 * 1024)
        print(f"{tar_size:.0f}MB in {elapsed:.0f}s")

        # Verify checksums (up to 50 random)
        check_count = min(50, len(batch))
        for fn in random.sample(batch, check_count):
            with open(os.path.join(PRIMARY_DIR, fn), "rb") as f:
                orig_md5 = hashlib.md5(f.read()).hexdigest()
            with tarfile.open(tar_path, "r") as tf:
                tar_md5 = hashlib.md5(tf.extractfile(fn).read()).hexdigest()
            if orig_md5 != tar_md5:
                print(f"  CHECKSUM MISMATCH: {fn}. STOPPING.")
                os.unlink(list_path)
                sys.exit(1)
        print(f"  Verified: {check_count} checksums OK")

        # Collect byte offsets for index
        offsets = {}
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                offsets[member.name] = member.offset
        forward_index[tar_name] = offsets

        for fn in batch:
            os.remove(os.path.join(PRIMARY_DIR, fn))
        print(f"  Deleted {len(batch)} loose files")

        os.unlink(list_path)

    # Build reverse lookup: listing_id -> [image_id, tar_name]
    reverse_index = {}
    for tar_name, offsets in forward_index.items():
        for filename in offsets:
            parts = filename.replace(".jpg", "").split("_")
            if len(parts) == 2:
                reverse_index[parts[0]] = [parts[1], tar_name]

    # Write primary_index.json
    index_path = os.path.join(PRIMARY_DIR, "primary_index.json")
    index_data = {"tars": forward_index, "reverse": reverse_index}
    with open(index_path, "w") as f:
        json.dump(index_data, f)
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"  Wrote primary_index.json ({size_mb:.1f}MB, {len(reverse_index):,} listings)")


if __name__ == "__main__":
    clear_primary_dir()
    extract_primaries()
    tar_primary_images()

    # Final summary
    tars = [f for f in os.listdir(PRIMARY_DIR) if f.endswith(".tar")]
    loose = [f for f in os.listdir(PRIMARY_DIR) if f.endswith(".jpg")]
    print(f"\n=== Done ===")
    print(f"  Tars: {len(tars)}")
    print(f"  Loose files: {len(loose)}")
