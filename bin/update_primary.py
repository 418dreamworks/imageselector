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
    """Extract all primary images from imagetarred/ into imageprimary/.

    Uses tar_index.json reverse lookup for O(1) image location.
    """
    print("\n=== Extracting primary images ===")

    # Load tar_index.json reverse lookup
    index_path = os.path.join(TAR_DIR, "tar_index.json")
    print("Loading tar_index.json...")
    with open(index_path, "r") as f:
        tar_index = json.load(f)
    reverse = tar_index["reverse"]
    print(f"  {len(reverse):,} images indexed")

    # Query DB for primary images
    print("Querying DB for primary images...")
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.execute(
        "SELECT listing_id, image_id FROM image_status WHERE is_primary = 1"
    )
    all_primaries = cursor.fetchall()
    conn.close()

    # Look up each primary in reverse index, group by tar
    by_tar = {}  # tar_name -> [(filename, offset), ...]
    skipped = 0
    for lid, iid in all_primaries:
        key = f"{lid}_{iid}"
        if key in reverse:
            tar_name, offset = reverse[key]
            by_tar.setdefault(tar_name, []).append((f"{key}.jpg", offset))
        else:
            skipped += 1

    total_to_extract = sum(len(v) for v in by_tar.values())
    print(f"  Primary images to extract: {total_to_extract:,} (skipped {skipped:,} not in tars)")
    print(f"  Extracting from {len(by_tar)} tar files...")
    extracted_total = 0

    for tar_name in sorted(by_tar.keys()):
        entries = by_tar[tar_name]
        tar_path = os.path.join(TAR_DIR, tar_name)

        if not os.path.exists(tar_path):
            print(f"  WARNING: {tar_path} not found, skipping {len(entries)} images")
            continue

        extracted = 0
        with tarfile.open(tar_path, "r") as tf:
            for fn, offset in entries:
                try:
                    tf.fileobj.seek(offset)
                    member = tarfile.TarInfo.fromtarfile(tf)
                    with tf.extractfile(member) as src:
                        data = src.read()
                    dst_path = os.path.join(PRIMARY_DIR, fn)
                    with open(dst_path, "wb") as dst:
                        dst.write(data)
                    extracted += 1
                except Exception as e:
                    print(f"  WARNING: {fn} failed in {tar_name}: {e}")

        extracted_total += extracted
        print(f"  {tar_name}: extracted {extracted}/{len(entries)}")

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
