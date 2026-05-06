#!/usr/bin/env python3
"""Append-only primary image update.

Finds new listings not yet in primary_index.json, extracts their primary
images from imagetarred/, and appends to new tar files. Existing tars are
never touched. Falls back to full rebuild if no existing index found.

Usage:
    venv/bin/python3 bin/update_primary.py
    venv/bin/python3 bin/update_primary.py --full   # force full rebuild
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
INDEX_PATH = os.path.join(PRIMARY_DIR, "primary_index.json")


def load_primary_index():
    """Load existing primary_index.json. Returns None if not found."""
    if not os.path.exists(INDEX_PATH):
        return None
    with open(INDEX_PATH) as f:
        return json.load(f)


def get_db_primaries():
    """Query DB for all current primary images. Returns dict: listing_id -> image_id."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    rows = conn.execute(
        "SELECT listing_id, image_id FROM image_status WHERE is_primary = 1"
    ).fetchall()
    conn.close()
    return {str(lid): str(iid) for lid, iid in rows}


def find_new_primaries(db_primaries, existing_index):
    """Find listings not yet in the primary index. Returns list of (lid, iid) to extract."""
    if existing_index is None:
        return [(lid, iid) for lid, iid in db_primaries.items()]

    existing_reverse = existing_index.get("reverse", {})
    return [(lid, iid) for lid, iid in db_primaries.items() if lid not in existing_reverse]


def extract_primaries(primaries_to_extract):
    """Extract specific primary images from imagetarred/ into imageprimary/.

    Args:
        primaries_to_extract: list of (listing_id_str, image_id_str)
    """
    print(f"\n=== Extracting {len(primaries_to_extract):,} new primary images ===")
    if not primaries_to_extract:
        print("  Nothing to extract")
        return 0

    # Load tar_index.json reverse lookup
    index_path = os.path.join(TAR_DIR, "tar_index.json")
    print("Loading tar_index.json...")
    with open(index_path, "r") as f:
        tar_index = json.load(f)
    reverse = tar_index["reverse"]

    # Look up each primary in reverse index, group by tar
    by_tar = {}  # tar_name -> [(src_filename, dst_filename, offset), ...]
    skipped = 0
    for lid, iid in primaries_to_extract:
        key = f"{lid}_{iid}"
        if key in reverse:
            tar_name, offset = reverse[key]
            by_tar.setdefault(tar_name, []).append((f"{key}.jpg", f"{lid}.jpg", offset))
        else:
            skipped += 1

    total_to_extract = sum(len(v) for v in by_tar.values())
    print(f"  Found in tars: {total_to_extract:,} (skipped {skipped:,} not in tars)")
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
            for src_fn, dst_fn, offset in entries:
                try:
                    tf.fileobj.seek(offset)
                    member = tarfile.TarInfo.fromtarfile(tf)
                    with tf.extractfile(member) as src:
                        data = src.read()
                    dst_path = os.path.join(PRIMARY_DIR, dst_fn)
                    with open(dst_path, "wb") as dst:
                        dst.write(data)
                    extracted += 1
                except Exception as e:
                    print(f"  WARNING: {src_fn} failed in {tar_name}: {e}")

        extracted_total += extracted
        print(f"  {tar_name}: extracted {extracted}/{len(entries)}")

    print(f"\nTotal extracted: {extracted_total:,}")
    return extracted_total


def tar_loose_images(existing_index):
    """Tar loose .jpg files in imageprimary/ into new tar batches.

    Continues numbering from existing tars. Updates and returns the full index."""
    loose = sorted(f for f in os.listdir(PRIMARY_DIR) if f.endswith(".jpg"))
    if not loose:
        print("\n  No new loose files to tar")
        return existing_index if existing_index else {"tars": {}, "reverse": {}}

    print(f"\n=== Tarring {len(loose):,} new primary images ===")

    # Start numbering after existing tars
    if existing_index and existing_index.get("tars"):
        last_tar = max(existing_index["tars"].keys())
        tarnum = int(last_tar.replace("imageprimary_", "").replace(".tar", ""))
    else:
        tarnum = -1

    forward_index = dict(existing_index["tars"]) if existing_index else {}
    reverse_index = dict(existing_index["reverse"]) if existing_index else {}

    while loose:
        tarnum += 1
        batch = loose[:BATCH_SIZE]
        loose = loose[BATCH_SIZE:]
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

        # Collect byte offsets
        offsets = {}
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                offsets[member.name] = member.offset
        forward_index[tar_name] = offsets

        # Update reverse index
        for filename, offset in offsets.items():
            listing_id = filename.replace(".jpg", "")
            reverse_index[listing_id] = [tar_name, offset]

        for fn in batch:
            os.remove(os.path.join(PRIMARY_DIR, fn))
        print(f"  Deleted {len(batch)} loose files")

        os.unlink(list_path)

    # Write updated primary_index.json
    index_data = {"tars": forward_index, "reverse": reverse_index}
    with open(INDEX_PATH, "w") as f:
        json.dump(index_data, f)
    size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"  Wrote primary_index.json ({size_mb:.1f}MB, {len(reverse_index):,} listings)")

    return index_data


def full_rebuild():
    """Full rebuild: clear everything and re-extract all primaries."""
    print("=== FULL REBUILD ===")

    # Clear
    print("Clearing imageprimary/...")
    removed = 0
    for fn in os.listdir(PRIMARY_DIR):
        fp = os.path.join(PRIMARY_DIR, fn)
        if os.path.isfile(fp):
            os.remove(fp)
            removed += 1
    print(f"  Removed {removed:,} files")

    # Extract all
    db_primaries = get_db_primaries()
    all_primaries = list(db_primaries.items())
    extract_primaries(all_primaries)

    # Tar all
    tar_loose_images(None)


def incremental_update():
    """Incremental: only extract and tar new primaries."""
    print("=== INCREMENTAL UPDATE ===")

    existing_index = load_primary_index()
    if existing_index is None:
        print("No existing primary_index.json — falling back to full rebuild")
        full_rebuild()
        return

    existing_count = len(existing_index.get("reverse", {}))
    print(f"Existing index: {existing_count:,} listings")

    db_primaries = get_db_primaries()
    print(f"DB primaries: {len(db_primaries):,} listings")

    new_primaries = find_new_primaries(db_primaries, existing_index)
    print(f"New primaries to add: {len(new_primaries):,}")

    if not new_primaries:
        print("Nothing to do")
        return

    extract_primaries(new_primaries)
    tar_loose_images(existing_index)


if __name__ == "__main__":
    if "--full" in sys.argv:
        full_rebuild()
    else:
        incremental_update()

    # Final summary
    tars = [f for f in os.listdir(PRIMARY_DIR) if f.endswith(".tar")]
    loose = [f for f in os.listdir(PRIMARY_DIR) if f.endswith(".jpg")]
    print(f"\n=== Done ===")
    print(f"  Tars: {len(tars)}")
    print(f"  Loose files: {len(loose)}")
