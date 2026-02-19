#!/usr/bin/env python3
"""Create 10K-image tar archives from imageembedded, ordered by FAISS row index.

Reads sharded image_index files one shard at a time (lower memory than monolithic).
Each shard of 500K rows produces exactly 50 tar batches.
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'embedding'))
from shard_utils import get_shard_dirs, load_shard_index, SHARD_SIZE

BASE_DIR = str(Path(__file__).parent.parent)
IMAGE_DIR = os.path.join(BASE_DIR, "images", "imageembedded")
TAR_DIR = os.path.join(BASE_DIR, "images", "imagetarred")
BATCH_SIZE = 10000
TARS_PER_SHARD = SHARD_SIZE // BATCH_SIZE  # 50


def create_tar_batch(batch_num, num_full_batches, filenames):
    """Create and verify one tar batch. Returns True on success."""
    tar_name = f"imageall_{batch_num:05d}.tar"
    tar_path = os.path.join(TAR_DIR, tar_name)

    if os.path.exists(tar_path):
        print(f"[{batch_num+1}/{num_full_batches}] {tar_name} already exists, skipping")
        return True

    # Write file list to temp file
    list_path = f"/tmp/tar_batch_{batch_num:05d}.txt"
    with open(list_path, "w") as f:
        for fn in filenames:
            f.write(fn + "\n")

    # Create tar
    t0 = time.time()
    start_row = batch_num * BATCH_SIZE
    end_row = start_row + BATCH_SIZE
    print(f"[{batch_num+1}/{num_full_batches}] Creating {tar_name} (rows {start_row}-{end_row-1})...", end=" ", flush=True)
    result = subprocess.run(
        ["tar", "--no-mac-metadata", "-cf", tar_path, "-T", list_path],
        cwd=IMAGE_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}")
        if os.path.exists(tar_path):
            os.unlink(tar_path)
            print(f"  Deleted corrupt {tar_name}")
        os.unlink(list_path)
        print(f"  STOPPING due to tar failure.")
        sys.exit(1)

    tar_size = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"{tar_size:.0f}MB in {elapsed:.0f}s")

    # Verify: check file count
    try:
        with tarfile.open(tar_path, 'r') as tf:
            tar_members = tf.getnames()
    except Exception as e:
        print(f"  CORRUPT TAR: {e}")
        os.unlink(tar_path)
        print(f"  Deleted corrupt {tar_name}. STOPPING.")
        sys.exit(1)

    if len(tar_members) != BATCH_SIZE:
        print(f"  VERIFY FAILED: expected {BATCH_SIZE} files, got {len(tar_members)}.")
        os.unlink(tar_path)
        print(f"  Deleted corrupt {tar_name}. STOPPING.")
        sys.exit(1)

    # Verify: checksum 50 random files
    check_files = random.sample(filenames, 50)
    checksum_ok = True
    with tarfile.open(tar_path, 'r') as tf:
        for fn in check_files:
            orig_path = os.path.join(IMAGE_DIR, fn)
            with open(orig_path, 'rb') as f:
                orig_md5 = hashlib.md5(f.read()).hexdigest()
            tar_md5 = hashlib.md5(tf.extractfile(fn).read()).hexdigest()
            if orig_md5 != tar_md5:
                print(f"  CHECKSUM MISMATCH: {fn}")
                checksum_ok = False
                break
    if not checksum_ok:
        os.unlink(tar_path)
        print(f"  Deleted corrupt {tar_name}. STOPPING.")
        sys.exit(1)
    print(f"  Verified: {BATCH_SIZE} files, 50 checksums OK")

    # Update tar_index.json with byte offsets + reverse lookup
    index_path = os.path.join(TAR_DIR, "tar_index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            tar_index = json.load(f)
    else:
        tar_index = {"tars": {}, "reverse": {}}
    offsets = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            offsets[member.name] = member.offset
            key = member.name.replace(".jpg", "")
            tar_index["reverse"][key] = [tar_name, member.offset]
    tar_index["tars"][tar_name] = offsets
    with open(index_path, "w") as f:
        json.dump(tar_index, f)
    print(f"  Updated tar_index.json ({len(offsets)} entries)")

    # Delete originals from SSD
    deleted = 0
    for fn in filenames:
        src = os.path.join(IMAGE_DIR, fn)
        if os.path.exists(src):
            os.remove(src)
            deleted += 1
    print(f"  Deleted {deleted}/{len(filenames)} from imageembedded/")

    os.unlink(list_path)
    return True


# Load shard directories
shard_dirs = get_shard_dirs()
if not shard_dirs:
    print("No shard directories found. Nothing to do.")
    sys.exit(0)

# Calculate total full batches across all shards
total_rows = 0
for sd in shard_dirs:
    idx = load_shard_index(sd)
    total_rows += len(idx)

num_full_batches = total_rows // BATCH_SIZE
remainder = total_rows % BATCH_SIZE
print(f"Total rows: {total_rows:,}, Full batches: {num_full_batches}, Remainder: {remainder} (skipping)")

# Check last existing tar for integrity
for check_num in range(num_full_batches - 1, -1, -1):
    check_path = os.path.join(TAR_DIR, f"imageall_{check_num:05d}.tar")
    if os.path.exists(check_path):
        print(f"Checking last tar: imageall_{check_num:05d}.tar...", end=" ", flush=True)
        try:
            with tarfile.open(check_path, 'r') as tf:
                members = tf.getnames()
            if len(members) != BATCH_SIZE:
                print(f"INCOMPLETE ({len(members)}/{BATCH_SIZE}). Deleting.")
                os.unlink(check_path)
            else:
                print(f"OK ({len(members)} files)")
        except Exception as e:
            print(f"CORRUPT ({e}). Deleting.")
            os.unlink(check_path)
        break

# Process shard by shard (only one shard's index in memory at a time)
for shard_dir in shard_dirs:
    shard_name = shard_dir.name
    shard_num = int(shard_name.split('_')[1])
    idx = load_shard_index(shard_dir)
    shard_rows = len(idx)

    # Calculate which tar batches this shard covers
    first_tar = shard_num * TARS_PER_SHARD
    shard_full_batches = shard_rows // BATCH_SIZE

    if shard_full_batches == 0:
        print(f"{shard_name}: {shard_rows} rows, no full batches to tar")
        continue

    print(f"\n{shard_name}: {shard_rows:,} rows, tars {first_tar}-{first_tar + shard_full_batches - 1}")

    for local_batch in range(shard_full_batches):
        batch_num = first_tar + local_batch
        local_start = local_batch * BATCH_SIZE
        local_end = local_start + BATCH_SIZE

        # Build file list from this shard's index
        filenames = []
        for row in range(local_start, local_end):
            lid, iid = idx[row][0], idx[row][1]
            filenames.append(f"{lid}_{iid}.jpg")

        create_tar_batch(batch_num, num_full_batches, filenames)

    # Free shard index memory
    del idx

print("\nDone!")
