#!/usr/bin/env python3
"""Create 10K-image tar archives from imageall, ordered by FAISS row index."""

import hashlib
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BASE_DIR = str(Path(__file__).parent.parent)
IMAGE_DIR = os.path.join(BASE_DIR, "images", "imageall_new")
TAR_DIR = os.path.join(BASE_DIR, "images", "imageall_tars")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "image_index.json")
BATCH_SIZE = 10000

LOCK_PATH = INDEX_PATH + ".lock"

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

print("Loading image_index.json...")
idx = load_index_safe()

total = len(idx)
num_full_batches = total // BATCH_SIZE
remainder = total % BATCH_SIZE
print(f"Total: {total}, Full batches: {num_full_batches}, Remainder: {remainder} (skipping)")

# Check last existing tar for integrity (could be corrupt from interrupted run)
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

for batch_num in range(num_full_batches):
    tar_name = f"imageall_{batch_num:05d}.tar"
    tar_path = os.path.join(TAR_DIR, tar_name)

    if os.path.exists(tar_path):
        print(f"[{batch_num+1}/{num_full_batches}] {tar_name} already exists, skipping")
        continue

    start_row = batch_num * BATCH_SIZE
    end_row = start_row + BATCH_SIZE

    # Build file list
    filenames = []
    for row in range(start_row, end_row):
        lid, iid = idx[row][0], idx[row][1]
        filenames.append(f"{lid}_{iid}.jpg")

    # Write file list to temp file
    list_path = f"/tmp/tar_batch_{batch_num:05d}.txt"
    with open(list_path, "w") as f:
        for fn in filenames:
            f.write(fn + "\n")

    # Create tar
    t0 = time.time()
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
        os.unlink(list_path)
        continue

    tar_size = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"{tar_size:.0f}MB in {elapsed:.0f}s")

    # Verify: check file count
    with tarfile.open(tar_path, 'r') as tf:
        tar_members = tf.getnames()
    if len(tar_members) != BATCH_SIZE:
        print(f"  VERIFY FAILED: expected {BATCH_SIZE} files, got {len(tar_members)}. STOPPING.")
        sys.exit(1)

    # Verify: checksum 50 random files
    check_files = random.sample(filenames, 50)
    checksum_ok = True
    with tarfile.open(tar_path, 'r') as tf:
        for fn in check_files:
            # Hash from original on disk
            orig_path = os.path.join(IMAGE_DIR, fn)
            with open(orig_path, 'rb') as f:
                orig_md5 = hashlib.md5(f.read()).hexdigest()
            # Hash from tar
            tar_md5 = hashlib.md5(tf.extractfile(fn).read()).hexdigest()
            if orig_md5 != tar_md5:
                print(f"  CHECKSUM MISMATCH: {fn}")
                checksum_ok = False
                break
    if not checksum_ok:
        print(f"  VERIFY FAILED: checksum mismatch. STOPPING.")
        sys.exit(1)
    print(f"  Verified: {BATCH_SIZE} files, 50 checksums OK")

    # Delete originals from SSD
    deleted = 0
    for fn in filenames:
        src = os.path.join(IMAGE_DIR, fn)
        if os.path.exists(src):
            os.remove(src)
            deleted += 1
    print(f"  Deleted {deleted}/{len(filenames)} from imageall_new/")

    os.unlink(list_path)

print("\nDone!")
