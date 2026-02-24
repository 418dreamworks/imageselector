#!/usr/bin/env python3
"""Create 10K-image tar archives from imageembedded, ordered by FAISS row index.

Runs in a continuous loop: processes all available shards, sleeps 5 min,
checks for new work. Touch KILL_TAR to stop gracefully.

Reads sharded image_index files one shard at a time (lower memory than monolithic).
Each shard of 500K rows produces exactly 50 tar batches.
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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'embedding'))
from shard_utils import get_shard_dirs, load_shard_index, SHARD_SIZE

BASE_DIR = str(Path(__file__).parent.parent)
IMAGE_DIR = os.path.join(BASE_DIR, "images", "imageembedded")
TAR_DIR = os.path.join(BASE_DIR, "images", "imagetarred")
DB_PATH = os.path.join(BASE_DIR, "data", "db", "etsy_data.db")
KILL_FILE = Path(BASE_DIR) / "KILL_TAR"
BATCH_SIZE = 10000
TARS_PER_SHARD = SHARD_SIZE // BATCH_SIZE  # 50


def ts():
    return datetime.now().strftime("%H:%M:%S")


def create_tar_batch(batch_num, num_full_batches, filenames):
    """Create and verify one tar batch. Returns True on success."""
    tar_name = f"imageall_{batch_num:05d}.tar"
    tar_path = os.path.join(TAR_DIR, tar_name)

    if os.path.exists(tar_path):
        print(f"[{batch_num+1}/{num_full_batches}] {tar_name} already exists, skipping")
        return True

    # Check all files exist before tarring
    missing = [fn for fn in filenames if not os.path.exists(os.path.join(IMAGE_DIR, fn))]
    if missing:
        print(f"[{batch_num+1}/{num_full_batches}] {tar_name}: {len(missing)}/{len(filenames)} files not in imageembedded/ yet, skipping")
        return False

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
        return False

    tar_size = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"{tar_size:.0f}MB in {elapsed:.0f}s")

    # Verify: check file count
    try:
        with tarfile.open(tar_path, 'r') as tf:
            tar_members = tf.getnames()
    except Exception as e:
        print(f"  CORRUPT TAR: {e}")
        os.unlink(tar_path)
        return False

    if len(tar_members) != BATCH_SIZE:
        print(f"  VERIFY FAILED: expected {BATCH_SIZE} files, got {len(tar_members)}.")
        os.unlink(tar_path)
        return False

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
        return False
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

    # Mark as tarred in DB (download_done=2 → 3)
    pairs = []
    for fn in filenames:
        parts = fn.replace(".jpg", "").split("_")
        if len(parts) == 2:
            pairs.append((int(parts[0]), int(parts[1])))
    conn = sqlite3.connect(DB_PATH, timeout=300.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executemany(
        "UPDATE image_status SET download_done = 3 "
        "WHERE listing_id = ? AND image_id = ? AND download_done = 2",
        pairs
    )
    conn.commit()
    conn.close()
    print(f"  Marked {len(pairs)} images as download_done=3")

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


def run_pass():
    """Run one pass: process all available shards. Returns number of tars created."""
    shard_dirs = get_shard_dirs()
    if not shard_dirs:
        print(f"[{ts()}] No shard directories found. Nothing to do.")
        return 0

    # Calculate total full batches across all shards
    total_rows = 0
    for sd in shard_dirs:
        idx = load_shard_index(sd)
        total_rows += len(idx)

    num_full_batches = total_rows // BATCH_SIZE
    remainder = total_rows % BATCH_SIZE
    print(f"[{ts()}] Total rows: {total_rows:,}, Full batches: {num_full_batches}, Remainder: {remainder} (skipping)")

    if num_full_batches == 0:
        return 0

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
    tars_created = 0
    for shard_dir in shard_dirs:
        if KILL_FILE.exists():
            break

        shard_name = shard_dir.name
        shard_num = int(shard_name.split('_')[1])
        first_tar = shard_num * TARS_PER_SHARD

        # Skip finalized shards: if the 50th tar exists, all tars are done
        last_possible = os.path.join(TAR_DIR, f"imageall_{first_tar + TARS_PER_SHARD - 1:05d}.tar")
        if os.path.exists(last_possible):
            continue

        idx = load_shard_index(shard_dir)
        shard_rows = len(idx)
        shard_full_batches = shard_rows // BATCH_SIZE

        if shard_full_batches == 0:
            print(f"{shard_name}: {shard_rows} rows, no full batches to tar")
            continue

        # Find first missing tar to skip already-completed batches
        start_batch = 0
        for local_batch in range(shard_full_batches):
            tar_path = os.path.join(TAR_DIR, f"imageall_{first_tar + local_batch:05d}.tar")
            if not os.path.exists(tar_path):
                start_batch = local_batch
                break
        else:
            del idx
            continue

        print(f"\n{shard_name}: {shard_rows:,} rows, tars {first_tar}-{first_tar + shard_full_batches - 1}"
              + (f" (resuming from tar {first_tar + start_batch})" if start_batch > 0 else ""))

        for local_batch in range(start_batch, shard_full_batches):
            if KILL_FILE.exists():
                break

            batch_num = first_tar + local_batch
            local_start = local_batch * BATCH_SIZE
            local_end = local_start + BATCH_SIZE

            # Build file list from this shard's index
            filenames = []
            for row in range(local_start, local_end):
                lid, iid = idx[row][0], idx[row][1]
                filenames.append(f"{lid}_{iid}.jpg")

            if create_tar_batch(batch_num, num_full_batches, filenames):
                tars_created += 1
            else:
                break  # Files not ready yet, try again next pass

        # Check if shard is now fully tarred (all 50 tars exist for 500K shard)
        if shard_rows == SHARD_SIZE:
            last_tar = os.path.join(TAR_DIR, f"imageall_{first_tar + TARS_PER_SHARD - 1:05d}.tar")
            if os.path.exists(last_tar):
                # All 50 tars exist — mark entire shard as finalized
                pairs = [(entry[0], entry[1]) for entry in idx]
                conn = sqlite3.connect(DB_PATH, timeout=300.0)
                conn.execute("PRAGMA journal_mode=WAL")
                for chunk_start in range(0, len(pairs), 10000):
                    chunk = pairs[chunk_start:chunk_start + 10000]
                    conn.executemany(
                        "UPDATE image_status SET download_done = 4 "
                        "WHERE listing_id = ? AND image_id = ?",
                        chunk
                    )
                    conn.commit()
                conn.close()
                print(f"  {shard_name}: all {TARS_PER_SHARD} tars complete, marked {len(pairs)} images as download_done=4")

        # Free shard index memory
        del idx

    return tars_created


def main():
    if KILL_FILE.exists():
        print(f"Error: Kill file exists ({KILL_FILE}). Remove it to start.")
        return

    print(f"{'='*60}")
    print(f"TAR IMAGES (continuous)")
    print(f"{'='*60}")
    print(f"Stop with: touch KILL_TAR")
    print(f"{'='*60}")

    pass_num = 0
    while not KILL_FILE.exists():
        pass_num += 1
        print(f"\n[{ts()}] === Pass {pass_num} ===")

        tars_created = run_pass()
        print(f"[{ts()}] Pass {pass_num}: {tars_created} tars created")

        if KILL_FILE.exists():
            break

        # Sleep 5 min between passes (30 × 10s), checking kill file
        print(f"[{ts()}] Sleeping 5 min...")
        for _ in range(30):
            if KILL_FILE.exists():
                break
            time.sleep(10)

    print(f"\n[{ts()}] tar_images stopped")


if __name__ == "__main__":
    main()
