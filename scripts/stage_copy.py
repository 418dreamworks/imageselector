#!/usr/bin/env python3
"""Copy files from SSD to HDD using tar for sequential I/O, staging through main drive."""
import os
import subprocess

SRC = '/Volumes/SSD_120/embeddedimages'
DST = '/Volumes/HDD1TB/nonprimary'
STAGE = '/Users/tzuohannlaw/Documents/418Dreamworks/imageselector/_stage.tar'
LIST_FILE = '/Users/tzuohannlaw/Documents/418Dreamworks/imageselector/_tarlist.txt'
CHUNK_BYTES = 5 * 1024 * 1024 * 1024  # 5GB

# Get files already on destination
print("Scanning destination...")
existing = set(os.listdir(DST))
print(f"Destination has {len(existing):,} files")

# Get files to copy with sizes
print("Scanning source...")
all_files = os.listdir(SRC)
to_copy = []
for f in all_files:
    if f not in existing:
        try:
            size = os.path.getsize(os.path.join(SRC, f))
            to_copy.append((f, size))
        except OSError:
            pass

print(f"Source has {len(all_files):,} files, {len(to_copy):,} need copying")
total_bytes = sum(s for _, s in to_copy)
print(f"Total to copy: {total_bytes / 1024 / 1024 / 1024:.1f} GB")

idx = 0
round_num = 0

while idx < len(to_copy):
    round_num += 1
    chunk = []
    chunk_bytes = 0

    while idx < len(to_copy) and chunk_bytes < CHUNK_BYTES:
        f, size = to_copy[idx]
        chunk.append(f)
        chunk_bytes += size
        idx += 1

    print(f"\n--- Round {round_num}: {len(chunk):,} files, {chunk_bytes / 1024 / 1024:.0f} MB ---")

    # Write file list
    with open(LIST_FILE, 'w') as fh:
        for f in chunk:
            fh.write(f + '\n')

    # Step 1: tar from SSD -> main drive (sequential USB read)
    print("  Tarring from SSD to main drive...")
    r = subprocess.run(['tar', 'cf', STAGE, '-C', SRC, '-T', LIST_FILE])
    if r.returncode != 0:
        print(f"  tar create failed")
        break
    print(f"  Tar: {os.path.getsize(STAGE) / 1024 / 1024:.0f} MB")

    # Step 2: untar from main drive -> HDD (sequential USB write)
    print("  Untarring to HDD...")
    r = subprocess.run(['tar', 'xf', STAGE, '-C', DST])
    if r.returncode != 0:
        print(f"  tar extract failed")
        break

    # Cleanup
    os.remove(STAGE)
    os.remove(LIST_FILE)
    print(f"  Done. {idx:,} / {len(to_copy):,} copied")

print(f"\nComplete!")
