#!/usr/bin/env python3
"""Backup data/ folder and embedding shards to backups/ (HDD1TB via symlink).

DB: SQLite backup API (safe for concurrent access).
Completed shards: backed up once (immutable, never changes).
Active shard: timestamped snapshot each run (for rollback).

Usage:
    venv/bin/python3 bin/backup_db.py
"""

import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'embedding'))
from shard_utils import get_shard_dirs, load_shard_index, SHARD_SIZE

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "db" / "etsy_data.db"
BACKUPS_DIR = BASE_DIR / "backups"


def make_readonly(path: Path):
    """Make all files under path read-only (chmod 444/555)."""
    for root, dirs, files in os.walk(path):
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o555)
    os.chmod(path, 0o555)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
date_stamp = datetime.now().strftime("%Y%m%d")

# ============================================================
# 1. DB backup (unchanged)
# ============================================================
backup_dir = BACKUPS_DIR / f"data_{timestamp}"
backup_dir.mkdir(parents=True)

# Copy data/ tree, skipping live DB and embeddings (shards backed up separately)
shutil.copytree(
    DATA_DIR, backup_dir / "data",
    ignore=shutil.ignore_patterns(
        "etsy_data.db", "etsy_data.db-wal", "etsy_data.db-shm",
        "shard_*",  # shards backed up separately below
    )
)
print(f"Copied data/ tree to {backup_dir.name}/data/")

# Backup DB safely using SQLite backup API
backup_db = backup_dir / "data" / "db" / "etsy_data.db"
src = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
dst = sqlite3.connect(str(backup_db))
src.backup(dst)
src.close()
dst.close()
print(f"  DB backed up ({backup_db.stat().st_size / (1024**3):.1f} GB)")

make_readonly(backup_dir)

# ============================================================
# 2. Embedding shard backups
# ============================================================
shard_dirs = get_shard_dirs()
if not shard_dirs:
    print("No shards to back up.")
else:
    for shard_dir in shard_dirs:
        idx = load_shard_index(shard_dir)
        shard_rows = len(idx)
        is_complete = shard_rows == SHARD_SIZE
        shard_name = shard_dir.name

        if is_complete:
            # Completed shard: back up once, skip if already exists
            dest = BACKUPS_DIR / shard_name
            if dest.exists():
                print(f"  {shard_name}: already backed up (immutable), skipping")
                continue

            print(f"  {shard_name}: backing up completed shard ({shard_rows:,} rows)...")
            shutil.copytree(shard_dir, dest)
            make_readonly(dest)
            print(f"  {shard_name}: backed up and locked")

        else:
            # Active shard: timestamped snapshot for rollback
            dest = BACKUPS_DIR / f"active_shard_{date_stamp}"
            if dest.exists():
                # Already have today's snapshot, overwrite
                # First remove read-only protection
                for root, dirs, files in os.walk(dest):
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o644)
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o755)
                os.chmod(dest, 0o755)
                shutil.rmtree(dest)

            print(f"  {shard_name}: backing up active shard ({shard_rows:,} rows) → active_shard_{date_stamp}/")
            shutil.copytree(shard_dir, dest)
            make_readonly(dest)
            print(f"  {shard_name}: snapshot saved")

print("\nBackup complete")
