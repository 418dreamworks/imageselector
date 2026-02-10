#!/usr/bin/env python3
"""Backup data/ folder to backups/ (HDD1TB via symlink).

Uses SQLite backup API for the DB (safe for concurrent access),
then copies the rest of data/ as-is. All files chmod 444.

Usage:
    venv/bin/python3 scratch/backup_db.py
"""

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "db" / "etsy_data.db"
BACKUPS_DIR = BASE_DIR / "backups"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = BACKUPS_DIR / f"data_{timestamp}"
backup_dir.mkdir(parents=True)

# Copy entire data/ tree (skipping the live DB — we'll use SQLite backup API for that)
shutil.copytree(DATA_DIR, backup_dir / "data", ignore=shutil.ignore_patterns("etsy_data.db", "etsy_data.db-wal", "etsy_data.db-shm"))
print(f"Copied data/ tree to {backup_dir.name}/data/")

# Backup DB safely using SQLite backup API
backup_db = backup_dir / "data" / "db" / "etsy_data.db"
src = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
dst = sqlite3.connect(str(backup_db))
src.backup(dst)
src.close()
dst.close()
print(f"  DB backed up ({backup_db.stat().st_size / (1024**3):.1f} GB)")

# Make everything read-only
for root, dirs, files in os.walk(backup_dir):
    for f in files:
        os.chmod(os.path.join(root, f), 0o444)
    for d in dirs:
        os.chmod(os.path.join(root, d), 0o555)
os.chmod(backup_dir, 0o555)

print("Backup complete")
