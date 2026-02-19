"""Shared database helpers for atomic image status updates.

All scripts use this module to update flags in image_status table.
Each UPDATE is atomic - no race conditions between scripts.

Schema (image_status):
    listing_id, image_id, is_primary, url, download_done
"""
import sqlite3
import time
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DB_FILE = BASE_DIR / "data" / "db" / "etsy_data.db"

# Retry settings for database lock handling
MAX_RETRIES = 30
BASE_DELAY = 0.5  # 500ms initial delay
MAX_DELAY = 10.0  # Max 10 seconds between retries


def _retry_on_lock(func):
    """Decorator to retry on database lock with exponential backoff."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.1), MAX_DELAY)
                    time.sleep(delay)
                else:
                    raise
    return wrapper


def get_connection() -> sqlite3.Connection:
    """Get a database connection with WAL mode for concurrent access."""
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def commit_with_retry(conn: sqlite3.Connection):
    """Commit with retry on database lock."""
    for attempt in range(MAX_RETRIES):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.1), MAX_DELAY)
                time.sleep(delay)
            else:
                raise


# ============================================================
# STATS
# ============================================================

def get_stats(conn: sqlite3.Connection) -> dict:
    """Get current status counts."""
    cursor = conn.cursor()

    stats = {}
    cursor.execute("SELECT COUNT(*) FROM image_status")
    stats["total"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 2")
    stats["downloaded"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 0")
    stats["pending_download"] = cursor.fetchone()[0]

    return stats


def print_stats():
    """Print current status counts."""
    conn = get_connection()
    stats = get_stats(conn)
    conn.close()

    print(f"Total images:       {stats['total']:,}")
    print(f"Downloaded:         {stats['downloaded']:,}")
    print(f"Pending download:   {stats['pending_download']:,}")


if __name__ == "__main__":
    print("=== Image Status ===\n")
    print_stats()
