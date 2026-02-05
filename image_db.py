"""Shared database helpers for atomic image status updates.

All scripts use this module to update flags in image_status table.
Each UPDATE is atomic - no race conditions between scripts.

Schema (image_status):
    listing_id, image_id, is_primary, url, download_done, faiss_row
"""
import sqlite3
import time
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "etsy_data.db"

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
# INSERT (only sync_data.py uses this)
# ============================================================

@_retry_on_lock
def insert_image(
    conn: sqlite3.Connection,
    listing_id: int,
    image_id: int,
    is_primary: bool,
    url: str,
):
    """Insert a new image. Used by sync_data.py when discovering new images."""
    conn.execute("""
        INSERT OR IGNORE INTO image_status (
            listing_id, image_id, is_primary, url
        ) VALUES (?, ?, ?, ?)
    """, (listing_id, image_id, int(is_primary), url))


# ============================================================
# ATOMIC FLAG UPDATES
# ============================================================

@_retry_on_lock
def mark_download_done(conn: sqlite3.Connection, listing_id: int, image_id: int) -> bool:
    """Mark image as downloaded. Returns True if flag was actually flipped."""
    cursor = conn.execute("""
        UPDATE image_status SET download_done = 1
        WHERE listing_id = ? AND image_id = ? AND download_done = 0
    """, (listing_id, image_id))
    return cursor.rowcount > 0


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

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE faiss_row IS NOT NULL")
    stats["embedded"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 0")
    stats["pending_download"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 2 AND faiss_row IS NULL")
    stats["pending_embed"] = cursor.fetchone()[0]

    return stats


def print_stats():
    """Print current status counts."""
    conn = get_connection()
    stats = get_stats(conn)
    conn.close()

    print(f"Total images:       {stats['total']:,}")
    print(f"Downloaded:         {stats['downloaded']:,}")
    print(f"Embedded (faiss):   {stats['embedded']:,}")
    print(f"Pending download:   {stats['pending_download']:,}")
    print(f"Pending embed:      {stats['pending_embed']:,}")


if __name__ == "__main__":
    print("=== Image Status ===\n")
    print_stats()
