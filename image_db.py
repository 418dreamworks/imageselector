"""Shared database helpers for atomic image status updates.

All scripts use this module to update flags in image_status table.
Each UPDATE is atomic - no race conditions between scripts.
"""
import sqlite3
import time
import random
from pathlib import Path
from typing import Iterator

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

# CDN URL template for Etsy images (570xN size)
CDN_URL_TEMPLATE = "https://i.etsystatic.com/il/{hex}/{image_id}/il_570xN.{image_id}_{suffix}.jpg"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with WAL mode for concurrent access."""
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def commit_with_retry(conn: sqlite3.Connection):
    """Commit with retry on database lock.

    The @_retry_on_lock decorator only wraps individual execute() calls,
    but commit() can also fail with 'database is locked'. This function
    provides the same retry logic for commits.
    """
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


def build_cdn_url(hex_val: str, image_id: int, suffix: str) -> str:
    """Build CDN URL for an image."""
    return f"https://i.etsystatic.com/il/{hex_val}/{image_id}/il_570xN.{image_id}_{suffix}.jpg"


# ============================================================
# INSERT (only sync_data.py uses this)
# ============================================================

@_retry_on_lock
def insert_image(
    conn: sqlite3.Connection,
    listing_id: int,
    image_id: int,
    shop_id: int,
    hex_val: str,
    suffix: str,
    is_primary: bool,
    when_made: str,
    price: float = None,
    to_download: bool = True,
):
    """Insert a new image. Used by sync_data.py when discovering new images."""
    conn.execute("""
        INSERT OR IGNORE INTO image_status (
            listing_id, image_id, shop_id, hex, suffix, is_primary, when_made, price, to_download
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (listing_id, image_id, shop_id, hex_val, suffix, int(is_primary), when_made, price, int(to_download)))


# ============================================================
# ATOMIC FLAG UPDATES (each script uses its own)
# ============================================================

@_retry_on_lock
def mark_download_done(conn: sqlite3.Connection, listing_id: int, image_id: int) -> bool:
    """Mark image as downloaded. Returns True if flag was actually flipped."""
    cursor = conn.execute("""
        UPDATE image_status SET download_done = 1
        WHERE listing_id = ? AND image_id = ? AND download_done = 0
    """, (listing_id, image_id))
    return cursor.rowcount > 0


@_retry_on_lock
def mark_bg_removed(conn: sqlite3.Connection, listing_id: int, image_id: int) -> bool:
    """Mark image as background removed. Returns True if flag was actually flipped."""
    cursor = conn.execute("""
        UPDATE image_status SET bg_removed = 1
        WHERE listing_id = ? AND image_id = ? AND bg_removed = 0
    """, (listing_id, image_id))
    return cursor.rowcount > 0


@_retry_on_lock
def mark_embedded(conn: sqlite3.Connection, listing_id: int, image_id: int, model_key: str) -> bool:
    """Mark image as embedded for a specific model. Returns True if flag was actually flipped."""
    # Validate model_key to prevent SQL injection
    valid_models = [
        "clip_vitb32", "clip_vitl14", "dinov2_base", "dinov2_large", "dinov3_base"
    ]
    if model_key not in valid_models:
        raise ValueError(f"Invalid model_key: {model_key}")

    col = f"embed_{model_key}"
    cursor = conn.execute(f"""
        UPDATE image_status SET {col} = 1
        WHERE listing_id = ? AND image_id = ? AND {col} = 0
    """, (listing_id, image_id))
    return cursor.rowcount > 0


# ============================================================
# QUERY WORK QUEUES (each script queries its own)
# ============================================================

def get_images_to_download(conn: sqlite3.Connection, limit: int = 1000) -> list[dict]:
    """Get images that need downloading."""
    cursor = conn.execute("""
        SELECT listing_id, image_id, hex, suffix
        FROM image_status
        WHERE to_download = 1 AND download_done = 0
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cursor.fetchall()]


def get_images_for_bg_removal(conn: sqlite3.Connection, limit: int = 1000) -> list[dict]:
    """Get downloaded images that need background removal."""
    cursor = conn.execute("""
        SELECT listing_id, image_id
        FROM image_status
        WHERE download_done = 1 AND bg_removed = 0
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cursor.fetchall()]


def get_images_for_embedding(conn: sqlite3.Connection, model_key: str, limit: int = 1000) -> list[dict]:
    """Get images that need embedding for a specific model.

    Results are ordered by (listing_id, image_id) for consistent ordering
    across models - this ensures FAISS row alignment.
    """
    valid_models = [
        "clip_vitb32", "clip_vitl14", "dinov2_base", "dinov2_large", "dinov3_base"
    ]
    if model_key not in valid_models:
        raise ValueError(f"Invalid model_key: {model_key}")

    col = f"embed_{model_key}"
    cursor = conn.execute(f"""
        SELECT listing_id, image_id
        FROM image_status
        WHERE bg_removed = 1 AND {col} = 0
        ORDER BY listing_id, image_id
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cursor.fetchall()]


def get_images_for_embedding_batch(conn: sqlite3.Connection, limit: int = 50000) -> list[dict]:
    """Get a batch of images that need embedding for ANY model.

    Returns images ordered by (listing_id, image_id) that have bg_removed=1
    but are missing at least one embedding. This ensures the same batch
    is processed through all models before moving to the next batch.
    """
    cursor = conn.execute("""
        SELECT listing_id, image_id
        FROM image_status
        WHERE bg_removed = 1 AND (
            embed_clip_vitb32 = 0 OR
            embed_clip_vitl14 = 0 OR
            embed_dinov2_base = 0 OR
            embed_dinov2_large = 0 OR
            embed_dinov3_base = 0
        )
        ORDER BY listing_id, image_id
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cursor.fetchall()]


def get_embedding_status_for_images(
    conn: sqlite3.Connection,
    image_ids: list[tuple[int, int]]
) -> dict[tuple[int, int], dict[str, bool]]:
    """Get embedding status for a list of specific images.

    Returns dict mapping (listing_id, image_id) -> {model_key: bool, ...}
    Queries in batches to avoid SQLite variable limit.
    """
    if not image_ids:
        return {}

    result = {}
    batch_size = 400  # Safe limit for (?, ?) pairs = 800 variables

    for i in range(0, len(image_ids), batch_size):
        batch = image_ids[i:i + batch_size]
        placeholders = ",".join(["(?, ?)"] * len(batch))
        flat_ids = [x for pair in batch for x in pair]

        cursor = conn.execute(f"""
            SELECT listing_id, image_id,
                   embed_clip_vitb32, embed_clip_vitl14,
                   embed_dinov2_base, embed_dinov2_large, embed_dinov3_base
            FROM image_status
            WHERE (listing_id, image_id) IN ({placeholders})
        """, flat_ids)

        for row in cursor.fetchall():
            key = (row[0], row[1])
            result[key] = {
                "clip_vitb32": bool(row[2]),
                "clip_vitl14": bool(row[3]),
                "dinov2_base": bool(row[4]),
                "dinov2_large": bool(row[5]),
                "dinov3_base": bool(row[6]),
            }

    return result


# ============================================================
# STATS
# ============================================================

def get_stats(conn: sqlite3.Connection) -> dict:
    """Get current status counts."""
    cursor = conn.cursor()

    stats = {}
    cursor.execute("SELECT COUNT(*) FROM image_status")
    stats["total"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE to_download = 1")
    stats["to_download"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE download_done = 1")
    stats["download_done"] = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM image_status WHERE bg_removed = 1")
    stats["bg_removed"] = cursor.fetchone()[0]

    for model in ["clip_vitb32", "clip_vitl14", "dinov2_base", "dinov2_large", "dinov3_base"]:
        cursor.execute(f"SELECT COUNT(*) FROM image_status WHERE embed_{model} = 1")
        stats[f"embed_{model}"] = cursor.fetchone()[0]

    return stats


def print_stats():
    """Print current status counts."""
    conn = get_connection()
    stats = get_stats(conn)
    conn.close()

    print(f"Total images:      {stats['total']:,}")
    print(f"To download:       {stats['to_download']:,}")
    print(f"Download done:     {stats['download_done']:,}")
    print(f"BG removed:        {stats['bg_removed']:,}")
    print(f"Embedded clip32:   {stats['embed_clip_vitb32']:,}")
    print(f"Embedded clip14:   {stats['embed_clip_vitl14']:,}")
    print(f"Embedded dino2b:   {stats['embed_dinov2_base']:,}")
    print(f"Embedded dino2l:   {stats['embed_dinov2_large']:,}")
    print(f"Embedded dino3b:   {stats['embed_dinov3_base']:,}")


if __name__ == "__main__":
    print("=== Image Status ===\n")
    print_stats()
