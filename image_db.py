"""Shared database helpers for atomic image status updates.

All scripts use this module to update flags in image_status table.
Each UPDATE is atomic - no race conditions between scripts.
"""
import sqlite3
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "etsy_data.db"

# CDN URL template for Etsy images (570xN size)
CDN_URL_TEMPLATE = "https://i.etsystatic.com/il/{hex}/{image_id}/il_570xN.{image_id}_{suffix}.jpg"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with WAL mode for concurrent access."""
    conn = sqlite3.connect(DB_FILE, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def build_cdn_url(hex_val: str, image_id: int, suffix: str) -> str:
    """Build CDN URL for an image."""
    return f"https://i.etsystatic.com/il/{hex_val}/{image_id}/il_570xN.{image_id}_{suffix}.jpg"


# ============================================================
# INSERT (only sync_data.py uses this)
# ============================================================

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

def mark_download_done(conn: sqlite3.Connection, listing_id: int, image_id: int) -> bool:
    """Mark image as downloaded. Returns True if flag was actually flipped."""
    cursor = conn.execute("""
        UPDATE image_status SET download_done = 1
        WHERE listing_id = ? AND image_id = ? AND download_done = 0
    """, (listing_id, image_id))
    return cursor.rowcount > 0


def mark_bg_removed(conn: sqlite3.Connection, listing_id: int, image_id: int) -> bool:
    """Mark image as background removed. Returns True if flag was actually flipped."""
    cursor = conn.execute("""
        UPDATE image_status SET bg_removed = 1
        WHERE listing_id = ? AND image_id = ? AND bg_removed = 0
    """, (listing_id, image_id))
    return cursor.rowcount > 0


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
    """Get images that need embedding for a specific model."""
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
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cursor.fetchall()]


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
