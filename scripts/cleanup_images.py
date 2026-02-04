#!/usr/bin/env python3
"""Cleanup non-primary images after embedding is complete.

For each listing where all images have all 5 embedded_* flags:
- Keep the primary image (is_primary = 1)
- Delete all non-primary images from disk
- Optionally delete from database

Uses SQLite image_status table for tracking.
"""
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
KILL_FILE = BASE_DIR / "KILL_CLEANUP"

# Add parent to path for imports
sys.path.insert(0, str(BASE_DIR))
from image_db import get_connection

# All embedding models that must be complete before cleanup
EMBED_MODELS = [
    "clip_vitb32",
    "clip_vitl14",
    "dinov2_base",
    "dinov2_large",
    "dinov3_base",
]


def check_kill_file() -> bool:
    """Check if kill file exists."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        print("\nKill file detected. Stopping...")
        return True
    return False


def get_cleanable_images(conn, limit: int = 10000) -> list[dict]:
    """Find non-primary images where listing has all embeddings complete.

    Returns images that can be deleted (non-primary, fully embedded).
    """
    # Build condition for all models embedded
    embed_conditions = " AND ".join([f"embed_{m} = 1" for m in EMBED_MODELS])

    # Find non-primary images from listings where ALL images are fully embedded
    # A listing is fully embedded if it has no images with any embed_* = 0
    query = f"""
        SELECT i.listing_id, i.image_id
        FROM image_status i
        WHERE i.is_primary = 0
          AND i.{embed_conditions.replace(' AND ', ' AND i.')}
          AND NOT EXISTS (
              SELECT 1 FROM image_status i2
              WHERE i2.listing_id = i.listing_id
                AND (i2.embed_clip_vitb32 = 0
                     OR i2.embed_clip_vitl14 = 0
                     OR i2.embed_dinov2_base = 0
                     OR i2.embed_dinov2_large = 0
                     OR i2.embed_dinov3_base = 0)
          )
        LIMIT ?
    """

    cursor = conn.execute(query, (limit,))
    return [dict(row) for row in cursor.fetchall()]


def delete_image_file(listing_id: int, image_id: int, dry_run: bool = True) -> bool:
    """Delete image file from disk."""
    img_path = IMAGES_DIR / f"{listing_id}_{image_id}.jpg"

    if img_path.exists():
        if dry_run:
            print(f"  Would delete: {img_path.name}")
        else:
            img_path.unlink()
            print(f"  Deleted: {img_path.name}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Cleanup non-primary images after embedding")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Limit number of images to process per run (default: 10000)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous watch mode - keep checking for cleanable images",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between checks in watch mode (default: 300)",
    )
    args = parser.parse_args()

    print("Cleanup: Delete non-primary images after embedding")
    print(f"Images dir: {IMAGES_DIR}")
    if args.dry_run:
        print("DRY RUN - no files will be deleted")
    print()

    conn = get_connection()

    while True:
        if check_kill_file():
            break

        print("Finding cleanable images...")
        cleanable = get_cleanable_images(conn, args.limit)
        print(f"Found {len(cleanable)} non-primary images ready for cleanup")

        if not cleanable:
            if args.watch:
                import time
                print(f"Nothing to clean. Waiting {args.interval}s...")
                time.sleep(args.interval)
                continue
            else:
                print("Nothing to clean up.")
                break

        deleted = 0
        for img in cleanable:
            if check_kill_file():
                break

            lid = img["listing_id"]
            iid = img["image_id"]

            if delete_image_file(lid, iid, dry_run=args.dry_run):
                deleted += 1

        print(f"\n{'Would delete' if args.dry_run else 'Deleted'}: {deleted} files")

        if not args.watch:
            break

        if deleted == 0:
            import time
            print(f"Waiting {args.interval}s before next check...")
            time.sleep(args.interval)

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
