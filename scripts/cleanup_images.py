#!/usr/bin/env python3
"""Cleanup non-primary images after embedding is complete.

For each listing where all images have all 5 embedded_* flags:
- Keep the primary image (is_primary = 1) in images/
- Move all non-primary images to backup location (HDD)

Uses SQLite image_status table for tracking.
"""
import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
IMAGES_DIR = BASE_DIR / "images"
KILL_FILE = BASE_DIR / "KILL_CLEANUP"

# Backup location for non-primary images on external SSD
BACKUP_DIR = Path("/Volumes/SSD_120/embeddedimages")

# Add parent to path for imports
sys.path.insert(0, str(BASE_DIR))
from image_db import get_connection

def check_kill_file() -> bool:
    """Check if kill file exists."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        print("\nKill file detected. Stopping...")
        return True
    return False


def get_cleanable_images(conn, limit: int = 10000) -> list[dict]:
    """Find non-primary images that have been embedded (faiss_row IS NOT NULL).

    Returns images that can be moved to backup.
    """
    query = """
        SELECT listing_id, image_id
        FROM image_status
        WHERE is_primary = 0
          AND faiss_row IS NOT NULL
        LIMIT ?
    """

    cursor = conn.execute(query, (limit,))
    return [dict(row) for row in cursor.fetchall()]


def move_image_file(listing_id: int, image_id: int, backup_dir: Path, dry_run: bool = True) -> bool:
    """Move image file from images/ to backup location."""
    src_path = IMAGES_DIR / f"{listing_id}_{image_id}.jpg"
    dst_path = backup_dir / f"{listing_id}_{image_id}.jpg"

    if not src_path.exists():
        return False

    if dry_run:
        print(f"  Would move: {src_path.name} -> {dst_path}")
    else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        print(f"  Moved: {src_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Move non-primary images to backup after embedding")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Limit number of images to process per run (default: 10000)",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(BACKUP_DIR),
        help=f"Backup directory (default: {BACKUP_DIR})",
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

    backup_dir = Path(args.backup_dir)

    print("Cleanup: Move non-primary images to backup after embedding")
    print(f"Source: {IMAGES_DIR}")
    print(f"Backup: {backup_dir}")
    if args.dry_run:
        print("DRY RUN - no files will be moved")
    print()

    # Check backup dir exists (or can be created)
    if not args.dry_run and not backup_dir.exists():
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created backup directory: {backup_dir}")
        except Exception as e:
            print(f"ERROR: Cannot create backup directory: {e}")
            return

    conn = get_connection()

    while True:
        if check_kill_file():
            break

        print("Finding cleanable images...")
        cleanable = get_cleanable_images(conn, args.limit)
        print(f"Found {len(cleanable)} non-primary images ready for backup")

        if not cleanable:
            if args.watch:
                import time
                print(f"Nothing to move. Waiting {args.interval}s...")
                time.sleep(args.interval)
                continue
            else:
                print("Nothing to clean up.")
                break

        moved = 0
        for img in cleanable:
            if check_kill_file():
                break

            lid = img["listing_id"]
            iid = img["image_id"]

            if move_image_file(lid, iid, backup_dir, dry_run=args.dry_run):
                moved += 1

        print(f"\n{'Would move' if args.dry_run else 'Moved'}: {moved} files")

        if not args.watch:
            break

        if moved == 0:
            import time
            print(f"Waiting {args.interval}s before next check...")
            time.sleep(args.interval)

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
