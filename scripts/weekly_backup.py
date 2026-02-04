#!/usr/bin/env python3
"""Weekly backup of critical files to HDD.

Backs up:
- etsy_data.db (SQLite database)
- image_metadata.json
- progress.json
- All FAISS embeddings (*.faiss, image_index.json)

Keeps 4 weekly backups, rotating oldest.
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
BACKUP_BASE = Path("/Volumes/HDD_1000/imageselector_backup")

# Files to back up (relative to BASE_DIR)
BACKUP_FILES = [
    "etsy_data.db",
    "image_metadata.json",
    "progress.json",
]

# Directories to back up (relative to BASE_DIR)
BACKUP_DIRS = [
    "embeddings",
]

MAX_BACKUPS = 4  # Keep 4 weekly backups


def get_backup_dir() -> Path:
    """Get backup directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BACKUP_BASE / f"backup_{timestamp}"


def rotate_backups():
    """Remove oldest backups if we have more than MAX_BACKUPS."""
    if not BACKUP_BASE.exists():
        return

    backups = sorted(BACKUP_BASE.glob("backup_*"), key=lambda p: p.name)

    while len(backups) >= MAX_BACKUPS:
        oldest = backups.pop(0)
        print(f"Removing old backup: {oldest.name}")
        shutil.rmtree(oldest)


def backup_file(src: Path, dest_dir: Path, dry_run: bool = False) -> bool:
    """Copy a file to backup directory."""
    if not src.exists():
        print(f"  Skip (not found): {src.name}")
        return False

    dest = dest_dir / src.name
    size_mb = src.stat().st_size / (1024 * 1024)

    if dry_run:
        print(f"  Would copy: {src.name} ({size_mb:.1f} MB)")
    else:
        print(f"  Copying: {src.name} ({size_mb:.1f} MB)...")
        shutil.copy2(src, dest)

    return True


def backup_dir(src: Path, dest_dir: Path, dry_run: bool = False) -> bool:
    """Copy a directory to backup location."""
    if not src.exists():
        print(f"  Skip (not found): {src.name}/")
        return False

    dest = dest_dir / src.name

    # Calculate total size
    total_size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    file_count = sum(1 for f in src.rglob("*") if f.is_file())

    if dry_run:
        print(f"  Would copy: {src.name}/ ({file_count} files, {size_mb:.1f} MB)")
    else:
        print(f"  Copying: {src.name}/ ({file_count} files, {size_mb:.1f} MB)...")
        shutil.copytree(src, dest)

    return True


def main():
    parser = argparse.ArgumentParser(description="Weekly backup to HDD")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backed up without doing it",
    )
    args = parser.parse_args()

    print(f"Base directory: {BASE_DIR}")
    print(f"Backup location: {BACKUP_BASE}")

    # Check HDD is mounted
    if not BACKUP_BASE.parent.exists():
        print(f"\nError: HDD not mounted at {BACKUP_BASE.parent}")
        return

    # Rotate old backups first
    if not args.dry_run:
        rotate_backups()

    # Create backup directory
    backup_dir_path = get_backup_dir()
    print(f"\nBackup directory: {backup_dir_path.name}")

    if not args.dry_run:
        backup_dir_path.mkdir(parents=True, exist_ok=True)

    # Backup files
    print("\nBacking up files:")
    for filename in BACKUP_FILES:
        src = BASE_DIR / filename
        backup_file(src, backup_dir_path, args.dry_run)

    # Backup directories
    print("\nBacking up directories:")
    for dirname in BACKUP_DIRS:
        src = BASE_DIR / dirname
        backup_dir(src, backup_dir_path, args.dry_run)

    # Summary
    if not args.dry_run:
        total_size = sum(
            f.stat().st_size for f in backup_dir_path.rglob("*") if f.is_file()
        )
        print(f"\nBackup complete: {total_size / (1024 * 1024):.1f} MB")
        print(f"Location: {backup_dir_path}")
    else:
        print("\nDry run complete. Use without --dry-run to perform backup.")


if __name__ == "__main__":
    main()
