#!/usr/bin/env python3
"""Verify source and mirror drives match by file count and total bytes.

Walks both sides of each mirror pair, comparing aggregate file counts and
byte totals. Exits non-zero on any mismatch so the pipeline's failure
alert fires.

macOS per-volume metadata (.Spotlight-V100, .fseventsd, .TemporaryItems)
is excluded from both sides — those are filesystem-private and not part
of the mirror by design.

Usage:
    venv/bin/python3 bin/verify_mirrors.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

PAIRS = [
    ("HDD1TB -> HDD3TB",   "/Volumes/HDD1TB",   "/Volumes/HDD3TB/HDD1TB_mirror"),
    ("SSD500GB -> HDD500GB", "/Volumes/SSD500GB", "/Volumes/HDD500GB"),
]

EXCLUDE_DIRS = {".Spotlight-V100", ".fseventsd", ".TemporaryItems", ".Trashes"}

BYTE_TOLERANCE = 1 * 1024 * 1024  # 1 MB; absorbs noise without hiding a missed file


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def walk_size(root):
    """Return (file_count, total_bytes) under root, skipping per-volume metadata."""
    count = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
                count += 1
            except OSError:
                pass
    return count, total


def main():
    print(f"[{ts()}] {'=' * 60}")
    print(f"[{ts()}] VERIFY MIRRORS")
    print(f"[{ts()}] {'=' * 60}")

    failures = []
    for name, source, mirror in PAIRS:
        print(f"[{ts()}] === {name} ===")
        if not Path(source).is_dir():
            print(f"[{ts()}]   ERROR: source {source} not mounted")
            failures.append(f"{name}: source {source} not mounted")
            continue
        if not Path(mirror).is_dir():
            print(f"[{ts()}]   ERROR: mirror {mirror} not mounted")
            failures.append(f"{name}: mirror {mirror} not mounted")
            continue

        s_count, s_bytes = walk_size(source)
        m_count, m_bytes = walk_size(mirror)

        delta_count = s_count - m_count
        delta_bytes = s_bytes - m_bytes

        print(f"[{ts()}]   source: {s_count:,} files, {s_bytes / 1024**3:.2f} GB")
        print(f"[{ts()}]   mirror: {m_count:,} files, {m_bytes / 1024**3:.2f} GB")
        print(f"[{ts()}]   delta:  {delta_count:+,} files, {delta_bytes / 1024**2:+.1f} MB")

        if delta_count != 0 or abs(delta_bytes) > BYTE_TOLERANCE:
            failures.append(
                f"{name}: {delta_count:+,} files, {delta_bytes / 1024**2:+.1f} MB"
            )
            print(f"[{ts()}]   MISMATCH")
        else:
            print(f"[{ts()}]   OK")

    print(f"[{ts()}] {'=' * 60}")
    if failures:
        print(f"[{ts()}] VERIFY FAILED:")
        for f in failures:
            print(f"[{ts()}]   - {f}")
        return 1
    print(f"[{ts()}] ALL MIRRORS IN SYNC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
