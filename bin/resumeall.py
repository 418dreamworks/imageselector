#!/usr/bin/env python3
"""Resume all pipeline scripts by removing pause/ack files.

Removes PAUSE_SD, PAUSE_DL, PAUSE_ORCH and their PAUSED_* ack files.
Scripts will resume on their next poll cycle (up to 60s).

Usage:
    venv/bin/python3 bin/resumeall.py
"""

from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

FILES = [
    BASE_DIR / "PAUSE_SD",
    BASE_DIR / "PAUSED_SD",
    BASE_DIR / "PAUSE_DL",
    BASE_DIR / "PAUSED_DL",
    BASE_DIR / "PAUSE_ORCH",
    BASE_DIR / "PAUSED_ORCH",
]


def ts():
    return datetime.now().strftime("%H:%M:%S")


def main():
    removed = []
    for f in FILES:
        if f.exists():
            f.unlink()
            removed.append(f.name)

    if removed:
        print(f"[{ts()}] Removed: {', '.join(removed)}")
    else:
        print(f"[{ts()}] No pause/ack files found")

    print(f"[{ts()}] Scripts will resume on next poll cycle (up to 60s)")


if __name__ == "__main__":
    main()
