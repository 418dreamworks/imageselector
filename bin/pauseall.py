#!/usr/bin/env python3
"""Pause all pipeline scripts and wait for acks.

Creates PAUSE_SD, PAUSE_DL, PAUSE_ORCH. For each running script,
waits until it touches its PAUSED_* ack file (confirming it has stopped).

Usage:
    venv/bin/python3 bin/pauseall.py
"""

import os
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

SCRIPTS = [
    {
        "name": "sync_data",
        "pause": BASE_DIR / "PAUSE_SD",
        "acked": BASE_DIR / "PAUSED_SD",
        "pid": BASE_DIR / "sync_data.pid",
    },
    {
        "name": "image_downloader",
        "pause": BASE_DIR / "PAUSE_DL",
        "acked": BASE_DIR / "PAUSED_DL",
        "pid": BASE_DIR / "image_downloader.pid",
    },
    {
        "name": "embed_orchestrator",
        "pause": BASE_DIR / "PAUSE_ORCH",
        "acked": BASE_DIR / "PAUSED_ORCH",
        "pid": BASE_DIR / "orchestrator.pid",
    },
]


def ts():
    return datetime.now().strftime("%H:%M:%S")


def is_running(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def main():
    # Create all pause files
    for s in SCRIPTS:
        s["pause"].touch()
    print(f"[{ts()}] Pause files created: PAUSE_SD, PAUSE_DL, PAUSE_ORCH")

    # Wait for acks from running scripts
    for s in SCRIPTS:
        if not is_running(s["pid"]):
            print(f"[{ts()}] {s['name']}: not running")
            continue

        print(f"[{ts()}] {s['name']}: running, waiting for ack...", end="", flush=True)
        waited = 0
        while not s["acked"].exists():
            time.sleep(5)
            waited += 5
            if not is_running(s["pid"]):
                print(f" exited after {waited}s")
                break
            if waited % 30 == 0:
                print(".", end="", flush=True)
        else:
            print(f" acked after {waited}s")

    print(f"[{ts()}] All scripts paused or not running.")
    print(f"Run 'venv/bin/python3 bin/resumeall.py' to resume.")


if __name__ == "__main__":
    main()
