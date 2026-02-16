#!/usr/bin/env python3
"""Rsync backups with pause/ack handshake to safely pause pipeline scripts.

Each invocation pauses all pipeline scripts, runs ONE rsync, then resumes.

Usage:
    venv/bin/python3 bin/backup_rsync.py data    # data/ → SSD500GB
    venv/bin/python3 bin/backup_rsync.py hdd     # HDD1TB → HDD3TB
    venv/bin/python3 bin/backup_rsync.py ssd     # SSD500GB → HDD500GB
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Pause file → ack file → PID file (to check if script is running)
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

RSYNCS = {
    "data": {
        "description": "data/ → SSD500GB",
        "cmd": ["rsync", "-a", "--exclude", "embeddings/",
                str(BASE_DIR / "data") + "/",
                "/Volumes/SSD500GB/imageselector/data/"],
    },
    "hdd": {
        "description": "HDD1TB → HDD3TB",
        "cmd": ["rsync", "-a", "--exclude", ".Spotlight-V100/",
                "/Volumes/HDD1TB/",
                "/Volumes/HDD3TB/HDD1TB_mirror/"],
    },
    "ssd": {
        "description": "SSD500GB → HDD500GB",
        "cmd": ["rsync", "-a", "--exclude", ".Spotlight-V100/",
                "/Volumes/SSD500GB/",
                "/Volumes/HDD500GB/"],
    },
}


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def is_running(pid_file: Path) -> bool:
    """Check if the process in pid_file is alive."""
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def cleanup():
    """Remove all pause and ack files."""
    for s in SCRIPTS:
        s["pause"].unlink(missing_ok=True)
        s["acked"].unlink(missing_ok=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in RSYNCS:
        print(f"Usage: {sys.argv[0]} {{data|hdd|ssd}}")
        return

    mode = sys.argv[1]
    rsync = RSYNCS[mode]

    # Check if any pause files already exist (another backup running?)
    existing = [s["name"] for s in SCRIPTS if s["pause"].exists()]
    if existing:
        print(f"[{ts()}] Error: Pause files already exist for: {', '.join(existing)}")
        print(f"[{ts()}] Another backup may be running. Remove PAUSE_* files to clear.")
        return

    print(f"[{ts()}] {'='*60}")
    print(f"[{ts()}] BACKUP RSYNC: {rsync['description']}")
    print(f"[{ts()}] {'='*60}")

    try:
        # Step 1: Create all pause files
        for s in SCRIPTS:
            s["pause"].touch()
        print(f"[{ts()}] Pause files created")

        # Step 2: Wait for acks from running scripts
        for s in SCRIPTS:
            if not is_running(s["pid"]):
                print(f"[{ts()}] {s['name']}: not running, skipping")
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

        # Step 3: Run rsync
        print(f"[{ts()}] All scripts paused. Starting: {rsync['description']}")
        print(f"  cmd: {' '.join(rsync['cmd'])}")
        t0 = time.time()
        result = subprocess.run(rsync["cmd"], capture_output=True, text=True)
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"[{ts()}] Done: {rsync['description']} ({elapsed:.0f}s)")
        else:
            print(f"[{ts()}] FAILED (exit {result.returncode}): {rsync['description']} ({elapsed:.0f}s)")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")

    finally:
        cleanup()
        print(f"[{ts()}] Pause/ack files removed")


if __name__ == "__main__":
    main()
