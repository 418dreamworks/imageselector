#!/usr/bin/env python3
"""Pipeline runner — chains all pipeline steps sequentially.

Each step runs as a subprocess, logging to its own log file.
Next step starts when previous exits. If a step fails, log it and continue.

Usage:
    venv/bin/python3 bin/pipeline.py

Kill: touch KILL_PIPELINE (checked between steps)
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
KILL_FILE = BASE_DIR / "KILL_PIPELINE"
VENV_PYTHON = str(BASE_DIR / "venv" / "bin" / "python3")

STEPS = [
    {"name": "sync_data",         "cmd": [VENV_PYTHON, "bin/sync_data.py"],                    "log": "logs/sync_data.log"},
    {"name": "image_downloader",  "cmd": [VENV_PYTHON, "bin/image_downloader.py"],              "log": "logs/image_downloader.log"},
    {"name": "embed_orchestrator","cmd": [VENV_PYTHON, "bin/embedding/embed_orchestrator.py"],  "log": "logs/embed_orchestrator.log"},
    {"name": "tar_images",        "cmd": [VENV_PYTHON, "bin/tar_images.py"],                    "log": "logs/tar_images.log"},
    {"name": "update_primary",    "cmd": [VENV_PYTHON, "bin/update_primary.py"],                "log": "logs/update_primary.log"},
    {"name": "backup_db",         "cmd": [VENV_PYTHON, "bin/backup_db.py"],                     "log": "logs/backup_db.log"},
    {"name": "backup_rsync_data", "cmd": [VENV_PYTHON, "bin/backup_rsync.py", "data"],          "log": "logs/rsync_ssd500.log"},
    {"name": "backup_rsync_hdd",  "cmd": [VENV_PYTHON, "bin/backup_rsync.py", "hdd"],           "log": "logs/rsync_hdd3tb.log"},
    {"name": "backup_rsync_ssd",  "cmd": [VENV_PYTHON, "bin/backup_rsync.py", "ssd"],           "log": "logs/rsync_hdd500.log"},
]


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def main():
    if KILL_FILE.exists():
        log(f"Error: Kill file exists ({KILL_FILE}). Remove it to start.")
        return

    log("=" * 60)
    log("PIPELINE START")
    log("=" * 60)

    results = []

    for step in STEPS:
        if KILL_FILE.exists():
            log("Kill file detected. Stopping pipeline.")
            break

        name = step["name"]
        log_path = BASE_DIR / step["log"]
        log_path.parent.mkdir(exist_ok=True)

        log(f"--- {name} ---")
        t0 = time.time()

        try:
            proc = subprocess.Popen(
                step["cmd"],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            with open(log_path, "a") as lf:
                for line in proc.stdout:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                    lf.write(line.decode("utf-8", errors="replace"))
                    lf.flush()
            proc.wait()
            elapsed = time.time() - t0

            if proc.returncode == 0:
                log(f"  {name}: OK ({elapsed:.0f}s)")
                results.append((name, "OK", elapsed))
            else:
                log(f"  {name}: FAILED (exit {proc.returncode}, {elapsed:.0f}s)")
                results.append((name, f"FAIL({proc.returncode})", elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            log(f"  {name}: ERROR ({e}, {elapsed:.0f}s)")
            results.append((name, f"ERROR", elapsed))

    log("=" * 60)
    log("PIPELINE SUMMARY")
    for name, status, elapsed in results:
        log(f"  {name}: {status} ({elapsed:.0f}s)")
    log("=" * 60)
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
