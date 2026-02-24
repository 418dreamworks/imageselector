#!/usr/bin/env python3
"""Pipeline runner — concurrent main steps, then sequential finishers.

Phase 1 (concurrent): sync_data, image_downloader, embed_orchestrator, tar_images
  - All four start simultaneously
  - sync_data is the driver: once it exits, the others get 5 min grace then KILL signals
Phase 2 (sequential): update_primary → backup_db → backup_rsync × 3

Kill: touch KILL_PIPELINE (signals all children, skips Phase 2)
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

# Kill files for each concurrent step
KILL_FILES = {
    "sync_data": BASE_DIR / "KILL_SD",
    "image_downloader": BASE_DIR / "KILL_DL",
    "embed_orchestrator": BASE_DIR / "KILL_ORCH",
    "tar_images": BASE_DIR / "KILL_TAR",
}

CONCURRENT_STEPS = [
    {"name": "sync_data",          "cmd": [VENV_PYTHON, "bin/sync_data.py"],                   "log": "logs/sync_data.log"},
    {"name": "image_downloader",   "cmd": [VENV_PYTHON, "bin/image_downloader.py"],             "log": "logs/image_downloader.log"},
    {"name": "embed_orchestrator", "cmd": [VENV_PYTHON, "bin/embedding/embed_orchestrator.py"], "log": "logs/embed_orchestrator.log"},
    {"name": "tar_images",         "cmd": [VENV_PYTHON, "bin/tar_images.py"],                   "log": "logs/tar_images.log"},
]

SEQUENTIAL_STEPS = [
    {"name": "update_primary",    "cmd": [VENV_PYTHON, "bin/update_primary.py"],       "log": "logs/update_primary.log"},
    {"name": "backup_db",         "cmd": [VENV_PYTHON, "bin/backup_db.py"],            "log": "logs/backup_db.log"},
    {"name": "backup_rsync_data", "cmd": [VENV_PYTHON, "bin/backup_rsync.py", "data"], "log": "logs/rsync_ssd500.log"},
    {"name": "backup_rsync_hdd",  "cmd": [VENV_PYTHON, "bin/backup_rsync.py", "hdd"],  "log": "logs/rsync_hdd3tb.log"},
    {"name": "backup_rsync_ssd",  "cmd": [VENV_PYTHON, "bin/backup_rsync.py", "ssd"],  "log": "logs/rsync_hdd500.log"},
]

GRACE_PERIOD = 300  # 5 min


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def start_subprocess(step):
    """Start a step as a subprocess writing to its own log file."""
    log_path = BASE_DIR / step["log"]
    log_path.parent.mkdir(exist_ok=True)
    lf = open(log_path, "a")
    proc = subprocess.Popen(
        step["cmd"],
        cwd=str(BASE_DIR),
        stdout=lf,
        stderr=lf,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return proc, lf


def signal_stop(names):
    """Touch kill files for the given step names."""
    for name in names:
        kf = KILL_FILES.get(name)
        if kf:
            kf.touch()
            log(f"  Touched {kf.name}")


def cleanup_kill_files():
    """Remove all kill files we manage."""
    for kf in KILL_FILES.values():
        try:
            kf.unlink()
        except FileNotFoundError:
            pass


def wait_for_procs(procs, timeout=None):
    """Wait for all procs to exit. Returns True if all exited, False on timeout."""
    deadline = time.time() + timeout if timeout else None
    while any(p.poll() is None for p in procs):
        if deadline and time.time() > deadline:
            return False
        time.sleep(2)
    return True


def run_sequential_step(step):
    """Run a single step, streaming output to log and stdout. Returns (status, elapsed)."""
    log_path = BASE_DIR / step["log"]
    log_path.parent.mkdir(exist_ok=True)
    name = step["name"]
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
            return "OK", elapsed
        else:
            return f"FAIL({proc.returncode})", elapsed
    except Exception as e:
        return f"ERROR({e})", time.time() - t0


def main():
    if KILL_FILE.exists():
        log(f"Error: Kill file exists ({KILL_FILE}). Remove it to start.")
        return

    log("=" * 60)
    log("PIPELINE START (concurrent mode)")
    log("=" * 60)

    results = []
    killed = False

    # ── Phase 1: Start all concurrent steps ──
    log("Phase 1: Starting concurrent steps")
    procs = {}   # name -> proc
    logfds = {}  # name -> file descriptor
    start_times = {}

    for step in CONCURRENT_STEPS:
        name = step["name"]
        proc, lf = start_subprocess(step)
        procs[name] = proc
        logfds[name] = lf
        start_times[name] = time.time()
        log(f"  Started {name} (PID {proc.pid})")

    # ── Wait for sync_data to finish (it's the driver) ──
    log("Waiting for sync_data to finish...")
    while procs["sync_data"].poll() is None:
        if KILL_FILE.exists():
            log("KILL_PIPELINE detected during Phase 1")
            killed = True
            break
        time.sleep(5)

    if killed:
        # Signal all concurrent steps to stop
        log("Signaling all concurrent steps to stop...")
        signal_stop(procs.keys())
        wait_for_procs(list(procs.values()), timeout=120)
        # Force-terminate anything still alive
        for name, proc in procs.items():
            if proc.poll() is None:
                log(f"  {name} still running after 120s, terminating...")
                proc.terminate()
                proc.wait(timeout=30)
    else:
        # sync_data finished naturally
        sd_elapsed = time.time() - start_times["sync_data"]
        sd_rc = procs["sync_data"].returncode
        sd_status = "OK" if sd_rc == 0 else f"FAIL({sd_rc})"
        results.append(("sync_data", sd_status, sd_elapsed))
        log(f"  sync_data: {sd_status} ({sd_elapsed:.0f}s)")

        # Grace period for others to process remaining work
        log(f"Grace period: {GRACE_PERIOD // 60} min for remaining work...")
        grace_end = time.time() + GRACE_PERIOD
        while time.time() < grace_end:
            if KILL_FILE.exists():
                log("KILL_PIPELINE detected during grace period")
                killed = True
                break
            # If all others already exited, no need to wait
            if all(procs[n].poll() is not None for n in procs if n != "sync_data"):
                log("  All concurrent steps already finished")
                break
            time.sleep(5)

        # Signal remaining concurrent steps to stop
        still_running = [n for n in procs if n != "sync_data" and procs[n].poll() is None]
        if still_running:
            log(f"Signaling stop: {', '.join(still_running)}")
            signal_stop(still_running)
            wait_for_procs([procs[n] for n in still_running], timeout=120)
            for name in still_running:
                if procs[name].poll() is None:
                    log(f"  {name} still running after 120s, terminating...")
                    procs[name].terminate()
                    procs[name].wait(timeout=30)

    # Collect results for concurrent steps (except sync_data already recorded)
    for step in CONCURRENT_STEPS:
        name = step["name"]
        if name == "sync_data" and not killed:
            continue  # Already recorded
        elapsed = time.time() - start_times[name]
        rc = procs[name].returncode
        if rc is None:
            status = "KILLED"
        elif rc == 0:
            status = "OK"
        else:
            status = f"FAIL({rc})"
        results.append((name, status, elapsed))
        log(f"  {name}: {status} ({elapsed:.0f}s)")

    # Close log file descriptors
    for lf in logfds.values():
        lf.close()

    # Clean up kill files
    cleanup_kill_files()

    # ── Phase 2: Sequential steps (skip if killed) ──
    if not killed:
        log("")
        log("Phase 2: Sequential steps")
        for step in SEQUENTIAL_STEPS:
            if KILL_FILE.exists():
                log("Kill file detected. Skipping remaining sequential steps.")
                killed = True
                break

            name = step["name"]
            log(f"--- {name} ---")
            status, elapsed = run_sequential_step(step)
            log(f"  {name}: {status} ({elapsed:.0f}s)")
            results.append((name, status, elapsed))
    else:
        log("Phase 2 skipped (pipeline killed)")

    # ── Summary ──
    log("")
    log("=" * 60)
    log("PIPELINE SUMMARY")
    for name, status, elapsed in results:
        log(f"  {name}: {status} ({elapsed:.0f}s)")
    log("=" * 60)
    log("PIPELINE COMPLETE" if not killed else "PIPELINE KILLED")


if __name__ == "__main__":
    main()
