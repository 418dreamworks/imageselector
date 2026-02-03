#!/usr/bin/env python3
"""Orchestrator for iMac pipeline: sync → bg_remove → embed.

Runs all three processes continuously with monitoring.
Uses subprocess to manage each component.
"""
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
KILL_FILE = BASE_DIR / "KILL"
LOG_FILE = BASE_DIR / "orchestrator.log"

# Process configurations
PROCESSES = {
    "sync_data": {
        "script": BASE_DIR / "sync_data.py",
        "args": [],
        "restart_delay": 60,  # Wait 1 min before restart
    },
    "bg_remover": {
        "script": BASE_DIR / "scripts" / "bg_remover.py",
        "args": ["--watch", "--gpu"],
        "restart_delay": 30,
    },
    "embed": {
        "script": BASE_DIR / "image_search_v2" / "embed.py",
        "args": ["--model", "all", "--batch-size", "8"],
        "restart_delay": 300,  # Wait 5 min between embed runs
    },
}


def log(msg: str):
    """Log message with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_kill_file() -> bool:
    """Check for kill file and delete if found."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        log("Kill file detected. Shutting down...")
        return True
    return False


def run_process(name: str, config: dict) -> subprocess.Popen:
    """Start a process and return the Popen object."""
    script = config["script"]
    args = config["args"]

    cmd = [sys.executable, str(script)] + args
    log(f"Starting {name}: {' '.join(cmd)}")

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(BASE_DIR),
    )


def monitor_process(proc: subprocess.Popen, name: str) -> bool:
    """Check if process is still running. Returns True if running."""
    if proc.poll() is None:
        return True

    # Process finished
    returncode = proc.returncode
    log(f"{name} exited with code {returncode}")
    return False


def main():
    log("=" * 60)
    log("Orchestrator starting")
    log("=" * 60)

    running_procs = {}
    last_restart = {}

    for name in PROCESSES:
        last_restart[name] = 0

    try:
        while True:
            if check_kill_file():
                break

            now = time.time()

            for name, config in PROCESSES.items():
                # Check if process needs to be started/restarted
                if name not in running_procs or not monitor_process(running_procs[name], name):
                    # Check restart delay
                    if now - last_restart[name] > config["restart_delay"]:
                        running_procs[name] = run_process(name, config)
                        last_restart[name] = now

            # Sleep before next check
            time.sleep(10)

    except KeyboardInterrupt:
        log("Interrupted by user")

    finally:
        # Terminate all running processes
        log("Stopping all processes...")
        for name, proc in running_procs.items():
            if proc.poll() is None:
                log(f"Terminating {name}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

        log("Orchestrator stopped")


if __name__ == "__main__":
    main()
