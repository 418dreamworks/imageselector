#!/usr/bin/env python3
"""Pipeline runner — concurrent main steps, then sequential finishers.

Phase 1 (concurrent): sync_data, image_downloader, embed_orchestrator, tar_images
  - All four start simultaneously
  - sync_data is the driver: once it exits, the others get 5 min grace then KILL signals
Phase 2 (sequential): update_primary → backup_db → backup_rsync × 3

Kill: touch KILL_PIPELINE (signals all children, skips Phase 2)
"""

import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
KILL_FILE = BASE_DIR / "KILL_PIPELINE"
PID_FILE = BASE_DIR / "pipeline.pid"
VENV_PYTHON = str(BASE_DIR / "venv" / "bin" / "python3")


_lock_acquired = False


def acquire_lock() -> bool:
    """Acquire PID lock. Returns False if another instance is running."""
    global _lock_acquired
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    PID_FILE.write_text(str(os.getpid()))
    _lock_acquired = True
    return True


def release_lock():
    """Release PID lock."""
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass

CHILD_PID_FILES = [
    BASE_DIR / "sync_data.pid",
    BASE_DIR / "image_downloader.pid",
    BASE_DIR / "orchestrator.pid",
]

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

TAR_DIR = BASE_DIR / "images" / "imagetarred"
TAR_IDLE_TIMEOUT = 7200  # 2 hours since last new tar (imac-only takes ~1.5hr per batch)

# SMS alerts via Gmail → T-Mobile gateway
GMAIL_USER = "418dreamworks@gmail.com"
GMAIL_PASS = "mkcsnpzojtgqfnpn"
SMS_TO = "2674743645@tmomail.net"


def send_alert(msg):
    """Send SMS alert. Fails silently."""
    try:
        email = MIMEText(msg)
        email["Subject"] = "Pipeline Alert"
        email["From"] = GMAIL_USER
        email["To"] = SMS_TO
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.starttls()
            s.login(GMAIL_USER, GMAIL_PASS)
            s.send_message(email)
        log(f"Alert sent: {msg}")
    except Exception as e:
        log(f"Warning: failed to send alert: {e}")


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def newest_tar_age():
    """Return seconds since the most recently created tar in imagetarred/, or None if no tars."""
    try:
        tars = [TAR_DIR / f for f in os.listdir(TAR_DIR) if f.endswith(".tar")]
        if not tars:
            return None
        newest_mtime = max(f.stat().st_mtime for f in tars)
        return time.time() - newest_mtime
    except Exception:
        return None


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


def cleanup_stale_pids():
    """Remove stale PID files from child processes (left over from previous runs)."""
    for pf in CHILD_PID_FILES:
        if not pf.exists():
            continue
        try:
            old_pid = int(pf.read_text().strip())
            os.kill(old_pid, 0)
            # Process is alive — don't touch it
            log(f"  Warning: {pf.name} PID {old_pid} is still alive")
        except (ValueError, ProcessLookupError, PermissionError):
            pf.unlink()
            log(f"  Removed stale {pf.name}")


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


REQUIRED_VOLUMES = ["/Volumes/SSD500GB", "/Volumes/HDD1TB"]


def mount_volume(name):
    """Try to mount a volume by name using diskutil. Returns True if mounted."""
    try:
        result = subprocess.run(
            ["diskutil", "list", "external"],
            capture_output=True, text=True, timeout=10
        )
        # Find the disk identifier for this volume name
        for line in result.stdout.splitlines():
            if name in line:
                parts = line.split()
                disk_id = parts[-1]  # e.g. disk4s2
                log(f"  Mounting {name} ({disk_id})...")
                mount_result = subprocess.run(
                    ["diskutil", "mount", disk_id],
                    capture_output=True, text=True, timeout=30
                )
                if mount_result.returncode == 0:
                    log(f"  Mounted {name}")
                    return True
                else:
                    log(f"  Failed to mount {name}: {mount_result.stderr.strip()}")
                    return False
        log(f"  {name} not found in external disks")
        return False
    except Exception as e:
        log(f"  Mount error for {name}: {e}")
        return False


def check_volumes():
    """Verify required external drives are mounted, attempt to mount if not.

    1. Check if mounted
    2. If not, attempt diskutil mount
    3. Wait 30s for drives to settle
    4. Re-check — if still missing, alert via SMS and abort
    """
    missing = [v for v in REQUIRED_VOLUMES if not Path(v).is_dir()]
    if not missing:
        return True

    log(f"Volumes not mounted: {', '.join(missing)}")
    for vol in missing:
        name = Path(vol).name
        mount_volume(name)

    # Wait for drives to settle
    time.sleep(30)

    # Re-check
    still_missing = [v for v in REQUIRED_VOLUMES if not Path(v).is_dir()]
    if still_missing:
        msg = f"Pipeline aborted: volumes not mounted: {', '.join(still_missing)}"
        log(f"Error: {msg}")
        send_alert(msg)
        return False
    return True


def main():
    if KILL_FILE.exists():
        msg = f"Pipeline aborted: kill file exists ({KILL_FILE})"
        log(msg)
        send_alert(msg)
        return

    if not check_volumes():
        return  # check_volumes already sends alert

    if not acquire_lock():
        msg = "Pipeline aborted: another instance already running"
        log(msg)
        send_alert(msg)
        return

    log("=" * 60)
    log("PIPELINE START (concurrent mode)")
    log("=" * 60)

    cleanup_stale_pids()

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

        # Wait for others to finish: poll newest tar mtime, kill after 1hr idle
        log("Waiting for remaining work (kill after 1hr with no new tars)...")
        while True:
            if KILL_FILE.exists():
                log("KILL_PIPELINE detected while waiting")
                killed = True
                break
            if all(procs[n].poll() is not None for n in procs if n != "sync_data"):
                log("  All concurrent steps already finished")
                break
            age = newest_tar_age()
            if age is not None and age > TAR_IDLE_TIMEOUT:
                log(f"  Last tar is {age:.0f}s old (>{TAR_IDLE_TIMEOUT}s). Stopping others.")
                break
            time.sleep(30)

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

    # ── Cleanup: remove dead URL rows before backup ──
    if not killed:
        try:
            import sqlite3
            db_path = BASE_DIR / "data" / "db" / "etsy_data.db"
            conn = sqlite3.connect(str(db_path), timeout=300)
            deleted = conn.execute("DELETE FROM image_status WHERE download_done = -1").rowcount
            conn.commit()
            conn.close()
            if deleted:
                log(f"Cleaned up {deleted} dead URL rows (dl=-1)")
        except Exception as e:
            log(f"Warning: failed to clean dead URLs: {e}")

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
    if killed:
        log("PIPELINE KILLED")
        send_alert("Pipeline was killed")
    else:
        failures = [n for n, s, _ in results if not s.startswith("OK")]
        if failures:
            msg = f"Pipeline done with failures: {', '.join(failures)}"
            log(msg)
            send_alert(msg)
        else:
            log("PIPELINE COMPLETE")
            # Clean up KILL_PIPELINE if it exists from a previous run
            try:
                KILL_FILE.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    try:
        main()
    finally:
        if _lock_acquired:
            release_lock()
