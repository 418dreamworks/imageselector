#!/usr/bin/env python3
"""Monitor Etsy API rate limit and dynamically adjust QPS.

Also monitors and restarts sync_data.py and bg_remover.py if stopped.

Runs every 10 minutes:
- If remaining < 90k: decrease QPS by 10%
- If remaining > 90k: increase QPS by 10% (max 5 QPS)
- Restart sync_data.py if not running
- Restart bg_remover.py if not running

Writes to qps_config.json which sync_data.py reads.
"""
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import httpx

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "qps_config.json"
LOG_FILE = BASE_DIR / "qps_monitor.log"
SYNC_LOG = BASE_DIR / "sync_data.log"
BG_LOG = BASE_DIR / "bg_remover.log"

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# Limits
MIN_DELAY = 0.2    # 5 QPS max
MAX_DELAY = 2.0    # 0.5 QPS min
THRESHOLD = 90000  # Rate limit threshold
CHECK_INTERVAL = 600  # 10 minutes


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    line = f"[{ts()}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"api_delay": 0.2}


def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def check_rate_limit() -> dict | None:
    """Make a lightweight API call and return rate limit info."""
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{BASE_URL}/application/openapi-ping",
                headers={"x-api-key": ETSY_API_KEY}
            )
            # Both per-second and per-day limits
            return {
                "per_second": {
                    "limit": int(response.headers.get("x-limit-per-second", 0)),
                    "remaining": int(response.headers.get("x-remaining-this-second", 0)),
                },
                "per_day": {
                    "limit": int(response.headers.get("x-limit-per-day", 0)),
                    "remaining": int(response.headers.get("x-remaining-today", 0)),
                }
            }
    except Exception as e:
        log(f"Error checking rate limit: {e}")
        return None


def adjust_qps(limits: dict, current_delay: float) -> float:
    """Adjust delay based on remaining rate limits."""
    per_sec = limits["per_second"]
    per_day = limits["per_day"]

    # Check which limit is the bottleneck
    sec_pct = per_sec["remaining"] / max(per_sec["limit"], 1) * 100
    day_pct = per_day["remaining"] / max(per_day["limit"], 1) * 100
    day_remaining = per_day["remaining"]

    # Determine action based on daily remaining (main concern)
    if day_remaining < THRESHOLD:
        new_delay = min(current_delay * 1.1, MAX_DELAY)
        action = "DECREASE"
        trigger = "DAILY"
    else:
        new_delay = max(current_delay * 0.9, MIN_DELAY)
        action = "INCREASE"
        trigger = "OK"

    new_qps = 1.0 / new_delay
    old_qps = 1.0 / current_delay

    # Always log status
    log(f"Limits: sec={per_sec['remaining']}/{per_sec['limit']} ({sec_pct:.0f}%), "
        f"day={per_day['remaining']:,}/{per_day['limit']:,} ({day_pct:.1f}%) | "
        f"{trigger} -> {action} QPS: {old_qps:.2f} -> {new_qps:.2f}")

    return new_delay


def is_process_running(name: str) -> bool:
    """Check if a process with given name is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def start_sync_data():
    """Start sync_data.py in background."""
    log("Starting sync_data.py...")
    subprocess.Popen(
        ["python", "-u", str(BASE_DIR / "sync_data.py")],
        stdout=open(SYNC_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        start_new_session=True
    )
    log("sync_data.py started")


def start_bg_remover():
    """Start bg_remover.py in background with GPU."""
    log("Starting bg_remover.py --gpu --watch...")
    subprocess.Popen(
        ["python", "-u", str(BASE_DIR / "scripts" / "bg_remover.py"), "--gpu", "--watch"],
        stdout=open(BG_LOG, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        start_new_session=True
    )
    log("bg_remover.py started")


def check_and_restart_processes():
    """Check if sync_data.py and bg_remover.py are running, restart if not."""
    # Check sync_data.py
    if not is_process_running("sync_data.py"):
        log("sync_data.py NOT running!")
        start_sync_data()

    # Check bg_remover.py
    if not is_process_running("bg_remover.py"):
        log("bg_remover.py NOT running!")
        start_bg_remover()


def main():
    log("=" * 50)
    log("QPS Monitor started")
    log(f"Threshold: {THRESHOLD:,} | Check interval: {CHECK_INTERVAL}s")
    log("=" * 50)

    while True:
        # Check and restart processes if needed
        check_and_restart_processes()

        # Check rate limit and adjust QPS
        config = load_config()
        current_delay = config.get("api_delay", 0.2)

        limits = check_rate_limit()
        if limits is not None:
            new_delay = adjust_qps(limits, current_delay)
            if abs(new_delay - current_delay) > 0.001:
                config["api_delay"] = new_delay
                config["last_updated"] = datetime.now().isoformat()
                config["limits"] = limits
                save_config(config)
        else:
            log("Could not get rate limit info")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
