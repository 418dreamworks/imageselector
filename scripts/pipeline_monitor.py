#!/usr/bin/env python3
"""Pipeline monitor: runs and restarts all 4 scripts + monitors API rate limits.

Scripts managed:
1. sync_data.py - discovers listings, adds to SQL
2. image_downloader.py - downloads images from CDN
3. bg_remover.py - removes backgrounds
4. embed.py - generates embeddings

Also adjusts QPS based on API rate limits.

Stop with: touch KILL
"""
import os
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
KILL_FILE = BASE_DIR / "KILL"
CONFIG_FILE = BASE_DIR / "qps_config.json"
LOG_FILE = BASE_DIR / "pipeline.log"

ETSY_API_KEY = os.getenv("ETSY_API_KEY")
BASE_URL = "https://openapi.etsy.com/v3"

# QPS limits
MIN_DELAY = 0.2    # 5 QPS max
MAX_DELAY = 2.0    # 0.5 QPS min
THRESHOLD = 10000  # Rate limit threshold

# Process configurations
PROCESSES = {
    "sync_data": {
        "script": BASE_DIR / "sync_data.py",
        "args": [],
        "restart_delay": 60,
        "log": BASE_DIR / "sync_data.log",
    },
    "image_downloader": {
        "script": BASE_DIR / "scripts" / "image_downloader.py",
        "args": ["--watch"],
        "restart_delay": 30,
        "log": BASE_DIR / "image_downloader.log",
    },
    "bg_remover": {
        "script": BASE_DIR / "scripts" / "bg_remover.py",
        "args": ["--watch", "--gpu"],
        "restart_delay": 30,
        "log": BASE_DIR / "bg_remover.log",
    },
    "embed": {
        "script": BASE_DIR / "image_search_v2" / "embed.py",
        "args": ["--model", "all", "--batch-size", "8"],
        "restart_delay": 300,
        "log": BASE_DIR / "embed.log",
    },
}


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    line = f"[{ts()}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def check_kill_file() -> bool:
    if KILL_FILE.exists():
        KILL_FILE.unlink()
        log("Kill file detected. Shutting down...")
        return True
    return False


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
    try:
        import httpx
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{BASE_URL}/application/openapi-ping",
                headers={"x-api-key": ETSY_API_KEY}
            )
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
    per_sec = limits["per_second"]
    per_day = limits["per_day"]
    day_remaining = per_day["remaining"]

    if day_remaining > THRESHOLD:
        new_delay = max(current_delay / 1.1, MIN_DELAY)
        action = "INCREASE"
    else:
        new_delay = min(current_delay / 0.9, MAX_DELAY)
        action = "DECREASE"

    log(f"API: {per_day['remaining']:,}/{per_day['limit']:,} remaining | {action} QPS: {1/current_delay:.2f} -> {1/new_delay:.2f}")
    return new_delay


def start_process(name: str, config: dict) -> subprocess.Popen:
    script = config["script"]
    args = config["args"]
    log_file = config["log"]

    cmd = [sys.executable, "-u", str(script)] + args
    log(f"Starting {name}")

    return subprocess.Popen(
        cmd,
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        start_new_session=True
    )


def main():
    log("=" * 50)
    log("Pipeline Monitor started")
    log("=" * 50)

    running = {}
    last_restart = {name: 0 for name in PROCESSES}
    last_rate_check = 0
    rate_check_interval = 600  # 10 minutes

    try:
        while True:
            if check_kill_file():
                break

            now = time.time()

            # Check/restart processes
            for name, config in PROCESSES.items():
                proc = running.get(name)

                # Check if needs restart
                if proc is None or proc.poll() is not None:
                    if proc is not None:
                        log(f"{name} exited with code {proc.returncode}")

                    if now - last_restart[name] > config["restart_delay"]:
                        running[name] = start_process(name, config)
                        last_restart[name] = now

            # Check rate limits periodically
            if now - last_rate_check > rate_check_interval:
                cfg = load_config()
                limits = check_rate_limit()
                if limits:
                    new_delay = adjust_qps(limits, cfg.get("api_delay", 0.2))
                    cfg["api_delay"] = new_delay
                    cfg["last_updated"] = datetime.now().isoformat()
                    save_config(cfg)
                last_rate_check = now

            time.sleep(10)

    except KeyboardInterrupt:
        log("Interrupted")

    finally:
        log("Stopping all processes via kill files...")
        # Create kill files for each script
        kill_files = {
            "sync_data": BASE_DIR / "KILL",
            "image_downloader": BASE_DIR / "KILL_DL",
            "bg_remover": BASE_DIR / "KILL_BG",
            "embed": BASE_DIR / "KILL_EMBED",
        }
        for name, kill_file in kill_files.items():
            if name in running and running[name] and running[name].poll() is None:
                log(f"Creating {kill_file.name} for {name}")
                kill_file.touch()

        # Wait for processes to exit gracefully
        for name, proc in running.items():
            if proc and proc.poll() is None:
                log(f"Waiting for {name} to exit...")
                try:
                    proc.wait(timeout=30)
                    log(f"{name} exited")
                except subprocess.TimeoutExpired:
                    log(f"{name} did not exit in time")

        log("Pipeline Monitor stopped")


if __name__ == "__main__":
    main()
