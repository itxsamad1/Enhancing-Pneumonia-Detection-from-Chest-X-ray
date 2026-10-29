"""
Shared pause/resume control file for training.
Read-only for the monitor; written by monitor API and read by train_cv / run_queue.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTROL_PATH = ROOT / "revision_results" / "control.json"
HEARTBEAT_PATH = ROOT / "revision_results" / "live_heartbeat.json"

DEFAULT = {
    "state": "running",  # running | paused | stop
    "message": "",
    "updated_at": None,
}


def read_control() -> dict:
    if not CONTROL_PATH.exists():
        return dict(DEFAULT)
    try:
        data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
        out = dict(DEFAULT)
        out.update(data)
        return out
    except Exception:
        return dict(DEFAULT)


def write_control(state: str, message: str = ""):
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "state": state,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CONTROL_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def wait_if_paused(poll_seconds: float = 5.0, label: str = ""):
    """Block while control state is 'paused'. Returns current state."""
    while True:
        ctrl = read_control()
        state = ctrl.get("state", "running")
        if state != "paused":
            return state
        msg = ctrl.get("message") or "paused by user"
        print(f"  [PAUSED] {label} waiting... ({msg})", flush=True)
        time.sleep(poll_seconds)


def write_heartbeat(**fields):
    """Live status for the web monitor. Never raises — must not kill training."""
    try:
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = dict(fields)
        data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["updated_ts"] = time.time()
        payload = json.dumps(data, indent=2)
        # Direct overwrite is more reliable on Windows than tmp+replace
        # (replace fails with WinError 5 if monitor/antivirus holds the file).
        HEARTBEAT_PATH.write_text(payload, encoding="utf-8")
    except Exception:
        pass


def read_heartbeat() -> dict:
    if not HEARTBEAT_PATH.exists():
        return {}
    try:
        return json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
