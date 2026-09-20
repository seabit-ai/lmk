"""Structured logs: one JSON object per line, every line carries `event`.

Same contract as kitten's Go side (xklib): a line without an event name is
unsearchable. Written to stderr AND to a file that survives a reboot.
"""
import json
import sys
import threading
from pathlib import Path

from lmk.clock import get_current_clock

_lock = threading.Lock()
_file = None


def open_log_file(log_dir: Path) -> None:
    global _file
    log_dir.mkdir(parents=True, exist_ok=True)
    _file = open(log_dir / "lmk.jsonl", "a", buffering=1)


def _emit(level: str, event: str, msg: str, fields: dict) -> None:
    record = {"time_ms": get_current_clock().wall_ms(), "level": level, "event": event, "msg": msg}
    record.update(fields)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock:
        print(line, file=sys.stderr, flush=True)
        if _file is not None:
            _file.write(line + "\n")


def info(event: str, msg: str, **fields) -> None:
    _emit("INFO", event, msg, fields)


def warn(event: str, msg: str, **fields) -> None:
    _emit("WARN", event, msg, fields)


def error(event: str, msg: str, **fields) -> None:
    _emit("ERROR", event, msg, fields)
