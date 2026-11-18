#!/usr/bin/env python3
"""Prefix console output with a CST timestamp and append it to a run log."""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path


CST = _dt.timezone(_dt.timedelta(hours=8), name="Asia/Shanghai")


def _now() -> str:
    return _dt.datetime.now(CST).isoformat(timespec="milliseconds")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: task_log_tee.py <run.log>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    with open(path, "a", encoding="utf-8", buffering=1) as handle:
        for raw in sys.stdin:
            text = raw.rstrip("\n")
            handle.write(f"[{_now()}] {text}\n")
            handle.flush()
            sys.stdout.write(raw)
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
