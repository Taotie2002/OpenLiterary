"""Round 158: shell tee/finalizer helper tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.task_logging import now_iso, sha256_file


ROOT = Path(__file__).resolve().parent.parent


def test_task_log_tee_prefixes_each_line_and_preserves_stdout(tmp_path):
    log_path = tmp_path / "run.log"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "task_log_tee.py"), str(log_path)],
        input="第一行\n第二行\n", text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout == "第一行\n第二行\n"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(line.startswith("[") and "+08:00]" in line for line in lines)
    assert "第一行" in lines[0]


def test_task_log_finalizer_recomputes_seal(tmp_path):
    run_id = "20261116T150401-12345"
    jsonl = tmp_path / f"{run_id}.jsonl"
    log = tmp_path / f"{run_id}.log"
    rows = [
        {"ts": now_iso(), "seq": 1, "run_id": run_id, "command": "pipeline",
         "stage": "RUN", "event": "run_start", "status": "running"},
        {"ts": now_iso(), "seq": 2, "run_id": run_id, "command": "pipeline",
         "stage": "RUN", "event": "run_end", "status": "ok"},
    ]
    jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                     encoding="utf-8")
    log.write_text("console line\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "task_log_finalize.py"),
         "--run-id", run_id, "--log-dir", str(tmp_path)],
        text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    seal = json.loads((tmp_path / f"{run_id}.sha256").read_text(encoding="utf-8"))
    assert seal["sealed_by"] == "shell_finalizer"
    assert seal["jsonl_sha256"] == sha256_file(jsonl)
    assert seal["log_sha256"] == sha256_file(log)
    assert seal["validation"]["ok"] is True
