"""Round 158: persistent task-log contract tests."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pytest

from src.task_logging import (
    CST,
    LOG_DIR_ENV,
    RUN_ID_ENV,
    TaskLoggingError,
    TaskLogger,
    log_artifact,
    make_logger,
    now_iso,
    prune_runs,
    set_active_logger,
    sha256_file,
    validate_jsonl,
)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_now_iso_has_milliseconds_and_cst_offset():
    value = now_iso()
    assert value.endswith("+08:00")
    parsed = dt.datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == dt.timedelta(hours=8)
    assert len(value.split(".")[-1]) == 9  # 3 ms digits + +08:00


def test_run_id_propagates_and_events_are_ordered(tmp_path, monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, "20261116T150401-12345")
    logger = TaskLogger(tmp_path, "pipeline", read_only=False)
    assert logger.run_id == "20261116T150401-12345"
    logger.start(chapter="ch03")
    logger.event("stage_started", stage="TRANSLATING_RAW", chunk_id="ch03_chunk001")
    logger.event("stage_finished", stage="TRANSLATING_RAW", chunk_id="ch03_chunk001")
    logger.finish(0)

    rows = _read_jsonl(logger.jsonl_path)
    assert [row["event"] for row in rows] == [
        "run_start", "stage_started", "stage_finished", "run_end"]
    assert [row["seq"] for row in rows] == [1, 2, 3, 4]
    assert {row["run_id"] for row in rows} == {"20261116T150401-12345"}
    for row in rows:
        for key in ("ts", "seq", "run_id", "command", "stage", "event", "status"):
            assert key in row
    text_lines = logger.text_path.read_text(encoding="utf-8").splitlines()
    assert len(text_lines) == len(rows)
    assert all(line.startswith("[") and "+08:00]" in line for line in text_lines)


def test_secret_and_content_redaction(tmp_path):
    logger = TaskLogger(tmp_path, "pipeline", read_only=False)
    logger.start()
    logger.event(
        "command_failed",
        stage="LLM",
        status="failed",
        message="Authorization: Bearer supersecret-token",
        api_key="sk-abcdefghijklmnop",
        text="正文" * 400,
    )
    logger.finish(1)
    raw = logger.jsonl_path.read_text(encoding="utf-8")
    assert "supersecret-token" not in raw
    assert "sk-abcdefghijklmnop" not in raw
    assert "[REDACTED]" in raw
    assert "content truncated" in raw


def test_run_end_seal_hashes_match(tmp_path):
    logger = TaskLogger(tmp_path, "pipeline", read_only=False)
    logger.start()
    logger.event("artifact_written", stage="FINALIZE", artifact="output/a.json")
    logger.finish(0)
    seal = json.loads(logger.seal_path.read_text(encoding="utf-8"))
    assert seal["run_id"] == logger.run_id
    assert seal["status"] == "ok"
    assert seal["jsonl_sha256"] == sha256_file(logger.jsonl_path)
    assert seal["log_sha256"] == sha256_file(logger.text_path)
    assert seal["seq_last"] == 3


def test_artifact_event_records_hash(tmp_path):
    artifact = tmp_path / "chunk_final.json"
    artifact.write_text('{"text":"ok"}', encoding="utf-8")
    logger = TaskLogger(tmp_path, "pipeline", read_only=False)
    set_active_logger(logger)
    try:
        logger.start()
        log_artifact(artifact, stage="FINALIZE", chunk_id="ch01_chunk001", kind="final")
        logger.finish(0)
    finally:
        set_active_logger(None)
    rows = _read_jsonl(logger.jsonl_path)
    event = [row for row in rows if row["event"] == "artifact_written"][0]
    assert event["artifact_sha256"] == "sha256:" + sha256_file(artifact)
    assert event["chunk_id"] == "ch01_chunk001"


def test_disabled_logger_creates_no_files(tmp_path):
    logger = TaskLogger(tmp_path, "gate", read_only=True, enabled=False)
    logger.start()
    logger.finish(0)
    assert not (tmp_path / "output").exists()


def test_context_manager_records_exception(tmp_path):
    logger = TaskLogger(tmp_path, "pipeline", read_only=False)
    with pytest.raises(RuntimeError):
        with logger:
            raise RuntimeError("boom")
    rows = _read_jsonl(logger.jsonl_path)
    assert rows[-1]["event"] == "run_end"
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["error_class"] == "RuntimeError"


def test_context_manager_records_system_exit_130(tmp_path):
    logger = TaskLogger(tmp_path, "pipeline", read_only=False)
    with pytest.raises(SystemExit):
        with logger:
            raise SystemExit(130)
    rows = _read_jsonl(logger.jsonl_path)
    assert rows[-1]["status"] == "interrupted"
    assert rows[-1]["exit_code"] == 130


def test_validate_jsonl_detects_seq_gap(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = [
        {"ts": now_iso(), "seq": 1, "run_id": "r1", "command": "gate",
         "stage": "RUN", "event": "run_start", "status": "running"},
        {"ts": now_iso(), "seq": 3, "run_id": "r1", "command": "gate",
         "stage": "RUN", "event": "run_end", "status": "ok"},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")
    result = validate_jsonl(path, expected_run_id="r1")
    assert result["ok"] is False
    assert any("seq_gap" in item for item in result["errors"])


def test_two_loggers_share_monotonic_sequence(tmp_path, monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, "shared-run")
    first = TaskLogger(tmp_path, "pipeline", read_only=False)
    second = TaskLogger(tmp_path, "pipeline", read_only=False)
    first.start()
    second.event("stage_started", stage="A")
    first.event("stage_started", stage="B")
    second.finish(0)
    first.finish(0)
    rows = _read_jsonl(first.jsonl_path)
    seqs = [row["seq"] for row in rows]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def test_write_logger_fails_closed_when_log_dir_is_not_a_directory(tmp_path):
    bogus = tmp_path / "not-a-dir"
    bogus.write_text("x", encoding="utf-8")
    with pytest.raises(TaskLoggingError):
        make_logger(tmp_path, "pipeline", read_only=False, log_dir=bogus)


def test_read_only_logger_degrades_without_raising(tmp_path, capsys):
    bogus = tmp_path / "not-a-dir"
    bogus.write_text("x", encoding="utf-8")
    logger = make_logger(tmp_path, "gate", read_only=True, log_dir=bogus)
    assert logger.enabled is False
    assert "任务日志不可写" in capsys.readouterr().err


def test_prune_runs_preserves_current_and_keeps_recent(tmp_path):
    log_dir = tmp_path / "runs"
    log_dir.mkdir()
    now = dt.datetime.now().timestamp()
    for index, run_id in enumerate(("old-1", "old-2", "new-1")):
        for suffix in (".jsonl", ".log", ".sha256"):
            path = log_dir / f"{run_id}{suffix}"
            path.write_text("x", encoding="utf-8")
            os.utime(path, (now - 400 * 86400 + index, now - 400 * 86400 + index))
    removed = prune_runs(log_dir, keep_last=1, keep_days=90, current_run_id="new-1")
    assert "new-1" not in removed
    assert (log_dir / "new-1.jsonl").exists()
    assert not (log_dir / "old-1.jsonl").exists()
