"""Round 158: read-only interrupted-inflight recovery audit tests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.inflight_recovery import audit_and_write, scan_inflight


def _make_db(root: Path):
    db_dir = root / "db"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "workflow.db")
    conn.execute(
        "CREATE TABLE chunk_tasks ("
        "chunk_id TEXT PRIMARY KEY, chapter_id TEXT, text_content TEXT, "
        "state TEXT NOT NULL, retries INTEGER DEFAULT 0, quality_retries INTEGER DEFAULT 0, "
        "last_error TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "backtrack_rounds INTEGER DEFAULT 0, retry_effort_overrides TEXT, "
        "pf_requeues INTEGER DEFAULT 0, rescue_attempts INTEGER DEFAULT 0, rescue_stage TEXT)"
    )
    conn.executemany(
        "INSERT INTO chunk_tasks (chunk_id, chapter_id, state, retries, quality_retries, "
        "last_error, backtrack_rounds, rescue_attempts, rescue_stage) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("ch03_chunk091", "ch03", "TRANSLATING_RAW", 1, 0, "interrupted", 0, 1, ""),
            ("ch03_chunk101", "ch03", "EXTRACTING_TERMS", 0, 0, "", 0, 0, ""),
            ("ch03_chunk102", "ch03", "EXTRACTING_TERMS", 0, 0, "", 0, 0, ""),
            ("ch03_chunk200", "ch03", "DIRTY", 0, 0, "decision", 1, 0, ""),
            ("ch01_chunk001", "ch01", "COMPLETED", 0, 0, "", 0, 0, ""),
        ],
    )
    conn.commit()
    conn.close()


def test_scan_inflight_finds_only_active_stages_and_preserves_backups(tmp_path):
    _make_db(tmp_path)
    out = tmp_path / "output" / "ch03"
    out.mkdir(parents=True)
    (out / "ch03_chunk091_final_backup.json").write_text('{"text":"old"}', encoding="utf-8")
    (out / "ch03_chunk091_critic_report_backup.json").write_text('{"score":8}', encoding="utf-8")
    (out / "ch03_chunk101_final.json").write_text('{"text":"current"}', encoding="utf-8")

    before = sqlite3.connect(tmp_path / "db" / "workflow.db")
    states_before = before.execute("SELECT chunk_id,state FROM chunk_tasks ORDER BY chunk_id").fetchall()
    before.close()

    report = scan_inflight(tmp_path, chapter_id="ch03", run_id="run-test")
    assert report["summary"]["nonterminal"] == 3
    assert report["summary"]["by_state"] == {"TRANSLATING_RAW": 1, "EXTRACTING_TERMS": 2}
    ids = {item["chunk_id"] for item in report["chunks"]}
    assert ids == {"ch03_chunk091", "ch03_chunk101", "ch03_chunk102"}
    assert "ch03_chunk200" not in ids
    assert "ch01_chunk001" not in ids
    row = [item for item in report["chunks"] if item["chunk_id"] == "ch03_chunk091"][0]
    assert row["final_exists"] is False
    assert row["final_backup_present"] is True
    assert row["critic_report_backup_present"] is True
    assert row["backup_preserved"] is True
    assert row["safe_to_skip_stage"] is False
    assert row["completed"] is False
    assert row["resume_stage"] == "TRANSLATING_RAW"

    after = sqlite3.connect(tmp_path / "db" / "workflow.db")
    states_after = after.execute("SELECT chunk_id,state FROM chunk_tasks ORDER BY chunk_id").fetchall()
    after.close()
    assert states_after == states_before
    assert (out / "ch03_chunk091_final_backup.json").exists()


def test_audit_and_write_publishes_run_report(tmp_path):
    _make_db(tmp_path)
    result = audit_and_write(tmp_path, chapter_id="ch03", run_id="20261116T150401-12345")
    path = Path(result["report_path"])
    assert path == tmp_path / "output" / "recovery" / "20261116T150401-12345.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "20261116T150401-12345"
    assert payload["summary"]["nonterminal"] == 3
    assert all(item["status"] == "interrupted_inflight" for item in payload["chunks"])


def test_scan_missing_db_is_non_destructive(tmp_path):
    report = scan_inflight(tmp_path, run_id="r1")
    assert report["summary"]["nonterminal"] == 0
    assert "workflow_db_missing" in report["warnings"]
