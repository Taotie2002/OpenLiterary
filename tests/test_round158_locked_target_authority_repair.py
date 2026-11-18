"""Round 158: corpus-backed repair for contradicted locked targets."""
from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path

from src.decision_authority_align import (
    apply_locked_target_authority_repairs,
    audit_locked_target_authority_repairs,
)
from src.main import _MUTATING_COMMANDS, _command_needs_lock


def _digest(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_project(root: Path, *, pending_fourth: bool = False,
                  fourth_final: str = "屏障等待。") -> None:
    db = root / "db"
    db.mkdir(parents=True)
    con = sqlite3.connect(str(db / "workflow.db"))
    try:
        con.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, "
            "chapter_id TEXT, text_content TEXT, state TEXT, last_error TEXT, "
            "backtrack_rounds INTEGER, rescue_stage TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        rows = [
            ("ch01_chunk001", "ch01", "The shield held.", "COMPLETED"),
            ("ch01_chunk002", "ch01", "Another shield rose.", "COMPLETED"),
            ("ch01_chunk003", "ch01", "A third shield fell.", "COMPLETED"),
            ("ch01_chunk004", "ch01",
             "The shield waited." if pending_fourth else "Nothing happened.",
             "COMPLETED"),
            ("ch01_chunk009", "ch01", "Nothing happened.", "COMPLETED"),
        ]
        con.executemany(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, "
            "state, backtrack_rounds) VALUES (?, ?, ?, ?, 0)", rows)
        con.commit()
    finally:
        con.close()

    con = sqlite3.connect(str(db / "decision_db.sqlite"))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, reason TEXT, updated_at TEXT, "
            "locked INTEGER DEFAULT 0)")
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, "
            "canonical_name TEXT, source_names TEXT, variant_names TEXT, "
            "quarantined_names TEXT, confidence TEXT, "
            "pending_review INTEGER DEFAULT 0)")
        con.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)")
        con.execute(
            "INSERT INTO decision_db VALUES (315, 2, 'shield', '屏蔽场', "
            "'legacy', '2026-11-17 00:00:00', 1)")
        pending = [
            ("ch01_chunk001", 315, "shield"),
            ("ch01_chunk002", 315, "shield"),
            ("ch01_chunk003", 315, "shield"),
        ]
        if pending_fourth:
            pending.append(("ch01_chunk004", 315, "shield"))
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, ?, ?, 1, "
            "'2026-11-17 00:00:00', '2026-11-17 00:00:00')", pending)
        con.commit()
    finally:
        con.close()

    texts = {
        "ch01_chunk001": "护盾撑住了。",
        "ch01_chunk002": "护盾升起来。",
        "ch01_chunk003": "护盾落下去。",
        "ch01_chunk004": fourth_final,
        "ch01_chunk009": "风声掠过了。",
    }
    out = root / "output" / "ch01"
    out.mkdir(parents=True)
    for chunk_id, text in texts.items():
        (out / f"{chunk_id}_final.json").write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    (root / "input").mkdir()
    (root / "input" / "ch01.md").write_text("# Chapter\n", encoding="utf-8")


def test_corpus_lock_repair_dry_run_is_read_only(tmp_path: Path):
    _make_project(tmp_path)
    db_path = tmp_path / "db" / "decision_db.sqlite"
    before = _digest(db_path)

    result = audit_locked_target_authority_repairs(tmp_path)

    assert _digest(db_path) == before
    assert result["mode"] == "read_only"
    assert result["write_attempted"] is False
    assert result["ok"] is True
    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["decision_id"] == 315
    assert candidate["old_translation"] == "屏蔽场"
    assert candidate["new_translation"] == "护盾"
    assert candidate["authority"]["positive_chunks"] == 3
    assert candidate["authority"]["authority_chunks"] == 3
    assert candidate["pending_chunks"] == [
        "ch01_chunk001", "ch01_chunk002", "ch01_chunk003"]


def test_corpus_lock_repair_apply_updates_only_decision(tmp_path: Path):
    _make_project(tmp_path)
    db_path = tmp_path / "db" / "decision_db.sqlite"
    workflow_path = tmp_path / "db" / "workflow.db"
    final_paths = sorted((tmp_path / "output" / "ch01").glob("*_final.json"))
    before_workflow = _digest(workflow_path)
    before_finals = {path.name: _digest(path) for path in final_paths}

    result = apply_locked_target_authority_repairs(
        tmp_path, apply=True, run_id="corpus-lock-fixture")

    assert result["ok"] is True
    assert result["write_attempted"] is True
    assert Path(result["backup_path"]).exists()
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        decision = con.execute(
            "SELECT translation, reason, locked FROM decision_db WHERE id=315"
        ).fetchone()
        assert decision["translation"] == "护盾"
        assert decision["locked"] == 1
        assert "Round 158 corpus lock authority: 屏蔽场 -> 护盾" in (
            decision["reason"])
        pending = {
            row["chunk_id"] for row in con.execute(
                "SELECT chunk_id FROM pending_backtrack")
        }
        assert pending == {
            "ch01_chunk001", "ch01_chunk002", "ch01_chunk003"}
    finally:
        con.close()
    assert _digest(workflow_path) == before_workflow
    assert {path.name: _digest(path) for path in final_paths} == before_finals
    after = audit_locked_target_authority_repairs(tmp_path)
    assert after["counts"]["candidates"] == 0
    assert after["counts"]["blocked"] == 0


def test_corpus_lock_repair_blocks_when_pending_final_lacks_candidate(
        tmp_path: Path):
    _make_project(tmp_path, pending_fourth=True, fourth_final="屏障等待。")
    db_path = tmp_path / "db" / "decision_db.sqlite"
    before = _digest(db_path)

    result = apply_locked_target_authority_repairs(tmp_path, apply=True)

    assert _digest(db_path) == before
    assert result["write_attempted"] is False
    assert result["ok"] is False
    blocked = result["blocked_candidates"][0]
    assert blocked["decision_id"] == 315
    assert blocked["reason"] == (
        "candidate_absent_from_pending_final:ch01_chunk004")


def test_corpus_lock_repair_command_holds_instance_lock():
    assert "locked-target-authority-repair" in _MUTATING_COMMANDS
    assert _command_needs_lock(
        "locked-target-authority-repair", Namespace()) is True
