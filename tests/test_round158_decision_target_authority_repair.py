"""Round 158: evidence-gated Decision target correction without prose rewrite."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.decision_target_authority_repair import (
    apply_decision_target_repairs,
    audit_decision_target_repairs,
)
from src.main import _MUTATING_COMMANDS, _command_needs_lock
from src.pending_backtrack_reconcile import (
    classify_pending_snapshot,
    load_pending_context,
)


def _make_fixture(root: Path, *, target_hits: int = 1) -> None:
    db = root / "db"
    db.mkdir(parents=True)
    workflow = sqlite3.connect(str(db / "workflow.db"))
    try:
        workflow.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, "
            "chapter_id TEXT, text_content TEXT, state TEXT, last_error TEXT, "
            "backtrack_rounds INTEGER, rescue_stage TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        rows = []
        for index in range(1, 6):
            rows.append((
                "ch01_chunk%03d" % index,
                ("A guard waited." if index == 5
                 else "The city of Carthag is described."),
                "COMPLETED",
            ))
        workflow.executemany(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, "
            "state, backtrack_rounds) VALUES (?, 'ch01', ?, ?, 0)",
            rows,
        )
        workflow.commit()
    finally:
        workflow.close()

    decisions = sqlite3.connect(str(db / "decision_db.sqlite"))
    try:
        decisions.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, reason TEXT, updated_at TEXT, "
            "locked INTEGER DEFAULT 0)"
        )
        decisions.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, "
            "canonical_name TEXT, source_names TEXT, context_names TEXT, "
            "variant_names TEXT, quarantined_names TEXT, entity_type TEXT, "
            "confidence TEXT, pending_review INTEGER, review_reason TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        decisions.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)"
        )
        decisions.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "reason, updated_at, locked) VALUES (1, 2, 'Carthag', '迦太基', "
            "'legacy city name', '2026-11-17 00:00:00', 0)"
        )
        decisions.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, 1, 'Carthag', 1, "
            "'2026-11-17 00:00:00', '2026-11-17 00:00:00')",
            [("ch01_chunk001",), ("ch01_chunk002",)],
        )
        decisions.commit()
    finally:
        decisions.close()

    texts = {
        "ch01_chunk001": "卡萨格城在东北方。",
        "ch01_chunk002": "他们从卡萨格出发。",
        "ch01_chunk003": "卡萨格是哈克南人的城市。",
        "ch01_chunk004": "迦太基城已被摧毁。",
        "ch01_chunk005": "一名卫兵等待着。",
    }
    if target_hits >= 2:
        texts["ch01_chunk003"] = "迦太基是哈克南人的城市。"
    if target_hits >= 3:
        texts["ch01_chunk002"] = "他们从迦太基出发。"
    out = root / "output" / "ch01"
    out.mkdir(parents=True)
    for chunk_id, text in texts.items():
        (out / (chunk_id + "_final.json")).write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    (root / "input").mkdir()
    (root / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")


def test_decision_target_repair_plans_only_strong_corpus_majority(
        tmp_path: Path):
    _make_fixture(tmp_path)
    before_final = (tmp_path / "output" / "ch01" /
                    "ch01_chunk001_final.json").read_bytes()

    result = audit_decision_target_repairs(tmp_path)

    assert result["ok"] is True
    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["decision_id"] == 1
    assert candidate["old_translation"] == "迦太基"
    assert candidate["new_translation"] == "卡萨格"
    assert candidate["stale_chunks"] == ["ch01_chunk004"]
    assert (tmp_path / "output" / "ch01" /
            "ch01_chunk001_final.json").read_bytes() == before_final


def test_decision_target_repair_updates_authority_and_queues_stale_final(
        tmp_path: Path):
    _make_fixture(tmp_path)
    finals_before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "output" / "ch01").glob("*_final.json")
    }

    result = apply_decision_target_repairs(tmp_path, apply=True, run_id="test")

    assert result["ok"] is True
    assert result["write_attempted"] is True
    action = result["actions"][0]
    assert action["new_translation"] == "卡萨格"
    assert action["queued_chunks"] == ["ch01_chunk004"]
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    con.row_factory = sqlite3.Row
    try:
        decision = con.execute(
            "SELECT translation, reason FROM decision_db WHERE id=1").fetchone()
        queued = con.execute(
            "SELECT decision_id, source_key FROM pending_backtrack "
            "WHERE chunk_id='ch01_chunk004'").fetchone()
    finally:
        con.close()
    assert decision["translation"] == "卡萨格"
    assert "Round 158 decision target authority repair" in decision["reason"]
    assert queued["decision_id"] == 1
    assert queued["source_key"] == "Carthag"
    finals_after = {
        path.name: path.read_bytes()
        for path in (tmp_path / "output" / "ch01").glob("*_final.json")
    }
    assert finals_after == finals_before

    context = load_pending_context(tmp_path)
    snapshot = classify_pending_snapshot(context)
    statuses = {row["chunk_id"]: row["status"]
                for row in snapshot["entries"]}
    assert statuses["ch01_chunk001"] == "satisfied"
    assert statuses["ch01_chunk002"] == "satisfied"
    assert statuses["ch01_chunk004"] == "outstanding"


def test_decision_target_repair_rejects_split_body(tmp_path: Path):
    _make_fixture(tmp_path, target_hits=2)

    result = audit_decision_target_repairs(tmp_path)

    assert result["ok"] is True
    assert result["counts"]["candidates"] == 0
    blocked = {row["source_key"]: row for row in result["blocked_groups"]}
    assert blocked["Carthag"]["reason"] == "no_safe_candidate"


def test_decision_target_repair_is_in_mutating_command_set():
    assert "decision-target-authority-repair" in _MUTATING_COMMANDS

    class Args:
        apply = False

    assert _command_needs_lock("decision-target-authority-repair", Args()) is True
