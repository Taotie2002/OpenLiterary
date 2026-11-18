"""Round 158: Decision target repair when registry variants disagree."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.decision_authority_align import (
    apply_decision_authority_alignment,
    audit_decision_authorities,
)


def _make_project(root: Path, *, third_form: bool = False) -> None:
    (root / "db").mkdir(parents=True)
    workflow = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        workflow.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, "
            "chapter_id TEXT, text_content TEXT, state TEXT)"
        )
        workflow.executemany(
            "INSERT INTO chunk_tasks VALUES (?, 'ch01', ?, ?)",
            [
                ("ch01_chunk000", "Leto waited.", "COMPLETED"),
                ("ch01_chunk001", "Leto spoke.", "COMPLETED"),
                ("ch01_chunk002", "Leto turned.", "COMPLETED"),
                ("ch01_chunk003", "Leto thought.", "COMPLETED"),
                ("ch01_chunk004", "Leto ordered it.", "DIRTY"),
            ],
        )
        workflow.commit()
    finally:
        workflow.close()

    decisions = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        decisions.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, reason TEXT, updated_at TEXT, "
            "locked INTEGER DEFAULT 0)"
        )
        decisions.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, "
            "canonical_name TEXT, source_names TEXT, variant_names TEXT, "
            "quarantined_names TEXT, confidence TEXT, pending_review INTEGER)"
        )
        decisions.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)"
        )
        decisions.execute(
            "INSERT INTO decision_db VALUES "
            "(113, 2, 'Leto', '莱托', 'legacy', '2026-11-16 00:00:00', 0)"
        )
        decisions.execute(
            "INSERT INTO entity_registry VALUES (27, '雷托·厄崔迪公爵', ?, ?, ?, "
            "'high', 0)",
            (
                json.dumps(["Duke Leto", "Leto Atreides"], ensure_ascii=False),
                json.dumps([
                    "雷托·厄崔迪", "雷托公爵", "莱托·厄崔迪", "莱托公爵",
                    "莱托·厄崔迪公爵",
                ], ensure_ascii=False),
                json.dumps(["Leto"], ensure_ascii=False),
            ),
        )
        decisions.commit()
    finally:
        decisions.close()

    out = root / "output" / "ch01"
    out.mkdir(parents=True)
    texts = {
        "ch01_chunk000": "雷托等待着。",
        "ch01_chunk001": "雷托开口了。",
        "ch01_chunk002": "莱托转过身。",
        "ch01_chunk003": "雷托思索着，至少这一点我可以指望，莱托想。",
    }
    if third_form:
        texts["ch01_chunk001"] = "另一个人开口了。"
    for chunk_id, text in texts.items():
        (out / f"{chunk_id}_final.json").write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


def test_authority_audit_requires_book_wide_form_consensus(tmp_path: Path):
    _make_project(tmp_path)

    result = audit_decision_authorities(tmp_path)

    assert result["ok"] is True
    assert result["counts"]["ready"] == 1
    candidate = result["candidates"][0]
    assert candidate["decision_id"] == 113
    assert candidate["old_translation"] == "莱托"
    assert candidate["new_translation"] == "雷托"
    assert [row["chunk_id"] for row in candidate["repair_chunks"]] == [
        "ch01_chunk002", "ch01_chunk003"]
    assert [row["chunk_id"] for row in candidate["rerun_chunks"]] == [
        "ch01_chunk004"]
    assert "莱托公爵" in candidate["variants_to_quarantine"]
    assert "雷托公爵" not in candidate["variants_to_quarantine"]
    assert result["mode"] == "read_only"


def test_authority_apply_updates_decision_registry_and_only_wrong_chunks(
        tmp_path: Path):
    _make_project(tmp_path)
    workflow_before = (tmp_path / "db" / "workflow.db").read_bytes()
    finals_before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "output" / "ch01").glob("*_final.json")
    }

    result = apply_decision_authority_alignment(
        tmp_path, apply=True, run_id="authority-fixture")

    assert result["ok"] is True
    assert result["write_attempted"] is True
    assert Path(result["backup_path"]).exists()
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    con.row_factory = sqlite3.Row
    try:
        decision = con.execute(
            "SELECT translation, locked FROM decision_db WHERE id=113").fetchone()
        assert tuple(decision) == ("雷托", 0)
        registry = con.execute(
            "SELECT variant_names, quarantined_names FROM entity_registry "
            "WHERE id=27").fetchone()
        variants = json.loads(registry["variant_names"])
        quarantined = json.loads(registry["quarantined_names"])
        assert "雷托公爵" in variants
        assert all("莱托" not in value for value in variants)
        assert "莱托" in quarantined
        assert "莱托公爵" in quarantined
        pending = {
            row["chunk_id"] for row in con.execute(
                "SELECT chunk_id FROM pending_backtrack")
        }
        assert pending == {"ch01_chunk002", "ch01_chunk003"}
    finally:
        con.close()
    assert (tmp_path / "db" / "workflow.db").read_bytes() == workflow_before
    assert {
        path.name: path.read_bytes()
        for path in (tmp_path / "output" / "ch01").glob("*_final.json")
    } == finals_before


def test_authority_audit_blocks_third_translation(tmp_path: Path):
    _make_project(tmp_path, third_form=True)

    result = audit_decision_authorities(tmp_path)

    assert result["ok"] is False
    assert result["counts"]["ready"] == 0
    assert result["blocked_candidates"][0]["reason"] == (
        "completed_final_uses_third_form")


def test_authority_accepts_ellipsis_truncated_source_name(tmp_path: Path):
    _make_project(tmp_path)
    workflow = sqlite3.connect(str(tmp_path / "db" / "workflow.db"))
    try:
        workflow.execute(
            "UPDATE chunk_tasks SET text_content=? WHERE chunk_id=?",
            ("‘ . . . eto turned.", "ch01_chunk002"),
        )
        workflow.commit()
    finally:
        workflow.close()

    result = audit_decision_authorities(tmp_path)

    assert result["ok"] is True
    candidate = result["candidates"][0]
    row = next(item for item in candidate["repair_chunks"]
               if item["chunk_id"] == "ch01_chunk002")
    assert row["source_occurrences"] == 0
    assert row["source_fragment_occurrences"] == 1


def test_authority_apply_is_idempotent_after_alignment(tmp_path: Path):
    _make_project(tmp_path)
    first = apply_decision_authority_alignment(
        tmp_path, apply=True, run_id="authority-first")
    assert first["ok"] is True

    second = audit_decision_authorities(tmp_path)

    assert second["ok"] is True
    assert second["counts"]["ready"] == 0
