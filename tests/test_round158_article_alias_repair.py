"""Round 158: article-led Decision rows cannot shadow exact-key authority."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.decision_authority_align import (
    apply_article_shadow_alias_repairs,
    audit_article_shadow_decisions,
)


def _make_project(root: Path, *, wrong_final: bool = False,
                  alias_source: str = "the Lady Jessica") -> None:
    (root / "db").mkdir(parents=True)
    workflow = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        workflow.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, "
            "chapter_id TEXT, text_content TEXT, state TEXT)"
        )
        workflow.executemany(
            "INSERT INTO chunk_tasks VALUES (?, ?, ?, ?)",
            [
                ("ch01_chunk007", "ch01",
                 f"{alias_source} watched Paul.", "COMPLETED"),
                ("ch02_chunk026", "ch02",
                 f"{alias_source} waited.", "PENDING"),
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
            "CREATE TABLE decision_impact (decision_id INTEGER, "
            "chunk_id TEXT, PRIMARY KEY (decision_id, chunk_id))"
        )
        decisions.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)"
        )
        decisions.executemany(
            "INSERT INTO decision_db VALUES (?, 2, ?, ?, 'fixture', "
            "'2026-11-18 00:00:00', 0)",
            [
                (54, "Lady Jessica", "杰西卡夫人"),
                (3488, alias_source, "杰西卡女士"),
            ],
        )
        decisions.executemany(
            "INSERT INTO decision_impact VALUES (?, ?)",
            [(54, "ch01_chunk007"), (3488, "ch01_chunk007"),
             (3488, "ch02_chunk026")],
        )
        decisions.executemany(
            "INSERT INTO pending_backtrack VALUES (?, 3488, ?, 1, '', '')",
            [
                ("ch01_chunk007", alias_source),
                ("ch02_chunk026", alias_source),
            ],
        )
        decisions.commit()
    finally:
        decisions.close()

    out = root / "output" / "ch01"
    out.mkdir(parents=True)
    text = "杰西卡女士望着保罗。" if wrong_final else "杰西卡夫人望着保罗。"
    (out / "ch01_chunk007_final.json").write_text(
        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


def test_article_shadow_audit_requires_completed_final_to_match_canonical(
        tmp_path: Path):
    _make_project(tmp_path)

    result = audit_article_shadow_decisions(tmp_path)

    assert result["ok"] is True
    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["decision_id"] == 3488
    assert candidate["canonical_decision_id"] == 54
    assert candidate["pending_chunks"] == ["ch01_chunk007", "ch02_chunk026"]
    assert candidate["inert_pending_chunks"] == ["ch02_chunk026"]


def test_article_shadow_apply_deletes_only_alias_and_keeps_canonical(
        tmp_path: Path):
    _make_project(tmp_path)

    result = apply_article_shadow_alias_repairs(
        tmp_path, apply=True, run_id="article-fixture")

    assert result["ok"] is True
    assert Path(result["backup_path"]).exists()
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        rows = con.execute(
            "SELECT id, source_key, translation FROM decision_db ORDER BY id"
        ).fetchall()
        assert rows == [(54, "Lady Jessica", "杰西卡夫人")]
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 0
        impacts = con.execute(
            "SELECT decision_id, chunk_id FROM decision_impact "
            "ORDER BY decision_id, chunk_id").fetchall()
        assert impacts == [(54, "ch01_chunk007"), (54, "ch02_chunk026")]
    finally:
        con.close()
    again = audit_article_shadow_decisions(tmp_path)
    assert again["counts"]["candidates"] == 0
    assert again["counts"]["blocked"] == 0


def test_article_shadow_blocks_if_alias_target_remains_in_prose(
        tmp_path: Path):
    _make_project(tmp_path, wrong_final=True)

    result = audit_article_shadow_decisions(tmp_path)

    assert result["ok"] is False
    assert result["counts"]["blocked"] == 1
    assert result["blocked_candidates"][0]["reason"] == (
        "alias_target_still_present:ch01_chunk007")


def test_article_shadow_audit_reports_all_blocked_source_chunks(
        tmp_path: Path):
    _make_project(tmp_path, wrong_final=True)
    workflow = sqlite3.connect(str(tmp_path / "db" / "workflow.db"))
    try:
        workflow.execute(
            "INSERT INTO chunk_tasks VALUES (?, ?, ?, ?)",
            ("ch01_chunk008", "ch01",
             "the Lady Jessica waited.", "COMPLETED"),
        )
        workflow.commit()
    finally:
        workflow.close()
    decisions = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        decisions.execute(
            "INSERT INTO decision_impact VALUES (?, ?)",
            (3488, "ch01_chunk008"),
        )
        decisions.execute(
            "INSERT INTO pending_backtrack VALUES (?, 3488, ?, 1, '', '')",
            ("ch01_chunk008", "the Lady Jessica"),
        )
        decisions.commit()
    finally:
        decisions.close()
    (tmp_path / "output" / "ch01" / "ch01_chunk008_final.json").write_text(
        json.dumps({"text": "杰西卡女士等待着。"}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = audit_article_shadow_decisions(tmp_path)

    assert result["ok"] is False
    blocked = result["blocked_candidates"][0]
    assert blocked["reason"] == "alias_target_still_present:ch01_chunk007"
    assert blocked["blocked_chunks"] == [
        "ch01_chunk007", "ch01_chunk008"]


def test_article_shadow_match_keeps_case_sensitive_remainder(
        tmp_path: Path):
    _make_project(tmp_path, alias_source="the one")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "UPDATE decision_db SET source_key='One', translation='唯一者' "
            "WHERE id=54")
        con.commit()
    finally:
        con.close()

    result = audit_article_shadow_decisions(tmp_path)

    assert result["counts"]["candidates"] == 0
    assert result["counts"]["blocked"] == 0
