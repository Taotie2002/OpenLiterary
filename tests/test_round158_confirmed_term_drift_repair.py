"""Round 158: confirmed specific-term drift repair is group-atomic."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import src.name_only_fastpath as nof
from src.core.scheduler import TaskScheduler
from src.judgment_layer import sha256_text


def _authority_group(*, negative_chunks: int = 0, second_form: bool = False) -> dict:
    candidates = [{
        "form": "盾墙",
        "positive_chunks": 4,
        "negative_chunks": negative_chunks,
        "positive_coverage": 0.25,
        "target_substring": False,
        "target_inside_candidate": False,
        "similarity_to_target": 0.4,
        "pending_chunks": ["ch02_chunk086", "ch02_chunk089", "ch03_chunk011"],
    }]
    if second_form:
        candidates.append({
            "form": "屏障",
            "positive_chunks": 4,
            "negative_chunks": 0,
            "positive_coverage": 0.25,
            "target_substring": False,
            "target_inside_candidate": False,
            "similarity_to_target": 0.4,
            "pending_chunks": ["ch02_chunk086", "ch02_chunk089", "ch03_chunk011"],
        })
    return {
        "authority_kind": "specific",
        "decision_id": 1103,
        "source_key": "Shield Wall",
        "target": "屏蔽墙",
        "classification": "confirmed",
        "reason": "locked_target_is_book_wide_majority",
        "pending_decision_id": 315,
        "pending_reason": "specific_decision_target_missing",
        "pending_chunks": ["ch02_chunk086", "ch02_chunk089"],
        "source_positive_chunks": 4,
        "authority_chunks": 4,
        "target_hits": 1,
        "candidates": candidates,
    }


def _make_project(tmp_path: Path, monkeypatch, *, negative_chunks: int = 0,
                  second_form: bool = False) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    sched = TaskScheduler(db_path=str(db_dir / "workflow.db"))
    rows = [
        ("ch02_chunk040", "ch02", "The Shield Wall cliffs.", "COMPLETED", "屏蔽墙峭壁。"),
        ("ch02_chunk086", "ch02", "The Shield Wall cliffs.", "COMPLETED", "盾墙峭壁。"),
        ("ch02_chunk089", "ch02", "Over the Shield Wall.", "COMPLETED", "越过盾墙。"),
        ("ch03_chunk011", "ch03", "Over the Shield Wall.", "COMPLETED", "飞越盾墙。"),
        ("ch03_chunk012", "ch03", "Nothing here.", "COMPLETED", "盾墙盾墙。"),
    ]
    for cid, chapter, source, state, final_text in rows:
        sched.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?, ?, ?, ?)", (cid, chapter, source, state))
        out = tmp_path / "output" / chapter
        out.mkdir(parents=True, exist_ok=True)
        (out / ("%s_final.json" % cid)).write_text(json.dumps({
            "text": final_text,
            "metadata": {},
        }, ensure_ascii=False), encoding="utf-8")
        report = {
            "_backend_path": "jev",
            "_audit_target": {"literary_sha256": sha256_text(final_text)},
            "scores": {},
        }
        (out / ("%s_critic_report.json" % cid)).write_text(
            json.dumps(report, ensure_ascii=False), encoding="utf-8")
    sched.conn.commit()

    con = sqlite3.connect(str(db_dir / "decision_db.sqlite"))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, reason TEXT, updated_at TEXT, "
            "locked INTEGER DEFAULT 0)")
        con.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)")
        con.execute(
            "INSERT INTO decision_db VALUES (315, 2, 'shield', '护盾', '', "
            "'2026-11-18 00:00:00', 1)")
        con.execute(
            "INSERT INTO decision_db VALUES (1103, 2, 'Shield Wall', '屏蔽墙', '', "
            "'2026-11-18 00:00:00', 0)")
        con.executemany(
            "INSERT INTO pending_backtrack VALUES (?, 315, 'shield', 2, "
            "'2026-11-18 00:00:00', '2026-11-18 00:00:00')",
            [("ch02_chunk086",), ("ch02_chunk089",)])
        con.commit()
    finally:
        con.close()

    group = _authority_group(
        negative_chunks=negative_chunks, second_form=second_form)
    monkeypatch.setattr(
        "src.locked_target_authority_audit.audit_locked_target_authorities",
        lambda *args, **kwargs: {
        "schema_version": 1,
        "groups": [group],
        "counts": {"groups": 1, "confirmed": 1},
        "errors": [],
        "ok": True,
        })


def _good_critic(candidate, source_text, raw_text, proposed_text, pipeline):
    return {
        "_backend_path": "jev",
        "_typesafe_verdict": {"decision": "PASS", "reject_reasons": []},
        "scores": {
            "voice_consistency": 8.5,
            "style_compliance": 8.5,
            "literal_residue": 8.5,
            "semantic_preservation": 8.5,
            "readability": 8.5,
            "fluency": 8.5,
            "entity_consistency": 9.5,
        },
    }


def _workflow_rows(path: Path) -> list:
    con = sqlite3.connect(str(path))
    try:
        return list(con.execute(
            "SELECT chunk_id, chapter_id, text_content, state, rescue_stage "
            "FROM chunk_tasks ORDER BY chunk_id"))
    finally:
        con.close()


def test_plan_repairs_completed_source_positive_chunks_without_pending(
        tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch)

    result = nof.audit_confirmed_term_drift_repair(tmp_path)

    assert result["ok"] is True
    assert result["counts"]["candidate_groups"] == 1
    assert result["counts"]["candidate_chunks"] == 3
    candidate = result["candidates"][0]
    assert candidate["decision_id"] == 1103
    assert candidate["wrong_form"] == "盾墙"
    assert [item["chunk_id"] for item in candidate["chunks"]] == [
        "ch02_chunk086", "ch02_chunk089", "ch03_chunk011"]
    assert "ch03_chunk012" not in {
        item["chunk_id"] for item in candidate["chunks"]}


def test_plan_blocks_candidate_with_negative_chunks(tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch, negative_chunks=1)

    result = nof.audit_confirmed_term_drift_repair(tmp_path)

    assert result["counts"]["candidate_groups"] == 0
    assert result["counts"]["blocked_groups"] == 1
    assert result["blocked"][0]["reason"].startswith(
        "no_unique_counted_candidate:")


def test_plan_blocks_ambiguous_second_wrong_form(tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch, second_form=True)
    final_path = tmp_path / "output" / "ch02" / "ch02_chunk086_final.json"
    final_path.write_text(json.dumps({
        "text": "盾墙与屏障并立。", "metadata": {},
    }, ensure_ascii=False), encoding="utf-8")

    result = nof.audit_confirmed_term_drift_repair(tmp_path)

    assert result["counts"]["candidate_groups"] == 0
    assert result["blocked"][0]["reason"].startswith(
        "no_unique_counted_candidate:ch02_chunk086")


def test_plan_accepts_linewrapped_source_only_with_full_cjk_gloss(
        tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch)
    con = sqlite3.connect(str(tmp_path / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET text_content=? WHERE chunk_id='ch03_chunk012'",
            ("Over the Shield\nWall.",))
        con.commit()
    finally:
        con.close()
    final_path = tmp_path / "output" / "ch03" / "ch03_chunk012_final.json"
    final_path.write_text(json.dumps({
        "text": (
            "公爵低头向左瞥了一眼护盾墙（Shield Wall）的裂谷。\n\n"
            "[7] 护盾墙（Shield Wall）：地貌名。"),
        "metadata": {},
    }, ensure_ascii=False), encoding="utf-8")

    result = nof.audit_confirmed_term_drift_repair(tmp_path)

    assert result["ok"] is True
    candidate = result["candidates"][0]
    assert set(candidate["wrong_forms"]) == {"盾墙", "护盾墙"}
    gloss_chunk = next(
        item for item in candidate["chunks"] if item["chunk_id"] == "ch03_chunk012")
    assert gloss_chunk["wrong_form"] == "护盾墙"
    assert gloss_chunk["gloss_bound"] is True


def test_plan_rejects_linewrapped_source_without_gloss(
        tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch)
    con = sqlite3.connect(str(tmp_path / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET text_content=? WHERE chunk_id='ch03_chunk012'",
            ("Over the Shield\nWall.",))
        con.commit()
    finally:
        con.close()
    final_path = tmp_path / "output" / "ch03" / "ch03_chunk012_final.json"
    final_path.write_text(json.dumps({
        "text": "护盾墙的裂谷。\n\n[7] 护盾墙：地貌名。",
        "metadata": {},
    }, ensure_ascii=False), encoding="utf-8")

    result = nof.audit_confirmed_term_drift_repair(tmp_path)

    assert result["counts"]["candidate_groups"] == 0
    assert result["blocked"][0]["reason"].endswith("ch03_chunk012")


def test_plan_rejects_single_gloss_with_welded_chinese_context(
        tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch)
    con = sqlite3.connect(str(tmp_path / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET text_content=? WHERE chunk_id='ch03_chunk012'",
            ("Over the Shield\nWall.",))
        con.commit()
    finally:
        con.close()
    final_path = tmp_path / "output" / "ch03" / "ch03_chunk012_final.json"
    final_path.write_text(json.dumps({
        "text": "公爵低头向左瞥了一眼护盾墙（Shield Wall）的裂谷。",
        "metadata": {},
    }, ensure_ascii=False), encoding="utf-8")

    result = nof.audit_confirmed_term_drift_repair(tmp_path)

    assert result["counts"]["candidate_groups"] == 0
    assert result["blocked"][0]["reason"].endswith("ch03_chunk012")


def test_apply_is_group_atomic_and_consumes_pending(tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch)
    workflow_before = _workflow_rows(tmp_path / "db" / "workflow.db")

    result = nof.apply_confirmed_term_drift_repair(
        tmp_path, apply=True, critic_fn=_good_critic,
        run_id="confirmed-shield-wall", require_authority_audit=False,
    )

    assert result["ok"] is True
    assert result["counts"] == {"repaired": 1}
    assert _workflow_rows(tmp_path / "db" / "workflow.db") == workflow_before
    for cid, chapter in [
        ("ch02_chunk086", "ch02"), ("ch02_chunk089", "ch02"),
        ("ch03_chunk011", "ch03"),
    ]:
        final = json.loads((tmp_path / "output" / chapter / (
            "%s_final.json" % cid)).read_text(encoding="utf-8"))
        assert "屏蔽墙" in final["text"]
        assert "盾墙" not in final["text"]
        report = json.loads((tmp_path / "output" / chapter / (
            "%s_critic_report.json" % cid)).read_text(encoding="utf-8"))
        assert report["_reaudit"]["tool"] == "confirmed-term-drift-repair"
        assert report["_final_binding"]["final_sha256"] == sha256_text(
            final["text"])
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 0
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1103").fetchone()[0] == 1
    finally:
        con.close()
    assert (tmp_path / "output" / "confirmed_term_drift_repair" /
            "confirmed-shield-wall" / "1103" /
            "ch03_chunk011_confirmed_term_drift_evidence.json").exists()


def test_critic_rejection_writes_nothing(tmp_path: Path, monkeypatch):
    _make_project(tmp_path, monkeypatch)
    paths = [
        tmp_path / "output" / chapter / ("%s_final.json" % cid)
        for cid, chapter in [
            ("ch02_chunk086", "ch02"), ("ch02_chunk089", "ch02"),
            ("ch03_chunk011", "ch03"),
        ]
    ]
    before = {path: path.read_bytes() for path in paths}
    decision_before = (tmp_path / "db" / "decision_db.sqlite").read_bytes()

    def bad(candidate, source_text, raw_text, proposed_text, pipeline):
        report = _good_critic(
            candidate, source_text, raw_text, proposed_text, pipeline)
        report["scores"] = dict(report["scores"], semantic_preservation=4.0)
        return report

    result = nof.apply_confirmed_term_drift_repair(
        tmp_path, apply=True, critic_fn=bad, run_id="confirmed-reject",
        require_authority_audit=False)

    assert result["ok"] is True
    assert result["counts"] == {"kept": 1}
    assert {path: path.read_bytes() for path in paths} == before
    assert (tmp_path / "db" / "decision_db.sqlite").read_bytes() == decision_before


def test_command_requires_instance_lock():
    from src.main import _command_needs_lock
    from argparse import Namespace

    assert _command_needs_lock(
        "confirmed-term-drift-repair", Namespace(apply=False)) is True
