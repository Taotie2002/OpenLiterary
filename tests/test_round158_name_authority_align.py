"""Round 158 Phase 1.5a: deterministic name authority alignment."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import src.name_authority_align as align
from src.core.decision_engine import (
    apply_name_unification,
    build_name_unification_maps,
    unify_entity_registry,
)
from src.main import _MUTATING_COMMANDS


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_db(root: Path, decisions, registry) -> Path:
    path = root / "db" / "decision_db.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT UNIQUE, translation TEXT, locked INTEGER DEFAULT 0, "
            "reason TEXT)"
        )
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, canonical_name TEXT, "
            "source_names TEXT, context_names TEXT, quarantined_names TEXT, "
            "variant_names TEXT, entity_type TEXT, confidence TEXT, "
            "pending_review INTEGER DEFAULT 0, review_reason TEXT, updated_at TEXT)"
        )
        con.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)"
        )
        con.executemany(
            "INSERT INTO decision_db (id, level, source_key, translation) "
            "VALUES (?, 2, ?, ?)", decisions)
        for item in registry:
            rid, canonical, sources = item[:3]
            pending = int(item[3]) if len(item) > 3 else 0
            confidence = str(item[4]) if len(item) > 4 else "high"
            con.execute(
                "INSERT INTO entity_registry "
                "(id, canonical_name, source_names, context_names, quarantined_names, "
                "variant_names, entity_type, confidence, pending_review) "
                "VALUES (?, ?, ?, '[]', '[]', '[]', 'person', ?, ?)",
                (rid, canonical, json.dumps(sources, ensure_ascii=False),
                 confidence, pending),
            )
        con.commit()
    finally:
        con.close()
    return path


def _clean_db(root: Path) -> Path:
    return _make_db(
        root,
        [(1, "Arrakis", "厄拉科斯")],
        [(1, "厄拉科斯", ["Arrakis"])],
    )


def _cross_person_db(root: Path) -> Path:
    return _make_db(
        root,
        [(1, "M’Lord Harkonnen", "哈克南大人")],
        [
            (1, "弗拉基米尔·哈克南男爵", ["M’Lord Harkonnen"]),
            (2, "哈克南男爵", ["M’Lord Harkonnen"]),
        ],
    )


def _form_drift_db(root: Path) -> Path:
    return _make_db(
        root,
        [
            (1, "Bene Gesserit", "贝尼·杰瑟里特"),
            (2, "B.G.", "贝尼·杰赛里特（B.G.）"),
        ],
        [(9, "贝尼·杰赛里特", ["Bene Gesserit", "B.G."])],
    )


def _target_owned_db(root: Path) -> Path:
    return _make_db(
        root,
        [(1, "Gaius Helen Mohiam", "盖乌斯·海伦·莫希阿姆")],
        [
            (1, "盖乌斯·海伦·莫希阿姆", ["Gaius Helen Mohiam"]),
            (2, "盖乌斯·海伦·莫希亚姆", ["Gaius Helen Mohiam"], 1),
        ],
    )


def _term_phrase_db(root: Path) -> Path:
    return _make_db(
        root,
        [
            (1, "Bene Gesserit", "贝尼·杰瑟里特"),
            (2, "Bene Gesserit Plan", "贝尼·杰瑟里特计划"),
            (3, "Bene Gesserit school", "贝尼·杰瑟里特学校"),
            (4, "Jessica", "杰西卡"),
            (5, "Lady Jessica", "杰西卡夫人"),
        ],
        [
            (9, "贝尼·杰瑟里特",
             ["Bene Gesserit", "the Bene Gesserit Plan", "Bene Gesserit school"]),
            (10, "杰西卡", ["Jessica", "Lady Jessica"]),
            (62, "贝尼·杰赛里特学校", [], 1),
            (97, "贝尼·杰瑟里特学校", [], 1),
        ],
    )


def _registry_row(db: Path, rid: int) -> dict:
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    try:
        row = dict(con.execute(
            "SELECT id, canonical_name, source_names, variant_names, "
            "quarantined_names, pending_review, review_reason "
            "FROM entity_registry WHERE id=?", (rid,)).fetchone())
        for field in ("source_names", "variant_names", "quarantined_names"):
            row[field] = json.loads(row[field] or "[]")
        return row
    finally:
        con.close()


def _locked_short_variant_db(root: Path, *, competing: bool = False) -> Path:
    registry = [(31, "杜菲·哈瓦特", ["Hawat"], 0, "high")]
    if competing:
        registry.append((40, "哈瓦特军团", [], 0, "high"))
    db = _make_db(root, [(153, "Hawat", "哈瓦特")], registry)
    con = sqlite3.connect(str(db))
    try:
        con.execute("UPDATE decision_db SET locked=1 WHERE id=153")
        con.commit()
    finally:
        con.close()
    return db


def _duplicate_form_db(
    root: Path,
    *,
    canonical: str = "法弗雷卢什",
    pending_review: int = 0,
    locked_decision: int = 0,
) -> Path:
    db = _make_db(
        root,
        [
            (1, "Faufreluches", "福弗雷卢什"),
            (2, "faufreluches", "法弗雷卢什"),
        ],
        [(9, canonical, ["Faufreluches"], pending_review)],
    )
    if locked_decision:
        con = sqlite3.connect(str(db))
        try:
            con.execute(
                "UPDATE decision_db SET locked=1 WHERE id=?",
                (locked_decision,))
            con.commit()
        finally:
            con.close()
    return db


def _resolved_pending_db(root: Path, *, canonical: str = "福弗雷卢什") -> Path:
    db = _make_db(
        root,
        [(1, "Faufreluches", canonical)],
        [(9, canonical, ["Faufreluches"], 1)],
    )
    con = sqlite3.connect(str(db))
    try:
        con.execute(
            "UPDATE entity_registry SET review_reason=? WHERE id=9",
            (f"称谓差异大: 新实体 法弗雷卢什 与现有 {canonical} 可能指向同一对象 | "
             f"Round158 duplicate source form consensus: 法弗雷卢什 -> {canonical}",))
        con.commit()
    finally:
        con.close()
    return db


def _write_finals(root: Path, texts) -> None:
    for chunk_id, text in texts.items():
        chapter = chunk_id.split("_", 1)[0]
        path = root / "output" / chapter / f"{chunk_id}_final.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"text": text}, ensure_ascii=False),
            encoding="utf-8",
        )


def _decision_translation(db: Path, decision_id: int) -> str:
    con = sqlite3.connect(str(db))
    try:
        row = con.execute(
            "SELECT translation FROM decision_db WHERE id=?",
            (decision_id,)).fetchone()
        return str(row[0]) if row else ""
    finally:
        con.close()


def _pending_rows(db: Path) -> list[tuple]:
    con = sqlite3.connect(str(db))
    try:
        return list(con.execute(
            "SELECT chunk_id, decision_id, source_key, queue_rounds "
            "FROM pending_backtrack ORDER BY chunk_id"))
    finally:
        con.close()


def test_clean_fixture_plans_no_changes(tmp_path: Path):
    db = _clean_db(tmp_path)
    before = _digest(db)

    result = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert _digest(db) == before
    assert result["mode"] == "read_only"
    assert result["write_attempted"] is False
    assert result["ok"] is True
    assert result["actions"] == []
    assert result["manual_review"] == []
    assert result["counts"] == {
        "quarantine_aliases": 0, "align_canonical": 0,
        "align_decision_forms": 0, "clear_resolved_pending_review": 0,
        "register_locked_variant": 0,
        "manual_review": 0,
    }
    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()


def test_cli_command_is_registered_as_mutating():
    assert "name-authority-align" in _MUTATING_COMMANDS


def test_cross_person_alias_is_quarantined_from_all_owners(tmp_path: Path):
    db = _cross_person_db(tmp_path)
    before = _digest(db)

    result = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                          run_id="cross-person")

    assert result["ok"] is True, result.get("errors")
    assert result["audit_before"]["ok"] is False
    assert result["audit_after"]["ok"] is True
    assert result["applied_count"] == 2
    assert _digest(db) != before
    for rid in (1, 2):
        row = _registry_row(db, rid)
        assert row["source_names"] == []
        assert row["quarantined_names"] == ["M’Lord Harkonnen"]
        assert row["pending_review"] == 0
        assert "Round158 authority align" in row["review_reason"]
    assert (Path(result["backup_path"])).exists()


def test_form_drift_aligns_canonical_and_predicts_secondary_conflict(tmp_path: Path):
    db = _form_drift_db(tmp_path)

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["ok"] is True
    assert plan["manual_review"] == []
    by_action = {item["action"]: item for item in plan["actions"]}
    assert by_action["align_canonical"]["canonical_before"] == "贝尼·杰赛里特"
    assert by_action["align_canonical"]["canonical_after"] == "贝尼·杰瑟里特"
    assert by_action["quarantine_aliases"]["aliases"] == ["B.G."]
    assert "post_form_align_conflict" in by_action["quarantine_aliases"]["kinds"]

    result = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                          run_id="form-drift")

    assert result["ok"] is True, result.get("errors")
    assert result["audit_after"]["ok"] is True
    row = _registry_row(db, 9)
    assert row["canonical_name"] == "贝尼·杰瑟里特"
    assert row["source_names"] == ["Bene Gesserit"]
    assert row["quarantined_names"] == ["B.G."]
    assert row["pending_review"] == 0
    con = sqlite3.connect(str(db))
    try:
        assert con.execute(
            "SELECT translation FROM decision_db WHERE id=2").fetchone()[0] == (
                "贝尼·杰赛里特（B.G.）")
    finally:
        con.close()


def test_target_owned_form_drift_quarantines_only_drifting_owner(tmp_path: Path):
    db = _target_owned_db(tmp_path)

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["manual_review"] == []
    assert plan["counts"]["align_canonical"] == 0
    assert plan["counts"]["quarantine_aliases"] == 1
    action = plan["actions"][0]
    assert action["registry_id"] == 2
    assert action["aliases"] == ["Gaius Helen Mohiam"]
    assert action["reasons"] == ["form_target_owned:盖乌斯·海伦·莫希阿姆"]

    result = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                          run_id="target-owned")

    assert result["ok"] is True, result.get("errors")
    correct = _registry_row(db, 1)
    drift = _registry_row(db, 2)
    assert correct["canonical_name"] == "盖乌斯·海伦·莫希阿姆"
    assert correct["source_names"] == ["Gaius Helen Mohiam"]
    assert drift["canonical_name"] == "盖乌斯·海伦·莫希亚姆"
    assert drift["source_names"] == []
    assert drift["quarantined_names"] == ["Gaius Helen Mohiam"]


def test_term_phrase_aliases_are_quarantined_without_touching_decisions(
        tmp_path: Path):
    db = _term_phrase_db(tmp_path)
    con = sqlite3.connect(str(db))
    try:
        con.execute(
            "UPDATE entity_registry SET variant_names=? WHERE id=9",
            (json.dumps(
                ["贝尼·杰赛里特学校", "贝尼·杰瑟里特学校"],
                ensure_ascii=False),),
        )
        con.commit()
    finally:
        con.close()

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["ok"] is True
    assert plan["manual_review"] == []
    action = next(item for item in plan["actions"]
                  if item["registry_id"] == 9)
    assert set(action["aliases"]) == {
        "Bene Gesserit school", "the Bene Gesserit Plan",
        "贝尼·杰赛里特学校", "贝尼·杰瑟里特学校"}
    assert "term_phrase_alias_quarantine" in action["kinds"]
    assert "term_phrase_variant_quarantine" in action["kinds"]

    result = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                          run_id="term-phrase")

    assert result["ok"] is True, result.get("errors")
    assert result["audit_after"]["ok"] is True
    row = _registry_row(db, 9)
    assert row["source_names"] == ["Bene Gesserit"]
    assert row["variant_names"] == []
    assert set(row["quarantined_names"]) == {
        "Bene Gesserit school", "the Bene Gesserit Plan",
        "贝尼·杰赛里特学校", "贝尼·杰瑟里特学校"}
    con = sqlite3.connect(str(db))
    try:
        assert con.execute(
            "SELECT translation FROM decision_db WHERE id=2").fetchone()[0] == (
                "贝尼·杰瑟里特计划")
        unify_entity_registry(con, dry_run=False)
    finally:
        con.close()
    assert _registry_row(db, 9)["variant_names"] == []
    con = sqlite3.connect(str(db))
    try:
        maps = build_name_unification_maps(con, "贝尼·杰瑟里特学校。")
    finally:
        con.close()
    after, _stats = apply_name_unification("贝尼·杰瑟里特学校。", maps)
    assert "学校" in after
    assert after != "贝尼·杰瑟里特。"


def test_pending_ambiguous_and_multi_owner_cases_require_review(tmp_path: Path):
    pending_root = tmp_path / "pending"
    pending_db = _make_db(
        pending_root,
        [(1, "Bene Gesserit", "贝尼·杰瑟里特")],
        [(9, "贝尼·杰赛里特", ["Bene Gesserit"], 1)],
    )
    pending = align.build_name_authority_alignment(
        pending_root, decision_db=pending_db)
    assert pending["actions"] == []
    assert pending["manual_review"][0]["reason"] == (
        "pending_registry_row_requires_review")
    refused = align.align_name_authorities(
        pending_root, decision_db=pending_db, apply=True, run_id="pending")
    assert refused["ok"] is False
    assert refused["write_attempted"] is False
    assert refused["applied_count"] == 0

    ambiguous_root = tmp_path / "ambiguous"
    ambiguous_db = _make_db(
        ambiguous_root,
        [
            (1, "Bene Gesserit", "贝尼·杰瑟里特"),
            (2, "Bene  Gesserit", "贝尼·杰赛里德"),
        ],
        [(9, "贝尼·杰赛里特", ["Bene Gesserit"])],
    )
    ambiguous = align.build_name_authority_alignment(
        ambiguous_root, decision_db=ambiguous_db)
    assert ambiguous["manual_review"][0]["reason"] == "ambiguous_decision_forms"

    multi_root = tmp_path / "multi"
    multi_db = _make_db(
        multi_root,
        [(1, "M’Lord Harkonnen", "哈克南大人")],
        [
            (1, "哈克南太人", ["M’Lord Harkonnen"]),
            (2, "哈克南大甲", ["M’Lord Harkonnen"]),
        ],
    )
    multi = align.build_name_authority_alignment(
        multi_root, decision_db=multi_db)
    assert multi["manual_review"][0]["reason"] == "form_alias_multiple_owners"


def test_apply_is_idempotent_after_clean_audit(tmp_path: Path):
    db = _cross_person_db(tmp_path)
    first = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                         run_id="first")
    assert first["ok"] is True
    after_first = _digest(db)

    second = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                          run_id="second")

    assert second["ok"] is True
    assert second["write_attempted"] is False
    assert second["applied_count"] == 0
    assert second.get("backup_path") is None
    assert _digest(db) == after_first


def test_locked_decision_short_form_is_registered_as_variant(tmp_path: Path):
    db = _locked_short_variant_db(tmp_path)

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["ok"] is True
    assert plan["counts"]["register_locked_variant"] == 1
    assert plan["actions"] == [{
        "action": "register_locked_variant",
        "registry_id": 31,
        "canonical_before": "杜菲·哈瓦特",
        "variant": "哈瓦特",
        "source_key": "Hawat",
        "decision_ids": [153],
        "reason": "locked_decision_short_form",
    }]

    result = align.align_name_authorities(
        tmp_path, decision_db=db, apply=True, run_id="locked-variant")

    assert result["ok"] is True
    assert result["applied_count"] == 1
    assert result["actions"][0]["status"] == "applied"
    assert _registry_row(db, 31)["variant_names"] == ["哈瓦特"]


def test_locked_short_variant_rejects_cross_person_collision(tmp_path: Path):
    db = _locked_short_variant_db(tmp_path, competing=True)

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["actions"] == []
    assert plan["ok"] is False
    assert plan["manual_review"][0]["reason"] == (
        "locked_variant_cross_person_conflict")


def test_stale_plan_fails_closed_without_writing(tmp_path: Path, monkeypatch):
    db = _clean_db(tmp_path)
    before = _digest(db)
    fake_plan = {
        "counts": {"quarantine_aliases": 1, "align_canonical": 0,
                   "manual_review": 0},
        "audit_before": {"ok": False, "status": "needs_alignment",
                         "blocking_count": 1},
        "manual_review": [],
        "errors": [],
        "actions": [{
            "action": "quarantine_aliases",
            "registry_id": 1,
            "canonical_before": "stale canonical",
            "aliases": ["Arrakis"],
            "kinds": ["test"],
            "decision_ids": [1],
            "reasons": ["test"],
        }],
    }
    monkeypatch.setattr(
        align, "build_name_authority_alignment", lambda *a, **k: fake_plan)

    result = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                          run_id="stale")

    assert result["ok"] is False
    assert result["applied_count"] == 0
    assert result["actions"][0]["status"] == "stale_canonical_changed"
    assert _digest(db) == before


def test_write_failure_rolls_back_every_applied_action(tmp_path: Path):
    db = _cross_person_db(tmp_path)
    con = sqlite3.connect(str(db))
    try:
        con.execute(
            "CREATE TRIGGER fail_second BEFORE UPDATE ON entity_registry "
            "WHEN OLD.id=2 BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
        con.commit()
    finally:
        con.close()
    before = _digest(db)

    result = align.align_name_authorities(tmp_path, decision_db=db, apply=True,
                                          run_id="rollback")

    assert result["ok"] is False
    assert result["applied_count"] == 0
    assert any(error.startswith("apply_failed:") for error in result["errors"])
    assert result["actions"][0]["status"] == "rolled_back"
    assert _digest(db) == before
    first = _registry_row(db, 1)
    assert first["source_names"] == ["M’Lord Harkonnen"]
    assert first["quarantined_names"] == []


def test_duplicate_form_corpus_majority_aligns_decision_and_registry(tmp_path: Path):
    db = _duplicate_form_db(tmp_path)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "福弗雷卢什",
        "ch05_chunk001": "法弗雷卢什",
    })

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["ok"] is True
    assert plan["counts"]["align_decision_forms"] == 1
    action = plan["actions"][0]
    assert action["action"] == "align_decision_forms"
    assert action["decision_ids"] == [2]
    assert action["new_translation"] == "福弗雷卢什"
    assert action["queued_chunks"] == ["ch05_chunk001"]

    result = align.align_name_authorities(
        tmp_path, decision_db=db, apply=True, run_id="duplicate-form")

    assert result["ok"] is True
    assert result["applied_count"] == 1
    assert _decision_translation(db, 1) == "福弗雷卢什"
    assert _decision_translation(db, 2) == "福弗雷卢什"
    assert _registry_row(db, 9)["canonical_name"] == "福弗雷卢什"
    assert _pending_rows(db) == [("ch05_chunk001", 2, "faufreluches", 1)]


def test_duplicate_form_allows_pending_row_when_canonical_is_winner(tmp_path: Path):
    db = _duplicate_form_db(
        tmp_path, canonical="福弗雷卢什", pending_review=1)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "福弗雷卢什",
        "ch05_chunk001": "法弗雷卢什",
    })

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["ok"] is True
    assert plan["counts"]["align_decision_forms"] == 1


def test_duplicate_form_rejects_locked_minority(tmp_path: Path):
    db = _duplicate_form_db(tmp_path, locked_decision=2)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "福弗雷卢什",
        "ch05_chunk001": "法弗雷卢什",
    })
    before = _digest(db)

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["actions"] == []
    assert plan["manual_review"][0]["blocked_reason"] == "minority_decision_locked"
    assert _digest(db) == before


def test_duplicate_form_rejects_narrow_corpus_majority(tmp_path: Path):
    db = _duplicate_form_db(tmp_path)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "法弗雷卢什",
        "ch05_chunk001": "法弗雷卢什",
    })

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["actions"] == []
    assert plan["manual_review"][0]["blocked_reason"] == (
        "corpus_majority_too_narrow")


def test_duplicate_form_rejects_registry_canonical_third_form(tmp_path: Path):
    db = _duplicate_form_db(tmp_path, canonical="福弗雷卢斯")
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "福弗雷卢什",
        "ch05_chunk001": "法弗雷卢什",
    })

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["counts"]["align_decision_forms"] == 0
    assert plan["manual_review"][0]["blocked_reason"] == (
        "registry_canonical_is_third_form")


def test_duplicate_form_rejects_winner_owned_by_another_registry_row(tmp_path: Path):
    db = _make_db(
        tmp_path,
        [
            (1, "Faufreluches", "福弗雷卢什"),
            (2, "faufreluches", "法弗雷卢什"),
        ],
        [
            (9, "法弗雷卢什", ["Faufreluches"]),
            (10, "福弗雷卢什", ["Another source"]),
        ],
    )
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "福弗雷卢什",
        "ch05_chunk001": "法弗雷卢什",
    })

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["actions"] == []
    assert plan["manual_review"][0]["blocked_reason"] == (
        "winner_registry_owned:10")


def test_duplicate_form_with_no_stale_chunks_still_cleans_decision(tmp_path: Path):
    db = _duplicate_form_db(tmp_path)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "福弗雷卢什",
    })

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["counts"]["align_decision_forms"] == 1
    assert plan["actions"][0]["queued_chunks"] == []

    result = align.align_name_authorities(
        tmp_path, decision_db=db, apply=True, run_id="no-stale")

    assert result["ok"] is True
    assert _decision_translation(db, 2) == "福弗雷卢什"
    assert _pending_rows(db) == []


def test_duplicate_form_read_only_plan_writes_no_db(tmp_path: Path):
    db = _duplicate_form_db(tmp_path)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
        "ch04_chunk001": "福弗雷卢什",
        "ch05_chunk001": "法弗雷卢什",
    })
    before = _digest(db)

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["counts"]["align_decision_forms"] == 1
    assert _digest(db) == before


def test_resolved_pending_review_clears_only_with_consensus_marker(tmp_path: Path):
    db = _resolved_pending_db(tmp_path)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
        "ch03_chunk001": "福弗雷卢什",
    })

    plan = align.build_name_authority_alignment(tmp_path, decision_db=db)

    assert plan["counts"]["clear_resolved_pending_review"] == 1
    action = next(a for a in plan["actions"]
                  if a["action"] == "clear_resolved_pending_review")
    assert action["registry_id"] == 9
    assert action["corpus_chunks"] == 3

    result = align.align_name_authorities(
        tmp_path, decision_db=db, apply=True, run_id="clear-review")

    assert result["ok"] is True
    assert result["applied_count"] == 1
    row = _registry_row(db, 9)
    assert row["pending_review"] == 0
    assert "pending review resolved" in str(row["review_reason"])


def test_resolved_pending_review_requires_corpus_and_exact_marker(tmp_path: Path):
    db = _resolved_pending_db(tmp_path)
    _write_finals(tmp_path, {
        "ch01_chunk001": "福弗雷卢什",
        "ch02_chunk001": "福弗雷卢什",
    })

    thin = align.build_name_authority_alignment(tmp_path, decision_db=db)
    assert thin["counts"]["clear_resolved_pending_review"] == 0

    con = sqlite3.connect(str(db))
    try:
        con.execute(
            "UPDATE entity_registry SET review_reason=? WHERE id=9",
            ("称谓差异大: 新实体 法弗雷卢什 与现有 福弗雷卢什 可能指向同一对象",))
        con.commit()
    finally:
        con.close()
    _write_finals(tmp_path, {
        "ch03_chunk001": "福弗雷卢什",
    })

    unmarked = align.build_name_authority_alignment(tmp_path, decision_db=db)
    assert unmarked["counts"]["clear_resolved_pending_review"] == 0
