"""Round 158 Phase 1.5a: name authority audit is evidence-only."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from src.name_authority_audit import audit_name_authorities


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_db(root: Path) -> Path:
    path = root / "db" / "decision_db.sqlite"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT UNIQUE, translation TEXT, locked INTEGER DEFAULT 0)"
        )
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, canonical_name TEXT, "
            "source_names TEXT, context_names TEXT, quarantined_names TEXT, "
            "variant_names TEXT, entity_type TEXT, confidence TEXT, "
            "pending_review INTEGER DEFAULT 0)"
        )
        decisions = [
            (1, 2, "Bene Gesserit", "贝尼·杰瑟里特"),
            (2, 2, "Leto", "莱托"),
            (3, 2, "Duke Leto", "雷托公爵"),
            (4, 2, "Duke Leto Atreides", "雷托·厄崔迪公爵"),
            (5, 2, "Noble Born", "贵族出身"),
            (6, 2, "Dukeling", "小公爵"),
            (7, 2, "M’Lord Harkonnen", "哈克南大人"),
            (8, 2, "Baron Harkonnen", "哈克南男爵"),
        ]
        con.executemany(
            "INSERT INTO decision_db (id, level, source_key, translation) "
            "VALUES (?, ?, ?, ?)", decisions)
        registry = [
            (9, "贝尼·杰赛里特", ["Bene Gesserit", "B.G."], []),
            (27, "雷托·厄崔迪公爵",
             ["Leto", "Duke Leto", "Duke Leto Atreides", "Noble Born"], []),
            (1, "保罗·厄崔迪", ["Dukeling", "Paul Atreides"], []),
            (39, "弗拉基米尔·哈克南男爵",
             ["Baron Harkonnen", "M’Lord Harkonnen"], []),
            (44, "哈克南男爵", ["Baron Harkonnen", "M’Lord Harkonnen"], []),
        ]
        for rid, canonical, sources, contexts in registry:
            con.execute(
                "INSERT INTO entity_registry "
                "(id, canonical_name, source_names, context_names, "
                "quarantined_names, variant_names, entity_type, confidence) "
                "VALUES (?, ?, ?, ?, '[]', '[]', 'person', 'high')",
                (rid, canonical, json.dumps(sources, ensure_ascii=False),
                 json.dumps(contexts, ensure_ascii=False)),
            )
        con.commit()
    finally:
        con.close()
    return path


def test_authority_audit_classifies_alignment_groups_and_quarantine(tmp_path: Path):
    db = _make_db(tmp_path)
    before = _digest(db)

    result = audit_name_authorities(tmp_path, decision_db=db)

    assert _digest(db) == before
    assert result["mode"] == "read_only"
    assert result["write_attempted"] is False
    assert result["ok"] is False
    assert result["counts"]["form_align_candidate"] >= 1
    assert result["counts"]["cross_person_alias_conflict"] >= 1
    assert result["counts"]["authority_mismatch_quarantine"] >= 1

    by_alias = {}
    for item in result["records"]:
        by_alias.setdefault(item["alias"], []).append(item)
    assert "form_align_candidate" in {
        item["kind"] for item in by_alias["Bene Gesserit"]}
    assert "cross_person_alias_conflict" in {
        item["kind"] for item in by_alias["M’Lord Harkonnen"]}
    assert any(
        item["alias"] == "Noble Born"
        and item["suggested_action"] == "remove_from_source_names_or_quarantine"
        for item in result["records"])

    group_sources = {tuple(g["sources"]) for g in result["multi_form_groups"]}
    assert ("Duke Leto", "Duke Leto Atreides", "Leto") in group_sources
    leto = next(g for g in result["multi_form_groups"]
                if tuple(g["sources"]) == ("Duke Leto", "Duke Leto Atreides", "Leto"))
    assert {"莱托", "雷托公爵", "雷托·厄崔迪公爵"} <= set(leto["forms"])
    assert leto["needs_form_contract"] is True

    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()


def test_clean_authority_audit_is_ok(tmp_path: Path):
    db = tmp_path / "db" / "decision_db.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(str(db))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, locked INTEGER DEFAULT 0)"
        )
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, canonical_name TEXT, "
            "source_names TEXT, context_names TEXT, quarantined_names TEXT, "
            "variant_names TEXT, entity_type TEXT, confidence TEXT, "
            "pending_review INTEGER DEFAULT 0)"
        )
        con.execute(
            "INSERT INTO decision_db VALUES (1, 1, 'Arrakis', '厄拉科斯', 1)")
        con.execute(
            "INSERT INTO entity_registry VALUES "
            "(1, '厄拉科斯', '[\"Arrakis\"]', '[]', '[]', '[]', 'place', 'high', 0)")
        con.commit()
    finally:
        con.close()

    result = audit_name_authorities(tmp_path, decision_db=db)

    assert result["ok"] is True
    assert result["status"] == "clean"
    assert result["blocking_count"] == 0


def _make_term_phrase_db(root: Path) -> Path:
    path = root / "db" / "decision_db.sqlite"
    path.parent.mkdir(parents=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT UNIQUE, translation TEXT, locked INTEGER DEFAULT 0)"
        )
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, canonical_name TEXT, "
            "source_names TEXT, context_names TEXT, quarantined_names TEXT, "
            "variant_names TEXT, entity_type TEXT, confidence TEXT, "
            "pending_review INTEGER DEFAULT 0)"
        )
        con.executemany(
            "INSERT INTO decision_db (id, level, source_key, translation) "
            "VALUES (?, 2, ?, ?)",
            [
                (1, "Bene Gesserit", "贝尼·杰瑟里特"),
                (2, "Bene Gesserit Plan", "贝尼·杰瑟里特计划"),
                (3, "Bene Gesserit school", "贝尼·杰瑟里特学校"),
                (4, "Jessica", "杰西卡"),
                (5, "Lady Jessica", "杰西卡夫人"),
                (6, "Kynes", "凯恩斯"),
                (7, "Dr. Kynes", "凯恩斯博士"),
            ],
        )
        registry = [
            (9, "贝尼·杰瑟里特",
             ["Bene Gesserit", "the Bene Gesserit Plan", "Bene Gesserit school"]),
            (10, "杰西卡", ["Jessica", "Lady Jessica"]),
            (11, "凯恩斯", ["Kynes", "Dr. Kynes"]),
        ]
        for rid, canonical, sources in registry:
            con.execute(
                "INSERT INTO entity_registry VALUES "
                "(?, ?, ?, '[]', '[]', '[]', 'person', 'high', 0)",
                (rid, canonical, json.dumps(sources, ensure_ascii=False)),
            )
        con.execute(
            "UPDATE entity_registry SET variant_names=? WHERE id=9",
            (json.dumps(
                ["贝尼·杰赛里特学校", "贝尼·杰瑟里特学校"],
                ensure_ascii=False),),
        )
        con.commit()
    finally:
        con.close()
    return path


def test_term_phrases_are_blocking_but_role_forms_are_not(tmp_path: Path):
    db = _make_term_phrase_db(tmp_path)

    result = audit_name_authorities(tmp_path, decision_db=db)

    term_records = [
        item for item in result["records"]
        if item["kind"] == "term_phrase_alias_quarantine"
    ]
    assert {item["alias"] for item in term_records} == {
        "the Bene Gesserit Plan", "Bene Gesserit school"}
    plan = next(item for item in term_records
                if item["alias"] == "the Bene Gesserit Plan")
    assert plan["suggested_action"] == "remove_from_source_names_or_quarantine"
    assert plan["term_phrase_evidence"] == {
        "base_decision_id": 1,
        "base_source_key": "Bene Gesserit",
        "base_translation": "贝尼·杰瑟里特",
        "extra_source_tokens": ["plan"],
        "inserted_translation": ["计划"],
    }
    assert not any(item["alias"] in {"Lady Jessica", "Dr. Kynes"}
                   for item in term_records)
    assert result["counts"]["term_phrase_alias_quarantine"] == 2
    variant_records = [
        item for item in result["records"]
        if item["kind"] == "term_phrase_variant_quarantine"
    ]
    assert {item["alias"] for item in variant_records} == {
        "贝尼·杰赛里特学校", "贝尼·杰瑟里特学校"}
    assert all(item["suggested_action"] ==
               "remove_from_variant_names_or_quarantine"
               for item in variant_records)
    assert result["counts"]["term_phrase_variant_quarantine"] == 2
    assert result["ok"] is False
    assert result["blocking_count"] >= 4

