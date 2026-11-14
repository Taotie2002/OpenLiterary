"""Round 149: undo only mechanically-detectable cross-person unify merges."""
from __future__ import annotations

import json
import sqlite3

from src.core.decision_engine import (
    apply_name_unification,
    build_name_unification_maps,
    repair_false_unify_merges,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            review_reason TEXT,
            pending_review INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    rows = [
        ("弗拉基米尔·哈克南男爵",
         ["Baron Harkonnen", "Baron Vladimir Harkonnen", "Vladimir Harkonnen"],
         ["皇帝", "皇帝陛下"], "", 0),
        ("沙达姆四世", ["Shaddam IV"], ["皇帝", "皇帝陛下"],
         "Round134 unify: 译名冲突并入 弗拉基米尔·哈克南男爵", 1),
        ("卡拉丹城堡", ["Castle Caladan"], ["卡拉丹"], "", 0),
        ("卡拉丹", ["Caladan"], [],
         "Round134 unify: 译名冲突并入 卡拉丹城堡", 1),
        ("禅苏尼", ["Zensunni"], ["曾苏尼"], "", 0),
        ("曾苏尼", ["Zensunni"], [],
         "Round134 unify: 译名冲突并入 禅苏尼", 1),
    ]
    conn.executemany(
        "INSERT INTO entity_registry "
        "(canonical_name, source_names, variant_names, review_reason, pending_review) "
        "VALUES (?, ?, ?, ?, ?)",
        [(name, json.dumps(src, ensure_ascii=False), json.dumps(var, ensure_ascii=False),
          reason, pending) for name, src, var, reason, pending in rows],
    )
    conn.commit()
    return conn


def test_dry_run_repairs_only_unrelated_proper_name_merge():
    conn = _conn()
    res = repair_false_unify_merges(conn, dry_run=True)
    assert res["repaired"] == 1
    assert [c["canonical"] for c in res["changes"]] == ["沙达姆四世"]
    assert conn.execute(
        "SELECT pending_review FROM entity_registry WHERE canonical_name='沙达姆四世'"
    ).fetchone()[0] == 1


def test_apply_restores_entity_and_removes_foreign_variants():
    conn = _conn()
    res = repair_false_unify_merges(conn, dry_run=False)
    assert res["repaired"] == 1
    pending = conn.execute(
        "SELECT pending_review, review_reason FROM entity_registry "
        "WHERE canonical_name='沙达姆四世'"
    ).fetchone()
    assert pending[0] == 0
    assert "Round149 unmerge" in pending[1]
    variants = json.loads(conn.execute(
        "SELECT variant_names FROM entity_registry "
        "WHERE canonical_name='弗拉基米尔·哈克南男爵'"
    ).fetchone()[0])
    assert variants == []
    # Legitimate short forms remain merged.
    for name in ("卡拉丹", "曾苏尼"):
        assert conn.execute(
            "SELECT pending_review FROM entity_registry WHERE canonical_name=?", (name,)
        ).fetchone()[0] == 1


def test_role_words_on_mohiam_never_become_cross_person_replacements():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            entity_type TEXT,
            confidence TEXT DEFAULT 'high',
            pending_review INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    rows = [
        ("盖乌斯·海伦·莫希阿姆", ["Gaius Helen Mohiam", "Reverend Mother"],
         ["圣母", "真言者", "尊母", "老妇人"]),
        ("杰西卡", ["Jessica"], []),
    ]
    conn.executemany(
        "INSERT INTO entity_registry "
        "(canonical_name, source_names, variant_names, entity_type) "
        "VALUES (?, ?, ?, 'person')",
        [(canonical, json.dumps(src, ensure_ascii=False),
          json.dumps(variants, ensure_ascii=False))
         for canonical, src, variants in rows],
    )
    conn.commit()

    maps = build_name_unification_maps(conn)
    replace_sources = {form[0] for form in maps["forms"]}
    for role in ("真言者", "尊母", "圣母", "老妇人"):
        assert role not in replace_sources
        assert role in maps["guards"]
    assert ["杰西卡", "盖乌斯", "盖乌斯·海伦·莫希阿姆"] not in maps["forms"]

    text = "杰西卡看见真言者。尊母到来。"
    new_text, _ = apply_name_unification(text, maps)
    assert new_text == text
