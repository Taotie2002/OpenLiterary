"""Round 158: chapter digest derivation from the deep-read audit."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.artifact_freshness import chapter_manifest, profile_source_freshness
from src.judgment_layer import sha256_text
import src.style_profiler as sp
from src.style_profiler import (
    build_chapter_digests,
    derive_profile_digests,
    normalize_profile,
)


def test_build_chapter_digests_complete_and_incomplete():
    texts = {"ch01": "abcdefghij", "ch02": "xyz"}
    payloads = [
        {"chapter_id": "ch01", "part_id": "ch01:1", "char_start": 0,
         "char_end": 6, "synopsis": "first", "register_summary": "A",
         "parse_error": ""},
        {"chapter_id": "ch01", "part_id": "ch01:2", "char_start": 4,
         "char_end": 10, "synopsis": "second", "register_summary": "B",
         "parse_error": ""},
        {"chapter_id": "ch02", "part_id": "ch02:1", "char_start": 0,
         "char_end": 3, "synopsis": "third", "register_summary": "C",
         "parse_error": "bad json"},
    ]
    res = build_chapter_digests(texts, payloads, model="m", generated_at="t")
    by_id = {d["chapter_id"]: d for d in res["digests"]}
    assert by_id["ch01"]["complete"] is True
    assert by_id["ch01"]["segment_coverage"] == 1.0
    assert by_id["ch01"]["digest"] == "first\n\nsecond"
    assert by_id["ch01"]["source_sha256"] == sha256_text("abcdefghij")
    assert by_id["ch02"]["complete"] is False
    assert "segment_parse_error" in by_id["ch02"]["missing_reason"]
    assert res["complete"] is False
    assert res["summary"]["missing"] == ["ch02"]


def test_build_chapter_digests_detects_coverage_gap():
    texts = {"ch01": "abcdefghij"}
    payloads = [
        {"chapter_id": "ch01", "part_id": "ch01:1", "char_start": 0,
         "char_end": 4, "synopsis": "first", "parse_error": ""},
        {"chapter_id": "ch01", "part_id": "ch01:2", "char_start": 6,
         "char_end": 10, "synopsis": "second", "parse_error": ""},
    ]
    res = build_chapter_digests(texts, payloads)
    entry = res["digests"][0]
    assert entry["complete"] is False
    assert entry["segment_coverage"] == 0.8
    assert "segment_coverage_incomplete" in entry["missing_reason"]


def test_derive_profile_digests_binds_current_chapter_bytes(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "ch01.md").write_text("abcdefghij", encoding="utf-8")
    (input_dir / "ch02.md").write_text("klm", encoding="utf-8")
    audit = {
        "generated_at": "2026-11-16T00:00:00",
        "generation": {"model": "m", "params": {"mode": "deep"}},
        "segment_payloads": [
            {"chapter_id": "ch01", "part_id": "ch01:1", "char_start": 0,
             "char_end": 10, "synopsis": "one", "register_summary": "r1",
             "parse_error": ""},
            {"chapter_id": "ch02", "part_id": "ch02:1", "char_start": 0,
             "char_end": 3, "synopsis": "two", "register_summary": "r2",
             "parse_error": ""},
        ],
    }
    profile = {"schema_version": 2, "source": "known_work",
               "generation": {"model": "m", "prompt_version": "r131.deep.v3",
                              "params": {"mode": "deep"}, "time": "t"}}
    res = derive_profile_digests(tmp_path, profile, audit)
    assert res["ok"] is True and res["complete"] is True
    out = res["profile"]
    manifest = chapter_manifest(input_dir)
    assert out["generation"]["inputs"]["chapters"] == {
        cid: rec["sha256"] for cid, rec in manifest.items()}
    assert {d["chapter_id"] for d in out["chapter_digests"]} == {"ch01", "ch02"}
    fresh = profile_source_freshness(out, tmp_path)
    assert fresh["ok"] is True and fresh["status"] == "fresh"

    (input_dir / "ch01.md").write_text("changed", encoding="utf-8")
    stale = profile_source_freshness(out, tmp_path)
    assert stale["status"] == "stale"


def test_normalize_profile_preserves_digests_and_inputs(tmp_path: Path):
    inputs = {"chapters": {"ch01": sha256_text("abc")},
              "profile_version": 2, "prompt_version": "r131.deep.v3"}
    data = {
        "schema_version": 2,
        "source": "known_work",
        "generation": {"model": "m", "params": {"mode": "deep"},
                       "prompt_version": "r131.deep.v3", "inputs": inputs},
        "chapter_digests": [{"chapter_id": "ch01", "source_sha256": sha256_text("abc"),
                             "source_chars": 3, "complete": True,
                             "digest": "one", "segment_ids": ["ch01:1"],
                             "segment_coverage": 1.0}],
        "chapter_digests_summary": {"complete": True, "chapters_total": 1,
                                    "chapters_complete": 1, "missing": []},
        "chapter_digests_complete": True,
    }
    out = normalize_profile(data)
    assert out["generation"]["inputs"] == inputs
    assert out["chapter_digests"][0]["chapter_id"] == "ch01"
    assert out["chapter_digests_complete"] is True
    assert out["chapter_digests_summary"]["complete"] is True


def _make_name_authority_db(
    tmp_path: Path,
    *,
    canonical: str = "贝尼·杰瑟里特",
    source: str = "Bene Gesserit",
    variant: str = "",
    extra_variant: str = "",
    locked_translation: str = "贝尼·杰赛里特",
) -> Path:
    path = tmp_path / "decision_db.sqlite"
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE entity_registry (canonical_name TEXT, source_names TEXT, "
            "variant_names TEXT, confidence TEXT, pending_review INTEGER)")
        con.execute(
            "CREATE TABLE decision_db (source_key TEXT, translation TEXT, locked INTEGER)")
        con.execute(
            "INSERT INTO entity_registry VALUES (?, ?, ?, 'high', 0)",
            (canonical, json.dumps([source], ensure_ascii=False),
             json.dumps([variant] if variant else [], ensure_ascii=False)),
        )
        if extra_variant:
            con.execute(
                "INSERT INTO entity_registry VALUES (?, ?, ?, 'high', 0)",
                ("另一个活跃实体", json.dumps(["Other"], ensure_ascii=False),
                 json.dumps([extra_variant], ensure_ascii=False)),
            )
        con.execute(
            "INSERT INTO decision_db VALUES (?, ?, 1)",
            (source, locked_translation),
        )
        con.commit()
    finally:
        con.close()
    return path


def test_profile_hints_sanitize_active_registry_name_drift(tmp_path: Path):
    db = _make_name_authority_db(tmp_path)
    profile = {
        "source": "known_work",
        "translation_hints": [
            "Bene Gesserit = 贝尼·杰赛里特／姐妹会：统一音译。",
            "Kwisatz Haderach = 贝尼·杰赛里特男性先知性称谓。",
        ],
    }
    before = json.dumps(profile, ensure_ascii=False, sort_keys=True)

    guide = sp.profile_to_style_guide(profile, base={}, decision_db=db)

    assert guide["translation_hints"][0].startswith(
        "Bene Gesserit = 贝尼·杰瑟里特／姐妹会")
    assert "贝尼·杰赛里特" not in "\n".join(guide["translation_hints"])
    assert guide["_translation_hint_name_repairs"] == [
        {"wrong": "贝尼·杰赛里特", "canonical": "贝尼·杰瑟里特"},
        {"wrong": "贝尼·杰赛里特", "canonical": "贝尼·杰瑟里特"},
    ]
    assert json.dumps(profile, ensure_ascii=False, sort_keys=True) == before


def test_profile_hints_do_not_rewrite_another_active_entity_variant(
        tmp_path: Path):
    db = _make_name_authority_db(
        tmp_path,
        canonical="盖乌斯·海伦·莫希阿姆",
        source="Gaius Helen Mohiam",
        extra_variant="盖乌斯·海伦·莫哈阿姆",
    )
    hint = "Gaius Helen Mohiam = 盖乌斯·海伦·莫哈阿姆：另一位活跃实体的合法变体。"
    profile = {"source": "known_work", "translation_hints": [hint]}

    guide = sp.profile_to_style_guide(profile, base={}, decision_db=db)

    assert guide["translation_hints"] == [hint]
    assert "_translation_hint_name_repairs" not in guide
