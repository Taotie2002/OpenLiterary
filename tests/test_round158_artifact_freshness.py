"""Round 158: deterministic artifact freshness contracts."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.artifact_freshness import (
    chapter_manifest,
    decision_db_fingerprint,
    eval_freshness,
    graph_freshness,
    load_story_graph_checked,
    profile_fingerprint,
    profile_notes_hash,
    profile_source_freshness,
)
from src.story_graph import build_story_graph, save_story_graph


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_db(path: Path, names=("Paul",)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE IF NOT EXISTS entity_registry ("
            "id INTEGER PRIMARY KEY, canonical_name TEXT, source_names TEXT,"
            "variant_names TEXT, entity_type TEXT, pronoun_gender TEXT,"
            "confidence REAL, pending_review INTEGER, translation_strategy TEXT,"
            "first_seen_chunk TEXT, quarantined_names TEXT)"
        )
        for name in names:
            con.execute(
                "INSERT INTO entity_registry (canonical_name, source_names,"
                "variant_names, entity_type, confidence, pending_review) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, json.dumps([name]), "[]", "person", 0.99, 0),
            )
        con.commit()
    finally:
        con.close()


def _profile(chapter_hashes=None):
    prof = {
        "schema_version": 2,
        "source": "known_work",
        "generation": {
            "model": "test-model",
            "merge_model": "test-model",
            "params": {"mode": "deep"},
            "prompt_version": "r131.deep.v3",
            "time": "2026-11-16T00:00:00",
        },
        "voice_notes": [
            {
                "id": "voice-1",
                "type": "voice",
                "chapters": ["ch01"],
                "claim": "叙述保持克制。",
                "evidence": [{"chapter": "ch01", "quote": "hello world",
                              "offset": 0, "match_kind": "exact"}],
                "confidence": 0.9,
                "status": "active",
                "l2_verdict": "confirmed",
            }
        ],
    }
    if chapter_hashes is not None:
        prof["generation"]["inputs"] = {
            "chapters": chapter_hashes,
            "profile_version": 2,
            "prompt_version": "r131.deep.v3",
        }
    return prof


def _fixture(tmp_path: Path):
    _write(tmp_path / "input" / "ch01.md", "hello world\n\nsecond scene")
    _write(tmp_path / "input" / "ch02.md", "another chapter")
    _make_db(tmp_path / "db" / "decision_db.sqlite")
    return tmp_path


def test_chapter_manifest_tracks_bytes_not_paths(tmp_path: Path):
    root = _fixture(tmp_path)
    first = chapter_manifest(root / "input")
    assert first["ch01"]["chars"] == len("hello world\n\nsecond scene")
    assert len(first["ch01"]["sha256"]) == 64
    # Touch only (same bytes) must not change identity.
    (root / "input" / "ch01.md").touch()
    assert chapter_manifest(root / "input") == first
    _write(root / "input" / "ch01.md", "hello world\n\nchanged scene")
    assert chapter_manifest(root / "input")["ch01"]["sha256"] != first["ch01"]["sha256"]


def test_profile_fingerprint_stable_and_notes_sensitive():
    base = _profile()
    first = profile_fingerprint(base)
    # Wall-clock fields do not participate.
    other = json.loads(json.dumps(base))
    other["generation"]["time"] = "2099-01-01T00:00:00"
    other["generated_at"] = "2099-01-01T00:00:00"
    assert profile_fingerprint(other) == first
    changed = json.loads(json.dumps(base))
    changed["voice_notes"][0]["claim"] = "不同判断"
    assert profile_fingerprint(changed) != first


def test_profile_source_freshness_legacy_fresh_stale(tmp_path: Path):
    root = _fixture(tmp_path)
    legacy = _profile()
    res = profile_source_freshness(legacy, root)
    assert res["ok"] is False and res["status"] == "untracked"

    hashes = {cid: rec["sha256"]
              for cid, rec in chapter_manifest(root / "input").items()}
    pinned = _profile(hashes)
    res = profile_source_freshness(pinned, root)
    assert res["ok"] is True and res["status"] == "fresh"

    _write(root / "input" / "ch02.md", "another chapter, revised")
    res = profile_source_freshness(pinned, root)
    assert res["ok"] is False and res["status"] == "stale"
    assert "chapter_sources_changed" in res["reasons"]
    # A chapter-scoped check can still be fresh for an unchanged chapter.
    scoped = profile_source_freshness(pinned, root, chapters=["ch01"])
    assert scoped["ok"] is True and scoped["status"] == "fresh"


def test_decision_db_fingerprint_tracks_registry_rows(tmp_path: Path):
    root = _fixture(tmp_path)
    db = root / "db" / "decision_db.sqlite"
    first = decision_db_fingerprint(db)
    assert first["entity_registry_rows"] == 1
    con = sqlite3.connect(str(db))
    try:
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names,"
            "variant_names, entity_type, confidence, pending_review) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("Jessica", json.dumps(["Jessica"]), "[]", "person", 0.95, 0),
        )
        con.commit()
    finally:
        con.close()
    second = decision_db_fingerprint(db)
    assert second["entity_registry_rows"] == 2
    assert second["rows_sha256"] != first["rows_sha256"]


def test_graph_freshness_reasons_and_allow_stale(tmp_path: Path):
    root = _fixture(tmp_path)
    hashes = {cid: rec["sha256"]
              for cid, rec in chapter_manifest(root / "input").items()}
    prof = _profile(hashes)
    graph = build_story_graph(root, profile=prof)
    fresh = graph_freshness(graph, root, prof)
    assert fresh["ok"] is True and fresh["status"] == "fresh"

    changed = json.loads(json.dumps(prof))
    changed["voice_notes"][0]["claim"] = "新的判断"
    stale = graph_freshness(graph, root, changed)
    assert stale["ok"] is False and stale["status"] == "stale"
    assert "profile_notes_changed" in stale["reasons"]

    _write(root / "input" / "ch01.md", "hello world changed")
    stale = graph_freshness(graph, root, prof)
    assert "chapter_sources_changed" in stale["reasons"]

    path = root / "input" / "story_graph.json"
    save_story_graph(graph, path=path, project_root=root)
    rejected = load_story_graph_checked(root, path, prof, require_current=True)
    assert rejected["ok"] is False
    allowed = load_story_graph_checked(root, path, prof, require_current=False)
    assert allowed["ok"] is True and allowed["status"] == "stale"


def test_graph_freshness_detects_db_row_change(tmp_path: Path):
    root = _fixture(tmp_path)
    hashes = {cid: rec["sha256"]
              for cid, rec in chapter_manifest(root / "input").items()}
    prof = _profile(hashes)
    graph = build_story_graph(root, profile=prof)
    _make_db(root / "db" / "decision_db.sqlite", names=("Paul", "Jessica"))
    res = graph_freshness(graph, root, prof)
    assert "decision_db_hash_changed" in res["reasons"]
    assert "decision_db_row_count_changed" in res["reasons"]


def test_eval_freshness_rejects_legacy_payload(tmp_path: Path):
    root = _fixture(tmp_path)
    prof = _profile()
    legacy = {"version": 1, "book": "Dune", "source": "round142",
              "cases": [{"id": f"case-{i}", "note_id": f"note:missing-{i}",
                         "question": "q", "gold": []} for i in range(26)]}
    res = eval_freshness(legacy, root, prof)
    assert res["ok"] is False and res["status"] == "invalid"
    assert "eval_schema_legacy" in res["reasons"]
    assert "eval_lineage_fields_missing" in res["reasons"]
    assert res["details"]["invalid_cases"] == 26
    assert res["details"]["stale_notes"] == 26


def test_eval_freshness_accepts_v2_with_verified_gold(tmp_path: Path):
    root = _fixture(tmp_path)
    hashes = {cid: rec["sha256"]
              for cid, rec in chapter_manifest(root / "input").items()}
    prof = _profile(hashes)
    graph = build_story_graph(root, profile=prof)
    payload = {
        "schema_version": 2,
        "book": "Dune",
        "source": "round158-test",
        "profile_notes_hash": profile_notes_hash(prof),
        "chapter_manifest": chapter_manifest(root / "input"),
        "graph_fingerprint": graph["fingerprint"],
        "cases": [{
            "id": "case-1",
            "note_id": "voice-1",
            "question": "叙述如何？",
            "gold": [{"chapter": "ch01", "quote": "hello world", "offset": 0}],
        }],
    }
    res = eval_freshness(payload, root, prof, graph=graph)
    assert res["ok"] is True, res
    assert res["status"] == "fresh"
    assert res["details"]["stale_notes"] == 0

    payload["cases"][0]["gold"][0]["quote"] = "quote that is not present"
    bad = eval_freshness(payload, root, prof, graph=graph)
    assert bad["ok"] is False
    assert any("gold_quote_unmatched" in e
               for item in bad["details"]["case_errors"]
               for e in item["errors"])
