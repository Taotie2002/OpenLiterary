"""Round 142: deterministic scene nodes and entity -> scene incidence."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.judgment_layer import sha256_text
from src.story_graph import (
    GRAPH_VERSION,
    _chapter_scenes,
    build_story_graph,
    story_graph_context_for_chapter,
)

CH01 = """# Chapter 1

Paul met Jessica in the hall.

They spoke about the desert.

***

Jessica walked away alone.

---

Paul watched her go.

= = = = = =

A final paragraph with no names.
"""


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "db").mkdir()
    (tmp_path / "input" / "ch01.md").write_text(CH01, encoding="utf-8")
    con = sqlite3.connect(tmp_path / "db" / "decision_db.sqlite")
    con.execute("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY, canonical_name TEXT, source_names TEXT,
            variant_names TEXT, entity_type TEXT, pronoun_gender TEXT,
            confidence TEXT, pending_review INTEGER, translation_strategy TEXT,
            first_seen_chunk TEXT
        )
    """)
    con.execute("INSERT INTO entity_registry VALUES "
                "(1,'保罗','[\"Paul\"]','[]','person','male','high',0,'literal','ch01')")
    con.execute("INSERT INTO entity_registry VALUES "
                "(2,'杰西卡','[\"Jessica\"]','[]','person','female','medium',1,'literal','ch01')")
    con.commit()
    con.close()
    return tmp_path


def _edge_counts(graph, edge_type):
    return {(e["source"], e["target"]): int(e.get("count") or 0)
            for e in graph["edges"] if e["type"] == edge_type}


def test_chapter_scenes_are_deterministic_and_offset_exact():
    scenes = _chapter_scenes(CH01, "ch01", scene_chars=2200)
    assert [s["id"] for s in scenes] == [
        "scene:ch01:0001", "scene:ch01:0002", "scene:ch01:0003", "scene:ch01:0004",
    ]
    assert [s["paragraphs"] for s in scenes] == [3, 2, 2, 2]
    for scene in scenes:
        fragment = CH01[scene["start"]:scene["end"]]
        assert fragment
        assert scene["chars"] == scene["end"] - scene["start"] == len(fragment)
        assert scene["hash"] == sha256_text(fragment)
        assert scene["scene_policy"].startswith("paragraph_grouping:v1;")
    assert CH01[scenes[1]["start"]:scenes[1]["end"]].strip().startswith("***")
    assert CH01[scenes[2]["start"]:scenes[2]["end"]].strip().startswith("---")
    assert CH01[scenes[3]["start"]:scenes[3]["end"]].strip().startswith("= = = = = =")
    assert scenes[0]["start"] == 0
    assert all(a["end"] <= b["start"] for a, b in zip(scenes, scenes[1:]))


def test_scene_grouping_never_splits_or_reconstructs_paragraphs():
    text = "A" * 80 + "\n\n" + "B" * 80 + "\n\n" + "C" * 400
    scenes = _chapter_scenes(text, "ch01", scene_chars=150)
    assert len(scenes) == 3
    assert text[scenes[0]["start"]:scenes[0]["end"]] == "A" * 80
    assert text[scenes[1]["start"]:scenes[1]["end"]] == "B" * 80
    assert text[scenes[2]["start"]:scenes[2]["end"]] == "C" * 400
    assert scenes[2]["chars"] == 400


def test_mentions_in_aggregates_by_entity_scene_and_keeps_provenance(tmp_path: Path):
    graph = build_story_graph(_make_project(tmp_path))
    assert graph["schema_version"] == GRAPH_VERSION == 5
    scenes = [n for n in graph["nodes"] if n["type"] == "scene"]
    assert len(scenes) == 4
    assert {n["attributes"]["chapter"] for n in scenes} == {"ch01"}

    mentions = _edge_counts(graph, "mentions_in")
    assert mentions == {
        ("entity:entity-1", "scene:ch01:0001"): 1,
        ("entity:entity-1", "scene:ch01:0003"): 1,
        ("entity:entity-2", "scene:ch01:0001"): 1,
        ("entity:entity-2", "scene:ch01:0002"): 1,
    }
    appears = _edge_counts(graph, "appears_in")
    assert sum(mentions.values()) == sum(appears.values()) == 4
    assert all(("entity:entity-%d" % i, "chapter:ch01") in appears for i in (1, 2))

    scene_by_id = {n["id"]: n for n in scenes}
    for edge in graph["edges"]:
        if edge["type"] != "mentions_in":
            continue
        scene = scene_by_id[edge["target"]]["attributes"]
        prov = edge["provenance"][0]
        assert scene["start"] <= prov["start"] < prov["end"] <= scene["end"]
        assert prov["offset"] == prov["start"]
        assert CH01[prov["start"]:prov["end"]] == prov["surface"]
        assert prov["surface"] in prov["quote"]
        assert prov["scene"] == edge["target"]

    by_source = {}
    for edge in graph["edges"]:
        if edge["type"] == "mentions_in":
            by_source[edge["source"]] = edge
    assert by_source["entity:entity-1"]["certification"] == "auto_verified"
    assert by_source["entity:entity-2"]["certification"] == "pending"
    assert by_source["entity:entity-2"]["entity_certification"] == "pending"


def test_scene_coverage_is_additive_and_does_not_change_old_coverage(tmp_path: Path):
    root = _make_project(tmp_path)
    small = build_story_graph(root, scene_chars=80)
    large = build_story_graph(root, scene_chars=5000)
    cov = small["coverage"]
    assert cov["scenes_total"] == 4
    assert cov["scenes_with_entities"] == 3
    assert cov["scene_coverage"] == 0.75
    assert cov["scenes_with_certified_entities"] == 2
    assert cov["scenes_without_certified_entities"] == [
        "scene:ch01:0002", "scene:ch01:0004",
    ]
    assert cov["entity_scene_coverage"] == 1.0
    assert cov["certified_entity_scene_coverage"] == 1.0
    assert cov["scene_coverage_by_chapter"]["ch01"] == 0.75

    for key in (
        "entities_total", "entities_with_appearance", "entity_coverage",
        "certified_entities_total", "certified_entities_with_appearance",
        "certified_entity_coverage", "certified_entity_coverage_of_trusted",
        "chapters_total", "chapters_with_entities", "chapter_coverage",
        "answerable", "partial_coverage",
    ):
        assert small["coverage"][key] == large["coverage"][key], key
    assert _edge_counts(small, "appears_in") == _edge_counts(large, "appears_in")
    assert _edge_counts(small, "mentions_in") == _edge_counts(large, "mentions_in")
    assert small["fingerprint"] != large["fingerprint"]
    assert small["generation"]["params"]["scene_chars"] == 80
    assert large["generation"]["params"]["scene_chars"] == 5000
    assert small["generation"]["params"]["scene_policy"]
    assert small["diagnostics"]["unassigned_scene_occurrences"] == 0
    assert small["diagnostics"]["scene_mentions_total"] == 4
    assert small["diagnostics"]["scene_count"] == small["coverage"]["scenes_total"]


def test_scene_segmentation_is_stable_across_rebuilds(tmp_path: Path):
    root = _make_project(tmp_path)
    first = build_story_graph(root)
    second = build_story_graph(root)
    first_scenes = [(n["id"], n["attributes"]) for n in first["nodes"]
                    if n["type"] == "scene"]
    second_scenes = [(n["id"], n["attributes"]) for n in second["nodes"]
                     if n["type"] == "scene"]
    assert first_scenes == second_scenes
    assert _edge_counts(first, "mentions_in") == _edge_counts(second, "mentions_in")
    assert first["fingerprint"] == second["fingerprint"]


def test_scene_incidence_never_guesses_ambiguous_alias(tmp_path: Path):
    root = _make_project(tmp_path)
    con = sqlite3.connect(root / "db" / "decision_db.sqlite")
    con.execute("INSERT INTO entity_registry VALUES "
                "(4,'另一人','[\"Paul\"]','[]','person','male','high',0,'literal','ch01')")
    con.commit()
    con.close()
    graph = build_story_graph(root)
    assert not [e for e in graph["edges"] if e["type"] == "mentions_in"
                and e["source"] in ("entity:entity-1", "entity:entity-4")]
    assert graph["diagnostics"]["unresolved_alias_count"] == 1
    assert graph["diagnostics"]["unassigned_scene_occurrences"] == 0


def test_v4_artifact_still_serves_chapter_context():
    graph = {
        "schema_version": 4,
        "coverage": {"answerable": True},
        "nodes": [
            {"id": "entity:entity-1", "type": "entity", "label": "保罗"},
            {"id": "chapter:ch01", "type": "chapter", "label": "ch01"},
        ],
        "edges": [{
            "type": "appears_in", "source": "entity:entity-1",
            "target": "chapter:ch01", "count": 2,
            "certification": "auto_verified",
        }],
    }
    context = story_graph_context_for_chapter(graph, "ch01")
    assert "保罗" in context and "本章出现 2 次" in context


def test_monolith_contains_scene_incidence():
    root = Path(__file__).resolve().parent.parent
    text = (root / "src" / "translator_agent.py").read_text(encoding="utf-8")
    assert "def _chapter_scenes" in text
    assert '"mentions_in"' in text or "'mentions_in'" in text
