"""Round 142 Phase B: read-only story graph query, trace, DPP, and eval."""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

from src.story_graph import build_story_graph
from src.story_graph_query import (
    StoryGraphQueryIndex,
    evaluate_story_graph_query,
    format_query_result,
)

CH01 = """# Chapter 1

Paul met Jessica in the hall.

They spoke about the desert.

***

Jessica walked away alone.

---

Gurney trained Paul at the table.

= = = = = =

A quiet room with no names.
"""

GURNEY_QUOTE = "Gurney trained Paul at the table."


def _make_project(tmp_path: Path):
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
    rows = [
        (1, "保罗·厄崔迪", '["Paul"]', "[]", "person", "male", "high", 0),
        (2, "杰西卡", '["Jessica"]', "[]", "person", "female", "high", 0),
        (3, "格尼·哈莱克", '["Gurney"]', "[]", "person", "male", "high", 0),
    ]
    for row in rows:
        con.execute("INSERT INTO entity_registry VALUES (?,?,?,?,?,?,?,?,?,?)",
                    row + ("literal", "ch01"))
    con.commit()
    con.close()
    profile = {
        "schema_version": 2,
        "voice_notes": [{
            "id": "voice-gurney", "type": "voice", "chapters": ["ch01"],
            "claim": "格尼在训练桌旁从戏谑转为严厉。", "confidence": 0.9,
            "status": "active", "l2_verdict": "confirmed",
            "evidence": [{"chapter": "ch01", "quote": GURNEY_QUOTE,
                          "offset": CH01.index(GURNEY_QUOTE),
                          "match_kind": "exact", "match_score": 1.0}],
        }],
    }
    return tmp_path, profile


def _graph(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    return build_story_graph(root, profile=profile)


def test_query_seeds_from_alias_and_returns_trace(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    result = index.query(query="格尼在训练桌旁教保罗什么？", top_k=5)
    assert result["results"]
    assert result["metrics"]["seed_entities"] >= 2
    assert result["metrics"]["candidate_scenes"] > 0
    assert result["trace"]["hops"][0]["entities"]
    assert result["trace"]["hops"][1]["scenes"]
    assert result["trace"]["dpp"]["enabled"] is True
    top = result["results"][0]
    assert top["scene_id"] == "scene:ch01:0003"
    assert top["entities"]
    assert top["anchor"]["quote"]
    assert top["score_components"]["hop"] == 1.0
    assert "格尼" in top["excerpt"] or "Gurney" in top["excerpt"]
    text = format_query_result(result)
    assert "story-graph-query" in text
    assert "scene:ch01:0003" in text


def test_note_evidence_seeds_gold_scene_and_second_hop_is_bounded(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    result = index.query(query="格尼如何训练保罗？", note_id="note:voice-gurney", top_k=5)
    assert result["results"][0]["scene_id"] == "scene:ch01:0003"
    assert any(e.get("source") == "note_evidence"
               for e in result["results"][0]["evidence"])
    hops = result["trace"]["hops"]
    assert hops[1]["scenes"]
    assert len(hops[2]["entities"]) <= 40
    assert result["trace"]["candidates"]["hop1"] >= 1


def test_query_without_seed_match_is_explicit_and_fail_open(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    result = index.query(query="完全不存在的陌生人", top_k=5)
    assert result["results"] == []
    assert result["trace"]["reason"] == "no_seed_match"
    assert result["metrics"]["candidate_scenes"] == 0


def test_query_never_mutates_graph_or_profile(tmp_path: Path):
    graph = _graph(tmp_path)
    before = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    index.query(query="格尼在训练桌旁教保罗什么？", top_k=10)
    after = json.dumps(graph, ensure_ascii=False, sort_keys=True)
    assert before == after


def test_llm_judge_failure_is_fail_open(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")

    def broken(_prompt: str) -> str:
        raise RuntimeError("offline")

    result = index.query(query="格尼在训练桌旁教保罗什么？",
                         top_k=5, llm_generate=broken)
    assert result["results"]
    assert result["trace"]["llm_judge"]["ok"] is False
    assert "offline" in result["trace"]["llm_judge"]["error"]


def test_llm_judge_rejected_scene_is_removed(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    baseline = index.query(query="格尼在训练桌旁教保罗什么？", top_k=5)
    first_id = baseline["results"][0]["scene_id"]

    def reject_first(_prompt: str) -> str:
        return json.dumps({"verdicts": [{
            "id": first_id, "verdict": "rejected", "score": 1.0,
            "reason": "thematic only",
        }]})

    judged = index.query(query="格尼在训练桌旁教保罗什么？",
                         top_k=5, llm_generate=reject_first)
    assert judged["trace"]["llm_judge"]["ok"] is True
    assert first_id not in [r["scene_id"] for r in judged["results"]]
    assert judged["trace"]["llm_judge"]["rejected"] == 1


def test_evaluation_reports_recall_and_anchor_metrics(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    cases = [{
        "id": "q1",
        "question": "格尼如何训练保罗？",
        "note_id": "note:voice-gurney",
        "gold": [{"chapter": "ch01", "quote": GURNEY_QUOTE}],
    }, {
        "id": "q2",
        "question": "格尼在训练桌旁教保罗什么？",
        "gold": [{"chapter": "ch01", "quote": GURNEY_QUOTE}],
    }]
    report = evaluate_story_graph_query(index, cases, top_k=10)
    metrics = report["metrics"]
    assert metrics["cases"] == 2
    assert metrics["note_evidence_cases"] == 1
    assert metrics["note_similarity_cases"] == 1
    assert metrics["note_evidence_R@5"] == 1.0
    assert metrics["note_similarity_R@5"] == 1.0
    assert metrics["R@5"] >= 0.5
    assert 0.0 <= metrics["anchor_precision@1"] <= 1.0
    assert report["rows"][0]["top_scene"] == "scene:ch01:0003"


def test_note_similarity_can_be_disabled_for_strict_alias_mode(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    auto = index.query(query="格尼在训练桌旁教保罗什么？", top_k=5)
    assert auto["trace"]["note_matches"]
    strict = index.query(query="格尼在训练桌旁教保罗什么？", top_k=5,
                         use_note_similarity=False)
    assert strict["trace"]["note_matches"] == []
    assert not any(seed.get("via") == "note_similarity"
                   for seed in strict["trace"]["seeds"])


def test_note_claim_tokens_are_weighted_over_long_evidence_quotes(tmp_path: Path):
    graph = _graph(tmp_path)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    counts = index.note_token_counts["note:voice-gurney"]
    # "格尼" appears once in the Chinese claim and cannot appear in the
    # English evidence quote; the 3x claim weight must survive.
    assert counts["格尼"] == 3
    matches = index._similar_notes("格尼在训练桌旁从戏谑转为严厉。")
    assert matches[0]["note_id"] == "note:voice-gurney"
    assert "cosine" in matches[0] and "entity_overlap" in matches[0]


def test_only_active_validated_notes_are_queryable(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    dropped = copy.deepcopy(profile["voice_notes"][0])
    dropped["id"] = "voice-dropped"
    dropped["status"] = "dropped"
    profile["voice_notes"].append(dropped)
    graph = build_story_graph(root, profile=profile)
    index = StoryGraphQueryIndex(graph, input_dir=root / "input")
    assert "note:voice-dropped" not in index.notes
    result = index.query(note_id="note:voice-dropped", top_k=5)
    assert result["results"] == []
    assert result["trace"]["reason"] == "no_seed_match"


def test_query_command_is_read_only_and_monolith_runtime_works(tmp_path: Path):
    from src.main import _MUTATING_COMMANDS
    assert "story-graph-query" not in _MUTATING_COMMANDS

    graph = _graph(tmp_path)
    import src.translator_agent as ta
    assert callable(ta.StoryGraphQueryIndex)
    mono_graph = ta.build_story_graph(tmp_path)
    mono_index = ta.StoryGraphQueryIndex(mono_graph, input_dir=tmp_path / "input")
    mono_result = mono_index.query(query="格尼在训练桌旁教保罗什么？", top_k=5)
    assert mono_result["results"]
    assert "def _greedy_dpp_select" in (
        Path(__file__).resolve().parent.parent / "src" / "translator_agent.py"
    ).read_text(encoding="utf-8")
