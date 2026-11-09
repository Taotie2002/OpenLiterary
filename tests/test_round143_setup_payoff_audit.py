"""Round 142 Phase C: setup/payoff candidate audit and controlled pairing."""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path

from src.setup_payoff_audit import (
    build_setup_payoff_audit,
    promote_setup_payoff_pairs,
    select_setup_payoff_candidates,
    verify_setup_payoff_selections,
)
from src.story_graph import build_story_graph
from src.story_graph_query import StoryGraphQueryIndex

CH01 = """# Chapter 1

Gurney promised Paul that he would teach him the weirding way before the desert journey.

Paul nodded and said he would remember the promise.
"""

CH02 = """# Chapter 2

Jessica walked through the hall and spoke quietly to the servants.

She said nothing about the promise.
"""

CH03 = """# Chapter 3

Gurney kept his promise and taught Paul the weirding way beside the desert rocks.

Paul finally understood the lesson.
"""

CH07 = """# Chapter 7

Paul said he would destroy the spice before the emperor could take it.

The emperor did not answer.
"""

SETUP_QUOTE = "Gurney promised Paul that he would teach him the weirding way before the desert journey."
PAYOFF_QUOTE = "Gurney kept his promise and taught Paul the weirding way beside the desert rocks."
TERMINAL_QUOTE = "Paul said he would destroy the spice before the emperor could take it."


def _note(note_id: str, chapter: str, quote: str, text: str, status: str = "dormant"):
    return {
        "id": note_id,
        "type": "setup_payoff",
        "chapters": [chapter],
        "claim": quote,
        "evidence": [{"chapter": chapter, "quote": quote,
                      "offset": text.index(quote), "match_kind": "exact",
                      "match_score": 1.0, "unique": True}],
        "confidence": 0.55,
        "status": status,
        "pairing_status": "unpaired",
        "meta": {"pairing_status": "unpaired"},
    }


def _make_project(tmp_path: Path):
    (tmp_path / "input").mkdir()
    (tmp_path / "db").mkdir()
    (tmp_path / "input" / "ch01.md").write_text(CH01, encoding="utf-8")
    (tmp_path / "input" / "ch02.md").write_text(CH02, encoding="utf-8")
    (tmp_path / "input" / "ch03.md").write_text(CH03, encoding="utf-8")
    (tmp_path / "input" / "ch07.md").write_text(CH07, encoding="utf-8")
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
        (2, "格尼·哈莱克", '["Gurney"]', "[]", "person", "male", "high", 0),
        (3, "杰西卡", '["Jessica"]', "[]", "person", "female", "high", 0),
    ]
    for row in rows:
        con.execute("INSERT INTO entity_registry VALUES (?,?,?,?,?,?,?,?,?,?)",
                    row + ("literal", "ch01"))
    con.commit()
    con.close()
    profile = {
        "schema_version": 2,
        "voice_notes": [],
        "narrative_arc": [],
        "setup_payoff": [
            _note("setup-ch01-gurney", "ch01", SETUP_QUOTE, CH01),
            _note("setup-ch07-spice", "ch07", TERMINAL_QUOTE, CH07),
        ],
    }
    return tmp_path, profile


def _graph_and_index(tmp_path: Path, profile):
    graph = build_story_graph(tmp_path, profile=profile)
    index = StoryGraphQueryIndex(graph, input_dir=tmp_path / "input")
    return graph, index


def test_candidate_audit_is_later_only_and_does_not_mutate(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    graph, index = _graph_and_index(root, profile)
    before = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    audit = build_setup_payoff_audit(profile, index, top_n=5)
    after = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    assert before == after
    cases = {c["setup_id"]: c for c in audit["cases"]}
    assert cases["setup-ch01-gurney"]["candidates"]
    assert all(str(c["chapter"]) > "ch01"
               for c in cases["setup-ch01-gurney"]["candidates"])
    assert any(c["chapter"] == "ch03"
               for c in cases["setup-ch01-gurney"]["candidates"])
    assert cases["setup-ch07-spice"]["terminal"] is True
    assert cases["setup-ch07-spice"]["candidates"] == []
    assert audit["summary"]["setups"] == 2
    assert audit["summary"]["terminal"] == 1


def test_llm_selection_and_l1_make_pair_eligible(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    graph, index = _graph_and_index(root, profile)
    audit = build_setup_payoff_audit(profile, index, top_n=5)
    target = next(c for c in audit["cases"] if c["setup_id"] == "setup-ch01-gurney")
    chosen = next(c for c in target["candidates"] if c["chapter"] == "ch03")

    def choose(_prompt: str) -> str:
        return json.dumps({"decisions": [{
            "setup_id": "setup-ch01-gurney",
            "candidate_id": chosen["candidate_id"],
            "verdict": "payoff",
            "reason": "Gurney fulfils the training promise",
        }]})

    selected = select_setup_payoff_candidates(audit, index.chapter_texts, choose)
    target = next(c for c in selected["cases"] if c["setup_id"] == "setup-ch01-gurney")
    assert target["selection"]["verdict"] == "payoff"
    assert target["selection"]["l1"]["ok"] is True
    assert selected["summary"]["selected"] == 1
    assert selected["summary"]["l1_ok"] == 1


def test_l2_confirmation_and_promotion_preserve_terminal_marker(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    graph, index = _graph_and_index(root, profile)
    audit = build_setup_payoff_audit(profile, index, top_n=5)
    target = next(c for c in audit["cases"] if c["setup_id"] == "setup-ch01-gurney")
    chosen = next(c for c in target["candidates"] if c["chapter"] == "ch03")

    def choose(_prompt: str) -> str:
        return json.dumps({"decisions": [{
            "setup_id": "setup-ch01-gurney",
            "candidate_id": chosen["candidate_id"],
            "verdict": "payoff",
            "reason": "payoff",
        }]})

    def confirm_all(prompt: str) -> str:
        ids = []
        for raw in prompt.split('"id": "')[1:]:
            nid = raw.split('"', 1)[0]
            if nid.startswith("setup-ch01"):
                ids.append(nid)
        return json.dumps({"decisions": [
            {"id": nid, "verdict": "confirmed", "reason": "evidence supports pairing"}
            for nid in ids]})

    selected = select_setup_payoff_candidates(audit, index.chapter_texts, choose)
    verified = verify_setup_payoff_selections(
        selected, index.chapter_texts, llm_generate=confirm_all)
    target = next(c for c in verified["cases"] if c["setup_id"] == "setup-ch01-gurney")
    assert target["selection"]["l2_verdict"] == "confirmed"
    assert target["selection"]["promotion_eligible"] is True

    result = promote_setup_payoff_pairs(profile, verified)
    promoted = {n["id"]: n for n in result["profile"]["setup_payoff"]}
    assert promoted["setup-ch01-gurney"]["status"] == "active"
    assert promoted["setup-ch01-gurney"]["chapters"] == ["ch01", "ch03"]
    assert len(promoted["setup-ch01-gurney"]["evidence"]) == 2
    assert promoted["setup-ch07-spice"]["status"] == "dormant"
    assert promoted["setup-ch07-spice"]["pairing_status"] == "terminal_unpaired"
    assert result["activated"][0]["setup_id"] == "setup-ch01-gurney"
    assert "setup-ch07-spice" in result["terminal"]


def test_l1_failure_never_promotes(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    graph, index = _graph_and_index(root, profile)
    audit = build_setup_payoff_audit(profile, index, top_n=5)
    target = next(c for c in audit["cases"] if c["setup_id"] == "setup-ch01-gurney")
    chosen = next(c for c in target["candidates"] if c["chapter"] == "ch03")
    chosen["anchor_quote"] = "this quote is not in the chapter"

    def choose(_prompt: str) -> str:
        return json.dumps({"decisions": [{
            "setup_id": "setup-ch01-gurney",
            "candidate_id": chosen["candidate_id"],
            "verdict": "payoff",
        }]})

    selected = select_setup_payoff_candidates(audit, index.chapter_texts, choose)
    target = next(c for c in selected["cases"] if c["setup_id"] == "setup-ch01-gurney")
    assert target["selection"]["l1"]["ok"] is False
    verified = verify_setup_payoff_selections(selected, index.chapter_texts)
    result = promote_setup_payoff_pairs(profile, verified)
    assert result["activated"] == []
    assert result["profile"]["setup_payoff"][0]["status"] == "dormant"


def test_l2_rejection_never_promotes(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    graph, index = _graph_and_index(root, profile)
    audit = build_setup_payoff_audit(profile, index, top_n=5)
    target = next(c for c in audit["cases"] if c["setup_id"] == "setup-ch01-gurney")
    chosen = next(c for c in target["candidates"] if c["chapter"] == "ch03")

    def choose(_prompt: str) -> str:
        return json.dumps({"decisions": [{
            "setup_id": "setup-ch01-gurney",
            "candidate_id": chosen["candidate_id"],
            "verdict": "payoff",
        }]})

    def reject(_prompt: str) -> str:
        return json.dumps({"decisions": [{
            "id": "setup-ch01-gurney", "verdict": "rejected",
            "reason": "generic scene, not a payoff",
        }]})

    selected = select_setup_payoff_candidates(audit, index.chapter_texts, choose)
    verified = verify_setup_payoff_selections(
        selected, index.chapter_texts, llm_generate=reject)
    result = promote_setup_payoff_pairs(profile, verified)
    assert result["activated"] == []
    assert result["profile"]["setup_payoff"][0]["status"] == "dormant"


def test_terminal_unpaired_setup_is_excluded_from_rerun(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    graph, index = _graph_and_index(root, profile)
    audit = build_setup_payoff_audit(profile, index, top_n=5)
    promoted = promote_setup_payoff_pairs(profile, audit)["profile"]
    terminal = next(n for n in promoted["setup_payoff"]
                    if n["id"] == "setup-ch07-spice")
    assert terminal["pairing_status"] == "terminal_unpaired"

    rerun = build_setup_payoff_audit(promoted, index, top_n=5)
    assert [c["setup_id"] for c in rerun["cases"]] != ["setup-ch07-spice"]
    assert all(c["setup_id"] != "setup-ch07-spice" for c in rerun["cases"])


def test_reapplying_old_audit_does_not_reactivate_resolved_pair(tmp_path: Path):
    root, profile = _make_project(tmp_path)
    graph, index = _graph_and_index(root, profile)
    audit = build_setup_payoff_audit(profile, index, top_n=5)
    target = next(c for c in audit["cases"] if c["setup_id"] == "setup-ch01-gurney")
    chosen = next(c for c in target["candidates"] if c["chapter"] == "ch03")

    def choose(_prompt: str) -> str:
        return json.dumps({"decisions": [{
            "setup_id": "setup-ch01-gurney",
            "candidate_id": chosen["candidate_id"],
            "verdict": "payoff",
        }]})

    def confirm_all(prompt: str) -> str:
        ids = []
        for raw in prompt.split('"id": "')[1:]:
            nid = raw.split('"', 1)[0]
            if nid.startswith("setup-ch01"):
                ids.append(nid)
        return json.dumps({"decisions": [
            {"id": nid, "verdict": "confirmed", "reason": "evidence supports pairing"}
            for nid in ids]})

    selected = select_setup_payoff_candidates(audit, index.chapter_texts, choose)
    verified = verify_setup_payoff_selections(
        selected, index.chapter_texts, llm_generate=confirm_all)
    first = promote_setup_payoff_pairs(profile, verified)
    assert len(first["activated"]) == 1
    before = json.dumps(first["profile"], ensure_ascii=False, sort_keys=True)

    second = promote_setup_payoff_pairs(first["profile"], verified)
    after = json.dumps(second["profile"], ensure_ascii=False, sort_keys=True)
    assert before == after
    assert second["activated"] == []
    assert "setup-ch01-gurney" in second["already_resolved"]
    assert second["existing_active"] == 1
    assert second["available_slots"] == 29


def test_setup_audit_cli_is_read_only_unless_apply_and_monolith_exports():
    from src.main import _command_needs_lock

    assert _command_needs_lock("setup-payoff-audit",
                               SimpleNamespace(apply=False)) is False
    assert _command_needs_lock("setup-payoff-audit",
                               SimpleNamespace(apply=True)) is True
    assert _command_needs_lock("story-graph-query",
                               SimpleNamespace(apply=False)) is False
    import src.translator_agent as ta
    assert callable(ta._command_needs_lock)
    assert callable(ta.build_setup_payoff_audit)
    assert callable(ta.select_setup_payoff_candidates)
    assert callable(ta.verify_setup_payoff_selections)
    assert callable(ta.promote_setup_payoff_pairs)
