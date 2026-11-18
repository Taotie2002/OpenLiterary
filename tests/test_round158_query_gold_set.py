"""Round 158 Phase 2: query gold-set builder and negative-case freshness."""
from __future__ import annotations

import json
from pathlib import Path

from src.artifact_freshness import eval_freshness
from src.query_gold_set import build_query_gold_set


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(tmp_path: Path, *, setup_status: str = "active"):
    ch01 = "Hello world.\n\nSetup clue here."
    ch02 = "Payoff arrives here."
    _write(tmp_path / "input" / "ch01.md", ch01)
    _write(tmp_path / "input" / "ch02.md", ch02)
    setup_ev1 = ch01.index("Setup clue here.")
    setup_ev2 = ch02.index("Payoff arrives here.")
    voice_ev = ch01.index("Hello world.")
    arc_ev = ch01.index("Hello world.")
    profile = {
        "schema_version": 2,
        "source": "known_work",
        "work_guess": {"title": "Fixture"},
        "voice_notes": [{
            "id": "voice-1", "type": "voice", "chapters": ["ch01"],
            "claim": "叙述语气克制。", "confidence": 0.9,
            "status": "active", "l2_verdict": "confirmed",
            "evidence": [{"chapter": "ch01", "quote": "Hello world.",
                          "offset": voice_ev, "match_kind": "exact"}],
        }],
        "narrative_arc": [{
            "id": "arc-1", "type": "arc", "chapters": ["ch01"],
            "claim": "主角从旁观转为行动。", "confidence": 0.9,
            "status": "active", "l2_verdict": "confirmed",
            "evidence": [{"chapter": "ch01", "quote": "Hello world.",
                          "offset": arc_ev, "match_kind": "exact"}],
        }],
        "setup_payoff": [{
            "id": "setup-1", "type": "setup_payoff",
            "chapters": ["ch01", "ch02"],
            "claim": "线索在开头埋下并在后文回收。", "confidence": 0.9,
            "status": setup_status, "l2_verdict": "confirmed",
            "evidence": [
                {"chapter": "ch01", "quote": "Setup clue here.",
                 "offset": setup_ev1, "match_kind": "exact"},
                {"chapter": "ch02", "quote": "Payoff arrives here.",
                 "offset": setup_ev2, "match_kind": "exact"},
            ],
        }],
    }
    graph = {"fingerprint": "graph-fixture", "schema_version": 5}
    return profile, graph


def test_builder_emits_fresh_lineage_bound_eval(tmp_path: Path):
    profile, graph = _fixture(tmp_path)
    eval_path = tmp_path / "docs" / "eval.json"

    result = build_query_gold_set(
        tmp_path, profile=profile, graph=graph,
        output_path=eval_path, report_path=tmp_path / "output" / "report.json",
        min_setup=1, min_arc=1, min_voice=1, min_negative=1, max_cases=10,
    )

    assert result["ok"] is True
    assert result["status"] == "fresh"
    assert result["eval_written"] is True
    assert result["case_count"] == 4
    assert result["counts"] == {
        "setup_payoff": 1, "arc": 1, "voice": 1, "negative": 1}
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["profile_notes_hash"] == result["profile_notes_hash"]
    assert payload["chapter_manifest"] == result["chapter_manifest"]
    assert payload["graph_fingerprint"] == "graph-fixture"
    assert payload["counts"] == result["counts"]
    setup_case = next(c for c in payload["cases"] if c["type"] == "setup_payoff")
    assert {g["chapter"] for g in setup_case["gold"]} == {"ch01", "ch02"}
    negative = next(c for c in payload["cases"] if c["type"] == "negative")
    assert "note_id" not in negative
    assert negative["answerable"] is False

    freshness = eval_freshness(payload, tmp_path, profile, graph=graph)
    assert freshness["ok"] is True
    assert freshness["status"] == "fresh"
    assert freshness["details"]["stale_notes"] == 0


def test_insufficient_active_setup_fails_closed_and_writes_no_eval(tmp_path: Path):
    profile, graph = _fixture(tmp_path)
    eval_path = tmp_path / "docs" / "eval.json"

    result = build_query_gold_set(
        tmp_path, profile=profile, graph=graph,
        output_path=eval_path, report_path=tmp_path / "output" / "report.json",
        min_setup=2, min_arc=1, min_voice=1, min_negative=1, max_cases=10,
    )

    assert result["ok"] is False
    assert result["status"] == "insufficient_case_mix"
    assert result["eval_written"] is False
    assert result["gaps"] == {"setup_payoff": 1}
    assert eval_path.exists() is False
    assert (tmp_path / "output" / "report.json").exists()


def test_partial_draft_is_never_reported_as_success(tmp_path: Path):
    profile, graph = _fixture(tmp_path)
    eval_path = tmp_path / "docs" / "eval.json"

    result = build_query_gold_set(
        tmp_path, profile=profile, graph=graph,
        output_path=eval_path, report_path=tmp_path / "output" / "report.json",
        min_setup=2, min_arc=1, min_voice=1, min_negative=1,
        max_cases=10, write_partial=True,
    )

    assert result["ok"] is False
    assert result["status"] == "partial_draft"
    assert result["eval_written"] is True
    assert eval_path.exists()


def test_dormant_setup_is_not_counted(tmp_path: Path):
    profile, graph = _fixture(tmp_path, setup_status="dormant")

    result = build_query_gold_set(
        tmp_path, profile=profile, graph=graph,
        output_path=tmp_path / "docs" / "eval.json",
        report_path=tmp_path / "output" / "report.json",
        min_setup=1, min_arc=1, min_voice=1, min_negative=1, max_cases=10,
    )

    assert result["ok"] is False
    assert result["available"]["setup_payoff"] == 0
    assert result["gaps"]["setup_payoff"] == 1


def test_negative_case_without_note_id_passes_freshness(tmp_path: Path):
    profile, graph = _fixture(tmp_path)
    result = build_query_gold_set(
        tmp_path, profile=profile, graph=graph,
        output_path=tmp_path / "docs" / "eval.json",
        report_path=tmp_path / "output" / "report.json",
        min_setup=1, min_arc=1, min_voice=1, min_negative=1, max_cases=10,
    )
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))
    negatives = [c for c in payload["cases"] if c["type"] == "negative"]
    payload["cases"] = negatives
    freshness = eval_freshness(payload, tmp_path, profile, graph=graph)
    assert freshness["ok"] is True
    assert freshness["details"]["invalid_cases"] == 0

