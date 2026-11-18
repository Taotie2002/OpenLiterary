"""Round 158: established Decision translations are stable by default.

A chunk-local Reference Agent proposal may create a new Decision, but it must
not silently rewrite an existing one.  Otherwise the same source key can
oscillate between chapter runs and requeue completed chunks indefinitely.
"""
from __future__ import annotations

import importlib

import pytest


ENGINE_CASES = [
    ("translator_agent", "src.translator_agent"),
    ("core", "src.core.decision_engine"),
]

REFERENCE_CASES = [
    ("translator_agent", "src.translator_agent"),
    ("agents", "src.agents.reference_agent"),
]


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, **_kwargs):
        return self.response


@pytest.fixture(params=ENGINE_CASES, ids=[case[0] for case in ENGINE_CASES])
def engine_module(request):
    module_name = request.param[1]
    return importlib.import_module(module_name), request.param[0]


@pytest.fixture(params=REFERENCE_CASES, ids=[case[0] for case in REFERENCE_CASES])
def reference_module(request):
    module_name = request.param[1]
    return importlib.import_module(module_name)


def _translation(engine, source_key: str) -> str:
    row = engine.conn.execute(
        "SELECT translation FROM decision_db WHERE source_key = ?",
        (source_key,),
    ).fetchone()
    return str(row[0])


def test_chunk_local_proposal_does_not_revise_established_translation(
        engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_{mirror}.sqlite"))

    assert engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "kindjal",
        "坎贾尔匕首",
        "book authority",
        affected_chunks=None,
    )
    before = engine.count_pending_backtracks()

    assert not engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "kindjal",
        "金贾尔短剑（kindjal）",
        "chunk-local proposal",
        affected_chunks=["ch01_chunk023", "ch01_chunk055"],
    )

    assert _translation(engine, "kindjal") == "坎贾尔匕首"
    assert engine.count_pending_backtracks() == before


def test_explicit_revision_is_the_only_engine_path_that_requeues(
        engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_{mirror}.sqlite"))
    engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "kindjal",
        "坎贾尔匕首",
        "book authority",
        affected_chunks=None,
    )

    assert engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "kindjal",
        "坎贾尔短剑",
        "audited authority repair",
        affected_chunks=["ch01_chunk091"],
        allow_revision=True,
    )

    assert _translation(engine, "kindjal") == "坎贾尔短剑"
    pending = engine.conn.execute(
        "SELECT chunk_id, source_key FROM pending_backtrack"
    ).fetchall()
    assert [tuple(row) for row in pending] == [("ch01_chunk091", "kindjal")]


def test_new_source_key_still_inserts_normally(engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_{mirror}.sqlite"))

    assert engine.add_decision(
        module.DecisionLevel.TERMINOLOGY,
        "crysknife",
        "晶刃刀",
        "new terminology",
        affected_chunks=None,
    )
    assert _translation(engine, "crysknife") == "晶刃刀"


def test_article_led_alias_cannot_shadow_established_source_key(
        engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_article_alias_{mirror}.sqlite"))
    engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "Lady Jessica",
        "杰西卡夫人",
        "book authority",
        affected_chunks=["ch01_chunk001"],
    )
    pending_before = engine.count_pending_backtracks()

    assert not engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "the Lady Jessica",
        "杰西卡女士",
        "chunk-local proposal",
        affected_chunks=["ch01_chunk007"],
    )

    row = engine.conn.execute(
        "SELECT COUNT(*) FROM decision_db WHERE source_key = ?",
        ("the Lady Jessica",),
    ).fetchone()
    assert row[0] == 0
    assert engine.count_pending_backtracks() == pending_before
    canonical_id = engine.conn.execute(
        "SELECT id FROM decision_db WHERE source_key = ?",
        ("Lady Jessica",),
    ).fetchone()[0]
    impacts = engine.conn.execute(
        "SELECT chunk_id FROM decision_impact WHERE decision_id = ? "
        "ORDER BY chunk_id", (canonical_id,)
    ).fetchall()
    assert [row[0] for row in impacts] == ["ch01_chunk001", "ch01_chunk007"]


def test_article_led_alias_with_same_target_is_merged_idempotently(
        engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_article_same_{mirror}.sqlite"))
    engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "Lady Jessica",
        "杰西卡夫人",
        "book authority",
    )

    assert engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "The Lady Jessica",
        "杰西卡夫人",
        "same authority with article",
        affected_chunks=["ch02_chunk026"],
    )

    assert engine.conn.execute(
        "SELECT COUNT(*) FROM decision_db WHERE source_key = ?",
        ("The Lady Jessica",),
    ).fetchone()[0] == 0
    assert engine.count_pending_backtracks() == 0


def test_article_alias_match_keeps_case_sensitive_source_keys(
        engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_article_case_{mirror}.sqlite"))
    engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "One",
        "唯一者",
        "proper title",
    )

    assert engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "the one",
        "那一位",
        "generic phrase",
    )
    assert _translation(engine, "the one") == "那一位"


def test_article_led_capitalised_decision_is_case_sensitive(
        engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_{mirror}.sqlite"))
    engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "the One",
        "那一位",
        "messianic title",
        affected_chunks=None,
    )

    generic = engine.format_prompt_context(
        "Your mother wanted me to be the one to tell you.")
    exact = engine.format_prompt_context("You are the One.")

    assert "the One" not in generic
    assert "the One" in exact


def test_multiword_decision_source_tolerates_line_wrap(engine_module, tmp_path):
    module, mirror = engine_module
    engine = module.DecisionEngine(
        db_path=str(tmp_path / f"decision_wrap_{mirror}.sqlite"))
    engine.add_decision(
        module.DecisionLevel.TERMINOLOGY,
        "Judge of the Change",
        "变革仲裁官",
        "book authority",
        affected_chunks=None,
    )

    assert engine._decision_source_hit(
        "Judge of the Change",
        "It cannot be because he is Judge of the\nChange. That is temporary.",
    )
    prompt = engine.format_prompt_context(
        "It cannot be because he is Judge of the\nChange.")
    assert "Judge of the Change" in prompt


def test_reference_agent_cannot_rewrite_existing_decision(
        reference_module, tmp_path):
    module = reference_module
    engine = module.DecisionEngine(db_path=str(tmp_path / "reference.sqlite"))
    engine.add_decision(
        module.DecisionLevel.REFERENCE,
        "kindjal",
        "坎贾尔匕首",
        "book authority",
        affected_chunks=None,
    )

    agent = module.ReferenceAgent.__new__(module.ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch01"
    agent.role = "reference_extraction"
    agent._scheduler = None
    agent.llm = _FakeLLM(
        '{"references": [{"source_text": "kindjal", '
        '"allusion_target": "Caucasian dagger", '
        '"strategy": "LITERAL_TRANSLATION", '
        '"translated_term": "金贾尔短剑（kindjal）", '
        '"reason": "chunk-local proposal"}], "entities": []}'
    )
    agent._run_confirm_stage = lambda *_args, **_kwargs: None

    agent.process_chunk(
        "ch01_chunk023",
        "Halleck lifted a naked kindjal from the table.",
        affected_chunks=["ch01_chunk023"],
    )

    assert _translation(engine, "kindjal") == "坎贾尔匕首"
    assert engine.count_pending_backtracks() == 0


def test_reference_agent_does_not_queue_the_chunk_consuming_the_decision(
        reference_module, tmp_path):
    module = reference_module
    engine = module.DecisionEngine(db_path=str(tmp_path / "reference.sqlite"))
    agent = module.ReferenceAgent.__new__(module.ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch01"
    agent.role = "reference_extraction"
    agent._scheduler = None
    agent.llm = _FakeLLM(
        '{"references": [{"source_text": "kindjal", '
        '"allusion_target": "Caucasian dagger", '
        '"strategy": "RETAIN_AND_ANNOTATE", '
        '"translated_term": "坎贾尔匕首", '
        '"reason": "new decision"}], "entities": []}'
    )
    agent._run_confirm_stage = lambda *_args, **_kwargs: None
    agent._find_chunks_containing = lambda _source: [
        "ch01_chunk023", "ch01_chunk053"]

    agent.process_chunk(
        "ch01_chunk023",
        "Halleck lifted a naked kindjal from the table.",
        affected_chunks=["ch01_chunk023"],
    )

    assert _translation(engine, "kindjal") == "坎贾尔匕首"
    pending = engine.conn.execute(
        "SELECT chunk_id FROM pending_backtrack ORDER BY chunk_id"
    ).fetchall()
    assert [row[0] for row in pending] == ["ch01_chunk053"]

