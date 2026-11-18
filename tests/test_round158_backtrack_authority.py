"""Round 158: authority decisions must not create avoidable retranslation."""
from __future__ import annotations

import json
from pathlib import Path

from src.agents.reference_agent import ReferenceAgent
from src.config import config as _cfg
from src.core.decision_engine import DecisionEngine, DecisionLevel
from src.core.scheduler import TaskScheduler
from src.pipeline import TranslationPipeline


def _engine(tmp_path: Path) -> DecisionEngine:
    return DecisionEngine(db_path=str(tmp_path / "db" / "decision_db.sqlite"))


def _final(tmp_path: Path, chunk_id: str, text: str) -> None:
    chapter = chunk_id.split("_chunk", 1)[0]
    out = tmp_path / "output" / chapter
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{chunk_id}_final.json").write_text(
        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


def test_article_led_title_allows_sentence_initial_capital():
    assert DecisionEngine._decision_source_hit(
        "the Duke", "The Duke must know when I encompass his doom.")
    assert not DecisionEngine._decision_source_hit(
        "the Duke", "The duke's men moved quietly.")


def test_new_decision_skips_chunks_whose_final_already_has_target(tmp_path):
    engine = _engine(tmp_path)
    chunk_id = "ch01_chunk001"
    _final(tmp_path, chunk_id, "帝国必须知道这件事。")

    engine.add_decision(
        DecisionLevel.TERMINOLOGY,
        "the Imperium",
        "帝国",
        "book authority",
        affected_chunks=[chunk_id],
    )

    assert engine.count_pending_backtracks() == 0


def test_new_decision_queues_chunk_that_still_misses_target(tmp_path):
    engine = _engine(tmp_path)
    chunk_id = "ch01_chunk001"
    _final(tmp_path, chunk_id, "皇帝必须知道这件事。")

    engine.add_decision(
        DecisionLevel.TERMINOLOGY,
        "the Imperium",
        "帝国",
        "book authority",
        affected_chunks=[chunk_id],
    )

    assert engine.count_pending_backtracks() == 1


def test_soft_literal_reference_is_advisory_not_a_retranslation_order(tmp_path):
    engine = _engine(tmp_path)
    chunk_id = "ch01_chunk001"
    _final(tmp_path, chunk_id, "阴影里藏着刺客。")

    assert engine.add_decision(
        DecisionLevel.REFERENCE,
        "shadows",
        "暗影",
        "【来源】文学隐喻 | 【策略】LITERAL_TRANSLATION | 【依据】阴影意象",
        affected_chunks=[chunk_id],
    )

    assert engine.count_pending_backtracks() == 0
    prompt = engine.format_prompt_context("They hide in the shadows.")
    assert "必须译为 '暗影'" not in prompt
    assert "不得机械替换" in prompt


def test_soft_cultural_reference_is_advisory_not_a_retranslation_order(tmp_path):
    engine = _engine(tmp_path)
    chunk_id = "ch01_chunk001"
    _final(tmp_path, chunk_id, "他们全是没用的土疙瘩。")

    assert engine.add_decision(
        DecisionLevel.REFERENCE,
        "clods",
        "泥腿子",
        "【来源】阶级蔑称 | 【策略】CULTURAL_EQUIVALENT | 【依据】泥土意象的贬称",
        affected_chunks=[chunk_id],
    )

    assert engine.count_pending_backtracks() == 0
    prompt = engine.format_prompt_context("They are useless clods.")
    assert "必须译为 '泥腿子'" not in prompt
    assert "不得机械替换" in prompt


def test_context_only_alias_cannot_mint_a_new_entity(tmp_path):
    engine = _engine(tmp_path)
    engine.add_decision(
        DecisionLevel.TERMINOLOGY,
        "Judge of the Change",
        "变革仲裁官",
        "book authority",
        affected_chunks=None,
    )
    agent = ReferenceAgent.__new__(ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch02"

    agent._register_entities(
        "ch02_chunk048",
        [{
            "canonical_name": "变更法官",
            "source_names": [],
            "context_names": ["Judge of the Change"],
            "entity_type": "person",
            "confidence": "medium",
        }],
        "The Judge of the Change will decide.",
    )

    names = {row[0] for row in engine.conn.execute(
        "SELECT canonical_name FROM entity_registry")}
    assert "变更法官" not in names
    assert "变革仲裁官" not in names


def test_specific_person_title_is_not_collapsed_to_generic_title(tmp_path):
    engine = _engine(tmp_path)
    engine.add_decision(
        DecisionLevel.TERMINOLOGY,
        "The Baron",
        "男爵",
        "book authority",
        affected_chunks=None,
    )
    agent = ReferenceAgent.__new__(ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch01"

    agent._register_entities(
        "ch01_chunk023",
        [{
            "canonical_name": "哈克南男爵",
            "source_names": [],
            "context_names": ["The Baron"],
            "entity_type": "person",
            "confidence": "high",
        }],
        "The Baron watched.",
    )

    names = {row[0] for row in engine.conn.execute(
        "SELECT canonical_name FROM entity_registry")}
    assert "男爵" not in names
    assert "哈克南男爵" not in names


def test_unattested_proper_alias_cannot_mint_context_only_entity(tmp_path):
    engine = _engine(tmp_path)
    agent = ReferenceAgent.__new__(ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch01"

    agent._register_entities(
        "ch01_chunk023",
        [{
            "canonical_name": "韦林·岳医生",
            "source_names": ["Wellington"],
            "context_names": ["the doctor"],
            "entity_type": "person",
            "confidence": "high",
        }],
        "The doctor watched.",
    )

    names = {row[0] for row in engine.conn.execute(
        "SELECT canonical_name FROM entity_registry")}
    assert "韦林·岳医生" not in names


def test_version_merit_accepts_trusted_full_name_for_short_locked_form(tmp_path):
    old_merit = dict(_cfg.pipeline.get("version_merit", {}) or {})
    try:
        _cfg.pipeline["version_merit"] = {"enabled": True, "tolerance": 0.0}
        engine = _engine(tmp_path)
        for source, target in (("kindjal", "坎贾尔匕首"), ("Piter", "皮特")):
            engine.add_decision(
                DecisionLevel.TERMINOLOGY, source, target, "book authority",
                affected_chunks=None)
            engine.conn.execute(
                "UPDATE decision_db SET locked=1 WHERE source_key=?", (source,))
        engine.conn.execute(
            "INSERT INTO entity_registry "
            "(canonical_name, source_names, context_names, quarantined_names, "
            "variant_names, pronoun_gender, entity_type, first_seen_chunk, "
            "confidence, pending_review, translation_strategy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("彼得·德弗里斯", json.dumps(["Piter"]), "[]", "[]", "[]",
             "male", "person", "ch01_chunk023", "high", 0, "transliteration"))
        engine.conn.commit()

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.output_dir = tmp_path
        pipeline.chapter_id = "ch01"
        pipeline.decision_engine = engine
        pipeline.scheduler = object.__new__(TaskScheduler)
        pipeline.scheduler.get_task = lambda _cid: {
            "chunk_id": "ch01_chunk023",
            "text_content": "Piter lifted a kindjal. The Duke watched.",
        }
        chunk_id = "ch01_chunk023"
        final = tmp_path / f"{chunk_id}_final.json"
        backup = tmp_path / f"{chunk_id}_final_backup.json"
        final.write_text(json.dumps({
            "text": "彼得·德弗里斯拿起坎贾尔匕首。公爵注视着。",
        }, ensure_ascii=False), encoding="utf-8")
        backup.write_text(json.dumps({
            "text": "皮特拿起金贾尔匕首。公爵注视着。",
        }, ensure_ascii=False), encoding="utf-8")
        (tmp_path / f"{chunk_id}_critic_report.json").write_text(
            json.dumps({"scores": {"fluency": 8.0}}), encoding="utf-8")
        (tmp_path / f"{chunk_id}_critic_report_backup.json").write_text(
            json.dumps({"scores": {"fluency": 9.5}}), encoding="utf-8")

        restored = pipeline._apply_version_merit(chunk_id, backup, final)

        assert restored is False
        assert "坎贾尔匕首" in json.loads(
            final.read_text(encoding="utf-8"))["text"]
    finally:
        _cfg.pipeline["version_merit"] = old_merit
