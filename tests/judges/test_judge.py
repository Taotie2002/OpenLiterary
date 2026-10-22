"""Round 49: 单元测试 - judges/judge.py (build_state + to_legacy_scores).

保护目标：
  - build_state：entity_registry 按 source_text 过滤（A4 修复契约）
  - build_state：None/空字典/无匹配 entity 都正确处理
  - to_legacy_scores：6 个 primitive → 旧 critic 6 维度分数映射
"""
from __future__ import annotations

import pytest

from src.judges.judge import build_state, to_legacy_scores, evaluate_chunk
from src.judges.typesafe_spec import CompositeVerdict, NoulAnswer, ChoiceAnswer, ScoreAnswer


# ---------------------------------------------------------------------------
# build_state
# ---------------------------------------------------------------------------

class TestBuildState:
    def test_basic_fields_present(self):
        state = build_state(
            source_text="Hello world.",
            raw_translation="你好世界。",
            literary_translation="你好，世界。",
            style_guide_stats={"avg_sentence_length": "短"},
        )
        assert state["source_text"] == "Hello world."
        assert state["raw_translation"] == "你好世界。"
        assert state["literary_translation"] == "你好，世界。"
        assert state["style_guide"] == {"avg_sentence_length": "短"}

    def test_entity_registry_none_yields_empty(self):
        state = build_state(
            source_text="anything",
            raw_translation="x",
            literary_translation="y",
            style_guide_stats={},
            entity_registry=None,
        )
        assert state["entity_registry"] == {}

    def test_entity_registry_empty_yields_empty(self):
        state = build_state(
            source_text="anything",
            raw_translation="x",
            literary_translation="y",
            style_guide_stats={},
            entity_registry={},
        )
        assert state["entity_registry"] == {}

    def test_entity_registry_filters_by_source_text(self):
        # A4 修复：仅注入 source_text 中实际出现的实体
        registry = {
            "Luffy": "鲁夫",
            "Zoro": "索隆",
            "Sanji": "山治",
        }
        state = build_state(
            source_text="Luffy and Zoro set sail.",
            raw_translation="x",
            literary_translation="y",
            style_guide_stats={},
            entity_registry=registry,
        )
        assert "Luffy" in state["entity_registry"]
        assert "Zoro" in state["entity_registry"]
        assert "Sanji" not in state["entity_registry"]

    def test_entity_registry_no_match_yields_empty(self):
        # 罕见：全新章节，registry 全部不匹配
        registry = {"Luffy": "鲁夫", "Zoro": "索隆"}
        state = build_state(
            source_text="本章完全不涉及路飞索隆。",
            raw_translation="x",
            literary_translation="y",
            style_guide_stats={},
            entity_registry=registry,
        )
        assert state["entity_registry"] == {}

    def test_entity_registry_skips_empty_source_name(self):
        registry = {"": "空名", "Luffy": "鲁夫"}
        state = build_state(
            source_text="Luffy is here.",
            raw_translation="x",
            literary_translation="y",
            style_guide_stats={},
            entity_registry=registry,
        )
        # 空 source_name 被跳过；Luffy 保留
        assert "" not in state["entity_registry"]
        assert "Luffy" in state["entity_registry"]


# ---------------------------------------------------------------------------
# to_legacy_scores
# ---------------------------------------------------------------------------

def _make_verdict(primitive_dict):
    """辅助：构造 CompositeVerdict with given primitive_summary"""
    return CompositeVerdict(
        decision="PASS",
        confidence=0.8,
        reject_reasons=[],
        primitive_summary=primitive_dict,
    )


class TestToLegacyScores:
    def test_voice_drift_none_maps_to_high(self):
        v = _make_verdict({"voice_drift": {"choice": "none", "confidence": 0.8}})
        scores = to_legacy_scores(v)
        assert scores["voice_consistency"] == 9.0

    def test_voice_drift_subtle_maps_to_mid_high(self):
        v = _make_verdict({"voice_drift": {"choice": "subtle", "confidence": 0.8}})
        scores = to_legacy_scores(v)
        assert scores["voice_consistency"] == 7.5

    def test_voice_drift_severe_maps_to_low(self):
        v = _make_verdict({"voice_drift": {"choice": "severe", "confidence": 0.8}})
        scores = to_legacy_scores(v)
        assert scores["voice_consistency"] == 4.0

    def test_voice_drift_missing_maps_to_default(self):
        v = _make_verdict({})
        scores = to_legacy_scores(v)
        assert scores["voice_consistency"] == 6.0

    def test_register_match_match_maps_to_high(self):
        v = _make_verdict({"register_match": {"choice": "match", "confidence": 0.8}})
        scores = to_legacy_scores(v)
        assert scores["style_compliance"] == 8.5

    def test_register_match_under_or_over_maps_to_low(self):
        for choice in ("under", "over"):
            v = _make_verdict({"register_match": {"choice": choice, "confidence": 0.8}})
            scores = to_legacy_scores(v)
            assert scores["style_compliance"] == 5.0

    def test_literal_residue_inverse_mapping(self):
        # noul=0.0 → 10.0；noul=1.0 → 0.0
        v = _make_verdict({"literal_residue": {"noul": 0.0}})
        assert to_legacy_scores(v)["literal_residue"] == 10.0
        v = _make_verdict({"literal_residue": {"noul": 1.0}})
        assert to_legacy_scores(v)["literal_residue"] == 0.0
        v = _make_verdict({"literal_residue": {"noul": 0.5}})
        assert to_legacy_scores(v)["literal_residue"] == 5.0

    def test_semantic_loss_inverse_mapping(self):
        v = _make_verdict({"semantic_loss": {"noul": 0.0}})
        assert to_legacy_scores(v)["semantic_preservation"] == 10.0
        v = _make_verdict({"semantic_loss": {"noul": 0.5}})
        assert to_legacy_scores(v)["semantic_preservation"] == 5.0

    def test_readability_from_style_score(self):
        # style_compliance_score=1.0 → readability=5.0
        v = _make_verdict({"style_compliance_score": {"score": 1.0}})
        assert to_legacy_scores(v)["readability"] == 5.0
        v = _make_verdict({"style_compliance_score": {"score": 2.0}})
        assert to_legacy_scores(v)["readability"] == 10.0
        v = _make_verdict({"style_compliance_score": {"score": 0.0}})
        assert to_legacy_scores(v)["readability"] == 0.0

    def test_fluency_equals_readability(self):
        v = _make_verdict({"style_compliance_score": {"score": 1.5}})
        scores = to_legacy_scores(v)
        assert scores["fluency"] == scores["readability"] == 7.5

    def test_all_six_dimensions_present(self):
        # 完整 6 primitive 输入应产出 6 维度 scores
        v = _make_verdict({
            "voice_drift": {"choice": "none"},
            "register_match": {"choice": "match"},
            "entity_consistency": {"noul": 0.0},
            "literal_residue": {"noul": 0.0},
            "semantic_loss": {"noul": 0.0},
            "style_compliance_score": {"score": 2.0},
        })
        scores = to_legacy_scores(v)
        assert set(scores.keys()) >= {
            "voice_consistency",
            "style_compliance",
            "literal_residue",
            "semantic_preservation",
            "readability",
            "fluency",
        }

    def test_missing_primitives_use_defaults(self):
        # 空 primitive_summary → 所有维度用默认（6.0 / 8.5 / 10.0 / 5.0）
        v = _make_verdict({})
        scores = to_legacy_scores(v)
        # voice_drift 缺失 → 6.0
        assert scores["voice_consistency"] == 6.0
        # register_match 缺失 → 5.0（非 match）
        assert scores["style_compliance"] == 5.0
        # literal_residue 缺失 → 10.0 (noul 默认 0)
        assert scores["literal_residue"] == 10.0
        # readability 默认 = 5.0 (score 默认 1.0 * 5)
        assert scores["readability"] == 5.0

    def test_score_rounded_to_1_decimal(self):
        # 0.33 noul * 10 = 3.3 → round 1 位
        v = _make_verdict({"literal_residue": {"noul": 0.33}})
        scores = to_legacy_scores(v)
        assert scores["literal_residue"] == 6.7  # 10 - 3.3 = 6.7
