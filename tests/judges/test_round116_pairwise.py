"""Round 116: Jev 配对仲裁原型单测（纯逻辑，不调网络）。"""
from __future__ import annotations

from src.judges.pairwise import arbitrate_answers
from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer


def _answers(entity, semantic, register_choice, register_conf, style_score, style_conf):
    return {
        "entity_consistency": NoulAnswer(noul=entity),
        "semantic_loss": NoulAnswer(noul=semantic),
        "register_match": ChoiceAnswer(
            choice=register_choice,
            probabilities=({"under": 0.05, "match": 0.9, "over": 0.05} if register_choice == "match"
                           else {"under": 0.1, "match": 0.3, "over": 0.6}),
            confidence=register_conf,
        ),
        "style_compliance_score": ScoreAnswer(
            score=style_score,
            probabilities=({0: 0.05, 1: 0.15, 2: 0.8} if style_score >= 1.5
                           else {0: 0.6, 1: 0.3, 2: 0.1}),
            confidence=style_conf,
        ),
    }


class TestArbitrate:
    def test_clear_winner_a(self):
        a = _answers(0.05, 0.1, "match", 0.9, 2.0, 0.8)
        b = _answers(0.8, 0.7, "over", 0.7, 0.0, 0.6)
        res = arbitrate_answers(a, b)
        assert res["winner"] == "A"
        assert res["votes"]["A"] == 4
        assert res["dimensions"]["entity_consistency"]["winner"] == "A"

    def test_low_confidence_skipped(self):
        a = _answers(0.05, 0.1, "match", 0.9, 2.0, 0.8)
        b = _answers(0.8, 0.7, "over", 0.7, 0.0, 0.6)
        b["semantic_loss"] = NoulAnswer(noul=0.9, confidence=0.3)
        res = arbitrate_answers(a, b, min_confidence=0.5)
        assert res["dimensions"]["semantic_loss"]["status"] == "skipped_low_confidence"
        assert res["skipped_reasons"]["semantic_loss"].startswith("low_confidence")
        assert res["votes"]["A"] == 3  # 4 个可投维度中 semantic_loss 被跳过
        assert res["winner"] == "A"

    def test_tie_when_margins_small(self):
        a = _answers(0.30, 0.30, "match", 0.9, 1.5, 0.8)
        b = _answers(0.32, 0.31, "match", 0.9, 1.52, 0.8)
        res = arbitrate_answers(a, b, tie_epsilon=0.05)
        assert res["winner"] == "tie"
        assert res["votes"]["tie"] == 4

    def test_dict_serialized_answers_supported(self):
        a = {"entity_consistency": {"type": "noul", "noul": 0.1, "confidence": 0.9}}
        b = {"entity_consistency": {"type": "noul", "noul": 0.6, "confidence": 0.9}}
        res = arbitrate_answers(a, b)
        assert res["winner"] == "A"
        assert res["dimensions"]["entity_consistency"]["value_a"] == 0.1

    def test_close_dimension_flagged(self):
        a = _answers(0.10, 0.10, "match", 0.9, 2.0, 0.8)
        b = _answers(0.22, 0.7, "over", 0.7, 0.0, 0.6)
        res = arbitrate_answers(a, b, close_margin=0.15)
        assert "entity_consistency" in res["close_dimensions"]
