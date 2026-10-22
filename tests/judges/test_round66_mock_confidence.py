"""Round 66: MockBackend confidence 透明化 —— 从概率分布导出，而非硬编码。

保护目标：
  - 旧实现：ChoiceAnswer/ScoreAnswer 硬编码 confidence=0.3/0.4/0.5，与 probabilities 脱节
  - 新实现：confidence = max(probabilities.values())，与 SDK 行为一致
  - 下游 verdict.low_conf_threshold=0.5 的跳过语义更诚实

为什么不动 NoulAnswer：
  - NoulAnswer.confidence 字段仅作向后兼容（Noul SDK schema 无 confidence）
  - NoulAnswer.to_dict() 已用 effective_conf = max(noul, 1-noul) 覆盖（Round 60）
  - verdict.py:107 明确 Noul 不参与 low_conf_threshold 门控
"""
from __future__ import annotations

import pytest

from src.judges.typesafe_backend import (
    MockBackend,
    _confidence_from_probs,
)


class TestConfidenceFromProbs:
    """_confidence_from_probs helper 数学正确性"""

    def test_peaked_distribution_high_confidence(self):
        """单峰分布 → 高 confidence"""
        probs = {"severe": 0.7, "subtle": 0.2, "none": 0.1}
        assert _confidence_from_probs(probs) == 0.7

    def test_uniform_distribution_low_confidence(self):
        """均匀分布 → 低 confidence（≈ 1/K）"""
        probs = {"a": 0.33, "b": 0.33, "c": 0.34}
        result = _confidence_from_probs(probs)
        assert 0.33 <= result <= 0.34

    def test_empty_probs_returns_zero(self):
        """空分布 → 0.0（避免下游 NoneType 异常）"""
        assert _confidence_from_probs({}) == 0.0

    def test_single_value(self):
        """只有一个选项 → confidence=1.0"""
        assert _confidence_from_probs({"only": 1.0}) == 1.0

    def test_handles_int_keys(self):
        """ScoreAnswer 的 probabilities key 是 int"""
        probs = {0: 0.1, 1: 0.3, 2: 0.6}
        assert _confidence_from_probs(probs) == 0.6


class TestMockBackendConfidenceTransparent:
    """MockBackend 返回的 confidence 必须与 probabilities 的 max 一致"""

    def setup_method(self):
        self.backend = MockBackend()

    def _eval(self, src="", raw="", lit=""):
        return self.backend.evaluate(
            state={"source_text": src, "raw_translation": raw, "literary_translation": lit},
            questions={"voice_drift": {"type": "choice", "options": ["none", "subtle", "severe"]},
                       "register_match": {"type": "choice", "options": ["match", "under", "over"]},
                       "style_compliance_score": {"type": "score", "levels": 3}},
        )

    def test_voice_drift_severe_confidence_matches_probs(self):
        """voice_drift=severe 的 probabilities max=0.7 → confidence=0.7"""
        # length_ratio < 0.3 → severe
        ans = self._eval(raw="x" * 1000, lit="y" * 100)["voice_drift"]
        assert ans.choice == "severe"
        assert ans.confidence == max(ans.probabilities.values())
        assert ans.confidence == pytest.approx(0.7)

    def test_voice_drift_subtle_confidence_matches_probs(self):
        """voice_drift=subtle 的 probabilities max=0.5 → confidence=0.5"""
        ans = self._eval(raw="x" * 100, lit="y" * 200)["voice_drift"]
        assert ans.choice == "subtle"
        assert ans.confidence == max(ans.probabilities.values())
        assert ans.confidence == pytest.approx(0.5)

    def test_voice_drift_none_confidence_matches_probs(self):
        """voice_drift=none 的 probabilities max=0.6 → confidence=0.6"""
        ans = self._eval(raw="x" * 100, lit="y" * 100)["voice_drift"]
        assert ans.choice == "none"
        assert ans.confidence == max(ans.probabilities.values())
        assert ans.confidence == pytest.approx(0.6)

    def test_register_match_over_confidence_matches_probs(self):
        """register_match=over max=0.5 → confidence=0.5"""
        ans = self._eval(lit="于是乎之")["register_match"]
        assert ans.choice == "over"
        assert ans.confidence == max(ans.probabilities.values())
        assert ans.confidence == pytest.approx(0.5)

    def test_register_match_match_confidence_matches_probs(self):
        """register_match=match max=0.7 → confidence=0.7"""
        ans = self._eval(lit="普通叙述")["register_match"]
        assert ans.choice == "match"
        assert ans.confidence == max(ans.probabilities.values())
        assert ans.confidence == pytest.approx(0.7)

    def test_no_hardcoded_constant_confidence(self):
        """不应再有任何 confidence=0.3 / 0.4 / 0.5 硬编码（仅 Noul 可保留向后兼容默认值）"""
        # 所有 ChoiceAnswer/ScoreAnswer 的 confidence 都应从 probs 推导
        result = self._eval(raw="x" * 100, lit="y" * 100)
        for qid, ans in result.items():
            if hasattr(ans, "probabilities") and ans.probabilities:
                # confidence 必须等于 probs 最大值
                assert ans.confidence == max(ans.probabilities.values()), (
                    f"{qid}: confidence={ans.confidence} != max(probs)={max(ans.probabilities.values())}"
                )


class TestMockBackendDownstreamLowConf:
    """MockBackend confidence 透明化后，verdict 的 low_conf_threshold 跳过行为"""

    def test_peaked_choice_not_skipped_by_verdict(self):
        """单峰 Choice (conf>0.5) → verdict 不跳过"""
        from src.judges.verdict import synthesize_verdict
        from src.judges.typesafe_spec import ChoiceAnswer

        # high_conf_peaked Choice
        answers = {
            "voice_drift": ChoiceAnswer(
                choice="severe",
                probabilities={"none": 0.1, "subtle": 0.2, "severe": 0.7},
                confidence=0.7,
            ),
        }
        v = synthesize_verdict(answers)
        # confidence 0.7 >= low_conf_threshold 0.5 → 不应被跳过
        assert "voice_drift" not in v.skipped_dimensions

    def test_uniform_choice_skipped_by_verdict(self):
        """均匀 Choice (conf<0.5) → verdict 跳过（low_conf）"""
        from src.judges.verdict import synthesize_verdict
        from src.judges.typesafe_spec import ChoiceAnswer

        # uniform Choice
        answers = {
            "voice_drift": ChoiceAnswer(
                choice="severe",
                probabilities={"none": 0.33, "subtle": 0.33, "severe": 0.34},
                confidence=0.34,
            ),
        }
        v = synthesize_verdict(answers)
        assert "voice_drift" in v.skipped_dimensions
        assert any("low_conf" in r for r in v.skipped_reasons.values())
