"""Round 67: OpenAICompatBackend 伪概率一致性 — confidence 必须与 max(probs) 严格一致。

保护目标（修复 self-contradiction）：
  - 旧实现: ChoiceAnswer 的 confidence = LLM 自报 conf，probabilities 是伪构造（max 与 conf 不一定匹配）
  - 新实现: chosen 不在 options 时自动归入；confidence 用 Round 66 _confidence_from_probs 派生
  - ScoreAnswer 同理：均匀分布 score 的 confidence 必须等于 1/n
"""
from __future__ import annotations

import pytest

from src.judges.typesafe_backend import OpenAICompatBackend, _confidence_from_probs
from src.judges.typesafe_spec import ChoiceAnswer, ScoreAnswer


class TestOpenAICompatParseAnswersChoice:
    """OpenAICompatBackend._parse_answers Choice 路径一致性"""

    def _backend(self):
        # 注入一个 fake llm client（不被实际调用，因为 _parse_answers 是纯函数）
        return OpenAICompatBackend.__new__(OpenAICompatBackend)

    def test_choice_in_options_confidence_aligns_max_prob(self):
        """chosen 在 options 内 → max(probs) == confidence"""
        b = self._backend()
        questions = {"voice_drift": {"type": "choice", "criteria": {"none": "", "subtle": "", "severe": ""}}}
        raw = '{"answers": {"voice_drift": {"type": "choice", "choice": "severe", "confidence": 0.7}}}'
        out = b._parse_answers(raw, questions)
        ans = out["voice_drift"]
        assert isinstance(ans, ChoiceAnswer)
        # confidence 来自 max(probs)
        assert ans.confidence == _confidence_from_probs(ans.probabilities)
        assert ans.confidence == pytest.approx(0.7, abs=0.01)
        # chosen 必须占最大值
        assert ans.probabilities["severe"] == max(ans.probabilities.values())

    def test_choice_not_in_options_auto_admitted(self):
        """chosen 不在 criteria options 内 → 自动归入 + confidence 对齐"""
        b = self._backend()
        questions = {"voice_drift": {"type": "choice", "criteria": {"none": "", "subtle": ""}}}
        # LLM 偶发输出 criteria 之外的 option（如 hallucinated）
        raw = '{"answers": {"voice_drift": {"type": "choice", "choice": "extreme", "confidence": 0.6}}}'
        out = b._parse_answers(raw, questions)
        ans = out["voice_drift"]
        # chosen 必须被保留（不能丢，silent），且在 probs 里
        assert ans.choice == "extreme"
        assert "extreme" in ans.probabilities
        assert ans.confidence == _confidence_from_probs(ans.probabilities)

    def test_choice_empty_options_does_not_crash(self):
        """criteria 为空 → 不应抛 KeyError 或 IndexError"""
        b = self._backend()
        questions = {"voice_drift": {"type": "choice", "criteria": {}}}
        raw = '{"answers": {"voice_drift": {"type": "choice", "choice": "x", "confidence": 0.5}}}'
        out = b._parse_answers(raw, questions)
        ans = out["voice_drift"]
        # 应当产生单点分布
        assert ans.probabilities == {"x": 1.0} or ans.probabilities.get("x") == 1.0
        assert ans.confidence == pytest.approx(1.0, abs=0.01)


class TestOpenAICompatParseAnswersScore:
    """OpenAICompatBackend._parse_answers Score 路径一致性"""

    def _backend(self):
        return OpenAICompatBackend.__new__(OpenAICompatBackend)

    def test_uniform_score_confidence_aligns_max_prob(self):
        """均匀分布 score → confidence == max(probs) == 1/n"""
        b = self._backend()
        questions = {"style": {"type": "score", "criteria": ["low", "mid", "high"]}}
        raw = '{"answers": {"style": {"type": "score", "score": 1, "confidence": 0.8}}}'
        ans = b._parse_answers(raw, questions)["style"]
        # 旧实现: confidence=0.8（与均匀分布 1/3=0.33 矛盾）
        # 新实现: confidence=max(probs)=1/3≈0.33
        assert isinstance(ans, ScoreAnswer)
        assert ans.confidence == _confidence_from_probs(ans.probabilities)
        assert ans.confidence == pytest.approx(1.0 / 3, abs=0.01)

    def test_uniform_score_does_not_use_llm_self_reported_conf(self):
        """Score 的 confidence 不应直接来自 LLM 自报（LLM 不能评估自己分布的真实置信度）"""
        b = self._backend()
        questions = {"style": {"type": "score", "criteria": ["a", "b"]}}
        raw_high = '{"answers": {"style": {"type": "score", "score": 1, "confidence": 0.95}}}'
        raw_low = '{"answers": {"style": {"type": "score", "score": 1, "confidence": 0.1}}}'
        out_high = b._parse_answers(raw_high, questions)["style"]
        out_low = b._parse_answers(raw_low, questions)["style"]
        # 两者 probs 相同（均匀）→ confidence 必须相同
        assert out_high.confidence == out_low.confidence
        assert out_high.confidence == pytest.approx(0.5, abs=0.01)


class TestDownstreamLowConfAlignment:
    """下游 verdict.low_conf_threshold 跳过行为"""

    def test_choice_with_high_confidence_passes_low_conf_check(self):
        """ChoiceAnswer.confidence = max(probs) = 0.7 → 不被 verdict 跳过"""
        from src.judges.verdict import synthesize_verdict
        # 模拟 OpenAICompatBackend 真实输出
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.15, "subtle": 0.15, "severe": 0.7},
            confidence=_confidence_from_probs({"none": 0.15, "subtle": 0.15, "severe": 0.7}),
        )
        v = synthesize_verdict({"voice_drift": ans})
        # confidence 0.7 >= low_conf_threshold 0.5 → 不跳过
        assert "voice_drift" not in v.skipped_dimensions

    def test_uniform_choice_skipped_by_low_conf(self):
        """均匀 Choice (max=0.33) → 被 verdict 跳过"""
        from src.judges.verdict import synthesize_verdict
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.33, "subtle": 0.33, "severe": 0.34},
            confidence=_confidence_from_probs({"none": 0.33, "subtle": 0.33, "severe": 0.34}),
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert "voice_drift" in v.skipped_dimensions
