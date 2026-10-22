"""Round 47+60: 单元测试 - judges/verdict.py (synthesize_verdict).

保护目标（typesafe verdict 合成）：
  - 6 个 primitives 各自的 reject 阈值（边界值 + 组合）
  - Noul 答案不参与 low_confidence 门控（SDK schema 限制）
  - Choice 答案 < low_confidence_threshold 时被跳过
  - Round 60: Noul uncertain zone [noul_uncertain_low, noul_uncertain_high]
    内的 Noul 不触发 reject（typesafe-ai consistency_noul_cookbook 推荐）
  - thresholds 参数可覆盖 DEFAULT_THRESHOLDS
  - overall_conf = min(confidences) — 最低把握维度主导
  - reject_reasons 反映所有违例维度
"""
from __future__ import annotations

import pytest

from src.judges.verdict import synthesize_verdict, DEFAULT_THRESHOLDS
from src.judges.typesafe_spec import (
    NoulAnswer, ChoiceAnswer, ScoreAnswer, CompositeVerdict,
)


# ---------------------------------------------------------------------------
# 基本合成
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictBasic:
    def test_empty_answers_returns_pass(self):
        v = synthesize_verdict({})
        assert v.decision == "PASS"
        assert v.confidence == 0.0
        assert v.reject_reasons == []

    def test_single_noul_below_uncertain_low_passes(self):
        # entity_consistency noul=0.1 < uncertain_low=0.3 → confident "no" → PASS
        answers = {"entity_consistency": NoulAnswer(noul=0.1)}
        v = synthesize_verdict(answers)
        assert v.decision == "PASS"
        assert v.reject_reasons == []

    def test_single_noul_above_uncertain_high_rejects(self):
        # entity_consistency noul=0.8 > uncertain_high=0.7 → confident "yes" → REJECT
        answers = {"entity_consistency": NoulAnswer(noul=0.8)}
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"
        assert any("实体一致性违反" in r for r in v.reject_reasons)


# ---------------------------------------------------------------------------
# 6 primitives 阈值
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictPrimitives:
    def test_voice_drift_severe_rejects(self):
        # severe prob >= 0.5 → REJECT
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.2, "subtle": 0.2, "severe": 0.6},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "REJECT"
        assert any("音色漂移（severe）" in r for r in v.reject_reasons)

    def test_voice_drift_severe_at_boundary_rejects(self):
        # severe prob == 0.5 → REJECT (>=)
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.0, "subtle": 0.5, "severe": 0.5},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "REJECT"

    def test_voice_drift_only_subtle_passes(self):
        # subtle 不触发 — 必须叠加其他维度
        ans = ChoiceAnswer(
            choice="subtle",
            probabilities={"none": 0.2, "subtle": 0.7, "severe": 0.1},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "PASS"

    def test_register_match_combined_rejects(self):
        # under + over >= 0.4 → REJECT (Round 90: 输出拆为具体类别, 不再是合并 语域失配)
        ans = ChoiceAnswer(
            choice="over",
            probabilities={"under": 0.2, "over": 0.25, "match": 0.55},
            confidence=0.8,
        )
        v = synthesize_verdict({"register_match": ans})
        assert v.decision == "REJECT"
        # Round 90: combined=0.45 >= 0.4 触发 reject, 拆分输出 语域偏离（over）
        # (over=0.25 >= dir_thr=0.2, under=0.20 == dir_thr 也达阈; 两个都输出)
        joined = "; ".join(v.reject_reasons)
        assert "语域失配" not in joined, (
            "Round 90 修复后不再输出合并标签 语域失配, 应拆为具体类别. "
            f"reasons={v.reject_reasons!r}"
        )
        assert "语域偏离" in joined, (
            f"拆分后类别应含 '语域偏离', 实际: {joined!r}"
        )

    def test_register_match_clean_passes(self):
        ans = ChoiceAnswer(
            choice="match",
            probabilities={"under": 0.1, "over": 0.1, "match": 0.8},
            confidence=0.8,
        )
        v = synthesize_verdict({"register_match": ans})
        assert v.decision == "PASS"

    def test_literal_residue_below_uncertain_passes(self):
        # noul=0.1 < uncertain_low=0.3 → confident "no" → PASS
        v = synthesize_verdict({"literal_residue": NoulAnswer(noul=0.1)})
        assert v.decision == "PASS"

    def test_literal_residue_in_uncertain_zone_passes(self):
        # noul=0.5 ∈ [0.3, 0.7] → uncertain → 不参与 reject → PASS
        v = synthesize_verdict({"literal_residue": NoulAnswer(noul=0.5)})
        assert v.decision == "PASS"

    def test_literal_residue_above_uncertain_rejects(self):
        # noul=0.8 > uncertain_high=0.7 → confident "yes" → REJECT
        v = synthesize_verdict({"literal_residue": NoulAnswer(noul=0.8)})
        assert v.decision == "REJECT"
        assert any("直译残留" in r for r in v.reject_reasons)

    def test_semantic_loss_in_uncertain_zone_passes(self):
        # noul=0.4 ∈ [0.3, 0.7] → uncertain → PASS
        v = synthesize_verdict({"semantic_loss": NoulAnswer(noul=0.4)})
        assert v.decision == "PASS"

    def test_semantic_loss_above_uncertain_rejects(self):
        # noul=0.75 > uncertain_high → REJECT
        v = synthesize_verdict({"semantic_loss": NoulAnswer(noul=0.75)})
        assert v.decision == "REJECT"

    def test_style_compliance_low_score_rejects(self):
        # score < 1.0 且 confidence >= 0.6 → REJECT
        ans = ScoreAnswer(score=0.5, probabilities={0: 0.3, 1: 0.5, 2: 0.2}, confidence=0.7)
        v = synthesize_verdict({"style_compliance_score": ans})
        assert v.decision == "REJECT"
        assert any("风格契合度低" in r for r in v.reject_reasons)

    def test_style_compliance_perfect_score_passes(self):
        ans = ScoreAnswer(score=1.0, probabilities={0: 0.0, 1: 0.0, 2: 1.0}, confidence=0.7)
        v = synthesize_verdict({"style_compliance_score": ans})
        assert v.decision == "PASS"


# ---------------------------------------------------------------------------
# 低置信度门控
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictLowConfidence:
    def test_choice_low_confidence_skipped(self):
        # Choice conf < 0.5 → skip（不 reject）
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.0, "subtle": 0.0, "severe": 1.0},
            confidence=0.3,  # < 0.5 → skip
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "PASS"

    def test_choice_high_confidence_triggers(self):
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.0, "subtle": 0.0, "severe": 1.0},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "REJECT"

    def test_noul_above_uncertain_still_rejects(self):
        # Noul 不参与 low_conf 门控（SDK 限制），但 threshold 判断照常进行
        # noul=0.8 > uncertain_high=0.7 → REJECT（即使 confidence=0）
        ans = NoulAnswer(noul=0.8, confidence=0.0)
        v = synthesize_verdict({"entity_consistency": ans})
        assert v.decision == "REJECT"

    def test_style_low_confidence_passes(self):
        # score < 1.0 但 confidence < 0.6 → skip（不 reject）
        ans = ScoreAnswer(score=0.3, probabilities={0: 0.7, 1: 0.3}, confidence=0.4)
        v = synthesize_verdict({"style_compliance_score": ans})
        assert v.decision == "PASS"


# ---------------------------------------------------------------------------
# 综合置信度
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictOverallConf:
    def test_overall_conf_is_min_of_confidences(self):
        # 3 个维度 confidence 分别为 0.8, 0.6, 0.7 → overall_conf = 0.6
        answers = {
            "voice_drift": ChoiceAnswer(
                choice="none",
                probabilities={"none": 1.0},
                confidence=0.8,
            ),
            "register_match": ChoiceAnswer(
                choice="match",
                probabilities={"match": 1.0},
                confidence=0.6,
            ),
            "literal_residue": NoulAnswer(noul=0.2, confidence=0.0),  # max(0.2, 0.8) = 0.8
        }
        v = synthesize_verdict(answers)
        assert v.confidence == pytest.approx(0.6, abs=0.01)

    def test_overall_conf_uses_max_of_noul_distance(self):
        # Noul confidence 用 max(noul, 1-noul) 聚合
        # noul=0.3 → 1-0.3 = 0.7 作为 conf
        answers = {
            "literal_residue": NoulAnswer(noul=0.3, confidence=0.0),
        }
        v = synthesize_verdict(answers)
        assert v.confidence == pytest.approx(0.7, abs=0.01)


# ---------------------------------------------------------------------------
# Round 60: Noul uncertain zone 边界测试
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictUncertainZone:
    """Round 60: 验证 Noul uncertain zone 行为
    参考 typesafe-ai/docs/cookbooks/consistency_noul_cookbook.md
    """

    def test_uncertain_low_inclusive(self):
        # noul 恰好等于 uncertain_low=0.3 → 在区间内 → PASS
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.3)})
        assert v.decision == "PASS"

    def test_uncertain_high_inclusive(self):
        # noul 恰好等于 uncertain_high=0.7 → 在区间内 → PASS
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.7)})
        assert v.decision == "PASS"

    def test_just_below_uncertain_low_rejects_when_threshold_low(self):
        # 紧贴 uncertain_low 下界：noul=0.29 → 区间外下侧
        # 加上 entity_consistency_noul=0.3 threshold → 0.29 < 0.3 → PASS
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.29)})
        assert v.decision == "PASS"

    def test_just_above_uncertain_high_rejects(self):
        # noul=0.71 → 区间外上侧 → entity_consistency_noul=0.3 → REJECT
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.71)})
        assert v.decision == "REJECT"

    def test_ch32_scenario_entity_consistency_uncertain_passes(self):
        """ch32 实测场景：noul=0.39 entity_consistency 落在区间内 → 不触发 reject
        旧行为：REJECT → rewriter 死循环
        新行为：PASS（由 local _check_entity_consistency 单独负责 entity 校对）
        """
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.39)})
        assert v.decision == "PASS"

    def test_ch32_scenario_semantic_loss_uncertain_passes(self):
        """ch32 实测场景：noul=0.59 semantic_loss 落在区间内 → 不触发 reject"""
        v = synthesize_verdict({"semantic_loss": NoulAnswer(noul=0.59)})
        assert v.decision == "PASS"

    def test_ch32_scenario_literal_residue_above_rejects(self):
        """ch32 实测场景：noul=0.71 literal_residue 在区间外上侧 → REJECT（单维度焦点）"""
        v = synthesize_verdict({"literal_residue": NoulAnswer(noul=0.71)})
        assert v.decision == "REJECT"
        assert any("直译残留" in r for r in v.reject_reasons)

    def test_uncertain_zone_can_be_disabled(self):
        """把 uncertain zone 设到 [0, 1] → 等于禁用区间。noul=0.5 → 区间内 → PASS
        把 uncertain zone 设到 [0.9, 1.1] → noul=0.5 → 区间外下侧 → 由 threshold 决定"""
        # 禁用区间：[0, 1] 覆盖整个 [0,1]，所有 noul 都"uncertain"
        answers = {"entity_consistency": NoulAnswer(noul=0.5)}
        v_all_uncertain = synthesize_verdict(
            answers,
            thresholds={"noul_uncertain_low": 0.0, "noul_uncertain_high": 1.0},
        )
        assert v_all_uncertain.decision == "PASS"

    def test_uncertain_zone_can_be_tightened(self):
        """收紧区间：[0.45, 0.55]：noul=0.5 在区间内（PASS），noul=0.4 在区间外下侧（按 threshold 走）"""
        # noul=0.4 不在区间 → 0.4 < entity_consistency_noul=0.3? 0.4 > 0.3 → REJECT
        v = synthesize_verdict(
            {"entity_consistency": NoulAnswer(noul=0.4)},
            thresholds={"noul_uncertain_low": 0.45, "noul_uncertain_high": 0.55},
        )
        assert v.decision == "REJECT"

    def test_summary_still_records_uncertain_noul(self):
        """即使 uncertain 区间内 Noul 不触发 reject，primitive_summary 仍记录完整信息"""
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.5)})
        # 决策 PASS，但 summary 里应有 entity_consistency
        assert v.decision == "PASS"
        assert "entity_consistency" in v.primitive_summary
        assert v.primitive_summary["entity_consistency"]["noul"] == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# 多违例累积
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictMultipleViolations:
    def test_only_above_uncertain_zone_triggers(self):
        # noul=0.8/0.85/0.75 → 全部在区间外上侧 → 3 个 reject reasons
        answers = {
            "entity_consistency": NoulAnswer(noul=0.8),
            "literal_residue": NoulAnswer(noul=0.85),
            "semantic_loss": NoulAnswer(noul=0.75),
        }
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"
        assert len(v.reject_reasons) == 3

    def test_mixed_uncertain_and_clear(self):
        # noul=0.8 (above zone → REJECT), noul=0.5 (in zone → skip), noul=0.6 (in zone → skip)
        # → 1 个 reject reason
        answers = {
            "entity_consistency": NoulAnswer(noul=0.8),
            "literal_residue": NoulAnswer(noul=0.5),
            "semantic_loss": NoulAnswer(noul=0.6),
        }
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"
        assert len(v.reject_reasons) == 1

    def test_primitive_summary_includes_all(self):
        answers = {
            "entity_consistency": NoulAnswer(noul=0.5),
            "voice_drift": ChoiceAnswer(
                choice="none",
                probabilities={"none": 1.0},
                confidence=0.8,
            ),
        }
        v = synthesize_verdict(answers)
        assert "entity_consistency" in v.primitive_summary
        assert "voice_drift" in v.primitive_summary


# ---------------------------------------------------------------------------
# thresholds 覆盖
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictThresholds:
    def test_custom_threshold_relaxes_entity_consistency(self):
        # noul=0.5 在 uncertain zone → uncertain 覆盖 threshold → PASS
        # 即使 custom threshold=0.9，因为 uncertain zone [0.3, 0.7] 仍生效
        answers = {"entity_consistency": NoulAnswer(noul=0.5)}
        v_relaxed = synthesize_verdict(
            answers,
            thresholds={"entity_consistency_noul": 0.9},
        )
        assert v_relaxed.decision == "PASS"

    def test_custom_threshold_can_break_uncertain_zone(self):
        # 把 uncertain zone 缩到 [0.45, 0.55]：noul=0.5 在区间内 → PASS
        # 把 uncertain zone 扩到 [0.0, 0.4]：noul=0.5 → 区间外上侧 → REJECT（threshold=0.3）
        answers = {"entity_consistency": NoulAnswer(noul=0.5)}
        v = synthesize_verdict(
            answers,
            thresholds={"noul_uncertain_low": 0.0, "noul_uncertain_high": 0.4},
        )
        assert v.decision == "REJECT"

    def test_custom_threshold_tightens_literal_residue(self):
        # 把 literal_residue uncertain zone 缩到 [0.3, 0.4]：noul=0.5 在区间外上侧
        # → 按 literal_residue_noul=0.5（默认）→ REJECT
        answers = {"literal_residue": NoulAnswer(noul=0.5)}
        v = synthesize_verdict(
            answers,
            thresholds={
                "literal_residue_noul": 0.5,
                "noul_uncertain_low": 0.3,
                "noul_uncertain_high": 0.4,
            },
        )
        assert v.decision == "REJECT"

    def test_unknown_threshold_key_ignored(self):
        # 未知 key 不影响 verdict
        answers = {"entity_consistency": NoulAnswer(noul=0.8)}
        v = synthesize_verdict(answers, thresholds={"unknown_key_xyz": 0.99})
        assert v.decision == "REJECT"


# ---------------------------------------------------------------------------
# DEFAULT_THRESHOLDS 完整性
# ---------------------------------------------------------------------------

class TestDefaultThresholds:
    def test_all_required_keys_present(self):
        required = [
            "voice_drift_severe_prob",
            "voice_drift_subtle_prob",
            "register_mismatch_prob",
            "entity_consistency_noul",
            "literal_residue_noul",
            "semantic_loss_noul",
            "style_compliance_min_score",
            "style_compliance_min_confidence",
            "low_confidence_threshold",
            # Round 60: 新增 Noul uncertain zone keys
            "noul_uncertain_low",
            "noul_uncertain_high",
        ]
        for k in required:
            assert k in DEFAULT_THRESHOLDS, f"missing key: {k}"

    def test_all_values_in_valid_range(self):
        # 标量阈值应是 0..1 浮点; dict 类型阈值 (Round 72 noul_uncertain_zones) 单独验证
        for k, v in DEFAULT_THRESHOLDS.items():
            if isinstance(v, dict):
                # Round 72: 嵌套 dict, 单独验证每个 zone 是 [low, high] 数字对
                for qid, zone in v.items():
                    assert isinstance(zone, (list, tuple)) and len(zone) == 2,                         f"{k}.{qid} not a (low, high) pair"
                    low, high = zone
                    assert isinstance(low, (int, float)) and isinstance(high, (int, float))
                    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
                    assert low < high, f"{k}.{qid}: low >= high"
                continue
            assert isinstance(v, (int, float)), f"{k} not numeric"
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_uncertain_zone_is_well_formed(self):
        # Round 60: uncertain_low < uncertain_high
        assert DEFAULT_THRESHOLDS["noul_uncertain_low"] < DEFAULT_THRESHOLDS["noul_uncertain_high"]
        # 且与 cookbook 一致：[0.3, 0.7]
        assert DEFAULT_THRESHOLDS["noul_uncertain_low"] == pytest.approx(0.3, abs=0.01)
        assert DEFAULT_THRESHOLDS["noul_uncertain_high"] == pytest.approx(0.7, abs=0.01)


# ---------------------------------------------------------------------------
# Round 60: NoulAnswer.to_dict() 透传 effective confidence
# ---------------------------------------------------------------------------

class TestNoulAnswerEffectiveConfidence:
    """Round 60: typesafe_sdk 不提供 Noul confidence，本地 to_dict 计算 effective
    max(noul, 1-noul)，让下游 consumer 看到校准信号。
    """

    def test_noul_high_eff_conf_is_noul(self):
        # noul=0.8 → eff_conf = max(0.8, 0.2) = 0.8
        a = NoulAnswer(noul=0.8, confidence=0.0)
        d = a.to_dict()
        assert d["noul"] == pytest.approx(0.8, abs=0.01)
        assert d["confidence"] == pytest.approx(0.8, abs=0.01)

    def test_noul_low_eff_conf_is_complement(self):
        # noul=0.2 → eff_conf = max(0.2, 0.8) = 0.8
        a = NoulAnswer(noul=0.2, confidence=0.0)
        d = a.to_dict()
        assert d["noul"] == pytest.approx(0.2, abs=0.01)
        assert d["confidence"] == pytest.approx(0.8, abs=0.01)

    def test_noul_midpoint_eff_conf_is_half(self):
        # noul=0.5 → eff_conf = max(0.5, 0.5) = 0.5（真正不确定）
        a = NoulAnswer(noul=0.5, confidence=0.0)
        d = a.to_dict()
        assert d["noul"] == pytest.approx(0.5, abs=0.01)
        assert d["confidence"] == pytest.approx(0.5, abs=0.01)

    def test_ch32_eff_conf_calibration(self):
        # ch32 实测：noul=0.39 → eff_conf = 0.61（不再显示 conf=0.00）
        a = NoulAnswer(noul=0.39, confidence=0.0)
        d = a.to_dict()
        assert d["confidence"] == pytest.approx(0.61, abs=0.01)


# ---------------------------------------------------------------------------
# Round 62: skipped_dimensions 追踪 — 让下游区分"明确 PASS"vs"不确定跳过"
# ---------------------------------------------------------------------------

class TestSkippedDimensions:
    """Round 62: verdict 显式记录被跳过的 dimension（uncertain zone / low_conf）。

    目的：让 rewriter / report 区分：
      - 明确 PASS（assert 0 violation，无 reject_reason）— 该维度已确认无问题
      - 不确定跳过（Noul ∈ [0.3, 0.7] 或 Choice conf < 0.5）— 该维度 Jev 不确定
        仍可能有问题，但未达到 reject 门槛
    """

    def test_noul_uncertain_records_skipped(self):
        # entity_consistency noul=0.5（区间内）→ 跳过 reject 但记录
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.5)})
        assert v.decision == "PASS"
        assert "entity_consistency" in v.skipped_dimensions
        assert "uncertain_noul" in v.skipped_reasons["entity_consistency"]

    def test_noul_clear_pass_does_not_record_skipped(self):
        # noul=0.0 → 明确 PASS（不在区间内）→ 不记录
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.0)})
        assert v.decision == "PASS"
        assert "entity_consistency" not in v.skipped_dimensions

    def test_choice_low_conf_records_skipped(self):
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.0, "subtle": 0.0, "severe": 1.0},
            confidence=0.3,  # < 0.5
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "PASS"
        assert "voice_drift" in v.skipped_dimensions
        assert "low_conf" in v.skipped_reasons["voice_drift"]

    def test_choice_high_conf_does_not_record_skipped(self):
        ans = ChoiceAnswer(
            choice="none",
            probabilities={"none": 1.0},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert "voice_drift" not in v.skipped_dimensions

    def test_ch32_scenario_skipped_dimensions(self):
        """ch32 实测场景：3 个 Noul 在不确定区间 → skipped_dimensions 含 3 项"""
        answers = {
            "voice_drift": ChoiceAnswer("subtle", {"none": 0.4, "subtle": 0.5, "severe": 0.1}, 0.6),
            "register_match": ChoiceAnswer("match", {"match": 0.85, "under": 0.1, "over": 0.05}, 0.85),
            "entity_consistency": NoulAnswer(noul=0.39),  # uncertain
            "literal_residue": NoulAnswer(noul=0.71),     # above zone → reject
            "semantic_loss": NoulAnswer(noul=0.59),       # uncertain
            "style_compliance_score": ScoreAnswer(1.5, {0: 0.1, 1: 0.5, 2: 0.4}, 0.7),
        }
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"
        assert "entity_consistency" in v.skipped_dimensions
        assert "semantic_loss" in v.skipped_dimensions
        assert "literal_residue" not in v.skipped_dimensions  # 触发了 reject，不算 skipped
        assert "voice_drift" not in v.skipped_dimensions  # conf=0.6 > 0.5
        assert len(v.skipped_dimensions) == 2

    def test_skipped_dimensions_default_empty_for_clean_pass(self):
        # 全部明确 PASS
        v = synthesize_verdict({})
        assert v.skipped_dimensions == []
        assert v.skipped_reasons == {}

    def test_skipped_dimensions_in_to_dict(self):
        v = synthesize_verdict({"entity_consistency": NoulAnswer(noul=0.5)})
        d = v.to_dict()
        assert "skipped_dimensions" in d
        assert "skipped_reasons" in d
        assert "entity_consistency" in d["skipped_dimensions"]
