"""Round 95: CJK confidence gating — Choice/Score 答案置信门槛默认高于英文工作负载.

typesafe.ai 官方 models.md / state.md 警告: Jev 在 CJK 上准确率较低.
为了让 CJK 边界判断不触发 hard reject (避免 noise-driven 死循环),
默认启用 CJK gating, 对 Choice/Score 答案使用 ≥0.6 的置信门槛.

可通过 thresholds.cjk_confidence_gating=False 关闭.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_modules():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        # Round 103 P6+: 动态 polluted 自动覆盖 src.* 父包 + 所有子模块,
        # 父包循环已是冗余, 不再单独遍历 parents.
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)



def _answers_with_low_conf_choice():
    """构造一组答案, 其中 ChoiceAnswer conf=0.55 (CJK 边界)."""
    from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer
    return {
        # voice_drift conf=0.55 (CJK 边界, 通用阈值 0.5 会通过, CJK 阈值 0.6 应跳过)
        "voice_drift": ChoiceAnswer("subtle", {"none": 0.4, "subtle": 0.55, "severe": 0.05}, 0.55),
        "register_match": ChoiceAnswer("match", {"match": 0.95, "under": 0.03, "over": 0.02}, 0.95),
        "entity_consistency": NoulAnswer(noul=0.05),
        "literal_residue": NoulAnswer(noul=0.10),
        "semantic_loss": NoulAnswer(noul=0.10),
        "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.85),
    }


class TestCJKConfidenceGating:
    def test_cjk_gating_skips_low_conf_choice(self):
        """CJK gating 默认启用: conf=0.55 的 Choice 应被跳过 (不 reject)."""
        from src.judges.verdict import synthesize_verdict

        v = synthesize_verdict(_answers_with_low_conf_choice())
        # voice_drift conf=0.55 < cjk 0.6 → skipped, 不进入 reject 路径
        assert "voice_drift" in v.skipped_dimensions, (
            f"CJK 模式下 conf=0.55 应被 skipped_dimensions 收录. "
            f"skipped: {v.skipped_dimensions}, reasons: {v.skipped_reasons}"
        )
        # 而且 skip reason 标记 (cjk)
        skip_reason = v.skipped_reasons.get("voice_drift", "")
        assert "(cjk)" in skip_reason, f"应标记 (cjk). 实际: {skip_reason}"

    def test_disable_cjk_gating_passes_low_conf(self):
        """关闭 CJK gating: 走通用 0.5 阈值, conf=0.55 通过."""
        from src.judges.verdict import synthesize_verdict

        v = synthesize_verdict(
            _answers_with_low_conf_choice(),
            thresholds={"cjk_confidence_gating": False},
        )
        # conf=0.55 > 0.5 → 通过, voice_drift 进入 reject 路径 (subtle 单独不 reject 但被记录)
        # 这里关键检查: skipped 标签不应带 (cjk)
        skip_reason = v.skipped_reasons.get("voice_drift", "")
        if "voice_drift" in v.skipped_dimensions:
            assert "(cjk)" not in skip_reason, f"关闭 CJK gating 时不应带 (cjk). 实际: {skip_reason}"

    def test_high_conf_choice_passes_cjk_gating(self):
        """高置信 (>= 0.6) 的 Choice 不被 CJK gating 跳过."""
        from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer
        from src.judges.verdict import synthesize_verdict

        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 0.85, "subtle": 0.10, "severe": 0.05}, 0.85),
            "register_match": ChoiceAnswer("match", {"match": 0.95, "under": 0.03, "over": 0.02}, 0.95),
            "entity_consistency": NoulAnswer(noul=0.05),
            "literal_residue": NoulAnswer(noul=0.10),
            "semantic_loss": NoulAnswer(noul=0.10),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.85),
        }
        v = synthesize_verdict(answers)
        assert "voice_drift" not in v.skipped_dimensions, (
            "高置信 Choice (conf=0.85) 不应被 skipped. "
            f"skipped: {v.skipped_dimensions}"
        )
