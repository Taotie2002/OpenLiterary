"""Round 82: 接线 voice_drift_subtle_prob 阈值 + 实现"叠加其它维度才升级"组合逻辑.

设计原则 (与 verdict.py 模块 docstring 对齐):
  - severe prob >= voice_drift_severe_prob → 单维度 REJECT (硬规则)
  - subtle prob >= voice_drift_subtle_prob 但 severe < severe_threshold → 单独不 reject
    (显式记入 skipped_dimensions 作为软跳过, 防止轻微漂移误触发 rewriter 死循环)
  - subtle prob >= subtle_threshold 且其它维度已 reject → 升级为 REJECT 原因 (组合升级)

修复前 BUG:
  - DEFAULT_THRESHOLDS 定义 voice_drift_subtle_prob=0.7 但代码完全未引用,
    "subtle 单独不触发，但叠加其它会" 的设计意图仅停留在注释, 实际运行时
    subtle 答案被静默忽略, 失去 Jev 的细微漂移信号

修复:
  - verdict.py::synthesize_verdict 在 voice_drift handler 增加 subtle 分支
  - subtle alone → skipped_dimensions (PASS)
  - subtle + 其它 reject reasons → reasons.append (REJECT)

验证:
  - subtle 单独 → decision=PASS, skipped_dimensions 含 voice_drift
  - subtle + 其它维度 reject → decision=REJECT, reject_reasons 含 "音色漂移（subtle+其它）"
  - severe 优先 → severe 命中时 subtle 不参与
  - 阈值边界 → subtle_prob == threshold (0.7) 触发升级
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.judges.typesafe_spec import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from src.judges.verdict import (
    DEFAULT_THRESHOLDS,
    synthesize_verdict,
)


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



# ---------------------------------------------------------------------------
# Round 82: subtle voice_drift 单独 → 不 reject (PASS), 记入 skipped_dimensions
# ---------------------------------------------------------------------------

class TestSubtleAloneDoesNotReject:
    """subtle 单维度: severe < threshold, subtle >= threshold, 无其它 reject → PASS."""

    def test_subtle_alone_passes(self):
        ans = ChoiceAnswer(
            choice="subtle",
            probabilities={"none": 0.2, "subtle": 0.7, "severe": 0.1},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "PASS"
        assert "voice_drift" in v.skipped_dimensions
        assert any("subtle" in r.lower() for r in v.skipped_reasons.values())
        # 不应产生 reject reasons
        assert v.reject_reasons == []

    def test_subtle_alone_at_threshold_passes(self):
        """边界: subtle_prob == voice_drift_subtle_prob (0.7) → 仍 alone 不 reject."""
        ans = ChoiceAnswer(
            choice="subtle",
            probabilities={"none": 0.15, "subtle": 0.7, "severe": 0.15},
            confidence=0.9,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "PASS"
        assert "voice_drift" in v.skipped_dimensions

    def test_subtle_below_threshold_just_skips(self):
        """subtle < threshold: 既不 reject 也不 escalate, 只跳过 (无信号)."""
        ans = ChoiceAnswer(
            choice="subtle",
            probabilities={"none": 0.4, "subtle": 0.5, "severe": 0.1},
            confidence=0.7,
        )
        v = synthesize_verdict({"voice_drift": ans})
        # 现有逻辑: subtle < threshold 时根本不进入 subtle 分支
        # 应该与"无 voice_drift"等价 — 既不 reject, 也不 escalated
        assert v.decision == "PASS"
        # 不应加入 skipped_dimensions (因为没有任何处理发生)
        # 这取决于现有 skipped 逻辑 — 不强制要求


# ---------------------------------------------------------------------------
# Round 82: subtle + 其它 reject → escalate (REJECT)
# ---------------------------------------------------------------------------

class TestSubtleWithOtherRejectsEscalates:
    """subtle 达到阈值, 且其它维度已 reject → 加入 reject_reasons (升级)."""

    def test_subtle_combined_with_literal_residue_rejects(self):
        """subtle voice_drift + literal_residue (noul >= threshold) → REJECT."""
        ans_voice = ChoiceAnswer(
            choice="subtle",
            probabilities={"none": 0.2, "subtle": 0.7, "severe": 0.1},
            confidence=0.8,
        )
        # literal_residue noul=0.6 >= threshold=0.5 → reject
        ans_lit = NoulAnswer(noul=0.8)  # > 0.7 uncertain_high
        v = synthesize_verdict({
            "voice_drift": ans_voice,
            "literal_residue": ans_lit,
        })
        assert v.decision == "REJECT"
        # 应有 literal_residue + subtle 两个原因
        assert any("直译残留" in r for r in v.reject_reasons)
        assert any("音色漂移（subtle+其它" in r for r in v.reject_reasons)

    def test_subtle_combined_with_semantic_loss_rejects(self):
        """subtle voice_drift + semantic_loss → REJECT."""
        ans_voice = ChoiceAnswer(
            choice="subtle",
            probabilities={"none": 0.1, "subtle": 0.75, "severe": 0.15},
            confidence=0.8,
        )
        ans_sem = NoulAnswer(noul=0.8)  # > 0.7 uncertain_high, ≥ 0.4 threshold
        v = synthesize_verdict({
            "voice_drift": ans_voice,
            "semantic_loss": ans_sem,
        })
        assert v.decision == "REJECT"
        assert any("关键语义丢失" in r for r in v.reject_reasons)
        assert any("音色漂移（subtle+其它" in r for r in v.reject_reasons)

    def test_subtle_combined_with_entity_consistency_rejects(self):
        """subtle + entity_consistency (noul >= 0.3) → REJECT."""
        ans_voice = ChoiceAnswer(
            choice="subtle",
            probabilities={"none": 0.2, "subtle": 0.7, "severe": 0.1},
            confidence=0.8,
        )
        ans_ent = NoulAnswer(noul=0.8)  # > 0.7 uncertain_high, ≥ 0.3 threshold
        v = synthesize_verdict({
            "voice_drift": ans_voice,
            "entity_consistency": ans_ent,
        })
        assert v.decision == "REJECT"
        assert any("音色漂移（subtle+其它" in r for r in v.reject_reasons)


# ---------------------------------------------------------------------------
# Round 82: severe 优先 — severe 命中时 subtle 不参与
# ---------------------------------------------------------------------------

class TestSevereTakesPrecedenceOverSubtle:
    """severe >= threshold 时, 只用 severe reason, subtle 不参与."""

    def test_severe_only_no_subtle_reason(self):
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.0, "subtle": 0.0, "severe": 1.0},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "REJECT"
        # 只有 severe 原因, 没有 subtle
        assert any("音色漂移（severe）" in r for r in v.reject_reasons)
        assert not any("subtle" in r.lower() and "其它" not in r for r in v.reject_reasons)

    def test_severe_and_subtle_combined_uses_severe(self):
        """severe=0.7, subtle=0.3 (subtle 不达阈值) → 只有 severe reject."""
        ans = ChoiceAnswer(
            choice="severe",
            probabilities={"none": 0.0, "subtle": 0.3, "severe": 0.7},
            confidence=0.8,
        )
        v = synthesize_verdict({"voice_drift": ans})
        assert v.decision == "REJECT"
        assert any("severe" in r for r in v.reject_reasons)


# ---------------------------------------------------------------------------
# Round 82: 阈值配置完整性 (DEFAULT_THRESHOLDS 仍含 voice_drift_subtle_prob)
# ---------------------------------------------------------------------------

class TestThresholdKeyStillPresent:
    """配置完整性: DEFAULT_THRESHOLDS 保留 voice_drift_subtle_prob 键 (向后兼容)."""

    def test_subtle_threshold_key_present(self):
        assert "voice_drift_subtle_prob" in DEFAULT_THRESHOLDS
        assert DEFAULT_THRESHOLDS["voice_drift_subtle_prob"] == 0.7


# ---------------------------------------------------------------------------
# Round 82: 整合脚本 src/translator_agent.py 同步修复
# ---------------------------------------------------------------------------

class TestIntegratedScriptSynced:
    """散装 src/judges/verdict.py 修复必须同步到整合脚本 src/translator_agent.py."""

    def test_round_82_marker_in_scattered(self):
        content = Path("src/judges/verdict.py").read_text(encoding="utf-8")
        assert "Round 82" in content, "散装 verdict.py 未标注 Round 82 修复"

    def test_round_82_marker_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 82" in content, "整合脚本未同步 Round 82 修复"

    def test_voice_drift_handler_in_scattered(self):
        content = Path("src/judges/verdict.py").read_text(encoding="utf-8")
        # 检查 subtle+其它 字符串出现
        assert "音色漂移（subtle+其它）" in content, (
            "散装 verdict.py 应有 '音色漂移（subtle+其它）' reason 格式"
        )

    def test_voice_drift_handler_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "音色漂移（subtle+其它）" in content, (
            "整合脚本 src/translator_agent.py 应有 '音色漂移（subtle+其它）' reason 格式"
        )
