"""Round 109: semantic_loss 长度交叉校验（ch08 chunk009 根因）。

ch08_chunk009: Jev 返回 semantic_loss noul=1.0 (100% 语义丢失, confidence=1.0),
但实际译文忠实流畅 (len(lit)/len(raw)=0.994, 未压缩)。judge 以低置信度 REJECT,
rewriter 两次分数完全相同 (8.2143) 无法改进 → NO_PROGRESS_STREAK → fallback。

根因: Jev 在 CJK 上对 semantic_loss 维度校准噪声大, 出现高置信度误判。

修复: 在 synthesize_verdict 的 semantic_loss 分支加长度交叉校验 —
若润色稿相对直译稿未明显压缩 (len(lit)/len(raw) >= ratio_min, 默认 0.8),
高 semantic_loss noul 视为 Jev 校准噪声, cap 到 reject 阈值以下 (默认 0.2)。
译文被压缩 (ratio < 0.8) 时不 cap, 保留 Jev 判断。

双库同步: 共享 src/judges/verdict.py + 单体 src/translator_agent.py 内联副本。
"""
from __future__ import annotations

import inspect

import pytest

from src.judges.typesafe_spec import NoulAnswer
from src.judges.verdict import synthesize_verdict, DEFAULT_THRESHOLDS


# ────────────────────────────────────────────────────────────────────────
# 共享版 (src/judges/verdict.py)
# ────────────────────────────────────────────────────────────────────────

class TestSharedCrosscheck:
    def test_uncompressed_caps_noul_and_passes(self):
        """未压缩 (ratio=0.994) + noul=1.0 → cap 到 0.2 → PASS。"""
        raw = "x" * 329
        lit = "译" * 327
        answers = {"semantic_loss": NoulAnswer(noul=1.0)}
        v = synthesize_verdict(answers, lit_text=lit, raw_text=raw)
        assert v.decision == "PASS"
        assert v.reject_reasons == []
        sl = v.primitive_summary["semantic_loss"]
        assert sl["noul"] == 0.2
        assert sl.get("_length_capped") is True

    def test_compressed_not_capped_and_rejects(self):
        """压缩 (ratio=0.4 < 0.8) + noul=1.0 → 不 cap → REJECT。"""
        raw = "x" * 329
        lit = "译" * 130  # ratio ≈ 0.395
        answers = {"semantic_loss": NoulAnswer(noul=1.0)}
        v = synthesize_verdict(answers, lit_text=lit, raw_text=raw)
        assert v.decision == "REJECT"
        assert any("关键语义丢失" in r for r in v.reject_reasons)
        assert v.primitive_summary["semantic_loss"]["noul"] == 1.0
        assert "_length_capped" not in v.primitive_summary["semantic_loss"]

    def test_no_raw_text_backward_compat_rejects(self):
        """未传 raw_text (向后兼容) → 不 cap → REJECT。"""
        lit = "译" * 327
        answers = {"semantic_loss": NoulAnswer(noul=1.0)}
        v = synthesize_verdict(answers, lit_text=lit)
        assert v.decision == "REJECT"
        assert v.primitive_summary["semantic_loss"]["noul"] == 1.0

    def test_crosscheck_disabled_rejects(self):
        """阈值关闭交叉校验 → 不 cap → REJECT。"""
        raw = "x" * 329
        lit = "译" * 327
        answers = {"semantic_loss": NoulAnswer(noul=1.0)}
        v = synthesize_verdict(
            answers,
            thresholds={"semantic_loss_length_crosscheck": False},
            lit_text=lit,
            raw_text=raw,
        )
        assert v.decision == "REJECT"
        assert v.primitive_summary["semantic_loss"]["noul"] == 1.0

    def test_low_noul_unaffected(self):
        """noul 本就低于阈值 → 不受 cap 影响, 仍 PASS。"""
        raw = "x" * 329
        lit = "译" * 327
        answers = {"semantic_loss": NoulAnswer(noul=0.1)}
        v = synthesize_verdict(answers, lit_text=lit, raw_text=raw)
        assert v.decision == "PASS"
        assert v.primitive_summary["semantic_loss"]["noul"] == 0.1
        assert "_length_capped" not in v.primitive_summary["semantic_loss"]

    def test_ratio_boundary(self):
        """ratio 恰好 = ratio_min (0.8) → 触发 cap (>= 语义)。"""
        raw = "x" * 100
        lit = "译" * 80  # ratio = 0.8
        answers = {"semantic_loss": NoulAnswer(noul=0.9)}
        v = synthesize_verdict(answers, lit_text=lit, raw_text=raw)
        assert v.decision == "PASS"
        assert v.primitive_summary["semantic_loss"]["noul"] == 0.2

    def test_custom_cap_and_ratio(self):
        """自定义 cap / ratio 阈值生效。"""
        raw = "x" * 100
        lit = "译" * 85  # ratio=0.85
        answers = {"semantic_loss": NoulAnswer(noul=0.9)}
        v = synthesize_verdict(
            answers,
            thresholds={
                "semantic_loss_length_ratio_min": 0.9,  # 0.85 < 0.9 → 不 cap
            },
            lit_text=lit,
            raw_text=raw,
        )
        assert v.decision == "REJECT"
        assert v.primitive_summary["semantic_loss"]["noul"] == 0.9


# ────────────────────────────────────────────────────────────────────────
# 单体版 (src/translator_agent.py 内联副本)
# ────────────────────────────────────────────────────────────────────────

class TestMonolithCrosscheck:
    @pytest.fixture(scope="class")
    def mono(self):
        import src.translator_agent as ta
        return ta

    def test_uncompressed_caps_and_passes(self, mono):
        raw = "x" * 329
        lit = "译" * 327
        answers = {"semantic_loss": NoulAnswer(noul=1.0)}
        v = mono.synthesize_verdict(answers, lit_text=lit, raw_text=raw)
        assert v.decision == "PASS"
        assert v.primitive_summary["semantic_loss"]["noul"] == 0.2
        assert v.primitive_summary["semantic_loss"].get("_length_capped") is True

    def test_compressed_rejects(self, mono):
        raw = "x" * 329
        lit = "译" * 130
        answers = {"semantic_loss": NoulAnswer(noul=1.0)}
        v = mono.synthesize_verdict(answers, lit_text=lit, raw_text=raw)
        assert v.decision == "REJECT"
        assert v.primitive_summary["semantic_loss"]["noul"] == 1.0

    def test_default_thresholds_have_crosscheck_keys(self, mono):
        assert mono.DEFAULT_THRESHOLDS["semantic_loss_length_crosscheck"] is True
        assert mono.DEFAULT_THRESHOLDS["semantic_loss_length_ratio_min"] == 0.8
        assert mono.DEFAULT_THRESHOLDS["semantic_loss_noul_cap"] == 0.2


# ────────────────────────────────────────────────────────────────────────
# 调用点接线 (judge / critic 传 raw_text)
# ────────────────────────────────────────────────────────────────────────

class TestCallSitesWireRawText:
    def test_shared_judge_passes_raw_text(self):
        from src.judges import judge
        src = inspect.getsource(judge.evaluate_chunk)
        assert "raw_text=raw_translation" in src

    def test_shared_critic_passes_raw_text(self):
        from src.agents import critic_agent
        src = inspect.getsource(critic_agent.CriticAgent.process_chunk)
        assert "raw_text=raw_trans" in src

    def test_monolith_judge_passes_raw_text(self):
        import src.translator_agent as ta
        src = inspect.getsource(ta.evaluate_chunk)
        assert "raw_text=raw_translation" in src

    def test_monolith_critic_passes_raw_text(self):
        import src.translator_agent as ta
        # 单体 critic 是 TranslationPipeline 的方法, 找 synthesize_verdict 调用
        src = inspect.getsource(ta)
        # 统计 raw_text=raw_trans 出现次数 (critic 两处)
        assert src.count("raw_text=raw_trans") >= 2
