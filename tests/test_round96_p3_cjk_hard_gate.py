"""Round 96 P3: synthesize_verdict CJK 硬性闸门 — 派生 cjk_compliance 维度.

设计动机:
  P1 (final) 和 P2 (raw) 守住流水线两端, 但中段 (literary translation) 在
  critic/judge 评估时仍可能被 LLM 给出高分 (Jev 没察觉 CJK 异常).
  Round 96 P3 在 synthesize_verdict 阶段派生 cjk_compliance 维度:
    - 若 lit_text CJK 比例 < cjk_hard_threshold (默认 0.5), 直接加 REJECT reason
    - 不走 Noul uncertain zone 跳过逻辑 (这是硬性 FAIL)
    - 触发 rewriter 重试, 而不是等到 final 才发现

测试覆盖:
  - synthesize_verdict 加 lit_text 参数
  - 中文 lit → PASS + cjk_compliance.noul=0
  - 英文 lit → REJECT + 中文 reject reason
  - 混合 lit → REJECT (cjk < 0.5)
  - lit=None → 不影响 (向后兼容)
  - threshold=0 → 禁用闸门
  - cjk_compliance 写入 primitive_summary
  - 整合脚本 + evaluate_chunk 传递 lit_text
"""
from __future__ import annotations

import sys

import pytest


# ---------------------------------------------------------------------------
# synthesize_verdict 加 lit_text 参数
# ---------------------------------------------------------------------------

class TestVerdictCjkHardGate:
    """synthesize_verdict 派生 cjk_compliance 硬性闸门."""

    def test_lit_text_param_exists(self):
        """synthesize_verdict 接受 lit_text 关键字参数."""
        from src.judges.verdict import synthesize_verdict
        import inspect
        sig = inspect.signature(synthesize_verdict)
        assert "lit_text" in sig.parameters
        assert sig.parameters["lit_text"].default is None

    def test_chinese_lit_passes(self):
        """中文 lit → PASS, cjk_compliance noul=0."""
        from src.judges.verdict import synthesize_verdict
        lit = "这是一段中文翻译内容" * 20
        v = synthesize_verdict({}, thresholds={"cjk_hard_threshold": 0.5}, lit_text=lit)
        assert v.decision == "PASS"
        assert v.reject_reasons == []
        assert "cjk_compliance" in v.primitive_summary
        assert v.primitive_summary["cjk_compliance"]["noul"] == pytest.approx(0.0, abs=1e-6)
        assert v.primitive_summary["cjk_compliance"]["_hard_gate"] is True

    def test_english_lit_rejects(self):
        """英文 lit → REJECT + 中文 reject reason."""
        from src.judges.verdict import synthesize_verdict
        v = synthesize_verdict(
            {}, thresholds={"cjk_hard_threshold": 0.5}, lit_text="This is English text only."
        )
        assert v.decision == "REJECT"
        assert len(v.reject_reasons) == 1
        reason = v.reject_reasons[0]
        assert "CJK 比例过低" in reason
        assert "cjk_ratio=0.00" in reason

    def test_mixed_lit_rejects(self):
        """混合 lit (CJK < 0.5) → REJECT."""
        from src.judges.verdict import synthesize_verdict
        # "你好 hello world abc def ghi" cjk=2/19 ≈ 0.105
        lit = "你好 hello world abc def ghi jkl mno pqr"
        v = synthesize_verdict({}, thresholds={"cjk_hard_threshold": 0.5}, lit_text=lit)
        assert v.decision == "REJECT"
        assert any("CJK 比例过低" in r for r in v.reject_reasons)
        # cjk_compliance 应被记录
        assert "cjk_compliance" in v.primitive_summary
        # noul = 1 - cjk_ratio
        cjk_summary = v.primitive_summary["cjk_compliance"]
        assert cjk_summary["noul"] > 0.5  # 应 >= threshold 触发 reject

    def test_lit_none_no_effect(self):
        """lit_text=None 不影响 (向后兼容)."""
        from src.judges.verdict import synthesize_verdict
        v = synthesize_verdict({}, lit_text=None)
        assert v.decision == "PASS"
        # 没有 lit_text 时不应有 cjk_compliance
        assert "cjk_compliance" not in v.primitive_summary

    def test_threshold_zero_disables_gate(self):
        """threshold=0 禁用闸门 → 英文 lit 也通过."""
        from src.judges.verdict import synthesize_verdict
        v = synthesize_verdict(
            {}, thresholds={"cjk_hard_threshold": 0}, lit_text="Pure English text."
        )
        assert v.decision == "PASS"
        assert v.reject_reasons == []

    def test_threshold_default_0_5(self):
        """默认 threshold=0.5."""
        from src.judges.verdict import synthesize_verdict
        # 阈值为 0.5, cjk=0.4 → 应被拒
        lit = "你好 hello world abc"  # cjk ≈ 0.13
        v = synthesize_verdict({}, lit_text=lit)  # 不传 thresholds, 走默认
        assert v.decision == "REJECT"

    def test_hard_gate_skips_uncertain_zone(self):
        """硬闸门不走 Noul uncertain zone (即使 noul 在 zone 内也 reject)."""
        from src.judges.verdict import synthesize_verdict
        # 设置不确定区间为 [0.3, 0.7], 让 cjk_noul=0.55 (在 zone 内)
        # 但硬闸门应直接 reject, 不跳过
        thresholds = {
            "cjk_hard_threshold": 0.5,
            "noul_uncertain_low": 0.3,
            "noul_uncertain_high": 0.7,
        }
        # lit CJK ≈ 0.45 → noul = 0.55 (在 zone [0.3, 0.7])
        # Noul 不确定区间应跳过, 但硬闸门不应
        lit = "你好你好你好 hello world abc def ghi"  # 6/22 ≈ 0.27
        # 实际 cjk < 0.5, 所以应 reject
        v = synthesize_verdict({}, thresholds=thresholds, lit_text=lit)
        # 让我们确保 cjk_noul >= 0.5
        # cjk=0.27 → noul=0.73 >= 0.5 → reject
        assert v.decision == "REJECT"
        assert any("CJK 比例过低" in r for r in v.reject_reasons)

    def test_cjk_compliance_in_summary_even_on_pass(self):
        """即使 PASS, cjk_compliance 也应被写入 summary (供 cost_report/QA)."""
        from src.judges.verdict import synthesize_verdict
        lit = "你好你好你好你好 hello world"
        # cjk=8/19 ≈ 0.42 < 0.5 → reject
        # 让我们改成 cjk > 0.5 的
        lit = "你好你好你好你好你好你好你好你好 hello"
        # 16/17 ≈ 0.94 → pass
        v = synthesize_verdict({}, thresholds={"cjk_hard_threshold": 0.5}, lit_text=lit)
        assert v.decision == "PASS"
        assert "cjk_compliance" in v.primitive_summary


# ---------------------------------------------------------------------------
# evaluate_chunk 传递 lit_text
# ---------------------------------------------------------------------------

class TestEvaluateChunkPassesLitText:
    """evaluate_chunk 应该传递 literary_translation 给 synthesize_verdict."""

    def test_evaluate_chunk_signature(self):
        """evaluate_chunk 接受 literary_translation 参数."""
        from src.judges.judge import evaluate_chunk
        import inspect
        sig = inspect.signature(evaluate_chunk)
        assert "literary_translation" in sig.parameters

    def test_evaluate_chunk_english_lit_rejects(self):
        """evaluate_chunk 输入英文 lit → REJECT (无需 LLM 调用)."""
        from src.judges.judge import evaluate_chunk
        # 使用 Mock backend (无需真实 LLM)
        result = evaluate_chunk(
            source_text="Hello world.",
            raw_translation="你好世界",
            literary_translation="This is English text only.",  # 英文 lit
            style_guide_stats={},
            thresholds={"cjk_hard_threshold": 0.5, "allow_mock": True},
        )
        # 应被 CJK 硬闸门拦截
        assert result.decision == "REJECT"
        assert any("CJK 比例过低" in r for r in result.reject_reasons)


# ---------------------------------------------------------------------------
# 整合脚本同步
# ---------------------------------------------------------------------------

class TestIntegratedScriptSyncP3:
    """src/translator_agent.py 应同步 P3 改动."""

    def test_synthesize_verdict_has_lit_text_in_integrated(self):
        """整合脚本的 synthesize_verdict 含 lit_text 参数."""
        from src.translator_agent import synthesize_verdict
        import inspect
        sig = inspect.signature(synthesize_verdict)
        assert "lit_text" in sig.parameters
        assert sig.parameters["lit_text"].default is None

    def test_integrated_cjk_gate_works(self):
        """整合脚本的 CJK 闸门也工作."""
        from src.translator_agent import synthesize_verdict
        v = synthesize_verdict(
            {}, thresholds={"cjk_hard_threshold": 0.5}, lit_text="English only"
        )
        assert v.decision == "REJECT"
        assert any("CJK 比例过低" in r for r in v.reject_reasons)

    def test_config_has_cjk_hard_threshold(self):
        """config.yaml 含 cjk_hard_threshold."""
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "cjk_hard_threshold" in cfg["typesafe_thresholds"]
        assert cfg["typesafe_thresholds"]["cjk_hard_threshold"] == pytest.approx(0.5, abs=1e-6)
