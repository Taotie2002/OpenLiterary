"""Round 107 P6: 短文本/前言 CJK 阈值放宽.

ch01 倒查:
  ch01 是前言/目录 (65 字符, 含英文公司名/版权符号/年份), CJK 比例天然低 (0.40).
  两处 CJK 检查阈值 0.50 误判:
  1. judge CJK hard gate: cjk_noul=0.60 >= 0.50 → REJECT → 触发 rewriter
  2. rewriter 对前言无法做有意义修改 → REWRITER_STAGNATED
  3. final sanity: cjk_ratio=0.40 < 0.50 → sanity_failed → PERMANENTLY_FAILED

修复:
  对短文本 (<100 字符) 或前言特征文本, CJK 阈值放宽:
  - judge hard gate (cjk_noul 门槛): 0.50 → 0.70 (即 cjk_ratio <= 0.30 才 reject)
  - final sanity (cjk_ratio 门槛): 0.50 → 0.30 (即 cjk_ratio < 0.30 才 fail)

测试覆盖:
  - 前言 (CJK=0.40, 短文本) → judge PASS
  - 长文本低 CJK (CJK=0.00) → judge REJECT (阈值保持 0.50)
  - 极短极低 CJK (CJK=0.00) → judge REJECT (阈值 0.70, cjk_noul=1.0)
  - 常规长文本高 CJK → judge PASS
  - 两版一致
"""
from __future__ import annotations

import pytest

FRONTMATTER_TEXT = (
    "# 目录\n\n"
    "《烟囱别墅的秘密》版权所有 © 1925 由 Dodd, Mead & Company, Inc.\n\n"
    "献给潘基\n\n"
    "目录"
)
LOW_CJK_LONG = "This is a long English text that should have been translated. " * 5
VERY_LOW_SHORT = "Hello World"
NORMAL_HIGH_CJK = "这是一个正常的中文段落，包含足够的汉字，用于测试 CJK 比例检查。" * 10


class TestCJKFrontmatterRelax:
    def test_frontmatter_passes_judge(self):
        """前言 (CJK=0.40, 短文本) 应通过 judge (不 reject)."""
        from src.judges.verdict import synthesize_verdict
        verdict = synthesize_verdict({}, lit_text=FRONTMATTER_TEXT)
        assert verdict.decision == "PASS", \
            f"前言应 PASS, 实际: {verdict.decision}, reasons: {verdict.reject_reasons}"

    def test_low_cjk_long_text_still_rejected(self):
        """长文本低 CJK (CJK=0.00) 仍被 reject (阈值保持 0.50)."""
        from src.judges.verdict import synthesize_verdict
        verdict = synthesize_verdict({}, lit_text=LOW_CJK_LONG)
        assert verdict.decision == "REJECT", \
            f"长文本低 CJK 应 REJECT, 实际: {verdict.decision}"
        assert any("CJK" in r for r in verdict.reject_reasons)

    def test_very_low_short_still_rejected(self):
        """极短极低 CJK (CJK=0.00) 仍被 reject (阈值 0.70, cjk_noul=1.0)."""
        from src.judges.verdict import synthesize_verdict
        verdict = synthesize_verdict({}, lit_text=VERY_LOW_SHORT)
        assert verdict.decision == "REJECT", \
            f"极短极低 CJK 应 REJECT, 实际: {verdict.decision}"
        assert any("CJK" in r for r in verdict.reject_reasons)

    def test_normal_high_cjk_passes(self):
        """常规长文本高 CJK 应 PASS."""
        from src.judges.verdict import synthesize_verdict
        verdict = synthesize_verdict({}, lit_text=NORMAL_HIGH_CJK)
        assert verdict.decision == "PASS", \
            f"常规高 CJK 应 PASS, 实际: {verdict.decision}, reasons: {verdict.reject_reasons}"

    def test_monolith_matches_scattered(self):
        """单体版与散装版 CJK 放宽行为一致."""
        from src.judges.verdict import synthesize_verdict as scattered_syn
        from src.translator_agent import synthesize_verdict as monolith_syn

        for text, name in [(FRONTMATTER_TEXT, "前言"), (LOW_CJK_LONG, "长文本低CJK"),
                           (VERY_LOW_SHORT, "极短极低CJK"), (NORMAL_HIGH_CJK, "常规高CJK")]:
            v_sc = scattered_syn({}, lit_text=text)
            v_mo = monolith_syn({}, lit_text=text)
            assert v_sc.decision == v_mo.decision, \
                f"{name}: 散装版={v_sc.decision}, 单体版={v_mo.decision}"
