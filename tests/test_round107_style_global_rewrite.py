"""Round 107 P5: 风格契合度低 → 整体重写策略.

ch02 chunk001 失败根因:
  Jev 判定 "风格契合度低" (score=0.99 conf=0.64), 全局风格问题.
  但 "风格契合度低" 不在 GLOBAL_REWRITE_CATEGORIES, retry 策略告诉 rewriter
  "只动被拒原因相关的段落" (针对性修改), rewriter 做小修改 (99.88% 相似),
  没解决风格根因, judge 持续拒绝, 振荡检测触发 (sim=0.9765).

修复:
  把 "风格契合度低" 加入 GLOBAL_REWRITE_CATEGORIES + _JEV_CATEGORY_STRATEGY,
  让 rewriter 对风格问题做整体重润色.

测试覆盖:
  - "风格契合度低" 在 GLOBAL_REWRITE_CATEGORIES 里
  - "风格契合度低" 在 _JEV_CATEGORY_STRATEGY 里
  - retry prompt 对 "风格契合度低" 触发整体重写策略
  - 两版一致
"""
from __future__ import annotations

import pytest


class TestStyleGlobalRewrite:
    def test_style_in_global_rewrite_categories(self):
        """"风格契合度低" 在 GLOBAL_REWRITE_CATEGORIES 里."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        # 通过 _build_retry_prompt 间接验证 (策略选择逻辑)
        prompt = agent._build_retry_prompt(
            raw_translation="直译底稿",
            prev_lit_text="上一版译文",
            decisions_context="决策上下文",
            style_guide="风格基准",
            reject_reason="风格契合度低: score=0.99 conf=0.64",
            critic_feedback="低分维度：style_compliance(5.0分)",
            retry_count=1,
            low_dims=["style_compliance"],
        )
        # 整体重写策略应包含 "整体重润色" 或 "整体重写"
        assert "整体重润色" in prompt or "整体重写" in prompt, \
            f"风格契合度低应触发整体重写策略, prompt 片段: {prompt[:300]}"

    def test_style_in_jev_category_strategy(self):
        """"风格契合度低" 在 _JEV_CATEGORY_STRATEGY 里."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        assert "风格契合度低" in agent._JEV_CATEGORY_STRATEGY, \
            "_JEV_CATEGORY_STRATEGY 应包含 风格契合度低"

    def test_jev_category_section_renders_style(self):
        """_build_jev_category_section 对 风格契合度低 渲染具体策略."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        section = agent._build_jev_category_section("风格契合度低: score=0.99 conf=0.64")
        assert "风格契合度低" in section, f"jev category section 应包含 风格契合度低: {section[:200]}"
        assert "整体" in section, "策略应提到整体重润色/重写"

    def test_monolith_and_scattered_match(self):
        """单体版与散装版 风格契合度低 策略一致."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent as ScatteredAgent
        from src.translator_agent import LiteraryRewriterAgent as MonolithAgent

        for AgentClass, name in [(ScatteredAgent, "散装版"), (MonolithAgent, "单体版")]:
            agent = AgentClass.__new__(AgentClass)
            assert "风格契合度低" in agent._JEV_CATEGORY_STRATEGY, \
                f"{name} _JEV_CATEGORY_STRATEGY 应包含 风格契合度低"
            prompt = agent._build_retry_prompt(
                raw_translation="直译底稿",
                prev_lit_text="上一版译文",
                decisions_context="决策上下文",
                style_guide="风格基准",
                reject_reason="风格契合度低: score=0.99 conf=0.64",
                critic_feedback="低分维度",
                retry_count=1,
                low_dims=["style_compliance"],
            )
            assert "整体重润色" in prompt or "整体重写" in prompt, \
                f"{name} 风格契合度低应触发整体重写策略"
