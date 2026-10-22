"""Round 107 P4: rewriter retry prompt 加"必须修改"指令.

ch02 chunk001 失败根因:
  Jev 真实评分 (style_compliance=5.0, literal_residue=4.2 等), judge REJECT.
  rewriter 收到反馈但 LLM 输出与上一版字节级相同 → REWRITER_STAGNATED → 失败.

  根因: retry prompt 说"其他部分保持稳定", LLM 认为译文已够好, 不做修改.

修复:
  retry prompt 末尾加【CRITICAL — 必须修改】指令, 强制 LLM 至少做一处实质性修改.

测试覆盖:
  - retry prompt 包含"必须修改"指令
  - 指令明确要求"至少一处实质性修改"
  - 指令说明"字节级相同 → REWRITER_STAGNATED"
  - 两版 prompt 一致
"""
from __future__ import annotations

import pytest


class TestRewriterMustChange:
    def test_retry_prompt_contains_must_change(self):
        """retry prompt 包含【CRITICAL — 必须修改】指令."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent

        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        prompt = agent._build_retry_prompt(
            raw_translation="直译底稿",
            prev_lit_text="上一版译文",
            decisions_context="决策上下文",
            style_guide="风格基准",
            reject_reason="风格契合度低",
            critic_feedback="低分维度：style_compliance(5.0分)",
            retry_count=1,
        )
        assert "【CRITICAL — 必须修改】" in prompt, "retry prompt 应包含必须修改指令"
        assert "至少一处实质性修改" in prompt, "指令应要求至少一处实质性修改"
        assert "REWRITER_STAGNATED" in prompt, "指令应说明字节级相同的后果"

    def test_first_prompt_no_must_change(self):
        """首次润色 prompt 不包含"必须修改"指令 (只有 retry 才需要)."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent

        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        agent._few_shot_examples = ""
        prompt = agent._build_prompt(
            "直译底稿", "决策上下文", "风格基准",
        )
        assert "【CRITICAL — 必须修改】" not in prompt, "首次 prompt 不应包含必须修改指令"

    def test_monolith_and_scattered_match(self):
        """单体版与散装版 retry prompt 都包含必须修改指令."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent as ScatteredAgent
        from src.translator_agent import LiteraryRewriterAgent as MonolithAgent

        for AgentClass, name in [(ScatteredAgent, "散装版"), (MonolithAgent, "单体版")]:
            agent = AgentClass.__new__(AgentClass)
            prompt = agent._build_retry_prompt(
                raw_translation="直译底稿",
                prev_lit_text="上一版译文",
                decisions_context="决策上下文",
                style_guide="风格基准",
                reject_reason="风格契合度低",
                critic_feedback="低分维度",
                retry_count=1,
            )
            assert "【CRITICAL — 必须修改】" in prompt, f"{name} retry prompt 应包含必须修改指令"
