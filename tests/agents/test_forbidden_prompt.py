"""Round 54: 单元测试 - forbidden_modernisms 注入 Rewriter prompt.

保护目标（修复 dead config）：
  - config.yaml 的 forbidden_modernisms 必须出现在 Rewriter prompt 中
  - 默认空列表 → prompt 中无禁用词段
  - 配置后 → 禁用词段出现，含全部禁用词
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.agents.rewriter_agent import LiteraryRewriterAgent
from src.core.decision_engine import DecisionEngine


@pytest.fixture
def rewriter():
    """构造一个不依赖 DB 的 Rewriter 实例（仅测试 _build_prompt）"""
    db = DecisionEngine(db_path=":memory:")
    return LiteraryRewriterAgent(db)


class TestForbiddenModernisms:
    def test_empty_forbidden_list_omits_section(self, rewriter, monkeypatch):
        # 配置为空 → prompt 中无禁用词段
        monkeypatch.setitem(rewriter.db._data if hasattr(rewriter.db, "_data") else {}, "style_guide", {})
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": []}
            prompt = rewriter._build_prompt(
                raw_translation="直译稿。",
                decisions_context="",
                style_guide="风格指南",
            )
            assert "禁用词列表" not in prompt
            assert "严禁出现" not in prompt

    def test_none_forbidden_list_omits_section(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {}
            prompt = rewriter._build_prompt(
                raw_translation="直译稿。",
                decisions_context="",
                style_guide="风格指南",
            )
            assert "禁用词列表" not in prompt

    def test_forbidden_words_appear_in_prompt(self, rewriter):
        forbidden = ["娘们儿", "卧槽", "开后宫"]
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": forbidden}
            prompt = rewriter._build_prompt(
                raw_translation="直译稿。",
                decisions_context="",
                style_guide="风格指南",
            )
            # 禁用词段必须出现
            assert "禁用词列表" in prompt
            assert "CRITICAL" in prompt
            # 所有禁用词必须列出
            for word in forbidden:
                assert word in prompt, f"{word} 缺失"

    def test_forbidden_words_joined_with_chinese_separator(self, rewriter):
        forbidden = ["娘们儿", "卧槽", "开后宫"]
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": forbidden}
            prompt = rewriter._build_prompt(
                raw_translation="x",
                decisions_context="",
                style_guide="",
            )
            # 中文顿号分隔
            assert "娘们儿、卧槽、开后宫" in prompt

    def test_section_includes_warning_about_rejection(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": ["X"]}
            prompt = rewriter._build_prompt(
                raw_translation="x",
                decisions_context="",
                style_guide="",
            )
            assert "reject" in prompt.lower()

    def test_section_placed_after_entity_registry(self, rewriter):
        # 验证顺序：禁用词段在 entity_registry 之后
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": ["X"]}
            prompt = rewriter._build_prompt(
                raw_translation="x",
                decisions_context="",
                style_guide="",
                entity_registry={"Luffy": "鲁夫"},
            )
            entity_pos = prompt.find("实体译名登记表")
            forbidden_pos = prompt.find("禁用词列表")
            assert entity_pos > 0
            assert forbidden_pos > entity_pos, "禁用词段应在 entity_registry 之后"


# ---------------------------------------------------------------------------
# Round 56: _build_retry_prompt 也必须含 forbidden_modernisms 段
# ---------------------------------------------------------------------------

class TestForbiddenModernismsInRetryPrompt:
    def test_retry_prompt_contains_forbidden_section(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": ["娘们儿"]}
            prompt = rewriter._build_retry_prompt(
                raw_translation="直译。",
                prev_lit_text="上一版译文。",
                decisions_context="",
                style_guide="",
                reject_reason="直译残留: conf=0.9",
                critic_feedback=None,
                retry_count=1,
            )
            assert "禁用词列表" in prompt, "retry prompt 漏 forbidden_section"
            assert "娘们儿" in prompt

    def test_retry_prompt_no_forbidden_when_list_empty(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": []}
            prompt = rewriter._build_retry_prompt(
                raw_translation="x",
                prev_lit_text="y",
                decisions_context="",
                style_guide="",
                reject_reason="",
                critic_feedback=None,
                retry_count=1,
            )
            assert "禁用词列表" not in prompt

    def test_retry_prompt_section_after_entity_registry(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {"forbidden_modernisms": ["X"]}
            prompt = rewriter._build_retry_prompt(
                raw_translation="x",
                prev_lit_text="y",
                decisions_context="",
                style_guide="",
                reject_reason="",
                critic_feedback=None,
                retry_count=1,
                entity_registry={"Luffy": "鲁夫"},
            )
            entity_pos = prompt.find("实体译名登记表")
            forbidden_pos = prompt.find("禁用词列表")
            assert entity_pos > 0
            assert forbidden_pos > entity_pos
