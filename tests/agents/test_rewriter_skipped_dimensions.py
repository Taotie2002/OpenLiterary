"""Round 64: 单元测试 - rewriter_agent 感知 skipped_dimensions.

保护目标：
  - _build_retry_prompt 在收到 skipped_dimensions 时，prompt 渲染"边界判断跳过"section
  - skipped_dimensions 为 None 或空 list 时，prompt 中不渲染该 section
  - process_chunk 签名接受 skipped_dimensions 并透传给 _build_retry_prompt
  - 验证 section 内容含"无需重点修改"语义，让 LLM 不要局部修复边界判断维度
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.agents.rewriter_agent import LiteraryRewriterAgent
from src.core.decision_engine import DecisionEngine


@pytest.fixture
def rewriter():
    db = DecisionEngine(db_path=":memory:")
    return LiteraryRewriterAgent(db)


class TestRetryPromptSkippedDimensions:
    """_build_retry_prompt 对 skipped_dimensions 的渲染行为"""

    def test_none_skipped_omits_section(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {}
            prompt = rewriter._build_retry_prompt(
                raw_translation="直译稿。",
                prev_lit_text="上一版。",
                decisions_context="",
                style_guide="风格",
                reject_reason="测试拒绝",
                critic_feedback=None,
                retry_count=1,
                skipped_dimensions=None,
            )
            assert "Jev 边界判断跳过" not in prompt
            assert "跳过维度" not in prompt

    def test_empty_skipped_omits_section(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {}
            prompt = rewriter._build_retry_prompt(
                raw_translation="直译稿。",
                prev_lit_text="上一版。",
                decisions_context="",
                style_guide="风格",
                reject_reason="测试拒绝",
                critic_feedback=None,
                retry_count=1,
                skipped_dimensions=[],
            )
            assert "Jev 边界判断跳过" not in prompt
            assert "跳过维度" not in prompt

    def test_skipped_dimensions_rendered_in_section(self, rewriter):
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {}
            prompt = rewriter._build_retry_prompt(
                raw_translation="直译稿。",
                prev_lit_text="上一版。",
                decisions_context="",
                style_guide="风格",
                reject_reason="测试拒绝",
                critic_feedback=None,
                retry_count=2,
                skipped_dimensions=["entity_consistency", "literal_residue"],
            )
            # Section 必须出现
            assert "Jev 边界判断跳过" in prompt
            # 必须含 2 个维度名
            assert "entity_consistency" in prompt
            assert "literal_residue" in prompt
            # 必须含"无需重点修改"语义（避免 LLM 局部修复边界判断维度）
            assert "无需重点修改" in prompt

    def test_skipped_section_explains_uncertain_zone(self, rewriter):
        """Section 必须解释 uncertain zone 是 noul ∈ [0.3, 0.7] / confidence < 0.5"""
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {}
            prompt = rewriter._build_retry_prompt(
                raw_translation="直译稿。",
                prev_lit_text="上一版。",
                decisions_context="",
                style_guide="风格",
                reject_reason="测试拒绝",
                critic_feedback=None,
                retry_count=1,
                skipped_dimensions=["voice_drift"],
            )
            assert "noul ∈ [0.3, 0.7]" in prompt or "noul ∈" in prompt
            assert "confidence < 0.5" in prompt

    def test_skipped_section_after_jev_categories(self, rewriter):
        """Section 应渲染在 jev_categories_section 之后、entity_feedback_section 之前"""
        with patch("src.agents.rewriter_agent._config") as mock_config:
            mock_config.style_guide = {}
            prompt = rewriter._build_retry_prompt(
                raw_translation="直译稿。",
                prev_lit_text="上一版。",
                decisions_context="",
                style_guide="风格",
                reject_reason="直译残留: 局部词换无效",  # 触发 jev category
                critic_feedback=None,
                retry_count=1,
                entity_feedback="实体反馈：xxx 应统一为 yyy",
                entity_registry=None,
                skipped_dimensions=["literal_residue"],
            )
            jev_pos = prompt.find("本轮核心问题维度")
            skipped_pos = prompt.find("Jev 边界判断跳过")
            entity_pos = prompt.find("实体一致性反馈")
            # 顺序：jev_categories < skipped < entity_feedback
            assert 0 <= jev_pos < skipped_pos < entity_pos, (
                f"位置异常: jev={jev_pos}, skipped={skipped_pos}, entity={entity_pos}"
            )


class TestProcessChunkSignature:
    """process_chunk 必须接受 skipped_dimensions 并透传"""

    def test_process_chunk_accepts_skipped_dimensions_kwarg(self, rewriter):
        """通过 kwargs 调用 process_chunk，不抛 TypeError"""
        # Mock 掉 _build_retry_prompt 以验证透传
        with patch.object(rewriter, "_build_retry_prompt", return_value="stubbed prompt") as mock_rp:
            with patch.object(rewriter, "_build_prompt", return_value="stubbed prompt"):
                with patch.object(rewriter, "llm") as mock_llm:
                    mock_llm.generate.return_value = "润色结果"
                    rewriter.process_chunk(
                        chunk_id="test_chunk",
                        raw_translation="raw",
                        style_guide_stats={},
                        source_text="source",
                        retry_count=1,
                        prev_lit_text="prev",
                        reject_reason="拒因",
                        skipped_dimensions=["entity_consistency"],
                    )
                    # _build_retry_prompt 必须收到 skipped_dimensions
                    assert mock_rp.called
                    _, kwargs = mock_rp.call_args
                    assert "skipped_dimensions" in kwargs
                    assert kwargs["skipped_dimensions"] == ["entity_consistency"]

    def test_process_chunk_default_skipped_dimensions_is_none(self, rewriter):
        """不传 skipped_dimensions 时，默认 None（不破坏现有调用）"""
        with patch.object(rewriter, "_build_retry_prompt", return_value="stubbed prompt") as mock_rp:
            with patch.object(rewriter, "_build_prompt", return_value="stubbed prompt"):
                with patch.object(rewriter, "llm") as mock_llm:
                    mock_llm.generate.return_value = "润色结果"
                    rewriter.process_chunk(
                        chunk_id="test_chunk",
                        raw_translation="raw",
                        style_guide_stats={},
                        source_text="source",
                        retry_count=1,
                        prev_lit_text="prev",
                        reject_reason="拒因",
                    )
                    _, kwargs = mock_rp.call_args
                    assert kwargs.get("skipped_dimensions") is None
