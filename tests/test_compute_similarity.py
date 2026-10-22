"""Round 57: 单元测试 - pipeline._compute_text_similarity (Round 6 引入).

保护目标（oscillation 检测）：
  - 相同字符串 → 1.0
  - 空字符串 → 0.0（避免除零）
  - 完全不同的字符串 → 较低相似度
  - 部分重叠 → 中等相似度
  - 长字符串 (>4000) 截断对比（避免 O(n²) 爆栈）
  - Unicode 字符串正确处理
  - 多次调用稳定（无副作用）
"""
from __future__ import annotations

import pytest

from src.pipeline import TranslationPipeline


# ---------------------------------------------------------------------------
# 基本行为
# ---------------------------------------------------------------------------

class TestComputeSimilarityBasic:
    def test_identical_strings_return_one(self):
        assert TranslationPipeline._compute_text_similarity("hello world", "hello world") == 1.0

    def test_completely_different_strings_low(self):
        sim = TranslationPipeline._compute_text_similarity("abcdef", "xyz123")
        assert sim < 0.5

    def test_partial_overlap_moderate(self):
        sim = TranslationPipeline._compute_text_similarity("hello world", "hello universe")
        # 公共部分 "hello " 占大头
        assert 0.3 < sim < 0.9

    def test_one_char_difference_very_high(self):
        sim = TranslationPipeline._compute_text_similarity("test123abc", "test123abd")
        assert sim > 0.8

    def test_single_char_strings_identical(self):
        assert TranslationPipeline._compute_text_similarity("a", "a") == 1.0

    def test_single_char_strings_different(self):
        assert TranslationPipeline._compute_text_similarity("a", "b") == 0.0


# ---------------------------------------------------------------------------
# 空字符串 / None 处理
# ---------------------------------------------------------------------------

class TestComputeSimilarityEmpty:
    def test_empty_a_returns_zero(self):
        assert TranslationPipeline._compute_text_similarity("", "hello") == 0.0

    def test_empty_b_returns_zero(self):
        assert TranslationPipeline._compute_text_similarity("hello", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert TranslationPipeline._compute_text_similarity("", "") == 0.0

    def test_no_division_by_zero(self):
        # 验证空字符串不会触发除零
        sim = TranslationPipeline._compute_text_similarity("", "")
        assert sim == 0.0


# ---------------------------------------------------------------------------
# 长字符串截断
# ---------------------------------------------------------------------------

class TestComputeSimilarityLong:
    def test_long_strings_truncated(self):
        # 两段前 4000 字符相同，后面不同 — 截断后应得 1.0
        prefix = "a" * 4000
        a = prefix + "different_ending_A"
        b = prefix + "different_ending_B"
        sim = TranslationPipeline._compute_text_similarity(a, b)
        # 前 4000 字符完全相同 → ratio = 1.0
        assert sim == 1.0

    def test_long_strings_different_prefix(self):
        # 前 4000 字符不同 → ratio 较低
        a = "x" * 4000 + "common_suffix"
        b = "y" * 4000 + "common_suffix"
        sim = TranslationPipeline._compute_text_similarity(a, b)
        # 公共部分只是后面的 common_suffix（占比小）
        assert sim < 0.5

    def test_exactly_4000_chars_handled(self):
        # 边界：恰好 4000 字符（不截断）
        a = "x" * 4000
        b = "x" * 4000
        sim = TranslationPipeline._compute_text_similarity(a, b)
        assert sim == 1.0

    def test_4001_chars_truncated(self):
        # 边界：4001 字符（截断 1 字符）
        a = "x" * 4001
        b = "x" * 4000  # 注意 b 只有 4000（同样截断）
        sim = TranslationPipeline._compute_text_similarity(a, b)
        # 截断后两者都是 "x" * 4000 → 1.0
        assert sim == 1.0


# ---------------------------------------------------------------------------
# Unicode 与中文
# ---------------------------------------------------------------------------

class TestComputeSimilarityUnicode:
    def test_identical_chinese(self):
        assert TranslationPipeline._compute_text_similarity("你好世界", "你好世界") == 1.0

    def test_different_chinese_low(self):
        sim = TranslationPipeline._compute_text_similarity("你好世界", "再见朋友")
        assert sim < 0.3

    def test_partial_chinese_overlap(self):
        sim = TranslationPipeline._compute_text_similarity("鲁夫和索隆", "鲁夫和山治")
        # 公共部分 "鲁夫和" + "山" vs "鲁夫和" + "索"
        assert 0.3 < sim < 0.8

    def test_mixed_cjk_and_ascii(self):
        sim = TranslationPipeline._compute_text_similarity("Chapter 1 第一章", "Chapter 1 第一章")
        assert sim == 1.0


# ---------------------------------------------------------------------------
# 副作用与稳定性
# ---------------------------------------------------------------------------

class TestComputeSimilarityStability:
    def test_pure_function_no_side_effects(self):
        a = "hello world"
        b = "hello there"
        # 多次调用应返回相同结果
        results = [TranslationPipeline._compute_text_similarity(a, b) for _ in range(5)]
        assert len(set(results)) == 1  # 所有结果一致

    def test_inputs_not_mutated(self):
        a = "hello world"
        b = "hello there"
        a_before = a
        b_before = b
        TranslationPipeline._compute_text_similarity(a, b)
        assert a == a_before
        assert b == b_before


# ---------------------------------------------------------------------------
# 阈值契约（oscillation_detection.similarity_threshold 默认 0.95）
# ---------------------------------------------------------------------------

class TestSimilarityThresholdContract:
    def test_near_identical_above_threshold(self):
        # 95% 相似度 → 触发 oscillation（默认阈值 0.95）
        # 改 1 个字符在 20 字符中 → sim = 0.95
        a = "abcdefghijklmnopqrst"
        b = "abcdefghijklmnopqrsu"
        sim = TranslationPipeline._compute_text_similarity(a, b)
        assert sim >= 0.95

    def test_far_below_threshold(self):
        a = "completely unique text one"
        b = "totally different content two"
        sim = TranslationPipeline._compute_text_similarity(a, b)
        assert sim < 0.5
