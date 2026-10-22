"""Round 43: 单元测试 - utils/text_match.py.

覆盖目标：
  - extract_reject_categories: 类别抽取、dedup、顺序、空值/无冒号/前置冒号边界
  - case_insensitive_find: ASCII 边界、大小写不敏感、非 ASCII 直匹配、空值

回归保护：
  - Round 22/26 边界（空 reason_text / ASCII `I` 触发）
  - Round 34 thin shim 签名 (as_set=False→list, as_set=True→set)
"""
from __future__ import annotations

import pytest

from src.utils.text_match import extract_reject_categories, case_insensitive_find


# ---------------------------------------------------------------------------
# extract_reject_categories
# ---------------------------------------------------------------------------

class TestExtractRejectCategoriesAsList:
    def test_empty_string_returns_empty_list(self):
        assert extract_reject_categories("") == []

    def test_none_returns_empty_list(self):
        assert extract_reject_categories(None) == []

    def test_single_category(self):
        assert extract_reject_categories("直译残留: conf=0.92") == ["直译残留"]

    def test_multiple_categories_preserves_order(self):
        text = "直译残留: noul=0.5; 关键语义丢失: noul=0.6; 风格失真: conf=0.4"
        assert extract_reject_categories(text) == ["直译残留", "关键语义丢失", "风格失真"]

    def test_dedup_keeps_first_occurrence(self):
        text = "直译残留: a; 风格失真: b; 直译残留: c"
        result = extract_reject_categories(text)
        assert result == ["直译残留", "风格失真"]
        assert len(result) == 2

    def test_segment_without_colon_ignored(self):
        text = "没有冒号的描述; 直译残留: conf=0.9"
        assert extract_reject_categories(text) == ["直译残留"]

    def test_segment_with_leading_colon_ignored(self):
        # "idx > 0" 要求冒号不在第一位
        text = ": conf=0.5; 直译残留: conf=0.9"
        assert extract_reject_categories(text) == ["直译残留"]

    def test_trailing_semicolon_handled(self):
        text = "直译残留: a; 风格失真: b;"
        assert extract_reject_categories(text) == ["直译残留", "风格失真"]

    def test_extra_whitespace_stripped(self):
        text = "  直译残留  : a;   风格失真  : b"
        assert extract_reject_categories(text) == ["直译残留", "风格失真"]

    def test_only_semicolons_returns_empty(self):
        assert extract_reject_categories(";;;") == []


class TestExtractRejectCategoriesAsSet:
    def test_empty_string_returns_empty_set(self):
        assert extract_reject_categories("", as_set=True) == set()

    def test_none_returns_empty_set(self):
        assert extract_reject_categories(None, as_set=True) == set()

    def test_dedup(self):
        text = "直译残留: a; 直译残留: b; 风格失真: c"
        result = extract_reject_categories(text, as_set=True)
        assert result == {"直译残留", "风格失真"}
        assert isinstance(result, set)

    def test_single_category(self):
        result = extract_reject_categories("直译残留: a", as_set=True)
        assert result == {"直译残留"}
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# case_insensitive_find
# ---------------------------------------------------------------------------

class TestCaseInsensitiveFind:
    def test_ascii_exact_match(self):
        assert case_insensitive_find("Luffy", "海賊王中的 Luffy 出场了") == "Luffy"

    def test_ascii_case_insensitive(self):
        assert case_insensitive_find("Luffy", "the character luffy is here") == "luffy"

    def test_ascii_word_boundary_rejects_partial(self):
        # "ruffy" 不是 "Luffy" - 边界检查应拒绝
        assert case_insensitive_find("Luffy", "fluffy fur") is None

    def test_ascii_word_boundary_misses_but_fallback_substring_returns_name(self):
        # "Luffyxxx" 中 word boundary 正则不匹配（"Luffy" 后是 "xxx"），
        # 但 fallback 精确子串检测命中 — 返回 name 本身。
        # 这是 Round 34 的有意契约：保证 name attribution 一定找得到。
        assert case_insensitive_find("Luffy", "Luffyxxx 出现") == "Luffy"

    def test_ascii_word_boundary_prefix_misses_but_fallback_returns_name(self):
        # "xxxLuffy" 中 word boundary 正则不匹配（前是字母数字），
        # fallback 命中，返回 name。
        assert case_insensitive_find("Luffy", "前缀 xxxLuffy") == "Luffy"

    def test_ascii_not_in_haystack_returns_none(self):
        assert case_insensitive_find("Zoro", "Luffy is here") is None

    def test_non_ascii_exact_substring(self):
        # 非 ASCII 直接做 substring 匹配
        assert case_insensitive_find("鲁夫", "船长鲁夫出发了") == "鲁夫"

    def test_non_ascii_not_in_haystack(self):
        assert case_insensitive_find("鲁夫", "娜美登场") is None

    def test_empty_name_returns_none(self):
        assert case_insensitive_find("", "anything") is None

    def test_none_name_returns_none(self):
        assert case_insensitive_find(None, "anything") is None

    def test_none_haystack_returns_none(self):
        assert case_insensitive_find("Luffy", None) is None

    def test_empty_haystack_returns_none(self):
        assert case_insensitive_find("Luffy", "") is None

    def test_mixed_case_word_boundary_misses_fallback_returns(self):
        # Round 26 回归保护: ASCII `I` 在 `Isolation` 中 word boundary 不匹配，
        # 但 fallback 精确子串命中，返回 "I"。
        assert case_insensitive_find("I", "Isolation ward") == "I"

    def test_mixed_case_word_boundary_still_matches_standalone(self):
        # 但独立的大写 I 应匹配
        assert case_insensitive_find("I", "I am here") is not None

    def test_ascii_with_punctuation_boundary(self):
        # 标点不算 word char
        result = case_insensitive_find("Luffy", "(Luffy) and friends")
        assert result is not None
        assert result.lower() == "luffy"
