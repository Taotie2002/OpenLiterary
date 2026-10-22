"""Round 59: 单元测试 - src.splitters._normalize_chapter_title.

保护目标（章节标题规范化）：
  - 罗马数字 + 英文紧接（如 II.The） → 插入空格（II. The）
  - 阿拉伯数字 + 英文紧接（如 1.Introduction） → 插入空格
  - 已规范标题保持不变
  - 空字符串 / 纯中文保持不变
  - 首尾空白被 strip
"""
from __future__ import annotations

import pytest

from src.splitters import _normalize_chapter_title


class TestRomanNumeralNormalization:
    def test_roman_numeral_with_dot_no_space(self):
        assert _normalize_chapter_title("II.The Machine") == "II. The Machine"

    def test_roman_numeral_already_spaced_unchanged(self):
        assert _normalize_chapter_title("II. The Machine") == "II. The Machine"

    def test_roman_numeral_iii(self):
        assert _normalize_chapter_title("III.The Story") == "III. The Story"

    def test_roman_numeral_xv_with_possessive(self):
        assert _normalize_chapter_title("XV.The Time Traveller's Return") == "XV. The Time Traveller's Return"


class TestArabicNumeralNormalization:
    def test_single_digit_with_letter(self):
        assert _normalize_chapter_title("1.Introduction") == "1. Introduction"

    def test_double_digit_with_letter(self):
        assert _normalize_chapter_title("10.Next Chapter") == "10. Next Chapter"

    def test_zero_digit_with_letter(self):
        assert _normalize_chapter_title("0.Preface") == "0. Preface"

    def test_three_digit_with_letter(self):
        assert _normalize_chapter_title("100.Conclusion") == "100. Conclusion"


class TestNonLetterAfterDot:
    def test_dot_followed_by_lowercase_unchanged(self):
        # 仅当后续是大写字母才插入空格
        assert _normalize_chapter_title("II.the machine") == "II.the machine"

    def test_dot_followed_by_number_unchanged(self):
        # 后续是数字，不插入
        assert _normalize_chapter_title("II.1") == "II.1"

    def test_dot_followed_by_chinese_unchanged(self):
        # 后续是中文，不插入
        assert _normalize_chapter_title("Chapter.第一章") == "Chapter.第一章"


class TestEdgeCases:
    def test_empty_string_returns_empty(self):
        assert _normalize_chapter_title("") == ""

    def test_none_input_returns_none(self):
        # 文档说非空才处理，None 也按"不变"返回
        # 实际代码：if not title: return title
        assert _normalize_chapter_title(None) is None

    def test_whitespace_stripped(self):
        assert _normalize_chapter_title("  II.The Machine  ") == "II. The Machine"

    def test_no_dot_unchanged(self):
        assert _normalize_chapter_title("Chapter One") == "Chapter One"

    def test_only_dot_unchanged(self):
        # 边界：单字符 "." 不会触发（[A-Za-z0-9]+ 至少 1 个字符）
        # 但代码不要求匹配，所以 "." 原样返回
        assert _normalize_chapter_title(".") == "."

    def test_already_normalized_unchanged(self):
        assert _normalize_chapter_title("V. The End") == "V. The End"

    def test_multiple_dots_only_first_processed(self):
        # 正则只处理第一个匹配（re.sub 默认不处理重叠）
        # "1.2.Chapter" — 第一个匹配是 "1.2"，但后续是小写 2
        # 所以不会触发
        assert _normalize_chapter_title("1.2.Chapter") == "1.2.Chapter"


class TestIdempotency:
    def test_already_normalized_then_normalized_unchanged(self):
        # 二次归一化结果不变（幂等性）
        first = _normalize_chapter_title("II.The Machine")
        second = _normalize_chapter_title(first)
        assert first == second

    def test_normalization_idempotent_arabic(self):
        first = _normalize_chapter_title("1.Introduction")
        second = _normalize_chapter_title(first)
        assert first == second
