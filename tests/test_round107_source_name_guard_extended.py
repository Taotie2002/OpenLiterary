"""Round 107: source_name 护栏扩展 — ch03 倒查补充.

ch03 chunk004 失败根因:
  实体登记表 source_names 污染:
  - "an old friend of yours" (描述性短语) 被登记为 绅士乔 的指称
  - "he/his/him", "himself" (代词) 被登记为 安东尼·凯德 的指称
  - "this young fellow", "my boy" (描述性短语) 被登记
  - "Gentleman Joe", "Jimmy McGrath" (其他实体的专名) 被跨实体污染

  judge 要求把 "an old friend of yours" 统一译为 "绅士乔", rewriter 无法对
  描述性短语做这种修改, 输出相同文本 → REWRITER_STAGNATED → PERMANENTLY_FAILED.

护栏扩展:
  1. _NON_ENTITY_SOURCE_NAMES: 加代词/反身代词/描述性短语
  2. _NON_ENTITY_POSSESSIVE_NOUNS: 加 boy/man/fellow/gentleman/lady
  3. _NON_ENTITY_PRONOUN_RE: 代词/反身代词 (含斜杠分隔)
  4. _NON_ENTITY_DESCR_RE: 冠词+形容词+名词+of+代词
  5. _NON_ENTITY_THIS_THAT_RE: this/that+形容词+名词

测试覆盖:
  - 新增的无效 source_names 被护栏拦截
  - 合法的 source_names 仍通过
  - 两版护栏行为一致
"""
from __future__ import annotations

import pytest


# ch03 倒查发现的无效 source_names (应被护栏拦截)
INVALID_SOURCE_NAMES = [
    "an old friend of yours",
    "a friend of yours",
    "an old friend",
    "he/his/him",
    "she/her/her",
    "it/its",
    "himself",
    "herself",
    "myself",
    "my boy",
    "my man",
    "my friend",
    "my old friend",
    # Round 111: 限定词+全小写普通名词短语统一判无效（ch09 chunk004 "the stairs"
    # 硬失败根因）。Round 107 曾以"特定语境下是专名指称"保留 "the waiter"，
    # 但该理由正是 ch03/ch06/ch07/ch09 反复硬失败的来源 — 描述性指称不应强制
    # 替换为规范主名（中文可用侍者/服务员/侍应生等多种自然表达）。
    "the waiter",
    "this young fellow",
    "that young fellow",
    "this man",
    "that woman",
    "his boy",
    "her man",
    "his fellow",
]

# 合法的 source_names (应通过护栏)
VALID_SOURCE_NAMES = [
    "Gentleman Joe",
    "Anthony Cade",
    "Mr. Cade",
    "Joe",
    "Jimmy McGrath",
    "Lady Eileen",
    "Baron Lollipop",
    "King Victor",
]


class TestSourceNameGuardExtended:
    @pytest.mark.parametrize("sn", INVALID_SOURCE_NAMES)
    def test_invalid_source_names_blocked(self, sn):
        """ch03 倒查发现的无效 source_names 应被护栏拦截."""
        from src.pipeline import TranslationPipeline
        assert not TranslationPipeline._is_valid_entity_source_name(sn), \
            f"{sn!r} 应被护栏拦截"

    @pytest.mark.parametrize("sn", VALID_SOURCE_NAMES)
    def test_valid_source_names_pass(self, sn):
        """合法的 source_names 应通过护栏."""
        from src.pipeline import TranslationPipeline
        assert TranslationPipeline._is_valid_entity_source_name(sn), \
            f"{sn!r} 应通过护栏"

    @pytest.mark.parametrize("sn", [
        "this country",
        "this house",
        "this passage",
        "that town",
        "that city",
        "this place",
    ])
    def test_this_that_single_noun_blocked(self, sn):
        """Round 107 P8: this/that + 单个普通名词 (2 词) 应被护栏拦截."""
        from src.pipeline import TranslationPipeline
        assert not TranslationPipeline._is_valid_entity_source_name(sn), \
            f"{sn!r} 应被护栏拦截"

    @pytest.mark.parametrize("sn", [
        "Like all women",
        "like a man",
        "like most people",
        "like his father before him",
    ])
    def test_like_phrases_blocked(self, sn):
        """Round 107 P3: like + 名词短语 (类比/比较) 应被护栏拦截."""
        from src.pipeline import TranslationPipeline
        assert not TranslationPipeline._is_valid_entity_source_name(sn), \
            f"{sn!r} 应被护栏拦截"

    def test_monolith_and_scattered_match(self):
        """单体版与散装版护栏行为一致."""
        from src.pipeline import TranslationPipeline as Scattered
        from src.translator_agent import TranslationPipeline as Monolith

        all_test = INVALID_SOURCE_NAMES + VALID_SOURCE_NAMES
        for sn in all_test:
            scattered_result = Scattered._is_valid_entity_source_name(sn)
            monolith_result = Monolith._is_valid_entity_source_name(sn)
            assert scattered_result == monolith_result, \
                f"{sn!r}: 散装版={scattered_result}, 单体版={monolith_result}"
