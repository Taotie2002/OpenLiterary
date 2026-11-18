"""Round 158: English source names in legitimate glosses are not residue."""
from __future__ import annotations

from src.pipeline import TranslationPipeline


def _pipeline() -> TranslationPipeline:
    return TranslationPipeline.__new__(TranslationPipeline)


def test_canonical_with_english_gloss_is_allowed():
    pipeline = _pipeline()

    assert pipeline._entity_source_name_residue_forms(
        "厄拉科斯（Arrakis）的风暴将至。", "Arrakis", "厄拉科斯",
    ) == []
    assert pipeline._entity_source_name_residue_forms(
        "Arrakis（厄拉科斯）的风暴将至。", "Arrakis", "厄拉科斯",
    ) == []


def test_footnote_source_name_is_allowed_but_body_residue_is_not():
    pipeline = _pipeline()

    footnote_only = (
        "厄拉科斯的风暴将至。\n\n"
        "[^1]: 原文 Arrakis 为行星名。"
    )
    assert pipeline._entity_source_name_residue_forms(
        footnote_only, "Arrakis", "厄拉科斯",
    ) == []

    assert pipeline._entity_source_name_residue_forms(
        "Arrakis 的风暴将至。", "Arrakis", "厄拉科斯",
    ) == ["Arrakis"]


def test_parenthetical_canonical_uses_stripped_gloss_base():
    pipeline = _pipeline()

    assert pipeline._entity_source_name_residue_forms(
        "贝尼·杰瑟里特（Bene Gesserit）的仪式。",
        "Bene Gesserit",
        "贝尼·杰瑟里特（Bene Gesserit）",
    ) == []
