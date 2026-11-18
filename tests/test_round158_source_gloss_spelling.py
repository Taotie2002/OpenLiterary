"""Round 158: deterministic repair of source-term typos in explicit glosses."""
from __future__ import annotations

from src.book_repair import (
    _br_plan_source_gloss_spelling,
    _br_plan_text,
)


def test_restores_unique_one_letter_typo_in_footnote_and_parenthetical():
    text = (
        '正文提到“蒸馏帐篷”（stiltent）[^1]。\n\n'
        '[^1]: “蒸馏帐篷”（stiltent）是弗雷曼人的设备。\n'
    )
    source = (
        'He crawled out of the darkened stilltent. The stilltent held the '
        'spare water.'
    )

    repaired, changes = _br_plan_source_gloss_spelling(text, source)

    assert "stiltent" not in repaired
    assert repaired.count("stilltent") == 2
    assert len(changes) == 1
    assert changes[0]["rule"] == "source_gloss_typo_restore"
    assert changes[0]["to"] == "stilltent"
    assert changes[0]["evidence"]["source_term"] == "stilltent"


def test_exact_source_gloss_is_not_changed():
    text = "他爬上蒸馏帐篷（stilltent）。\n"
    source = "He crawled out of the stilltent."

    repaired, changes = _br_plan_source_gloss_spelling(text, source)

    assert repaired == text
    assert changes == []


def test_ambiguous_near_source_spellings_are_left_untouched():
    text = "机械术语（stiltent）保持不变。\n"
    source = "The stilltent and the spilltent were both present."

    repaired, changes = _br_plan_source_gloss_spelling(text, source)

    assert repaired == text
    assert changes == []


def test_gloss_without_a_close_source_token_is_left_untouched():
    text = "脚注提到（Mahdi）这一称号。\n"
    source = "Muad'Dib is the name used by the Fremen."

    repaired, changes = _br_plan_source_gloss_spelling(text, source)

    assert repaired == text
    assert changes == []


def test_valid_singular_gloss_is_not_forced_to_source_plural():
    text = "“预言与弥赛亚（Messiah）”涉及末世救世主。\n"
    source = "About prophecies and messiahs, the Fremen dreamed."

    repaired, changes = _br_plan_source_gloss_spelling(text, source)

    assert repaired == text
    assert changes == []


def test_plan_text_wires_in_the_gloss_repair_before_delivery():
    final = "正文提到（stiltent）。\n\n[^1]: （stiltent）是设备。\n"
    source = "He crawled out of the stilltent. The stilltent held water."

    plan = _br_plan_text("ch04_chunk000", final, "", "", source)

    assert "stiltent" not in plan["repaired_text"]
    assert plan["repaired_text"].count("stilltent") == 2
    assert any(change["rule"] == "source_gloss_typo_restore"
               for change in plan["changes"])
