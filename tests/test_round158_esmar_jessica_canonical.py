"""Round 158: source-gated canonical repair for Esmar Tuek and Lady Jessica."""
from __future__ import annotations

from src.book_repair import _br_plan_canonical_aliases, _br_plan_text


def test_esmar_spelling_drift_is_repaired_only_with_source_evidence():
    text = "那个走私者叫图克，埃斯马·图克。\n"
    source = "The smuggler’s called Tuek, Esmar Tuek.\n"

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert "埃斯马·图克" not in repaired
    assert "埃斯玛·图克" in repaired
    assert len(changes) == 1
    assert changes[0]["rule"] == "canonical_surface_restore"
    assert changes[0]["from"] == "埃斯马·图克"
    assert changes[0]["to"] == "埃斯玛·图克"
    assert changes[0]["evidence"]["source_term"] == "Esmar Tuek"


def test_esmar_spelling_drift_is_left_untouched_without_source_evidence():
    text = "那个走私者叫图克，埃斯马·图克。\n"
    source = "The smuggler’s called Tuek.\n"

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert repaired == text
    assert changes == []


def test_lady_jessica_shadow_title_is_repaired_only_with_source_evidence():
    text = "杜菲·哈瓦特怀疑过杰西卡女士。\n"
    source = "Thufir Hawat suspected the Lady Jessica.\n"

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert "杰西卡女士" not in repaired
    assert "杰西卡夫人" in repaired
    assert len(changes) == 1
    assert changes[0]["rule"] == "canonical_surface_restore"
    assert changes[0]["from"] == "杰西卡女士"
    assert changes[0]["to"] == "杰西卡夫人"
    assert changes[0]["evidence"]["source_term"] == "Lady Jessica"


def test_lady_jessica_shadow_title_is_left_untouched_without_source_evidence():
    text = "她被称为杰西卡女士。\n"
    source = "Jessica waited in the hall.\n"

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert repaired == text
    assert changes == []


def test_book_repair_wires_both_canonical_rules_before_delivery():
    final = "埃斯马·图克说，杰西卡女士保持沉默。\n"
    source = "Esmar Tuek spoke. Lady Jessica remained silent.\n"

    plan = _br_plan_text("ch04_chunk098", final, "", "", source)

    assert "埃斯马·图克" not in plan["repaired_text"]
    assert "杰西卡女士" not in plan["repaired_text"]
    assert "埃斯玛·图克" in plan["repaired_text"]
    assert "杰西卡夫人" in plan["repaired_text"]
