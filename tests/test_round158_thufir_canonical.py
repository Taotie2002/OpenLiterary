"""Round 158: source-gated canonical repair for the Thufir short form."""
from __future__ import annotations

from src.book_repair import _br_plan_canonical_aliases, _br_plan_text


def test_replaces_thufir_vocative_when_source_has_thufir():
    text = "“谢谢你，图菲尔。”他说。\n"
    source = "“Thank you, Thufir,” he said.\n"

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert repaired == "“谢谢你，杜菲。”他说。\n"
    assert len(changes) == 1
    assert changes[0]["rule"] == "canonical_surface_restore"
    assert changes[0]["from"] == "图菲尔"
    assert changes[0]["to"] == "杜菲"
    assert changes[0]["evidence"]["source_term"] == "Thufir"


def test_does_not_replace_without_source_support():
    text = "“谢谢你，图菲尔。”他说。\n"
    source = "“Thank you,” he said.\n"

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert repaired == text
    assert changes == []


def test_plan_text_wires_source_gated_thufir_repair():
    final = "“谢谢你，图菲尔。”\n"
    source = "“Thank you, Thufir,” he said.\n"

    plan = _br_plan_text("ch03_chunk091", final, "", "", source)

    assert "图菲尔" not in plan["repaired_text"]
    assert "杜菲" in plan["repaired_text"]
    assert any(change["rule"] == "canonical_surface_restore"
               for change in plan["changes"])
