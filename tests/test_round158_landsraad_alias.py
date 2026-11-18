"""Round 158: source-gated canonical repair for the Landsraad rendering."""
from __future__ import annotations

from src.book_repair import _br_plan_canonical_aliases, _br_plan_text


def test_replaces_landsraad_alias_only_with_source_support():
    text = "雷托公爵在兰萨德的各大家族中很受欢迎；兰萨德最高议会保持沉默。\n"
    source = (
        "The Duke Leto was popular among the Great Houses of the Landsraad. "
        "The High Council of the Landsraad remained silent.\n"
    )

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert "兰萨德" not in repaired
    assert repaired.count("兰兹拉德") == 2
    assert len(changes) == 1
    assert changes[0]["rule"] == "canonical_surface_restore"
    assert changes[0]["from"] == "兰萨德"
    assert changes[0]["to"] == "兰兹拉德"
    assert changes[0]["count"] == 2
    assert changes[0]["evidence"]["source_term"] == "Landsraad"


def test_does_not_replace_landsraad_alias_without_source_term():
    text = "兰萨德这一写法不应被无源替换。\n"
    source = "An unrelated paragraph contains no faction name.\n"

    repaired, changes = _br_plan_canonical_aliases(text, source)

    assert repaired == text
    assert changes == []


def test_landsraad_repair_is_idempotent():
    repaired, _changes = _br_plan_canonical_aliases(
        "兰萨德各大家族。\n", source="The Landsraad Great Houses.\n")
    again, changes = _br_plan_canonical_aliases(
        repaired, source="The Landsraad Great Houses.\n")

    assert repaired == "兰兹拉德各大家族。\n"
    assert again == repaired
    assert changes == []


def test_plan_text_wires_landsraad_alias_repair():
    final = "雷托公爵在兰萨德的各大家族中很受欢迎。\n"
    source = "The Duke Leto was popular among the Great Houses of the Landsraad.\n"

    plan = _br_plan_text("ch01_chunk002", final, "", "", source)

    assert "兰萨德" not in plan["repaired_text"]
    assert "兰兹拉德" in plan["repaired_text"]
    assert any(change["rule"] == "canonical_surface_restore"
               for change in plan["changes"])
