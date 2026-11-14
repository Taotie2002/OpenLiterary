"""Round 151: deterministic repair of impossible residual name surfaces."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repair_residual_name_damage.py"
_SPEC = importlib.util.spec_from_file_location("repair_residual_name_damage", _SCRIPT)
repair = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(repair)


def test_invalid_lady_gaius_and_malformed_old_helen():
    text = "盖乌斯夫人看着老海伦盖乌斯·海伦·莫希阿姆。"
    repaired, changes, skipped = repair.repair_text(
        text, "Jessica looked at the old Reverend Mother.")
    assert repaired == "杰西卡夫人看着老圣母。"
    assert not skipped
    assert sum(x["count"] for x in changes) == 2


def test_paul_leto_surface_uses_source_duke_identity():
    leto, _changes, _skipped = repair.repair_text(
        "保罗·雷托公爵走了进来。", "The Duke Leto Atreides entered.")
    paul, _changes, _skipped = repair.repair_text(
        "保罗·雷托公爵走了进来。", "The Duke Paul Atreides entered.")
    assert leto == "雷托·厄崔迪公爵走了进来。"
    assert paul == "保罗·厄崔迪公爵走了进来。"


def test_ambiguous_duke_source_is_not_repaired():
    text = "保罗·雷托公爵走了进来。"
    repaired, changes, skipped = repair.repair_text(
        text, "The Duke Leto met the Duke Paul.")
    assert repaired == text
    assert not changes
    assert "ambiguous_paul_leto_without_unique_source_duke" in skipped


def test_contaminated_footnotes_require_jessica_source():
    text = "盖乌斯：保罗的母亲。提供有关盖乌斯报告的女性。"
    repaired, _changes, _skipped = repair.repair_text(
        text, "Jessica was Paul's mother. A report about Jessica.")
    assert repaired == "杰西卡：保罗的母亲。提供有关杰西卡报告的女性。"
    unchanged, _changes, _skipped = repair.repair_text(text, "Mohiam spoke.")
    assert unchanged == text


def test_malformed_truthsayer_lady_is_repaired_to_jessica():
    repaired, _changes, _skipped = repair.repair_text(
        "真言者夫人看着盖乌斯。", "Jessica looked at the Truthsayer.")
    assert repaired == "杰西卡看着盖乌斯。"


def test_residual_gaius_truthsayer_requires_truthsayer_source():
    repaired, _changes, _skipped = repair.repair_text(
        "他也怕盖乌斯。", "He too fears the Truthsayer.")
    assert repaired == "他也怕真言者。"


def test_save_your_son_and_your_woman_restores_jessica():
    repaired, _changes, _skipped = repair.repair_text(
        "我会救你的儿子和盖乌斯·海伦·莫希阿姆。",
        "I will save your son and your woman.")
    assert repaired == "我会救你的儿子和杰西卡。"


def test_sayyadina_body_token_is_restored_but_footnote_is_kept():
    text = "你不得干涉，莫希阿姆。\n\n[^1]: 盖乌斯·海伦·莫希阿姆是另一位圣母。"
    repaired, _changes, _skipped = repair.repair_text(
        text, "You must not interfere, Sayyadina.")
    assert repaired.startswith("你不得干涉，赛亚迪娜。")
    assert "盖乌斯·海伦·莫希阿姆" in repaired


def test_generic_reverend_mother_mystery_drops_mohiam_name():
    repaired, _changes, _skipped = repair.repair_text(
        "圣母盖乌斯·海伦·莫希阿姆之谜。",
        "I must probe this Reverend Mother mystery.")
    assert repaired == "圣母之谜。"
