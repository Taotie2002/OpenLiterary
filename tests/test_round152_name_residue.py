"""Round 152: exact repairs for remaining title/name aliasing."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "repair_round152_name_residue.py")
_SPEC = importlib.util.spec_from_file_location("repair_round152_name_residue", _SCRIPT)
repair = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(repair)


def test_truthsayer_is_not_gaius():
    text = ("你听说过盖乌斯之药[^4]吗？一位盖乌斯[^6]天赋。"
            "任何盖乌斯都看不见。\n"
            "[^4]: 盖乌斯之药：说明。\n"
            "[^5]: 真言恍惚：盖乌斯服用药物。\n"
            "[^6]: 盖乌斯：说明。")
    fixed, changes, skipped = repair.repair_text("ch01_chunk019", text)
    assert "真言者之药" in fixed
    assert "任何真言者" in fixed
    assert not skipped
    assert changes


def test_reverend_mother_is_not_helen():
    text = ("海伦盖乌斯·海伦·莫希阿姆独自坐在入口正对面。\n"
            "[^1]: **海伦**：贝尼·杰瑟里特姐妹会中高阶女性成员或女先知的敬称。")
    fixed, _changes, skipped = repair.repair_text("ch07_chunk012", text)
    assert fixed.startswith("圣母独自坐在入口正对面。")
    assert "**圣母**" in fixed
    assert not skipped


def test_fenring_is_not_helen():
    text = ("哈西米尔·海伦伯爵。海伦应对导致此事。海伦是沙达姆四世的朋友。"
            "海伦伯爵担任过的帝国职任包括代理人。")
    fixed, _changes, skipped = repair.repair_text("ch07_chunk146", text)
    assert fixed == ("哈西米尔·芬林伯爵。芬林应对导致此事。芬林是沙达姆四世的朋友。"
                     "芬林伯爵担任过的帝国职任包括代理人。")
    assert not skipped


def test_mother_is_jessica_not_mohiam():
    text = ("男爵承认过，盖乌斯·海伦·莫希阿姆——那个贝尼·杰瑟里特姐妹会"
            "——曾经是他的武器。而那只能意味着一切的终结——甚至包括"
            "盖乌斯自己的儿子。")
    fixed, _changes, skipped = repair.repair_text("ch06_chunk024", text)
    assert "那个贝尼·杰瑟里特女巫" in fixed
    assert "她自己的儿子" in fixed
    assert "莫希阿姆" not in fixed
    assert not skipped


def test_paul_leto_action_is_source_grounded():
    fixed, _changes, skipped = repair.repair_text(
        "ch03_chunk100", "保罗把左手放在护盾带上。")
    assert fixed == "雷托把左手放在护盾带上。"
    assert not skipped


def test_atreides_transliterations_are_unified():
    fixed, changes, skipped = repair.repair_text(
        "ch99_chunk000", "阿特雷迪斯家族、阿特雷德斯、阿特雷德、亚崔迪、厄崔迪。")
    assert fixed == "厄崔迪家族、厄崔迪、厄崔迪、厄崔迪、厄崔迪。"
    assert changes[0]["count"] == 4
    assert not skipped


def test_truthsayer_dialogue_is_not_a_question():
    text = ("“真言者？干完今晚这桩活，我可不愿去想还要面对那个杰西卡，”疤脸说。"
            "真言者听着他们的话，观察着，记录着。")
    fixed, changes, skipped = repair.repair_text("ch03_chunk114", text)
    assert fixed.startswith("“我嘛，干完今晚这桩活")
    assert "面对那个真言者" in fixed
    assert fixed.endswith("杰西卡听着他们的话，观察着，记录着。")
    assert len(changes) == 2
    assert not skipped


def test_legitimate_mohiam_name_is_not_rewritten():
    text = "盖乌斯·海伦·莫希阿姆看着杰西卡。"
    fixed, changes, skipped = repair.repair_text("ch99_chunk001", text)
    assert fixed == text
    assert not changes
    assert not skipped


def test_missing_exact_context_is_reported_not_guessed():
    fixed, changes, skipped = repair.repair_text(
        "ch03_chunk100", "文本已经改变。")
    assert fixed == "文本已经改变。"
    assert not changes
    assert skipped and skipped[0]["rule"] == "leto_action_not_paul"
