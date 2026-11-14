"""Round 153: source-grounded final repairs."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "repair_round153_final_gaps.py")
_SPEC = importlib.util.spec_from_file_location("repair_round153_final_gaps", _SCRIPT)
repair = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(repair)


def test_arrakeen_and_customs_are_source_grounded():
    text = ("“我们已经同这些民族联合了。”\n“水税不一样。”\n"
            "“他们还会帮我们抵达阿拉基斯吗？”")
    fixed, changes, skipped = repair.repair_text("ch04_chunk024", text)
    assert "这些人" in fixed
    assert "习俗不一样" in fixed
    assert "抵达阿拉肯吗" in fixed
    assert len(changes) == 3
    assert not skipped


def test_crowd_is_not_nation():
    text = ("随着民族看到仆人们把死者抬走，人群沸腾。"
            "正确的。民族简直像发了疯。不能就这样让民族回家。")
    fixed, changes, skipped = repair.repair_text("ch05_chunk113", text)
    assert "随着人们看到仆人们" in fixed
    assert "人群简直像发了疯" in fixed
    assert "让人们回家" in fixed
    assert "民族" not in fixed
    assert len(changes) == 3
    assert not skipped


def test_truncated_final_restores_complete_literary_body():
    literary = ("第一段完整内容。\n\n第二段也完整，并以句号结束。\n\n"
                "[^1]: 脚注。")
    truncated = "第一段完整内容。\n\n公爵朝儿子"
    fixed, changes, skipped = repair.repair_text(
        "ch01_chunk072", truncated, literary)
    assert fixed == literary
    assert any(c["rule"] == "restore_complete_body" for c in changes)
    assert not skipped


def test_complete_final_is_not_replaced():
    literary = "第一段。\n\n第二段完整。\n\n[^1]: 注。"
    final = "第一段。\n\n第二段完整。"
    fixed, changes, skipped = repair.repair_text(
        "ch01_chunk072", final, literary)
    assert fixed == final
    assert not changes
    assert skipped


def test_duplicate_title_is_collapsed():
    fixed, changes, skipped = repair.repair_text(
        "ch02_chunk042", "雷托公爵公爵靠在墙上。")
    assert fixed == "雷托公爵靠在墙上。"
    assert changes[0]["rule"] == "duplicate_title_after_name_unification"
    assert not skipped
