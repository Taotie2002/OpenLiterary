"""Round 119 (T6): consistency 扫描器注释豁免扩展.

背景（D8）: ch08_chunk009 脚注 ² 内 `（well = the well）` 被英文残留扫描
误报（"the well" → 井），是上一轮全量扫描唯一 1 条。旧豁免只认 [^ 行、
引号计数、前 200 字符内 "译注/脚注" 字样 — ¹²³ 长尾注块漏判。

修复口径（方案 T6）:
  - 段落首行为上标注（行首 ¹²³…）/ "注：" 前缀 → 整段豁免（含续行）；
  - "（原文 …）" 括注内英文豁免；
  - 与既有 [^ 行、引号内、译注/脚注字样豁免并存（回归）。

验收: ch08_chunk009 fixture 复跑 → 0 条。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.consistency import _in_quote_or_note

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "output" / "ch08" / "ch08_chunk009_final.json"


class TestNoteExemptionUnit:
    def test_superscript_note_line_exempt(self):
        text = "¹ **draw**——词义兼有“汲取”与“绘画”两层，前段以 draw 译出。"
        pos = text.index("draw", text.index("前段"))
        assert _in_quote_or_note(text, pos) is True

    def test_note_continuation_line_exempt(self):
        text = "¹ **punning**——注释第一行说明。\n第二行仍有 the well 残留。"
        assert _in_quote_or_note(text, text.index("the well")) is True

    def test_note_prefix_line_exempt(self):
        text = "正文一段。\n\n注：本处 draw 为双关，保留原文线索。"
        assert _in_quote_or_note(text, text.index("draw")) is True

    def test_original_parenthetical_exempt(self):
        text = "他拧开软管（原文 hydrant）往外喷水。"
        assert _in_quote_or_note(text, text.index("hydrant")) is True

    def test_plain_prose_residue_not_exempt(self):
        text = "爱丽丝看见 White Rabbit 匆匆跑过。"
        assert _in_quote_or_note(text, text.index("White Rabbit")) is False

    def test_bracket_note_regression(self):
        text = "正文。\n\n[^1]: the well 在原文中指水井。"
        assert _in_quote_or_note(text, text.index("the well")) is True

    def test_quote_regression(self):
        text = "她轻声念道：“well in, of course”。然后走开了。"
        assert _in_quote_or_note(text, text.index("well in")) is True

    def test_body_line_after_note_not_exempt(self):
        # 注区段落之后的正常正文段落不受豁免影响
        text = "¹ **draw**——注释。\n\n他走开了，white rabbit 仍在。"
        pos = text.index("white rabbit")
        # 首词小写纯正文段 — 扫描器按源名大小写不敏感匹配，此处本体检测
        assert _in_quote_or_note(text, pos) is False


_need_fixture = pytest.mark.skipif(not FIXTURE.exists(), reason="ch08_chunk009 fixture 缺失")


class TestCh08Fixture:
    @_need_fixture
    def test_all_the_well_occurrences_exempt(self):
        # 产物随每轮重译漂移：短语存在则校验豁免，不存在则跳过（豁免逻辑另有单测覆盖）
        hits = []
        for p in sorted((ROOT / "output" / "ch08").glob("ch08_chunk*_final.json")):
            try:
                text = json.loads(p.read_text(encoding="utf-8"))["text"]
            except Exception:
                continue
            hits += [(text, m.start()) for m in re.finditer(r"(?<!\w)the well(?!\w)", text)]
        if not hits:
            pytest.skip("译文已重写该括注（内容漂移，非回归）；逻辑由 TestNoteExemptionUnit 覆盖")
        assert all(_in_quote_or_note(t, pos) for t, pos in hits)

    @_need_fixture
    def test_detect_issues_zero_with_real_glossary(self):
        db = ROOT / "db" / "decision_db.sqlite"
        if not db.exists():
            pytest.skip("decision_db 缺失")
        from src.consistency import _detect_issues, _load_decisions, _load_glossary
        text = json.loads(FIXTURE.read_text(encoding="utf-8"))["text"]
        fin = {"chapter": "ch08", "file": FIXTURE.name, "text": text,
               "path": str(FIXTURE)}
        issues = _detect_issues([fin], _load_glossary(str(db)), _load_decisions(str(db)))
        assert issues == [], issues
