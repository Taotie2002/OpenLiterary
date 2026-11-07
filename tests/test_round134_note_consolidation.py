"""Round 134: 成书 lint/notes-normalize 结构修复回归。

覆盖：
- 裸标题 `# 第二章` / `# 第一部` 不再回退成英文；
- 连续堆叠的 `[^n]:` 定义按行拆单元，同词条跨 chunk 注按章合并；
- 悬空引用清理、孤儿定义转译者注；
- 跨 chunk 重复注在 merged 存在时只告警，章内重复仍失败。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.book_lint import (
    _scan_lines,
    chapter_heading,
    lint_book,
    normalize_chapter_notes,
)


def _mk_chapter(root: Path, ch: str, orig_h1: str, first_text: str) -> None:
    (root / "input").mkdir(exist_ok=True)
    (root / "output" / ch).mkdir(parents=True, exist_ok=True)
    (root / "input" / f"{ch}.md").write_text(f"{orig_h1}\n\n正文。\n", encoding="utf-8")
    (root / "output" / ch / f"{ch}_chunk000_final.json").write_text(
        json.dumps({"text": first_text}, ensure_ascii=False), encoding="utf-8")


def _mk_final(root: Path, ch: str, cid: str, text: str) -> None:
    d = root / "output" / ch
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{cid}_final.json").write_text(
        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


def test_chapter_heading_bare_part_and_chapter(tmp_path):
    _mk_chapter(tmp_path, "ch01", "# Book 1", "# 第一部\n\n正文。\n")
    assert chapter_heading(tmp_path, "ch01") == "第一部（Book 1）"

    _mk_chapter(tmp_path, "ch02", "# Chapter 2", "# 第二章\n\n正文。\n")
    assert chapter_heading(tmp_path, "ch02") == "第二章（Chapter 2）"


def test_lint_accepts_part_and_chapter_headings(tmp_path):
    merged = (
        "# 第一部（Book 1）\n\n正文一。\n\n"
        "# 第二章（Chapter 2）\n\n正文二。\n\n"
        "# 第三部（Book Three）\n\n正文三。\n"
    )
    out = tmp_path / "output"
    out.mkdir()
    (out / "translated_full.md").write_text(merged, encoding="utf-8")
    res = lint_book(project_root=str(tmp_path))
    assert res["ok"] is True, res["failures"]


def test_packed_defs_same_key_merged_and_remapped():
    body = (
        "正文甲[^a]与正文乙[^b]，都足够长的叙事内容用于测试。\n\n"
        "---\n\n"
        "[^a]: “门泰特”：指受过人类计算与推理能力强化训练者。\n"
        "[^b]: 门泰特（Mentat）指受过人类式计算与推理训练的人。\n"
    )
    new, ch, _ = normalize_chapter_notes(body, 1)
    _, defs, refs = _scan_lines(new)
    assert ch["same_key_merges"] >= 1
    assert len(defs) == 1
    assert refs == {"1"}
    assert new.count("[^1]") == 3


def test_dangling_ref_removed_and_orphan_relocated():
    body = (
        "正文提到[^missing]，但没有对应定义；另一处正文足够长以通过扫描。\n\n"
        "---\n\n"
        "[^orphan]: 一条没有引用的考据注释，讲的是某个只在此处出现的罕见典故名词。\n"
    )
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["dangling_refs_removed"] == 1
    assert "[^missing]" not in new
    assert "[^orphan]" not in new
    assert "> **译者注：** 一条没有引用的考据注释" in new


def test_cross_chunk_dup_warns_when_merged_exists(tmp_path):
    note = "[^1]: **同一注**：完全相同的注释文本，长度足够用于重复检测。\n\n正文[^1]。\n"
    _mk_final(tmp_path, "ch01", "ch01_chunk000", note)
    _mk_final(tmp_path, "ch01", "ch01_chunk001", note)
    out = tmp_path / "output"
    (out / "translated_full.md").write_text(
        "# 第一章 起点\n\n正文[^1]。\n\n[^1]: **同一注**：完全相同的注释文本，长度足够用于重复检测。\n",
        encoding="utf-8")
    res = lint_book(project_root=str(tmp_path))
    assert res["ok"] is True, res["failures"]
    assert any("重复注" in w for w in res["warnings"])


def test_same_section_same_key_still_fails(tmp_path):
    merged = (
        "# 第一章 起点\n\n"
        "正文甲[^1]与正文乙[^2]。\n\n"
        "[^1]: “门泰特”：指受过人类计算与推理训练者。\n"
        "[^2]: 门泰特（Mentat）指受过人类式计算与推理训练的人。\n"
    )
    out = tmp_path / "output"
    out.mkdir()
    (out / "translated_full.md").write_text(merged, encoding="utf-8")
    res = lint_book(project_root=str(tmp_path))
    assert res["ok"] is False
    assert any("同键重复注" in f for f in res["failures"])
