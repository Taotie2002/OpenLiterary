"""Round 119 (T5): merge 前处理与章节标题制式（chapter_heading / postprocess_chunk_text）.

背景（D6/D7）: merged 出现 Gutenberg 页眉残留、版本行、标题粘连
（"…eBook ofAlice…"）、"CHAPTER I.Down…" 缺空格、chunk 内嵌错位标题;
章节标题制式此前无决策点（默认双语 中文主标题＋原文括注）。

测试目标（对应方案 T5-4/5）:
  1. postprocess_chunk_text: 页眉/版本行/目录行/内嵌"第N章"标题行剥离;
     正文行（含"关于第三章…"型）保留; 行序与空行不重排。
  2. chapter_heading: bilingual/zh/keep 三制式; Gutenberg 前缀清理;
     zh 缺失回退原文; 全缺回退 chapter_id; 多 chunk 取首个; 默认读 config。
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.book_lint import chapter_heading, postprocess_chunk_text, sup_run_to_int

ROOT = Path(__file__).resolve().parent.parent


def _configured_style() -> str:
    data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    return (data.get("book") or {}).get("chapter_titles", "bilingual")


# ── 1. postprocess_chunk_text ────────────────────────────────────────

def test_postprocess_strips_frontmatter_and_inner_heading():
    src_text = "\n".join([
        "# The Project Gutenberg eBook of Alice",
        "",
        "THE MILLENNIUM FULCRUM EDITION 3.0",
        "",
        "第一部 山间小屋",
        "第三章 山间小屋",
        "",
        "他说：“早上好。”",
        "关于第三章的内容，他早已忘记。",
        "目录",
        "Contents",
    ])
    out = postprocess_chunk_text(src_text, "ch03", 0)
    assert out == "\n".join([
        "", "", "第一部 山间小屋", "", "他说：“早上好。”",
        "关于第三章的内容，他早已忘记。",
    ])


def test_postprocess_identity_on_clean_text():
    t = "他说：“你好。”\n\n“早安，”她说。\n"
    assert postprocess_chunk_text(t, "ch01", 0) == t


def test_sup_run_to_int():
    assert sup_run_to_int("¹") == 1
    assert sup_run_to_int("¹²") == 12
    assert sup_run_to_int("¹²³⁴⁵⁶") == 123456


# ── 2. chapter_heading ───────────────────────────────────────────────

def _mk_chapter(root, ch, orig_h1, first_text):
    (root / "input").mkdir(exist_ok=True)
    (root / "output" / ch).mkdir(parents=True, exist_ok=True)
    (root / "input" / f"{ch}.md").write_text(f"{orig_h1}\n\n正文。\n", encoding="utf-8")
    (root / "output" / ch / f"{ch}_chunk000_final.json").write_text(
        json.dumps({"text": first_text}, ensure_ascii=False), encoding="utf-8")


def test_chapter_heading_three_styles(tmp_path):
    _mk_chapter(tmp_path, "ch01", "# CHAPTER I. Down the Rabbit-Hole",
                "# 第一章 掉进兔子洞\n\n正文。\n")
    assert chapter_heading(tmp_path, "ch01", style="bilingual") == "第一章 掉进兔子洞（Down the Rabbit-Hole）"
    assert chapter_heading(tmp_path, "ch01", style="zh") == "第一章 掉进兔子洞"
    assert chapter_heading(tmp_path, "ch01", style="keep") == "CHAPTER I. Down the Rabbit-Hole"


def test_chapter_heading_default_reads_config(tmp_path):
    _mk_chapter(tmp_path, "ch01", "# CHAPTER I. Down the Rabbit-Hole",
                "# 第一章 掉进兔子洞\n\n正文。\n")
    style = _configured_style()
    got = chapter_heading(tmp_path, "ch01")
    assert got == chapter_heading(tmp_path, "ch01", style=style)
    if style == "bilingual":
        assert got == "第一章 掉进兔子洞（Down the Rabbit-Hole）"


def test_chapter_heading_gutenberg_prefix_and_fallbacks(tmp_path):
    _mk_chapter(tmp_path, "ch02", "# The Project Gutenberg eBook of Alice",
                "正文，无标题行。\n")
    assert chapter_heading(tmp_path, "ch02", style="keep") == "Alice"
    assert chapter_heading(tmp_path, "ch02", style="bilingual") == "Alice"

    _mk_chapter(tmp_path, "ch03", "# CHAPTER II. The Pool of Tears",
                "正文，无标题行。\n")
    assert chapter_heading(tmp_path, "ch03", style="bilingual") == "The Pool of Tears"
    assert chapter_heading(tmp_path, "ch03", style="zh") == "CHAPTER II. The Pool of Tears"

    assert chapter_heading(tmp_path, "ch99", style="bilingual") == "ch99"
    assert chapter_heading(tmp_path, "ch99", style="keep") == "ch99"


def test_chapter_heading_prefers_first_chunk(tmp_path):
    _mk_chapter(tmp_path, "ch04", "# CHAPTER IV. The Rabbit Sends in a Little Bill",
                "# 第四章 兔子派来小比尔\n\n正文。\n")
    d = tmp_path / "output" / "ch04"
    (d / "ch04_chunk001_final.json").write_text(
        json.dumps({"text": "# 第四章 另一个标题\n\n正文二。\n"}, ensure_ascii=False),
        encoding="utf-8")
    assert chapter_heading(tmp_path, "ch04", style="zh") == "第四章 兔子派来小比尔"
