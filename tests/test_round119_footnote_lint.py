"""Round 119 (T5): 成书 lint — 脚注制式/orphan/重复注/策略元语/页眉残留/标题制式.

背景（D5/D6/D9）: 脚注三制式混用（¹²³ / [^N] / 编号列表）; 策略元语
10 处入正文（"保留并加注""按约束"）; ch08_chunk000 6 定义 0 引用;
merged 页眉残留（Gutenberg/版本行/标题粘连/缺空格）; D9 悬空承诺无检测。

测试目标（对应方案 T5 验收 + T8 守卫）:
  1. lint_texts 逐文本: 直引号 / 「」 / 上标定义 / 悬空定义 / 策略元语 /
     同键重复注 → failures; 无定义引用 → warnings（fail-closed 分流）。
  2. lint_book 合成项目: 全绿案例; Gutenberg 残留 + 标题制式 → 红;
     跨 chunk 重复注（同文/近似/不同）→ 失败/警告分流; merged 缺失告警。
  3. 报告 output/book_lint.md 落盘 + 结论行。
  4. prompt 禁用策略元语守卫（D5 根因：prompt 未禁 → 现双落点各 2 处）。
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.book_lint import lint_book, lint_texts

ROOT = Path(__file__).resolve().parent.parent


def _configured_style() -> str:
    data = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
    return (data.get("book") or {}).get("chapter_titles", "bilingual")


def _mk(tmp_path, finals, merged=None):
    out = tmp_path / "output"
    for ch, cid, text in finals:
        d = out / ch
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cid}_final.json").write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    if merged is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "translated_full.md").write_text(merged, encoding="utf-8")
    return out


# ── 1. lint_texts 逐文本 ─────────────────────────────────────────────

def test_lint_texts_problem_classes():
    text = (
        '他说"old style"并喊「快跑」。\n\n'
        '正文引用[^1]和[^2]，以及元语"保留并加注""按约束"。\n\n'
        '¹ 上标定义。\n\n'
        '[^9]: 悬空定义。\n\n'
        '[^1]: **凯撒**：罗马统帅。\n\n'
        '[^2]: **凯撒**：另一种写法的注释文本。\n'
    )
    res = lint_texts([("lbl", text)])
    assert res["ok"] is False
    joined = "\n".join(res["failures"])
    assert len(res["failures"]) == 6, res["failures"]
    for s in ("直引号 6 处", "「」/『』引号 2 处", "上标脚注定义 1 行",
              "悬空脚注定义 1 条: ['9']", "策略元语 2 处", "同键重复注释: ['凯撒']"):
        assert s in joined, s


def test_lint_texts_dangling_ref_is_warning_not_failure():
    res = lint_texts([("lbl", "正文引用了[^7]，但没有定义。\n")])
    assert res["ok"] is True and res["failures"] == []
    assert res["warnings"] == ["lbl: 无定义引用 ['7']"], res["warnings"]


# ── 2. lint_book 合成项目 ────────────────────────────────────────────

_CLEAN_FINAL = "# 第一章 起点\n\n他说：“你好。”然后[^1]。\n\n[^1]: 说明。\n"


def test_lint_book_green_project(tmp_path):
    merged = "# 全书\n\n# 第一章 起点\n\n他说：“你好。”然后[^1]。\n\n[^1]: 说明。\n"
    _mk(tmp_path, [("ch01", "ch01_chunk000", _CLEAN_FINAL)], merged=merged)
    res = lint_book(project_root=str(tmp_path))
    assert res["ok"] is True, res["failures"]
    assert res["failures"] == [] and res["warnings"] == []
    assert res["stats"]["finals"] == 1 and res["stats"]["headings"] == 2
    rep = (tmp_path / "output" / "book_lint.md").read_text(encoding="utf-8")
    assert "全绿" in rep


def test_lint_book_red_on_residue_orphan_and_heading(tmp_path):
    merged = ("# The Project Gutenberg eBook of Alice\n\n"
              "The Project Gutenberg eBook\n\n"
              "# Chapter One\n\n正文。\n")
    _mk(tmp_path, [("ch01", "ch01_chunk000", "正文。\n\n¹ 无引用。\n")], merged=merged)
    res = lint_book(project_root=str(tmp_path))
    assert res["ok"] is False
    joined = "\n".join(res["failures"])
    assert "Gutenberg 英文残留 2 处" in joined
    assert "悬空脚注定义 1 条: ['1']" in joined
    if _configured_style() in ("bilingual", "zh"):
        assert "章节标题不符合制式" in joined
    rep = (tmp_path / "output" / "book_lint.md").read_text(encoding="utf-8")
    assert "未通过" in rep


def test_lint_book_duplicate_notes_across_chunks(tmp_path):
    finals = [
        ("ch01", "ch01_chunk000", "[^1]: **同一注**：完全相同的文本。\n\n正文[^1]。\n"),
        ("ch01", "ch01_chunk001", "[^1]: **同一注**：完全相同的文本。\n\n正文[^1]。\n"),
        ("ch02", "ch02_chunk000", "[^1]: **凯撒**：罗马统帅，也是作家和演说家。\n\n正文[^1]。\n"),
        ("ch02", "ch02_chunk001", "[^1]: **凯撒**：罗马统帅，同时是作家和演说家。\n\n正文[^1]。\n"),
        ("ch03", "ch03_chunk000", "[^1]: **凯撒**：罗马统帅。\n\n正文[^1]。\n"),
        ("ch03", "ch03_chunk001", "[^1]: **凯撒**：与沙拉有关的另一种解释。\n\n正文[^1]。\n"),
    ]
    _mk(tmp_path, finals)
    res = lint_book(project_root=str(tmp_path))
    assert res["ok"] is False
    joined_f = "\n".join(res["failures"])
    joined_w = "\n".join(res["warnings"])
    assert "重复注（同键同文）: 同一注" in joined_f
    assert "重复注（同键近似 r=0.93）: 凯撒" in joined_f
    assert "同键不同注: 凯撒" in joined_w
    assert "merged 未生成" in joined_w


def test_lint_book_merged_missing_warns(tmp_path):
    _mk(tmp_path, [("ch01", "ch01_chunk000", _CLEAN_FINAL)])
    res = lint_book(project_root=str(tmp_path))
    assert res["ok"] is True and res["failures"] == []
    assert any("merged 未生成" in w for w in res["warnings"])


# ── 3. D5 根因守卫：prompt 禁流程元语 ────────────────────────────────

def test_prompt_bans_translation_meta_language():
    ban = "脚注只陈述典故/文化背景/双关机制本身"
    for path in ("src/translator_agent.py", "src/agents/rewriter_agent.py"):
        src = (ROOT / path).read_text(encoding="utf-8")
        assert src.count(ban) == 2, f"{path} 应含 2 处禁元语指令"
