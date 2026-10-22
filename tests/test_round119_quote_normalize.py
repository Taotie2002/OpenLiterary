"""Round 119 (T5): 样式规范化 pass — 引号统一 / 上标脚注转 [^n] / 悬空定义转译者注.

背景（D5）: 旧全量跑 finalize 干跑显示 16 个 final 待变更 — 注释标题行 4、
上标定义 17、悬空定义 29、「」72、直引号 28；此前无规范化 pass、无 lint。

测试目标（对应方案 T5 验收）:
  1. normalize_final_style: 「」/『』→“”/‘’; 直引号确定性配对;
     上标定义（"¹ 内容"）有引用→[^n]:、无引用→章末译者注; 注释标题行删除;
     编号列表定义（header 后 "1. …"）→译者注。
  2. 幂等: 二次规范化输出逐字节稳定（md_defs_kept 为信息计数，允许非零）。
  3. finalize_outputs: dry-run 不落盘、apply 写回 + metadata.style_normalized、
     二次 apply changed=0; 报告 output/finalize_diff.md。
  4. 规范化后 lint_texts 全绿（样式类问题闭环）。
"""
from __future__ import annotations

import json

from src.book_lint import finalize_outputs, lint_texts, normalize_final_style

_MUTATING_KEYS = ("headers_removed", "sup_defs_converted", "orphans_relocated",
                  "cjk_quotes", "straight_quotes")


def _stable_rerun(text: str):
    new2, ch2 = normalize_final_style(text)
    assert new2 == text, "二次规范化输出应稳定（幂等）"
    for k in _MUTATING_KEYS:
        assert ch2[k] == 0, (k, ch2)


# ── 1. 引号 ──────────────────────────────────────────────────────────

def test_cjk_quotes_converted_to_curly():
    new, ch = normalize_final_style("他喊「快跑」，又低声说『别回头』。")
    assert new == "他喊“快跑”，又低声说‘别回头’。"
    assert ch["cjk_quotes"] == 4
    _stable_rerun(new)


def test_straight_quotes_even_pairs():
    new, ch = normalize_final_style('他说"你好"，然后"再见"。')
    assert new == "他说“你好”，然后“再见”。"
    assert ch["straight_quotes"] == 4
    _stable_rerun(new)


def test_straight_quotes_odd_pairing_deterministic():
    new, ch = normalize_final_style('甲"乙"丙"丁')
    assert new == "甲“乙”丙“丁"
    assert ch["straight_quotes"] == 3
    _stable_rerun(new)


# ── 2. 脚注制式 ───────────────────────────────────────────────────────

def test_sup_def_with_ref_converted_to_md():
    new, ch = normalize_final_style("正文参考。¹\n\n¹ 原注：某典故出处。\n")
    assert new == "正文参考。[^1]\n\n[^1]: 原注：某典故出处。\n"
    assert ch["sup_defs_converted"] == 1
    _stable_rerun(new)


def test_md_def_with_ref_kept_in_place():
    text = "正文[^2]。\n\n[^2]: 已有定义。\n"
    new, ch = normalize_final_style(text)
    assert new == text
    assert ch["md_defs_kept"] == 1 and ch["sup_defs_converted"] == 0
    _stable_rerun(text)


def test_header_removed_and_orphan_def_relocated():
    new, ch = normalize_final_style("**脚注：**\n\n¹ 这条注释无人引用。\n\n正文段落。\n")
    assert new == "\n\n正文段落。\n\n> **译者注：** 这条注释无人引用。"
    assert ch["headers_removed"] == 1 and ch["orphans_relocated"] == 1
    assert "[^1]" not in new
    _stable_rerun(new)


def test_footnote_header_variants_all_removed():
    variants = [
        "**脚注：**", "**脚注**：", "**【脚注】**", "【脚注】",
        "**脚注**（典故保留并加注）：", "**脚注（保留与加注策略）**",
        "**脚注**", "**脚注（…策略）**：",
    ]
    for h in variants:
        new, ch = normalize_final_style(h + "\n\n正文。\n")
        assert ch["headers_removed"] == 1, h
        assert h not in new, h
        assert new == "\n正文。\n", h


def test_numbered_list_after_header_relocated():
    new, ch = normalize_final_style(
        "**脚注（保留与加注策略）**\n\n1. 第一条注释文字。\n2. 第二条注释文字。\n\n正文。\n")
    assert new == ("\n\n正文。\n\n> **译者注：** 第一条注释文字。\n"
                   "> **译者注：** 第二条注释文字。")
    assert ch["headers_removed"] == 1 and ch["orphans_relocated"] == 2
    assert "保留与加注" not in new
    _stable_rerun(new)


def test_messy_document_second_pass_stable():
    text = ('**脚注（保留与加注策略）**\n「引文」和"straight"\n\n'
            '¹ 未引用定义。\n\n¹ 第二条未引用定义。\n')
    new, ch = normalize_final_style(text)
    assert new == ('“引文”和“straight”\n\n> **译者注：** 未引用定义。\n'
                   '> **译者注：** 第二条未引用定义。')
    assert ch["headers_removed"] == 1 and ch["orphans_relocated"] == 2
    assert ch["cjk_quotes"] == 2 and ch["straight_quotes"] == 2
    _stable_rerun(new)


# ── 3. finalize_outputs ──────────────────────────────────────────────

def test_finalize_dry_run_then_apply_and_idempotent(tmp_path):
    out = tmp_path / "output"
    d = out / "ch01"
    d.mkdir(parents=True)
    fp = d / "ch01_chunk000_final.json"
    fp.write_text(json.dumps({"text": "他说「你好」。\n", "metadata": {"x": 1}},
                             ensure_ascii=False), encoding="utf-8")
    before = fp.read_text(encoding="utf-8")

    dry = finalize_outputs(project_root=str(tmp_path))
    assert dry["applied"] is False
    assert dry["stats"]["files"] == 1 and dry["stats"]["changed"] == 1
    assert dry["stats"]["cjk_quotes"] == 2
    assert fp.read_text(encoding="utf-8") == before, "dry-run 不得写回 final"
    rep = (out / "finalize_diff.md").read_text(encoding="utf-8")
    assert "DRY RUN" in rep and "ch01/ch01_chunk000_final.json" in rep

    a1 = finalize_outputs(project_root=str(tmp_path), apply=True)
    assert a1["applied"] is True and a1["stats"]["changed"] == 1
    data = json.loads(fp.read_text(encoding="utf-8"))
    assert data["text"] == "他说“你好”。\n"
    assert data["metadata"]["style_normalized"] is True
    assert data["metadata"]["x"] == 1, "apply 不得丢原有 metadata 字段"
    assert "APPLY" in (out / "finalize_diff.md").read_text(encoding="utf-8")

    a2 = finalize_outputs(project_root=str(tmp_path), apply=True)
    assert a2["stats"]["changed"] == 0, "二次 apply 应零变更（幂等）"


def test_normalize_then_lint_green():
    raw = "**脚注：**\n\n他说「你好」。¹\n\n¹ 原注：注释内容。\n"
    new, _ = normalize_final_style(raw)
    res = lint_texts([("t", new)])
    assert res["ok"] is True and res["failures"] == []
