#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L5 editor_review — 成书级编辑审读（merge 层级, 逐章, MiniMax-M3 thinking on）。

chunk 级 QA 查不出 merge 级缺陷（悬空承诺/重复注释/译文腔密集/声音断裂/注释喧宾夺主）。
本脚本在 merge 后逐章把「原文 + 各 chunk final 拼接译文」交给编辑审读角色，输出五类问题：

  dangling       悬空承诺（"如下：——"类中断、内容缺失）
  dup_note       重复注释 / 注释载体丢失（注了没引、引了没注）
  translationese 译文腔密集段
  voice          声音断裂（叙述者/角色口吻与基调突变）
  note_heavy     注释喧宾夺主（密度打断叙事节奏）

产物：output/editor_review.md / output/editor_review.json
默认只报告（退出码 0）；--fail-on-p0 时存在 P0 问题 → 退出码 1（可选门禁）。

用法：
  python3 scripts/editor_review.py [--project-root .] [--chapters ch04,ch08]
      [--md output/editor_review.md] [--json output/editor_review.json]
      [--max-chars 60000] [--fail-on-p0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

VALID_SEVERITIES = ("P0", "P1", "P2")
VALID_CATEGORIES = ("dangling", "dup_note", "translationese", "voice", "note_heavy", "other")


def _default_generate(prompt: str) -> str:
    """editor_review 角色（MiniMax-M3, thinking on）单次生成。"""
    from src.config import get_config
    from src.utils.llm_adapter import (
        get_llm_client_for_task, get_role_model_name,
    )
    cfg = get_config()
    model_key, params = cfg.resolve_task_model("editor_review")
    role = cfg.get_task_role("editor_review")
    llm = get_llm_client_for_task("editor_review")
    model_name = get_role_model_name(role) or model_key
    return llm.generate(prompt, model_name=model_name, **params)


def _model_label() -> str:
    try:
        from src.config import get_config
        from src.utils.llm_adapter import get_role_model_name
        cfg = get_config()
        role = cfg.get_task_role("editor_review")
        return get_role_model_name(role) or "MiniMax-M3"
    except Exception:
        return "MiniMax-M3"


def build_review_prompt(chapter_id, source_text, translated_text, max_chars=60000) -> str:
    src = source_text or "（原文缺失，仅审读译文）"
    tgt = translated_text or ""
    if len(src) > max_chars:
        src = src[:max_chars] + "\n……（原文超长截断）"
    if len(tgt) > max_chars:
        tgt = tgt[:max_chars] + "\n……（译文超长截断）"
    return f"""你是资深中文文学编辑，负责译著付印前的成书级审读。只报告问题，不改写全文、不润色。

请审读第 {chapter_id} 章（英译中译稿），聚焦五类成书级缺陷：
1. dangling 悬空承诺：如"……就成了下面这个样子：——"后无内容、句子/段落异常中断、内容缺失；
2. dup_note 重复注释 / 注释载体丢失：同一词条重复加注、有注无引、有引无注；
3. translationese 译文腔密集段：连续两句以上明显翻译腔（"对于…来说""当…的时候""被…所…""进行了一次"等）；
4. voice 声音断裂：叙述者/角色口吻与该章基调骤然不一致（如童趣叙述中混入公文腔）；
5. note_heavy 注释喧宾夺主：注释密度过高、频繁打断叙事节奏。

【原文】
{src}

【译文】
{tgt}

只输出一个 JSON 对象（不要任何额外文字）：
{{"issues": [{{"severity": "P0|P1|P2", "category": "dangling|dup_note|translationese|voice|note_heavy|other",
  "quote": "译文中逐字可检索的片段（≤40字）", "note": "问题说明（≤60字）", "suggestion": "最小修复建议（≤60字）"}}]}}

判定约定：内容缺失/悬空中断/注释重复 = P0；译文腔密集/声音断裂 = P1；其余轻微 = P2。
quote 必须是译文中真实出现的片段；无问题输出 {{"issues": []}}。"""


def _norm_issues(items) -> list:
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        sev = str(it.get("severity") or "P2").strip().upper()
        if sev not in VALID_SEVERITIES:
            sev = "P2"
        cat = str(it.get("category") or "other").strip().lower()
        if cat not in VALID_CATEGORIES:
            cat = "other"
        out.append({
            "severity": sev,
            "category": cat,
            "quote": str(it.get("quote") or "").strip()[:120],
            "note": str(it.get("note") or "").strip()[:200],
            "suggestion": str(it.get("suggestion") or "").strip()[:200],
        })
    return out


def review_chapter(chapter_id: str, source_text: str, translated_text: str,
                   generate, max_chars: int = 60000) -> dict:
    """单章审读；LLM 失败/解析失败均落 error 字段，不抛出。"""
    if not translated_text:
        return {"chapter": chapter_id, "error": "no_final_texts", "issues": []}
    prompt = build_review_prompt(chapter_id, source_text, translated_text, max_chars=max_chars)
    try:
        raw = generate(prompt)
    except Exception as exc:  # noqa: BLE001
        return {"chapter": chapter_id, "error": f"llm_error: {type(exc).__name__}: {exc}",
                "issues": []}
    from src.utils.llm_adapter import robust_json_loads
    data = robust_json_loads(raw, expected_type=dict)
    issues = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(issues, list):
        return {"chapter": chapter_id, "error": "parse_error", "issues": []}
    return {"chapter": chapter_id, "error": None, "issues": _norm_issues(issues)}


def discover_chapters(root: Path, chapters_arg: str = "") -> list:
    out_root = root / "output"
    if chapters_arg:
        return [c.strip() for c in chapters_arg.split(",") if c.strip()]
    return sorted(p.name for p in out_root.glob("ch*") if p.is_dir())


def collect_translated_text(chapter_dir: Path) -> str:
    from src.drift_audit import collect_chunk_texts
    return "\n\n".join(t for _cid, t in collect_chunk_texts(chapter_dir) if t)


def _md_cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def build_report_md(results, model_label: str, generated_at: str) -> str:
    counts = {"P0": 0, "P1": 0, "P2": 0}
    errors = [r for r in results if r.get("error")]
    for r in results:
        for it in r.get("issues") or []:
            counts[it["severity"]] = counts.get(it["severity"], 0) + 1
    lines = [
        "# 成书级编辑审读报告（editor_review）",
        "",
        f"- 生成时间: {generated_at}",
        f"- 审读模型: {model_label}",
        f"- 审读章节: {len(results)}（失败 {len(errors)}）",
        f"- 问题计数: P0 {counts['P0']} / P1 {counts['P1']} / P2 {counts['P2']}",
    ]
    p0_rows = []
    for r in results:
        for it in r.get("issues") or []:
            if it["severity"] == "P0":
                p0_rows.append((r["chapter"], it))
    if p0_rows:
        lines += ["", "## P0 汇总", "", "| 章 | 类别 | 引文 | 说明 |", "|---|---|---|---|"]
        for ch, it in p0_rows:
            lines.append(
                f"| {ch} | {it['category']} | {_md_cell(it['quote'])} | {_md_cell(it['note'])} |")
    for r in results:
        lines += ["", f"## {r['chapter']}"]
        if r.get("error"):
            lines.append(f"- ⚠️ 审读失败: {_md_cell(r['error'])}")
            continue
        issues = r.get("issues") or []
        if not issues:
            lines.append("- 未发现问题。")
            continue
        lines += ["", "| severity | category | 引文 | 说明 | 建议 |", "|---|---|---|---|---|"]
        for it in issues:
            lines.append(
                f"| {it['severity']} | {it['category']} | {_md_cell(it['quote'])} "
                f"| {_md_cell(it['note'])} | {_md_cell(it['suggestion'])} |")
    lines += ["", "---", "*由 scripts/editor_review.py 生成（默认只报告；--fail-on-p0 可作门禁）*", ""]
    return "\n".join(lines)


def main(argv=None, generate=None) -> int:
    ap = argparse.ArgumentParser(description="L5 editor_review 成书级编辑审读")
    ap.add_argument("--project-root", default=str(_ROOT))
    ap.add_argument("--chapters", default="", help="逗号分隔章节 id（默认全部 output/ch*）")
    ap.add_argument("--md", default="output/editor_review.md")
    ap.add_argument("--json", default="output/editor_review.json")
    ap.add_argument("--max-chars", type=int, default=60000, help="原文/译文各自截断上限")
    ap.add_argument("--fail-on-p0", action="store_true", help="存在 P0 时退出码 1")
    args = ap.parse_args(argv)

    root = Path(args.project_root)
    gen = generate or _default_generate
    chapter_ids = discover_chapters(root, args.chapters)

    # 优先审读交付文本（merge + notes-normalize 后的切片）；缺失时回退 finals 拼接
    merged_map = {}
    merged_path = root / "output" / "translated_full.md"
    if merged_path.exists():
        try:
            from src.book_lint import _split_h1_sections, chapter_heading
            for label, body in _split_h1_sections(merged_path.read_text(encoding="utf-8")):
                merged_map[label] = body
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ [editor_review] merged 加载失败，回退 finals 拼接（{type(exc).__name__}: {exc}）")

    results = []
    for ch in chapter_ids:
        src_path = root / "input" / f"{ch}.md"
        src = src_path.read_text(encoding="utf-8") if src_path.exists() else ""
        ch_dir = root / "output" / ch
        tgt = ""
        if merged_map:
            try:
                exp = ("# " + chapter_heading(root, ch)).strip()
            except Exception:  # noqa: BLE001
                exp = ""
            if exp in merged_map:
                tgt = exp + "\n\n" + merged_map[exp]
        if not tgt:
            tgt = collect_translated_text(ch_dir) if ch_dir.exists() else ""
        results.append(review_chapter(ch, src, tgt, gen, max_chars=args.max_chars))
        status = results[-1].get("error") or f"{len(results[-1].get('issues') or [])} issues"
        print(f"📖 [editor_review] {ch}: {status}")

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    model_label = _model_label()
    md = build_report_md(results, model_label, generated_at)
    counts = {"P0": 0, "P1": 0, "P2": 0}
    for r in results:
        for it in r.get("issues") or []:
            counts[it["severity"]] += 1
    payload = {
        "generated_at": generated_at,
        "model": model_label,
        "summary": {
            "chapters": len(results),
            "errors": sum(1 for r in results if r.get("error")),
            "total_issues": sum(counts.values()),
            "p0": counts["P0"], "p1": counts["P1"], "p2": counts["P2"],
        },
        "chapters": results,
    }
    md_path = root / args.md
    json_path = root / args.json
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📖 [editor_review] P0={counts['P0']} P1={counts['P1']} P2={counts['P2']} → {md_path}")
    return 1 if (args.fail_on_p0 and counts["P0"] > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
