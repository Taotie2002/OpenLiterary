#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L5 editor_review — 成书级编辑审读（merge 层级, 逐章, MiniMax-M3 thinking on）。

chunk 级 QA 查不出 merge 级缺陷（悬空承诺/重复注释/译文腔密集/声音断裂/注释喧宾夺主）。
本脚本在 merge 后把「原文 + 译文」分段交给编辑审读角色，输出六类问题：

  dangling       悬空承诺（"如下：——"类中断、内容缺失）
  dup_note       重复注释 / 注释载体丢失（注了没引、引了没注）
  translationese 译文腔密集段
  voice          声音断裂（叙述者/角色口吻与基调突变）
  note_heavy     注释喧宾夺主（密度打断叙事节奏）
  arc            Round 131：人物/伏笔/语气弧线违背（--global 时结合全书判断层）

产物：output/editor_review.md / output/editor_review.json
默认只报告（退出码 0）；--fail-on-p0 时存在 P0 问题 → 退出码 1（可选门禁）。

用法：
  python3 scripts/editor_review.py [--project-root .] [--chapters ch04,ch08]
      [--md output/editor_review.md] [--json output/editor_review.json]
      [--max-chars 60000] [--overlap-chars 2000] [--global]
      [--fail-on-p0]
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
VALID_CATEGORIES = ("dangling", "dup_note", "translationese", "voice", "note_heavy",
                    "arc", "other")


def _make_default_generate(task_name: str = "editor_review"):
    """Build a generator for a specific review task."""
    if task_name not in ("editor_review", "editor_review_global"):
        task_name = "editor_review"
    from src.config import get_config
    from src.utils.llm_adapter import (
        get_llm_client_for_task, get_role_model_name,
    )
    cfg = get_config()
    actual = task_name if task_name in (cfg.task_routing or {}) else "editor_review"
    model_key, params = cfg.resolve_task_model(actual)
    role = cfg.get_task_role(actual)
    llm = get_llm_client_for_task(actual)
    model_name = get_role_model_name(role) or model_key

    def generate(prompt: str) -> str:
        return llm.generate(prompt, model_name=model_name, **params)

    return generate


def _default_generate(prompt: str) -> str:
    return _make_default_generate("editor_review")(prompt)


def _model_label(task_name: str = "editor_review") -> str:
    try:
        from src.config import get_config
        from src.utils.llm_adapter import get_role_model_name
        cfg = get_config()
        actual = task_name if task_name in (cfg.task_routing or {}) else "editor_review"
        role = cfg.get_task_role(actual)
        return get_role_model_name(role) or "MiniMax-M3"
    except Exception:
        return "MiniMax-M3"


def build_review_prompt(chapter_id, source_text, translated_text, max_chars=60000,
                        judgment_context: str = "", global_mode: bool = False,
                        truncate: bool = True) -> str:
    src = source_text or "（原文缺失，仅审读译文）"
    tgt = translated_text or ""
    if truncate and max_chars > 0:
        if len(src) > max_chars:
            src = src[:max_chars] + "\n……（原文超长截断）"
        if len(tgt) > max_chars:
            tgt = tgt[:max_chars] + "\n……（译文超长截断）"
    focus = "全书弧线/伏笔闭环" if global_mode else "本章成书质量"
    judgment_block = (judgment_context.rstrip() + "\n\n") if judgment_context else ""
    return f"""你是资深中文文学编辑，负责译著付印前的成书级审读。只报告问题，不改写全文、不润色。

请审读第 {chapter_id} 章（英译中译稿），本轮焦点：{focus}。检查以下六类缺陷：
1. dangling 悬空承诺：如"……就成了下面这个样子：——"后无内容、句子/段落异常中断、内容缺失；
2. dup_note 重复注释 / 注释载体丢失：同一词条重复加注、有注无引、有引无注；
3. translationese 译文腔密集段：连续两句以上明显翻译腔（"对于…来说""当…的时候""被…所…""进行了一次"等）；
4. voice 声音断裂：叙述者/角色口吻与该章基调骤然不一致（如童趣叙述中混入公文腔）；
5. note_heavy 注释喧宾夺主：注释密度过高、频繁打断叙事节奏；
6. arc 弧线违背：人物语气/关系演变、伏笔与回收、主题意象的阶段性变化与全书判断层冲突。

{judgment_block}【原文】
{src}

【译文】
{tgt}

只输出一个 JSON 对象（不要任何额外文字）：
{{"issues": [{{"severity": "P0|P1|P2", "category": "dangling|dup_note|translationese|voice|note_heavy|arc|other",
  "quote": "译文中逐字可检索的片段（≤40字）", "note": "问题说明（≤60字）", "suggestion": "最小修复建议（≤60字）"}}]}}

判定约定：内容缺失/悬空中断/注释重复 = P0；译文腔密集/声音断裂/弧线违背 = P1；其余轻微 = P2。
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
            "section_id": str(it.get("section_id") or "").strip(),
            "char_start": it.get("char_start"),
            "char_end": it.get("char_end"),
            "source_char_start": it.get("source_char_start"),
            "source_char_end": it.get("source_char_end"),
            "anchor_verified": bool(it.get("anchor_verified", False)),
            "anchor_offset": it.get("anchor_offset"),
        })
    return out


def split_text_windows(text: str, max_chars: int, overlap_chars: int = 2000) -> list:
    """Split a long chapter into review windows.  No source text is truncated."""
    src = str(text or "")
    limit = max(1000, int(max_chars or 60000))
    overlap = max(0, min(int(overlap_chars or 0), limit // 2))
    if len(src) <= limit:
        return [{"text": src, "char_start": 0, "char_end": len(src), "index": 1, "total": 1}]
    raw = []
    start = 0
    while start < len(src):
        end = min(len(src), start + limit)
        if end < len(src):
            floor = start + int(limit * 0.65)
            best = -1
            for marker in ("\n\n", "\n", ". ", "。", "! ", "? "):
                idx = src.rfind(marker, floor, end)
                if idx > best:
                    best = idx + (len(marker) if marker in ("\n\n", ". ", "! ", "? ") else 1)
            if best > start + 200:
                end = best
        raw.append({"text": src[start:end], "char_start": start,
                    "char_end": end, "index": len(raw) + 1, "total": 0})
        if end >= len(src):
            break
        start = max(start + 1, end - overlap)
    if len(raw) > 1 and len(raw[-1]["text"]) < max(500, overlap // 2):
        prev = raw[-2]
        prev["text"] = src[prev["char_start"]:]
        prev["char_end"] = len(src)
        raw.pop()
    for i, item in enumerate(raw, 1):
        item["index"] = i
        item["total"] = len(raw)
    return raw


def _source_window(source_text: str, target_start: int, target_end: int,
                   target_len: int, overlap: int) -> tuple:
    src = str(source_text or "")
    if not src:
        return "", 0, 0
    ratio = len(src) / max(1, target_len)
    start = max(0, int(target_start * ratio) - overlap)
    end = min(len(src), int(target_end * ratio) + overlap)
    if end <= start:
        end = min(len(src), start + 1)
    return src[start:end], start, end


def review_chapter(chapter_id: str, source_text: str, translated_text: str,
                   generate, max_chars: int = 60000, overlap_chars: int = 2000,
                   judgment_context: str = "", global_mode: bool = False) -> dict:
    """Review a whole chapter via paragraph-aware windows; never silently truncate."""
    if not translated_text:
        return {"chapter": chapter_id, "error": "no_final_texts", "issues": []}
    windows = split_text_windows(translated_text, max_chars=max_chars,
                                 overlap_chars=overlap_chars)
    from src.utils.llm_adapter import robust_json_loads
    all_issues = []
    section_meta = []
    errors = []
    for win in windows:
        src_seg, src_start, src_end = _source_window(
            source_text, win["char_start"], win["char_end"],
            len(translated_text), max(400, int(overlap_chars or 0)),
        )
        section_id = f"{chapter_id}:s{int(win['index']):03d}"
        prompt = build_review_prompt(
            chapter_id, src_seg, win["text"], max_chars=0,
            judgment_context=judgment_context, global_mode=global_mode,
            truncate=False,
        )
        section_meta.append({
            "section_id": section_id,
            "char_start": win["char_start"], "char_end": win["char_end"],
            "source_char_start": src_start, "source_char_end": src_end,
        })
        try:
            raw = generate(prompt)
            data = robust_json_loads(raw, expected_type=dict)
            items = data.get("issues") if isinstance(data, dict) else None
            if not isinstance(items, list):
                errors.append({"section_id": section_id, "error": "parse_error"})
                continue
        except Exception as exc:  # noqa: BLE001
            errors.append({"section_id": section_id,
                           "error": f"llm_error: {type(exc).__name__}: {exc}"})
            continue
        issues = _norm_issues(items)
        from src.judgment_layer import match_quote
        for it in issues:
            it["section_id"] = section_id
            it["char_start"] = win["char_start"]
            it["char_end"] = win["char_end"]
            it["source_char_start"] = src_start
            it["source_char_end"] = src_end
            hit = match_quote(translated_text, it.get("quote") or "")
            if hit.get("ok"):
                it["anchor_verified"] = True
                it["anchor_offset"] = hit.get("offset")
                it["char_start"] = int(hit.get("offset") or 0)
                it["char_end"] = int(hit.get("end") or it["char_start"])
            else:
                it["anchor_verified"] = False
                it["anchor_offset"] = None
        all_issues.extend(issues)

    # Overlap windows may return the same issue twice.  Verified copy wins.
    deduped = []
    seen = set()
    for it in all_issues:
        sig = (it.get("category"), it.get("quote"), it.get("note"))
        if sig in seen:
            for existing in deduped:
                if (existing.get("category"), existing.get("quote"),
                        existing.get("note")) == sig and not existing.get("anchor_verified"):
                    existing.update(it)
            continue
        seen.add(sig)
        deduped.append(it)
    err = errors[0]["error"] if errors and len(errors) == len(windows) else None
    return {
        "chapter": chapter_id,
        "error": err,
        "errors": errors,
        "sections": section_meta,
        "issues": deduped,
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "global_mode": bool(global_mode),
    }


def discover_chapters(root: Path, chapters_arg: str = "") -> list:
    out_root = root / "output"
    if chapters_arg:
        return [c.strip() for c in chapters_arg.split(",") if c.strip()]
    return sorted(p.name for p in out_root.glob("ch*") if p.is_dir())


def collect_translated_text(chapter_dir: Path) -> str:
    from src.drift_audit import collect_chunk_texts
    return "\n\n".join(t for _cid, t in collect_chunk_texts(chapter_dir) if t)


def _load_final_chunk_texts(root: Path, chapter_id: str) -> dict:
    """Load chunk-level final texts for quote-to-chunk mapping."""
    out = {}
    ch_dir = Path(root) / "output" / chapter_id
    if not ch_dir.exists():
        return out
    for path in sorted(ch_dir.glob("*_final.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            text = data.get("text") or data.get("final_text") or ""
        else:
            text = data if isinstance(data, str) else ""
        if text:
            out[path.stem.replace("_final", "")] = str(text)
    return out


def build_review_dirty(root: Path, results: list) -> dict:
    """Map mechanically verified structural P0 issues to one final chunk.

    This artifact is advisory input for a future ``pipeline --fix-from-review``.
    It never changes task state itself. Ambiguous or unverified anchors are
    recorded as skipped so the caller can audit why no automatic fix happened.
    """
    from src.judgment_layer import STRUCTURAL_P0_CATEGORIES, map_quote_to_chunks

    entries = []
    skipped = []
    for chapter in results or []:
        cid = str(chapter.get("chapter") or "")
        chunk_texts = _load_final_chunk_texts(root, cid)
        for issue in chapter.get("issues") or []:
            if issue.get("severity") != "P0":
                continue
            category = str(issue.get("category") or "")
            if category not in STRUCTURAL_P0_CATEGORIES:
                continue
            base = {
                "chapter": cid,
                "section_id": issue.get("section_id", ""),
                "category": category,
                "quote": issue.get("quote", ""),
                "note": issue.get("note", ""),
                "suggestion": issue.get("suggestion", ""),
            }
            if not issue.get("anchor_verified"):
                skipped.append({**base, "reason": "anchor_not_verified"})
                continue
            mapped = map_quote_to_chunks(chunk_texts, issue.get("quote") or "")
            if not mapped.get("ok"):
                skipped.append({**base, "reason": mapped.get("reason") or "unmapped",
                                "hits": mapped.get("hits") or []})
                continue
            if not mapped.get("unique_in_chunk"):
                skipped.append({**base, "reason": "quote_not_unique_in_chunk",
                                "chunk_id": mapped.get("chunk_id"),
                                "hits": mapped.get("hits") or []})
                continue
            entries.append({**base, "chunk_id": mapped.get("chunk_id"),
                            "match_kind": mapped.get("match_kind"),
                            "match_score": mapped.get("match_score"),
                            "offset": mapped.get("offset"),
                            "matched_quote": mapped.get("matched_quote")})
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "profile": "structural_p0_verified",
        "summary": {"entries": len(entries), "skipped": len(skipped)},
        "entries": entries,
        "skipped": skipped,
    }


def _md_cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def build_report_md(results, model_label: str, generated_at: str) -> str:
    counts = {"P0": 0, "P1": 0, "P2": 0}
    errors = [r for r in results if r.get("error")]
    section_errors = sum(len(r.get("errors") or []) for r in results)
    for r in results:
        for it in r.get("issues") or []:
            counts[it["severity"]] = counts.get(it["severity"], 0) + 1
    lines = [
        "# 成书级编辑审读报告（editor_review）",
        "",
        f"- 生成时间: {generated_at}",
        f"- 审读模型: {model_label}",
        f"- 审读章节: {len(results)}（失败 {len(errors)}）",
        f"- 分段失败: {section_errors}",
        f"- 问题计数: P0 {counts['P0']} / P1 {counts['P1']} / P2 {counts['P2']}",
    ]
    p0_rows = []
    for r in results:
        for it in r.get("issues") or []:
            if it["severity"] == "P0":
                p0_rows.append((r["chapter"], it))
    if p0_rows:
        lines += ["", "## P0 汇总", "",
                  "| 章 | 段 | 类别 | 锚句 | 引文 | 说明 |", "|---|---|---|---|---|---|"]
        for ch, it in p0_rows:
            lines.append(
                f"| {ch} | {it.get('section_id', '')} | {it['category']} "
                f"| {'verified' if it.get('anchor_verified') else 'UNVERIFIED'} "
                f"| {_md_cell(it['quote'])} | {_md_cell(it['note'])} |")
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
    ap.add_argument("--overlap-chars", type=int, default=2000,
                    help="分段审读窗口重叠字符数（默认 2000）")
    ap.add_argument("--global", dest="global_mode", action="store_true",
                    help="全书弧线/伏笔审读；加载 book_profile.json 并使用 editor_review_global 角色")
    ap.add_argument("--review-dirty", action="store_true",
                    help="额外生成 review_dirty.json（仅机械校验过的结构 P0，不直接改状态）")
    ap.add_argument("--review-dirty-path", default="output/review_dirty.json")
    ap.add_argument("--fail-on-p0", action="store_true", help="存在 P0 时退出码 1")
    args = ap.parse_args(argv)

    root = Path(args.project_root)
    task_name = "editor_review_global" if args.global_mode else "editor_review"
    gen = generate or _make_default_generate(task_name)
    chapter_ids = discover_chapters(root, args.chapters)

    judgment_context = ""
    if args.global_mode:
        try:
            from src.judgment_layer import render_global_review_context
            from src.style_profiler import load_profile
            prof = load_profile(root / "input" / "book_profile.json")
            judgment_context = render_global_review_context(prof)
            if not judgment_context:
                print("⚠️ [editor_review] --global 未找到可注入的判断层注记（继续普通全局审读）")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ [editor_review] 判断层加载失败，继续普通审读（{type(exc).__name__}: {exc}）")

    # 优先审读交付文本（merge + notes-normalize 后的切片）；缺失时回退 finals 拼接
    merged_map = {}
    from src.book_lint import resolve_book_output_path
    merged_path = resolve_book_output_path(root, root / "output")
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
        results.append(review_chapter(
            ch, src, tgt, gen, max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
            judgment_context=judgment_context, global_mode=args.global_mode,
        ))
        status = results[-1].get("error") or f"{len(results[-1].get('issues') or [])} issues"
        print(f"📖 [editor_review] {ch}: {status}")

    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    model_label = _model_label(task_name)
    md = build_report_md(results, model_label, generated_at)
    counts = {"P0": 0, "P1": 0, "P2": 0}
    for r in results:
        for it in r.get("issues") or []:
            counts[it["severity"]] += 1
    payload = {
        "generated_at": generated_at,
        "model": model_label,
        "global_mode": bool(args.global_mode),
        "judgment_context_chars": len(judgment_context),
        "summary": {
            "chapters": len(results),
            "errors": sum(1 for r in results if r.get("error")),
            "section_errors": sum(len(r.get("errors") or []) for r in results),
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
    if args.review_dirty:
        dirty = build_review_dirty(root, results)
        dirty_path = root / args.review_dirty_path
        dirty_path.parent.mkdir(parents=True, exist_ok=True)
        dirty_path.write_text(json.dumps(dirty, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🧹 [editor_review] review_dirty: {dirty['summary']['entries']} mapped / "
              f"{dirty['summary']['skipped']} skipped → {dirty_path}")
    print(f"📖 [editor_review] P0={counts['P0']} P1={counts['P1']} P2={counts['P2']} → {md_path}")
    return 1 if (args.fail_on_p0 and counts["P0"] > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
