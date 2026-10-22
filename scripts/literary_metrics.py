#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L6 literary_metrics — 文学度量（AI 味/翻译腔/标点/注释/重复/诗体）。

用法：
  python3 scripts/literary_metrics.py [--merged output/translated_full.md]
      [--json output/literary_metrics.json] [--baseline docs/baselines/literary_metrics_2026-10-14.json]
      [--input-dir input] [--print]

指标以中文正文为对象（剥离 markdown/注释后按 CJK 字符统计）：
  ai_flavor_counts/密度（每千字）/强度词/翻译腔句式/标点/引号/脚注/4-gram 重复率/韵文 manifest 概况。
对比模式（--baseline）：与基线同名数值比 ±tolerance，输出 worse 列表供 gate/人工判断。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

AI_WORDS = ["仿佛", "彷佛", "似乎", "深深地", "顿时", "瞬间", "微微",
            "不禁", "蓦然", "旋即", "倏然", "恍惚"]
INTENSITY_WORDS = ["非常", "极其", "十分", "格外", "异常", "分外", "无比", "极为", "更加"]

TRANSLATIONESE_PATTERNS = OrderedDict([
    ("进行一次", r"进行了?一次"),
    ("对于…来说", r"对于.{0,8}来说"),
    ("当…的时候", r"当.{0,14}的时候"),
    ("在…的时候", r"在.{0,14}的时候"),
    ("作为一个", r"作为一[个名位]"),
    ("被…所…", r"被.{0,10}所[\u4e00-\u9fff]"),
    ("…的话", r"的话[，。]"),
    ("使得", r"使得"),
    ("的的", r"的的"),
])


def strip_markup(text: str) -> str:
    t = re.sub(r"\[[^\]]*\]\([^)]*\)", "", text)
    t = re.sub(r"\[\^[^\]]+\]:?[^\n]*", " ", t)
    t = re.sub(r"^#{1,6}\s.*$", "", t, flags=re.M)
    t = re.sub(r"^\s*[¹²³⁴⁵⁶⁷⁸⁹]+\s.*$", "", t, flags=re.M)
    return t


def compute_metrics(text: str, verse_manifest: dict | None = None) -> dict:
    body = strip_markup(text)
    cjk_only = "".join(re.findall(r"[\u4e00-\u9fff]", body))
    n = len(cjk_only)

    ai_counts = {w: body.count(w) for w in AI_WORDS}
    ai_total = sum(ai_counts.values())
    intensity = {w: body.count(w) for w in INTENSITY_WORDS}
    trans = {name: len(re.findall(pat, body)) for name, pat in TRANSLATIONESE_PATTERNS.items()}

    punct = {
        "dash_em": body.count("——"),
        "ellipsis": body.count("……"),
        "exclaim": body.count("！"),
        "question": body.count("？"),
    }
    quotes = {
        "ascii_double": body.count('"'),
        "curly_open": body.count("“"),
        "curly_close": body.count("”"),
        "corner": sum(body.count(c) for c in "「」『』"),
    }

    try:
        from src.book_lint import _scan_lines
        _items, defs, refs = _scan_lines(text)
        notes = {
            "defs": len(defs),
            "sup": sum(1 for d in defs if d["style"] == "sup"),
            "md": sum(1 for d in defs if d["style"] == "md"),
            "num": sum(1 for d in defs if d["style"] == "num"),
            "orphans": sum(1 for d in defs if d["id"] not in refs),
        }
    except Exception:
        notes = {"defs": 0, "sup": 0, "md": 0, "num": 0, "orphans": 0}

    grams: dict = {}
    for i in range(max(0, n - 3)):
        g = cjk_only[i:i + 4]
        grams[g] = grams.get(g, 0) + 1
    total_g = sum(grams.values())
    distinct_g = len(grams)
    top = sorted(grams.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    repeat_rate = round(1 - distinct_g / total_g, 6) if total_g else 0.0

    verse = {"manifest_blocks": 0, "manifest_lines": 0}
    if isinstance(verse_manifest, dict):
        for _ch, blocks in (verse_manifest.get("chapters") or {}).items():
            for b in (blocks or []):
                verse["manifest_blocks"] += 1
                verse["manifest_lines"] += len(b.get("lines") or [])

    return {
        "chars_cjk": n,
        "ai_flavor_counts": ai_counts,
        "ai_flavor_total": ai_total,
        "ai_density_per_1k": round(ai_total / n * 1000, 3) if n else 0.0,
        "intensity_counts": intensity,
        "translationese_counts": trans,
        "translationese_total": sum(trans.values()),
        "punct": punct,
        "quotes": quotes,
        "notes": notes,
        "ngram4": {
            "total": total_g,
            "distinct": distinct_g,
            "repeat_rate": repeat_rate,
            "top": [{"gram": g, "count": c} for g, c in top],
        },
        "verse": verse,
    }


def _get_path(d: dict, path: str):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def compare_to_baseline(current: dict, baseline: dict, tolerance: float = 0.05) -> list:
    """与基线对比：worse=True 表示劣于基线且超出容差（lower 型指标）。"""
    rows = []
    checks = [
        ("ai_density_per_1k", "lower"),
        ("translationese_total", "lower"),
        ("quotes.ascii_double", "lower"),
        ("notes.orphans", "lower"),
        ("ngram4.repeat_rate", "lower"),
    ]
    for path, direction in checks:
        b = _get_path(baseline, path)
        c = _get_path(current, path)
        if b is None or c is None:
            continue
        if b == 0:
            delta_pct = 0.0 if c == 0 else None
            worse = bool(c > 0)
        else:
            delta_pct = round((c - b) / b * 100, 2)
            worse = delta_pct > tolerance * 100
        rows.append({"metric": path, "baseline": b, "current": c,
                     "delta_pct": delta_pct, "worse": worse})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L6 literary_metrics 文学度量")
    ap.add_argument("--project-root", default=str(_ROOT))
    ap.add_argument("--merged", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.project_root)
    merged = Path(args.merged) if args.merged else root / "output" / "translated_full.md"
    json_path = Path(args.json_out) if args.json_out else root / "output" / "literary_metrics.json"
    input_dir = Path(args.input_dir) if args.input_dir else root / "input"

    if not merged.exists():
        print(f"❌ merged 不存在: {merged}", file=sys.stderr)
        return 2
    text = merged.read_text(encoding="utf-8")

    verse_manifest = None
    vm_path = input_dir / "verse_manifest.json"
    if vm_path.exists():
        try:
            verse_manifest = json.loads(vm_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            verse_manifest = None

    metrics = compute_metrics(text, verse_manifest=verse_manifest)
    metrics["source"] = str(merged)
    metrics["generated_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")

    if args.baseline:
        try:
            baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            baseline = None
        if isinstance(baseline, dict):
            metrics["baseline_comparison"] = compare_to_baseline(metrics, baseline)
            metrics["baseline_path"] = str(args.baseline)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"🎯 [literary_metrics] chars={metrics['chars_cjk']} "
          f"ai_density={metrics['ai_density_per_1k']}/1k "
          f"translationese={metrics['translationese_total']} "
          f"repeat_rate={metrics['ngram4']['repeat_rate']}")
    print(f"  报告: {json_path}")
    if args.do_print:
        top = ", ".join(f"{g['gram']}x{g['count']}" for g in metrics["ngram4"]["top"][:5])
        print(f"  4-gram top: {top}")
        cmp_rows = metrics.get("baseline_comparison") or []
        for row in cmp_rows:
            flag = "⚠️ 劣" if row["worse"] else "✅"
            print(f"  {flag} {row['metric']}: {row['baseline']} → {row['current']} "
                  f"({row['delta_pct']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
