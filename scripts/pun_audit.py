#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 pun_audit — 双关登记核销审计（纯静态，无 LLM 调用）。

对 input/book_profile.json 的 pun_watchlist 逐项核销三类事实：
  1) decision_ok: decision_db 存在含该条目 anchor 的决策记录
     （pipeline._seed_pun_decisions 命中 chunk 时强制落库；ReferenceAgent 机制复用）；
  2) landed     : merged / 各章 final 的正文或注释中出现 anchor 落点
                  （anchor = watchlist anchors ∪ 预注册决定译文片段）；
  3) dup_free   : 同一 anchor 的注释未重复（同键、跨文件近似）。

产物：
  output/pun_audit.md   — 人读报告
  output/pun_audit.json — 机读（gate ⑦ 读 summary.uncovered_nonwaived）
未核销且未豁免 → 退出码 1（--no-fail 关闭；watchlist 为空 → 退出码 0）。

用法：
  python3 scripts/pun_audit.py [--project-root .] [--profile input/book_profile.json]
      [--merged output/translated_full.md] [--waivers input/pun_waivers.json]
      [--md output/pun_audit.md] [--json output/pun_audit.json] [--no-fail]
豁免文件 input/pun_waivers.json 格式：{"waived": ["tale/tail", "game"], "note": "理由"}
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.book_lint import _scan_lines, _def_key  # noqa: E402


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_watchlist(profile_path) -> list:
    try:
        from src.style_profiler import load_profile
        prof = load_profile(Path(profile_path))
        if isinstance(prof, dict):
            return list(prof.get("pun_watchlist") or [])
    except Exception:
        pass
    prof = _read_json(profile_path, {}) or {}
    if isinstance(prof, dict):
        return list(prof.get("pun_watchlist") or [])
    return []


def load_waivers(path) -> set:
    data = _read_json(path, {}) or {}
    out = set()
    if isinstance(data, dict):
        for k in (data.get("waived") or []):
            s = str(k).strip().lower()
            if s:
                out.add(s)
    return out


def load_decision_texts(db_path) -> list:
    p = Path(db_path)
    if not p.exists():
        return []
    try:
        con = sqlite3.connect(str(p))
        rows = con.execute(
            "SELECT source_key, translation, COALESCE(reason, '') FROM decision_db"
        ).fetchall()
        con.close()
    except sqlite3.Error:
        return []
    return [f"{a} {b} {c}".lower() for a, b, c in rows]


_RENDER_META_RE = re.compile(r"策略|双关|保留|注释|正文|尾注|【|\[|\]|《|》")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _render_fragments(translation: str) -> list:
    """预注册决定译文 → 合法落点片段（中文短片段；滤除策略说明式文本）。"""
    out = []
    for part in re.split(r"[/、；;,，\s]+", str(translation or "")):
        frag = part.strip().strip("“”\"'（）()【】[]")
        if not (2 <= len(frag) <= 14):
            continue
        if not _CJK_RE.search(frag) or _RENDER_META_RE.search(frag):
            continue
        if frag not in out:
            out.append(frag)
    return out


def load_decision_renders(db_path) -> list:
    """[(source_key_lower, translation), ...] 供落点判定派生片段。"""
    p = Path(db_path)
    if not p.exists():
        return []
    try:
        con = sqlite3.connect(str(p))
        rows = con.execute("SELECT source_key, translation FROM decision_db").fetchall()
        con.close()
    except sqlite3.Error:
        return []
    return [(str(a or "").lower(), str(b or "")) for a, b in rows]


_ASCII_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\- ]*$")


def _anchor_hit(anchor: str, text: str) -> bool:
    if not anchor or len(anchor) < 2 or not text:
        return False
    a = str(anchor).strip()
    if _ASCII_WORD_RE.match(a):
        return re.search(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])",
                         text, re.IGNORECASE) is not None
    return a in text


def collect_texts(out_root: Path, merged_path: Path):
    texts = []
    if merged_path.exists():
        texts.append(("merged", merged_path.read_text(encoding="utf-8")))
    for fp in sorted(out_root.glob("ch*/*_final.json")):
        data = _read_json(fp, None)
        if data is None:
            continue
        text = data.get("text") if isinstance(data, dict) else str(data)
        if text:
            texts.append((f"{fp.parent.name}/{fp.stem}", text))
    return texts


def audit(watchlist, texts, decision_texts, waivers, decision_renders=None) -> dict:
    from src.style_profiler import _split_anchors

    items = []
    for it in watchlist:
        if not isinstance(it, dict):
            continue
        source = str(it.get("source") or "").strip()
        anchors = [str(a).strip() for a in (it.get("anchors") or []) if str(a).strip()]
        anchors = [a for a in anchors if len(a) >= 2]
        if not anchors:
            anchors = _split_anchors(source)
        src_anchors = [str(a).strip() for a in (it.get("src_anchors") or []) if str(a).strip()]
        src_anchors = [a for a in src_anchors if len(a) >= 2]
        key = source.lower()
        waived = key in waivers or any(a.lower() in waivers for a in anchors + src_anchors)
        decision_ok = bool(decision_texts) and any(
            any(a.lower() in d for a in anchors + src_anchors) for d in decision_texts
        )
        render_anchors = []
        for sk, tr in (decision_renders or []):
            if any(a.lower() in sk for a in src_anchors):
                for frag in _render_fragments(tr):
                    if frag not in render_anchors:
                        render_anchors.append(frag)
        hit_anchors = anchors + render_anchors
        landed_labels = [label for label, text in texts
                         if any(_anchor_hit(a, text) for a in hit_anchors)]
        notes_seen = {}
        dups = []
        for label, text in texts:
            if label == "merged":
                continue
            _items, defs, _refs = _scan_lines(text)
            for d in defs:
                content = d.get("content") or ""
                if not any(_anchor_hit(a, content) for a in hit_anchors):
                    continue
                nk = _def_key(content)
                if nk in notes_seen and notes_seen[nk][0] != label:
                    dups.append(f"{nk} @ {notes_seen[nk][0]} / {label}")
                else:
                    notes_seen[nk] = (label, content)
        covered = decision_ok and bool(landed_labels) and not dups
        items.append({
            "source": source,
            "kind": it.get("kind") or "",
            "anchors": anchors,
            "src_anchors": src_anchors,
            "strategy": it.get("strategy") or "",
            "hint": it.get("hint") or "",
            "decision_ok": decision_ok,
            "landed": bool(landed_labels),
            "landed_labels": landed_labels[:4],
            "render_anchors": render_anchors[:8],
            "dup_notes": dups[:4],
            "waived": waived,
            "covered": covered,
        })
    total = len(items)
    covered_n = sum(1 for i in items if i["covered"])
    waived_n = sum(1 for i in items if i["waived"] and not i["covered"])
    uncovered = [i["source"] for i in items if not i["covered"] and not i["waived"]]
    return {
        "summary": {
            "total": total,
            "covered": covered_n,
            "waived": waived_n,
            "uncovered_nonwaived": len(uncovered),
            "uncovered_sources": uncovered[:20],
        },
        "items": items,
    }


def render_markdown(result: dict) -> str:
    s = result["summary"]
    lines = ["# 双关核销报告 (pun_audit)", "",
             f"- watchlist 总数: {s['total']}",
             f"- 已核销: {s['covered']}",
             f"- 豁免: {s['waived']}",
             f"- 未核销（非豁免）: {s['uncovered_nonwaived']}", ""]
    if s["total"]:
        lines += ["| # | source | decision | landed | dup | 覆盖 | 策略 |",
                  "|---|--------|----------|--------|-----|------|------|"]
        for i, it in enumerate(result["items"], 1):
            mark = "✅" if it["covered"] else ("🚫 豁免" if it["waived"] else "❌")
            landed = ",".join(it["landed_labels"]) or "—"
            dup = "; ".join(it["dup_notes"]) or "—"
            strategy = (it["strategy"] or "").replace("|", "/")[:40]
            lines.append(
                f"| {i} | {it['source']} | {'✅' if it['decision_ok'] else '❌'} | "
                f"{'✅' if it['landed'] else '—'}（{landed}） | {dup} | {mark} | {strategy} |"
            )
    else:
        lines.append("（无 pun_watchlist 条目，跳过）")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="L2 pun_audit 双关核销审计")
    ap.add_argument("--project-root", default=str(_ROOT))
    ap.add_argument("--profile", default=None)
    ap.add_argument("--merged", default=None)
    ap.add_argument("--waivers", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--no-fail", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.project_root)
    profile_path = Path(args.profile) if args.profile else root / "input" / "book_profile.json"
    merged_path = Path(args.merged) if args.merged else root / "output" / "translated_full.md"
    waivers_path = Path(args.waivers) if args.waivers else root / "input" / "pun_waivers.json"
    md_path = Path(args.md) if args.md else root / "output" / "pun_audit.md"
    json_path = Path(args.json_out) if args.json_out else root / "output" / "pun_audit.json"

    watchlist = load_watchlist(profile_path)
    waivers = load_waivers(waivers_path)
    if not watchlist:
        result = {
            "summary": {"total": 0, "covered": 0, "waived": 0,
                        "uncovered_nonwaived": 0, "uncovered_sources": []},
            "items": [],
            "note": "book_profile 无 pun_watchlist（或剖面不存在），跳过",
        }
    else:
        texts = collect_texts(root / "output", merged_path)
        decisions = load_decision_texts(root / "db" / "decision_db.sqlite")
        renders = load_decision_renders(root / "db" / "decision_db.sqlite")
        result = audit(watchlist, texts, decisions, waivers, decision_renders=renders)
        result["texts_scanned"] = len(texts)
        result["decisions_scanned"] = len(decisions)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    s = result["summary"]
    print(f"🎯 [pun_audit] watchlist={s['total']} covered={s['covered']} "
          f"waived={s['waived']} uncovered={s['uncovered_nonwaived']}")
    for src in s.get("uncovered_sources", [])[:5]:
        print(f"  ❌ 未核销: {src}")
    if result.get("note"):
        print(f"  ℹ️ {result['note']}")
    print(f"  报告: {md_path}")
    if s["total"] and s["uncovered_nonwaived"] and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
