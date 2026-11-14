#!/usr/bin/env python3
"""Repair mechanically provable residue from historical name unification.

Only exact, invalid surfaces are repaired.  Proper names that can legitimately
refer to Mohiam (for example "老圣母盖乌斯·海伦·莫希阿姆" in a scene using the
title "Reverend Mother") are deliberately left untouched for later auditing.

Dry-run is the default; --apply writes final.json with per-chunk backups.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp-residual-name")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_sources(root: Path) -> Dict[str, str]:
    db = root / "db" / "workflow.db"
    out: Dict[str, str] = {}
    if not db.exists():
        return out
    con = sqlite3.connect(str(db))
    try:
        for cid, text in con.execute("SELECT chunk_id, text_content FROM chunk_tasks"):
            if isinstance(cid, str) and isinstance(text, str):
                out[cid] = text
    finally:
        con.close()
    return out


def _has_source_name(source: str, name: str) -> bool:
    return bool(re.search(r"\b" + re.escape(name) + r"\b", source, re.IGNORECASE))


def _replace_once(text: str, before: str, after: str, rule: str,
                  changes: List[Dict[str, Any]]) -> str:
    count = text.count(before)
    if not count:
        return text
    text = text.replace(before, after)
    changes.append({"rule": rule, "from": before, "to": after, "count": count})
    return text


def _replace_body_token(text: str, before: str, after: str, rule: str,
                        changes: List[Dict[str, Any]]) -> str:
    """Replace a token outside footnote definition lines."""
    out: List[str] = []
    count = 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("[^"):
            out.append(line)
            continue
        n = line.count(before)
        if n:
            count += n
            line = line.replace(before, after)
        out.append(line)
    if count:
        text = "".join(out)
        changes.append({"rule": rule, "from": before, "to": after, "count": count})
    return text


def repair_text(text: str, source: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    changes: List[Dict[str, Any]] = []
    skipped: List[str] = []
    source_flat = re.sub(r"\s+", " ", source.replace("-?", "")).casefold()

    # "Lady Gaius" is not a valid title in the novel.  Every occurrence is a
    # historical Jessica->Mohiam substitution.
    text = _replace_once(text, "盖乌斯夫人", "杰西卡夫人",
                          "invalid_lady_gaius", changes)

    # Concatenation artifacts produced when a role token was substituted into a
    # full Mohiam name.  No such surface is legitimate.
    text = _replace_once(text, "老海伦盖乌斯·海伦·莫希阿姆", "老圣母",
                          "malformed_old_helen_full", changes)
    text = _replace_once(text, "老海伦", "老圣母",
                          "malformed_old_helen", changes)

    no_mohiam_source = (not _has_source_name(source, "Mohiam")
                        and not _has_source_name(source, "Gaius"))
    has_jessica_source = _has_source_name(source, "Jessica")

    if no_mohiam_source and has_jessica_source:
        text = _replace_once(text, "真言者夫人", "杰西卡",
                              "malformed_truthsayer_lady", changes)
        text = _replace_once(text, "盖乌斯报告", "杰西卡报告",
                              "gaius_report_footnote", changes)
        text = _replace_once(
            text,
            "盖乌斯（Jessica），全名盖乌斯·海伦·莫希阿姆（Gaius Helen Mohiam）",
            "杰西卡（Jessica）",
            "contaminated_jessica_footnote",
            changes,
        )
        text = _replace_once(text, "盖乌斯：保罗的母亲", "杰西卡：保罗的母亲",
                              "contaminated_mother_footnote", changes)

    if (no_mohiam_source
            and _has_source_name(source, "Truthsayer")
            and "他也怕盖乌斯" in text):
        text = _replace_once(text, "他也怕盖乌斯", "他也怕真言者",
                              "residual_gaius_truthsayer", changes)

    if no_mohiam_source:
        # Source/raw both describe Jessica without naming her.  These exact
        # contexts are mechanically recoverable and have no legitimate Mohiam
        # reading in the novel.
        if "save your son and your woman" in source_flat:
            text = _replace_once(
                text,
                "我会救你的儿子和盖乌斯·海伦·莫希阿姆",
                "我会救你的儿子和杰西卡",
                "woman_save_jessica",
                changes,
            )
        if "his mother" in source_flat:
            text = _replace_once(text, "他的母亲盖乌斯·海伦·莫希阿姆",
                                  "他的母亲杰西卡", "mother_jessica", changes)
        if "jessica arose" in source_flat:
            text = _replace_once(text, "盖乌斯·海伦·莫希阿姆起身", "杰西卡起身",
                                  "jessica_arose", changes)
        if "the stranger might think" in source_flat:
            text = _replace_once(text, "盖乌斯·海伦·莫希阿姆初来乍到",
                                  "新来者初来乍到", "stranger_not_mohiam", changes)
        if "brood-mother" in source_flat or "broodmother" in source_flat:
            text = _replace_once(text, "我的小繁育母亲——盖乌斯·海伦·莫希阿姆",
                                  "我的小繁育母亲", "brood_mother_not_mohiam", changes)
        if "reverend mother mystery" in source_flat:
            text = _replace_once(text, "圣母盖乌斯·海伦·莫希阿姆", "圣母",
                                  "generic_reverend_mother", changes)
            text = _replace_once(text, "你的圣母盖乌斯·海伦·莫希阿姆", "你的圣母",
                                  "generic_your_reverend_mother", changes)
            text = _replace_once(
                text,
                "[^1]: 盖乌斯·海伦·莫希阿姆，贝尼·杰瑟里特姐妹会的领袖人物；“圣母”是其职衔尊称。",
                "[^1]: “圣母”是弗雷曼宗教语境中对女性精神领袖的尊称。",
                "generic_reverend_mother_footnote",
                changes,
            )
        text = _replace_once(text, "“盖乌斯”（Truthsayer）", "“真言者”（Truthsayer）",
                              "truthsayer_label", changes)

        if (re.search(r"\bSayyadina\b", source, re.IGNORECASE)
                and "莫希阿姆" in text):
            text = _replace_body_token(text, "莫希阿姆", "赛亚迪娜",
                                       "sayyadina_not_mohiam", changes)

    has_leto_duke = bool(re.search(r"\bDuke\s+Leto\b", source, re.IGNORECASE))
    has_paul_duke = bool(re.search(r"\bDuke\s+Paul\b", source, re.IGNORECASE))

    if has_leto_duke and not has_paul_duke:
        text = _replace_once(text, "保罗·亚崔迪公爵", "雷托·厄崔迪公爵",
                              "pauls_wrong_atreides_duke", changes)
        text = _replace_once(text, "保罗·雷托公爵", "雷托·厄崔迪公爵",
                              "concatenated_leto_duke", changes)
    elif has_paul_duke and not has_leto_duke:
        text = _replace_once(text, "保罗·雷托公爵", "保罗·厄崔迪公爵",
                              "concatenated_paul_duke", changes)
        text = _replace_once(text, "保罗·雷托", "保罗·厄崔迪",
                              "concatenated_paul", changes)
    elif "保罗·雷托" in text:
        skipped.append("ambiguous_paul_leto_without_unique_source_duke")

    return text, changes, skipped


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# 残留姓名损坏确定性修复报告",
        "",
        f"- 模式: {payload['mode']}",
        f"- 扫描 final: {payload['scanned']}",
        f"- 命中 chunk: {payload['matched_chunks']}",
        f"- 替换数: {payload['replacements']}",
        f"- 已写入 chunk: {payload['applied_chunks']}",
        "",
        "只修复作品内不可能成立的精确表面形；有歧义的全名指代不改。",
        "",
    ]
    rows = payload.get("rows") or []
    if not rows:
        lines.append("（无命中）")
    for row in rows:
        lines.append(f"## `{row['chunk_id']}`")
        for change in row.get("changes") or []:
            lines.append(
                f"- `{change['rule']}`: `{change['from']}` → `{change['to']}` × {change['count']}"
            )
        for skip in row.get("skipped") or []:
            lines.append(f"- 跳过: `{skip}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="修复残留的确定性姓名损坏（默认 dry-run）")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--chapter", default=None)
    parser.add_argument("--chunks", default=None, help="逗号分隔的 chunk_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output"
    json_out = Path(args.json_out) if args.json_out else output_dir / "residual_name_damage_repair.json"
    report = Path(args.report) if args.report else output_dir / "residual_name_damage_repair.md"
    sources = _load_sources(root)
    now = _dt.datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    backup_root = output_dir / "repair_backups" / f"residual_name_{stamp}"

    if args.chunks:
        paths = []
        for cid in (x.strip() for x in args.chunks.split(",")):
            if cid:
                paths.append(output_dir / cid.split("_", 1)[0] / f"{cid}_final.json")
    else:
        pattern = f"{args.chapter}/*_final.json" if args.chapter else "ch*/*_final.json"
        paths = sorted(output_dir.glob(pattern))

    rows: List[Dict[str, Any]] = []
    scanned = 0
    applied_chunks = 0
    replacements = 0
    for final_path in paths:
        cid = final_path.stem.rsplit("_final", 1)[0]
        source = sources.get(cid, "")
        final_data = _read_json(final_path)
        if not source or not final_data:
            continue
        final = str(final_data.get("text") or "")
        if not final:
            continue
        scanned += 1
        repaired, changes, skipped = repair_text(final, source)
        if not changes and not skipped:
            continue
        row = {
            "chunk_id": cid,
            "changes": changes,
            "skipped": skipped,
            "before_sha256": __import__("hashlib").sha256(final.encode("utf-8")).hexdigest(),
            "after_sha256": __import__("hashlib").sha256(repaired.encode("utf-8")).hexdigest(),
            "applied": False,
        }
        replacements += sum(x["count"] for x in changes)
        rows.append(row)
        if args.limit is not None and len(rows) > max(0, args.limit):
            rows.pop()
            break
        if args.apply and changes:
            chapter = cid.split("_", 1)[0]
            backup_path = backup_root / chapter / final_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_path, backup_path)
            final_data["text"] = repaired
            meta = final_data.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["residual_name_damage_repair"] = {
                "schema_version": 1,
                "applied_at": now.isoformat(),
                "rules": changes,
                "backup": str(backup_path.relative_to(root)),
            }
            final_data["metadata"] = meta
            _atomic_write_json(final_path, final_data)
            row["applied"] = True
            applied_chunks += 1

    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "scanned": scanned,
        "matched_chunks": len(rows),
        "replacements": replacements,
        "applied_chunks": applied_chunks,
        "rows": rows,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, payload)
    print(json.dumps({
        "mode": payload["mode"],
        "scanned": scanned,
        "matched_chunks": len(rows),
        "replacements": replacements,
        "applied_chunks": applied_chunks,
        "json": str(json_out),
        "report": str(report),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
