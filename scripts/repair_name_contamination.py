#!/usr/bin/env python3
"""Repair mechanically-detectable cross-person name contamination in finals.

Only a conservative subset is eligible: the final and literary name sequences
must have the same length, every non-suspect name must match positionally, and
the literary text must not itself contain a standalone suspect name.  All other
chunks are reported as rescue candidates and are left untouched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


NAME_TOKENS = [
    "盖乌斯·海伦·莫希阿姆",
    "盖乌斯海伦莫希阿姆",
    "盖乌斯夫人",
    "杰西卡夫人",
    "杰西卡",
    "盖乌斯",
    "海伦",
    "尊母",
    "真言者",
    "芬伦",
    "芬林",
    "莫希阿姆",
    "圣母",
    "老妇人",
]
NAME_TOKENS.sort(key=len, reverse=True)
SUSPECT_NAMES = frozenset({"盖乌斯", "盖乌斯夫人", "海伦"})


def _split_names(text: str) -> List[str]:
    names: List[str] = []
    i = 0
    while i < len(text):
        token = next((t for t in NAME_TOKENS if text.startswith(t, i)), None)
        if token is None:
            i += 1
            continue
        names.append(token)
        i += len(token)
    return names


def _name_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        token = next((t for t in NAME_TOKENS if text.startswith(t, i)), None)
        if token is None:
            i += 1
            continue
        spans.append((i, i + len(token), token))
        i += len(token)
    return spans


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp-name-repair")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _repair_text(final_text: str, literary_text: str) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    final_names = _split_names(final_text)
    literary_names = _split_names(literary_text)
    if len(final_names) != len(literary_names):
        return None
    if any(name in SUSPECT_NAMES for name in literary_names):
        return None
    for final_name, literary_name in zip(final_names, literary_names):
        if final_name not in SUSPECT_NAMES and final_name != literary_name:
            return None

    spans = _name_spans(final_text)
    if len(spans) != len(literary_names):
        return None
    out: List[str] = []
    cursor = 0
    changes: List[Dict[str, Any]] = []
    for (start, end, final_name), literary_name in zip(spans, literary_names):
        out.append(final_text[cursor:start])
        replacement = literary_name if final_name in SUSPECT_NAMES else final_name
        out.append(replacement)
        cursor = end
        if replacement != final_name:
            changes.append({
                "from": final_name,
                "to": replacement,
                "offset": start,
            })
    out.append(final_text[cursor:])
    repaired = "".join(out)
    if _split_names(repaired) != literary_names:
        return None
    if any(name in SUSPECT_NAMES for name in _split_names(repaired)):
        return None
    return repaired, changes


def _iter_finals(output_dir: Path, chapter: Optional[str]) -> Iterable[Path]:
    pattern = f"{chapter}/*_final.json" if chapter else "ch*/*_final.json"
    return sorted(output_dir.glob(pattern))


def _write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# 姓名污染确定性修复报告",
        "",
        f"- 模式: {payload['mode']}",
        f"- 扫描 final: {payload['finals_scanned']}",
        f"- 含可疑姓名: {payload['suspect_chunks']}",
        f"- 可确定性修复: {payload['repair_candidates']}",
        f"- 已写入: {payload['repaired']}",
        f"- 跳过: {json.dumps(payload['skipped'], ensure_ascii=False, sort_keys=True)}",
        f"- 替换统计: {json.dumps(payload['replacements'], ensure_ascii=False, sort_keys=True)}",
        "",
        "其余姓名结构不一致或 literary 仍含可疑姓名的块不做正文修改，进入定向 rescue 名单。",
        "",
        "## 待 rescue 块",
        "",
    ]
    rescue = payload.get("rescue_candidates") or []
    if rescue:
        lines.extend(f"- `{cid}`" for cid in rescue)
    else:
        lines.append("（无）")
    lines.extend(["", "## 已修复明细", ""])
    rows = payload.get("rows") or []
    if rows:
        for row in rows:
            mapping = ", ".join(f"{x['from']}→{x['to']}" for x in row["changes"])
            lines.append(f"- `{row['chunk_id']}`: {mapping}")
    else:
        lines.append("（无）")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="确定性修复 final 中的姓名污染")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--chapter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="写入 final.json（默认 dry-run）")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output"
    json_out = Path(args.json_out) if args.json_out else output_dir / "name_contamination_repair.json"
    report_path = Path(args.report) if args.report else output_dir / "name_contamination_repair.md"
    now = _dt.datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    backup_root = output_dir / "repair_backups" / f"name_contamination_{stamp}"

    skipped: Counter = Counter()
    replacements: Counter = Counter()
    rescue_candidates: List[str] = []
    rows: List[Dict[str, Any]] = []
    finals = list(_iter_finals(output_dir, args.chapter))
    suspect_chunks = 0
    candidates = 0
    repaired = 0

    for final_path in finals:
        cid = final_path.stem.rsplit("_final", 1)[0]
        chapter = final_path.parent.name
        final_data = _read_json(final_path)
        if not final_data or not isinstance(final_data.get("text"), str):
            skipped["invalid_final"] += 1
            continue
        final_text = final_data["text"]
        final_names = _split_names(final_text)
        if not any(name in SUSPECT_NAMES for name in final_names):
            continue
        suspect_chunks += 1

        literary_path = final_path.with_name(f"{cid}_literary.json")
        literary_data = _read_json(literary_path)
        if not literary_data or not isinstance(literary_data.get("text"), str):
            skipped["missing_literary"] += 1
            rescue_candidates.append(cid)
            continue
        literary_text = literary_data["text"]
        literary_names = _split_names(literary_text)
        if any(name in SUSPECT_NAMES for name in literary_names):
            skipped["literary_has_suspect"] += 1
            rescue_candidates.append(cid)
            continue
        if len(final_names) != len(literary_names):
            skipped["name_count_mismatch"] += 1
            rescue_candidates.append(cid)
            continue
        if any(f != l for f, l in zip(final_names, literary_names)
               if f not in SUSPECT_NAMES):
            skipped["nonsuspect_sequence_mismatch"] += 1
            rescue_candidates.append(cid)
            continue

        result = _repair_text(final_text, literary_text)
        if result is None:
            skipped["repair_validation_failed"] += 1
            rescue_candidates.append(cid)
            continue
        repaired_text, changes = result
        if not changes:
            skipped["no_change"] += 1
            continue

        candidates += 1
        if args.limit is not None and candidates > max(0, args.limit):
            continue

        if args.apply:
            backup_path = backup_root / chapter / final_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_path, backup_path)
            final_data["text"] = repaired_text
            meta = final_data.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["name_contamination_repair"] = {
                "schema_version": 1,
                "applied_at": now.isoformat(),
                "source": "literary",
                "replacements": changes,
                "backup": str(backup_path.relative_to(root)),
            }
            final_data["metadata"] = meta
            _atomic_write_json(final_path, final_data)
            repaired += 1

        for change in changes:
            replacements[f"{change['from']}->{change['to']}"] += 1
        rows.append({
            "chunk_id": cid,
            "chapter": chapter,
            "before_sha256": _sha256_text(final_text),
            "after_sha256": _sha256_text(repaired_text),
            "changes": changes,
        })

    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "finals_scanned": len(finals),
        "suspect_chunks": suspect_chunks,
        "repair_candidates": candidates,
        "repaired": repaired,
        "skipped": dict(sorted(skipped.items())),
        "replacements": dict(sorted(replacements.items())),
        "rescue_candidates": sorted(set(rescue_candidates)),
        "rows": rows,
    }
    _write_report(json_out, payload)
    _write_markdown(report_path, payload)
    print(json.dumps({
        "mode": payload["mode"],
        "finals_scanned": payload["finals_scanned"],
        "suspect_chunks": payload["suspect_chunks"],
        "repair_candidates": payload["repair_candidates"],
        "repaired": payload["repaired"],
        "skipped": payload["skipped"],
        "report": str(report_path),
        "json": str(json_out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
