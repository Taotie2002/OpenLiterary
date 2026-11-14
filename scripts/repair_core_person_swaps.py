#!/usr/bin/env python3
"""Restore mechanically-provable core-person swaps in final.json.

The historical name-unification pass could turn Jessica into Gaius/Mohiam and
Leto into Paul.  This script repairs only those two directions, and only when
the final/literary person-mention sequences align one-for-one and the raw draft
plus the English source support the literary person.

It never rewrites prose.  It replaces a wrong surface name with the surface
name already used by the literary draft.  Dry-run is the default.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

CORE_ALIASES: Dict[str, str] = {
    "雷托": "LETO",
    "雷托公爵": "LETO",
    "雷托·厄崔迪": "LETO",
    "雷托·厄崔迪斯": "LETO",
    "雷托·厄崔迪公爵": "LETO",
    "雷托·亚崔迪公爵": "LETO",
    "雷托·阿特雷德斯公爵": "LETO",
    "雷托·阿特雷德公爵": "LETO",
    "保罗": "PAUL",
    "保罗·厄崔迪": "PAUL",
    "保罗·阿特雷迪斯": "PAUL",
    "保罗·亚崔迪": "PAUL",
    "保罗·厄崔迪斯": "PAUL",
    "保罗·阿特雷德斯": "PAUL",
    "杰西卡": "JESSICA",
    "杰西卡夫人": "JESSICA",
    "盖乌斯": "MOHIAM",
    "盖乌斯夫人": "MOHIAM",
    "盖乌斯·海伦·莫希阿姆": "MOHIAM",
    "莫希阿姆": "MOHIAM",
    "海伦": "MOHIAM",
    "真言者": "TRUTHSAYER",
    "真言师": "TRUTHSAYER",
}
TERMS = sorted(CORE_ALIASES, key=len, reverse=True)
REPAIR_DIRECTIONS = {
    "MOHIAM": "JESSICA",
    "PAUL": "LETO",
}
SOURCE_NAMES = {
    "LETO": ("Leto",),
    "PAUL": ("Paul",),
    "JESSICA": ("Jessica",),
    "MOHIAM": ("Mohiam", "Gaius"),
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp-core-person")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _spans(text: str) -> List[Tuple[int, int, str, str]]:
    out: List[Tuple[int, int, str, str]] = []
    i = 0
    while i < len(text):
        token = next((t for t in TERMS if text.startswith(t, i)), None)
        if token is None:
            i += 1
            continue
        out.append((i, i + len(token), CORE_ALIASES[token], token))
        i += len(token)
    return out


def _key_counts(text: str) -> Dict[str, int]:
    counts = {k: 0 for k in SOURCE_NAMES}
    for _s, _e, key, _tok in _spans(text):
        counts[key] = counts.get(key, 0) + 1
    return counts


def _source_counts(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, needles in SOURCE_NAMES.items():
        total = 0
        for needle in needles:
            total += len(re.findall(r"\b" + re.escape(needle) + r"\b", text))
        out[key] = total
    return out


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


def _support(raw_counts: Dict[str, int], source_counts: Dict[str, int],
             target: str, wrong: str) -> bool:
    if source_counts.get(target, 0) <= 0:
        return False
    if raw_counts.get(target, 0) <= 0:
        return False
    if target == "JESSICA":
        # A Mohiam mention in the source can be introduced by the title
        # "Reverend Mother", so the safe check is that raw has no Mohiam at
        # all in this chunk.
        return raw_counts.get("MOHIAM", 0) == 0 and source_counts.get("MOHIAM", 0) == 0
    if target == "LETO":
        return raw_counts.get("LETO", 0) >= raw_counts.get("PAUL", 0)
    return False


def _plan_chunk(cid: str, final: str, literary: str, raw: str,
                source: str) -> Dict[str, Any]:
    sf = _spans(final)
    sl = _spans(literary)
    kf = [x[2] for x in sf]
    kl = [x[2] for x in sl]
    if not kf or not kl or kf == kl:
        return {"chunk_id": cid, "changes": [], "reason": "no_core_difference"}

    raw_counts = _key_counts(raw)
    source_counts = _source_counts(source)
    import difflib
    matcher = difflib.SequenceMatcher(None, kl, kf, autojunk=False)
    changes: List[Dict[str, Any]] = []
    rejected: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            rejected.append(f"{tag}:{i1}-{i2}/{j1}-{j2}")
            continue
        for li, fi in zip(range(i1, i2), range(j1, j2)):
            lk, fk = kl[li], kf[fi]
            if lk == fk:
                continue
            target = REPAIR_DIRECTIONS.get(fk)
            if target != lk:
                rejected.append(f"direction:{fk}->{lk}")
                continue
            if not _support(raw_counts, source_counts, target, fk):
                rejected.append(f"unsupported:{fk}->{lk}")
                continue
            fs, fe, _fk, ftok = sf[fi]
            _ls, _le, _lk, ltok = sl[li]
            changes.append({
                "from": ftok,
                "to": ltok,
                "start": fs,
                "end": fe,
                "final_key": fk,
                "literary_key": lk,
                "before": final[max(0, fs - 45):min(len(final), fe + 45)],
            })
    if not changes:
        return {"chunk_id": cid, "changes": [], "reason": "no_safe_change",
                "rejected": rejected[:8]}
    if rejected:
        return {"chunk_id": cid, "changes": [], "reason": "complex_alignment",
                "rejected": rejected[:8]}

    out: List[str] = []
    cursor = 0
    for change in changes:
        out.append(final[cursor:change["start"]])
        out.append(change["to"])
        cursor = change["end"]
    out.append(final[cursor:])
    repaired = "".join(out)
    verify = _spans(repaired)
    if [x[2] for x in verify] != kl:
        return {"chunk_id": cid, "changes": [], "reason": "verification_failed"}
    before_counts = _key_counts(final)
    after_counts = _key_counts(repaired)
    if after_counts == before_counts:
        return {"chunk_id": cid, "changes": [], "reason": "no_effective_change"}
    for change in changes:
        change.pop("start", None)
        change.pop("end", None)
    return {
        "chunk_id": cid,
        "changes": changes,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "literary_counts": _key_counts(literary),
        "raw_counts": raw_counts,
        "source_counts": source_counts,
        "before_sha256": _sha256(final),
        "after_sha256": _sha256(repaired),
        "repaired_text": repaired,
        "reason": "safe_change",
    }


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# 核心人物错位修复报告",
        "",
        f"- 模式: {payload['mode']}",
        f"- 扫描 final: {payload['scanned']}",
        f"- 可修复 chunk: {payload['candidate_chunks']}",
        f"- 替换数: {payload['candidate_changes']}",
        f"- 已写入 chunk: {payload['applied_chunks']}",
        "",
        "只修复 `盖乌斯/莫希阿姆→杰西卡` 与 `保罗→雷托` 两类，且要求 final/literary",
        "人物序列一一对齐、raw 与英文源文支持 literary 人物。其余块只报告不改写。",
        "",
        "## 候选明细",
        "",
    ]
    rows = payload.get("rows") or []
    if not rows:
        lines.append("（无）")
    for row in rows:
        lines.append(f"### `{row['chunk_id']}`")
        if row.get("reason") != "safe_change":
            lines.append(f"- 跳过: {row.get('reason')} {row.get('rejected') or ''}")
            lines.append("")
            continue
        for ch in row["changes"]:
            lines.append(f"- `{ch['from']}→{ch['to']}`：{ch['before']}")
        lines.append("")
    lines.extend(["## 复杂对齐（不改）", ""])
    complex_rows = payload.get("complex_rows") or []
    if not complex_rows:
        lines.append("（无）")
    for row in complex_rows:
        lines.append(f"- `{row['chunk_id']}`: {row.get('reason')} {row.get('rejected') or ''}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="修复核心人物机械错位（默认 dry-run）")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--chapter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output"
    json_out = Path(args.json_out) if args.json_out else output_dir / "core_person_swap_repair.json"
    report = Path(args.report) if args.report else output_dir / "core_person_swap_repair.md"
    sources = _load_sources(root)
    now = _dt.datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    backup_root = output_dir / "repair_backups" / f"core_person_swaps_{stamp}"
    pattern = f"{args.chapter}/*_final.json" if args.chapter else "ch*/*_final.json"

    rows: List[Dict[str, Any]] = []
    complex_rows: List[Dict[str, Any]] = []
    applied_chunks = 0
    scanned = 0
    candidates = 0
    candidate_changes = 0
    for final_path in sorted(output_dir.glob(pattern)):
        cid = final_path.stem.rsplit("_final", 1)[0]
        source = sources.get(cid, "")
        if not source:
            continue
        scanned += 1
        final_data = _read_json(final_path)
        literary_data = _read_json(final_path.with_name(f"{cid}_literary.json"))
        raw_data = _read_json(final_path.with_name(f"{cid}_raw.json"))
        if not final_data or not literary_data or not raw_data:
            continue
        final = final_data.get("text")
        literary = literary_data.get("text")
        raw = raw_data.get("text")
        if not all(isinstance(x, str) and x for x in (final, literary, raw)):
            continue
        plan = _plan_chunk(cid, final, literary, raw, source)
        if plan.get("reason") != "safe_change":
            if plan.get("reason") in ("complex_alignment", "verification_failed"):
                complex_rows.append(plan)
            continue
        candidates += 1
        candidate_changes += len(plan["changes"])
        if args.limit is not None and candidates > max(0, args.limit):
            continue
        row = {k: v for k, v in plan.items() if k != "repaired_text"}
        rows.append(row)
        if not args.apply:
            continue
        backup_path = backup_root / final_path.parent.name / final_path.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_path, backup_path)
        final_data["text"] = plan["repaired_text"]
        meta = final_data.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta["core_person_swap_repair"] = {
            "schema_version": 1,
            "applied_at": now.isoformat(),
            "source": "literary",
            "changes": plan["changes"],
            "backup": str(backup_path.relative_to(root)),
        }
        final_data["metadata"] = meta
        _atomic_write_json(final_path, final_data)
        applied_chunks += 1

    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "scanned": scanned,
        "candidate_chunks": candidates,
        "candidate_changes": candidate_changes,
        "applied_chunks": applied_chunks,
        "rows": rows,
        "complex_rows": complex_rows,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, payload)
    print(json.dumps({
        "mode": payload["mode"],
        "scanned": scanned,
        "candidate_chunks": candidates,
        "candidate_changes": candidate_changes,
        "applied_chunks": applied_chunks,
        "complex": len(complex_rows),
        "json": str(json_out),
        "report": str(report),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
