#!/usr/bin/env python3
"""Round 153: repair final texts with source-grounded defects.

The rules are deliberately chunk-scoped and exact.  ``ch01_chunk072`` is the
only final whose body stops mid-sentence while a complete literary candidate
exists; the other two chunks contain mistranslations proven by the English
source (``Customs differ``, ``Arrakeen``, and ``crowd`` rendered as 民族).

Dry-run by default.  ``--apply`` backs up every changed final before writing.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ROOT = Path(__file__).resolve().parent.parent


_TEXT_RULES: Dict[str, List[Tuple[str, str, str]]] = {
    "ch04_chunk024": [
        ("我们已经同这些民族联合了。", "我们已经同这些人联合了。",
         "people_not_nation"),
        ("“水税不一样。”", "“习俗不一样。”", "customs_not_water_tax"),
        ("他们还会帮我们抵达阿拉基斯吗？", "他们还会帮我们抵达阿拉肯吗？",
         "arrakeen_city_not_arrakis"),
    ],
    "ch05_chunk113": [
        ("随着民族看到仆人们", "随着人们看到仆人们", "crowd_not_nation"),
        ("正确的。民族简直像发了疯", "正确的。人群简直像发了疯",
         "crowd_not_nation"),
        ("不能就这样让民族回家", "不能就这样让人们回家",
         "crowd_not_nation"),
    ],
}

_GLOBAL_RULES: List[Tuple[str, str, str]] = [
    ("雷托公爵公爵", "雷托公爵", "duplicate_title_after_name_unification"),
]

_LITERARY_REPLACEMENTS = (
    ("阿特雷迪斯家族", "厄崔迪家族"),
    ("就连我们卡拉丹城堡可怜的潘迪米也不例外。",
     "就连我们卡拉丹那可怜的潘迪米也不例外。"),
)


def _body(text: str) -> str:
    marker = text.find("\n[^")
    return text[:marker] if marker >= 0 else text


def repair_text(chunk_id: str, text: str,
                literary_text: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    changes: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    if literary_text is not None:
        final_body = _body(text).rstrip()
        literary_body = _body(literary_text).rstrip()
        terminal = "。！？…”’\"」』）]】"
        is_gap = (
            len(final_body) < 0.65 * len(literary_body)
            and final_body
            and final_body[-1] not in terminal
            and literary_body
            and literary_body[-1] in terminal
        )
        if is_gap:
            restored = literary_text
            for old, new in _LITERARY_REPLACEMENTS:
                if old in restored:
                    restored = restored.replace(old, new)
                    changes.append({"rule": "complete_from_literary", "old": old,
                                    "new": new, "count": 1})
            changes.append({
                "rule": "restore_complete_body",
                "old": f"final body chars={len(final_body)}",
                "new": f"literary body chars={len(literary_body)}",
                "count": 1,
            })
            return restored, changes, skipped
        skipped.append({"rule": "restore_complete_body",
                        "reason": "not_a_truncated_final"})

    fixed = text
    for old, new, rule in _GLOBAL_RULES:
        count = fixed.count(old)
        if count:
            fixed = fixed.replace(old, new)
            changes.append({"rule": rule, "old": old, "new": new, "count": count})
    for old, new, rule in _TEXT_RULES.get(chunk_id, []):
        count = fixed.count(old)
        if count != 1:
            skipped.append({"rule": rule, "expected": 1, "found": count})
            continue
        fixed = fixed.replace(old, new)
        changes.append({"rule": rule, "old": old, "new": new, "count": count})
    return fixed, changes, skipped


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp-round153")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Round 153 final-gap repair")
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-out", default="output/round153_final_gaps.json")
    parser.add_argument("--md-out", default="output/round153_final_gaps.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    now = _dt.datetime.now().astimezone()
    backup_dir = root / "output" / "repair_backups" / (
        "round153_final_gaps_" + now.strftime("%Y%m%d_%H%M%S"))
    rows: List[Dict[str, Any]] = []

    target_ids = sorted(set(_TEXT_RULES) | {
        "ch01_chunk072", "ch01_chunk069", "ch02_chunk042", "ch04_chunk002",
    })
    for cid in target_ids:
        chapter = cid.split("_", 1)[0]
        final_path = root / "output" / chapter / (cid + "_final.json")
        literary_path = root / "output" / chapter / (cid + "_literary.json")
        final_data = _read_json(final_path)
        if not final_data or not isinstance(final_data.get("text"), str):
            rows.append({"chunk_id": cid, "status": "missing_final"})
            continue
        literary_data = _read_json(literary_path)
        literary_text = (literary_data or {}).get("text")
        fixed, changes, skipped = repair_text(
            cid, final_data["text"],
            literary_text if isinstance(literary_text, str) else None,
        )
        row = {"chunk_id": cid, "changes": changes, "skipped": skipped,
               "status": "planned" if changes else "unchanged"}
        if changes and args.apply:
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / chapter / final_path.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_path, backup)
            final_data["text"] = fixed
            metadata = dict(final_data.get("metadata") or {})
            metadata["round153_final_gap_repair"] = {
                "applied_at": now.isoformat(),
                "rules": changes,
                "backup": str(backup.relative_to(root)),
            }
            final_data["metadata"] = metadata
            _atomic_write_json(final_path, final_data)
            row["status"] = "applied"
            row["backup"] = str(backup.relative_to(root))
        rows.append(row)

    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "rows": rows,
        "changed": sum(1 for r in rows if r.get("status") == "applied"),
        "planned": sum(1 for r in rows if r.get("status") == "planned"),
    }
    out = root / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    md = root / args.md_out
    lines = ["# Round 153 final gaps", "",
             f"- mode: {payload['mode']}",
             f"- changed: {payload['changed']}",
             f"- planned: {payload['planned']}", ""]
    for row in rows:
        lines.append(f"## {row['chunk_id']} — {row.get('status')}")
        for change in row.get("changes") or []:
            lines.append(f"- {change['rule']}: {change['old']} → {change['new']}")
        for skip in row.get("skipped") or []:
            lines.append(f"- skipped {skip['rule']}: {skip.get('reason') or skip}")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"mode": payload["mode"], "changed": payload["changed"],
                      "planned": payload["planned"], "json": str(out),
                      "report": str(md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
