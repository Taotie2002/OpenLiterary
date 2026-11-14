#!/usr/bin/env python3
"""Read-only audit: compare core entity mentions in final/raw/literary with source.

The audit is deliberately conservative.  It only raises a strong flag when the
English source and both independent Chinese drafts agree on an entity presence
that the final contradicts, or when the final introduces a core person absent
from all three references.  It never edits text.

Usage:
  python3 scripts/audit_name_grounding.py
  python3 scripts/audit_name_grounding.py --json-out output/name_grounding_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CORE: Dict[str, Dict[str, Any]] = {
    "LETO": {
        "zh": ["雷托"],
        "en": ["Leto"],
        "label": "雷托/Leto",
    },
    "PAUL": {
        "zh": ["保罗"],
        "en": ["Paul"],
        "label": "保罗/Paul",
    },
    "JESSICA": {
        "zh": ["杰西卡"],
        "en": ["Jessica"],
        "label": "杰西卡/Jessica",
    },
    "MOHIAM": {
        # 盖乌斯/莫希阿姆 are proper names.  Reverend Mother is a title that
        # may refer to Mohiam, so it is counted separately and never used as
        # the sole evidence for an extra-person flag.
        "zh": ["盖乌斯", "莫希阿姆"],
        "en": ["Mohiam", "Gaius"],
        "label": "盖乌斯·海伦·莫希阿姆/Mohiam",
    },
    "PITER": {
        "zh": ["皮特", "彼得"],
        "en": ["Piter"],
        "label": "皮特/Piter",
    },
    "TRUTHSAYER": {
        "zh": ["真言者", "真言师"],
        "en": ["Truthsayer"],
        "label": "真言者/Truthsayer（角色）",
    },
}

# Atreides variants are tracked separately because they are transliteration
# variants of one house, not different people.
ATREIDES_ZH = ("厄崔迪", "亚崔迪", "阿特雷德", "阿特雷迪斯")
ATREIDES_CANONICAL = "厄崔迪"


def _count_en(text: str, needles: Iterable[str]) -> int:
    total = 0
    for needle in needles:
        total += len(re.findall(r"\b" + re.escape(needle) + r"\b", text))
    return total


def _count_zh(text: str, needles: Iterable[str]) -> int:
    # Overlapping variants are not expected for these core names.  Count each
    # occurrence once; the values are used only for differential comparisons.
    return sum(text.count(needle) for needle in needles)


def _read_text(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return data.get("text", "") if isinstance(data, dict) else ""


def _load_source_chunks(root: Path) -> Dict[str, str]:
    db = root / "db" / "workflow.db"
    out: Dict[str, str] = {}
    if not db.exists():
        return out
    con = sqlite3.connect(str(db))
    try:
        for chunk_id, text in con.execute(
                "SELECT chunk_id, text_content FROM chunk_tasks"):
            if isinstance(chunk_id, str) and isinstance(text, str):
                out[chunk_id] = text
    finally:
        con.close()
    return out


def audit_chunk(chunk_id: str, source: str, final: str,
                literary: str, raw: str) -> Dict[str, Any]:
    counts = {}
    for key, spec in CORE.items():
        counts[key] = {
            "source": _count_en(source, spec["en"]),
            "final": _count_zh(final, spec["zh"]),
            "literary": _count_zh(literary, spec["zh"]),
            "raw": _count_zh(raw, spec["zh"]),
        }
    atreides = {
        "source": _count_en(source, ["Atreides"]),
        "final": _count_zh(final, ATREIDES_ZH),
        "literary": _count_zh(literary, ATREIDES_ZH),
        "raw": _count_zh(raw, ATREIDES_ZH),
        "final_variants": {v: final.count(v) for v in ATREIDES_ZH if final.count(v)},
    }

    flags: List[Dict[str, Any]] = []

    def add(kind: str, entity: str, severity: str, detail: str) -> None:
        flags.append({
            "kind": kind,
            "entity": entity,
            "severity": severity,
            "detail": detail,
        })

    # 1. Entity present in source and both references, absent from final.
    for key, c in counts.items():
        if c["source"] > 0 and c["final"] == 0 and c["literary"] > 0 and c["raw"] > 0:
            add("missing_in_final", key, "high",
                f"source={c['source']} literary={c['literary']} raw={c['raw']} final=0")

    # 2. Final-only core person (all three references agree it is absent).
    for key, c in counts.items():
        if c["source"] == 0 and c["final"] > 0 and c["literary"] == 0 and c["raw"] == 0:
            add("extra_in_final", key, "high",
                f"final={c['final']} source/literary/raw=0")

    # 3. Leto -> Paul substitution.  Two forms are reported:
    #    - total substitution: final contains no Leto at all;
    #    - balanced shift: final loses exactly as many Leto mentions as it
    #      gains Paul mentions relative to the conservative reference
    #      min(literary, raw).  This catches mixed scenes where the source
    #      contains both names.
    leto, paul = counts["LETO"], counts["PAUL"]
    if leto["source"] > paul["source"]:
        leto_ref = min(leto["literary"], leto["raw"])
        paul_ref = min(paul["literary"], paul["raw"])
        delta_leto = leto["final"] - leto_ref
        delta_paul = paul["final"] - paul_ref
        if delta_leto < 0 and delta_paul > 0 and abs(delta_leto) == delta_paul:
            add("leto_to_paul", "LETO", "critical",
                f"source Leto={leto['source']} Paul={paul['source']}; "
                f"final Leto={leto['final']} Paul={paul['final']}; "
                f"ref Leto={leto_ref} Paul={paul_ref}; "
                f"shift Leto{delta_leto} Paul+{delta_paul}")

    # 4. Jessica -> Mohiam substitution (proper-name evidence only).  A
    #    balanced shift is the mechanical signature of the historical
    #    "盖乌斯夫人" contamination: one Jessica mention disappears and one
    #    Mohiam mention appears.
    jess, moh = counts["JESSICA"], counts["MOHIAM"]
    if jess["source"] > 0 and moh["source"] == 0:
        jess_ref = min(jess["literary"], jess["raw"])
        moh_ref = min(moh["literary"], moh["raw"])
        delta_jess = jess["final"] - jess_ref
        delta_moh = moh["final"] - moh_ref
        if delta_jess < 0 and delta_moh > 0 and abs(delta_jess) == delta_moh:
            add("jessica_to_mohiam", "JESSICA", "critical",
                f"source Jessica={jess['source']} Mohiam=0; "
                f"final Jessica={jess['final']} Mohiam={moh['final']}; "
                f"ref Jessica={jess_ref} Mohiam={moh_ref}; "
                f"shift Jessica{delta_jess} Mohiam+{delta_moh}")

    # 5. Extra proper Mohiam in final when source and both references lack it.
    if moh["source"] == 0 and moh["final"] > 0 and moh["literary"] == 0 and moh["raw"] == 0:
        add("mohiam_inserted", "MOHIAM", "high",
            f"final Mohiam={moh['final']} but source/literary/raw=0")

    # 6. Atreides transliteration drift inside one chunk.
    variants = atreides["final_variants"]
    if len(variants) > 1:
        add("atreides_variant_mix", "ATREIDES", "medium",
            json.dumps(variants, ensure_ascii=False, sort_keys=True))

    severity_rank = {"critical": 0, "high": 1, "medium": 2}
    flags.sort(key=lambda f: (severity_rank.get(f["severity"], 9), f["kind"], f["entity"]))
    return {
        "chunk_id": chunk_id,
        "chapter": chunk_id.split("_", 1)[0] if "_" in chunk_id else "",
        "flags": flags,
        "counts": counts,
        "atreides": atreides,
        "missing_artifacts": {
            "final": not bool(final),
            "literary": not bool(literary),
            "raw": not bool(raw),
        },
    }


def run(root: Path, output_dir: Path, chapter: str | None, limit: int | None) -> Dict[str, Any]:
    sources = _load_source_chunks(root)
    pattern = f"{chapter}/*_final.json" if chapter else "ch*/*_final.json"
    rows: List[Dict[str, Any]] = []
    scanned = 0
    for final_path in sorted(output_dir.glob(pattern)):
        chunk_id = final_path.stem.rsplit("_final", 1)[0]
        source = sources.get(chunk_id, "")
        if not source:
            continue
        scanned += 1
        final = _read_text(final_path)
        literary = _read_text(final_path.with_name(f"{chunk_id}_literary.json"))
        raw = _read_text(final_path.with_name(f"{chunk_id}_raw.json"))
        row = audit_chunk(chunk_id, source, final, literary, raw)
        if row["flags"] or any(row["missing_artifacts"].values()):
            rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    by_kind: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for row in rows:
        for flag in row["flags"]:
            by_kind[flag["kind"]] = by_kind.get(flag["kind"], 0) + 1
            by_severity[flag["severity"]] = by_severity.get(flag["severity"], 0) + 1
    return {
        "schema_version": 1,
        "scanned": scanned,
        "flagged_chunks": len(rows),
        "flag_counts": dict(sorted(by_kind.items())),
        "severity_counts": dict(sorted(by_severity.items())),
        "rows": rows,
    }


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# 源文对照姓名审计",
        "",
        f"- 扫描 chunk: {payload['scanned']}",
        f"- 命中 chunk: {payload['flagged_chunks']}",
        f"- 按类型: `{json.dumps(payload['flag_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- 按严重度: `{json.dumps(payload['severity_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "规则：只有英文源文与 raw/literary 两份独立中文稿同时支持、而 final 相反，",
        "或 final 引入三份参考都不存在的核心人物时，才记强证据。",
        "",
        "## 命中 chunk",
        "",
    ]
    rows = payload["rows"]
    if not rows:
        lines.append("（无）")
    for row in rows:
        lines.append(f"### `{row['chunk_id']}`")
        for flag in row["flags"]:
            lines.append(f"- [{flag['severity']}] {flag['kind']} / {flag['entity']}: {flag['detail']}")
        miss = [k for k, v in row["missing_artifacts"].items() if v]
        if miss:
            lines.append(f"- 缺制品: {', '.join(miss)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="源文对照姓名审计（只读）")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--chapter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output"
    json_out = Path(args.json_out) if args.json_out else output_dir / "name_grounding_audit.json"
    report = Path(args.report) if args.report else output_dir / "name_grounding_audit.md"
    payload = run(root, output_dir, args.chapter, args.limit)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, payload)
    print(json.dumps({
        "scanned": payload["scanned"],
        "flagged_chunks": payload["flagged_chunks"],
        "flag_counts": payload["flag_counts"],
        "severity_counts": payload["severity_counts"],
        "json": str(json_out),
        "report": str(report),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
