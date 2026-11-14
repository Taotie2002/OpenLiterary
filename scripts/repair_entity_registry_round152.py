#!/usr/bin/env python3
"""Round 152: remove known cross-person sources from entity_registry.

This is deliberately narrower than the general cleaner.  It only repairs
surfaces established by source/final comparison:
  * Paul must not own Duke Leto source aliases;
  * Mohiam must not own role words as replaceable variants;
  * Atreides-family duplicates keep one canonical winner.

Dry-run by default.  ``--apply`` backs up the SQLite database first.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROLE_VARIANTS = frozenset({
    "圣母", "真言者", "尊母", "老妇人", "老妪", "老圣母", "圣母们",
})
ATREIDES_FAMILY_WINNER = "厄崔迪家族"


def _loads(raw: Any) -> List[str]:
    if not isinstance(raw, str):
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [x for x in value if isinstance(x, str)] if isinstance(value, list) else []


def _dumps(values: Iterable[str]) -> str:
    return json.dumps(list(dict.fromkeys(values)), ensure_ascii=False)


def repair_registry(conn: sqlite3.Connection, dry_run: bool = True) -> Dict[str, Any]:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(entity_registry)")}
    required = {"id", "canonical_name", "source_names", "variant_names",
                "context_names", "quarantined_names", "pending_review",
                "review_reason"}
    if not required <= cols:
        raise ValueError("entity_registry schema missing: " + repr(sorted(required - cols)))

    rows = conn.execute(
        "SELECT id, canonical_name, source_names, variant_names, context_names, "
        "quarantined_names, pending_review, review_reason FROM entity_registry"
    ).fetchall()
    changes: List[Dict[str, Any]] = []
    updates: List[tuple] = []

    for rid, canonical, sn_raw, vn_raw, ctx_raw, q_raw, pending, reason in rows:
        source_names = _loads(sn_raw)
        variants = _loads(vn_raw)
        contexts = _loads(ctx_raw)
        quarantined = _loads(q_raw)
        new_sources, new_variants = list(source_names), list(variants)
        new_contexts, new_quarantined = list(contexts), list(quarantined)
        local: Dict[str, Any] = {}

        if canonical == "保罗·厄崔迪":
            bad = [s for s in source_names
                   if "Leto" in s and "Paul" not in s]
            if bad:
                new_sources = [s for s in new_sources if s not in bad]
                new_quarantined.extend(bad)
                local["removed_paul_foreign_sources"] = bad

        if canonical == "盖乌斯·海伦·莫希阿姆":
            bad = [v for v in variants if v in ROLE_VARIANTS]
            if bad:
                new_variants = [v for v in new_variants if v not in bad]
                new_contexts.extend(bad)
                local["moved_role_variants_to_context"] = bad

        is_atreides_family = (
            canonical.endswith("家族")
            and (
                any("Atreides" in s for s in source_names)
                or any(token in canonical for token in (
                    "厄崔迪", "阿特雷迪斯", "阿特雷德斯", "阿特雷德", "亚崔迪"))
            )
        )
        if is_atreides_family and canonical != ATREIDES_FAMILY_WINNER:
            if pending != 1 or not str(reason or "").startswith("Round152 "):
                local["demoted_atreides_duplicate"] = True
                pending = 1
                reason = "Round152 duplicate Atreides family; canonical=厄崔迪家族"
        elif canonical == ATREIDES_FAMILY_WINNER:
            if pending != 0:
                local["promoted_atreides_winner"] = True
                pending = 0
                reason = "Round152 canonical Atreides family winner"

        if local:
            changes.append({
                "id": rid,
                "canonical": canonical,
                **local,
                "source_names": new_sources,
                "variant_names": new_variants,
                "context_names": new_contexts,
                "quarantined_names": new_quarantined,
                "pending_review": pending,
                "review_reason": reason,
            })
            updates.append((
                _dumps(new_sources), _dumps(new_variants),
                _dumps(new_contexts), _dumps(new_quarantined),
                int(pending), reason, rid,
            ))

    if not dry_run and updates:
        conn.executemany(
            "UPDATE entity_registry SET source_names=?, variant_names=?, "
            "context_names=?, quarantined_names=?, pending_review=?, "
            "review_reason=? WHERE id=?",
            updates,
        )
        conn.commit()

    return {
        "rows": len(rows),
        "changed_rows": len(changes),
        "changes": changes,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Round 152 entity_registry 定向清理")
    parser.add_argument("--db", default=str(root / "db" / "decision_db.sqlite"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-out", default=str(root / "output" / "round152_entity_registry.json"))
    args = parser.parse_args()

    db = Path(args.db)
    now = _dt.datetime.now().astimezone()
    backup = None
    if args.apply:
        backup = db.parent / "repair_backups" / (
            "decision_db_round152_" + now.strftime("%Y%m%d_%H%M%S") + ".sqlite")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db, backup)

    conn = sqlite3.connect(str(db))
    try:
        result = repair_registry(conn, dry_run=not args.apply)
    finally:
        conn.close()
    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "db": str(db),
        "backup": str(backup) if backup else None,
        **result,
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "mode", "rows", "changed_rows", "backup", "json_out") if k in payload}
        | {"json": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
