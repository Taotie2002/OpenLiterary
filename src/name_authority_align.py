"""Round 158 Phase 1.5a: deterministic, two-phase name authority alignment.

The planner treats the Decision DB as the hard terminology authority and the
entity registry as a derived index.  It never rewrites translated prose.  A
conflicting registry alias is either:

* quarantined from ``source_names`` / ``variant_names`` while the Decision DB
  mapping remains untouched, or
* when the registry's canonical form itself is the only drift and the target
  form has no competing owner, renamed to the Decision DB translation.
* when two case/spacing-equivalent Decision rows disagree, a book-wide corpus
  consensus may align the minority Decision and registry canonical.  This is
  deliberately limited to plain CJK targets with a >=3-chunk majority and no
  locked decision or competing registry canonical.

Ambiguous multi-decision aliases, multi-owner aliases and conflicting target
canonicals are emitted as ``manual_review`` and never applied.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.name_authority_audit import (
    FORM_ALIGN_SIMILARITY,
    _similarity,
    _decision_for_alias,
    _decision_rows,
    audit_name_authorities,
)
from src.core.decision_engine import canonical_is_title_only, canonical_tokens
from src.pending_backtrack_reconcile import (
    _name_compatible,
    _name_key,
    _phrase_key,
    _target_core,
    body_before_footnotes,
)


ALIGNMENT_SCHEMA_VERSION = 1
_ALIGN_ANNOTATION_RE = re.compile(r"[（(][^（()）]*[）)]")
_ALIGN_PLAIN_TARGET_RE = re.compile(r"^[\u4e00-\u9fff·]+$")
_ALIGN_CONSENSUS_MARKER_RE = re.compile(
    r"Round158 duplicate source form consensus:\s*([^|]+?)\s*->\s*([^|]+?)(?:\s*\||$)")


def _align_now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _align_read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _align_atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _align_atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _align_json_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        values = raw
    else:
        try:
            values = json.loads(str(raw) or "[]")
        except (TypeError, json.JSONDecodeError):
            return []
    if not isinstance(values, list):
        return []
    return [str(x).strip() for x in values if str(x).strip()]


def _align_json_dump(values: Iterable[str]) -> str:
    deduped: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        key = _phrase_key(text)
        if text and key and key not in seen:
            seen.add(key)
            deduped.append(text)
    return json.dumps(deduped, ensure_ascii=False)


def _align_plain_target(value: str) -> bool:
    return bool(_ALIGN_PLAIN_TARGET_RE.fullmatch(str(value or "").strip()))


def _align_form_corpus_evidence(output_dir: Path, targets: Iterable[str]) -> Dict[str, dict]:
    """Count each candidate form in delivered finals (footnote definitions excluded)."""
    wanted = [str(value).strip() for value in targets if str(value).strip()]
    evidence = {
        target: {"chunks": [], "occurrences": 0}
        for target in wanted
    }
    for path in sorted(Path(output_dir).glob("ch*/*_final.json")):
        payload = _align_read_json(path) or {}
        text = body_before_footnotes(str(payload.get("text") or ""))
        if not text:
            continue
        chunk_id = path.stem[:-6] if path.stem.endswith("_final") else path.stem
        for target in wanted:
            occurrences = text.count(target)
            if not occurrences:
                continue
            evidence[target]["chunks"].append(chunk_id)
            evidence[target]["occurrences"] += occurrences
    return evidence


def _align_duplicate_form_action(
    *,
    rows: List[dict],
    registry_row: dict,
    alias: str,
    records: List[dict],
    decisions: List[dict],
    evidence: Dict[str, dict],
    output_dir: Path,
) -> Tuple[Optional[dict], str]:
    """Plan a corpus-backed alignment for duplicate case/spacing Decisions."""
    targets = sorted(evidence)
    if len(targets) < 2 or not all(_align_plain_target(target) for target in targets):
        return None, "non_plain_duplicate_targets"
    for index, left in enumerate(targets):
        for right in targets[index + 1:]:
            if _similarity(left, right) < FORM_ALIGN_SIMILARITY:
                return None, "non_form_equivalent_targets"
    ranked = sorted(
        targets,
        key=lambda target: (
            len(evidence[target]["chunks"]),
            evidence[target]["occurrences"],
            -len(target),
        ),
        reverse=True,
    )
    winner = ranked[0]
    winner_chunks = len(evidence[winner]["chunks"])
    if winner_chunks < 3 or evidence[winner]["occurrences"] < 3:
        return None, "insufficient_corpus_majority"
    minority_limit = max(1, winner_chunks // 4)
    losers = ranked[1:]
    if any(len(evidence[loser]["chunks"]) > minority_limit for loser in losers):
        return None, "corpus_majority_too_narrow"

    canonical = str(registry_row.get("canonical_name") or "")
    if canonical not in targets:
        return None, "registry_canonical_is_third_form"
    conflicting = [
        decision for decision in decisions
        if str(decision.get("translation") or "").strip() in targets
        and str(decision.get("translation") or "").strip() != winner
    ]
    if not conflicting:
        return None, "no_stale_decision_form"
    if any(int(decision.get("locked") or 0) for decision in conflicting):
        return None, "minority_decision_locked"
    if canonical != winner and int(registry_row.get("pending_review") or 0):
        return None, "pending_registry_row_requires_review"
    target_owner = _align_target_owner(
        rows, winner, int(registry_row.get("id") or 0))
    if target_owner is not None:
        return None, f"winner_registry_owned:{target_owner}"

    stale_chunks = sorted({
        chunk_id
        for loser in losers
        for chunk_id in evidence[loser]["chunks"]
    })
    action = {
        "action": "align_decision_forms",
        "registry_id": int(registry_row.get("id") or 0),
        "alias": alias,
        "canonical_before": canonical,
        "canonical_after": winner,
        "decision_ids": [int(decision["id"]) for decision in conflicting],
        "decisions_before": [
            {"decision_id": int(decision["id"]),
             "source_key": str(decision.get("source_key") or ""),
             "translation": str(decision.get("translation") or "")}
            for decision in conflicting
        ],
        "new_translation": winner,
        "corpus_evidence": {
            target: {
                "chunks": len(evidence[target]["chunks"]),
                "occurrences": evidence[target]["occurrences"],
            }
            for target in targets
        },
        "queued_chunks": stale_chunks,
        "aliases_to_quarantine": ([canonical] if canonical != winner else []),
        "reason": "duplicate_source_form_corpus_consensus",
    }
    return action, ""


def _align_resolved_pending_review_actions(
    *,
    rows: List[dict],
    decision_index: Dict[str, List[dict]],
    output_dir: Path,
) -> List[dict]:
    """Clear a pending registry review only when a prior consensus fixed it.

    This does not infer that arbitrary ``pending_review`` rows are safe.  It
    trusts only the machine-readable marker written by
    ``align_decision_forms`` and rechecks the Decision translations and the
    delivered corpus at the current turn.
    """
    candidates: List[tuple[dict, str]] = []
    canonicals: List[str] = []
    canonical_keys: Counter = Counter(
        _name_key(str(row.get("canonical_name") or "")) for row in rows)
    for row in rows:
        if not int(row.get("pending_review") or 0):
            continue
        canonical = str(row.get("canonical_name") or "").strip()
        if not canonical or not _align_plain_target(canonical):
            continue
        if canonical_keys.get(_name_key(canonical), 0) != 1:
            continue
        markers = _ALIGN_CONSENSUS_MARKER_RE.findall(
            str(row.get("review_reason") or ""))
        if not markers:
            continue
        _old_form, aligned = markers[-1]
        if _name_key(aligned) != _name_key(canonical):
            continue
        sources = [str(value) for value in row.get("source_names") or []]
        if not sources:
            continue
        matched: List[dict] = []
        for source in sources:
            matched.extend(_decision_for_alias(source, decision_index))
        translations = {
            str(decision.get("translation") or "").strip()
            for decision in matched
            if str(decision.get("translation") or "").strip()
        }
        if not translations or any(
                not _name_compatible(canonical, translation)
                for translation in translations):
            continue
        candidates.append((row, canonical))
        canonicals.append(canonical)
    evidence = _align_form_corpus_evidence(output_dir, canonicals)
    actions: List[dict] = []
    for row, canonical in candidates:
        chunks = len((evidence.get(canonical) or {}).get("chunks") or [])
        if chunks < 3:
            continue
        actions.append({
            "action": "clear_resolved_pending_review",
            "registry_id": int(row.get("id") or 0),
            "canonical_before": canonical,
            "pending_review_before": 1,
            "review_reason_before": str(row.get("review_reason") or ""),
            "corpus_chunks": chunks,
            "reason": "duplicate_source_form_consensus_resolved",
        })
    return actions


def _align_table_exists(con: sqlite3.Connection, name: str) -> bool:
    return bool(con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())


def _align_load_registry(con: sqlite3.Connection) -> Tuple[List[dict], str]:
    if not _align_table_exists(con, "entity_registry"):
        return [], "entity_registry_missing_table"
    cols = {row[1] for row in con.execute("PRAGMA table_info(entity_registry)")}
    required = {"id", "canonical_name", "source_names"}
    missing = sorted(required - cols)
    if missing:
        return [], "entity_registry_schema_incompatible:" + ",".join(missing)
    if "quarantined_names" not in cols:
        return [], "entity_registry_missing_quarantined_names"

    def expr(name: str, default: str) -> str:
        return name if name in cols else f"{default} AS {name}"

    select = [
        "id", "canonical_name", "source_names",
        expr("variant_names", "NULL"),
        "quarantined_names",
        expr("confidence", "'low'"),
        expr("pending_review", "0"),
        expr("review_reason", "NULL"),
        expr("updated_at", "NULL"),
    ]
    rows: List[dict] = []
    for row in con.execute(
            f"SELECT {', '.join(select)} FROM entity_registry ORDER BY id"):
        item = dict(row)
        for field in ("source_names", "variant_names", "quarantined_names"):
            item[field] = _align_json_list(item.get(field))
        rows.append(item)
    return rows, ""


def _align_owners_for_alias(rows: List[dict], alias: str) -> List[int]:
    key = _phrase_key(alias)
    return [
        int(row["id"]) for row in rows
        if any(_phrase_key(value) == key for value in row.get("source_names") or [])
    ]


def _align_target_owner(rows: List[dict], target: str,
                        exclude_id: int) -> Optional[int]:
    key = _name_key(target)
    for row in rows:
        if int(row["id"]) == int(exclude_id):
            continue
        if _name_key(row.get("canonical_name") or "") == key:
            return int(row["id"])
    return None


def _align_rows_by_canonical(rows: List[dict], canonical: str) -> List[dict]:
    key = _name_key(canonical)
    if not key:
        return []
    return [
        row for row in rows
        if _name_key(row.get("canonical_name") or "") == key
    ]


def build_name_authority_alignment(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a deterministic alignment proposal without mutating the DB."""
    root = Path(project_root)
    db_path = (Path(decision_db) if decision_db
               else root / "db" / "decision_db.sqlite")
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "name_authority_align.json"
    mpath = Path(md_out) if md_out else out_dir / "name_authority_align.md"
    result: Dict[str, Any] = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _align_now_iso(),
        "decision_db": str(db_path),
        "actions": [],
        "manual_review": [],
        "counts": {},
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    audit = audit_name_authorities(root, decision_db=db_path, output_dir=out_dir)
    result["audit_before"] = {
        "ok": bool(audit.get("ok")),
        "status": audit.get("status"),
        "blocking_count": int(audit.get("blocking_count") or 0),
        "counts": audit.get("counts") or {},
        "report": audit.get("md_path"),
    }
    if not db_path.exists():
        result["errors"].append("decision_db_missing")
        _align_atomic_json(jpath, result)
        _align_atomic_text(mpath, "# 名称权威对齐计划\n\nDecision DB 不存在。\n")
        return result

    for error in audit.get("errors") or []:
        result["errors"].append("audit_before:" + str(error))

    con = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows, schema_error = _align_load_registry(con)
        decisions, decision_index, decision_error = _decision_rows(con)
    finally:
        con.close()
    if schema_error or decision_error:
        for error in (schema_error, decision_error):
            if error:
                result["errors"].append(error)
        _align_atomic_json(jpath, result)
        _align_atomic_text(mpath, "# 名称权威对齐计划\n\n" + "\n".join(
            f"- 错误: `{error}`" for error in result["errors"]) + "\n")
        return result

    by_id = {int(row["id"]): row for row in rows}
    quarantine: Dict[int, Dict[str, Any]] = {}
    form_groups: Dict[Tuple[int, str], List[dict]] = {}
    decision_form_actions: List[dict] = []
    manual: List[dict] = []

    def add_quarantine(registry_id: int, alias: str, record: dict,
                       reason: str) -> None:
        if registry_id not in by_id:
            manual.append({
                "registry_id": registry_id, "alias": alias,
                "reason": "registry_row_missing", "evidence": record,
            })
            return
        slot = quarantine.get(registry_id)
        if slot is None:
            slot = {"aliases": {}, "kinds": set(), "decision_ids": set(),
                    "reasons": []}
            quarantine[registry_id] = slot
        slot["aliases"][_phrase_key(alias)] = alias
        slot["kinds"].add(str(record.get("kind") or ""))
        slot["decision_ids"].add(int(record.get("decision_id") or 0))
        slot["reasons"].append(reason)

    for record in audit.get("records") or []:
        rid = int(record.get("registry_id") or 0)
        alias = str(record.get("alias") or "")
        kind = str(record.get("kind") or "")
        if not rid or not alias:
            continue
        if kind == "form_align_candidate":
            form_groups.setdefault((rid, _phrase_key(alias)), []).append(record)
            continue
        if kind == "cross_person_alias_conflict":
            owner_ids = _align_owners_for_alias(rows, alias)
            if not owner_ids:
                owner_ids = [rid]
            for owner_id in sorted(set(owner_ids)):
                add_quarantine(
                    owner_id, alias, record,
                    f"cross_person:{','.join(record.get('alias_registry_owners') or [])}")
            continue
        add_quarantine(rid, alias, record, kind)

    align_by_id: Dict[int, Dict[str, Any]] = {}
    for (registry_id, alias_key), records in sorted(form_groups.items()):
        row = by_id.get(registry_id)
        alias = str(records[0].get("alias") or "")
        if not row:
            manual.append({"registry_id": registry_id, "alias": alias,
                           "reason": "registry_row_missing"})
            continue
        matched_decisions = _decision_for_alias(alias, decision_index)
        translations = {
            str(decision.get("translation") or "").strip()
            for decision in matched_decisions
            if str(decision.get("translation") or "").strip()
        }
        owners = set()
        for record in records:
            owners.update(record.get("alias_registry_owners") or [])
        if len(translations) != 1:
            evidence = _align_form_corpus_evidence(out_dir, translations)
            action, reason = _align_duplicate_form_action(
                rows=rows,
                registry_row=row,
                alias=alias,
                records=records,
                decisions=matched_decisions,
                evidence=evidence,
                output_dir=out_dir,
            )
            if action is not None:
                decision_form_actions.append(action)
            else:
                manual.append({
                    "registry_id": registry_id, "alias": alias,
                    "reason": "ambiguous_decision_forms",
                    "decision_translations": sorted(translations),
                    "corpus_evidence": evidence,
                    "blocked_reason": reason,
                })
            continue
        target = next(iter(translations))
        if _name_key(target) == _name_key(row.get("canonical_name") or ""):
            continue
        # A source alias may legitimately be shared by a short/titled/full form
        # family.  When one owner already carries the authoritative canonical
        # and is active, keep that owner and quarantine the alias only from the
        # drifting owners.  This is deterministic and avoids a false "manual
        # review" for a mere spelling drift.
        target_rows = _align_rows_by_canonical(rows, target)
        active_target_rows = [
            candidate for candidate in target_rows
            if not int(candidate.get("pending_review") or 0)
        ]
        if len(owners) > 1:
            if not target_rows:
                manual.append({
                    "registry_id": registry_id, "alias": alias,
                    "reason": "form_alias_multiple_owners",
                    "owners": sorted(owners),
                })
                continue
            if not active_target_rows:
                manual.append({
                    "registry_id": registry_id, "alias": alias,
                    "reason": "form_alias_target_owner_pending",
                    "owners": sorted(owners),
                    "target": target,
                })
                continue
            quarantined_any = False
            for owner in sorted(owners):
                if _name_key(owner) == _name_key(target):
                    continue
                owner_rows = _align_rows_by_canonical(rows, owner)
                if not owner_rows:
                    manual.append({
                        "registry_id": registry_id, "alias": alias,
                        "reason": "form_alias_owner_row_missing",
                        "owner": owner,
                    })
                    continue
                for owner_row in owner_rows:
                    if int(owner_row.get("id") or 0) == int(registry_id):
                        add_quarantine(
                            registry_id, alias, records[0],
                            f"form_target_owned:{target}")
                    else:
                        add_quarantine(
                            int(owner_row["id"]), alias, records[0],
                            f"form_target_owned:{target}")
                    quarantined_any = True
            if not quarantined_any:
                manual.append({
                    "registry_id": registry_id, "alias": alias,
                    "reason": "form_alias_no_drift_owner",
                    "owners": sorted(owners),
                    "target": target,
                })
            continue
        if int(row.get("pending_review") or 0):
            manual.append({
                "registry_id": registry_id, "alias": alias,
                "reason": "pending_registry_row_requires_review",
            })
            continue
        target_owner = _align_target_owner(rows, target, registry_id)
        if target_owner is not None:
            record = records[0]
            add_quarantine(
                registry_id, alias, record,
                f"target_canonical_owned_by:{target_owner}")
            continue
        previous = align_by_id.get(registry_id)
        if previous and _name_key(previous.get("canonical_after") or "") != _name_key(target):
            manual.append({
                "registry_id": registry_id,
                "aliases": sorted({previous.get("alias"), alias} - {None, ""}),
                "reason": "multiple_form_targets",
                "targets": sorted({
                    str(previous.get("canonical_after") or ""), target}),
            })
            align_by_id.pop(registry_id, None)
            continue
        align_by_id[registry_id] = {
            "action": "align_canonical",
            "registry_id": registry_id,
            "alias": alias,
            "canonical_before": str(row.get("canonical_name") or ""),
            "canonical_after": target,
            "decision_ids": sorted({
                int(record.get("decision_id") or 0) for record in records
                if int(record.get("decision_id") or 0)}),
            "reason": "single_form_drift_with_unique_target",
        }

    # A canonical rename is atomic at row level, but the row may carry other
    # hard aliases whose Decision translations still embed the old canonical
    # (for example ``B.G. -> 贝尼·杰赛里特（B.G.）``).  Keeping such an alias
    # after the rename would manufacture a new authority conflict.  Predict
    # that second-order effect now and quarantine only the affected aliases;
    # the Decision DB remains untouched and fail-closed.
    for registry_id, action in list(align_by_id.items()):
        row = by_id.get(registry_id)
        if not row:
            continue
        target = str(action.get("canonical_after") or "")
        form_alias_key = _phrase_key(str(action.get("alias") or ""))
        for alias in row.get("source_names") or []:
            if _phrase_key(alias) == form_alias_key:
                continue
            matches = _decision_for_alias(alias, decision_index)
            if not matches:
                continue
            translations = [
                str(decision.get("translation") or "").strip()
                for decision in matches
                if str(decision.get("translation") or "").strip()
            ]
            if not translations or all(
                    _name_compatible(target, translation)
                    for translation in translations):
                continue
            add_quarantine(
                registry_id, alias, {
                    "kind": "post_form_align_conflict",
                    "decision_id": int(matches[0].get("id") or 0),
                },
                "post_form_align_conflict:" + target)

    variant_actions, variant_manual = _align_locked_short_variants(
        rows, decisions)
    variant_actions = [
        action for action in variant_actions
        if int(action.get("registry_id") or 0) not in align_by_id
    ]
    manual.extend(variant_manual)

    actions: List[dict] = []
    for registry_id, slot in sorted(quarantine.items()):
        row = by_id.get(registry_id)
        if not row:
            continue
        aliases = sorted(slot["aliases"].values(), key=_phrase_key)
        actions.append({
            "action": "quarantine_aliases",
            "registry_id": registry_id,
            "canonical_before": str(row.get("canonical_name") or ""),
            "aliases": aliases,
            "kinds": sorted(slot["kinds"]),
            "decision_ids": sorted(did for did in slot["decision_ids"] if did),
            "reasons": list(dict.fromkeys(slot["reasons"])),
        })
    actions.extend(align_by_id.values())
    actions.extend(decision_form_actions)
    actions.extend(_align_resolved_pending_review_actions(
        rows=rows,
        decision_index=decision_index,
        output_dir=out_dir,
    ))
    actions.extend(variant_actions)
    # Apply alias removals before canonical renames on the same row: the rename
    # is guarded by the pre-plan canonical value, while quarantine only touches
    # alias columns and is safe to run first.
    action_order = {
        "quarantine_aliases": 0,
        "align_canonical": 1,
        "align_decision_forms": 2,
        "clear_resolved_pending_review": 2,
        "register_locked_variant": 3,
    }
    actions.sort(key=lambda item: (
        int(item.get("registry_id", 0)),
        action_order.get(str(item.get("action") or ""), 9),
        str(item.get("action") or ""),
    ))
    result["actions"] = actions
    result["manual_review"] = manual
    counts = Counter(str(action.get("action") or "") for action in actions)
    result["counts"] = {
        "quarantine_aliases": counts.get("quarantine_aliases", 0),
        "align_canonical": counts.get("align_canonical", 0),
        "align_decision_forms": counts.get("align_decision_forms", 0),
        "clear_resolved_pending_review": counts.get(
            "clear_resolved_pending_review", 0),
        "register_locked_variant": counts.get("register_locked_variant", 0),
        "manual_review": len(manual),
    }
    result["ok"] = not result["errors"] and not manual
    _align_atomic_json(jpath, result)
    _align_atomic_text(mpath, _align_markdown(result))
    return result


def _align_remove_alias(values: List[str], aliases: Iterable[str]) -> List[str]:
    remove = {_phrase_key(alias) for alias in aliases if _phrase_key(alias)}
    return [value for value in values if _phrase_key(value) not in remove]


def _align_db_backup(db_path: Path, backup_path: Path) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(str(db_path))
    destination = sqlite3.connect(str(backup_path))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def _align_active(rows: Iterable[dict]) -> List[dict]:
    return [
        row for row in rows
        if not int(row.get("pending_review") or 0)
        and str(row.get("confidence") or "high").lower() == "high"
    ]


def _align_locked_short_variants(
    rows: List[dict],
    decisions: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """Register a locked Decision short form as its active owner's variant.

    This is deliberately narrower than general alias synthesis: the source
    alias must have exactly one active owner, the locked target must be one
    non-generic CJK token already present in that owner's canonical, and the
    token must not be owned by any other active entity.
    """
    active = _align_active(rows)
    by_source: Dict[str, List[dict]] = {}
    for row in active:
        for alias in row.get("source_names") or []:
            key = _phrase_key(alias)
            if key:
                by_source.setdefault(key, []).append(row)

    actions: List[dict] = []
    manual: List[dict] = []
    seen: set = set()
    for decision in decisions:
        if not int(decision.get("locked") or 0):
            continue
        source_key = _phrase_key(decision.get("source_key") or "")
        translation = _target_core(str(decision.get("translation") or ""))
        if not source_key or not translation:
            continue
        if _ALIGN_ANNOTATION_RE.search(translation):
            translation = _ALIGN_ANNOTATION_RE.sub("", translation).strip()
        tokens = canonical_tokens(translation)
        if len(tokens) != 1:
            continue
        variant = tokens[0].strip()
        if (not variant or len(variant) < 2
                or canonical_is_title_only(variant)):
            continue
        owners = by_source.get(source_key) or []
        if len(owners) != 1:
            if len(owners) > 1 and any(
                    variant in canonical_tokens(row.get("canonical_name") or "")
                    for row in owners):
                manual.append({
                    "source_key": str(decision.get("source_key") or ""),
                    "decision_id": int(decision.get("id") or 0),
                    "variant": variant,
                    "reason": "locked_variant_multiple_active_owners",
                    "registry_ids": sorted(int(row.get("id") or 0)
                                           for row in owners),
                })
            continue
        owner = owners[0]
        rid = int(owner.get("id") or 0)
        canonical = str(owner.get("canonical_name") or "")
        canonical_parts = canonical_tokens(canonical)
        variant_key = _name_key(variant)
        canonical_key = _name_key(canonical)
        if not rid or variant_key == canonical_key:
            continue
        if variant not in canonical_parts:
            continue
        quarantined = _align_json_list(owner.get("quarantined_names"))
        if any(_name_key(value) == variant_key for value in quarantined):
            manual.append({
                "registry_id": rid,
                "canonical": canonical,
                "variant": variant,
                "decision_id": int(decision.get("id") or 0),
                "reason": "locked_variant_is_quarantined",
            })
            continue
        variants = _align_json_list(owner.get("variant_names"))
        if any(_name_key(value) == variant_key for value in variants):
            continue
        competing = []
        for other in active:
            if int(other.get("id") or 0) == rid:
                continue
            other_canonical = str(other.get("canonical_name") or "")
            other_forms = [other_canonical]
            other_forms.extend(_align_json_list(other.get("variant_names")))
            other_forms.extend(_align_json_list(other.get("quarantined_names")))
            if any(variant in str(value or "") for value in other_forms):
                competing.append(int(other.get("id") or 0))
        if competing:
            manual.append({
                "registry_id": rid,
                "canonical": canonical,
                "variant": variant,
                "decision_id": int(decision.get("id") or 0),
                "reason": "locked_variant_cross_person_conflict",
                "competing_registry_ids": sorted(competing),
            })
            continue
        key = (rid, variant_key)
        if key in seen:
            continue
        seen.add(key)
        actions.append({
            "action": "register_locked_variant",
            "registry_id": rid,
            "canonical_before": canonical,
            "variant": variant,
            "source_key": str(decision.get("source_key") or ""),
            "decision_ids": [int(decision.get("id") or 0)],
            "reason": "locked_decision_short_form",
        })
    return actions, manual


def align_name_authorities(
    project_root: Path,
    *,
    apply: bool = False,
    decision_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply the deterministic alignment proposal, or produce a read-only plan."""
    root = Path(project_root)
    db_path = (Path(decision_db) if decision_db
               else root / "db" / "decision_db.sqlite")
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "name_authority_align_apply.json"
    mpath = Path(md_out) if md_out else out_dir / "name_authority_align_apply.md"
    if not apply:
        return build_name_authority_alignment(
            root, decision_db=db_path, output_dir=out_dir,
            json_out=(Path(json_out) if json_out else out_dir / "name_authority_align.json"),
            md_out=(Path(md_out) if md_out else out_dir / "name_authority_align.md"),
        )

    plan = build_name_authority_alignment(root, decision_db=db_path,
                                          output_dir=out_dir)
    result: Dict[str, Any] = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "mode": "apply",
        "write_attempted": False,
        "generated_at": _align_now_iso(),
        "run_id": str(run_id or (
            _dt.datetime.now().strftime("name-align-%Y%m%dT%H%M%S") +
            f"-{os.getpid()}")).replace("/", "_"),
        "decision_db": str(db_path),
        "plan_counts": plan.get("counts") or {},
        "audit_before": plan.get("audit_before") or {},
        "actions": [],
        "manual_review": plan.get("manual_review") or [],
        "errors": list(plan.get("errors") or []),
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if result["errors"]:
        _align_atomic_json(jpath, result)
        _align_atomic_text(mpath, _align_markdown(result))
        return result
    eligible = [
        action for action in (plan.get("actions") or [])
        if action.get("action") in {
            "quarantine_aliases", "align_canonical", "align_decision_forms",
            "clear_resolved_pending_review", "register_locked_variant",
        }
    ]
    result["write_attempted"] = bool(eligible)
    if not eligible:
        result["applied_count"] = 0
        result["ok"] = bool((plan.get("audit_before") or {}).get("ok"))
        result["after_audit"] = plan.get("audit_before") or {}
        result["audit_after"] = plan.get("audit_before") or {}
        _align_atomic_json(jpath, result)
        _align_atomic_text(mpath, _align_markdown(result))
        return result

    backup_path = out_dir / "name_authority_align" / result["run_id"] / "decision_db.sqlite.bak"
    try:
        _align_db_backup(db_path, backup_path)
    except Exception as exc:  # noqa: BLE001 - do not mutate without a backup
        result["errors"].append(f"backup_failed:{type(exc).__name__}:{exc}")
        _align_atomic_json(jpath, result)
        _align_atomic_text(mpath, _align_markdown(result))
        return result
    result["backup_path"] = str(backup_path)

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(entity_registry)")}
        updated_expr = ", updated_at = CURRENT_TIMESTAMP" if "updated_at" in cols else ""
        variant_expr = "variant_names" if "variant_names" in cols else "NULL AS variant_names"
        reason_expr = "review_reason" if "review_reason" in cols else "NULL AS review_reason"
        con.execute("BEGIN IMMEDIATE")
        for action in eligible:
            out_action = dict(action)
            out_action["status"] = "kept"
            rid = int(action.get("registry_id") or 0)
            row = con.execute(
                f"SELECT id, canonical_name, source_names, {variant_expr}, "
                f"quarantined_names, pending_review, {reason_expr} "
                "FROM entity_registry WHERE id=?", (rid,)).fetchone()
            if not row:
                out_action["status"] = "stale_row_missing"
                result["actions"].append(out_action)
                continue
            canonical = str(row["canonical_name"] or "")
            if canonical != str(action.get("canonical_before") or ""):
                out_action["status"] = "stale_canonical_changed"
                result["actions"].append(out_action)
                continue

            if action.get("action") == "align_decision_forms":
                winner = str(action.get("new_translation") or "").strip()
                before_rows = action.get("decisions_before") or []
                if not winner or not before_rows:
                    out_action["status"] = "invalid_decision_plan"
                    result["actions"].append(out_action)
                    continue
                if canonical != winner:
                    winner_key = _name_key(winner)
                    duplicate = None
                    for other_id, other_canonical in con.execute(
                            "SELECT id, canonical_name FROM entity_registry "
                            "WHERE id!=?", (rid,)):
                        if _name_key(other_canonical or "") == winner_key:
                            duplicate = other_id
                            break
                    if duplicate:
                        raise RuntimeError(
                            f"decision_form_target_registry_conflict:{duplicate}")
                for before in before_rows:
                    did = int(before.get("decision_id") or 0)
                    decision = con.execute(
                        "SELECT source_key, translation, locked, reason "
                        "FROM decision_db WHERE id=?", (did,)).fetchone()
                    if not decision:
                        raise RuntimeError(f"decision_form_row_missing:{did}")
                    if int(decision["locked"] or 0):
                        raise RuntimeError(f"decision_form_locked:{did}")
                    if str(decision["source_key"]) != str(
                            before.get("source_key") or ""):
                        raise RuntimeError(f"decision_form_source_changed:{did}")
                    if str(decision["translation"]) != str(
                            before.get("translation") or ""):
                        raise RuntimeError(
                            f"decision_form_translation_changed:{did}")
                    reason = str(decision["reason"] or "")
                    marker = (
                        "Round 158 duplicate source form consensus: "
                        f"{decision['translation']} -> {winner}"
                    )
                    if marker not in reason:
                        reason = (reason.rstrip() + " | " + marker).strip(" |")
                    con.execute(
                        "UPDATE decision_db SET translation=?, reason=? WHERE id=?",
                        (winner, reason, did))

                sources = _align_json_list(row["source_names"])
                variants = _align_json_list(row["variant_names"])
                quarantined = _align_json_list(row["quarantined_names"])
                aliases_to_quarantine = [
                    str(value) for value in
                    action.get("aliases_to_quarantine") or []
                    if str(value).strip()
                ]
                if aliases_to_quarantine:
                    sources = _align_remove_alias(sources, aliases_to_quarantine)
                    variants = _align_remove_alias(variants, aliases_to_quarantine)
                    qkeys = {_phrase_key(value) for value in quarantined}
                    for alias in aliases_to_quarantine:
                        key = _phrase_key(alias)
                        if key and key not in qkeys:
                            quarantined.append(alias)
                            qkeys.add(key)
                assignments = ["canonical_name=?",
                               "source_names=?", "quarantined_names=?"]
                params: List[Any] = [
                    winner, _align_json_dump(sources),
                    _align_json_dump(quarantined)]
                if "variant_names" in cols:
                    assignments.append("variant_names=?")
                    params.append(_align_json_dump(variants))
                if "review_reason" in cols:
                    old_reason = str(row["review_reason"] or "")
                    note = (
                        "Round158 duplicate source form consensus: "
                        f"{canonical} -> {winner}"
                    )
                    assignments.append("review_reason=?")
                    params.append(
                        (old_reason + " | " if old_reason else "") + note)
                if "updated_at" in cols:
                    assignments.append("updated_at=CURRENT_TIMESTAMP")
                params.append(rid)
                con.execute(
                    "UPDATE entity_registry SET " + ", ".join(assignments)
                    + " WHERE id=?", tuple(params))

                queued = [str(value) for value in action.get("queued_chunks") or []]
                first = before_rows[0]
                did = int(first.get("decision_id") or 0)
                source_key = str(first.get("source_key") or "")
                for chunk_id in queued:
                    existing = con.execute(
                        "SELECT decision_id, source_key FROM pending_backtrack "
                        "WHERE chunk_id=?", (chunk_id,)).fetchone()
                    if existing and (
                            int(existing["decision_id"] or 0) != did
                            or str(existing["source_key"]) != source_key):
                        raise RuntimeError(
                            f"decision_form_pending_conflict:{chunk_id}")
                    con.execute(
                        "INSERT INTO pending_backtrack (chunk_id, decision_id, "
                        "source_key, queue_rounds, created_at, updated_at) "
                        "VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                        "ON CONFLICT(chunk_id) DO UPDATE SET "
                        "decision_id=excluded.decision_id, "
                        "source_key=excluded.source_key, "
                        "queue_rounds=COALESCE(pending_backtrack.queue_rounds,0)+1, "
                        "updated_at=CURRENT_TIMESTAMP",
                        (chunk_id, did, source_key))
                out_action["status"] = "applied"
                result["actions"].append(out_action)
                continue

            if action.get("action") == "clear_resolved_pending_review":
                expected_reason = str(action.get("review_reason_before") or "")
                if int(row["pending_review"] or 0) != 1:
                    out_action["status"] = "stale_pending_review_changed"
                    result["actions"].append(out_action)
                    continue
                if str(row["review_reason"] or "") != expected_reason:
                    out_action["status"] = "stale_review_reason_changed"
                    result["actions"].append(out_action)
                    continue
                note = (
                    "Round158 duplicate source form consensus: "
                    "pending review resolved by Decision and corpus evidence"
                )
                new_reason = (expected_reason.rstrip() + " | " + note).strip(" |")
                assignments = ["pending_review=0", "review_reason=?"]
                params: List[Any] = [new_reason]
                if "updated_at" in cols:
                    assignments.append("updated_at=CURRENT_TIMESTAMP")
                params.append(rid)
                con.execute(
                    "UPDATE entity_registry SET " + ", ".join(assignments)
                    + " WHERE id=? AND pending_review=1 AND review_reason=?",
                    tuple(params + [expected_reason]))
                out_action["status"] = "applied"
                result["actions"].append(out_action)
                continue

            if action.get("action") == "align_canonical":
                target = str(action.get("canonical_after") or "")
                target_key = _name_key(target)
                duplicate = None
                for other_id, other_canonical in con.execute(
                        "SELECT id, canonical_name FROM entity_registry WHERE id!=?",
                        (rid,)):
                    if _name_key(other_canonical or "") == target_key:
                        duplicate = other_id
                        break
                if duplicate:
                    out_action["status"] = "target_canonical_conflict"
                    result["actions"].append(out_action)
                    continue
                con.execute(
                    f"UPDATE entity_registry SET canonical_name=?{updated_expr} "
                    "WHERE id=? AND canonical_name=?",
                    (target, rid, canonical))
                out_action["status"] = "applied"
                result["actions"].append(out_action)
                continue

            if action.get("action") == "register_locked_variant":
                variant = str(action.get("variant") or "").strip()
                variant_key = _name_key(variant)
                sources = _align_json_list(row["source_names"])
                if not any(_phrase_key(source) == _phrase_key(
                        action.get("source_key") or "") for source in sources):
                    out_action["status"] = "stale_source_alias_missing"
                    result["actions"].append(out_action)
                    continue
                variants = _align_json_list(row["variant_names"])
                quarantined = _align_json_list(row["quarantined_names"])
                if any(_name_key(value) == variant_key for value in quarantined):
                    out_action["status"] = "variant_is_quarantined"
                    result["actions"].append(out_action)
                    continue
                if any(_name_key(value) == variant_key for value in variants):
                    out_action["status"] = "already_registered"
                    result["actions"].append(out_action)
                    continue
                variants.append(variant)
                reason_note = (
                    "Round158 authority align: register locked short variant "
                    f"{variant} from decision {','.join(str(x) for x in action.get('decision_ids') or [])}")
                old_reason = str(row["review_reason"] or "")
                review_reason = (old_reason + " | " if old_reason else "") + reason_note
                assignments = ["variant_names=?"]
                params: List[Any] = [_align_json_dump(variants)]
                if "review_reason" in cols:
                    assignments.append("review_reason=?")
                    params.append(review_reason)
                if "updated_at" in cols:
                    assignments.append("updated_at=CURRENT_TIMESTAMP")
                params.append(rid)
                con.execute(
                    "UPDATE entity_registry SET " + ", ".join(assignments)
                    + " WHERE id=?", tuple(params))
                out_action["status"] = "applied"
                result["actions"].append(out_action)
                continue

            aliases = [str(value) for value in action.get("aliases") or []]
            sources = _align_remove_alias(
                _align_json_list(row["source_names"]), aliases)
            variants = _align_remove_alias(
                _align_json_list(row["variant_names"]), aliases)
            quarantined = _align_json_list(row["quarantined_names"])
            qkeys = {_phrase_key(value) for value in quarantined}
            changed = False
            for alias in aliases:
                key = _phrase_key(alias)
                if key and key not in qkeys:
                    quarantined.append(alias)
                    qkeys.add(key)
                    changed = True
            if (sources != _align_json_list(row["source_names"])
                    or variants != _align_json_list(row["variant_names"])):
                changed = True
            if not changed:
                out_action["status"] = "already_quarantined"
                result["actions"].append(out_action)
                continue
            reason_note = (
                "Round158 authority align: quarantine "
                + ",".join(aliases) + "; kinds="
                + ",".join(action.get("kinds") or []))
            old_reason = str(row["review_reason"] or "")
            review_reason = (old_reason + " | " if old_reason else "") + reason_note
            assignments = ["source_names=?", "quarantined_names=?"]
            params: List[Any] = [
                _align_json_dump(sources), _align_json_dump(quarantined)]
            if "variant_names" in cols:
                assignments.append("variant_names=?")
                params.append(_align_json_dump(variants))
            if "review_reason" in cols:
                assignments.append("review_reason=?")
                params.append(review_reason)
            if "updated_at" in cols:
                assignments.append("updated_at=CURRENT_TIMESTAMP")
            params.append(rid)
            con.execute(
                "UPDATE entity_registry SET " + ", ".join(assignments)
                + " WHERE id=?", tuple(params))
            out_action["status"] = "applied"
            result["actions"].append(out_action)
        con.commit()
    except Exception as exc:  # noqa: BLE001 - preserve a truthful failure report
        try:
            con.rollback()
        except sqlite3.Error:
            pass
        for action in result["actions"]:
            if action.get("status") == "applied":
                action["status"] = "rolled_back"
        result["errors"].append(f"apply_failed:{type(exc).__name__}:{exc}")
    finally:
        con.close()

    after = audit_name_authorities(root, decision_db=db_path, output_dir=out_dir)
    result["audit_after"] = {
        "ok": bool(after.get("ok")),
        "status": after.get("status"),
        "blocking_count": int(after.get("blocking_count") or 0),
        "counts": after.get("counts") or {},
        "report": after.get("md_path"),
    }
    applied = sum(1 for action in result["actions"]
                  if action.get("status") == "applied")
    result["applied_count"] = applied
    result["ok"] = (not result["errors"] and bool(after.get("ok"))
                    and applied == len(eligible))
    _align_atomic_json(jpath, result)
    _align_atomic_text(mpath, _align_markdown(result))
    return result


def _align_markdown(result: Dict[str, Any]) -> str:
    lines = ["# 名称权威对齐（Round 158 Phase 1.5a）", ""]
    lines.append(f"- 生成时间: {result.get('generated_at')}")
    lines.append(f"- 模式: **{result.get('mode')}** / ok=`{result.get('ok')}`")
    if result.get("run_id"):
        lines.append(f"- run_id: `{result.get('run_id')}`")
    if result.get("backup_path"):
        lines.append(f"- 备份: `{result.get('backup_path')}`")
    before = result.get("audit_before") or {}
    if before:
        lines.append(f"- 对齐前: blocking={before.get('blocking_count', 0)} "
                     f"status={before.get('status')}")
    after = result.get("audit_after") or {}
    if after:
        lines.append(f"- 对齐后: blocking={after.get('blocking_count', 0)} "
                     f"status={after.get('status')}")
    counts = result.get("counts") or {}
    if counts:
        lines.append("- 计划计数: " + ", ".join(
            f"`{key}`={value}" for key, value in counts.items()))
    actions = result.get("actions") or []
    if actions:
        lines.extend(["", f"## 动作（{len(actions)}）"])
        for action in actions[:80]:
            if action.get("action") == "quarantine_aliases":
                detail = ",".join(action.get("aliases") or [])
            elif action.get("action") == "register_locked_variant":
                detail = (
                    f"{action.get('variant')} ← decision "
                    f"{','.join(str(x) for x in action.get('decision_ids') or [])}")
            elif action.get("action") == "align_decision_forms":
                detail = (
                    f"decision "
                    f"{','.join(str(x) for x in action.get('decision_ids') or [])} "
                    f"→ {action.get('new_translation')}; queued="
                    f"{','.join(action.get('queued_chunks') or []) or '-'}")
            elif action.get("action") == "clear_resolved_pending_review":
                detail = (
                    f"pending_review 1→0; corpus_chunks="
                    f"{action.get('corpus_chunks')}")
            else:
                detail = f"{action.get('canonical_before')} → {action.get('canonical_after')}"
            lines.append(
                f"- id={action.get('registry_id')} `{action.get('action')}` "
                f"{detail} / status={action.get('status', 'planned')}")
    manual = result.get("manual_review") or []
    if manual:
        lines.extend(["", f"## 需人工/LLM 复核（{len(manual)}）"])
        for item in manual[:60]:
            lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
    if result.get("errors"):
        lines.extend(["", "## 错误"])
        lines.extend(f"- `{error}`" for error in result["errors"])
    lines.extend([
        "",
        "本工具不改正文；仅 corpus-consensus 动作会同步 Decision DB 与派生 registry。",
        "任何 ambiguous / multi-owner 项均保持原样。",
        "",
    ])
    return "\n".join(lines)
