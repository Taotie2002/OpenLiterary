"""Round 158: controlled repair of Decision targets contradicted by registry.

The Decision DB is normally the hard terminology authority.  Legacy rows can
still contain a translation that was later retained only as a *variant* of a
different active entity and an unlocked decision may continue enforcing that
wrong form.  The narrow example in this project is ``Leto -> 莱托`` while the
active registry canonical is ``雷托·厄崔迪公爵``; 92 completed chunks already
use ``雷托`` and five contain ``莱托``.

This module does not rewrite prose.  It only plans/applies a deterministic
authority correction when all book-wide evidence agrees:

* the decision is unlocked;
* exactly one active high-confidence registry row owns the wrong translation
  as a Chinese variant while the source key is quarantined on that row;
* the canonical's first CJK token is a one-character correction of the wrong
  target and is not owned by another active entity;
* every chunk containing the English source key ends in either the proposed
  form or the wrong form, with no third spelling;
* at least one COMPLETED final needs repair and every such repair is exactly
  the registered wrong target to the proposed target.

The apply phase updates the Decision row, quarantines the wrong registry
variants, and queues only the COMPLETED chunks that contain the wrong target.
Those chunks are then handled by ``completed-name-repair`` with a real critic
re-audit before any final bytes change.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.decision_engine import canonical_tokens
from src.locked_target_authority_audit import (
    audit_locked_target_authorities,
)
from src.pending_backtrack_reconcile import (
    _name_compatible,
    _name_key,
    _source_phrase_spans,
    _source_tokens,
    _target_core,
)


AUTHORITY_ALIGN_SCHEMA_VERSION = 1
_ARTICLE_ALIAS_RE = re.compile(
    r"^(?:the|a|an)\s+(?P<rest>\S.*)$", re.IGNORECASE)


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _json_list(raw: Any) -> List[str]:
    if isinstance(raw, list):
        values = raw
    else:
        try:
            values = json.loads(str(raw or "[]"))
        except (TypeError, json.JSONDecodeError):
            return []
    return [str(value).strip() for value in values
            if isinstance(value, str) and str(value).strip()]


def _json_dump(values: List[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return bool(con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())


def _columns(con: sqlite3.Connection, table: str) -> set:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def _update_row(
    con: sqlite3.Connection,
    table: str,
    row_id: int,
    values: Dict[str, Any],
) -> None:
    columns = _columns(con, table)
    assignments: List[str] = []
    params: List[Any] = []
    for column, value in values.items():
        if column in columns:
            assignments.append(column + "=?")
            params.append(value)
    if "updated_at" in columns:
        assignments.append("updated_at=CURRENT_TIMESTAMP")
    if not assignments:
        return
    con.execute(
        "UPDATE " + table + " SET " + ", ".join(assignments) + " WHERE id=?",
        tuple(params + [row_id]))


def _connect_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _cjk_only(value: str) -> str:
    text = re.sub(r"（[^）]*）|\([^)]*\)", "", str(value or "")).strip()
    return text if text and re.fullmatch(r"[\u4e00-\u9fff]+", text) else ""


def _one_char_cjk_drift(left: str, right: str) -> bool:
    a, b = _cjk_only(left), _cjk_only(right)
    return bool(
        a and b and len(a) == len(b) and a != b
        and sum(x != y for x, y in zip(a, b)) == 1
    )


def _source_fragment_spans(text: str, source_key: str) -> List[Tuple[int, int]]:
    tokens = _source_tokens(source_key)
    if len(tokens) != 1 or len(tokens[0]) < 4:
        return []
    token = tokens[0]
    fragment = token[1:]
    pattern = re.compile(
        r"(?:(?:\.\s*){2,}|…\s*)" + re.escape(fragment)
        + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return [match.span() for match in pattern.finditer(str(text or ""))]


def _load_registry(con: sqlite3.Connection) -> Tuple[List[dict], str]:
    if not _table_exists(con, "entity_registry"):
        return [], "entity_registry_missing"
    cols = _columns(con, "entity_registry")
    required = {"id", "canonical_name", "source_names", "variant_names",
                "quarantined_names", "confidence", "pending_review"}
    missing = sorted(required - cols)
    if missing:
        return [], "entity_registry_schema_incompatible:" + ",".join(missing)
    rows: List[dict] = []
    for row in con.execute(
            "SELECT id, canonical_name, source_names, variant_names, "
            "quarantined_names, confidence, pending_review "
            "FROM entity_registry ORDER BY id"):
        item = dict(row)
        for field in ("source_names", "variant_names", "quarantined_names"):
            item[field] = _json_list(item.get(field))
        rows.append(item)
    return rows, ""


def _load_decisions(con: sqlite3.Connection) -> Tuple[List[dict], str]:
    if not _table_exists(con, "decision_db"):
        return [], "decision_db_missing"
    cols = _columns(con, "decision_db")
    required = {"id", "source_key", "translation", "locked"}
    missing = sorted(required - cols)
    if missing:
        return [], "decision_db_schema_incompatible:" + ",".join(missing)
    reason_expr = "reason" if "reason" in cols else "NULL AS reason"
    rows = [dict(row) for row in con.execute(
        "SELECT id, source_key, translation, locked, "
        f"{reason_expr} FROM decision_db ORDER BY id")]
    return rows, ""


def _load_tasks(path: Path) -> Tuple[List[dict], str]:
    if not Path(path).exists():
        return [], "workflow_db_missing"
    con = _connect_ro(path)
    try:
        if not _table_exists(con, "chunk_tasks"):
            return [], "workflow_chunk_tasks_missing"
        cols = _columns(con, "chunk_tasks")
        required = {"chunk_id", "chapter_id", "text_content", "state"}
        missing = sorted(required - cols)
        if missing:
            return [], "workflow_schema_incompatible:" + ",".join(missing)
        return [dict(row) for row in con.execute(
            "SELECT chunk_id, chapter_id, text_content, state "
            "FROM chunk_tasks ORDER BY chapter_id, chunk_id")], ""
    finally:
        con.close()


def _load_pending(con: sqlite3.Connection) -> Tuple[Dict[str, dict], str]:
    if not _table_exists(con, "pending_backtrack"):
        return {}, "pending_backtrack_missing"
    cols = _columns(con, "pending_backtrack")
    required = {"chunk_id", "decision_id", "source_key"}
    missing = sorted(required - cols)
    if missing:
        return {}, "pending_backtrack_schema_incompatible:" + ",".join(missing)
    return {str(row["chunk_id"]): dict(row) for row in con.execute(
        "SELECT chunk_id, decision_id, source_key FROM pending_backtrack")}, ""


def _load_impacts(con: sqlite3.Connection) -> Tuple[Dict[int, List[str]], str]:
    if not _table_exists(con, "decision_impact"):
        return {}, "decision_impact_missing"
    cols = _columns(con, "decision_impact")
    required = {"decision_id", "chunk_id"}
    missing = sorted(required - cols)
    if missing:
        return {}, "decision_impact_schema_incompatible:" + ",".join(missing)
    impacts: Dict[int, List[str]] = {}
    for row in con.execute(
            "SELECT decision_id, chunk_id FROM decision_impact "
            "ORDER BY decision_id, chunk_id"):
        impacts.setdefault(int(row[0]), []).append(str(row[1]))
    return impacts, ""


def _active_form_keys(rows: List[dict], *, exclude_id: int) -> set:
    keys = set()
    for row in rows:
        if int(row["id"]) == int(exclude_id):
            continue
        if str(row.get("confidence") or "") != "high":
            continue
        if int(row.get("pending_review") or 0):
            continue
        canonical = str(row.get("canonical_name") or "")
        key = _name_key(canonical)
        if key:
            keys.add(key)
        for variant in row.get("variant_names") or []:
            key = _name_key(variant)
            if key:
                keys.add(key)
    return keys


def _candidate_repair_text(
    root: Path,
    task: dict,
    source_key: str,
    old_target: str,
    new_target: str,
) -> Dict[str, Any]:
    cid = str(task["chunk_id"])
    chapter = str(task["chapter_id"])
    state = str(task["state"])
    final_path = root / "output" / chapter / f"{cid}_final.json"
    row: Dict[str, Any] = {
        "chunk_id": cid,
        "chapter": chapter,
        "state": state,
        "final_path": str(final_path),
        "wrong_occurrences": 0,
        "correct_occurrences": 0,
        "repair_required": False,
        "status": "ok",
        "reason": "",
    }
    if state != "COMPLETED":
        row.update(status="rerun", reason="nonterminal_will_use_updated_decision")
        return row
    final = _read_json(final_path)
    text = str((final or {}).get("text") or "")
    if not final or not text:
        row.update(status="blocked", reason="completed_final_missing")
        return row
    row["wrong_occurrences"] = text.count(old_target)
    row["correct_occurrences"] = text.count(new_target)
    if not row["wrong_occurrences"] and not row["correct_occurrences"]:
        row.update(status="blocked", reason="completed_final_uses_third_form")
        return row
    if row["wrong_occurrences"]:
        source_spans = _source_phrase_spans(str(task.get("text_content") or ""),
                                            source_key)
        fragment_spans = _source_fragment_spans(
            str(task.get("text_content") or ""), source_key)
        row["source_occurrences"] = len(source_spans)
        row["source_fragment_occurrences"] = len(fragment_spans)
        if row["wrong_occurrences"] > len(source_spans) + len(fragment_spans):
            row.update(status="blocked", reason="wrong_form_count_exceeds_source")
            return row
        row["repair_required"] = True
    return row


def _plan_decision(
    root: Path,
    decision: dict,
    registry: List[dict],
    tasks: List[dict],
    pending: Dict[str, dict],
) -> Optional[Dict[str, Any]]:
    source_key = str(decision.get("source_key") or "").strip()
    old_target = str(decision.get("translation") or "").strip()
    if not source_key or not _cjk_only(old_target):
        return None
    if _source_tokens(source_key) and len(_source_tokens(source_key)) != 1:
        return None
    if int(decision.get("locked") or 0):
        return None
    old_key = _name_key(old_target)
    owners = [
        row for row in registry
        if str(row.get("confidence") or "") == "high"
        and int(row.get("pending_review") or 0) == 0
        and any(_name_key(value).startswith(old_key)
                for value in row.get("variant_names") or [])
        and any(_name_key(value) == _name_key(source_key)
                for value in row.get("quarantined_names") or [])
    ]
    if len(owners) != 1:
        return None
    owner = owners[0]
    canonical = str(owner.get("canonical_name") or "")
    tokens = [token for token in canonical_tokens(canonical) if _cjk_only(token)]
    if not tokens:
        return None
    new_target = tokens[0]
    if not _one_char_cjk_drift(old_target, new_target):
        return None
    if not _name_compatible(new_target, canonical):
        return None
    if _name_compatible(old_target, canonical):
        return None
    protected = _active_form_keys(registry, exclude_id=int(owner["id"]))
    if _name_key(new_target) in protected or _name_key(old_target) in protected:
        return None
    variants_to_quarantine = [
        value for value in owner.get("variant_names") or []
        if _name_key(value).startswith(old_key)
        and _name_key(value) != _name_key(new_target)
    ]
    if old_target not in variants_to_quarantine:
        variants_to_quarantine.append(old_target)
    if not variants_to_quarantine:
        return None

    active_tasks = [
        task for task in tasks
        if _source_phrase_spans(str(task.get("text_content") or ""), source_key)
        or _source_fragment_spans(str(task.get("text_content") or ""), source_key)
    ]
    if not active_tasks:
        return None
    evidence_rows = [
        _candidate_repair_text(root, task, source_key, old_target, new_target)
        for task in active_tasks
    ]
    blocked = [row for row in evidence_rows if row["status"] == "blocked"]
    repairs = [row for row in evidence_rows if row["repair_required"]]
    correct = [row for row in evidence_rows
               if row["status"] == "ok" and not row["repair_required"]]
    rerun = [row for row in evidence_rows if row["status"] == "rerun"]
    if blocked:
        return {
            "status": "blocked",
            "reason": blocked[0]["reason"],
            "decision_id": int(decision["id"]),
            "source_key": source_key,
            "old_translation": old_target,
            "new_translation": new_target,
            "registry_id": int(owner["id"]),
            "registry_canonical": canonical,
            "blocked_chunks": blocked,
            "repair_chunks": repairs,
            "correct_chunks": correct,
            "rerun_chunks": rerun,
            "variants_to_quarantine": variants_to_quarantine,
        }
    if len(correct) < 2 or not repairs:
        return None
    pending_conflicts = []
    for row in repairs:
        existing = pending.get(row["chunk_id"])
        if existing and (
                int(existing.get("decision_id") or 0) != int(decision["id"])
                or str(existing.get("source_key") or "") != source_key):
            pending_conflicts.append(row["chunk_id"])
    if pending_conflicts:
        return {
            "status": "blocked",
            "reason": "repair_chunk_has_other_pending_decision",
            "decision_id": int(decision["id"]),
            "source_key": source_key,
            "old_translation": old_target,
            "new_translation": new_target,
            "registry_id": int(owner["id"]),
            "registry_canonical": canonical,
            "pending_conflicts": pending_conflicts,
            "repair_chunks": repairs,
            "correct_chunks": correct,
            "rerun_chunks": rerun,
            "variants_to_quarantine": variants_to_quarantine,
        }
    return {
        "status": "ready",
        "reason": "book_wide_form_consensus",
        "decision_id": int(decision["id"]),
        "source_key": source_key,
        "old_translation": old_target,
        "new_translation": new_target,
        "registry_id": int(owner["id"]),
        "registry_canonical": canonical,
        "correct_chunks": correct,
        "repair_chunks": repairs,
        "rerun_chunks": rerun,
        "variants_to_quarantine": variants_to_quarantine,
        "evidence": {
            "active_source_chunks": len(active_tasks),
            "correct_completed": len(correct),
            "repair_completed": len(repairs),
            "rerun_pending": len(rerun),
            "wrong_occurrences": sum(int(row["wrong_occurrences"]) for row in repairs),
            "correct_occurrences": sum(int(row["correct_occurrences"]) for row in correct),
        },
    }


def audit_decision_authorities(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    db_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "decision_authority_align.json"
    mpath = Path(md_out) if md_out else out_dir / "decision_authority_align.md"
    result: Dict[str, Any] = {
        "schema_version": AUTHORITY_ALIGN_SCHEMA_VERSION,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "decision_db": str(db_path),
        "workflow_db": str(wf_path),
        "candidates": [],
        "blocked_candidates": [],
        "errors": [],
        "counts": {},
        "ok": False,
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if not db_path.exists():
        result["errors"].append("decision_db_missing")
        _write_authority_reports(result, jpath, mpath)
        return result
    con = _connect_ro(db_path)
    try:
        registry, registry_error = _load_registry(con)
        decisions, decision_error = _load_decisions(con)
        pending, pending_error = _load_pending(con)
    finally:
        con.close()
    tasks, task_error = _load_tasks(wf_path)
    for error in (registry_error, decision_error, pending_error, task_error):
        if error:
            result["errors"].append(error)
    if result["errors"]:
        _write_authority_reports(result, jpath, mpath)
        return result
    for decision in decisions:
        candidate = _plan_decision(
            root, decision, registry, tasks, pending)
        if candidate is None:
            continue
        if candidate["status"] == "ready":
            result["candidates"].append(candidate)
        else:
            result["blocked_candidates"].append(candidate)
    result["counts"] = {
        "ready": len(result["candidates"]),
        "blocked": len(result["blocked_candidates"]),
        "repair_chunks": sum(len(c.get("repair_chunks") or [])
                             for c in result["candidates"]),
        "variant_quarantines": sum(len(c.get("variants_to_quarantine") or [])
                                   for c in result["candidates"]),
    }
    result["ok"] = not result["errors"] and not result["blocked_candidates"]
    _write_authority_reports(result, jpath, mpath)
    return result


def _write_authority_reports(result: Dict[str, Any], jpath: Path,
                             mpath: Path) -> None:
    _atomic_json(Path(jpath), result)
    lines = ["# Decision 权威对齐计划", "",
             f"- 生成时间: {result.get('generated_at')}",
             f"- 模式: **{result.get('mode')}**",
             f"- ok: `{result.get('ok')}`",
             f"- counts: `{result.get('counts') or {}}`", ""]
    for candidate in result.get("candidates") or []:
        lines.extend([
            f"## Decision #{candidate.get('decision_id')} "
            f"{candidate.get('source_key')}",
            f"- 旧目标: `{candidate.get('old_translation')}` → "
            f"`{candidate.get('new_translation')}`",
            f"- Registry #{candidate.get('registry_id')} canonical: "
            f"`{candidate.get('registry_canonical')}`",
            f"- 已完成证据: `{len(candidate.get('correct_chunks') or [])}` 正确，"
            f"`{len(candidate.get('repair_chunks') or [])}` 待修复，"
            f"`{len(candidate.get('rerun_chunks') or [])}` 在途重跑",
            "- 待隔离 variants: " + ", ".join(
                f"`{value}`" for value in candidate.get("variants_to_quarantine") or []),
            "- 待修复 chunks: " + ", ".join(
                f"`{row['chunk_id']}`" for row in candidate.get("repair_chunks") or []),
            "",
        ])
    for candidate in result.get("blocked_candidates") or []:
        lines.extend([
            f"## BLOCKED Decision #{candidate.get('decision_id')} "
            f"{candidate.get('source_key')}",
            f"- 原因: `{candidate.get('reason')}`",
            "",
        ])
    if result.get("errors"):
        lines.extend(["## 错误", ""] + [
            f"- `{error}`" for error in result["errors"]])
    _atomic_text(Path(mpath), "\n".join(lines).rstrip() + "\n")


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(source))
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def apply_decision_authority_alignment(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    apply: bool = False,
    run_id: Optional[str] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    db_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "decision_authority_align_apply.json"
    mpath = Path(md_out) if md_out else out_dir / "decision_authority_align_apply.md"
    plan = audit_decision_authorities(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir)
    result: Dict[str, Any] = {
        "schema_version": AUTHORITY_ALIGN_SCHEMA_VERSION,
        "mode": "apply" if apply else "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "run_id": run_id or _now_iso().replace(":", ""),
        "plan_counts": plan.get("counts") or {},
        "planned": plan.get("candidates") or [],
        "blocked_candidates": plan.get("blocked_candidates") or [],
        "actions": [],
        "errors": list(plan.get("errors") or []),
        "backup_path": "",
        "json_path": str(jpath),
        "md_path": str(mpath),
        "ok": False,
    }
    if not apply:
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result
    if not plan.get("candidates"):
        result["ok"] = not plan.get("blocked_candidates") and not result["errors"]
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result
    if result["errors"] or plan.get("blocked_candidates"):
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result
    backup_path = (out_dir / "decision_authority_align" /
                   str(result["run_id"]) / "decision_db.sqlite.bak")
    try:
        _sqlite_backup(db_path, backup_path)
    except Exception as exc:
        result["errors"].append("backup_failed:%s:%s" % (type(exc).__name__, exc))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result
    result["backup_path"] = str(backup_path)
    result["write_attempted"] = True
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        for candidate in plan["candidates"]:
            did = int(candidate["decision_id"])
            rid = int(candidate["registry_id"])
            decision = con.execute(
                "SELECT source_key, translation, locked, reason "
                "FROM decision_db WHERE id=?", (did,)).fetchone()
            registry = con.execute(
                "SELECT canonical_name, variant_names, quarantined_names, "
                "confidence, pending_review FROM entity_registry WHERE id=?",
                (rid,)).fetchone()
            if not decision or not registry:
                raise RuntimeError("authority_row_missing:%s" % did)
            if int(decision["locked"] or 0):
                raise RuntimeError("decision_locked:%s" % did)
            if str(decision["source_key"]) != candidate["source_key"]:
                raise RuntimeError("decision_source_changed:%s" % did)
            if str(decision["translation"]) != candidate["old_translation"]:
                raise RuntimeError("decision_translation_changed:%s" % did)
            if str(registry["canonical_name"]) != candidate["registry_canonical"]:
                raise RuntimeError("registry_canonical_changed:%s" % rid)
            if str(registry["confidence"]) != "high" or int(registry["pending_review"] or 0):
                raise RuntimeError("registry_owner_not_active:%s" % rid)
            variants = _json_list(registry["variant_names"])
            quarantined = _json_list(registry["quarantined_names"])
            to_quarantine = set(candidate.get("variants_to_quarantine") or [])
            new_variants = [value for value in variants
                            if value not in to_quarantine]
            quarantined = list(dict.fromkeys(quarantined + sorted(to_quarantine)))
            reason = str(decision["reason"] or "")
            marker = "Round 158 decision authority align: %s -> %s" % (
                candidate["old_translation"], candidate["new_translation"])
            if marker not in reason:
                reason = (reason.rstrip() + " | " + marker).strip(" |")
            _update_row(
                con, "decision_db", did,
                {"translation": candidate["new_translation"], "reason": reason})
            _update_row(
                con, "entity_registry", rid,
                {"variant_names": _json_dump(new_variants),
                 "quarantined_names": _json_dump(quarantined)})
            for row in candidate.get("repair_chunks") or []:
                cid = str(row["chunk_id"])
                existing = con.execute(
                    "SELECT decision_id, source_key FROM pending_backtrack "
                    "WHERE chunk_id=?", (cid,)).fetchone()
                if existing and (
                        int(existing["decision_id"] or 0) != did
                        or str(existing["source_key"]) != candidate["source_key"]):
                    raise RuntimeError("pending_conflict:%s" % cid)
                con.execute(
                    "INSERT INTO pending_backtrack (chunk_id, decision_id, "
                    "source_key, queue_rounds, created_at, updated_at) "
                    "VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET "
                    "decision_id=excluded.decision_id, "
                    "source_key=excluded.source_key, "
                    "queue_rounds=COALESCE(pending_backtrack.queue_rounds,0)+1, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (cid, did, candidate["source_key"]))
            result["actions"].append({
                "decision_id": did,
                "source_key": candidate["source_key"],
                "old_translation": candidate["old_translation"],
                "new_translation": candidate["new_translation"],
                "registry_id": rid,
                "variants_quarantined": sorted(to_quarantine),
                "queued_chunks": [row["chunk_id"]
                                  for row in candidate.get("repair_chunks") or []],
                "status": "applied",
            })
        con.commit()
    except Exception as exc:
        try:
            con.rollback()
        except sqlite3.Error:
            pass
        result["errors"].append("%s:%s" % (type(exc).__name__, exc))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result
    finally:
        con.close()
    result["ok"] = not result["errors"]
    _atomic_json(jpath, result)
    _atomic_text(mpath, _apply_markdown(result))
    return result


def _apply_markdown(result: Dict[str, Any]) -> str:
    lines = ["# Decision 权威对齐（受控写入）", "",
             f"- 生成时间: {result.get('generated_at')}",
             f"- 模式: **{result.get('mode')}**",
             f"- ok: `{result.get('ok')}`；写入尝试: `{result.get('write_attempted')}`",
             f"- 备份: `{result.get('backup_path') or '(无)'}`", ""]
    for action in result.get("actions") or []:
        lines.extend([
            f"## Decision #{action.get('decision_id')} {action.get('source_key')}",
            f"- `{action.get('old_translation')}` → `{action.get('new_translation')}`",
            f"- Registry #{action.get('registry_id')}",
            "- 隔离 variants: " + ", ".join(
                f"`{value}`" for value in action.get("variants_quarantined") or []),
            "- 排队待重审: " + ", ".join(
                f"`{value}`" for value in action.get("queued_chunks") or []),
            "",
        ])
    if result.get("errors"):
        lines.extend(["## 错误", ""] + [
            f"- `{error}`" for error in result["errors"]])
    return "\n".join(lines).rstrip() + "\n"


def _final_text_for_task(root: Path, task: Dict[str, Any]) -> str:
    chunk_id = str(task.get("chunk_id") or "")
    chapter = str(task.get("chapter_id") or chunk_id[:4])
    payload = _read_json(root / "output" / chapter / f"{chunk_id}_final.json")
    return str((payload or {}).get("text") or "")


def _active_registry_form_sources(rows: List[dict]) -> Dict[str, set]:
    owners: Dict[str, set] = {}
    for row in rows:
        if str(row.get("confidence") or "") != "high":
            continue
        if int(row.get("pending_review") or 0):
            continue
        sources = {
            str(value).strip() for value in row.get("source_names") or []
            if str(value).strip()
        }
        forms = [str(row.get("canonical_name") or "")]
        forms.extend(str(value) for value in row.get("variant_names") or [])
        for form in forms:
            key = _name_key(form)
            if key:
                owners.setdefault(key, set()).update(sources)
    return owners


def _article_alias_remainder(source_key: str) -> str:
    match = _ARTICLE_ALIAS_RE.match(str(source_key or "").strip())
    return match.group("rest") if match else ""


def audit_article_shadow_decisions(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
    write_reports: bool = True,
) -> Dict[str, Any]:
    """Plan deletion of pending article-led shadow rows.

    The match is deliberately narrow and case-sensitive after the article:
    ``the Lady Jessica`` can defer to ``Lady Jessica``, while ``the one`` does
    not defer to ``One``.  A row is eligible only when it is unlocked, already
    has pending work, and every source-positive final agrees with the existing
    exact-key authority.
    """
    root = Path(project_root)
    db_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "decision_article_alias_repair.json"
    mpath = Path(md_out) if md_out else out_dir / "decision_article_alias_repair.md"
    result: Dict[str, Any] = {
        "schema_version": AUTHORITY_ALIGN_SCHEMA_VERSION,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "decision_db": str(db_path),
        "workflow_db": str(wf_path),
        "candidates": [],
        "blocked_candidates": [],
        "errors": [],
        "counts": {},
        "ok": False,
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if not db_path.exists():
        result["errors"].append("decision_db_missing")
        if write_reports:
            _atomic_json(jpath, result)
            _atomic_text(mpath, _article_alias_markdown(result))
        return result
    con = _connect_ro(db_path)
    try:
        decisions, decision_error = _load_decisions(con)
        pending, pending_error = _load_pending(con)
        impacts, impact_error = _load_impacts(con)
    finally:
        con.close()
    tasks, task_error = _load_tasks(wf_path)
    for error in (decision_error, pending_error, impact_error, task_error):
        if error:
            result["errors"].append(error)
    if result["errors"]:
        if write_reports:
            _atomic_json(jpath, result)
            _atomic_text(mpath, _article_alias_markdown(result))
        return result

    by_source = {str(row.get("source_key") or ""): row for row in decisions}
    tasks_by_id = {str(row.get("chunk_id") or ""): row for row in tasks}
    for alias in decisions:
        source_key = str(alias.get("source_key") or "").strip()
        remainder = _article_alias_remainder(source_key)
        canonical = by_source.get(remainder)
        if not canonical or int(canonical["id"]) == int(alias["id"]):
            continue
        if int(alias.get("locked") or 0):
            continue
        alias_target = str(alias.get("translation") or "").strip()
        canonical_target = str(canonical.get("translation") or "").strip()
        if not alias_target or not canonical_target:
            continue
        alias_pending = [
            row for row in pending.values()
            if int(row.get("decision_id") or 0) == int(alias["id"])
        ]
        if not alias_pending:
            continue

        blocked_reason = ""
        wrong_pending_sources = sorted({
            str(row.get("source_key") or "") for row in alias_pending
            if str(row.get("source_key") or "") != source_key
        })
        if wrong_pending_sources:
            blocked_reason = "pending_source_key_mismatch:" + ",".join(wrong_pending_sources)

        source_tasks = []
        for task in tasks:
            source_text = str(task.get("text_content") or "")
            if (_source_phrase_spans(source_text, source_key)
                    or _source_fragment_spans(source_text, source_key)):
                source_tasks.append(task)
        completed = []
        inert = []
        blocked_chunks = set()
        if not source_tasks and not blocked_reason:
            blocked_reason = "no_source_positive_tasks"
        for task in source_tasks:
            state = str(task.get("state") or "")
            chunk_id = str(task.get("chunk_id") or "")
            if state == "PENDING":
                inert.append(chunk_id)
                continue
            if state != "COMPLETED":
                if not blocked_reason:
                    blocked_reason = (
                        "nonterminal_source_task:%s:%s" % (chunk_id, state))
                continue
            text = _final_text_for_task(root, task)
            if not text:
                blocked_chunks.add(chunk_id)
                if not blocked_reason:
                    blocked_reason = "completed_final_missing:%s" % chunk_id
                continue
            if alias_target in text:
                blocked_chunks.add(chunk_id)
                if not blocked_reason:
                    blocked_reason = "alias_target_still_present:%s" % chunk_id
                continue
            if canonical_target not in text:
                blocked_chunks.add(chunk_id)
                if not blocked_reason:
                    blocked_reason = "canonical_target_absent:%s" % chunk_id
                continue
            completed.append(chunk_id)
        if not completed and not blocked_reason:
            blocked_reason = "no_completed_authority_evidence"

        if blocked_reason:
            result["blocked_candidates"].append({
                "decision_id": int(alias["id"]),
                "source_key": source_key,
                "canonical_decision_id": int(canonical["id"]),
                "canonical_source_key": remainder,
                "alias_translation": alias_target,
                "canonical_translation": canonical_target,
                "reason": blocked_reason,
                "blocked_chunks": sorted(blocked_chunks),
                "pending_chunks": sorted(row["chunk_id"] for row in alias_pending),
            })
            continue
        result["candidates"].append({
            "decision_id": int(alias["id"]),
            "source_key": source_key,
            "canonical_decision_id": int(canonical["id"]),
            "canonical_source_key": remainder,
            "alias_translation": alias_target,
            "canonical_translation": canonical_target,
            "completed_chunks": sorted(completed),
            "inert_pending_chunks": sorted(inert),
            "pending_chunks": sorted(row["chunk_id"] for row in alias_pending),
            "impact_chunks": sorted(impacts.get(int(alias["id"]), [])),
        })
    result["counts"] = {
        "candidates": len(result["candidates"]),
        "blocked": len(result["blocked_candidates"]),
        "pending_rows": sum(len(row.get("pending_chunks") or [])
                            for row in result["candidates"]),
        "completed_evidence": sum(len(row.get("completed_chunks") or [])
                                  for row in result["candidates"]),
    }
    result["ok"] = not result["errors"] and not result["blocked_candidates"]
    if write_reports:
        _atomic_json(jpath, result)
        _atomic_text(mpath, _article_alias_markdown(result))
    return result


def _article_alias_markdown(result: Dict[str, Any]) -> str:
    lines = ["# Decision 冠词影子归并（只读/受控写入）", "",
             f"- 生成时间: {result.get('generated_at')}",
             f"- 模式: **{result.get('mode')}**",
             f"- ok: `{result.get('ok')}`；写入尝试: `{result.get('write_attempted')}`",
             f"- 备份: `{result.get('backup_path') or '(无)'}`",
             f"- counts: `{result.get('counts') or {}}`", ""]
    for candidate in result.get("candidates") or []:
        lines.extend([
            f"## Decision #{candidate.get('decision_id')} "
            f"{candidate.get('source_key')}",
            f"- 归并到 #{candidate.get('canonical_decision_id')} "
            f"`{candidate.get('canonical_source_key')}`",
            f"- 影子译文 `{candidate.get('alias_translation')}`；"
            f"既有权威 `{candidate.get('canonical_translation')}`",
            f"- COMPLETED 证据: `{len(candidate.get('completed_chunks') or [])}`；"
            f"PENDING: `{len(candidate.get('inert_pending_chunks') or [])}`；"
            f"待清 pending: `{len(candidate.get('pending_chunks') or [])}`",
            "",
        ])
    for blocked in result.get("blocked_candidates") or []:
        lines.extend([
            f"## BLOCKED Decision #{blocked.get('decision_id')} "
            f"{blocked.get('source_key')}",
            f"- 原因: `{blocked.get('reason')}`",
            "",
        ])
    if result.get("errors"):
        lines.extend(["## 错误", ""] + [
            f"- `{error}`" for error in result["errors"]])
    lines.extend([
        "", "归并只删除已证实无原文正文依赖的影子 Decision 及其 pending 行；"
        "不修改 canonical Decision、正文、registry 或 workflow 状态。", ""])
    return "\n".join(lines).rstrip() + "\n"


def apply_article_shadow_alias_repairs(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
    apply: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    db_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "decision_article_alias_repair_apply.json"
    mpath = Path(md_out) if md_out else out_dir / "decision_article_alias_repair_apply.md"
    plan = audit_article_shadow_decisions(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir,
        write_reports=False)
    result: Dict[str, Any] = {
        "schema_version": AUTHORITY_ALIGN_SCHEMA_VERSION,
        "mode": "apply" if apply else "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "run_id": run_id or _now_iso().replace(":", ""),
        "decision_db": str(db_path),
        "workflow_db": str(wf_path),
        "candidates": plan.get("candidates") or [],
        "blocked_candidates": plan.get("blocked_candidates") or [],
        "errors": list(plan.get("errors") or []),
        "counts": dict(plan.get("counts") or {}),
        "actions": [],
        "backup_path": "",
        "ok": False,
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if not apply:
        result["ok"] = bool(plan.get("ok"))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _article_alias_markdown(result))
        return result
    if not result["candidates"]:
        result["ok"] = not result["errors"] and not result["blocked_candidates"]
        _atomic_json(jpath, result)
        _atomic_text(mpath, _article_alias_markdown(result))
        return result
    if result["errors"] or result["blocked_candidates"]:
        _atomic_json(jpath, result)
        _atomic_text(mpath, _article_alias_markdown(result))
        return result

    backup_path = (out_dir / "decision_article_alias_repair" /
                   str(result["run_id"]) / "decision_db.sqlite.bak")
    try:
        _sqlite_backup(db_path, backup_path)
    except Exception as exc:
        result["errors"].append("backup_failed:%s:%s" % (type(exc).__name__, exc))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _article_alias_markdown(result))
        return result
    result["backup_path"] = str(backup_path)
    result["write_attempted"] = True
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        for candidate in result["candidates"]:
            alias_id = int(candidate["decision_id"])
            canonical_id = int(candidate["canonical_decision_id"])
            source_key = str(candidate["source_key"])
            alias_target = str(candidate["alias_translation"])
            alias_row = con.execute(
                "SELECT source_key, translation, locked FROM decision_db WHERE id=?",
                (alias_id,)).fetchone()
            canonical_row = con.execute(
                "SELECT source_key, translation FROM decision_db WHERE id=?",
                (canonical_id,)).fetchone()
            if not alias_row or not canonical_row:
                raise RuntimeError("article_alias_authority_missing:%s" % alias_id)
            if (str(alias_row["source_key"]) != source_key
                    or str(alias_row["translation"]) != alias_target
                    or int(alias_row["locked"] or 0)):
                raise RuntimeError("article_alias_changed:%s" % alias_id)
            if str(canonical_row["source_key"]) != candidate["canonical_source_key"]:
                raise RuntimeError("article_alias_canonical_changed:%s" % canonical_id)
            con.execute(
                "INSERT OR IGNORE INTO decision_impact (decision_id, chunk_id) "
                "SELECT ?, chunk_id FROM decision_impact WHERE decision_id=?",
                (canonical_id, alias_id))
            con.execute("DELETE FROM decision_impact WHERE decision_id=?", (alias_id,))
            con.execute(
                "DELETE FROM pending_backtrack WHERE decision_id=? AND source_key=?",
                (alias_id, source_key))
            cursor = con.execute(
                "DELETE FROM decision_db WHERE id=? AND source_key=? "
                "AND translation=? AND COALESCE(locked, 0)=0",
                (alias_id, source_key, alias_target))
            if cursor.rowcount != 1:
                raise RuntimeError("article_alias_delete_cas_failed:%s" % alias_id)
            result["actions"].append({
                "decision_id": alias_id,
                "source_key": source_key,
                "canonical_decision_id": canonical_id,
                "status": "merged",
            })
        con.commit()
    except Exception as exc:
        con.rollback()
        result["errors"].append("%s:%s" % (type(exc).__name__, exc))
    finally:
        con.close()

    after = audit_article_shadow_decisions(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir,
        write_reports=False)
    result["after_counts"] = after.get("counts") or {}
    result["after_ok"] = bool(after.get("ok"))
    result["ok"] = (not result["errors"] and bool(after.get("ok"))
                    and not (after.get("blocked_candidates") or []))
    _atomic_json(jpath, result)
    _atomic_text(mpath, _article_alias_markdown(result))
    return result


def _corpus_repair_markdown(result: Dict[str, Any]) -> str:
    lines = ["# 锁目标 corpus 权威修复（受控写入）", "",
             f"- 生成时间: {result.get('generated_at')}",
             f"- 模式: **{result.get('mode')}**",
             f"- ok: `{result.get('ok')}`；写入尝试: `{result.get('write_attempted')}`",
             f"- 备份: `{result.get('backup_path') or '(无)'}`",
             f"- 计划统计: `{result.get('counts') or {}}`", ""]
    for candidate in result.get("candidates") or []:
        evidence = candidate.get("authority") or {}
        lines.extend([
            f"## Decision #{candidate.get('decision_id')} "
            f"{candidate.get('source_key')}",
            f"- `{candidate.get('old_translation')}` → "
            f"`{candidate.get('new_translation')}`",
            f"- 全书证据: `{evidence.get('positive_chunks')}` / "
            f"`{evidence.get('authority_chunks')}` 个 source-positive 块；"
            f"第二候选 `{evidence.get('second_positive_chunks')}` 块",
            "- 已验证 pending finals: " + ", ".join(
                f"`{value}`" for value in candidate.get("pending_chunks") or []),
            "",
        ])
    for blocked in result.get("blocked_candidates") or []:
        lines.extend([
            f"## BLOCKED Decision #{blocked.get('decision_id')} "
            f"{blocked.get('source_key')}",
            f"- 原因: `{blocked.get('reason')}`",
            "",
        ])
    if result.get("errors"):
        lines.extend(["## 错误", ""] + [
            f"- `{error}`" for error in result["errors"]])
    lines.extend([
        "", "本命令只更新 Decision 的 translation/reason/lock 证据；"
        "不修改正文、registry、pending 或 workflow 状态。", ""])
    return "\n".join(lines).rstrip() + "\n"


def audit_locked_target_authority_repairs(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
    majority_ratio: float = 0.70,
) -> Dict[str, Any]:
    """Plan corpus-backed repairs for locked generic Decisions only.

    The auditor already separates generic keys from concepts covered by a more
    specific Decision.  This planner adds the write preconditions: a clear
    candidate majority, every affected pending final already using the
    candidate, no third form, a locked Decision, and CAS-friendly identity.
    """
    root = Path(project_root)
    db_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "locked_target_authority_repair.json"
    mpath = Path(md_out) if md_out else out_dir / "locked_target_authority_repair.md"
    result: Dict[str, Any] = {
        "schema_version": AUTHORITY_ALIGN_SCHEMA_VERSION,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "candidates": [],
        "blocked_candidates": [],
        "residual_groups": [],
        "counts": {},
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
        "ok": False,
    }
    audit = audit_locked_target_authorities(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir)
    result["audit_counts"] = audit.get("counts") or {}
    result["errors"].extend(str(error) for error in audit.get("errors") or [])
    if not audit.get("ok"):
        result["counts"] = {"candidates": 0, "blocked": 0}
        _atomic_json(jpath, result)
        _atomic_text(mpath, _corpus_repair_markdown(result))
        return result

    ro = _connect_ro(db_path)
    try:
        decisions_rows, decision_error = _load_decisions(ro)
        registry_rows, registry_error = _load_registry(ro)
        pending, pending_error = _load_pending(ro)
    finally:
        ro.close()
    if decision_error:
        result["errors"].append(decision_error)
    if registry_error:
        result["errors"].append(registry_error)
    tasks, task_error = _load_tasks(wf_path)
    if task_error:
        result["errors"].append(task_error)
    if pending_error:
        result["errors"].append(pending_error)
    if result["errors"]:
        _atomic_json(jpath, result)
        _atomic_text(mpath, _corpus_repair_markdown(result))
        return result

    decisions = {int(row["id"]): row for row in decisions_rows}
    tasks_by_id = {str(row["chunk_id"]): row for row in tasks}
    specific_targets = {
        _name_key(_target_core(row.get("translation")))
        for row in decisions_rows
        if _name_key(_target_core(row.get("translation")))
    }
    registry_sources = _active_registry_form_sources(registry_rows)
    contradicted_groups = [
        group for group in audit.get("groups") or []
        if group.get("authority_kind") == "generic"
        and group.get("classification") == "contradicted"
    ]
    residual = [
        {key: group.get(key) for key in (
            "decision_id", "source_key", "target", "classification",
            "reason", "authority_chunks", "target_hits")}
        for group in audit.get("groups") or []
        if group.get("classification") != "contradicted"
    ]
    for group in contradicted_groups:
        did = int(group.get("decision_id") or 0)
        source_key = str(group.get("source_key") or "")
        old_target = str(group.get("target") or "")
        decision = decisions.get(did)
        blocked_reason = ""
        if not decision:
            blocked_reason = "decision_missing"
        elif str(decision.get("source_key") or "") != source_key:
            blocked_reason = "decision_source_changed"
        elif str(decision.get("translation") or "") != old_target:
            blocked_reason = "decision_translation_changed"
        elif not int(decision.get("locked") or 0):
            blocked_reason = "decision_not_locked"

        candidates = list(group.get("candidates") or [])
        best = candidates[0] if candidates else None
        new_target = str((best or {}).get("form") or "")
        authority_chunks = int(group.get("authority_chunks") or 0)
        best_hits = int((best or {}).get("positive_chunks") or 0)
        second_hits = int(
            (candidates[1] or {}).get("positive_chunks") or 0
            if len(candidates) > 1 else 0)
        threshold = max(2, int(math.ceil(
            authority_chunks * max(0.5, min(0.9, float(majority_ratio))))))
        if not blocked_reason:
            if not new_target or new_target == old_target:
                blocked_reason = "no_candidate_form"
            elif authority_chunks < 3:
                blocked_reason = "insufficient_authority_chunks"
            elif best_hits < threshold:
                blocked_reason = "candidate_below_majority_threshold"
            elif second_hits and best_hits < second_hits * 2:
                blocked_reason = "second_candidate_too_close"
            elif _name_key(new_target) in specific_targets:
                blocked_reason = "candidate_owned_by_specific_decision"
            else:
                owners = registry_sources.get(_name_key(new_target)) or set()
                if owners and _name_key(source_key) not in {
                        _name_key(value) for value in owners}:
                    blocked_reason = "candidate_owned_by_other_registry_source"

        pending_chunks = [str(value) for value in group.get("pending_chunks") or []]
        verified_chunks: List[str] = []
        if not blocked_reason:
            if not pending_chunks:
                blocked_reason = "no_pending_chunks"
            for chunk_id in pending_chunks:
                row = pending.get(chunk_id)
                task = tasks_by_id.get(chunk_id)
                if not row or int(row.get("decision_id") or 0) != did:
                    blocked_reason = "pending_decision_mismatch:%s" % chunk_id
                    break
                if not task or str(task.get("state") or "") != "COMPLETED":
                    blocked_reason = "pending_chunk_not_completed:%s" % chunk_id
                    break
                text = _final_text_for_task(root, task)
                if not text:
                    blocked_reason = "pending_final_missing:%s" % chunk_id
                    break
                if old_target in text:
                    blocked_reason = "old_target_still_present:%s" % chunk_id
                    break
                if new_target not in text:
                    blocked_reason = "candidate_absent_from_pending_final:%s" % chunk_id
                    break
                verified_chunks.append(chunk_id)

        if blocked_reason:
            result["blocked_candidates"].append({
                "decision_id": did,
                "source_key": source_key,
                "old_translation": old_target,
                "new_translation": new_target,
                "reason": blocked_reason,
            })
            continue
        result["candidates"].append({
            "decision_id": did,
            "source_key": source_key,
            "old_translation": old_target,
            "new_translation": new_target,
            "pending_chunks": verified_chunks,
            "authority": {
                "positive_chunks": best_hits,
                "authority_chunks": authority_chunks,
                "second_positive_chunks": second_hits,
                "threshold": threshold,
                "classification": group.get("classification"),
                "reason": group.get("reason"),
            },
        })
    result["residual_groups"] = residual
    result["counts"] = {
        "groups": len(audit.get("groups") or []),
        "contradicted": len(contradicted_groups),
        "candidates": len(result["candidates"]),
        "blocked": len(result["blocked_candidates"]),
        "residual": len(residual),
    }
    result["ok"] = not result["errors"] and not result["blocked_candidates"]
    _atomic_json(jpath, result)
    _atomic_text(mpath, _corpus_repair_markdown(result))
    return result


def apply_locked_target_authority_repairs(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    apply: bool = False,
    run_id: Optional[str] = None,
    majority_ratio: float = 0.70,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    db_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "locked_target_authority_repair_apply.json"
    mpath = Path(md_out) if md_out else out_dir / "locked_target_authority_repair_apply.md"
    plan = audit_locked_target_authority_repairs(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir,
        majority_ratio=majority_ratio)
    result: Dict[str, Any] = {
        "schema_version": AUTHORITY_ALIGN_SCHEMA_VERSION,
        "mode": "apply" if apply else "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "run_id": run_id or _now_iso().replace(":", ""),
        "plan_counts": plan.get("counts") or {},
        "counts": plan.get("counts") or {},
        "candidates": plan.get("candidates") or [],
        "blocked_candidates": plan.get("blocked_candidates") or [],
        "residual_groups": plan.get("residual_groups") or [],
        "actions": [],
        "errors": list(plan.get("errors") or []),
        "backup_path": "",
        "audit_after_counts": {},
        "json_path": str(jpath),
        "md_path": str(mpath),
        "ok": bool(plan.get("ok")),
    }
    if not apply:
        _atomic_json(jpath, result)
        _atomic_text(mpath, _corpus_repair_markdown(result))
        return result
    if result["errors"] or result["blocked_candidates"]:
        _atomic_json(jpath, result)
        _atomic_text(mpath, _corpus_repair_markdown(result))
        return result
    if not result["candidates"]:
        result["ok"] = True
        _atomic_json(jpath, result)
        _atomic_text(mpath, _corpus_repair_markdown(result))
        return result

    backup_path = (out_dir / "locked_target_authority_repair" /
                   str(result["run_id"]) / "decision_db.sqlite.bak")
    try:
        _sqlite_backup(db_path, backup_path)
    except Exception as exc:
        result["errors"].append(
            "backup_failed:%s:%s" % (type(exc).__name__, exc))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _corpus_repair_markdown(result))
        return result
    result["backup_path"] = str(backup_path)
    result["write_attempted"] = True
    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        for candidate in result["candidates"]:
            did = int(candidate["decision_id"])
            row = con.execute(
                "SELECT source_key, translation, reason, locked "
                "FROM decision_db WHERE id=?", (did,)).fetchone()
            if not row:
                raise RuntimeError("decision_missing:%s" % did)
            if str(row["source_key"]) != candidate["source_key"]:
                raise RuntimeError("decision_source_changed:%s" % did)
            if str(row["translation"]) != candidate["old_translation"]:
                raise RuntimeError("decision_translation_changed:%s" % did)
            if not int(row["locked"] or 0):
                raise RuntimeError("decision_not_locked:%s" % did)
            reason = str(row["reason"] or "")
            marker = "Round 158 corpus lock authority: %s -> %s" % (
                candidate["old_translation"], candidate["new_translation"])
            if marker not in reason:
                reason = (reason.rstrip() + " | " + marker).strip(" |")
            _update_row(
                con, "decision_db", did,
                {"translation": candidate["new_translation"], "reason": reason})
            result["actions"].append({
                "decision_id": did,
                "source_key": candidate["source_key"],
                "old_translation": candidate["old_translation"],
                "new_translation": candidate["new_translation"],
                "pending_chunks_verified": candidate["pending_chunks"],
                "status": "applied",
            })
        con.commit()
    except Exception as exc:
        try:
            con.rollback()
        except sqlite3.Error:
            pass
        result["errors"].append("%s:%s" % (type(exc).__name__, exc))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _corpus_repair_markdown(result))
        return result
    finally:
        con.close()

    audit_after = audit_locked_target_authorities(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir)
    result["audit_after_counts"] = audit_after.get("counts") or {}
    result["ok"] = not result["errors"]
    _atomic_json(jpath, result)
    _atomic_text(mpath, _corpus_repair_markdown(result))
    return result


__all__ = [
    "AUTHORITY_ALIGN_SCHEMA_VERSION",
    "audit_decision_authorities",
    "apply_decision_authority_alignment",
    "audit_locked_target_authority_repairs",
    "apply_locked_target_authority_repairs",
]
