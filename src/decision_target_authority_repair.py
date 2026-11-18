"""Round 158: mechanically evidence-gated repair of a stale Decision target.

A Decision normally owns the hard terminology target.  A legacy row can still
be contradicted by the completed book while all pending finals already agree on
a different surface form.  Rewriting prose in that situation is both expensive
and wrong: the authority row, not the prose, is stale.

This module plans/applies only the narrow case where all of the following hold:

* the Decision target is pure CJK and has no editorial prose/annotation;
* the Decision is unlocked;
* exactly one term-audit group is ``target_missing_contract``;
* all source-positive chunks are COMPLETED;
* every pending final contains one common alternate CJK form;
* that alternate appears in at least two more source-positive chunks than the
  stale target and covers at least 75% of the target/alternate union;
* the alternate is not a substring of the target (or vice versa), is not used
  as another Decision target, and is not a high-frequency unrelated n-gram.

The apply phase changes ``decision_db.translation`` with compare-and-swap and
an audit marker.  It never rewrites a final, changes workflow state, deletes
pending, or locks a Decision.  Completed chunks that still contain only the
stale target are queued to ``pending_backtrack`` for the normal repair path; a
subsequent safe-only reconciliation consumes rows the corrected authority now
satisfies.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pending_backtrack_reconcile import (
    _name_key,
    _source_phrase_spans,
    _target_core,
    body_before_footnotes,
    classify_pending_snapshot,
    load_pending_context,
)
from src.term_form_audit import audit_term_forms


DECISION_TARGET_REPAIR_SCHEMA_VERSION = 1
_PURE_CJK_RE = re.compile(r"^[\u4e00-\u9fff]{2,16}$")


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _run_id() -> str:
    now = _dt.datetime.now().astimezone()
    return now.strftime("%Y%m%dT%H%M%S-%f")


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


def _read_final_text(output_dir: Path, chapter: str, chunk_id: str) -> str:
    payload = _read_json(Path(output_dir) / chapter / f"{chunk_id}_final.json")
    if not payload:
        return ""
    return body_before_footnotes(payload.get("text"))


def _source_positive_ids(tasks: Dict[str, dict], source_key: str) -> set:
    out = set()
    for chunk_id, task in tasks.items():
        if _source_phrase_spans(
                str(task.get("text_content") or ""), source_key):
            out.add(str(chunk_id))
    return out


def _candidate_evidence(
    *,
    source_key: str,
    target: str,
    form: str,
    tasks: Dict[str, dict],
    corpus: Dict[str, str],
    pending_chunk_ids: List[str],
    pending_texts: Dict[str, str],
    other_target_keys: set,
) -> Tuple[Optional[dict], str]:
    if not _PURE_CJK_RE.fullmatch(form):
        return None, "candidate_not_pure_cjk"
    target_key = _name_key(target)
    form_key = _name_key(form)
    if not form_key or form_key == target_key:
        return None, "candidate_equals_target"
    if form in target or target in form:
        return None, "candidate_is_target_fragment"
    if form_key in other_target_keys:
        return None, "candidate_owned_by_other_decision"
    if any(not pending_texts.get(chunk_id)
           or form not in pending_texts[chunk_id]
           for chunk_id in pending_chunk_ids):
        return None, "candidate_missing_from_pending_final"

    positive_ids = _source_positive_ids(tasks, source_key)
    if len(positive_ids) < 3:
        return None, "insufficient_source_positive_chunks"
    if any(str((tasks.get(chunk_id) or {}).get("state") or "") != "COMPLETED"
           for chunk_id in positive_ids):
        return None, "source_positive_chunk_not_completed"

    target_ids = {chunk_id for chunk_id in positive_ids
                  if target in corpus.get(chunk_id, "")}
    form_ids = {chunk_id for chunk_id in positive_ids
                if form in corpus.get(chunk_id, "")}
    if target_ids & form_ids:
        return None, "target_and_candidate_coexist"
    union_ids = target_ids | form_ids
    negative_ids = {
        chunk_id for chunk_id, text in corpus.items()
        if chunk_id not in positive_ids and form in text
    }
    if len(form_ids) < len(target_ids) + 2:
        return None, "candidate_gain_below_two_chunks"
    if not union_ids or len(form_ids) * 4 < len(union_ids) * 3:
        return None, "candidate_below_three_quarter_share"
    if len(negative_ids) > max(2, len(form_ids) // 2):
        return None, "candidate_too_frequent_outside_source_positive"

    return {
        "form": form,
        "source_positive_chunks": len(positive_ids),
        "target_chunks": sorted(target_ids),
        "candidate_chunks": sorted(form_ids),
        "target_hits": len(target_ids),
        "candidate_hits": len(form_ids),
        "stale_chunks": sorted(target_ids),
        "explicit_union_chunks": sorted(union_ids),
        "candidate_share_of_union": round(
            len(form_ids) / max(1, len(union_ids)), 4),
        "negative_chunks": sorted(negative_ids),
    }, ""


def audit_decision_target_repairs(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    term_report_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    db_path = (Path(decision_db) if decision_db
               else root / "db" / "decision_db.sqlite")
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    run_id = _run_id()
    nested = (Path(term_report_dir) if term_report_dir
              else out_dir / "decision_target_authority_repair" / run_id)
    jpath = (Path(json_out) if json_out
             else out_dir / "decision_target_authority_repair.json")
    mpath = (Path(md_out) if md_out
             else out_dir / "decision_target_authority_repair.md")
    result: Dict[str, Any] = {
        "schema_version": DECISION_TARGET_REPAIR_SCHEMA_VERSION,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "run_id": run_id,
        "candidates": [],
        "blocked_groups": [],
        "counts": {},
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
        "ok": False,
    }
    context = load_pending_context(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir)
    if not context.get("available"):
        result["errors"].append(
            str(context.get("reason") or "pending_context_unavailable"))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_markdown(result))
        return result

    term_audit = audit_term_forms(
        root,
        decision_db=db_path,
        workflow_db=wf_path,
        output_dir=out_dir,
        json_out=nested / "term_form_audit.json",
        md_out=nested / "term_form_audit.md",
    )
    result["term_audit_counts"] = term_audit.get("counts") or {}
    result["errors"].extend(str(error) for error in term_audit.get("errors") or [])
    if not term_audit.get("ok"):
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_markdown(result))
        return result

    decisions = context.get("decisions") or {}
    tasks = context.get("tasks") or {}
    pending_by_chunk = {
        str(row.get("chunk_id") or ""): row
        for row in context.get("pending") or []
    }
    corpus = {}
    for chunk_id, task in tasks.items():
        if str(task.get("state") or "") != "COMPLETED":
            continue
        chapter = str(task.get("chapter_id") or str(chunk_id)[:4])
        corpus[str(chunk_id)] = _read_final_text(
            out_dir, chapter, str(chunk_id))

    all_target_keys = {
        _name_key(_target_core(str(row.get("translation") or "")))
        for row in decisions.values()
        if _name_key(_target_core(str(row.get("translation") or "")))
    }
    for group in term_audit.get("groups") or []:
        decision_id = int(group.get("decision_id") or 0)
        source_key = str(group.get("source_key") or "")
        target = str(group.get("target") or "")
        pending_chunks = [str(value) for value in
                          group.get("pending_chunks") or []]
        blocked: Dict[str, Any] = {
            "decision_id": decision_id,
            "source_key": source_key,
            "target": target,
            "pending_chunks": pending_chunks,
            "reason": "",
        }
        decision = decisions.get(decision_id)
        core = _target_core(target)
        if str(group.get("classification") or "") != "target_missing_contract":
            blocked["reason"] = "classification:" + str(
                group.get("classification") or "")
        elif not decision:
            blocked["reason"] = "decision_missing"
        elif str(decision.get("source_key") or "") != source_key:
            blocked["reason"] = "decision_source_changed"
        elif str(decision.get("translation") or "") != target:
            blocked["reason"] = "decision_translation_changed"
        elif int(decision.get("locked") or 0):
            blocked["reason"] = "decision_locked"
        elif core.strip() != target.strip() or not _PURE_CJK_RE.fullmatch(core):
            blocked["reason"] = "decision_target_has_editorial_prose"
        elif not pending_chunks or any(
                chunk_id not in pending_by_chunk for chunk_id in pending_chunks):
            blocked["reason"] = "pending_context_incomplete"

        pending_texts = {}
        if not blocked["reason"]:
            for chunk_id in pending_chunks:
                if int(pending_by_chunk[chunk_id].get("decision_id") or 0) != decision_id:
                    blocked["reason"] = "pending_decision_mismatch:" + chunk_id
                    break
                pending_texts[chunk_id] = corpus.get(chunk_id, "")
                if not pending_texts[chunk_id]:
                    blocked["reason"] = "pending_final_missing:" + chunk_id
                    break

        if not blocked["reason"]:
            other_targets = set(all_target_keys)
            other_targets.discard(_name_key(core))
            chosen = None
            rejected: List[Dict[str, str]] = []
            for candidate in group.get("candidates") or []:
                form = str(candidate.get("form") or "").strip()
                evidence, reason = _candidate_evidence(
                    source_key=source_key,
                    target=core,
                    form=form,
                    tasks=tasks,
                    corpus=corpus,
                    pending_chunk_ids=pending_chunks,
                    pending_texts=pending_texts,
                    other_target_keys=other_targets,
                )
                if evidence:
                    chosen = evidence
                    break
                rejected.append({"form": form, "reason": reason})
            if not chosen:
                blocked["reason"] = "no_safe_candidate"
                blocked["rejected_candidates"] = rejected[:8]
            else:
                result["candidates"].append({
                    "decision_id": decision_id,
                    "source_key": source_key,
                    "old_translation": target,
                    "new_translation": chosen["form"],
                    "pending_chunks": pending_chunks,
                    "stale_chunks": chosen["stale_chunks"],
                    "evidence": chosen,
                })
                continue
        result["blocked_groups"].append(blocked)

    result.update({
        "counts": {
            "groups": len(term_audit.get("groups") or []),
            "candidates": len(result["candidates"]),
            "blocked": len(result["blocked_groups"]),
        },
        "ok": not result["errors"],
    })
    _atomic_json(jpath, result)
    _atomic_text(mpath, _render_markdown(result))
    return result


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(source))
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def apply_decision_target_repairs(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    apply: bool = False,
    run_id: str = "",
) -> Dict[str, Any]:
    root = Path(project_root)
    db_path = (Path(decision_db) if decision_db
               else root / "db" / "decision_db.sqlite")
    wf_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    plan = audit_decision_target_repairs(
        root,
        decision_db=db_path,
        workflow_db=wf_path,
        output_dir=out_dir,
        json_out=out_dir / "decision_target_authority_repair_plan.json",
        md_out=out_dir / "decision_target_authority_repair_plan.md",
    )
    effective_run_id = run_id or str(plan.get("run_id") or _run_id())
    jpath = out_dir / "decision_target_authority_repair_apply.json"
    mpath = out_dir / "decision_target_authority_repair_apply.md"
    result: Dict[str, Any] = {
        "schema_version": DECISION_TARGET_REPAIR_SCHEMA_VERSION,
        "mode": "apply" if apply else "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "run_id": effective_run_id,
        "plan_counts": plan.get("counts") or {},
        "actions": [],
        "blocked_groups": plan.get("blocked_groups") or [],
        "errors": list(plan.get("errors") or []),
        "backup_path": "",
        "json_path": str(jpath),
        "md_path": str(mpath),
        "ok": False,
    }
    if not apply:
        result["ok"] = bool(plan.get("ok"))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_apply_markdown(result))
        return result
    if result["errors"] or not plan.get("candidates"):
        result["ok"] = not result["errors"]
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_apply_markdown(result))
        return result

    backup_path = (out_dir / "decision_target_authority_repair" /
                   effective_run_id / "decision_db.sqlite.bak")
    try:
        _sqlite_backup(db_path, backup_path)
    except Exception as exc:
        result["errors"].append(
            "backup_failed:%s:%s" % (type(exc).__name__, exc))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_apply_markdown(result))
        return result
    result["backup_path"] = str(backup_path)
    result["write_attempted"] = True

    con = sqlite3.connect(str(db_path), timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        for candidate in plan["candidates"]:
            did = int(candidate["decision_id"])
            row = con.execute(
                "SELECT source_key, translation, locked, reason "
                "FROM decision_db WHERE id=?", (did,)).fetchone()
            if not row:
                raise RuntimeError("decision_missing:%s" % did)
            if str(row["source_key"]) != candidate["source_key"]:
                raise RuntimeError("decision_source_changed:%s" % did)
            if str(row["translation"]) != candidate["old_translation"]:
                raise RuntimeError("decision_translation_changed:%s" % did)
            if int(row["locked"] or 0):
                raise RuntimeError("decision_locked:%s" % did)
            marker = ("Round 158 decision target authority repair: %s -> %s "
                      "(pending=%d)" % (
                          candidate["old_translation"],
                          candidate["new_translation"],
                          len(candidate.get("pending_chunks") or [])))
            reason = str(row["reason"] or "")
            if marker not in reason:
                reason = (reason.rstrip() + " | " + marker).strip(" |")
            con.execute(
                "UPDATE decision_db SET translation=?, reason=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (candidate["new_translation"], reason, did))
            for chunk_id in candidate.get("stale_chunks") or []:
                existing = con.execute(
                    "SELECT decision_id, source_key FROM pending_backtrack "
                    "WHERE chunk_id=?", (chunk_id,)).fetchone()
                if existing and (
                        int(existing["decision_id"] or 0) != did
                        or str(existing["source_key"]) != candidate["source_key"]):
                    raise RuntimeError("pending_conflict:%s" % chunk_id)
                con.execute(
                    "INSERT INTO pending_backtrack (chunk_id, decision_id, "
                    "source_key, queue_rounds, created_at, updated_at) "
                    "VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET "
                    "decision_id=excluded.decision_id, "
                    "source_key=excluded.source_key, "
                    "queue_rounds=COALESCE(pending_backtrack.queue_rounds,0)+1, "
                    "updated_at=CURRENT_TIMESTAMP",
                    (chunk_id, did, candidate["source_key"]))
            result["actions"].append({
                "decision_id": did,
                "source_key": candidate["source_key"],
                "old_translation": candidate["old_translation"],
                "new_translation": candidate["new_translation"],
                "pending_chunks": candidate.get("pending_chunks") or [],
                "queued_chunks": candidate.get("stale_chunks") or [],
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
        _atomic_text(mpath, _render_apply_markdown(result))
        return result
    finally:
        con.close()

    post = load_pending_context(
        root, decision_db=db_path, workflow_db=wf_path, output_dir=out_dir)
    if post.get("available"):
        snapshot = classify_pending_snapshot(post)
        impacted = {
            str(chunk_id)
            for action in result["actions"]
            for chunk_id in ((action.get("pending_chunks") or [])
                             + (action.get("queued_chunks") or []))
        }
        result["post_pending"] = [
            entry for entry in snapshot.get("entries") or []
            if str(entry.get("chunk_id") or "") in impacted
        ]
    result["ok"] = not result["errors"]
    _atomic_json(jpath, result)
    _atomic_text(mpath, _render_apply_markdown(result))
    return result


def _render_markdown(result: Dict[str, Any]) -> str:
    lines = ["# Decision 目标权威修正（只读）", "",
             "- 生成时间: %s" % result.get("generated_at"),
             "- 模式: **read_only**",
             "- 统计: `%s`" % (result.get("counts") or {}), ""]
    for candidate in result.get("candidates") or []:
        evidence = candidate.get("evidence") or {}
        lines.extend([
            "## #%s %s" % (candidate.get("decision_id"),
                            candidate.get("source_key")),
            "- 目标: `%s` → `%s`" % (
                candidate.get("old_translation"),
                candidate.get("new_translation")),
            "- 证据: target=%s, candidate=%s, union=%s, share=%s" % (
                evidence.get("target_hits"), evidence.get("candidate_hits"),
                len(evidence.get("explicit_union_chunks") or []),
                evidence.get("candidate_share_of_union")),
            "- pending: " + ", ".join(
                "`%s`" % value for value in candidate.get("pending_chunks") or []),
            "- 待重跑旧形态: " + ", ".join(
                "`%s`" % value for value in candidate.get("stale_chunks") or []),
            "",
        ])
    for blocked in result.get("blocked_groups") or []:
        lines.append("- BLOCKED #%s %s: `%s`" % (
            blocked.get("decision_id"), blocked.get("source_key"),
            blocked.get("reason")))
    if result.get("errors"):
        lines.extend(["", "## 错误", ""] + [
            "- `%s`" % error for error in result["errors"]])
    lines.extend(["", "本报告不修改正文、Decision、registry、pending 或 workflow。", ""])
    return "\n".join(lines).rstrip() + "\n"


def _render_apply_markdown(result: Dict[str, Any]) -> str:
    lines = ["# Decision 目标权威修正（受控写入）", "",
             "- 生成时间: %s" % result.get("generated_at"),
             "- 模式: **%s**" % result.get("mode"),
             "- ok: `%s`；写入尝试: `%s`" % (
                 result.get("ok"), result.get("write_attempted")),
             "- 备份: `%s`" % (result.get("backup_path") or "(无)"), ""]
    for action in result.get("actions") or []:
        lines.extend([
            "## #%s %s" % (action.get("decision_id"), action.get("source_key")),
            "- `%s` → `%s`" % (
                action.get("old_translation"), action.get("new_translation")),
            "- pending: " + ", ".join(
                "`%s`" % value for value in action.get("pending_chunks") or []),
            "- queued: " + ", ".join(
                "`%s`" % value for value in action.get("queued_chunks") or []),
            "- 状态: `%s`" % action.get("status"),
            "",
        ])
    for entry in result.get("post_pending") or []:
        lines.append("- post-pending `%s`: `%s`" % (
            entry.get("chunk_id"), entry.get("status")))
    for blocked in result.get("blocked_groups") or []:
        lines.append("- BLOCKED #%s %s: `%s`" % (
            blocked.get("decision_id"), blocked.get("source_key"),
            blocked.get("reason")))
    if result.get("errors"):
        lines.extend(["", "## 错误", ""] + [
            "- `%s`" % error for error in result["errors"]])
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DECISION_TARGET_REPAIR_SCHEMA_VERSION",
    "apply_decision_target_repairs",
    "audit_decision_target_repairs",
]
