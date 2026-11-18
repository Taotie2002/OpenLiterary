"""Read-only recovery audit for chunks left in an active pipeline stage.

This module deliberately does not reset states or delete backups.  It records
what a previous run left behind so the next run can resume by the declared
stage instead of treating an interrupted chunk as completed.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.task_logging import now_iso, sha256_file


RECOVERY_DIR = Path("output") / "recovery"
NONTERMINAL_STATES = (
    "EXTRACTING_TERMS", "TRANSLATING_RAW", "REWRITING_LITERARY", "AUDITING", "JUDGING",
)
RESUME_STAGE = {
    "EXTRACTING_TERMS": "EXTRACTING_TERMS",
    "TRANSLATING_RAW": "TRANSLATING_RAW",
    "REWRITING_LITERARY": "REWRITING_LITERARY",
    "AUDITING": "AUDITING",
    "JUDGING": "JUDGING",
}


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = "file:" + str(Path(db_path).resolve()) + "?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return sqlite3.connect(str(db_path))


def _artifact_info(path: Path) -> Dict[str, Any]:
    info = {"path": str(path), "exists": False, "bytes": 0, "sha256": ""}
    try:
        if path.is_file():
            info["exists"] = True
            info["bytes"] = path.stat().st_size
            info["sha256"] = "sha256:" + sha256_file(path)
    except OSError:
        pass
    return info


def _pending_counts(root: Path, chunk_ids: List[str]) -> Dict[str, int]:
    if not chunk_ids:
        return {}
    db_path = Path(root) / "db" / "decision_db.sqlite"
    if not db_path.exists():
        return {}
    counts: Dict[str, int] = {}
    try:
        conn = _connect_readonly(db_path)
        try:
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = conn.execute(
                "SELECT chunk_id, COUNT(*) FROM pending_backtrack "
                f"WHERE chunk_id IN ({placeholders}) GROUP BY chunk_id",
                chunk_ids,
            ).fetchall()
            counts = {str(row[0]): int(row[1]) for row in rows}
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    return counts


def scan_inflight(root: Path, chapter_id: Optional[str] = None,
                  run_id: Optional[str] = None) -> Dict[str, Any]:
    """Scan workflow.db without changing any row and return a recovery report."""
    root = Path(root).resolve()
    db_path = root / "db" / "workflow.db"
    report: Dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(run_id or os.environ.get("SAGATRANS_RUN_ID") or ""),
        "generated_at": now_iso(),
        "chapter": chapter_id or "all",
        "db_path": str(db_path),
        "nonterminal_states": list(NONTERMINAL_STATES),
        "chunks": [],
        "summary": {"nonterminal": 0, "by_state": {}, "by_chapter": {},
                    "with_final": 0, "with_final_backup": 0,
                    "with_critic_report": 0, "with_critic_report_backup": 0},
        "warnings": [],
    }
    if not db_path.exists():
        report["warnings"].append("workflow_db_missing")
        return report
    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error as exc:
        report["warnings"].append(f"workflow_db_open_failed:{exc}")
        return report

    rows: List[tuple] = []
    try:
        placeholders = ",".join("?" for _ in NONTERMINAL_STATES)
        sql = (
            "SELECT chapter_id, chunk_id, state, retries, quality_retries, "
            "last_error, updated_at, backtrack_rounds, rescue_attempts, rescue_stage "
            "FROM chunk_tasks WHERE state IN (" + placeholders + ")"
        )
        params: List[Any] = list(NONTERMINAL_STATES)
        if chapter_id:
            sql += " AND chapter_id = ?"
            params.append(chapter_id)
        sql += " ORDER BY chapter_id, chunk_id"
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        report["warnings"].append(f"workflow_query_failed:{exc}")
    finally:
        conn.close()

    chunk_ids = [str(row[1]) for row in rows]
    pending = _pending_counts(root, chunk_ids)
    for row in rows:
        chapter, chunk_id, state, retries, quality_retries, last_error, updated_at, \
            backtrack_rounds, rescue_attempts, rescue_stage = row
        out_dir = root / "output" / str(chapter)
        final_path = out_dir / f"{chunk_id}_final.json"
        final_backup = out_dir / f"{chunk_id}_final_backup.json"
        critic_path = out_dir / f"{chunk_id}_critic_report.json"
        critic_backup = out_dir / f"{chunk_id}_critic_report_backup.json"
        final_info = _artifact_info(final_path)
        backup_info = _artifact_info(final_backup)
        critic_info = _artifact_info(critic_path)
        critic_backup_info = _artifact_info(critic_backup)
        report["chunks"].append({
            "chapter": str(chapter),
            "chunk_id": str(chunk_id),
            "state": str(state),
            "status": "interrupted_inflight",
            "resume_stage": RESUME_STAGE.get(str(state), str(state)),
            "final_exists": bool(final_info["exists"]),
            "final": final_info,
            "final_backup_present": bool(backup_info["exists"]),
            "final_backup": backup_info,
            "critic_report_exists": bool(critic_info["exists"]),
            "critic_report": critic_info,
            "critic_report_backup_present": bool(critic_backup_info["exists"]),
            "critic_report_backup": critic_backup_info,
            "pending_backtrack_rows": int(pending.get(str(chunk_id), 0)),
            "retries": int(retries or 0),
            "quality_retries": int(quality_retries or 0),
            "backtrack_rounds": int(backtrack_rounds or 0),
            "rescue_attempts": int(rescue_attempts or 0),
            "rescue_stage": str(rescue_stage or ""),
            "last_error": str(last_error or "")[:500],
            "updated_at": str(updated_at or ""),
            "safe_to_skip_stage": False,
            "backup_preserved": True,
            "completed": False,
        })
        summary = report["summary"]
        summary["nonterminal"] += 1
        summary["by_state"][str(state)] = summary["by_state"].get(str(state), 0) + 1
        summary["by_chapter"][str(chapter)] = summary["by_chapter"].get(str(chapter), 0) + 1
        if final_info["exists"]:
            summary["with_final"] += 1
        if backup_info["exists"]:
            summary["with_final_backup"] += 1
        if critic_info["exists"]:
            summary["with_critic_report"] += 1
        if critic_backup_info["exists"]:
            summary["with_critic_report_backup"] += 1
    return report


def write_recovery_report(root: Path, report: Dict[str, Any],
                          run_id: Optional[str] = None) -> Path:
    """Atomically publish the recovery report; never touches workflow rows."""
    root = Path(root).resolve()
    directory = root / RECOVERY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    identifier = str(run_id or report.get("run_id") or "manual")
    path = directory / f"{identifier}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def audit_and_write(root: Path, chapter_id: Optional[str] = None,
                    run_id: Optional[str] = None) -> Dict[str, Any]:
    report = scan_inflight(root, chapter_id=chapter_id, run_id=run_id)
    path = write_recovery_report(root, report, run_id=run_id)
    report["report_path"] = str(path)
    return report


__all__ = [
    "RECOVERY_DIR", "NONTERMINAL_STATES", "RESUME_STAGE",
    "scan_inflight", "write_recovery_report", "audit_and_write",
]
