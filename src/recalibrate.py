"""Read-only critic re-audit for calibrated delivery evidence.

The delivery gate checks ``*_critic_report.json`` and requires the calibrated
System One backend (the legacy ``jev`` label, currently served by MoArk d3).
Reports produced during earlier outages may still carry ``legacy`` or
``openai_compat``, even though the final translations are complete.

This module re-runs only the critic and replaces only the critic report.  It
never edits ``*_final.json``, the workflow database, statuses, or the merged
book.  The final text is the object under audit: stale literary drafts are not
accepted as evidence for the delivered translation.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp-recalibrate")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _score_average(report: Optional[dict]) -> Optional[float]:
    scores = (report or {}).get("scores") or {}
    vals = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _calibrated_backends() -> set:
    try:
        from src.config import config as _config
        gate_cfg = _config.get("pipeline.final_gate", {}) or {}
        raw = gate_cfg.get("calibrated_backends") or ["jev"]
    except Exception:
        raw = ["jev"]
    if isinstance(raw, str):
        raw = [raw]
    return {str(x) for x in raw}


def _iter_chapter_ids(scheduler) -> List[str]:
    rows = scheduler.conn.execute(
        "SELECT DISTINCT chapter_id FROM chunk_tasks ORDER BY chapter_id"
    ).fetchall()
    return [str(row[0]) for row in rows]


def _report_matches_final(report: Optional[dict], final_path: Path) -> bool:
    """Return False when a re-audit report was produced for different final text.

    Reports written by ``recalibrate-critic`` carry the SHA-256 of the final
    text they audited.  Repairs performed after that audit make the scores
    stale even though ``_backend_path`` still says ``jev``.  Older reports
    without ``_reaudit.final_sha256`` cannot be checked this way; they are
    treated as current unless ``--force`` is requested.
    """
    audited_sha = str(((report or {}).get("_reaudit") or {}).get("final_sha256") or "")
    if not audited_sha:
        return True
    final_text = _load_final_text(final_path)
    if final_text is None:
        return False
    return _sha256_text(final_text) == audited_sha


def critic_evidence_state(
    report: Optional[dict],
    final_path: Path,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Classify whether a critic report is evidence for the *current* final.

    A critic report written before ``finalize`` / ``book-repair`` is not
    sufficient merely because it still says ``_backend_path=jev``.  When a
    report carries an explicit final hash, the gate must either (a) match the
    current final exactly, or (b) prove a chain of verified ``book_repair``
    rewrites from the audited text to the current text.  The chain is only
    accepted when every link's before/after hashes and byte backup agree.

    Reports without any explicit binding are returned as ``unbound``.  They
    are legacy evidence: the backend provenance is retained, but they cannot
    claim to have audited the delivered bytes.
    """
    final_path = Path(final_path)
    final_payload = _read_json(final_path)
    if not final_payload or not isinstance(final_payload.get("text"), str):
        return {
            "status": "missing_final",
            "audited_sha256": None,
            "current_sha256": None,
            "field": None,
            "reason": "final 缺少可用 text",
        }

    current_sha = _sha256_text(final_payload["text"])
    report = report or {}
    binding = report.get("_final_binding") or {}
    reaudit = report.get("_reaudit") or {}
    target = report.get("_audit_target") or {}

    # An explicit final binding is the strongest statement.  If it names a
    # different byte sequence than the one on disk, a later stage edited the
    # final without recording a replacement binding.
    binding_final = str(binding.get("final_sha256") or "")
    if binding_final:
        if binding_final != current_sha:
            return {
                "status": "stale",
                "audited_sha256": binding_final,
                "current_sha256": current_sha,
                "field": "_final_binding.final_sha256",
                "reason": "final binding 与当前 final 哈希不一致",
            }
        # A binding may carry the original audited hash as well.  If it does
        # not, the binding itself is the only available final-level evidence.
        audited_sha = str(binding.get("audited_sha256") or "")
        if not audited_sha:
            return {
                "status": "exact",
                "audited_sha256": current_sha,
                "current_sha256": current_sha,
                "field": "_final_binding.final_sha256",
                "reason": "final binding 与当前 final 完全一致",
            }
    else:
        audited_sha = ""

    field = ""
    if not audited_sha:
        audited_sha = str(reaudit.get("final_sha256") or "")
        if audited_sha:
            field = "_reaudit.final_sha256"
    if not audited_sha:
        audited_sha = str(target.get("literary_sha256") or "")
        if audited_sha:
            field = "_audit_target.literary_sha256"
    if not audited_sha:
        audited_sha = str(binding.get("before_sha256") or "")
        if audited_sha:
            field = "_final_binding.before_sha256"

    if not audited_sha:
        return {
            "status": "unbound",
            "audited_sha256": None,
            "current_sha256": current_sha,
            "field": None,
            "reason": "critic_report 未记录 audited final 哈希",
        }

    if audited_sha == current_sha:
        return {
            "status": "exact",
            "audited_sha256": audited_sha,
            "current_sha256": current_sha,
            "field": field,
            "reason": "audited final 哈希与当前 final 完全一致",
        }

    root = Path(project_root) if project_root else final_path.parent.parent.parent
    payload = final_payload
    path = final_path
    chain: List[Dict[str, str]] = []
    seen = set()
    for _depth in range(16):
        path_key = str(path.resolve())
        if path_key in seen:
            break
        seen.add(path_key)
        metadata = payload.get("metadata") or {}
        repair = metadata.get("book_repair") or {}
        before = str(repair.get("before_sha256") or "")
        after = str(repair.get("after_sha256") or "")
        text = payload.get("text") if isinstance(payload.get("text"), str) else ""
        actual = _sha256_text(text)
        if not before or not after or after != actual:
            break
        chain.append({
            "path": str(path),
            "before_sha256": before,
            "after_sha256": after,
        })
        backup = repair.get("backup")
        if not backup:
            break
        backup_path = Path(str(backup))
        if not backup_path.is_absolute():
            backup_path = root / backup_path
        backup_payload = _read_json(backup_path)
        if not backup_payload or not isinstance(backup_payload.get("text"), str):
            break
        if _sha256_text(backup_payload["text"]) != before:
            break
        if before == audited_sha:
            return {
                "status": "repaired",
                "audited_sha256": audited_sha,
                "current_sha256": current_sha,
                "field": field,
                "repair_chain": chain,
                "reason": "audited final 经可验证 book-repair 链到达当前 final",
            }
        path = backup_path
        payload = backup_payload

    return {
        "status": "stale",
        "audited_sha256": audited_sha,
        "current_sha256": current_sha,
        "field": field,
        "repair_chain": chain,
        "reason": "audited final 与当前 final 不同，且无完整 book-repair 链",
    }


def _select_targets(root: Path, scheduler, chapters: Iterable[str],
                    chunk_ids: Optional[Iterable[str]], calibrated: set,
                    force: bool = False) -> List[dict]:
    wanted = {str(c) for c in (chunk_ids or []) if str(c)}
    targets: List[dict] = []
    for chapter_id in chapters:
        out_dir = root / "output" / chapter_id
        for task in scheduler.get_all_tasks_by_chapter(chapter_id):
            cid = str(task.get("chunk_id") or "")
            if not cid or (wanted and cid not in wanted):
                continue
            final_path = out_dir / f"{cid}_final.json"
            report_path = out_dir / f"{cid}_critic_report.json"
            if not final_path.exists():
                targets.append({
                    "chapter": chapter_id,
                    "chunk_id": cid,
                    "reason": "missing_final",
                    "old_backend": None,
                    "report_path": str(report_path),
                    "final_path": str(final_path),
                })
                continue
            report = _read_json(report_path) if report_path.exists() else None
            old_backend = str((report or {}).get("_backend_path") or "missing")
            # Round 158: a normal pipeline report carries
            # ``_audit_target.literary_sha256`` but no ``_reaudit.final_sha256``.
            # The old helper therefore treated a report from before finalize /
            # name-unification as current.  Use the same evidence contract as
            # the delivery gate so stale reports are selected without --force.
            evidence = (critic_evidence_state(report, final_path, project_root=root)
                        if report is not None else {})
            evidence_status = str(evidence.get("status") or "")
            stale = evidence_status in {"stale", "missing_final"}
            if report is not None and old_backend in calibrated and not force and not stale:
                continue
            reason = "missing_report" if report is None else "uncalibrated_report"
            if stale:
                reason = "stale_report"
            targets.append({
                "chapter": chapter_id,
                "chunk_id": cid,
                "reason": reason,
                "old_backend": None if report is None else old_backend,
                "old_avg": _score_average(report),
                "report_path": str(report_path),
                "final_path": str(final_path),
                "evidence_status": evidence_status,
            })
    return targets


def _load_final_text(path: Path) -> Optional[str]:
    data = _read_json(path)
    if not data:
        return None
    text = data.get("text")
    return text if isinstance(text, str) and text.strip() else None


def _load_intermediate_text(path: Path) -> Optional[str]:
    data = _read_json(path)
    if not data:
        return None
    text = data.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return None


def _is_retryable_result(report: dict) -> bool:
    diag = report.get("_backend_diagnostics") or {}
    categories = set()
    for value in diag.values():
        if isinstance(value, dict) and value.get("category"):
            categories.add(str(value.get("category")))
    if categories & {"auth", "dependency", "model_access", "request_schema"}:
        return False
    return True


def _is_retryable_exception(exc: Exception) -> bool:
    try:
        from src.judges.typesafe_backend import classify_backend_error
        category = str((classify_backend_error(exc) or {}).get("category") or "")
    except Exception:
        category = ""
    return category not in {"auth", "dependency", "model_access", "request_schema"}


class _WorkerState:
    def __init__(self, root: Path, low_score_threshold: float) -> None:
        self.root = root
        self.low_score_threshold = low_score_threshold
        self.local = threading.local()

    def pipeline(self, chapter_id: str):
        if not hasattr(self.local, "pipelines"):
            self.local.pipelines = {}
        pipelines = self.local.pipelines
        if chapter_id not in pipelines:
            from src.pipeline import TranslationPipeline
            pipelines[chapter_id] = TranslationPipeline(chapter_id=chapter_id)
        return pipelines[chapter_id]


def _process_one(state: _WorkerState, target: dict, calibrated: set,
                 retries: int) -> dict:
    chapter_id = target["chapter"]
    cid = target["chunk_id"]
    final_path = Path(target["final_path"])
    report_path = Path(target["report_path"])
    row = dict(target)
    row.update({"status": "failed", "new_backend": None, "new_avg": None,
                "error": None})

    final_text = _load_final_text(final_path)
    if final_text is None:
        row["error"] = "final.json 缺少可用 text"
        return row
    final_hash_before = _sha256_text(final_text)
    row["final_sha256"] = final_hash_before

    try:
        pipeline = state.pipeline(chapter_id)
        task = pipeline.scheduler.get_task(cid) or {}
        source_text = task.get("text_content") or ""
        if not source_text.strip():
            raise RuntimeError("workflow.db 缺少 chunk source_text")

        raw_text = _load_intermediate_text(final_path.with_name(f"{cid}_raw.json"))
        raw_source = "raw" if raw_text is not None else "final"
        if raw_text is None:
            raw_text = final_text

        entity_registry = pipeline._build_entity_registry_for_rewriter(source_text)
        entity_variants = pipeline._build_entity_variants_for_critic(source_text)

        report: dict = {}
        attempts_used = 0
        attempts = max(0, int(retries)) + 1
        for attempt in range(attempts):
            attempts_used = attempt + 1
            try:
                report = pipeline.critic_agent.process_chunk(
                    cid,
                    source_text,
                    raw_text,
                    final_text,
                    pipeline.style_guide,
                    entity_registry=entity_registry,
                    entity_variants=entity_variants,
                    allow_legacy_fallback=False,
                )
            except Exception as exc:
                if attempt + 1 >= attempts or not _is_retryable_exception(exc):
                    raise
                time.sleep(min(2 ** attempt, 8))
                continue
            backend = str((report or {}).get("_backend_path") or "")
            scores = (report or {}).get("scores") or {}
            if backend in calibrated and scores:
                break
            if attempt + 1 >= attempts or not _is_retryable_result(report or {}):
                break
            time.sleep(min(2 ** attempt, 8))

        from src.splitters import _normalize_scores
        if isinstance(report, dict) and "scores" in report:
            report["scores"] = _normalize_scores(report.get("scores"))

        backend = str((report or {}).get("_backend_path") or "unknown")
        avg = _score_average(report)
        row["new_backend"] = backend
        row["new_avg"] = avg
        row["low_score"] = bool(avg is not None and avg < state.low_score_threshold)

        if backend not in calibrated or not (report or {}).get("scores"):
            row["error"] = f"重审仍非校准后端: {backend}"
            return row

        report["_reaudit"] = {
            "tool": "recalibrate-critic",
            "schema_version": 1,
            "at": _utc_now(),
            "previous_backend_path": target.get("old_backend"),
            "previous_score_avg": target.get("old_avg"),
            "final_sha256": final_hash_before,
            "raw_input": raw_source,
            "literary_input": "final",
            "attempts_allowed": attempts,
            "attempts_used": attempts_used,
        }
        report["_audit_target"] = {
            "schema_version": 1,
            "source_sha256": _sha256_text(source_text),
            "raw_sha256": _sha256_text(raw_text),
            "literary_sha256": final_hash_before,
        }
        _atomic_write_json(report_path, report)

        final_text_after = _load_final_text(final_path)
        final_hash_after = _sha256_text(final_text_after or "")
        if final_hash_after != final_hash_before:
            row["error"] = "内部错误：final 文本哈希发生变化"
            return row
        row["status"] = "recalibrated"
        return row
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def run_recalibrate_critic(
    *,
    chapter_id: Optional[str] = None,
    all_chapters: bool = False,
    chunk_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    workers: int = 1,
    apply: bool = False,
    low_score_threshold: float = 7.0,
    retries: int = 1,
    force: bool = False,
    project_root: Optional[Path] = None,
) -> dict:
    """Re-audit missing/uncalibrated critic reports without touching finals."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent.parent
    if not all_chapters and not chapter_id:
        raise ValueError("recalibrate-critic 需要 --chapter 或 --all")

    # In the generated single-file snapshot, ``src.core`` is a shim that the
    # build script intentionally removes when probing for the real scheduler.
    # The scheduler is already inlined in that deployment; fall back to the
    # inlined class when the import cannot resolve.
    try:
        from src.core.scheduler import TaskScheduler
    except ImportError:
        TaskScheduler = globals().get("TaskScheduler")
        if TaskScheduler is None:
            raise
    scheduler = TaskScheduler(db_path=str(root / "db" / "workflow.db"))
    try:
        chapters = _iter_chapter_ids(scheduler) if all_chapters else [str(chapter_id)]
        calibrated = _calibrated_backends()
        targets = _select_targets(root, scheduler, chapters, chunk_ids, calibrated,
                                  force=force)
        if limit is not None:
            targets = targets[:max(0, int(limit))]

        summary = {
            "ok": True,
            "applied": bool(apply),
            "chapters": chapters,
            "calibrated_backends": sorted(calibrated),
            "selected": len(targets),
            "recalibrated": 0,
            "failed": 0,
            "low_score": 0,
            "low_score_threshold": float(low_score_threshold),
            "workers": max(1, int(workers)),
            "rows": [],
            "errors": [],
            "low_scores": [],
            "started_at": _utc_now(),
        }
        if not apply:
            summary["dry_run"] = True
            summary["rows"] = targets
            return summary

        merged_before = None
        merged_path = root / "output" / "沙丘.md"
        if merged_path.exists():
            merged_before = _sha256_file(merged_path)

        state = _WorkerState(root, float(low_score_threshold))
        rows: List[dict] = []
        total = len(targets)
        if total:
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                futures = {
                    pool.submit(_process_one, state, target, calibrated, retries): target
                    for target in targets
                }
                done = 0
                for fut in as_completed(futures):
                    row = fut.result()
                    rows.append(row)
                    done += 1
                    if row.get("status") != "recalibrated" or row.get("low_score"):
                        print(
                            f"  ⚠️ {row['chunk_id']}: {row.get('status')} "
                            f"backend={row.get('new_backend')} avg={row.get('new_avg')} "
                            f"{row.get('error') or ''}".rstrip()
                        )
                    if done % 10 == 0 or done == total:
                        print(f"  … recalibrate-critic {done}/{total}")

        rows.sort(key=lambda r: (r.get("chapter") or "", r.get("chunk_id") or ""))
        summary["rows"] = rows
        summary["recalibrated"] = sum(1 for r in rows if r.get("status") == "recalibrated")
        summary["failed"] = sum(1 for r in rows if r.get("status") != "recalibrated")
        summary["low_scores"] = [
            {"chapter": r.get("chapter"), "chunk_id": r.get("chunk_id"),
             "avg": r.get("new_avg"), "backend": r.get("new_backend")}
            for r in rows if r.get("low_score")
        ]
        summary["low_score"] = len(summary["low_scores"])
        summary["errors"] = [
            {"chapter": r.get("chapter"), "chunk_id": r.get("chunk_id"),
             "error": r.get("error")}
            for r in rows if r.get("status") != "recalibrated"
        ]

        if merged_before is not None:
            merged_after = _sha256_file(merged_path) if merged_path.exists() else None
            summary["merged_sha256_before"] = merged_before
            summary["merged_sha256_after"] = merged_after
            if merged_after != merged_before:
                summary["ok"] = False
                summary["errors"].append({"error": "output/沙丘.md 哈希发生变化"})
        if summary["failed"]:
            summary["ok"] = False
        summary["finished_at"] = _utc_now()

        report_path = root / "output" / "recalibrate_critic_report.json"
        _atomic_write_json(report_path, summary)
        summary["report_path"] = str(report_path)
        return summary
    finally:
        try:
            scheduler.conn.close()
        except Exception:
            pass
