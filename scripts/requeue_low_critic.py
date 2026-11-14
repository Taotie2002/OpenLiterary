#!/usr/bin/env python3
"""Queue completed chunks whose calibrated critic average is below threshold.

The queue uses ``TaskScheduler.queue_rescue`` with ``REWRITING_LITERARY``:
the existing raw translation is reused, so this is a targeted rewrite rather
than a full retranslation.  Dry-run by default.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _avg(report: dict) -> float | None:
    scores = report.get("scores") or {}
    vals = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else None


def select_candidates(root: Path, threshold: float,
                      chapters: List[str] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = "output/ch*/*_critic_report.json"
    for path in sorted(root.glob(pattern)):
        chunk_id = path.name[: -len("_critic_report.json")]
        chapter = path.parent.name
        if chapters and chapter not in chapters:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        avg = _avg(report)
        if avg is None or avg >= threshold:
            continue
        rows.append({
            "chapter": chapter,
            "chunk_id": chunk_id,
            "avg": avg,
            "backend": str(report.get("_backend_path") or "unknown"),
            "scores": report.get("scores") or {},
        })
    rows.sort(key=lambda r: (r["avg"], r["chunk_id"]))
    return rows


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="queue low-critic chunks for targeted rewrite")
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--chapters", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-out", default="output/low_critic_requeue.json")
    args = parser.parse_args()

    chapters = [c.strip() for c in args.chapters.split(",") if c.strip()]
    candidates = select_candidates(root, args.threshold, chapters or None)
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    scheduled: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    if args.apply:
        sys.path.insert(0, str(root))
        from src.core.scheduler import TaskScheduler

        scheduler = TaskScheduler(db_path=str(root / "db" / "workflow.db"))
        try:
            for item in candidates:
                task = scheduler.get_task(item["chunk_id"]) or {}
                state = str(task.get("state") or "missing")
                if state != "COMPLETED":
                    skipped.append({**item, "reason": f"state={state}"})
                    continue
                res = scheduler.queue_rescue(
                    item["chunk_id"], "REWRITING_LITERARY",
                    f"low_critic_score:{item['avg']:.2f}; round153_quality_rescue",
                    reset_counters=True,
                )
                if res.get("ok"):
                    scheduled.append({**item, "attempt": res.get("attempt")})
                else:
                    skipped.append({**item, "reason": str(res.get("reason"))})
        finally:
            scheduler.conn.close()

    payload = {
        "schema_version": 1,
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "threshold": args.threshold,
        "selected": len(candidates),
        "scheduled": scheduled,
        "skipped": skipped,
    }
    out = root / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    by_chapter: Dict[str, int] = {}
    for item in candidates:
        by_chapter[item["chapter"]] = by_chapter.get(item["chapter"], 0) + 1
    print(json.dumps({
        "mode": payload["mode"], "selected": len(candidates),
        "scheduled": len(scheduled), "skipped": len(skipped),
        "by_chapter": by_chapter, "json": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
