"""Round 158: read-only evidence audit for unresolved term/form contracts.

The pending outbox records a Decision id and its expected translation.  When a
COMPLETED final does not contain that exact target, a single substring check
cannot tell whether the Decision target is too narrow, the book legitimately
uses several forms, or the final contains a real translation error.

This module does not rewrite prose, alter Decisions, consume pending rows, or
lock anything.  It builds a deterministic, corpus-backed report:

* source-positive chunks are all COMPLETED chunks containing the Decision's
  English source key;
* candidate Chinese forms are ranked by document-frequency lift against the
  source-negative corpus;
* every unresolved pending chunk gets its near-target forms and matching
  corpus candidates;
* groups are classified as stem contract, multi-form contract, likely body
  drift, target missing, or weak evidence.

The report is evidence for a later controlled repair/alignment, never a write
permission by itself.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.pending_backtrack_reconcile import (
    _name_key,
    _source_phrase_spans,
    _target_core,
    body_before_footnotes,
    classify_pending_snapshot,
    load_pending_context,
)


TERM_FORM_AUDIT_SCHEMA_VERSION = 1
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


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


def _cjk_ngrams(text: str) -> set:
    out = set()
    for run in _CJK_RE.findall(str(text or "")):
        limit = min(8, len(run))
        for size in range(2, limit + 1):
            for start in range(len(run) - size + 1):
                out.add(run[start:start + size])
    return out


def _similarity(left: str, right: str) -> float:
    a, b = _name_key(left), _name_key(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return float(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio())


def _read_final_text(context: Dict[str, Any], chunk_id: str,
                     chapter: str) -> str:
    path = ((context.get("output_dir") or Path(".")) / chapter /
            ("%s_final.json" % chunk_id))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return str(payload.get("text") or "") if isinstance(payload, dict) else ""


def _completed_corpus(context: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    corpus: Dict[str, Dict[str, str]] = {}
    for chunk_id, task in (context.get("tasks") or {}).items():
        if str(task.get("state") or "") != "COMPLETED":
            continue
        chapter = str(task.get("chapter_id") or str(chunk_id)[:4])
        final_text = _read_final_text(context, str(chunk_id), chapter)
        corpus[str(chunk_id)] = {
            "chapter": chapter,
            "source": str(task.get("text_content") or ""),
            "final": body_before_footnotes(final_text),
        }
    return corpus


def _rank_candidates(
    positive_ids: set,
    gram_cache: Dict[str, set],
    limit: int = 12,
) -> List[Dict[str, Any]]:
    total_positive = len(positive_ids)
    total_negative = max(0, len(gram_cache) - total_positive)
    if not total_positive or not total_negative:
        return []
    positive_df: Counter = Counter()
    negative_df: Counter = Counter()
    for chunk_id, grams in gram_cache.items():
        bucket = positive_df if chunk_id in positive_ids else negative_df
        for gram in grams:
            bucket[gram] += 1
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for gram, positive_hits in positive_df.items():
        if len(gram) < 2:
            continue
        if total_positive >= 3 and positive_hits < 2:
            continue
        negative_hits = int(negative_df.get(gram, 0))
        if (total_positive >= 3 and total_negative
                and negative_hits > max(2, int(total_negative * 0.08))):
            continue
        coverage = positive_hits / float(total_positive)
        background = negative_hits / float(total_negative)
        lift = coverage - background * 1.5
        adjusted = lift + min(len(gram), 6) * 0.02
        ranked.append((adjusted, {
            "form": gram,
            "positive_chunks": int(positive_hits),
            "positive_coverage": round(coverage, 4),
            "negative_chunks": negative_hits,
            "negative_coverage": round(background, 4),
            "lift": round(lift, 4),
            "score": round(adjusted, 4),
        }))
    ranked.sort(key=lambda item: (-item[0], -len(item[1]["form"]),
                                  item[1]["form"]))
    selected: List[Dict[str, Any]] = []
    keys = set()
    for _score, row in ranked:
        form = str(row["form"])
        key = _name_key(form)
        if not key or key in keys:
            continue
        keys.add(key)
        selected.append(row)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def _near_target_forms(text: str, target: str, *,
                       limit: int = 6) -> List[Dict[str, Any]]:
    target_key = _name_key(target)
    if not target_key:
        return []
    rows: List[Tuple[float, int, str]] = []
    for gram in _cjk_ngrams(text):
        key = _name_key(gram)
        if not key or key == target_key:
            continue
        similarity = _similarity(key, target_key)
        if similarity < 0.45:
            continue
        rows.append((similarity, len(key), gram))
    rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
    unique: List[Dict[str, Any]] = []
    seen = set()
    for similarity, length, form in rows:
        key = _name_key(form)
        if key in seen:
            continue
        seen.add(key)
        unique.append({
            "form": form,
            "similarity": round(similarity, 4),
            "length": length,
            "key": key,
            "target_substring": bool(key in target_key and key != target_key),
            "target_stem": bool(
                key != target_key
                and len(key) >= 3
                and len(target_key) >= 4
                and (target_key.startswith(key) or target_key.endswith(key))),
        })
    out: List[Dict[str, Any]] = []
    for row in unique:
        key = str(row.get("key") or "")
        similarity = float(row.get("similarity") or 0.0)
        if any(
                len(other["key"]) > len(key)
                and key in other["key"]
                and float(other.get("similarity") or 0.0) >= similarity - 0.04
                for other in unique):
            continue
        payload = dict(row)
        payload.pop("key", None)
        out.append(payload)
        if len(out) >= limit:
            break
    return out


def _candidate_stats(candidate: str, gram_cache: Dict[str, set],
                     positive_ids: set) -> Dict[str, Any]:
    key = _name_key(candidate)
    hits = [cid for cid, grams in gram_cache.items() if key in grams]
    positive_hits = sum(1 for cid in hits if cid in positive_ids)
    return {
        "form": candidate,
        "positive_chunks": positive_hits,
        "negative_chunks": len(hits) - positive_hits,
    }


def _classify_group(
    *,
    target: str,
    target_hits: int,
    total_positive: int,
    pending_rows: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Tuple[str, str]:
    if not pending_rows or not candidates:
        return "weak_evidence", "no_group_candidate_or_pending_row"
    all_stem = all(
        any(form.get("target_stem") for form in row.get("near_target_forms") or [])
        for row in pending_rows)
    if all_stem and (target_hits or total_positive):
        return "target_stem_contract", "every_pending_chunk_uses_target_stem"
    near_forms = [
        form for row in pending_rows
        for form in row.get("near_target_forms") or []
        if not form.get("target_stem") and not form.get("target_substring")
    ]
    if target_hits == 0:
        return "target_missing_contract", "decision_target_absent_from_completed_corpus"
    if near_forms:
        best = max(near_forms, key=lambda form: form.get("similarity") or 0.0)
        stats = best.get("corpus") or {}
        alternate_hits = int(stats.get("positive_chunks") or 0)
        if alternate_hits * 2 > max(1, target_hits):
            return "multi_form_contract", "alternate_form_is_materially_frequent"
        return "body_drift_candidate", "target_is_book_majority_but_pending_uses_near_form"
    return "target_missing_contract", "pending_rows_have_no_near_target_form"


def audit_term_forms(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "term_form_audit.json"
    mpath = Path(md_out) if md_out else out_dir / "term_form_audit.md"
    result: Dict[str, Any] = {
        "schema_version": TERM_FORM_AUDIT_SCHEMA_VERSION,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "groups": [],
        "counts": {},
        "errors": [],
        "ok": False,
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    context = load_pending_context(
        root, decision_db=decision_db, workflow_db=workflow_db,
        output_dir=out_dir)
    if not context.get("available"):
        result["errors"].append(str(context.get("reason") or "pending_context_unavailable"))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_markdown(result))
        return result
    snapshot = classify_pending_snapshot(context)
    outstanding = [
        entry for entry in snapshot.get("entries") or []
        if entry.get("status") == "outstanding"
    ]
    corpus = _completed_corpus(context)
    gram_cache = {
        chunk_id: _cjk_ngrams(row.get("final") or "")
        for chunk_id, row in corpus.items()
    }
    grouped: Dict[Tuple[int, str, str], List[dict]] = defaultdict(list)
    for entry in outstanding:
        grouped[(
            int(entry.get("decision_id") or 0),
            str(entry.get("decision_source_key") or ""),
            str(entry.get("decision_translation") or ""),
        )].append(entry)
    groups: List[Dict[str, Any]] = []
    for (decision_id, source_key, target), pending_rows in sorted(
            grouped.items(), key=lambda item: (-len(item[1]), item[0][1])):
        positive_ids = {
            chunk_id for chunk_id, row in corpus.items()
            if _source_phrase_spans(row.get("source") or "", source_key)
        }
        if not positive_ids:
            groups.append({
                "decision_id": decision_id,
                "source_key": source_key,
                "target": target,
                "classification": "weak_evidence",
                "reason": "source_phrase_absent_from_completed_corpus",
                "pending_chunks": [row.get("chunk_id") for row in pending_rows],
                "pending_rows": [],
                "candidates": [],
                "target_hits": 0,
                "source_positive_chunks": 0,
            })
            continue
        candidates = _rank_candidates(positive_ids, gram_cache)
        target_key = _name_key(_target_core(target))
        target_hits = sum(
            1 for chunk_id in positive_ids
            if target_key in _name_key(corpus[chunk_id].get("final") or ""))
        enriched_rows: List[Dict[str, Any]] = []
        for pending_row in pending_rows:
            chunk_id = str(pending_row.get("chunk_id") or "")
            text = corpus.get(chunk_id, {}).get("final") or ""
            near = _near_target_forms(text, _target_core(target))
            for form in near:
                form["corpus"] = _candidate_stats(
                    str(form.get("form") or ""), gram_cache, positive_ids)
            matching_candidates = []
            for candidate in candidates:
                form = str(candidate.get("form") or "")
                if _name_key(form) and _name_key(form) in _name_key(text):
                    matching_candidates.append(candidate)
            enriched_rows.append({
                "chunk_id": chunk_id,
                "state": pending_row.get("state"),
                "queue_rounds": pending_row.get("queue_rounds"),
                "final_chars": len(text),
                "near_target_forms": near,
                "matching_group_candidates": matching_candidates[:6],
            })
        classification, reason = _classify_group(
            target=target, target_hits=target_hits,
            total_positive=len(positive_ids), pending_rows=enriched_rows,
            candidates=candidates)
        groups.append({
            "decision_id": decision_id,
            "source_key": source_key,
            "target": target,
            "classification": classification,
            "reason": reason,
            "pending_chunks": [row.get("chunk_id") for row in pending_rows],
            "pending_rows": enriched_rows,
            "candidates": candidates,
            "target_hits": target_hits,
            "source_positive_chunks": len(positive_ids),
        })
    by_class = Counter(str(group.get("classification") or "") for group in groups)
    result.update({
        "groups": groups,
        "counts": {
            "outstanding_rows": len(outstanding),
            "groups": len(groups),
            "completed_corpus": len(corpus),
            **{key: int(value) for key, value in sorted(by_class.items())},
        },
        "ok": not result["errors"],
    })
    _atomic_json(jpath, result)
    _atomic_text(mpath, _render_markdown(result))
    return result


def _render_markdown(result: Dict[str, Any]) -> str:
    lines = ["# 术语形态只读审计", "",
             "- 模式: **read_only**",
             "- 生成时间: %s" % result.get("generated_at"),
             "- 统计: `%s`" % (result.get("counts") or {}), ""]
    groups = result.get("groups") or []
    if groups:
        lines.extend(["## 分诊（前 40 组）"])
        for group in groups[:40]:
            lines.append(
                "- `#%s %s → %s` / %s / target_hits=%s/%s / pending=%s"
                % (group.get("decision_id"), group.get("source_key"),
                   group.get("target"), group.get("classification"),
                   group.get("target_hits"), group.get("source_positive_chunks"),
                   ",".join(group.get("pending_chunks") or [])))
            candidates = group.get("candidates") or []
            if candidates:
                lines.append("  - 候选: " + ", ".join(
                    "`%s`(pos=%s,neg=%s)" % (
                        row.get("form"), row.get("positive_chunks"),
                        row.get("negative_chunks"))
                    for row in candidates[:6]))
            for pending in (group.get("pending_rows") or [])[:4]:
                near = pending.get("near_target_forms") or []
                if near:
                    lines.append("  - `%s` 近形: %s" % (
                        pending.get("chunk_id"), ", ".join(
                            "`%s`(sim=%.2f%s)" % (
                                row.get("form"), row.get("similarity") or 0.0,
                                ",stem" if row.get("target_stem") else "")
                            for row in near[:4])))
    if result.get("errors"):
        lines.extend(["", "## 错误", ""] + [
            "- `%s`" % error for error in result["errors"]])
    lines.extend(["", "本报告不修改正文、Decision、registry、pending 或 workflow。", ""])
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "TERM_FORM_AUDIT_SCHEMA_VERSION",
    "audit_term_forms",
]
