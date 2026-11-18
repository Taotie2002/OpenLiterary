"""Round 158 Phase 1.5b: read-only authority audit for locked Decisions.

The pending reconciler intentionally treats a ``locked`` Decision as hard
authority.  That is only safe if the lock itself was acquired from book-wide
evidence.  Round 158 found the opposite: a local name-only repair could lock a
Decision merely because its target was proposed for one chunk.  A later
book-wide scan then found that the locked target occurred nowhere while the
book consistently used a different form.

This module does not unlock or rewrite anything.  It inspects every pending row
blocked by a locked target (including a more-specific target behind a generic
key) and classifies the authority using source-positive COMPLETED chunks:

* ``confirmed``: the locked target is the book-wide majority;
* ``contradicted``: the locked target is absent and a different form is a clear
  book-wide majority;
* ``ambiguous``: two or more meaningful forms split the corpus;
* ``legitimate_variant``: the pending prose uses the same characters in a
  legitimate title/word-order arrangement;
* ``insufficient_evidence``: the corpus is too small or has no usable form.

The report is evidence for a later controlled unlock/lock-repair path.  It is
not permission to change a Decision, a final, pending_backtrack or workflow
state by itself.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pending_backtrack_reconcile import (
    _all_source_occurrences_have_more_specific_decision,
    _name_key,
    _source_phrase_spans,
    _strictly_more_specific_source,
    _target_core,
    classify_pending_snapshot,
    load_pending_context,
)
from src.term_form_audit import (
    _candidate_stats,
    _cjk_ngrams,
    _completed_corpus,
    _near_target_forms,
    _rank_candidates,
    _similarity,
)
from src.term_form_contracts import (
    _FUNCTION_INFIXES,
    _FUNCTION_PREFIXES,
    _FUNCTION_SUFFIXES,
)


LOCKED_TARGET_AUDIT_SCHEMA_VERSION = 1
LOCK_AUTHORITY_REASONS = {
    "locked_decision_target_missing",
    "specific_decision_target_missing",
}


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


def _source_positive_ids(corpus: Dict[str, Dict[str, str]],
                         source_key: str) -> List[str]:
    return [
        chunk_id for chunk_id, row in corpus.items()
        if _source_phrase_spans(row.get("source") or "", source_key)
    ]


def _authority_ids_for_decision(
    *,
    context: Dict[str, Any],
    corpus: Dict[str, Dict[str, str]],
    decision_id: int,
    source_key: str,
) -> Tuple[List[str], List[str]]:
    source_positive = _source_positive_ids(corpus, source_key)
    generic_positive = [
        chunk_id for chunk_id in source_positive
        if not _all_source_occurrences_have_more_specific_decision(
            pending_source=source_key,
            pending_decision_id=int(decision_id),
            source_text=corpus[chunk_id].get("source") or "",
            decisions=context.get("decisions") or {},
        )
    ]
    return source_positive, generic_positive


def _target_hit_ids(corpus: Dict[str, Dict[str, str]],
                    positive_ids: List[str], target: str) -> List[str]:
    target_key = _name_key(_target_core(target))
    if not target_key:
        return []
    return [
        chunk_id for chunk_id in positive_ids
        if target_key in _name_key(corpus[chunk_id].get("final") or "")
    ]


def _same_character_multiset(left: str, right: str) -> bool:
    a, b = _name_key(left), _name_key(right)
    return bool(a and b and a != b and sorted(a) == sorted(b))


def _contains_character_permutation(text: str, target: str) -> bool:
    """Detect title/word-order variants such as 圣母盖乌斯… vs 盖乌斯…圣母."""
    text_key = _name_key(text)
    target_key = _name_key(_target_core(target))
    if not text_key or not target_key or len(text_key) < len(target_key):
        return False
    wanted = sorted(target_key)
    size = len(target_key)
    for start in range(len(text_key) - size + 1):
        if sorted(text_key[start:start + size]) == wanted:
            return True
    return False


def _specific_covering_decisions(
    *,
    source_text: str,
    generic_source: str,
    generic_decision_id: int,
    decisions: Dict[int, dict],
) -> List[Dict[str, Any]]:
    """Return longer Decisions that cover every generic source occurrence."""
    generic_spans = _source_phrase_spans(source_text, generic_source)
    if not generic_spans:
        return []
    covering: List[Dict[str, Any]] = []
    for decision_id, decision in (decisions or {}).items():
        if int(decision_id) == int(generic_decision_id):
            continue
        candidate_source = str(decision.get("source_key") or "")
        if not _strictly_more_specific_source(generic_source, candidate_source):
            continue
        spans = _source_phrase_spans(source_text, candidate_source)
        if not spans:
            continue
        covering.append({
            "decision_id": int(decision_id),
            "source_key": candidate_source,
            "target": _target_core(decision.get("translation")),
            "spans": spans,
        })
    if not covering:
        return []
    if not all(
        any(start <= span[0] and span[1] <= end
            for item in covering for start, end in item["spans"])
        for span in generic_spans
    ):
        return []
    out: List[Dict[str, Any]] = []
    for item in covering:
        payload = dict(item)
        payload.pop("spans", None)
        out.append(payload)
    return sorted(out, key=lambda row: (len(row["source_key"]), row["decision_id"]))


def _ranked_lock_candidates(
    *,
    target: str,
    positive_ids: List[str],
    corpus: Dict[str, Dict[str, str]],
    gram_cache: Dict[str, set],
) -> List[Dict[str, Any]]:
    target_key = _name_key(_target_core(target))
    ranked = _rank_candidates(set(positive_ids), gram_cache, limit=64)
    out: List[Dict[str, Any]] = []
    for row in ranked:
        form = str(row.get("form") or "")
        key = _name_key(form)
        if not key or key == target_key:
            continue
        if key in target_key or target_key in key:
            continue
        if form.startswith(_FUNCTION_PREFIXES):
            continue
        if form.endswith(_FUNCTION_SUFFIXES):
            continue
        if any(char in form[1:-1] for char in _FUNCTION_INFIXES):
            continue
        if len(key) > len(target_key) + 1:
            continue
        target_inside_candidate = bool(target_key and target_key in key)
        evidence = _candidate_stats(form, gram_cache, set(positive_ids))
        coverage = evidence["positive_chunks"] / float(max(1, len(positive_ids)))
        similarity = _similarity(form, target_key)
        if coverage < 0.5 and similarity < 0.3:
            continue
        evidence.update({
            "positive_coverage": round(coverage, 4),
            "target_substring": bool(key in target_key),
            "target_inside_candidate": target_inside_candidate,
            "similarity_to_target": round(similarity, 4),
            "pending_chunks": [
                chunk_id for chunk_id in positive_ids
                if key in _name_key(corpus[chunk_id].get("final") or "")
            ],
        })
        out.append(evidence)
    filtered: List[Dict[str, Any]] = []
    for row in out:
        form = str(row.get("form") or "")
        hits = int(row.get("positive_chunks") or 0)
        if any(
            form != str(other.get("form") or "")
            and form in str(other.get("form") or "")
            and int(other.get("positive_chunks") or 0) >= hits - 1
            for other in out
        ):
            continue
        filtered.append(row)
    return filtered


def _pending_form_evidence(
    *,
    pending_entries: List[Dict[str, Any]],
    corpus: Dict[str, Dict[str, str]],
    target: str,
    candidates: List[Dict[str, Any]],
    gram_cache: Dict[str, set],
    positive_ids: List[str],
) -> Tuple[List[Dict[str, Any]], bool]:
    rows: List[Dict[str, Any]] = []
    legitimate_variant = False
    positive_set = set(positive_ids)
    for entry in pending_entries:
        chunk_id = str(entry.get("chunk_id") or "")
        text = corpus.get(chunk_id, {}).get("final") or ""
        if _contains_character_permutation(text, target):
            legitimate_variant = True
        near = _near_target_forms(text, target)
        for form in near:
            form["corpus"] = _candidate_stats(
                str(form.get("form") or ""), gram_cache, positive_set)
            if _same_character_multiset(
                    str(form.get("form") or ""), _target_core(target)):
                legitimate_variant = True
        matching = [
            row for row in candidates
            if _name_key(str(row.get("form") or "")) in _name_key(text)
        ]
        rows.append({
            "chunk_id": chunk_id,
            "reason": entry.get("reason"),
            "state": entry.get("state"),
            "queue_rounds": entry.get("queue_rounds"),
            "final_chars": len(text),
            "near_target_forms": near,
            "matching_candidates": matching[:8],
        })
    return rows, legitimate_variant


def _classify_lock_authority(
    *,
    target: str,
    target_hits: int,
    total_positive: int,
    candidates: List[Dict[str, Any]],
    legitimate_variant: bool,
) -> Tuple[str, str]:
    if total_positive <= 0:
        return "insufficient_evidence", "source_phrase_absent_from_completed_corpus"
    usable = [
        row for row in candidates
        if int(row.get("positive_chunks") or 0) >= 2
        and not bool(row.get("target_substring"))
        and not bool(row.get("target_inside_candidate"))
        and (target_hits == 0 or float(row.get("similarity_to_target") or 0.0) >= 0.2)
    ]
    usable.sort(key=lambda row: (
        -int(row.get("positive_chunks") or 0),
        -float(row.get("positive_coverage") or 0.0),
        -len(str(row.get("form") or "")),
        str(row.get("form") or ""),
    ))
    best = usable[0] if usable else None
    second_hits = int(usable[1].get("positive_chunks") or 0) if len(usable) > 1 else 0
    best_hits = int(best.get("positive_chunks") or 0) if best else 0
    majority = max(2, int(math.ceil(total_positive * 0.5)))

    if target_hits == 0:
        if legitimate_variant:
            return "legitimate_variant", "pending_uses_same_characters_in_another_order"
        if best and best_hits >= majority and best_hits >= second_hits * 2:
            return "contradicted", "locked_target_absent_and_alternate_form_dominates"
        if best and best_hits >= 2:
            return "ambiguous", "locked_target_absent_and_alternate_forms_split"
        return "insufficient_evidence", "locked_target_absent_without_strong_alternate"

    if target_hits >= majority:
        return "confirmed", "locked_target_is_book_wide_majority"
    return "ambiguous", "locked_target_is_minority_in_completed_corpus"


def _build_group(
    *,
    authority_kind: str,
    decision_id: int,
    source_key: str,
    target: str,
    pending_entries: List[Dict[str, Any]],
    context: Dict[str, Any],
    corpus: Dict[str, Dict[str, str]],
    gram_cache: Dict[str, set],
    pending_decision_id: int = 0,
    pending_reason: str = "",
) -> Dict[str, Any]:
    source_positive, generic_positive = _authority_ids_for_decision(
        context=context, corpus=corpus, decision_id=decision_id,
        source_key=source_key)
    authority_ids = generic_positive
    if authority_kind != "generic":
        authority_ids = source_positive
    target_hits = _target_hit_ids(corpus, authority_ids, target)
    candidates = _ranked_lock_candidates(
        target=target, positive_ids=authority_ids,
        corpus=corpus, gram_cache=gram_cache)
    if target_hits:
        target_chars = set(_name_key(_target_core(target)))
        candidates = [
            row for row in candidates
            if target_chars & set(_name_key(str(row.get("form") or "")))
        ]
    if authority_kind == "generic":
        specific_target_keys = {
            _name_key(_target_core(decision.get("translation")))
            for decision in (context.get("decisions") or {}).values()
            if _strictly_more_specific_source(
                source_key, str(decision.get("source_key") or ""))
        }
        candidates = [
            row for row in candidates
            if _name_key(str(row.get("form") or "")) not in specific_target_keys
        ]
    pending_rows, legitimate_variant = _pending_form_evidence(
        pending_entries=pending_entries, corpus=corpus, target=target,
        candidates=candidates, gram_cache=gram_cache,
        positive_ids=authority_ids)
    classification, reason = _classify_lock_authority(
        target=target, target_hits=len(target_hits),
        total_positive=len(authority_ids), candidates=candidates,
        legitimate_variant=legitimate_variant)
    if classification == "legitimate_variant":
        target_chars = set(_name_key(_target_core(target)))
        candidates = [
            row for row in candidates
            if len(target_chars & set(_name_key(str(row.get("form") or "")))) >= 2
        ]
    return {
        "authority_kind": authority_kind,
        "decision_id": int(decision_id),
        "source_key": source_key,
        "target": _target_core(target),
        "classification": classification,
        "reason": reason,
        "pending_decision_id": int(pending_decision_id or 0),
        "pending_reason": pending_reason,
        "pending_chunks": [str(e.get("chunk_id") or "") for e in pending_entries],
        "pending_rows": pending_rows,
        "source_positive_chunks": len(source_positive),
        "authority_chunks": len(authority_ids),
        "target_hits": len(target_hits),
        "target_hit_chunks": target_hits,
        "candidates": candidates[:12],
    }


def build_lock_authority_corpus(
    context: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, set]]:
    """Build the COMPLETED final cache used by lock-acquisition checks."""
    corpus = _completed_corpus(context)
    gram_cache = {
        chunk_id: _cjk_ngrams(row.get("final") or "")
        for chunk_id, row in corpus.items()
    }
    return corpus, gram_cache


def evaluate_decision_lock_authority(
    *,
    decision_id: int,
    source_key: str,
    target: str,
    context: Dict[str, Any],
    corpus: Dict[str, Dict[str, str]],
    gram_cache: Dict[str, set],
    proposed_chunk_id: str = "",
    proposed_text: str = "",
    low_corpus_threshold: int = 1,
) -> Dict[str, Any]:
    """Decide whether a Decision target may acquire/reinforce a lock.

    The current chunk is treated as COMPLETED with ``proposed_text`` so a
    genuine one-chunk entity can still lock its first authoritative form.  A
    book with several source-positive chunks must independently support the
    target; one local repair cannot manufacture a majority by itself.
    """
    effective = dict(corpus)
    effective_grams = dict(gram_cache)
    proposed_chunk_id = str(proposed_chunk_id or "")
    if proposed_chunk_id:
        task = (context.get("tasks") or {}).get(proposed_chunk_id) or {}
        chapter = str(task.get("chapter_id") or proposed_chunk_id[:4])
        proposed_final = str(proposed_text or "")
        effective[proposed_chunk_id] = {
            "chapter": chapter,
            "source": str(task.get("text_content") or ""),
            "final": proposed_final,
        }
        # Candidate ranking must see the bytes that the mutating path is about
        # to write.  Otherwise a proposed majority can be counted in
        # ``target_hits`` while its form is absent from the positive-gram
        # statistics, producing a contradictory decision from the same input.
        effective_grams[proposed_chunk_id] = _cjk_ngrams(proposed_final)
    source_positive, generic_positive = _authority_ids_for_decision(
        context=context, corpus=effective, decision_id=int(decision_id),
        source_key=source_key)
    authority_ids = generic_positive
    if not authority_ids and not source_positive:
        authority_ids = []
    target_hits = _target_hit_ids(effective, authority_ids, target)
    book_authority_ids = [
        chunk_id for chunk_id in authority_ids
        if chunk_id != proposed_chunk_id
    ]
    book_target_hits = _target_hit_ids(effective, book_authority_ids, target)
    target_key = _name_key(_target_core(target))
    legitimate_variant = any(
        _contains_character_permutation(
            effective[chunk_id].get("final") or "", target)
        for chunk_id in (book_authority_ids or authority_ids)
    )
    candidates = _ranked_lock_candidates(
        target=target, positive_ids=authority_ids,
        corpus=effective, gram_cache=effective_grams)
    classification_target_hits = (
        book_target_hits if book_authority_ids else target_hits)
    classification_authority_total = (
        len(book_authority_ids) if book_authority_ids else len(authority_ids))
    if classification_target_hits:
        target_chars = set(target_key)
        candidates = [
            row for row in candidates
            if target_chars & set(_name_key(str(row.get("form") or "")))
        ]
    total = classification_authority_total
    if total > 0:
        classification, reason = _classify_lock_authority(
            target=target, target_hits=len(classification_target_hits),
            total_positive=total, candidates=candidates,
            legitimate_variant=legitimate_variant)
    else:
        classification = "insufficient_evidence"
        reason = "source_phrase_absent_from_completed_corpus"
    low_corpus = 0 < total <= max(1, int(low_corpus_threshold))
    if (total > 0 and low_corpus and target_hits
            and not book_authority_ids):
        classification = "confirmed"
        reason = "low_corpus_target_confirmed_by_proposed_final"
    if classification == "legitimate_variant":
        target_chars = set(target_key)
        candidates = [
            row for row in candidates
            if len(target_chars & set(_name_key(str(row.get("form") or "")))) >= 2
        ]
    lock_allowed = classification == "confirmed" and bool(target_hits)
    return {
        "decision_id": int(decision_id),
        "source_key": str(source_key),
        "target": _target_core(target),
        "status": classification,
        "reason": reason,
        "lock_allowed": lock_allowed,
        "low_corpus": low_corpus,
        "source_positive_chunks": len(source_positive),
        "authority_chunks": total,
        "target_hits": len(target_hits),
        "target_hit_chunks": target_hits,
        "book_authority_chunks": len(book_authority_ids),
        "book_target_hits": len(book_target_hits),
        "book_target_hit_chunks": book_target_hits,
        "proposed_target_hit": bool(
            proposed_chunk_id and proposed_chunk_id in target_hits),
        "proposed_chunk_id": proposed_chunk_id,
        "candidates": candidates[:12],
    }


def audit_locked_target_authorities(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    include_marked_repair_sweep: bool = False,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit locked-target blocking rows without changing project state."""
    root = Path(project_root)
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "locked_target_authority_audit.json"
    mpath = Path(md_out) if md_out else out_dir / "locked_target_authority_audit.md"
    result: Dict[str, Any] = {
        "schema_version": LOCKED_TARGET_AUDIT_SCHEMA_VERSION,
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
    locked_entries = [
        entry for entry in snapshot.get("entries") or []
        if entry.get("reason") in LOCK_AUTHORITY_REASONS
    ]
    corpus = _completed_corpus(context)
    gram_cache = {
        chunk_id: _cjk_ngrams(row.get("final") or "")
        for chunk_id, row in corpus.items()
    }
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for entry in locked_entries:
        source_key = str(entry.get("decision_source_key") or
                         entry.get("pending_source_key") or "")
        target = str(entry.get("decision_translation") or "")
        decision_id = int(entry.get("decision_id") or 0)
        reason = str(entry.get("reason") or "")
        task = (context.get("tasks") or {}).get(str(entry.get("chunk_id") or ""))
        source_text = str((task or {}).get("text_content") or "")
        if reason == "specific_decision_target_missing":
            covers = _specific_covering_decisions(
                source_text=source_text, generic_source=source_key,
                generic_decision_id=decision_id,
                decisions=context.get("decisions") or {})
            if covers:
                for cover in covers:
                    grouped[(
                        "specific", int(cover["decision_id"]), str(cover["source_key"]),
                        str(cover["target"]), decision_id, reason,
                    )].append(entry)
                continue
        grouped[("generic", decision_id, source_key, target, 0, reason)].append(entry)

    groups: List[Dict[str, Any]] = []
    for key, entries in sorted(grouped.items(), key=lambda item: (
            item[0][0], item[0][1], item[0][2])):
        authority_kind, decision_id, source_key, target, generic_id, reason = key
        groups.append(_build_group(
            authority_kind=authority_kind, decision_id=int(decision_id),
            source_key=str(source_key), target=str(target),
            pending_entries=entries, context=context, corpus=corpus,
            gram_cache=gram_cache, pending_decision_id=int(generic_id),
            pending_reason=str(reason)))

    if include_marked_repair_sweep:
        # A successful repair consumes its pending row, but a *different*
        # source-positive COMPLETED chunk may still carry the same drift with
        # no outbox row (the Shield Wall / ch03_chunk012 case).  Sweep locked
        # Decisions independently of pending only when this repairer left its
        # explicit marker.  Sweeping every locked proper name makes unrelated
        # names look like repair candidates and violates least authority.
        seen_authority = {
            (int(group.get("decision_id") or 0), str(group.get("source_key") or ""))
            for group in groups
        }
        for decision_id, decision in sorted((context.get("decisions") or {}).items()):
            if not int(decision.get("locked") or 0):
                continue
            reason = str(decision.get("reason") or "")
            if "Round 158 confirmed term drift lock:" not in reason:
                continue
            source_key = str(decision.get("source_key") or "")
            target = str(decision.get("translation") or "")
            if ((int(decision_id), source_key) in seen_authority
                    or not source_key.strip() or not target.strip()):
                continue
            group = _build_group(
                authority_kind="specific", decision_id=int(decision_id),
                source_key=source_key, target=target, pending_entries=[],
                context=context, corpus=corpus, gram_cache=gram_cache,
                pending_decision_id=0,
                pending_reason="specific_decision_target_missing_no_pending",
            )
            if int(group.get("source_positive_chunks") or 0) <= 0:
                continue
            if int(group.get("target_hits") or 0) >= int(
                    group.get("authority_chunks") or 0):
                continue
            group["sweep"] = True
            groups.append(group)
        groups.sort(key=lambda group: (
            str(group.get("authority_kind") or ""),
            int(group.get("decision_id") or 0), str(group.get("source_key") or "")))
    by_class = Counter(str(group.get("classification") or "") for group in groups)
    result.update({
        "groups": groups,
        "counts": {
            "pending_rows": len(locked_entries),
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
    lines = ["# 锁定目标权威审计（只读）", "",
             "- 模式: **read_only**",
             "- 生成时间: %s" % result.get("generated_at"),
             "- 统计: `%s`" % (result.get("counts") or {}), ""]
    groups = result.get("groups") or []
    if groups:
        lines.append("## 分组")
        for group in groups:
            lines.append(
                "- `#%s %s → %s` / **%s** / %s / target_hits=%s/%s / pending=%s"
                % (group.get("decision_id"), group.get("source_key"),
                   group.get("target"), group.get("classification"),
                   group.get("reason"), group.get("target_hits"),
                   group.get("authority_chunks"),
                   ",".join(group.get("pending_chunks") or [])))
            candidates = group.get("candidates") or []
            if candidates:
                lines.append("  - 候选: " + ", ".join(
                    "`%s`(pos=%s,neg=%s)" % (
                        row.get("form"), row.get("positive_chunks"),
                        row.get("negative_chunks")) for row in candidates[:6]))
    if result.get("errors"):
        lines.extend(["", "## 错误", ""] + [
            "- `%s`" % error for error in result["errors"]])
    lines.extend(["", "本报告不修改正文、Decision、registry、pending 或 workflow。", ""])
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "LOCKED_TARGET_AUDIT_SCHEMA_VERSION",
    "audit_locked_target_authorities",
    "build_lock_authority_corpus",
    "evaluate_decision_lock_authority",
]
