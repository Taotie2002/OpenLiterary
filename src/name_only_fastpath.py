"""Round 158 Phase 1.5: strict name-only DIRTY classification.

This module is intentionally a dry-run planner.  It never rewrites a final,
critic report, pending row, decision lock, or workflow state.  A chunk becomes a
``candidate`` only when all of the following are true:

* it is DIRTY, has a parseable non-fallback final and no rescue stage;
* its current critic report is calibrated and bound to the current final bytes,
  and every score is above the configured floor;
* applying the existing name-unification maps changes text, but only through
  separator/form rules (token-level drift is held for a later, separately
  validated slice);
* the Decision DB and high-confidence entity registry do not disagree about an
  active source name;
* the chunk's pending outbox row is satisfied by the proposed final (or there
  is no pending row), rather than being blindly cleared.

Even a candidate still needs a real critic re-audit of the proposed bytes in
the later mutating phase.  The predicted text hash is therefore an input to
that audit, never evidence that the audit already happened.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.config import config as _config
from src.core.decision_engine import (
    _NAME_SEP_CHARS,
    apply_name_unification,
    build_name_unification_maps,
    canonical_is_title_only,
    canonical_tokens,
    unify_entity_registry,
)
from src.judgment_layer import sha256_text
from src.locked_target_authority_audit import (
    build_lock_authority_corpus,
    evaluate_decision_lock_authority,
)
from src.pending_backtrack_reconcile import (
    _delete_pending_row_if_unchanged,
    _name_compatible,
    _name_key,
    _phrase_key,
    _source_phrase_spans,
    _target_core,
    body_before_footnotes,
    classify_pending_snapshot,
    load_pending_context,
)
from src.recalibrate import critic_evidence_state
from src.utils.entity_stopwords import (
    is_likely_common_word,
    is_plausible_name_variant,
)


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


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


def _calibrated_backends() -> set:
    gate_cfg = _config.get("pipeline.final_gate", {}) or {}
    raw = gate_cfg.get("calibrated_backends") or ["jev"]
    if isinstance(raw, str):
        raw = [raw]
    return {str(x) for x in raw}


def _score_floor() -> Dict[str, float]:
    floors = dict(_config.critic_thresholds or {})
    floors.setdefault("entity_consistency", 6.0)
    return {str(k): float(v) for k, v in floors.items()}


def _scores_below_floor(report: Dict[str, Any]) -> List[str]:
    scores = (report or {}).get("scores") or {}
    out: List[str] = []
    for dim, floor in _score_floor().items():
        try:
            value = float(scores.get(dim))
        except (TypeError, ValueError):
            out.append(f"{dim}:missing")
            continue
        if value < floor:
            out.append(f"{dim}:{value:g}<{floor:g}")
    return out


def _residual_authority_forms(text: str, authorities: Iterable[str]) -> List[dict]:
    """Find registered canonical names with a one-character CJK drift.

    The critic's per-entity check is source-scoped: a translator footnote may
    mention ``贝尼·杰瑟里特`` even when the source chunk does not, and an LLM
    score can therefore miss a wrong ``贝尼杰赛里特`` spelled in that note.
    Name-only completion is a state-changing fast path, so it additionally
    requires every active canonical form to be free of a mechanically
    plausible one-character drift.

    Separator folding deliberately matches both ``贝尼·杰瑟里特`` and
    ``贝尼杰赛里特``.  Only canonicals with at least two canonical tokens and
    four CJK characters participate; this keeps short CJK words out of the
    no-space scan, where the false-positive rate would be unacceptable.
    """
    text_key = _name_key(text)
    if not text_key:
        return []
    findings: List[dict] = []
    seen = set()
    for authority in authorities:
        canonical = str(authority or "").strip()
        if not canonical:
            continue
        base = re.sub(r"（[^）]*）|\([^)]*\)", "", canonical).strip()
        if len(canonical_tokens(base)) < 2:
            continue
        canonical_key = _name_key(base)
        if (len(canonical_key) < 4
                or not re.fullmatch(r"[\u4e00-\u9fff]+", canonical_key)):
            continue
        length = len(canonical_key)
        for start in range(0, max(0, len(text_key) - length + 1)):
            found_key = text_key[start:start + length]
            if found_key == canonical_key:
                continue
            if not re.fullmatch(r"[\u4e00-\u9fff]+", found_key):
                continue
            if sum(a != b for a, b in zip(canonical_key, found_key)) != 1:
                continue
            plausible, _reason = is_plausible_name_variant(
                canonical_key, found_key)
            if not plausible:
                continue
            key = (canonical, found_key)
            if key in seen:
                continue
            seen.add(key)
            findings.append({
                "canonical": canonical,
                "found": found_key,
                "kind": "one_char_cjk_drift",
            })
            if len(findings) >= 20:
                return findings
    return findings


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_fsync(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _open_decision_write_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def _decision_index(context: Dict[str, Any]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for decision in (context.get("decisions") or {}).values():
        key = _phrase_key(decision.get("source_key") or "")
        if key:
            out.setdefault(key, []).append(decision)
    return out


def _active_authority_conflicts(
    source_text: str,
    context: Dict[str, Any],
    decision_index: Dict[str, List[dict]],
    targets: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """Find authority conflicts for the names this chunk would actually change.

    A polluted registry row for an unrelated character must not veto a local
    separator repair.  Scope the check to the target names touched by the
    deterministic mapping.
    """
    conflicts: List[Dict[str, str]] = []
    sources_by_target = context.get("registry_sources_by_target") or {}
    wanted = {_name_key(t) for t in (targets or []) if _name_key(t)}
    if not wanted:
        return conflicts
    source_keys = set()
    for target_key in wanted:
        source_keys.update(sources_by_target.get(target_key) or set())
    by_source = context.get("registry_by_source") or {}
    trusted_by_source = context.get("registry_trusted_by_source") or {}
    for source_key in sorted(source_keys):
        canonicals = by_source.get(source_key) or set()
        if not canonicals:
            continue
        trusted = trusted_by_source.get(source_key) or set()
        trusted_keys = {_name_key(c) for c in trusted if _name_key(c)}
        if len(trusted_keys) > 1:
            conflicts.append({
                "source_key": source_key,
                "reason": "cross_person_alias_conflict",
                "registry_targets": " | ".join(sorted(trusted)),
                "decision_targets": "",
            })
            continue
        trusted_target = next(iter(trusted), "")
        if not trusted_target and len(
                {_name_key(c) for c in canonicals if _name_key(c)}) > 1:
            conflicts.append({
                "source_key": source_key,
                "reason": "untrusted_registry_alias_ambiguity",
                "registry_targets": " | ".join(sorted(canonicals)),
                "decision_targets": "",
            })
            continue
        decisions = decision_index.get(source_key) or []
        decision_targets = [str(d.get("translation") or "") for d in decisions]
        compatible = any(
            _name_compatible(str(target), d_target)
            for target in wanted for d_target in decision_targets if d_target)
        if decision_targets and not compatible:
            conflicts.append({
                "source_key": source_key,
                "reason": "decision_registry_target_conflict",
                "registry_targets": trusted_target or " | ".join(sorted(canonicals)),
                "decision_targets": " | ".join(sorted(decision_targets)),
            })
    return conflicts


def _build_maps(
    root: Path,
    decision_db: Path,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build maps on a disposable Decision DB copy (production DB is read-only)."""
    fd, tmp_name = tempfile.mkstemp(prefix="sagatrans_r158_nu_", suffix=".sqlite")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        source_con = sqlite3.connect(str(decision_db))
        copy_con = sqlite3.connect(str(tmp_path))
        try:
            source_con.backup(copy_con)
            unify_entity_registry(copy_con, dry_run=False)
            corpus = "\n".join(str(c.get("text") or "") for c in chunks)
            preferred = (_config.get("pipeline.name_unification.preferred_forms", {})
                         or {})
            maps = build_name_unification_maps(
                copy_con, corpus, preferred_forms=preferred)
        finally:
            copy_con.close()
            source_con.close()
        return maps
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _changed_spans(before: str, after: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        spans.append({
            "op": tag,
            "start": i1,
            "end": i2,
            "before": before[i1:i2],
            "after": after[j1:j2],
        })
        if len(spans) >= 80:
            break
    return spans


def _mapping_targets(maps: Dict[str, Any], stats: Dict[str, Any]) -> List[str]:
    """Targets touched by the actual deterministic changes in this chunk."""
    targets: List[str] = [str(t) for t in (stats.get("targets") or []) if t]
    for text in stats.get("examples") or []:
        if "→" not in str(text):
            continue
        target = str(text).split("→", 1)[1].split("×", 1)[0]
        if target:
            targets.append(target)
    return list(dict.fromkeys(targets))


def _lockable_decisions(
    source_text: str,
    maps: Dict[str, Any],
    stats: Dict[str, Any],
    context: Dict[str, Any],
    decision_index: Dict[str, List[dict]],
    *,
    chunk_id: str = "",
    proposed_text: str = "",
    authority_corpus: Optional[Dict[str, Dict[str, str]]] = None,
    authority_gram_cache: Optional[Dict[str, set]] = None,
) -> List[dict]:
    """Decisions touched by this name change and supported by chunk/registry evidence."""
    targets = _mapping_targets(maps, stats)
    target_keys = {_name_key(t) for t in targets if _name_key(t)}
    registry_sources = context.get("registry_sources_by_target") or {}
    evidence_sources = set()
    for target_key in target_keys:
        evidence_sources.update(registry_sources.get(target_key) or set())
    source_key = _phrase_key(source_text)
    corpus = authority_corpus
    gram_cache = authority_gram_cache
    if corpus is None or gram_cache is None:
        corpus, gram_cache = build_lock_authority_corpus(context)
    out: List[dict] = []
    seen = set()
    for decision in (context.get("decisions") or {}).values():
        did = int(decision.get("id") or 0)
        if not did or did in seen:
            continue
        translation = str(decision.get("translation") or "")
        if not translation or not any(
                _name_compatible(translation, target) for target in targets):
            continue
        dec_source = _phrase_key(decision.get("source_key") or "")
        if dec_source not in evidence_sources and dec_source not in source_key:
            continue
        seen.add(did)
        authority = evaluate_decision_lock_authority(
            decision_id=did,
            source_key=str(decision.get("source_key") or ""),
            target=translation,
            context=context,
            corpus=corpus,
            gram_cache=gram_cache,
            proposed_chunk_id=chunk_id,
            proposed_text=proposed_text,
        )
        out.append({
            "id": did,
            "source_key": str(decision.get("source_key") or ""),
            "translation": translation,
            "locked": int(decision.get("locked") or 0),
            "authority": authority,
            "lock_allowed": bool(authority.get("lock_allowed")),
        })
    return out


def _validate_reauthored_report(
    report: Optional[dict],
    *,
    proposed_text: str,
) -> List[str]:
    reasons: List[str] = []
    report = report or {}
    backend = str(report.get("_backend_path") or "missing")
    if backend not in _calibrated_backends():
        reasons.append(f"critic_backend_uncalibrated:{backend}")
    scores = report.get("scores") or {}
    if not isinstance(scores, dict) or not scores:
        reasons.append("critic_scores_missing")
    low = _scores_below_floor(report)
    if low:
        reasons.extend(f"critic_floor:{item}" for item in low)
    verdict = report.get("_typesafe_verdict") or {}
    decision = str(verdict.get("decision") or "")
    if decision and decision != "PASS":
        reasons.append(f"critic_verdict:{decision}")
    entity_notes = report.get("entity_consistency_notes") or []
    unresolved_notes = [
        note for note in entity_notes
        if not (isinstance(note, dict)
                and _entity_note_satisfied(note, proposed_text))
    ]
    if unresolved_notes:
        reasons.append(f"entity_consistency_notes:{len(unresolved_notes)}")
    if report.get("_entity_check_error"):
        reasons.append(f"entity_check_error:{report['_entity_check_error']}")
    if not proposed_text.strip():
        reasons.append("proposed_text_empty")
    return reasons


def _entity_note_satisfied(note: Dict[str, Any], text: str) -> bool:
    """Mirror the pipeline's allowed-form tolerance for entity notes.

    A translated chunk may legitimately keep the source name in a parenthetical
    gloss or translator footnote while using the canonical Chinese form in the
    prose.  The pipeline already suppresses those notes when the canonical (or
    a canonical token) is present; the name-only fast path must use the same
    rule or it would hold otherwise-clean chunks DIRTY forever.
    """
    if not isinstance(note, dict) or not text:
        return False
    entity = note.get("entity") or note.get("expected") or ""
    if entity:
        if entity in text:
            return True
        for token in canonical_tokens(entity):
            if len(token) >= 2 and token in text:
                return True
    found = note.get("found")
    if (isinstance(found, str) and found
            and re.search(r"[\u4e00-\u9fff]", found) and found in text):
        return True
    return False


def _pending_rows_for_chunk(context: Dict[str, Any], chunk_id: str,
                            proposed_text: str) -> List[dict]:
    classified = classify_pending_snapshot(
        context, final_overrides={str(chunk_id): proposed_text})
    return [entry for entry in (classified.get("entries") or [])
            if str(entry.get("chunk_id") or "") == str(chunk_id)]


def _default_critic_fn(
    record: Dict[str, Any],
    source_text: str,
    raw_text: str,
    proposed_text: str,
    pipeline: Any,
) -> Dict[str, Any]:
    cid = str(record["chunk_id"])
    from src.splitters import _normalize_scores
    registry = pipeline._build_entity_registry_for_rewriter(source_text)
    variants = pipeline._build_entity_variants_for_critic(source_text)
    report = pipeline.critic_agent.process_chunk(
        cid,
        source_text,
        raw_text,
        proposed_text,
        pipeline.style_guide,
        entity_registry=registry,
        entity_variants=variants,
        allow_legacy_fallback=False,
    )
    if isinstance(report, dict) and "scores" in report:
        report["scores"] = _normalize_scores(report.get("scores"))
    if isinstance(report, dict):
        try:
            notes = pipeline._check_entity_consistency(
                cid, proposed_text, source_text)
            if notes:
                report["entity_consistency_notes"] = notes
        except Exception as exc:  # noqa: BLE001 - fail closed on verifier error
            report["_entity_check_error"] = f"{type(exc).__name__}: {exc}"
    return report


def _load_dirty_items(
    workflow_path: Path,
    out_dir: Path,
    chapter_id: Optional[str],
) -> List[Dict[str, Any]]:
    wcon = sqlite3.connect(f"file:{workflow_path.resolve()}?mode=ro", uri=True)
    wcon.row_factory = sqlite3.Row
    try:
        sql = ("SELECT chunk_id, chapter_id, text_content, state, rescue_stage "
               "FROM chunk_tasks WHERE state='DIRTY'")
        params: tuple = ()
        if chapter_id:
            sql += " AND chapter_id=?"
            params = (chapter_id,)
        tasks = [dict(r) for r in wcon.execute(sql + " ORDER BY chunk_id", params)]
    finally:
        wcon.close()
    loaded: List[Dict[str, Any]] = []
    for task in tasks:
        cid = str(task["chunk_id"])
        ch = str(task["chapter_id"])
        final_path = out_dir / ch / f"{cid}_final.json"
        final = _read_json(final_path)
        loaded.append({
            "task": task,
            "final_path": final_path,
            "final": final,
            "text": str((final or {}).get("text") or ""),
            "critic": _read_json(out_dir / ch / f"{cid}_critic_report.json"),
        })
    return loaded


def _load_completed_items(
    workflow_path: Path,
    out_dir: Path,
    chapter_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Load COMPLETED chunks for the completed-name repair planner.

    This is deliberately separate from ``_load_dirty_items``: the completed
    repair path must never broaden the DIRTY fast path's state contract.
    """
    wcon = sqlite3.connect(f"file:{workflow_path.resolve()}?mode=ro", uri=True)
    wcon.row_factory = sqlite3.Row
    try:
        sql = ("SELECT chunk_id, chapter_id, text_content, state, rescue_stage "
               "FROM chunk_tasks WHERE state='COMPLETED'")
        params: tuple = ()
        if chapter_id:
            sql += " AND chapter_id=?"
            params = (chapter_id,)
        tasks = [dict(r) for r in wcon.execute(sql + " ORDER BY chunk_id", params)]
    finally:
        wcon.close()
    loaded: List[Dict[str, Any]] = []
    for task in tasks:
        cid = str(task["chunk_id"])
        ch = str(task["chapter_id"])
        final_path = out_dir / ch / f"{cid}_final.json"
        final = _read_json(final_path)
        loaded.append({
            "task": task,
            "final_path": final_path,
            "final": final,
            "text": str((final or {}).get("text") or ""),
            "critic": _read_json(out_dir / ch / f"{cid}_critic_report.json"),
        })
    return loaded


def _canonical_name_chars(canonical: str, min_chars: int) -> List[str]:
    """Return the CJK character sequence for a pure-Chinese canonical name.

    Mixed Latin/Chinese targets are intentionally excluded.  The completed
    repair path is a deterministic one-character CJK correction, not a general
    term rewriter.
    """
    base = re.sub(r"（[^）]*）|\([^)]*\)", "", str(canonical or "")).strip()
    tokens = canonical_tokens(base)
    chars = [ch for token in tokens for ch in token]
    if len(chars) < max(1, int(min_chars)):
        return []
    if any(not re.fullmatch(r"[\u4e00-\u9fff]", ch) for ch in chars):
        return []
    if len("".join(tokens)) != len(chars):
        return []
    return chars


def _one_char_name_drift_matches(
    text: str,
    canonical: str,
    *,
    min_chars: int = 4,
) -> List[Dict[str, Any]]:
    """Find names that differ from ``canonical`` by exactly one CJK character.

    The match keeps the canonical token boundaries but permits the project's
    legal name separators (``·``, ``-`` etc.) between tokens.  It never matches
    the canonical form itself, and it requires equal normalized length plus a
    Hamming distance of exactly one; a broader fuzzy match is not admissible
    for a text-rewriting operation.
    """
    target_chars = _canonical_name_chars(canonical, min_chars)
    if not target_chars or not text:
        return []
    tokens = canonical_tokens(re.sub(
        r"（[^）]*）|\([^)]*\)", "", str(canonical or "")).strip())
    sep = r"[\s" + re.escape(_NAME_SEP_CHARS) + r"]*"
    patterns: List[str] = []
    for changed in range(len(target_chars)):
        pieces: List[str] = []
        char_index = 0
        for token_index, token in enumerate(tokens):
            if token_index:
                pieces.append(sep)
            for char in token:
                if char_index == changed:
                    pieces.append(r"[\u4e00-\u9fff]")
                else:
                    pieces.append(re.escape(char))
                char_index += 1
        patterns.append("(?:%s)" % "".join(pieces))
    matcher = re.compile("|".join(patterns))
    target_key = _name_key(canonical)
    out: List[Dict[str, Any]] = []
    seen = set()
    for match in matcher.finditer(text):
        form = match.group(0)
        key = _name_key(form)
        if len(key) != len(target_key) or key == target_key:
            continue
        if sum(a != b for a, b in zip(key, target_key)) != 1:
            continue
        signature = (match.start(), match.end(), key)
        if signature in seen:
            continue
        seen.add(signature)
        out.append({
            "start": match.start(),
            "end": match.end(),
            "form": form,
            "normalized": key,
        })
    return out


def _replace_name_drift_spans(
    text: str,
    matches: Iterable[Dict[str, Any]],
    canonical: str,
) -> str:
    after = str(text)
    for match in sorted(matches, key=lambda item: int(item["start"]), reverse=True):
        start = int(match["start"])
        end = int(match["end"])
        after = after[:start] + str(canonical) + after[end:]
    return after


def _active_name_owners(context: Dict[str, Any], source_key: str) -> List[Dict[str, Any]]:
    owners = (context.get("registry_form_owners") or {}).get(
        _phrase_key(source_key)) or []
    unique: Dict[str, Dict[str, Any]] = {}
    for owner in owners:
        if int(owner.get("pending_review") or 0):
            continue
        key = _name_key(owner.get("canonical") or "")
        if key:
            unique.setdefault(key, owner)
    return list(unique.values())


def _protected_active_name_keys(
    context: Dict[str, Any],
    target_key: str,
) -> set:
    """Active canonical/variant keys that may not be overwritten by this repair."""
    protected = set()
    for owners in (context.get("registry_form_owners") or {}).values():
        for owner in owners:
            if int(owner.get("pending_review") or 0):
                continue
            key = _name_key(owner.get("canonical") or "")
            if not key or key == target_key:
                continue
            protected.add(key)
            for variant in owner.get("variants") or []:
                variant_key = _name_key(variant)
                if variant_key:
                    protected.add(variant_key)
    return protected


def _source_key_occurrence_count(text: str, source_key: str) -> int:
    """Count a simple ASCII source term without matching inside another word.

    EPUB source paragraphs contain soft line wraps.  The source key is a
    semantic phrase, not a literal byte string, so internal whitespace is
    normalised exactly as it is in ``_confirmed_term_source_count``.
    """
    key = str(source_key or "").strip()
    if (len(key) < 2
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9'’ .-]*", key)):
        return 0
    parts = [part for part in key.split() if part]
    if not parts:
        return 0
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + r"\s+".join(
            re.escape(part) for part in parts) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return len(pattern.findall(_body_before_footnotes(text)))


_TARGET_MIGRATION_MARKERS = (
    re.compile(
        r"Round 158 decision target authority repair:\s*([^|]+?)\s*->\s*([^|]+?)(?:\s*\||$)"),
    re.compile(
        r"Round 158 corpus lock authority:\s*([^|]+?)\s*->\s*([^|]+?)(?:\s*\||$)"),
)


def _decision_target_migration_matches(
    item: Dict[str, Any],
    context: Dict[str, Any],
    pending_entry: Optional[dict],
    canonical: str,
) -> Dict[str, Any]:
    """Plan a post-authority target migration for a COMPLETED final.

    ``decision-target-authority-repair`` and the locked-target repairer write a
    machine-readable ``old -> new`` marker into the Decision reason.  A target
    repair can still leave an already COMPLETED chunk containing the old form;
    this helper recognises only that audited migration.  It deliberately does
    not infer an old target from similarity, so ordinary prose cannot opt into
    the broader rewrite path by accident.
    """
    if not pending_entry:
        return {"status": "unavailable", "reason": "no_pending_backtrack"}
    decision_id = pending_entry.get("decision_id")
    try:
        decision = (context.get("decisions") or {}).get(int(decision_id))
    except (TypeError, ValueError):
        decision = None
    if not decision:
        return {"status": "unavailable", "reason": "decision_missing"}
    reason = str(decision.get("reason") or "")
    migrations: List[Tuple[str, str]] = []
    for pattern in _TARGET_MIGRATION_MARKERS:
        migrations.extend(
            (str(old).strip(), str(new).strip())
            for old, new in pattern.findall(reason)
        )
    if not migrations:
        return {"status": "unavailable", "reason": "no_authority_migration_marker"}
    old_raw, new_raw = migrations[-1]
    old_core = _target_core(old_raw)
    new_core = _target_core(new_raw)
    canonical_key = _name_key(canonical)
    new_key = _name_key(new_core)
    old_key = _name_key(old_core)
    if not canonical_key or canonical_key != new_key:
        return {
            "status": "unavailable",
            "reason": "migration_new_target_mismatch",
            "old_target": old_core,
            "new_target": new_core,
        }
    if (not old_key or old_key == new_key or not old_core
            or not re.fullmatch(r"[\u4e00-\u9fff·・－\-]+", old_core)):
        return {
            "status": "unavailable",
            "reason": "migration_old_target_not_mechanical",
            "old_target": old_core,
            "new_target": new_core,
        }
    source_text = str((item.get("task") or {}).get("text_content") or "")
    source_key = str((item.get("task") or {}).get("source_key") or "")
    # The pending row carries the authoritative Decision source key; prefer it
    # over any task-local fixture field.
    source_key = str(pending_entry.get("decision_source_key") or source_key)
    source_spans = _source_phrase_spans(source_text, source_key)
    body = body_before_footnotes(item.get("text") or "")
    spans = [
        (match.start(), match.end(), match.group(0))
        for match in re.finditer(re.escape(old_core), body)
    ]
    if not source_spans:
        return {
            "status": "unavailable",
            "reason": "migration_source_absent",
            "old_target": old_core,
            "new_target": new_core,
        }
    if len(spans) != len(source_spans):
        return {
            "status": "unavailable",
            "reason": "migration_occurrence_mismatch",
            "source_occurrences": len(source_spans),
            "old_target_occurrences": len(spans),
            "old_target": old_core,
            "new_target": new_core,
        }
    protected = _protected_active_name_keys(context, canonical_key)
    if old_key in protected:
        return {
            "status": "unavailable",
            "reason": "migration_old_form_owned_by_active_name",
            "old_target": old_core,
            "new_target": new_core,
        }
    return {
        "status": "eligible",
        "reason": "round158_authority_migration",
        "old_target": old_core,
        "new_target": new_core,
        "source_occurrences": len(source_spans),
        "old_target_occurrences": len(spans),
        "matches": [
            {"start": start, "end": end, "form": form,
             "normalized": _name_key(form)}
            for start, end, form in spans
        ],
    }


def _best_glossed_term_form(form: str, target_key: str) -> str:
    """Trim context accidentally welded to an explicit ``term（source）`` gloss.

    A fixed-width scan can include a preceding verb when the Chinese term is
    shorter than the Decision target (``看见运输机（carryall）``).  The literal
    gloss is still strong provenance, so choose the suffix that is most similar
    to the target instead of replacing the verb together with the term.
    """
    best = form
    best_score = (
        difflib.SequenceMatcher(
            None, form, target_key, autojunk=False).ratio(),
        len(form),
    )
    for offset in range(1, len(form) - 1):
        suffix = form[offset:]
        if is_likely_common_word(suffix):
            continue
        score = (
            difflib.SequenceMatcher(
                None, suffix, target_key, autojunk=False).ratio(),
            len(suffix),
        )
        if score > best_score:
            best = suffix
            best_score = score
    return best


def _pending_source_scoped_term_matches(
    item: Dict[str, Any],
    canonical: str,
    *,
    source_key: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Find a repeated final form whose count matches the source term count.

    This is the missing bridge between a one-character typo repair and a full
    chunk rerun.  A rejected spelling such as ``梅普斯`` may differ from the
    Decision target by two characters, yet the chunk still gives mechanical
    provenance: the source contains ``Mapes`` N times and the body contains
    that one competing CJK form exactly N times.  A second safe form is a
    translator-glossed term such as ``真实感（truthsense）``: every occurrence
    of the candidate is directly bound to the literal source key.

    The helper is read-only and intentionally returns at most one form.  The
    caller still revalidates the proposed bytes with a calibrated critic and
    only then clears pending/locks the Decision.
    """
    source_text = str(item.get("task", {}).get("text_content") or "")
    final_text = str(item.get("text") or "")
    target = _term_target_core(canonical)
    target_key = _name_key(target)
    evidence: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "",
        "target": target,
        "source_key": str(source_key or ""),
        "source_count": 0,
        "matches": [],
        "paired_gloss": False,
    }
    if not source_text.strip() or not final_text.strip():
        evidence["reason"] = "source_or_final_missing"
        return evidence
    if (not target_key or len(target_key) < 2
            or not all(re.fullmatch(r"[\u4e00-\u9fff]", ch)
                       for ch in target_key)):
        evidence["reason"] = "target_not_pure_cjk_or_too_short"
        return evidence
    source_count = _source_key_occurrence_count(source_text, source_key)
    evidence["source_count"] = source_count
    if source_count <= 0:
        evidence["reason"] = "source_term_not_present"
        return evidence
    final_body = _body_before_footnotes(final_text)
    if target_key in _name_key(final_body):
        evidence["reason"] = "target_already_present"
        return evidence

    protected = _protected_active_name_keys(context, target_key)
    candidate_spans: Dict[str, List[Tuple[int, int]]] = {}
    candidate_meta: Dict[str, Dict[str, Any]] = {}
    # A same-width CJK token is the only admissible form here.  Allowing a
    # longer n-gram can include a preceding particle (``的真实感``) and then a
    # replacement would delete grammatical text instead of just the term.
    width = len(target_key)
    for run in re.finditer(r"[\u4e00-\u9fff]+", final_text):
        chars = run.group(0)
        if len(chars) < width:
            continue
        for offset in range(0, len(chars) - width + 1):
            form = chars[offset:offset + width]
            normalized = _name_key(form)
            if (not normalized or normalized == target_key
                    or normalized in protected):
                continue
            if any(normalized in key or key in normalized
                   for key in protected if key):
                continue
            similarity = difflib.SequenceMatcher(
                None, normalized, target_key, autojunk=False).ratio()
            pair_re = re.compile(
                r"(?P<form>" + re.escape(form) + r")"
                + r"""["'”’」』]*\s*[（(]\s*"""
                + re.escape(str(source_key or ""))
                + r"\s*[）)]",
                re.IGNORECASE,
            )
            paired_in_body = len(pair_re.findall(final_body))
            pair_matches = list(pair_re.finditer(final_text))
            total_form = len(re.findall(re.escape(form), final_text))
            paired_gloss = bool(paired_in_body and len(pair_matches) == total_form)
            if paired_gloss:
                trimmed = _best_glossed_term_form(form, target_key)
                if trimmed != form:
                    form = trimmed
                    normalized = _name_key(form)
                    if (not normalized or normalized == target_key
                            or normalized in protected):
                        continue
                    if any(normalized in key or key in normalized
                           for key in protected if key):
                        continue
                    similarity = difflib.SequenceMatcher(
                        None, normalized, target_key, autojunk=False).ratio()
                    pair_re = re.compile(
                        r"(?P<form>" + re.escape(form) + r")"
                        + r"""["'”’」』]*\s*[（(]\s*"""
                        + re.escape(str(source_key or ""))
                        + r"\s*[）)]",
                        re.IGNORECASE,
                    )
                    paired_in_body = len(pair_re.findall(final_body))
                    pair_matches = list(pair_re.finditer(final_text))
                    total_form = len(pair_matches)
                    paired_gloss = bool(paired_in_body and total_form)
            if not paired_gloss:
                positional = sum(
                    left == right
                    for left, right in zip(normalized, target_key))
                if positional < 1:
                    continue
                if normalized[-1] != target_key[-1]:
                    continue
                if total_form != source_count:
                    continue
                # A one-position match (usually the shared final character) is
                # admissible only when the source term is repeated.  A one-off
                # common phrase such as ``的能力`` must fall through to the
                # stricter legacy paths.
                if positional < 2 and source_count < 2:
                    continue
                if is_likely_common_word(form):
                    continue
            term_spans: List[Tuple[int, int]] = []
            if paired_gloss:
                all_form_matches = list(re.finditer(re.escape(form), final_text))
                if len(all_form_matches) == source_count:
                    # The gloss proves the term identity; an exact occurrence
                    # count matching the English source count safely extends the
                    # replacement to unglossed repetitions in the same chunk.
                    term_spans = [match.span() for match in all_form_matches]
                    total_form = source_count
                else:
                    term_spans = [match.span("form") for match in pair_matches]
                    total_form = len(pair_matches)
            else:
                all_form_matches = list(re.finditer(re.escape(form), final_text))
                term_spans = [match.span() for match in all_form_matches]
            if normalized not in candidate_meta:
                candidate_meta[normalized] = {
                    "form": form,
                    "similarity": round(similarity, 4),
                    "total_form": total_form,
                    "paired_gloss": paired_gloss,
                }
            candidate_spans.setdefault(normalized, [])
            candidate_spans[normalized].extend(term_spans)
            candidate_spans[normalized] = sorted(
                set(candidate_spans[normalized]))
    # A literal ``form（source_key）`` gloss is exact provenance and outranks a
    # count-only neighbour that merely happens to share the target's last CJK
    # character.  If several glossed forms remain, the caller still rejects.
    paired_keys = {
        normalized for normalized, meta in candidate_meta.items()
        if meta.get("paired_gloss")
    }
    if paired_keys:
        candidate_meta = {
            normalized: meta for normalized, meta in candidate_meta.items()
            if normalized in paired_keys
        }
        candidate_spans = {
            normalized: spans for normalized, spans in candidate_spans.items()
            if normalized in paired_keys
        }
    if not candidate_meta:
        evidence["reason"] = "no_counted_term_variant"
        return evidence
    if len(candidate_meta) != 1:
        evidence["reason"] = "multiple_counted_term_variants"
        evidence["candidates"] = sorted(candidate_meta.values(),
                                        key=lambda row: row["form"])
        return evidence
    normalized, meta = next(iter(candidate_meta.items()))
    spans = candidate_spans.get(normalized) or []
    evidence.update(
        status="eligible",
        reason="",
        paired_gloss=bool(meta["paired_gloss"]),
        matches=[{
            "start": start,
            "end": end,
            "form": final_text[start:end],
            "normalized": normalized,
            "source_count": source_count,
            "similarity": meta["similarity"],
            "paired_gloss": bool(meta["paired_gloss"]),
        } for start, end in spans],
    )
    return evidence


def _registered_alias_drift_keys(
    *,
    source_key: str,
    canonical: str,
    context: Dict[str, Any],
    min_chars: int = 3,
) -> set:
    """Return exact pending registry aliases one CJK character from canonical."""
    target_key = _name_key(canonical)
    if not target_key or not _canonical_name_chars(
            canonical, min(2, max(1, int(min_chars)))):
        return set()
    owners = (context.get("registry_form_owners") or {}).get(
        _phrase_key(source_key)) or []
    protected = _protected_active_name_keys(context, target_key)
    wrong_keys = set()
    for owner in owners:
        candidates = []
        if int(owner.get("pending_review") or 0):
            candidates.append(str(owner.get("canonical") or "").strip())
        candidates.extend(str(value).strip()
                          for value in owner.get("quarantined") or [])
        for candidate in candidates:
            candidate_key = _name_key(candidate)
            if not candidate_key or candidate_key in protected:
                continue
            if len(candidate_key) != len(target_key) or candidate_key == target_key:
                continue
            if sum(a != b for a, b in zip(candidate_key, target_key)) != 1:
                continue
            if not all(re.fullmatch(r"[\u4e00-\u9fff]", ch)
                       for ch in candidate_key):
                continue
            wrong_keys.add(candidate_key)
    return wrong_keys


def _registered_alias_drift_matches(
    text: str,
    *,
    source_key: str,
    canonical: str,
    context: Dict[str, Any],
    min_chars: int = 3,
) -> List[Dict[str, Any]]:
    """Find exact pending registry aliases that differ from active canonical.

    A 3-character canonical name is too short for unrestricted fuzzy scanning:
    nearby names such as ``厄拉科斯`` contain many one-character windows.  This
    helper therefore accepts only an exact alias already attached to the same
    source key in the registry, owned by a non-active row, and mechanically one
    CJK character away from the single active canonical.
    """
    wrong_keys = _registered_alias_drift_keys(
        source_key=source_key,
        canonical=canonical,
        context=context,
        min_chars=min_chars,
    )
    if not wrong_keys:
        return []
    return [
        match for match in _one_char_name_drift_matches(
            text, canonical, min_chars=min_chars)
        if str(match.get("normalized") or "") in wrong_keys
    ]


def _term_target_core(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    core = re.sub(r"（[^）]*）|\([^)]*\)", "", raw).strip()
    return core or raw


def _body_before_footnotes(text: str) -> str:
    """Drop definition-style footnote lines from provenance checks.

    Raw and final may both contain an exact target only in a translator note.
    A note is evidence about the term, not evidence that the prose once used
    the target.  Definition-style notes are therefore outside this repair's
    witness region.
    """
    return body_before_footnotes(text)


def _literal_name_spans(text: str, canonical: str) -> List[Tuple[int, int]]:
    """Locate a Decision target without separator/case normalization drift."""
    core = _term_target_core(canonical)
    tokens = canonical_tokens(core)
    if not core or not tokens:
        return []
    sep = r"[\s" + re.escape(_NAME_SEP_CHARS) + r"]*"
    pattern = re.compile(sep.join(re.escape(token) for token in tokens))
    target_key = _name_key(core)
    return [
        match.span() for match in pattern.finditer(str(text or ""))
        if _name_key(match.group(0)) == target_key
    ]


def _mapped_range_through_diff(
    raw_text: str,
    final_text: str,
    start: int,
    end: int,
) -> Optional[Tuple[int, int, str]]:
    """Map one raw span to final without accepting an edit outside that span."""
    matcher = difflib.SequenceMatcher(
        None, raw_text, final_text, autojunk=False)
    pieces: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlap_start = max(start, i1)
        overlap_end = min(end, i2)
        if overlap_start >= overlap_end:
            # Insertions have zero width in raw; include only an insertion that
            # is wholly inside the target, never text inserted at its boundary.
            if tag == "insert" and start < i1 < end:
                pieces.append((j1, j2))
            continue
        if tag == "equal":
            pieces.append((
                j1 + (overlap_start - i1),
                j1 + (overlap_end - i1),
            ))
            continue
        if tag == "replace":
            # A replacement that also rewrites surrounding prose cannot tell us
            # which final characters belong to the name.
            if i1 < start or i2 > end:
                return None
            pieces.append((j1, j2))
            continue
        if tag == "delete":
            continue
        return None
    if not pieces:
        return None
    pieces.sort()
    final_start = pieces[0][0]
    final_end = pieces[-1][1]
    for left, right in zip(pieces, pieces[1:]):
        if left[1] != right[0]:
            return None
    candidate = final_text[final_start:final_end]
    return final_start, final_end, candidate


def _raw_provenance_name_drift_matches(
    item: Dict[str, Any],
    canonical: str,
    *,
    min_chars: int = 3,
) -> Dict[str, Any]:
    """Find a final rewrite drift that raw states to undo.

    This is deliberately narrower than a general name repair.  It requires the
    same chunk's raw artifact to contain the exact Decision target, the final
    to have lost it, and a high raw/final similarity so a stale intermediate
    cannot authorize a rewrite.  The final replacement must be a unique, near
    target CJK form that maps to the raw target through a bounded diff.
    """
    final_text = str(item.get("text") or "")
    final_path = Path(item.get("final_path") or "")
    raw_text = ""
    raw_payload = _read_json(final_path.with_name(
        "%s_raw.json" % str(item.get("task", {}).get("chunk_id") or "")))
    if raw_payload:
        raw_text = str(raw_payload.get("text") or "")
    target = _term_target_core(canonical)
    target_chars = _canonical_name_chars(
        target, min(3, max(1, int(min_chars))))
    evidence: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "",
        "target": target,
        "raw_path": str(final_path.with_name(
            "%s_raw.json" % str(item.get("task", {}).get("chunk_id") or ""))),
        "raw_final_similarity": None,
        "matches": [],
    }
    if not raw_text.strip():
        evidence["reason"] = "raw_missing"
        return evidence
    if not target_chars or len(target_chars) < 3:
        evidence["reason"] = "target_not_pure_cjk_or_too_short"
        return evidence
    if not final_text.strip():
        evidence["reason"] = "final_missing"
        return evidence
    raw_body = _body_before_footnotes(raw_text)
    final_body = _body_before_footnotes(final_text)
    target_key = _name_key(target)
    if target_key in _name_key(final_body):
        evidence["reason"] = "target_already_present"
        return evidence
    target_spans = _literal_name_spans(raw_body, target)
    if not target_spans:
        evidence["reason"] = "target_not_in_raw_body"
        return evidence
    raw_final_similarity = difflib.SequenceMatcher(
        None, raw_text, final_text, autojunk=False).ratio()
    evidence["raw_final_similarity"] = round(raw_final_similarity, 4)
    if raw_final_similarity < 0.80:
        evidence["reason"] = "raw_final_drift_too_large"
        return evidence

    candidates: List[Dict[str, Any]] = []
    for start, end in target_spans:
        mapped = _mapped_range_through_diff(raw_text, final_body, start, end)
        if mapped is None:
            continue
        final_start, final_end, candidate = mapped
        candidate_key = _name_key(candidate)
        if not candidate_key or candidate_key == target_key:
            continue
        if candidate_key in target_key or target_key in candidate_key:
            # Suffix-only common nouns ("医生") and target fragments are not
            # repairs; the candidate must be a substitution of the name form.
            if candidate_key in target_key:
                continue
        if len(candidate_key) > len(target_key) + 2:
            continue
        if not all(re.fullmatch(r"[\u4e00-\u9fff]", ch)
                   for ch in candidate_key):
            continue
        if candidate_key in _name_key(raw_body):
            continue
        similarity = difflib.SequenceMatcher(
            None, candidate_key, target_key, autojunk=False).ratio()
        if similarity < 0.55:
            continue
        common = difflib.SequenceMatcher(
            None, candidate_key, target_key, autojunk=False).find_longest_match(
                0, len(candidate_key), 0, len(target_key))
        if common.size < 2:
            continue
        if len(_literal_name_spans(final_body, candidate)) != 1:
            continue
        candidates.append({
            "start": final_start,
            "end": final_end,
            "form": candidate,
            "normalized": candidate_key,
            "similarity": round(similarity, 4),
            "raw_target": target,
        })
    forms = {str(match["normalized"]) for match in candidates}
    if not candidates:
        evidence["reason"] = "no_bounded_raw_to_final_name_drift"
        return evidence
    if len(forms) != 1:
        evidence["reason"] = "multiple_raw_provenance_drift_forms"
        evidence["matches"] = candidates
        return evidence
    evidence.update(
        status="eligible",
        reason="",
        matches=candidates,
        wrong_forms=sorted({str(match["form"]) for match in candidates}),
    )
    return evidence


def _raw_rewrite_term_drift_matches(
    item: Dict[str, Any],
    canonical: str,
    *,
    min_chars: int = 3,
) -> Dict[str, Any]:
    """Recover a term span that raw had but a later rewrite replaced entirely.

    ``crysknife`` is the motivating case: raw/literary say ``晶牙匕首`` while the
    final says ``啸刃`` at the same position.  Raw does not contain the Decision
    target itself, so the exact-witness mode above must decline.  This second
    mode is limited to one small diff replacement whose raw side is a bounded
    CJK form near the Decision target and whose final side is another non-
    dictionary CJK word.  The final is restored to the Decision target only
    after the normal critic re-audit and pending CAS.
    """
    final_text = str(item.get("text") or "")
    final_path = Path(item.get("final_path") or "")
    cid = str(item.get("task", {}).get("chunk_id") or "")
    raw_path = final_path.with_name("%s_raw.json" % cid)
    raw_payload = _read_json(raw_path)
    raw_text = str((raw_payload or {}).get("text") or "")
    target = _term_target_core(canonical)
    target_key = _name_key(target)
    evidence: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "",
        "target": target,
        "raw_path": str(raw_path),
        "raw_final_similarity": None,
        "matches": [],
    }
    target_chars = _canonical_name_chars(
        target, min(3, max(1, int(min_chars))))
    if not raw_text.strip() or not final_text.strip():
        evidence["reason"] = "raw_or_final_missing"
        return evidence
    if not target_chars or len(target_chars) < 3:
        evidence["reason"] = "target_not_pure_cjk_or_too_short"
        return evidence
    raw_body = _body_before_footnotes(raw_text)
    final_body = _body_before_footnotes(final_text)
    if target_key in _name_key(final_body):
        evidence["reason"] = "target_already_present"
        return evidence
    similarity = difflib.SequenceMatcher(
        None, raw_text, final_text, autojunk=False).ratio()
    evidence["raw_final_similarity"] = round(similarity, 4)
    if similarity < 0.80:
        evidence["reason"] = "raw_final_drift_too_large"
        return evidence

    candidates: List[Dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(
        None, raw_text, final_text, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace" or i2 <= i1 or j2 <= j1:
            continue
        if i2 > len(raw_body) or j2 > len(final_body):
            continue
        raw_candidate = raw_text[i1:i2].strip()
        final_candidate = final_text[j1:j2].strip()
        raw_key = _name_key(raw_candidate)
        final_key = _name_key(final_candidate)
        if not raw_key or not final_key:
            continue
        if not all(re.fullmatch(r"[\u4e00-\u9fff]", ch) for ch in raw_key):
            continue
        if not all(re.fullmatch(r"[\u4e00-\u9fff]", ch) for ch in final_key):
            continue
        if len(raw_key) > len(target_key) + 1:
            continue
        if len(final_key) < 2 or len(final_key) > len(target_key) + 1:
            continue
        if final_key == target_key or final_key in raw_key:
            continue
        if final_key in target_key or target_key in final_key:
            continue
        if is_likely_common_word(final_key):
            continue
        raw_similarity = difflib.SequenceMatcher(
            None, raw_key, target_key, autojunk=False).ratio()
        if raw_similarity < 0.55:
            continue
        common = difflib.SequenceMatcher(
            None, raw_key, target_key, autojunk=False).find_longest_match(
                0, len(raw_key), 0, len(target_key))
        if common.size < 2:
            continue
        if _name_key(final_candidate) in _name_key(raw_body):
            continue
        if len(_literal_name_spans(final_body, final_candidate)) != 1:
            continue
        candidates.append({
            "start": j1,
            "end": j2,
            "form": final_candidate,
            "normalized": final_key,
            "raw_form": raw_candidate,
            "raw_similarity": round(raw_similarity, 4),
        })
    forms = {str(match["normalized"]) for match in candidates}
    if not candidates:
        evidence["reason"] = "no_bounded_term_rewrite_span"
        return evidence
    if len(candidates) != 1 or len(forms) != 1:
        evidence["reason"] = "multiple_term_rewrite_spans"
        evidence["matches"] = candidates
        return evidence
    evidence.update(
        status="eligible",
        reason="",
        matches=candidates,
        wrong_forms=sorted({str(match["form"]) for match in candidates}),
    )
    return evidence


def _stable_payload_sha256(payload: Any) -> str:
    """Hash a JSON-like payload without depending on file formatting."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return sha256_text(encoded)


def _completed_reject_memo_path(root: Path, out_dir: Path) -> Path:
    cfg = (_config.get("pipeline.completed_name_repair", {}) or {})
    raw = str(cfg.get("reject_memo_path") or "").strip()
    if not raw:
        return Path(out_dir) / "completed_name_repair_rejects.json"
    path = Path(raw)
    return path if path.is_absolute() else Path(root) / path


def _load_completed_reject_memo(path: Path) -> Dict[str, Any]:
    payload = _read_json(Path(path)) or {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        entries = []
    return {
        "schema_version": 1,
        "updated_at": str(payload.get("updated_at") or ""),
        "entries": [entry for entry in entries if isinstance(entry, dict)],
    }


def _completed_repair_policy_fingerprint() -> Dict[str, Any]:
    return {
        "critic_score_floor": _score_floor(),
        "calibrated_backends": sorted(_calibrated_backends()),
    }


def _completed_repair_fingerprint(record: Dict[str, Any]) -> Dict[str, Any]:
    decisions = record.get("applied_decisions") or []
    decision = decisions[0] if decisions and isinstance(decisions[0], dict) else {}
    return {
        "chunk_id": str(record.get("chunk_id") or ""),
        "chapter": str(record.get("chapter") or ""),
        "final_sha256_before": str(record.get("final_sha256_before") or ""),
        "final_sha256_after": str(record.get("final_sha256_after") or ""),
        "critic_sha256": str(record.get("critic_sha256") or ""),
        "decision_id": record.get("decision_id"),
        "decision_source_key": str(decision.get("source_key") or ""),
        "decision_translation": str(decision.get("translation") or ""),
        "decision_locked": int(decision.get("locked") or 0),
        "policy": _completed_repair_policy_fingerprint(),
    }


def _completed_repair_fingerprint_key(record: Dict[str, Any]) -> str:
    return _stable_payload_sha256(_completed_repair_fingerprint(record))


def _completed_reject_memo_entry(
    memo: Dict[str, Any],
    record: Dict[str, Any],
) -> Optional[dict]:
    key = _completed_repair_fingerprint_key(record)
    for entry in memo.get("entries") or []:
        if str(entry.get("key") or "") == key:
            return entry
    return None


def _stable_critic_rejection(
    report: Optional[dict],
    reasons: Iterable[str],
) -> bool:
    """Return True only for a quality rejection that is safe to memoize.

    Missing backends, malformed reports, verifier exceptions and empty text are
    infrastructure or programming failures.  Caching them would turn a
    transient outage into a permanent repair veto, so they remain retryable.
    """
    report = report or {}
    if str(report.get("_backend_path") or "") not in _calibrated_backends():
        return False
    unstable_prefixes = (
        "critic_backend_uncalibrated:",
        "critic_scores_missing",
        "entity_check_error:",
        "proposed_text_empty",
    )
    return not any(
        str(reason).startswith(unstable_prefixes)
        for reason in reasons
    )


def _record_completed_reject(
    path: Path,
    memo: Dict[str, Any],
    record: Dict[str, Any],
    reasons: Iterable[str],
    *,
    run_id: str,
) -> str:
    key = _completed_repair_fingerprint_key(record)
    entries = [
        entry for entry in (memo.get("entries") or [])
        if str(entry.get("key") or "") != key
    ]
    entries.append({
        "key": key,
        "fingerprint": _completed_repair_fingerprint(record),
        "chunk_id": str(record.get("chunk_id") or ""),
        "chapter": str(record.get("chapter") or ""),
        "reasons": sorted({str(reason) for reason in reasons if reason}),
        "run_id": run_id,
        "rejected_at": _now_iso(),
    })
    payload = {
        "schema_version": 1,
        "updated_at": _now_iso(),
        "entries": entries[-5000:],
    }
    _atomic_json_fsync(path, payload)
    memo.clear()
    memo.update(payload)
    return key


def _classify_completed_name_candidate(
    item: Dict[str, Any],
    *,
    context: Dict[str, Any],
    pending_entry: Optional[dict],
    min_chars: int = 4,
    authority_corpus: Optional[Dict[str, Dict[str, str]]] = None,
    authority_gram_cache: Optional[Dict[str, set]] = None,
) -> Dict[str, Any]:
    """Build a strict candidate record for one COMPLETED chunk.

    A candidate is not permission to write.  It is a frozen plan that the
    mutating phase must revalidate against the current files and a fresh
    calibrated critic report before touching any artifact.
    """
    task = item["task"]
    cid = str(task.get("chunk_id") or "")
    chapter = str(task.get("chapter_id") or "")
    before = str(item.get("text") or "")
    record: Dict[str, Any] = {
        "chunk_id": cid,
        "chapter": chapter,
        "state_before": str(task.get("state") or ""),
        "final_sha256_before": sha256_text(before),
        "final_sha256_after": "",
        "decision_id": None,
        "source_key": "",
        "canonical_name": "",
        "wrong_forms": [],
        "changed_spans": [],
        "critic_before": {
            "backend_path": (item.get("critic") or {}).get("_backend_path"),
            "scores": (item.get("critic") or {}).get("scores") or {},
        },
        "critic_sha256": _stable_payload_sha256(item.get("critic") or {}),
        "pending_before": [pending_entry] if pending_entry else [],
        "pending_after": [],
        "applied_decisions": [],
        "reaudit_required": True,
        "status": "kept",
        "reasons": [],
    }
    reasons = record["reasons"]

    if not pending_entry:
        reasons.append("no_pending_backtrack")
        return record
    record["decision_id"] = pending_entry.get("decision_id")
    record["source_key"] = str(pending_entry.get("decision_source_key") or "")
    canonical = str(pending_entry.get("decision_translation") or "")
    record["canonical_name"] = canonical
    pending_status = str(pending_entry.get("status") or "")
    pending_reason = str(pending_entry.get("reason") or "")
    # A locked decision turns every other same-name outstanding row into
    # ``locked_decision_target_missing``.  That is not a separate blocker: it is
    # the same deterministic one-character drift, now protected by the lock.
    if pending_status == "outstanding":
        if pending_reason != "completed_final_still_misses_target":
            reasons.append("pending_reason:%s" % pending_reason)
            return record
    elif (pending_status == "satisfied"
          and pending_reason == "target_present"):
        # A chunk may contain the canonical form and a registered rejected
        # variant at the same time.  Presence of the canonical form satisfies
        # the outbox check, but the rejected form still needs the strict
        # one-character drift repair below.
        pass
    elif (pending_status == "blocked"
          and pending_reason == "locked_decision_target_missing"):
        # eligible: the lock is the reason the target is a hard invariant.
        pass
    else:
        reasons.append("pending_not_outstanding:%s" % pending_status)
        return record
    if not bool(pending_entry.get("source_active")):
        reasons.append("source_not_active")
        return record
    if str(task.get("state") or "") != "COMPLETED":
        reasons.append("state_not_completed:%s" % task.get("state"))
    if task.get("rescue_stage"):
        reasons.append("rescue_stage_present")
    if not before.strip():
        reasons.append("final_missing_or_unreadable")
        return record
    final = item.get("final") or {}
    metadata = final.get("metadata") if isinstance(final.get("metadata"), dict) else {}
    if metadata.get("fallback") or metadata.get("fallback_used"):
        reasons.append("final_fallback")
    if metadata.get("quality_waived"):
        reasons.append("final_quality_waived")

    report = item.get("critic") or {}
    backend = str(report.get("_backend_path") or "missing")
    if not report:
        reasons.append("critic_report_missing")
    elif backend not in _calibrated_backends():
        reasons.append("critic_backend_uncalibrated:%s" % backend)
    low = _scores_below_floor(report) if report else ["critic_scores_missing"]
    entity_only_low = bool(low) and all(
        str(item).startswith("entity_consistency:") for item in low)
    if low and not entity_only_low:
        reasons.append("critic_scores_below_floor")
        record["critic_before"]["low_scores"] = low
    elif low:
        record["critic_before"]["low_scores"] = low

    source_key = record["source_key"]
    if not source_key or not canonical:
        reasons.append("empty_decision")
        return record
    target_key = _name_key(canonical)
    raw_provenance = _raw_provenance_name_drift_matches(
        item, canonical, min_chars=min_chars)
    record["raw_provenance"] = raw_provenance
    target_migration = _decision_target_migration_matches(
        item, context, pending_entry, canonical)
    record["target_migration"] = target_migration
    raw_rewrite: Dict[str, Any] = {}
    if raw_provenance.get("status") != "eligible":
        raw_rewrite = _raw_rewrite_term_drift_matches(
            item, canonical, min_chars=min_chars)
        record["raw_rewrite"] = raw_rewrite
    drifts: List[Dict[str, Any]] = []
    drift_mode = ""
    alias_min_chars = min(2, max(1, int(min_chars)))
    raw_mode = ""
    raw_evidence: Dict[str, Any] = {}
    if target_migration.get("status") == "eligible":
        # A target change can be a complete spelling change (Carthag:
        # 迦太基 -> 卡萨格), so the one-character-drift helpers do not apply.
        # The Decision reason marker plus an exact source/old-target occurrence
        # count is the authority boundary for this mode.
        drift_mode = "decision_target_migration"
        drifts = list(target_migration.get("matches") or [])
    elif raw_provenance.get("status") == "eligible":
        raw_mode = "raw_provenance"
        raw_evidence = raw_provenance
    elif raw_rewrite.get("status") == "eligible":
        raw_mode = "raw_rewrite"
        raw_evidence = raw_rewrite
    if raw_mode:
        # ``registry_trusted_by_source`` contains only non-pending hard source
        # aliases.  A source form present only in ``quarantined_names`` is an
        # explicit non-authority and must not veto a Decision-backed raw
        # witness.  This is the Yueh/威灵顿·岳 pollution case.
        trusted_owners = set(
            (context.get("registry_trusted_by_source") or {}).get(
                _phrase_key(source_key)) or set())
        if len(trusted_owners) > 1:
            reasons.append("registry_active_owner_count:%d" % len(trusted_owners))
            return record
        if len(trusted_owners) == 1:
            owner = next(iter(trusted_owners))
            if not _name_compatible(canonical, owner):
                reasons.append("registry_target_mismatch")
                return record
        drift_mode = raw_mode
        drifts = list(raw_evidence.get("matches") or [])
    elif target_migration.get("status") != "eligible":
        active_owners = _active_name_owners(context, source_key)
        if len(active_owners) > 1:
            reasons.append("registry_active_owner_count:%d" % len(active_owners))
            return record
        if len(active_owners) == 1:
            owner_key = _name_key(active_owners[0].get("canonical") or "")
        else:
            owner_key = ""
        if owner_key and not _name_compatible(canonical, owner_key):
            reasons.append("registry_target_mismatch")
            return record
        # Preserve the existing one-character / registered-alias behaviour
        # first.  The source-counted mode is a fallback for a term whose
        # registry owner is still pending review (Mapes) or absent
        # (truthsense), not a replacement for an exact narrow alias repair.
        if len(active_owners) == 1:
            registered = {
                _name_key(value)
                for value in ((context.get("registry_by_source") or {}).get(
                    _phrase_key(source_key)) or set())
                if _name_key(value)
            }
            allowed_alias_keys = _registered_alias_drift_keys(
                source_key=source_key,
                canonical=canonical,
                context=context,
                min_chars=min(3, max(1, int(min_chars))),
            )
            if registered and (registered - {target_key} - allowed_alias_keys):
                reasons.append("registry_source_target_ambiguous")
                return record
            alias_drifts = _registered_alias_drift_matches(
                before,
                source_key=source_key,
                canonical=canonical,
                context=context,
                min_chars=alias_min_chars,
            )
            fuzzy_drifts = (
                _one_char_name_drift_matches(before, canonical, min_chars=min_chars)
                if _canonical_name_chars(canonical, min_chars) else []
            )
            drift_mode = "registered_alias" if alias_drifts else "one_char_cjk"
            drifts = alias_drifts or fuzzy_drifts
        if not drifts:
            source_scoped = _pending_source_scoped_term_matches(
                item, canonical, source_key=source_key, context=context)
            record["source_scoped_variant"] = source_scoped
            if source_scoped.get("status") == "eligible":
                drift_mode = "source_scoped_variant"
                drifts = list(source_scoped.get("matches") or [])
            elif len(active_owners) != 1:
                reasons.append(
                    "registry_active_owner_count:%d" % len(active_owners))
                return record
    if not drifts:
        if entity_only_low:
            reasons.append("critic_scores_below_floor")
        migration_reason = str(target_migration.get("reason") or "")
        if migration_reason:
            reasons.append("target_migration:%s" % migration_reason)
        reasons.append(
            "canonical_not_pure_cjk_or_too_short"
            if drift_mode not in {"raw_provenance", "raw_rewrite"}
            and not _canonical_name_chars(canonical, min_chars)
            else "no_one_char_name_drift")
        return record
    unique_forms = {str(match["normalized"]) for match in drifts}
    if len(unique_forms) != 1:
        reasons.append("multiple_drift_forms")
        return record
    if entity_only_low and drift_mode != "source_scoped_variant":
        reasons.append("critic_scores_below_floor")
    wrong_key = next(iter(unique_forms))
    if wrong_key in _protected_active_name_keys(context, target_key):
        reasons.append("drift_form_belongs_to_another_active_name")
        return record

    after = _replace_name_drift_spans(before, drifts, canonical)
    if after == before:
        reasons.append("no_name_change")
        return record
    if drift_mode == "decision_target_migration":
        old_target = str(target_migration.get("old_target") or "")
        if old_target and old_target in after:
            reasons.append("residual_target_migration_form")
            return record
    elif drift_mode == "registered_alias":
        if _registered_alias_drift_matches(
                after,
                source_key=source_key,
                canonical=canonical,
                context=context,
                min_chars=alias_min_chars):
            reasons.append("residual_registered_alias_drift")
            return record
    elif drift_mode == "one_char_cjk" and _one_char_name_drift_matches(
            after, canonical, min_chars=min_chars):
        reasons.append("residual_one_char_name_drift")
        return record
    elif drift_mode == "source_scoped_variant":
        wrong_forms = {
            str(match.get("form") or "")
            for match in drifts
            if str(match.get("form") or "")
        }
        if any(form in after for form in wrong_forms):
            reasons.append("residual_source_scoped_variant")
            return record
    if not canonical or canonical not in after:
        reasons.append("canonical_not_present_after_repair")
        return record

    proposed_entries = _pending_rows_for_chunk(context, cid, after)
    record["pending_after"] = proposed_entries
    if len(proposed_entries) != 1 or proposed_entries[0].get("status") != "satisfied":
        reasons.append("pending_not_satisfied_by_proposed_final")
        return record

    decision = (context.get("decisions") or {}).get(
        int(record["decision_id"]) if record["decision_id"] is not None else -1)
    if not decision:
        reasons.append("decision_missing")
        return record
    corpus = authority_corpus
    gram_cache = authority_gram_cache
    if corpus is None or gram_cache is None:
        corpus, gram_cache = build_lock_authority_corpus(context)
    authority = evaluate_decision_lock_authority(
        decision_id=int(decision["id"]),
        source_key=str(decision.get("source_key") or ""),
        target=str(decision.get("translation") or ""),
        context=context,
        corpus=corpus,
        gram_cache=gram_cache,
        proposed_chunk_id=cid,
        proposed_text=after,
    )
    record["lock_authority"] = authority
    if not authority.get("lock_allowed"):
        reasons.append(
            "decision_authority_not_confirmed:%s"
            % str(authority.get("status") or "unknown"))
        return record
    record["applied_decisions"] = [{
        "id": int(decision["id"]),
        "source_key": str(decision.get("source_key") or ""),
        "translation": str(decision.get("translation") or ""),
        "locked": int(decision.get("locked") or 0),
        "authority": authority,
    }]
    record.update({
        "final_sha256_after": sha256_text(after),
        "wrong_forms": sorted({str(match["form"]) for match in drifts}),
        "repair_mode": drift_mode,
        "drift_matches": drifts,
        "changed_spans": _changed_spans(before, after),
        "status": "candidate" if not reasons else "kept",
    })
    return record


def _completed_repair_plan(
    root: Path,
    *,
    chapter_id: Optional[str] = None,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_chars: int = 4,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    root = Path(root)
    decision_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    workflow_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "chapter": chapter_id,
        "counts": {},
        "reject_memo_skipped": 0,
        "candidates": [],
        "kept": [],
        "failed": [],
        "ok": False,
    }
    if not decision_path.exists() or not workflow_path.exists():
        result["failed"].append({"reason": "database_missing"})
        result["counts"] = {"failed": 1}
        return result, {}, {}
    context = load_pending_context(
        root, decision_db=decision_path, workflow_db=workflow_path,
        output_dir=out_dir,
    )
    if not context.get("available"):
        result["failed"].append({
            "reason": "pending_context_unavailable",
            "detail": context.get("reason") or "",
        })
        result["counts"] = {"failed": 1}
        return result, {}, context
    classification = classify_pending_snapshot(context)
    pending_by_chunk = {
        str(entry.get("chunk_id") or ""): entry
        for entry in (classification.get("entries") or [])
    }
    memo_path = _completed_reject_memo_path(root, out_dir)
    reject_memo = _load_completed_reject_memo(memo_path)
    result["reject_memo_path"] = str(memo_path)
    items = _load_completed_items(workflow_path, out_dir, chapter_id)
    items_by_cid = {str(item["task"]["chunk_id"]): item for item in items}
    authority_corpus, authority_gram_cache = build_lock_authority_corpus(context)
    for item in items:
        cid = str(item["task"]["chunk_id"])
        try:
            record = _classify_completed_name_candidate(
                item, context=context,
                pending_entry=pending_by_chunk.get(cid),
                min_chars=min_chars,
                authority_corpus=authority_corpus,
                authority_gram_cache=authority_gram_cache,
            )
        except Exception as exc:  # noqa: BLE001 - planner must report truthfully
            result["failed"].append({
                "chunk_id": cid,
                "chapter": str(item["task"].get("chapter_id") or ""),
                "reason": "%s: %s" % (type(exc).__name__, exc),
            })
            continue
        if record.get("status") == "candidate":
            memo_entry = _completed_reject_memo_entry(reject_memo, record)
            if memo_entry:
                record["status"] = "kept"
                record["reasons"] = ["reaudit_reject_memo"]
                record["reject_memo"] = {
                    "key": memo_entry.get("key"),
                    "reject_reasons": memo_entry.get("reasons") or [],
                    "rejected_at": memo_entry.get("rejected_at"),
                    "run_id": memo_entry.get("run_id"),
                }
                result["reject_memo_skipped"] += 1
                result["kept"].append(record)
            else:
                result["candidates"].append(record)
        else:
            # Only surface kept rows that were plausibly in scope; otherwise
            # every ordinary COMPLETED chunk would drown the report.
            if record.get("pending_before"):
                result["kept"].append(record)
    reason_counts: Dict[str, int] = {}
    for record in result["kept"]:
        for reason in record.get("reasons") or []:
            key = str(reason).split(":", 1)[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    result["reason_counts"] = dict(sorted(reason_counts.items()))
    result["counts"] = {
        "completed_scanned": len(items),
        "candidates": len(result["candidates"]),
        "kept": len(result["kept"]),
        "failed": len(result["failed"]),
        "memo_skipped": result["reject_memo_skipped"],
    }
    result["ok"] = not result["failed"]
    result["pending_counts"] = dict(classification.get("counts") or {})
    return result, items_by_cid, context


def _completed_repair_markdown(result: Dict[str, Any]) -> str:
    lines = ["# COMPLETED 专名漂移修复计划（只读）", ""]
    lines.append("- 生成时间: %s" % result.get("generated_at"))
    lines.append("- 章节: %s" % (result.get("chapter") or "全书"))
    lines.append("- 结论: **%s**" % (
        "存在可修复候选" if result.get("candidates") else "无候选"))
    counts = result.get("counts") or {}
    if counts:
        lines.append("- 扫描: " + ", ".join(
            "`%s`=%s" % (key, value) for key, value in counts.items()))
    if result.get("reason_counts"):
        lines.append("- 保留原因: " + ", ".join(
            "`%s`=%s" % (key, value)
            for key, value in result["reason_counts"].items()))
    if result.get("reject_memo_skipped"):
        lines.append("- 拒绝备忘录跳过: `%s`；文件: `%s`" % (
            result.get("reject_memo_skipped"), result.get("reject_memo_path")))
    candidates = result.get("candidates") or []
    if candidates:
        lines.extend(["", "## 候选（前 40 / 共 %d）" % len(candidates)])
        for record in candidates[:40]:
            lines.append(
                "- `%s` %s → %s；错形 %s；出现 %d 处；pending=%s"
                % (record.get("chunk_id"), record.get("source_key"),
                   record.get("canonical_name"),
                   " / ".join(record.get("wrong_forms") or []),
                   len(record.get("changed_spans") or []),
                   record.get("decision_id")))
        if len(candidates) > 40:
            lines.append("- 其余 %d 条见 JSON。" % (len(candidates) - 40))
    kept = result.get("kept") or []
    if kept:
        lines.extend(["", "## 明确保留（前 30 / 共 %d）" % len(kept)])
        for record in kept[:30]:
            lines.append(
                "- `%s` %s / %s" % (
                    record.get("chunk_id"),
                    record.get("source_key") or "-",
                    ",".join(record.get("reasons") or [])))
        if len(kept) > 30:
            lines.append("- 其余 %d 条见 JSON。" % (len(kept) - 30))
    lines.append("")
    return "\n".join(lines)


def audit_completed_name_repair(
    project_root: Path,
    *,
    chapter_id: Optional[str] = None,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_chars: int = 4,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "completed_name_repair.json"
    mpath = Path(md_out) if md_out else out_dir / "completed_name_repair.md"
    result, _items, _context = _completed_repair_plan(
        root, chapter_id=chapter_id, decision_db=decision_db,
        workflow_db=workflow_db, output_dir=out_dir, min_chars=min_chars,
    )
    result["json_path"] = str(jpath)
    result["md_path"] = str(mpath)
    _atomic_json(jpath, result)
    _atomic_text(mpath, _completed_repair_markdown(result))
    return result


def _blocked_reasons(
    *,
    root: Path,
    task: Dict[str, Any],
    final_path: Path,
    final: Optional[dict],
    critic_report: Optional[dict],
    maps: Dict[str, Any],
    stats: Dict[str, Any],
    source_text: str,
    context: Dict[str, Any],
    decision_index: Dict[str, List[dict]],
    predicted_text: str,
    allow_state_only: bool = False,
    authority_corpus: Optional[Dict[str, Dict[str, str]]] = None,
    authority_gram_cache: Optional[Dict[str, set]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    reasons: List[str] = []
    diagnostics: Dict[str, Any] = {}
    if str(task.get("state") or "") != "DIRTY":
        reasons.append(f"state_not_dirty:{task.get('state')}")
    if task.get("rescue_stage"):
        reasons.append("rescue_stage_present")
    if not context.get("available"):
        reasons.append("pending_context_unavailable")
    elif context.get("decisions_complete") is False:
        reasons.append("decision_db_schema_incomplete")
    if final is None or not str(final.get("text") or "").strip():
        reasons.append("final_missing_or_unreadable")
        return reasons, diagnostics
    meta = final.get("metadata") if isinstance(final.get("metadata"), dict) else {}
    if meta.get("fallback") or meta.get("fallback_used"):
        reasons.append("final_fallback")
    if meta.get("quality_waived"):
        reasons.append("final_quality_waived")

    report = critic_report or {}
    evidence = critic_evidence_state(report, final_path, project_root=root)
    diagnostics["critic_evidence"] = evidence
    if evidence.get("status") not in {"exact", "repaired"}:
        reasons.append(f"critic_evidence_{evidence.get('status') or 'missing'}")
    backend = str(report.get("_backend_path") or "missing")
    diagnostics["critic_backend"] = backend
    if backend not in _calibrated_backends():
        reasons.append(f"critic_backend_uncalibrated:{backend}")
    low = _scores_below_floor(report)
    if low:
        diagnostics["low_scores"] = low
        reasons.append("critic_scores_below_floor")

    n_sep = int(stats.get("sep_restore") or 0)
    n_forms = int(stats.get("forms") or 0)
    n_tokens = int(stats.get("tokens") or 0)
    diagnostics["mapping_counts"] = {
        "sep_restore": n_sep, "forms": n_forms, "tokens": n_tokens}
    text_changed = predicted_text != str(final.get("text") or "")
    diagnostics["change_mode"] = "name_only" if text_changed else "state_only"
    if (n_sep + n_forms + n_tokens <= 0 or not text_changed) \
            and not allow_state_only:
        reasons.append("no_name_change")
    if n_tokens:
        # The first production slice is deliberately narrower than the long-term
        # contract: token drift needs its own false-positive measurement.
        reasons.append("token_level_change_requires_separate_validation")

    mapping_targets = _mapping_targets(maps, stats)
    for target in mapping_targets:
        if canonical_is_title_only(target):
            reasons.append(f"role_only_target:{target}")
            break

    residual_forms = _residual_authority_forms(
        predicted_text, maps.get("authorities") or [])
    if residual_forms:
        diagnostics["residual_authority_forms"] = residual_forms
        reasons.append("residual_authority_form")

    conflicts = _active_authority_conflicts(
        source_text, context, decision_index, targets=mapping_targets)
    if conflicts:
        diagnostics["authority_conflicts"] = conflicts
        reasons.append("authority_conflict")

    lockable = _lockable_decisions(
        source_text, maps, stats, context, decision_index,
        chunk_id=str(task.get("chunk_id") or ""),
        proposed_text=predicted_text,
        authority_corpus=authority_corpus,
        authority_gram_cache=authority_gram_cache,
    )
    diagnostics["decision_authority"] = lockable
    allowed_lockable = [row for row in lockable if row.get("lock_allowed")]
    diagnostics["lockable_decisions"] = allowed_lockable
    blocked_authority = [row for row in lockable if not row.get("lock_allowed")]
    if blocked_authority:
        diagnostics["blocked_decision_authority"] = blocked_authority
        reasons.append("decision_authority_not_confirmed")
    if not lockable and (text_changed or not allow_state_only):
        reasons.append("no_lockable_decision")
    elif not allowed_lockable and (text_changed or not allow_state_only):
        reasons.append("no_authorized_lockable_decision")

    pending_before = classify_pending_snapshot(
        context, final_overrides={str(task["chunk_id"]): predicted_text})
    chunk_entries = [e for e in pending_before.get("entries") or []
                     if e.get("chunk_id") == task.get("chunk_id")]
    diagnostics["pending"] = chunk_entries
    # A fast path bypasses the normal DIRTY rerun, so "covered by the upcoming
    # run" is no longer a sufficient explanation.  The proposed final itself
    # must satisfy the outbox row.
    pending_blocking = [e for e in chunk_entries if e.get("status") not in {
        "satisfied"}]
    if pending_blocking:
        reasons.append("pending_not_satisfied_by_proposed_final")
        diagnostics["pending_blocking"] = pending_blocking
    return list(dict.fromkeys(reasons)), diagnostics


def audit_name_only_fastpath(
    project_root: Path,
    *,
    chapter_id: Optional[str] = None,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
    allow_state_only: bool = False,
) -> Dict[str, Any]:
    """Classify DIRTY chunks without mutating any production artifact."""
    root = Path(project_root)
    decision_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    workflow_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "name_only_fastpath.json"
    mpath = Path(md_out) if md_out else out_dir / "name_only_fastpath.md"
    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "read_only",
        "generated_at": _now_iso(),
        "chapter": chapter_id,
        "candidates": [],
        "kept_dirty": [],
        "failed": [],
        "counts": {},
        "write_attempted": False,
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if not decision_path.exists() or not workflow_path.exists():
        result["counts"] = {"failed": 1}
        result["failed"].append({"reason": "database_missing"})
        result["ok"] = False
        _atomic_json(jpath, result)
        _atomic_text(mpath, "# name-only fast path dry-run\n\n数据库缺失；未执行。\n")
        return result

    wcon = sqlite3.connect(f"file:{workflow_path.resolve()}?mode=ro", uri=True)
    wcon.row_factory = sqlite3.Row
    try:
        sql = ("SELECT chunk_id, chapter_id, text_content, state, rescue_stage "
               "FROM chunk_tasks WHERE state='DIRTY'")
        params: tuple = ()
        if chapter_id:
            sql += " AND chapter_id=?"
            params = (chapter_id,)
        tasks = [dict(r) for r in wcon.execute(sql + " ORDER BY chunk_id", params)]
    finally:
        wcon.close()

    loaded: List[Dict[str, Any]] = []
    for task in tasks:
        cid = str(task["chunk_id"])
        ch = str(task["chapter_id"])
        final_path = out_dir / ch / f"{cid}_final.json"
        final = _read_json(final_path)
        loaded.append({
            "task": task,
            "final_path": final_path,
            "final": final,
            "text": str((final or {}).get("text") or ""),
            "critic": _read_json(out_dir / ch / f"{cid}_critic_report.json"),
        })

    try:
        maps = _build_maps(root, decision_path, loaded)
    except Exception as exc:  # noqa: BLE001 - report a truthful dry-run failure
        result["ok"] = False
        result["failed"].append({"reason": "map_build_failed", "error": str(exc)})
        result["counts"] = {"failed": 1}
        _atomic_json(jpath, result)
        _atomic_text(mpath, f"# name-only fast path dry-run\n\n映射构建失败：{exc}\n")
        return result

    context = load_pending_context(
        root, decision_db=decision_path, workflow_db=workflow_path,
        output_dir=out_dir,
    )
    decision_index = _decision_index(context)
    authority_corpus, authority_gram_cache = build_lock_authority_corpus(context)
    for item in loaded:
        task = item["task"]
        cid = str(task["chunk_id"])
        before = item["text"]
        try:
            after, stats = apply_name_unification(before, maps)
            reasons, diagnostics = _blocked_reasons(
                root=root, task=task, final_path=item["final_path"],
                final=item["final"], critic_report=item["critic"],
                maps=maps, stats=stats, source_text=str(task.get("text_content") or ""),
                context=context, decision_index=decision_index,
                predicted_text=after,
                allow_state_only=allow_state_only,
                authority_corpus=authority_corpus,
                authority_gram_cache=authority_gram_cache,
            )
            record = {
                "chunk_id": cid,
                "chapter": str(task["chapter_id"]),
                "state_before": str(task.get("state") or ""),
                "final_sha256_before": sha256_text(before),
                "final_sha256_after": sha256_text(after),
                "mapping_counts": diagnostics.get("mapping_counts") or {},
                "change_mode": diagnostics.get("change_mode") or "name_only",
                "changed_spans": _changed_spans(before, after),
                "critic_before": {
                    "backend_path": diagnostics.get("critic_backend"),
                    "evidence": diagnostics.get("critic_evidence"),
                    "scores": (item["critic"] or {}).get("scores") or {},
                },
                "pending_before": diagnostics.get("pending") or [],
                "authority_conflicts": diagnostics.get("authority_conflicts") or [],
                "residual_authority_forms": (
                    diagnostics.get("residual_authority_forms") or []),
                "decision_authority": diagnostics.get("decision_authority") or [],
                "applied_decisions": diagnostics.get("lockable_decisions") or [],
                "reaudit_required": True,
                "status": "candidate" if not reasons else "kept_dirty",
                "reasons": reasons,
            }
            if reasons:
                result["kept_dirty"].append(record)
            else:
                result["candidates"].append(record)
        except Exception as exc:  # noqa: BLE001
            result["failed"].append({
                "chunk_id": cid,
                "chapter": str(task["chapter_id"]),
                "reason": f"{type(exc).__name__}: {exc}",
            })

    result["counts"] = {
        "dirty_scanned": len(loaded),
        "candidates": len(result["candidates"]),
        "kept_dirty": len(result["kept_dirty"]),
        "failed": len(result["failed"]),
    }
    reason_counts: Dict[str, int] = {}
    for record in result["kept_dirty"]:
        for reason in record.get("reasons") or []:
            key = str(reason).split(":", 1)[0]
            reason_counts[key] = reason_counts.get(key, 0) + 1
    result["reason_counts"] = dict(sorted(reason_counts.items()))
    result["ok"] = not result["failed"]
    _atomic_json(jpath, result)
    _atomic_text(mpath, _name_only_markdown(result))
    return result


def _restore_file_from_backup(path: Path, backup: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".rollback.tmp")
    tmp.write_bytes(Path(backup).read_bytes())
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_bytes(Path(source).read_bytes())
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, destination)


def apply_name_only_fastpath(
    project_root: Path,
    *,
    chapter_id: Optional[str] = None,
    apply: bool = False,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
    critic_fn: Optional[Any] = None,
    pipeline_factory: Optional[Any] = None,
    scheduler: Any = None,
    require_authority_audit: Optional[bool] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
    allow_state_only: bool = False,
) -> Dict[str, Any]:
    """Apply strict name-only repairs, or return the read-only plan.

    Every candidate is revalidated from scratch under the caller's flock.  The
    final and critic report are only replaced after a real calibrated re-audit
    succeeds.  If the workflow transition fails, both files are restored from
    the isolated run backup and the task remains DIRTY.
    """
    root = Path(project_root)
    decision_path = root / "db" / "decision_db.sqlite"
    workflow_path = root / "db" / "workflow.db"
    out_dir = root / "output"
    cfg = (_config.get("pipeline.name_only_fastpath", {}) or {})
    jpath = Path(json_out) if json_out else out_dir / "name_only_fastpath_apply.json"
    mpath = Path(md_out) if md_out else out_dir / "name_only_fastpath_apply.md"
    if not apply:
        return audit_name_only_fastpath(
            root, chapter_id=chapter_id, decision_db=decision_path,
            workflow_db=workflow_path, output_dir=out_dir,
            allow_state_only=allow_state_only)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "apply",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "chapter": chapter_id,
        "run_id": str(run_id or (
            _dt.datetime.now().strftime("name-only-%Y%m%dT%H%M%S") +
            f"-{os.getpid()}")).replace("/", "_"),
        "actions": [],
        "counts": {},
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if not bool(cfg.get("enabled", True)):
        result.update(ok=True, disabled=True, reason="name_only_fastpath_disabled")
        _atomic_json(jpath, result)
        _atomic_text(mpath, "# name-only fast path apply\n\n配置已关闭；未执行。\n")
        return result

    require_auth = (bool(cfg.get("require_authority_audit", True))
                    if require_authority_audit is None
                    else bool(require_authority_audit))
    if require_auth:
        from src.name_authority_audit import audit_name_authorities
        authority = audit_name_authorities(root)
        result["authority"] = {
            "ok": bool(authority.get("ok")),
            "status": authority.get("status"),
            "counts": authority.get("counts") or {},
            "blocking_count": int(authority.get("blocking_count") or 0),
            "report": authority.get("md_path"),
        }
        if not authority.get("ok"):
            result.update(
                ok=False,
                authority_blocked=True,
                reason="name_authority_audit_not_clean",
            )
            _atomic_json(jpath, result)
            _atomic_text(mpath, _apply_markdown(result))
            return result

    plan = audit_name_only_fastpath(
        root, chapter_id=chapter_id, decision_db=decision_path,
        workflow_db=workflow_path, output_dir=out_dir,
        allow_state_only=allow_state_only)
    candidates = list(plan.get("candidates") or [])
    try:
        cap = max(0, int(limit if limit is not None
                         else cfg.get("max_chunks_per_run", 50)))
    except (TypeError, ValueError):
        cap = 50
    candidates = candidates[:cap]
    result["planned_candidates"] = len(plan.get("candidates") or [])
    result["planned_kept_dirty"] = len(plan.get("kept_dirty") or [])
    result["planned_failed"] = len(plan.get("failed") or [])
    result["write_attempted"] = bool(candidates)
    if not candidates:
        result.update(ok=not bool(plan.get("failed")), no_candidates=True)
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result

    if not decision_path.exists() or not workflow_path.exists():
        result.update(ok=False, errors=["database_missing"])
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result

    loaded = _load_dirty_items(workflow_path, out_dir, chapter_id)
    loaded_by_cid = {str(item["task"]["chunk_id"]): item for item in loaded}
    try:
        maps = _build_maps(root, decision_path, loaded)
    except Exception as exc:  # noqa: BLE001
        result.update(ok=False, errors=[f"map_build_failed:{exc}"])
        _atomic_json(jpath, result)
        _atomic_text(mpath, _apply_markdown(result))
        return result
    context = load_pending_context(
        root, decision_db=decision_path, workflow_db=workflow_path,
        output_dir=out_dir)
    decision_index = _decision_index(context)
    authority_corpus, authority_gram_cache = build_lock_authority_corpus(context)
    if scheduler is None:
        from src.core.scheduler import TaskScheduler
        scheduler = TaskScheduler(db_path=str(workflow_path))
    if pipeline_factory is None:
        from src.pipeline import TranslationPipeline
        pipeline_factory = lambda ch: TranslationPipeline(chapter_id=ch)
    pipelines: Dict[str, Any] = {}
    backup_root = out_dir / "name_only_fastpath" / str(result["run_id"])

    for candidate in candidates:
        cid = str(candidate.get("chunk_id") or "")
        item = loaded_by_cid.get(cid)
        action: Dict[str, Any] = {
            "chunk_id": cid,
            "chapter": candidate.get("chapter"),
            "status": "failed",
            "reason": "",
            "final_sha256_before": candidate.get("final_sha256_before"),
            "final_sha256_after": candidate.get("final_sha256_after"),
            "applied_decisions": candidate.get("applied_decisions") or [],
            "change_mode": candidate.get("change_mode") or "name_only",
        }
        result["actions"].append(action)
        if item is None:
            action.update(status="kept_dirty", reason="chunk_no_longer_dirty")
            continue
        task = item["task"]
        before = str(item.get("text") or "")
        final_payload = item.get("final")
        critic_payload = item.get("critic")
        final_path = Path(item["final_path"])
        report_path = final_path.with_name(f"{cid}_critic_report.json")
        if not final_payload or not critic_payload:
            action.update(status="kept_dirty", reason="final_or_critic_missing")
            continue
        if sha256_text(before) != str(candidate.get("final_sha256_before") or ""):
            action.update(status="kept_dirty", reason="final_changed_since_plan")
            continue
        try:
            after, stats = apply_name_unification(before, maps)
        except Exception as exc:  # noqa: BLE001
            action.update(status="failed", reason=f"apply_error:{exc}")
            continue
        if sha256_text(after) != str(candidate.get("final_sha256_after") or ""):
            action.update(status="kept_dirty", reason="predicted_final_hash_changed")
            continue
        reasons, diagnostics = _blocked_reasons(
            root=root, task=task, final_path=final_path,
            final=final_payload, critic_report=critic_payload,
            maps=maps, stats=stats,
            source_text=str(task.get("text_content") or ""),
            context=context, decision_index=decision_index,
            predicted_text=after,
            allow_state_only=allow_state_only,
            authority_corpus=authority_corpus,
            authority_gram_cache=authority_gram_cache,
        )
        if reasons:
            action.update(status="kept_dirty", reason=",".join(reasons))
            action["diagnostics"] = diagnostics
            continue
        applied_decisions = diagnostics.get("lockable_decisions") or []
        change_mode = str(diagnostics.get("change_mode") or "name_only")
        action["change_mode"] = change_mode
        if not applied_decisions and change_mode != "state_only":
            action.update(status="kept_dirty", reason="no_lockable_decision")
            continue
        action["applied_decisions"] = applied_decisions

        chapter = str(task.get("chapter_id") or candidate.get("chapter") or "")
        pipeline = None
        if critic_fn is None:
            pipeline = pipelines.get(chapter)
            if pipeline is None:
                pipeline = pipeline_factory(chapter)
                pipelines[chapter] = pipeline
        raw_payload = _read_json(final_path.with_name(f"{cid}_raw.json")) or {}
        raw_text = str(raw_payload.get("text") or before)
        try:
            if critic_fn is None:
                report = _default_critic_fn(
                    candidate, str(task.get("text_content") or ""),
                    raw_text, after, pipeline)
            else:
                report = critic_fn(
                    candidate, str(task.get("text_content") or ""),
                    raw_text, after, pipeline)
        except Exception as exc:  # noqa: BLE001 - no final has been written yet
            action.update(status="kept_dirty", reason=f"critic_reaudit_failed:{exc}")
            continue
        critic_reasons = _validate_reauthored_report(report, proposed_text=after)
        if critic_reasons:
            action.update(status="kept_dirty", reason=",".join(critic_reasons))
            action["critic_after"] = {
                "backend_path": (report or {}).get("_backend_path"),
                "scores": (report or {}).get("scores") or {},
            }
            continue

        after_sha = sha256_text(after)
        report = dict(report or {})
        report["_reaudit"] = {
            "tool": "name-only-fastpath",
            "schema_version": 1,
            "at": _now_iso(),
            "previous_backend_path": diagnostics.get("critic_backend"),
            "final_sha256": after_sha,
            "raw_input": "raw" if raw_payload else "final_before",
            "literary_input": "proposed_final",
        }
        report["_final_binding"] = {
            "schema_version": 1,
            "final_sha256": after_sha,
            "before_sha256": sha256_text(before),
            "reason": "name-only-fastpath real critic re-audit",
        }
        report["_audit_target"] = {
            "schema_version": 1,
            "source_sha256": sha256_text(str(task.get("text_content") or "")),
            "raw_sha256": sha256_text(raw_text),
            "literary_sha256": after_sha,
        }

        chunk_backup_dir = backup_root / chapter
        chunk_backup_dir.mkdir(parents=True, exist_ok=True)
        final_backup_path = chunk_backup_dir / f"{cid}_final_before.json"
        report_backup_path = chunk_backup_dir / f"{cid}_critic_report_before.json"
        _copy_file_atomic(final_path, final_backup_path)
        _copy_file_atomic(report_path, report_backup_path)
        new_final = dict(final_payload)
        if change_mode == "name_only":
            meta = dict(new_final.get("metadata") or {})
            meta.update({
                "name_unification_run_id": result["run_id"],
                "name_unification_at": _now_iso(),
                "name_unification_counts": diagnostics.get("mapping_counts") or {},
            })
            new_final["text"] = after
            new_final["metadata"] = meta

        final_written = False
        report_written = False
        state_completed = False
        try:
            if change_mode == "name_only":
                _atomic_json_fsync(final_path, new_final)
                final_written = True
            _atomic_json_fsync(report_path, report)
            report_written = True
            pending_entries = _pending_rows_for_chunk(context, cid, after)
            pending_blocking = [e for e in pending_entries
                                if e.get("status") != "satisfied"]
            if pending_blocking:
                raise RuntimeError(
                    "pending_not_satisfied:" + ",".join(
                        str(e.get("status")) for e in pending_blocking))
            evidence_path = chunk_backup_dir / f"{cid}_name_only_evidence.json"
            evidence = {
                "schema_version": 1,
                "chunk_id": cid,
                "status": "verified_for_completion",
                "run_id": result["run_id"],
                "final_sha256_before": sha256_text(before),
                "final_sha256_after": after_sha,
                "critic_final_sha256": after_sha,
                "critic_report_sha256": _sha256_file(report_path),
                "pending_checked": True,
                "pending_before": pending_entries,
                "applied_decisions": applied_decisions,
                "decision_authority": diagnostics.get("decision_authority") or [],
                "change_mode": change_mode,
                "changed_spans": candidate.get("changed_spans") or [],
                "mapping_counts": diagnostics.get("mapping_counts") or {},
            }
            _atomic_json_fsync(evidence_path, evidence)

            dcon = _open_decision_write_db(decision_path)
            try:
                dcon.execute("BEGIN IMMEDIATE")
                outbox = None
                for row in (context.get("pending") or []):
                    if str(row.get("chunk_id") or "") == cid:
                        outbox = row
                        break
                if outbox is not None and not _delete_pending_row_if_unchanged(
                        dcon, outbox):
                    raise RuntimeError("pending_cas_failed")
                decision_ids = sorted({int(d["id"]) for d in applied_decisions})
                if decision_ids:
                    placeholders = ",".join("?" for _ in decision_ids)
                    dcon.execute(
                        f"UPDATE decision_db SET locked = 1 "
                        f"WHERE id IN ({placeholders})", tuple(decision_ids))
                completed = scheduler.complete_name_only(
                    cid,
                    final_path=final_path,
                    critic_report_path=report_path,
                    evidence_path=evidence_path,
                    final_sha256=after_sha,
                )
                if not completed.get("ok"):
                    dcon.rollback()
                    raise RuntimeError(
                        "state_completion_failed:" + str(completed.get("reason")))
                state_completed = True
                try:
                    dcon.commit()
                except sqlite3.Error as exc:
                    try:
                        dcon.rollback()
                    except sqlite3.Error:
                        pass
                    reopened = scheduler.reopen_name_only(cid)
                    if reopened.get("ok"):
                        rollback_errors = []
                        try:
                            _restore_file_from_backup(final_path, final_backup_path)
                        except Exception as rollback_exc:  # noqa: BLE001
                            rollback_errors.append(
                                f"final_rollback_failed:{rollback_exc}")
                        try:
                            _restore_file_from_backup(report_path, report_backup_path)
                        except Exception as rollback_exc:  # noqa: BLE001
                            rollback_errors.append(
                                f"critic_rollback_failed:{rollback_exc}")
                        action.update(
                            status=("rolled_back" if not rollback_errors
                                    else "decision_commit_failed_rollback_partial"),
                            reason=f"decision_commit_failed:{exc}",
                        )
                        result["errors"].append(
                            f"{cid}:decision_commit_failed:{exc}")
                        result["errors"].extend(
                            f"{cid}:{item}" for item in rollback_errors)
                    else:
                        action.update(
                            status="completed_decision_commit_failed",
                            reason=(f"decision_commit_failed:{exc};"
                                    f"reopen_failed:{reopened.get('reason')}"),
                        )
                        result["errors"].append(
                            f"{cid}:decision_commit_failed:{exc};"
                            f"reopen_failed:{reopened.get('reason')}")
                    continue
            except Exception:
                try:
                    dcon.rollback()
                except sqlite3.Error:
                    pass
                raise
            finally:
                dcon.close()

            action.update(
                status="completed",
                reason="name_only_completed",
                critic_report_sha256=_sha256_file(report_path),
                evidence_path=str(evidence_path),
                pending_before=pending_entries,
                critic_after={
                    "backend_path": report.get("_backend_path"),
                    "scores": report.get("scores") or {},
                    "evidence": "exact",
                },
                decision_locked=True,
            )
        except Exception as exc:  # noqa: BLE001
            if state_completed:
                action.update(
                    status="completed_postcommit_error",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                result["errors"].append(f"{cid}:postcommit:{exc}")
                continue
            if final_written:
                try:
                    _restore_file_from_backup(final_path, final_backup_path)
                except Exception as rollback_exc:  # noqa: BLE001
                    result["errors"].append(
                        f"{cid}:final_rollback_failed:{rollback_exc}")
            if report_written:
                try:
                    _restore_file_from_backup(report_path, report_backup_path)
                except Exception as rollback_exc:  # noqa: BLE001
                    result["errors"].append(
                        f"{cid}:critic_rollback_failed:{rollback_exc}")
            action.update(status="rolled_back", reason=f"{type(exc).__name__}: {exc}")

    counts: Dict[str, int] = {}
    for action in result["actions"]:
        key = str(action.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    result["counts"] = dict(sorted(counts.items()))
    result["completed"] = counts.get("completed", 0)
    result["kept_dirty"] = counts.get("kept_dirty", 0)
    result["rolled_back"] = counts.get("rolled_back", 0)
    result["failed"] = sum(
        value for key, value in counts.items()
        if key not in {"completed", "kept_dirty"})
    result["ok"] = result["failed"] == 0 and not result["errors"]
    result["write_attempted"] = bool(candidates)
    _atomic_json(jpath, result)
    _atomic_text(mpath, _apply_markdown(result))
    return result


def _completed_repair_apply_markdown(result: Dict[str, Any]) -> str:
    lines = ["# COMPLETED 专名漂移修复（受控写入）", ""]
    lines.append("- 生成时间: %s" % result.get("generated_at"))
    lines.append("- 模式: **%s**" % result.get("mode"))
    lines.append("- ok: `%s`；写入尝试: `%s`" % (
        result.get("ok"), result.get("write_attempted")))
    counts = result.get("counts") or {}
    if counts:
        lines.append("- 动作: " + ", ".join(
            "`%s`=%s" % (key, value) for key, value in counts.items()))
    if result.get("after_counts"):
        lines.append("- 修复后 pending: " + ", ".join(
            "`%s`=%s" % (key, value)
            for key, value in result["after_counts"].items()))
    if result.get("reject_memo_skipped") or result.get("reject_memo_recorded"):
        lines.append(
            "- 拒绝备忘录: skipped=`%s`, recorded=`%s`；文件: `%s`" % (
                result.get("reject_memo_skipped"),
                result.get("reject_memo_recorded"),
                result.get("reject_memo_path")))
    actions = result.get("actions") or []
    if actions:
        lines.extend(["", "## 动作（前 60 / 共 %d）" % len(actions)])
        for action in actions[:60]:
            lines.append(
                "- `%s` %s / %s%s" % (
                    action.get("chunk_id"), action.get("status"),
                    action.get("reason"),
                    " / DIRTY" if action.get("state_after") == "DIRTY" else ""))
        if len(actions) > 60:
            lines.append("- 其余 %d 条见 JSON。" % (len(actions) - 60))
    errors = result.get("errors") or []
    if errors:
        lines.extend(["", "## 错误", ""])
        lines.extend("- %s" % error for error in errors[:40])
    memo_errors = result.get("reject_memo_errors") or []
    if memo_errors:
        lines.extend(["", "## 备忘录告警", ""])
        lines.extend("- %s" % error for error in memo_errors[:20])
    lines.append("")
    return "\n".join(lines)


def apply_completed_name_repair(
    project_root: Path,
    *,
    chapter_id: Optional[str] = None,
    apply: bool = False,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
    critic_fn: Optional[Any] = None,
    pipeline_factory: Optional[Any] = None,
    require_authority_audit: Optional[bool] = None,
    min_chars: Optional[int] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Repair a COMPLETED final's one-character CJK name drift.

    Unlike ``apply_name_only_fastpath``, this path never changes a workflow
    state.  It is only allowed when the chunk is already COMPLETED, the
    Decision DB outbox still says the canonical target is missing, and the
    proposed change is a single active-name spelling correction.  The final,
    the critic report and the outbox/decision lock are all updated together;
    any failure restores final/report and leaves the pending row intact.
    """
    root = Path(project_root)
    cfg = (_config.get("pipeline.completed_name_repair", {}) or {})
    min_chars_value = max(1, int(
        min_chars if min_chars is not None else cfg.get("min_cjk_chars", 4)))
    if not apply:
        return audit_completed_name_repair(
            root, chapter_id=chapter_id, min_chars=min_chars_value,
            json_out=json_out, md_out=md_out)

    decision_path = root / "db" / "decision_db.sqlite"
    workflow_path = root / "db" / "workflow.db"
    out_dir = root / "output"
    jpath = Path(json_out) if json_out else out_dir / "completed_name_repair_apply.json"
    mpath = Path(md_out) if md_out else out_dir / "completed_name_repair_apply.md"
    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "apply",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "chapter": chapter_id,
        "run_id": str(run_id or (
            _dt.datetime.now().strftime("completed-name-%Y%m%dT%H%M%S") +
            "-" + str(os.getpid()))).replace("/", "_"),
        "actions": [],
        "counts": {},
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if not bool(cfg.get("enabled", True)):
        result.update(ok=True, disabled=True, reason="completed_name_repair_disabled")
        _atomic_json(jpath, result)
        _atomic_text(mpath, _completed_repair_apply_markdown(result))
        return result

    require_auth = (bool(cfg.get("require_authority_audit", True))
                    if require_authority_audit is None
                    else bool(require_authority_audit))
    if require_auth:
        from src.name_authority_audit import audit_name_authorities
        authority = audit_name_authorities(root)
        result["authority"] = {
            "ok": bool(authority.get("ok")),
            "status": authority.get("status"),
            "counts": authority.get("counts") or {},
            "blocking_count": int(authority.get("blocking_count") or 0),
            "report": authority.get("md_path"),
        }
        if not authority.get("ok"):
            result.update(
                ok=False,
                authority_blocked=True,
                reason="name_authority_audit_not_clean",
            )
            _atomic_json(jpath, result)
            _atomic_text(mpath, _completed_repair_apply_markdown(result))
            return result

    plan, items_by_cid, context = _completed_repair_plan(
        root, chapter_id=chapter_id, decision_db=decision_path,
        workflow_db=workflow_path, output_dir=out_dir,
        min_chars=min_chars_value,
    )
    result["plan_counts"] = plan.get("counts") or {}
    result["planned_candidates"] = len(plan.get("candidates") or [])
    result["planned_kept"] = len(plan.get("kept") or [])
    result["reject_memo_skipped"] = int(plan.get("reject_memo_skipped") or 0)
    result["reject_memo_recorded"] = 0
    result["reject_memo_errors"] = []
    memo_path = _completed_reject_memo_path(root, out_dir)
    result["reject_memo_path"] = str(memo_path)
    reject_memo = _load_completed_reject_memo(memo_path)
    if plan.get("failed"):
        result["errors"].extend(
            "plan:%s" % item.get("reason") for item in plan.get("failed") or [])
    try:
        cap = max(0, int(
            limit if limit is not None else cfg.get("max_chunks_per_run", 50)))
    except (TypeError, ValueError):
        cap = 50
    candidates = list(plan.get("candidates") or [])[:cap]
    result["write_attempted"] = bool(candidates)
    if not candidates:
        result.update(ok=not bool(result["errors"]), no_candidates=True)
        _atomic_json(jpath, result)
        _atomic_text(mpath, _completed_repair_apply_markdown(result))
        return result
    if not decision_path.exists() or not workflow_path.exists():
        result.update(ok=False, errors=["database_missing"])
        _atomic_json(jpath, result)
        _atomic_text(mpath, _completed_repair_apply_markdown(result))
        return result

    pending_by_chunk = {
        str(entry.get("chunk_id") or ""): entry
        for entry in (classify_pending_snapshot(context).get("entries") or [])
    }
    outbox_by_chunk = {
        str(row.get("chunk_id") or ""): row
        for row in (context.get("pending") or [])
    }
    authority_corpus, authority_gram_cache = build_lock_authority_corpus(context)
    backup_root = out_dir / "completed_name_repair" / result["run_id"]
    pipeline = None
    pipelines: Dict[str, Any] = {}
    for planned in candidates:
        cid = str(planned.get("chunk_id") or "")
        action: Dict[str, Any] = {
            "chunk_id": cid,
            "chapter": planned.get("chapter"),
            "status": "failed",
            "reason": "",
            "final_sha256_before": planned.get("final_sha256_before"),
            "final_sha256_after": planned.get("final_sha256_after"),
            "decision_id": planned.get("decision_id"),
            "source_key": planned.get("source_key"),
            "canonical_name": planned.get("canonical_name"),
        }
        result["actions"].append(action)
        item = items_by_cid.get(cid)
        if item is None:
            action.update(status="kept", reason="chunk_no_longer_loaded")
            continue
        final_path = Path(item["final_path"])
        report_path = final_path.with_name("%s_critic_report.json" % cid)
        current_final = _read_json(final_path)
        current_text = str((current_final or {}).get("text") or "")
        if not current_final or not current_text:
            action.update(status="kept", reason="final_missing_or_unreadable")
            continue
        if sha256_text(current_text) != str(planned.get("final_sha256_before") or ""):
            action.update(status="kept", reason="final_changed_since_plan")
            continue
        # Re-run every mechanical gate against the bytes currently on disk.
        current_item = dict(item)
        current_item.update({
            "text": current_text,
            "final": current_final,
            "critic": _read_json(report_path),
        })
        fresh = _classify_completed_name_candidate(
            current_item, context=context,
            pending_entry=pending_by_chunk.get(cid),
            min_chars=min_chars_value,
            authority_corpus=authority_corpus,
            authority_gram_cache=authority_gram_cache,
        )
        if fresh.get("status") != "candidate":
            action.update(status="kept", reason=",".join(fresh.get("reasons") or []))
            continue
        after = _replace_name_drift_spans(
            current_text,
            fresh.get("drift_matches") or [],
            str(fresh.get("canonical_name") or ""),
        )
        if sha256_text(after) != str(fresh.get("final_sha256_after") or ""):
            action.update(status="kept", reason="predicted_final_hash_changed")
            continue
        chapter = str(fresh.get("chapter") or "")
        if critic_fn is None:
            pipeline = pipelines.get(chapter)
            if pipeline is None:
                if pipeline_factory is None:
                    from src.pipeline import TranslationPipeline
                    pipeline_factory = lambda ch: TranslationPipeline(chapter_id=ch)
                pipeline = pipeline_factory(chapter)
                pipelines[chapter] = pipeline
        raw_payload = _read_json(final_path.with_name("%s_raw.json" % cid)) or {}
        raw_text = str(raw_payload.get("text") or current_text)
        source_text = str(item["task"].get("text_content") or "")
        try:
            if critic_fn is None:
                report = _default_critic_fn(
                    fresh, source_text, raw_text, after, pipeline)
            else:
                report = critic_fn(fresh, source_text, raw_text, after, pipeline)
        except Exception as exc:  # noqa: BLE001 - no final has been written yet
            action.update(status="kept", reason="critic_reaudit_failed:%s" % exc)
            continue
        critic_reasons = _validate_reauthored_report(report, proposed_text=after)
        if critic_reasons:
            action.update(status="kept", reason=",".join(critic_reasons))
            action["critic_after"] = {
                "backend_path": (report or {}).get("_backend_path"),
                "scores": (report or {}).get("scores") or {},
            }
            if _stable_critic_rejection(report, critic_reasons):
                try:
                    action["reject_memo_key"] = _record_completed_reject(
                        memo_path, reject_memo, fresh, critic_reasons,
                        run_id=str(result["run_id"]),
                    )
                    result["reject_memo_recorded"] += 1
                except Exception as memo_exc:  # noqa: BLE001 - memo is an optimization
                    result["reject_memo_errors"].append(
                        "%s:%s" % (cid, memo_exc))
            continue

        after_sha = sha256_text(after)
        report = dict(report or {})
        report["_reaudit"] = {
            "tool": "completed-name-repair",
            "schema_version": 1,
            "at": _now_iso(),
            "previous_backend_path": (current_item.get("critic") or {}).get(
                "_backend_path"),
            "final_sha256": after_sha,
            "raw_input": "raw" if raw_payload else "final_before",
            "literary_input": "proposed_final",
        }
        report["_final_binding"] = {
            "schema_version": 1,
            "final_sha256": after_sha,
            "before_sha256": sha256_text(current_text),
            "reason": "completed-name-repair real critic re-audit",
        }
        report["_audit_target"] = {
            "schema_version": 1,
            "source_sha256": sha256_text(source_text),
            "raw_sha256": sha256_text(raw_text),
            "literary_sha256": after_sha,
        }

        backup_dir = backup_root / chapter
        backup_dir.mkdir(parents=True, exist_ok=True)
        final_backup = backup_dir / ("%s_final_before.json" % cid)
        report_backup = backup_dir / ("%s_critic_report_before.json" % cid)
        evidence_path = backup_dir / ("%s_completed_name_repair_evidence.json" % cid)
        _copy_file_atomic(final_path, final_backup)
        _copy_file_atomic(report_path, report_backup)
        new_final = dict(current_final)
        metadata = dict(new_final.get("metadata") or {})
        metadata.update({
            "completed_name_repair_run_id": result["run_id"],
            "completed_name_repair_at": _now_iso(),
            "completed_name_repair_before_sha256": sha256_text(current_text),
            "completed_name_repair_decision_id": fresh.get("decision_id"),
            "completed_name_repair_forms": fresh.get("wrong_forms") or [],
        })
        new_final["text"] = after
        new_final["metadata"] = metadata
        final_written = False
        report_written = False
        evidence_written = False
        decision_done = False
        try:
            _atomic_json_fsync(final_path, new_final)
            final_written = True
            _atomic_json_fsync(report_path, report)
            report_written = True
            evidence = {
                "schema_version": 1,
                "chunk_id": cid,
                "status": "verified_completed_name_repair",
                "run_id": result["run_id"],
                "state_before": "COMPLETED",
                "state_after": "COMPLETED",
                "decision_id": fresh.get("decision_id"),
                "source_key": fresh.get("source_key"),
                "canonical_name": fresh.get("canonical_name"),
                "wrong_forms": fresh.get("wrong_forms") or [],
                "final_sha256_before": sha256_text(current_text),
                "final_sha256_after": after_sha,
                "critic_final_sha256": after_sha,
                "critic_report_sha256": _sha256_file(report_path),
                "pending_before": fresh.get("pending_before") or [],
                "pending_checked": True,
                "applied_decisions": fresh.get("applied_decisions") or [],
                "lock_authority": fresh.get("lock_authority") or {},
                "changed_spans": fresh.get("changed_spans") or [],
            }
            _atomic_json_fsync(evidence_path, evidence)
            evidence_written = True

            dcon = _open_decision_write_db(decision_path)
            try:
                dcon.execute("BEGIN IMMEDIATE")
                outbox = outbox_by_chunk.get(cid)
                if outbox is None:
                    raise RuntimeError("pending_row_missing_at_apply")
                if not _delete_pending_row_if_unchanged(dcon, outbox):
                    raise RuntimeError("pending_cas_failed")
                cursor = dcon.execute(
                    "UPDATE decision_db SET locked = 1 WHERE id = ?",
                    (int(fresh["decision_id"]),),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("decision_lock_cas_failed")
                dcon.commit()
                decision_done = True
            finally:
                dcon.close()

            # The completed state is an invariant, not a side effect of this
            # repair.  Verify it after the cross-DB transaction committed.
            wcon = sqlite3.connect(
                "file:%s?mode=ro" % workflow_path.resolve(), uri=True)
            try:
                row = wcon.execute(
                    "SELECT state, rescue_stage FROM chunk_tasks WHERE chunk_id=?",
                    (cid,)).fetchone()
            finally:
                wcon.close()
            state_after = str(row[0]) if row else "MISSING"
            action["state_after"] = state_after
            if state_after != "COMPLETED" or (row and row[1]):
                raise RuntimeError("completed_state_changed:%s" % state_after)
            action.update(
                status="repaired",
                reason="completed_name_repaired",
                evidence_path=str(evidence_path),
                critic_report_sha256=_sha256_file(report_path),
                critic_after={
                    "backend_path": report.get("_backend_path"),
                    "scores": report.get("scores") or {},
                    "evidence": "exact",
                },
                decision_locked=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep evidence truthfully
            if decision_done:
                action.update(
                    status="repaired_postcommit_error",
                    reason="%s: %s" % (type(exc).__name__, exc),
                )
                result["errors"].append("%s:postcommit:%s" % (cid, exc))
                continue
            for flag, path, backup, label in (
                (final_written, final_path, final_backup, "final"),
                (report_written, report_path, report_backup, "critic"),
            ):
                if not flag:
                    continue
                try:
                    _restore_file_from_backup(path, backup)
                except Exception as rollback_exc:  # noqa: BLE001
                    result["errors"].append(
                        "%s:%s_rollback_failed:%s" % (cid, label, rollback_exc))
            if evidence_written:
                try:
                    evidence_path.unlink()
                except OSError:
                    pass
            action.update(
                status="rolled_back",
                reason="%s: %s" % (type(exc).__name__, exc),
            )

    counts: Dict[str, int] = {}
    for action in result["actions"]:
        key = str(action.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    result["counts"] = dict(sorted(counts.items()))
    result["repaired"] = counts.get("repaired", 0)
    result["kept"] = counts.get("kept", 0)
    result["failed_count"] = sum(
        value for key, value in counts.items()
        if key not in {"repaired", "kept"})
    try:
        after_context = load_pending_context(
            root, decision_db=decision_path, workflow_db=workflow_path,
            output_dir=out_dir,
        )
        after_classification = classify_pending_snapshot(after_context)
        result["after_counts"] = dict(after_classification.get("counts") or {})
        result["pending_ok"] = bool(after_classification.get("ok"))
    except Exception as exc:  # noqa: BLE001 - report failure without hiding it
        result["errors"].append("post_audit:%s" % exc)
    result["ok"] = result["failed_count"] == 0 and not result["errors"]
    _atomic_json(jpath, result)
    _atomic_text(mpath, _completed_repair_apply_markdown(result))
    return result


def _confirmed_term_plan_markdown(result: Dict[str, Any]) -> str:
    lines = ["# confirmed specific term drift repair", ""]
    lines.append("- 生成时间: `%s`" % result.get("generated_at"))
    lines.append("- 模式: **%s**" % result.get("mode"))
    lines.append("- 统计: `%s`" % (result.get("counts") or {}))
    candidates = result.get("candidates") or []
    if candidates:
        lines.extend(["", "## 候选组"])
        for item in candidates:
            lines.append(
                "- Decision #%s `%s` → `%s`; wrong=`%s`; chunks=%s"
                % (item.get("decision_id"), item.get("source_key"),
                   item.get("canonical_name"), item.get("wrong_form"),
                   ",".join(str(chunk.get("chunk_id") or "")
                            for chunk in item.get("chunks") or [])))
    blocked = result.get("blocked") or []
    if blocked:
        lines.extend(["", "## 阻断组"])
        for item in blocked[:40]:
            lines.append(
                "- Decision #%s `%s` → `%s`: `%s`"
                % (item.get("decision_id"), item.get("source_key"),
                   item.get("canonical_name"), item.get("reason")))
        if len(blocked) > 40:
            lines.append("- 其余 %d 组见 JSON。" % (len(blocked) - 40))
    residual = result.get("residual") or []
    if residual:
        lines.extend(["", "## 无需修复 / 延后"])
        for item in residual[:40]:
            lines.append(
                "- Decision #%s `%s`: `%s`"
                % (item.get("decision_id"), item.get("source_key"),
                   item.get("reason")))
        if len(residual) > 40:
            lines.append("- 其余 %d 组见 JSON。" % (len(residual) - 40))
    if result.get("errors"):
        lines.extend(["", "## 错误"])
        lines.extend("- `%s`" % error for error in result["errors"][:40])
    lines.extend([
        "", "本命令只接受锁目标权威审计中的 `confirmed + specific` 组。"
        "错误形必须在每个 source-positive 块中与 source 出现次数完全一致，"
        "且不得到达其他活跃专名的保护集。", ""])
    return "\n".join(lines)


def _gloss_term_from_raw_forms(
    forms: List[str],
    target_key: str,
) -> str:
    """Recover the glossed term without swallowing preceding Chinese prose.

    A CJK run immediately before ``（Source）`` has no whitespace boundary, so a
    regex cannot know where the term starts.  Repeated glosses provide the
    evidence: the shorter surface form (usually the footnote form) is a suffix
    of the longer body form.  With only one occurrence, accept the raw form
    only when no trimming is needed; a welded context is ambiguous and must
    fall back to a normal rerun instead of deleting prose.
    """
    values = [str(form or "") for form in forms if str(form or "")]
    if not values:
        return ""
    if len(values) == 1:
        raw = values[0]
        return raw if _best_glossed_term_form(raw, target_key) == raw else ""
    shortest = min(values, key=len)
    if all(value.endswith(shortest) for value in values):
        term = shortest
    else:
        term = shortest
        for candidate_length in range(len(shortest), 0, -1):
            suffix = shortest[-candidate_length:]
            if all(value.endswith(suffix) for value in values):
                term = suffix
                break
    if len(term) < 2 or is_likely_common_word(term):
        return ""
    if term[0] in "的了着过在和与把被向从对将给为是个一":
        return ""
    return term


def _confirmed_term_chunk_occurrences(
    source_text: str,
    final_text: str,
    *,
    source_key: str,
    target: str,
    candidate_rows: List[dict],
    protected_keys: set,
) -> Tuple[int, List[Dict[str, Any]]]:
    source_count = _confirmed_term_source_count(source_text, source_key)
    if source_count <= 0:
        return 0, []
    body = _body_before_footnotes(final_text)
    target_key = _name_key(target)
    if target_key and target_key in _name_key(body):
        return source_count, []
    matches: List[Dict[str, Any]] = []
    for row in candidate_rows:
        if int(row.get("negative_chunks") or 0):
            continue
        form = str(row.get("form") or "").strip()
        normalized = _name_key(form)
        if (not form or not normalized or normalized == target_key
                or normalized in protected_keys
                or any(normalized in key or key in normalized
                       for key in protected_keys if key)):
            continue
        if not all(re.fullmatch(r"[\u4e00-\u9fff]", char)
                   for char in normalized):
            continue
        if len(normalized) < 2 or is_likely_common_word(form):
            continue
        all_spans = [match.span() for match in re.finditer(
            re.escape(form), final_text)]
        body_spans = [match.span() for match in re.finditer(
            re.escape(form), body)]
        if len(all_spans) != source_count or len(body_spans) != source_count:
            continue
        matches.append({
            "start": all_spans[0][0],
            "end": all_spans[0][1],
            "form": form,
            "normalized": normalized,
            "source_count": source_count,
            "spans": all_spans,
            "authority_positive_chunks": int(row.get("positive_chunks") or 0),
            "authority_similarity": float(row.get("similarity_to_target") or 0.0),
            "chunk_id": str(row.get("chunk_id") or ""),
        })
    # A translator gloss is independent, exact provenance: ``护盾墙（Shield
    # Wall）`` binds the Chinese form to this source phrase even when the English
    # phrase was wrapped across a newline in the source.  Accept it only when
    # every occurrence of that CJK form in the final is glossed, so a common
    # unglossed word cannot enter through this route.
    source_pattern = re.compile(
        r"(?<![A-Za-z0-9])"
        + r"\s+".join(re.escape(part) for part in source_key.split())
        + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    gloss_groups: Dict[str, List[Tuple[str, Tuple[int, int]]]] = {}
    for gloss in re.finditer(
            r"(?P<form>[\u4e00-\u9fff]{2,12})"
            r"[\"'”’」』]*\s*[（(]\s*(?P<source>[^）)]+?)\s*[）)]",
            final_text):
        if not source_pattern.fullmatch(gloss.group("source").strip()):
            continue
        gloss_groups.setdefault(
            _phrase_key(gloss.group("source")), []
        ).append((gloss.group("form"), gloss.span("form")))

    glossed: Dict[str, List[Tuple[int, int]]] = {}
    for raw_forms in gloss_groups.values():
        forms = [form for form, _span in raw_forms]
        term = _gloss_term_from_raw_forms(forms, target_key)
        normalized = _name_key(term)
        if (not term or not normalized or normalized == target_key
                or normalized in protected_keys
                or any(normalized in key or key in normalized
                       for key in protected_keys if key)
                or is_likely_common_word(term)):
            continue
        term_spans: List[Tuple[int, int]] = []
        for raw, (_start, end) in raw_forms:
            if not raw.endswith(term):
                term_spans = []
                break
            start = end - len(term)
            if final_text[start:end] != term:
                term_spans = []
                break
            term_spans.append((start, end))
        if not term_spans:
            continue
        glossed.setdefault(term, []).extend(term_spans)
    for form, spans in glossed.items():
        all_spans = [match.span() for match in re.finditer(
            re.escape(form), final_text)]
        if not all_spans or len(spans) != len(all_spans):
            continue
        if len(spans) < source_count or not source_count:
            continue
        normalized = _name_key(form)
        matches.append({
            "start": spans[0][0],
            "end": spans[0][1],
            "form": form,
            "normalized": normalized,
            "source_count": source_count,
            "spans": all_spans,
            "gloss_bound": True,
            "authority_positive_chunks": 0,
            "authority_similarity": 0.0,
            "chunk_id": "",
        })
    if gloss_groups:
        # An explicit ``term（Source）`` gloss is stronger evidence than a
        # shorter authority form that happens to occur inside the glossed CJK
        # run.  If the gloss term cannot be recovered safely, fail closed rather
        # than falling back to that substring (``护盾墙`` must not become the
        # two-character candidate ``盾墙`` with a dangling ``护`` prefix).
        if not glossed:
            return source_count, []
        matches = [
            item for item in matches if bool(item.get("gloss_bound"))
        ]
    forms = {str(item.get("normalized") or "") for item in matches}
    if len(matches) != 1 or len(forms) != 1:
        return source_count, []
    return source_count, matches


def _confirmed_term_source_count(text: str, source_key: str) -> int:
    """Count a Decision phrase with source line wraps normalized."""
    parts = [part for part in str(source_key or "").split() if part]
    body = _body_before_footnotes(text)
    if len(parts) < 1 or not body:
        return 0
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + r"\s+".join(
            re.escape(part) for part in parts) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return len(pattern.findall(body))


def _confirmed_term_group_plan(
    group: Dict[str, Any],
    *,
    context: Dict[str, Any],
    snapshot_by_chunk: Dict[str, dict],
    out_dir: Path,
    chapter_id: Optional[str],
) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
    decision_id = int(group.get("decision_id") or 0)
    source_key = str(group.get("source_key") or "")
    target = _term_target_core(group.get("target"))
    target_key = _name_key(target)
    base = {
        "decision_id": decision_id,
        "source_key": source_key,
        "canonical_name": target,
        "authority": {
            "classification": group.get("classification"),
            "reason": group.get("reason"),
            "source_positive_chunks": int(group.get("source_positive_chunks") or 0),
            "authority_chunks": int(group.get("authority_chunks") or 0),
            "target_hits": int(group.get("target_hits") or 0),
        },
    }
    if (group.get("classification") != "confirmed"
            or group.get("authority_kind") != "specific"
            or group.get("pending_reason") not in {
                "specific_decision_target_missing",
                "specific_decision_target_missing_no_pending",
            }):
        base["reason"] = "group_not_confirmed_specific"
        return None, None, base
    if (not source_key or not target or not target_key
            or not all(re.fullmatch(r"[\u4e00-\u9fff]", char)
                       for char in target_key)):
        base["reason"] = "target_not_pure_cjk"
        return None, None, base
    decision = (context.get("decisions") or {}).get(decision_id)
    if not decision:
        base["reason"] = "decision_missing"
        return None, base, None
    if (str(decision.get("source_key") or "") != source_key
            or _name_key(decision.get("translation") or "") != target_key):
        base["reason"] = "decision_target_changed"
        return None, base, None

    protected = _protected_active_name_keys(context, target_key)
    candidate_rows = list(group.get("candidates") or [])
    chunk_candidates: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    for cid, task in sorted((context.get("tasks") or {}).items()):
        task_chapter = str(task.get("chapter_id") or "")
        if chapter_id and task_chapter != chapter_id:
            continue
        source_text = str(task.get("text_content") or "")
        source_count = _confirmed_term_source_count(source_text, source_key)
        if source_count <= 0:
            continue
        state = str(task.get("state") or "")
        final_path = out_dir / task_chapter / ("%s_final.json" % cid)
        final_payload = _read_json(final_path)
        final_text = str((final_payload or {}).get("text") or "")
        if not final_text.strip():
            if state == "COMPLETED":
                base["reason"] = "final_missing:%s" % cid
                return None, base, None
            deferred.append({"chunk_id": cid, "reason": "final_missing"})
            continue
        if target_key in _name_key(_body_before_footnotes(final_text)):
            continue
        if state != "COMPLETED":
            deferred.append({
                "chunk_id": cid,
                "state": state,
                "reason": "noncompleted_missing_target",
            })
            continue
        if task.get("rescue_stage"):
            base["reason"] = "rescue_stage:%s" % cid
            return None, base, None
        metadata = (final_payload or {}).get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("fallback") or metadata.get("fallback_used"):
            base["reason"] = "final_fallback:%s" % cid
            return None, base, None
        if metadata.get("quality_waived"):
            base["reason"] = "final_quality_waived:%s" % cid
            return None, base, None
        pending = snapshot_by_chunk.get(cid)
        if pending and str(pending.get("reason") or "") != (
                "specific_decision_target_missing"):
            base["reason"] = "pending_reason:%s" % (
                pending.get("reason") or "missing")
            return None, base, None
        resolved_source_count, matches = _confirmed_term_chunk_occurrences(
            source_text,
            final_text,
            source_key=source_key,
            target=target,
            candidate_rows=candidate_rows,
            protected_keys=protected,
        )
        if resolved_source_count != source_count or len(matches) != 1:
            base["reason"] = "no_unique_counted_candidate:%s" % cid
            return None, base, None
        match = matches[0]
        replacement_matches = [
            {"start": start, "end": end, "form": match.get("form"),
             "normalized": match.get("normalized")}
            for start, end in match.get("spans") or []
        ]
        proposed_text = _replace_name_drift_spans(
            final_text, replacement_matches, target)
        wrong_form = str(match.get("form") or "")
        if (wrong_form in proposed_text or target not in proposed_text):
            base["reason"] = "proposed_text_invariant_failed:%s" % cid
            return None, base, None
        chunk_candidates.append({
            "chunk_id": cid,
            "chapter": task_chapter,
            "state_before": state,
            "final_path": str(final_path),
            "final_sha256_before": sha256_text(final_text),
            "final_sha256_after": sha256_text(proposed_text),
            "source_count": source_count,
            "wrong_form": wrong_form,
            "gloss_bound": bool(match.get("gloss_bound")),
            "wrong_normalized": str(match.get("normalized") or ""),
            "changed_spans": _changed_spans(final_text, proposed_text),
            "pending_before": [pending] if pending else [],
            "critic_sha256": _stable_payload_sha256(
                _read_json(final_path.with_name(
                    "%s_critic_report.json" % cid)) or {}),
            "authority": {
                "positive_chunks": int(match.get("authority_positive_chunks") or 0),
                "similarity_to_target": float(
                    match.get("authority_similarity") or 0.0),
            },
        })
    if not chunk_candidates:
        base["reason"] = "no_completed_source_positive_missing_target"
        return None, None, base
    wrong_forms = sorted({str(item.get("wrong_form") or "")
                          for item in chunk_candidates if item.get("wrong_form")})
    if deferred:
        base["deferred"] = deferred
    candidate = dict(base)
    candidate.update({
        "wrong_form": ", ".join(wrong_forms),
        "wrong_forms": wrong_forms,
        "chunks": chunk_candidates,
    })
    return candidate, None, None


def audit_confirmed_term_drift_repair(
    project_root: Path,
    *,
    chapter_id: Optional[str] = None,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Plan repairs for confirmed specific-term drift without writing prose."""
    root = Path(project_root)
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "confirmed_term_drift_repair.json"
    mpath = Path(md_out) if md_out else out_dir / "confirmed_term_drift_repair.md"
    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "chapter": chapter_id,
        "counts": {},
        "candidates": [],
        "blocked": [],
        "residual": [],
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
        "ok": False,
    }
    context = load_pending_context(
        root, decision_db=decision_db, workflow_db=workflow_db,
        output_dir=out_dir,
    )
    if not context.get("available"):
        result["errors"].append(
            str(context.get("reason") or "pending_context_unavailable"))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _confirmed_term_plan_markdown(result))
        return result
    snapshot = classify_pending_snapshot(context)
    snapshot_by_chunk = {
        str(entry.get("chunk_id") or ""): entry
        for entry in snapshot.get("entries") or []
    }
    from src.locked_target_authority_audit import audit_locked_target_authorities
    authority = audit_locked_target_authorities(
        root, decision_db=decision_db, workflow_db=workflow_db,
        output_dir=out_dir, include_marked_repair_sweep=True,
    )
    result["authority_counts"] = authority.get("counts") or {}
    if authority.get("errors"):
        result["errors"].extend(
            "authority:%s" % error for error in authority.get("errors") or [])
    for group in authority.get("groups") or []:
        candidate, blocked, residual = _confirmed_term_group_plan(
            group, context=context, snapshot_by_chunk=snapshot_by_chunk,
            out_dir=out_dir, chapter_id=chapter_id,
        )
        if candidate is not None:
            result["candidates"].append(candidate)
        if blocked is not None:
            result["blocked"].append(blocked)
        if residual is not None:
            result["residual"].append(residual)
    result["counts"] = {
        "authority_groups": len(authority.get("groups") or []),
        "candidate_groups": len(result["candidates"]),
        "candidate_chunks": sum(
            len(item.get("chunks") or []) for item in result["candidates"]),
        "blocked_groups": len(result["blocked"]),
        "residual_groups": len(result["residual"]),
        "errors": len(result["errors"]),
    }
    result["ok"] = not result["errors"]
    _atomic_json(jpath, result)
    _atomic_text(mpath, _confirmed_term_plan_markdown(result))
    return result


def apply_confirmed_term_drift_repair(
    project_root: Path,
    *,
    chapter_id: Optional[str] = None,
    apply: bool = False,
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
    critic_fn: Optional[Any] = None,
    pipeline_factory: Optional[Any] = None,
    require_authority_audit: Optional[bool] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Apply a confirmed specific-term drift group atomically.

    Every affected chunk must pass a fresh calibrated critic re-audit before
    any prose is written.  The final, critic report, evidence, pending CAS and
    Decision lock are one group transaction; a failure restores every final and
    report written by this group.
    """
    root = Path(project_root)
    out_dir = root / "output"
    jpath = Path(json_out) if json_out else out_dir / "confirmed_term_drift_repair_apply.json"
    mpath = Path(md_out) if md_out else out_dir / "confirmed_term_drift_repair_apply.md"
    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "apply" if apply else "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "chapter": chapter_id,
        "run_id": str(run_id or (
            _dt.datetime.now().strftime("confirmed-term-%Y%m%dT%H%M%S") +
            "-" + str(os.getpid()))),
        "actions": [],
        "counts": {},
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
        "ok": False,
    }
    if not apply:
        plan = audit_confirmed_term_drift_repair(
            root, chapter_id=chapter_id, output_dir=out_dir,
            json_out=jpath, md_out=mpath,
        )
        plan["mode"] = "read_only"
        return plan
    cfg = (_config.get("pipeline.confirmed_term_drift_repair", {}) or {})
    if not bool(cfg.get("enabled", True)):
        result.update(ok=True, disabled=True,
                      reason="confirmed_term_drift_repair_disabled")
        _atomic_json(jpath, result)
        _atomic_text(mpath, _confirmed_term_plan_markdown(result))
        return result

    require_auth = (bool(cfg.get("require_authority_audit", True))
                    if require_authority_audit is None
                    else bool(require_authority_audit))
    if require_auth:
        from src.name_authority_audit import audit_name_authorities
        authority = audit_name_authorities(root)
        result["authority"] = {
            "ok": bool(authority.get("ok")),
            "status": authority.get("status"),
            "counts": authority.get("counts") or {},
            "blocking_count": int(authority.get("blocking_count") or 0),
            "report": authority.get("md_path"),
        }
        if not authority.get("ok"):
            result.update(
                ok=False,
                authority_blocked=True,
                reason="name_authority_audit_not_clean",
            )
            _atomic_json(jpath, result)
            _atomic_text(mpath, _confirmed_term_plan_markdown(result))
            return result

    plan = audit_confirmed_term_drift_repair(
        root, chapter_id=chapter_id, output_dir=out_dir,
        json_out=(out_dir / "confirmed_term_drift_repair.json"),
        md_out=(out_dir / "confirmed_term_drift_repair.md"),
    )
    result["plan_counts"] = plan.get("counts") or {}
    result["blocked"] = plan.get("blocked") or []
    result["residual"] = plan.get("residual") or []
    result["errors"].extend(plan.get("errors") or [])
    candidates = list(plan.get("candidates") or [])
    try:
        cap = max(0, int(
            limit if limit is not None else cfg.get("max_groups_per_run", 10)))
    except (TypeError, ValueError):
        cap = 10
    candidates = candidates[:cap]
    if plan.get("errors") or plan.get("blocked"):
        result["ok"] = False
        _atomic_json(jpath, result)
        _atomic_text(mpath, _confirmed_term_plan_markdown(result))
        return result
    if not candidates:
        result.update(ok=True, no_candidates=True)
        _atomic_json(jpath, result)
        _atomic_text(mpath, _confirmed_term_plan_markdown(result))
        return result

    decision_path = root / "db" / "decision_db.sqlite"
    workflow_path = root / "db" / "workflow.db"
    context = load_pending_context(
        root, decision_db=decision_path, workflow_db=workflow_path,
        output_dir=out_dir,
    )
    raw_pending = {
        str(row.get("chunk_id") or ""): row
        for row in context.get("pending") or []
    }
    pipelines: Dict[str, Any] = {}
    backup_root = out_dir / "confirmed_term_drift_repair" / result["run_id"]
    for candidate in candidates:
        action: Dict[str, Any] = {
            "decision_id": candidate.get("decision_id"),
            "source_key": candidate.get("source_key"),
            "canonical_name": candidate.get("canonical_name"),
            "wrong_form": candidate.get("wrong_form"),
            "status": "failed",
            "reason": "",
            "chunks": [],
        }
        result["actions"].append(action)
        prepared: List[Dict[str, Any]] = []
        for chunk in candidate.get("chunks") or []:
            cid = str(chunk.get("chunk_id") or "")
            chapter = str(chunk.get("chapter") or "")
            final_path = out_dir / chapter / ("%s_final.json" % cid)
            report_path = out_dir / chapter / ("%s_critic_report.json" % cid)
            current_final = _read_json(final_path)
            current_text = str((current_final or {}).get("text") or "")
            if not current_final or sha256_text(current_text) != str(
                    chunk.get("final_sha256_before") or ""):
                action.update(status="kept", reason="final_changed_since_plan")
                break
            task = (context.get("tasks") or {}).get(cid) or {}
            if str(task.get("state") or "") != "COMPLETED":
                action.update(status="kept", reason="state_changed_since_plan")
                break
            matches = [{
                "start": int(span.get("start")),
                "end": int(span.get("end")),
                "form": chunk.get("wrong_form"),
                "normalized": chunk.get("wrong_normalized"),
            } for span in chunk.get("changed_spans") or []]
            # Recover exact spans from the current bytes instead of trusting the
            # flattened report summary.
            wrong = str(chunk.get("wrong_form") or "")
            current_matches = [
                {"start": match.start(), "end": match.end(),
                 "form": match.group(0), "normalized": _name_key(match.group(0))}
                for match in re.finditer(re.escape(wrong), current_text)
            ]
            proposed_text = _replace_name_drift_spans(
                current_text, current_matches,
                str(candidate.get("canonical_name") or ""))
            if sha256_text(proposed_text) != str(
                    chunk.get("final_sha256_after") or ""):
                action.update(status="kept", reason="proposed_hash_changed")
                break
            raw_payload = _read_json(final_path.with_name("%s_raw.json" % cid)) or {}
            raw_text = str(raw_payload.get("text") or current_text)
            pipeline = None
            if critic_fn is None:
                pipeline = pipelines.get(chapter)
                if pipeline is None:
                    if pipeline_factory is None:
                        from src.pipeline import TranslationPipeline
                        pipeline_factory = lambda ch: TranslationPipeline(chapter_id=ch)
                    pipeline = pipeline_factory(chapter)
                    pipelines[chapter] = pipeline
            critic_record = dict(chunk)
            critic_record.update({
                "decision_id": candidate.get("decision_id"),
                "source_key": candidate.get("source_key"),
                "canonical_name": candidate.get("canonical_name"),
            })
            try:
                if critic_fn is None:
                    report = _default_critic_fn(
                        critic_record, str(task.get("text_content") or ""),
                        raw_text, proposed_text, pipeline)
                else:
                    report = critic_fn(
                        critic_record, str(task.get("text_content") or ""),
                        raw_text, proposed_text, pipeline)
            except Exception as exc:  # noqa: BLE001 - no prose written yet
                action.update(status="kept",
                              reason="critic_reaudit_failed:%s" % exc)
                break
            critic_reasons = _validate_reauthored_report(
                report, proposed_text=proposed_text)
            if critic_reasons:
                action.update(status="kept", reason=",".join(critic_reasons))
                action["critic_after"] = {
                    "backend_path": (report or {}).get("_backend_path"),
                    "scores": (report or {}).get("scores") or {},
                }
                break
            prepared.append({
                "chunk": chunk,
                "cid": cid,
                "chapter": chapter,
                "final_path": final_path,
                "report_path": report_path,
                "current_final": current_final,
                "current_text": current_text,
                "proposed_text": proposed_text,
                "proposed_sha256": sha256_text(proposed_text),
                "report": dict(report or {}),
                "source_text": str(task.get("text_content") or ""),
                "raw_text": raw_text,
                "raw_present": bool(raw_payload),
            })
        else:
            # Python for/else: all chunks passed re-audit.
            if prepared:
                written: List[Tuple[Path, Path]] = []
                evidence_paths: List[Path] = []
                final_done = False
                db_done = False
                try:
                    backup_dir = backup_root / str(
                        candidate.get("decision_id"))
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    for item in prepared:
                        final_backup = backup_dir / (
                            "%s_final_before.json" % item["cid"])
                        report_backup = backup_dir / (
                            "%s_critic_report_before.json" % item["cid"])
                        _copy_file_atomic(item["final_path"], final_backup)
                        _copy_file_atomic(item["report_path"], report_backup)
                        written.append((item["final_path"], final_backup))
                        written.append((item["report_path"], report_backup))
                    for item in prepared:
                        report = item["report"]
                        report["_reaudit"] = {
                            "tool": "confirmed-term-drift-repair",
                            "schema_version": 1,
                            "at": _now_iso(),
                            "previous_backend_path": (
                                _read_json(item["report_path"]) or {}).get(
                                    "_backend_path"),
                            "final_sha256": item["proposed_sha256"],
                            "raw_input": "raw" if item["raw_present"] else "final_before",
                            "literary_input": "proposed_final",
                        }
                        report["_final_binding"] = {
                            "schema_version": 1,
                            "final_sha256": item["proposed_sha256"],
                            "before_sha256": sha256_text(item["current_text"]),
                            "reason": "confirmed term drift real critic re-audit",
                        }
                        report["_audit_target"] = {
                            "schema_version": 1,
                            "source_sha256": sha256_text(item["source_text"]),
                            "raw_sha256": sha256_text(item["raw_text"]),
                            "literary_sha256": item["proposed_sha256"],
                        }
                        new_final = dict(item["current_final"])
                        metadata = dict(new_final.get("metadata") or {})
                        metadata.update({
                            "confirmed_term_drift_repair_run_id": result["run_id"],
                            "confirmed_term_drift_repair_at": _now_iso(),
                            "confirmed_term_drift_repair_decision_id": (
                                candidate.get("decision_id")),
                            "confirmed_term_drift_repair_forms": [
                                item["chunk"].get("wrong_form")],
                        })
                        new_final["text"] = item["proposed_text"]
                        new_final["metadata"] = metadata
                        _atomic_json_fsync(item["final_path"], new_final)
                        _atomic_json_fsync(item["report_path"], report)
                        evidence_path = backup_dir / (
                            "%s_confirmed_term_drift_evidence.json" % item["cid"])
                        _atomic_json_fsync(evidence_path, {
                            "schema_version": 1,
                            "chunk_id": item["cid"],
                            "status": "verified_confirmed_term_drift_repair",
                            "run_id": result["run_id"],
                            "state_before": "COMPLETED",
                            "state_after": "COMPLETED",
                            "decision_id": candidate.get("decision_id"),
                            "source_key": candidate.get("source_key"),
                            "canonical_name": candidate.get("canonical_name"),
                            "wrong_form": item["chunk"].get("wrong_form"),
                            "gloss_bound": bool(item["chunk"].get("gloss_bound")),
                            "source_count": item["chunk"].get("source_count"),
                            "final_sha256_before": sha256_text(item["current_text"]),
                            "final_sha256_after": item["proposed_sha256"],
                            "critic_report_sha256": _sha256_file(item["report_path"]),
                            "pending_before": item["chunk"].get("pending_before") or [],
                            "changed_spans": item["chunk"].get("changed_spans") or [],
                            "lock_authority": candidate.get("authority") or {},
                        })
                        evidence_paths.append(evidence_path)
                    final_done = True

                    wcon = sqlite3.connect(
                        "file:%s?mode=ro" % workflow_path.resolve(), uri=True)
                    try:
                        for item in prepared:
                            row = wcon.execute(
                                "SELECT state, rescue_stage FROM chunk_tasks "
                                "WHERE chunk_id=?", (item["cid"],)).fetchone()
                            if not row or str(row[0]) != "COMPLETED" or row[1]:
                                raise RuntimeError(
                                    "completed_state_changed:%s" % item["cid"])
                    finally:
                        wcon.close()

                    dcon = _open_decision_write_db(decision_path)
                    try:
                        dcon.execute("BEGIN IMMEDIATE")
                        for item in prepared:
                            outbox = raw_pending.get(item["cid"])
                            if outbox is None:
                                continue
                            if not _delete_pending_row_if_unchanged(dcon, outbox):
                                raise RuntimeError(
                                    "pending_cas_failed:%s" % item["cid"])
                        did = int(candidate.get("decision_id") or 0)
                        row = dcon.execute(
                            "SELECT source_key, translation, reason, locked "
                            "FROM decision_db WHERE id=?", (did,)).fetchone()
                        if not row:
                            raise RuntimeError("decision_missing:%s" % did)
                        if str(row[0]) != str(candidate.get("source_key") or ""):
                            raise RuntimeError("decision_source_changed:%s" % did)
                        if _name_key(row[1]) != _name_key(
                                candidate.get("canonical_name") or ""):
                            raise RuntimeError("decision_target_changed:%s" % did)
                        reason = str(row[2] or "")
                        marker = "Round 158 confirmed term drift lock: %s" % (
                            candidate.get("canonical_name"))
                        if marker not in reason:
                            reason = (reason.rstrip() + " | " + marker).strip(" |")
                        cursor = dcon.execute(
                            "UPDATE decision_db SET locked=1, reason=? "
                            "WHERE id=? AND source_key=? AND translation=?",
                            (reason, did, candidate.get("source_key"),
                             candidate.get("canonical_name")))
                        if cursor.rowcount != 1:
                            raise RuntimeError("decision_lock_cas_failed:%s" % did)
                        dcon.commit()
                        db_done = True
                    finally:
                        dcon.close()
                    action.update(
                        status="repaired",
                        reason="confirmed_term_drift_repaired",
                        decision_locked=True,
                        chunks=[{
                            "chunk_id": item["cid"],
                            "final_sha256_before": sha256_text(item["current_text"]),
                            "final_sha256_after": item["proposed_sha256"],
                            "evidence_path": str(evidence_paths[index]),
                            "critic_backend_path": item["report"].get("_backend_path"),
                        } for index, item in enumerate(prepared)],
                    )
                except Exception as exc:  # noqa: BLE001 - rollback group writes
                    if db_done:
                        action.update(
                            status="repaired_postcommit_error",
                            reason="%s: %s" % (type(exc).__name__, exc))
                        result["errors"].append(
                            "decision#%s:postcommit:%s" % (
                                candidate.get("decision_id"), exc))
                    else:
                        for path, backup in written:
                            try:
                                _restore_file_from_backup(path, backup)
                            except Exception as rollback_exc:  # noqa: BLE001
                                result["errors"].append(
                                    "decision#%s:rollback_failed:%s:%s" % (
                                        candidate.get("decision_id"),
                                        path, rollback_exc))
                        for evidence_path in evidence_paths:
                            try:
                                evidence_path.unlink()
                            except OSError:
                                pass
                        action.update(
                            status="rolled_back",
                            reason="%s: %s" % (type(exc).__name__, exc))

    counts: Dict[str, int] = {}
    for action in result["actions"]:
        key = str(action.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    result["counts"] = dict(sorted(counts.items()))
    result["ok"] = not result["errors"] and not any(
        str(action.get("status") or "") in {
            "failed", "rolled_back", "repaired_postcommit_error"}
        for action in result["actions"])
    result["write_attempted"] = any(
        str(action.get("status") or "") == "repaired"
        for action in result["actions"])
    try:
        after_context = load_pending_context(
            root, decision_db=decision_path, workflow_db=workflow_path,
            output_dir=out_dir,
        )
        after_snapshot = classify_pending_snapshot(after_context)
        result["after_counts"] = dict(after_snapshot.get("counts") or {})
    except Exception as exc:  # noqa: BLE001
        result["errors"].append("post_audit:%s" % exc)
    _atomic_json(jpath, result)
    _atomic_text(mpath, _confirmed_term_apply_markdown(result))
    return result


def _confirmed_term_apply_markdown(result: Dict[str, Any]) -> str:
    lines = ["# confirmed specific term drift repair（受控写入）", ""]
    lines.append("- 生成时间: `%s`" % result.get("generated_at"))
    lines.append("- 模式: **%s**；ok=`%s`；write=`%s`" % (
        result.get("mode"), result.get("ok"), result.get("write_attempted")))
    lines.append("- 计划统计: `%s`" % (result.get("plan_counts") or {}))
    lines.append("- 动作统计: `%s`" % (result.get("counts") or {}))
    actions = result.get("actions") or []
    if actions:
        lines.extend(["", "## 动作"])
        for action in actions[:40]:
            lines.append(
                "- Decision #%s `%s` → `%s`: **%s** / `%s`; chunks=%s"
                % (action.get("decision_id"), action.get("source_key"),
                   action.get("canonical_name"), action.get("status"),
                   action.get("reason"), len(action.get("chunks") or [])))
            for chunk in action.get("chunks") or []:
                lines.append(
                    "  - `%s` before=%s after=%s backend=`%s`"
                    % (chunk.get("chunk_id"),
                       str(chunk.get("final_sha256_before"))[:12],
                       str(chunk.get("final_sha256_after"))[:12],
                       chunk.get("critic_backend_path")))
    if result.get("blocked"):
        lines.extend(["", "## 计划阻断"])
        for item in result.get("blocked")[:30]:
            lines.append(
                "- Decision #%s `%s`: `%s`"
                % (item.get("decision_id"), item.get("source_key"),
                   item.get("reason")))
    if result.get("after_counts"):
        lines.append("- 修复后 pending: `%s`" % result.get("after_counts"))
    if result.get("errors"):
        lines.extend(["", "## 错误"])
        lines.extend("- `%s`" % error for error in result["errors"][:40])
    lines.append("")
    return "\n".join(lines)


def _apply_markdown(result: Dict[str, Any]) -> str:
    lines = ["# name-only fast path（受控写入）", ""]
    lines.append(f"- 生成时间: {result.get('generated_at')}")
    lines.append(f"- 状态: `{result.get('mode')}` / ok=`{result.get('ok')}`")
    lines.append(f"- run_id: `{result.get('run_id')}`")
    if result.get("authority"):
        auth = result["authority"]
        lines.append(f"- 权威审计: ok=`{auth.get('ok')}` "
                     f"blocking={auth.get('blocking_count')} report={auth.get('report')}")
    if result.get("reason"):
        lines.append(f"- 未执行原因: `{result.get('reason')}`")
    counts = result.get("counts") or {}
    if counts:
        lines.append("- 动作计数: " + ", ".join(
            f"`{k}`={v}" for k, v in counts.items()))
    actions = result.get("actions") or []
    if actions:
        lines.append("")
        lines.append(f"## Chunks（{len(actions)}）")
        for action in actions[:80]:
            lines.append(f"- `{action.get('chunk_id')}` {action.get('status')} / "
                         f"{action.get('reason')}")
    if result.get("errors"):
        lines.append("")
        lines.append("## Errors")
        for error in result["errors"][:20]:
            lines.append(f"- {error}")
    lines.append("")
    return "\n".join(lines)


def _name_only_markdown(result: Dict[str, Any]) -> str:
    lines = ["# name-only fast path 只读分类", ""]
    lines.append(f"- 生成时间: {result.get('generated_at')}")
    lines.append(f"- 范围: `{result.get('chapter') or '全书'}`")
    counts = result.get("counts") or {}
    lines.append("- 计数: " + ", ".join(f"`{k}`={v}" for k, v in counts.items()))
    reason_counts = result.get("reason_counts") or {}
    if reason_counts:
        lines.append("- 未准入原因: " + ", ".join(
            f"`{k}`={v}" for k, v in reason_counts.items()))
    candidates = result.get("candidates") or []
    lines.append("")
    lines.append(f"## Candidates（{len(candidates)}）")
    lines.append("这些块仍必须在后续受控阶段对预测 final 重新调用 calibrated critic；"
                 "本报告不代表已完成或已获准写状态。")
    for item in candidates[:80]:
        mc = item.get("mapping_counts") or {}
        lines.append(
            f"- `{item['chunk_id']}` sep={mc.get('sep_restore', 0)}, "
            f"forms={mc.get('forms', 0)}, tokens={mc.get('tokens', 0)}; "
            f"before={str(item.get('final_sha256_before'))[:12]}, "
            f"after={str(item.get('final_sha256_after'))[:12]}")
    lines.append("")
    lines.append("## 阻断样例")
    blocked = [r for r in result.get("kept_dirty") or [] if r.get("reasons")]
    if not blocked:
        lines.append("- 无")
    else:
        for item in blocked[:30]:
            lines.append(f"- `{item['chunk_id']}`: {', '.join(item['reasons'])}")
        if len(blocked) > 30:
            lines.append(f"- 其余 {len(blocked) - 30} 个 kept_dirty 详见 JSON。")
    if result.get("failed"):
        lines.append("")
        lines.append("## Failed")
        for item in result["failed"][:20]:
            lines.append(f"- `{item.get('chunk_id', '-')}`: {item.get('reason')}")
    lines.append("")
    return "\n".join(lines)

