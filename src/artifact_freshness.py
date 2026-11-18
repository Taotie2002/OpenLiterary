"""Round 158 deterministic artifact identity and freshness contracts.

This module is deliberately LLM-free.  It answers one question for every
derived artifact: *was it built from the bytes that are on disk now?*

The contracts are:

* chapter sources are identified by ``chars + sha256``;
* a book profile records the chapter hashes it actually read;
* a story graph records the Decision DB row hash (never an absolute path),
  chapter hashes, profile note hash and build parameters;
* a gold evaluation records the chapter manifest, profile notes hash and graph
  fingerprint it was written against.

Callers may keep consuming legacy artifacts for research, but must not present
them as current delivery evidence.  Every result is a structured dict with
``ok``, ``status``, ``reasons``, ``expected`` and ``actual``.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.judgment_layer import all_notes, match_quote, sha256_text

EVAL_SCHEMA_VERSION = 2

_REGISTRY_COLUMNS = (
    "id", "canonical_name", "source_names", "variant_names", "entity_type",
    "pronoun_gender", "confidence", "pending_review", "translation_strategy",
    "first_seen_chunk", "quarantined_names",
)


def _result(ok: bool, status: str, reasons: Optional[Sequence[str]] = None,
            expected: Optional[Dict[str, Any]] = None,
            actual: Optional[Dict[str, Any]] = None,
            details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": bool(ok),
        "status": status,
        "reasons": list(reasons or []),
        "expected": expected or {},
        "actual": actual or {},
        "details": details or {},
    }


def chapter_manifest(input_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return ``chapter_id -> {chars, sha256}`` for ``input/ch*.md``.

    Only the chapter file bytes are hashed.  Paths and mtimes are deliberately
    excluded so the manifest is stable across checkouts and copies.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(input_dir).glob("ch*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        out[path.stem] = {"chars": len(text), "sha256": sha256_text(text)}
    return out


def chapter_hash_map(input_dir: Path) -> Dict[str, str]:
    """Compact ``chapter_id -> sha256`` map used by profile/graph artifacts."""
    return {cid: rec["sha256"] for cid, rec in chapter_manifest(input_dir).items()}


def _canonical_note(note: Dict[str, Any]) -> Dict[str, Any]:
    evidence: List[Dict[str, Any]] = []
    for ev in note.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        evidence.append({
            "chapter": str(ev.get("chapter") or ""),
            "quote": str(ev.get("quote") or ""),
            "offset": ev.get("offset"),
            "match_kind": str(ev.get("match_kind") or ""),
        })
    evidence.sort(key=lambda e: (e["chapter"], str(e["offset"]), e["quote"]))
    return {
        "id": str(note.get("id") or ""),
        "type": str(note.get("type") or ""),
        "chapters": [str(c) for c in (note.get("chapters") or [])],
        "claim": str(note.get("claim") or ""),
        "evidence": evidence,
        "confidence": note.get("confidence"),
        "status": str(note.get("status") or ""),
        "l2_verdict": str(note.get("l2_verdict") or ""),
    }


def profile_notes_hash(profile: Optional[Dict[str, Any]]) -> str:
    """Hash the semantic identity of the notes the graph consumes.

    Volatile bookkeeping (``created_at`` / ``source_model``) is excluded: a
    legacy profile loaded twice would otherwise receive a new ``created_at``
    on each normalization and make an unchanged graph look stale.
    """
    notes = [_canonical_note(n) for n in all_notes(profile) if isinstance(n, dict)]
    notes.sort(key=lambda n: (n["type"], n["id"]))
    return sha256_text(json.dumps(notes, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")))


def profile_fingerprint(profile: Optional[Dict[str, Any]]) -> str:
    """Content fingerprint of a book profile.

    Excludes ``generated_at``/``time`` and other wall-clock fields; includes the
    prompt version, generation parameters and canonicalized notes.  Rewriting a
    profile with the same content therefore yields the same fingerprint.
    """
    if not isinstance(profile, dict):
        return sha256_text("{}")
    generation = profile.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    notes = [_canonical_note(n) for n in all_notes(profile) if isinstance(n, dict)]
    notes.sort(key=lambda n: (n["type"], n["id"]))
    payload = {
        "schema_version": profile.get("schema_version"),
        "source": profile.get("source"),
        "model": generation.get("model"),
        "merge_model": generation.get("merge_model"),
        "prompt_version": generation.get("prompt_version"),
        "params": generation.get("params") or {},
        "notes": notes,
        "chapter_digests": [
            {
                "chapter_id": d.get("chapter_id"),
                "source_sha256": d.get("source_sha256"),
                "source_chars": d.get("source_chars"),
                "segment_ids": d.get("segment_ids") or [],
                "digest": d.get("digest"),
                "complete": bool(d.get("complete")),
            }
            for d in profile.get("chapter_digests") or []
            if isinstance(d, dict)
        ],
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), default=str))


def registry_rows(db_path: Path) -> List[Dict[str, Any]]:
    """Read the exact ``entity_registry`` rows consumed by the story graph."""
    path = Path(db_path)
    if not path.exists():
        return []
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(entity_registry)")}
        if not cols:
            return []
        select = ", ".join(c for c in _REGISTRY_COLUMNS if c in cols)
        rows = con.execute(f"SELECT {select} FROM entity_registry ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def rows_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    """Content hash of registry rows; stable across SQLite page metadata."""
    canonical = json.dumps(list(rows or []), ensure_ascii=False,
                           sort_keys=True, default=str, separators=(",", ":"))
    return sha256_text(canonical)


def decision_db_fingerprint(db_path: Path) -> Dict[str, Any]:
    """Fingerprint the Decision DB content the graph would read right now."""
    path = Path(db_path)
    exists = path.exists()
    rows = registry_rows(path) if exists else []
    return {
        "path_exists": bool(exists),
        "entity_registry_rows": len(rows),
        "rows_sha256": rows_fingerprint(rows),
    }


def profile_source_freshness(profile: Optional[Dict[str, Any]], root: Path,
                             input_dir: Optional[Path] = None,
                             chapters: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Compare ``generation.inputs.chapters`` with the chapter files on disk.

    ``chapters`` may restrict the comparison to the chapters a caller is about
    to consume.  A missing input block is ``untracked`` (legacy, readable but
    not evidence of current-source understanding), never ``fresh``.
    """
    root = Path(root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    current = chapter_manifest(in_dir)
    if chapters is not None:
        wanted = [str(c) for c in chapters]
        current = {cid: current[cid] for cid in wanted if cid in current}
    if not isinstance(profile, dict):
        return _result(False, "missing", ["profile_missing"],
                       expected={"chapters": current},
                       actual={"chapters": {}})
    generation = profile.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    inputs = generation.get("inputs")
    declared = inputs.get("chapters") if isinstance(inputs, dict) else None
    if not isinstance(declared, dict) or not declared:
        return _result(False, "untracked", ["profile_chapter_inputs_missing"],
                       expected={"chapters": current},
                       actual={"chapters": {}})

    reasons: List[str] = []
    missing: List[str] = []
    extra: List[str] = []
    changed: List[str] = []
    for cid, rec in current.items():
        got = declared.get(cid)
        if not isinstance(got, str) or not got:
            missing.append(cid)
        elif got != rec["sha256"]:
            changed.append(cid)
    if chapters is None:
        for cid in declared:
            if cid not in current:
                extra.append(str(cid))
    if missing:
        reasons.append("profile_chapter_missing")
    if extra:
        reasons.append("profile_chapter_unexpected")
    if changed:
        reasons.append("chapter_sources_changed")
    status = "fresh" if not reasons else "stale"
    return _result(not reasons, status, reasons,
                   expected={"chapters": current},
                   actual={"chapters": declared},
                   details={"missing": missing, "extra": extra, "changed": changed})


def _story_graph_params_from_config(root: Path) -> Dict[str, Any]:
    path = Path(root) / "config.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - config is advisory here
        return {}
    cfg = data.get("story_graph") if isinstance(data, dict) else None
    if not isinstance(cfg, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, caster in (("cooccur_window", int), ("max_cooccur_edges", int),
                        ("entity_coverage_min", float), ("scene_chars", int)):
        if key in cfg:
            try:
                out[key] = caster(cfg[key])
            except (TypeError, ValueError):
                pass
    return out


def graph_freshness(graph: Optional[Dict[str, Any]], root: Path,
                    profile: Optional[Dict[str, Any]],
                    input_dir: Optional[Path] = None,
                    db_path: Optional[Path] = None,
                    expected_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check that an on-disk story graph matches current sources/DB/profile."""
    root = Path(root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    dbp = Path(db_path) if db_path else root / "db" / "decision_db.sqlite"
    chapters = chapter_manifest(in_dir)
    db_fp = decision_db_fingerprint(dbp)
    notes_hash = profile_notes_hash(profile)
    params = dict(expected_params or _story_graph_params_from_config(root))

    if not isinstance(graph, dict) or not graph:
        return _result(False, "missing", ["graph_artifact_missing"],
                       expected={"chapters": chapters, "decision_db": db_fp,
                                 "profile_notes_hash": notes_hash},
                       actual={})
    generation = graph.get("generation")
    if not isinstance(generation, dict):
        generation = {}
    inputs = generation.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        return _result(False, "untracked", ["graph_generation_inputs_missing"],
                       expected={"chapters": chapters, "decision_db": db_fp,
                                 "profile_notes_hash": notes_hash},
                       actual={})

    reasons: List[str] = []
    details: Dict[str, Any] = {}
    actual_chapters = inputs.get("chapters")
    if not isinstance(actual_chapters, dict) or not actual_chapters:
        reasons.append("graph_chapter_inputs_missing")
        actual_chapters = {}
    else:
        missing = [cid for cid in chapters if cid not in actual_chapters]
        extra = [str(cid) for cid in actual_chapters if cid not in chapters]
        changed = [cid for cid, rec in chapters.items()
                   if cid in actual_chapters and actual_chapters.get(cid) != rec["sha256"]]
        if missing or extra or changed:
            reasons.append("chapter_sources_changed")
            details["chapters"] = {"missing": missing, "extra": extra,
                                   "changed": changed}
    if not db_fp["path_exists"]:
        reasons.append("decision_db_missing")
    if inputs.get("decision_db_hash") != db_fp["rows_sha256"]:
        reasons.append("decision_db_hash_changed")
    if int(inputs.get("entity_registry_rows") or 0) != int(db_fp["entity_registry_rows"]):
        reasons.append("decision_db_row_count_changed")
    if inputs.get("profile_notes_hash") != notes_hash:
        reasons.append("profile_notes_changed")

    declared_params = generation.get("params") if isinstance(generation.get("params"), dict) else {}
    param_diffs: Dict[str, Any] = {}
    for key, expected in params.items():
        actual = declared_params.get(key)
        if actual != expected:
            param_diffs[key] = {"expected": expected, "actual": actual}
    if param_diffs:
        reasons.append("graph_params_changed")
        details["params"] = param_diffs

    stored_fp = str(graph.get("fingerprint") or "")
    if stored_fp:
        version = graph.get("schema_version") or generation.get("graph_version")
        payload = {"graph_version": version,
                   "inputs": inputs,
                   "params": declared_params}
        recomputed = sha256_text(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if stored_fp != recomputed:
            reasons.append("graph_fingerprint_mismatch")

    status = "fresh" if not reasons else "stale"
    return _result(not reasons, status, reasons,
                   expected={"chapters": chapters, "decision_db": db_fp,
                             "profile_notes_hash": notes_hash, "params": params},
                   actual={"chapters": actual_chapters,
                           "decision_db_hash": inputs.get("decision_db_hash"),
                           "entity_registry_rows": inputs.get("entity_registry_rows"),
                           "profile_notes_hash": inputs.get("profile_notes_hash"),
                           "params": declared_params},
                   details=details)


def load_story_graph_checked(root: Path, path: Path,
                             profile: Optional[Dict[str, Any]],
                             require_current: bool = True,
                             input_dir: Optional[Path] = None,
                             db_path: Optional[Path] = None,
                             graph: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a graph and attach its freshness verdict.

    ``require_current=True`` makes an untracked/stale graph unusable for
    consumers that would present the result as current evidence.  Research
    callers may pass ``False`` (CLI ``--allow-stale``).
    """
    if graph is None:
        from src.story_graph import load_story_graph
        graph = load_story_graph(Path(path))
    if not graph:
        return {"ok": False, "status": "missing", "graph": None,
                "reasons": ["graph_artifact_missing"], "freshness": {}}
    freshness = graph_freshness(graph, root, profile, input_dir=input_dir,
                                db_path=db_path)
    usable = bool(freshness.get("ok")) or not require_current
    return {
        "ok": usable,
        "status": freshness.get("status", "unknown"),
        "graph": graph,
        "reasons": list(freshness.get("reasons") or []),
        "freshness": freshness,
    }


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _gold_errors(case: Dict[str, Any], chapter_texts: Dict[str, str],
                 min_similarity: float = 0.9) -> List[str]:
    gold = _as_list(case.get("gold") or case.get("gold_spans"))
    if not gold:
        if case.get("answerable") is False or case.get("negative") is True:
            return []
        return ["gold_missing"]
    errors: List[str] = []
    for item in gold:
        if not isinstance(item, dict):
            errors.append("gold_not_object")
            continue
        chapter = str(item.get("chapter") or "")
        quote = str(item.get("quote") or "")
        text = chapter_texts.get(chapter)
        if text is None:
            errors.append(f"gold_chapter_missing:{chapter}")
            continue
        if not quote:
            errors.append(f"gold_quote_missing:{chapter}")
            continue
        hit = match_quote(text, quote, min_similarity=min_similarity,
                          preferred_offset=item.get("offset"))
        if not hit.get("ok"):
            errors.append(f"gold_quote_unmatched:{chapter}")
    return errors


def eval_freshness(payload: Any, root: Path,
                   profile: Optional[Dict[str, Any]],
                   graph: Optional[Dict[str, Any]] = None,
                   input_dir: Optional[Path] = None,
                   require_active_notes: bool = True,
                   min_similarity: float = 0.9) -> Dict[str, Any]:
    """Fail-closed validation for a gold-span evaluation file.

    Legacy list payloads, missing lineage fields, unknown note IDs, inactive
    notes and unverifiable gold quotes all make the whole evaluation invalid.
    No metrics may be computed from an invalid file.
    """
    root = Path(root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    chapters = chapter_manifest(in_dir)
    notes_hash = profile_notes_hash(profile)
    graph_fp = str((graph or {}).get("fingerprint") or "")
    note_status: Dict[str, str] = {}
    for note in all_notes(profile):
        nid = str(note.get("id") or "")
        if nid:
            note_status[nid] = str(note.get("status") or "")

    if isinstance(payload, list):
        return _result(False, "invalid", ["eval_payload_legacy_list"],
                       expected={"schema_version": EVAL_SCHEMA_VERSION,
                                 "chapter_manifest": chapters,
                                 "profile_notes_hash": notes_hash,
                                 "graph_fingerprint": graph_fp},
                       actual={"schema_version": 0, "cases": len(payload)},
                       details={"invalid_cases": len(payload)})
    if not isinstance(payload, dict):
        return _result(False, "invalid", ["eval_payload_not_object"],
                       expected={"schema_version": EVAL_SCHEMA_VERSION},
                       actual={})

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return _result(False, "invalid", ["eval_cases_empty"],
                       expected={"schema_version": EVAL_SCHEMA_VERSION},
                       actual={"cases": len(cases) if isinstance(cases, list) else 0})

    reasons: List[str] = []
    version = payload.get("schema_version")
    if version is None:
        version = payload.get("version")
    try:
        version_int = int(version or 0)
    except (TypeError, ValueError):
        version_int = 0
    if version_int < EVAL_SCHEMA_VERSION:
        reasons.append("eval_schema_legacy")

    required = ("book", "profile_notes_hash", "chapter_manifest", "source",
                "graph_fingerprint")
    missing_fields = [key for key in required if not payload.get(key)]
    if missing_fields:
        reasons.append("eval_lineage_fields_missing")

    declared_manifest = payload.get("chapter_manifest")
    if isinstance(declared_manifest, dict) and declared_manifest:
        normalized: Dict[str, Any] = {}
        for cid, value in declared_manifest.items():
            if isinstance(value, dict):
                normalized[str(cid)] = value
            else:
                normalized[str(cid)] = {"sha256": str(value)}
        if normalized != chapters:
            reasons.append("eval_chapter_manifest_changed")
    elif "eval_lineage_fields_missing" not in reasons:
        reasons.append("eval_chapter_manifest_missing")

    if payload.get("profile_notes_hash") and payload.get("profile_notes_hash") != notes_hash:
        reasons.append("eval_profile_changed")
    if graph_fp and payload.get("graph_fingerprint") and \
            payload.get("graph_fingerprint") != graph_fp:
        reasons.append("eval_graph_changed")

    chapter_texts: Dict[str, str] = {}
    for cid in chapters:
        path = in_dir / f"{cid}.md"
        try:
            chapter_texts[cid] = path.read_text(encoding="utf-8")
        except OSError:
            continue

    case_errors: List[Dict[str, Any]] = []
    stale_notes = 0
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            case_errors.append({"index": index, "errors": ["case_not_object"]})
            continue
        errors: List[str] = []
        note_id = str(case.get("note_id") or "")
        negative = (case.get("answerable") is False
                    or case.get("negative") is True)
        if note_id:
            status = note_status.get(note_id)
            if status is None:
                errors.append("note_id_missing")
                stale_notes += 1
            elif require_active_notes and status != "active":
                errors.append(f"note_not_active:{status or 'unknown'}")
                stale_notes += 1
        elif not negative:
            errors.append("note_id_missing")
            stale_notes += 1
        errors.extend(_gold_errors(case, chapter_texts, min_similarity))
        if errors:
            case_errors.append({"index": index, "id": str(case.get("id") or ""),
                                "errors": errors})
    if stale_notes:
        reasons.append("eval_stale_notes")
    if case_errors:
        reasons.append("eval_cases_invalid")

    status = "fresh" if not reasons else "invalid"
    return _result(not reasons, status, reasons,
                   expected={"schema_version": EVAL_SCHEMA_VERSION,
                             "chapter_manifest": chapters,
                             "profile_notes_hash": notes_hash,
                             "graph_fingerprint": graph_fp},
                   actual={"schema_version": version_int,
                           "book": payload.get("book"),
                           "source": payload.get("source"),
                           "chapter_manifest": declared_manifest or {},
                           "profile_notes_hash": payload.get("profile_notes_hash"),
                           "graph_fingerprint": payload.get("graph_fingerprint"),
                           "cases": len(cases)},
                   details={"missing_fields": missing_fields,
                            "stale_notes": stale_notes,
                            "invalid_cases": len(case_errors),
                            "case_errors": case_errors[:50]})
