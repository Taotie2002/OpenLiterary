"""Round 131 judgment layer.

The judgment layer is deliberately advisory and artifact-based:

* deep reading produces immutable ``input/book_profile.json`` notes and
  ``input/story_graph.json``;
* L1 validates every evidence quote against the split chapter source file;
* L2 asks a cheap second model to confirm that evidence supports the claim;
* execution prompts receive only a small, deterministic slice of active notes.

Nothing here may override the decision DB.  Terminology/prescriptive claims are
rejected at schema and L2 time.  Failures are fail-open for the translation
workflow: a note becomes dormant and is simply not injected.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

JUDGMENT_SCHEMA_VERSION = 2
VALID_NOTE_TYPES = ("voice", "setup_payoff", "arc")
VALID_NOTE_STATUSES = ("active", "dormant", "dropped")
PROFILE_NOTE_KEYS = {
    "voice": "voice_notes",
    "setup_payoff": "setup_payoff",
    "arc": "narrative_arc",
}
TYPE_PRIORITY = {"voice": 3, "setup_payoff": 2, "arc": 1}

# Output caps keep one deep read from producing an unverifiable note flood.
NOTE_TYPE_LIMITS = {"voice": 50, "setup_payoff": 30, "arc": 20}
VOICE_NOTES_PER_CHAPTER = 8

# These belong to the judgment artifact/profile, never to critic scoring or
# literal-translation style state.  profile_to_style_guide already omits them;
# this list is the defensive boundary for other callers.
ADVISORY_STYLE_KEYS = {
    "narrative_arc", "setup_payoff", "voice_notes", "judgment_notes",
    "judgment", "generation", "deep_read", "schema_version",
}

STRUCTURAL_P0_CATEGORIES = ("dangling", "dup_note")

# Phase 3 repair sidecar.  The sidecar is valid only for the exact final text
# that produced the review anchors; the pipeline binds it to the pre-rerun
# final_backup.json hash before allowing the feedback into a prompt.
REVIEW_FIX_SCHEMA_VERSION = 1
REVIEW_FIX_SIDECAR_STEP = "review_fix"

# A normative terminology assertion belongs to the decision DB, not here.
# The pattern intentionally allows narrative clauses that merely name a
# character ("保罗在第一章仍显得...").
_PRESCRIPTIVE_RE = re.compile(
    r"(?:术语|译名|名称|专名|人名|地名)\s*(?:应|应当|必须|须|一律|统一|固定|标准)"
    r"|(?:应|应当|必须|须|一律|统一|固定|标准|建议)\s*(?:译作|译成|译为|采用|使用)"
    r"|(?:统一|固定)(?:译名|术语|名称|译法)"
    r"|(?:该|此)(?:词|术语|名称).{0,18}(?:译|作|采用)"
    r"|should\s+be\s+translated|must\s+be\s+translated|standard\s+translation"
    r"|preferred\s+term|canonical\s+(?:term|translation)",
    re.IGNORECASE,
)


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON via replace so a crashed run never leaves a torn artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(text or ""))
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def is_prescriptive_claim(claim: str) -> bool:
    return bool(_PRESCRIPTIVE_RE.search(str(claim or "")))


def _clean_chapter_id(value: Any) -> str:
    s = str(value or "").strip()
    return s if re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", s) else ""


def normalize_chapters(value: Any) -> List[str]:
    """Normalize chapter ids/ranges while preserving first-seen order.

    A two-element list is an inclusive narrative span.  Explicit dictionaries
    such as ``{"start": "ch01", "end": "ch03"}`` are accepted for convenience.
    """
    out: List[str] = []
    if isinstance(value, dict):
        a = _clean_chapter_id(value.get("start") or value.get("from"))
        b = _clean_chapter_id(value.get("end") or value.get("to"))
        value = [a, b]
    elif isinstance(value, str):
        value = re.split(r"[,，\s]+", value)
    if not isinstance(value, list):
        return out
    for item in value:
        cid = _clean_chapter_id(item)
        if cid and cid not in out:
            out.append(cid)
    return out


def chapter_ids_between(chapters: Sequence[str]) -> List[str]:
    """Return the lexicographic inclusive span for a note's chapter list.

    Chapter ids in SagaTrans are fixed-width (ch01, ch02, ...).  Treating a
    multi-chapter list as a span is what makes an arc visible to intermediate
    chapters without exploding the profile into one record per chapter.
    """
    clean = normalize_chapters(list(chapters))
    if len(clean) < 2:
        return clean
    try:
        lo = min(clean)
        hi = max(clean)
        m_lo = re.fullmatch(r"([^\d]*)(\d+)", lo)
        m_hi = re.fullmatch(r"([^\d]*)(\d+)", hi)
        if m_lo and m_hi and m_lo.group(1) == m_hi.group(1):
            width = max(len(m_lo.group(2)), len(m_hi.group(2)))
            return [f"{m_lo.group(1)}{i:0{width}d}"
                    for i in range(int(m_lo.group(2)), int(m_hi.group(2)) + 1)]
    except Exception:
        pass
    return sorted(clean)


def note_covers_chapter(note: Dict[str, Any], chapter_id: str) -> bool:
    cid = _clean_chapter_id(chapter_id)
    if not cid:
        return False
    return cid in chapter_ids_between(note.get("chapters") or [])


def normalize_evidence(value: Any, default_chapters: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        value = [value]
    elif isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    defaults = normalize_chapters(list(default_chapters or []))
    for idx, item in enumerate(value):
        chapter = ""
        quote = ""
        offset: Optional[int] = None
        if isinstance(item, str):
            quote = item.strip()
            chapter = defaults[min(idx, len(defaults) - 1)] if defaults else ""
        elif isinstance(item, dict):
            chapter = _clean_chapter_id(item.get("chapter") or item.get("chapter_id"))
            quote = str(item.get("quote") or item.get("text") or "").strip()
            try:
                if item.get("offset") is not None:
                    offset = max(0, int(item.get("offset")))
            except (TypeError, ValueError):
                offset = None
        if not chapter and defaults:
            chapter = defaults[0]
        if not chapter or not quote:
            continue
        ev = {"chapter": chapter, "quote": quote}
        if offset is not None:
            ev["offset"] = offset
        for key in ("match_kind", "match_score", "matched_quote", "unique"):
            if isinstance(item, dict) and key in item:
                ev[key] = item[key]
        out.append(ev)
    return out


def normalize_note(
    raw: Any,
    type_hint: str = "",
    index: int = 1,
    source_model: str = "",
    created_at: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    ntype = str(raw.get("type") or type_hint or "").strip().lower()
    if ntype not in VALID_NOTE_TYPES:
        return None
    claim = re.sub(r"\s+", " ", str(
        raw.get("claim") or raw.get("note") or raw.get("summary") or ""
    )).strip()
    if not claim:
        return None
    chapters = normalize_chapters(raw.get("chapters") or [])
    # Evidence items may omit their chapter while the note carries the span.
    # Normalize chapters first and use the span as a default, otherwise a valid
    # evidence quote is silently discarded before L1 can validate it.
    ev = normalize_evidence(raw.get("evidence") or raw.get("evidence_quotes"),
                            default_chapters=chapters)
    if not chapters:
        seen = []
        for e in ev:
            if e["chapter"] not in seen:
                seen.append(e["chapter"])
        chapters = seen
    if not chapters or not ev:
        return None
    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    status = str(raw.get("status") or "active").strip().lower()
    if status not in VALID_NOTE_STATUSES:
        status = "dormant"
    nid = str(raw.get("id") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", nid):
        nid = f"{ntype}-{index:03d}"
    note = {
        "id": nid,
        "type": ntype,
        "chapters": chapters,
        "claim": claim[:600],
        "evidence": ev,
        "confidence": conf,
        "status": status,
        "source_model": str(raw.get("source_model") or source_model or "").strip(),
        "created_at": str(raw.get("created_at") or created_at or _now()).strip(),
    }
    meta = raw.get("meta")
    if isinstance(meta, dict) and meta:
        note["meta"] = meta
    # Preserve validation/provenance fields across save -> load round-trips.
    # Without these fields a persisted active note is indistinguishable from a
    # hand-edited/unvalidated note, so note_is_validated() rejects it at runtime.
    for key in ("l1_reject_reason", "l1_note", "l2_verdict", "l2_reason",
                "l2_reject_reason", "schema_reject_reason",
                "limit_reject_reason", "pairing_status"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            note[key] = value.strip()
    if is_prescriptive_claim(claim):
        note["status"] = "dropped"
        note["schema_reject_reason"] = "prescriptive_terminology_claim"
    return note


def dedupe_notes(notes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop exact duplicate observations while preserving first-seen order."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for note in notes:
        normalized = normalize_text_for_match(str(note.get("claim") or ""))
        evidence = tuple(sorted(
            (str(e.get("chapter") or ""), normalize_text_for_match(str(e.get("quote") or "")))
            for e in (note.get("evidence") or [])
        ))
        sig = (str(note.get("type") or ""), normalized, evidence)
        if not normalized or sig in seen:
            continue
        seen.add(sig)
        out.append(note)
    return out


def ensure_unique_note_ids(notes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make note ids globally unique after merging independently extracted parts."""
    out: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for raw in notes:
        note = dict(raw)
        base = str(note.get("id") or "note").strip() or "note"
        count = seen.get(base, 0)
        if count:
            note["id"] = f"{base}-{count + 1}"
        seen[base] = count + 1
        out.append(note)
    return out


def restrict_notes_to_candidate_quotes(
    notes: Sequence[Dict[str, Any]],
    candidate_notes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Enforce merge-layer provenance: only quotes emitted by extraction survive.

    L1 alone cannot catch this failure mode: a merge model may paste a real
    sentence from a different part of the book and still pass substring
    matching. The extraction stage is the candidate boundary, so merge output
    may only re-use those exact quotes (whitespace/case normalized). Notes
    whose evidence is entirely invented by the merge layer are retained as
    ``dropped`` audit records instead of disappearing silently.
    """
    allowed = set()
    for note in candidate_notes or []:
        for ev in note.get("evidence") or []:
            qn = normalize_text_for_match(str(ev.get("quote") or ""))
            if qn:
                allowed.add(qn)
    out: List[Dict[str, Any]] = []
    for raw in notes or []:
        note = dict(raw)
        kept = []
        for ev in note.get("evidence") or []:
            qn = normalize_text_for_match(str(ev.get("quote") or ""))
            if qn and qn in allowed:
                kept.append(dict(ev))
        note["evidence"] = kept
        if not kept and note.get("status") == "active":
            note["status"] = "dropped"
            note["schema_reject_reason"] = "merge_quote_not_from_extraction_candidates"
        out.append(note)
    return out


def enforce_note_limits(notes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply per-type and per-chapter voice caps deterministically.

    Highest confidence wins; ties are broken by id so repeated runs are stable.
    Notes beyond a cap are retained as ``dropped`` for audit, never silently
    removed from the artifact.
    """
    ranked = sorted(
        [dict(n) for n in notes],
        key=lambda n: (-float(n.get("confidence", 0.0) or 0.0), str(n.get("id") or "")),
    )
    counts: Dict[str, int] = {k: 0 for k in VALID_NOTE_TYPES}
    voice_chapter_counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for note in ranked:
        # Non-active notes are audit-only and must not consume an injection
        # quota before L2 has even run.
        if str(note.get("status") or "") != "active":
            out.append(note)
            continue
        ntype = str(note.get("type") or "")
        if ntype not in counts:
            note["status"] = "dropped"
            note["limit_reject_reason"] = "unknown_type"
            out.append(note)
            continue
        reason = ""
        if counts[ntype] >= NOTE_TYPE_LIMITS[ntype]:
            reason = f"cap:{ntype}"
        if ntype == "voice":
            for cid in normalize_chapters(note.get("chapters") or []):
                if voice_chapter_counts.get(cid, 0) >= VOICE_NOTES_PER_CHAPTER:
                    reason = f"cap:voice_per_chapter:{cid}"
                    break
        if reason:
            note["status"] = "dropped"
            note["limit_reject_reason"] = reason
            out.append(note)
            continue
        counts[ntype] += 1
        if ntype == "voice":
            for cid in normalize_chapters(note.get("chapters") or []):
                voice_chapter_counts[cid] = voice_chapter_counts.get(cid, 0) + 1
        out.append(note)
    out.sort(key=lambda n: str(n.get("id") or ""))
    return {"notes": out, "counts": counts,
            "voice_chapter_counts": voice_chapter_counts}


def normalize_notes(value: Any, type_hint: str = "", source_model: str = "",
                    created_at: str = "") -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        # Accept {"notes": [...]} and grouped-by-chapter dictionaries.
        if isinstance(value.get("notes"), list):
            value = value["notes"]
        else:
            value = list(value.values())
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_ids: Dict[str, int] = {}
    for idx, raw in enumerate(value, 1):
        note = normalize_note(raw, type_hint=type_hint, index=idx,
                              source_model=source_model, created_at=created_at)
        if not note:
            continue
        base = note["id"]
        n = seen_ids.get(base, 0)
        if n:
            note["id"] = f"{base}-{n + 1}"
        seen_ids[base] = n + 1
        out.append(note)
    return out


def normalize_judgment_fields(data: Dict[str, Any], source_model: str = "",
                              created_at: str = "") -> Dict[str, List[Dict[str, Any]]]:
    raw_by_type = {
        "voice": data.get("voice_notes") or [],
        "setup_payoff": data.get("setup_payoff") or [],
        "arc": data.get("narrative_arc") or data.get("arc_notes") or [],
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for ntype in VALID_NOTE_TYPES:
        out[PROFILE_NOTE_KEYS[ntype]] = normalize_notes(
            raw_by_type[ntype], type_hint=ntype,
            source_model=source_model, created_at=created_at,
        )
    return out


def all_notes(profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(profile, dict):
        return []
    out: List[Dict[str, Any]] = []
    for key in PROFILE_NOTE_KEYS.values():
        val = profile.get(key) or []
        if isinstance(val, list):
            out.extend(n for n in val if isinstance(n, dict))
    return out


def note_is_validated(note: Dict[str, Any]) -> bool:
    """True only when L1 matched every quote and L2 explicitly confirmed it.

    A loaded v1 profile (or a hand-edited v2 file) may contain ``status:
    active`` without ever passing the validation chain.  Status alone is not a
    provenance proof, so runtime injection and graph certification require both
    markers produced by the actual pipeline.
    """
    if not isinstance(note, dict) or note.get("status") != "active":
        return False
    if note.get("l2_verdict") != "confirmed":
        return False
    evidence = note.get("evidence") or []
    return bool(evidence) and all(
        isinstance(ev, dict) and bool(ev.get("match_kind")) for ev in evidence
    )


def active_notes(profile: Optional[Dict[str, Any]],
                 min_confidence: float = 0.0,
                 require_validated: bool = True) -> List[Dict[str, Any]]:
    out = []
    for note in all_notes(profile):
        if note.get("status") != "active":
            continue
        if require_validated and not note_is_validated(note):
            continue
        try:
            conf = float(note.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf + 1e-9 < min_confidence:
            continue
        if is_prescriptive_claim(str(note.get("claim") or "")):
            continue
        out.append(note)
    return out


def sanitize_style_guide_for_scoring(style_guide: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a scorer-safe style guide with all advisory judgment fields removed."""
    if not isinstance(style_guide, dict):
        return {}
    return {k: v for k, v in style_guide.items() if k not in ADVISORY_STYLE_KEYS}


# ────────────────────────────────────────────────────────────────────
# L1: mechanical evidence matching against split chapter files
# ────────────────────────────────────────────────────────────────────

_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
}


def _normalized_with_map(text: str) -> Tuple[str, List[int]]:
    """Whitespace-free normalized text plus a normalized->original index map."""
    chars: List[str] = []
    positions: List[int] = []
    for i, ch in enumerate(str(text or "")):
        if ch.isspace():
            continue
        rep = _REPLACEMENTS.get(ch, ch)
        rep = unicodedata.normalize("NFKC", rep)
        for rc in rep:
            if rc.isspace():
                continue
            chars.append(rc.lower())
            positions.append(i)
    return "".join(chars), positions


def normalize_text_for_match(text: str) -> str:
    return _normalized_with_map(text)[0]


def _ascii_alnum(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalnum()


def _boundary_ok(text: str, start: int, end: int, quote: str) -> bool:
    if start < 0 or end > len(text) or end <= start:
        return False
    q = str(quote or "")
    if q and _ascii_alnum(q[0]) and start > 0 and _ascii_alnum(text[start - 1]):
        return False
    if q and _ascii_alnum(q[-1]) and end < len(text) and _ascii_alnum(text[end]):
        return False
    return True


def _exact_spans(text: str, quote: str, limit: int = 50) -> List[Tuple[int, int]]:
    if not quote:
        return []
    spans: List[Tuple[int, int]] = []
    pos = 0
    while len(spans) < limit:
        idx = text.find(quote, pos)
        if idx < 0:
            break
        end = idx + len(quote)
        if _boundary_ok(text, idx, end, quote):
            spans.append((idx, end))
        pos = idx + 1
    return spans


def _normalized_spans(text: str, quote: str, limit: int = 50) -> List[Tuple[int, int]]:
    qn = normalize_text_for_match(quote)
    tn, pos = _normalized_with_map(text)
    if not qn or not tn:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    while len(spans) < limit:
        idx = tn.find(qn, start)
        if idx < 0:
            break
        end_norm = idx + len(qn) - 1
        orig_start = pos[idx]
        orig_end = pos[end_norm] + 1
        if _boundary_ok(text, orig_start, orig_end, quote):
            spans.append((orig_start, orig_end))
        start = idx + 1
    return spans


def _quote_tokens(quote: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]{3,}|[\u4e00-\u9fff]{2,}", quote)
    toks.sort(key=len, reverse=True)
    return toks


def _similar_spans(text: str, quote: str, min_ratio: float = 0.9,
                   limit: int = 12) -> List[Tuple[int, int, float]]:
    qn = normalize_text_for_match(quote)
    if len(qn) < 8:
        return []
    tokens = _quote_tokens(quote)
    candidates: List[Tuple[int, str, int]] = []
    if tokens:
        for tok in tokens[:2]:
            tok_n = normalize_text_for_match(tok)
            relative = qn.find(tok_n)
            if relative < 0:
                continue
            try:
                for m in re.finditer(re.escape(tok), text, re.IGNORECASE):
                    candidates.append((m.start(), tok_n, relative))
                    if len(candidates) >= 30:
                        break
            except re.error:
                pass
            if len(candidates) >= 30:
                break
    if not candidates:
        return []
    margin = max(24, int(len(qn) * 0.6) + 12)
    best: List[Tuple[int, int, float]] = []
    seen: Dict[Tuple[int, int], float] = {}
    # Compare quote-length slices anchored at a source token.  Comparing the
    # quote with a fixed oversized context window dilutes a one-character
    # punctuation/typo difference below the threshold; token anchoring keeps
    # the comparison local while still allowing small insertions/deletions.
    for cpos, tok_n, relative in candidates:
        start = max(0, cpos - margin)
        end = min(len(text), cpos + margin + len(qn))
        window = text[start:end]
        wn, wmap = _normalized_with_map(window)
        tok_pos = wn.find(tok_n)
        if tok_pos < 0:
            continue
        base = max(0, tok_pos - relative)
        for offset in range(-3, 4):
            nstart = max(0, base + offset)
            for delta in (-2, 0, 2, 4, 8):
                nend = min(len(wn), nstart + len(qn) + delta)
                if nend <= nstart:
                    continue
                ratio = difflib.SequenceMatcher(
                    None, qn, wn[nstart:nend], autojunk=False).ratio()
                if ratio + 1e-9 < min_ratio:
                    continue
                orig_start = start + wmap[nstart]
                orig_end = start + wmap[nend - 1] + 1
                if not _boundary_ok(text, orig_start, orig_end, quote):
                    continue
                key = (orig_start, orig_end)
                seen[key] = max(seen.get(key, 0.0), ratio)
    for (orig_start, orig_end), ratio in seen.items():
        best.append((orig_start, orig_end, ratio))
    best.sort(key=lambda x: (-x[2], x[0]))
    return best[:limit]


def match_quote(text: str, quote: str, min_similarity: float = 0.9,
                preferred_offset: Optional[int] = None) -> Dict[str, Any]:
    """Match an evidence quote against one chapter source file.

    Order: exact -> whitespace/quote-normalized exact -> anchored fuzzy.
    A short ASCII token is never allowed to match inside a larger word.
    """
    src = str(text or "")
    q = str(quote or "").strip()
    if not src or len(q) < 2:
        return {"ok": False, "reason": "empty_quote_or_text"}

    def _choose(spans: List[Tuple[int, int]], kind: str, score: float) -> Dict[str, Any]:
        if not spans:
            return {}
        if preferred_offset is not None:
            spans = sorted(spans, key=lambda s: (abs(s[0] - preferred_offset), s[0]))
        start, end = spans[0]
        return {
            "ok": True,
            "match_kind": kind,
            "match_score": float(score),
            "offset": int(start),
            "end": int(end),
            "matched_quote": src[start:end],
            "unique": len(spans) == 1,
            "candidate_count": len(spans),
        }

    hit = _choose(_exact_spans(src, q), "exact", 1.0)
    if hit:
        return hit
    hit = _choose(_normalized_spans(src, q), "normalized_exact", 1.0)
    if hit:
        return hit
    fuzzy = _similar_spans(src, q, min_ratio=min_similarity)
    if fuzzy:
        spans = [(s, e) for s, e, _ in fuzzy]
        chosen = _choose(spans, "fuzzy", fuzzy[0][2])
        if chosen:
            return chosen
    return {"ok": False, "reason": "evidence_not_found", "quote": q}


def map_quote_to_chunks(chunk_texts: Dict[str, str], quote: str,
                        min_similarity: float = 0.9) -> Dict[str, Any]:
    """Map one verified review quote to exactly one final chunk.

    Exact/normalized matches are preferred over fuzzy matches.  When the same
    quote occurs in more than one chunk the result is deliberately not
    actionable: auto-DIRTY must never guess which chunk the editor meant.
    """
    hits: List[Dict[str, Any]] = []
    for chunk_id, text in (chunk_texts or {}).items():
        hit = match_quote(text or "", quote, min_similarity=min_similarity)
        if hit.get("ok"):
            hits.append({
                "chunk_id": str(chunk_id),
                "match_kind": hit.get("match_kind", ""),
                "match_score": hit.get("match_score", 0.0),
                "offset": hit.get("offset"),
                "matched_quote": hit.get("matched_quote", ""),
                "unique_in_chunk": bool(hit.get("unique")),
            })
    if not hits:
        return {"ok": False, "reason": "quote_not_in_final_chunks", "hits": []}
    if len(hits) > 1:
        return {"ok": False, "reason": "ambiguous_across_chunks", "hits": hits}
    out = dict(hits[0])
    out.update({"ok": True, "reason": ""})
    return out


def validate_note_evidence(notes: Sequence[Dict[str, Any]],
                           chapter_texts: Dict[str, str],
                           min_similarity: float = 0.9) -> Dict[str, Any]:
    """L1 validation.  A note with any unverifiable evidence is dropped.

    Besides quote matching, L1 enforces two structural rules which an LLM merge
    can otherwise violate: evidence must sit inside the note's declared chapter
    span, and a setup/payoff note must actually carry evidence from at least
    two distinct chapters.
    """
    validated: List[Dict[str, Any]] = []
    audit_notes: List[Dict[str, Any]] = []
    for note in notes:
        n = dict(note)
        n["evidence"] = [dict(e) for e in (note.get("evidence") or [])]
        reason = ""
        soft_reason = ""
        ev_audit: List[Dict[str, Any]] = []
        if is_prescriptive_claim(str(n.get("claim") or "")):
            n["status"] = "dropped"
            reason = "schema:prescriptive_terminology_claim"
        else:
            for ev in n["evidence"]:
                cid = str(ev.get("chapter") or "")
                txt = chapter_texts.get(cid)
                if txt is None:
                    reason = f"l1:chapter_missing:{cid or '?'}"
                    ev_audit.append({"chapter": cid, "quote": ev.get("quote"), "ok": False,
                                     "reason": reason})
                    break
                hit = match_quote(txt, str(ev.get("quote") or ""),
                                  min_similarity=min_similarity,
                                  preferred_offset=ev.get("offset"))
                ev_audit.append({
                    "chapter": cid,
                    "quote": ev.get("quote"),
                    "ok": bool(hit.get("ok")),
                    "match_kind": hit.get("match_kind", ""),
                    "match_score": hit.get("match_score", 0.0),
                    "offset": hit.get("offset"),
                    "unique": hit.get("unique", False),
                    "matched_quote": hit.get("matched_quote", ""),
                    "reason": hit.get("reason", ""),
                })
                if not hit.get("ok"):
                    reason = f"l1:{hit.get('reason') or 'evidence_not_found'}:{cid}"
                    break
                ev["offset"] = hit["offset"]
                ev["match_kind"] = hit["match_kind"]
                ev["match_score"] = hit["match_score"]
                ev["matched_quote"] = hit["matched_quote"]
                ev["unique"] = hit["unique"]
            if not n["evidence"]:
                reason = reason or "l1:no_evidence"
            if not reason:
                evidence_chapters: List[str] = []
                for ev in n["evidence"]:
                    cid = str(ev.get("chapter") or "")
                    if cid and cid not in evidence_chapters:
                        evidence_chapters.append(cid)
                span = set(chapter_ids_between(n.get("chapters") or []))
                outside = [cid for cid in evidence_chapters if cid not in span]
                if outside:
                    reason = f"l1:evidence_outside_chapter_range:{outside[0]}"
                elif n.get("type") == "setup_payoff" and len(evidence_chapters) < 2:
                    # Merge marks a one-sided setup as dormant/unpaired when it
                    # cannot find a payoff candidate.  That is an audit-only
                    # knowledge gap, not a fabricated active claim: keep the
                    # quote validation and leave it dormant.  An active
                    # one-chapter setup is still invalid and must be dropped.
                    if str(n.get("status") or "").lower() == "dormant":
                        soft_reason = "l1:setup_payoff_unpaired"
                    else:
                        reason = "l1:setup_payoff_requires_two_chapters"
            if reason:
                n["status"] = "dropped"
                n["l1_reject_reason"] = reason
            elif soft_reason:
                n["status"] = "dormant"
                n["l1_note"] = soft_reason
        validated.append(n)
        audit_notes.append({
            "id": n.get("id"), "type": n.get("type"), "status": n.get("status"),
            "confidence": n.get("confidence"),
            "reason": reason or n.get("l1_reject_reason", "") or n.get("l1_note", ""),
            "evidence": ev_audit,
        })

    limited = enforce_note_limits(validated)
    validated = limited["notes"]
    counts = {"active": 0, "dormant": 0, "dropped": 0}
    audit_by_id = {str(a.get("id") or ""): a for a in audit_notes}
    for n in validated:
        status = str(n.get("status") or "dormant")
        counts[status] = counts.get(status, 0) + 1
        item = audit_by_id.get(str(n.get("id") or ""))
        if item is None:
            item = {"id": n.get("id"), "type": n.get("type"), "evidence": []}
            audit_notes.append(item)
        item["status"] = status
        if n.get("limit_reject_reason"):
            item["reason"] = n["limit_reject_reason"]
    return {
        "notes": validated,
        "audit": {
            "stage": "l1",
            "checked": len(notes),
            "counts": counts,
            "type_counts": limited.get("counts") or {},
            "notes": audit_notes,
        },
    }


# ────────────────────────────────────────────────────────────────────
# L2: cheap cross-model corroboration
# ────────────────────────────────────────────────────────────────────

def _context_for_evidence(chapter_texts: Dict[str, str], evidence: Dict[str, Any],
                          radius: int = 120) -> str:
    text = chapter_texts.get(str(evidence.get("chapter") or "")) or ""
    try:
        off = int(evidence.get("offset", 0))
    except (TypeError, ValueError):
        off = 0
    if not text:
        return ""
    start = max(0, off - radius)
    end = min(len(text), off + len(str(evidence.get("quote") or "")) + radius)
    return text[start:end].replace("\n", " ")


def build_l2_prompt(notes: Sequence[Dict[str, Any]], chapter_texts: Dict[str, str]) -> str:
    payload = []
    for n in notes:
        evs = []
        for ev in (n.get("evidence") or [])[:4]:
            evs.append({
                "chapter": ev.get("chapter"),
                "quote": ev.get("quote"),
                "matched_quote": ev.get("matched_quote") or ev.get("quote"),
                "context": _context_for_evidence(chapter_texts, ev),
            })
        payload.append({
            "id": n.get("id"),
            "type": n.get("type"),
            "chapters": n.get("chapters"),
            "claim": n.get("claim"),
            "evidence": evs,
        })
    return f"""你是文学翻译项目的证据复核员。以下是从全书深读中抽出的“判断层注记”。
你的任务只有一个：判断每条 claim 是否被 evidence 原文引文直接支撑。

判定规则：
- confirmed：引文/上下文确实支撑该叙事、伏笔或音色观察；
- unsure：证据不足、范围过宽或无法判断；
- rejected：claim 与引文矛盾，或引文不支持 claim。
- 注记只允许描述语气、人物弧线、伏笔/回收、叙事视角。若 claim 规定“术语/译名应如何翻译”、
  “统一译作”“标准译名”等规范性内容，必须 rejected（这些约束属于 decision DB）。
- 仅因 claim 提到人名/地名（如“保罗在结尾更沉默”）不得 rejected；要看证据是否支撑叙述判断。

只输出 JSON，不要 markdown：
{{"decisions":[{{"id":"...","verdict":"confirmed|unsure|rejected","reason":"≤60字"}}]}}

【待复核注记】
{json.dumps(payload, ensure_ascii=False)}
"""


def default_l2_generate(prompt: str) -> str:
    from src.config import get_config
    from src.utils.llm_adapter import get_llm_client_for_task, get_role_model_name

    cfg = get_config()
    task = "judgment_verify" if "judgment_verify" in (cfg.task_routing or {}) else "literal_translation"
    model_key, params = cfg.resolve_task_model(task)
    role = cfg.get_task_role(task)
    llm = get_llm_client_for_task(task)
    model_name = get_role_model_name(role) or model_key
    return llm.generate(prompt, model_name=model_name, **params)


def _parse_l2_decisions(raw: str) -> Dict[str, Dict[str, str]]:
    from src.utils.llm_adapter import robust_json_loads

    try:
        payload = robust_json_loads(raw)
    except Exception:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("decisions"), list):
        items = payload["decisions"]
    elif isinstance(payload, list):
        items = payload
    else:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        nid = str(it.get("id") or "").strip()
        verdict = str(it.get("verdict") or it.get("status") or "").strip().lower()
        if not nid or verdict not in ("confirmed", "unsure", "rejected"):
            continue
        out[nid] = {"verdict": verdict,
                    "reason": str(it.get("reason") or "").strip()[:160]}
    return out


def verify_notes_l2(notes: Sequence[Dict[str, Any]], chapter_texts: Dict[str, str],
                    llm_generate: Optional[Callable[[str], str]] = None,
                    batch_size: int = 10) -> Dict[str, Any]:
    """Confirm notes in batches of <=10.  Unconfirmed notes become dormant.

    L2 is advisory and fail-open: an API/parse failure never blocks translation;
    it only prevents injection for that batch.
    """
    notes_out = [dict(n) for n in notes]
    gen = llm_generate or default_l2_generate
    audit: List[Dict[str, Any]] = []
    batch_size = max(1, min(10, int(batch_size or 10)))
    for base in range(0, len(notes_out), batch_size):
        batch = notes_out[base:base + batch_size]
        pending = [n for n in batch if n.get("status") == "active"]
        if not pending:
            continue
        try:
            raw = gen(build_l2_prompt(pending, chapter_texts))
            decisions = _parse_l2_decisions(raw)
            parse_error = "" if decisions else "empty_or_invalid_json"
        except Exception as exc:  # noqa: BLE001
            decisions = {}
            parse_error = f"llm_error:{type(exc).__name__}: {exc}"
        for n in pending:
            nid = str(n.get("id") or "")
            d = decisions.get(nid)
            if not d:
                n["status"] = "dormant"
                n["l2_reject_reason"] = f"l2_missing_decision:{parse_error or 'id_not_returned'}"
                audit.append({"id": nid, "verdict": "missing", "reason": n["l2_reject_reason"]})
                continue
            verdict = d["verdict"]
            reason = d.get("reason", "")
            if verdict == "confirmed" and not is_prescriptive_claim(str(n.get("claim") or "")):
                n["status"] = "active"
                n["l2_verdict"] = "confirmed"
                n["l2_reason"] = reason
            else:
                n["status"] = "dormant"
                if verdict == "confirmed" and is_prescriptive_claim(str(n.get("claim") or "")):
                    reason = "prescriptive_terminology_claim"
                    verdict = "rejected"
                n["l2_verdict"] = verdict
                n["l2_reject_reason"] = reason or verdict
            audit.append({"id": nid, "verdict": verdict, "reason": reason})
    counts = {"active": 0, "dormant": 0, "dropped": 0}
    for n in notes_out:
        st = str(n.get("status") or "dormant")
        counts[st] = counts.get(st, 0) + 1
    return {"notes": notes_out,
            "audit": {"stage": "l2", "checked": len(notes_out), "counts": counts,
                      "decisions": audit}}


def replace_notes_in_profile(profile: Dict[str, Any], notes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    prof = dict(profile)
    for ntype, key in PROFILE_NOTE_KEYS.items():
        prof[key] = [dict(n) for n in notes if n.get("type") == ntype]
    return prof


# ────────────────────────────────────────────────────────────────────
# Runtime slicing / injection
# ────────────────────────────────────────────────────────────────────

@dataclass
class JudgmentContext:
    block: str = ""
    selected: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    chars: int = 0
    budget: int = 0
    cache_key: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chars": self.chars,
            "budget": self.budget,
            "cache_key": self.cache_key,
            "selected": [{k: n.get(k) for k in
                          ("id", "type", "chapters", "claim", "confidence")}
                         for n in self.selected],
            "skipped": self.skipped,
        }


def _note_line(note: Dict[str, Any]) -> str:
    typ = note.get("type")
    chapters = "→".join(str(c) for c in (note.get("chapters") or []))
    label = {"voice": "音色", "setup_payoff": "伏笔/回收", "arc": "叙事弧线"}.get(typ, "注记")
    claim = re.sub(r"\s+", " ", str(note.get("claim") or "")).strip()
    return f"- {label}[{chapters}] {claim}"


def render_review_fix_feedback(issues: Sequence[Dict[str, Any]]) -> str:
    """Render verified structural review issues as retry-prompt instructions."""
    lines = ["【成书终审结构修复（P0，必须逐条落实）】"]
    for idx, issue in enumerate(issues, 1):
        category = str(issue.get("category") or "other")
        quote = re.sub(r"\s+", " ", str(issue.get("quote") or "")).strip()
        note = re.sub(r"\s+", " ", str(issue.get("note") or "")).strip()
        suggestion = re.sub(r"\s+", " ", str(issue.get("suggestion") or "")).strip()
        line = f"{idx}. [{category}] 锚句: {quote}"
        if note:
            line += f"；问题: {note}"
        if suggestion:
            line += f"；最小修复: {suggestion}"
        lines.append(line)
    lines.append("约束：保留未涉及段落；不得删除或改写锚句之外的正常内容；不得新增无原文依据的信息。")
    return "\n".join(lines) + "\n"


def _quote_present_in_chunk(note: Dict[str, Any], chunk_norm: str) -> bool:
    for ev in note.get("evidence") or []:
        # L1 stores the source slice it actually matched.  A note may carry a
        # slightly different ``quote`` (typo/punctuation mismatch) that still
        # passed the fuzzy branch; using the original quote here would then
        # miss the very chunk the evidence points at and lose anchor priority.
        qn = normalize_text_for_match(
            str(ev.get("matched_quote") or ev.get("quote") or "")
        )
        if len(qn) >= 2 and qn in chunk_norm:
            return True
    return False


def build_judgment_context(profile: Optional[Dict[str, Any]], chapter_id: str,
                           chunk_text: str = "", budget_chars: int = 300,
                           min_confidence: float = 0.55,
                           include_types: Optional[Sequence[str]] = None,
                           max_notes: int = 6,
                           prompt_version: str = "r131") -> JudgmentContext:
    """Select a deterministic, whole-note-only slice for one chunk.

    Notes whose evidence quote occurs in this exact chunk are considered first;
    among equally anchored (or equally unanchored) notes, selection is
    confidence-first (ties: voice > setup_payoff > arc).  Without the anchor
    gate, a high-confidence chapter-wide voice note can crowd out the setup or
    payoff note that actually matches the chunk being translated.  Notes are
    never cut mid-sentence; an over-budget note is skipped and recorded.
    """
    budget = max(0, int(budget_chars or 0))
    ctx = JudgmentContext(budget=budget)
    if isinstance(profile, dict) and profile.get("schema_version") not in (None, 1, 2):
        ctx.skipped.append({"reason": "unsupported_schema_version",
                            "value": profile.get("schema_version")})
        return ctx
    include = set(include_types or VALID_NOTE_TYPES)
    chunk_norm = normalize_text_for_match(chunk_text)
    candidates: List[Tuple[Tuple[float, int, int, int, str], Dict[str, Any], bool]] = []
    for note in active_notes(profile, min_confidence=min_confidence):
        if note.get("type") not in include:
            continue
        if not note_covers_chapter(note, chapter_id):
            continue
        anchor_hit = _quote_present_in_chunk(note, chunk_norm) if chunk_norm else False
        exact_evidence_chapter = any(
            str(ev.get("chapter") or "") == str(chapter_id)
            for ev in (note.get("evidence") or [])
        )
        try:
            conf = float(note.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        key = (int(anchor_hit), round(conf, 6), TYPE_PRIORITY.get(note.get("type"), 0),
               int(exact_evidence_chapter), str(note.get("id") or ""))
        candidates.append((key, note, anchor_hit))
    candidates.sort(key=lambda x: x[0], reverse=True)

    header = "【全书判断层注记（仅作参考；术语/译名以决策库为准）】\n"
    if budget <= len(header):
        if candidates:
            ctx.skipped.append({"reason": "budget_smaller_than_header", "count": len(candidates)})
        return ctx
    lines: List[str] = []
    used = len(header)
    for key, note, _ in candidates:
        if len(ctx.selected) >= max(1, int(max_notes or 1)):
            ctx.skipped.append({"id": note.get("id"), "reason": "max_notes",
                                "confidence": note.get("confidence")})
            continue
        line = _note_line(note) + "\n"
        if used + len(line) > budget:
            ctx.skipped.append({"id": note.get("id"), "reason": "budget",
                                "confidence": note.get("confidence"), "chars": len(line)})
            continue
        lines.append(line)
        ctx.selected.append(note)
        used += len(line)
    if not lines:
        return ctx
    ctx.block = header + "".join(lines)
    ctx.chars = len(ctx.block)
    fingerprint = {
        "schema_version": (profile or {}).get("schema_version") if isinstance(profile, dict) else None,
        "generation": (profile or {}).get("generation") if isinstance(profile, dict) else None,
        "chapter": chapter_id,
        "notes": [n.get("id") for n in ctx.selected],
        "prompt_version": prompt_version,
        "budget": budget,
    }
    ctx.cache_key = hashlib.sha256(json.dumps(
        fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return ctx


def render_global_review_context(profile: Optional[Dict[str, Any]],
                                 budget_chars: int = 4000,
                                 min_confidence: float = 0.55) -> str:
    """Compact global narrative context for editor_review --global."""
    notes = active_notes(profile, min_confidence=min_confidence)
    lines: List[str] = []
    used = 0
    header = "【全书判断层（供弧线/语气审读参考，不是硬约束）】\n"
    used += len(header)
    for n in sorted(notes, key=lambda x: (-float(x.get("confidence") or 0),
                                           -TYPE_PRIORITY.get(x.get("type"), 0))):
        line = _note_line(n) + "\n"
        if used + len(line) > budget_chars:
            continue
        lines.append(line)
        used += len(line)
    return header + "".join(lines) if lines else ""


def build_audit_payload(profile: Dict[str, Any], l1_audit: Optional[Dict[str, Any]] = None,
                        l2_audit: Optional[Dict[str, Any]] = None,
                        generation: Optional[Dict[str, Any]] = None,
                        extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    notes = all_notes(profile)
    counts = {"active": 0, "dormant": 0, "dropped": 0}
    for n in notes:
        st = str(n.get("status") or "dormant")
        counts[st] = counts.get(st, 0) + 1
    payload = {
        "generated_at": _now(),
        "schema_version": JUDGMENT_SCHEMA_VERSION,
        "counts": counts,
        "notes": notes,
        "l1": l1_audit or {},
        "l2": l2_audit or {},
        "generation": generation or profile.get("generation") or {},
    }
    if extra:
        payload.update(extra)
    return payload


def build_audit_markdown(audit: Dict[str, Any]) -> str:
    counts = audit.get("counts") or {}
    lines = [
        "# 判断层注记审计（Round 131）",
        "",
        f"- 生成时间: {audit.get('generated_at', '')}",
        f"- active: {counts.get('active', 0)} / dormant: {counts.get('dormant', 0)}"
        f" / dropped: {counts.get('dropped', 0)}",
    ]
    gen = audit.get("generation") or {}
    if gen:
        lines.append(f"- 生成模型: {gen.get('model', '')}；参数: {json.dumps(gen.get('params') or {}, ensure_ascii=False)}")
    lines += ["", "## 非 active 注记", ""]
    rows = [n for n in (audit.get("notes") or []) if n.get("status") != "active"]
    if not rows:
        lines.append("- 无。")
    else:
        lines += ["| id | type | status | 原因 | claim |", "|---|---|---|---|---|"]
        for n in rows:
            reason = (n.get("l1_reject_reason") or n.get("l2_reject_reason")
                      or n.get("schema_reject_reason") or n.get("pairing_status") or "")
            claim = str(n.get("claim") or "").replace("|", "\\|")[:120]
            lines.append(f"| {n.get('id', '')} | {n.get('type', '')} | {n.get('status', '')} "
                         f"| {reason} | {claim} |")
    lines += ["", "---", "*注记是 advisory 制品；L1/L2 未通过的注记只保留审计，不进入 prompt。*", ""]
    return "\n".join(lines)


def write_judgment_audit(project_root: Path, audit: Dict[str, Any],
                         json_path: Optional[Path] = None,
                         md_path: Optional[Path] = None) -> Dict[str, str]:
    root = Path(project_root)
    jp = Path(json_path) if json_path else root / "output" / "style_profile_audit.json"
    mp = Path(md_path) if md_path else root / "output" / "style_profile_audit.md"
    atomic_write_json(jp, audit)
    atomic_write_text(mp, build_audit_markdown(audit))
    return {"json": str(jp), "md": str(mp)}


def judgment_cache_key(profile: Optional[Dict[str, Any]], chapter_id: str,
                       chapter_text: str, prompt_version: str = "r131") -> str:
    payload = {
        "profile": sha256_text(json.dumps(profile or {}, ensure_ascii=False, sort_keys=True)),
        "chapter_id": chapter_id,
        "chapter_hash": sha256_text(chapter_text),
        "prompt_version": prompt_version,
        "schema_version": JUDGMENT_SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
