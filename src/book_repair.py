"""Deterministic, evidence-gated repair of finished chunks (Round 156).

This module is the book-level repair stage for defects that are invisible to
ordinary lint/proofread checks: person-name substitutions introduced by a
rewrite, a mechanically provable appositive contamination, a small set of
factually corrupted glossary notes, Markdown footnote definitions that lost
their colon, and source paragraphs that disappeared between two well-supported
neighbours, plus a deliberately small set of canonical surface-name aliases
confirmed by the work profile.  It never rewrites prose freely and never uses
an LLM.

Two principles keep the stage safe:

1. A name substitution is repaired only when the source chunk, raw draft and
   final/literary sentence or sequence provide aligning evidence.  Sentence
   evidence is accepted only for one-for-one safe substitutions with local
   context and, where needed, paragraph-level source corroboration.  A missing
   raw paragraph is restored only when it has no final counterpart and both
   surrounding raw paragraphs map to adjacent final paragraphs; multiple
   plausible insertions or unresolved footnote dependencies block delivery.
   If the evidence is merely suggestive, the chunk becomes a blocking finding
   instead of a guess.
2. Every mutation is preceded by a byte-for-byte backup and recorded in a
   manifest containing before/after hashes.  ``verify_book_repairs`` fails when
   the manifest no longer describes the current finals.

Dry-run is the default.  The one-click workflow runs ``--apply`` after
``finalize`` and before ``merge``; the final gate re-runs verification from the
current files, so a stale repair report cannot certify a later edit.
"""
from __future__ import annotations

import difflib
import datetime
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BOOK_REPAIR_SCHEMA_VERSION = 1
BOOK_REPAIR_REPORT_NAME = "book_repair_report.json"

_BR_ALIASES: Dict[str, Tuple[str, ...]] = {
    "LETO": (
        "雷托·厄崔迪公爵", "雷托·厄崔迪斯", "雷托·厄崔迪", "雷托公爵", "雷托",
    ),
    "PAUL": (
        "保罗·厄崔迪", "保罗·亚崔迪", "保罗·阿特雷迪斯", "保罗·厄崔迪斯",
        "保罗·阿特雷德斯", "保罗",
    ),
    "JESSICA": ("杰西卡夫人", "杰西卡"),
    "MOHIAM": (
        "盖乌斯·海伦·莫希阿姆", "盖乌斯夫人", "莫希阿姆", "盖乌斯", "海伦",
    ),
    "FEYD": (
        "菲德-罗萨", "费德-罗萨", "菲德·罗萨", "费德·罗萨", "菲德", "费德",
    ),
    # Bare "戈姆" is intentionally omitted: it also occurs in legitimate
    # weapon/glossary contexts.  Only the full person-shaped surface is audited.
    "GOM": ("戈姆·贾巴尔",),
    "PITER": ("皮特", "彼得"),
    "TRUTHSAYER": ("真言者", "真言师"),
}
_BR_TERMS: Tuple[Tuple[str, str], ...] = tuple(sorted(
    ((surface, key) for key, aliases in _BR_ALIASES.items() for surface in aliases),
    key=lambda item: len(item[0]), reverse=True,
))

_BR_EN_NAMES: Dict[str, Tuple[str, ...]] = {
    "LETO": ("Leto",),
    "PAUL": ("Paul",),
    "JESSICA": ("Jessica",),
    "MOHIAM": ("Mohiam", "Gaius"),
    "FEYD": ("Feyd",),
    "GOM": ("Gom Jabbar",),
    "PITER": ("Piter",),
    "TRUTHSAYER": ("Truthsayer",),
}

# Only these directions are accepted by the sequence-based core-person rule.
# FEYD is handled separately because the literary draft itself can carry the
# historical Gom contamination.
_BR_SAFE_SUBSTITUTIONS: Dict[str, str] = {
    "MOHIAM": "JESSICA",
    "PAUL": "LETO",
}
_BR_CANONICAL_SURFACE: Dict[str, str] = {
    "FEYD": "菲德-罗萨",
    "LETO": "雷托",
    "PAUL": "保罗",
    "JESSICA": "杰西卡",
    "MOHIAM": "盖乌斯·海伦·莫希阿姆",
}

_BR_GOM_NOTE = "戈姆·贾巴尔：贝尼·杰瑟里特姐妹会用于测试人类与动物之别的毒针武器。"
_BR_GOM_ALIAS_NOTE = (
    "戈姆刺（戈姆·贾巴尔，gom jabbar）：贝尼·杰瑟里特姐妹会用于测试人类与动物之别的毒针武器。"
)
_BR_GOM_SURFACES = ("戈姆·贾巴尔", "戈姆刺")
_BR_FAUFRELUCHES_NOTE = (
    "“法弗勒鲁奇”（Faufreluches）：帝国社会等级制度的统称，规定各阶层与家族的地位及往来规范。"
)
_BR_GOM_WRONG_MARKERS = (
    "弗雷曼传说", "弗雷曼的传说", "禁忌体系", "沙虫", "沙虫之牙",
)

# These are not contextual titles or free translation choices.  They are
# alternate surface forms for the same source entity, with the canonical
# direction established by the work profile and entity registry.  Keep this
# list deliberately small: ambiguous person names belong in a separate,
# evidence-mapped review rather than a generic find/replace pass.
_BR_CANONICAL_ALIAS_RULES: Tuple[Tuple[str, str, str, str], ...] = (
    ("阿拉基斯", "厄拉科斯", "Arrakis",
     "decision id=5 translation=厄拉科斯; registry id=6 canonical=厄拉科斯"),
    ("贝尼杰瑟里特", "贝尼·杰瑟里特", "Bene Gesserit",
     "profile: Bene Gesserit=贝尼·杰瑟里特; registry variant"),
    ("沙杜特梅普斯", "沙杜特·梅普斯", "Shadout Mapes",
     "registry: Shadout Mapes canonical=沙杜特·梅普斯"),
    # Feyd-Rautha has three surface forms in the current final corpus:
    # 菲德-罗萨 (first full introduction in ch01_chunk021), 费德-劳萨
    # (the dominant later drift), and 菲德-劳萨.  Use the stable first-use
    # transliteration; the question-mark form is an EPUB extraction artifact.
    ("费德-劳萨", "菲德-罗萨", "Feyd",
     "first-use ch01_chunk021; Feyd-Rautha; variants 费德-劳萨/菲德-劳萨/费德-？劳萨"),
    ("菲德-劳萨", "菲德-罗萨", "Feyd",
     "first-use ch01_chunk021; Feyd-Rautha; variants 费德-劳萨/菲德-劳萨/费德-？劳萨"),
    ("费德-？劳萨", "菲德-罗萨", "Feyd",
     "first-use ch01_chunk021; Feyd-Rautha; EPUB extraction artifact"),
    # The corpus contains the isolated vocative "图菲尔" in 16 completed
    # chunks, while the active registry owner and Decision 236 use
    # "杜菲·哈瓦特".  Every affected chunk's source contains "Thufir", so the
    # short-form repair remains source-gated and never applies to free prose.
    ("图菲尔", "杜菲", "Thufir",
     "registry id=31 canonical=杜菲·哈瓦特; decision id=236; source-gated Thufir"),
    # Two legacy Decisions disagree on the Landsraad rendering.  The active
    # registry owner and the book-wide Decisions use 兰兹拉德; the older
    # 兰萨德 form survives in completed prose.  The source gate prevents this
    # rule from touching unrelated CJK prose.
    ("兰萨德", "兰兹拉德", "Landsraad",
     "registry id=266 canonical=兰兹拉德; decision id=22/3442; source-gated Landsraad"),
)

# English glosses in parentheses and footnotes are evidence, not prose.  A
# model can transcribe one letter incorrectly (``stiltent`` for ``stilltent``)
# while the surrounding Chinese sentence remains correct.  Repair only when
# the gloss is absent from this chunk's source and exactly one source token is
# a close spelling variant; ambiguous candidates remain untouched.
_BR_GLOSS_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9’'\-?]{4,}")
_BR_GLOSS_PAREN_RE = re.compile(
    r"[（(]\s*([A-Za-z][A-Za-z0-9’'\-?]{4,})\s*[）)]")
_BR_GLOSS_QUOTE_RE = re.compile(
    r"原文(?:作|为)\s*[“\"]([A-Za-z][A-Za-z0-9’'\-?]{4,})[”\"]")


def _br_normalize_gloss_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.casefold())


def _br_plan_source_gloss_spelling(
    text: str, source: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Restore a single-letter source-term typo inside an explicit gloss.

    The pass is deliberately narrow: it considers only ASCII tokens enclosed
    in parentheses or quoted by ``原文作/原文为``.  It never rewrites the
    translated prose around the gloss and never guesses when more than one
    source spelling is close enough.
    """
    if not text or not source:
        return text, []
    source_tokens = _BR_GLOSS_TOKEN_RE.findall(source)
    source_norm = {_br_normalize_gloss_token(token): token
                   for token in source_tokens}
    candidates = set(_BR_GLOSS_PAREN_RE.findall(text))
    candidates.update(_BR_GLOSS_QUOTE_RE.findall(text))
    working = text
    changes: List[Dict[str, Any]] = []
    for candidate in sorted(candidates):
        norm = _br_normalize_gloss_token(candidate)
        if not norm or norm in source_norm:
            continue
        scored = sorted(
            ((difflib.SequenceMatcher(None, norm, source_norm_key).ratio(),
              source_token)
             for source_norm_key, source_token in source_norm.items()),
            reverse=True,
        )
        if not scored:
            continue
        best_ratio, best_token = scored[0]
        second_ratio = scored[1][0] if len(scored) > 1 else 0.0
        best_norm = _br_normalize_gloss_token(best_token)
        if (best_ratio < 0.90
                or best_ratio - second_ratio < 0.12
                or abs(len(norm) - len(best_norm)) > 2):
            continue
        # A singular gloss such as Messiah versus source messiahs is a valid
        # lexical form, not a transcription typo.  Prefix-only differences are
        # therefore left untouched; the stiltent/stilltent case is a true
        # internal substitution and still passes this guard.
        if norm.startswith(best_norm) or best_norm.startswith(norm):
            continue
        before = working.count(candidate)
        if not before:
            continue
        # Restrict the substitution to the two recognised gloss contexts.
        working = _BR_GLOSS_PAREN_RE.sub(
            lambda match: match.group(0).replace(candidate, best_token), working)
        working = _BR_GLOSS_QUOTE_RE.sub(
            lambda match: match.group(0).replace(candidate, best_token), working)
        count = before - working.count(candidate)
        if count <= 0:
            continue
        changes.append({
            "rule": "source_gloss_typo_restore",
            "from": candidate,
            "to": best_token,
            "count": count,
            "context": "source_term_gloss",
            "evidence": {
                "source_term": best_token,
                "similarity": round(best_ratio, 3),
                "runner_up_similarity": round(second_ratio, 3),
            },
        })
    return working, changes

_BR_MD_DEF_RE = re.compile(r"^\s*\[\^([^\]]+)\]:\s?")
_BR_MALFORMED_MD_DEF_RE = re.compile(
    r"^(\s*\[\^([^\]]+)\]\s+)(\S.*?)(\r?\n?)$")
_BR_BODY_REF_RE = re.compile(r"\[\^([^\]]+)\](?!:)")
_BR_LEGACY_NOTE_RE = re.compile(
    r"^\s*[\u00b9\u00b2\u00b3\u2070-\u2079]+(?:[\.、:：]|\s)")
_BR_OMISSION_MIN_CHARS = 60
_BR_OMISSION_MAX_MATCH = 0.03
_BR_OMISSION_MIN_ANCHOR = 0.20
_BR_OMISSION_NGRAM_SIZE = 4


def _br_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _br_excerpt(text: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()[:limit]


def _br_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _br_atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp-book-repair")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _br_load_sources(root: Path) -> Dict[str, str]:
    db = root / "db" / "workflow.db"
    if not db.exists():
        return {}
    out: Dict[str, str] = {}
    connection = sqlite3.connect(str(db))
    try:
        for chunk_id, text in connection.execute(
                "SELECT chunk_id, text_content FROM chunk_tasks"):
            if isinstance(chunk_id, str) and isinstance(text, str):
                out[chunk_id] = text
    finally:
        connection.close()
    return out


def _br_source_counts(source: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for key, names in _BR_EN_NAMES.items():
        total = 0
        for name in names:
            pattern = r"\b" + re.escape(name).replace(r"\ ", r"[-\s]+") + r"\b"
            total += len(re.findall(pattern, source, re.IGNORECASE))
        counts[key] = total
    return counts


def _br_note_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"(?m)^\s*\[\^[^\]]+\]:.*$", text):
        spans.append((match.start(), match.end()))
    return spans


def _br_scan_mentions(
    text: str, body_only: bool = False,
) -> List[Tuple[int, int, str, str]]:
    """Return ``(start, end, entity_key, surface)`` for explicit names."""
    note_spans = _br_note_spans(text) if body_only else []

    def in_note(position: int) -> bool:
        return any(start <= position < end for start, end in note_spans)

    out: List[Tuple[int, int, str, str]] = []
    cursor = 0
    length = len(text)
    while cursor < length:
        hit = next((item for item in _BR_TERMS if text.startswith(item[0], cursor)), None)
        if hit is None:
            cursor += 1
            continue
        surface, key = hit
        end = cursor + len(surface)
        if not body_only or not in_note(cursor):
            out.append((cursor, end, key, surface))
        cursor = end
    return out


def _br_key_counts(text: str, body_only: bool = False) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _start, _end, key, _surface in _br_scan_mentions(text, body_only=body_only):
        counts[key] = counts.get(key, 0) + 1
    return counts


def _br_apply_span_changes(
    text: str, changes: Iterable[Dict[str, Any]],
) -> str:
    ordered = sorted(changes, key=lambda item: int(item["start"]), reverse=True)
    repaired = text
    for change in ordered:
        repaired = (repaired[:int(change["start"])] + str(change["to"])
                    + repaired[int(change["end"]):])
    return repaired


_BR_SENTENCE_DELIMS = "\n。！？!?"


def _br_sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char not in _BR_SENTENCE_DELIMS:
            continue
        end = index + 1
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _br_sentence_mentions(
    text: str, start: int, end: int,
    mentions: Iterable[Tuple[int, int, str, str]],
) -> List[Tuple[int, int, str, str]]:
    return [item for item in mentions if start <= item[0] < end]


def _br_mask_entity_surfaces(text: str) -> str:
    masked = text
    for surface, _key in _BR_TERMS:
        masked = masked.replace(surface, "<ENTITY>")
    return re.sub(r"\s+", "", masked)


def _br_paragraphs(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _br_paragraph_index(text: str, position: int) -> int:
    return len(re.findall(r"\n\s*\n", text[:position]))


def _br_is_note_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if (_BR_MD_DEF_RE.match(lines[0])
            or _BR_MALFORMED_MD_DEF_RE.match(lines[0])
            or _BR_LEGACY_NOTE_RE.match(lines[0])):
        return True
    return all(
        _BR_MD_DEF_RE.match(line)
        or _BR_MALFORMED_MD_DEF_RE.match(line)
        or _BR_LEGACY_NOTE_RE.match(line)
        for line in lines
    )


def _br_paragraph_records(text: str) -> List[Dict[str, Any]]:
    """Return non-empty paragraphs with exact source offsets and note kind."""
    records: List[Dict[str, Any]] = []
    offset = 0
    for part in re.split(r"(\n\s*\n)", text):
        if not part:
            continue
        if re.fullmatch(r"\n\s*\n", part):
            offset += len(part)
            continue
        if part.strip():
            lead = len(part) - len(part.lstrip())
            trail = len(part) - len(part.rstrip())
            content = part[lead:len(part) - trail]
            records.append({
                "text": content,
                "start": offset + lead,
                "end": offset + len(part) - trail,
                "kind": "note" if _br_is_note_block(content) else "body",
            })
        offset += len(part)
    return records


def _br_similarity_text(text: str) -> str:
    """Normalize only format noise that cannot carry paragraph identity."""
    clean = re.sub(r"(?m)^\s*#+\s*", "", text)
    clean = re.sub(r"(?m)^\s*\[\^[^\]]+\]:\s*", "", clean)
    clean = re.sub(r"(?m)^\s*[\u00b9\u00b2\u00b3\u2070-\u2079]+\s*", "", clean)
    clean = re.sub(r"\[\^[^\]]+\]", "", clean)
    return re.sub(r"\s+", "", clean)


def _br_ngrams(text: str) -> set:
    normalized = _br_similarity_text(text)
    if not normalized:
        return set()
    if len(normalized) <= _BR_OMISSION_NGRAM_SIZE:
        return {normalized}
    return {
        normalized[index:index + _BR_OMISSION_NGRAM_SIZE]
        for index in range(len(normalized) - _BR_OMISSION_NGRAM_SIZE + 1)
    }


def _br_jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _br_best_paragraph_matches(
    grams: set, candidate_grams: List[set], limit: int = 2,
) -> List[Tuple[int, float]]:
    scores = [
        (_index, _br_jaccard(grams, candidate))
        for _index, candidate in enumerate(candidate_grams)
    ]
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:limit]


def _br_omission_insert_position(
    text: str, final_records: List[Dict[str, Any]],
    final_body: List[Dict[str, Any]], gap_index: int,
) -> int:
    if gap_index < len(final_body):
        return int(final_body[gap_index]["start"])
    if final_body:
        last_end = int(final_body[-1]["end"])
        for record in final_records:
            if record["kind"] == "note" and int(record["start"]) >= last_end:
                return int(record["start"])
    return len(text)


def _br_apply_paragraph_insertions(
    text: str, insertions: List[Tuple[int, List[str]]],
) -> str:
    repaired = text
    for position, paragraphs in sorted(
            insertions, key=lambda item: item[0], reverse=True):
        block = "\n\n".join(paragraphs)
        before = repaired[:position]
        prefix = "" if (not before or re.search(r"\n\s*\n\Z", before)) else "\n\n"
        suffix = "\n\n" if position < len(repaired) else "\n"
        repaired = (before + prefix + block + suffix + repaired[position:])
    return repaired


def _br_plan_omissions(
    text: str, raw: str, source: str, literary: str,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Restore a raw paragraph only when a final gap is structurally provable.

    Raw and final share a language, while ordinary literary rewriting keeps
    enough four-character overlap for reliable paragraph anchors.  A candidate
    must be substantial, have no match anywhere in final (including footnote
    blocks), and sit between two raw neighbours that map one paragraph apart.
    Multiple adjacent mappings mean the insertion point is ambiguous and block
    delivery instead of being guessed.
    """
    if not text.strip() or not raw.strip():
        return text, [], []
    final_records = _br_paragraph_records(text)
    raw_records = _br_paragraph_records(raw)
    final_body = [record for record in final_records if record["kind"] == "body"]
    raw_body = [record for record in raw_records if record["kind"] == "body"]
    if len(final_body) < 2 or len(raw_body) < 3:
        return text, [], []

    final_body_grams = [_br_ngrams(record["text"]) for record in final_body]
    final_all_grams = [_br_ngrams(record["text"]) for record in final_records]
    raw_matches: List[Dict[str, Any]] = []
    for record in raw_body:
        grams = _br_ngrams(record["text"])
        raw_matches.append({
            "grams": grams,
            "normalized_chars": len(_br_similarity_text(record["text"])),
            "all_best": _br_best_paragraph_matches(grams, final_all_grams, limit=1),
            "body_best": _br_best_paragraph_matches(grams, final_body_grams, limit=2),
        })

    candidate_indices = []
    for index, match in enumerate(raw_matches):
        if match["normalized_chars"] < _BR_OMISSION_MIN_CHARS:
            continue
        all_best = match["all_best"]
        body_best = match["body_best"]
        if not all_best or not body_best:
            continue
        if (all_best[0][1] < _BR_OMISSION_MAX_MATCH
                and body_best[0][1] < _BR_OMISSION_MAX_MATCH):
            candidate_indices.append(index)
    if not candidate_indices:
        return text, [], []

    groups: List[List[int]] = []
    for index in candidate_indices:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    source_body = [
        record for record in _br_paragraph_records(source)
        if record["kind"] == "body"
    ]
    source_aligned = len(source_body) == len(raw_body)
    literary_body = [
        record for record in _br_paragraph_records(literary)
        if record["kind"] == "body"
    ]
    literary_grams = [_br_ngrams(record["text"]) for record in literary_body]
    final_defs = set()
    for record in final_records:
        if record["kind"] != "note":
            continue
        final_defs.update(re.findall(r"(?m)^\s*\[\^([^\]]+)\]:", record["text"]))

    changes: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []
    insertions: List[Tuple[int, List[str]]] = []
    for group in groups:
        start, end = group[0], group[-1]
        if start <= 0 or end + 1 >= len(raw_body):
            continue
        previous = raw_matches[start - 1]["body_best"]
        following = raw_matches[end + 1]["body_best"]
        if not previous or not following:
            continue
        previous_index, previous_score = previous[0]
        following_index, following_score = following[0]
        if (previous_score < _BR_OMISSION_MIN_ANCHOR
                or following_score < _BR_OMISSION_MIN_ANCHOR
                or following_index != previous_index + 1):
            continue

        plausible = sorted({
            (prev_index, next_index)
            for prev_index, prev_score in previous
            if prev_score >= _BR_OMISSION_MIN_ANCHOR
            for next_index, next_score in following
            if next_score >= _BR_OMISSION_MIN_ANCHOR
            and next_index == prev_index + 1
        })
        gap_index = following_index
        group_texts = [raw_body[index]["text"] for index in group]
        excerpt = _br_excerpt(" / ".join(group_texts))
        if len(plausible) > 1:
            blocking.append({
                "kind": "source_paragraph_omitted",
                "severity": "high",
                "detail": (
                    f"raw 段 {group} 无 final 对应，但候选插入位置不唯一: "
                    f"{plausible[:4]}；拒绝猜测"),
            })
            continue

        unresolved_refs: List[str] = []
        has_note_lines = False
        for index in group:
            candidate_text = raw_body[index]["text"]
            if re.search(
                    r"(?m)^\s*(?:\[\^[^\]]+\]:|[\u00b9\u00b2\u00b3\u2070-\u2079]+\s)",
                    candidate_text):
                has_note_lines = True
            refs = set(_BR_BODY_REF_RE.findall(candidate_text))
            unresolved_refs.extend(sorted(refs - final_defs))
        if has_note_lines or unresolved_refs:
            reason = ("含脚注定义，需先确定注释归属"
                      if has_note_lines
                      else f"脚注引用未在 final 定义: {sorted(set(unresolved_refs))}")
            blocking.append({
                "kind": "source_paragraph_omitted",
                "severity": "high",
                "detail": f"raw 段 {group} 无 final 对应，但{reason}；拒绝猜测",
            })
            continue

        source_supported = bool(
            source_aligned
            and all(_br_similarity_text(source_body[index]["text"])
                    for index in group)
        )
        literary_supported = any(
            _br_best_paragraph_matches(
                raw_matches[index]["grams"], literary_grams, limit=1)[0][1]
            >= _BR_OMISSION_MIN_ANCHOR
            for index in group
        ) if literary_grams else False
        insert_position = _br_omission_insert_position(
            text, final_records, final_body, gap_index)
        insertions.append((insert_position, group_texts))
        for index in group:
            match = raw_matches[index]
            source_index = index if source_aligned else None
            changes.append({
                "rule": "source_paragraph_restored",
                "from": "(final 中缺失)",
                "to": _br_excerpt(raw_body[index]["text"]),
                "count": 1,
                "start": insert_position,
                "raw_paragraph_index": index,
                "final_insert_before": gap_index,
                "evidence": {
                    "best_final_similarity": round(match["body_best"][0][1], 4),
                    "previous_anchor_similarity": round(previous_score, 4),
                    "next_anchor_similarity": round(following_score, 4),
                    "source_paragraph_index": source_index,
                    "source_supported": source_supported,
                    "literary_supported": literary_supported,
                },
                "context": (
                    f"raw 段 {index} 位于 final 段 {previous_index} 与 "
                    f"{following_index} 之间；候选位置 {plausible}"),
            })

    if not changes:
        return text, [], blocking
    repaired = _br_apply_paragraph_insertions(text, insertions)
    return repaired, changes, blocking


def _br_english_count(text: str, key: str) -> int:
    total = 0
    for name in _BR_EN_NAMES.get(key, ()):
        pattern = r"\b" + re.escape(name).replace(r"\ ", r"[-\s]+") + r"\b"
        total += len(re.findall(pattern, text, re.IGNORECASE))
    return total


def _br_plan_sentence_core_swaps(
    text: str, literary: str, source: str,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Repair one-for-one identity swaps inside locally matching sentences.

    Whole-chunk mention sequences can drift when a translator splits paragraphs,
    so the older sequence-only matcher misses genuine cases and rejects some
    corroborated ones.  This pass aligns complete sentences instead: all entity
    differences must be accepted safe-substitution directions, and a sentence
    is changed only when its masked text closely matches a literary sentence.
    A lower similarity threshold is allowed only when the corresponding source
    paragraph independently supports the target name over the wrong one.
    """
    final_mentions = _br_scan_mentions(text, body_only=True)
    literary_mentions = _br_scan_mentions(literary, body_only=True)
    if not final_mentions or not literary_mentions:
        return text, [], []

    final_records: List[Dict[str, Any]] = []
    for start, end in _br_sentence_spans(text):
        mentions = _br_sentence_mentions(
            text, start, end, final_mentions)
        keys = [item[2] for item in mentions]
        if not keys or not any(key in _BR_SAFE_SUBSTITUTIONS for key in keys):
            continue
        final_records.append({
            "start": start,
            "end": end,
            "mentions": mentions,
            "keys": keys,
            "masked": _br_mask_entity_surfaces(text[start:end]),
        })
    if not final_records:
        return text, [], []

    literary_by_length: Dict[int, List[Dict[str, Any]]] = {}
    for start, end in _br_sentence_spans(literary):
        mentions = _br_sentence_mentions(
            literary, start, end, literary_mentions)
        if not mentions:
            continue
        keys = [item[2] for item in mentions]
        literary_by_length.setdefault(len(keys), []).append({
            "start": start,
            "end": end,
            "mentions": mentions,
            "keys": keys,
            "masked": _br_mask_entity_surfaces(literary[start:end]),
        })

    source_paragraphs = _br_paragraphs(source)
    changes: List[Dict[str, Any]] = []
    for final_record in final_records:
        final_keys = final_record["keys"]
        scored: List[Tuple[float, Dict[str, Any], List[Tuple[int, str, str]]]] = []
        for literary_record in literary_by_length.get(len(final_keys), ()):
            literary_keys = literary_record["keys"]
            diffs = [
                (index, wrong, target)
                for index, (wrong, target) in enumerate(
                    zip(final_keys, literary_keys))
                if wrong != target
            ]
            if not diffs:
                continue
            if any(_BR_SAFE_SUBSTITUTIONS.get(wrong) != target
                   for _index, wrong, target in diffs):
                continue
            if any(_br_english_count(source, target) <= 0
                   for _index, _wrong, target in diffs):
                continue
            ratio = difflib.SequenceMatcher(
                None, final_record["masked"], literary_record["masked"],
                autojunk=False).ratio()
            scored.append((ratio, literary_record, diffs))
        if not scored:
            continue
        scored.sort(key=lambda item: item[0], reverse=True)
        best_ratio, best_literary, best_diffs = scored[0]
        second_ratio = scored[1][0] if len(scored) > 1 else 0.0
        if best_ratio - second_ratio < 0.02 and best_ratio < 0.90:
            continue

        paragraph_index = min(
            _br_paragraph_index(text, final_record["start"]),
            max(0, len(source_paragraphs) - 1),
        )
        source_paragraph = (source_paragraphs[paragraph_index]
                            if source_paragraphs else "")
        paragraph_support = all(
            _br_english_count(source_paragraph, target)
            > _br_english_count(source_paragraph, wrong)
            for _index, wrong, target in best_diffs
        )
        if not (best_ratio >= 0.75 or
                (best_ratio >= 0.55 and paragraph_support)):
            continue

        for index, wrong, target in best_diffs:
            start, end, _key, surface = final_record["mentions"][index]
            literary_surface = best_literary["mentions"][index][3]
            changes.append({
                "rule": f"sentence_core_person_{wrong.lower()}_to_{target.lower()}",
                "from": surface,
                "to": literary_surface,
                "count": 1,
                "start": start,
                "end": end,
                "entity": target,
                "evidence": {
                    "sentence_similarity": round(best_ratio, 4),
                    "paragraph_supported": bool(paragraph_support),
                },
                "context": _br_excerpt(
                    text[max(0, start - 50):min(len(text), end + 70)]),
            })
    if not changes:
        return text, [], []
    repaired = _br_apply_span_changes(text, changes)
    return repaired, changes, []


def _br_plan_note_repairs(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Correct only note definitions whose factual predicate is provably wrong."""
    changes: List[Dict[str, Any]] = []
    out_lines: List[str] = []
    for line in text.splitlines(keepends=True):
        match = re.match(r"^(\s*\[\^[^\]]+\]:\s*)(.*?)(\r?\n?)$", line)
        if match is None:
            out_lines.append(line)
            continue
        prefix, content, suffix = match.groups()
        replacement: Optional[str] = None
        rule = ""
        gom_surface = next(
            (surface for surface in _BR_GOM_SURFACES if surface in content), None)
        if (gom_surface is not None
                and any(marker in content for marker in _BR_GOM_WRONG_MARKERS)
                and "贝尼" not in content and "姐妹会" not in content):
            replacement = (_BR_GOM_NOTE if gom_surface == "戈姆·贾巴尔"
                           else _BR_GOM_ALIAS_NOTE)
            rule = "gom_jabbar_fact_repair"
        elif ("法弗勒鲁奇" in content
              and ("弗雷曼对外来者" in content or "贬称" in content)):
            replacement = _BR_FAUFRELUCHES_NOTE
            rule = "faufreluches_fact_repair"
        if replacement is None:
            out_lines.append(line)
            continue
        out_lines.append(prefix + replacement + suffix)
        changes.append({
            "rule": rule,
            "from": _br_excerpt(content),
            "to": _br_excerpt(replacement),
            "count": 1,
            "context": "footnote_definition",
        })
    return "".join(out_lines), changes


def _br_source_term_count(source: str, term: str) -> int:
    pattern = r"\b" + re.escape(term).replace(r"\ ", r"[-\s]+") + r"\b"
    return len(re.findall(pattern, source, re.IGNORECASE))


def _br_plan_canonical_aliases(
    text: str, source: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Normalize only canonical surface aliases confirmed by the work profile.

    This pass intentionally excludes person names whose canonical choice is
    disputed.  Every entry here is a spelling/transliteration drift for the
    same source entity, not a contextual title such as ``公爵``.

    A registry/profile match alone is insufficient: each alias is applied only
    when the corresponding source term occurs in this chunk's source evidence.
    """
    working = text
    changes: List[Dict[str, Any]] = []
    for wrong, right, source_term, basis in _BR_CANONICAL_ALIAS_RULES:
        count = working.count(wrong)
        if not count:
            continue
        source_count = _br_source_term_count(source, source_term)
        if source_count <= 0:
            # A profile/registry alias is not sufficient by itself: the source
            # segment must contain the corresponding term.  Otherwise a stale
            # rule or mismatched source slice could rewrite unrelated prose.
            continue
        first = working.find(wrong)
        working = working.replace(wrong, right)
        changes.append({
            "rule": "canonical_surface_restore",
            "from": wrong,
            "to": right,
            "count": count,
            "context": _br_excerpt(
                text[max(0, first - 50):min(len(text), first + len(wrong) + 80)]),
            "evidence": {
                "source_term": source_term,
                "source_occurrences": source_count,
                "source_supported": source_count > 0,
                "basis": basis,
            },
        })
    return working, changes


def _br_plan_malformed_footnotes(
    text: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Restore the colon in ``[^id] note`` definitions when the body cites ``id``.

    A missing colon makes the line ordinary prose to Markdown and the reader.
    The repair is deliberately limited to a definition-shaped line whose id is
    cited in the body, is not already defined properly, and whose content looks
    like a note rather than a normal paragraph beginning with a reference.
    """
    lines = text.splitlines(keepends=True)
    valid_defs = {
        match.group(1)
        for line in lines
        if (match := _BR_MD_DEF_RE.match(line))
    }
    body_parts = []
    malformed = []
    for index, line in enumerate(lines):
        if _BR_MD_DEF_RE.match(line):
            continue
        match = _BR_MALFORMED_MD_DEF_RE.match(line)
        if match is not None:
            malformed.append((index, match))
            continue
        body_parts.append(line)
    refs = set(_BR_BODY_REF_RE.findall("".join(body_parts)))
    if not malformed or not refs:
        return text, []

    changes: List[Dict[str, Any]] = []
    repaired_lines = list(lines)
    for index, match in malformed:
        prefix, ident, content, suffix = match.groups()
        stripped = content.strip()
        if ident not in refs or ident in valid_defs or len(stripped) < 4:
            continue
        first = stripped[:1]
        if first not in '“"「『*（(' and not re.match(
                r"[\u4e00-\u9fffA-Za-z]", first):
            continue
        replacement = prefix.rstrip() + ": " + stripped + suffix
        repaired_lines[index] = replacement
        changes.append({
            "rule": "malformed_md_footnote_definition",
            "from": _br_excerpt(lines[index]),
            "to": _br_excerpt(replacement),
            "count": 1,
            "footnote_id": ident,
            "context": "footnote_definition",
        })
    if not changes:
        return text, []
    return "".join(repaired_lines), changes


def _br_first_content_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("\ufeff")
        if stripped:
            return stripped
    return ""


def _br_plan_appositive(
    text: str, raw: str, literary: str, source: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Remove ``杰西卡，那位老妇人`` when all drafts and source say old woman."""
    source_flat = re.sub(r"\s+", " ", source)
    if "The old woman was let in" not in source_flat:
        return text, []
    if re.search(r"Jessica\s+was\s+let\s+in", source_flat, re.IGNORECASE):
        return text, []
    if not text.lstrip("\ufeff \t\r\n").startswith("杰西卡，那位老妇人"):
        return text, []
    raw_lead = _br_first_content_line(raw)
    literary_lead = _br_first_content_line(literary)
    if not raw_lead.startswith(("老妇人", "那位老妇人", "那个老妇人")):
        return text, []
    if not literary_lead.startswith(("老妇人", "那位老妇人", "那个老妇人")):
        return text, []
    before = "杰西卡，那位老妇人"
    position = text.find(before)
    if position < 0:
        return text, []
    repaired = text[:position] + "那位老妇人" + text[position + len(before):]
    change = {
        "rule": "appositive_old_woman_not_jessica",
        "from": before,
        "to": "那位老妇人",
        "count": 1,
        "context": _br_excerpt(text[max(0, position - 40):position + 100]),
    }
    return repaired, [change]


def _br_plan_gom_to_feyd(
    text: str, raw: str, source: str,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_counts = _br_source_counts(source)
    final_mentions = _br_scan_mentions(text, body_only=True)
    final_has_gom = any(item[2] == "GOM" for item in final_mentions)
    if not final_has_gom:
        return text, [], []
    if source_counts.get("GOM", 0) or source_counts.get("FEYD", 0) <= 0:
        return text, [], []

    raw_mentions = _br_scan_mentions(raw, body_only=True)
    if not raw_mentions:
        return text, [], [{
            "kind": "feyd_gom_unresolved",
            "severity": "high",
            "detail": ("final 将 Feyd 写成戈姆·贾巴尔，但 raw 缺失，"
                       "无法完成逐项序列证明"),
        }]

    final_keys = [item[2] for item in final_mentions]
    raw_keys = [item[2] for item in raw_mentions]
    normalized = ["FEYD" if key == "GOM" else key for key in final_keys]
    if normalized != raw_keys:
        return text, [], [{
            "kind": "feyd_gom_alignment_requires_review",
            "severity": "high",
            "detail": f"final={final_keys[:24]} raw={raw_keys[:24]}",
        }]

    changes: List[Dict[str, Any]] = []
    for index, mention in enumerate(final_mentions):
        start, end, key, surface = mention
        if key != "GOM":
            continue
        raw_key = raw_mentions[index][2]
        if raw_key != "FEYD":
            return text, [], [{
                "kind": "feyd_gom_alignment_requires_review",
                "severity": "high",
                "detail": f"occurrence {index}: final=GOM raw={raw_key}",
            }]
        target_surface = raw_mentions[index][3] or _BR_CANONICAL_SURFACE["FEYD"]
        changes.append({
            "rule": "gom_jabbar_to_feyd",
            "from": surface,
            "to": target_surface,
            "count": 1,
            "start": start,
            "end": end,
            "entity": "FEYD",
            "context": _br_excerpt(text[max(0, start - 50):min(len(text), end + 70)]),
        })
    if not changes:
        return text, [], []
    repaired = _br_apply_span_changes(text, changes)
    if any(item[2] == "GOM" for item in _br_scan_mentions(repaired, body_only=True)):
        return text, [], [{
            "kind": "feyd_gom_verification_failed",
            "severity": "high",
            "detail": "替换后仍残留戈姆·贾巴尔",
        }]
    return repaired, changes, []


def _br_support_core_swap(
    target: str, raw_counts: Dict[str, int], source_counts: Dict[str, int],
) -> bool:
    if source_counts.get(target, 0) <= 0 or raw_counts.get(target, 0) <= 0:
        return False
    if target == "JESSICA":
        return (raw_counts.get("MOHIAM", 0) == 0
                and source_counts.get("MOHIAM", 0) == 0)
    if target == "LETO":
        return raw_counts.get("LETO", 0) >= raw_counts.get("PAUL", 0)
    return False


def _br_plan_core_swaps(
    text: str, literary: str, raw: str, source: str,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    final_mentions = _br_scan_mentions(text, body_only=True)
    literary_mentions = _br_scan_mentions(literary, body_only=True)
    if not final_mentions or not literary_mentions:
        return text, [], []
    final_keys = [item[2] for item in final_mentions]
    literary_keys = [item[2] for item in literary_mentions]
    if final_keys == literary_keys:
        return text, [], []

    raw_counts = _br_key_counts(raw, body_only=True)
    source_counts = _br_source_counts(source)
    changes: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(None, literary_keys, final_keys, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue
        for literary_index, final_index in zip(range(i1, i2), range(j1, j2)):
            target = literary_keys[literary_index]
            wrong = final_keys[final_index]
            expected_target = _BR_SAFE_SUBSTITUTIONS.get(wrong)
            if expected_target is None:
                # Unrelated name differences are not treated as substitutions.
                continue
            if expected_target != target:
                blocking.append({
                    "kind": "identity_swap_direction_mismatch",
                    "severity": "critical",
                    "detail": f"{wrong}->{target} (expected {expected_target})",
                })
                continue
            if not _br_support_core_swap(target, raw_counts, source_counts):
                blocking.append({
                    "kind": "identity_swap_without_corroboration",
                    "severity": "high",
                    "detail": f"{wrong}->{target} not supported by source/raw",
                })
                continue
            start, end, _key, surface = final_mentions[final_index]
            target_surface = literary_mentions[literary_index][3]
            changes.append({
                "rule": f"core_person_{wrong.lower()}_to_{target.lower()}",
                "from": surface,
                "to": target_surface,
                "count": 1,
                "start": start,
                "end": end,
                "entity": target,
                "context": _br_excerpt(text[max(0, start - 50):min(len(text), end + 70)]),
            })
    if not changes:
        return text, [], blocking
    repaired = _br_apply_span_changes(text, changes)
    repaired_keys = [item[2] for item in _br_scan_mentions(repaired, body_only=True)]
    if repaired_keys != literary_keys:
        return text, [], blocking + [{
            "kind": "identity_swap_verification_failed",
            "severity": "high",
            "detail": "sequence after repair does not match literary draft",
        }]
    return repaired, changes, blocking


def _br_remaining_blocking(
    text: str, raw: str, literary: str, source: str,
) -> List[Dict[str, Any]]:
    blocking: List[Dict[str, Any]] = []
    source_counts = _br_source_counts(source)
    final_counts = _br_key_counts(text, body_only=True)
    if (source_counts.get("GOM", 0) == 0
            and final_counts.get("GOM", 0) > 0
            and source_counts.get("FEYD", 0) > 0):
        blocking.append({
            "kind": "feyd_gom_unresolved",
            "severity": "high",
            "detail": ("source 以 Feyd 为主且无 Gom Jabbar，final 仍有完整"
                       "戈姆·贾巴尔人名，需人工/证据复核"),
        })
    return blocking


def _br_plan_text(
    chunk_id: str, final: str, literary: str, raw: str, source: str,
) -> Dict[str, Any]:
    working = final
    changes: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []

    working, note_changes = _br_plan_note_repairs(working)
    changes.extend(note_changes)

    working, footnote_changes = _br_plan_malformed_footnotes(working)
    changes.extend(footnote_changes)

    working, gloss_changes = _br_plan_source_gloss_spelling(
        working, source=source)
    changes.extend(gloss_changes)

    working, omission_changes, omission_blocking = _br_plan_omissions(
        working, raw=raw, source=source, literary=literary)
    changes.extend(omission_changes)
    blocking.extend(omission_blocking)

    working, alias_changes = _br_plan_canonical_aliases(working, source=source)
    changes.extend(alias_changes)

    working, appositive_changes = _br_plan_appositive(
        working, raw=raw, literary=literary, source=source)
    changes.extend(appositive_changes)

    working, gom_changes, gom_blocking = _br_plan_gom_to_feyd(
        working, raw=raw, source=source)
    changes.extend(gom_changes)
    blocking.extend(gom_blocking)

    working, sentence_changes, sentence_blocking = _br_plan_sentence_core_swaps(
        working, literary=literary, source=source)
    changes.extend(sentence_changes)
    blocking.extend(sentence_blocking)

    working, core_changes, core_blocking = _br_plan_core_swaps(
        working, literary=literary, raw=raw, source=source)
    changes.extend(core_changes)
    blocking.extend(core_blocking)

    blocking.extend(_br_remaining_blocking(
        working, raw=raw, literary=literary, source=source))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for issue in blocking:
        marker = (issue.get("kind"), issue.get("detail"))
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(issue)
    return {
        "chunk_id": chunk_id,
        "before_text": final,
        "repaired_text": working,
        "changes": changes,
        "blocking": deduped,
        "before_sha256": _br_sha256(final),
        "repaired_sha256": _br_sha256(working),
    }


def _br_scan_all(
    root: Path, out_root: Path, chapter: Optional[str] = None,
) -> Dict[str, Any]:
    sources = _br_load_sources(root)
    pattern = f"{chapter}/*_final.json" if chapter else "ch*/*_final.json"
    rows: List[Dict[str, Any]] = []
    blocking_issues: List[Dict[str, Any]] = []
    current_hashes: Dict[str, str] = {}
    missing_sources: List[str] = []
    malformed: List[str] = []

    for final_path in sorted(out_root.glob(pattern)):
        chunk_id = final_path.stem.rsplit("_final", 1)[0]
        data = _br_read_json(final_path)
        if data is None or not isinstance(data.get("text"), str):
            malformed.append(str(final_path))
            continue
        final_text = data["text"]
        current_hashes[chunk_id] = _br_sha256(final_text)
        source = sources.get(chunk_id, "")
        if not source:
            missing_sources.append(chunk_id)
            continue
        literary_data = _br_read_json(final_path.with_name(f"{chunk_id}_literary.json"))
        raw_data = _br_read_json(final_path.with_name(f"{chunk_id}_raw.json"))
        literary = (literary_data or {}).get("text", "")
        raw = (raw_data or {}).get("text", "")
        if not isinstance(literary, str):
            literary = ""
        if not isinstance(raw, str):
            raw = ""
        plan = _br_plan_text(chunk_id, final_text, literary, raw, source)
        if plan["changes"] or plan["blocking"]:
            row = {
                "chunk_id": chunk_id,
                "chapter": chunk_id.split("_", 1)[0],
                "path": str(final_path.relative_to(root)),
                "before_sha256": plan["before_sha256"],
                "repaired_sha256": plan["repaired_sha256"],
                "changes": plan["changes"],
                "blocking": plan["blocking"],
                "repaired_text": plan["repaired_text"],
            }
            rows.append(row)
            for issue in plan["blocking"]:
                blocking_issues.append({
                    "chunk_id": chunk_id,
                    **issue,
                })

    return {
        "scanned": len(current_hashes),
        "rows": rows,
        "blocking_issues": blocking_issues,
        "missing_sources": missing_sources,
        "malformed_finals": malformed,
        "current_hashes": current_hashes,
    }


def _br_public_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    public: List[Dict[str, Any]] = []
    for row in rows:
        item = {key: value for key, value in row.items() if key != "repaired_text"}
        public.append(item)
    return public


def _br_render_markdown(payload: Dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    lines = [
        "# 全书终检修复报告 (book_repair)",
        "",
        f"- 模式: {payload.get('mode')}",
        f"- 扫描 final: {summary.get('scanned', 0)} 个",
        f"- 有修复计划/已修复 chunk: {summary.get('changed_chunks', 0)} 个",
        f"- 替换项: {summary.get('changes', 0)} 个",
        f"- 待修 chunk: {summary.get('pending_chunks', 0)} 个",
        f"- 未决高/严重风险: {summary.get('blocking_issues', 0)} 个",
        f"- 缺少 DB source 证据: {summary.get('missing_sources', 0)} 个",
        f"- 无法解析 final: {summary.get('malformed_finals', 0)} 个",
        f"- 修复凭证: {'当前' if summary.get('manifest_current') else '缺失/陈旧'}",
        "",
    ]
    by_rule = summary.get("by_rule") or {}
    if by_rule:
        lines.append("## 规则统计")
        lines.append("")
        for rule, count in sorted(by_rule.items()):
            lines.append(f"- `{rule}`: {count}")
        lines.append("")

    changes = payload.get("changes") or []
    if changes:
        lines.append("## 修复明细")
        lines.append("")
        for row in changes[:120]:
            lines.append(f"### `{row['chunk_id']}`")
            for change in row.get("changes") or []:
                lines.append(
                    f"- `{change.get('rule')}`: `{change.get('from')}` → "
                    f"`{change.get('to')}`")
                if change.get("context"):
                    lines.append(f"  - 语境: {change['context']}")
                if change.get("evidence"):
                    evidence = change["evidence"]
                    rendered = ", ".join(
                        f"{key}={value}" for key, value in evidence.items()
                        if value is not None)
                    lines.append(f"  - 证据: {rendered}")
            if row.get("backup"):
                lines.append(f"- 备份: `{row['backup']}`")
            lines.append("")
        if len(changes) > 120:
            lines.append(f"（另有 {len(changes) - 120} 个 chunk，见 JSON）")
            lines.append("")

    blocking = payload.get("blocking_issues") or []
    if blocking:
        lines.append("## 未决风险（阻断）")
        lines.append("")
        for issue in blocking[:120]:
            lines.append(
                f"- `{issue.get('chunk_id')}` [{issue.get('severity')}] "
                f"`{issue.get('kind')}`: {issue.get('detail')}")
        if len(blocking) > 120:
            lines.append(f"（另有 {len(blocking) - 120} 项，见 JSON）")
        lines.append("")
    elif not changes:
        lines.append("（未发现可修或未决项目）")
        lines.append("")

    missing_sources = payload.get("missing_sources") or []
    malformed_finals = payload.get("malformed_finals") or []
    if missing_sources or malformed_finals:
        lines.append("## 证据缺口（阻断）")
        lines.append("")
        if missing_sources:
            lines.append("- 缺少 DB source: " + ", ".join(
                f"`{chunk_id}`" for chunk_id in missing_sources[:80]))
        if malformed_finals:
            lines.append("- 无法解析 final: " + ", ".join(
                f"`{path}`" for path in malformed_finals[:80]))
        lines.append("")

    lines += ["---", "*由 book-repair 子命令自动生成；修复凭证以 current_hashes 为准*", ""]
    return "\n".join(lines)


def apply_book_repairs(
    project_root: Optional[Any] = None,
    output_dir: Optional[Any] = None,
    apply: bool = False,
    chapter: Optional[str] = None,
    json_out: Optional[Any] = None,
    report: Optional[Any] = None,
) -> Dict[str, Any]:
    """Plan or apply deterministic book repairs; always write a manifest report."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    planned = _br_scan_all(root, out_root, chapter=chapter)
    public_rows = _br_public_rows(planned["rows"])
    changed_chunks = sum(1 for row in public_rows if row.get("changes"))
    pending_before = changed_chunks
    applied_chunks = 0
    backup_root: Optional[Path] = None
    now = datetime.datetime.now().astimezone()

    if apply and changed_chunks:
        backup_root = out_root / "repair_backups" / (
            "book_repair_" + now.strftime("%Y%m%d_%H%M%S"))
        for row in public_rows:
            if not row.get("changes"):
                continue
            source_row = next(
                item for item in planned["rows"] if item["chunk_id"] == row["chunk_id"])
            final_path = root / row["path"]
            try:
                original_bytes = final_path.read_bytes()
            except OSError as exc:
                row["status"] = f"read_error: {exc}"
                continue
            data = _br_read_json(final_path)
            if data is None or not isinstance(data.get("text"), str):
                row["status"] = "json_error"
                continue
            if _br_sha256(data["text"]) != row["before_sha256"]:
                row["status"] = "stale_since_plan"
                continue
            backup_path = backup_root / row["chapter"] / final_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path.write_bytes(original_bytes)
            data["text"] = source_row["repaired_text"]
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata["book_repair"] = {
                "schema_version": BOOK_REPAIR_SCHEMA_VERSION,
                "applied_at": now.isoformat(),
                "before_sha256": row["before_sha256"],
                "after_sha256": row["repaired_sha256"],
                "rules": sorted({c.get("rule", "") for c in row.get("changes") or []}),
                "backup": str(backup_path.relative_to(root)),
            }
            data["metadata"] = metadata
            _br_atomic_write_json(final_path, data)
            row["status"] = "applied"
            row["backup"] = str(backup_path.relative_to(root))
            applied_chunks += 1

    verified = _br_scan_all(root, out_root, chapter=chapter)
    pending_chunks = sum(1 for row in verified["rows"] if row.get("changes"))
    blocking_issues = verified["blocking_issues"]
    by_rule: Dict[str, int] = {}
    for row in public_rows:
        for change in row.get("changes") or []:
            rule = str(change.get("rule") or "unknown")
            by_rule[rule] = by_rule.get(rule, 0) + 1
    mode = "apply" if apply else "dry-run"
    summary = {
        "scanned": verified["scanned"],
        "changed_chunks": changed_chunks,
        "changes": sum(len(row.get("changes") or []) for row in public_rows),
        "pending_chunks": pending_chunks,
        "blocking_issues": len(blocking_issues),
        "applied_chunks": applied_chunks,
        "by_rule": by_rule,
        "missing_sources": len(verified["missing_sources"]),
        "malformed_finals": len(verified["malformed_finals"]),
        "manifest_current": True,
    }
    missing_sources = verified["missing_sources"]
    malformed_finals = verified["malformed_finals"]
    ok = (not blocking_issues and not missing_sources and not malformed_finals
          and (not apply or pending_chunks == 0))
    payload = {
        "schema_version": BOOK_REPAIR_SCHEMA_VERSION,
        "generated_at": now.isoformat(),
        "mode": mode,
        "ok": ok,
        "summary": summary,
        "current_hashes": verified["current_hashes"],
        "changes": public_rows,
        "pending_chunks": [
            row["chunk_id"] for row in verified["rows"] if row.get("changes")
        ],
        "blocking_issues": blocking_issues,
        "missing_sources": missing_sources,
        "malformed_finals": malformed_finals,
    }
    json_path = Path(json_out) if json_out else out_root / BOOK_REPAIR_REPORT_NAME
    report_path = (Path(report) if report
                   else json_path.with_suffix(".md"))
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _br_atomic_write_json(json_path, payload)
        report_path.write_text(_br_render_markdown(payload), encoding="utf-8")
    except OSError as exc:
        return {
            "ok": False,
            "error": f"report_write_error: {exc}",
            "report_path": str(json_path),
            "summary": summary,
            "changes": public_rows,
            "blocking_issues": blocking_issues,
            "missing_sources": missing_sources,
            "malformed_finals": malformed_finals,
        }
    print(
        f"🩺 book-repair({mode}): scanned={summary['scanned']} "
        f"changed_chunks={changed_chunks} changes={summary['changes']} "
        f"pending={pending_chunks} blocking={len(blocking_issues)}")
    print(f"   报告: {json_path}")
    return {
        "ok": ok,
        "error": None,
        "report_path": str(json_path),
        "markdown_path": str(report_path),
        "summary": summary,
        "changes": public_rows,
        "blocking_issues": blocking_issues,
        "pending_chunks": payload["pending_chunks"],
        "current_hashes": verified["current_hashes"],
        "missing_sources": missing_sources,
        "malformed_finals": malformed_finals,
        "manifest_current": True,
    }


def verify_book_repairs(
    project_root: Optional[Any] = None,
    output_dir: Optional[Any] = None,
    chapter: Optional[str] = None,
    manifest_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Re-plan current finals and verify they still match the repair manifest."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    planned = _br_scan_all(root, out_root, chapter=chapter)
    pending = [row["chunk_id"] for row in planned["rows"] if row.get("changes")]
    blocking = planned["blocking_issues"]
    manifest_file = (Path(manifest_path) if manifest_path
                     else out_root / BOOK_REPAIR_REPORT_NAME)
    manifest = _br_read_json(manifest_file) if manifest_file.exists() else None
    stale: List[Dict[str, Any]] = []
    if manifest is not None:
        expected = manifest.get("current_hashes")
        if not isinstance(expected, dict):
            stale.append({"kind": "manifest_schema", "detail": "current_hashes missing"})
        else:
            actual = planned["current_hashes"]
            for chunk_id in sorted(set(expected) | set(actual)):
                if expected.get(chunk_id) != actual.get(chunk_id):
                    stale.append({
                        "kind": "manifest_stale",
                        "chunk_id": chunk_id,
                        "expected": expected.get(chunk_id),
                        "actual": actual.get(chunk_id),
                    })
    missing_sources = planned["missing_sources"]
    malformed_finals = planned["malformed_finals"]
    ok = (not pending and not blocking and not stale
          and not missing_sources and not malformed_finals)
    return {
        "ok": ok,
        "mode": "verify",
        "manifest_path": str(manifest_file),
        "manifest_present": manifest is not None,
        "manifest_current": bool(manifest is not None and not stale),
        "stale_manifest": stale,
        "scanned": planned["scanned"],
        "pending_chunks": pending,
        "blocking_issues": blocking,
        "current_hashes": planned["current_hashes"],
        "missing_sources": missing_sources,
        "malformed_finals": malformed_finals,
    }
