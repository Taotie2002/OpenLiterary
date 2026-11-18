"""Round 158 Phase 1.5: read-only reconciliation of pending backtracks.

``pending_backtrack`` is intentionally a one-row-per-chunk outbox.  Its row is
the *latest* deferred decision for that chunk, not a history.  A global
reconciler therefore has to be conservative:

* never delete a row before its state transition (or its proof of satisfaction)
  succeeds;
* never treat a substring hit alone as proof when the Decision DB and entity
  registry disagree about the canonical form;
* leave non-terminal chunks covered by their upcoming run instead of clearing
  evidence behind a running pipeline.

The mutating path is explicit and conservative: a row is deleted only after its
exact outbox snapshot matches and the required state transition (if any) has
already succeeded.  Blocked and orphan evidence is never cleared automatically.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.artifact_freshness import chapter_hash_map
from src.core.decision_engine import (
    _NAME_SEP_CHARS,
    canonical_is_title_only,
    canonical_tokens,
    classify_entity_source_name,
    source_name_proper_tokens,
)


TERMINAL_STATES = {"COMPLETED", "PERMANENTLY_FAILED"}
NONTERMINAL_STATES = {
    "PENDING", "EXTRACTING_TERMS", "TRANSLATING_RAW", "REWRITING_LITERARY",
    "AUDITING", "JUDGING", "DIRTY", "FAILED",
}

_FOOTNOTE_DEF_PATTERNS = (
    re.compile(r"(?m)^\[\^[^\]]+\]:"),
    re.compile(r"(?ms)^[ \t]*---[ \t]*\r?\n(?:[ \t]*\r?\n)*[ \t]*\[\d+\][ \t]+"),
    re.compile(r"(?m)^[ \t]*\[\d+\][ \t]+[^\n]"),
)

_SOURCE_GLOSS_RE = re.compile(
    r"(?P<cjk>[\u4e00-\u9fff·・－\-]{2,40})\s*[（(]\s*"
    r"(?P<source>[A-Za-z][A-Za-z0-9'’\- .]{1,100})\s*[）)]"
)
_ARTICLE_ALIAS_RE = re.compile(
    r"^(?:the|a|an)\s+(?P<rest>\S.*)$", re.IGNORECASE)


def _article_shadow_authority_id(
    decision: Dict[str, Any], decisions: Dict[int, dict]
) -> int:
    """Return an exact-key Decision shadowed by this article-led source."""
    match = _ARTICLE_ALIAS_RE.match(
        str(decision.get("source_key") or "").strip())
    if not match:
        return 0
    remainder = match.group("rest")
    decision_id = int(decision.get("id") or 0)
    for other_id, other in decisions.items():
        if int(other_id) == decision_id:
            continue
        if str(other.get("source_key") or "") == remainder:
            return int(other_id)
    return 0


def body_before_footnotes(text: Any) -> str:
    """Return prose before markdown or numbered translator-note definitions.

    Both ``[^1]: ...`` and the book's current ``---\\n\\n[1] ...`` form are
    note definitions, not translated prose.  A canonical term that appears only
    in an explanatory note must not satisfy a pending Decision or authorise a
    corpus-derived surface form.
    """
    raw = str(text or "")
    cuts = []
    for pattern in _FOOTNOTE_DEF_PATTERNS:
        match = pattern.search(raw)
        if match:
            cuts.append(match.start())
    return raw[:min(cuts)] if cuts else raw


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _name_key(text: Any) -> str:
    """Comparison key for names: width, quote, case, whitespace and separators.

    It keeps phrase boundaries useful (``贝尼杰赛里特`` contains no separator)
    while making the many legal Dune spellings ``·`` / ``-`` / ``－`` equal for
    an audit.  Inline ``[^n]`` references are markup, not part of a Chinese
    name: ``蒸馏服[^1]的集水袋`` must match ``蒸馏服的集水袋``.  This helper is
    not a translation policy and never rewrites text.
    """
    raw = re.sub(r"\[\^[^\]]+\]", "", str(text or ""))
    raw = unicodedata.normalize("NFKC", raw).casefold()
    raw = (raw.replace("’", "'").replace("‘", "'")
              .replace("“", '"').replace("”", '"'))
    raw = re.sub(r"[\s" + re.escape(_NAME_SEP_CHARS) + r"]+", "", raw)
    return raw.strip()


def _phrase_key(text: Any) -> str:
    raw = unicodedata.normalize("NFKC", str(text or "")).casefold()
    raw = raw.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", raw).strip()


def _contains_name(text: str, name: str) -> bool:
    key = _name_key(name)
    return bool(key and key in _name_key(text))


def _contains_phrase(text: str, phrase: str) -> bool:
    key = _phrase_key(phrase)
    return bool(key and key in _phrase_key(text))


def _source_phrase_present(text: str, phrase: str) -> bool:
    """Treat ASCII source keys as tokens, not unconstrained substrings."""
    if _source_tokens(phrase):
        return bool(_source_phrase_spans(text, phrase))
    return _contains_phrase(text, phrase)


_TARGET_ANNOTATION_RE = re.compile(r"[（(][^（()）]*[）)]")
_TARGET_INSTRUCTION_RE = re.compile(
    r"(?:首次写作|此后统一|统一为|统一)[^“]*“([^”]+)”")
_TARGET_SURROUNDING_QUOTES = "《》“”\"'‘’「」『』"


def _target_core(value: Any) -> str:
    """Return the executable part of a Decision target.

    Decisions often record a reader annotation alongside the term itself, for
    example ``音言（The Voice）``.  The translated prose is expected to contain
    ``音言``, never the annotation.  Comparing the raw decision string would
    turn a correct final into an endless outstanding row.  Parenthetical
    annotations are therefore ignored for presence/form matching only; the
    Decision DB remains unchanged.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    # A few high-value Decisions store an executable form inside editorial
    # prose (Kwisatz Haderach: "首次写作...，此后统一‘短距者’").  Treating the
    # whole sentence as a literal target creates permanent false outstanding
    # rows.  The last explicit instruction is the post-first-occurrence form.
    instructed = _TARGET_INSTRUCTION_RE.findall(raw)
    if instructed:
        raw = instructed[-1].strip()
    raw = raw.strip(_TARGET_SURROUNDING_QUOTES).strip()
    core = _TARGET_ANNOTATION_RE.sub("", raw).strip()
    return core or raw


def _name_compatible(a: str, b: str) -> bool:
    """True when one name form contains the other after separator folding."""
    ka, kb = _name_key(a), _name_key(b)
    if not ka or not kb:
        return False
    if ka == kb or ka in kb or kb in ka:
        return True
    # A decision may deliberately use a short title form ("雷托公爵") while
    # the canonical registry name is fuller ("雷托·厄崔迪公爵").  The short
    # form's characters must appear in order in the fuller target; a one-char
    # transliteration conflict ("杰赛" vs "杰瑟") still fails this test.
    pos = 0
    for ch in kb:
        pos = ka.find(ch, pos)
        if pos < 0:
            return False
        pos += 1
    return True


def _connect_ro(path: Path) -> sqlite3.Connection:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    con = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_term_contracts(
    root: Path,
    path: Optional[Path] = None,
) -> Tuple[Dict[Tuple[int, str, str], dict], str]:
    """Load the optional Round 158 term/form artifact.

    A malformed artifact is reported but does not change pre-contract pending
    classification.  This keeps a bad evidence file fail-open for diagnosis and
    fail-closed for delivery: rows still appear outstanding/blocked and gate ⑬
    remains red until the artifact is fixed.
    """
    cpath = Path(path) if path else Path(root) / "input" / "term_form_contracts.json"
    if not cpath.exists():
        return {}, ""
    payload = _read_json(cpath)
    if not payload:
        return {}, "term_contract_artifact_unreadable"
    if int(payload.get("schema_version") or 0) != 1:
        return {}, "term_contract_schema_incompatible"
    out: Dict[Tuple[int, str, str], dict] = {}
    for contract in payload.get("contracts") or []:
        if not isinstance(contract, dict):
            continue
        try:
            decision_id = int(contract.get("decision_id") or 0)
        except (TypeError, ValueError):
            continue
        source_key = str(contract.get("source_key") or "")
        translation = str(contract.get("decision_translation") or "")
        forms = []
        for item in contract.get("forms") or []:
            if isinstance(item, dict):
                form = str(item.get("form") or "").strip()
            else:
                form = str(item or "").strip()
            if form:
                forms.append(form)
        match_mode = str(contract.get("match_mode") or "forms")
        reference = str(contract.get("reference") or "")
        if not decision_id or not source_key or not translation:
            continue
        if match_mode == "fuzzy_text" and not reference:
            continue
        if match_mode != "fuzzy_text" and not forms:
            continue
        key = (decision_id, _phrase_key(source_key), _phrase_key(translation))
        out[key] = dict(contract, forms=forms)
    return out, ""


def _cjk_only(text: Any) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", str(text or "")))


def term_text_similarity(text: str, reference: str) -> float:
    """Best similarity between a reference passage and a quoted final segment."""
    ref = _cjk_only(reference)
    if not ref:
        return 0.0
    segments = re.findall(r"[“\"]([^”\"]{10,})[”\"]", str(text or ""))
    if not segments:
        segments = [str(text or "")]
    best = 0.0
    for segment in segments:
        probe = _cjk_only(segment)
        if not probe:
            continue
        best = max(best, difflib.SequenceMatcher(
            None, ref, probe, autojunk=False).ratio())
    return best


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return bool(row)


def _registry_source_targets(
    con: sqlite3.Connection,
) -> Tuple[Dict[str, set], Dict[str, str], Dict[str, set], Dict[str, set]]:
    """Return source→canonical, source→trusted canonical, target→sources,
    and source→trusted canonical set."""
    by_source: Dict[str, set] = {}
    state: Dict[str, str] = {}
    sources_by_target: Dict[str, set] = {}
    trusted_by_source: Dict[str, set] = {}
    if not _table_exists(con, "entity_registry"):
        return by_source, state, sources_by_target, trusted_by_source
    cols = {r[1] for r in con.execute("PRAGMA table_info(entity_registry)")}
    pending_col = "pending_review" if "pending_review" in cols else "0"
    for row in con.execute(
        "SELECT canonical_name, source_names, confidence, "
        f"{pending_col} AS pending_review FROM entity_registry"
    ):
        if str(row["confidence"] or "") != "high":
            continue
        canonical = str(row["canonical_name"] or "").strip()
        if not canonical:
            continue
        canonical_key = _name_key(canonical)
        if canonical_key:
            sources_by_target.setdefault(canonical_key, set())
        try:
            sources = json.loads(row["source_names"] or "[]")
        except (TypeError, json.JSONDecodeError):
            sources = []
        if not isinstance(sources, list):
            continue
        for source in sources:
            key = _phrase_key(source)
            if not key:
                continue
            by_source.setdefault(key, set()).add(canonical)
            if canonical_key:
                sources_by_target.setdefault(canonical_key, set()).add(key)
            if not int(row["pending_review"] or 0):
                state.setdefault(key, canonical)
                trusted_by_source.setdefault(key, set()).add(canonical)
    return by_source, state, sources_by_target, trusted_by_source


def _registry_form_owners(con: sqlite3.Connection) -> Dict[str, List[dict]]:
    """Index every registry form (source/variant/quarantined) to its owner.

    A Decision key can be quarantined as an ambiguous alias while one of its
    translations is still a non-canonical spelling of a different active name
    (the ``Leto -> 莱托`` vs ``雷托·厄崔迪公爵`` case).  Full-book pending
    reconciliation needs the quarantined form as evidence *against* enforcing
    that Decision target, not as proof that the entity is absent.
    """
    out: Dict[str, List[dict]] = {}
    if not _table_exists(con, "entity_registry"):
        return out
    cols = {r[1] for r in con.execute("PRAGMA table_info(entity_registry)")}
    pending_col = "pending_review" if "pending_review" in cols else "0"
    variant_expr = "variant_names" if "variant_names" in cols else "NULL"
    quarantined_expr = ("quarantined_names"
                        if "quarantined_names" in cols else "NULL")
    for row in con.execute(
        "SELECT canonical_name, source_names, "
        f"{variant_expr} AS variant_names, "
        f"{quarantined_expr} AS quarantined_names, confidence, "
        f"{pending_col} AS pending_review FROM entity_registry"
    ):
        if str(row["confidence"] or "") != "high":
            continue
        canonical = str(row["canonical_name"] or "").strip()
        if not canonical:
            continue
        parsed: Dict[str, List[str]] = {}
        for field in ("source_names", "variant_names", "quarantined_names"):
            try:
                values = json.loads(row[field] or "[]")
            except (TypeError, json.JSONDecodeError):
                values = []
            parsed[field] = [str(v).strip() for v in values
                             if isinstance(v, str) and str(v).strip()]
        owner = {
            "canonical": canonical,
            "variants": parsed["variant_names"],
            "quarantined": parsed["quarantined_names"],
            "pending_review": int(row["pending_review"] or 0),
        }
        for field in ("source_names", "variant_names", "quarantined_names"):
            for form in parsed[field]:
                key = _phrase_key(form)
                if not key:
                    continue
                bucket = out.setdefault(key, [])
                if not any(item["canonical"] == canonical for item in bucket):
                    bucket.append(owner)
    return out


def _profile_source_keys(root: Path) -> set:
    """Return source keys explicitly covered by the profile's term hints."""
    profile = _read_json(Path(root) / "input" / "book_profile.json") or {}
    out = set()
    for hint in profile.get("translation_hints") or []:
        if not isinstance(hint, str) or "=" not in hint:
            continue
        source = hint.split("=", 1)[0].strip()
        key = _phrase_key(source)
        if key:
            out.add(key)
    return out


def _completed_gloss_keys(tasks: Dict[str, dict], out_dir: Path) -> set:
    """Collect exact ``中文（English source）`` glosses from completed finals.

    This is a read-only term-authority signal.  It deliberately uses only
    source-positive completed chunks, so an outstanding chunk cannot grant
    itself an exemption by writing a gloss in its own final.
    """
    out = set()
    for chunk_id, task in (tasks or {}).items():
        if str(task.get("state") or "") != "COMPLETED":
            continue
        final_path = (Path(out_dir) / str(task.get("chapter_id") or chunk_id[:4])
                      / f"{chunk_id}_final.json")
        final = _read_json(final_path) or {}
        text = body_before_footnotes(final.get("text") or "")
        if not text:
            continue
        for match in _SOURCE_GLOSS_RE.finditer(text):
            key = _phrase_key(match.group("source"))
            if key:
                out.add(key)
    return out


def _canonical_present(text: str, canonical: str) -> bool:
    if not text or not canonical:
        return False
    if _contains_name(text, canonical):
        return True
    return any(len(token) >= 2 and _contains_name(text, token)
               for token in canonical_tokens(canonical))


def _decision_conflict(
    source_key: str,
    translation: str,
    source_text: str,
    final_text: str,
    registry_targets: Dict[str, str],
    trusted_by_source: Dict[str, set],
    form_owners: Dict[str, List[dict]],
) -> str:
    key = _phrase_key(source_key)
    if not key or not _source_phrase_present(source_text, source_key):
        return ""
    trusted_targets = trusted_by_source.get(key) or set()
    if len({_name_key(c) for c in trusted_targets}) > 1:
        return "cross_person_alias_conflict"
    registry_target = registry_targets.get(key) or ""
    if registry_target and not _name_compatible(registry_target, translation):
        return "decision_registry_target_conflict"
    for owner in form_owners.get(key) or []:
        canonical = str(owner.get("canonical") or "")
        if not canonical or _name_compatible(canonical, translation):
            continue
        if any(_name_compatible(variant, translation)
               for variant in owner.get("variants") or []):
            if _canonical_present(final_text, canonical):
                return "decision_target_noncanonical_variant"
    return ""


def _proper_core_forms(
    context: Dict[str, Any],
    source_key: str,
) -> List[str]:
    """Return registry-owned translations for the proper-name part of a phrase.

    ``Bene Gesserit witch`` and ``Harkonnen agents`` are descriptive phrases,
    not executable jargon.  Their proper-name cores still have to be present,
    but the common-noun tail may legitimately vary with Chinese syntax.
    """
    proper_tokens = set(source_name_proper_tokens(source_key))
    if not proper_tokens:
        return []
    by_source = context.get("registry_by_source") or {}
    trusted = context.get("registry_trusted_by_source") or {}
    candidates = []
    for key, forms in by_source.items():
        key_tokens = set(_source_tokens(key))
        if not key_tokens or not key_tokens <= proper_tokens:
            continue
        if not _source_phrase_spans(source_key, key):
            continue
        trusted_forms = trusted.get(key) or set()
        use_forms = trusted_forms if trusted_forms else set(forms or ())
        if len({_name_key(form) for form in use_forms if _name_key(form)}) != 1:
            continue
        candidates.append((len(key_tokens), key, tuple(sorted(use_forms))))
    if not candidates:
        return []
    max_tokens = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == max_tokens]
    if len(best) != 1:
        return []
    return list(best[0][2])


def _has_hard_term_authority(
    context: Dict[str, Any],
    source_key: str,
    decision: Dict[str, Any],
) -> bool:
    """True when a Decision is backed by a hard term/name authority."""
    key = _phrase_key(source_key)
    if not key:
        return True
    if int(decision.get("locked") or 0):
        return True
    if key in (context.get("profile_sources") or set()):
        return True
    if key in (context.get("glossed_sources") or set()):
        return True
    if (context.get("registry_by_source") or {}).get(key):
        return True
    return False


def _contextual_reference_kind(source_key: str, target: str) -> str:
    """Classify a narrow, non-jargon Reference phrase.

    This is intentionally lexical rather than model-based.  Technical nouns
    such as ``suspensors`` / ``shield`` must never be relaxed just because
    their source has no explicit gloss.  Eligible shapes are:

    * a proper-name prefix plus a contextual common-noun tail;
    * a grammatical participle/adjective (``conditioned``);
    * a coordinated descriptive phrase (``green and black banner``);
    * a slash-separated editorial target (``妾室/情妇``).
    """
    raw = str(source_key or "").strip()
    if not raw:
        return ""
    proper_tokens = set(source_name_proper_tokens(raw))
    tokens = _source_tokens(raw)
    if proper_tokens and any(token not in proper_tokens for token in tokens):
        return "proper_core"
    if not proper_tokens and re.fullmatch(r"[A-Za-z]+(?:ed|ing)", raw):
        return "common"
    if not proper_tokens and re.search(r"\b(?:and|or)\b", raw, re.I):
        return "common"
    if "/" in str(target or ""):
        return "common"
    return ""


def _soft_reference_strategy(source_key: str, decision: Dict[str, Any]) -> str:
    """Return the generic-reference strategy for a lower-case prose phrase.

    ``shadows -> 暗影`` and ``cells -> 小组`` are useful lexical observations,
    not mandatory surface forms.  The same applies to a culturally equivalent
    metaphor such as ``clods -> 泥腿子`` when another context naturally uses
    ``土疙瘩``.  The Decision reason records the extractor's strategy; only
    LITERAL_TRANSLATION or CULTURAL_EQUIVALENT without retain/terminology
    tags, quoted text, or capitalised proper tokens is eligible.  Hard term
    authorities are checked by the caller and always win.
    """
    reason = str(decision.get("reason") or "")
    if "【策略】" not in reason:
        return ""
    strategy = reason.upper()
    if any(tag in reason.upper() for tag in
           ("RETAIN_AND_ANNOTATE", "TRANSLITERATION", "TERMINOLOGY")):
        return ""
    if "LITERAL_TRANSLATION" in strategy:
        kind = "literal"
    elif "CULTURAL_EQUIVALENT" in strategy:
        kind = "cultural"
    else:
        return ""
    raw = str(source_key or "").strip()
    if not raw or re.search(r"[“”\"‘’]", raw):
        return ""
    tokens = _EN_SOURCE_TOKEN_RE.findall(raw)
    if not tokens or any(token and token[0].isupper() for token in tokens):
        return ""
    return kind


def _soft_literal_reference_decision(source_key: str, decision: Dict[str, Any]) -> bool:
    """Backward-compatible predicate for literal generic references."""
    return _soft_reference_strategy(source_key, decision) == "literal"


_EN_SOURCE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_EN_SOURCE_JOINER = r"[\s'’‘\-‐‑‒–—]+"


def _source_tokens(text: Any) -> List[str]:
    """Tokenize an English source key for strict phrase containment."""
    return [token.casefold() for token in _EN_SOURCE_TOKEN_RE.findall(str(text or ""))]


def _source_phrase_spans(text: str, phrase: str) -> List[Tuple[int, int]]:
    """Locate an English source phrase without rewriting the source text.

    Decision source keys may use straight/curly apostrophes and different
    whitespace from the EPUB text.  Only ASCII alphanumeric tokens are matched;
    Chinese term targets are deliberately outside this helper's scope.
    """
    raw_tokens = _EN_SOURCE_TOKEN_RE.findall(str(phrase or ""))
    tokens = [token.casefold() for token in raw_tokens]
    if not tokens or not str(text or ""):
        return []
    body = _EN_SOURCE_JOINER.join(re.escape(token) for token in raw_tokens)
    # Article-led names such as ``the Voice`` must not consume the ordinary
    # lowercase phrase ``the voice``.  The capitalisation is part of the term
    # identity here; otherwise source-positive audits are inflated by generic
    # narration and every such chunk becomes a false pending/contract signal.
    case_sensitive = bool(
        len(raw_tokens) > 1
        and raw_tokens[0].casefold() in {"the", "a", "an"}
        and any(any(ch.isupper() for ch in token) for token in raw_tokens[1:])
    )
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])",
        0 if case_sensitive else re.IGNORECASE,
    )
    spans = []
    for match in pattern.finditer(str(text)):
        if case_sensitive and _is_title_head_span(
                str(text), match.end(), raw_tokens):
            continue
        spans.append(match.span())
    return spans


def _is_title_head_span(
    text: str,
    end: int,
    raw_tokens: Sequence[str],
) -> bool:
    """Return True when an article-led term heads a longer proper title.

    ``the Voice from the Outer World`` is a title, not the Bene Gesserit
    ``the Voice``.  A generic Decision must not create a pending row for the
    longer phrase merely because its first two words match.  The exception is
    deliberately limited to article-led keys followed by a lowercase
    preposition and a capitalised title token (optionally after an article).
    """
    if not raw_tokens or raw_tokens[0].casefold() not in {"the", "a", "an"}:
        return False
    tail = text[end:end + 120]
    return bool(re.match(
        r"\s+(?:of|from|for|in|on|at|by|with)\s+"
        r"(?:(?:the|a|an)\s+)?[A-Z][A-Za-z0-9'’-]*",
        tail,
    ))


def _strictly_more_specific_source(pending_source: str, candidate_source: str) -> bool:
    """True when a decision key extends the pending key by whole tokens.

    Case, articles and simple plural forms are *not* more specific: those are
    competing authorities for the same lexical key and need an explicit form
    contract.  This helper only accepts a genuine longer phrase such as
    ``Mapes`` -> ``Shadout Mapes`` or ``flagon`` -> ``water flagon``.
    """
    pending = _source_tokens(pending_source)
    candidate = _source_tokens(candidate_source)
    if not pending or len(candidate) <= len(pending):
        return False
    for start in range(len(candidate) - len(pending) + 1):
        if candidate[start:start + len(pending)] == pending:
            return True
    return False


def _specific_decision_coverage(
    *,
    pending_source: str,
    pending_decision_id: int,
    source_text: str,
    final_text: str,
    decisions: Dict[int, dict],
) -> Dict[str, Any]:
    """Prove that every generic source occurrence is covered by a specific one.

    A pending row stores only the latest deferred decision for a chunk.  A
    generic key such as ``Mapes`` or ``flagon`` can be a substring of a more
    specific decision (``Shadout Mapes`` / ``water flagon``) whose target is
    already present.  Clearing the generic row is safe only when:

    * every occurrence of the generic key is contained in a specific phrase;
    * the specific Decision targets are present in the final text; and
    * all competing targets covering the same occurrence are compatible.

    If any occurrence lacks a specific cover, the row remains outstanding.  A
    partial hit must never erase a whole-book invariant.
    """
    pending_spans = _source_phrase_spans(source_text, pending_source)
    if not pending_spans:
        return {"ok": False, "reason": "pending_source_span_missing"}

    per_span: List[List[Dict[str, Any]]] = [[] for _ in pending_spans]
    for decision_id, decision in (decisions or {}).items():
        if int(decision_id) == int(pending_decision_id):
            continue
        candidate_source = str(decision.get("source_key") or "")
        candidate_target = _target_core(decision.get("translation"))
        if not candidate_target:
            continue
        if not _strictly_more_specific_source(pending_source, candidate_source):
            continue
        candidate_spans = _source_phrase_spans(source_text, candidate_source)
        if not candidate_spans:
            continue
        if not _contains_name(final_text, candidate_target):
            continue
        evidence = {
            "decision_id": int(decision_id),
            "source_key": candidate_source,
            "translation": candidate_target,
        }
        for index, pending_span in enumerate(pending_spans):
            if any(start <= pending_span[0] and pending_span[1] <= end
                   for start, end in candidate_spans):
                per_span[index].append(evidence)

    if any(not candidates for candidates in per_span):
        return {"ok": False, "reason": "pending_source_not_fully_covered"}

    for candidates in per_span:
        targets = [str(item["translation"]) for item in candidates]
        for left in targets:
            for right in targets:
                if not _name_compatible(left, right):
                    return {"ok": False, "reason": "specific_target_conflict"}

    unique: Dict[int, Dict[str, Any]] = {}
    for candidates in per_span:
        for item in candidates:
            unique[int(item["decision_id"])] = item
    return {
        "ok": True,
        "reason": "specific_decision_covered",
        "covered_by_decisions": [unique[key] for key in sorted(unique)],
        "covered_source_occurrences": len(pending_spans),
    }


def _all_source_occurrences_have_more_specific_decision(
    *,
    pending_source: str,
    pending_decision_id: int,
    source_text: str,
    decisions: Dict[int, dict],
) -> bool:
    """True when every generic source span is inside a longer Decision key.

    This distinguishes a missing *specific* target (``Shield Wall``) from a
    genuinely missing generic target (``shield``).  The generic locked row must
    not masquerade as authority over the longer phrase and force a whole-chunk
    retranslation of the wrong term.
    """
    pending_spans = _source_phrase_spans(source_text, pending_source)
    if not pending_spans:
        return False
    candidate_spans: List[Tuple[int, int]] = []
    for decision_id, decision in (decisions or {}).items():
        if int(decision_id) == int(pending_decision_id):
            continue
        candidate_source = str(decision.get("source_key") or "")
        if not _strictly_more_specific_source(pending_source, candidate_source):
            continue
        candidate_spans.extend(_source_phrase_spans(
            source_text, candidate_source))
    if not candidate_spans:
        return False
    return all(
        any(start <= span[0] and span[1] <= end
            for start, end in candidate_spans)
        for span in pending_spans
    )


def load_pending_context(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    term_contracts_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load a coherent read-only snapshot for pending classification."""
    root = Path(project_root)
    decision_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    workflow_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    term_contracts, term_contract_error = _load_term_contracts(
        root, term_contracts_path)

    if not decision_path.exists():
        return {"available": False, "reason": "decision_db_missing",
                "pending": [], "pending_rows": 0, "tasks": {}, "decisions": {},
                "decisions_complete": False, "chapters": {}}
    if not workflow_path.exists():
        return {"available": False, "reason": "workflow_db_missing",
                "pending": [], "pending_rows": 0, "tasks": {}, "decisions": {},
                "decisions_complete": False, "chapters": {}}

    dcon = _connect_ro(decision_path)
    wcon = _connect_ro(workflow_path)
    try:
        if not _table_exists(dcon, "pending_backtrack"):
            pending: List[dict] = []
        else:
            pending = [dict(r) for r in dcon.execute(
                "SELECT chunk_id, decision_id, source_key, "
                "COALESCE(queue_rounds, 0) AS queue_rounds, "
                "created_at, updated_at FROM pending_backtrack "
                "ORDER BY chunk_id"
            )]
        # A minimal/legacy project may have no decision_db table at all, and old
        # fixtures declare only source_key/translation/reason.  Pending cleanup
        # has no rows to reconcile in that case, but name-only auditing still
        # needs decisions when the authoritative columns exist.  A schema gap is
        # blocking only when there is a real pending row that cannot be checked.
        decision_cols = {r[1] for r in dcon.execute("PRAGMA table_info(decision_db)")}
        required_cols = {"id", "source_key", "translation"}
        missing_cols = sorted(required_cols - decision_cols)
        schema_error = ""
        if missing_cols:
            decisions: Dict[int, dict] = {}
            # name-only classification still needs decisions even when the
            # outbox is empty.  Treat the schema gap as unavailable only when
            # there is an actual pending row to reconcile.
            if pending:
                schema_error = "decision_db_schema_incompatible:" + ",".join(missing_cols)
        else:
            locked_expr = ("COALESCE(locked, 0)" if "locked" in decision_cols else "0")
            updated_expr = ("updated_at" if "updated_at" in decision_cols else "NULL")
            reason_expr = ("COALESCE(reason, '')" if "reason" in decision_cols else "''")
            decisions = {
                int(r["id"]): dict(r)
                for r in dcon.execute(
                    "SELECT id, source_key, translation, "
                    f"{reason_expr} AS reason, "
                    f"{updated_expr} AS updated_at, "
                    f"{locked_expr} AS locked FROM decision_db"
                )
            }
        registry_by_source, registry_targets, sources_by_target, trusted_by_source = (
            _registry_source_targets(dcon))
        registry_form_owners = _registry_form_owners(dcon)
    finally:
        dcon.close()

    try:
        tasks = {
            str(r["chunk_id"]): dict(r)
            for r in wcon.execute(
                "SELECT chunk_id, chapter_id, text_content, state, last_error, "
                "COALESCE(backtrack_rounds, 0) AS backtrack_rounds, rescue_stage "
                "FROM chunk_tasks"
            )
        }
    finally:
        wcon.close()

    profile_sources = _profile_source_keys(root)
    glossed_sources = _completed_gloss_keys(tasks, out_dir)

    if schema_error:
        return {
            "available": False,
            "reason": schema_error,
            "pending": pending,
            "pending_rows": len(pending),
            "tasks": tasks,
            "decisions": {},
            "decisions_complete": False,
            "term_contracts": {},
            "term_contract_error": term_contract_error,
            "chapters": chapter_hash_map(root / "input"),
            "pending_table_present": True,
            "profile_sources": profile_sources,
            "glossed_sources": glossed_sources,
        }

    return {
        "available": True,
        "reason": "",
        "root": root,
        "decision_db": decision_path,
        "workflow_db": workflow_path,
        "output_dir": out_dir,
        "pending": pending,
        "pending_rows": len(pending),
        "tasks": tasks,
        "decisions": decisions,
        "decisions_complete": not bool(missing_cols),
        "term_contracts": term_contracts,
        "term_contract_error": term_contract_error,
        "registry_targets": registry_targets,
        "registry_by_source": registry_by_source,
        "registry_sources_by_target": sources_by_target,
        "registry_trusted_by_source": trusted_by_source,
        "registry_form_owners": registry_form_owners,
        "chapters": chapter_hash_map(root / "input"),
        "profile_sources": profile_sources,
        "glossed_sources": glossed_sources,
    }


def classify_pending_row(
    row: Dict[str, Any],
    context: Dict[str, Any],
    *,
    final_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Classify one outbox row as satisfied/covered/outstanding/orphan/blocked."""
    chunk_id = str(row.get("chunk_id") or "")
    decision_id = row.get("decision_id")
    override_map = final_overrides or {}
    task = (context.get("tasks") or {}).get(chunk_id)
    decision = (context.get("decisions") or {}).get(
        int(decision_id) if decision_id is not None else -1
    )
    base = {
        "chunk_id": chunk_id,
        "chapter": str((task or {}).get("chapter_id") or chunk_id[:4]),
        "decision_id": decision_id,
        "pending_source_key": str(row.get("source_key") or ""),
        "queue_rounds": int(row.get("queue_rounds") or 0),
        "updated_at": row.get("updated_at"),
        "state": str((task or {}).get("state") or "MISSING"),
        "status": "blocked",
        "reason": "",
        "decision_source_key": "",
        "decision_translation": "",
        "target_present": False,
        "source_active": False,
        "covered_by_decisions": [],
        "covered_source_occurrences": 0,
        "term_contract_id": "",
        "matched_term_form": "",
    }
    if task is None:
        base.update(status="orphan", reason="chunk_missing")
        return base
    if decision is None:
        base.update(status="blocked", reason="decision_missing")
        return base
    source_key = str(decision.get("source_key") or "")
    translation = str(decision.get("translation") or "")
    pending_source = str(row.get("source_key") or "")
    base.update(decision_source_key=source_key, decision_translation=translation)
    if pending_source and _phrase_key(pending_source) != _phrase_key(source_key):
        base.update(status="blocked", reason="pending_decision_key_mismatch")
        return base
    if not source_key.strip() or not translation.strip():
        base.update(status="blocked", reason="empty_decision")
        return base

    source_text = str(task.get("text_content") or "")
    final_path = (context.get("output_dir") or Path(".")) / base["chapter"] / f"{chunk_id}_final.json"
    final_text = override_map.get(chunk_id)
    if final_text is None:
        final = _read_json(final_path)
        final_text = str(final.get("text") or "") if final else ""
    final_prose = body_before_footnotes(final_text)
    translation_core = _target_core(translation)
    target_present = _contains_name(final_prose, translation_core)
    source_active = _source_phrase_present(source_text, source_key)
    shadow_authority_id = _article_shadow_authority_id(
        decision, context.get("decisions") or {})
    shadow_decision = ((context.get("decisions") or {}).get(
        int(shadow_authority_id)) if shadow_authority_id else {}) or {}
    shadow_target = _target_core(str(shadow_decision.get("translation") or ""))
    shadow_canonical_present = bool(
        shadow_target and _contains_name(final_prose, shadow_target))
    base.update(
        target_present=target_present,
        source_active=source_active,
        article_shadow_authority_id=shadow_authority_id,
        article_shadow_canonical_id=int(shadow_authority_id or 0),
        article_shadow_canonical_target=shadow_target,
        article_shadow_canonical_present=shadow_canonical_present,
    )

    # Round 158: title/address decisions are advisory because the same English
    # form legitimately takes different Chinese forms by grammar position
    # (Sire -> 陛下/大人/老爷).  They must not keep a global outbox row alive.
    # This is deliberately narrower than "any non-proper source": technical
    # terms such as ``shield`` remain hard consistency decisions.
    source_class = classify_entity_source_name(source_key)
    if source_class in {"title", "relation"} or (
            source_class != "proper" and canonical_is_title_only(translation)):
        base.update(
            status="satisfied",
            reason="contextual_decision_not_unified",
        )
        return base

    if not str(final_text).strip():
        state = base["state"]
        if state in NONTERMINAL_STATES:
            base.update(status="covered", reason="nonterminal_final_pending")
        else:
            base.update(status="blocked", reason="final_missing_or_unreadable")
        return base

    # An article-led Decision (``the Lady Jessica``) is a shadow of the exact
    # key (``Lady Jessica``).  Its own target may be a bad title variant.  Close
    # the row when the canonical target is already in prose; if the shadow
    # target is the only form present, keep one repair signal instead of
    # treating the wrong variant as satisfaction.
    if shadow_authority_id and source_active:
        if shadow_canonical_present:
            base.update(
                status="satisfied",
                reason="article_shadow_canonical_present",
            )
            return base
        state = base["state"]
        if state in NONTERMINAL_STATES:
            base.update(
                status="covered",
                reason="nonterminal_article_shadow_target_pending",
            )
        elif state == "COMPLETED":
            base.update(
                status="outstanding",
                reason="article_shadow_canonical_missing",
            )
        elif state == "PERMANENTLY_FAILED":
            base.update(status="blocked", reason="permanently_failed_task")
        else:
            base.update(status="blocked", reason=f"unknown_state:{state}")
        return base

    # Level-2 Reference rows are not all hard terminology.  A descriptive
    # lowercase phrase with no profile hint, registry source, explicit gloss,
    # or lock is context-dependent prose (``conditioned`` / ``concubine``).
    # Proper-name prefixes still need their registered Chinese core; common
    # noun tails such as ``witch`` / ``agents`` may follow Chinese syntax.
    contextual_kind = _contextual_reference_kind(source_key, translation)
    soft_reference = _soft_reference_strategy(source_key, decision)
    if (source_class == "description"
            and not _has_hard_term_authority(context, source_key, decision)
            and soft_reference):
        base.update(
            status="satisfied",
            reason=("soft_literal_reference_not_unified"
                    if soft_reference == "literal"
                    else "soft_cultural_reference_not_unified"),
        )
        return base
    if (contextual_kind and source_class == "description"
            and not _has_hard_term_authority(context, source_key, decision)):
        if contextual_kind == "common":
            base.update(
                status="satisfied",
                reason="contextual_reference_not_unified",
            )
            return base
        core_forms = _proper_core_forms(context, source_key)
        if core_forms and all(
                _contains_name(final_prose, form) for form in core_forms):
            base.update(
                status="satisfied",
                reason="contextual_reference_proper_core_present",
                matched_term_form="|".join(core_forms),
            )
            return base

    conflict = _decision_conflict(
        source_key, translation, source_text, final_prose,
        context.get("registry_targets") or {},
        context.get("registry_trusted_by_source") or {},
        context.get("registry_form_owners") or {},
    )
    if conflict:
        base.update(status="blocked", reason=conflict)
        return base
    contract = (context.get("term_contracts") or {}).get((
        int(decision_id or 0), _phrase_key(source_key), _phrase_key(translation)))
    if shadow_authority_id:
        contract = None
    # A locked Decision is an explicit book-wide authority.  A partial contract
    # may still be useful for normal pending rows, but it must never be enough
    # to close a locked target by itself: otherwise a partial pending snapshot
    # would silently downgrade the lock and hide missing prose.
    contract_can_close = (
        not int(decision.get("locked") or 0)
        or str(contract.get("status") or "") == "complete"
    ) if contract else False
    if source_active and contract and contract_can_close:
        if str(contract.get("match_mode") or "forms") == "fuzzy_text":
            similarity = term_text_similarity(
                final_prose, str(contract.get("reference") or ""))
            minimum = float(contract.get("min_similarity") or 0.8)
            if similarity >= minimum:
                base.update(
                    status="satisfied",
                    reason="term_quote_contract_present",
                    term_contract_id=str(contract.get("contract_id") or ""),
                    matched_term_form="fuzzy_text:%.4f" % similarity,
                )
                return base
        for form in contract.get("forms") or []:
            if _contains_name(final_prose, str(form)):
                base.update(
                    status="satisfied",
                    reason="term_form_contract_present",
                    term_contract_id=str(contract.get("contract_id") or ""),
                    matched_term_form=str(form),
                )
                return base
    specific: Optional[Dict[str, Any]] = None
    if int(decision.get("locked") or 0) and not target_present:
        specific = _specific_decision_coverage(
            pending_source=pending_source or source_key,
            pending_decision_id=int(row.get("decision_id") or 0),
            source_text=source_text,
            final_text=final_prose,
            decisions=context.get("decisions") or {},
        )
        if specific.get("ok"):
            base.update(
                status="satisfied",
                reason="specific_decision_covered",
                covered_by_decisions=specific.get("covered_by_decisions") or [],
                covered_source_occurrences=int(
                    specific.get("covered_source_occurrences") or 0),
            )
            return base
        if _all_source_occurrences_have_more_specific_decision(
                pending_source=pending_source or source_key,
                pending_decision_id=int(row.get("decision_id") or 0),
                source_text=source_text,
                decisions=context.get("decisions") or {}):
            base.update(
                status="blocked",
                reason="specific_decision_target_missing",
            )
            return base
        base.update(status="blocked", reason="locked_decision_target_missing")
        return base
    if target_present:
        base.update(status="satisfied", reason="target_present")
        return base
    if not source_active:
        base.update(status="satisfied", reason="source_absent")
        return base
    if specific is None:
        specific = _specific_decision_coverage(
            pending_source=pending_source or source_key,
            pending_decision_id=int(row.get("decision_id") or 0),
            source_text=source_text,
            final_text=final_prose,
            decisions=context.get("decisions") or {},
        )
    if specific.get("ok"):
        base.update(
            status="satisfied",
            reason="specific_decision_covered",
            covered_by_decisions=specific.get("covered_by_decisions") or [],
            covered_source_occurrences=int(
                specific.get("covered_source_occurrences") or 0),
        )
        return base

    state = base["state"]
    if state == "FAILED":
        base.update(status="blocked", reason="failed_task_not_in_backtrack_path")
    elif state == "PERMANENTLY_FAILED":
        base.update(status="blocked", reason="permanently_failed_task")
    elif state in NONTERMINAL_STATES:
        base.update(status="covered", reason="nonterminal_task_will_consume_decisions")
    elif state == "COMPLETED":
        base.update(status="outstanding", reason="completed_final_still_misses_target")
    else:
        base.update(status="blocked", reason=f"unknown_state:{state}")
    return base


def classify_pending_snapshot(
    context: Dict[str, Any],
    *,
    final_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if not context.get("available"):
        pending_rows = int(context.get("pending_rows") or
                           len(context.get("pending") or []))
        return {
            "ok": pending_rows == 0,
            "available": False,
            "reason": context.get("reason") or "pending_backtrack_unavailable",
            "total": pending_rows,
            "pending_rows": pending_rows,
            "counts": {},
            "by_chapter": {},
            "entries": [],
            "blocking": [],
        }
    entries = [classify_pending_row(row, context, final_overrides=final_overrides)
               for row in (context.get("pending") or [])]
    counts: Dict[str, int] = {}
    by_chapter: Dict[str, Dict[str, int]] = {}
    for entry in entries:
        status = str(entry["status"])
        counts[status] = counts.get(status, 0) + 1
        bucket = by_chapter.setdefault(entry["chapter"], {})
        bucket[status] = bucket.get(status, 0) + 1
    blocking = [e for e in entries if e["status"] in {"outstanding", "blocked", "orphan"}]
    return {
        "ok": not blocking,
        "available": True,
        "reason": "",
        "counts": dict(sorted(counts.items())),
        "by_chapter": {k: dict(sorted(v.items())) for k, v in sorted(by_chapter.items())},
        "entries": entries,
        "blocking": blocking,
        "total": len(entries),
    }


def pending_report_markdown(result: Dict[str, Any]) -> str:
    lines = ["# pending_backtrack 全书核对（只读）", ""]
    lines.append(f"- 生成时间: {_now_iso()}")
    lines.append(f"- 队列总数: **{result.get('total', 0)}**")
    lines.append(f"- 结论: **{'可清账/可收敛' if result.get('ok') else '存在未决项'}**")
    if result.get("reason"):
        lines.append(f"- 不可用原因: `{result['reason']}`")
    counts = result.get("counts") or {}
    if counts:
        lines.append("- 分类: " + ", ".join(f"`{k}`={v}" for k, v in counts.items()))
    by_chapter = result.get("by_chapter") or {}
    if by_chapter:
        lines.append("")
        lines.append("## 按章")
        for chapter, bucket in by_chapter.items():
            lines.append(f"- {chapter}: " + ", ".join(
                f"{k}={v}" for k, v in bucket.items()))
    blocking = result.get("blocking") or []
    if blocking:
        lines.append("")
        lines.append(f"## 需要处理（仅列前 40 / 共 {len(blocking)}）")
        for entry in blocking[:40]:
            lines.append(
                f"- `{entry['chunk_id']}` {entry['status']} / {entry['reason']} / "
                f"{entry.get('decision_source_key', '')} → "
                f"{entry.get('decision_translation', '')}"
            )
    else:
        lines.append("")
        lines.append("所有队列项均有机械证据解释；仍应由受控 apply 路径在状态迁移成功后消费。")
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def audit_pending_backtracks(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read-only audit; writes reports but never mutates DB/final/pending."""
    root = Path(project_root)
    out_dir = Path(output_dir) if output_dir else root / "output"
    context = load_pending_context(
        root, decision_db=decision_db, workflow_db=workflow_db,
        output_dir=out_dir,
    )
    result = classify_pending_snapshot(context)
    result["generated_at"] = _now_iso()
    result["mode"] = "read_only"
    json_path = Path(json_out) if json_out else out_dir / "pending_backtrack_reconcile.json"
    md_path = Path(md_out) if md_out else out_dir / "pending_backtrack_reconcile.md"
    result["json_path"] = str(json_path)
    result["md_path"] = str(md_path)
    _atomic_write(json_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(md_path, pending_report_markdown(result))
    return result


def _delete_pending_row_if_unchanged(
    con: sqlite3.Connection,
    row: Dict[str, Any],
) -> bool:
    """CAS-delete one outbox row; a concurrent refresh makes this a no-op."""
    cursor = con.execute(
        "DELETE FROM pending_backtrack WHERE chunk_id = ? "
        "AND decision_id IS ? AND source_key IS ? "
        "AND COALESCE(queue_rounds, 0) = ? "
        "AND COALESCE(updated_at, '') = COALESCE(?, '')",
        (
            str(row.get("chunk_id") or ""),
            row.get("decision_id"),
            row.get("source_key"),
            int(row.get("queue_rounds") or 0),
            row.get("updated_at"),
        ),
    )
    return cursor.rowcount == 1


def _pending_apply_markdown(result: Dict[str, Any]) -> str:
    lines = ["# pending_backtrack 全书清账（受控写入）", ""]
    lines.append(f"- 生成时间: {result.get('generated_at')}")
    lines.append(f"- 模式: **{result.get('mode')}**")
    lines.append(f"- 写入尝试: `{result.get('write_attempted')}`")
    if result.get("safe_only"):
        lines.append("- 范围: **仅消费机械满足项；outstanding 不触发 DIRTY**")
    if result.get("locked_target_only"):
        lines.append("- 范围: **仅处理 locked_decision_target_missing；其余行原样保留**")
    before = result.get("before_counts") or {}
    after = result.get("after_counts") or {}
    lines.append("- 清账前: " + (", ".join(
        f"`{k}`={v}" for k, v in before.items()) or "无"))
    lines.append("- 清账后: " + (", ".join(
        f"`{k}`={v}" for k, v in after.items()) or "无"))
    actions = result.get("actions") or []
    consumed = sum(1 for a in actions if a.get("status") == "consumed")
    triggered = sum(1 for a in actions if a.get("triggered_dirty"))
    lines.append(f"- 已消费: **{consumed}**；转 DIRTY: **{triggered}**")
    if actions:
        lines.append("")
        lines.append(f"## 动作（仅列前 60 / 共 {len(actions)}）")
        for action in actions[:60]:
            lines.append(
                f"- `{action.get('chunk_id')}` {action.get('before_status')} → "
                f"{action.get('status')} / {action.get('reason')}"
                + (" / triggered DIRTY" if action.get("triggered_dirty") else ""))
    if not result.get("ok"):
        lines.append("")
        lines.append("仍有 outstanding/blocked/orphan；gate ⑬ 必须保持红色。")
    lines.append("")
    return "\n".join(lines)


def apply_pending_backtrack_reconcile(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
    scheduler: Any = None,
    max_actions: Optional[int] = None,
    safe_only: bool = False,
    locked_target_only: bool = False,
    apply: bool = False,
) -> Dict[str, Any]:
    """Reconcile the complete book-wide outbox, optionally consuming safe rows.

    ``satisfied`` rows are removed after an exact snapshot CAS.  A ``covered``
    row is removed only when the chunk is already DIRTY, because that run will
    consume the latest decision.  Nonterminal translation states are left alone.
    An ``outstanding`` row is consumed only after ``COMPLETED -> DIRTY``
    succeeds.  With ``safe_only=True`` it is retained without a state change.
    ``blocked`` rows caused by a locked Decision losing its target are
    actionable: the lock is the authority and the current final is known not to
    satisfy it.  Other blocked rows and orphan rows always remain visible.
    ``locked_target_only=True`` narrows a mutating run to exactly that slice;
    every other row is retained untouched, so the CLI cannot accidentally turn
    all outstanding rows into DIRTY work.
    """
    root = Path(project_root)
    decision_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    workflow_path = Path(workflow_db) if workflow_db else root / "db" / "workflow.db"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "pending_backtrack_reconcile.json"
    mpath = Path(md_out) if md_out else out_dir / "pending_backtrack_reconcile.md"

    before_context = load_pending_context(
        root, decision_db=decision_path, workflow_db=workflow_path,
        output_dir=out_dir,
    )
    before = classify_pending_snapshot(before_context)
    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "apply" if apply else "read_only",
        "safe_only": bool(safe_only),
        "locked_target_only": bool(locked_target_only),
        "write_attempted": bool(apply),
        "generated_at": _now_iso(),
        "before_counts": dict(before.get("counts") or {}),
        "actions": [],
        "errors": [],
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    if not apply:
        result.update(before)
        result["mode"] = "read_only"
        result["write_attempted"] = False
        _atomic_write(jpath, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        _atomic_write(mpath, pending_report_markdown(result))
        return result

    if not before.get("available"):
        result.update(before)
        result["mode"] = "apply"
        result["write_attempted"] = True
        result["errors"].append(str(before.get("reason") or "pending_context_unavailable"))
        _atomic_write(jpath, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        _atomic_write(mpath, _pending_apply_markdown(result))
        return result

    if scheduler is None:
        from src.core.scheduler import TaskScheduler
        scheduler = TaskScheduler(db_path=str(workflow_path))

    try:
        limit = None if max_actions is None else max(0, int(max_actions))
    except (TypeError, ValueError):
        limit = None

    dcon = sqlite3.connect(str(decision_path))
    dcon.row_factory = sqlite3.Row
    outbox_by_chunk = {
        str(row.get("chunk_id") or ""): row
        for row in (before_context.get("pending") or [])
    }
    used = 0
    try:
        dcon.execute("BEGIN IMMEDIATE")
        for entry in before.get("entries") or []:
            if limit is not None and used >= limit:
                result["actions"].append({
                    "chunk_id": entry.get("chunk_id"),
                    "before_status": entry.get("status"),
                    "status": "deferred_by_limit",
                    "reason": entry.get("reason"),
                    "triggered_dirty": False,
                })
                continue
            before_status = str(entry.get("status") or "")
            outbox_row = outbox_by_chunk.get(str(entry.get("chunk_id") or ""), {})
            actionable_blocked = (
                before_status == "blocked"
                and entry.get("reason") == "locked_decision_target_missing")
            action = {
                "chunk_id": entry.get("chunk_id"),
                "chapter": entry.get("chapter"),
                "before_status": before_status,
                "reason": entry.get("reason"),
                "status": "kept",
                "triggered_dirty": False,
            }
            if locked_target_only and not actionable_blocked:
                action.update(
                    status="deferred_not_locked_target",
                    reason=entry.get("reason"),
                )
                result["actions"].append(action)
                continue
            if before_status == "satisfied":
                action["status"] = (
                    "consumed" if _delete_pending_row_if_unchanged(dcon, outbox_row)
                    else "stale_snapshot")
                if action["status"] == "consumed":
                    used += 1
            elif before_status == "covered" and entry.get("state") == "DIRTY":
                action["status"] = (
                    "consumed" if _delete_pending_row_if_unchanged(dcon, outbox_row)
                    else "stale_snapshot")
                if action["status"] == "consumed":
                    used += 1
            elif before_status in {"outstanding", "blocked"}:
                if before_status == "blocked" and not actionable_blocked:
                    action["status"] = "blocked_evidence_retained"
                    result["actions"].append(action)
                    continue
                if safe_only:
                    action.update(
                        status="retained_safe_only",
                        reason=(
                            "outstanding_requires_triage"
                            if before_status == "outstanding"
                            else "locked_target_repair_requires_apply"),
                    )
                else:
                    task = scheduler.get_task(
                        str(entry.get("chunk_id") or "")) or {}
                    if task.get("state") != "COMPLETED":
                        action["status"] = "state_changed"
                    else:
                        try:
                            transition = scheduler.trigger_backtrack(
                                [str(entry.get("chunk_id") or "")])
                            triggered = str(entry.get("chunk_id") or "") in set(
                                transition.get("dirty") or [])
                            action["triggered_dirty"] = triggered
                            if triggered:
                                task_after = scheduler.get_task(
                                    str(entry.get("chunk_id") or "")) or {}
                                if task_after.get("state") != "DIRTY":
                                    action["status"] = "transition_unverified"
                                elif _delete_pending_row_if_unchanged(
                                        dcon, outbox_row):
                                    action["status"] = "consumed"
                                    used += 1
                                else:
                                    action["status"] = "dirty_pending_stale"
                            else:
                                action["status"] = "blocked_by_backtrack_budget"
                        except Exception as exc:  # noqa: BLE001
                            action["status"] = "transition_failed"
                            action["error"] = f"{type(exc).__name__}: {exc}"
            elif before_status == "orphan":
                action["status"] = "orphan_retained"
            else:
                action["status"] = "unsupported_status_retained"
            result["actions"].append(action)
        dcon.commit()
    except Exception as exc:  # noqa: BLE001 - a failed apply must not hide evidence
        try:
            dcon.rollback()
        except sqlite3.Error:
            pass
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        dcon.close()

    after_context = load_pending_context(
        root, decision_db=decision_path, workflow_db=workflow_path,
        output_dir=out_dir,
    )
    after = classify_pending_snapshot(after_context)
    result.update(after)
    result["mode"] = "apply"
    result["safe_only"] = bool(safe_only)
    result["locked_target_only"] = bool(locked_target_only)
    result["write_attempted"] = True
    result["after_counts"] = dict(after.get("counts") or {})
    result["ok"] = bool(after.get("ok")) and not result["errors"]
    result["consumed"] = sum(1 for a in result["actions"] if a.get("status") == "consumed")
    result["triggered_dirty"] = sum(
        1 for a in result["actions"] if a.get("triggered_dirty"))
    _atomic_write(jpath, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(mpath, _pending_apply_markdown(result))
    return result

