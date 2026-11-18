"""Round 158: evidence-bound term/form contracts for pending reconciliation.

Decision DB rows are deliberately small, single-string facts.  Literary prose
also needs controlled short forms and the profile sometimes authorises two
spellings (for example ``stillsuit``).  Treating the Decision string as the
only legal surface form creates false pending rows and eventually causes whole
chunks to be retranslated for no semantic reason.

This module builds a separate, immutable contract artifact.  Every additional
form must have a named authority:

* an explicit translation hint in ``book_profile.json``;
* a pun-watchlist anchor in that profile;
* a high-confidence, active entity-registry canonical/variant; or
* an explicit ``中文（English source）`` gloss in a source-positive COMPLETED
  final, which is independent of the pending row being reconciled; or
* a narrowly defined short form of the Decision target (family name, doctor
  address, instrumental name, or a registered multi-part name head) that is
  also attested across source-positive COMPLETED chunks.

The builder never changes prose, Decisions, registry, pending rows, or task
state.  ``apply_term_form_contracts`` only writes the contract JSON atomically.
Consumption is implemented by ``pending_backtrack_reconcile``.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pending_backtrack_reconcile import (
    _article_shadow_authority_id,
    _contains_name,
    _name_key,
    _name_compatible,
    _phrase_key,
    _source_phrase_spans,
    _source_tokens,
    _target_core,
    body_before_footnotes,
    load_pending_context,
    term_text_similarity,
)
from src.term_form_audit import _near_target_forms, _similarity


TERM_CONTRACT_SCHEMA_VERSION = 1
DEFAULT_CONTRACT_PATH = "input/term_form_contracts.json"
DEFAULT_AUDIT_PATH = "output/term_form_contracts.json"
DEFAULT_AUDIT_MD_PATH = "output/term_form_contracts.md"

_CJK_FORM_RE = re.compile(r"^[\u4e00-\u9fff·・－\-]+$")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ROLE_SUFFIXES = ("家族", "医生", "琴")
_FUNCTION_PREFIXES = (
    "的", "了", "着", "地", "得", "在", "和", "与", "把", "被", "是",
    "这", "那", "一", "些", "个", "其", "他", "她", "它", "向", "从",
    "没", "有", "无", "像", "将", "只", "道", "给", "对",
    "会", "能", "可", "要", "已", "还", "又", "也", "都", "就", "才",
    "仅", "正", "仍", "却", "便", "再", "更", "很", "太", "不", "未",
    "别", "请", "让", "叫", "使", "为",
)
_FUNCTION_SUFFIXES = (
    "的", "地", "得", "了", "着", "是", "在", "和", "与", "这", "那",
    "个", "些", "们", "一", "样",
)
_FUNCTION_INFIXES = ("的", "地", "得", "了", "着", "之")
_GLOSS_CLAUSE_CHARS = frozenset("的地得了着之在是我你他她它您这那")
_GLOSS_BOUNDARIES = (
    "成为", "作为", "称为", "叫做", "尝试", "采用", "看到", "看见",
    "拥有", "使用", "写作", "译作", "叫作", "借助", "通过", "利用",
    "凭借",
    "是", "为", "有", "用", "将", "把", "被", "叫", "称", "能", "着",
    "在", "和", "与", "从", "向", "对", "给", "让", "使",
    "我", "你", "您", "他", "她", "它",
)

_SOURCE_GLOSS_RE = re.compile(
    r"(?P<cjk>[\u4e00-\u9fff·・－\-]{2,40})\s*[（(]\s*"
    r"(?P<source>[A-Za-z][A-Za-z0-9'’\- .]{1,100}?)"
    r"\s*(?:\[\^[^\]]+\]\s*)*[）)]"
)


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _cjk_edit_distance(left: str, right: str) -> int:
    """Small-string Levenshtein distance for lexical near-form checks."""
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return 2
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right))
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    i = j = edits = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return 2
        j += 1
    return edits + (len(longer) - j)


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


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _source_match_key(value: Any) -> str:
    key = re.sub(r"\s+", " ", str(value or "").casefold()).strip()
    if key.endswith("'s"):
        key = key[:-2]
    # Profile hints use the collective plural while Decision keys often use the
    # singular (Harkonnens vs Harkonnen).  Only trim a simple final ``s``.
    if len(key) > 3 and key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key


def _clean_form(value: Any) -> str:
    form = str(value or "").strip().strip("《》“”\"'")
    if not form or len(form) > 24 or not _CJK_FORM_RE.fullmatch(form):
        return ""
    return form


def _forms_from_hint(source: str, hint: str) -> List[str]:
    """Extract only surface forms, not the explanatory description."""
    if "=" not in hint:
        return []
    left, right = hint.split("=", 1)
    if _source_match_key(left) != _source_match_key(source):
        return []
    head = re.split(r"[：;；]", right, maxsplit=1)[0].strip()
    forms: List[str] = []
    for part in re.split(r"[／/]", head):
        form = _clean_form(re.sub(r"[（(][^（()）]*[）)]", "", part))
        if form:
            forms.append(form)
    # Some hints put the executable form in quotes after prose
    # (Kwisatz Haderach).  Do not harvest every quoted word: clauses such as
    # "不宜仅译“生态学家”" are explicitly rejected forms.
    for quoted in re.findall(r"(?:可统一为|统一为|统一|首次写作)[^“]*“([^”]+)”", right):
        form = _clean_form(re.sub(r"[（(][^（()）]*[）)]", "", quoted))
        if form:
            forms.append(form)
    return forms


def _profile_forms(profile: dict) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for hint in profile.get("translation_hints") or []:
        if not isinstance(hint, str) or "=" not in hint:
            continue
        source = hint.split("=", 1)[0].strip()
        for form in _forms_from_hint(source, hint):
            out.setdefault(_source_match_key(source), []).append({
                "form": form,
                "authority": "profile_translation_hint",
                "evidence": hint,
            })
    for item in profile.get("pun_watchlist") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "")
        key = _source_match_key(source)
        hint_keys = {_name_key(row.get("form") or "")
                     for row in out.get(key, []) if row.get("form")}
        for anchor in item.get("anchors") or []:
            form = _clean_form(anchor)
            # Pun explanations often contain ordinary phrases ("缩短距离").
            # An anchor becomes an executable form only when the same source's
            # translation hint already lists it as a form.
            if (form and _name_key(form) in hint_keys
                    and form.casefold() != source.casefold()):
                out.setdefault(key, []).append({
                    "form": form,
                    "authority": "profile_pun_anchor",
                    "evidence": str(item.get("strategy") or source),
                })
    return out


def _same_cjk_word_order(left: str, right: str) -> bool:
    """True for a title word-order variant with identical CJK characters."""
    left_key = _name_key(left)
    right_key = _name_key(right)
    return bool(left_key and right_key and left_key != right_key
                and sorted(left_key) == sorted(right_key))


def _gloss_surface_form(prefix: str, target: str) -> str:
    """Pick the noun phrase immediately before an explicit source gloss.

    A gloss such as ``会用杂酚灌木（creosote bush）`` has a verb in the
    same CJK run.  Keep only a suffix that has no function-word boundary and
    is closest to the Decision target.  The returned form is evidence for a
    contract, never a rewrite instruction.
    """
    target_key = _name_key(target)
    if not target_key:
        return ""
    # The gloss prefix can be exactly the target.  Do not let a literal
    # function character inside the target (``之`` in ``外世界之声``) cause the
    # heuristic below to trim a valid full form down to its tail (``之声``).
    if _name_key(prefix).endswith(target_key):
        return target
    candidates: List[Tuple[float, int, str]] = []
    limit = min(12, len(prefix))
    for size in range(2, limit + 1):
        form = prefix[-size:]
        if not _CJK_FORM_RE.fullmatch(form):
            continue
        if form.startswith(_FUNCTION_PREFIXES):
            continue
        if form.endswith(_FUNCTION_SUFFIXES):
            continue
        if any(char in form[1:-1] for char in _FUNCTION_INFIXES):
            continue
        candidates.append((_similarity(form, target), len(form), form))
    if not candidates:
        return ""
    # When a short, clean CJK run is immediately followed by its source gloss,
    # prefer the whole run over a target-similar suffix.  ``宾杜悬浮（Bindu
    # suspension）`` must yield the full term, not the generic ``悬浮``.  Clause
    # markers veto this shortcut so ``如果我在沙虫活动区（worm zone）`` still
    # reaches the boundary-based extraction below.
    full = prefix
    if (2 <= len(full) <= 8
            and _CJK_FORM_RE.fullmatch(full)
            and not any(char in full for char in _GLOSS_CLAUSE_CHARS)
            and not full.startswith(_FUNCTION_PREFIXES)
            and not full.endswith(_FUNCTION_SUFFIXES)
            and not any(char in full[1:-1] for char in _FUNCTION_INFIXES)):
        has_boundary = any(boundary in prefix for boundary in _GLOSS_BOUNDARIES)
        if not has_boundary:
            return full
    # A verb or function word often shares the CJK run
    # (会用杂酚灌木 / 你以为我是奎萨茨·哈德拉赫).  Prefer the suffix after
    # the last such boundary when it is a plausible term.
    boundary_forms: List[Tuple[int, str]] = []
    for boundary in _GLOSS_BOUNDARIES:
        pos = prefix.rfind(boundary)
        if pos < 0:
            continue
        form = prefix[pos + len(boundary):]
        if 2 <= len(form) <= 12 and _CJK_FORM_RE.fullmatch(form):
            if (not form.startswith(_FUNCTION_PREFIXES)
                    and not form.endswith(_FUNCTION_SUFFIXES)
                    and not any(char in form[1:-1]
                                for char in _FUNCTION_INFIXES)):
                boundary_forms.append((pos, form))
    if boundary_forms:
        return max(boundary_forms, key=lambda item: item[0])[1]
    # If the gloss shares the target's head, the longer form is the compound
    # (杂酚灌木, not the bare 灌木).  Otherwise use similarity, then length for
    # alternate names such as Kwisatz Haderach -> 奎萨茨·哈德拉赫.
    target_key = _name_key(target)
    for _score, _length, form in candidates:
        if _name_key(form) == target_key:
            return form
    head = target_key[-2:] if len(target_key) >= 2 else target_key
    head_forms = [item for item in candidates
                  if head and _name_key(item[2]).endswith(head)]
    if head_forms:
        return max(head_forms, key=lambda item: (item[1], item[0]))[2]
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _source_gloss_forms(
    corpus: Dict[str, Dict[str, str]],
    source_key: str,
    target: str,
) -> List[Dict[str, Any]]:
    """Harvest forms explicitly bound to the source key by a parenthetical.

    The source phrase must match the complete parenthetical text, not merely a
    token inside it.  This prevents ``carryall wing`` from becoming evidence
    for the shorter ``carryall`` Decision.  Only source-positive COMPLETED
    finals participate, so a pending chunk cannot validate itself.
    """
    wanted = _phrase_key(source_key)
    if not wanted:
        return []
    forms: List[Dict[str, Any]] = []
    seen = set()
    for chunk_id, row in corpus.items():
        if not _source_phrase_spans(row.get("source") or "", source_key):
            continue
        final = body_before_footnotes(row.get("final") or "")
        for match in _SOURCE_GLOSS_RE.finditer(final):
            if _phrase_key(match.group("source")) != wanted:
                continue
            form = _gloss_surface_form(match.group("cjk"), target)
            if not form or form in seen:
                continue
            seen.add(form)
            forms.append({
                "form": form,
                "authority": "source_gloss",
                "evidence": "explicit gloss in %s: %s（%s）" % (
                    chunk_id, form, match.group("source")),
            })
    return forms


def _embedded_source_gloss_forms(
    corpus: Dict[str, Dict[str, str]],
    source_key: str,
    target: str,
    decisions: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Allow a glossed longer phrase whose final token is the Decision key.

    Examples are ``Great Erg`` for the Decision key ``erg`` and ``Bene
    Gesserit litany`` for ``litany``.  The source chapter must contain the full
    gloss phrase, the pending text must contain the extracted CJK form, and the
    phrase must not already be owned by a more specific Decision (for example
    ``body shield`` must not authorise the generic key ``shield``).
    """
    key_tokens = _source_tokens(source_key)
    if len(key_tokens) != 1 or len(key_tokens[0]) < 3:
        return []
    key = key_tokens[0]
    target_key = _name_key(_target_core(target))
    decision_keys = {
        _phrase_key(str(item.get("source_key") or ""))
        for item in (decisions or {}).values()
        if str(item.get("source_key") or "").strip()
    }
    forms: List[Dict[str, Any]] = []
    seen = set()
    for chunk_id, row in corpus.items():
        source_text = str(row.get("source") or "")
        if not _source_phrase_spans(source_text, source_key):
            continue
        final = body_before_footnotes(str(row.get("final") or ""))
        for match in _SOURCE_GLOSS_RE.finditer(final):
            phrase = str(match.group("source") or "").strip()
            phrase_tokens = _source_tokens(phrase)
            if (len(phrase_tokens) < 2 or phrase_tokens[-1] != key
                    or _phrase_key(phrase) in decision_keys
                    or not _source_phrase_spans(source_text, phrase)):
                continue
            form = _gloss_surface_form(match.group("cjk"), target)
            form_key = _name_key(form)
            if (not form_key or len(form_key) < len(target_key)
                    or form_key == _name_key(source_key)):
                continue
            dedupe = (form_key, _phrase_key(phrase))
            if dedupe in seen:
                continue
            seen.add(dedupe)
            forms.append({
                "form": form,
                "authority": "source_gloss_embedded_term",
                "evidence": "explicit gloss in %s: %s（%s）; Decision key %s is the phrase head" % (
                    chunk_id, form, phrase, source_key),
            })
    return forms


def _gloss_extension_forms(
    pending_texts: List[str],
    source_key: str,
    target: str,
    corpus: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Allow one CJK insertion into an explicitly glossed term.

    ``杂酚灌木（creosote bush）`` and the body form ``杂酚油灌木`` are
    morphological variants of the same plant name.  The base gloss is
    independently attested in a completed source-positive chunk; the pending
    final supplies only the one-character extension.
    """
    joined = "\n".join(body_before_footnotes(text) for text in pending_texts)
    if not joined:
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in _source_gloss_forms(corpus, source_key, target):
        base = str(item.get("form") or "")
        if not base or len(base) < 2 or len(base) > 16:
            continue
        # Candidate generation is bounded by the CJK characters actually in
        # the pending text; never synthesize an unseen string.
        chars = set("".join(_CJK_RUN_RE.findall(joined)))
        for pos in range(1, len(base)):
            for inserted in sorted(chars):
                candidate = base[:pos] + inserted + base[pos:]
                if candidate == base or candidate in seen:
                    continue
                if not _CJK_FORM_RE.fullmatch(candidate):
                    continue
                if not _contains_name(joined, candidate):
                    continue
                if candidate.startswith(_FUNCTION_PREFIXES) or candidate.endswith(
                        _FUNCTION_SUFFIXES):
                    continue
                if any(value in candidate[1:-1] for value in _FUNCTION_INFIXES):
                    continue
                seen.add(candidate)
                out.append({
                    "form": candidate,
                    "authority": "source_gloss_one_char_extension",
                    "evidence": "%s; one-character body extension of %s" % (
                        item.get("evidence") or "explicit source gloss", base),
                })
    return out


def _gloss_modifier_elision_forms(
    pending_texts: List[str],
    source_key: str,
    target: str,
    corpus: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Allow a short, bounded modifier elision from an explicit gloss."""
    joined = "\n".join(body_before_footnotes(text) for text in pending_texts)
    if not joined:
        return []
    modifiers = ("两色", "色的", "活动", "区域", "地区", "家族", "的")
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in _source_gloss_forms(corpus, source_key, target):
        base = str(item.get("form") or "")
        if not base:
            continue
        for modifier in modifiers:
            if modifier not in base:
                continue
            candidate = _clean_form(base.replace(modifier, "", 1))
            if not candidate or candidate == base or candidate in seen:
                continue
            if not _contains_name(joined, candidate):
                continue
            seen.add(candidate)
            out.append({
                "form": candidate,
                "authority": "source_gloss_modifier_elision",
                "evidence": "%s; bounded modifier elision %s" % (
                    item.get("evidence") or "explicit source gloss", modifier),
            })
    return out


def _registry_word_order_forms(
    context: Dict[str, Any],
    source_key: str,
    target: str,
) -> List[Dict[str, Any]]:
    """Return full active registry variants that only reorder title characters."""
    owners = ((context.get("registry_form_owners") or {}).get(
        _phrase_key(source_key)) or [])
    out: List[Dict[str, Any]] = []
    for owner in owners:
        if int(owner.get("pending_review") or 0):
            continue
        canonical = str(owner.get("canonical") or "")
        if not canonical or not _name_compatible(canonical, target):
            continue
        for variant in owner.get("variants") or []:
            if not _same_cjk_word_order(str(variant), target):
                continue
            out.append({
                "form": str(variant),
                "authority": "registry_active_word_order_variant",
                "evidence": "entity_registry canonical %s" % canonical,
            })
    return out


def _completed_corpus(context: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    out_dir = context.get("output_dir") or Path(".")
    for chunk_id, task in (context.get("tasks") or {}).items():
        if str(task.get("state") or "") != "COMPLETED":
            continue
        chapter = str(task.get("chapter_id") or str(chunk_id)[:4])
        path = Path(out_dir) / chapter / f"{chunk_id}_final.json"
        payload = _read_json(path)
        out[str(chunk_id)] = {
            "source": str(task.get("text_content") or ""),
            "final": body_before_footnotes(payload.get("text")),
        }
    return out


def _corpus_coverage(
    source_key: str,
    form: str,
    corpus: Dict[str, Dict[str, str]],
    *,
    exclude_chunk_ids: Optional[set] = None,
) -> Tuple[int, int]:
    positive = 0
    hits = 0
    excluded = exclude_chunk_ids or set()
    for chunk_id, row in corpus.items():
        if chunk_id in excluded:
            continue
        if not _source_phrase_spans(row.get("source") or "", source_key):
            continue
        positive += 1
        if _contains_name(body_before_footnotes(row.get("final")), form):
            hits += 1
    return hits, positive


def _registry_forms(
    context: Dict[str, Any],
    source_key: str,
) -> List[Dict[str, str]]:
    trusted = ((context.get("registry_trusted_by_source") or {}).get(
        _phrase_key(source_key)) or set())
    canonicals = sorted({str(value).strip() for value in trusted
                         if str(value).strip()})
    if len({_name_key(value) for value in canonicals}) != 1:
        return []
    canonical = _clean_form(canonicals[0])
    if not canonical:
        return []
    return [{
        "form": canonical,
        "authority": "registry_trusted_canonical",
        "evidence": "entity_registry source_names + pending_review=0",
    }]


def _sibling_decision_forms(
    context: Dict[str, Any],
    source_key: str,
    current_decision_id: int,
) -> List[Dict[str, str]]:
    """Other active Decisions for the exact same source key.

    A source key occasionally has two Decision rows with distinct surface
    forms (for example crysknife -> 晶牙刀 / 晶牙匕).  Both rows are explicit
    human-curated authorities, unlike corpus guesses.  They are safe as
    evidence for a form contract, but they do not change either Decision.
    """
    wanted = _source_match_key(source_key)
    out: List[Dict[str, str]] = []
    for decision in (context.get("decisions") or {}).values():
        try:
            decision_id = int(decision.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if decision_id == int(current_decision_id or 0):
            continue
        if _source_match_key(decision.get("source_key")) != wanted:
            continue
        form = _clean_form(_target_core(
            str(decision.get("translation") or "")))
        if not form:
            continue
        out.append({
            "form": form,
            "authority": "sibling_decision_alias",
            "evidence": "Decision #%d shares source_key %s" % (
                decision_id, source_key),
        })
    return out


def _short_forms(
    target: str,
    source_key: str,
    corpus: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    core = _target_core(target)
    candidates: List[Dict[str, Any]] = []
    for suffix in _ROLE_SUFFIXES:
        if core.endswith(suffix) and len(core) > len(suffix):
            stem = core[:-len(suffix)]
            if suffix == "医生" and len(stem) != 1:
                continue
            if len(stem) >= 2 or suffix == "医生":
                hits, positive = _corpus_coverage(source_key, stem, corpus)
                if positive and hits * 2 >= positive:
                    candidates.append({
                        "form": stem,
                        "authority": "target_" + suffix + "_short_form",
                        "evidence": "attested %d/%d source-positive COMPLETED chunks" % (
                            hits, positive),
                    })
    if "·" in core:
        head = _clean_form(core.split("·", 1)[0])
        if head:
            core_key = _name_key(core)
            head_key = _name_key(head)
            # A family-name head is useful for ``吉迪·普莱姆 -> 吉迪`` but
            # not for a faction-plus-noun title such as
            # ``贝尼·杰瑟里特妖妇 -> 贝尼``.  Require the head to retain at
            # least 40% of the canonical CJK form.
            ratio_ok = bool(head_key) and len(head_key) * 5 >= len(core_key) * 2
            hits, positive = _corpus_coverage(source_key, head, corpus)
            if ratio_ok and positive and hits * 2 >= positive:
                candidates.append({
                    "form": head,
                    "authority": "multi_part_name_head",
                    "evidence": "attested %d/%d source-positive COMPLETED chunks" % (
                        hits, positive),
                })
    return candidates


def _corpus_consensus_forms(
    pending_texts: List[str],
    source_key: str,
    target: str,
    corpus: Dict[str, Dict[str, str]],
    *,
    pending_chunk_ids: Optional[List[str]] = None,
    allow_self_attestation: bool = False,
    require_target_length: bool = False,
) -> List[Dict[str, Any]]:
    """Return stable alternate surface forms observed in the book body.

    This is intentionally stricter than "contains a similar n-gram".  A form
    must occur in the pending final and in a material share of *other*
    source-positive COMPLETED chunks.  The pending chunk itself is excluded:
    counting it as evidence would make every local substring a circular 1/1
    "consensus".  Typos such as 马普斯 vs 马佩斯 (0 independent positive chunks)
    therefore never qualify.
    """
    core = _target_core(target)
    target_key = _name_key(core)
    target_len = len(target_key)
    source_lower = bool(source_key) and any(ch.isalpha() for ch in source_key) and all(
        (not ch.isalpha()) or ch.islower() for ch in source_key)
    if not target_key:
        return []
    out: List[Dict[str, Any]] = []
    for text_index, text in enumerate(pending_texts):
        text = body_before_footnotes(text)
        exclude = set()
        if (not allow_self_attestation and pending_chunk_ids
                and text_index < len(pending_chunk_ids)):
            chunk_id = str(pending_chunk_ids[text_index] or "")
            if chunk_id:
                exclude.add(chunk_id)
        candidates = set()
        for near in _near_target_forms(text, core, limit=12):
            form = str(near.get("form") or "")
            if not form:
                continue
            for run in _CJK_RUN_RE.findall(text):
                if form not in run:
                    continue
                run_len = len(run)
                for size in range(2, min(10, run_len) + 1):
                    for start in range(0, run_len - size + 1):
                        candidate = run[start:start + size]
                        if form in candidate:
                            candidates.add(candidate)
        scored: List[Tuple[int, int, float, int, str, int]] = []
        for candidate in candidates:
            key = _name_key(candidate)
            if not key or key == target_key:
                continue
            if candidate.startswith(_FUNCTION_PREFIXES):
                continue
            if candidate.endswith(_FUNCTION_SUFFIXES):
                continue
            if any(char in candidate[1:-1] for char in _FUNCTION_INFIXES):
                continue
            # A short generic fragment inside the canonical target is already
            # handled by the explicit short-form policy, not corpus consensus.
            if key in target_key and key != target_key:
                # A lowercase compound may legitimately drop a descriptive
                # prefix (factory crawler -> 采集车); uppercase proper names may
                # not be shortened through this corpus-only path.
                if not (source_lower and len(key) >= 3 and target_key.endswith(key)):
                    continue
            # Consensus is only a fallback for an alternate surface form of the
            # same term.  Allowing several extra characters lets a neighbouring
            # clause become a "term" (e.g. 会为卫星控制 instead of 天气控制).
            if len(key) > target_len + 1:
                continue
            if require_target_length and len(key) != target_len:
                continue
            shared_bigram = any(
                key[index:index + 2] in target_key
                for index in range(max(0, len(key) - 1)))
            same_head = key[-1:] == target_key[-1:]
            if not shared_bigram and not same_head:
                continue
            embedded = key in target_key or target_key in key
            lexical_near = embedded or _cjk_edit_distance(key, target_key) <= 1
            hits, positive = _corpus_coverage(
                source_key, candidate, corpus, exclude_chunk_ids=exclude)
            if not positive:
                continue
            if lexical_near:
                # Exact substrings and one-edit forms (保护传教团 vs
                # 保护性传教团) may rely on the ordinary majority threshold.
                if positive <= 4:
                    if hits < 1 or hits * 2 < positive:
                        continue
                elif hits < 2 or hits * 5 < positive * 2:
                    continue
            elif positive == 1 and source_lower and same_head and hits >= 1:
                # A singleton lowercase context can still justify a close
                # contextual noun (酒壶/水壶).  Multi-positive paraphrases use
                # the stricter consensus check below.
                pass
            elif positive <= 4:
                if hits < 1 or hits * 2 < positive:
                    continue
            elif hits < 2 or hits * 5 < positive * 2:
                continue
            # Non-lexical paraphrases need a 2/3 majority.  The former 50%
            # threshold allowed wrong forms such as ``desert power -> 空中力量``
            # (2/4) and ``worm zone -> 沙虫区启动`` (1/2) to become contracts.
            if not lexical_near and positive > 1 and (
                    hits < 2 or hits * 3 < positive * 2):
                continue
            scored.append((
                hits,
                len(candidate),
                _similarity(candidate, core),
                len(candidate),
                candidate,
                positive,
            ))
        if not scored:
            continue
        hits, _preferred_length, _similarity_sort, _len_sort, form, positive = max(
            scored, key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
        out.append({
            "form": form,
            "authority": "corpus_consensus",
            "evidence": "attested %d/%d source-positive COMPLETED chunks" % (
                hits, positive),
        })
    return out


def _single_context_common_noun_forms(
    pending_texts: List[str],
    source_key: str,
    target: str,
    corpus: Dict[str, Dict[str, str]],
    pending_chunk_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Accept a near-target form for a lowercase common noun in its own context.

    This is deliberately narrower than proper-name normalization.  Lowercase
    source phrases such as ``flagon of water`` may have a contextually better
    Chinese noun than the single Decision string (``水壶`` versus ``酒壶``);
    forcing the Decision target would make the prose worse.  Uppercase proper
    names are never eligible here, and the same prefix/suffix/length filters as
    corpus consensus still apply.
    """
    has_alpha = any(ch.isalpha() for ch in (source_key or ""))
    source_lower = bool(source_key) and has_alpha and all(
        (not ch.isalpha()) or ch.islower() for ch in source_key)
    if not source_lower:
        return []
    forms = _corpus_consensus_forms(
        pending_texts,
        source_key,
        target,
        corpus,
        pending_chunk_ids=pending_chunk_ids,
        allow_self_attestation=True,
        require_target_length=True,
    )
    for item in forms:
        item["authority"] = "single_context_common_noun_variant"
        item["evidence"] = (
            "%s; scoped to a lowercase common noun in this source context"
            % item.get("evidence", "")
        )
    return forms


def _title_variant_forms(
    pending_texts: List[str],
    source_key: str,
    target: str,
    corpus: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Contextual honorific/title variants, separate from lexical spelling."""
    core = _target_core(target)
    key = _source_match_key(source_key).casefold()
    candidates: set = set()
    if key == "my lord":
        candidates.update({"大人", "阁下", "陛下", "主公"})
    if key.endswith("mother") and core.endswith("母"):
        for text in pending_texts:
            for near in _near_target_forms(text, core, limit=12):
                form = str(near.get("form") or "")
                if len(form) == 2 and form.endswith("母"):
                    candidates.add(form)
    out: List[Dict[str, Any]] = []
    joined = "\n".join(pending_texts)
    for candidate in sorted(candidates):
        if candidate == core or not _contains_name(joined, candidate):
            continue
        hits, positive = _corpus_coverage(source_key, candidate, corpus)
        if hits >= 2 and positive:
            out.append({
                "form": candidate,
                "authority": "contextual_title_variant",
                "evidence": "attested %d/%d source-positive COMPLETED chunks" % (
                    hits, positive),
            })
    return out


def _registry_name_short_forms(
    pending_texts: List[str],
    source_key: str,
    target: str,
    corpus: Dict[str, Dict[str, str]],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Short heads of active registry variants compatible with the Decision."""
    owners = ((context.get("registry_form_owners") or {}).get(
        _phrase_key(source_key)) or [])
    joined = "\n".join(pending_texts)
    out: List[Dict[str, Any]] = []
    for owner in owners:
        if int(owner.get("pending_review") or 0):
            continue
        canonical = str(owner.get("canonical") or "")
        if not canonical or not _name_compatible(canonical, target):
            continue
        for full_form in [canonical] + list(owner.get("variants") or []):
            if "·" not in str(full_form):
                continue
            head = _clean_form(str(full_form).split("·", 1)[0])
            if not head or not _contains_name(joined, head):
                continue
            hits, positive = _corpus_coverage(source_key, head, corpus)
            if hits >= 1 and positive:
                out.append({
                    "form": head,
                    "authority": "registry_variant_short_form",
                    "evidence": "%s / attested %d/%d source-positive COMPLETED chunks" % (
                        full_form, hits, positive),
                })
    return out


def _merge_forms(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in items:
        form = _clean_form(item.get("form"))
        if not form:
            continue
        key = _name_key(form)
        if not key:
            continue
        row = merged.setdefault(key, {
            "form": form,
            "authorities": [],
            "evidence": [],
        })
        authority = str(item.get("authority") or "")
        if authority and authority not in row["authorities"]:
            row["authorities"].append(authority)
        evidence = str(item.get("evidence") or "")
        if evidence and evidence not in row["evidence"]:
            row["evidence"].append(evidence)
    return list(merged.values())


def build_term_form_contracts(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    workflow_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    profile_path: Optional[Path] = None,
    contracts_path: Optional[Path] = None,
    report_json: Optional[Path] = None,
    report_md: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(project_root)
    out_dir = Path(output_dir) if output_dir else root / "output"
    cpath = (Path(contracts_path) if contracts_path
             else root / DEFAULT_CONTRACT_PATH)
    jpath = Path(report_json) if report_json else out_dir / "term_form_contracts.json"
    mpath = Path(report_md) if report_md else out_dir / "term_form_contracts.md"
    ppath = (Path(profile_path) if profile_path
             else root / "input" / "book_profile.json")
    result: Dict[str, Any] = {
        "schema_version": TERM_CONTRACT_SCHEMA_VERSION,
        "mode": "read_only",
        "write_attempted": False,
        "generated_at": _now_iso(),
        "profile_path": str(ppath),
        "contracts_path": str(cpath),
        "contracts": [],
        "counts": {},
        "errors": [],
        "ok": False,
        "json_path": str(jpath),
        "md_path": str(mpath),
    }
    context = load_pending_context(
        root, decision_db=decision_db, workflow_db=workflow_db,
        output_dir=out_dir,
    )
    if not context.get("available"):
        result["errors"].append(str(context.get("reason") or "pending_context_unavailable"))
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_markdown(result))
        return result
    profile = _read_json(ppath)
    if not profile:
        result["errors"].append("profile_missing_or_invalid")
        _atomic_json(jpath, result)
        _atomic_text(mpath, _render_markdown(result))
        return result
    profile_forms = _profile_forms(profile)
    corpus = _completed_corpus(context)
    decisions = context.get("decisions") or {}

    # Unresolved rows need contracts.  For a locked Decision, a contract is
    # only emitted when every pending row is covered; the lock remains the
    # primary authority and a contract must not leave a mixed lock state.
    from src.pending_backtrack_reconcile import classify_pending_snapshot
    snapshot = classify_pending_snapshot(context)
    grouped: Dict[Tuple[int, str, str], List[dict]] = {}
    for entry in snapshot.get("entries") or []:
        status = str(entry.get("status") or "")
        reason = str(entry.get("reason") or "")
        if not (status == "outstanding"
                or (status == "blocked" and reason == "locked_decision_target_missing")):
            continue
        key = (int(entry.get("decision_id") or 0),
               str(entry.get("decision_source_key") or ""),
               str(entry.get("decision_translation") or ""))
        grouped.setdefault(key, []).append(entry)

    contracts: List[Dict[str, Any]] = []
    for (decision_id, source_key, target), pending_rows in sorted(grouped.items()):
        if _article_shadow_authority_id(
                decisions.get(int(decision_id)) or {}, decisions):
            continue
        core = _target_core(target)
        pending_texts = [
            body_before_footnotes(_read_json(
                Path(out_dir) / str(row.get("chapter") or str(row.get("chunk_id"))[:4])
                / f"{row.get('chunk_id')}_final.json").get("text"))
            for row in pending_rows]
        source_is_quote = (
            len(source_key) >= 80
            or (len(source_key.split()) >= 12 and (";" in source_key or "!" in source_key))
        )
        if source_is_quote:
            matched_rows: List[Dict[str, str]] = []
            uncovered: List[str] = []
            minimum = 0.8
            for row, text in zip(pending_rows, pending_texts):
                chunk_id = str(row.get("chunk_id") or "")
                similarity = term_text_similarity(text, core)
                if similarity >= minimum:
                    matched_rows.append({
                        "chunk_id": chunk_id,
                        "form": "fuzzy_text:%.4f" % similarity,
                    })
                else:
                    uncovered.append(chunk_id)
            if matched_rows:
                contracts.append({
                    "contract_id": "term-quote-%d" % decision_id,
                    "decision_id": decision_id,
                    "source_key": source_key,
                    "decision_translation": target,
                    "forms": [],
                    "match_mode": "fuzzy_text",
                    "reference": core,
                    "min_similarity": minimum,
                    "pending_chunks": [
                        str(row.get("chunk_id") or "") for row in pending_rows],
                    "matched_chunks": matched_rows,
                    "uncovered_chunks": uncovered,
                    "status": "complete" if not uncovered else "partial",
                })
            continue
        evidence_items = [{
            "form": core,
            "authority": "decision_target",
            "evidence": "Decision #%d" % decision_id,
        }]
        evidence_items.extend(profile_forms.get(_source_match_key(source_key), []))
        evidence_items.extend(_registry_forms(context, source_key))
        evidence_items.extend(_registry_word_order_forms(
            context, source_key, target))
        evidence_items.extend(_sibling_decision_forms(
            context, source_key, decision_id))
        evidence_items.extend(_source_gloss_forms(corpus, source_key, target))
        evidence_items.extend(_embedded_source_gloss_forms(
            corpus, source_key, target, decisions))
        evidence_items.extend(_gloss_extension_forms(
            pending_texts, source_key, target, corpus))
        evidence_items.extend(_gloss_modifier_elision_forms(
            pending_texts, source_key, target, corpus))
        evidence_items.extend(_short_forms(core, source_key, corpus))
        evidence_items.extend(_title_variant_forms(
            pending_texts, source_key, core, corpus))
        evidence_items.extend(_registry_name_short_forms(
            pending_texts, source_key, core, corpus, context))
        explicit_forms = _merge_forms(evidence_items)
        explicit_covered = all(
            any(_contains_name(text, item.get("form") or "")
                for item in explicit_forms)
            for text in pending_texts)
        if not explicit_covered:
            evidence_items.extend(_corpus_consensus_forms(
                pending_texts, source_key, core, corpus,
                pending_chunk_ids=[
                    str(row.get("chunk_id") or "") for row in pending_rows]))
            evidence_items.extend(_single_context_common_noun_forms(
                pending_texts, source_key, core, corpus,
                pending_chunk_ids=[
                    str(row.get("chunk_id") or "") for row in pending_rows]))
        forms = _merge_forms(evidence_items)
        # This artifact is scoped to currently unresolved rows.  Keeping only
        # forms actually present in one of those finals prevents descriptive
        # profile prose (for example "男性先知性称谓" or a pun explanation)
        # from becoming an executable surface form.
        observed_texts = pending_texts
        forms = [
            item for item in forms
            if "decision_target" in item.get("authorities", [])
            or any(_contains_name(text, item.get("form") or "")
                   for text in observed_texts)
        ]
        # A long proper name must not be silently reduced to a 3-character
        # head just because that head occurs somewhere in the pending prose.
        # Shorter registered forms are still allowed for ordinary short names.
        core_key = _name_key(core)
        if len(core_key) >= 8:
            forms = [item for item in forms
                     if len(_name_key(item.get("form") or ""))
                     >= len(core_key) - 3]
        if len(forms) <= 1:
            continue
        matched_rows: List[Dict[str, str]] = []
        uncovered: List[str] = []
        for row in pending_rows:
            chunk_id = str(row.get("chunk_id") or "")
            chapter = str(row.get("chapter") or chunk_id[:4])
            final = _read_json(Path(out_dir) / chapter / f"{chunk_id}_final.json")
            text = body_before_footnotes(final.get("text"))
            match = next((item for item in forms
                          if _contains_name(text, item["form"])), None)
            if match:
                matched_rows.append({"chunk_id": chunk_id,
                                     "form": str(match["form"])})
            else:
                uncovered.append(chunk_id)
        decision_locked = bool(int(((context.get("decisions") or {}).get(
            decision_id) or {}).get("locked") or 0))
        if not matched_rows or (decision_locked and uncovered):
            continue
        contracts.append({
            "contract_id": "term-%d-%s" % (
                decision_id, re.sub(r"[^a-z0-9]+", "-", source_key.casefold())[:40]),
            "decision_id": decision_id,
            "source_key": source_key,
            "decision_translation": target,
            "forms": forms,
            "pending_chunks": [str(row.get("chunk_id") or "") for row in pending_rows],
            "matched_chunks": matched_rows,
            "uncovered_chunks": uncovered,
            "status": "complete" if not uncovered else "partial",
        })
    complete = sum(1 for contract in contracts if contract["status"] == "complete")
    partial = len(contracts) - complete
    covered_rows = sum(len(contract.get("matched_chunks") or [])
                       for contract in contracts)
    result.update({
        "contracts": contracts,
        "counts": {
            "outstanding_rows": sum(len(rows) for rows in grouped.values()),
            "groups": len(grouped),
            "contracts": len(contracts),
            "complete_contracts": complete,
            "partial_contracts": partial,
            "covered_rows": covered_rows,
            "uncovered_rows": sum(len(contract.get("uncovered_chunks") or [])
                                  for contract in contracts),
        },
        "ok": not result["errors"],
    })
    _atomic_json(jpath, result)
    _atomic_text(mpath, _render_markdown(result))
    return result


def apply_term_form_contracts(
    project_root: Path,
    *,
    contracts_path: Optional[Path] = None,
    run_id: str = "",
) -> Dict[str, Any]:
    root = Path(project_root)
    cpath = (Path(contracts_path) if contracts_path
             else root / DEFAULT_CONTRACT_PATH)
    result = build_term_form_contracts(root, contracts_path=cpath)
    result["mode"] = "apply"
    result["write_attempted"] = True
    result["run_id"] = run_id
    if not result.get("ok"):
        result["applied"] = False
        return result
    existing_contracts: List[dict] = []
    if cpath.exists():
        existing = _read_json(cpath)
        if int(existing.get("schema_version") or 0) == TERM_CONTRACT_SCHEMA_VERSION:
            existing_contracts = [
                item for item in existing.get("contracts") or []
                if isinstance(item, dict)
            ]
    merged: Dict[Tuple[int, str, str], dict] = {}
    for contract in existing_contracts:
        try:
            decision_id = int(contract.get("decision_id") or 0)
        except (TypeError, ValueError):
            continue
        source_key = str(contract.get("source_key") or "")
        translation = str(contract.get("decision_translation") or "")
        if decision_id and source_key and translation:
            merged[(decision_id, _phrase_key(source_key),
                    _phrase_key(translation))] = contract
    for contract in result.get("contracts") or []:
        try:
            decision_id = int(contract.get("decision_id") or 0)
        except (TypeError, ValueError):
            continue
        source_key = str(contract.get("source_key") or "")
        translation = str(contract.get("decision_translation") or "")
        if decision_id and source_key and translation:
            merged[(decision_id, _phrase_key(source_key),
                    _phrase_key(translation))] = contract
    payload = {
        "schema_version": TERM_CONTRACT_SCHEMA_VERSION,
        "generated_at": result.get("generated_at"),
        "generation": {
            "module": "src.term_form_contracts",
            "run_id": run_id,
            "profile_path": result.get("profile_path"),
        },
        "contracts": list(merged.values()),
    }
    backup = ""
    if cpath.exists():
        bpath = cpath.with_name(cpath.name + ".bak")
        shutil.copy2(str(cpath), str(bpath))
        backup = str(bpath)
    _atomic_json(cpath, payload)
    result.update({"applied": True, "artifact_path": str(cpath),
                   "backup_path": backup})
    _atomic_json(Path(result["json_path"]), result)
    _atomic_text(Path(result["md_path"]), _render_markdown(result))
    return result


def _render_markdown(result: Dict[str, Any]) -> str:
    lines = ["# 术语形态合同（证据约束）", "",
             "- 模式: **%s**" % result.get("mode"),
             "- 写入尝试: `%s`" % result.get("write_attempted"),
             "- 生成时间: %s" % result.get("generated_at"),
             "- 统计: `%s`" % (result.get("counts") or {}), ""]
    if result.get("artifact_path"):
        lines.append("- 合同制品: `%s`" % result["artifact_path"])
        lines.append("- 备份: `%s`" % (result.get("backup_path") or ""))
        lines.append("")
    contracts = result.get("contracts") or []
    if contracts:
        lines.extend(["## 合同（前 40 组）"])
        for contract in contracts[:40]:
            lines.append(
                "- `#%s %s → %s` / %s / covered=%s uncovered=%s"
                % (contract.get("decision_id"), contract.get("source_key"),
                   contract.get("decision_translation"), contract.get("status"),
                   len(contract.get("matched_chunks") or []),
                   len(contract.get("uncovered_chunks") or [])))
            lines.append("  - 合法形态: " + ", ".join(
                "`%s`(%s)" % (item.get("form"),
                              ",".join(item.get("authorities") or []))
                for item in contract.get("forms") or []))
            if contract.get("uncovered_chunks"):
                lines.append("  - 仍待修: " + ", ".join(
                    "`%s`" % chunk for chunk in contract["uncovered_chunks"]))
    if result.get("errors"):
        lines.extend(["", "## 错误", ""] + [
            "- `%s`" % error for error in result["errors"]])
    lines.extend(["", "本制品只声明有权威证据的合法形态；不修改正文、Decision、registry、pending 或 workflow。", ""])
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "TERM_CONTRACT_SCHEMA_VERSION",
    "DEFAULT_CONTRACT_PATH",
    "build_term_form_contracts",
    "apply_term_form_contracts",
]
