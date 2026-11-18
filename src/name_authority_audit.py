"""Round 158 Phase 1.5a: read-only name authority audit.

The Decision DB is the hard authority for translated names.  The entity
registry is an execution cache: it may carry extra aliases, but it must not
silently contradict an authoritative translation.  This module compares those
two stores without modifying either one.

It deliberately emits evidence, not repairs.  A caller must not use the audit
as permission to rewrite finals or unlock DIRTY chunks.  In particular:

* ``form_align_candidate`` means two spellings are highly similar and one may
  be aligned after a separate, reviewed registry migration;
* ``multi_form_authority`` means one source family legitimately has short,
  titled, and full translated forms; no single global replacement is safe;
* ``alias_quarantine_candidate`` means a registry hard alias disagrees with the
  decision translation and must be moved out of ``source_names`` or quarantined;
* ``cross_person_alias_conflict`` means the same alias currently belongs to
  multiple registry canonicals.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.decision_engine import (
    canonical_is_title_only,
    canonical_tokens,
    classify_entity_source_name,
    name_char_similarity,
    source_name_normalized,
    source_name_proper_tokens,
    _EN_DETERMINERS,
    _EN_GENERIC_MODIFIERS,
    _EN_GROUP_TERM_WORDS,
    _EN_ROLE_WORDS,
    _ZH_GENERIC_MODIFIERS,
    _ZH_ROLE_WORDS,
)
from src.pending_backtrack_reconcile import _name_compatible, _name_key, _phrase_key


FORM_ALIGN_SIMILARITY = 0.78
_NAME_AUTHORITY_ARTICLE_PREFIX_RE = re.compile(
    r"^(?:the|a|an)\s+", re.IGNORECASE)

# Generic group/category vocabulary shared with the registry merger.  These
# words describe a category of entity rather than the entity identity and must
# never bridge two unrelated source families (House A / House B / House Troop)
# or turn a distinct term phrase into an alias of its base name.
_GENERIC_GROUP_ANCHORS = _EN_GROUP_TERM_WORDS


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


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


def _authority_table_exists(con: sqlite3.Connection, name: str) -> bool:
    return bool(con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())


def _json_list(raw: Any) -> List[str]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if isinstance(x, str) and str(x).strip()]


def _decision_source_key(text: str) -> str:
    return source_name_normalized(text) or _phrase_key(text)


def _decision_lookup_keys(text: str) -> List[str]:
    """Keys used to join a registry alias to an authoritative decision.

    Article stripping mirrors ``build_name_unification_maps``: a registry alias
    such as ``the Bene Gesserit Plan`` and a Decision DB source key ``Bene
    Gesserit Plan`` are the same executable source.  Without this join the
    audit can report a clean registry while the name-unification map still
    deletes the distinct term ``Plan`` from prose.
    """
    keys: List[str] = []
    seen = set()
    for key in (_decision_source_key(text), _phrase_key(text)):
        if not key:
            continue
        candidates = (key, _NAME_AUTHORITY_ARTICLE_PREFIX_RE.sub("", key))
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                keys.append(candidate)
    return keys


def _alias_owner_key(text: str) -> str:
    return _NAME_AUTHORITY_ARTICLE_PREFIX_RE.sub(
        "", _decision_source_key(text)).strip()


def _similarity(a: str, b: str) -> float:
    ka, kb = _name_key(a), _name_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    try:
        value = float(name_char_similarity(ka, kb))
        if value > 1.0:
            value = value / 100.0
        if value > 0.0:
            return value
    except Exception:
        pass
    return float(difflib.SequenceMatcher(None, ka, kb, autojunk=False).ratio())


def _decision_rows(con: sqlite3.Connection) -> Tuple[List[dict], Dict[str, List[dict]], str]:
    if not _authority_table_exists(con, "decision_db"):
        return [], {}, "decision_db_missing_table"
    cols = {r[1] for r in con.execute("PRAGMA table_info(decision_db)")}
    required = {"id", "source_key", "translation"}
    missing = sorted(required - cols)
    if missing:
        return [], {}, "decision_db_schema_incompatible:" + ",".join(missing)
    level_expr = "level" if "level" in cols else "NULL AS level"
    lock_expr = "COALESCE(locked, 0) AS locked" if "locked" in cols else "0 AS locked"
    rows = [dict(r) for r in con.execute(
        "SELECT id, source_key, translation, "
        f"{level_expr}, {lock_expr} FROM decision_db ORDER BY id"
    )]
    index: Dict[str, List[dict]] = {}
    for row in rows:
        keys = {
            _decision_source_key(str(row.get("source_key") or "")),
            _phrase_key(str(row.get("source_key") or "")),
        }
        for key in keys:
            if key:
                index.setdefault(key, []).append(row)
    return rows, index, ""


def _authority_registry_rows(con: sqlite3.Connection) -> Tuple[List[dict], str]:
    if not _authority_table_exists(con, "entity_registry"):
        return [], "entity_registry_missing_table"
    cols = {r[1] for r in con.execute("PRAGMA table_info(entity_registry)")}
    required = {"id", "canonical_name", "source_names"}
    missing = sorted(required - cols)
    if missing:
        return [], "entity_registry_schema_incompatible:" + ",".join(missing)

    def col(name: str, default: str) -> str:
        return name if name in cols else f"{default} AS {name}"

    select = [
        "id", "canonical_name", "source_names",
        col("context_names", "'[]'"),
        col("quarantined_names", "'[]'"),
        col("variant_names", "'[]'"),
        col("entity_type", "''"),
        col("confidence", "'low'"),
        col("pending_review", "0"),
    ]
    rows: List[dict] = []
    for row in con.execute(f"SELECT {', '.join(select)} FROM entity_registry ORDER BY id"):
        item = dict(row)
        for field in ("source_names", "context_names", "quarantined_names", "variant_names"):
            item[field] = _json_list(item.get(field))
        rows.append(item)
    return rows, ""


def _decision_for_alias(alias: str, index: Dict[str, List[dict]]) -> List[dict]:
    seen: Dict[int, dict] = {}
    for key in _decision_lookup_keys(alias):
        for row in index.get(key, []):
            seen[int(row["id"])] = row
    return list(seen.values())


def _alias_owner_index(registry: List[dict]) -> Dict[str, set]:
    owners: Dict[str, set] = {}
    for row in registry:
        canonical = str(row.get("canonical_name") or "").strip()
        if not canonical:
            continue
        for alias in row.get("source_names") or []:
            key = _alias_owner_key(alias)
            if key:
                owners.setdefault(key, set()).add(canonical)
    return owners


def _proper_token_groups(decisions: List[dict]) -> List[dict]:
    """Group short/titled/full source forms that share the same proper-name core."""
    items = []
    for row in decisions:
        source = str(row.get("source_key") or "").strip()
        if classify_entity_source_name(source) != "proper":
            continue
        tokens = frozenset(source_name_proper_tokens(source))
        words = tuple(
            word for word in re_find_words(source_name_normalized(source))
            if word in tokens
        )
        if source and words:
            items.append((source, words, row))

    parent: Dict[str, str] = {source: source for source, _tokens, _row in items}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Only a contiguous prefix/short-form relation is evidence of a legitimate
    # title/full-name family.  A shared token in the middle of two names is not
    # enough (e.g. unrelated houses that both contain "Atreides").
    for i in range(len(items)):
        source_a, words_a, _row_a = items[i]
        for j in range(i + 1, len(items)):
            source_b, words_b, _row_b = items[j]
            short, long_ = ((words_a, words_b) if len(words_a) <= len(words_b)
                            else (words_b, words_a))
            if not short or long_[:len(short)] != short:
                continue
            if short[0] in _GENERIC_GROUP_ANCHORS:
                continue
            if any(token in _GENERIC_GROUP_ANCHORS for token in long_[len(short):]):
                continue
            union(source_a, source_b)

    groups: Dict[str, List[Tuple[str, Tuple[str, ...], dict]]] = {}
    for item in items:
        groups.setdefault(find(item[0]), []).append(item)
    out: List[dict] = []
    for members in groups.values():
        sources = sorted({source for source, _tokens, _row in members})
        if len(sources) < 2:
            continue
        rows = []
        seen_ids = set()
        for _source, _tokens, row in members:
            rid = int(row["id"])
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rows.append({
                "id": rid,
                "source_key": str(row.get("source_key") or ""),
                "translation": str(row.get("translation") or ""),
            })
        rows.sort(key=lambda r: (len(r["source_key"]), r["source_key"]))
        translations = [r["translation"] for r in rows if r["translation"]]
        incompatible = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a = rows[i]["translation"]
                b = rows[j]["translation"]
                if a and b and not _name_compatible(a, b):
                    incompatible.append({
                        "left": rows[i]["source_key"],
                        "right": rows[j]["source_key"],
                        "left_translation": a,
                        "right_translation": b,
                    })
        out.append({
            "sources": sources,
            "decisions": rows,
            "forms": list(dict.fromkeys(translations)),
            "incompatible_pairs": incompatible,
            "needs_form_contract": bool(incompatible),
        })
    return sorted(out, key=lambda g: (g["sources"][0], g["sources"]))


def re_find_words(text: str) -> Tuple[str, ...]:
    """Small local word tokenizer for normalized English source keys."""
    return tuple(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _translation_extras(base: str, current: str) -> List[str]:
    """Return inserted CJK content that is not merely a title/role suffix.

    Insertions are inspected separately from replacements so a one-character
    transliteration repair (杰赛 -> 杰瑟) is not mistaken for new meaning.  A
    title such as 夫人, 博士 or 男爵 is harmless in a short/full name form; a
    term such as 计划, 学校 or 护航舰 changes the referent and must not be a
    hard alias of the shorter entity.
    """
    base_key = _name_key(base)
    current_key = _name_key(current)
    if not base_key or not current_key or base_key == current_key:
        return []
    out: List[str] = []
    matcher = difflib.SequenceMatcher(None, base_key, current_key, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "insert":
            continue
        inserted = current_key[j1:j2]
        residual = inserted
        for word in sorted(set(_ZH_ROLE_WORDS) | set(_ZH_GENERIC_MODIFIERS),
                           key=len, reverse=True):
            residual = residual.replace(word, "")
        if any("\u4e00" <= ch <= "\u9fff" for ch in residual):
            out.append(inserted)
    return out


def _source_content_tokens(text: str) -> set:
    """Case-insensitive source tokens that can add semantic content."""
    return {
        token for token in source_name_normalized(text).split()
        if token and token not in _EN_DETERMINERS
        and token not in _EN_ROLE_WORDS
        and token not in _EN_GENERIC_MODIFIERS
    }


def _term_phrase_alias_evidence(alias: str, canonical: str, translation: str,
                                decisions: List[dict]) -> Optional[dict]:
    """Detect a distinct term phrase accidentally owned by a shorter entity.

    The proof is deliberately narrow and mechanical:

    * a shorter Decision key is a contiguous sub-phrase of the alias;
    * that shorter key's translation is exactly the registry canonical;
    * the alias contributes a generic group/term token (``plan``, ``school``,
      ``way`` … rather than a surname or role); and
    * the alias decision adds CJK content beyond title/role words.

    ``Bene Gesserit Plan -> 贝尼·杰瑟里特计划`` meets all four conditions.
    ``Lady Jessica -> 杰西卡夫人`` and ``Dr. Kynes -> 凯恩斯博士`` do not,
    because 夫人/博士 are role forms rather than new semantic content.
    """
    alias_source = source_name_normalized(alias)
    alias_tokens = _source_content_tokens(alias)
    canonical_key = _name_key(canonical)
    if not alias_source or not alias_tokens or not canonical_key:
        return None

    candidates: List[dict] = []
    for decision in decisions:
        base_source = str(decision.get("source_key") or "")
        base_normalized = source_name_normalized(base_source)
        if (not base_normalized or base_normalized == alias_source
                or base_normalized not in alias_source):
            continue
        base_translation = str(decision.get("translation") or "").strip()
        if not base_translation or _name_key(base_translation) != canonical_key:
            continue
        base_tokens = _source_content_tokens(base_source)
        extra_tokens = sorted(alias_tokens - base_tokens)
        if not extra_tokens:
            continue
        if not any(token in _GENERIC_GROUP_ANCHORS for token in extra_tokens):
            continue
        extras = _translation_extras(base_translation, translation)
        if not extras:
            continue
        candidates.append({
            "base_decision_id": int(decision.get("id") or 0),
            "base_source_key": base_source,
            "base_translation": base_translation,
            "extra_source_tokens": extra_tokens,
            "inserted_translation": extras,
        })
    if not candidates:
        return None
    candidates.sort(key=lambda item: (
        -len(str(item.get("base_source_key") or "")), item.get("base_decision_id") or 0))
    return candidates[0]


def _term_phrase_variant_evidence(variant: str, canonical: str,
                                  decisions: List[dict]) -> Optional[dict]:
    """Map a Chinese registry variant back to a flagged term-phrase decision.

    ``variant_names`` is also executable: name unification may replace a
    variant with the canonical.  A variant such as ``贝尼·杰瑟里特学校`` must
    therefore be quarantined just like its English source alias, while a full
    personal name or a title form remains untouched.
    """
    variant_key = _name_key(variant)
    if not variant_key:
        return None
    for decision in decisions:
        translation = str(decision.get("translation") or "").strip()
        if not translation:
            continue
        if (_name_key(translation) != variant_key
                and _similarity(variant, translation) < FORM_ALIGN_SIMILARITY):
            continue
        evidence = _term_phrase_alias_evidence(
            str(decision.get("source_key") or ""), canonical, translation,
            decisions)
        if not evidence:
            continue
        out = dict(evidence)
        out.update({
            "term_decision_id": int(decision.get("id") or 0),
            "term_source_key": str(decision.get("source_key") or ""),
            "variant_translation": translation,
            "variant_similarity": round(_similarity(variant, translation), 4),
        })
        return out
    return None


def _classify_alias_relation(alias: str, canonical: str, translation: str,
                            owners: Dict[str, set],
                            term_evidence: Optional[dict] = None) -> Tuple[str, float]:
    if term_evidence:
        return "term_phrase_alias_quarantine", _similarity(canonical, translation)
    if _name_compatible(canonical, translation):
        return "compatible", _similarity(canonical, translation)
    similarity = _similarity(canonical, translation)
    key = _alias_owner_key(alias)
    if similarity >= FORM_ALIGN_SIMILARITY:
        return "form_align_candidate", similarity
    if len(owners.get(key) or set()) > 1:
        return "cross_person_alias_conflict", similarity
    # A title/relation/description phrase or a decision translation with no
    # proper-name core is contextual.  Otherwise this is still quarantined,
    # but labelled as an authority mismatch rather than inventing a person.
    if classify_entity_source_name(alias) != "proper" or canonical_is_title_only(translation):
        return "context_alias_quarantine", similarity
    return "authority_mismatch_quarantine", similarity


def audit_name_authorities(
    project_root: Path,
    *,
    decision_db: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    json_out: Optional[Path] = None,
    md_out: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare Decision DB names with registry hard aliases; never mutate DBs."""
    root = Path(project_root)
    db_path = Path(decision_db) if decision_db else root / "db" / "decision_db.sqlite"
    out_dir = Path(output_dir) if output_dir else root / "output"
    jpath = Path(json_out) if json_out else out_dir / "name_authority_audit.json"
    mpath = Path(md_out) if md_out else out_dir / "name_authority_audit.md"
    result: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "read_only",
        "generated_at": _now_iso(),
        "write_attempted": False,
        "decision_db": str(db_path),
        "records": [],
        "multi_form_groups": [],
        "counts": {},
        "blocking_count": 0,
        "ok": False,
        "json_path": str(jpath),
        "md_path": str(mpath),
        "errors": [],
    }
    if not db_path.exists():
        result["errors"].append("decision_db_missing")
        _atomic_json(jpath, result)
        _atomic_text(mpath, "# 名称权威审计\n\nDecision DB 不存在；未执行。\n")
        return result

    con = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        decisions, decision_index, decision_error = _decision_rows(con)
        registry, registry_error = _authority_registry_rows(con)
    finally:
        con.close()
    for error in (decision_error, registry_error):
        if error:
            result["errors"].append(error)
    if decision_error or registry_error:
        _atomic_json(jpath, result)
        _atomic_text(mpath, "# 名称权威审计\n\n" + "\n".join(
            f"- 错误: `{e}`" for e in result["errors"]) + "\n")
        return result

    owners = _alias_owner_index(registry)
    records: List[dict] = []
    for row in registry:
        canonical = str(row.get("canonical_name") or "").strip()
        if not canonical:
            continue
        if str(row.get("confidence") or "") not in {"high", "medium"}:
            continue
        for alias in row.get("source_names") or []:
            matches = _decision_for_alias(alias, decision_index)
            if not matches:
                continue
            # One alias can match more than one decision only in legacy DBs;
            # all mappings are retained as evidence.
            for decision in matches:
                translation = str(decision.get("translation") or "").strip()
                if not translation:
                    continue
                term_evidence = _term_phrase_alias_evidence(
                    alias, canonical, translation, decisions)
                relation, similarity = _classify_alias_relation(
                    alias, canonical, translation, owners,
                    term_evidence=term_evidence)
                if relation == "compatible":
                    continue
                action = "align_registry_form"
                if relation in {"context_alias_quarantine",
                                 "authority_mismatch_quarantine",
                                 "term_phrase_alias_quarantine"}:
                    action = "remove_from_source_names_or_quarantine"
                elif relation == "cross_person_alias_conflict":
                    action = "quarantine_alias_until_single_owner"
                record = {
                    "kind": relation,
                    "registry_id": int(row["id"]),
                    "registry_canonical": canonical,
                    "registry_confidence": str(row.get("confidence") or ""),
                    "registry_pending_review": int(row.get("pending_review") or 0),
                    "alias": alias,
                    "alias_class": classify_entity_source_name(alias),
                    "decision_id": int(decision["id"]),
                    "decision_source_key": str(decision.get("source_key") or ""),
                    "decision_translation": translation,
                    "similarity": round(float(similarity), 4),
                    "alias_registry_owners": sorted(owners.get(_alias_owner_key(alias), set())),
                    "suggested_action": action,
                }
                if term_evidence:
                    record["term_phrase_evidence"] = term_evidence
                records.append(record)

        for variant in row.get("variant_names") or []:
            term_evidence = _term_phrase_variant_evidence(
                variant, canonical, decisions)
            if not term_evidence:
                continue
            translation = str(term_evidence.get("variant_translation") or "")
            records.append({
                "kind": "term_phrase_variant_quarantine",
                "registry_id": int(row["id"]),
                "registry_canonical": canonical,
                "registry_confidence": str(row.get("confidence") or ""),
                "registry_pending_review": int(row.get("pending_review") or 0),
                "alias": variant,
                "alias_class": "variant",
                "decision_id": int(term_evidence.get("term_decision_id") or 0),
                "decision_source_key": str(
                    term_evidence.get("term_source_key") or ""),
                "decision_translation": translation,
                "similarity": float(
                    term_evidence.get("variant_similarity") or 0.0),
                "alias_registry_owners": [canonical],
                "suggested_action": "remove_from_variant_names_or_quarantine",
                "term_phrase_evidence": term_evidence,
            })

    multi_form = _proper_token_groups(decisions)
    result["records"] = sorted(
        records,
        key=lambda r: (r["kind"], r["alias"], r["registry_canonical"], r["decision_id"]))
    result["multi_form_groups"] = multi_form
    counts = Counter(r["kind"] for r in records)
    result["counts"] = dict(sorted(counts.items()))
    result["multi_form_count"] = len(multi_form)
    result["multi_form_needing_contract"] = sum(
        1 for group in multi_form if group.get("needs_form_contract"))
    blocking_kinds = {
        "form_align_candidate",
        "cross_person_alias_conflict",
        "context_alias_quarantine",
        "authority_mismatch_quarantine",
        "term_phrase_alias_quarantine",
        "term_phrase_variant_quarantine",
    }
    result["blocking_count"] = sum(
        int(v) for k, v in counts.items() if k in blocking_kinds)
    result["ok"] = result["blocking_count"] == 0 and not result["errors"]
    result["status"] = "clean" if result["ok"] else "needs_alignment"
    _atomic_json(jpath, result)
    _atomic_text(mpath, _authority_markdown(result))
    return result


def _authority_markdown(result: Dict[str, Any]) -> str:
    lines = ["# 名称权威审计（只读）", ""]
    lines.append(f"- 生成时间: {result.get('generated_at')}")
    lines.append("- 模式: **read-only**（Decision DB / entity_registry 均未写入）")
    counts = result.get("counts") or {}
    lines.append("- 硬别名分歧: " + (", ".join(
        f"`{k}`={v}" for k, v in counts.items()) or "无"))
    lines.append(f"- 多重称谓权威组: **{result.get('multi_form_count', 0)}** "
                 f"(需 form contract: {result.get('multi_form_needing_contract', 0)})")
    lines.append(f"- 结论: **{'可进入下一阶段' if result.get('ok') else '需先对齐/隔离'}**")
    if result.get("errors"):
        lines.append("- 错误: " + ", ".join(f"`{e}`" for e in result["errors"]))

    records = result.get("records") or []
    if records:
        lines.append("")
        lines.append(f"## 硬别名分歧（仅列前 40 / 共 {len(records)}）")
        for item in records[:40]:
            lines.append(
                f"- `{item['alias']}` {item['registry_canonical']} ↔ "
                f"{item['decision_translation']} / {item['kind']} / "
                f"sim={item['similarity']:.3f} / action={item['suggested_action']}")
    groups = result.get("multi_form_groups") or []
    if groups:
        lines.append("")
        lines.append(f"## 多重称谓权威（仅列前 30 / 共 {len(groups)}）")
        for group in groups[:30]:
            forms = " | ".join(group.get("forms") or [])
            lines.append(f"- {' + '.join(group.get('sources') or [])}: {forms}")
    lines.append("")
    lines.append("本报告不提供自动修复。若存在阻断项，`name-only-fastpath` 不得进入写闭环。")
    lines.append("")
    return "\n".join(lines)

