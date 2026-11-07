"""Deterministic + validated narrative graph for a book (Round 131).

Inspired by NVIDIA Auto Ontology's useful separation, not its stack:

* deterministic evidence first: ``entity_registry`` seeds entity/chapter and
  co-occurrence facts;
* LLM output is only a residual layer: validated narrative notes add
  setup/payoff and arc relationships;
* every edge carries provenance and coverage is explicit;
* the graph is never a second terminology source of truth.

Artifact: ``input/story_graph.json``.  The graph can be rebuilt without an LLM
from the decision DB and the current ``book_profile.json``.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.judgment_layer import (
    active_notes,
    all_notes,
    atomic_write_json,
    chapter_ids_between,
    sha256_text,
)

GRAPH_VERSION = 4
DEFAULT_GRAPH_PATH = "input/story_graph.json"


def _story_graph_config() -> Dict[str, Any]:
    """Read optional graph settings without making config a hard dependency."""
    try:
        from src.config import get_config
        cfg = get_config().get("story_graph", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}
    except Exception:
        return {}

_GENERIC_EN = {
    "the", "and", "for", "with", "from", "that", "this", "they", "them",
    "their", "there", "then", "than", "have", "has", "had", "were", "was",
    "will", "would", "could", "should", "what", "when", "where", "which",
    "who", "whom", "why", "how", "his", "her", "him", "she", "you", "your",
    "our", "ours", "its", "it's", "not", "but", "are", "is", "be", "been",
    "being", "one", "two", "all", "any", "some", "into", "about", "after",
    "before", "over", "under", "again", "also", "more", "most", "very",
}


def _load_json_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, dict):
        out = []
        for v in raw.values():
            out.extend(_load_json_list(v))
        return out
    s = str(raw).strip()
    if not s:
        return []
    if s[:1] in "[{":
        try:
            return _load_json_list(json.loads(s))
        except Exception:
            pass
    for sep in ("\n", ";", "；", "|", "、"):
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def _rows_fingerprint(rows: Sequence[Dict[str, Any]]) -> str:
    """Content hash of the rows the graph actually consumed.

    The database path alone is not a version: new entities can be added to the
    same file, and a rebuilt graph must change its fingerprint when that
    happens.  Hashing the selected registry rows keeps the check stable across
    unrelated SQLite metadata/page changes.
    """
    canonical = json.dumps(list(rows or []), ensure_ascii=False,
                           sort_keys=True, default=str, separators=(",", ":"))
    return sha256_text(canonical)


def _registry_rows(db_path: Path) -> List[Dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(entity_registry)")}
        if not cols:
            return []
        wanted = ["id", "canonical_name", "source_names", "variant_names", "entity_type",
                  "pronoun_gender", "confidence", "pending_review", "translation_strategy",
                  "first_seen_chunk"]
        select = ", ".join(c for c in wanted if c in cols)
        rows = con.execute(f"SELECT {select} FROM entity_registry ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _entity_aliases(row: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    canonical = str(row.get("canonical_name") or "").strip()
    if canonical:
        names.append(canonical)
    for raw in (row.get("source_names"), row.get("variant_names")):
        names.extend(_load_json_list(raw))
    out: List[str] = []
    seen: set = set()
    for name in names:
        name = str(name).strip()
        if len(name) < 2:
            continue
        key = name.casefold()
        if key in seen:
            continue
        if name.isascii() and key in _GENERIC_EN:
            continue
        seen.add(key)
        out.append(name)
    return out


def _compile_alias_pattern(alias_map: Dict[str, List[str]]) -> Optional[re.Pattern]:
    if not alias_map:
        return None
    tokens = sorted(alias_map, key=lambda s: (-len(s), s))
    parts = []
    for token in tokens:
        escaped = re.escape(token)
        left = r"(?<![A-Za-z0-9])" if token[0].isascii() and token[0].isalnum() else ""
        right = r"(?![A-Za-z0-9])" if token[-1].isascii() and token[-1].isalnum() else ""
        parts.append(left + escaped + right)
    try:
        return re.compile("(?:%s)" % "|".join(parts), re.IGNORECASE)
    except re.error:
        return None


def _chapter_files(input_dir: Path) -> List[Path]:
    return sorted(Path(input_dir).glob("ch*.md"))


def _chapter_texts(input_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in _chapter_files(input_dir):
        try:
            out[p.stem] = p.read_text(encoding="utf-8")
        except OSError:
            continue
    return out


def _first_quote(text: str, start: int, end: int, radius: int = 40) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b].replace("\n", " ").strip()[:220]


def _node_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{value}"


def _note_certification(note: Dict[str, Any], validated: bool) -> str:
    """Derive one certification value for a note and every edge it owns."""
    if str(note.get("status") or "") == "dropped":
        return "dropped"
    return "l1_l2_verified" if validated else "pending"


def _entity_confidence(row: Dict[str, Any]) -> Tuple[float, str]:
    """Normalize the registry's text confidence without losing its label.

    ``entity_registry.confidence`` is a TEXT column with the project-wide
    values ``high|medium|low`` (see DecisionEngine.create_tables), not a
    float.  The first story-graph implementation parsed it with ``float()``,
    which made every production entity pending and left the chapter context
    empty even though 286 entities were trusted by the rest of the project.
    Keep numeric artifacts working too, and retain the original label for
    auditability.
    """
    raw = row.get("confidence")
    if isinstance(raw, bool):
        return (1.0 if raw else 0.0), ""
    if isinstance(raw, (int, float)):
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(1.0, score)), ""
    label = str(raw or "").strip().casefold()
    if label in ("high", "medium", "low"):
        return {"high": 0.9, "medium": 0.6, "low": 0.3}[label], label
    try:
        score = float(label)
    except (TypeError, ValueError):
        return 0.0, label
    return max(0.0, min(1.0, score)), ""


def _entity_is_certified(confidence: float, label: str, pending: bool) -> bool:
    """Match the project's existing trusted-entity rule.

    ``query_high_confidence_entities`` and consistency checks treat only
    ``confidence='high' AND pending_review=0`` as trusted.  Numeric test
    fixtures use a >=0.8 threshold; medium/low labels stay pending even when
    their numeric score would otherwise clear the threshold.
    """
    if pending:
        return False
    if label:
        return label == "high"
    return confidence >= 0.8


def _structural_certification(*values: str) -> str:
    """Roll endpoint facts up into one certification for a deterministic edge.

    A surface occurrence is mechanical, but it is not an independently verified
    identity.  If either endpoint is still pending, the edge is pending too.
    This mirrors Auto Ontology's single certification rollup rather than letting
    each consumer infer trust from a different field.
    """
    return "auto_verified" if values and all(
        str(value or "") == "auto_verified" for value in values
    ) else "pending"


def _note_evidence_provenance(note: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep source quotes and offsets on narrative edges, not just the claim."""
    out: List[Dict[str, Any]] = []
    for ev in note.get("evidence") or []:
        if not isinstance(ev, dict):
            continue
        out.append({
            "chapter": ev.get("chapter"),
            "offset": ev.get("offset"),
            "quote": ev.get("quote"),
            "matched_quote": ev.get("matched_quote"),
            "match_kind": ev.get("match_kind"),
            "match_score": ev.get("match_score"),
            "unique": ev.get("unique"),
        })
    return out


def _knowledge_gap(code: str, message: str, items: Sequence[Any],
                   sample_key: str = "", sample_limit: int = 6) -> Dict[str, Any]:
    """One actionable gap row; lists stay in coverage/diagnostics, samples here."""
    sample: List[str] = []
    for item in list(items)[:max(0, int(sample_limit))]:
        if isinstance(item, dict):
            value = item.get(sample_key) if sample_key else ""
        else:
            value = item
        if value not in (None, ""):
            sample.append(str(value))
    return {"code": code, "count": len(items), "sample": sample,
            "message": message}


def build_story_graph(
    project_root: Path = Path("."),
    profile: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    cooccur_window: int = 600,
    max_cooccur_edges: int = 600,
    entity_coverage_min: float = 0.5,
) -> Dict[str, Any]:
    root = Path(project_root)
    dbp = Path(db_path) if db_path else root / "db" / "decision_db.sqlite"
    indir = Path(input_dir) if input_dir else root / "input"
    texts = _chapter_texts(indir)
    rows = _registry_rows(dbp)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    for cid, text in texts.items():
        nodes.append({
            "id": _node_id("chapter", cid),
            "type": "chapter",
            "label": cid,
            "attributes": {"chars": len(text), "hash": sha256_text(text)},
            "provenance": [{"artifact": "input", "path": f"input/{cid}.md"}],
        })

    alias_map: Dict[str, List[str]] = {}
    entity_by_id: Dict[str, Dict[str, Any]] = {}
    aliases_by_entity: Dict[str, List[str]] = {}
    entity_certification: Dict[str, str] = {}
    for row in rows:
        eid = f"entity-{row.get('id')}"
        canonical = str(row.get("canonical_name") or "").strip()
        if not canonical:
            continue
        aliases = _entity_aliases(row)
        aliases_by_entity[eid] = aliases
        entity_by_id[eid] = row
        conf, conf_label = _entity_confidence(row)
        pending = bool(row.get("pending_review"))
        certification = "auto_verified" if _entity_is_certified(
            conf, conf_label, pending) else "pending"
        entity_certification[eid] = certification
        nodes.append({
            "id": _node_id("entity", eid),
            "type": "entity",
            "label": canonical,
            "attributes": {
                "canonical_name": canonical,
                "aliases": aliases,
                "entity_type": row.get("entity_type") or "",
                "pronoun_gender": row.get("pronoun_gender") or "",
                "confidence": conf,
                "confidence_label": conf_label,
                "raw_confidence": row.get("confidence"),
                "pending_review": pending,
                "certification": certification,
                "translation_strategy": row.get("translation_strategy") or "",
                "first_seen_chunk": row.get("first_seen_chunk") or "",
            },
            "provenance": [{"artifact": "decision_db", "table": "entity_registry",
                            "row_id": row.get("id")}],
        })
        for alias in aliases:
            if alias == canonical:
                continue
            aid = "alias-" + hashlib.sha1(
                (eid + "\x00" + alias).encode("utf-8")).hexdigest()[:16]
            nodes.append({
                "id": _node_id("alias", aid),
                "type": "alias",
                "label": alias,
                "attributes": {"canonical_node": _node_id("entity", eid)},
                "provenance": [{"artifact": "decision_db", "table": "entity_registry",
                                "row_id": row.get("id"), "field": "source_names/variant_names"}],
            })
            edges.append({
                "id": f"edge:alias_of:{aid}",
                "type": "alias_of",
                "source": _node_id("alias", aid),
                "target": _node_id("entity", eid),
                "count": 1,
                "provenance": [{"artifact": "decision_db", "row_id": row.get("id")}],
                "certification": certification,
            })
        for alias in aliases:
            alias_map.setdefault(alias.casefold(), []).append(eid)

    # Never average two entities that merely share a surface alias.  A unique
    # canonical owner wins (the decision DB is authoritative); otherwise the
    # alias stays explicitly unresolved and contributes no occurrence edges.
    canonical_owners: Dict[str, List[str]] = {}
    for eid, row in entity_by_id.items():
        canon = str(row.get("canonical_name") or "").strip().casefold()
        if canon:
            canonical_owners.setdefault(canon, []).append(eid)
    unresolved_aliases: List[Dict[str, Any]] = []
    resolved_by_canonical: List[Dict[str, Any]] = []
    for alias_key, eids in list(alias_map.items()):
        unique_eids = sorted(set(eids))
        if len(unique_eids) <= 1:
            alias_map[alias_key] = unique_eids
            continue
        owners = canonical_owners.get(alias_key) or []
        if len(owners) == 1 and owners[0] in unique_eids:
            alias_map[alias_key] = [owners[0]]
            resolved_by_canonical.append({
                "alias": alias_key, "entities": unique_eids,
                "selected": owners[0], "reason": "unique_canonical_owner",
            })
            continue
        alias_map.pop(alias_key, None)
        unresolved_aliases.append({
            "alias": alias_key, "entities": unique_eids,
            "reason": "alias_collision_without_unique_canonical_owner",
        })
    unresolved_keys = {str(x.get("alias") or "") for x in unresolved_aliases}
    # Certification is inherited from the entity, not copied blindly.  An alias
    # collision that the decision DB resolves in favour of one canonical owner
    # must leave the losing alias_of edges pending too; otherwise the graph still
    # presents the old owner as verified even though its mapping was rejected.
    node_by_id = {str(n.get("id")): n for n in nodes}
    for edge in edges:
        if edge.get("type") != "alias_of":
            continue
        alias_node = node_by_id.get(str(edge.get("source")))
        if not alias_node:
            continue
        alias_key = str(alias_node.get("label") or "").casefold()
        owners = alias_map.get(alias_key) or []
        selected_owner = _node_id("entity", owners[0]) if len(owners) == 1 else ""
        target = str(edge.get("target") or "")
        target_entity = target.split(":", 1)[1] if target.startswith("entity:") else target
        edge["certification"] = entity_certification.get(target_entity, "pending")
        if alias_key in unresolved_keys:
            edge["certification"] = "pending"
            edge["ambiguous"] = True
        elif selected_owner and target != selected_owner:
            edge["certification"] = "pending"
            edge["ambiguous"] = True
            edge["collision_owner"] = selected_owner

    pattern = _compile_alias_pattern(alias_map)
    occurrences: Dict[str, List[Dict[str, Any]]] = {eid: [] for eid in entity_by_id}
    chapter_occurrences: Dict[str, List[Tuple[int, int, str, str]]] = {}
    if pattern:
        for cid, text in texts.items():
            co = chapter_occurrences.setdefault(cid, [])
            for m in pattern.finditer(text):
                matched = m.group(0)
                eids = alias_map.get(matched.casefold()) or []
                for eid in eids:
                    occurrences[eid].append({
                        "chapter": cid, "start": m.start(), "end": m.end(),
                        "surface": matched,
                    })
                    co.append((m.start(), m.end(), eid, matched))

    appeared = 0
    for eid, occs in occurrences.items():
        if not occs:
            continue
        appeared += 1
        per_ch: Dict[str, List[Dict[str, Any]]] = {}
        for o in occs:
            per_ch.setdefault(o["chapter"], []).append(o)
        for cid, items in sorted(per_ch.items()):
            first = min(items, key=lambda x: x["start"])
            src = texts.get(cid, "")
            edges.append({
                "id": f"edge:appears_in:{eid}:{cid}",
                "type": "appears_in",
                "source": _node_id("entity", eid),
                "target": _node_id("chapter", cid),
                "count": len(items),
                "provenance": [{
                    "chapter": cid,
                    "offset": first["start"],
                    "quote": _first_quote(src, first["start"], first["end"]),
                    "surface": first["surface"],
                }],
                "certification": entity_certification.get(eid, "pending"),
                "entity_certification": entity_certification.get(eid, "pending"),
            })

    pair_counts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for cid, co in chapter_occurrences.items():
        co.sort(key=lambda x: (x[0], x[1], x[2]))
        # Deduplicate repeated mentions of the same entity at one offset.
        uniq: List[Tuple[int, int, str, str]] = []
        seen_occ: set = set()
        for item in co:
            key = (item[0], item[1], item[2])
            if key not in seen_occ:
                seen_occ.add(key)
                uniq.append(item)
        for i, (s1, e1, a, _) in enumerate(uniq):
            for s2, e2, b, _ in uniq[i + 1:i + 21]:
                if s2 - e1 > cooccur_window:
                    break
                if a == b:
                    continue
                lo, hi = (a, b) if a < b else (b, a)
                key = (lo, hi)
                rec = pair_counts.get(key)
                if rec is None:
                    rec = {"count": 0, "first": (cid, s1, e2), "chapters": set()}
                    pair_counts[key] = rec
                    if len(pair_counts) > max_cooccur_edges * 8:
                        # Keep memory bounded; final pruning below is deterministic.
                        pass
                rec["count"] += 1
                rec["chapters"].add(cid)
                if (cid, s1) < (rec["first"][0], rec["first"][1]):
                    rec["first"] = (cid, s1, e2)
    ranked_pairs = sorted(pair_counts.items(),
                          key=lambda kv: (-kv[1]["count"], kv[0][0], kv[0][1]))
    for (a, b), rec in ranked_pairs[:max(0, int(max_cooccur_edges))]:
        cid, s, e = rec["first"]
        src = texts.get(cid, "")
        edge_certification = _structural_certification(
            entity_certification.get(a, "pending"),
            entity_certification.get(b, "pending"),
        )
        edges.append({
            "id": f"edge:co_occurs_with:{a}:{b}",
            "type": "co_occurs_with",
            "source": _node_id("entity", a),
            "target": _node_id("entity", b),
            "count": rec["count"],
            "chapters": sorted(rec["chapters"]),
            "provenance": [{"chapter": cid, "offset": s, "window": cooccur_window,
                            "quote": _first_quote(src, s, e)}],
            "certification": edge_certification,
            "entity_certifications": [
                {"entity": _node_id("entity", a),
                 "certification": entity_certification.get(a, "pending")},
                {"entity": _node_id("entity", b),
                 "certification": entity_certification.get(b, "pending")},
            ],
        })

    profile_notes = all_notes(profile)
    active_ids = {str(n.get("id")) for n in active_notes(profile)}
    for n in profile_notes:
        nid = str(n.get("id") or "")
        if not nid:
            continue
        note_node = _node_id("note", nid)
        validated = nid in active_ids
        certification = _note_certification(n, validated)
        match_kinds = []
        for ev in n.get("evidence") or []:
            kind = str(ev.get("match_kind") or "")
            if kind and kind not in match_kinds:
                match_kinds.append(kind)
        nodes.append({
            "id": note_node,
            "type": "judgment_note",
            "label": str(n.get("claim") or "")[:120],
            "attributes": {
                "note_type": n.get("type"),
                "chapters": n.get("chapters") or [],
                "confidence": n.get("confidence"),
                "status": n.get("status"),
                "claim": n.get("claim"),
                "validated": validated,
                "certification": certification,
                "l1_match_kinds": match_kinds,
                "l2_verdict": n.get("l2_verdict", ""),
            },
            "certification": certification,
            "provenance": [{"artifact": "book_profile", "note_id": nid,
                            "source_model": n.get("source_model", ""),
                            "created_at": n.get("created_at", "")}],
        })
        for ev in n.get("evidence") or []:
            cid = str(ev.get("chapter") or "")
            if cid not in texts:
                continue
            edges.append({
                "id": f"edge:evidenced_in:{nid}:{cid}",
                "type": "evidenced_in",
                "source": note_node,
                "target": _node_id("chapter", cid),
                "count": 1,
                "provenance": [{"chapter": cid, "offset": ev.get("offset"),
                                "quote": ev.get("quote")}],
                # ``auto_verified`` is reserved for deterministic extraction
                # (registry aliases and chapter occurrences).  A narrative
                # note is only verified after the L1 quote match and L2 model
                # review, so keep that provenance visible in the artifact.
                "certification": certification,
            })
        if n.get("type") == "setup_payoff" and validated:
            ev_chs = []
            for ev in n.get("evidence") or []:
                c = str(ev.get("chapter") or "")
                if c in texts and c not in ev_chs:
                    ev_chs.append(c)
            if len(ev_chs) >= 2:
                ev_chs.sort()
                edges.append({
                    "id": f"edge:sets_up:{nid}",
                    "type": "sets_up",
                    "source": _node_id("chapter", ev_chs[0]),
                    "target": _node_id("chapter", ev_chs[-1]),
                    "count": 1,
                    "note_id": nid,
                    "provenance": [{"artifact": "book_profile", "note_id": nid,
                                    "source_model": n.get("source_model", ""),
                                    "created_at": n.get("created_at", ""),
                                    "evidence": _note_evidence_provenance(n)}],
                    "certification": certification,
                })
        elif n.get("type") == "arc" and validated:
            span = chapter_ids_between(n.get("chapters") or [])
            if len(span) >= 2 and span[0] in texts and span[-1] in texts:
                edges.append({
                    "id": f"edge:evolves_from:{nid}",
                    "type": "evolves_from",
                    "source": _node_id("chapter", span[-1]),
                    "target": _node_id("chapter", span[0]),
                    "count": 1,
                    "note_id": nid,
                    "provenance": [{"artifact": "book_profile", "note_id": nid,
                                    "source_model": n.get("source_model", ""),
                                    "created_at": n.get("created_at", ""),
                                    "evidence": _note_evidence_provenance(n)}],
                    "certification": certification,
                })

    total_entities = len(entity_by_id)
    entity_coverage = (appeared / total_entities) if total_entities else 0.0
    certified_entity_ids = {
        eid for eid, value in entity_certification.items()
        if value == "auto_verified"
    }
    appeared_certified = sum(
        1 for eid, occs in occurrences.items()
        if occs and eid in certified_entity_ids
    )
    certified_entity_coverage = (
        appeared_certified / total_entities if total_entities else 0.0
    )
    certified_entity_coverage_of_trusted = (
        appeared_certified / len(certified_entity_ids)
        if certified_entity_ids else 0.0
    )
    chapters_with_entities = sorted({
        str(e["target"]).split(":", 1)[-1]
        for e in edges if e["type"] == "appears_in"
    })
    uncovered_chapters = sorted(set(texts) - set(chapters_with_entities))
    chapter_coverage = (len(chapters_with_entities) / len(texts)) if texts else 0.0
    entity_labels = {
        eid: str((row or {}).get("canonical_name") or eid)
        for eid, row in entity_by_id.items()
    }
    uncovered_entity_ids = sorted(
        eid for eid, occs in occurrences.items() if not occs
    )
    uncovered_entities = [entity_labels.get(eid, eid) for eid in uncovered_entity_ids]
    uncertified_entity_ids = sorted(
        eid for eid in entity_by_id if eid not in certified_entity_ids
    )
    uncertified_entities = [
        entity_labels.get(eid, eid) for eid in uncertified_entity_ids
    ]
    active_narrative = [n for n in profile_notes if str(n.get("id")) in active_ids]
    unpaired_setup = [
        n for n in profile_notes
        if n.get("type") == "setup_payoff"
        and ((n.get("meta") or {}).get("pairing_status") == "unpaired"
             or n.get("pairing_status") == "unpaired")
    ]
    inactive_notes = [
        n for n in profile_notes
        if str(n.get("id")) not in active_ids
    ]
    narrative_cov = (len(active_narrative) / len(profile_notes)) if profile_notes else 0.0
    narrative_chapters = set()
    for note in active_narrative:
        for cid in chapter_ids_between(note.get("chapters") or []):
            if cid in texts:
                narrative_chapters.add(cid)
    uncovered_narrative_chapters = sorted(set(texts) - narrative_chapters)
    narrative_chapter_coverage = (
        len(narrative_chapters) / len(texts) if texts else 0.0
    )
    chapter_entity_ids: Dict[str, set] = {cid: set() for cid in texts}
    for eid, occs in occurrences.items():
        for occ in occs:
            cid = str(occ.get("chapter") or "")
            if cid in chapter_entity_ids:
                chapter_entity_ids[cid].add(eid)
    chapter_entity_coverage = {
        cid: round(len(eids) / total_entities, 4) if total_entities else 0.0
        for cid, eids in sorted(chapter_entity_ids.items())
    }
    chapter_certified_entity_coverage = {
        cid: round(len(eids & certified_entity_ids) / total_entities, 4)
        if total_entities else 0.0
        for cid, eids in sorted(chapter_entity_ids.items())
    }
    narrative_answerable_reasons: List[str] = []
    if not profile_notes:
        narrative_answerable_reasons.append("no_narrative_notes")
    elif not active_narrative:
        narrative_answerable_reasons.append("no_validated_active_notes")
    narrative_answerable = not narrative_answerable_reasons
    narrative_partial_coverage = bool(
        narrative_answerable
        and (narrative_cov < 1.0 or narrative_chapter_coverage < 1.0)
    )
    answerable_reasons: List[str] = []
    if not texts:
        answerable_reasons.append("no_chapters")
    if total_entities == 0:
        answerable_reasons.append("no_registered_entities")
    if appeared_certified <= 0:
        answerable_reasons.append("no_certified_entity_occurrences")
    # Coverage is an index over the entities we actually trust.  The registry
    # also contains pending candidates and unresolved duplicate rows; letting
    # those dilute the denominator would make a healthy certified index
    # unanswerable forever.  Keep all-row coverage as a diagnostic instead.
    if certified_entity_coverage_of_trusted < float(entity_coverage_min):
        answerable_reasons.append("trusted_entity_coverage_below_threshold")
    answerable = not answerable_reasons
    knowledge_gaps = [
        _knowledge_gap(
            "uncertified_entities",
            "实体未达到 high+pending_review=0，图谱不把它当作认证事实",
            uncertified_entities,
        ),
        _knowledge_gap(
            "uncovered_entities",
            "实体已登记但未在已切分章节中找到任何出现",
            uncovered_entities,
        ),
        _knowledge_gap(
            "uncovered_chapters",
            "章节没有任何已登记实体出现",
            uncovered_chapters,
        ),
        _knowledge_gap(
            "unresolved_aliases",
            "别名存在多 owner 且无法由唯一 canonical owner 裁决；图谱不猜归属",
            unresolved_aliases,
            sample_key="alias",
        ),
        _knowledge_gap(
            "unpaired_setup_payoff",
            "伏笔候选没有找到可机械组装的回收侧候选，保留 dormant",
            unpaired_setup,
            sample_key="id",
        ),
        _knowledge_gap(
            "inactive_judgment_notes",
            "注记未通过 L1/L2 或已 dropped，运行时不会注入",
            inactive_notes,
            sample_key="id",
        ),
    ]
    fingerprint_inputs = {
        "chapters": {cid: sha256_text(t) for cid, t in sorted(texts.items())},
        "decision_db": str(dbp),
        "decision_db_hash": _rows_fingerprint(rows),
        "entity_registry_rows": len(rows),
        "profile_notes_hash": sha256_text(json.dumps(profile_notes, ensure_ascii=False,
                                                     sort_keys=True)),
    }
    fingerprint_params = {
        "cooccur_window": cooccur_window,
        "max_cooccur_edges": max_cooccur_edges,
        "entity_coverage_min": entity_coverage_min,
    }
    fingerprint = sha256_text(json.dumps(
        {"graph_version": GRAPH_VERSION, "inputs": fingerprint_inputs,
         "params": fingerprint_params},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    generation_payload = {
        "graph_version": GRAPH_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint": fingerprint,
        "inputs": fingerprint_inputs,
        "params": fingerprint_params,
    }
    return {
        "schema_version": GRAPH_VERSION,
        "generation": generation_payload,
        "fingerprint": fingerprint,
        "coverage": {
            "entities_total": total_entities,
            "entities_with_appearance": appeared,
            "entity_coverage": round(entity_coverage, 4),
            "certified_entities_total": len(certified_entity_ids),
            "certified_entities_with_appearance": appeared_certified,
            "certified_entity_coverage": round(certified_entity_coverage, 4),
            "certified_entity_coverage_of_trusted": round(
                certified_entity_coverage_of_trusted, 4
            ),
            "answerable_threshold_basis": "certified_entity_coverage_of_trusted",
            "uncovered_entities": uncovered_entities,
            "uncovered_entity_ids": uncovered_entity_ids,
            "uncertified_entities": uncertified_entities,
            "uncertified_entity_ids": uncertified_entity_ids,
            "chapters_total": len(texts),
            "chapters_with_entities": len(chapters_with_entities),
            "chapter_coverage": round(chapter_coverage, 4),
            "uncovered_chapters": uncovered_chapters,
            "entities_per_chapter": {
                cid: len(eids) for cid, eids in sorted(chapter_entity_ids.items())
            },
            "chapter_entity_coverage": chapter_entity_coverage,
            "chapter_certified_entity_coverage": chapter_certified_entity_coverage,
            "narrative_notes_total": len(profile_notes),
            "narrative_notes_active": len(active_narrative),
            "narrative_note_coverage": round(narrative_cov, 4),
            "narrative_chapters_total": len(texts),
            "narrative_chapters_with_active_notes": len(narrative_chapters),
            "narrative_chapter_coverage": round(narrative_chapter_coverage, 4),
            "uncovered_narrative_chapters": uncovered_narrative_chapters,
            "narrative_answerable_for": "validated_note_index",
            "narrative_answerable": narrative_answerable,
            "narrative_answerable_reasons": narrative_answerable_reasons,
            "narrative_partial_coverage": narrative_partial_coverage,
            "narration_source": "decision_db+validated_book_profile",
            "answerable_for": "entity_index",
            "answerable": answerable,
            "answerable_reasons": answerable_reasons,
            # A graph can be answerable for the chapters it saw yet still be
            # incomplete for the book.  Report either kind of hole instead of
            # equating partial coverage with missing entities only.
            "partial_coverage": bool(answerable and (
                uncovered_entities or uncovered_chapters or uncertified_entities
            )),
            "threshold": float(entity_coverage_min),
        },
        "knowledge_gaps": knowledge_gaps,
        "nodes": nodes,
        "edges": edges,
        "diagnostics": {
            "alias_tokens": len(alias_map),
            "entity_nodes": total_entities,
            "chapter_nodes": len(texts),
            "cooccurrence_pairs_kept": min(len(ranked_pairs), max(0, int(max_cooccur_edges))),
            "unresolved_alias_count": len(unresolved_aliases),
            "unresolved_aliases": unresolved_aliases,
            "collisions_resolved_by_canonical": resolved_by_canonical,
            "certified_entity_count": sum(
                1 for value in entity_certification.values() if value == "auto_verified"
            ),
            "pending_entity_count": sum(
                1 for value in entity_certification.values() if value != "auto_verified"
            ),
            "decision_db_hash": fingerprint_inputs["decision_db_hash"],
        },
    }


def save_story_graph(graph: Dict[str, Any], path: Optional[Path] = None,
                     project_root: Path = Path(".")) -> Path:
    out = Path(path) if path else Path(project_root) / DEFAULT_GRAPH_PATH
    atomic_write_json(out, graph)
    return out


def load_story_graph(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        return None
    return data


def build_and_save_story_graph(project_root: Path = Path("."),
                               profile: Optional[Dict[str, Any]] = None,
                               path: Optional[Path] = None,
                               use_config: bool = True,
                               **kwargs: Any) -> Dict[str, Any]:
    root = Path(project_root)
    cfg = _story_graph_config() if use_config else {}
    for key in ("cooccur_window", "max_cooccur_edges", "entity_coverage_min"):
        if key in cfg and key not in kwargs:
            kwargs[key] = cfg[key]
    graph = build_story_graph(root, profile=profile, **kwargs)
    rel = str(cfg.get("path") or DEFAULT_GRAPH_PATH)
    out = Path(path) if path else root / rel
    save_story_graph(graph, path=out, project_root=root)
    return graph


def story_graph_context_for_chapter(graph: Optional[Dict[str, Any]], chapter_id: str,
                                    max_entities: int = 12,
                                    max_chars: int = 1200) -> str:
    """Return top deterministic entities for a chapter, gated by coverage.

    This is used by global review/aids, not as a terminology hard constraint.
    """
    if not isinstance(graph, dict):
        return ""
    cov = graph.get("coverage") or {}
    if not cov.get("answerable"):
        return ""
    # ``chapters_with_entities`` is a count.  Use the per-chapter maps for the
    # actual gate; keep a fallback for older graph artifacts.
    chapter_counts = cov.get("chapter_certified_entity_coverage")
    if not isinstance(chapter_counts, dict) or not chapter_counts:
        chapter_counts = cov.get("chapter_entity_coverage") or \
            cov.get("entities_per_chapter") or {}
    if isinstance(chapter_counts, dict) and chapter_id in chapter_counts:
        try:
            if float(chapter_counts.get(chapter_id) or 0) <= 0:
                return ""
        except (TypeError, ValueError):
            return ""
    if chapter_id in set(str(x) for x in (cov.get("uncovered_chapters") or [])):
        return ""
    target = _node_id("chapter", chapter_id)
    rows = [e for e in graph.get("edges") or []
            if e.get("type") == "appears_in" and e.get("target") == target
            and e.get("certification") == "auto_verified"]
    rows.sort(key=lambda e: (-int(e.get("count") or 0), str(e.get("source"))))
    entity_labels: Dict[str, str] = {}
    for n in graph.get("nodes") or []:
        if n.get("type") == "entity":
            entity_labels[str(n.get("id"))] = str(n.get("label") or "")
    lines = []
    used = 0
    header = "【确定性实体覆盖（仅定位参考；译名以决策库为准）】\n"
    used += len(header)
    for e in rows[:max_entities]:
        label = entity_labels.get(str(e.get("source"))) or str(e.get("source"))
        line = f"- {label}（本章出现 {int(e.get('count') or 0)} 次）\n"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return header + "".join(lines) if lines else ""
