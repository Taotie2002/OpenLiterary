"""Read-only retrieval over the Round 142 story graph (Phase B).

The query layer is intentionally not a graph writer.  It turns a question,
claim, or validated narrative note into seed entities and evidence scenes,
expands at most two hops through ``mentions_in`` incidence, scores candidates
with deterministic heuristics, optionally asks a local model for a
necessary-evidence verdict, and returns a trace.

It never creates entities, changes certification, or writes the graph/profile.
When the graph lacks scene incidence, the caller gets an explicit reason instead
of a lexical fallback that would silently replace the entity layer.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.judgment_layer import match_quote

QUERY_VERSION = 1
DEFAULT_TOP_K = 10
DEFAULT_MAX_HOPS = 2
MAX_DPP_POOL = 80
MAX_JUDGE_CANDIDATES = 30
MAX_EXPANSION_ENTITIES = 40
MAX_CANDIDATE_SCENES = 240
NOTE_SIMILARITY_MIN = 0.10
NOTE_SEED_LIMIT = 2
NOTE_ENTITY_BOOST = 0.06

_ASCII_WORD_RE = re.compile(r"[a-z][a-z0-9'_-]+")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "they", "them",
    "their", "there", "then", "than", "have", "has", "had", "were", "was",
    "will", "would", "could", "should", "what", "when", "where", "which",
    "who", "whom", "why", "how", "his", "her", "him", "she", "you", "your",
    "our", "ours", "its", "not", "but", "are", "is", "be", "been", "being",
    "one", "two", "all", "any", "some", "into", "about", "after", "before",
    "over", "under", "again", "also", "more", "most", "very", "does", "did",
    "can", "may", "might", "must", "shall", "upon", "while", "because",
}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _load_chapter_texts(input_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(Path(input_dir).glob("ch*.md")):
        try:
            out[path.stem] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return out


def _tokens(text: str) -> Counter:
    """Sparse bilingual tokens: ASCII words plus CJK character bigrams."""
    tokens: Counter = Counter()
    lowered = str(text or "").casefold()
    for word in _ASCII_WORD_RE.findall(lowered):
        if len(word) >= 2 and word not in _STOPWORDS:
            tokens[word] += 1
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens[run] += 1
            continue
        for i in range(len(run) - 1):
            tokens[run[i:i + 2]] += 1
    return tokens


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for key, value in a.items():
        dot += value * float(b.get(key) or 0.0)
    if dot <= 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0 or nb <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _compile_name_pattern(names: Iterable[str]) -> Optional[re.Pattern]:
    unique: List[str] = []
    seen: set = set()
    for raw in names:
        name = str(raw or "").strip()
        key = name.casefold()
        if len(name) < 2 or key in seen:
            continue
        seen.add(key)
        unique.append(name)
    if not unique:
        return None
    parts: List[str] = []
    for token in sorted(unique, key=lambda s: (-len(s), s.casefold(), s)):
        escaped = re.escape(token)
        left = r"(?<![A-Za-z0-9])" if token[0].isascii() and token[0].isalnum() else ""
        right = r"(?![A-Za-z0-9])" if token[-1].isascii() and token[-1].isalnum() else ""
        parts.append(left + escaped + right)
    try:
        return re.compile("(?:%s)" % "|".join(parts), re.IGNORECASE)
    except re.error:
        return None


def _chapter_sort_key(chapter_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", str(chapter_id or ""))
    if match:
        return int(match.group(1)), str(chapter_id)
    return 10 ** 9, str(chapter_id)


def _span_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if str(a.get("chapter") or "") != str(b.get("chapter") or ""):
        return False
    try:
        a0, a1 = int(a.get("start")), int(a.get("end"))
        b0, b1 = int(b.get("start")), int(b.get("end"))
    except (TypeError, ValueError):
        return False
    return max(a0, b0) < min(a1, b1)


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    """Small dense determinant with partial pivoting (DPP greedy MAP)."""
    n = len(matrix)
    if n == 0:
        return 1.0
    a = [[float(x) for x in row] for row in matrix]
    det = 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return 0.0
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            det = -det
        value = a[col][col]
        det *= value
        for row in range(col + 1, n):
            factor = a[row][col] / value
            if factor == 0:
                continue
            for j in range(col, n):
                a[row][j] -= factor * a[col][j]
    return det


def _greedy_dpp_select(rows: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Greedy MAP-DPP over already-scored scene candidates.

    The kernel is quality x quality x similarity; the first row is fixed to the
    highest heuristic score, matching the Phase B rule "keep top-1, diversify
    the remaining top-k".
    """
    if top_k <= 0 or not rows:
        return []
    ranked = sorted(rows, key=lambda r: (-float(r.get("score") or 0.0),
                                         str(r.get("scene_id") or "")))
    if len(ranked) <= top_k:
        return ranked
    pool = ranked[:MAX_DPP_POOL]
    for left in pool:
        sims: Dict[str, float] = {}
        for right in pool:
            if left is right:
                continue
            sims[str(right.get("scene_id"))] = round(
                _cosine_sparse(left.get("_vector") or {},
                               right.get("_vector") or {}), 6)
        left["_text_similarity"] = sims
    max_score = max(float(r.get("score") or 0.0) for r in pool) or 1.0
    quality = [max(0.02, float(r.get("score") or 0.0) / max_score) for r in pool]
    kernel: List[List[float]] = []
    for i, left in enumerate(pool):
        row: List[float] = []
        for j, right in enumerate(pool):
            if i == j:
                row.append(quality[i] * quality[i])
                continue
            sim = float(left.get("_text_similarity", {}).get(str(right.get("scene_id")), 0.0))
            jac = _jaccard(left.get("_entity_set") or set(), right.get("_entity_set") or set())
            adjacency = 0.10 if (
                left.get("chapter") == right.get("chapter")
                and abs(int(left.get("index") or 0) - int(right.get("index") or 0)) == 1
            ) else 0.0
            sim = max(0.0, min(1.0, 0.65 * sim + 0.30 * jac + adjacency))
            row.append(quality[i] * quality[j] * sim)
        kernel.append(row)
    selected = [0]
    remaining = set(range(1, len(pool)))
    while len(selected) < min(top_k, len(pool)) and remaining:
        best_idx = -1
        best_gain = -1.0
        for idx in sorted(remaining):
            trial = selected + [idx]
            gain = _determinant([[kernel[r][c] for c in trial] for r in trial])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_idx = idx
        if best_idx < 0:
            best_idx = min(remaining,
                           key=lambda i: (-float(pool[i].get("score") or 0.0),
                                          str(pool[i].get("scene_id") or "")))
        selected.append(best_idx)
        remaining.discard(best_idx)
    return [pool[i] for i in selected]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _tfidf_vectors(token_counts: Dict[str, Counter], query_tokens: Counter) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    doc_ids = sorted(token_counts)
    n = len(doc_ids)
    df: Counter = Counter()
    for doc_id in doc_ids:
        df.update(set(token_counts[doc_id]))
    idf: Dict[str, float] = {}
    for token in set(df) | set(query_tokens):
        idf[token] = math.log((n + 1.0) / (float(df.get(token, 0)) + 1.0)) + 1.0
    q_total = sum(query_tokens.values()) or 1
    qvec = {
        token: (count / float(q_total)) * idf.get(token, 1.0)
        for token, count in query_tokens.items()
    }
    vectors = {
        doc_id: {
            token: (count / float(sum(token_counts[doc_id].values()) or 1))
            * idf.get(token, 1.0)
            for token, count in token_counts[doc_id].items()
        }
        for doc_id in doc_ids
    }
    return vectors, qvec


class StoryGraphQueryIndex:
    """Immutable in-memory index over one ``story_graph.json`` artifact."""

    def __init__(self, graph: Dict[str, Any], input_dir: Path = Path("input")):
        self.graph = graph if isinstance(graph, dict) else {}
        self.input_dir = Path(input_dir)
        self.nodes = [n for n in (self.graph.get("nodes") or []) if isinstance(n, dict)]
        self.edges = [e for e in (self.graph.get("edges") or []) if isinstance(e, dict)]
        self.node_by_id = {str(n.get("id")): n for n in self.nodes}
        self.chapter_texts = _load_chapter_texts(self.input_dir)
        self.chapter_order = sorted(self.chapter_texts, key=_chapter_sort_key)
        self.chapter_index = {cid: i for i, cid in enumerate(self.chapter_order)}
        self.scenes: Dict[str, Dict[str, Any]] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.notes: Dict[str, Dict[str, Any]] = {}
        self._build_nodes()
        self.mentions_by_entity: Dict[str, List[Dict[str, Any]]] = {}
        self.mentions_by_scene: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.note_evidence: Dict[str, List[Dict[str, Any]]] = {}
        self._build_incidence()
        self._build_alias_map()
        self.note_entities: Dict[str, set] = {
            nid: set(self._match_names(str(note.get("claim") or "")))
            for nid, note in self.notes.items()
        }
        self.scene_tokens = {
            sid: _tokens(scene.get("text") or "") for sid, scene in self.scenes.items()
        }
        # Questions and note claims are usually Chinese while evidence quotes
        # are the source-language text.  Weight the claim so the long quote
        # cannot dilute the part of the note a user question is matched against.
        self.note_token_counts = {}
        for nid, note in self.notes.items():
            claim_tokens = _tokens(str(note.get("claim") or ""))
            weighted_claim = Counter({token: count * 3
                                      for token, count in claim_tokens.items()})
            quote_tokens = _tokens(" ".join(
                str(ev.get("quote") or "") for ev in self.note_evidence.get(nid, [])))
            self.note_token_counts[nid] = weighted_claim + quote_tokens

    def _build_nodes(self) -> None:
        for node in self.nodes:
            node_type = str(node.get("type") or "")
            node_id = str(node.get("id") or "")
            attrs = node.get("attributes") or {}
            if node_type == "scene":
                cid = str(attrs.get("chapter") or "")
                start = int(attrs.get("start") or 0)
                end = int(attrs.get("end") or 0)
                text = (self.chapter_texts.get(cid) or "")[start:end]
                self.scenes[node_id] = {
                    "id": node_id,
                    "chapter": cid,
                    "index": int(attrs.get("index") or 0),
                    "start": start,
                    "end": end,
                    "chars": int(attrs.get("chars") or max(0, end - start)),
                    "text": text,
                    "certification": str(node.get("certification") or "auto_verified"),
                }
            elif node_type == "entity":
                aliases = []
                raw_aliases = attrs.get("aliases") or []
                if isinstance(raw_aliases, list):
                    aliases.extend(str(x).strip() for x in raw_aliases if str(x).strip())
                label = str(node.get("label") or "").strip()
                if label:
                    aliases.insert(0, label)
                self.entities[node_id] = {
                    "id": node_id,
                    "label": label,
                    "aliases": aliases,
                    "certification": str(attrs.get("certification")
                                         or node.get("certification") or "pending"),
                }
            elif node_type == "judgment_note":
                status = str(attrs.get("status") or "active")
                validated = attrs.get("validated")
                if status != "active" or validated is False:
                    continue
                self.notes[node_id] = {
                    "id": node_id,
                    "claim": str(attrs.get("claim") or node.get("label") or ""),
                    "chapters": _as_list(attrs.get("chapters")),
                    "certification": str(node.get("certification") or "pending"),
                    "status": status,
                }

    def _build_incidence(self) -> None:
        for edge in self.edges:
            edge_type = str(edge.get("type") or "")
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if edge_type == "mentions_in":
                if source not in self.entities or target not in self.scenes:
                    continue
                prov = (edge.get("provenance") or [{}])[0]
                rec = {
                    "scene_id": target,
                    "count": int(edge.get("count") or 0),
                    "certification": str(edge.get("entity_certification")
                                         or edge.get("certification") or "pending"),
                    "provenance": prov if isinstance(prov, dict) else {},
                }
                self.mentions_by_entity.setdefault(source, []).append(rec)
                self.mentions_by_scene.setdefault(target, {})[source] = rec
            elif edge_type == "evidenced_in":
                if source not in self.notes:
                    continue
                chapter = target.split(":", 1)[1] if target.startswith("chapter:") else target
                for raw in (edge.get("provenance") or []):
                    if not isinstance(raw, dict):
                        continue
                    offset = raw.get("offset")
                    quote = str(raw.get("quote") or "")
                    scene_id = self._scene_for_offset(chapter, offset)
                    match_kind = "offset"
                    matched_quote = quote
                    resolved_offset = int(offset) if offset is not None else None
                    resolved_end = None
                    if scene_id is None and quote:
                        matched = match_quote(self.chapter_texts.get(chapter) or "",
                                              quote, preferred_offset=offset)
                        if matched.get("ok"):
                            scene_id = self._scene_for_offset(
                                chapter, matched.get("offset"))
                            match_kind = str(matched.get("match_kind") or "quote")
                            matched_quote = str(matched.get("matched_quote") or quote)
                            resolved_offset = int(matched.get("offset"))
                            resolved_end = int(matched.get("end"))
                    if resolved_offset is not None and resolved_end is None:
                        # The offset came from the graph's already-validated
                        # evidence edge.  Approximate the end from the quote
                        # length rather than re-running fuzzy L1 matching for
                        # every note (that made index construction ~8s).
                        if matched_quote:
                            resolved_end = resolved_offset + len(matched_quote)
                        elif scene_id and scene_id in self.scenes:
                            resolved_end = int(self.scenes[scene_id].get("end") or resolved_offset)
                    self.note_evidence.setdefault(source, []).append({
                        "chapter": chapter,
                        "offset": resolved_offset,
                        "end": resolved_end,
                        "quote": quote,
                        "matched_quote": matched_quote,
                        "match_kind": match_kind,
                        "scene_id": scene_id,
                    })

    def _build_alias_map(self) -> None:
        alias_to_entity: Dict[str, List[Tuple[str, str]]] = {}
        for edge in self.edges:
            if str(edge.get("type") or "") != "alias_of":
                continue
            if edge.get("ambiguous"):
                continue
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if target not in self.entities:
                continue
            alias_node = self.node_by_id.get(source) or {}
            label = str(alias_node.get("label") or "").strip()
            if not label:
                continue
            cert = str(self.entities[target].get("certification") or "pending")
            alias_to_entity.setdefault(label.casefold(), []).append((target, cert))
        for eid, entity in self.entities.items():
            cert = str(entity.get("certification") or "pending")
            for name in entity.get("aliases") or []:
                key = str(name or "").strip().casefold()
                if len(key) < 2:
                    continue
                alias_to_entity.setdefault(key, []).append((eid, cert))
        # Chinese canonical names are often written as "given-name·house".
        # Add mechanically split segments only when exactly one entity owns the
        # segment; ambiguous short names remain unresolved instead of guessed.
        segment_owners: Dict[str, set] = {}
        for eid, entity in self.entities.items():
            label = str(entity.get("label") or "").strip()
            parts = [p.strip() for p in re.split(r"[·•・\\-\\s]+", label) if p.strip()]
            if len(parts) <= 1:
                continue
            for part in parts:
                key = part.casefold()
                if len(key) >= 2 and key != label.casefold():
                    segment_owners.setdefault(key, set()).add(eid)
        for key, owners in sorted(segment_owners.items()):
            if len(owners) != 1:
                continue
            eid = next(iter(owners))
            cert = str(self.entities[eid].get("certification") or "pending")
            alias_to_entity.setdefault(key, []).append((eid, cert))
        self.alias_to_entity: Dict[str, List[Tuple[str, str]]] = {}
        names: List[str] = []
        for key, rows in alias_to_entity.items():
            unique = sorted(set(rows), key=lambda x: (-(x[1] == "auto_verified"), x[0]))
            self.alias_to_entity[key] = unique
            for eid, _cert in unique:
                label = str(self.entities[eid].get("label") or "")
                if label:
                    names.append(label)
            # The original surface is recovered from entity aliases below; the
            # casefold key is still enough for regex matching.
            names.append(key)
        self.name_pattern = _compile_name_pattern(names)

    def _scene_for_offset(self, chapter: str, offset: Any) -> Optional[str]:
        if offset is None:
            return None
        try:
            value = int(offset)
        except (TypeError, ValueError):
            return None
        rows = [s for s in self.scenes.values() if s.get("chapter") == chapter]
        rows.sort(key=lambda s: int(s.get("start") or 0))
        starts = [int(s.get("start") or 0) for s in rows]
        pos = bisect_right(starts, value) - 1
        if pos >= 0 and value < int(rows[pos].get("end") or 0):
            return str(rows[pos].get("id"))
        return None

    def _match_names(self, text: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if not self.name_pattern or not text:
            return out
        for match in self.name_pattern.finditer(str(text)):
            surface = match.group(0)
            for eid, cert in self.alias_to_entity.get(surface.casefold(), []):
                rec = out.get(eid)
                weight = 1.0 if cert == "auto_verified" else 0.72
                if rec is None:
                    out[eid] = {"entity": eid, "via": "alias", "weight": weight,
                                "surfaces": [surface]}
                else:
                    rec["weight"] = max(float(rec["weight"]), weight)
                    if surface not in rec["surfaces"]:
                        rec["surfaces"].append(surface)
        return out

    def _note_scene_entities(self, note_id: str, limit: int = 12) -> List[str]:
        rows: List[Tuple[float, str]] = []
        seen: set = set()
        for evidence in self.note_evidence.get(note_id, []):
            sid = evidence.get("scene_id")
            if not sid:
                continue
            for eid, mention in (self.mentions_by_scene.get(str(sid)) or {}).items():
                if eid in seen:
                    continue
                cert = str(self.entities.get(eid, {}).get("certification") or "pending")
                score = (2.0 if cert == "auto_verified" else 1.0) + \
                    min(3.0, float(mention.get("count") or 0))
                rows.append((score, eid))
                seen.add(eid)
        rows.sort(key=lambda x: (-x[0], x[1]))
        return [eid for _score, eid in rows[:max(0, limit)]]

    def _similar_notes(self, text: str,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit is None:
            limit = NOTE_SEED_LIMIT
        query_tokens = _tokens(text)
        if not query_tokens or not self.note_token_counts:
            return []
        query_entities = set(self._match_names(text))
        vectors, qvec = _tfidf_vectors(self.note_token_counts, query_tokens)
        scored: List[Tuple[float, float, int, str]] = []
        for nid in self.note_token_counts:
            cosine = _cosine_sparse(vectors.get(nid, {}), qvec)
            overlap = 0
            if cosine >= 0.05 and query_entities:
                overlap = len(query_entities & self.note_entities.get(nid, set()))
            boost = NOTE_ENTITY_BOOST * overlap / float(len(query_entities) or 1)
            scored.append((cosine + boost, cosine, overlap, nid))
        scored.sort(key=lambda item: (-item[0], -item[1], item[3]))
        out: List[Dict[str, Any]] = []
        for score, cosine, overlap, nid in scored[:max(1, limit)]:
            if score < NOTE_SIMILARITY_MIN:
                continue
            note = self.notes.get(nid) or {}
            out.append({"note_id": nid, "score": round(score, 4),
                        "cosine": round(cosine, 4), "entity_overlap": overlap,
                        "claim": str(note.get("claim") or "")[:300]})
        return out

    def seeds_for(self, query: str, note_id: Optional[str] = None,
                  use_note_similarity: bool = True) -> Dict[str, Any]:
        note = self.notes.get(str(note_id or "")) if note_id else None
        text = str(query or "")
        if note:
            text = (text + "\n" + str(note.get("claim") or "")).strip()
        seeds = self._match_names(text)
        evidence_rows: List[Dict[str, Any]] = []
        note_matches: List[Dict[str, Any]] = []
        if note:
            evidence_rows = [dict(ev, note_weight=1.0, note_id=str(note_id))
                             for ev in self.note_evidence.get(str(note_id), [])]
            note_matches = [{"note_id": str(note_id), "score": 1.0,
                             "claim": str(note.get("claim") or "")[:300]}]
            # Note evidence contributes scene-local entities, but only as a
            # weaker seed than an explicit query alias.
            for eid in self._note_scene_entities(str(note_id)):
                rec = seeds.get(eid)
                if rec is None:
                    seeds[eid] = {"entity": eid, "via": "note_evidence", "weight": 0.58,
                                  "surfaces": []}
                else:
                    rec["weight"] = max(float(rec["weight"]), 0.58)
        elif use_note_similarity and text:
            note_matches = self._similar_notes(text, limit=NOTE_SEED_LIMIT)
            top_score = float(note_matches[0].get("score") or 1.0) if note_matches else 1.0
            for rank, match in enumerate(note_matches):
                nid = str(match.get("note_id") or "")
                relative = min(1.0, float(match.get("score") or 0.0) / top_score)
                note_weight = relative * relative
                evidence_rows.extend(
                    dict(ev, note_weight=note_weight, note_id=nid)
                    for ev in self.note_evidence.get(nid, []))
                if rank > 0:
                    continue
                for eid in self._note_scene_entities(nid):
                    rec = seeds.get(eid)
                    if rec is None:
                        seeds[eid] = {"entity": eid, "via": "note_similarity",
                                      "weight": 0.48, "surfaces": []}
                    else:
                        rec["weight"] = max(float(rec["weight"]), 0.48)
        seen_evidence: set = set()
        deduped_evidence: List[Dict[str, Any]] = []
        for ev in evidence_rows:
            key = (str(ev.get("chapter") or ""), ev.get("offset"),
                   str(ev.get("quote") or "")[:80], str(ev.get("scene_id") or ""))
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            deduped_evidence.append(ev)
        return {"text": text, "note": note,
                "note_evidence": deduped_evidence, "note_matches": note_matches,
                "entities": sorted(seeds.values(), key=lambda r: (-float(r["weight"]), r["entity"]))}

    def query(self, query: str = "", note_id: Optional[str] = None,
              top_k: int = DEFAULT_TOP_K, max_hops: int = DEFAULT_MAX_HOPS,
              use_dpp: bool = True,
              use_note_similarity: bool = True,
              llm_generate: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
        top_k = max(0, int(top_k))
        max_hops = max(1, min(2, int(max_hops)))
        seed_payload = self.seeds_for(
            query, note_id=note_id, use_note_similarity=use_note_similarity)
        seeds: List[Dict[str, Any]] = seed_payload["entities"]
        note_evidence: List[Dict[str, Any]] = seed_payload["note_evidence"]
        if not seeds and not any(e.get("scene_id") for e in note_evidence):
            return self._empty_result(seed_payload, "no_seed_match")
        if not self.scenes:
            return self._empty_result(seed_payload, "graph_has_no_scene_incidence")

        candidates = self._seed_candidates(seeds, note_evidence)
        hop1_count = len([c for c in candidates.values() if c.get("hop") == 1])
        hop2_entities: List[str] = []
        hop2_count = 0
        if max_hops >= 2:
            hop2_entities = self._expansion_entities(candidates, {s["entity"] for s in seeds})
            hop2_count = self._expand_hop2(candidates, hop2_entities, seeds)

        candidate_rows = self._score_candidates(candidates, seeds, note_evidence,
                                                seed_payload["text"])
        candidate_rows = [r for r in candidate_rows if self._is_eligible(r)]
        candidate_rows.sort(key=lambda r: (-float(r.get("score") or 0.0),
                                           str(r.get("scene_id") or "")))
        if len(candidate_rows) > MAX_CANDIDATE_SCENES:
            candidate_rows = candidate_rows[:MAX_CANDIDATE_SCENES]

        judge_trace: Dict[str, Any] = {"enabled": bool(llm_generate), "ok": None}
        if llm_generate is not None and candidate_rows:
            judged, judge_trace = self._apply_llm_judge(
                candidate_rows, seed_payload["text"], llm_generate)
            if judged:
                candidate_rows = judged
                candidate_rows.sort(key=lambda r: (-float(r.get("score") or 0.0),
                                                   str(r.get("scene_id") or "")))

        dpp_trace: Dict[str, Any] = {"enabled": bool(use_dpp), "pool": 0,
                                     "first_fixed": "", "selected": []}
        if use_dpp:
            selected = _greedy_dpp_select(candidate_rows, top_k)
            dpp_trace["pool"] = min(len(candidate_rows), MAX_DPP_POOL)
            dpp_trace["first_fixed"] = str(selected[0].get("scene_id") or "") if selected else ""
            dpp_trace["selected"] = [str(r.get("scene_id") or "") for r in selected]
        else:
            selected = candidate_rows[:top_k]

        results = [self._result_row(row, seeds, note_evidence) for row in selected]
        trace = {
            "query": {
                "text": seed_payload["text"],
                "note_id": str(note_id or ""),
                "tokens": sorted(_tokens(seed_payload["text"]))[:48],
            },
            "seeds": seeds,
            "note_matches": seed_payload.get("note_matches") or [],
            "note_evidence": note_evidence,
            "hops": [
                {"hop": 0, "entities": [s["entity"] for s in seeds],
                 "note_evidence_scenes": [e.get("scene_id") for e in note_evidence
                                          if e.get("scene_id")]},
                {"hop": 1, "scenes": [sid for sid, c in candidates.items()
                                      if c.get("hop") == 1]},
                {"hop": 2, "entities": hop2_entities,
                 "scenes": [sid for sid, c in candidates.items()
                            if c.get("hop") == 2]},
            ],
            "candidates": {"hop1": hop1_count, "hop2": hop2_count,
                           "total": len(candidate_rows)},
            "dpp": dpp_trace,
            "llm_judge": judge_trace,
            "graph": {
                "schema_version": self.graph.get("schema_version"),
                "fingerprint": self.graph.get("fingerprint"),
                "source": "input/story_graph.json",
            },
        }
        return {
            "query_version": QUERY_VERSION,
            "query": trace["query"],
            "results": results,
            "trace": trace,
            "metrics": {
                "seed_entities": len(seeds),
                "note_evidence_items": len(note_evidence),
                "candidate_scenes": len(candidate_rows),
                "returned_scenes": len(results),
                "dpp_enabled": bool(use_dpp),
                "llm_judge_ok": judge_trace.get("ok") if llm_generate else None,
            },
        }

    def _seed_candidates(self, seeds: Sequence[Dict[str, Any]],
                         note_evidence: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for seed in seeds:
            eid = str(seed.get("entity") or "")
            for mention in self.mentions_by_entity.get(eid, []):
                sid = str(mention.get("scene_id") or "")
                if sid not in self.scenes:
                    continue
                rec = candidates.setdefault(sid, {
                    "scene_id": sid, "hop": 1, "seed_entities": {},
                    "expansion_entities": {}, "note_evidence": [],
                })
                rec["seed_entities"][eid] = max(
                    float(rec["seed_entities"].get(eid) or 0.0),
                    float(seed.get("weight") or 0.0))
        for evidence in note_evidence:
            sid = str(evidence.get("scene_id") or "")
            if sid not in self.scenes:
                continue
            rec = candidates.setdefault(sid, {
                "scene_id": sid, "hop": 1, "seed_entities": {},
                "expansion_entities": {}, "note_evidence": [],
            })
            rec["note_evidence"].append(evidence)
        return candidates

    def _expansion_entities(self, candidates: Dict[str, Dict[str, Any]],
                            seed_ids: set) -> List[str]:
        weights: Dict[str, float] = {}
        for sid, candidate in candidates.items():
            if candidate.get("hop") != 1:
                continue
            for eid, mention in (self.mentions_by_scene.get(sid) or {}).items():
                if eid in seed_ids or eid not in self.entities:
                    continue
                entity = self.entities[eid]
                cert = str(entity.get("certification") or "pending")
                count = int(mention.get("count") or 0)
                if cert != "auto_verified" and count < 2:
                    continue
                weights[eid] = weights.get(eid, 0.0) + \
                    (1.0 if cert == "auto_verified" else 0.45) * min(3, max(1, count))
        ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
        out: List[str] = []
        for eid, _weight in ranked:
            if len(out) >= MAX_EXPANSION_ENTITIES:
                break
            # Hub entities would turn two hops into a full-book expansion.
            if len(self.mentions_by_entity.get(eid, [])) > 60:
                continue
            out.append(eid)
        return out

    def _expand_hop2(self, candidates: Dict[str, Dict[str, Any]],
                     expansion_entities: Sequence[str],
                     seeds: Sequence[Dict[str, Any]]) -> int:
        added = 0
        seed_ids = {str(s.get("entity") or "") for s in seeds}
        for eid in expansion_entities:
            mentions = list(self.mentions_by_entity.get(eid, []))
            mentions.sort(key=lambda m: (-int(m.get("count") or 0), str(m.get("scene_id"))))
            for mention in mentions[:24]:
                sid = str(mention.get("scene_id") or "")
                if sid not in self.scenes or sid in candidates:
                    continue
                cert = str(self.entities.get(eid, {}).get("certification") or "pending")
                candidates[sid] = {
                    "scene_id": sid, "hop": 2, "seed_entities": {},
                    "expansion_entities": {eid: 1.0 if cert == "auto_verified" else 0.45},
                    "note_evidence": [],
                }
                added += 1
                if len(candidates) >= MAX_CANDIDATE_SCENES:
                    return added
        return added

    def _score_candidates(self, candidates: Dict[str, Dict[str, Any]],
                          seeds: Sequence[Dict[str, Any]],
                          note_evidence: Sequence[Dict[str, Any]],
                          query_text: str) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        active_seed_ids = {
            str(s.get("entity") or "") for s in seeds
            if self.mentions_by_entity.get(str(s.get("entity") or ""))
        }
        seed_weight_total = sum(
            float(s.get("weight") or 0.0) for s in seeds
            if str(s.get("entity") or "") in active_seed_ids
        ) or 1.0
        note_weights: Dict[str, float] = {}
        for evidence in note_evidence:
            sid = str(evidence.get("scene_id") or "")
            if not sid:
                continue
            note_weights[sid] = max(
                note_weights.get(sid, 0.0),
                float(evidence.get("note_weight") if evidence.get("note_weight") is not None
                      else 1.0),
            )
        note_scene_ids = set(note_weights)
        note_chapters = {
            str(self.scenes[sid].get("chapter")) for sid in note_scene_ids
            if sid in self.scenes
        }
        seed_chapters = note_chapters or {
            str(self.scenes[sid].get("chapter"))
            for sid in candidates
            if candidates[sid].get("hop") == 1
        }
        query_tokens = _tokens(query_text)
        token_counts = {sid: self.scene_tokens.get(sid, Counter()) for sid in candidates}
        vectors, qvec = _tfidf_vectors(token_counts, query_tokens)
        rows: List[Dict[str, Any]] = []
        for sid, candidate in candidates.items():
            scene = self.scenes[sid]
            scene_entities = self.mentions_by_scene.get(sid) or {}
            seed_hit_weight = sum(
                float(candidate.get("seed_entities", {}).get(eid) or 0.0)
                for eid in candidate.get("seed_entities", {})
                if eid in scene_entities
            )
            seed_ratio = min(1.0, seed_hit_weight / seed_weight_total)
            note_bonus = note_weights.get(sid, 0.0)
            total_mentions = sum(int(m.get("count") or 0) for m in scene_entities.values())
            certified = sum(
                1 for eid in scene_entities
                if str(self.entities.get(eid, {}).get("certification") or "") == "auto_verified"
            )
            cert_ratio = (certified / float(len(scene_entities))) if scene_entities else 0.0
            density = min(1.0, total_mentions / 10.0)
            lexical = _cosine_sparse(vectors.get(sid, {}), qvec)
            chapter = str(scene.get("chapter") or "")
            if chapter in self.chapter_index and seed_chapters:
                distance = min(
                    abs(self.chapter_index[chapter] - self.chapter_index.get(c, -99))
                    for c in seed_chapters
                )
                proximity = 1.0 / (1.0 + float(distance))
            else:
                proximity = 0.0
            expansion_weight = sum(float(v) for v in candidate.get("expansion_entities", {}).values())
            expansion_ratio = min(1.0, expansion_weight / max(1.0, float(len(candidate.get("expansion_entities", {})))))
            hop_factor = 1.0 if candidate.get("hop") == 1 else 0.72
            components = {
                "seed": round(seed_ratio, 4),
                "note_evidence": round(note_bonus, 4),
                "certified": round(cert_ratio, 4),
                "density": round(density, 4),
                "lexical": round(lexical, 4),
                "chapter_proximity": round(proximity, 4),
                "expansion": round(expansion_ratio, 4),
                "hop": float(candidate.get("hop") or 1),
            }
            score = hop_factor * (
                0.30 * seed_ratio + 0.40 * note_bonus + 0.10 * cert_ratio
                + 0.06 * density + 0.06 * lexical + 0.05 * proximity
                + 0.03 * expansion_ratio
            )
            rows.append({
                "scene_id": sid,
                "chapter": chapter,
                "index": int(scene.get("index") or 0),
                "start": int(scene.get("start") or 0),
                "end": int(scene.get("end") or 0),
                "chars": int(scene.get("chars") or 0),
                "hop": int(candidate.get("hop") or 1),
                "score": round(score, 6),
                "score_components": components,
                "seed_entities": sorted(candidate.get("seed_entities", {})),
                "expansion_entities": sorted(candidate.get("expansion_entities", {})),
                "note_evidence": candidate.get("note_evidence") or [],
                "_entity_set": set(scene_entities),
                "_text_similarity": {},
                "_vector": vectors.get(sid, {}),
                "_scene_text": scene.get("text") or "",
            })
        return rows

    def _is_eligible(self, row: Dict[str, Any]) -> bool:
        if row.get("hop") == 1:
            return True
        if row.get("note_evidence") or row.get("seed_entities"):
            return True
        components = row.get("score_components") or {}
        expansion_count = len(row.get("expansion_entities") or [])
        return bool(expansion_count and (
            float(components.get("certified") or 0.0) > 0.0
            or expansion_count >= 2
            or float(components.get("lexical") or 0.0) >= 0.05
        ))

    def _apply_llm_judge(self, rows: List[Dict[str, Any]], query_text: str,
                         llm_generate: Callable[[str], str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        trace: Dict[str, Any] = {"enabled": True, "ok": False, "batches": 0,
                                 "verdicts": {}, "error": ""}
        top = rows[:MAX_JUDGE_CANDIDATES]
        verdicts: Dict[str, Dict[str, Any]] = {}
        try:
            for start in range(0, len(top), 10):
                batch = top[start:start + 10]
                prompt = self._build_judge_prompt(query_text, batch)
                raw = llm_generate(prompt)
                from src.utils.llm_adapter import robust_json_loads
                data = robust_json_loads(raw, expected_type=dict)
                for item in (data.get("verdicts") or []):
                    if not isinstance(item, dict):
                        continue
                    sid = str(item.get("id") or item.get("scene_id") or "")
                    verdict = str(item.get("verdict") or "").strip().lower()
                    if sid and verdict in ("necessary", "context", "rejected"):
                        verdicts[sid] = {
                            "verdict": verdict,
                            "score": max(0.0, min(1.0, float(item.get("score") or 0.0))),
                            "reason": str(item.get("reason") or "")[:300],
                        }
                trace["batches"] += 1
        except Exception as exc:  # noqa: BLE001 - advisory layer is fail-open
            trace["error"] = "%s: %s" % (type(exc).__name__, exc)
            trace["ok"] = False
            return rows, trace
        if not verdicts:
            trace["error"] = "no_valid_verdicts"
            return rows, trace
        kept: List[Dict[str, Any]] = []
        rejected = 0
        for row in rows:
            sid = str(row.get("scene_id") or "")
            verdict = verdicts.get(sid)
            if verdict:
                row["judge_verdict"] = verdict["verdict"]
                row["judge_score"] = verdict["score"]
                row["judge_reason"] = verdict["reason"]
                if verdict["verdict"] == "necessary":
                    row["score"] = round(float(row.get("score") or 0.0)
                                         * (1.0 + 0.25 * verdict["score"]), 6)
                elif verdict["verdict"] == "context":
                    row["score"] = round(float(row.get("score") or 0.0) * 0.78, 6)
                else:
                    rejected += 1
                    continue
            kept.append(row)
        trace["verdicts"] = verdicts
        trace["ok"] = True
        trace["rejected"] = rejected
        if not kept:
            trace["ok"] = False
            trace["error"] = "all_candidates_rejected"
            return rows, trace
        return kept, trace

    def _build_judge_prompt(self, query_text: str, rows: Sequence[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for row in rows:
            scene = self.scenes.get(str(row.get("scene_id"))) or {}
            excerpt = str(scene.get("text") or "")[:720].replace("\n", " ")
            entities = []
            for eid in sorted((self.mentions_by_scene.get(str(row.get("scene_id"))) or {}))[:16]:
                entities.append(str(self.entities.get(eid, {}).get("label") or eid))
            blocks.append(
                "id=%s\nchapter=%s\nentities=%s\ntext=%s" % (
                    row.get("scene_id"), row.get("chapter"),
                    ", ".join(entities), excerpt))
        return """你是小说故事图的证据复核器。给定问题/主张和候选场景，只判断候选场景是否提供了回答问题所需的必要事实。

判定规则：
1. necessary：场景本身包含人物、行动、因果或结果等必要事实，能直接支撑问题/主张。
2. context：只提供背景、氛围、重复信息或可由其他场景替代的内容，不能单独支撑。
3. rejected：主题相似但事实不必要、时间线不符，或只涉及术语/译名断言。
4. 不得把上下文场景的事实归给候选；不得补写原文没有的实体、事件或译名。
5. 术语与译名不归本层裁决。

只输出 JSON 对象：
{"verdicts":[{"id":"scene:...","verdict":"necessary|context|rejected","score":0.0,"reason":"一句话"}]}

问题/主张：
%s

候选场景：
%s
""" % (str(query_text or "")[:1200], "\n\n".join(blocks))

    def _result_row(self, row: Dict[str, Any], seeds: Sequence[Dict[str, Any]],
                    note_evidence: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        sid = str(row.get("scene_id") or "")
        evidence: List[Dict[str, Any]] = []
        for ev in note_evidence:
            if str(ev.get("scene_id") or "") == sid:
                evidence.append({
                    "source": "note_evidence",
                    "chapter": ev.get("chapter"),
                    "start": ev.get("offset"),
                    "end": ev.get("end"),
                    "quote": ev.get("matched_quote") or ev.get("quote"),
                    "match_kind": ev.get("match_kind"),
                })
        scene_mentions = self.mentions_by_scene.get(sid) or {}
        seed_ids = {str(s.get("entity") or "") for s in seeds}
        rows: List[Tuple[int, str, Dict[str, Any]]] = []
        for eid, mention in scene_mentions.items():
            entity = self.entities.get(eid) or {}
            prov = mention.get("provenance") or {}
            priority = 0
            if eid in seed_ids:
                priority += 2
            if str(entity.get("certification") or "") == "auto_verified":
                priority += 1
            rows.append((-priority, eid, {
                "entity": eid,
                "label": str(entity.get("label") or eid),
                "certification": str(entity.get("certification") or "pending"),
                "count": int(mention.get("count") or 0),
                "chapter": prov.get("chapter"),
                "start": prov.get("start"),
                "end": prov.get("end"),
                "offset": prov.get("offset"),
                "quote": prov.get("quote"),
                "surface": prov.get("surface"),
            }))
        rows.sort(key=lambda item: (item[0], -int(item[2].get("count") or 0), item[1]))
        evidence.extend([item[2] for item in rows[:3]])
        anchor = evidence[0] if evidence else {
            "source": "scene", "chapter": row.get("chapter"),
            "start": row.get("start"), "end": row.get("end"),
            "quote": str(row.get("_scene_text") or "")[:180].replace("\n", " "),
        }
        entities_out = []
        for _prio, eid, item in rows:
            entities_out.append({
                "entity": eid,
                "label": item["label"],
                "certification": item["certification"],
                "count": item["count"],
                "role": "seed" if eid in seed_ids else "expansion",
            })
        return {
            "scene_id": sid,
            "chapter": row.get("chapter"),
            "start": row.get("start"),
            "end": row.get("end"),
            "chars": row.get("chars"),
            "hop": row.get("hop"),
            "score": row.get("score"),
            "score_components": row.get("score_components"),
            "certification": "auto_verified" if any(
                e.get("certification") == "auto_verified" for e in entities_out
            ) else "pending",
            "entities": entities_out,
            "evidence": evidence[:4],
            "anchor": anchor,
            "excerpt": str(row.get("_scene_text") or "")[:240].replace("\n", " "),
            "judge_verdict": row.get("judge_verdict"),
            "judge_score": row.get("judge_score"),
            "judge_reason": row.get("judge_reason"),
        }

    def _best_scene_quote(self, text: str, query_text: str,
                          max_chars: int = 240) -> str:
        """Pick a deterministic, verbatim sentence for candidate evidence."""
        raw = str(text or "")
        if not raw.strip():
            return ""
        parts = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", raw)
        query_tokens = _tokens(query_text)
        best = ("", -1.0, -1)
        for idx, raw_part in enumerate(parts):
            part = " ".join(str(raw_part or "").split())
            if len(part) < 16:
                continue
            score = _cosine_sparse(dict(_tokens(part)), dict(query_tokens))
            if (score, -idx) > (best[1], best[2]):
                best = (part, score, -idx)
        quote = best[0]
        if not quote:
            quote = " ".join(raw.split())
        return quote[:max(1, int(max_chars))].strip()

    def setup_payoff_candidates(self, setup_note: Dict[str, Any],
                                top_n: int = 5,
                                max_hops: int = DEFAULT_MAX_HOPS,
                                seed_entity_limit: int = 12) -> Dict[str, Any]:
        """Return deterministically ranked *later-chapter* payoff candidates.

        This is the Phase C read-only primitive.  It accepts a dormant/dropped
        single-chapter setup note, builds temporary seeds from the claim plus
        the setup evidence scene, expands at most two hops, and never promotes
        or mutates any note.
        """
        setup = setup_note if isinstance(setup_note, dict) else {}
        setup_id = str(setup.get("id") or "")
        evidence = [dict(e) for e in (setup.get("evidence") or [])
                    if isinstance(e, dict)]
        chapters = [str(c) for c in (setup.get("chapters") or []) if str(c)]
        setup_chapter = (chapters[0] if chapters else
                         str((evidence[0] if evidence else {}).get("chapter") or ""))
        setup_idx = self.chapter_index.get(setup_chapter)
        claim = str(setup.get("claim") or "")
        query_text = (claim + "\n" + "\n".join(
            str(e.get("quote") or "") for e in evidence)).strip()
        seeds = self._match_names(query_text)
        for ev in evidence[:2]:
            sid = self._scene_for_offset(
                str(ev.get("chapter") or setup_chapter), ev.get("offset"))
            if not sid:
                continue
            mentions = sorted(
                (self.mentions_by_scene.get(sid) or {}).items(),
                key=lambda kv: (-int(kv[1].get("count") or 0), kv[0]))
            for eid, _mention in mentions[:max(1, int(seed_entity_limit))]:
                rec = seeds.get(eid)
                if rec is None:
                    seeds[eid] = {"entity": eid, "via": "setup_scene",
                                  "weight": 0.62, "surfaces": []}
                else:
                    rec["weight"] = max(float(rec["weight"]), 0.62)
        seed_rows = sorted(seeds.values(), key=lambda r: (-float(r["weight"]), r["entity"]))
        later_rows: List[Dict[str, Any]] = []
        if setup_idx is not None and seed_rows:
            candidates = self._seed_candidates(seed_rows, [])
            if max_hops >= 2 and candidates:
                expansion = self._expansion_entities(
                    candidates, {str(s.get("entity") or "") for s in seed_rows})
                self._expand_hop2(candidates, expansion, seed_rows)
            rows = self._score_candidates(candidates, seed_rows, [], query_text)
            for row in rows:
                if not self._is_eligible(row):
                    continue
                chapter = str(row.get("chapter") or "")
                chapter_idx = self.chapter_index.get(chapter)
                if chapter_idx is None or setup_idx is None or chapter_idx <= setup_idx:
                    continue
                later_rows.append(row)
            later_rows.sort(key=lambda r: (-float(r.get("score") or 0.0),
                                           str(r.get("scene_id") or "")))
        selected: List[Dict[str, Any]] = []
        per_chapter: Counter = Counter()
        for row in later_rows:
            chapter = str(row.get("chapter") or "")
            if per_chapter[chapter] >= 2:
                continue
            per_chapter[chapter] += 1
            text = str(row.get("_scene_text") or "")
            anchor_quote = self._best_scene_quote(text, query_text, max_chars=240)
            pos = text.find(anchor_quote) if anchor_quote else -1
            if pos >= 0:
                excerpt = text[max(0, pos - 120):pos + len(anchor_quote) + 220]
            else:
                excerpt = text[:420]
            selected.append({
                "candidate_id": "cand-%03d" % (len(selected) + 1),
                "scene_id": str(row.get("scene_id") or ""),
                "chapter": chapter,
                "start": int(row.get("start") or 0),
                "end": int(row.get("end") or 0),
                "score": round(float(row.get("score") or 0.0), 6),
                "score_components": row.get("score_components") or {},
                "anchor_quote": anchor_quote,
                "excerpt": " ".join(excerpt.split())[:520],
            })
            if len(selected) >= max(1, int(top_n)):
                break
        terminal = setup_idx is None or setup_idx >= max(self.chapter_index.values() or [setup_idx])
        return {
            "setup_id": setup_id,
            "setup_chapter": setup_chapter,
            "setup_chapters": chapters or ([setup_chapter] if setup_chapter else []),
            "seeds": seed_rows,
            "later_candidate_count": len(later_rows),
            "candidates": selected,
            "terminal": bool(terminal),
            "terminal_reason": "no_later_chapter" if terminal else "",
        }

    def _empty_result(self, seed_payload: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "query_version": QUERY_VERSION,
            "query": {"text": seed_payload.get("text") or "",
                      "note_id": str((seed_payload.get("note") or {}).get("id") or "")},
            "results": [],
            "trace": {
                "seeds": seed_payload.get("entities") or [],
                "note_evidence": seed_payload.get("note_evidence") or [],
                "note_matches": seed_payload.get("note_matches") or [],
                "hops": [], "candidates": {"hop1": 0, "hop2": 0, "total": 0},
                "dpp": {"enabled": False, "pool": 0, "selected": []},
                "llm_judge": {"enabled": False, "ok": None},
                "reason": reason,
                "graph": {"schema_version": self.graph.get("schema_version"),
                          "fingerprint": self.graph.get("fingerprint")},
            },
            "metrics": {"seed_entities": len(seed_payload.get("entities") or []),
                        "note_evidence_items": len(seed_payload.get("note_evidence") or []),
                        "candidate_scenes": 0, "returned_scenes": 0},
        }


def query_story_graph(graph: Dict[str, Any], query: str = "",
                      input_dir: Path = Path("input"),
                      note_id: Optional[str] = None,
                      top_k: int = DEFAULT_TOP_K,
                      max_hops: int = DEFAULT_MAX_HOPS,
                      use_dpp: bool = True,
                      use_note_similarity: bool = True,
                      llm_generate: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    index = StoryGraphQueryIndex(graph, input_dir=input_dir)
    return index.query(query=query, note_id=note_id, top_k=top_k,
                       max_hops=max_hops, use_dpp=use_dpp,
                       use_note_similarity=use_note_similarity,
                       llm_generate=llm_generate)


def default_story_graph_query_generate(prompt: str,
                                       task_name: str = "story_graph_query") -> str:
    from src.config import get_config
    from src.utils.llm_adapter import get_llm_client_for_task, get_role_model_name

    cfg = get_config()
    if task_name not in (cfg.task_routing or {}):
        task_name = "style_profiling_merge"
    model_key, params = cfg.resolve_task_model(task_name)
    role = cfg.get_task_role(task_name)
    llm = get_llm_client_for_task(task_name)
    model_name = get_role_model_name(role) or model_key
    return llm.generate(prompt, model_name=model_name, **params)


def _resolve_gold_span(index: StoryGraphQueryIndex, raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, str):
        scene = index.scenes.get(raw)
        if scene:
            return {"scene_id": raw, "chapter": scene.get("chapter"),
                    "start": scene.get("start"), "end": scene.get("end")}
        return None
    if not isinstance(raw, dict):
        return None
    sid = str(raw.get("scene_id") or "")
    if sid and sid in index.scenes:
        scene = index.scenes[sid]
        return {"scene_id": sid, "chapter": scene.get("chapter"),
                "start": scene.get("start"), "end": scene.get("end")}
    chapter = str(raw.get("chapter") or "")
    if raw.get("start") is not None and raw.get("end") is not None:
        return {"chapter": chapter, "start": int(raw["start"]), "end": int(raw["end"])}
    quote = str(raw.get("quote") or "")
    if chapter and quote:
        hit = match_quote(index.chapter_texts.get(chapter) or "", quote,
                          preferred_offset=raw.get("offset"))
        if hit.get("ok"):
            return {"chapter": chapter, "start": int(hit["offset"]),
                    "end": int(hit["end"]), "quote": quote,
                    "match_kind": hit.get("match_kind")}
    return None


def evaluate_story_graph_query(index: StoryGraphQueryIndex,
                               cases: Sequence[Dict[str, Any]],
                               top_k: int = DEFAULT_TOP_K,
                               max_hops: int = DEFAULT_MAX_HOPS,
                               use_dpp: bool = True,
                               use_note_similarity: bool = True,
                               llm_generate: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        cid = str(case.get("id") or "")
        question = str(case.get("question") or case.get("claim") or "")
        note_id = str(case.get("note_id") or "") or None
        gold_raw = case.get("gold") or case.get("gold_spans") or []
        gold = [g for g in (_resolve_gold_span(index, item) for item in _as_list(gold_raw)) if g]
        if not question and not note_id:
            continue
        result = index.query(query=question, note_id=note_id, top_k=top_k,
                             max_hops=max_hops, use_dpp=use_dpp,
                             use_note_similarity=use_note_similarity,
                             llm_generate=llm_generate)
        note_matches = (result.get("trace") or {}).get("note_matches") or []
        results = result.get("results") or []
        result_spans: List[Dict[str, Any]] = []
        for row in results:
            anchor = row.get("anchor") or {}
            result_spans.append({
                "scene_id": str(row.get("scene_id") or ""),
                "span": {"chapter": row.get("chapter"), "start": row.get("start"),
                         "end": row.get("end")},
                "anchor": {"chapter": anchor.get("chapter"),
                           "start": anchor.get("start"), "end": anchor.get("end")},
            })

        def recall_at(k: int) -> float:
            if not gold:
                return 0.0
            top = result_spans[:max(0, k)]
            hit = sum(1 for gold_span in gold
                      if any(_span_overlap(r["span"], gold_span) for r in top))
            return hit / float(len(gold))

        def anchor_hit_at(k: int) -> bool:
            top = result_spans[:max(0, k)]
            return any(_span_overlap(r["anchor"], gold_span)
                       for r in top for gold_span in gold)

        def precision_at(k: int) -> float:
            top = result_spans[:max(0, k)]
            if not top:
                return 0.0
            matched = sum(1 for r in top
                          if any(_span_overlap(r["span"], gold_span) for gold_span in gold))
            return matched / float(len(top))

        matched_at_10 = sum(
            1 for r in result_spans[:10]
            if any(_span_overlap(r["span"], gold_span) for gold_span in gold)
        )

        rows.append({
            "id": cid,
            "question": question,
            "note_id": note_id or "",
            "mode": ("note_evidence" if note_id else
                     ("note_similarity" if note_matches else "alias_only")),
            "note_matches": note_matches,
            "gold_spans": gold,
            "R@5": round(recall_at(5), 4),
            "R@10": round(recall_at(10), 4),
            "AE@5": anchor_hit_at(5),
            "AE@10": anchor_hit_at(10),
            "precision@1": round(precision_at(1), 4),
            "precision@3": round(precision_at(3), 4),
            "precision@5": round(precision_at(5), 4),
            "precision@10": round(precision_at(10), 4),
            "matched@10": matched_at_10,
            "returned": len(results),
            "top_scene": str((results[0] if results else {}).get("scene_id") or ""),
            "reason": str((result.get("trace") or {}).get("reason") or ""),
        })
    if not rows:
        return {"cases": 0, "metrics": {}, "rows": []}

    def mean(key: str) -> float:
        return round(sum(float(r.get(key) or 0.0) for r in rows) / len(rows), 4)

    def subset_mean(items: Sequence[Dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return round(sum(float(r.get(key) or 0.0) for r in items) / len(items), 4)

    metrics = {
        "cases": len(rows),
        "R@5": mean("R@5"),
        "R@10": mean("R@10"),
        "AE@5": round(sum(1 for r in rows if r["AE@5"]) / float(len(rows)), 4),
        "AE@10": round(sum(1 for r in rows if r["AE@10"]) / float(len(rows)), 4),
        "anchor_precision@1": mean("precision@1"),
        "anchor_precision@3": mean("precision@3"),
        "anchor_precision@5": mean("precision@5"),
        "anchor_precision@10": mean("precision@10"),
    }
    # Micro precision across the first ten returned scenes: this is the
    # "anchor precision" audit number, not a relevance judgement for a book.
    total_returned = sum(min(10, int(r.get("returned") or 0)) for r in rows)
    total_matched = sum(int(r.get("matched@10") or 0) for r in rows)
    metrics["anchor_precision"] = round(
        total_matched / float(total_returned or 1), 4)
    for mode in ("alias_only", "note_similarity", "note_evidence"):
        subset = [r for r in rows if r.get("mode") == mode]
        if not subset:
            continue
        metrics[mode + "_cases"] = len(subset)
        metrics[mode + "_R@5"] = subset_mean(subset, "R@5")
        metrics[mode + "_R@10"] = subset_mean(subset, "R@10")
        metrics[mode + "_AE@5"] = round(
            sum(1 for r in subset if r["AE@5"]) / float(len(subset)), 4)
        metrics[mode + "_AE@10"] = round(
            sum(1 for r in subset if r["AE@10"]) / float(len(subset)), 4)
        metrics[mode + "_anchor_precision@1"] = subset_mean(subset, "precision@1")
    return {"cases": len(rows), "metrics": metrics, "rows": rows}


def format_query_result(result: Dict[str, Any], max_items: int = 5) -> str:
    rows = result.get("results") or []
    trace = result.get("trace") or {}
    metrics = result.get("metrics") or {}
    lines = [
        "🔎 story-graph-query: seeds=%s candidates=%s returned=%s" % (
            metrics.get("seed_entities", 0), metrics.get("candidate_scenes", 0),
            metrics.get("returned_scenes", len(rows))),
    ]
    reason = trace.get("reason")
    if reason:
        lines.append("   reason=%s" % reason)
    note_matches = trace.get("note_matches") or []
    if note_matches:
        brief = ["%s(%.3f)" % (m.get("note_id"), float(m.get("score") or 0.0))
                 for m in note_matches[:3]]
        lines.append("   notes: %s" % ", ".join(brief))
    for i, row in enumerate(rows[:max_items], 1):
        labels = [str(e.get("label") or e.get("entity")) for e in (row.get("entities") or [])[:5]]
        anchor = row.get("anchor") or {}
        lines.append(
            "%2d. %s %s [%s-%s] score=%.4f hop=%s entities=%s" % (
                i, row.get("chapter"), row.get("scene_id"), row.get("start"),
                row.get("end"), float(row.get("score") or 0.0), row.get("hop"),
                ", ".join(labels)))
        quote = str(anchor.get("quote") or "").replace("\n", " ")[:160]
        if quote:
            lines.append("    anchor: %s" % quote)
        verdict = row.get("judge_verdict")
        if verdict:
            lines.append("    judge: %s (%.2f) %s" % (
                verdict, float(row.get("judge_score") or 0.0), row.get("judge_reason") or ""))
    return "\n".join(lines)
