"""Round 158 Phase 2: fail-closed query gold-set builder.

The evaluator must not consume a hand-written list that no longer matches the
current profile, chapter sources, or story graph.  This module derives cases
from validated active notes, re-checks every evidence quote against the split
chapter files, and binds the result to the current freshness fingerprints.

Setup/payoff cases require at least two distinct evidence chapters.  A profile
with no active setup/payoff notes produces an explicit coverage gap and no
formal eval artifact; it must not silently become an arc/voice-only benchmark.
"""
from __future__ import annotations

import datetime as _dt
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.artifact_freshness import (
    EVAL_SCHEMA_VERSION,
    chapter_manifest,
    profile_notes_hash,
)
from src.judgment_layer import all_notes, atomic_write_json, match_quote, note_is_validated
from src.style_profiler import load_profile


BUILDER_VERSION = 1
DEFAULT_OUTPUT_PATH = "docs/ROUND158_PHASE2_EVAL.json"
DEFAULT_REPORT_PATH = "output/query_gold_set_build.json"
DEFAULT_MIN_SETUP = 10
DEFAULT_MIN_ARC = 10
DEFAULT_MIN_VOICE = 5
DEFAULT_MIN_NEGATIVE = 5
DEFAULT_MAX_CASES = 50


_NEGATIVE_QUESTIONS = (
    "这本书是否包含会说话的机械龙与厄崔迪家族结盟的章节？",
    "书中是否描写了主角使用量子计算机编写操作系统？",
    "这本小说是否讲述了星际舰队在银河中心举行编程竞赛？",
    "书中是否有一段关于现代股票交易所高频交易的完整教程？",
    "故事里是否出现过用魔法咒语启动宇宙飞船的场景？",
)


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _chapter_sort_key(chapter: str) -> Tuple[int, str]:
    text = str(chapter or "")
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else 10 ** 9, text)


def _read_chapters(input_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in sorted(Path(input_dir).glob("ch*.md")):
        try:
            out[path.stem] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return out


def _validated_gold(note: Dict[str, Any], chapters: Dict[str, str],
                    min_similarity: float = 0.9) -> List[dict]:
    gold: List[dict] = []
    seen = set()
    for evidence in note.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        chapter = str(evidence.get("chapter") or "")
        quote = str(evidence.get("quote") or "")
        text = chapters.get(chapter)
        if not text or not quote:
            continue
        hit = match_quote(text, quote, min_similarity=min_similarity,
                          preferred_offset=evidence.get("offset"))
        if not hit.get("ok"):
            continue
        offset = hit.get("offset")
        key = (chapter, int(offset or 0), quote)
        if key in seen:
            continue
        seen.add(key)
        gold.append({
            "chapter": chapter,
            "quote": quote,
            "offset": offset,
            "match_kind": hit.get("match_kind") or "",
            "match_score": hit.get("match_score"),
        })
    gold.sort(key=lambda item: (_chapter_sort_key(item["chapter"]),
                                int(item.get("offset") or 0), item["quote"]))
    return gold


def _question_for(note: Dict[str, Any]) -> str:
    claim = str(note.get("claim") or "").strip()
    ntype = str(note.get("type") or "")
    if ntype == "setup_payoff":
        return "找出这个伏笔的埋设点以及它在后文被回收的具体场景：" + claim
    if ntype == "arc":
        return "找出能支撑以下叙事弧线判断的具体原文场景：" + claim
    if ntype == "voice":
        return "找出能验证以下叙述语气判断的具体原文场景：" + claim
    return "找出能支撑以下判断的具体原文场景：" + claim


def _note_pool(profile: Dict[str, Any], chapters: Dict[str, str]) -> Dict[str, List[dict]]:
    pool: Dict[str, List[dict]] = {"setup_payoff": [], "arc": [], "voice": []}
    for note in all_notes(profile):
        if not note_is_validated(note):
            continue
        ntype = str(note.get("type") or "")
        if ntype not in pool:
            continue
        gold = _validated_gold(note, chapters)
        if not gold:
            continue
        if ntype == "setup_payoff":
            evidence_chapters = {item["chapter"] for item in gold}
            if len(evidence_chapters) < 2:
                continue
        item = {
            "id": str(note.get("id") or ""),
            "type": ntype,
            "claim": str(note.get("claim") or ""),
            "chapters": [str(c) for c in (note.get("chapters") or [])],
            "confidence": float(note.get("confidence") or 0.0),
            "status": str(note.get("status") or ""),
            "gold": gold,
        }
        if item["id"]:
            pool[ntype].append(item)
    return pool


def _select_diverse(notes: Sequence[dict], count: int) -> List[dict]:
    """Pick highest-confidence notes while spreading chapters round-robin."""
    if count <= 0:
        return []
    ordered = sorted(
        notes,
        key=lambda n: (-float(n.get("confidence") or 0.0),
                       _chapter_sort_key((n.get("chapters") or [""])[0]),
                       str(n.get("id") or "")),
    )
    by_chapter: Dict[str, List[dict]] = {}
    for note in ordered:
        primary = str((note.get("chapters") or [""])[0] or "")
        by_chapter.setdefault(primary, []).append(note)
    selected: List[dict] = []
    seen = set()
    chapter_order = sorted(by_chapter, key=_chapter_sort_key)
    while len(selected) < count and chapter_order:
        progressed = False
        for chapter in list(chapter_order):
            bucket = by_chapter.get(chapter) or []
            if not bucket:
                chapter_order.remove(chapter)
                continue
            note = bucket.pop(0)
            if note["id"] in seen:
                progressed = True
                continue
            seen.add(note["id"])
            selected.append(note)
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def _case_from_note(note: dict) -> dict:
    return {
        "id": "r158-" + str(note["type"]) + "-" + str(note["id"]).replace(":", "-"),
        "note_id": str(note["id"]),
        "question": _question_for(note),
        "type": str(note["type"]),
        "gold": note["gold"],
        "answerable": True,
        "lineage": {
            "note_status": note.get("status"),
            "note_confidence": note.get("confidence"),
            "question_source": "note.claim",
            "gold_source": "note.evidence:L1_revalidated",
        },
    }


def _negative_cases(count: int) -> List[dict]:
    out: List[dict] = []
    for index, question in enumerate(_NEGATIVE_QUESTIONS[:max(0, int(count))], 1):
        out.append({
            "id": f"r158-negative-{index:02d}",
            "question": question,
            "type": "negative",
            "gold": [],
            "answerable": False,
            "negative": True,
            "lineage": {"question_source": "deterministic_out_of_domain",
                        "gold_source": "none"},
        })
    return out


def build_query_gold_set(
    project_root: Path,
    *,
    profile: Optional[Dict[str, Any]] = None,
    graph: Optional[Dict[str, Any]] = None,
    profile_path: Optional[Path] = None,
    graph_path: Optional[Path] = None,
    input_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    min_setup: int = DEFAULT_MIN_SETUP,
    min_arc: int = DEFAULT_MIN_ARC,
    min_voice: int = DEFAULT_MIN_VOICE,
    min_negative: int = DEFAULT_MIN_NEGATIVE,
    max_cases: int = DEFAULT_MAX_CASES,
    write_partial: bool = False,
) -> Dict[str, Any]:
    root = Path(project_root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    prof_path = Path(profile_path) if profile_path else in_dir / "book_profile.json"
    if profile is None:
        profile = load_profile(prof_path)
    graph_payload = graph
    if graph_payload is None and graph_path and Path(graph_path).exists():
        try:
            graph_payload = json.loads(Path(graph_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            graph_payload = None
    out_path = Path(output_path) if output_path else root / DEFAULT_OUTPUT_PATH
    rep_path = Path(report_path) if report_path else root / DEFAULT_REPORT_PATH
    result: Dict[str, Any] = {
        "builder_version": BUILDER_VERSION,
        "ok": False,
        "status": "invalid",
        "generated_at": _now_iso(),
        "profile_path": str(prof_path),
        "input_dir": str(in_dir),
        "output_path": str(out_path),
        "report_path": str(rep_path),
        "eval_written": False,
        "counts": {},
        "required": {
            "setup_payoff": int(min_setup), "arc": int(min_arc),
            "voice": int(min_voice), "negative": int(min_negative),
            "total": int(min_setup + min_arc + min_voice + min_negative),
        },
        "gaps": {},
        "errors": [],
    }
    if not isinstance(profile, dict):
        result["errors"].append("profile_missing_or_unreadable")
        atomic_write_json(rep_path, result)
        return result
    chapters = _read_chapters(in_dir)
    if not graph_payload:
        result["errors"].append("graph_missing_or_unreadable")
    if result["errors"]:
        atomic_write_json(rep_path, result)
        return result

    pool = _note_pool(profile, chapters)
    setup = _select_diverse(pool["setup_payoff"], int(min_setup))
    arc = _select_diverse(pool["arc"], int(min_arc))
    voice = _select_diverse(pool["voice"], int(min_voice))
    negatives = _negative_cases(int(min_negative))
    available = {
        "setup_payoff": len(pool["setup_payoff"]),
        "arc": len(pool["arc"]),
        "voice": len(pool["voice"]),
        "negative": len(negatives),
    }
    gaps = {
        "setup_payoff": max(0, int(min_setup) - len(setup)),
        "arc": max(0, int(min_arc) - len(arc)),
        "voice": max(0, int(min_voice) - len(voice)),
        "negative": max(0, int(min_negative) - len(negatives)),
    }
    cases = ([_case_from_note(n) for n in setup]
             + [_case_from_note(n) for n in arc]
             + [_case_from_note(n) for n in voice]
             + negatives)
    if len(cases) > int(max_cases):
        result["status"] = "too_many_cases"
        result["errors"].append("case_count_exceeds_max")
        atomic_write_json(rep_path, result)
        return result
    result["counts"] = dict(Counter(case["type"] for case in cases))
    result["available"] = available
    result["gaps"] = {key: value for key, value in gaps.items() if value}
    case_chapters = sorted({
        str(gold.get("chapter") or "")
        for case in cases for gold in (case.get("gold") or [])
        if gold.get("chapter")
    }, key=_chapter_sort_key)
    result["chapter_coverage"] = case_chapters
    result["chapter_manifest"] = chapter_manifest(in_dir)
    result["profile_notes_hash"] = profile_notes_hash(profile)
    result["graph_fingerprint"] = str(graph_payload.get("fingerprint") or "")
    if result["gaps"] or len(cases) < int(result["required"]["total"]):
        result["status"] = "insufficient_case_mix"
        result["errors"].append("case_mix_below_required")
        if not write_partial:
            atomic_write_json(rep_path, result)
            return result
    title = str(((profile.get("work_guess") or {}).get("title")) or "unknown")
    eval_payload = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "book": title,
        "source": "validated active notes + deterministic negatives",
        "generated_at": result["generated_at"],
        "builder_version": BUILDER_VERSION,
        "profile_notes_hash": result["profile_notes_hash"],
        "chapter_manifest": result["chapter_manifest"],
        "graph_fingerprint": result["graph_fingerprint"],
        "counts": result["counts"],
        "chapter_coverage": case_chapters,
        "cases": cases,
    }
    atomic_write_json(out_path, eval_payload)
    result["case_count"] = len(cases)
    result["eval_written"] = True
    if result["gaps"]:
        result["status"] = "partial_draft"
        atomic_write_json(rep_path, result)
        return result
    result["ok"] = True
    result["status"] = "fresh"
    atomic_write_json(rep_path, result)
    return result

