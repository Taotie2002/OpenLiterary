"""Round 142 Phase C0/C1: setup/payoff candidate audit and controlled pairing.

The audit is read-only by default.  It takes the 49 dormant/dropped single-
chapter setup notes, retrieves only *later-chapter* scenes through the story
graph, lets a model select at most one candidate id per setup, then validates
the candidate quote with L1.  L2 is a separate, explicit step.  Only an
``--apply`` run may promote a pair, and it must preserve graph coverage
invariants.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.judgment_layer import (
    NOTE_TYPE_LIMITS,
    match_quote,
    validate_note_evidence,
    verify_notes_l2,
)

AUDIT_VERSION = 1
DEFAULT_TOP_N = 5
DEFAULT_SELECTION_BATCH = 8
DEFAULT_L2_BATCH = 10
MAX_SELECTION_PROMPT_CHARS = 45000
DEFAULT_AUDIT_PATH = "output/setup_payoff_candidates.json"


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _chapter_order_key(chapter: str) -> tuple:
    import re as _re

    m = _re.fullmatch(r"([^\d]*)(\d+)", str(chapter or ""))
    if m:
        return (m.group(1), int(m.group(2)))
    return (str(chapter or ""), -1)


def is_unpaired_setup(note: Dict[str, Any]) -> bool:
    if not isinstance(note, dict) or str(note.get("type") or "") != "setup_payoff":
        return False
    if str(note.get("status") or "") == "active":
        return False
    meta = note.get("meta") or {}
    pairing = str(note.get("pairing_status") or meta.get("pairing_status") or "")
    if pairing in ("paired", "terminal_unpaired"):
        return False
    if pairing == "unpaired":
        return True
    chapters = {str(c) for c in (note.get("chapters") or []) if str(c)}
    evidence_chapters = {str(e.get("chapter") or "") for e in (note.get("evidence") or [])
                         if isinstance(e, dict) and e.get("chapter")}
    return len(chapters | evidence_chapters) < 2


def _strip_setup_note(note: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("id", "type", "chapters", "claim", "evidence", "confidence",
            "status", "pairing_status", "meta", "source_model", "created_at",
            "l1_reject_reason", "l1_note")
    return {k: copy.deepcopy(note.get(k)) for k in keep if k in note}


def build_setup_payoff_audit(profile: Dict[str, Any],
                             index: Any,
                             top_n: int = DEFAULT_TOP_N,
                             max_hops: int = 2) -> Dict[str, Any]:
    """Build deterministic candidate lists without mutating profile or graph."""
    notes = [n for n in (profile.get("setup_payoff") or [])
             if isinstance(n, dict) and is_unpaired_setup(n)]
    cases: List[Dict[str, Any]] = []
    for note in sorted(notes, key=lambda n: (_chapter_order_key(
            str((n.get("chapters") or [""])[0])), str(n.get("id") or ""))):
        result = index.setup_payoff_candidates(note, top_n=top_n,
                                               max_hops=max_hops)
        case = dict(result)
        case.update({
            "claim": str(note.get("claim") or ""),
            "confidence": note.get("confidence"),
            "status": str(note.get("status") or ""),
            "pairing_status": str(note.get("pairing_status")
                                  or (note.get("meta") or {}).get("pairing_status") or ""),
            "setup_evidence": copy.deepcopy(note.get("evidence") or []),
            "setup_note": _strip_setup_note(note),
            "selection": None,
        })
        if not case.get("candidates") and not case.get("terminal"):
            case["terminal_reason"] = "no_seed_or_no_later_candidate"
        cases.append(case)
    summary = summarize_setup_payoff_audit(cases)
    return {
        "audit_version": AUDIT_VERSION,
        "stage": "c0_candidate_audit",
        "generated_at": _now(),
        "settings": {"top_n": int(top_n), "max_hops": int(max_hops)},
        "graph": {
            "schema_version": getattr(index, "graph", {}).get("schema_version")
            if isinstance(getattr(index, "graph", None), dict) else None,
            "fingerprint": getattr(index, "graph", {}).get("fingerprint")
            if isinstance(getattr(index, "graph", None), dict) else None,
        },
        "summary": summary,
        "cases": cases,
    }


def summarize_setup_payoff_audit(cases_or_audit: Any) -> Dict[str, Any]:
    cases = (cases_or_audit.get("cases") if isinstance(cases_or_audit, dict)
             else cases_or_audit) or []
    selected = 0
    l1_ok = 0
    l2_confirmed = 0
    eligible = 0
    none = 0
    errors = 0
    candidates_total = 0
    for case in cases:
        candidates_total += len(case.get("candidates") or [])
        sel = case.get("selection") or {}
        if sel.get("candidate_id"):
            selected += 1
        elif sel:
            none += 1
        if (sel.get("l1") or {}).get("ok"):
            l1_ok += 1
        if str(sel.get("l2_verdict") or "") == "confirmed":
            l2_confirmed += 1
        if bool(sel.get("promotion_eligible")):
            eligible += 1
        if sel.get("error"):
            errors += 1
    return {
        "setups": len(cases),
        "actionable": sum(1 for c in cases if c.get("candidates")),
        "terminal": sum(1 for c in cases if c.get("terminal")),
        "with_candidates": sum(1 for c in cases if c.get("candidates")),
        "candidates_total": candidates_total,
        "selected": selected,
        "none": none,
        "l1_ok": l1_ok,
        "l2_confirmed": l2_confirmed,
        "promotion_eligible": eligible,
        "selection_errors": errors,
    }


def _selection_payload(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for case in cases:
        candidates = []
        for cand in (case.get("candidates") or [])[:DEFAULT_TOP_N]:
            candidates.append({
                "candidate_id": cand.get("candidate_id"),
                "chapter": cand.get("chapter"),
                "score": cand.get("score"),
                "quote": cand.get("anchor_quote"),
                "excerpt": str(cand.get("excerpt") or "")[:300],
            })
        payload.append({
            "setup_id": case.get("setup_id"),
            "setup_chapter": case.get("setup_chapter"),
            "claim": case.get("claim"),
            "setup_evidence": [str(e.get("quote") or "")[:240]
                               for e in (case.get("setup_evidence") or [])[:2]],
            "candidates": candidates,
        })
    return payload


def build_setup_selection_prompt(cases: Sequence[Dict[str, Any]]) -> str:
    payload = _selection_payload(cases)
    return f"""你是长篇小说的伏笔回收审计员。每个 setup 是较早章节留下的承诺、悬念或预置条件；候选场景全部来自更后章节。

任务：为每个 setup 最多选择一个真正“回收/兑现”它的候选场景。只允许选择候选 ID，不得发明 ID、章节或引文。

判定规则：
1. payoff：候选场景直接回答、兑现、反转或明确推进了 setup 提出的具体悬念/承诺。
2. none：候选只共享人物、地点、主题、词汇，或只是泛化背景，不能算回收；不确定时一律 none。
3. 不得因为候选“看起来相关”就选；必须能说清 setup 的哪个具体点在后文被兑现。
4. 同一 setup 最多选一个候选。
5. 输出只含 setup_id、candidate_id、verdict、reason；不要复制 quote 或 excerpt。

只输出 JSON：
{{"decisions":[{{"setup_id":"...","candidate_id":"cand-001 或 null","verdict":"payoff|none","reason":"≤60字"}}]}}

【setup 与候选】
{json.dumps(payload, ensure_ascii=False)}
"""


def _parse_selection_decisions(raw: str) -> Dict[str, Dict[str, Any]]:
    from src.utils.llm_adapter import robust_json_loads

    try:
        data = robust_json_loads(raw)
    except Exception:
        return {}
    if isinstance(data, dict) and isinstance(data.get("decisions"), list):
        items = data["decisions"]
    elif isinstance(data, list):
        items = data
    else:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("setup_id") or "").strip()
        if not sid:
            continue
        cid = item.get("candidate_id")
        cid = "" if cid is None else str(cid).strip()
        verdict = str(item.get("verdict") or ("payoff" if cid else "none")).strip().lower()
        if verdict not in ("payoff", "none"):
            verdict = "none"
        out[sid] = {"candidate_id": cid, "verdict": verdict,
                    "reason": str(item.get("reason") or "").strip()[:160]}
    return out


def _validate_candidate_l1(candidate: Dict[str, Any],
                           chapter_texts: Dict[str, str],
                           min_similarity: float = 0.9) -> Dict[str, Any]:
    chapter = str(candidate.get("chapter") or "")
    quote = str(candidate.get("anchor_quote") or "")
    text = chapter_texts.get(chapter) or ""
    hit = match_quote(text, quote, min_similarity=min_similarity,
                      preferred_offset=candidate.get("start"))
    return {
        "ok": bool(hit.get("ok")),
        "match_kind": hit.get("match_kind", ""),
        "match_score": hit.get("match_score", 0.0),
        "offset": hit.get("offset"),
        "end": hit.get("end"),
        "matched_quote": hit.get("matched_quote", ""),
        "unique": hit.get("unique", False),
        "reason": hit.get("reason", ""),
    }


def select_setup_payoff_candidates(audit: Dict[str, Any],
                                   chapter_texts: Dict[str, str],
                                   llm_generate: Callable[[str], str],
                                   batch_size: int = DEFAULT_SELECTION_BATCH,
                                   min_similarity: float = 0.9) -> Dict[str, Any]:
    """Ask a model to select candidate IDs; code performs the L1 check."""
    out = copy.deepcopy(audit)
    cases = [c for c in (out.get("cases") or []) if c.get("candidates")]
    by_id = {str(c.get("setup_id") or ""): c for c in cases}
    batch_size = max(1, min(10, int(batch_size or DEFAULT_SELECTION_BATCH)))
    errors: List[str] = []

    def _run_batch(batch: Sequence[Dict[str, Any]]) -> None:
        prompt = build_setup_selection_prompt(batch)
        if len(prompt) > MAX_SELECTION_PROMPT_CHARS and len(batch) > 1:
            mid = max(1, len(batch) // 2)
            _run_batch(batch[:mid])
            _run_batch(batch[mid:])
            return
        try:
            raw = llm_generate(prompt)
            decisions = _parse_selection_decisions(raw or "")
            parse_error = "" if decisions else "empty_or_invalid_json"
        except Exception as exc:  # noqa: BLE001 - audit is fail-open
            decisions = {}
            parse_error = f"llm_error:{type(exc).__name__}: {exc}"
        if parse_error:
            errors.append(parse_error)
        for case in batch:
            sid = str(case.get("setup_id") or "")
            decision = decisions.get(sid)
            if not decision:
                case["selection"] = {"verdict": "none",
                                     "reason": "not_returned",
                                     "error": parse_error}
                continue
            cid = str(decision.get("candidate_id") or "")
            verdict = str(decision.get("verdict") or "none")
            candidate = next((c for c in (case.get("candidates") or [])
                              if str(c.get("candidate_id") or "") == cid), None)
            if verdict != "payoff" or candidate is None:
                case["selection"] = {"verdict": "none",
                                     "candidate_id": "",
                                     "reason": decision.get("reason") or
                                               ("invalid_candidate_id" if cid else "no_payoff"),
                                     "error": parse_error}
                continue
            l1 = _validate_candidate_l1(candidate, chapter_texts, min_similarity)
            case["selection"] = {
                "verdict": "payoff",
                "candidate_id": cid,
                "reason": decision.get("reason") or "",
                "candidate": candidate,
                "l1": l1,
                "error": parse_error,
            }

    for base in range(0, len(cases), batch_size):
        _run_batch(cases[base:base + batch_size])
    out["stage"] = "c0_llm_selected"
    out["selection_errors"] = errors[-12:]
    out["summary"] = summarize_setup_payoff_audit(out)
    return out


def _assemble_pair_note(case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selection = case.get("selection") or {}
    candidate = selection.get("candidate") or {}
    l1 = selection.get("l1") or {}
    if selection.get("verdict") != "payoff" or not l1.get("ok"):
        return None
    raw = case.get("setup_note") or {}
    if not raw:
        return None
    note = copy.deepcopy(raw)
    setup_ch = str(case.get("setup_chapter") or "")
    payoff_ch = str(candidate.get("chapter") or "")
    if not setup_ch or not payoff_ch:
        return None
    if _chapter_order_key(setup_ch) >= _chapter_order_key(payoff_ch):
        return None
    evidence = [copy.deepcopy(e) for e in (note.get("evidence") or [])
                if isinstance(e, dict)]
    payoff_ev = {
        "chapter": payoff_ch,
        "quote": candidate.get("anchor_quote") or "",
        "offset": l1.get("offset"),
        "end": l1.get("end"),
        "match_kind": l1.get("match_kind") or "",
        "match_score": l1.get("match_score") or 0.0,
        "matched_quote": l1.get("matched_quote") or candidate.get("anchor_quote") or "",
        "unique": bool(l1.get("unique")),
        "role": "payoff",
        "scene_id": candidate.get("scene_id") or "",
    }
    evidence.append(payoff_ev)
    note["chapters"] = [setup_ch, payoff_ch]
    note["evidence"] = evidence
    note["status"] = "active"
    note["pairing_status"] = "paired"
    note.setdefault("meta", {})["pairing_status"] = "paired"
    note["meta"].update({
        "setup_id": str(note.get("id") or ""),
        "payoff_scene_id": candidate.get("scene_id") or "",
        "payoff_candidate_id": selection.get("candidate_id") or "",
        "payoff_reason": selection.get("reason") or "",
        "payoff_audit_version": AUDIT_VERSION,
        "paired_at": _now(),
    })
    return note


def verify_setup_payoff_selections(audit: Dict[str, Any],
                                   chapter_texts: Dict[str, str],
                                   llm_generate: Optional[Callable[[str], str]] = None,
                                   batch_size: int = DEFAULT_L2_BATCH,
                                   min_similarity: float = 0.9,
                                   max_pairs: int = NOTE_TYPE_LIMITS["setup_payoff"]
                                   ) -> Dict[str, Any]:
    """Assemble dual-chapter notes, run L1, then run L2 in <=10 batches."""
    out = copy.deepcopy(audit)
    drafted: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for case in (out.get("cases") or []):
        note = _assemble_pair_note(case)
        if note:
            drafted.append(note)
            by_id[str(note.get("id") or "")] = case
    drafted.sort(key=lambda n: -float(
        ((by_id.get(str(n.get("id") or "")) or {}).get("selection") or {})
        .get("candidate", {}).get("score") or 0.0))
    drafted = drafted[:max(1, int(max_pairs))]
    l1 = validate_note_evidence(drafted, chapter_texts,
                                min_similarity=min_similarity) if drafted else {"notes": [], "audit": {}}
    l1_rows = {str(n.get("id") or ""): n for n in (l1.get("notes") or [])}
    active_notes = [n for n in l1_rows.values() if n.get("status") == "active"]
    l2 = verify_notes_l2(active_notes, chapter_texts,
                         llm_generate=llm_generate, batch_size=batch_size) \
        if active_notes else {"notes": [], "audit": {}}
    l2_rows = {str(n.get("id") or ""): n for n in (l2.get("notes") or [])}
    for sid, case in by_id.items():
        selection = case.get("selection") or {}
        note = l2_rows.get(sid) or l1_rows.get(sid)
        if note is None:
            selection["l1"] = selection.get("l1") or {"ok": False}
            selection["l2_verdict"] = "missing"
            selection["promotion_eligible"] = False
            continue
        selection["pair_note"] = note
        selection["l1"] = {
            "ok": note.get("status") == "active" or not note.get("l1_reject_reason"),
            "reject_reason": note.get("l1_reject_reason") or note.get("l1_note") or "",
            "evidence": (note.get("evidence") or []),
        }
        verdict = str(note.get("l2_verdict") or "")
        selection["l2_verdict"] = verdict or "not_run"
        selection["l2_reason"] = note.get("l2_reason") or note.get("l2_reject_reason") or ""
        selection["promotion_eligible"] = bool(
            note.get("status") == "active" and verdict == "confirmed")
    out["stage"] = "c1_l1_l2_verified"
    out["summary"] = summarize_setup_payoff_audit(out)
    return out


def promote_setup_payoff_pairs(profile: Dict[str, Any],
                               audit: Dict[str, Any]) -> Dict[str, Any]:
    """Pure profile transform: activate only L1+L2 confirmed pairs, cap total."""
    prof = copy.deepcopy(profile)
    notes = [n for n in (prof.get("setup_payoff") or []) if isinstance(n, dict)]
    by_id = {str(n.get("id") or ""): n for n in notes}
    cases = [c for c in (audit.get("cases") or []) if isinstance(c, dict)]
    existing_active = sum(1 for n in notes if str(n.get("status") or "") == "active")
    slots = max(0, int(NOTE_TYPE_LIMITS["setup_payoff"]) - existing_active)
    eligible = []
    for case in cases:
        sel = case.get("selection") or {}
        if bool(sel.get("promotion_eligible")) and sel.get("pair_note"):
            eligible.append(case)
    eligible.sort(key=lambda c: -float(
        ((c.get("selection") or {}).get("candidate") or {}).get("score") or 0.0))
    activated: List[Dict[str, Any]] = []
    already_resolved: List[str] = []
    skipped_cap: List[str] = []
    for case in eligible:
        sid = str(case.get("setup_id") or "")
        note = by_id.get(sid)
        if note is None:
            continue
        if str(note.get("status") or "") == "active":
            already_resolved.append(sid)
            continue
        if len(activated) >= slots:
            skipped_cap.append(sid)
            continue
        pair_note = copy.deepcopy((case.get("selection") or {}).get("pair_note") or {})
        if not pair_note:
            continue
        note.clear()
        note.update(pair_note)
        activated.append({"setup_id": sid,
                          "chapters": note.get("chapters") or [],
                          "confidence": note.get("confidence"),
                          "payoff_scene_id": (note.get("meta") or {}).get("payoff_scene_id")})
    terminal: List[str] = []
    unpaired: List[str] = []
    for case in cases:
        sid = str(case.get("setup_id") or "")
        note = by_id.get(sid)
        if (note is None or sid in {x["setup_id"] for x in activated}
                or sid in already_resolved):
            continue
        if str(note.get("status") or "") == "active":
            already_resolved.append(sid)
            continue
        meta = note.setdefault("meta", {})
        if bool(case.get("terminal")):
            note["status"] = "dormant"
            note["pairing_status"] = "terminal_unpaired"
            meta["pairing_status"] = "terminal_unpaired"
            meta["terminal_reason"] = case.get("terminal_reason") or "no_later_chapter"
            terminal.append(sid)
        else:
            note["pairing_status"] = "unpaired"
            meta["pairing_status"] = "unpaired"
            sel = case.get("selection") or {}
            meta["pairing_attempt"] = {
                "verdict": sel.get("verdict") or "not_run",
                "reason": sel.get("reason") or sel.get("error") or "",
                "l1_ok": bool((sel.get("l1") or {}).get("ok")),
                "l2_verdict": sel.get("l2_verdict") or "not_run",
            }
            unpaired.append(sid)
    prof["setup_payoff"] = notes
    return {
        "profile": prof,
        "activated": activated,
        "already_resolved": already_resolved,
        "terminal": terminal,
        "skipped_cap": skipped_cap,
        "unpaired": unpaired,
        "existing_active": existing_active,
        "available_slots": slots,
    }


def default_setup_payoff_audit_generate(prompt: str) -> str:
    from src.config import get_config
    from src.utils.llm_adapter import get_llm_client_for_task, get_role_model_name

    cfg = get_config()
    task = "setup_payoff_audit" if "setup_payoff_audit" in (cfg.task_routing or {}) \
        else "style_profiling_merge"
    model_key, params = cfg.resolve_task_model(task)
    role = cfg.get_task_role(task)
    llm = get_llm_client_for_task(task)
    model_name = get_role_model_name(role) or model_key
    return llm.generate(prompt, model_name=model_name, **params)


def format_setup_payoff_audit_summary(audit: Dict[str, Any]) -> str:
    m = audit.get("summary") or summarize_setup_payoff_audit(audit)
    lines = [
        "🧭 setup-payoff audit: setups=%s actionable=%s terminal=%s candidates=%s" % (
            m.get("setups", 0), m.get("actionable", 0),
            m.get("terminal", 0), m.get("candidates_total", 0)),
    ]
    if m.get("selected") or m.get("l1_ok") or m.get("l2_confirmed"):
        lines.append("   selected=%s l1_ok=%s l2_confirmed=%s eligible=%s" % (
            m.get("selected", 0), m.get("l1_ok", 0),
            m.get("l2_confirmed", 0), m.get("promotion_eligible", 0)))
    return "\n".join(lines)
