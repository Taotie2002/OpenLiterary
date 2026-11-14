#!/usr/bin/env python3
"""LLM-assisted, mechanically-validated repair of cross-person names.

The LLM only diagnoses.  An edit is applied only when all of the following hold:

1. the reported `wrong` token is a known suspect person/role surface;
2. the reported `right` token is a curated target surface;
3. the English source for this exact chunk contains the target evidence
   (Jessica for 杰西卡, Leto for 雷托, Truthsayer for 真言者, ...); a match
   only in the LLM's explanation is not enough;
4. the exact `context` occurs in the final text and contains `wrong`;
5. the replacement is a plain token substitution at that exact occurrence.

If any check fails, the issue is recorded as rejected.  No free-form rewrite is
ever performed.  Dry-run is the default; `--apply` writes final.json with
backups and metadata.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.llm_adapter import get_llm_client, get_role_model_name, robust_json_loads

SUSPECT_TOKENS = {
    "盖乌斯",
    "盖乌斯夫人",
    "盖乌斯·海伦·莫希阿姆",
    "莫希阿姆",
    "海伦",
    "保罗",
    "保罗·厄崔迪",
    "保罗·阿特雷迪斯",
}
PROPER_TARGETS = {
    "杰西卡": ("Jessica",),
    "杰西卡夫人": ("Jessica",),
    "雷托": ("Leto",),
    "雷托公爵": ("Leto",),
    "雷托·厄崔迪": ("Leto",),
    "雷托·厄崔迪公爵": ("Leto",),
    "雷托·厄崔迪斯": ("Leto",),
    "保罗": ("Paul",),
    "保罗·厄崔迪": ("Paul",),
    "盖乌斯": ("Gaius", "Mohiam"),
    "盖乌斯夫人": ("Gaius", "Mohiam"),
    "莫希阿姆": ("Mohiam",),
    "皮特": ("Piter",),
    "彼得": ("Piter",),
}
ROLE_TARGETS = {
    "真言者": ("Truthsayer",),
    "真言师": ("Truthsayer",),
    "老妇人": ("old woman", "old witch", "old Truthsayer"),
    "老妪": ("old woman", "old witch"),
    "母亲": ("mother",),
    "你母亲": ("your mother",),
    "他的母亲": ("his mother",),
}
ALL_TARGETS = set(PROPER_TARGETS) | set(ROLE_TARGETS)
PERSON_CLUSTER = {
    "保罗": "PAUL",
    "保罗·厄崔迪": "PAUL",
    "保罗·阿特雷迪斯": "PAUL",
    "雷托": "LETO",
    "雷托公爵": "LETO",
    "雷托·厄崔迪": "LETO",
    "雷托·厄崔迪公爵": "LETO",
    "雷托·厄崔迪斯": "LETO",
    "杰西卡": "JESSICA",
    "杰西卡夫人": "JESSICA",
    "盖乌斯": "MOHIAM",
    "盖乌斯夫人": "MOHIAM",
    "盖乌斯·海伦·莫希阿姆": "MOHIAM",
    "莫希阿姆": "MOHIAM",
    "海伦": "MOHIAM",
    "皮特": "PITER",
    "彼得": "PITER",
}
EVIDENCE_TERMS = {
    "杰西卡": ("Jessica",),
    "杰西卡夫人": ("Jessica",),
    "雷托": ("Leto",),
    "雷托公爵": ("Leto",),
    "雷托·厄崔迪": ("Leto",),
    "雷托·厄崔迪公爵": ("Leto",),
    "雷托·厄崔迪斯": ("Leto",),
    "保罗": ("Paul", "Muad"),
    "保罗·厄崔迪": ("Paul", "Muad"),
    "盖乌斯": ("Gaius", "Mohiam"),
    "盖乌斯夫人": ("Gaius", "Mohiam"),
    "莫希阿姆": ("Mohiam",),
    "皮特": ("Piter",),
    "彼得": ("Piter",),
    "真言者": ("Truthsayer",),
    "真言师": ("Truthsayer",),
    "老妇人": ("old woman", "old witch", "old Truthsayer"),
    "老妪": ("old woman", "old witch"),
    "母亲": ("mother",),
    "你母亲": ("your mother",),
    "他的母亲": ("his mother",),
}
PROMPT_TEMPLATE = """你是中英文学翻译的专名校对器。下面给出英文原文、当前终稿、直译稿、文学稿。
只检查人物名/角色指称是否与英文原文一致。列出终稿中每一个错误的人物名。
输出严格 JSON：{{"issues":[{{"wrong":"终稿中的错误名字","right":"应按英文原文使用的名字","context":"终稿中该名字前后约30字","reason":"对应英文原句"}}]}}
规则：
- 只报人物名/角色错位，不报风格、标点、省略、增译。
- 如果终稿用代词而英文用名字，不算错。
- 如果终稿把角色称谓（如真言者、老妇人）换成具体名字但指代正确，不算错；反之亦然。
- 如果无法确定，不要报。
- 没有错误时输出 {{"issues":[]}}。

英文原文：
{source}

当前终稿：
{final}

直译稿：
{raw}

文学稿：
{literary}
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp-name-llm")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_sources(root: Path) -> Dict[str, str]:
    db = root / "db" / "workflow.db"
    out: Dict[str, str] = {}
    if not db.exists():
        return out
    con = sqlite3.connect(str(db))
    try:
        for cid, text in con.execute("SELECT chunk_id, text_content FROM chunk_tasks"):
            if isinstance(cid, str) and isinstance(text, str):
                out[cid] = text
    finally:
        con.close()
    return out


def _evidence_ok(right: str, reason: str) -> bool:
    needles = EVIDENCE_TERMS.get(right)
    if not needles:
        return False
    hay = (reason or "").casefold()
    return any(n.casefold() in hay for n in needles)


def _source_evidence_ok(right: str, source: str) -> bool:
    """Require target-name evidence in the actual source, not model prose."""
    needles = EVIDENCE_TERMS.get(right)
    if not needles or not source:
        return False
    for needle in needles:
        if re.search(r"\b" + re.escape(needle) + r"\b", source, re.IGNORECASE):
            return True
    return False


def _validate_issue(issue: Dict[str, Any], final: str, source: str,
                    ) -> Tuple[Optional[Dict[str, Any]], str]:
    wrong = str(issue.get("wrong") or "").strip()
    right = str(issue.get("right") or "").strip()
    context = str(issue.get("context") or "").strip()
    reason = str(issue.get("reason") or "").strip()
    if not wrong or not right or wrong == right:
        return None, "empty_or_same"
    if wrong not in SUSPECT_TOKENS:
        return None, f"wrong_not_suspect:{wrong}"
    if right not in ALL_TARGETS:
        return None, f"right_not_target:{right}"
    if PERSON_CLUSTER.get(wrong) and PERSON_CLUSTER.get(wrong) == PERSON_CLUSTER.get(right):
        return None, f"same_person:{wrong}->{right}"
    if not context:
        return None, "empty_context"
    if wrong not in context:
        return None, "wrong_not_in_context"
    wrong_at = context.find(wrong)
    if context.find(wrong, wrong_at + 1) >= 0:
        return None, "ambiguous_wrong_in_context"
    if not _source_evidence_ok(right, source):
        return None, f"source_evidence_missing:{right}"
    if not _evidence_ok(right, reason):
        return None, f"evidence_mismatch:{right}"
    # Locate the exact context.  If it repeats, prefer the occurrence nearest
    # the end (LLM contexts usually include trailing quote/paragraph text).
    start = final.find(context)
    if start < 0:
        return None, "context_not_found"
    second = final.find(context, start + 1)
    if second >= 0:
        return None, "ambiguous_context"
    token_at = wrong_at
    if token_at < 0:
        return None, "token_offset_missing"
    abs_start = start + token_at
    if final[abs_start:abs_start + len(wrong)] != wrong:
        return None, "token_offset_mismatch"
    return {
        "wrong": wrong,
        "right": right,
        "reason": reason,
        "context": context,
        "start": abs_start,
        "end": abs_start + len(wrong),
    }, "ok"


def _apply_issues(final: str, issues: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    # Deduplicate by span; later duplicates lose.
    by_span: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for issue in issues:
        key = (issue["start"], issue["end"])
        if key not in by_span:
            by_span[key] = issue
    ordered = sorted(by_span.values(), key=lambda x: x["start"], reverse=True)
    out = final
    applied: List[Dict[str, Any]] = []
    for issue in ordered:
        s, e = issue["start"], issue["end"]
        if out[s:e] != issue["wrong"]:
            continue
        out = out[:s] + issue["right"] + out[e:]
        applied.append(issue)
    # Applying from the end keeps earlier offsets valid only when edits do not
    # change lengths before them.  Restore ascending order in the report.
    applied.sort(key=lambda x: x["start"])
    return out, applied


def _call_llm(llm: Any, model: str, source: str, final: str,
              raw: str, literary: str) -> Dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        source=source, final=final, raw=raw, literary=literary)
    raw_out = llm.generate(
        prompt, model, max_tokens=4000, temperature=0.0, effort="none")
    data = robust_json_loads(raw_out, dict)
    issues = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(issues, list):
        return {"issues": [], "raw": raw_out[:500], "parse_error": True}
    return {"issues": issues, "raw": raw_out[:500], "parse_error": False}


def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            key = str(data.get("key") or "")
            if key:
                out[key] = data
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return out


def _append_cache(path: Path, record: Dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# LLM 姓名诊断 + 机械校验报告",
        "",
        f"- 模式: {payload['mode']}",
        f"- 候选 chunk: {payload['selected']}",
        f"- LLM 完成: {payload['llm_done']}",
        f"- 接受 issue: {payload['accepted_issues']}",
        f"- 拒绝 issue: {payload['rejected_issues']}",
        f"- 已写入 chunk: {payload['applied_chunks']}",
        "",
        "LLM 只负责诊断；写入前校验错误名、正确名、英文证据、精确 context 与唯一位置。",
        "",
        "## 已接受",
        "",
    ]
    rows = payload.get("rows") or []
    accepted_rows = [r for r in rows if r.get("accepted")]
    if not accepted_rows:
        lines.append("（无）")
    for row in accepted_rows:
        lines.append(f"### `{row['chunk_id']}`")
        for issue in row["accepted"]:
            lines.append(f"- `{issue['wrong']}→{issue['right']}`：{issue['reason'][:220]}")
        lines.append("")
    lines.extend(["## 已拒绝（原因摘要）", ""])
    rej: Dict[str, int] = {}
    for row in rows:
        for issue in row.get("rejected") or []:
            key = str(issue.get("reason") or "unknown")
            rej[key] = rej.get(key, 0) + 1
    if not rej:
        lines.append("（无）")
    for key, count in sorted(rej.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {key}: {count}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 诊断 + 机械校验的姓名修复（默认 dry-run）")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--audit", default=None, help="name_grounding_audit.json")
    parser.add_argument("--chapter", default=None)
    parser.add_argument("--chunks", default=None, help="逗号分隔的 chunk_id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output"
    audit_path = Path(args.audit) if args.audit else output_dir / "name_grounding_audit.json"
    json_out = Path(args.json_out) if args.json_out else output_dir / "name_repair_llm_report.json"
    report = Path(args.report) if args.report else output_dir / "name_repair_llm_report.md"
    cache_path = Path(args.cache) if args.cache else output_dir / "name_repair_llm_cache.jsonl"
    sources = _load_sources(root)

    if args.chunks:
        chunk_ids = [x.strip() for x in args.chunks.split(",") if x.strip()]
    else:
        audit = _read_json(audit_path)
        if not audit:
            print(f"❌ 找不到审计文件: {audit_path}", flush=True)
            return 2
        chunk_ids = []
        for row in audit.get("rows") or []:
            cid = str(row.get("chunk_id") or "")
            if not cid:
                continue
            if args.chapter and not cid.startswith(args.chapter + "_"):
                continue
            chunk_ids.append(cid)
    chunk_ids = sorted(dict.fromkeys(chunk_ids))
    if args.limit is not None:
        chunk_ids = chunk_ids[:max(0, args.limit)]

    cache = _load_cache(cache_path)
    cache_lock = threading.Lock()
    llm = get_llm_client("literal_translator")
    model = get_role_model_name("literal_translator")
    now = _dt.datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    backup_root = output_dir / "repair_backups" / f"name_llm_{stamp}"

    tasks: List[Tuple[str, Path, str, Dict[str, str]]] = []
    for cid in chunk_ids:
        chapter = cid.split("_", 1)[0]
        final_path = output_dir / chapter / f"{cid}_final.json"
        final_data = _read_json(final_path)
        literary_data = _read_json(final_path.with_name(f"{cid}_literary.json"))
        raw_data = _read_json(final_path.with_name(f"{cid}_raw.json"))
        source = sources.get(cid, "")
        if not final_data or not literary_data or not raw_data or not source:
            continue
        texts = {
            "final": str(final_data.get("text") or ""),
            "literary": str(literary_data.get("text") or ""),
            "raw": str(raw_data.get("text") or ""),
        }
        if not texts["final"]:
            continue
        key = f"{cid}:{_sha256(texts['final'])}"
        cached = cache.get(key)
        if cached and not cached.get("parse_error"):
            tasks.append((cid, final_path, source, texts))
            continue
        tasks.append((cid, final_path, source, texts))

    def work(item: Tuple[str, Path, str, Dict[str, str]]) -> Dict[str, Any]:
        cid, final_path, source, texts = item
        key = f"{cid}:{_sha256(texts['final'])}"
        cached = cache.get(key)
        if cached and not cached.get("parse_error"):
            result = cached.get("result") or {}
            return {"chunk_id": cid, "key": key, "result": result, "cached": True}
        result = _call_llm(
            llm, model, source, texts["final"], texts["raw"], texts["literary"])
        record = {"key": key, "chunk_id": cid, "result": result,
                  "created_at": now.isoformat()}
        _append_cache(cache_path, record, cache_lock)
        return {"chunk_id": cid, "key": key, "result": result, "cached": False}

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(work, item) for item in tasks]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - per-chunk isolation
                results.append({"chunk_id": "?", "key": "", "result": {},
                                "error": f"{type(exc).__name__}: {exc}"})

    results.sort(key=lambda x: x.get("chunk_id", ""))
    rows: List[Dict[str, Any]] = []
    accepted_total = 0
    rejected_total = 0
    applied_chunks = 0
    for result in results:
        cid = result.get("chunk_id", "")
        if cid == "?":
            rows.append({"chunk_id": cid, "accepted": [], "rejected": [],
                         "error": result.get("error")})
            continue
        chapter = cid.split("_", 1)[0]
        final_path = output_dir / chapter / f"{cid}_final.json"
        final_data = _read_json(final_path) or {}
        final = str(final_data.get("text") or "")
        if not final:
            continue
        issues = (result.get("result") or {}).get("issues") or []
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for issue in issues:
            if not isinstance(issue, dict):
                rejected.append({"issue": issue, "reason": "not_object"})
                continue
            validated, reason = _validate_issue(issue, final, sources.get(cid, ""))
            if validated:
                accepted.append(validated)
            else:
                rejected.append({
                    "wrong": issue.get("wrong"),
                    "right": issue.get("right"),
                    "reason": reason,
                })
        if accepted:
            repaired, applied = _apply_issues(final, accepted)
            accepted_total += len(accepted)
            if args.apply and applied:
                backup_path = backup_root / chapter / final_path.name
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final_path, backup_path)
                final_data["text"] = repaired
                meta = final_data.get("metadata")
                if not isinstance(meta, dict):
                    meta = {}
                meta["name_llm_repair"] = {
                    "schema_version": 1,
                    "applied_at": now.isoformat(),
                    "source": "llm_diagnosis+mechanical_validation",
                    "issues": [
                        {k: x[k] for k in ("wrong", "right", "reason")}
                        for x in applied
                    ],
                    "backup": str(backup_path.relative_to(root)),
                }
                final_data["metadata"] = meta
                _atomic_write_json(final_path, final_data)
                applied_chunks += 1
            rows.append({
                "chunk_id": cid,
                "accepted": [
                    {k: x[k] for k in ("wrong", "right", "reason")}
                    for x in accepted
                ],
                "rejected": rejected,
                "applied": bool(args.apply and applied),
                "cached": result.get("cached", False),
            })
        else:
            rows.append({
                "chunk_id": cid,
                "accepted": [],
                "rejected": rejected,
                "applied": False,
                "cached": result.get("cached", False),
            })
        rejected_total += len(rejected)

    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "selected": len(chunk_ids),
        "llm_done": len(results),
        "accepted_issues": accepted_total,
        "rejected_issues": rejected_total,
        "applied_chunks": applied_chunks,
        "rows": rows,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, payload)
    print(json.dumps({
        "mode": payload["mode"],
        "selected": payload["selected"],
        "llm_done": payload["llm_done"],
        "accepted_issues": payload["accepted_issues"],
        "rejected_issues": payload["rejected_issues"],
        "applied_chunks": payload["applied_chunks"],
        "json": str(json_out),
        "report": str(report),
        "cache": str(cache_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
