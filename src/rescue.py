"""失败块自动救援与交付证据（Round 147）。

本模块只做四件事：
  1. 统一失败分类，决定从哪个阶段做定向 rescue；
  2. 维护 append-only 失败账本 output/rescue/failure_ledger.jsonl；
  3. 从当前 final.json 重建 QA_REPORT.md（带机器可核验 JSON）；
  4. 提供 gate 使用的 fallback/QA 一致性扫描。

救援本身由 TranslationPipeline 执行；这里不调用 LLM，也不修改正文。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.config import config as _config


RESCUE_SCHEMA_VERSION = 1
QA_SCHEMA_VERSION = 1
QA_JSON_BEGIN = "<!-- SAGATRANS_QA_JSON_BEGIN -->"
QA_JSON_END = "<!-- SAGATRANS_QA_JSON_END -->"

_API_MARKERS = (
    "api_failure", "api 调用失败", "connection", "timeout", "timed out",
    "rate limit", "quota", "429", "502", "503", "504", "no_content",
    "仅推理", "无正文",
)
_SENSITIVE_MARKERS = (
    "new_sensitive", "content_sensitive", "content policy", "content_policy",
    "sensitive", "内容敏感", "敏感内容",
)
_CJK_MARKERS = ("cjk_retry_exhausted", "cjk 比例过低", "cjk=", "raw_not_chinese")
_ENTITY_MARKERS = ("entity_violation", "实体一致性", "实体违规", "entity_hard_fail")
_QUALITY_MARKERS = (
    "audit_early_stop", "early_stop", "score_decline", "no_progress",
    "stagnation", "stagnated", "质量重试耗尽", "max_quality_retries",
    "safety_net_passed", "permanently_failed_after_max_retries",
)
_STRUCTURE_MARKERS = ("lint", "footnote", "脚注", "structure", "json 解析失败")


def rescue_config() -> Dict[str, Any]:
    """读取 pipeline.rescue，并补齐不影响旧配置的默认值。"""
    raw = _config.get("pipeline.rescue", {}) or {}
    cfg = dict(raw) if isinstance(raw, dict) else {}
    cfg.setdefault("enabled", True)
    cfg.setdefault("max_targeted_attempts", 2)
    cfg.setdefault("max_full_requeues", 1)
    cfg.setdefault("chapter_budget_seconds", 1800)
    return cfg


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_int_safe(value: Any, default: int) -> int:
    return _as_int(value, default)


def classify_failure(reason: str, stage: str = "") -> str:
    """把多种历史错误文本归并为稳定的救援类别。"""
    text = f"{stage or ''} {reason or ''}".lower()
    if any(m in text for m in _SENSITIVE_MARKERS):
        return "content_sensitive"
    if any(m in text for m in _STRUCTURE_MARKERS):
        return "structure"
    if any(m in text for m in _CJK_MARKERS):
        return "cjk"
    if any(m in text for m in _ENTITY_MARKERS):
        return "entity"
    if any(m in text for m in _API_MARKERS):
        return "api_or_no_content"
    if any(m in text for m in _QUALITY_MARKERS):
        return "quality"
    return "unknown"


def _artifact_exists(out_dir: Path, chunk_id: str, name: str) -> bool:
    return (out_dir / f"{chunk_id}_{name}.json").exists()


def recommended_resume_stage(task: Dict[str, Any], failure_class: str,
                              out_dir: Path, attempt: int = 0) -> str:
    """按 cheap-first 原则选择续跑阶段。

    第一轮优先复用 raw/literary 重新审计；第二轮才重写润色；没有任何可用
    中间产物时才回到直译/术语提取。这样评分/后端恢复类失败不会从零重跑。
    """
    cid = str(task.get("chunk_id") or "")
    has_raw = _artifact_exists(out_dir, cid, "raw")
    has_lit = _artifact_exists(out_dir, cid, "literary")
    last_error = str(task.get("last_error") or "")
    if failure_class == "content_sensitive":
        stage_match = re.match(
            r"^\[(EXTRACTING_TERMS|TRANSLATING_RAW|REWRITING_LITERARY)\]", last_error)
        if stage_match and stage_match.group(1) == "TRANSLATING_RAW":
            return "TRANSLATING_RAW"
        if has_raw:
            return "REWRITING_LITERARY"
        return "TRANSLATING_RAW"
    if failure_class == "api_or_no_content":
        m = re.match(r"^\[(EXTRACTING_TERMS|TRANSLATING_RAW|REWRITING_LITERARY|AUDITING|JUDGING)\]", last_error)
        if m:
            stage = m.group(1)
            if stage == "TRANSLATING_RAW" or not has_raw:
                return "TRANSLATING_RAW"
            if stage == "REWRITING_LITERARY" or not has_lit:
                return "REWRITING_LITERARY"
            return "AUDITING"
        if has_raw and has_lit:
            return "AUDITING"
        if has_raw:
            return "REWRITING_LITERARY"
        return "TRANSLATING_RAW"
    if failure_class == "cjk":
        return "REWRITING_LITERARY" if has_raw else "TRANSLATING_RAW"
    if failure_class in ("quality", "entity", "unknown") and attempt <= 0:
        if has_raw and has_lit:
            return "AUDITING"
        if has_raw:
            return "REWRITING_LITERARY"
        return "TRANSLATING_RAW"
    if has_raw:
        return "REWRITING_LITERARY"
    return "EXTRACTING_TERMS"


def _json_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def collect_final_status(root: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """扫描所有 final.json，返回 fallback 证据与损坏文件。

    这是 gate ⑨ 和 QA_REPORT 的唯一事实源。只读，不依赖 DB 状态或 QA 历史。
    """
    out_root = Path(output_dir) if output_dir else Path(root) / "output"
    finals = sorted(out_root.glob("ch*/*_final.json"))
    fallbacks: List[Dict[str, Any]] = []
    final_entries: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for fp in finals:
        try:
            raw = fp.read_bytes()
            data = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append({"path": str(fp), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(data, dict):
            errors.append({"path": str(fp), "error": "final root is not an object"})
            continue
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        fallback = bool(
            meta.get("fallback")
            or meta.get("quality_waived")
            or meta.get("quality_degraded")
            or data.get("fallback")
        )
        chunk_id = fp.stem.rsplit("_final", 1)[0]
        final_entries.append({
            "chunk_id": chunk_id,
            "chapter": fp.parent.name,
            "final_sha256": _json_digest(raw),
            "fallback": fallback,
        })
        if not fallback:
            continue
        fallbacks.append({
            "chunk_id": chunk_id,
            "chapter": fp.parent.name,
            "reason": str(meta.get("reason") or meta.get("judge_reason") or "unknown")[:500],
            "final_sha256": _json_digest(raw),
            "path": str(fp),
        })
    fallbacks.sort(key=lambda x: x["chunk_id"])
    final_entries.sort(key=lambda x: x["chunk_id"])
    return {
        "final_count": len(finals),
        "finals": final_entries,
        "fallbacks": fallbacks,
        "errors": errors,
    }


def write_qa_report(root: Path, output_dir: Optional[Path] = None,
                    qa_path: Optional[Path] = None) -> Dict[str, Any]:
    """从当前 final 集合重建 QA_REPORT.md；无 fallback 时仍写机器证据。"""
    root = Path(root)
    out_root = Path(output_dir) if output_dir else root / "output"
    qa_file = Path(qa_path) if qa_path else out_root / "QA_REPORT.md"
    status = collect_final_status(root, out_root)
    payload = {
        "schema_version": QA_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "final_count": status["final_count"],
        "finals": status["finals"],
        "fallbacks": status["fallbacks"],
        "errors": status["errors"],
    }
    lines = [
        "# 译文质量验收报告 (QA Report)",
        "",
        "本文件由当前 final.json 重新生成；正文文件不包含本报告内容。",
        "以下 JSON 块是 gate 的机器可核验证据，请勿手工编辑。",
        "",
        QA_JSON_BEGIN,
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        QA_JSON_END,
        "",
    ]
    fallbacks = status["fallbacks"]
    if fallbacks:
        lines.append(f"## Fallback 明细 ({len(fallbacks)} 个 chunk)")
        lines.append("")
        for item in fallbacks:
            lines.append(f"### {item['chunk_id']}")
            lines.append(f"- 状态: 未通过自动化验收（fallback 稿已收录）")
            lines.append(f"- 原因: {item['reason']}")
            lines.append(f"- final SHA-256: `{item['final_sha256']}`")
            lines.append("")
    else:
        lines.append("## 结论")
        lines.append("")
        lines.append("- 无 fallback chunk；全部 final 来自通过验收的候选稿。")
        lines.append("")
    if status["errors"]:
        lines.append(f"## 无法解析的 final ({len(status['errors'])})")
        lines.append("")
        for item in status["errors"]:
            lines.append(f"- `{item['path']}`: {item['error']}")
        lines.append("")
    qa_file.parent.mkdir(parents=True, exist_ok=True)
    qa_file.write_text("\n".join(lines), encoding="utf-8")
    return {"ok": not status["errors"], "path": str(qa_file), **status}


def parse_qa_report(qa_file: Path) -> Optional[Dict[str, Any]]:
    """解析 QA 机器 JSON；旧报告或无机器块返回 None。"""
    try:
        text = Path(qa_file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    start = text.find(QA_JSON_BEGIN)
    end = text.find(QA_JSON_END, start + len(QA_JSON_BEGIN)) if start >= 0 else -1
    if start < 0 or end < 0:
        return None
    body = text[start + len(QA_JSON_BEGIN):end].strip()
    body = re.sub(r"^```(?:json)?\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"\s*```$", "", body)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def qa_report_mismatches(root: Path, output_dir: Optional[Path],
                         qa_file: Path) -> Tuple[List[str], Dict[str, Any]]:
    """对比 QA 机器证据与当前 final 集合，返回 stale/inconsistent 明细。"""
    status = collect_final_status(root, output_dir)
    expected = {x["chunk_id"]: x for x in status["fallbacks"]}
    payload = parse_qa_report(qa_file)
    problems: List[str] = []
    info: Dict[str, Any] = {
        "machine_readable": payload is not None,
        "expected_fallbacks": sorted(expected),
        "reported_fallbacks": [],
        "expected_final_count": status["final_count"],
        "reported_final_count": None,
    }
    if payload is not None:
        try:
            reported_final_count = int(payload.get("final_count"))
        except (TypeError, ValueError):
            reported_final_count = -1
        info["reported_final_count"] = reported_final_count
        if reported_final_count != status["final_count"]:
            problems.append(
                f"QA final_count={reported_final_count} != 当前 {status['final_count']}（报告陈旧）")
        reported_list = payload.get("fallbacks")
        reported_list = reported_list if isinstance(reported_list, list) else []
        reported: Dict[str, Dict[str, Any]] = {}
        for item in reported_list:
            if isinstance(item, dict) and item.get("chunk_id"):
                reported[str(item["chunk_id"])] = item
        info["reported_fallbacks"] = sorted(reported)
        reported_finals_raw = payload.get("finals")
        reported_finals: Dict[str, Dict[str, Any]] = {}
        if isinstance(reported_finals_raw, list):
            for item in reported_finals_raw:
                if isinstance(item, dict) and item.get("chunk_id"):
                    reported_finals[str(item["chunk_id"])] = item
        expected_finals = {x["chunk_id"]: x for x in (status.get("finals") or [])}
        if not reported_finals:
            problems.append("QA 缺少全量 final 哈希证据")
        else:
            missing_finals = sorted(set(expected_finals) - set(reported_finals))
            extra_finals = sorted(set(reported_finals) - set(expected_finals))
            if missing_finals:
                problems.append(f"QA 漏记 final: {missing_finals[:5]}")
            if extra_finals:
                problems.append(f"QA 残留过期 final: {extra_finals[:5]}")
            for cid in sorted(set(expected_finals) & set(reported_finals)):
                if str(expected_finals[cid].get("final_sha256") or "") != str(
                        reported_finals[cid].get("final_sha256") or ""):
                    problems.append(f"QA 与 final 哈希不一致: {cid}")
        missing = sorted(set(expected) - set(reported))
        extra = sorted(set(reported) - set(expected))
        if missing:
            problems.append(f"QA 漏记 fallback: {missing[:5]}")
        if extra:
            problems.append(f"QA 残留过期 fallback: {extra[:5]}")
        for cid in sorted(set(expected) & set(reported)):
            exp_sha = str(expected[cid].get("final_sha256") or "")
            got_sha = str(reported[cid].get("final_sha256") or "")
            if exp_sha != got_sha:
                problems.append(f"QA 与 final 哈希不一致: {cid}")
        reported_errors = payload.get("errors")
        reported_errors = reported_errors if isinstance(reported_errors, list) else []
        if len(reported_errors) != len(status["errors"]):
            problems.append(
                f"QA 损坏文件数 {len(reported_errors)} != 当前 {len(status['errors'])}")
        return problems, info

    # 旧格式只有章节级 ## 标题。无 fallback 时它仍可证明“当时全绿”；
    # 一旦当前出现 fallback，旧报告无法提供 chunk 级证据，必须重生成。
    try:
        qa_text = Path(qa_file).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        qa_text = ""
    legacy = [ln[3:].strip() for ln in qa_text.splitlines() if ln.startswith("## ")]
    info["legacy_headings"] = legacy
    if expected:
        problems.append("QA 无机器可读证据，无法核对 fallback chunk")
    elif legacy:
        problems.append(f"QA 旧报告仍含 {len(legacy)} 个 fallback 段（报告陈旧）")
    return problems, info


def ledger_path(root: Path) -> Path:
    return Path(root) / "output" / "rescue" / "failure_ledger.jsonl"


def append_rescue_event(root: Path, event: Dict[str, Any]) -> bool:
    """append-only 写入失败账本；任何 I/O 异常都 fail-open。"""
    try:
        path = ledger_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("schema_version", RESCUE_SCHEMA_VERSION)
        payload.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8") as fh:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            fh.write(line)
            fh.flush()
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        return True
    except OSError:
        return False


def read_rescue_events(root: Path) -> List[Dict[str, Any]]:
    """读取账本；损坏行跳过，不阻塞救援与 gate。"""
    path = ledger_path(root)
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
    except OSError:
        return []
    return events


def latest_events_by_chunk(root: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for event in read_rescue_events(root):
        cid = str(event.get("chunk_id") or "")
        if cid:
            latest[cid] = event
    return latest


def ledger_requeue_candidates(root: Path, chapter_id: str) -> List[str]:
    """返回账本中最近一次仍为隔离/失败的 PF 候选。"""
    out = []
    for cid, event in latest_events_by_chunk(root).items():
        if not cid.startswith(f"{chapter_id}_"):
            continue
        if str(event.get("result") or "") in ("quarantined", "failed"):
            out.append(cid)
    return sorted(out)


def failure_cohorts(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 failure_class + critic backend + 模型角色聚类，供批量修复与报告。"""
    cohorts: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for event in events:
        cls = str(event.get("failure_class") or "unknown")
        backend = str(event.get("critic_backend") or "unknown")
        model = str(event.get("model") or event.get("model_role") or "unknown")
        key = (cls, backend, model)
        row = cohorts.setdefault(key, {
            "failure_class": cls,
            "critic_backend": backend,
            "model": model,
            "count": 0,
            "chunks": [],
        })
        row["count"] += 1
        cid = str(event.get("chunk_id") or "")
        if cid and cid not in row["chunks"]:
            row["chunks"].append(cid)
    return sorted(cohorts.values(), key=lambda x: (-x["count"], x["failure_class"]))


def write_rescue_report(root: Path) -> Path:
    """把账本摘要写到 output/rescue/rescue_report.md。"""
    events = read_rescue_events(root)
    latest = latest_events_by_chunk(root)
    by_result: Dict[str, int] = {}
    for event in latest.values():
        result = str(event.get("result") or "unknown")
        by_result[result] = by_result.get(result, 0) + 1
    cohorts = failure_cohorts(events)
    lines = [
        "# Failure Rescue Report",
        "",
        f"- ledger events: {len(events)}",
        f"- chunks tracked: {len(latest)}",
        f"- latest results: {json.dumps(by_result, ensure_ascii=False, sort_keys=True)}",
        f"- ledger: `{ledger_path(root)}`",
        "",
        "## Root-cause cohorts",
        "",
    ]
    if cohorts:
        for row in cohorts[:40]:
            sample = ", ".join(row["chunks"][:8])
            lines.append(
                f"- {row['failure_class']} / {row['critic_backend']} / {row['model']}: "
                f"{row['count']} event(s); {sample}")
    else:
        lines.append("- 无失败账本记录。")
    lines.append("")
    path = Path(root) / "output" / "rescue" / "rescue_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
