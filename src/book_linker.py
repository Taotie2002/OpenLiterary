"""Round 158 build manifest and whole-book link audit.

The merge step is the only place where the delivery artifact is assembled from
per-chunk finals.  This module makes that assembly reproducible:

``output/build_manifest.json`` records the exact chapter order, DB tasks,
source hashes, final hashes, merged hash and run-log identity consumed by the
merge.  ``book-link-audit`` re-derives the same facts from disk and fails when
the merged book is no longer the product of the finals the manifest claims.

No LLM is involved.  Every check is deterministic and local.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.artifact_freshness import (
    chapter_manifest,
    graph_freshness,
    profile_fingerprint,
    profile_source_freshness,
)
from src.judgment_layer import atomic_write_json, sha256_text
from src.task_logging import now_iso, sha256_file, sha256_prefix, validate_jsonl

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = "output/build_manifest.json"
DEFAULT_AUDIT_JSON = "output/book_link_audit.json"
DEFAULT_AUDIT_MD = "output/book_link_audit.md"


def _natural_key(value: str) -> List[Any]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def _rel(root: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except (ValueError, OSError):
        return str(path)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_profile(root: Path):
    try:
        from src.style_profiler import load_profile
        return load_profile(Path(root) / "input" / "book_profile.json")
    except Exception:  # noqa: BLE001 - audit remains usable without a profile
        return None


def _load_graph(root: Path):
    try:
        from src.story_graph import load_story_graph
        return load_story_graph(Path(root) / "input" / "story_graph.json")
    except Exception:  # noqa: BLE001
        return None


def _chapter_heading(root: Path, chapter_id: str) -> str:
    try:
        from src.book_lint import chapter_heading
        value = str(chapter_heading(str(root), chapter_id) or "").strip()
        if value:
            return value
    except Exception:  # noqa: BLE001 - heading check is advisory if lint is absent
        pass
    return chapter_id


def _load_tasks(db_path: Path) -> Dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"rows": [], "error": f"workflow.db missing: {path}"}
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT chunk_id, chapter_id, state, text_content, updated_at "
            "FROM chunk_tasks ORDER BY chapter_id, chunk_id"
        ).fetchall()
        return {"rows": [dict(row) for row in rows], "error": ""}
    except sqlite3.Error as exc:
        return {"rows": [], "error": f"workflow.db unreadable: {exc}"}
    finally:
        con.close()


def _json_chars(path: Path) -> int:
    data = _read_json(path)
    if not isinstance(data, dict):
        return 0
    return len(str(data.get("text") or ""))


def _is_fallback(path: Path) -> bool:
    data = _read_json(path)
    if not isinstance(data, dict):
        return False
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    return bool(meta.get("fallback") or meta.get("quality_waived"))


def _run_evidence(root: Path, run_id: Optional[str]) -> Dict[str, Any]:
    log_dir = root / "output" / "logs" / "runs"
    rid = str(run_id or os.environ.get("SAGATRANS_RUN_ID") or "").strip()
    jsonl = log_dir / f"{rid}.jsonl" if rid else None
    text = log_dir / f"{rid}.log" if rid else None
    seal = log_dir / f"{rid}.sha256" if rid else None
    sealed = _read_json(seal) if seal and seal.exists() else None
    payload = {
        "run_id": rid,
        "jsonl_path": _rel(root, jsonl) if jsonl else "",
        "log_path": _rel(root, text) if text else "",
        "seal_path": _rel(root, seal) if seal else "",
        "jsonl_sha256": str((sealed or {}).get("jsonl_sha256") or ""),
        "jsonl_bytes": int((sealed or {}).get("jsonl_bytes") or 0),
        "log_sha256": str((sealed or {}).get("log_sha256") or ""),
        "log_bytes": int((sealed or {}).get("log_bytes") or 0),
        "sealed": bool(sealed),
        "sealed_at": str((sealed or {}).get("sealed_at") or ""),
        "seq_last": int((sealed or {}).get("seq_last") or 0),
    }
    return payload


def build_manifest(root: Path, merged_path: Path,
                   input_dir: Optional[Path] = None,
                   output_dir: Optional[Path] = None,
                   db_path: Optional[Path] = None,
                   run_id: Optional[str] = None) -> Dict[str, Any]:
    """Derive the manifest from the current disk state (no writes)."""
    root = Path(root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    out_dir = Path(output_dir) if output_dir else root / "output"
    dbp = Path(db_path) if db_path else root / "db" / "workflow.db"
    merged = Path(merged_path)

    chapter_sources = chapter_manifest(in_dir)
    chapter_order = sorted(chapter_sources, key=_natural_key)
    tasks = _load_tasks(dbp)
    rows_by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    for row in tasks.get("rows") or []:
        rows_by_chapter.setdefault(str(row.get("chapter_id") or ""), []).append(row)

    chapters: List[Dict[str, Any]] = []
    totals = {"chapters": len(chapter_order), "db_chunks": 0,
              "finals_present": 0, "finals_missing": 0,
              "fallback_finals": 0, "orphan_finals": 0}
    for cid in chapter_order:
        src = chapter_sources[cid]
        rows = sorted(rows_by_chapter.get(cid, []),
                      key=lambda r: _natural_key(r.get("chunk_id") or ""))
        manifest_chunks: List[Dict[str, Any]] = []
        expected_ids = set()
        for index, row in enumerate(rows):
            chunk_id = str(row.get("chunk_id") or "")
            expected_ids.add(chunk_id)
            final_path = out_dir / cid / f"{chunk_id}_final.json"
            present = final_path.exists()
            manifest_chunks.append({
                "index": index,
                "chunk_id": chunk_id,
                "state": str(row.get("state") or ""),
                "source_sha256": sha256_text(str(row.get("text_content") or "")),
                "source_chars": len(str(row.get("text_content") or "")),
                "final_path": _rel(root, final_path),
                "final_sha256": sha256_file(final_path) if present else "",
                "final_chars": _json_chars(final_path) if present else 0,
                "fallback": _is_fallback(final_path) if present else False,
                "updated_at": str(row.get("updated_at") or ""),
            })
            totals["db_chunks"] += 1
            if present:
                totals["finals_present"] += 1
                if _is_fallback(final_path):
                    totals["fallback_finals"] += 1
            else:
                totals["finals_missing"] += 1
        orphan_finals = []
        for final_path in sorted((out_dir / cid).glob("*_final.json"),
                                 key=lambda p: _natural_key(p.name)):
            chunk_id = final_path.name[:-len("_final.json")]
            if chunk_id not in expected_ids:
                orphan_finals.append({
                    "chunk_id": chunk_id,
                    "final_path": _rel(root, final_path),
                    "final_sha256": sha256_file(final_path),
                    "final_chars": _json_chars(final_path),
                })
        totals["orphan_finals"] += len(orphan_finals)
        chapters.append({
            "chapter_id": cid,
            "source_path": _rel(root, in_dir / f"{cid}.md"),
            "source_sha256": src["sha256"],
            "source_chars": src["chars"],
            "heading": _chapter_heading(root, cid),
            "chunks": manifest_chunks,
            "orphan_finals": orphan_finals,
        })

    merged_exists = merged.exists()
    merged_info = {
        "path": _rel(root, merged),
        "sha256": sha256_file(merged) if merged_exists else "",
        "chars": len(merged.read_text(encoding="utf-8")) if merged_exists else 0,
        "exists": merged_exists,
        "updated_at": now_iso(),
    }
    profile = _load_profile(root)
    graph = _load_graph(root)
    profile_status = (profile_source_freshness(profile, root, input_dir=in_dir)
                      if profile else {"status": "missing", "reasons": ["profile_missing"]})
    graph_status = (graph_freshness(graph, root, profile, input_dir=in_dir, db_path=dbp)
                    if graph else {"status": "missing", "reasons": ["graph_missing"]})
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "run_evidence": _run_evidence(root, run_id),
        "chapter_order": chapter_order,
        "chapters": chapters,
        "totals": totals,
        "merged": merged_info,
        "merged_history": [{
            "stage": "merge",
            "sha256": merged_info["sha256"],
            "chars": merged_info["chars"],
            "at": merged_info["updated_at"],
        }],
        "artifacts": {
            "profile_fingerprint": profile_fingerprint(profile) if profile else "",
            "profile_status": profile_status.get("status"),
            "profile_reasons": profile_status.get("reasons") or [],
            "profile_chapter_digests_complete": bool(
                (profile or {}).get("chapter_digests_complete")),
            "graph_fingerprint": str((graph or {}).get("fingerprint") or ""),
            "graph_status": graph_status.get("status"),
            "graph_reasons": graph_status.get("reasons") or [],
        },
        "workflow_db": {"path": _rel(root, dbp), "error": tasks.get("error") or ""},
    }


def write_manifest(manifest: Dict[str, Any], path: Path) -> Path:
    out = Path(path)
    atomic_write_json(out, manifest)
    return out


def update_manifest_run_evidence(root: Path, manifest_path: Path,
                                 run_id: str, seal: Dict[str, Any]) -> bool:
    """Attach the post-run sealed log hash without touching any final hash."""
    root = Path(root)
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    if not manifest:
        return False
    evidence = manifest.get("run_evidence") if isinstance(manifest.get("run_evidence"), dict) else {}
    recorded = str(evidence.get("run_id") or "")
    if recorded and recorded != run_id:
        return False
    log_dir = root / "output" / "logs" / "runs"
    evidence.update({
        "run_id": run_id,
        "jsonl_path": _rel(root, log_dir / f"{run_id}.jsonl"),
        "log_path": _rel(root, log_dir / f"{run_id}.log"),
        "seal_path": _rel(root, log_dir / f"{run_id}.sha256"),
        "jsonl_sha256": str(seal.get("jsonl_sha256") or ""),
        "jsonl_bytes": int(seal.get("jsonl_bytes") or 0),
        "log_sha256": str(seal.get("log_sha256") or ""),
        "log_bytes": int(seal.get("log_bytes") or 0),
        "sealed": True,
        "sealed_at": str(seal.get("sealed_at") or ""),
        "seq_last": int(seal.get("seq_last") or 0),
    })
    manifest["run_evidence"] = evidence
    manifest["updated_at"] = now_iso()
    atomic_write_json(manifest_path, manifest)
    return True


def _verify_final_links(manifest: Dict[str, Any], root: Path) -> List[str]:
    errors: List[str] = []
    for chapter in manifest.get("chapters") or []:
        for chunk in chapter.get("chunks") or []:
            path = root / str(chunk.get("final_path") or "")
            if not path.exists():
                errors.append(f"missing_final:{chunk.get('chunk_id')}")
            elif sha256_file(path) != chunk.get("final_sha256"):
                errors.append(f"changed_final:{chunk.get('chunk_id')}")
    return errors


def update_manifest_merged(root: Path, manifest_path: Path, merged_path: Path,
                           stage: str = "postprocess") -> Dict[str, Any]:
    """Record a merged-only transform (notes-normalize / proofread)."""
    root = Path(root)
    manifest_path = Path(manifest_path)
    merged = Path(merged_path)
    manifest = _read_json(manifest_path)
    if not manifest:
        return {"ok": False, "error": "manifest_missing"}
    if not merged.exists():
        return {"ok": False, "error": f"merged_missing:{merged}"}
    final_errors = _verify_final_links(manifest, root)
    if final_errors:
        return {"ok": False, "error": "final_links_changed",
                "details": final_errors[:20]}
    digest = sha256_file(merged)
    chars = len(merged.read_text(encoding="utf-8"))
    history = manifest.get("merged_history")
    if not isinstance(history, list):
        history = []
    history.append({"stage": str(stage), "sha256": digest,
                    "chars": chars, "at": now_iso()})
    manifest["merged_history"] = history[-20:]
    manifest["merged"] = {
        "path": _rel(root, merged),
        "sha256": digest,
        "chars": chars,
        "exists": True,
        "updated_at": now_iso(),
    }
    manifest["updated_at"] = now_iso()
    atomic_write_json(manifest_path, manifest)
    return {"ok": True, "sha256": digest, "chars": chars,
            "stage": str(stage), "manifest_path": str(manifest_path)}


def _item(name: str, status: str, message: str = "",
          **details: Any) -> Dict[str, Any]:
    return {"name": name, "status": status, "message": message,
            "details": details}


def _check_source_links(root: Path, manifest: Dict[str, Any],
                       input_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    current = chapter_manifest(input_dir)
    expected_order = sorted(current, key=_natural_key)
    declared = [str(c) for c in (manifest.get("chapter_order") or [])]
    if declared != expected_order:
        items.append(_item("source.chapter_order", "fail",
                           "manifest 章序与 input/ch*.md 不一致",
                           expected=expected_order, actual=declared))
    else:
        items.append(_item("source.chapter_order", "pass",
                           "章序一致", chapters=declared))
    by_id = {str(c.get("chapter_id")): c for c in manifest.get("chapters") or []}
    for cid in expected_order:
        entry = by_id.get(cid)
        if not entry:
            items.append(_item(f"source.{cid}", "fail", "manifest 缺章"))
            continue
        rec = current[cid]
        if entry.get("source_sha256") != rec["sha256"] or \
                int(entry.get("source_chars") or 0) != rec["chars"]:
            items.append(_item(f"source.{cid}", "fail", "章源哈希/字数已变化",
                               expected=rec, actual={
                                   "sha256": entry.get("source_sha256"),
                                   "chars": entry.get("source_chars")}))
        else:
            items.append(_item(f"source.{cid}", "pass", "章源一致"))
    for cid in declared:
        if cid not in current:
            items.append(_item(f"source.{cid}", "fail", "manifest 引用了不存在的章"))
    return items


def _check_chunk_links(root: Path, manifest: Dict[str, Any], db_path: Path,
                       output_dir: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    tasks = _load_tasks(db_path)
    if tasks.get("error"):
        return [_item("chunks.workflow_db", "fail", tasks["error"])]
    rows_by_chapter: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in tasks.get("rows") or []:
        rows_by_chapter.setdefault(str(row.get("chapter_id") or ""), {})[
            str(row.get("chunk_id") or "")] = row
    for chapter in manifest.get("chapters") or []:
        cid = str(chapter.get("chapter_id") or "")
        expected_rows = rows_by_chapter.get(cid, {})
        expected_ids = sorted(expected_rows, key=_natural_key)
        chunks = chapter.get("chunks") or []
        actual_ids = [str(c.get("chunk_id") or "") for c in chunks]
        if actual_ids != expected_ids:
            items.append(_item(f"chunks.{cid}.set", "fail",
                               "manifest chunk 集合/顺序与 DB 不一致",
                               expected=expected_ids, actual=actual_ids))
        else:
            items.append(_item(f"chunks.{cid}.set", "pass",
                               "chunk 集合与顺序一致", chunks=len(actual_ids)))
        seen = set()
        for index, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id") or "")
            if chunk_id in seen:
                items.append(_item(f"chunks.{cid}.{chunk_id}", "fail", "重复 chunk"))
                continue
            seen.add(chunk_id)
            row = expected_rows.get(chunk_id)
            if row is None:
                items.append(_item(f"chunks.{cid}.{chunk_id}", "fail", "DB 无此任务"))
                continue
            raw_index = chunk.get("index")
            try:
                chunk_index = int(raw_index) if raw_index is not None else -1
            except (TypeError, ValueError):
                chunk_index = -1
            if chunk_index != index:
                items.append(_item(f"chunks.{cid}.{chunk_id}", "fail",
                                   "chunk index 不连续"))
            if str(chunk.get("state") or "") != str(row.get("state") or ""):
                items.append(_item(f"chunks.{cid}.{chunk_id}", "fail",
                                   "DB 状态与 manifest 不一致",
                                   expected=row.get("state"),
                                   actual=chunk.get("state")))
            source_hash = sha256_text(str(row.get("text_content") or ""))
            if chunk.get("source_sha256") != source_hash:
                items.append(_item(f"chunks.{cid}.{chunk_id}", "fail",
                                   "DB source 与 manifest 哈希不一致"))
            final_path = output_dir / cid / f"{chunk_id}_final.json"
            if not final_path.exists():
                items.append(_item(f"final.{cid}.{chunk_id}", "fail", "final 缺失"))
            else:
                actual = sha256_file(final_path)
                if actual != chunk.get("final_sha256"):
                    items.append(_item(f"final.{cid}.{chunk_id}", "fail",
                                       "final 哈希已变化",
                                       expected=chunk.get("final_sha256"),
                                       actual=actual))
                elif _is_fallback(final_path) != bool(chunk.get("fallback")):
                    items.append(_item(f"final.{cid}.{chunk_id}", "fail",
                                       "fallback 标记与 manifest 不一致"))
        orphan = chapter.get("orphan_finals") or []
        disk_orphans = []
        known = set(expected_ids)
        for path in sorted((output_dir / cid).glob("*_final.json"),
                           key=lambda p: _natural_key(p.name)):
            chunk_id = path.name[:-len("_final.json")]
            if chunk_id not in known:
                disk_orphans.append(chunk_id)
        if orphan or disk_orphans:
            items.append(_item(f"chunks.{cid}.orphan", "fail",
                               "存在未登记的 final",
                               manifest=[o.get("chunk_id") for o in orphan],
                               disk=disk_orphans))
        else:
            items.append(_item(f"chunks.{cid}.orphan", "pass", "无孤立 final"))
    extra_chapters = sorted(set(rows_by_chapter) -
                            {str(c.get("chapter_id")) for c in manifest.get("chapters") or []},
                            key=_natural_key)
    if extra_chapters:
        items.append(_item("chunks.extra_chapters", "fail",
                           "DB 有 manifest 未覆盖的章", chapters=extra_chapters))
    return items


def _headings_from_merged(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            out.append(match.group(1).strip())
    return out


def _check_merged(root: Path, manifest: Dict[str, Any], merged_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    recorded = manifest.get("merged") if isinstance(manifest.get("merged"), dict) else {}
    if not merged_path.exists():
        return [_item("merged.exists", "fail", "merged 不存在")]
    actual_hash = sha256_file(merged_path)
    if actual_hash != recorded.get("sha256"):
        items.append(_item("merged.hash", "fail", "merged 哈希与 manifest 不一致",
                           expected=recorded.get("sha256"), actual=actual_hash))
    else:
        items.append(_item("merged.hash", "pass", "merged 哈希一致"))
    if Path(str(recorded.get("path") or "")) != Path(_rel(root, merged_path)):
        # The recorded path is intentionally relative; compare its normalized form.
        recorded_rel = str(recorded.get("path") or "")
        if recorded_rel and recorded_rel != _rel(root, merged_path):
            items.append(_item("merged.path", "fail", "merged 路径与 manifest 不一致",
                               expected=recorded_rel, actual=_rel(root, merged_path)))
    text = merged_path.read_text(encoding="utf-8")
    headings = _headings_from_merged(text)
    expected = [str(c.get("heading") or "") for c in manifest.get("chapters") or []]
    if expected and headings != expected:
        items.append(_item("merged.headings", "fail", "merged H1 章序/标题不一致",
                           expected=expected, actual=headings))
    else:
        items.append(_item("merged.headings", "pass", "merged H1 一致",
                           headings=headings))
    return items


def _check_artifacts(root: Path, manifest: Dict[str, Any],
                     require_current: bool) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    recorded = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    profile = _load_profile(root)
    graph = _load_graph(root)
    profile_fresh = (profile_source_freshness(profile, root) if profile
                     else {"status": "missing", "reasons": ["profile_missing"]})
    graph_fresh = (graph_freshness(graph, root, profile) if graph
                   else {"status": "missing", "reasons": ["graph_missing"]})
    for name, actual, current, recorded_fp in (
        ("profile", profile_fresh, profile_fingerprint(profile) if profile else "",
         recorded.get("profile_fingerprint")),
        ("graph", graph_fresh, str((graph or {}).get("fingerprint") or ""),
         recorded.get("graph_fingerprint")),
    ):
        if actual.get("status") == "fresh":
            items.append(_item(f"artifacts.{name}", "pass", "当前制品新鲜"))
        else:
            status = "fail" if require_current else "warn"
            items.append(_item(f"artifacts.{name}", status,
                               f"制品 {actual.get('status')}",
                               reasons=actual.get("reasons") or []))
        if recorded_fp and current and recorded_fp != current:
            items.append(_item(f"artifacts.{name}.fingerprint", "warn",
                               "manifest 记录与当前指纹不同",
                               expected=recorded_fp, actual=current))
    return items


def _check_run_evidence(root: Path, manifest: Dict[str, Any],
                        require_sealed: bool) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    evidence = manifest.get("run_evidence") if isinstance(manifest.get("run_evidence"), dict) else {}
    run_id = str(evidence.get("run_id") or "")
    if not run_id:
        return [_item("run.id", "fail", "manifest 未记录 run_id")]
    items.append(_item("run.id", "pass", "run_id 已记录", run_id=run_id))
    jsonl = root / str(evidence.get("jsonl_path") or "")
    text = root / str(evidence.get("log_path") or "")
    seal_path = root / str(evidence.get("seal_path") or "")
    if not jsonl.exists():
        items.append(_item("run.jsonl", "fail", "任务 JSONL 缺失", path=str(jsonl)))
        return items
    if not text.exists():
        items.append(_item("run.log", "fail", "任务文本日志缺失", path=str(text)))
    else:
        items.append(_item("run.log", "pass", "任务文本日志存在"))
    sealed = bool(evidence.get("sealed")) and seal_path.exists()
    validation = validate_jsonl(
        jsonl, expected_run_id=run_id, require_run_start=True,
        require_run_end=bool(sealed or require_sealed))
    if validation.get("ok"):
        items.append(_item("run.jsonl.events", "pass", "JSONL 事件序列有效",
                           seq_last=validation.get("seq_last")))
    else:
        items.append(_item("run.jsonl.events", "fail", "JSONL 事件序列无效",
                           errors=validation.get("errors") or []))
    if sealed:
        seal = _read_json(seal_path) or {}
        actual = sha256_file(jsonl)
        actual_bytes = jsonl.stat().st_size

        def _seal_matches(record: Dict[str, Any], kind: str) -> Tuple[bool, str]:
            expected = str(record.get("jsonl_sha256") or "")
            if not expected:
                return False, "missing_hash"
            expected_bytes = int(record.get("jsonl_bytes") or 0)
            if expected_bytes:
                if actual_bytes < expected_bytes:
                    return False, f"{kind}_truncated"
                if sha256_prefix(jsonl, expected_bytes) != expected:
                    return False, f"{kind}_prefix_changed"
                return True, ("exact" if actual_bytes == expected_bytes
                              else f"{kind}_append_only")
            # Legacy seals have no byte count: require an exact match so a
            # silently truncated or rewritten file can never pass.
            return (actual == expected), ("exact" if actual == expected
                                          else f"{kind}_hash_changed")

        seal_ok, seal_kind = _seal_matches(seal, "seal")
        evidence_ok, evidence_kind = (True, "absent")
        if evidence.get("jsonl_sha256"):
            evidence_ok, evidence_kind = _seal_matches(evidence, "manifest")
        if not seal_ok:
            items.append(_item("run.seal", "fail", "封存哈希与 JSONL 不一致",
                               expected=seal.get("jsonl_sha256"), actual=actual,
                               expected_bytes=seal.get("jsonl_bytes"),
                               actual_bytes=actual_bytes,
                               reason=seal_kind))
        elif not evidence_ok:
            items.append(_item("run.seal", "fail", "manifest 封存哈希已过期",
                               expected=evidence.get("jsonl_sha256"), actual=actual,
                               expected_bytes=evidence.get("jsonl_bytes"),
                               actual_bytes=actual_bytes,
                               reason=evidence_kind))
        else:
            extended = seal_kind.endswith("_append_only") or evidence_kind.endswith("_append_only")
            recorded_sizes = [
                int(value) for value in
                (seal.get("jsonl_bytes"), evidence.get("jsonl_bytes"))
                if int(value or 0) > 0
            ]
            base_bytes = min(recorded_sizes) if recorded_sizes else 0
            items.append(_item(
                "run.seal", "pass",
                ("封存哈希一致" if not extended
                 else "封存前缀一致（封存后仅追加审计事件）"),
                sealed_bytes=base_bytes,
                actual_bytes=actual_bytes,
                extension_bytes=max(0, actual_bytes - base_bytes),
            ))
    else:
        status = "fail" if require_sealed else "warn"
        items.append(_item("run.seal", status,
                           "运行尚未封存（交付审计需 --require-sealed）"))
    return items


def audit_book_links(root: Path, manifest_path: Optional[Path] = None,
                     merged_path: Optional[Path] = None,
                     input_dir: Optional[Path] = None,
                     output_dir: Optional[Path] = None,
                     db_path: Optional[Path] = None,
                     require_current: bool = False,
                     require_sealed: bool = False) -> Dict[str, Any]:
    """Re-derive every source -> chunk -> final -> merged link."""
    root = Path(root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    out_dir = Path(output_dir) if output_dir else root / "output"
    dbp = Path(db_path) if db_path else root / "db" / "workflow.db"
    mp = Path(manifest_path) if manifest_path else root / DEFAULT_MANIFEST_PATH
    manifest = _read_json(mp)
    if not manifest:
        return {
            "ok": False, "status": "failed", "manifest_path": _rel(root, mp),
            "checks": [_item("manifest.read", "fail",
                             f"无法读取 build manifest: {mp}")],
            "passed": [], "warnings": [], "failed": ["manifest.read"],
            "skipped": [],
        }
    checks: List[Dict[str, Any]] = []
    checks.extend(_check_source_links(root, manifest, in_dir))
    checks.extend(_check_chunk_links(root, manifest, dbp, out_dir))
    recorded_merged = (manifest.get("merged") or {}).get("path")
    actual_merged = (Path(merged_path) if merged_path else
                     (root / str(recorded_merged) if recorded_merged else None))
    if actual_merged is None:
        checks.append(_item("merged.path", "fail", "manifest 未记录 merged 路径"))
    else:
        checks.extend(_check_merged(root, manifest, actual_merged))
    checks.extend(_check_artifacts(root, manifest, require_current))
    checks.extend(_check_run_evidence(root, manifest, require_sealed))
    passed = [c["name"] for c in checks if c["status"] == "pass"]
    warnings = [c["name"] for c in checks if c["status"] == "warn"]
    failed = [c["name"] for c in checks if c["status"] == "fail"]
    skipped = [c["name"] for c in checks if c["status"] == "skip"]
    status = "failed" if failed else ("warning" if warnings else "pass")
    return {
        "ok": not failed,
        "status": status,
        "manifest_path": _rel(root, mp),
        "manifest_schema_version": manifest.get("schema_version"),
        "run_id": (manifest.get("run_evidence") or {}).get("run_id"),
        "checks": checks,
        "passed": passed,
        "warnings": warnings,
        "failed": failed,
        "skipped": skipped,
    }


def format_book_link_audit_md(result: Dict[str, Any]) -> str:
    lines = ["# 全书链接审计 (book_link_audit)", "",
             f"- 状态: {result.get('status')}",
             f"- manifest: {result.get('manifest_path')}",
             f"- run_id: {result.get('run_id') or '-'}",
             f"- passed: {len(result.get('passed') or [])}",
             f"- warnings: {len(result.get('warnings') or [])}",
             f"- failed: {len(result.get('failed') or [])}", ""]
    for check in result.get("checks") or []:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}.get(
            str(check.get("status")), "•")
        lines.append(f"- {icon} {check.get('name')}: {check.get('message')}")
        details = check.get("details") or {}
        if check.get("status") in {"fail", "warn"} and details:
            text = json.dumps(details, ensure_ascii=False, sort_keys=True)
            lines.append(f"  - {text[:500]}")
    return "\n".join(lines) + "\n"


def write_book_link_audit(result: Dict[str, Any], json_path: Optional[Path] = None,
                          md_path: Optional[Path] = None) -> Tuple[Path, Path]:
    jp = Path(json_path) if json_path else Path(DEFAULT_AUDIT_JSON)
    mp = Path(md_path) if md_path else Path(DEFAULT_AUDIT_MD)
    atomic_write_json(jp, result)
    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp = mp.with_name(mp.name + ".tmp")
    tmp.write_text(format_book_link_audit_md(result), encoding="utf-8")
    os.replace(tmp, mp)
    return jp, mp
