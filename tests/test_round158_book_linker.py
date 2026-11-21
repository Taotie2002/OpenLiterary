"""Round 158: build manifest and whole-book link audit."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from src.artifact_freshness import chapter_manifest
from src.book_linker import (
    audit_book_links,
    build_manifest,
    update_manifest_merged,
    write_manifest,
)
from src.judgment_layer import sha256_text
from src.story_graph import build_story_graph
from src.style_profiler import load_profile
from src.task_logging import sha256_file, sha256_prefix


RUN_ID = "20261116T120000-12345"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    _write(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _make_workflow_db(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, chapter_id TEXT,"
            " text_content TEXT, state TEXT, retries INTEGER, quality_retries INTEGER,"
            " last_error TEXT, updated_at TEXT, backtrack_rounds INTEGER,"
            " retry_effort_overrides TEXT, pf_requeues INTEGER, rescue_attempts INTEGER,"
            " rescue_stage TEXT)"
        )
        for chunk_id, chapter_id, text, state in rows:
            con.execute(
                "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state,"
                " retries, quality_retries, updated_at) VALUES (?, ?, ?, ?, 0, 0, ?)",
                (chunk_id, chapter_id, text, state, "2026-11-16 12:00:00"),
            )
        con.commit()
    finally:
        con.close()


def _make_decision_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, canonical_name TEXT,"
            " source_names TEXT, variant_names TEXT, entity_type TEXT,"
            " pronoun_gender TEXT, confidence REAL, pending_review INTEGER,"
            " translation_strategy TEXT, first_seen_chunk TEXT, quarantined_names TEXT)"
        )
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, variant_names,"
            " entity_type, confidence, pending_review) VALUES (?, ?, '[]', 'person',"
            " 0.99, 0)",
            ("Paul", json.dumps(["Paul"])),
        )
        con.commit()
    finally:
        con.close()


def _make_run_log(root: Path) -> None:
    log_dir = root / "output" / "logs" / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = log_dir / f"{RUN_ID}.jsonl"
    events = [
        {"ts": "2026-11-16T12:00:00.000+08:00", "seq": 1, "run_id": RUN_ID,
         "command": "merge", "stage": "RUN", "event": "run_start", "status": "ok"},
        {"ts": "2026-11-16T12:00:01.000+08:00", "seq": 2, "run_id": RUN_ID,
         "command": "merge", "stage": "MERGE", "event": "stage_finished",
         "status": "ok"},
        {"ts": "2026-11-16T12:00:02.000+08:00", "seq": 3, "run_id": RUN_ID,
         "command": "merge", "stage": "RUN", "event": "run_end", "status": "ok"},
    ]
    _write(jsonl, "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
    _write(log_dir / f"{RUN_ID}.log",
           "[2026-11-16T12:00:00.000+08:00] run_start\n"
           f"[2026-11-16T12:00:02.000+08:00] run_end\n")
    seal = {
        "run_id": RUN_ID,
        "sealed_at": "2026-11-16T12:00:03.000+08:00",
        "jsonl_sha256": sha256_file(jsonl),
        "jsonl_bytes": jsonl.stat().st_size,
        "log_sha256": sha256_file(log_dir / f"{RUN_ID}.log"),
        "log_bytes": (log_dir / f"{RUN_ID}.log").stat().st_size,
        "seq_last": 3,
    }
    _write_json(log_dir / f"{RUN_ID}.sha256", seal)


def _make_fixture(tmp_path: Path) -> Path:
    root = tmp_path
    _write(root / "input" / "ch01.md", "# One\n\nalpha scene")
    _write(root / "input" / "ch02.md", "# Two\n\nbeta scene")
    _make_workflow_db(root / "db" / "workflow.db", [
        ("ch01_chunk000", "ch01", "alpha scene", "COMPLETED"),
        ("ch01_chunk001", "ch01", "omega scene", "COMPLETED"),
        ("ch02_chunk000", "ch02", "beta scene", "COMPLETED"),
    ])
    _make_decision_db(root / "db" / "decision_db.sqlite")
    _write_json(root / "output" / "ch01" / "ch01_chunk000_final.json",
                {"text": "阿尔法。\n", "metadata": {"fallback": False}})
    _write_json(root / "output" / "ch01" / "ch01_chunk001_final.json",
                {"text": "欧米伽。\n", "metadata": {"fallback": False}})
    _write_json(root / "output" / "ch02" / "ch02_chunk000_final.json",
                {"text": "贝塔。\n", "metadata": {"fallback": False}})
    hashes = {cid: rec["sha256"]
              for cid, rec in chapter_manifest(root / "input").items()}
    profile = {
        "schema_version": 2,
        "source": "known_work",
        "generation": {
            "model": "test", "params": {"mode": "deep"},
            "prompt_version": "r131.deep.v3",
            "inputs": {"chapters": hashes, "profile_version": 2,
                       "prompt_version": "r131.deep.v3"},
        },
        "voice_notes": [{
            "id": "voice-1", "type": "voice", "chapters": ["ch01"],
            "claim": "保持克制", "status": "active", "l2_verdict": "confirmed",
            "evidence": [{"chapter": "ch01", "quote": "alpha scene",
                          "match_kind": "exact"}],
        }],
    }
    _write_json(root / "input" / "book_profile.json", profile)
    prof = load_profile(root / "input" / "book_profile.json")
    graph = build_story_graph(root, profile=prof)
    _write_json(root / "input" / "story_graph.json", graph)
    _make_run_log(root)
    from src.book_lint import chapter_heading
    headings = [chapter_heading(str(root), cid) for cid in ("ch01", "ch02")]
    merged = f"# {headings[0]}\n\n阿尔法。\n\n欧米伽。\n\n# {headings[1]}\n\n贝塔。\n"
    _write(root / "output" / "book.md", merged)
    return root


def _build(root: Path) -> Path:
    manifest = build_manifest(root, root / "output" / "book.md", run_id=RUN_ID)
    path = root / "output" / "build_manifest.json"
    write_manifest(manifest, path)
    return path


def test_book_link_fixture_is_green(tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    res = audit_book_links(root, manifest_path, require_sealed=True,
                           require_current=True)
    assert res["ok"] is True, res
    assert res["status"] == "pass"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chapter_order"] == ["ch01", "ch02"]
    assert manifest["totals"]["db_chunks"] == 3
    assert manifest["totals"]["finals_missing"] == 0
    assert manifest["run_evidence"]["sealed"] is True


def test_book_link_detects_deleted_and_duplicated_final(tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    (root / "output" / "ch01" / "ch01_chunk001_final.json").unlink()
    res = audit_book_links(root, manifest_path)
    assert res["ok"] is False
    assert any(name.startswith("final.ch01.ch01_chunk001")
               for name in res["failed"])

    _write_json(root / "output" / "ch01" / "ch01_chunk001_final.json",
                {"text": "欧米伽。\n", "metadata": {"fallback": False}})
    shutil.copy2(root / "output" / "ch01" / "ch01_chunk000_final.json",
                 root / "output" / "ch01" / "ch01_chunk999_final.json")
    res = audit_book_links(root, manifest_path)
    assert res["ok"] is False
    assert "chunks.ch01.orphan" in res["failed"]


def test_book_link_detects_changed_final_and_merged(tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    _write_json(root / "output" / "ch01" / "ch01_chunk000_final.json",
                {"text": "被篡改。\n", "metadata": {"fallback": False}})
    res = audit_book_links(root, manifest_path)
    assert res["ok"] is False
    assert any(name.startswith("final.ch01.ch01_chunk000") for name in res["failed"])

    _write_json(root / "output" / "ch01" / "ch01_chunk000_final.json",
                {"text": "阿尔法。\n", "metadata": {"fallback": False}})
    _write(root / "output" / "book.md", "# 伪标题\n\n换了内容\n")
    res = audit_book_links(root, manifest_path)
    assert res["ok"] is False
    assert "merged.hash" in res["failed"]
    assert "merged.headings" in res["failed"]


def test_book_link_detects_chapter_order_change(tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chapter_order"] = ["ch02", "ch01"]
    write_manifest(manifest, manifest_path)
    res = audit_book_links(root, manifest_path)
    assert res["ok"] is False
    assert "source.chapter_order" in res["failed"]


def test_book_link_detects_log_loss_seq_gap_and_seal_tamper(tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    jsonl = root / "output" / "logs" / "runs" / f"{RUN_ID}.jsonl"
    shutil.copy2(jsonl, tmp_path / "jsonl.keep")
    jsonl.unlink()
    res = audit_book_links(root, manifest_path, require_sealed=True)
    assert res["ok"] is False and "run.jsonl" in res["failed"]

    shutil.copy2(tmp_path / "jsonl.keep", jsonl)
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    events[1]["seq"] = 9
    _write(jsonl, "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
    res = audit_book_links(root, manifest_path, require_sealed=True)
    assert res["ok"] is False and "run.jsonl.events" in res["failed"]

    _make_run_log(root)
    seal_path = root / "output" / "logs" / "runs" / f"{RUN_ID}.sha256"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["jsonl_sha256"] = "0" * 64
    _write_json(seal_path, seal)
    res = audit_book_links(root, manifest_path, require_sealed=True)
    assert res["ok"] is False and "run.seal" in res["failed"]


def test_book_link_accepts_append_only_extension_but_rejects_truncation(
        tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    jsonl = root / "output" / "logs" / "runs" / f"{RUN_ID}.jsonl"
    seal_path = root / "output" / "logs" / "runs" / f"{RUN_ID}.sha256"
    sealed_bytes = jsonl.stat().st_size
    sealed_text = jsonl.read_text(encoding="utf-8")
    extra = [
        {"ts": "2026-11-16T12:00:04.000+08:00", "seq": 4, "run_id": RUN_ID,
         "command": "gate", "stage": "RUN", "event": "run_start",
         "status": "ok"},
        {"ts": "2026-11-16T12:00:05.000+08:00", "seq": 5, "run_id": RUN_ID,
         "command": "gate", "stage": "RUN", "event": "run_end",
         "status": "ok"},
    ]
    appended = sealed_text + "".join(
        json.dumps(event, ensure_ascii=False) + "\n" for event in extra)
    jsonl.write_text(appended, encoding="utf-8")
    current_bytes = jsonl.stat().st_size
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal.update({
        "jsonl_sha256": sha256_prefix(jsonl, current_bytes),
        "jsonl_bytes": current_bytes,
        "sealed_at": "2026-11-16T12:00:06.000+08:00",
        "seq_last": 5,
    })
    _write_json(seal_path, seal)

    res = audit_book_links(root, manifest_path, require_sealed=True)
    assert res["ok"] is True, res
    item = next(item for item in res["checks"] if item["name"] == "run.seal")
    assert item["details"]["extension_bytes"] == current_bytes - sealed_bytes

    jsonl.write_bytes(jsonl.read_bytes()[:sealed_bytes - 1])
    res = audit_book_links(root, manifest_path, require_sealed=True)
    assert res["ok"] is False and "run.seal" in res["failed"]

    tampered = sealed_text.replace('"merge"', '"mergX"', 1) + "".join(
        json.dumps(event, ensure_ascii=False) + "\n" for event in extra)
    jsonl.write_text(tampered, encoding="utf-8")
    res = audit_book_links(root, manifest_path, require_sealed=True)
    assert res["ok"] is False and "run.seal" in res["failed"]


def test_book_link_requires_seal_when_requested(tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    (root / "output" / "logs" / "runs" / f"{RUN_ID}.sha256").unlink()
    res = audit_book_links(root, manifest_path, require_sealed=True)
    assert res["ok"] is False and "run.seal" in res["failed"]


def test_update_merged_records_history_and_refuses_changed_final(tmp_path: Path):
    root = _make_fixture(tmp_path)
    manifest_path = _build(root)
    _write(root / "output" / "book.md",
           (root / "output" / "book.md").read_text(encoding="utf-8") + "\n校订一行。\n")
    upd = update_manifest_merged(root, manifest_path, root / "output" / "book.md",
                                 stage="proofread")
    assert upd["ok"] is True
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["merged_history"][-1]["stage"] == "proofread"
    assert audit_book_links(root, manifest_path)["failed"] == []

    _write_json(root / "output" / "ch01" / "ch01_chunk000_final.json",
                {"text": "改后。\n", "metadata": {"fallback": False}})
    upd = update_manifest_merged(root, manifest_path, root / "output" / "book.md",
                                 stage="proofread")
    assert upd["ok"] is False and upd["error"] == "final_links_changed"


def test_final_gate_item_12_rejects_stale_final(tmp_path: Path):
    root = _make_fixture(tmp_path)
    _build(root)
    _write_json(root / "output" / "ch01" / "ch01_chunk000_final.json",
                {"text": "被替换。\n", "metadata": {"fallback": False}})
    from src.pipeline import run_final_gate
    res = run_final_gate(project_root=str(root))
    assert any(str(item).startswith("⑫") for item in res["failures"])
    assert int((res["stats"].get("book_link") or {}).get("failed") or 0) >= 1


def test_final_gate_item_12_missing_manifest_is_advisory_without_config(tmp_path: Path):
    root = _make_fixture(tmp_path)
    from src.pipeline import run_final_gate
    res = run_final_gate(project_root=str(root))
    assert not any(str(item).startswith("⑫") for item in res["failures"])
    assert any(str(item).startswith("⑫") for item in res["warnings"])
    assert res["stats"]["book_link"]["required"] is False
