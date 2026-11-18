"""Round 158: consistency must not preempt the strict name-only closure."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import src.consistency as consistency


def _make_decision_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    try:
        con.executescript("""
            CREATE TABLE entity_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                source_names TEXT NOT NULL,
                variant_names TEXT,
                entity_type TEXT,
                confidence TEXT DEFAULT 'high',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pending_review INTEGER DEFAULT 0,
                review_reason TEXT,
                updated_at TIMESTAMP
            );
        """)
        rows = [
            ("佐恩侯爵", ["the Marquis", "Marquis Zorn"]),
            ("佐恩·巴列", ["Marquis Zorn", "Zorn Barley"]),
            ("佐恩·巴勒", ["Marquis Zorn", "Zorn Barley"]),
        ]
        for canonical, sources in rows:
            con.execute(
                "INSERT INTO entity_registry (canonical_name, source_names, "
                "variant_names, entity_type, confidence, pending_review) "
                "VALUES (?, ?, NULL, 'person', 'high', 0)",
                (canonical, json.dumps(sources, ensure_ascii=False)),
            )
        con.commit()
    finally:
        con.close()


def _make_workflow_db(path: Path, dirty_chunk: str) -> None:
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, state TEXT)")
        con.execute(
            "INSERT INTO chunk_tasks (chunk_id, state) VALUES (?, 'DIRTY')",
            (dirty_chunk,),
        )
        con.commit()
    finally:
        con.close()


def _write_final(root: Path, chunk_id: str, text: str) -> dict:
    chapter = chunk_id.split("_", 1)[0]
    out = root / "output" / chapter
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{chunk_id}_final.json"
    path.write_text(
        json.dumps({"text": text, "metadata": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "chapter": chapter,
        "file": path.name,
        "text": text,
        "path": str(path),
    }


def test_name_unification_defers_dirty_final_but_updates_completed(
        tmp_path: Path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    decision_db = db_dir / "decision_db.sqlite"
    workflow_db = db_dir / "workflow.db"
    _make_decision_db(decision_db)
    _make_workflow_db(workflow_db, "ch09_chunk000")
    dirty_final = _write_final(tmp_path, "ch09_chunk000", "佐恩·巴勒出场。")
    clean_final = _write_final(tmp_path, "ch09_chunk001", "佐恩·巴勒离开。")
    dirty_before = Path(dirty_final["path"]).read_bytes()
    monkeypatch.setattr(
        consistency, "_load_name_unification_cfg",
        lambda: {"enabled": True, "llm_vetting": False, "preferred_forms": {}},
    )

    dirty_ids = consistency._load_dirty_chunk_ids(workflow_db)
    result = consistency._apply_name_unification_pass(
        [dirty_final, clean_final], str(decision_db),
        report_path=tmp_path / "name_unification_report.md",
        cache_path=tmp_path / "cache.json",
        dry_run=False,
        skip_chunk_ids=dirty_ids,
    )

    assert dirty_ids == {"ch09_chunk000"}
    assert result["skipped_dirty_chunks"] == ["ch09_chunk000"]
    assert Path(dirty_final["path"]).read_bytes() == dirty_before
    assert "佐恩·巴列离开" in json.loads(
        Path(clean_final["path"]).read_text(encoding="utf-8"))["text"]
    assert result["files"] == 1


def test_legacy_issue_patch_skips_dirty_path(tmp_path: Path):
    final = _write_final(tmp_path, "ch09_chunk000", "旧名出场。")
    path = Path(final["path"])
    before = path.read_bytes()
    issue = {
        "issue_type": "alias_mixed",
        "path": str(path),
        "position": 0,
        "variant": "旧名",
        "canonical": "新名",
    }

    patched = consistency._apply_issues(
        [final], [issue], skip_paths={str(path)})

    assert patched == 0
    assert path.read_bytes() == before
