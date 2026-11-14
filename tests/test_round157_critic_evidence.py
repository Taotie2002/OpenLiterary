"""Round 157: bind critic reports to the delivered final bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.core.scheduler import TaskScheduler
from src.pipeline import run_final_gate
from src.recalibrate import critic_evidence_state
from src.rescue import write_qa_report


ROOT = Path(__file__).resolve().parent.parent
CHUNK = "ch01_chunk000"
CLEAN_BOOK = "# 第一章 起点\n\n他说：“你好。”然后[^1]。\n\n[^1]: 说明。\n"
AUDIT_PASS = (
    "# 转换完整性审计报告 (T1)\n\n"
    "- 全书覆盖率: 99.900%\n"
    "- 结论: ✅ 通过\n"
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, text: str, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "text": text,
        "metadata": metadata or {},
    }, ensure_ascii=False), encoding="utf-8")


def _mk_gate_project(tmp_path: Path, final_text: str, report: dict) -> None:
    (tmp_path / "db").mkdir()
    (tmp_path / "input").mkdir()
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    sched = TaskScheduler(db_path=str(tmp_path / "db" / "workflow.db"))
    sched.conn.execute(
        "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
        "VALUES (?, ?, ?, ?)",
        (CHUNK, "ch01", "source", "COMPLETED"),
    )
    sched.conn.commit()
    sched.conn.close()
    chapter = out / "ch01"
    chapter.mkdir()
    _write_json(chapter / f"{CHUNK}_final.json", final_text)
    (chapter / f"{CHUNK}_critic_report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "input" / "_conversion_audit.md").write_text(
        AUDIT_PASS, encoding="utf-8")
    (out / "translated_full.md").write_text(final_text, encoding="utf-8")
    write_qa_report(tmp_path, out)


def _repair_payload() -> dict:
    return {
        "ok": True,
        "scanned": 1,
        "pending_chunks": [],
        "blocking_issues": [],
        "manifest_present": True,
        "manifest_current": True,
        "stale_manifest": [],
        "missing_sources": [],
        "malformed_finals": [],
    }


class TestCriticEvidenceState:
    def test_reaudit_hash_matches_current_final(self, tmp_path):
        final = tmp_path / "output" / "ch01" / f"{CHUNK}_final.json"
        _write_json(final, CLEAN_BOOK)
        state = critic_evidence_state(
            {"_reaudit": {"final_sha256": _sha(CLEAN_BOOK)}},
            final,
            project_root=tmp_path,
        )
        assert state["status"] == "exact"

    def test_verified_book_repair_chain_is_repaired(self, tmp_path):
        old_text = CLEAN_BOOK + "旧稿。\n"
        backup = tmp_path / "output" / "repair_backups" / "old.json"
        _write_json(backup, old_text)
        final = tmp_path / "output" / "ch01" / f"{CHUNK}_final.json"
        _write_json(final, CLEAN_BOOK, metadata={
            "book_repair": {
                "before_sha256": _sha(old_text),
                "after_sha256": _sha(CLEAN_BOOK),
                "backup": "output/repair_backups/old.json",
            },
        })
        state = critic_evidence_state(
            {"_reaudit": {"final_sha256": _sha(old_text)}},
            final,
            project_root=tmp_path,
        )
        assert state["status"] == "repaired"
        assert len(state["repair_chain"]) == 1

    def test_chain_with_missing_backup_is_stale(self, tmp_path):
        final = tmp_path / "output" / "ch01" / f"{CHUNK}_final.json"
        _write_json(final, CLEAN_BOOK, metadata={
            "book_repair": {
                "before_sha256": _sha("unavailable old text"),
                "after_sha256": _sha(CLEAN_BOOK),
                "backup": "output/repair_backups/missing.json",
            },
        })
        state = critic_evidence_state(
            {"_reaudit": {"final_sha256": _sha("unavailable old text")}},
            final,
            project_root=tmp_path,
        )
        assert state["status"] == "stale"


class TestGateCriticEvidence:
    def test_gate_rejects_stale_reaudit_evidence(self, tmp_path, monkeypatch):
        _mk_gate_project(tmp_path, CLEAN_BOOK, {
            "_backend_path": "jev",
            "_reaudit": {"final_sha256": "0" * 64},
        })
        import src.book_repair as book_repair
        monkeypatch.setattr(
            book_repair, "verify_book_repairs", lambda **_kwargs: _repair_payload())

        result = run_final_gate(project_root=str(tmp_path))

        assert result["ok"] is False
        assert any("⑧ critic 证据与当前 final 不符" in failure
                   for failure in result["failures"]), result["failures"]
        assert result["stats"]["critic_evidence"]["stale"] == 1

    def test_gate_accepts_verified_repair_chain(self, tmp_path, monkeypatch):
        old_text = CLEAN_BOOK + "旧稿。\n"
        backup = tmp_path / "output" / "repair_backups" / "old.json"
        _write_json(backup, old_text)
        _mk_gate_project(tmp_path, CLEAN_BOOK, {
            "_backend_path": "jev",
            "_reaudit": {"final_sha256": _sha(old_text)},
        })
        final = tmp_path / "output" / "ch01" / f"{CHUNK}_final.json"
        _write_json(final, CLEAN_BOOK, metadata={
            "book_repair": {
                "before_sha256": _sha(old_text),
                "after_sha256": _sha(CLEAN_BOOK),
                "backup": "output/repair_backups/old.json",
            },
        })
        write_qa_report(tmp_path, tmp_path / "output")
        import src.book_repair as book_repair
        monkeypatch.setattr(
            book_repair, "verify_book_repairs", lambda **_kwargs: _repair_payload())

        result = run_final_gate(project_root=str(tmp_path))

        assert result["ok"] is True, result["failures"]
        assert result["stats"]["critic_evidence"]["repaired"] == 1
