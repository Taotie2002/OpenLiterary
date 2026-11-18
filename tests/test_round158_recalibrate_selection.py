"""Round 158: stale critic reports must be selected without --force."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.core.scheduler import TaskScheduler
from src.recalibrate import run_recalibrate_critic


ROOT = Path(__file__).resolve().parent.parent


def _make_project(tmp_path: Path, chunk_ids: list[str]) -> tuple[Path, Path]:
    root = tmp_path / "project"
    out = root / "output" / "ch01"
    out.mkdir(parents=True)
    scheduler = TaskScheduler(db_path=str(root / "db" / "workflow.db"))
    for cid in chunk_ids:
        scheduler.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?, 'ch01', ?, 'COMPLETED')",
            (cid, "source " + cid),
        )
        (out / f"{cid}_final.json").write_text(
            json.dumps({"text": "final " + cid}, ensure_ascii=False),
            encoding="utf-8",
        )
    scheduler.conn.commit()
    scheduler.conn.close()
    return root, out


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_report(out: Path, cid: str, target_hash: str | None) -> None:
    report = {"scores": {"fluency": 8.0}, "_backend_path": "jev"}
    if target_hash is not None:
        report["_audit_target"] = {"schema_version": 1,
                                   "literary_sha256": target_hash}
    (out / f"{cid}_critic_report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8"
    )


class TestRecalibrateSelection:
    def test_literary_target_hash_staleness_is_detected_without_force(self, tmp_path):
        root, out = _make_project(tmp_path, ["c0", "c1"])
        _write_report(out, "c0", _sha("final c0"))
        _write_report(out, "c1", _sha("an older literary draft"))
        # c0 carries the current target hash, c1 carries a stale target hash.
        res = run_recalibrate_critic(
            chapter_id="ch01", apply=False, project_root=root,
        )
        assert res["selected"] == 1
        assert res["rows"][0]["chunk_id"] == "c1"
        assert res["rows"][0]["reason"] == "stale_report"
        assert res["rows"][0]["evidence_status"] == "stale"

    def test_exact_target_hash_is_not_selected(self, tmp_path):
        root, out = _make_project(tmp_path, ["c0"])
        _write_report(out, "c0", _sha("final c0"))
        res = run_recalibrate_critic(
            chapter_id="ch01", apply=False, project_root=root,
        )
        assert res["selected"] == 0

    def test_force_still_selects_unbound_legacy_report(self, tmp_path):
        root, out = _make_project(tmp_path, ["c0"])
        _write_report(out, "c0", None)
        plain = run_recalibrate_critic(
            chapter_id="ch01", apply=False, project_root=root,
        )
        forced = run_recalibrate_critic(
            chapter_id="ch01", apply=False, force=True, project_root=root,
        )
        assert plain["selected"] == 0
        assert forced["selected"] == 1
        assert forced["rows"][0]["reason"] == "uncalibrated_report"


def test_shell_recalibrates_stale_evidence_after_repair_before_merge():
    shell = (ROOT / "scripts" / "translate_book.sh").read_text(encoding="utf-8")
    repair_at = shell.index("translator_agent book-repair --apply")
    recalibrate_at = shell.index("translator_agent recalibrate-critic", repair_at)
    merge_at = shell.index('merge_output "$CHAPTERS_DIR"', recalibrate_at)
    assert repair_at < recalibrate_at < merge_at
    command = shell[recalibrate_at:merge_at]
    assert "--all --apply" in command
    assert 'SAGATRANS_RECALIBRATE_WORKERS:-2' in command
    assert "gate ⑧ 将硬拦截" in command
