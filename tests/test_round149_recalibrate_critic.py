"""Round 149: read-only calibrated critic re-audit before delivery."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from src.core.scheduler import TaskScheduler
from src.main import _MUTATING_COMMANDS
from src.recalibrate import run_recalibrate_critic


def _make_project(tmp_path: Path, chunk_ids):
    root = tmp_path / "project"
    out = root / "output" / "ch01"
    out.mkdir(parents=True)
    scheduler = TaskScheduler(db_path=str(root / "db" / "workflow.db"))
    for cid in chunk_ids:
        scheduler.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?, 'ch01', ?, 'COMPLETED')",
            (cid, "Source text for " + cid),
        )
        (out / f"{cid}_final.json").write_text(
            json.dumps({"text": "最终译文 " + cid}, ensure_ascii=False), encoding="utf-8")
        (out / f"{cid}_raw.json").write_text(
            json.dumps({"text": "直译译文 " + cid}, ensure_ascii=False), encoding="utf-8")
    scheduler.conn.commit()
    scheduler.conn.close()
    return root, out


def _write_report(out: Path, cid: str, backend: str, scores=None):
    payload = {"scores": scores or {"fluency": 8}, "_backend_path": backend}
    (out / f"{cid}_critic_report.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class TestSelection:
    def test_dry_run_selects_missing_and_uncalibrated_only(self, tmp_path):
        root, out = _make_project(tmp_path, ["c0", "c1", "c2"])
        _write_report(out, "c0", "jev")
        _write_report(out, "c1", "legacy")
        res = run_recalibrate_critic(chapter_id="ch01", apply=False, project_root=root)
        assert res["selected"] == 2
        assert [r["chunk_id"] for r in res["rows"]] == ["c1", "c2"]
        assert res["rows"][0]["reason"] == "uncalibrated_report"
        assert res["rows"][1]["reason"] == "missing_report"

    def test_explicit_chunk_limit_is_respected(self, tmp_path):
        root, out = _make_project(tmp_path, ["c0", "c1", "c2"])
        _write_report(out, "c0", "legacy")
        _write_report(out, "c1", "legacy")
        _write_report(out, "c2", "legacy")
        res = run_recalibrate_critic(
            chapter_id="ch01", chunk_ids=["c2", "c1"], limit=1,
            apply=False, project_root=root,
        )
        assert res["selected"] == 1
        assert res["rows"][0]["chunk_id"] == "c1"

    def test_force_selects_already_calibrated_chunk(self, tmp_path):
        root, out = _make_project(tmp_path, ["c0", "c1"])
        _write_report(out, "c0", "jev")
        _write_report(out, "c1", "legacy")
        res = run_recalibrate_critic(
            chapter_id="ch01", chunk_ids=["c0"], apply=False,
            force=True, project_root=root,
        )
        assert res["selected"] == 1
        assert res["rows"][0]["chunk_id"] == "c0"

    def test_calibrated_report_is_selected_when_final_changed(self, tmp_path):
        root, out = _make_project(tmp_path, ["c0", "c1"])
        _write_report(out, "c0", "jev")
        _write_report(out, "c1", "jev")
        old_sha = hashlib.sha256(b"older final text").hexdigest()
        for cid, sha in (("c0", None), ("c1", old_sha)):
            final_text = json.loads(
                (out / f"{cid}_final.json").read_text(encoding="utf-8"))["text"]
            sha = sha or hashlib.sha256(final_text.encode("utf-8")).hexdigest()
            path = out / f"{cid}_critic_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["_reaudit"] = {"final_sha256": sha}
            path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

        res = run_recalibrate_critic(chapter_id="ch01", apply=False, project_root=root)
        assert res["selected"] == 1
        assert res["rows"][0]["chunk_id"] == "c1"
        assert res["rows"][0]["reason"] == "stale_report"


class _FakeCritic:
    def __init__(self, backend="jev"):
        self.backend = backend
        self.calls = []

    def process_chunk(self, chunk_id, source_text, raw_text, lit_text,
                      style_guide, **kwargs):
        self.calls.append((chunk_id, lit_text, kwargs))
        return {"scores": {"fluency": 8}, "_backend_path": self.backend}


def _install_fake_pipeline(monkeypatch, root: Path, critic):
    import src.pipeline as pipeline_mod

    class FakePipeline:
        def __init__(self, chapter_id):
            self.chapter_id = chapter_id
            self.scheduler = TaskScheduler(db_path=str(root / "db" / "workflow.db"))
            self.style_guide = {}
            self.critic_agent = critic

        def _build_entity_registry_for_rewriter(self, source_text):
            return {}

        def _build_entity_variants_for_critic(self, source_text):
            return {}

    monkeypatch.setattr(pipeline_mod, "TranslationPipeline", FakePipeline)


class TestApply:
    def test_apply_replaces_report_but_final_bytes_are_unchanged(self, tmp_path, monkeypatch):
        root, out = _make_project(tmp_path, ["c0"])
        _write_report(out, "c0", "legacy", {"fluency": 5})
        final_path = out / "c0_final.json"
        before = final_path.read_bytes()
        critic = _FakeCritic("jev")
        _install_fake_pipeline(monkeypatch, root, critic)

        res = run_recalibrate_critic(
            chapter_id="ch01", apply=True, workers=1, retries=0, project_root=root,
        )
        assert res["ok"] is True
        assert res["recalibrated"] == 1
        assert final_path.read_bytes() == before
        report = json.loads((out / "c0_critic_report.json").read_text(encoding="utf-8"))
        assert report["_backend_path"] == "jev"
        assert report["_reaudit"]["previous_backend_path"] == "legacy"
        assert report["_reaudit"]["literary_input"] == "final"
        assert critic.calls[0][1] == "最终译文 c0"
        assert critic.calls[0][2]["allow_legacy_fallback"] is False

    def test_uncalibrated_result_does_not_overwrite_existing_report(self, tmp_path, monkeypatch):
        root, out = _make_project(tmp_path, ["c0"])
        _write_report(out, "c0", "legacy", {"fluency": 5})
        report_path = out / "c0_critic_report.json"
        before = report_path.read_bytes()
        _install_fake_pipeline(monkeypatch, root, _FakeCritic("legacy"))

        res = run_recalibrate_critic(
            chapter_id="ch01", apply=True, workers=1, retries=0, project_root=root,
        )
        assert res["ok"] is False
        assert res["failed"] == 1
        assert report_path.read_bytes() == before

    def test_recalibrate_command_uses_instance_lock(self):
        assert "recalibrate-critic" in _MUTATING_COMMANDS
