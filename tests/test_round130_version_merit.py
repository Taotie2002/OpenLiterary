"""Round 130: 版本择优 — 回溯重跑新版 critic 均分低于备份版时保留旧版。

背景：ch09 长跑期间重跑用"新版本"反复覆盖已通过的老版本（置换风险），
且无任何对比机制。修复：
  1. _process_dirty_batch 把 critic_report 保留为 critic_report_backup.json；
  2. _restore_dirty_backups（章末）裁决：新版均分 < 旧版均分 - tolerance →
     备份版覆盖回 final.json 并保留旧版评分；否则维持既有"删除无用 backup"行为。

配置：pipeline.version_merit.{enabled,tolerance}（默认 true / 0.0）。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.config import config as _cfg


@pytest.fixture
def set_vm_cfg():
    """临时覆盖 pipeline.version_merit 配置并恢复。"""
    pipe = _cfg._data.setdefault("pipeline", {})
    backup = dict(pipe.get("version_merit", {}) or {})
    pipe["version_merit"] = dict(backup)

    def _apply(**kw):
        pipe["version_merit"].update(kw)

    yield _apply
    pipe["version_merit"] = backup


def _make_pipeline_stub(tmp_path):
    """构造 TranslationPipeline stub (不触发 __init__)."""
    from src.pipeline import TranslationPipeline
    from src.core.scheduler import TaskScheduler

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = tmp_path
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock(spec=TaskScheduler)
    pipeline._score_history = {}
    pipeline._best_lit = {}
    return pipeline


def _write_critic(tmp_path, chunk_id, scores, suffix=""):
    path = tmp_path / f"{chunk_id}_critic_report{suffix}.json"
    path.write_text(json.dumps({"scores": scores}), encoding="utf-8")
    return path


class TestDirtyBatchCriticBackup:
    def test_critic_report_preserved_as_backup(self, tmp_path):
        pipeline = _make_pipeline_stub(tmp_path)
        cid = "ch_test_chunk000"
        (tmp_path / f"{cid}_final.json").write_text(json.dumps({"text": "旧译文"}))
        (tmp_path / f"{cid}_raw.json").write_text("{}")
        (tmp_path / f"{cid}_literary.json").write_text("{}")
        _write_critic(tmp_path, cid, {"fluency": 8.0})

        pipeline._process_dirty_batch([{"chunk_id": cid}])

        assert not (tmp_path / f"{cid}_critic_report.json").exists()
        assert (tmp_path / f"{cid}_critic_report_backup.json").exists()
        assert not (tmp_path / f"{cid}_raw.json").exists()
        assert not (tmp_path / f"{cid}_literary.json").exists()


class TestVersionMerit:
    def test_merit_restores_old_when_new_worse(self, tmp_path, set_vm_cfg):
        set_vm_cfg(enabled=True, tolerance=0.0)
        pipeline = _make_pipeline_stub(tmp_path)
        cid = "ch_test_chunk000"
        final_path = tmp_path / f"{cid}_final.json"
        final_path.write_text(json.dumps({"text": "新译文"}))
        backup_path = tmp_path / f"{cid}_final_backup.json"
        backup_path.write_text(json.dumps({"text": "旧译文"}))
        _write_critic(tmp_path, cid, {"fluency": 6.0, "readability": 6.5})          # 新版 6.25
        _write_critic(tmp_path, cid, {"fluency": 8.5, "readability": 9.0}, "_backup")  # 旧版 8.75

        restored = pipeline._restore_dirty_backups()

        assert restored == []
        assert json.loads(final_path.read_text())["text"] == "旧译文"
        assert not backup_path.exists()
        assert not (tmp_path / f"{cid}_critic_report_backup.json").exists(), "旧版评分应归位, 不留备份"
        # 旧版评分归位为正式 critic_report
        assert json.loads((tmp_path / f"{cid}_critic_report.json").read_text())["scores"]["fluency"] == 8.5

    def test_merit_keeps_new_when_not_worse(self, tmp_path, set_vm_cfg):
        set_vm_cfg(enabled=True, tolerance=0.0)
        pipeline = _make_pipeline_stub(tmp_path)
        cid = "ch_test_chunk000"
        final_path = tmp_path / f"{cid}_final.json"
        final_path.write_text(json.dumps({"text": "新译文"}))
        backup_path = tmp_path / f"{cid}_final_backup.json"
        backup_path.write_text(json.dumps({"text": "旧译文"}))
        _write_critic(tmp_path, cid, {"fluency": 9.0})                 # 新版 9.0
        _write_critic(tmp_path, cid, {"fluency": 8.0}, "_backup")      # 旧版 8.0

        pipeline._restore_dirty_backups()

        assert json.loads(final_path.read_text())["text"] == "新译文"
        assert not backup_path.exists()
        assert not (tmp_path / f"{cid}_critic_report_backup.json").exists()

    def test_merit_tolerance_allows_small_regression(self, tmp_path, set_vm_cfg):
        set_vm_cfg(enabled=True, tolerance=0.5)
        pipeline = _make_pipeline_stub(tmp_path)
        cid = "ch_test_chunk000"
        final_path = tmp_path / f"{cid}_final.json"
        final_path.write_text(json.dumps({"text": "新译文"}))
        (tmp_path / f"{cid}_final_backup.json").write_text(json.dumps({"text": "旧译文"}))
        _write_critic(tmp_path, cid, {"fluency": 8.2})                 # 新版 8.2
        _write_critic(tmp_path, cid, {"fluency": 8.5}, "_backup")      # 旧版 8.5, 差 0.3 < 0.5

        pipeline._restore_dirty_backups()

        assert json.loads(final_path.read_text())["text"] == "新译文"

    def test_merit_disabled_keeps_legacy_behavior(self, tmp_path, set_vm_cfg):
        set_vm_cfg(enabled=False, tolerance=0.0)
        pipeline = _make_pipeline_stub(tmp_path)
        cid = "ch_test_chunk000"
        final_path = tmp_path / f"{cid}_final.json"
        final_path.write_text(json.dumps({"text": "新译文"}))
        (tmp_path / f"{cid}_final_backup.json").write_text(json.dumps({"text": "旧译文"}))
        _write_critic(tmp_path, cid, {"fluency": 1.0})                 # 新版很差
        _write_critic(tmp_path, cid, {"fluency": 9.9}, "_backup")      # 旧版很好

        pipeline._restore_dirty_backups()

        # 关闭择优 → 维持旧行为：新版保留、备份删除
        assert json.loads(final_path.read_text())["text"] == "新译文"
        assert not (tmp_path / f"{cid}_final_backup.json").exists()

    def test_merit_no_scores_falls_back(self, tmp_path, set_vm_cfg):
        set_vm_cfg(enabled=True, tolerance=0.0)
        pipeline = _make_pipeline_stub(tmp_path)
        cid = "ch_test_chunk000"
        final_path = tmp_path / f"{cid}_final.json"
        final_path.write_text(json.dumps({"text": "新译文"}))
        (tmp_path / f"{cid}_final_backup.json").write_text(json.dumps({"text": "旧译文"}))
        # 无 critic 文件（legacy chunk）→ 不裁决，维持旧行为

        pipeline._restore_dirty_backups()

        assert json.loads(final_path.read_text())["text"] == "新译文"
        assert not (tmp_path / f"{cid}_final_backup.json").exists()

    def test_merit_restore_failure_path_untouched(self, tmp_path, set_vm_cfg):
        """重跑失败（无 final.json）→ 仍走原有恢复路径, 不受择优影响."""
        set_vm_cfg(enabled=True, tolerance=0.0)
        pipeline = _make_pipeline_stub(tmp_path)
        cid = "ch_test_chunk000"
        (tmp_path / f"{cid}_final_backup.json").write_text(json.dumps({"text": "旧译文"}))

        restored = pipeline._restore_dirty_backups()

        assert cid in restored
        assert json.loads((tmp_path / f"{cid}_final.json").read_text())["text"] == "旧译文"
