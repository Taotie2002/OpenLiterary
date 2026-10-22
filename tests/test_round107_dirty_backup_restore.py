"""Round 107 P7: 回溯重跑 final.json 备份 + 恢复.

ch07 chunk000 失败根因:
  chunk000 首次 COMPLETED, 但回溯重跑机制 (COMPLETED→DIRTY) 删除 final.json 后
  重新生成译文, 重跑失败 (NO_PROGRESS_STREAK), chunk 从 COMPLETED 降级为
  PERMANENTLY_FAILED 且无产出.

修复:
  1. _process_dirty_batch: final.json 不删除, 重命名为 final_backup.json
  2. _restore_dirty_backups: 重跑失败时恢复 backup (final_backup.json → final.json,
     状态 PERMANENTLY_FAILED → COMPLETED)
  3. 重跑成功时删除无用 backup

测试覆盖:
  - DIRTY 重置时 final.json 被重命名为 final_backup.json (不删除)
  - 重跑失败 (无 final.json) → 恢复 backup
  - 重跑成功 (有 final.json) → 删除无用 backup
  - 两版一致
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_pipeline_stub(tmp_path):
    """构造 TranslationPipeline stub (不触发 __init__)."""
    from src.pipeline import TranslationPipeline
    from src.core.scheduler import TaskScheduler

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = tmp_path
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock(spec=TaskScheduler)
    # Round 108: _process_dirty_batch 会清空分数历史/最佳版本
    pipeline._score_history = {}
    pipeline._best_lit = {}
    return pipeline


class TestDirtyBackupRestore:
    def test_dirty_batch_backs_up_final(self, tmp_path):
        """DIRTY 重置时 final.json 被重命名为 final_backup.json (不删除)."""
        pipeline = _make_pipeline_stub(tmp_path)
        # 创建 final.json
        final_path = tmp_path / "ch_test_chunk000_final.json"
        final_path.write_text(json.dumps({"text": "之前的译文", "metadata": {}}))
        # 创建其他 intermediate
        (tmp_path / "ch_test_chunk000_raw.json").write_text("{}")
        (tmp_path / "ch_test_chunk000_literary.json").write_text("{}")

        pipeline._process_dirty_batch([{"chunk_id": "ch_test_chunk000"}])

        # final.json 应被重命名为 final_backup.json
        assert not final_path.exists(), "final.json 应被重命名"
        backup_path = tmp_path / "ch_test_chunk000_final_backup.json"
        assert backup_path.exists(), "final_backup.json 应存在"
        # 其他 intermediate 应被删除
        assert not (tmp_path / "ch_test_chunk000_raw.json").exists()
        assert not (tmp_path / "ch_test_chunk000_literary.json").exists()

    def test_restore_backup_on_failure(self, tmp_path):
        """重跑失败 (无 final.json) → 恢复 backup."""
        pipeline = _make_pipeline_stub(tmp_path)
        # 创建 backup (模拟重跑失败)
        backup_path = tmp_path / "ch_test_chunk000_final_backup.json"
        backup_path.write_text(json.dumps({"text": "之前的译文", "metadata": {}}))

        restored = pipeline._restore_dirty_backups()

        # backup 应被恢复为 final.json
        assert "ch_test_chunk000" in restored
        final_path = tmp_path / "ch_test_chunk000_final.json"
        assert final_path.exists(), "final.json 应被恢复"
        assert not backup_path.exists(), "backup 应被重命名"
        # 状态应被更新为 COMPLETED
        pipeline.scheduler.update_task_state.assert_called()

    def test_cleanup_backup_on_success(self, tmp_path):
        """重跑成功 (有 final.json) → 删除无用 backup."""
        pipeline = _make_pipeline_stub(tmp_path)
        # 创建 final.json + backup (模拟重跑成功)
        (tmp_path / "ch_test_chunk000_final.json").write_text(json.dumps({"text": "新译文"}))
        backup_path = tmp_path / "ch_test_chunk000_final_backup.json"
        backup_path.write_text(json.dumps({"text": "旧译文"}))

        restored = pipeline._restore_dirty_backups()

        # backup 应被删除 (重跑成功)
        assert "ch_test_chunk000" not in restored
        assert not backup_path.exists(), "无用 backup 应被删除"
        # final.json 应保留 (新译文)
        assert (tmp_path / "ch_test_chunk000_final.json").exists()

    def test_no_backup_no_action(self, tmp_path):
        """无 backup 文件 → 无操作."""
        pipeline = _make_pipeline_stub(tmp_path)
        restored = pipeline._restore_dirty_backups()
        assert restored == []
