"""Round 104: 恢复"COMPLETED 但 final.json 缺失"的 chunk.

根因: 回溯 (COMPLETED -> DIRTY) 会删除 final.json (_process_dirty_batch),
      若重跑因 bug 未重新落盘 final.json, chunk 会卡在 COMPLETED 且无产出.
      此时 --force (smart-force) 会保留 COMPLETED chunk, 用户无法恢复,
      产出校验永远报"缺失 1", 形成死锁.

修复: run() 启动时检测这类 chunk 并重置为 DIRTY (合法转移), 让其重新走完整流水线.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
    """隔离父包, 避免测试间相互污染 (历史教训)."""
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)


class _StubScheduler:
    """跟踪状态变更的 scheduler stub (仅实现恢复方法用到的两个接口)."""

    def __init__(self, tasks):
        self._tasks = {t["chunk_id"]: dict(t) for t in tasks}

    def get_all_tasks_by_chapter(self, chapter_id):
        return list(self._tasks.values())

    def update_task_state(self, chunk_id, new_state, error_msg=None, quality_retry=False):
        task = self._tasks.get(chunk_id)
        if task is None:
            return
        # 与 VALID_TRANSITIONS 对齐: 仅允许 COMPLETED -> DIRTY
        if task["state"] == "COMPLETED" and new_state.value == "DIRTY":
            task["state"] = "DIRTY"


def _make_pipeline(output_dir, tasks):
    from src.translator_agent import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.chapter_id = "ch_test"
    pipeline.output_dir = output_dir
    pipeline.scheduler = _StubScheduler(tasks=tasks)
    return pipeline


class TestSourcePatch:
    """散装 src/pipeline.py 与整合脚本 src/translator_agent.py 必须同步含修复."""

    def test_method_present_in_source(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "def _recover_completed_without_final" in content

    def test_method_present_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "def _recover_completed_without_final" in content

    def test_called_in_run(self):
        for path in ("src/pipeline.py", "src/translator_agent.py"):
            content = Path(path).read_text(encoding="utf-8")
            assert "self._recover_completed_without_final()" in content, (
                f"{path} 的 run() 必须调用恢复方法"
            )

    def test_syntax_check(self):
        import ast
        ast.parse(Path("src/pipeline.py").read_text(encoding="utf-8"))
        ast.parse(Path("src/translator_agent.py").read_text(encoding="utf-8"))


class TestRecoverBehavior:
    """_recover_completed_without_final 行为测试."""

    def test_completed_without_final_reset_to_dirty(self, tmp_path, capsys):
        """COMPLETED 且无 final.json → 重置为 DIRTY; 有 final.json 的保持 COMPLETED."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "COMPLETED"},  # 无 final
            {"chunk_id": "ch_test_chunk001", "state": "COMPLETED"},  # 有 final
            {"chunk_id": "ch_test_chunk002", "state": "COMPLETED"},  # 无 final
        ]
        (output_dir / "ch_test_chunk001_final.json").write_text('{"text":"ok"}')

        pipeline = _make_pipeline(output_dir, tasks)
        recovered = pipeline._recover_completed_without_final()

        assert sorted(recovered) == ["ch_test_chunk000", "ch_test_chunk002"]
        states = {
            t["chunk_id"]: t["state"]
            for t in pipeline.scheduler.get_all_tasks_by_chapter("ch_test")
        }
        assert states["ch_test_chunk000"] == "DIRTY"
        assert states["ch_test_chunk001"] == "COMPLETED"
        assert states["ch_test_chunk002"] == "DIRTY"

        captured = capsys.readouterr()
        assert "ch_test_chunk000" in captured.out
        assert "ch_test_chunk002" in captured.out
        # 有 final 的 chunk001 不应被点名恢复
        assert "ch_test_chunk001 状态为 COMPLETED" not in captured.out

    def test_idempotent(self, tmp_path):
        """第二次调用不应重复恢复 (chunk 已变 DIRTY)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [{"chunk_id": "ch_test_chunk000", "state": "COMPLETED"}]
        pipeline = _make_pipeline(output_dir, tasks)
        first = pipeline._recover_completed_without_final()
        assert first == ["ch_test_chunk000"]
        second = pipeline._recover_completed_without_final()
        assert second == []

    def test_non_completed_not_touched(self, tmp_path):
        """非 COMPLETED 状态 (PENDING / PERMANENTLY_FAILED) 不受影响."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "PENDING"},
            {"chunk_id": "ch_test_chunk001", "state": "PERMANENTLY_FAILED"},
        ]
        pipeline = _make_pipeline(output_dir, tasks)
        recovered = pipeline._recover_completed_without_final()
        assert recovered == []
        states = {
            t["chunk_id"]: t["state"]
            for t in pipeline.scheduler.get_all_tasks_by_chapter("ch_test")
        }
        assert states["ch_test_chunk000"] == "PENDING"
        assert states["ch_test_chunk001"] == "PERMANENTLY_FAILED"
