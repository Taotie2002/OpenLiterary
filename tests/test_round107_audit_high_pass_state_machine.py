"""Round 107: 状态机允许 AUDITING → COMPLETED (高分直通).

背景:
  R101 P1 高分直通从 AUDITING 直接 COMPLETED (跳过 judge), 但 VALID_TRANSITIONS
  缺 COMPLETED, 导致转移被拒, chunk 卡 AUDITING 死循环, 反复重审烧 critic 调用,
  最终被 NO_PROGRESS_STREAK 误判 PERMANENTLY_FAILED (分数其实是 9.4-9.6 高分).

  ch03 实测: 8 个 chunk 触发高分直通, 全部被拒, chunk006 重复 28 次.

修复:
  VALID_TRANSITIONS[TaskState.AUDITING] 加入 TaskState.COMPLETED.

测试覆盖:
  - AUDITING → COMPLETED 合法 (高分直通路径)
  - AUDITING → JUDGING 仍合法 (正常路径)
  - AUDITING → REWRITING_LITERARY 仍合法 (实体硬失败退回)
  - COMPLETED 是终态 (不可再转移, 除 DIRTY 回溯)
  - 单体版与散装版 VALID_TRANSITIONS 一致
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def scheduler(tmp_path):
    """构造真实 TaskScheduler (temp DB)."""
    from src.core.scheduler import TaskScheduler
    db_path = tmp_path / "test_workflow.db"
    sched = TaskScheduler(db_path=str(db_path))
    yield sched
    # cleanup
    try:
        sched.conn.close()
    except Exception:
        pass


def _insert_task(sched, chunk_id, state):
    """直接插入任务到指定状态 (绕过状态机, 用于测试)."""
    cursor = sched.conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO chunk_tasks
        (chunk_id, chapter_id, text_content, state, retries, quality_retries, last_error)
        VALUES (?, ?, ?, ?, 0, 0, NULL)
    ''', (chunk_id, "ch_test", "测试文本", state))
    sched.conn.commit()


def _get_state(sched, chunk_id):
    cursor = sched.conn.cursor()
    cursor.execute('SELECT state FROM chunk_tasks WHERE chunk_id = ?', (chunk_id,))
    row = cursor.fetchone()
    return row[0] if row else None


class TestAuditHighPassStateMachine:
    def test_auditing_to_completed_is_legal(self, scheduler):
        """高分直通: AUDITING → COMPLETED 必须合法."""
        from src.core.scheduler import TaskState
        chunk_id = "ch_test_chunk000"
        _insert_task(scheduler, chunk_id, TaskState.AUDITING.value)

        scheduler.update_task_state(
            chunk_id, TaskState.COMPLETED,
            error_msg="[AUDIT_HIGH_PASS] all_scores≥8.0, cjk=0.85, judge skipped"
        )

        assert _get_state(scheduler, chunk_id) == TaskState.COMPLETED.value

    def test_auditing_to_judging_still_legal(self, scheduler):
        """正常路径: AUDITING → JUDGING 仍合法."""
        from src.core.scheduler import TaskState
        chunk_id = "ch_test_chunk001"
        _insert_task(scheduler, chunk_id, TaskState.AUDITING.value)

        scheduler.update_task_state(chunk_id, TaskState.JUDGING)

        assert _get_state(scheduler, chunk_id) == TaskState.JUDGING.value

    def test_auditing_to_rewriting_still_legal(self, scheduler):
        """实体硬失败退回: AUDITING → REWRITING_LITERARY 仍合法."""
        from src.core.scheduler import TaskState
        chunk_id = "ch_test_chunk002"
        _insert_task(scheduler, chunk_id, TaskState.AUDITING.value)

        scheduler.update_task_state(
            chunk_id, TaskState.REWRITING_LITERARY,
            error_msg='{"feedback": "test"}', quality_retry=True
        )

        assert _get_state(scheduler, chunk_id) == TaskState.REWRITING_LITERARY.value

    def test_completed_is_terminal_except_dirty(self, scheduler):
        """COMPLETED 是终态: 只能转移到 DIRTY (回溯), 不能回到 AUDITING."""
        from src.core.scheduler import TaskState
        chunk_id = "ch_test_chunk003"
        _insert_task(scheduler, chunk_id, TaskState.COMPLETED.value)

        # COMPLETED → AUDITING 非法
        scheduler.update_task_state(chunk_id, TaskState.AUDITING)
        assert _get_state(scheduler, chunk_id) == TaskState.COMPLETED.value

        # COMPLETED → DIRTY 合法 (回溯)
        scheduler.update_task_state(chunk_id, TaskState.DIRTY)
        assert _get_state(scheduler, chunk_id) == TaskState.DIRTY.value

    def test_monolith_and_scattered_transitions_match(self):
        """单体版与散装版 VALID_TRANSITIONS 必须一致."""
        from src.core.scheduler import VALID_TRANSITIONS as scattered_vt
        from src.translator_agent import VALID_TRANSITIONS as monolith_vt
        from src.core.scheduler import TaskState

        # 关键: AUDITING 的允许列表必须包含 COMPLETED
        assert TaskState.COMPLETED in scattered_vt[TaskState.AUDITING]
        assert TaskState.COMPLETED in monolith_vt[TaskState.AUDITING]

        # 两版的 AUDITING 允许列表必须完全一致
        assert scattered_vt[TaskState.AUDITING] == monolith_vt[TaskState.AUDITING]

        # 两版的所有状态允许列表必须完全一致
        for state in TaskState:
            assert scattered_vt.get(state) == monolith_vt.get(state), \
                f"状态 {state} 的允许列表不一致: {scattered_vt.get(state)} vs {monolith_vt.get(state)}"
