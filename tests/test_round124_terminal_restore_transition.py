"""Round 124: 终态恢复转移 PF→COMPLETED 被静默拒绝（ch09 run2 根因之一）。

ch09 run2 日志 8 条「非法转移 ch09_chunkXXX: PERMANENTLY_FAILED -> COMPLETED」
+ 紧随其后 8 条假报「已恢复之前的 final.json (COMPLETED)」：
`_restore_dirty_backups`（R107 P7）把 final_backup.json 恢复成 final.json 后
调 `update_task_state(COMPLETED)`, 但 `VALID_TRANSITIONS[PERMANENTLY_FAILED]=[]`
（终态）→ 转移被拒绝且仅打印警告 → 状态残留 PF, 产出校验把 8 个已恢复合格旧稿的
chunk 继续计为 PERMANENTLY_FAILED。

修复: update_task_state 新增 allow_terminal_restore 参数, 仅放行
PERMANENTLY_FAILED → COMPLETED（恢复语义）；其余非法转移不放松。
本测试锁定: 默认仍拒绝 / 显式恢复放行 / 其它状态不松绑 / 双落点同步。
"""
from __future__ import annotations

import inspect

import pytest


def _mk_scheduler(tmp_path, module):
    scheduler = module.TaskScheduler(db_path=str(tmp_path / "wf.db"))
    cursor = scheduler.conn.cursor()
    cursor.execute(
        "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
        "VALUES ('c000', 'ch01', 'src', ?)",
        (module.TaskState.PERMANENTLY_FAILED.value,),
    )
    scheduler.conn.commit()
    return scheduler


def _state(scheduler, chunk_id="c000"):
    row = scheduler.conn.execute(
        "SELECT state FROM chunk_tasks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return row[0]


@pytest.fixture(params=["core", "monolith"])
def sched_module(request):
    if request.param == "core":
        import src.core.scheduler as core
        return core
    import src.translator_agent as mono
    return mono


class TestTerminalRestore:
    def test_pf_to_completed_blocked_by_default(self, tmp_path, sched_module):
        scheduler = _mk_scheduler(tmp_path, sched_module)
        scheduler.update_task_state("c000", sched_module.TaskState.COMPLETED)
        assert _state(scheduler) == sched_module.TaskState.PERMANENTLY_FAILED.value, \
            "默认调用不得放宽终态（回归 R107 语义）"

    def test_pf_to_completed_allowed_with_restore_flag(self, tmp_path, sched_module):
        scheduler = _mk_scheduler(tmp_path, sched_module)
        scheduler.update_task_state(
            "c000", sched_module.TaskState.COMPLETED, allow_terminal_restore=True)
        assert _state(scheduler) == sched_module.TaskState.COMPLETED.value

    def test_restore_flag_does_not_bypass_other_states(self, tmp_path, sched_module):
        scheduler = _mk_scheduler(tmp_path, sched_module)
        cursor = scheduler.conn.cursor()
        cursor.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES ('c001', 'ch01', 'src', ?)",
            (sched_module.TaskState.PENDING.value,))
        scheduler.conn.commit()
        scheduler.update_task_state(
            "c001", sched_module.TaskState.COMPLETED, allow_terminal_restore=True)
        assert _state(scheduler, "c001") == sched_module.TaskState.PENDING.value, \
            "allow_terminal_restore 只应放行 PF→COMPLETED"

    def test_normal_transition_unaffected(self, tmp_path, sched_module):
        scheduler = _mk_scheduler(tmp_path, sched_module)
        cursor = scheduler.conn.cursor()
        cursor.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES ('c002', 'ch01', 'src', ?)",
            (sched_module.TaskState.AUDITING.value,))
        scheduler.conn.commit()
        scheduler.update_task_state("c002", sched_module.TaskState.COMPLETED)
        assert _state(scheduler, "c002") == sched_module.TaskState.COMPLETED.value


class TestRestoreSiteSync:
    def test_restore_sites_pass_flag(self):
        from src.pipeline import TranslationPipeline as Module
        from src.translator_agent import TranslationPipeline as Monolith
        for cls in (Module, Monolith):
            source = inspect.getsource(cls._restore_dirty_backups)
            assert "allow_terminal_restore=True" in source, cls.__module__
