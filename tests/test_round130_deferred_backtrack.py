"""Round 130: 回溯治理 — deferred 章末合并队列 + 单块回溯轮数预算。

背景：ch09 全章重跑出现"决策翻转 → 回溯已完成块 → 重跑再翻转"自激循环，
连续 32.5h 不收敛（631 次重写 / 54 个 final_backup / 决策写入持续到中断前 8 分钟）。

修复（三件套）：
  1. backtrack_mode=deferred（默认）: 决策变更不再即时标 DIRTY，而是登记
     pending_backtrack 队列，pipeline 在"队列清空点"（章末）合并应用一次；
  2. scheduler.trigger_backtrack 治理计数：单块累计回溯轮数
     （backtrack_rounds，跨 episode 持久）达到 max_backtrack_rounds_per_chunk 即冻结；
  3. pipeline._consolidate_deferred_backtracks 章级合并预算
     （max_backtrack_sweeps_per_chapter，默认 1 轮；用尽后冻结登记到下次运行）。

镜像：队列/调度器测试覆盖 src/ 散装树 与 src/translator_agent.py 单体树。
"""
from __future__ import annotations

import importlib

import pytest

from src.config import config as _cfg


ENGINE_MIRROR = [
    ("monolith", "src.translator_agent"),
    ("modular", "src.core.decision_engine"),
]
SCHED_MIRROR = [
    ("monolith", "src.translator_agent"),
    ("modular", "src.core.scheduler"),
]


@pytest.fixture
def set_de_cfg():
    """临时覆盖 decision_engine 配置并在用例结束后恢复。"""
    de = _cfg._data.setdefault("decision_engine", {})
    backup = dict(de)

    def _apply(**kw):
        de.update(kw)

    yield _apply
    de.clear()
    de.update(backup)


class _FakeScheduler:
    def __init__(self):
        self.calls = []

    def trigger_backtrack(self, chunk_ids):
        self.calls.append(list(chunk_ids))
        return {"dirty": list(chunk_ids), "over_cap": [], "skipped_state": []}


@pytest.fixture(params=ENGINE_MIRROR, ids=[m[0] for m in ENGINE_MIRROR])
def engine_mod(request):
    return importlib.import_module(request.param[1])


@pytest.fixture(params=SCHED_MIRROR, ids=[m[0] for m in SCHED_MIRROR])
def sched_mod(request):
    return importlib.import_module(request.param[1])


class TestDeferredQueue:
    def test_deferred_queues_instead_of_dirty(self, engine_mod, tmp_path, set_de_cfg):
        set_de_cfg(backtrack_mode="deferred")
        sched = _FakeScheduler()
        engine = engine_mod.DecisionEngine(
            db_path=str(tmp_path / "d.sqlite"), scheduler_factory=lambda: sched)
        engine.add_decision(engine_mod.DecisionLevel.REFERENCE, "Mock Turtle", "假海龟",
                            "r1", affected_chunks=["ch10_c000"])
        engine.add_decision(engine_mod.DecisionLevel.REFERENCE, "Mock Turtle", "素甲鱼",
                            "r2", affected_chunks=["ch10_c000", "ch10_c002"])
        # 未即时回溯（deferred 模式）
        assert sched.calls == []

        pending = engine.take_pending_backtracks(["ch10_c000", "ch10_c002", "ch10_c999"])
        by_id = {p["chunk_id"]: p for p in pending}
        assert set(by_id) == {"ch10_c000", "ch10_c002"}
        assert by_id["ch10_c000"]["queue_rounds"] == 2   # 幂等合并累计
        assert by_id["ch10_c002"]["queue_rounds"] == 1
        # take 后队列清空
        assert engine.count_pending_backtracks(["ch10_c000", "ch10_c002"]) == 0

    def test_immediate_mode_triggers_directly(self, engine_mod, tmp_path, set_de_cfg):
        set_de_cfg(backtrack_mode="immediate")
        sched = _FakeScheduler()
        engine = engine_mod.DecisionEngine(
            db_path=str(tmp_path / "d.sqlite"), scheduler_factory=lambda: sched)
        engine.add_decision(engine_mod.DecisionLevel.REFERENCE, "Gryphon", "狮鹫",
                            "r1", affected_chunks=["ch10_c001"])
        engine.add_decision(engine_mod.DecisionLevel.REFERENCE, "Gryphon", "鹰头狮",
                            "r2", affected_chunks=["ch10_c001"])
        assert sched.calls == [["ch10_c001"], ["ch10_c001"]]
        assert engine.count_pending_backtracks() == 0


def _seed_completed(sched_mod, db_path, chunk_ids):
    sched = sched_mod.TaskScheduler(db_path=str(db_path))
    cur = sched.conn.cursor()
    for cid in chunk_ids:
        cur.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) VALUES (?,?,?,?)",
            (cid, cid.split("_")[0], "x", sched_mod.TaskState.COMPLETED.value))
    sched.conn.commit()
    return sched


def _force_completed(sched, chunk_id):
    cur = sched.conn.cursor()
    cur.execute("UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id=?", (chunk_id,))
    sched.conn.commit()


class TestGovernanceCap:
    def test_backtrack_rounds_cap_freeze(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(max_backtrack_rounds_per_chunk=2)
        sched = _seed_completed(sched_mod, tmp_path / "w.db", ["ch01_chunk000"])

        r1 = sched.trigger_backtrack(["ch01_chunk000"])
        assert r1["dirty"] == ["ch01_chunk000"] and r1["over_cap"] == []
        assert sched.get_task("ch01_chunk000")["backtrack_rounds"] == 1

        _force_completed(sched, "ch01_chunk000")   # 模拟重跑完成
        r2 = sched.trigger_backtrack(["ch01_chunk000"])
        assert r2["dirty"] == ["ch01_chunk000"]
        assert sched.get_task("ch01_chunk000")["backtrack_rounds"] == 2

        _force_completed(sched, "ch01_chunk000")
        r3 = sched.trigger_backtrack(["ch01_chunk000"])
        assert r3["dirty"] == [] and r3["over_cap"] == ["ch01_chunk000"]
        # 被冻结：状态保持 COMPLETED，轮数不再增长
        task = sched.get_task("ch01_chunk000")
        assert task["state"] == "COMPLETED" and task["backtrack_rounds"] == 2

    def test_non_completed_and_unknown_ids_skipped(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(max_backtrack_rounds_per_chunk=2)
        sched = _seed_completed(sched_mod, tmp_path / "w.db", ["ch01_chunk000"])
        cur = sched.conn.cursor()
        cur.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) VALUES (?,?,?,?)",
            ("ch01_chunk001", "ch01", "x", sched_mod.TaskState.REWRITING_LITERARY.value))
        sched.conn.commit()

        stats = sched.trigger_backtrack(["ch01_chunk000", "ch01_chunk001", "ch99_chunk404"])
        assert stats["dirty"] == ["ch01_chunk000"]
        assert stats["skipped_state"] == ["ch01_chunk001"]


class TestChapterConsolidation:
    def _make_pipeline(self, tmp_path):
        from src.pipeline import TranslationPipeline
        from src.core import scheduler as sched_module
        from src.core.decision_engine import DecisionEngine

        sched = _seed_completed(sched_module, tmp_path / "w.db", ["ch09_chunk000", "ch09_chunk001"])
        engine = DecisionEngine(db_path=str(tmp_path / "d.sqlite"), scheduler_factory=lambda: sched)
        pipe = TranslationPipeline.__new__(TranslationPipeline)
        pipe.chapter_id = "ch09"
        pipe.output_dir = tmp_path
        pipe.scheduler = sched
        pipe.decision_engine = engine
        pipe._backtrack_sweeps_done = 0
        return pipe, sched, engine

    def test_consolidation_applies_once_then_freezes(self, tmp_path, set_de_cfg):
        set_de_cfg(backtrack_mode="deferred",
                   max_backtrack_sweeps_per_chapter=1,
                   max_backtrack_rounds_per_chunk=2)
        pipe, sched, engine = self._make_pipeline(tmp_path)
        engine._queue_backtrack(["ch09_chunk000"], 1, "institutions")

        assert pipe._consolidate_deferred_backtracks() is True
        task = sched.get_task("ch09_chunk000")
        assert task["state"] == "DIRTY" and task["backtrack_rounds"] == 1

        # 预算（1 轮）用尽：再登记的项被冻结（队列保留到下次运行）
        engine._queue_backtrack(["ch09_chunk001"], 2, "property rights")
        assert pipe._consolidate_deferred_backtracks() is False
        assert engine.count_pending_backtracks(["ch09_chunk001"]) == 1
        assert sched.get_task("ch09_chunk001")["state"] == "COMPLETED"

    def test_sweeps_zero_disables_consolidation(self, tmp_path, set_de_cfg):
        set_de_cfg(backtrack_mode="deferred",
                   max_backtrack_sweeps_per_chapter=0,
                   max_backtrack_rounds_per_chunk=2)
        pipe, sched, engine = self._make_pipeline(tmp_path)
        engine._queue_backtrack(["ch09_chunk000"], 1, "institutions")

        assert pipe._consolidate_deferred_backtracks() is False
        # 登记保留，未打 DIRTY
        assert engine.count_pending_backtracks(["ch09_chunk000"]) == 1
        assert sched.get_task("ch09_chunk000")["state"] == "COMPLETED"

    def test_over_cap_frozen_and_report_written(self, tmp_path, set_de_cfg):
        set_de_cfg(backtrack_mode="deferred",
                   max_backtrack_sweeps_per_chapter=1,
                   max_backtrack_rounds_per_chunk=1)
        pipe, sched, engine = self._make_pipeline(tmp_path)
        cur = sched.conn.cursor()
        cur.execute("UPDATE chunk_tasks SET backtrack_rounds=1 WHERE chunk_id='ch09_chunk000'")
        sched.conn.commit()
        engine._queue_backtrack(["ch09_chunk000"], 1, "institutions")

        assert pipe._consolidate_deferred_backtracks() is False
        # 冻结：未打 DIRTY、登记已消费、报告已落盘
        assert sched.get_task("ch09_chunk000")["state"] == "COMPLETED"
        assert engine.count_pending_backtracks(["ch09_chunk000"]) == 0
        assert (tmp_path / "backtrack_frozen.md").exists()

    def test_immediate_mode_noop(self, tmp_path, set_de_cfg):
        set_de_cfg(backtrack_mode="immediate")
        pipe, sched, engine = self._make_pipeline(tmp_path)
        engine._queue_backtrack(["ch09_chunk000"], 1, "institutions")
        assert pipe._consolidate_deferred_backtracks() is False
        assert sched.get_task("ch09_chunk000")["state"] == "COMPLETED"
