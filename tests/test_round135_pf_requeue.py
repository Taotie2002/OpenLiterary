"""Round 135: PF 终态块复活通道（ch01 复跑根因：终态冻结）。

背景：ch01 复跑时 6 个 PERMANENTLY_FAILED 块（chunk031/034/072/074/075/078）
带 2 轮停滞签名冻结于 Round 134 生效之前；trigger_backtrack 只标记 COMPLETED，
PF 是终态 → 当日回溯全部「跳过非 COMPLETED」，Round 134 的白名单/软通过
无法触达它们（同人归一/实体软化只对重跑块生效）。

修复（通用机制，零作品硬编码）：
  1. scheduler.trigger_backtrack 遇 PERMANENTLY_FAILED：预算内（decision_engine.
     pf_requeue_max，默认 1，按块 pf_requeues 计）→ 复活 DIRTY（走既有 dirty 通道：
     final.json→backup、quality_retries 在批次入口清零、失败由 _restore_dirty_backups
     恢复旧版）；预算用尽 → pf_over_cap（冻结）。
  2. PF 复活不受 backtrack_rounds 上限约束（终态块与回溯冻结是两类根因）。
  3. CLI `requeue [--force]`：显式复活通道（force 绕预算）。
"""
from __future__ import annotations

import importlib

import pytest

from src.config import config as _cfg


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


@pytest.fixture(params=SCHED_MIRROR, ids=[m[0] for m in SCHED_MIRROR])
def sched_mod(request):
    return importlib.import_module(request.param[1])


def _seed(sched_mod, tmp_path, rows):
    """rows: [(chunk_id, TaskState, backtrack_rounds, pf_requeues)]"""
    sched = sched_mod.TaskScheduler(db_path=str(tmp_path / "wf.db"))
    cur = sched.conn.cursor()
    for cid, state, rounds, pf_n in rows:
        cur.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state, "
            "backtrack_rounds, pf_requeues) VALUES (?,?,?,?,?,?)",
            (cid, cid.split("_")[0], "x", state.value, rounds, pf_n))
    sched.conn.commit()
    return sched


def _row(sched, chunk_id):
    return sched.conn.execute(
        "SELECT state, COALESCE(backtrack_rounds,0), COALESCE(pf_requeues,0) "
        "FROM chunk_tasks WHERE chunk_id=?", (chunk_id,)).fetchone()


class TestPfReviveViaBacktrack:
    def test_pf_revived_within_budget(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=1, max_backtrack_rounds_per_chunk=2)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [
            ("ch01_c1", S.PERMANENTLY_FAILED, 2, 0),   # 已到回溯上限的终态块
            ("ch01_c2", S.COMPLETED, 0, 0),
        ])
        res = sched.trigger_backtrack(["ch01_c1", "ch01_c2"])
        assert res["pf_requeued"] == ["ch01_c1"]
        assert "ch01_c1" in res["dirty"], "PF 复活应计入 dirty（驱动主循环重跑）"
        st, rounds, pf_n = _row(sched, "ch01_c1")
        assert st == S.DIRTY.value
        assert pf_n == 1
        assert rounds == 3, "PF 复活不受回溯上限约束（轮数仅记录）"
        assert _row(sched, "ch01_c2")[0] == S.DIRTY.value  # 常规回溯不受影响

    def test_second_failure_over_cap(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=1, max_backtrack_rounds_per_chunk=2)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [("ch01_c1", S.PERMANENTLY_FAILED, 3, 1)])
        res = sched.trigger_backtrack(["ch01_c1"])
        assert res["pf_over_cap"] == ["ch01_c1"]
        assert res["pf_requeued"] == [] and res["dirty"] == []
        st, _, pf_n = _row(sched, "ch01_c1")
        assert st == S.PERMANENTLY_FAILED.value and pf_n == 1, "超预算不得改状态"

    def test_budget_zero_disables_revive(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=0, max_backtrack_rounds_per_chunk=2)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [("ch01_c1", S.PERMANENTLY_FAILED, 0, 0)])
        res = sched.trigger_backtrack(["ch01_c1"])
        assert res["pf_over_cap"] == ["ch01_c1"] and res["pf_requeued"] == []
        assert _row(sched, "ch01_c1")[0] == S.PERMANENTLY_FAILED.value


class TestExplicitRequeue:
    def test_requeue_chapter_scan(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=1)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [
            ("ch01_c1", S.PERMANENTLY_FAILED, 2, 0),
            ("ch01_c2", S.COMPLETED, 0, 0),
            ("ch01_c3", S.PENDING, 0, 0),
        ])
        res = sched.requeue_permanently_failed("ch01")
        assert res["requeued"] == ["ch01_c1"]
        assert set(res["skipped"]) == {"ch01_c2", "ch01_c3"}
        assert _row(sched, "ch01_c1")[0] == S.DIRTY.value

    def test_requeue_force_bypasses_budget(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=1)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [("ch01_c1", S.PERMANENTLY_FAILED, 3, 5)])
        # 无 force → 超预算
        res0 = sched.requeue_permanently_failed("ch01")
        assert res0["over_cap"] == ["ch01_c1"] and res0["requeued"] == []
        # --force → 强制复活
        res1 = sched.requeue_permanently_failed("ch01", force=True)
        assert res1["requeued"] == ["ch01_c1"]
        st, _, pf_n = _row(sched, "ch01_c1")
        assert st == S.DIRTY.value and pf_n == 6

    def test_requeue_explicit_chunk_ids(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=1)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [
            ("ch01_c1", S.PERMANENTLY_FAILED, 0, 0),
            ("ch01_c2", S.PERMANENTLY_FAILED, 0, 0),
        ])
        res = sched.requeue_permanently_failed("ch01", chunk_ids=["ch01_c2"])
        assert res["requeued"] == ["ch01_c2"]
        assert _row(sched, "ch01_c1")[0] == S.PERMANENTLY_FAILED.value
        assert _row(sched, "ch01_c2")[0] == S.DIRTY.value


class TestNormalGovernanceIntact:
    def test_completed_over_cap_still_frozen(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=1, max_backtrack_rounds_per_chunk=2)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [("ch01_c1", S.COMPLETED, 2, 0)])
        res = sched.trigger_backtrack(["ch01_c1"])
        assert res["over_cap"] == ["ch01_c1"] and res["dirty"] == []
        assert _row(sched, "ch01_c1")[0] == S.COMPLETED.value

    def test_mid_states_still_skipped(self, sched_mod, tmp_path, set_de_cfg):
        set_de_cfg(pf_requeue_max=1, max_backtrack_rounds_per_chunk=2)
        S = sched_mod.TaskState
        sched = _seed(sched_mod, tmp_path, [("ch01_c1", S.AUDITING, 0, 0)])
        res = sched.trigger_backtrack(["ch01_c1"])
        assert res["skipped_state"] == ["ch01_c1"]
        assert _row(sched, "ch01_c1")[0] == S.AUDITING.value
