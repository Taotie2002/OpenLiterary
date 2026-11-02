"""Round 135b: 章末执行顺序 — 备份恢复/版本择优必须先于章末归一与审计。

根因（ch01 Round 135 run 实测）：run() 章末原顺序为
[归一 → 一致性 → 冲突审计 → 实体审计 → cost → cleanup → 备份恢复/择优 → verify]。
版本择优回滚（final.json ← final_backup.json）发生在归一之后，会把刚应用到
新版 final.json 的归一替换静默抹掉（本次 5 块择优回滚：chunk031/034/072/074/
075/078）；且 entity_audit / consistency 基于回滚前文本判定，口径失真。
本次侥幸无残留（备份恰为上一轮已归一文本），但一旦出现"本轮新注册映射"
（如 Piter 新译名）即成真。修复：_restore_dirty_backups() 提前至章末归一之前。

本测试用 stub Pipeline 驱动 run() 的真实章末序列，锁定顺序不变式。
"""
from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

from src.core.scheduler import TaskScheduler


MIRROR = [
    ("monolith", "src.translator_agent"),
    ("modular", "src.pipeline"),
]


@pytest.fixture(params=MIRROR, ids=[m[0] for m in MIRROR])
def pipeline_mod(request):
    return importlib.import_module(request.param[1])


def _make_run_stub(module, tmp_path, order):
    """驱动 run() 走完主循环（无任务）直达章末序列，记录各步调用顺序。"""
    Pipeline = module.TranslationPipeline
    p = Pipeline.__new__(Pipeline)
    p.chapter_id = "ch_test"
    p.output_dir = tmp_path
    p.decision_engine = None
    sched = MagicMock(spec=TaskScheduler)
    sched.get_all_tasks_by_chapter.return_value = [
        {"chunk_id": "ch_test_c000", "state": "PENDING", "text_content": "x"}
    ]
    sched.get_tasks_by_state.return_value = []
    p.scheduler = sched

    def rec(name, rv=None):
        def _fn(*args, **kwargs):
            order.append(name)
            return rv
        return _fn

    p._restore_dirty_backups = rec("restore", [])
    p._maybe_unify_entity_registry = rec("unify", 0)
    p._run_post_consistency = rec("consistency")
    p._entity_registry_conflict_audit = rec("conflicts", [])
    p._post_pipeline_entity_audit = rec("audit", {})
    p._print_llm_cost_report = rec("cost")
    p._cleanup_intermediates = rec("cleanup")
    p._verify_final_outputs = rec("verify", True)
    return p


class TestChapterEndOrder:
    def test_restore_before_unify_and_audits(self, pipeline_mod, tmp_path):
        order = []
        p = _make_run_stub(pipeline_mod, tmp_path, order)
        ok = p.run()
        assert ok is True
        assert order == [
            "restore",     # Round 135b: 备份恢复/择优（先落地最终内容）
            "unify",       # 归一（对最终内容生效，替换不被回滚）
            "consistency",
            "conflicts",
            "audit",       # 实体审计基于最终内容（口径准确）
            "cost",
            "cleanup",
            "verify",
        ], f"章末顺序被改坏: {order}"

    def test_restore_strictly_before_unify(self, pipeline_mod, tmp_path):
        order = []
        p = _make_run_stub(pipeline_mod, tmp_path, order)
        p.run()
        assert order.index("restore") < order.index("unify"), \
            "版本择优回滚会抹掉归一替换（Round 135b 根因回归）"
