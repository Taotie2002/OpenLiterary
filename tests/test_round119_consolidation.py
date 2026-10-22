"""Round 119 (T2): 全书决策收敛闭环 (consolidate).

背景（D2/D3）: 上一轮全量跑结束时 97/126 chunk 仍 DIRTY（回溯标记
无消费点），merge 不检查状态，会把"回溯前"陈旧 final 收录进成书。

测试目标:
  1. 合成场景（chA 决策把已完成的 chB 标脏）→ consolidate 重跑 chB →
     DIRTY=0 + consolidation_diff.md 记录旧/新差异。
  2. 级联: 第 1 轮重跑又标脏另一章 → 第 2 轮继续收敛。
  3. 不收敛: 管线无法清掉 DIRTY → converged=False + remaining 清单 +
     轮数封顶 max_rounds。
"""
from __future__ import annotations

import json

from src.core.scheduler import TaskState, TaskScheduler
from src.pipeline import consolidate_dirty_chapters


def _mk_project(tmp_path, rows, finals):
    (tmp_path / "db").mkdir(exist_ok=True)
    out = tmp_path / "output"
    sched = TaskScheduler(db_path=str(tmp_path / "db" / "workflow.db"))
    for cid, ch, st in rows:
        sched.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?,?,?,?)", (cid, ch, "source " + cid, st))
    sched.conn.commit()
    for ch, cid, text in finals:
        d = out / ch
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cid}_final.json").write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    return sched, out


def _fake_factory(handler):
    def factory(chapter_id):
        class _Fake:
            def run(self):
                return handler(chapter_id)
        return _Fake()
    return factory


def _set_state(sched, chunk_id, state):
    sched.conn.execute("UPDATE chunk_tasks SET state=? WHERE chunk_id=?",
                       (state.value, chunk_id))
    sched.conn.commit()


def _write_final(out, ch, cid, text):
    (out / ch / f"{cid}_final.json").write_text(
        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


def _dirty_count(sched):
    return sched.conn.execute(
        "SELECT COUNT(*) FROM chunk_tasks WHERE state='DIRTY'").fetchone()[0]


def test_single_round_converges_and_reports(tmp_path):
    sched, out = _mk_project(
        tmp_path,
        rows=[("ch01_chunk000", "ch01", "COMPLETED"),
              ("ch02_chunk000", "ch02", "DIRTY")],
        finals=[("ch01", "ch01_chunk000", "定稿一"),
                ("ch02", "ch02_chunk000", "回溯前旧稿")],
    )

    def handler(ch):
        _set_state(sched, f"{ch}_chunk000", TaskState.COMPLETED)
        _write_final(out, ch, f"{ch}_chunk000", "决策收敛后新稿 " + ch)
        return True

    res = consolidate_dirty_chapters(
        project_root=tmp_path, max_rounds=3, pipeline_factory=_fake_factory(handler))

    assert res["converged"] is True
    assert res["rounds"] == 1
    assert res["rerun"] == ["ch02_chunk000"]
    assert res["remaining"] == {}
    assert _dirty_count(sched) == 0
    report_path = out / "consolidation_diff.md"
    assert report_path.exists()
    report = report_path.read_text(encoding="utf-8")
    assert "ch02_chunk000" in report
    assert "文本变化" in report
    assert "回溯前旧稿" in report and "决策收敛后新稿" in report
    assert res["diff_path"] == str(report_path)


def test_cascade_marks_other_chapter_takes_two_rounds(tmp_path):
    sched, out = _mk_project(
        tmp_path,
        rows=[("ch01_chunk000", "ch01", "COMPLETED"),
              ("ch02_chunk000", "ch02", "DIRTY")],
        finals=[("ch01", "ch01_chunk000", "旧稿A"),
                ("ch02", "ch02_chunk000", "旧稿B")],
    )

    def handler(ch):
        if ch == "ch02":
            _set_state(sched, "ch02_chunk000", TaskState.COMPLETED)
            _write_final(out, ch, "ch02_chunk000", "新稿B")
            # 级联：ch02 的新决策把已完成的 ch01 标脏
            _set_state(sched, "ch01_chunk000", TaskState.DIRTY)
        else:
            _set_state(sched, "ch01_chunk000", TaskState.COMPLETED)
            _write_final(out, ch, "ch01_chunk000", "新稿A")
        return True

    res = consolidate_dirty_chapters(
        project_root=tmp_path, max_rounds=3, pipeline_factory=_fake_factory(handler))

    assert res["converged"] is True
    assert res["rounds"] == 2
    assert res["rerun"] == ["ch02_chunk000", "ch01_chunk000"]
    assert _dirty_count(sched) == 0


def test_non_convergence_reports_remaining(tmp_path):
    sched, out = _mk_project(
        tmp_path,
        rows=[("ch02_chunk000", "ch02", "DIRTY")],
        finals=[("ch02", "ch02_chunk000", "旧稿")],
    )

    res = consolidate_dirty_chapters(
        project_root=tmp_path, max_rounds=2,
        pipeline_factory=_fake_factory(lambda ch: True))

    assert res["converged"] is False
    assert res["rounds"] == 2
    assert res["remaining"] == {"ch02": ["ch02_chunk000"]}
    assert _dirty_count(sched) == 1
    report = (out / "consolidation_diff.md").read_text(encoding="utf-8")
    assert "未收敛残留" in report
    assert "ch02_chunk000" in report
    assert "文本无变化" in report  # handler 未写新稿 → old == new


def test_zero_dirty_is_noop(tmp_path):
    sched, out = _mk_project(
        tmp_path,
        rows=[("ch01_chunk000", "ch01", "COMPLETED")],
        finals=[("ch01", "ch01_chunk000", "定稿")],
    )
    res = consolidate_dirty_chapters(
        project_root=tmp_path, max_rounds=3,
        pipeline_factory=_fake_factory(lambda ch: True))
    assert res["converged"] is True
    assert res["rounds"] == 0
    assert res["rerun"] == []
    report = (out / "consolidation_diff.md").read_text(encoding="utf-8")
    assert "无重跑" in report
