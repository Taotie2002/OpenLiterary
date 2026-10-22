"""Round 119 (T4): 完成信号不变量化 — _verify_final_outputs DIRTY 语义 + CLI gate.

背景（D2/D3）: 旧流程章节 verify 把 DIRTY 且带 final 的"回溯前陈旧稿"当
fallback 放行（return True），merge 不检查状态，97/126 DIRTY 直接进成书；
完成信号（退出码/🎉）与实际不变量不一致。

测试目标（对应方案 T4 验收）:
  1. _verify_final_outputs: DIRTY 且带 final → False（"陈旧待整合"）;
     DIRTY 收敛为 COMPLETED 后 → True; fallback 语义（round76）不动。
  2. run_final_gate 五类不变量: ①状态机(COMPLETED/DIRTY) ②产物(final 数/
     final_backup) ③QA_REPORT ④lint ⑤T1 转换审计 → 全绿 ok=True；
     任一红 ok=False + 失败项定位（构造 DIRTY=1 必红）。
"""
from __future__ import annotations

import json

from src.core.scheduler import TaskScheduler
from src.pipeline import TranslationPipeline, run_final_gate

_CLEAN = "# 第一章 起点\n\n他说：“你好。”然后[^1]。\n\n[^1]: 说明。\n"
_AUDIT_PASS = "# 转换完整性审计报告 (T1)\n\n- 全书覆盖率: 99.900%\n- 结论: ✅ 通过\n"
_AUDIT_FAIL = "# 转换完整性审计报告 (T1)\n\n- 全书覆盖率: 97.000%\n- 结论: ❌ 未通过\n"


def _mk_project(tmp_path, tasks, finals, merged=_CLEAN, audit=_AUDIT_PASS,
                qa="", backups=0):
    db = tmp_path / "db"
    db.mkdir(exist_ok=True)
    sched = TaskScheduler(db_path=str(db / "workflow.db"))
    for cid, ch, st in tasks:
        sched.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?,?,?,?)", (cid, ch, "src " + cid, st))
    sched.conn.commit()
    out = tmp_path / "output"
    for ch, cid, text in finals:
        d = out / ch
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cid}_final.json").write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    out.mkdir(exist_ok=True)
    if merged is not None:
        (out / "translated_full.md").write_text(merged, encoding="utf-8")
    (out / "QA_REPORT.md").write_text(
        "# 译文质量验收报告 (QA Report)\n\n" + qa, encoding="utf-8")
    if audit is not None:
        (tmp_path / "input").mkdir(exist_ok=True)
        (tmp_path / "input" / "_conversion_audit.md").write_text(audit, encoding="utf-8")
    d01 = out / "ch01"
    d01.mkdir(exist_ok=True)
    for i in range(backups):
        (d01 / f"ch01_chunk9{i:02d}_final_backup.json").write_text("{}", encoding="utf-8")
    return sched


# ── 1. _verify_final_outputs DIRTY 语义 ──────────────────────────────

def test_verify_final_outputs_dirty_then_converged(tmp_path, capsys):
    db = tmp_path / "db"
    db.mkdir()
    sched = TaskScheduler(db_path=str(db / "workflow.db"))
    for cid, st in [("ch_test_chunk000", "COMPLETED"),
                    ("ch_test_chunk001", "DIRTY")]:
        sched.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?,?,?,?)", (cid, "ch_test", "src", st))
    sched.conn.commit()
    out = tmp_path / "output" / "ch_test"
    out.mkdir(parents=True)
    for cid in ("ch_test_chunk000", "ch_test_chunk001"):
        (out / f"{cid}_final.json").write_text(
            json.dumps({"text": "译"}, ensure_ascii=False), encoding="utf-8")

    p = TranslationPipeline.__new__(TranslationPipeline)
    p.chapter_id = "ch_test"
    p.output_dir = out
    p.scheduler = sched

    assert p._verify_final_outputs(2, strict=True) is False
    captured = capsys.readouterr().out
    assert "DIRTY" in captured and "陈旧待整合" in captured

    sched.conn.execute("UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id=?",
                       ("ch_test_chunk001",))
    sched.conn.commit()
    assert p._verify_final_outputs(2, strict=True) is True


# ── 2. run_final_gate 五类不变量 ─────────────────────────────────────

def test_gate_green(tmp_path, capsys):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED")],
                finals=[("ch01", "ch01_chunk000", _CLEAN)])
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is True, res["failures"]
    assert res["failures"] == []
    assert res["stats"]["tasks"] == 1
    assert res["stats"]["states"] == {"COMPLETED": 1}
    assert res["stats"]["qa_fallbacks"] == 0
    assert res["stats"]["audit_coverage"] == 99.9
    assert "gate 全绿" in capsys.readouterr().out


def test_gate_dirty_red(tmp_path):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED"),
                       ("ch01_chunk001", "ch01", "DIRTY")],
                finals=[("ch01", "ch01_chunk000", _CLEAN),
                        ("ch01", "ch01_chunk001", _CLEAN)])
    res = run_final_gate(project_root=str(tmp_path))
    joined = "\n".join(res["failures"])
    assert res["ok"] is False
    assert "① DIRTY=1" in joined
    assert "非 COMPLETED 任务 1 个" in joined


def test_gate_non_completed_red(tmp_path):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED"),
                       ("ch01_chunk001", "ch01", "PERMANENTLY_FAILED")],
                finals=[("ch01", "ch01_chunk000", _CLEAN),
                        ("ch01", "ch01_chunk001", _CLEAN)])
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is False
    assert "PERMANENTLY_FAILED" in "\n".join(res["failures"])


def test_gate_missing_final_red(tmp_path):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED")],
                finals=[])
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is False
    assert "② ch01: final.json 0 个 != DB 任务 1 个" in "\n".join(res["failures"])


def test_gate_backup_red(tmp_path):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED")],
                finals=[("ch01", "ch01_chunk000", _CLEAN)],
                backups=1)
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is False
    assert "② 残留 final_backup 1 个" in "\n".join(res["failures"])


def test_gate_qa_fallback_red(tmp_path):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED")],
                finals=[("ch01", "ch01_chunk000", _CLEAN)],
                qa="## ch02\n- 状态: 未通过自动化验收（已 fallback）\n")
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is False
    joined = "\n".join(res["failures"])
    assert "③ QA_REPORT 记录 1 个 fallback 章节" in joined
    assert "ch02" in joined


def test_gate_lint_red(tmp_path):
    dirty_text = '他说"你好"。\n'
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED")],
                finals=[("ch01", "ch01_chunk000", dirty_text)])
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is False
    assert "④ lint 未通过" in "\n".join(res["failures"])


def test_gate_audit_missing_red(tmp_path):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED")],
                finals=[("ch01", "ch01_chunk000", _CLEAN)],
                audit=None)
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is False
    assert "⑤ 转换审计报告不存在" in "\n".join(res["failures"])


def test_gate_audit_not_passed_red(tmp_path):
    _mk_project(tmp_path,
                tasks=[("ch01_chunk000", "ch01", "COMPLETED")],
                finals=[("ch01", "ch01_chunk000", _CLEAN)],
                audit=_AUDIT_FAIL)
    res = run_final_gate(project_root=str(tmp_path))
    assert res["ok"] is False
    assert "⑤ 转换审计未通过" in "\n".join(res["failures"])
