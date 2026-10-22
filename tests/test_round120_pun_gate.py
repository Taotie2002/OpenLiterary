"""Round 120 (L2): gate ⑦ + _seed_pun_decisions（双文件镜像验证）。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.core.scheduler import TaskScheduler
from src.pipeline import run_final_gate

_CLEAN = "# 第一章 起点\n\n他说：“你好。”然后[^1]。\n\n[^1]: 说明。\n"
_AUDIT_PASS = "# 转换完整性审计报告 (T1)\n\n- 全书覆盖率: 99.900%\n- 结论: ✅ 通过\n"


def _mk(tmp_path, watch=True, pun_json=None):
    db = tmp_path / "db"
    db.mkdir(exist_ok=True)
    sched = TaskScheduler(db_path=str(db / "workflow.db"))
    sched.conn.execute(
        "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) VALUES (?,?,?,?)",
        ("ch01_chunk000", "ch01", "src", "COMPLETED"))
    sched.conn.commit()
    out = tmp_path / "output"
    (out / "ch01").mkdir(parents=True, exist_ok=True)
    (out / "ch01" / "ch01_chunk000_final.json").write_text(
        json.dumps({"text": _CLEAN}, ensure_ascii=False), encoding="utf-8")
    (out / "translated_full.md").write_text(_CLEAN, encoding="utf-8")
    (out / "QA_REPORT.md").write_text("# 译文质量验收报告\n", encoding="utf-8")
    inp = tmp_path / "input"
    inp.mkdir(exist_ok=True)
    (inp / "_conversion_audit.md").write_text(_AUDIT_PASS, encoding="utf-8")
    if watch:
        (inp / "book_profile.json").write_text(json.dumps({
            "genre": "童话", "source": "sampled", "confidence": 0.5,
            "pun_watchlist": [{"source": "tale/tail", "anchors": ["tale"], "strategy": "x"}],
        }, ensure_ascii=False), encoding="utf-8")
    if pun_json is not None:
        (out / "pun_audit.json").write_text(json.dumps(pun_json, ensure_ascii=False),
                                            encoding="utf-8")
    return tmp_path


def test_gate_pun_missing_red(tmp_path):
    root = _mk(tmp_path, watch=True, pun_json=None)
    res = run_final_gate(project_root=str(root))
    assert not res["ok"]
    assert any("⑦ pun_audit 未运行" in f for f in res["failures"])


def test_gate_pun_uncovered_red(tmp_path):
    root = _mk(tmp_path, watch=True,
               pun_json={"summary": {"total": 1, "covered": 0, "waived": 0,
                                     "uncovered_nonwaived": 1}})
    res = run_final_gate(project_root=str(root))
    assert not res["ok"]
    assert any("⑦ pun_watchlist 未核销" in f for f in res["failures"])


def test_gate_pun_covered_green(tmp_path):
    root = _mk(tmp_path, watch=True,
               pun_json={"summary": {"total": 1, "covered": 1, "waived": 0,
                                     "uncovered_nonwaived": 0}})
    res = run_final_gate(project_root=str(root))
    assert not any("⑦" in f for f in res["failures"])


def test_gate_pun_skipped_without_watchlist(tmp_path):
    root = _mk(tmp_path, watch=False)
    res = run_final_gate(project_root=str(root))
    assert not any("⑦" in f for f in res["failures"])


def _pipeline(guide, engine):
    from src.translator_agent import TranslationPipeline
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.style_guide = guide
    p.decision_engine = engine
    return p


def test_seed_pun_decisions_idempotent():
    engine = MagicMock()
    p = _pipeline({"pun_watchlist": [
        {"source": "tale/tail", "anchors": ["tale", "tail"], "strategy": "双关落点"}]}, engine)
    assert p._seed_pun_decisions("A long and sad tale!") == 1
    args = engine.add_decision.call_args
    assert args.args[1] == "tale"
    assert args.kwargs.get("affected_chunks") is None
    assert p._seed_pun_decisions("A long and sad tale!") == 0
    assert engine.add_decision.call_count == 1


def test_seed_pun_decisions_tolerates_bad_engine():
    engine = MagicMock()
    engine.add_decision.side_effect = RuntimeError("boom")
    p = _pipeline({"pun_watchlist": [
        {"source": "draw", "anchors": ["draw"], "strategy": "x"}]}, engine)
    assert p._seed_pun_decisions("draw a picture") == 0
    p2 = _pipeline("", engine)
    assert p2._seed_pun_decisions("draw") == 0


def test_seed_pun_decisions_uses_src_anchors():
    engine = MagicMock()
    p = _pipeline({"pun_watchlist": [
        {"source": "whiting (fish vs. shoe-blacking)", "src_anchors": ["whiting"],
         "anchors": ["牙鳕", "白垩粉"], "strategy": "双关"}]}, engine)
    assert p._seed_pun_decisions("‘The whiting,’ said the Gryphon.") == 1
    assert engine.add_decision.call_args.args[1] == "whiting"
