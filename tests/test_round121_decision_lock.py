"""Round 121: 决策锁定机制回归测试（双镜像）。

背景：跨章 rerun 时 LLM 对同一 source_key 的译文翻转（如
Mock Turtle -> 假海龟/假甲鱼/仿海龟）会经 _find_chunks_containing
放大成全书回溯，触发 DIRTY 乒乓不收敛（R5 consolidate 硬闸根因）。

语义：decision_db.locked=1 的决策拒绝覆盖性更新，且不触发 backtrack；
同译文写入与新建 source_key 不受影响。

镜像：src/translator_agent.py 与 src/core/decision_engine.py 行为须一致。
"""
import sqlite3
import importlib

import pytest


@pytest.fixture(autouse=True)
def _force_immediate_backtrack_mode():
    """Round 130: 决策引擎默认已切换 deferred（章末合并）模式；
    本文件回归的是 immediate 模式的即时回溯语义，显式固定之（含单体镜像树）。"""
    from src.config import config as _cfg
    de = _cfg._data.setdefault("decision_engine", {})
    old = de.get("backtrack_mode")
    de["backtrack_mode"] = "immediate"
    yield
    if old is None:
        de.pop("backtrack_mode", None)
    else:
        de["backtrack_mode"] = old


DECISION_LOCK_CASES = [
    ("translator_agent", "src.translator_agent"),
    ("core", "src.core.decision_engine"),
]


class _FakeScheduler:
    def __init__(self):
        self.calls = []

    def trigger_backtrack(self, chunk_ids):
        self.calls.append(list(chunk_ids))


@pytest.fixture(params=DECISION_LOCK_CASES, ids=[c[0] for c in DECISION_LOCK_CASES])
def engine_mod(request, tmp_path):
    name, modname = request.param
    mod = importlib.import_module(modname)
    return mod, tmp_path / f"decision_{name}.sqlite"


def _lock(engine, source_key):
    cur = engine.conn.cursor()
    cur.execute("UPDATE decision_db SET locked = 1 WHERE source_key = ?", (source_key,))
    engine.conn.commit()


def _translation(engine, source_key):
    cur = engine.conn.cursor()
    cur.execute("SELECT translation FROM decision_db WHERE source_key = ?", (source_key,))
    return cur.fetchone()[0]


def test_migration_adds_locked_column(engine_mod):
    mod, db_path = engine_mod
    engine = mod.DecisionEngine(db_path=str(db_path))
    cur = engine.conn.cursor()
    cur.execute("PRAGMA table_info(decision_db)")
    cols = {row[1] for row in cur.fetchall()}
    assert "locked" in cols


def test_unlocked_flip_updates_and_backtracks(engine_mod):
    mod, db_path = engine_mod
    sched = _FakeScheduler()
    engine = mod.DecisionEngine(db_path=str(db_path), scheduler_factory=lambda: sched)
    engine.add_decision(mod.DecisionLevel.REFERENCE, "Mock Turtle", "假海龟",
                        "r1", affected_chunks=["ch10_c000"])
    engine.add_decision(mod.DecisionLevel.REFERENCE, "Mock Turtle", "素甲鱼",
                        "r2", affected_chunks=["ch10_c000"])
    assert _translation(engine, "Mock Turtle") == "素甲鱼"
    assert sched.calls == [["ch10_c000"], ["ch10_c000"]]


def test_locked_flip_ignored_without_backtrack(engine_mod):
    mod, db_path = engine_mod
    sched = _FakeScheduler()
    engine = mod.DecisionEngine(db_path=str(db_path), scheduler_factory=lambda: sched)
    engine.add_decision(mod.DecisionLevel.REFERENCE, "Gryphon", "狮鹫",
                        "r1", affected_chunks=["ch10_c001"])
    _lock(engine, "Gryphon")

    engine.add_decision(mod.DecisionLevel.REFERENCE, "Gryphon", "鹰头狮",
                        "flip", affected_chunks=["ch10_c001", "ch13_c008"])
    assert _translation(engine, "Gryphon") == "狮鹫"
    # 锁定后的翻转不触发回溯（此前只有首次写入触发 1 次）
    assert sched.calls == [["ch10_c001"]]


def test_locked_same_translation_still_ok(engine_mod):
    mod, db_path = engine_mod
    sched = _FakeScheduler()
    engine = mod.DecisionEngine(db_path=str(db_path), scheduler_factory=lambda: sched)
    engine.add_decision(mod.DecisionLevel.REFERENCE, "Gryphon", "狮鹫",
                        "r1", affected_chunks=["ch10_c001"])
    _lock(engine, "Gryphon")

    # 同译文重复写入：不报错、不改写、不新增回溯
    engine.add_decision(mod.DecisionLevel.REFERENCE, "Gryphon", "狮鹫",
                        "r2", affected_chunks=["ch10_c001"])
    assert _translation(engine, "Gryphon") == "狮鹫"
    assert sched.calls == [["ch10_c001"]]


def test_new_source_key_insertable_while_locked_rows_exist(engine_mod):
    mod, db_path = engine_mod
    sched = _FakeScheduler()
    engine = mod.DecisionEngine(db_path=str(db_path), scheduler_factory=lambda: sched)
    engine.add_decision(mod.DecisionLevel.REFERENCE, "Gryphon", "狮鹫", "r1",
                        affected_chunks=["ch10_c001"])
    _lock(engine, "Gryphon")

    engine.add_decision(mod.DecisionLevel.TERMINOLOGY, "whiting", "牙鳕", "new",
                        affected_chunks=["ch11_c005"])
    assert _translation(engine, "whiting") == "牙鳕"
    assert sched.calls == [["ch10_c001"], ["ch11_c005"]]


def test_locked_row_survives_reopen(engine_mod):
    """锁是持久化的：重开引擎后翻转仍被拒。"""
    mod, db_path = engine_mod
    sched = _FakeScheduler()
    engine = mod.DecisionEngine(db_path=str(db_path), scheduler_factory=lambda: sched)
    engine.add_decision(mod.DecisionLevel.REFERENCE, "Mock Turtle", "假海龟", "r1",
                        affected_chunks=["ch10_c000"])
    _lock(engine, "Mock Turtle")
    del engine

    engine2 = mod.DecisionEngine(db_path=str(db_path), scheduler_factory=lambda: _FakeScheduler())
    engine2.add_decision(mod.DecisionLevel.REFERENCE, "Mock Turtle", "仿海龟", "flip",
                         affected_chunks=["ch10_c000"])
    assert _translation(engine2, "Mock Turtle") == "假海龟"


REFAGENT_CASES = [
    ("monolith", "src.translator_agent"),
    ("agents", "src.agents.reference_agent"),
]


@pytest.fixture(params=REFAGENT_CASES, ids=[c[0] for c in REFAGENT_CASES])
def refagent_mod(request):
    name, modname = request.param
    return importlib.import_module(modname)


def _seed_row(engine, canonical, source_json, variants_json, confidence="high"):
    cur = engine.conn.cursor()
    cur.execute(
        "INSERT INTO entity_registry (canonical_name, source_names, entity_type, "
        "confidence, pending_review, variant_names) VALUES (?, ?, 'person', ?, 0, ?)",
        (canonical, source_json, confidence, variants_json))
    engine.conn.commit()


def test_known_variant_merge_no_repending(refagent_mod, tmp_path):
    """新中文名已是登记异体 → 并入 source_names，不再标待核查。"""
    mod = refagent_mod
    engine = mod.DecisionEngine(db_path=str(tmp_path / "d.sqlite"))
    _seed_row(engine, "三月兔", '["March Hare"]', '["三月野兔"]')

    agent = mod.ReferenceAgent.__new__(mod.ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch07"
    agent._register_entities("ch07_chunk001", [{
        "canonical_name": "三月野兔",
        "source_names": ["a March Hare"],
        "entity_type": "person",
        "confidence": "high",
    }])

    cur = engine.conn.cursor()
    cur.execute("select pending_review, source_names from entity_registry where canonical_name='三月兔'")
    pending, names = cur.fetchone()
    assert pending == 0, "已知异体不应重标待核查"
    assert "a March Hare" in names
    cur.execute("select count(*) from entity_registry where canonical_name='三月野兔'")
    assert cur.fetchone()[0] == 0, "已知异体不应新建条目"


def test_unknown_similar_still_flagged(refagent_mod, tmp_path):
    """新中文名不在 variant_names → 仍按旧行为标待核查（保守）。"""
    mod = refagent_mod
    engine = mod.DecisionEngine(db_path=str(tmp_path / "d.sqlite"))
    _seed_row(engine, "三月兔", '["March Hare"]', '[]')

    agent = mod.ReferenceAgent.__new__(mod.ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch07"
    agent._register_entities("ch07_chunk001", [{
        "canonical_name": "三月野兔",
        "source_names": ["a March Hare"],
        "entity_type": "person",
        "confidence": "high",
    }])

    cur = engine.conn.cursor()
    cur.execute("select pending_review from entity_registry where canonical_name='三月兔'")
    assert cur.fetchone()[0] == 1


def test_wh_word_source_names_filtered(refagent_mod, tmp_path):
    """疑问词泄漏护栏：'who' 等不得进入 source_names（ch06 硬失败循环根因）。"""
    mod = refagent_mod
    engine = mod.DecisionEngine(db_path=str(tmp_path / "d.sqlite"))
    agent = mod.ReferenceAgent.__new__(mod.ReferenceAgent)
    agent.db = engine
    agent.chapter_id = "ch06"
    agent._register_entities("ch06_chunk000", [{
        "canonical_name": "新角色",
        "source_names": ["who", "whom", "the Newcomer"],
        "entity_type": "person",
        "confidence": "high",
    }])
    cur = engine.conn.cursor()
    cur.execute("select source_names from entity_registry where canonical_name='新角色'")
    names = cur.fetchone()[0]
    assert "who" not in names and "whom" not in names
    assert "the Newcomer" in names
