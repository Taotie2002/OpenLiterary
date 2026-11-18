"""Round 158 Phase 1.5: read-only pending_backtrack reconciliation."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.core.scheduler import TaskScheduler
from src.judgment_layer import sha256_text
from src.pending_backtrack_reconcile import (
    _source_phrase_spans,
    _target_core,
    apply_pending_backtrack_reconcile,
    audit_pending_backtracks,
    classify_pending_snapshot,
    load_pending_context,
)


def test_article_led_term_requires_capitalisation():
    assert _source_phrase_spans("Using the Voice on me.", "the Voice") == [(6, 15)]
    assert _source_phrase_spans("The voice was low.", "the Voice") == []
    assert _source_phrase_spans(
        "the Voice from the Outer World", "the Voice") == []
    assert _source_phrase_spans(
        "She used the Voice on him, then looked at the outer world.",
        "the Voice",
    ) == [(9, 18)]


def test_target_core_extracts_latest_explicit_instruction():
    translation = (
        "[双关·保留笑点] 正文首次写作“短距者（Kwisatz Haderach）”，"
        "此后统一“短距者”；用尾注解释“缩短距离”的字面双关。"
    )

    assert _target_core(translation) == "短距者"
    assert _target_core("“保护性传教团（Missionaria Protectiva）”") == "保护性传教团"


def test_article_shadow_closes_only_when_canonical_target_is_present(tmp_path: Path):
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk040", "ch01", "the Lady Jessica spoke.", "COMPLETED"),
        ("ch01_chunk041", "ch01", "the Lady Jessica waited.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("DELETE FROM pending_backtrack")
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (54, 2, 'Lady Jessica', '杰西卡夫人', "
            "'2026-11-16 12:00:00', 0)")
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (3488, 2, 'the Lady Jessica', '杰西卡女士', "
            "'2026-11-16 12:00:00', 0)")
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, 3488, "
            "'the Lady Jessica', 1, '2026-11-16 12:00:00', "
            "'2026-11-16 12:00:00')",
            [("ch01_chunk040",), ("ch01_chunk041",)],
        )
        con.commit()
    finally:
        con.close()
    _write_final(tmp_path, "ch01_chunk040", "杰西卡夫人开口。")
    _write_final(tmp_path, "ch01_chunk041", "杰西卡女士等待着。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")

    result = classify_pending_snapshot(load_pending_context(tmp_path))
    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}

    assert by_id["ch01_chunk040"]["status"] == "satisfied"
    assert by_id["ch01_chunk040"]["reason"] == "article_shadow_canonical_present"
    assert by_id["ch01_chunk041"]["status"] == "outstanding"
    assert by_id["ch01_chunk041"]["reason"] == "article_shadow_canonical_missing"


def _make_workflow(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE chunk_tasks ("
            "chunk_id TEXT PRIMARY KEY, chapter_id TEXT, text_content TEXT, "
            "state TEXT, last_error TEXT, backtrack_rounds INTEGER, rescue_stage TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        con.executemany(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state, "
            "backtrack_rounds) VALUES (?, ?, ?, ?, 0)",
            rows,
        )
        con.commit()
    finally:
        con.close()


def _make_decision_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, reason TEXT, updated_at TEXT, "
            "locked INTEGER DEFAULT 0)"
        )
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, "
            "canonical_name TEXT, source_names TEXT, confidence TEXT, "
            "pending_review INTEGER DEFAULT 0)"
        )
        con.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)"
        )
        decisions = [
            (1, 2, "Yueh", "岳医生"),
            (2, 2, "suspensors", "悬浮灯"),
            (3, 2, "Bene Gesserit", "贝尼·杰瑟里特"),
        ]
        con.executemany(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (?, ?, ?, ?, '2026-11-16 12:00:00', 0)",
            decisions,
        )
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, "
            "confidence, pending_review) VALUES (?, ?, 'high', 0)",
            ("贝尼·杰赛里特", json.dumps(["Bene Gesserit"])),
        )
        pending = [
            ("ch01_chunk000", 1, "Yueh", 1),
            ("ch01_chunk001", 2, "suspensors", 2),
            ("ch01_chunk002", 2, "suspensors", 3),
            ("ch01_chunk999", 1, "Yueh", 4),
            ("ch01_chunk003", 3, "Bene Gesserit", 5),
        ]
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, ?, ?, ?, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')",
            pending,
        )
        con.commit()
    finally:
        con.close()


def _write_final(root: Path, cid: str, text: str) -> None:
    path = root / "output" / "ch01" / f"{cid}_final.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk000", "ch01", "Yueh spoke.", "COMPLETED"),
        ("ch01_chunk001", "ch01", "suspensors hum.", "DIRTY"),
        ("ch01_chunk002", "ch01", "suspensors hum.", "COMPLETED"),
        ("ch01_chunk003", "ch01", "Bene Gesserit school.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    _write_final(tmp_path, "ch01_chunk000", "岳医生开口。")
    _write_final(tmp_path, "ch01_chunk001", "悬浮器低鸣。")
    _write_final(tmp_path, "ch01_chunk002", "悬浮器低鸣。")
    _write_final(tmp_path, "ch01_chunk003", "该学校。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    return tmp_path


def _locked_target_fixture(tmp_path: Path) -> Path:
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk020", "ch01", "The shield held.", "COMPLETED"),
        ("ch01_chunk021", "ch01", "The carryall landed.", "COMPLETED"),
        ("ch01_chunk022", "ch01", "Someone spoke.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("DELETE FROM pending_backtrack")
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (40, 2, 'shield', '屏蔽场', "
            "'2026-11-16 12:00:00', 1)")
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (41, 2, 'carryall', '运载飞行器', "
            "'2026-11-16 12:00:00', 0)")
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, ?, ?, 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')",
            [
                ("ch01_chunk020", 40, "shield"),
                ("ch01_chunk021", 41, "carryall"),
                ("ch01_chunk022", 999, "missing-decision"),
            ],
        )
        con.commit()
    finally:
        con.close()
    _write_final(tmp_path, "ch01_chunk020", "护盾撑住了。")
    _write_final(tmp_path, "ch01_chunk021", "运输机降落了。")
    _write_final(tmp_path, "ch01_chunk022", "有人开口。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    return tmp_path


def _specific_locked_fixture(tmp_path: Path) -> Path:
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk030", "ch01", "Beyond the Shield Wall.", "COMPLETED"),
        ("ch01_chunk031", "ch01", "Beyond the Shield Wall.", "COMPLETED"),
        ("ch01_chunk032", "ch01", "Their shield belts glinted.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("DELETE FROM pending_backtrack")
        con.executemany(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (?, 2, ?, ?, "
            "'2026-11-16 12:00:00', ?)",
            [
                (40, "shield", "屏蔽场", 1),
                (50, "Shield Wall", "屏蔽墙", 1),
                (51, "shield belts", "护盾腰带", 0),
            ],
        )
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, 40, 'shield', 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')",
            [("ch01_chunk030",), ("ch01_chunk031",), ("ch01_chunk032",)],
        )
        con.commit()
    finally:
        con.close()
    _write_final(tmp_path, "ch01_chunk030", "屏蔽墙之外。")
    _write_final(tmp_path, "ch01_chunk031", "盾墙之外。")
    _write_final(tmp_path, "ch01_chunk032", "他们的护盾腰带闪闪发光。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    return tmp_path


def _specific_coverage_fixture(tmp_path: Path) -> Path:
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk010", "ch01", "The Shadout Mapes spoke.", "COMPLETED"),
        ("ch01_chunk011", "ch01",
         "Mapes spoke; later Shadout Mapes bowed.", "COMPLETED"),
        ("ch01_chunk012", "ch01", "The Shadout Mapes bowed.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("DELETE FROM pending_backtrack")
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (10, 2, 'Mapes', '马佩斯', "
            "'2026-11-16 12:00:00', 0)")
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (11, 2, 'Shadout Mapes', "
            "'沙杜特·梅普斯', '2026-11-16 12:00:00', 0)")
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, 10, 'Mapes', 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')",
            [("ch01_chunk010",), ("ch01_chunk011",), ("ch01_chunk012",)],
        )
        con.commit()
    finally:
        con.close()
    _write_final(tmp_path, "ch01_chunk010", "沙杜特·梅普斯开口。")
    _write_final(tmp_path, "ch01_chunk011", "沙杜特·梅普斯开口；随后沙杜特·梅普斯鞠躬。")
    _write_final(tmp_path, "ch01_chunk012", "另一个人鞠了一躬。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    return tmp_path


def test_pending_classes_are_evidence_based(tmp_path: Path):
    root = _fixture(tmp_path)
    context = load_pending_context(root)
    result = classify_pending_snapshot(context)
    by_id = {e["chunk_id"]: e for e in result["entries"]}
    assert by_id["ch01_chunk000"]["status"] == "satisfied"
    assert by_id["ch01_chunk000"]["reason"] == "target_present"
    assert by_id["ch01_chunk001"]["status"] == "covered"
    assert by_id["ch01_chunk002"]["status"] == "outstanding"
    assert by_id["ch01_chunk999"]["status"] == "orphan"
    assert by_id["ch01_chunk003"]["status"] == "blocked"
    assert by_id["ch01_chunk003"]["reason"] == "decision_registry_target_conflict"
    assert result["counts"] == {
        "blocked": 1, "covered": 1, "orphan": 1, "outstanding": 1, "satisfied": 1,
    }
    assert result["ok"] is False


def test_contextual_title_pending_is_nonblocking(tmp_path: Path):
    """Sire -> 陛下 must not be treated as a global one-form invariant."""
    root = _fixture(tmp_path)
    db = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        db.execute(
            "DELETE FROM pending_backtrack WHERE chunk_id='ch01_chunk000'")
        db.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (30, 2, 'Sire', '陛下', "
            "'2026-11-16 12:00:00', 0)")
        db.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES "
            "('ch01_chunk000', 30, 'Sire', 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')")
        db.commit()
    finally:
        db.close()
    wf = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        wf.execute(
            "UPDATE chunk_tasks SET text_content='Sire spoke.' "
            "WHERE chunk_id='ch01_chunk000'")
        wf.commit()
    finally:
        wf.close()

    result = classify_pending_snapshot(load_pending_context(root))
    entry = next(e for e in result["entries"]
                 if e["chunk_id"] == "ch01_chunk000")
    assert entry["status"] == "satisfied"
    assert entry["reason"] == "contextual_decision_not_unified"


def _single_pending_fixture(tmp_path: Path, *, source_key: str, target: str,
                            source_text: str, final_text: str) -> Path:
    root = _fixture(tmp_path)
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        con.execute("DELETE FROM pending_backtrack")
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (60, 2, ?, ?, "
            "'2026-11-16 12:00:00', 0)",
            (source_key, target),
        )
        con.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES "
            "('ch01_chunk002', 60, ?, 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')",
            (source_key,),
        )
        con.commit()
    finally:
        con.close()
    wf = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        wf.execute(
            "UPDATE chunk_tasks SET text_content=? WHERE chunk_id='ch01_chunk002'",
            (source_text,),
        )
        wf.commit()
    finally:
        wf.close()
    _write_final(root, "ch01_chunk002", final_text)
    return root


def test_descriptive_reference_without_gloss_is_contextual(tmp_path: Path):
    root = _single_pending_fixture(
        tmp_path,
        source_key="conditioned",
        target="条件化",
        source_text="It was conditioned to serve.",
        final_text="它经过训练和调适，以履行职责。",
    )

    result = classify_pending_snapshot(load_pending_context(root))
    entry = next(e for e in result["entries"] if e["chunk_id"] == "ch01_chunk002")

    assert entry["status"] == "satisfied"
    assert entry["reason"] == "contextual_reference_not_unified"


def test_explicit_gloss_keeps_descriptive_reference_hard(tmp_path: Path):
    root = _single_pending_fixture(
        tmp_path,
        source_key="conditioned",
        target="条件化",
        source_text="It was conditioned to serve.",
        final_text="它经过训练和调适，以履行职责。",
    )
    _write_final(root, "ch01_chunk000", "它受过条件化（conditioned）训练。")

    result = classify_pending_snapshot(load_pending_context(root))
    entry = next(e for e in result["entries"] if e["chunk_id"] == "ch01_chunk002")

    assert entry["status"] != "satisfied"
    assert entry["reason"] == "completed_final_still_misses_target"


def test_proper_name_phrase_accepts_contextual_common_noun(tmp_path: Path):
    root = _single_pending_fixture(
        tmp_path,
        source_key="Harkonnen agents",
        target="哈克南间谍",
        source_text="Harkonnen agents watched the road.",
        final_text="哈克南家族的特工监视着道路。",
    )
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, "
            "confidence, pending_review) VALUES (?, ?, 'high', 0)",
            ("哈克南家族", json.dumps(["Harkonnen"])),
        )
        con.commit()
    finally:
        con.close()

    result = classify_pending_snapshot(load_pending_context(root))
    entry = next(e for e in result["entries"] if e["chunk_id"] == "ch01_chunk002")

    assert entry["status"] == "satisfied"
    assert entry["reason"] == "contextual_reference_proper_core_present"


def test_proper_name_phrase_still_requires_registered_core(tmp_path: Path):
    root = _single_pending_fixture(
        tmp_path,
        source_key="Harkonnen agents",
        target="哈克南间谍",
        source_text="Harkonnen agents watched the road.",
        final_text="特工监视着道路。",
    )
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, "
            "confidence, pending_review) VALUES (?, ?, 'high', 0)",
            ("哈克南家族", json.dumps(["Harkonnen"])),
        )
        con.commit()
    finally:
        con.close()

    result = classify_pending_snapshot(load_pending_context(root))
    entry = next(e for e in result["entries"] if e["chunk_id"] == "ch01_chunk002")

    assert entry["status"] == "outstanding"
    assert entry["target_present"] is False


def test_decision_target_annotation_is_not_part_of_final_presence(tmp_path: Path):
    """A reader annotation in Decision DB must not mask a valid Chinese term."""
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk020", "ch01", "She used the Voice.", "COMPLETED"),
        ("ch01_chunk021", "ch01", "She used the Voice.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (20, 2, 'the Voice', "
            "'音言（The Voice）', '2026-11-16 12:00:00', 0)"
        )
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, 20, 'the Voice', "
            "1, '2026-11-16 12:00:00', '2026-11-16 12:00:00')",
            [("ch01_chunk020",), ("ch01_chunk021",)],
        )
        con.commit()
    finally:
        con.close()
    _write_final(tmp_path, "ch01_chunk020", "她用音言下了命令。")
    _write_final(tmp_path, "ch01_chunk021", "她下了命令。")

    result = classify_pending_snapshot(load_pending_context(tmp_path))
    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}

    assert by_id["ch01_chunk020"]["status"] == "satisfied"
    assert by_id["ch01_chunk020"]["reason"] == "target_present"
    assert by_id["ch01_chunk021"]["status"] == "outstanding"
    assert by_id["ch01_chunk021"]["target_present"] is False


def test_numbered_footnote_does_not_satisfy_decision_target(tmp_path: Path):
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk022", "ch01", "She used the Voice.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (22, 2, 'the Voice', '音言', "
            "'2026-11-16 12:00:00', 0)")
        con.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES "
            "('ch01_chunk022', 22, 'the Voice', 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')")
        con.commit()
    finally:
        con.close()
    _write_final(
        tmp_path, "ch01_chunk022",
        "她压低了声音。\n\n---\n\n[1] 此处的“音言”是贝尼·杰瑟里特秘术。")

    result = classify_pending_snapshot(load_pending_context(tmp_path))
    entry = next(e for e in result["entries"]
                 if e["chunk_id"] == "ch01_chunk022")

    assert entry["status"] == "outstanding"
    assert entry["target_present"] is False


def test_inline_footnote_marker_does_not_break_decision_target(tmp_path: Path):
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk023", "ch01", "She sipped from her stillsuit’s catchpockets.",
         "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (23, 2, 'stillsuit’s catchpockets', "
            "'蒸馏服的集水袋', '2026-11-16 12:00:00', 0)")
        con.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES "
            "('ch01_chunk023', 23, 'stillsuit’s catchpockets', 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')")
        con.commit()
    finally:
        con.close()
    _write_final(
        tmp_path, "ch01_chunk023",
        "她停下来，从蒸馏服[^1]的集水袋里抿了一口。\n\n"
        "[^1]: 蒸馏服：回收水分的生命维持服装。")

    result = classify_pending_snapshot(load_pending_context(tmp_path))
    entry = next(e for e in result["entries"]
                 if e["chunk_id"] == "ch01_chunk023")

    assert entry["status"] == "satisfied"
    assert entry["reason"] == "target_present"
    assert entry["target_present"] is True


def test_missing_final_on_nonterminal_chunk_is_covered_not_blocked(tmp_path: Path):
    root = _fixture(tmp_path)
    wf = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        wf.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, "
            "state, backtrack_rounds) VALUES "
            "('ch01_chunk004', 'ch01', 'Yueh spoke.', 'EXTRACTING_TERMS', 0)"
        )
        wf.commit()
    finally:
        wf.close()
    db = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        db.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES "
            "('ch01_chunk004', 1, 'Yueh', 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')")
        db.commit()
    finally:
        db.close()

    result = classify_pending_snapshot(load_pending_context(root))
    entry = next(e for e in result["entries"] if e["chunk_id"] == "ch01_chunk004")
    assert entry["status"] == "covered"
    assert entry["reason"] == "nonterminal_final_pending"


def test_specific_decision_cover_is_mechanical_not_substring(tmp_path: Path):
    root = _specific_coverage_fixture(tmp_path)

    result = classify_pending_snapshot(load_pending_context(root))
    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}

    covered = by_id["ch01_chunk010"]
    assert covered["status"] == "satisfied"
    assert covered["reason"] == "specific_decision_covered"
    assert covered["covered_source_occurrences"] == 1
    assert covered["covered_by_decisions"] == [{
        "decision_id": 11,
        "source_key": "Shadout Mapes",
        "translation": "沙杜特·梅普斯",
    }]

    # The second occurrence is bare ``Mapes`` and has no specific decision.
    assert by_id["ch01_chunk011"]["status"] == "outstanding"
    assert by_id["ch01_chunk011"]["reason"] == "completed_final_still_misses_target"

    # Complete source coverage is insufficient if the specific target is absent.
    assert by_id["ch01_chunk012"]["status"] == "outstanding"


def test_locked_generic_does_not_override_longer_specific_decision(
        tmp_path: Path):
    root = _specific_locked_fixture(tmp_path)

    result = classify_pending_snapshot(load_pending_context(root))
    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}

    assert by_id["ch01_chunk030"]["status"] == "satisfied"
    assert by_id["ch01_chunk030"]["reason"] == "specific_decision_covered"
    assert by_id["ch01_chunk031"]["status"] == "blocked"
    assert by_id["ch01_chunk031"]["reason"] == "specific_decision_target_missing"
    assert by_id["ch01_chunk032"]["status"] == "satisfied"
    assert by_id["ch01_chunk032"]["reason"] == "specific_decision_covered"


def test_safe_only_consumes_specific_decision_cover_without_dirty(tmp_path: Path):
    root = _specific_coverage_fixture(tmp_path)

    class MustNotRunScheduler:
        def get_task(self, chunk_id):
            raise AssertionError("safe-only must not inspect outstanding state")

        def trigger_backtrack(self, chunk_ids):
            raise AssertionError("safe-only must not trigger DIRTY")

    result = apply_pending_backtrack_reconcile(
        root, apply=True, safe_only=True, scheduler=MustNotRunScheduler())

    assert result["consumed"] == 1
    assert result["triggered_dirty"] == 0
    assert result["after_counts"] == {"outstanding": 2}
    action = next(a for a in result["actions"] if a["chunk_id"] == "ch01_chunk010")
    assert action["status"] == "consumed"
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        rows = con.execute(
            "SELECT chunk_id FROM pending_backtrack ORDER BY chunk_id").fetchall()
    finally:
        con.close()
    assert rows == [("ch01_chunk011",), ("ch01_chunk012",)]


def test_decision_target_noncanonical_variant_is_blocked(tmp_path: Path):
    root = _fixture(tmp_path)
    wf = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        wf.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, "
            "state, backtrack_rounds) VALUES "
            "('ch01_chunk004', 'ch01', 'Leto spoke.', 'COMPLETED', 0)"
        )
        wf.commit()
    finally:
        wf.close()
    db = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        db.execute("ALTER TABLE entity_registry ADD COLUMN variant_names TEXT")
        db.execute("ALTER TABLE entity_registry ADD COLUMN quarantined_names TEXT")
        db.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (4, 2, 'Leto', '莱托', "
            "'2026-11-16 12:00:00', 0)"
        )
        db.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, "
            "variant_names, quarantined_names, confidence, pending_review) "
            "VALUES ('雷托·厄崔迪公爵', ?, ?, ?, 'high', 0)",
            (
                json.dumps(["Duke Leto of Arrakis"], ensure_ascii=False),
                json.dumps(["莱托·厄崔迪公爵"], ensure_ascii=False),
                json.dumps(["Leto"], ensure_ascii=False),
            ),
        )
        db.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES "
            "('ch01_chunk004', 4, 'Leto', 1, "
            "'2026-11-16 12:00:00', '2026-11-16 12:00:00')"
        )
        db.commit()
    finally:
        db.close()
    _write_final(root, "ch01_chunk004", "雷托开口。")

    result = classify_pending_snapshot(load_pending_context(root))

    entry = next(e for e in result["entries"]
                 if e["chunk_id"] == "ch01_chunk004")
    assert entry["status"] == "blocked"
    assert entry["reason"] == "decision_target_noncanonical_variant"


def test_pending_audit_writes_reports_without_mutating_db(tmp_path: Path):
    root = _fixture(tmp_path)
    db = root / "db" / "decision_db.sqlite"
    before = db.read_bytes()
    con = sqlite3.connect(str(db))
    before_count = con.execute("SELECT COUNT(*) FROM pending_backtrack").fetchone()[0]
    con.close()
    result = audit_pending_backtracks(root)
    con = sqlite3.connect(str(db))
    after_count = con.execute("SELECT COUNT(*) FROM pending_backtrack").fetchone()[0]
    con.close()
    assert db.read_bytes() == before
    assert before_count == after_count == 5
    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()
    assert result["mode"] == "read_only"


def test_pending_apply_consumes_only_evidence_safe_rows(tmp_path: Path):
    root = _fixture(tmp_path)

    result = apply_pending_backtrack_reconcile(root, apply=True)

    assert result["mode"] == "apply"
    assert result["write_attempted"] is True
    assert result["consumed"] == 3
    assert result["triggered_dirty"] == 1
    assert result["after_counts"] == {
        "blocked": 1, "orphan": 1,
    }
    assert result["ok"] is False
    by_id = {a["chunk_id"]: a for a in result["actions"]}
    assert by_id["ch01_chunk000"]["status"] == "consumed"
    assert by_id["ch01_chunk001"]["status"] == "consumed"
    assert by_id["ch01_chunk002"]["status"] == "consumed"
    assert by_id["ch01_chunk002"]["triggered_dirty"] is True
    assert by_id["ch01_chunk003"]["status"] == "blocked_evidence_retained"
    assert by_id["ch01_chunk999"]["status"] == "orphan_retained"

    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        rows = con.execute(
            "SELECT chunk_id FROM pending_backtrack ORDER BY chunk_id").fetchall()
    finally:
        con.close()
    assert rows == [("ch01_chunk003",), ("ch01_chunk999",)]
    task = TaskScheduler(str(root / "db" / "workflow.db")).get_task("ch01_chunk002")
    assert task["state"] == "DIRTY"
    assert task["backtrack_rounds"] == 1


def test_pending_safe_only_never_triggers_dirty(tmp_path: Path):
    root = _fixture(tmp_path)

    class MustNotRunScheduler:
        def get_task(self, chunk_id):
            raise AssertionError("safe-only must not inspect outstanding state")

        def trigger_backtrack(self, chunk_ids):
            raise AssertionError("safe-only must not trigger DIRTY")

    result = apply_pending_backtrack_reconcile(
        root, apply=True, safe_only=True, scheduler=MustNotRunScheduler())

    assert result["safe_only"] is True
    assert result["consumed"] == 2
    assert result["triggered_dirty"] == 0
    assert result["after_counts"] == {
        "blocked": 1, "orphan": 1, "outstanding": 1,
    }
    by_id = {a["chunk_id"]: a for a in result["actions"]}
    assert by_id["ch01_chunk000"]["status"] == "consumed"
    assert by_id["ch01_chunk001"]["status"] == "consumed"
    assert by_id["ch01_chunk002"]["status"] == "retained_safe_only"
    assert by_id["ch01_chunk002"]["reason"] == "outstanding_requires_triage"
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        rows = con.execute(
            "SELECT chunk_id FROM pending_backtrack ORDER BY chunk_id").fetchall()
    finally:
        con.close()
    assert rows == [("ch01_chunk002",), ("ch01_chunk003",), ("ch01_chunk999",)]


def test_locked_target_only_dirty_never_mass_requeues_outstanding(tmp_path: Path):
    root = _locked_target_fixture(tmp_path)

    result = apply_pending_backtrack_reconcile(
        root, apply=True, locked_target_only=True)

    assert result["locked_target_only"] is True
    assert result["consumed"] == 1
    assert result["triggered_dirty"] == 1
    assert result["after_counts"] == {"blocked": 1, "outstanding": 1}
    by_id = {a["chunk_id"]: a for a in result["actions"]}
    assert by_id["ch01_chunk020"]["status"] == "consumed"
    assert by_id["ch01_chunk020"]["triggered_dirty"] is True
    assert by_id["ch01_chunk021"]["status"] == "deferred_not_locked_target"
    assert by_id["ch01_chunk022"]["status"] == "deferred_not_locked_target"
    sched = TaskScheduler(str(root / "db" / "workflow.db"))
    assert sched.get_task("ch01_chunk020")["state"] == "DIRTY"
    assert sched.get_task("ch01_chunk021")["state"] == "COMPLETED"
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        rows = con.execute(
            "SELECT chunk_id FROM pending_backtrack ORDER BY chunk_id").fetchall()
    finally:
        con.close()
    assert rows == [("ch01_chunk021",), ("ch01_chunk022",)]


def test_locked_target_only_safe_mode_keeps_lock_evidence(tmp_path: Path):
    root = _locked_target_fixture(tmp_path)

    class MustNotRunScheduler:
        def get_task(self, chunk_id):
            raise AssertionError("safe-only must not inspect task state")

        def trigger_backtrack(self, chunk_ids):
            raise AssertionError("safe-only must not trigger DIRTY")

    result = apply_pending_backtrack_reconcile(
        root, apply=True, safe_only=True, locked_target_only=True,
        scheduler=MustNotRunScheduler())

    assert result["triggered_dirty"] == 0
    assert result["after_counts"] == {"blocked": 2, "outstanding": 1}
    lock_action = next(a for a in result["actions"]
                       if a["chunk_id"] == "ch01_chunk020")
    assert lock_action["status"] == "retained_safe_only"
    assert lock_action["reason"] == "locked_target_repair_requires_apply"
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 3
    finally:
        con.close()


def test_normal_apply_keeps_non_lock_blocked_evidence(tmp_path: Path):
    root = _locked_target_fixture(tmp_path)

    result = apply_pending_backtrack_reconcile(root, apply=True)

    by_id = {a["chunk_id"]: a for a in result["actions"]}
    assert by_id["ch01_chunk022"]["status"] == "blocked_evidence_retained"
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack "
            "WHERE chunk_id='ch01_chunk022'").fetchone()[0] == 1
    finally:
        con.close()


def test_pending_apply_is_idempotent(tmp_path: Path):
    root = _fixture(tmp_path)
    first = apply_pending_backtrack_reconcile(root, apply=True)
    second = apply_pending_backtrack_reconcile(root, apply=True)

    assert first["consumed"] == 3
    assert second["consumed"] == 0
    assert second["triggered_dirty"] == 0
    assert second["after_counts"] == {"blocked": 1, "orphan": 1}
    task = TaskScheduler(str(root / "db" / "workflow.db")).get_task("ch01_chunk002")
    assert task["backtrack_rounds"] == 1


def test_pending_apply_keeps_row_when_transition_fails(tmp_path: Path):
    root = _fixture(tmp_path)

    class BrokenScheduler:
        def get_task(self, chunk_id):
            return {"chunk_id": chunk_id, "state": "COMPLETED"}

        def trigger_backtrack(self, chunk_ids):
            raise RuntimeError("simulated transition failure")

    result = apply_pending_backtrack_reconcile(
        root, apply=True, scheduler=BrokenScheduler())

    action = next(a for a in result["actions"]
                  if a["chunk_id"] == "ch01_chunk002")
    assert action["status"] == "transition_failed"
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack WHERE chunk_id='ch01_chunk002'"
        ).fetchone()[0] == 1
    finally:
        con.close()


def test_empty_legacy_decision_db_is_a_clean_audit(tmp_path: Path):
    """No pending table means an empty outbox, not a schema failure."""
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk000", "ch01", "Yueh spoke.", "COMPLETED"),
    ])
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "CREATE TABLE decision_db "
            "(source_key TEXT, translation TEXT, reason TEXT)"
        )
        con.commit()
    finally:
        con.close()

    context = load_pending_context(tmp_path)
    result = classify_pending_snapshot(context)

    assert context["available"] is True
    assert context["decisions_complete"] is False
    assert result["total"] == 0
    assert result["ok"] is True
    assert result["blocking"] == []


def test_term_form_contract_satisfies_pending_without_mutating_db(tmp_path: Path):
    root = _fixture(tmp_path)
    workflow_before = (root / "db" / "workflow.db").read_bytes()
    decisions_before = (root / "db" / "decision_db.sqlite").read_bytes()
    contract_path = root / "input" / "term_form_contracts.json"
    contract_path.write_text(json.dumps({
        "schema_version": 1,
        "contracts": [{
            "contract_id": "term-2-suspensors",
            "decision_id": 2,
            "source_key": "suspensors",
            "decision_translation": "悬浮灯",
            "forms": [{"form": "悬浮灯"}, {"form": "悬浮器"}],
        }],
    }, ensure_ascii=False), encoding="utf-8")

    context = load_pending_context(root)
    result = classify_pending_snapshot(context)

    assert context["term_contract_error"] == ""
    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}
    assert by_id["ch01_chunk002"]["status"] == "satisfied"
    assert by_id["ch01_chunk002"]["reason"] == "term_form_contract_present"
    assert by_id["ch01_chunk002"]["matched_term_form"] == "悬浮器"
    assert (root / "db" / "workflow.db").read_bytes() == workflow_before
    assert (root / "db" / "decision_db.sqlite").read_bytes() == decisions_before


def test_partial_term_contract_cannot_close_locked_decision(tmp_path: Path):
    root = _locked_target_fixture(tmp_path)
    contract_path = root / "input" / "term_form_contracts.json"
    contract_path.write_text(json.dumps({
        "schema_version": 1,
        "contracts": [{
            "contract_id": "term-40-shield-partial",
            "decision_id": 40,
            "source_key": "shield",
            "decision_translation": "屏蔽场",
            "status": "partial",
            "forms": [{"form": "护盾"}],
        }],
    }, ensure_ascii=False), encoding="utf-8")

    context = load_pending_context(root)
    result = classify_pending_snapshot(context)

    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}
    assert by_id["ch01_chunk020"]["status"] == "blocked"
    assert by_id["ch01_chunk020"]["reason"] == "locked_decision_target_missing"


def test_complete_term_contract_can_close_locked_decision(tmp_path: Path):
    root = _locked_target_fixture(tmp_path)
    contract_path = root / "input" / "term_form_contracts.json"
    contract_path.write_text(json.dumps({
        "schema_version": 1,
        "contracts": [{
            "contract_id": "term-40-shield-complete",
            "decision_id": 40,
            "source_key": "shield",
            "decision_translation": "屏蔽场",
            "status": "complete",
            "forms": [{"form": "护盾"}],
        }],
    }, ensure_ascii=False), encoding="utf-8")

    context = load_pending_context(root)
    result = classify_pending_snapshot(context)

    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}
    assert by_id["ch01_chunk020"]["status"] == "satisfied"
    assert by_id["ch01_chunk020"]["reason"] == "term_form_contract_present"


def test_malformed_term_contract_does_not_relax_pending(tmp_path: Path):
    root = _fixture(tmp_path)
    (root / "input" / "term_form_contracts.json").write_text(
        '{"schema_version": 999, "contracts": []}', encoding="utf-8")

    context = load_pending_context(root)
    result = classify_pending_snapshot(context)

    assert context["term_contract_error"] == "term_contract_schema_incompatible"
    by_id = {entry["chunk_id"]: entry for entry in result["entries"]}
    assert by_id["ch01_chunk002"]["status"] == "outstanding"


def test_pending_rows_with_incompatible_decision_schema_fail_closed(tmp_path: Path):
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk000", "ch01", "Yueh spoke.", "COMPLETED"),
    ])
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "CREATE TABLE decision_db "
            "(source_key TEXT, translation TEXT, reason TEXT)"
        )
        con.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)"
        )
        con.execute(
            "INSERT INTO pending_backtrack VALUES "
            "('ch01_chunk000', 1, 'Yueh', 1, NULL, NULL)"
        )
        con.commit()
    finally:
        con.close()

    context = load_pending_context(tmp_path)
    result = classify_pending_snapshot(context)

    assert context["available"] is False
    assert context["pending_rows"] == 1
    assert context["decisions_complete"] is False
    assert result["total"] == 1
    assert result["ok"] is False


def test_soft_literal_reference_pending_row_is_advisory(tmp_path: Path):
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk040", "ch01", "Assassins hid in the shadows.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("DELETE FROM pending_backtrack")
        con.execute(
            "UPDATE decision_db SET source_key='shadows', translation='暗影', "
            "reason='【来源】文学隐喻 | 【策略】LITERAL_TRANSLATION | 【依据】阴影意象' "
            "WHERE id=2"
        )
        con.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES ("
            "'ch01_chunk040', 2, 'shadows', 1, '2026-11-16 12:00:00', "
            "'2026-11-16 12:00:00')"
        )
        con.commit()
    finally:
        con.close()
    _write_final(tmp_path, "ch01_chunk040", "刺客藏在阴影里。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")

    result = classify_pending_snapshot(load_pending_context(tmp_path))
    entry = {e["chunk_id"]: e for e in result["entries"]}["ch01_chunk040"]

    assert entry["status"] == "satisfied"
    assert entry["reason"] == "soft_literal_reference_not_unified"


def test_soft_cultural_reference_pending_row_is_advisory(tmp_path: Path):
    _make_workflow(tmp_path / "db" / "workflow.db", [
        ("ch01_chunk040", "ch01", "They are useless clods.", "COMPLETED"),
    ])
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("DELETE FROM pending_backtrack")
        con.execute(
            "UPDATE decision_db SET source_key='clods', translation='泥腿子', "
            "reason='【来源】阶级蔑称 | 【策略】CULTURAL_EQUIVALENT | "
            "【依据】泥土意象的贬称' WHERE id=2"
        )
        con.execute(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES ("
            "'ch01_chunk040', 2, 'clods', 1, '2026-11-16 12:00:00', "
            "'2026-11-16 12:00:00')"
        )
        con.commit()
    finally:
        con.close()
    _write_final(tmp_path, "ch01_chunk040", "他们全是没用的土疙瘩。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")

    result = classify_pending_snapshot(load_pending_context(tmp_path))
    entry = {e["chunk_id"]: e for e in result["entries"]}["ch01_chunk040"]

    assert entry["status"] == "satisfied"
    assert entry["reason"] == "soft_cultural_reference_not_unified"


def test_final_gate_item_13_rejects_outstanding_pending(tmp_path: Path):
    from src.pipeline import run_final_gate

    sched = TaskScheduler(db_path=str(tmp_path / "db" / "workflow.db"))
    sched.conn.execute(
        "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
        "VALUES ('ch01_chunk000', 'ch01', 'Yueh spoke.', 'COMPLETED')"
    )
    sched.conn.commit()
    out = tmp_path / "output" / "ch01"
    out.mkdir(parents=True)
    final_text = "有人开口。\n"
    (out / "ch01_chunk000_final.json").write_text(
        json.dumps({"text": final_text}, ensure_ascii=False), encoding="utf-8")
    (out / "ch01_chunk000_critic_report.json").write_text(
        json.dumps({
            "_backend_path": "jev",
            "_audit_target": {"literary_sha256": sha256_text(final_text)},
            "scores": {"readability": 8.0},
        }, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "output" / "translated_full.md").write_text(final_text, encoding="utf-8")
    (tmp_path / "output" / "QA_REPORT.md").write_text(
        "# 译文质量验收报告 (QA Report)\n", encoding="utf-8")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n\nYueh spoke.\n", encoding="utf-8")
    (tmp_path / "input" / "_conversion_audit.md").write_text(
        "# 转换完整性审计报告 (T1)\n\n- 全书覆盖率: 100.000%\n- 结论: ✅ 通过\n",
        encoding="utf-8")
    _make_decision_db(tmp_path / "db" / "decision_db.sqlite")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    con.execute("DELETE FROM pending_backtrack")
    con.execute(
        "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
        "queue_rounds, created_at, updated_at) VALUES ("
        "'ch01_chunk000', 1, 'Yueh', 1, '2026-11-16 12:00:00', "
        "'2026-11-16 12:00:00')")
    con.commit()
    con.close()

    result = run_final_gate(project_root=str(tmp_path))

    assert result["stats"]["pending_backtrack"]["counts"]["outstanding"] == 1
    assert any(str(f).startswith("⑬ pending_backtrack 未决") for f in result["failures"])

