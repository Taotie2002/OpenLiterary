"""Round 158 Phase 1.5: consolidate ordering and convergence contract."""
from __future__ import annotations

import json

from src.core.scheduler import TaskScheduler
from src.pipeline import consolidate_dirty_chapters


def _mk_project(tmp_path, rows):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    sched = TaskScheduler(db_path=str(db_dir / "workflow.db"))
    for cid, chapter, state in rows:
        sched.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?, ?, ?, ?)",
            (cid, chapter, "source " + cid, state),
        )
    sched.conn.commit()
    (db_dir / "decision_db.sqlite").touch()
    return sched


def _set_state(sched, chunk_id, state):
    sched.conn.execute(
        "UPDATE chunk_tasks SET state=? WHERE chunk_id=?", (state, chunk_id))
    sched.conn.commit()


def _factory(handler, events):
    def factory(chapter_id):
        class _Pipeline:
            def run(self):
                events.append(f"pipeline:{chapter_id}")
                return handler(chapter_id)

        return _Pipeline()

    return factory


def test_pending_then_name_only_then_dirty_rerun(tmp_path):
    sched = _mk_project(tmp_path, [
        ("ch01_chunk000", "ch01", "DIRTY"),
        ("ch02_chunk000", "ch02", "DIRTY"),
    ])
    events = []

    def pending():
        stage = "before_name_only" if not any(
            str(e).startswith("name_only") for e in events) else "after_rerun"
        events.append(f"pending:{stage}")
        return {"ok": True, "after_counts": {}}

    def completed():
        events.append("completed_name_repair")
        return {"ok": True, "counts": {"repaired": 0}}

    def confirmed():
        events.append("confirmed_term_repair")
        return {"ok": True, "counts": {"repaired": 0}}

    def article_alias():
        stage = ("article_alias_repair_before"
                 if not any(str(e).startswith("pipeline:") for e in events)
                 else "article_alias_repair_after")
        events.append(stage)
        return {"ok": True, "counts": {"candidates": 0}}

    def name_authority():
        events.append("name_authority_align")
        return {"ok": True, "counts": {"quarantine_aliases": 0}}

    def term_contracts():
        events.append("term_form_contracts")
        return {"ok": True, "counts": {"contracts": 0}}

    def name_only():
        events.append("name_only")
        _set_state(sched, "ch01_chunk000", "COMPLETED")
        return {"ok": True, "counts": {"completed": 1}}

    def handler(chapter_id):
        _set_state(sched, f"{chapter_id}_chunk000", "COMPLETED")
        return True

    result = consolidate_dirty_chapters(
        project_root=tmp_path,
        pipeline_factory=_factory(handler, events),
        pending_applier=pending,
        name_only_applier=name_only,
        confirmed_term_applier=confirmed,
        completed_name_applier=completed,
        article_alias_applier=article_alias,
        name_authority_applier=name_authority,
        term_contract_applier=term_contracts,
    )

    assert events == [
        "article_alias_repair_before", "name_authority_align",
        "confirmed_term_repair", "completed_name_repair",
        "term_form_contracts",
        "pending:before_name_only", "name_only",
        "pipeline:ch02", "article_alias_repair_after", "pending:after_rerun",
    ]
    assert result["converged"] is True
    assert result["rounds"] == 1
    assert result["rerun"] == ["ch02_chunk000"]
    assert result["remaining"] == {}
    assert [p["stage"] for p in result["pending_reconcile"]] == [
        "before_name_only", "after_rerun"]
    assert result["name_only_fastpath"]["counts"] == {"completed": 1}
    assert result["confirmed_term_drift_repair"]["counts"] == {"repaired": 0}
    assert result["completed_name_repair"]["counts"] == {"repaired": 0}


def test_zero_dirty_but_unresolved_pending_is_not_converged(tmp_path):
    sched = _mk_project(tmp_path, [("ch01_chunk000", "ch01", "DIRTY")])
    calls = []

    def pending():
        calls.append("pending")
        return {
            "ok": len(calls) == 1,
            "after_counts": {"outstanding": 1},
        }

    def name_only():
        _set_state(sched, "ch01_chunk000", "COMPLETED")
        return {"ok": True, "counts": {"completed": 1}}

    result = consolidate_dirty_chapters(
        project_root=tmp_path,
        pipeline_factory=_factory(lambda chapter_id: True, []),
        pending_applier=pending,
        name_only_applier=name_only,
        confirmed_term_applier=lambda: {"ok": True, "counts": {}},
        completed_name_applier=lambda: {"ok": True, "counts": {}},
        term_contract_applier=lambda: {"ok": True, "counts": {}},
    )

    assert result["rounds"] == 0
    assert result["rerun"] == []
    assert result["remaining"] == {}
    assert result["converged"] is False


def test_interrupted_active_stage_is_resumable_by_consolidate(tmp_path):
    sched = _mk_project(tmp_path, [
        ("ch01_chunk000", "ch01", "TRANSLATING_RAW"),
    ])
    events = []

    def handler(chapter_id):
        events.append(f"resume:{chapter_id}")
        _set_state(sched, "ch01_chunk000", "COMPLETED")
        return True

    result = consolidate_dirty_chapters(
        project_root=tmp_path,
        pipeline_factory=_factory(handler, events),
        pending_applier=lambda: {"ok": True, "after_counts": {}},
        name_only_applier=lambda: {"ok": True, "counts": {"completed": 0}},
        confirmed_term_applier=lambda: {"ok": True, "counts": {}},
        completed_name_applier=lambda: {"ok": True, "counts": {}},
        term_contract_applier=lambda: {"ok": True, "counts": {}},
    )

    assert events == ["pipeline:ch01", "resume:ch01"]
    assert result["converged"] is True
    assert result["rerun"] == ["ch01_chunk000"]


def test_article_alias_blocker_queues_only_source_positive_chunk(tmp_path):
    sched = _mk_project(tmp_path, [
        ("ch01_chunk000", "ch01", "COMPLETED"),
        ("ch01_chunk001", "ch01", "COMPLETED"),
        ("ch01_chunk002", "ch01", "COMPLETED"),
    ])
    sched.conn.execute(
        "UPDATE chunk_tasks SET text_content=? WHERE chunk_id=?",
        ("the Lady Jessica watched Paul.", "ch01_chunk000"),
    )
    sched.conn.execute(
        "UPDATE chunk_tasks SET text_content=? WHERE chunk_id=?",
        ("An unrelated completed passage.", "ch01_chunk001"),
    )
    sched.conn.execute(
        "UPDATE chunk_tasks SET text_content=? WHERE chunk_id=?",
        ("Paul watched while the Lady Jessica waited.", "ch01_chunk002"),
    )
    sched.conn.commit()
    out = tmp_path / "output" / "ch01"
    out.mkdir(parents=True)
    for chunk_id, text in (
        ("ch01_chunk000", "杰西卡女士看着保罗。"),
        ("ch01_chunk001", "一段无关的完成文本。"),
        ("ch01_chunk002", "杰西卡女士等待着。"),
    ):
        (out / f"{chunk_id}_final.json").write_text(
            json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    events = []
    article_calls = []

    def article_alias():
        article_calls.append("audit")
        if len(article_calls) == 1:
            return {
                "ok": False,
                "blocked_candidates": [
                    {
                        "source_key": "the Lady Jessica",
                        "alias_translation": "杰西卡女士",
                        "reason": "alias_target_still_present:ch01_chunk000",
                        "blocked_chunks": [
                            "ch01_chunk000", "ch01_chunk001", "ch01_chunk002"],
                    },
                ],
            }
        return {"ok": True, "blocked_candidates": []}

    def handler(chapter_id):
        _set_state(sched, "ch01_chunk000", "COMPLETED")
        _set_state(sched, "ch01_chunk002", "COMPLETED")
        return True

    result = consolidate_dirty_chapters(
        project_root=tmp_path,
        pipeline_factory=_factory(handler, events),
        pending_applier=lambda: {"ok": True, "after_counts": {}},
        name_only_applier=lambda: {"ok": True, "counts": {}},
        confirmed_term_applier=lambda: {"ok": True, "counts": {}},
        completed_name_applier=lambda: {"ok": True, "counts": {}},
        article_alias_applier=article_alias,
        term_contract_applier=lambda: {"ok": True, "counts": {}},
    )

    assert result["article_alias_requeue"]["chunk_ids"] == [
        "ch01_chunk000", "ch01_chunk002"]
    assert result["article_alias_requeue"]["ok"] is True
    assert result["rerun"] == ["ch01_chunk000", "ch01_chunk002"]
    assert events == ["pipeline:ch01"]
    assert article_calls == ["audit", "audit"]
    assert sched.get_task("ch01_chunk001")["state"] == "COMPLETED"
    assert result["converged"] is True
