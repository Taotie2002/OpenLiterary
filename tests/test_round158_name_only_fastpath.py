"""Round 158 Phase 1.5: strict read-only name-only fast-path classification."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import src.name_only_fastpath as nof
from src.judgment_layer import sha256_text


def test_source_key_occurrence_count_normalizes_line_wraps():
    text = "the Judge of\nthe Change is allowing it."
    assert nof._source_key_occurrence_count(text, "Judge of the Change") == 1


def _make_dbs(root: Path, *, canonical: str = "盖乌斯·海伦·莫希阿姆",
              decision_target: str = "盖乌斯·海伦·莫希阿姆",
              source_key: str = "Gaius Helen Mohiam",
              source_text: str = "Gaius Helen Mohiam watched.",
              pending=None) -> None:
    db = root / "db"
    db.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db / "workflow.db"))
    try:
        con.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, chapter_id TEXT, "
            "text_content TEXT, state TEXT, last_error TEXT, backtrack_rounds INTEGER, "
            "rescue_stage TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state, "
            "backtrack_rounds) VALUES ('ch01_chunk000', 'ch01', ?, 'DIRTY', 1)",
            (source_text,),
        )
        con.commit()
    finally:
        con.close()

    con = sqlite3.connect(str(db / "decision_db.sqlite"))
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
        con.execute(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (1, 2, ?, ?, '2026-11-16 12:00:00', 0)",
            (source_key, decision_target),
        )
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, confidence, "
            "pending_review) VALUES (?, ?, 'high', 0)",
            (canonical, json.dumps([source_key])),
        )
        if pending:
            con.executemany(
                "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
                "queue_rounds, created_at, updated_at) VALUES (?, ?, ?, ?, "
                "'2026-11-16 12:00:00', '2026-11-16 12:00:00')",
                pending,
            )
        con.commit()
    finally:
        con.close()


def _write_artifacts(root: Path, text: str, *, score: float = 8.5,
                     backend: str = "jev") -> None:
    out = root / "output" / "ch01"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ch01_chunk000_final.json").write_text(json.dumps({
        "text": text,
        "metadata": {"fallback": False},
    }, ensure_ascii=False), encoding="utf-8")
    report = {
        "_backend_path": backend,
        "_audit_target": {"literary_sha256": sha256_text(text)},
        "scores": {
            "voice_consistency": score,
            "style_compliance": score,
            "literal_residue": score,
            "semantic_preservation": score,
            "readability": score,
            "fluency": score,
            "entity_consistency": score,
        },
    }
    (out / "ch01_chunk000_critic_report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8")


def _add_completed_chunk(
    root: Path,
    chunk_id: str,
    *,
    source_text: str,
    final_text: str,
) -> None:
    con = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        con.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state, "
            "backtrack_rounds) VALUES (?, ?, ?, 'COMPLETED', 0)",
            (chunk_id, chunk_id[:4], source_text),
        )
        con.commit()
    finally:
        con.close()
    out = root / "output" / chunk_id[:4]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{chunk_id}_final.json").write_text(
        json.dumps({"text": final_text}, ensure_ascii=False), encoding="utf-8")


def _maps(target: str):
    return {
        "forms": [], "tokens": {}, "token_candidates": {},
        "authorities": [target], "guards": [], "semantic_pending": [],
    }


def test_pure_name_candidate_is_read_only(tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    text = "盖乌斯海伦莫希阿姆在观察。"
    _write_artifacts(tmp_path, text)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    db_path = tmp_path / "db" / "workflow.db"
    decision_path = tmp_path / "db" / "decision_db.sqlite"
    before_final = final_path.read_bytes()
    before_workflow = db_path.read_bytes()
    before_decision = decision_path.read_bytes()

    result = nof.audit_name_only_fastpath(tmp_path)

    assert result["ok"] is True
    assert result["mode"] == "read_only"
    assert result["write_attempted"] is False
    assert result["counts"] == {
        "dirty_scanned": 1, "candidates": 1, "kept_dirty": 0, "failed": 0,
    }
    candidate = result["candidates"][0]
    assert candidate["chunk_id"] == "ch01_chunk000"
    assert candidate["mapping_counts"]["sep_restore"] == 1
    assert candidate["reaudit_required"] is True
    assert final_path.read_bytes() == before_final
    assert db_path.read_bytes() == before_workflow
    assert decision_path.read_bytes() == before_decision


def test_residual_cjk_name_drift_keeps_dirty(tmp_path: Path, monkeypatch):
    _make_dbs(
        tmp_path,
        canonical="奎萨茨·哈德拉赫",
        decision_target="奎萨茨·哈德拉赫",
        source_key="Kwisatz Haderach",
        source_text="Kwisatz Haderach watched.",
    )
    _write_artifacts(tmp_path, "奎萨茨哈德拉赫在观察；贝尼杰赛里特修会也在场。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    maps = _maps("奎萨茨·哈德拉赫")
    maps["authorities"].append("贝尼·杰瑟里特")
    monkeypatch.setattr(nof, "_build_maps", lambda root, db, chunks: maps)

    result = nof.audit_name_only_fastpath(tmp_path)

    assert result["counts"]["candidates"] == 0
    assert result["counts"]["kept_dirty"] == 1
    record = result["kept_dirty"][0]
    assert "residual_authority_form" in record["reasons"]
    assert record["residual_authority_forms"] == [{
        "canonical": "贝尼·杰瑟里特",
        "found": "贝尼杰赛里特",
        "kind": "one_char_cjk_drift",
    }]


def test_low_score_keeps_dirty(tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    _write_artifacts(tmp_path, "盖乌斯海伦莫希阿姆在观察。", score=5.0)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    result = nof.audit_name_only_fastpath(tmp_path)
    assert result["counts"]["candidates"] == 0
    assert result["counts"]["kept_dirty"] == 1
    assert "critic_scores_below_floor" in result["kept_dirty"][0]["reasons"]


def test_decision_registry_conflict_blocks(tmp_path: Path, monkeypatch):
    _make_dbs(
        tmp_path,
        canonical="贝尼·杰赛里特",
        decision_target="贝尼·杰瑟里特",
        source_key="Bene Gesserit",
        source_text="Bene Gesserit school.",
    )
    _write_artifacts(tmp_path, "贝尼杰赛里特学校。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("贝尼·杰赛里特"))
    result = nof.audit_name_only_fastpath(tmp_path)
    assert result["counts"]["candidates"] == 0
    reasons = result["kept_dirty"][0]["reasons"]
    assert "authority_conflict" in reasons
    assert result["kept_dirty"][0]["authority_conflicts"][0]["reason"] == (
        "decision_registry_target_conflict")


def test_unresolved_pending_blocks_name_only_completion(tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path, source_text="Yueh and Gaius Helen Mohiam watched.", pending=[
        ("ch01_chunk000", 1, "Gaius Helen Mohiam", 1),
    ])
    # Replace the decision with a pending name that the proposed text does not
    # satisfy, while still changing the unrelated Mohiam separator below.
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    con.execute("UPDATE decision_db SET source_key='Yueh', translation='岳医生' WHERE id=1")
    con.execute("UPDATE pending_backtrack SET source_key='Yueh' WHERE chunk_id='ch01_chunk000'")
    con.execute(
        "INSERT INTO entity_registry (canonical_name, source_names, confidence, "
        "pending_review) VALUES ('岳医生', ?, 'high', 0)",
        (json.dumps(["Yueh"]),),
    )
    con.commit()
    con.close()
    _write_artifacts(tmp_path, "盖乌斯海伦莫希阿姆在观察。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    result = nof.audit_name_only_fastpath(tmp_path)
    assert result["counts"]["candidates"] == 0
    reasons = result["kept_dirty"][0]["reasons"]
    assert "pending_not_satisfied_by_proposed_final" in reasons


def test_contradicted_generic_authority_blocks_dirty_name_candidate(
        tmp_path: Path, monkeypatch):
    _make_dbs(
        tmp_path,
        canonical="屏蔽力场",
        decision_target="屏蔽力场",
        source_key="shield",
        source_text="The shield held.",
    )
    _write_artifacts(tmp_path, "屏蔽立场在挡光。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    for chunk_id, final_text in (
        ("ch01_chunk001", "护盾撑住了。"),
        ("ch01_chunk002", "护盾升起来。"),
        ("ch01_chunk003", "护盾落下去。"),
        ("ch01_chunk009", "风声掠过了。"),
    ):
        _add_completed_chunk(
            tmp_path, chunk_id,
            source_text=("Nothing happened." if chunk_id == "ch01_chunk009"
                         else "The shield held."),
            final_text=final_text)
    maps = _maps("屏蔽力场")
    maps["forms"] = [["屏蔽立场", "屏蔽力场", "屏蔽力场"]]
    monkeypatch.setattr(nof, "_build_maps", lambda root, db, chunks: maps)

    result = nof.audit_name_only_fastpath(tmp_path)

    assert result["counts"]["candidates"] == 0
    assert result["counts"]["kept_dirty"] == 1
    record = result["kept_dirty"][0]
    assert "decision_authority_not_confirmed" in record["reasons"]
    assert "no_authorized_lockable_decision" in record["reasons"]
    authority = record["decision_authority"][0]["authority"]
    assert authority["status"] == "contradicted"
    assert authority["lock_allowed"] is False


def test_contradicted_generic_authority_blocks_completed_name_repair(
        tmp_path: Path):
    _make_completed_fixture(
        tmp_path,
        final_text="屏蔽立场在挡光。",
        source_key="shield",
        canonical="屏蔽力场",
        source_text="The shield held.",
    )
    for chunk_id, final_text in (
        ("ch01_chunk001", "护盾撑住了。"),
        ("ch01_chunk002", "护盾升起来。"),
        ("ch01_chunk003", "护盾落下去。"),
        ("ch01_chunk009", "风声掠过了。"),
    ):
        _add_completed_chunk(
            tmp_path, chunk_id,
            source_text=("Nothing happened." if chunk_id == "ch01_chunk009"
                         else "The shield held."),
            final_text=final_text)

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 0
    kept = next(row for row in result["kept"]
                if row["chunk_id"] == "ch01_chunk000")
    assert kept["reasons"] == [
        "decision_authority_not_confirmed:contradicted"]
    assert kept["lock_authority"]["book_target_hits"] == 0


def test_completed_name_repair_allows_book_majority_authority(tmp_path: Path):
    _make_completed_fixture(
        tmp_path,
        final_text="屏蔽立场在挡光。",
        source_key="shield",
        canonical="屏蔽力场",
        source_text="The shield held.",
    )
    for chunk_id, final_text in (
        ("ch01_chunk001", "屏蔽力场撑住了。"),
        ("ch01_chunk002", "屏蔽力场升起来。"),
        ("ch01_chunk003", "屏蔽力场落下去。"),
        ("ch01_chunk009", "风声掠过了。"),
    ):
        _add_completed_chunk(
            tmp_path, chunk_id,
            source_text=("Nothing happened." if chunk_id == "ch01_chunk009"
                         else "The shield held."),
            final_text=final_text)

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["lock_authority"]["status"] == "confirmed"
    assert candidate["lock_authority"]["target_hits"] == 4


def _good_critic(candidate, source_text, raw_text, proposed_text, pipeline):
    return {
        "_backend_path": "jev",
        "_typesafe_verdict": {"decision": "PASS", "reject_reasons": []},
        "scores": {
            "voice_consistency": 8.5,
            "style_compliance": 8.5,
            "literal_residue": 8.5,
            "semantic_preservation": 8.5,
            "readability": 8.5,
            "fluency": 8.5,
            "entity_consistency": 10.0,
        },
    }


def test_apply_closes_dirty_with_real_reaudit_evidence(tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    text = "盖乌斯海伦莫希阿姆在观察。"
    _write_artifacts(tmp_path, text)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"
    from src.core.scheduler import TaskState
    assert nof_task_scheduler(tmp_path).update_task_state(
        "ch01_chunk000", TaskState.COMPLETED) is False
    assert nof_task_scheduler(tmp_path).get_task("ch01_chunk000")["state"] == "DIRTY"

    result = nof.apply_name_only_fastpath(
        tmp_path,
        apply=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="fixture-run",
    )

    assert result["ok"] is True
    assert result["counts"] == {"completed": 1}
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["text"] == "盖乌斯·海伦·莫希阿姆在观察。"
    assert final["metadata"]["name_unification_run_id"] == "fixture-run"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["_final_binding"]["final_sha256"] == sha256_text(final["text"])
    assert report["_reaudit"]["literary_input"] == "proposed_final"
    sched = nof_task_scheduler(tmp_path)
    assert sched.get_task("ch01_chunk000")["state"] == "COMPLETED"
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 1
    finally:
        con.close()
    backup = (tmp_path / "output" / "name_only_fastpath" / "fixture-run" / "ch01")
    assert (backup / "ch01_chunk000_final_before.json").exists()
    assert (backup / "ch01_chunk000_critic_report_before.json").exists()
    assert (backup / "ch01_chunk000_name_only_evidence.json").exists()


def test_state_only_closes_satisfied_dirty_without_rewriting_final(
        tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path, pending=[
        ("ch01_chunk000", 1, "Gaius Helen Mohiam", 1),
    ])
    text = "盖乌斯·海伦·莫希阿姆在观察。"
    _write_artifacts(tmp_path, text)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    before_final = final_path.read_bytes()

    default_plan = nof.audit_name_only_fastpath(tmp_path)
    assert default_plan["candidates"] == []
    assert "no_name_change" in default_plan["kept_dirty"][0]["reasons"]

    plan = nof.audit_name_only_fastpath(tmp_path, allow_state_only=True)
    assert plan["counts"]["candidates"] == 1
    assert plan["candidates"][0]["change_mode"] == "state_only"

    result = nof.apply_name_only_fastpath(
        tmp_path,
        apply=True,
        allow_state_only=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="state-only-run",
    )

    assert result["ok"] is True
    assert result["counts"] == {"completed": 1}
    assert result["actions"][0]["change_mode"] == "state_only"
    assert final_path.read_bytes() == before_final
    assert nof_task_scheduler(tmp_path).get_task(
        "ch01_chunk000")["state"] == "COMPLETED"
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute("SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 0
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 0
    finally:
        con.close()


def nof_task_scheduler(root: Path):
    from src.core.scheduler import TaskScheduler
    return TaskScheduler(db_path=str(root / "db" / "workflow.db"))


def test_apply_rolls_back_files_when_state_transition_fails(tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    _write_artifacts(tmp_path, "盖乌斯海伦莫希阿姆在观察。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"
    before_final = final_path.read_bytes()
    before_report = report_path.read_bytes()

    class RejectCompletion:
        def __init__(self, db_path):
            self.db_path = db_path

        def complete_name_only(self, *args, **kwargs):
            return {"ok": False, "reason": "simulated_state_failure"}

    result = nof.apply_name_only_fastpath(
        tmp_path,
        apply=True,
        scheduler=RejectCompletion(str(tmp_path / "db" / "workflow.db")),
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="rollback-run",
    )

    assert result["ok"] is False
    assert result["counts"] == {"rolled_back": 1}
    assert final_path.read_bytes() == before_final
    assert report_path.read_bytes() == before_report
    assert nof_task_scheduler(tmp_path).get_task("ch01_chunk000")["state"] == "DIRTY"
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 0
    finally:
        con.close()


def test_decision_commit_failure_reopens_state_and_restores_files(
        tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    _write_artifacts(tmp_path, "盖乌斯海伦莫希阿姆在观察。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"
    before_final = final_path.read_bytes()
    before_report = report_path.read_bytes()

    class CommitFailConnection:
        def __init__(self, connection):
            self._connection = connection

        def commit(self):
            self._connection.rollback()
            raise sqlite3.OperationalError("simulated decision commit failure")

        def __getattr__(self, name):
            return getattr(self._connection, name)

    monkeypatch.setattr(
        nof, "_open_decision_write_db",
        lambda path: CommitFailConnection(sqlite3.connect(str(path))),
    )

    result = nof.apply_name_only_fastpath(
        tmp_path,
        apply=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="decision-commit-failure-run",
    )

    assert result["ok"] is False
    assert result["counts"] == {"rolled_back": 1}
    assert "decision_commit_failed" in result["actions"][0]["reason"]
    assert final_path.read_bytes() == before_final
    assert report_path.read_bytes() == before_report
    assert nof_task_scheduler(tmp_path).get_task("ch01_chunk000")["state"] == "DIRTY"
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 0
    finally:
        con.close()


def test_apply_does_not_write_final_when_reaudit_fails(tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    _write_artifacts(tmp_path, "盖乌斯海伦莫希阿姆在观察。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    before_final = final_path.read_bytes()

    def bad_score(candidate, source_text, raw_text, proposed_text, pipeline):
        report = _good_critic(candidate, source_text, raw_text, proposed_text, pipeline)
        report["scores"] = dict(report["scores"], readability=4.0)
        return report

    result = nof.apply_name_only_fastpath(
        tmp_path,
        apply=True,
        critic_fn=bad_score,
        require_authority_audit=False,
        run_id="bad-score-run",
    )

    assert result["ok"] is True
    assert result["counts"] == {"kept_dirty": 1}
    assert final_path.read_bytes() == before_final
    assert nof_task_scheduler(tmp_path).get_task("ch01_chunk000")["state"] == "DIRTY"


def test_residual_entity_notes_keep_dirty(tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    _write_artifacts(tmp_path, "盖乌斯海伦莫希阿姆在观察。")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    before_final = final_path.read_bytes()

    def residual_name(candidate, source_text, raw_text, proposed_text, pipeline):
        report = _good_critic(
            candidate, source_text, raw_text, proposed_text, pipeline)
        report["entity_consistency_notes"] = [{
            "entity": "Gaius Helen Mohiam",
            "expected": "盖乌斯·海伦·莫希阿姆",
            "found": "盖乌斯海伦莫希阿姆",
        }]
        return report

    result = nof.apply_name_only_fastpath(
        tmp_path,
        apply=True,
        critic_fn=residual_name,
        require_authority_audit=False,
        run_id="residual-name-run",
    )

    assert result["ok"] is True
    assert result["counts"] == {"kept_dirty": 1}
    assert "entity_consistency_notes" in result["actions"][0]["reason"]
    assert final_path.read_bytes() == before_final
    assert nof_task_scheduler(tmp_path).get_task("ch01_chunk000")["state"] == "DIRTY"


def test_parenthetical_source_name_note_does_not_block_completion(
        tmp_path: Path, monkeypatch):
    _make_dbs(tmp_path)
    _write_artifacts(
        tmp_path,
        "盖乌斯海伦莫希阿姆在观察（Gaius Helen Mohiam）。",
    )
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")
    monkeypatch.setattr(
        nof, "_build_maps", lambda root, db, chunks: _maps("盖乌斯·海伦·莫希阿姆"))

    def source_name_in_parens(candidate, source_text, raw_text,
                              proposed_text, pipeline):
        report = _good_critic(
            candidate, source_text, raw_text, proposed_text, pipeline)
        report["entity_consistency_notes"] = [{
            "entity": "盖乌斯·海伦·莫希阿姆",
            "expected": "盖乌斯·海伦·莫希阿姆",
            "found": "Gaius Helen Mohiam",
            "type": "variant_instead_of_canonical",
        }]
        return report

    result = nof.apply_name_only_fastpath(
        tmp_path,
        apply=True,
        critic_fn=source_name_in_parens,
        require_authority_audit=False,
        run_id="parenthetical-source-run",
    )

    assert result["ok"] is True
    assert result["counts"] == {"completed": 1}
    final = json.loads(
        (tmp_path / "output" / "ch01" / "ch01_chunk000_final.json").read_text(
            encoding="utf-8"))
    assert final["text"].startswith("盖乌斯·海伦·莫希阿姆")


def _make_completed_fixture(
    root: Path,
    *,
    final_text: str = "贝尼·杰赛里特在观察。",
    source_key: str = "Bene Gesserit",
    canonical: str = "贝尼·杰瑟里特",
    source_text: str = "Bene Gesserit watched.",
) -> None:
    _make_dbs(
        root,
        canonical=canonical,
        decision_target=canonical,
        source_key=source_key,
        source_text=source_text,
        pending=[("ch01_chunk000", 1, source_key, 1)],
    )
    con = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id='ch01_chunk000'")
        con.commit()
    finally:
        con.close()
    _write_artifacts(root, final_text)


def _make_completed_short_alias_fixture(root: Path) -> None:
    _make_dbs(
        root,
        canonical="厄拉肯",
        decision_target="厄拉肯",
        source_key="Arrakeen",
        source_text="At Arrakeen the Duke waited.",
        pending=[("ch01_chunk000", 1, "Arrakeen", 1)],
    )
    con = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id='ch01_chunk000'")
        con.commit()
    finally:
        con.close()
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, confidence, "
            "pending_review) VALUES ('阿拉肯', ?, 'high', 1)",
            (json.dumps(["Arrakeen"]),),
        )
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, confidence, "
            "pending_review) VALUES ('厄拉科斯', ?, 'high', 0)",
            (json.dumps(["Arrakis"]),),
        )
        con.commit()
    finally:
        con.close()
    _write_artifacts(root, "在阿拉肯，公爵等待；厄拉科斯的风吹过。")


def _make_completed_raw_provenance_fixture(
    root: Path,
    *,
    final_text: str = "他的老师尤厄医生曾暗示。",
    raw_text: str = "他的老师岳医生曾暗示。",
) -> None:
    _make_dbs(
        root,
        canonical="岳医生",
        decision_target="岳医生",
        source_key="Yueh",
        source_text="Yueh watched.",
        pending=[("ch01_chunk000", 1, "Yueh", 1)],
    )
    con = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id='ch01_chunk000'")
        con.commit()
    finally:
        con.close()
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        con.execute("ALTER TABLE entity_registry ADD COLUMN variant_names TEXT")
        con.execute("ALTER TABLE entity_registry ADD COLUMN quarantined_names TEXT")
        con.execute(
            "UPDATE entity_registry SET canonical_name=?, source_names='[]', "
            "variant_names='[]', quarantined_names=?",
            ("威灵顿·岳", json.dumps(["Yueh"], ensure_ascii=False)),
        )
        con.commit()
    finally:
        con.close()
    _write_artifacts(root, final_text)
    raw_path = root / "output" / "ch01" / "ch01_chunk000_raw.json"
    raw_path.write_text(
        json.dumps({"text": raw_text}, ensure_ascii=False), encoding="utf-8")


def test_completed_name_repair_uses_raw_provenance_for_quarantined_alias(
        tmp_path: Path):
    _make_completed_raw_provenance_fixture(tmp_path)

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["repair_mode"] == "raw_provenance"
    assert candidate["wrong_forms"] == ["尤厄医生"]
    assert candidate["raw_provenance"]["status"] == "eligible"
    assert candidate["raw_provenance"]["raw_final_similarity"] >= 0.80
    assert candidate["pending_after"][0]["status"] == "satisfied"


def test_completed_name_repair_applies_raw_provenance_with_real_reaudit(
        tmp_path: Path):
    _make_completed_raw_provenance_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="completed-raw-provenance",
    )

    assert result["ok"] is True
    assert result["counts"] == {"repaired": 1}
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["text"] == "他的老师岳医生曾暗示。"
    assert final["metadata"]["completed_name_repair_forms"] == ["尤厄医生"]
    assert nof_task_scheduler(tmp_path).get_task("ch01_chunk000")["state"] == (
        "COMPLETED")
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 0
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 1
    finally:
        con.close()


def test_completed_name_repair_rejects_raw_candidate_owned_by_active_name(
        tmp_path: Path):
    _make_completed_raw_provenance_fixture(tmp_path)
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, "
            "variant_names, quarantined_names, confidence, pending_review) "
            "VALUES ('尤厄医生', ?, '[]', '[]', 'high', 0)",
            (json.dumps(["Another Yueh spelling"]),),
        )
        con.commit()
    finally:
        con.close()

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 0
    kept = next(row for row in result["kept"]
                if row["chunk_id"] == "ch01_chunk000")
    assert "drift_form_belongs_to_another_active_name" in kept["reasons"]


def _make_completed_source_scoped_fixture(
    root: Path,
    *,
    source_key: str,
    canonical: str,
    source_text: str,
    final_text: str,
) -> None:
    _make_dbs(
        root,
        canonical=canonical,
        decision_target=canonical,
        source_key=source_key,
        source_text=source_text,
        pending=[("ch01_chunk000", 1, source_key, 1)],
    )
    con = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id='ch01_chunk000'")
        con.commit()
    finally:
        con.close()
    _write_artifacts(root, final_text)


def test_completed_name_repair_uses_source_counted_variant(tmp_path: Path):
    _make_completed_source_scoped_fixture(
        tmp_path,
        source_key="Mapes",
        canonical="马佩斯",
        source_text="Mapes spoke. Mapes waited.",
        final_text="马普斯说了一句话。马普斯仍在等待。",
    )

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["repair_mode"] == "source_scoped_variant"
    assert candidate["wrong_forms"] == ["马普斯"]
    assert candidate["source_scoped_variant"]["source_count"] == 2
    assert len(candidate["changed_spans"]) == 2
    assert candidate["pending_after"][0]["status"] == "satisfied"


def test_completed_name_repair_uses_paired_source_gloss(tmp_path: Path):
    _make_completed_source_scoped_fixture(
        tmp_path,
        source_key="truthsense",
        canonical="辨真力",
        source_text="His truthsense warned him.",
        final_text=(
            "他的真实感（truthsense）提醒了他。\n"
            "[^1]: “真实感”（truthsense）：辨识真伪的能力。"
        ),
    )

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["repair_mode"] == "source_scoped_variant"
    assert candidate["wrong_forms"] == ["真实感"]
    assert candidate["source_scoped_variant"]["paired_gloss"] is True
    assert len(candidate["source_scoped_variant"]["matches"]) == 2


def test_completed_name_repair_trims_verb_from_paired_source_gloss(tmp_path: Path):
    _make_completed_source_scoped_fixture(
        tmp_path,
        source_key="carryall",
        canonical="运载飞行器",
        source_text="No one saw the carryall leave. The carryall was gone.",
        final_text="没人看见运输机（carryall）离开。运输机不见了。",
    )

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["repair_mode"] == "source_scoped_variant"
    assert candidate["wrong_forms"] == ["运输机"]
    assert candidate["source_scoped_variant"]["paired_gloss"] is True
    assert len(candidate["source_scoped_variant"]["matches"]) == 2


def test_completed_name_repair_rejects_source_count_mismatch(tmp_path: Path):
    _make_completed_source_scoped_fixture(
        tmp_path,
        source_key="Mapes",
        canonical="马佩斯",
        source_text="Mapes spoke. Mapes waited.",
        final_text="马普斯说了一句话。",
    )

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 0
    kept = next(row for row in result["kept"]
                if row["chunk_id"] == "ch01_chunk000")
    assert kept["source_scoped_variant"]["reason"] == "no_counted_term_variant"


def test_completed_name_repair_does_not_invent_raw_witness(tmp_path: Path):
    _make_completed_raw_provenance_fixture(
        tmp_path,
        final_text="他的老师尤厄医生曾暗示。",
        raw_text="他的老师尤赫医生曾暗示。",
    )

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 0
    kept = next(row for row in result["kept"]
                if row["chunk_id"] == "ch01_chunk000")
    assert kept["raw_provenance"]["reason"] == "target_not_in_raw_body"


def _make_completed_raw_rewrite_fixture(root: Path) -> None:
    _make_dbs(
        root,
        canonical="晶牙刀",
        decision_target="晶牙刀",
        source_key="crysknife",
        source_text="She felt the crysknife.",
        pending=[("ch01_chunk000", 1, "crysknife", 1)],
    )
    con = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id='ch01_chunk000'")
        con.commit()
    finally:
        con.close()
    _write_artifacts(root, "她感受着啸刃——冰冷的刀刃。")
    raw_path = root / "output" / "ch01" / "ch01_chunk000_raw.json"
    raw_path.write_text(json.dumps(
        {"text": "她感受着晶牙匕首——冰冷的刀刃。"},
        ensure_ascii=False), encoding="utf-8")


def test_completed_name_repair_uses_raw_rewrite_term_span(tmp_path: Path):
    _make_completed_raw_rewrite_fixture(tmp_path)

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["repair_mode"] == "raw_rewrite"
    assert candidate["wrong_forms"] == ["啸刃"]
    assert candidate["raw_rewrite"]["raw_final_similarity"] >= 0.80
    assert candidate["raw_rewrite"]["matches"][0]["raw_form"] == "晶牙匕首"


def test_completed_name_repair_applies_raw_rewrite_to_decision_target(
        tmp_path: Path):
    _make_completed_raw_rewrite_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="completed-raw-rewrite",
    )

    assert result["ok"] is True
    assert result["counts"] == {"repaired": 1}
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["text"] == "她感受着晶牙刀——冰冷的刀刃。"
    assert final["metadata"]["completed_name_repair_forms"] == ["啸刃"]
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 0
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 1
    finally:
        con.close()


def test_completed_name_repair_rejects_unbounded_raw_rewrite(tmp_path: Path):
    _make_completed_raw_rewrite_fixture(tmp_path)
    raw_path = tmp_path / "output" / "ch01" / "ch01_chunk000_raw.json"
    raw_path.write_text(json.dumps({
        "text": "她感受着晶牙匕首——冰冷的刀刃。\n\n" + ("无关旁白。" * 40),
    }, ensure_ascii=False), encoding="utf-8")

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 0
    kept = next(row for row in result["kept"]
                if row["chunk_id"] == "ch01_chunk000")
    assert kept["raw_rewrite"]["reason"] == "raw_final_drift_too_large"


def test_completed_name_repair_audit_is_read_only_and_selects_one_char_drift(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    workflow_path = tmp_path / "db" / "workflow.db"
    decision_path = tmp_path / "db" / "decision_db.sqlite"
    before = (final_path.read_bytes(), workflow_path.read_bytes(),
              decision_path.read_bytes())

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["mode"] == "read_only"
    assert result["write_attempted"] is False
    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["state_before"] == "COMPLETED"
    assert candidate["source_key"] == "Bene Gesserit"
    assert candidate["wrong_forms"] == ["贝尼·杰赛里特"]
    assert candidate["pending_after"][0]["status"] == "satisfied"
    assert candidate["final_sha256_after"] != candidate["final_sha256_before"]
    assert (final_path.read_bytes(), workflow_path.read_bytes(),
            decision_path.read_bytes()) == before


def test_completed_name_repair_accepts_exact_short_registry_alias(tmp_path: Path):
    _make_completed_short_alias_fixture(tmp_path)

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["repair_mode"] == "registered_alias"
    assert candidate["wrong_forms"] == ["阿拉肯"]
    assert candidate["pending_after"][0]["reason"] == "target_present"


def test_completed_name_repair_repairs_mixed_canonical_and_rejected_alias(
        tmp_path: Path):
    _make_completed_short_alias_fixture(tmp_path)
    _write_artifacts(tmp_path, "在阿拉肯，公爵等待；厄拉肯没有风。")

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["wrong_forms"] == ["阿拉肯"]
    assert candidate["pending_before"][0]["status"] == "satisfied"
    assert candidate["pending_after"][0]["status"] == "satisfied"


def test_completed_name_repair_applies_exact_short_registry_alias(tmp_path: Path):
    _make_completed_short_alias_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="completed-short-alias-run",
    )

    assert result["ok"] is True
    assert result["counts"] == {"repaired": 1}
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert "厄拉肯" in final["text"]
    assert "阿拉肯" not in final["text"]
    assert "厄拉科斯" in final["text"]
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 0
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 1
    finally:
        con.close()


def test_completed_name_repair_accepts_locked_missing_target_block(tmp_path: Path):
    _make_completed_fixture(tmp_path)
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("UPDATE decision_db SET locked=1 WHERE id=1")
        con.commit()
    finally:
        con.close()

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    assert result["candidates"][0]["pending_before"][0]["status"] == "blocked"
    assert result["candidates"][0]["pending_before"][0]["reason"] == (
        "locked_decision_target_missing")


def test_completed_name_repair_apply_keeps_completed_and_consumes_pending(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="completed-repair-fixture",
    )

    assert result["ok"] is True
    assert result["counts"] == {"repaired": 1}
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["text"] == "贝尼·杰瑟里特在观察。"
    assert final["metadata"]["completed_name_repair_run_id"] == (
        "completed-repair-fixture")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["_final_binding"]["final_sha256"] == sha256_text(final["text"])
    assert report["_reaudit"]["tool"] == "completed-name-repair"
    sched = nof_task_scheduler(tmp_path)
    assert sched.get_task("ch01_chunk000")["state"] == "COMPLETED"
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 0
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 1
    finally:
        con.close()
    assert (tmp_path / "output" / "completed_name_repair" /
            "completed-repair-fixture" / "ch01" /
            "ch01_chunk000_completed_name_repair_evidence.json").exists()


def test_completed_name_repair_reaudit_failure_leaves_everything_intact(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"
    before_final = final_path.read_bytes()
    before_report = report_path.read_bytes()

    def bad_score(candidate, source_text, raw_text, proposed_text, pipeline):
        report = _good_critic(
            candidate, source_text, raw_text, proposed_text, pipeline)
        report["scores"] = dict(report["scores"], readability=4.0)
        return report

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=bad_score,
        require_authority_audit=False,
        run_id="completed-bad-score-run",
    )

    assert result["ok"] is True
    assert result["counts"] == {"kept": 1}
    assert final_path.read_bytes() == before_final
    assert report_path.read_bytes() == before_report
    sched = nof_task_scheduler(tmp_path)
    assert sched.get_task("ch01_chunk000")["state"] == "COMPLETED"
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 1
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 0
    finally:
        con.close()


def test_completed_name_repair_infrastructure_failure_is_not_success(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"
    before_final = final_path.read_bytes()
    before_report = report_path.read_bytes()

    def unavailable(candidate, source_text, raw_text, proposed_text, pipeline):
        raise RuntimeError("network unavailable")

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=unavailable,
        require_authority_audit=False,
        run_id="completed-unavailable-run",
    )

    assert result["ok"] is False
    assert result["counts"] == {"failed": 1}
    assert any("network unavailable" in error for error in result["errors"])
    assert final_path.read_bytes() == before_final
    assert report_path.read_bytes() == before_report


def _bad_completed_critic(candidate, source_text, raw_text, proposed_text, pipeline):
    report = _good_critic(
        candidate, source_text, raw_text, proposed_text, pipeline)
    report["scores"] = dict(report["scores"], readability=4.0)
    return report


def test_completed_name_repair_reject_memo_skips_identical_reaudit(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)

    first = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_bad_completed_critic,
        require_authority_audit=False,
        run_id="completed-reject-memo-first",
    )
    memo_path = tmp_path / "output" / "completed_name_repair_rejects.json"

    assert first["counts"] == {"kept": 1}
    assert first["reject_memo_recorded"] == 1
    assert memo_path.exists()
    second = nof.audit_completed_name_repair(tmp_path)
    assert second["counts"]["candidates"] == 0
    assert second["counts"]["memo_skipped"] == 1
    assert second["kept"][0]["reasons"] == ["reaudit_reject_memo"]


def test_completed_name_repair_reject_memo_invalidated_by_critic_change(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)
    nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_bad_completed_critic,
        require_authority_audit=False,
        run_id="completed-reject-memo-critic",
    )
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["scores"]["fluency"] = 8.6
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    assert result["counts"]["memo_skipped"] == 0


def test_completed_name_repair_reject_memo_invalidated_by_final_change(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)
    nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_bad_completed_critic,
        require_authority_audit=False,
        run_id="completed-reject-memo-final",
    )
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["text"] += "夜色。"
    final_path.write_text(json.dumps(final, ensure_ascii=False), encoding="utf-8")

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    assert result["counts"]["memo_skipped"] == 0


def test_completed_name_repair_reject_memo_invalidated_by_decision_change(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)
    nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_bad_completed_critic,
        require_authority_audit=False,
        run_id="completed-reject-memo-decision",
    )
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        con.execute("UPDATE decision_db SET locked=1 WHERE id=1")
        con.commit()
    finally:
        con.close()

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    assert result["counts"]["memo_skipped"] == 0


def test_completed_name_repair_transient_backend_failure_is_not_memoized(
        tmp_path: Path):
    _make_completed_fixture(tmp_path)

    def transient(candidate, source_text, raw_text, proposed_text, pipeline):
        report = _good_critic(
            candidate, source_text, raw_text, proposed_text, pipeline)
        report.pop("_backend_path")
        return report

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=transient,
        require_authority_audit=False,
        run_id="completed-transient-failure",
    )

    assert result["counts"] == {"kept": 1}
    assert result["reject_memo_recorded"] == 0
    assert not (tmp_path / "output" / "completed_name_repair_rejects.json").exists()


def test_completed_name_repair_pending_cas_failure_rolls_back_files(
        tmp_path: Path, monkeypatch):
    _make_completed_fixture(tmp_path)
    final_path = tmp_path / "output" / "ch01" / "ch01_chunk000_final.json"
    report_path = tmp_path / "output" / "ch01" / "ch01_chunk000_critic_report.json"
    before_final = final_path.read_bytes()
    before_report = report_path.read_bytes()
    monkeypatch.setattr(
        nof, "_delete_pending_row_if_unchanged", lambda con, row: False)

    result = nof.apply_completed_name_repair(
        tmp_path,
        apply=True,
        critic_fn=_good_critic,
        require_authority_audit=False,
        run_id="completed-cas-failure-run",
    )

    assert result["ok"] is False
    assert result["counts"] == {"rolled_back": 1}
    assert final_path.read_bytes() == before_final
    assert report_path.read_bytes() == before_report
    sched = nof_task_scheduler(tmp_path)
    assert sched.get_task("ch01_chunk000")["state"] == "COMPLETED"
    con = sqlite3.connect(str(tmp_path / "db" / "decision_db.sqlite"))
    try:
        assert con.execute(
            "SELECT COUNT(*) FROM pending_backtrack").fetchone()[0] == 1
        assert con.execute(
            "SELECT locked FROM decision_db WHERE id=1").fetchone()[0] == 0
    finally:
        con.close()


def test_completed_name_repair_rejects_short_common_term(tmp_path: Path):
    _make_completed_fixture(
        tmp_path,
        final_text="水壶在桌上。",
        source_key="flagon",
        canonical="酒壶",
        source_text="A flagon stood on the table.",
    )

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 0
    assert result["counts"]["kept"] == 1
    assert "canonical_not_pure_cjk_or_too_short" in (
        result["kept"][0]["reasons"])



def _make_completed_target_migration_fixture(root: Path) -> None:
    _make_dbs(
        root,
        canonical="卡萨格",
        decision_target="卡萨格",
        source_key="Carthag",
        source_text="He came from Carthag with a stillsuit.",
        pending=[("ch01_chunk000", 1, "Carthag", 1)],
    )
    con = sqlite3.connect(str(root / "db" / "workflow.db"))
    try:
        con.execute(
            "UPDATE chunk_tasks SET state='COMPLETED' WHERE chunk_id='ch01_chunk000'")
        con.commit()
    finally:
        con.close()
    con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
    try:
        con.execute(
            "UPDATE decision_db SET reason=? WHERE id=1",
            ("Round 158 decision target authority repair: 迦太基 -> 卡萨格",),
        )
        con.commit()
    finally:
        con.close()
    _write_artifacts(root, "他来自迦太基，带着一套蒸馏服。")


def test_completed_name_repair_accepts_round158_target_migration(tmp_path: Path):
    _make_completed_target_migration_fixture(tmp_path)

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 1
    candidate = result["candidates"][0]
    assert candidate["repair_mode"] == "decision_target_migration"
    assert candidate["source_key"] == "Carthag"
    assert candidate["wrong_forms"] == ["迦太基"]
    assert candidate["target_migration"]["old_target"] == "迦太基"
    assert candidate["target_migration"]["new_target"] == "卡萨格"
    assert candidate["pending_after"][0]["status"] == "satisfied"


def test_completed_name_repair_rejects_target_migration_count_mismatch(
        tmp_path: Path):
    _make_completed_target_migration_fixture(tmp_path)
    _write_artifacts(tmp_path, "他来自迦太基，旧迦太基的名字仍在。")

    result = nof.audit_completed_name_repair(tmp_path)

    assert result["counts"]["candidates"] == 0
    record = result["kept"][0]
    assert "target_migration:migration_occurrence_mismatch" in record["reasons"]
