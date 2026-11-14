"""Round 147: failed-chunk rescue, ledger evidence, and final-gate closure.

These tests keep the production-first invariant: a failed chunk must be
classified, retried from the cheapest viable stage, and quarantined only when
the bounded rescue budget is exhausted.  A fallback final is never deliverable,
even when the database says COMPLETED.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.scheduler import TaskScheduler, TaskState
from src.pipeline import TranslationPipeline, run_final_gate
from src.rescue import (
    append_rescue_event,
    collect_final_status,
    failure_cohorts,
    latest_events_by_chunk,
    ledger_requeue_candidates,
    qa_report_mismatches,
    recommended_resume_stage,
    write_qa_report,
)


def _make_pipeline(tmp_path: Path):
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.chapter_id = "ch01"
    pipeline._root_dir = tmp_path
    pipeline.output_dir = tmp_path / "output" / "ch01"
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.scheduler = TaskScheduler(db_path=str(tmp_path / "db" / "workflow.db"))
    return pipeline


@pytest.fixture
def pipeline(tmp_path):
    p = _make_pipeline(tmp_path)
    yield p
    p.scheduler.conn.close()


def _insert_task(pipeline, chunk_id: str, state: str, last_error: str = "",
                 retries: int = 0, rescue_attempts: int = 0,
                 pf_requeues: int = 0) -> None:
    pipeline.scheduler.conn.execute(
        "INSERT OR REPLACE INTO chunk_tasks "
        "(chunk_id, chapter_id, text_content, state, retries, last_error, "
        " rescue_attempts, pf_requeues) "
        "VALUES (?, 'ch01', '源文本', ?, ?, ?, ?, ?)",
        (chunk_id, state, retries, last_error, rescue_attempts, pf_requeues),
    )
    pipeline.scheduler.conn.commit()


def _write_final(pipeline, chunk_id: str, text: str, *, fallback: bool = False,
                 reason: str = "") -> Path:
    path = pipeline.output_dir / f"{chunk_id}_final.json"
    metadata = {"fallback": True, "reason": reason} if fallback else {}
    path.write_text(
        json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _insert_critic_report(pipeline, chunk_id: str, backend: str = "jev") -> None:
    (pipeline.output_dir / f"{chunk_id}_critic_report.json").write_text(
        json.dumps({"_backend_path": backend}), encoding="utf-8")


class TestFailureStageSelection:
    def test_api_failure_resumes_at_the_failed_stage(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        (out / "ch01_chunk000_raw.json").write_text("{}", encoding="utf-8")
        (out / "ch01_chunk000_literary.json").write_text("{}", encoding="utf-8")
        task = {"chunk_id": "ch01_chunk000",
                "last_error": "[AUDITING] timeout while calling critic"}
        assert recommended_resume_stage(task, "api_or_no_content", out) == "AUDITING"

    def test_quality_failure_reuses_literary_before_retranslating(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        (out / "ch01_chunk000_raw.json").write_text("{}", encoding="utf-8")
        (out / "ch01_chunk000_literary.json").write_text("{}", encoding="utf-8")
        task = {"chunk_id": "ch01_chunk000", "last_error": ""}
        assert recommended_resume_stage(task, "quality", out) == "AUDITING"
        (out / "ch01_chunk000_literary.json").unlink()
        assert recommended_resume_stage(task, "quality", out) == "REWRITING_LITERARY"
        (out / "ch01_chunk000_raw.json").unlink()
        assert recommended_resume_stage(task, "quality", out) == "TRANSLATING_RAW"

    def test_cjk_failure_rewrites_only_when_raw_exists(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        task = {"chunk_id": "ch01_chunk000", "last_error": ""}
        (out / "ch01_chunk000_raw.json").write_text("{}", encoding="utf-8")
        assert recommended_resume_stage(task, "cjk", out) == "REWRITING_LITERARY"
        (out / "ch01_chunk000_raw.json").unlink()
        assert recommended_resume_stage(task, "cjk", out) == "TRANSLATING_RAW"


class TestRescueQueue:
    def test_stub_without_root_does_not_fall_back_to_cwd(self, pipeline):
        del pipeline._root_dir
        # output/<chapter>/ 可安全反推到临时项目根；关键不变量是绝不写 cwd。
        assert pipeline._rescue_project_root() == pipeline.output_dir.parent.parent
        assert pipeline._rescue_project_root() != Path.cwd()

    def test_pf_is_revived_as_a_targeted_dirty_task(self, pipeline):
        _insert_task(pipeline, "ch01_chunk000", "PERMANENTLY_FAILED",
                     "[AUDITING] timeout", rescue_attempts=0)
        (pipeline.output_dir / "ch01_chunk000_raw.json").write_text("{}", encoding="utf-8")
        (pipeline.output_dir / "ch01_chunk000_literary.json").write_text("{}", encoding="utf-8")

        queued = pipeline._auto_rescue_existing_failures()

        assert queued == 1
        row = pipeline.scheduler.get_task("ch01_chunk000")
        assert row["state"] == "DIRTY"
        assert row["rescue_stage"] == "AUDITING"
        assert row["rescue_attempts"] == 1
        events = latest_events_by_chunk(pipeline._root_dir)
        assert events["ch01_chunk000"]["action"] == "targeted_rescue"
        assert events["ch01_chunk000"]["result"] == "queued"

    def test_completed_fallback_final_enters_rescue(self, pipeline):
        _insert_task(pipeline, "ch01_chunk001", "COMPLETED")
        _write_final(pipeline, "ch01_chunk001", "降级稿", fallback=True,
                     reason="SAFETY_NET_PASSED")

        queued = pipeline._auto_rescue_existing_failures()

        assert queued == 1
        row = pipeline.scheduler.get_task("ch01_chunk001")
        assert row["state"] == "DIRTY"
        assert row["rescue_stage"] == "TRANSLATING_RAW"
        assert row["rescue_attempts"] == 1

    def test_targeted_and_full_budget_then_quarantine(self, pipeline):
        _insert_task(pipeline, "ch01_chunk002", "FAILED",
                     "[AUDITING] timeout", rescue_attempts=2, pf_requeues=1)
        task = pipeline.scheduler.get_task("ch01_chunk002")

        assert pipeline._enter_rescue_or_pf(
            "ch01_chunk002", "[AUDITING] timeout", "api_or_no_content") is False

        row = pipeline.scheduler.get_task("ch01_chunk002")
        assert row["state"] == "PERMANENTLY_FAILED"
        event = latest_events_by_chunk(pipeline._root_dir)["ch01_chunk002"]
        assert event["action"] == "quarantine"
        assert event["result"] == "quarantined"

    def test_dirty_rescue_is_staged_and_clears_one_shot_marker(self, pipeline):
        _insert_task(pipeline, "ch01_chunk003", "PERMANENTLY_FAILED",
                     "[REWRITING_LITERARY] score_decline", rescue_attempts=0)
        (pipeline.output_dir / "ch01_chunk003_raw.json").write_text("{}", encoding="utf-8")
        (pipeline.output_dir / "ch01_chunk003_literary.json").write_text("{}", encoding="utf-8")
        pipeline._auto_rescue_existing_failures()
        pipeline._process_dirty_batch(pipeline.scheduler.get_tasks_by_state(TaskState.DIRTY, 10))

        row = pipeline.scheduler.get_task("ch01_chunk003")
        assert row["state"] == "AUDITING"
        assert row["rescue_stage"] is None
        assert row["quality_retries"] == 1


class TestQAEvidence:
    def test_qa_report_hash_roundtrip_and_stale_detection(self, pipeline):
        _write_final(pipeline, "ch01_chunk000", "合格稿")
        _write_final(pipeline, "ch01_chunk001", "降级稿", fallback=True,
                     reason="EARLY_STOP")
        result = write_qa_report(pipeline._root_dir, pipeline._root_dir / "output")
        qa_file = Path(result["path"])

        assert result["final_count"] == 2
        problems, info = qa_report_mismatches(
            pipeline._root_dir, pipeline._root_dir / "output", qa_file)
        assert problems == []
        assert info["expected_fallbacks"] == ["ch01_chunk001"]
        assert info["reported_final_count"] == 2

        # A later normalization pass must not leave a stale hash in QA.
        _write_final(pipeline, "ch01_chunk000", "规范化后的合格稿")
        problems, _ = qa_report_mismatches(
            pipeline._root_dir, pipeline._root_dir / "output", qa_file)
        assert any("哈希不一致" in p for p in problems)

        write_qa_report(pipeline._root_dir, pipeline._root_dir / "output")
        problems, _ = qa_report_mismatches(
            pipeline._root_dir, pipeline._root_dir / "output", qa_file)
        assert problems == []

    def test_qa_report_detects_missing_and_extra_final(self, pipeline):
        _write_final(pipeline, "ch01_chunk000", "合格稿")
        result = write_qa_report(pipeline._root_dir, pipeline._root_dir / "output")
        qa_file = Path(result["path"])
        _write_final(pipeline, "ch01_chunk001", "新增稿")

        problems, _ = qa_report_mismatches(
            pipeline._root_dir, pipeline._root_dir / "output", qa_file)
        assert any("漏记 final" in p for p in problems)

    def test_qa_report_detects_malformed_final(self, pipeline):
        (pipeline.output_dir / "ch01_chunk000_final.json").write_text(
            "{broken", encoding="utf-8")
        result = write_qa_report(pipeline._root_dir, pipeline._root_dir / "output")
        assert result["ok"] is False
        status = collect_final_status(pipeline._root_dir, pipeline._root_dir / "output")
        assert len(status["errors"]) == 1


class TestFinalGateClosure:
    def test_fallback_final_is_red_even_when_database_is_completed(self, pipeline, tmp_path):
        _insert_task(pipeline, "ch01_chunk000", "COMPLETED")
        _write_final(pipeline, "ch01_chunk000", "他说：“你好。”", fallback=True,
                     reason="SAFETY_NET_PASSED")
        _insert_critic_report(pipeline, "ch01_chunk000")
        write_qa_report(pipeline._root_dir, pipeline._root_dir / "output")
        (tmp_path / "output" / "translated_full.md").write_text(
            "第一章\n\n他说：“你好。”\n", encoding="utf-8")
        (tmp_path / "input").mkdir(exist_ok=True)
        (tmp_path / "input" / "_conversion_audit.md").write_text(
            "# 转换完整性审计报告\n\n- 全书覆盖率: 99.900%\n- 结论: ✅ 通过\n",
            encoding="utf-8",
        )

        result = run_final_gate(project_root=str(tmp_path))

        assert result["ok"] is False
        joined = "\n".join(result["failures"])
        assert "⑨ 检测到 fallback" in joined
        assert "ch01_chunk000" in joined

    def test_stale_qa_is_red_after_final_changes(self, pipeline, tmp_path):
        _insert_task(pipeline, "ch01_chunk000", "COMPLETED")
        _write_final(pipeline, "ch01_chunk000", "第一版")
        _insert_critic_report(pipeline, "ch01_chunk000")
        write_qa_report(pipeline._root_dir, pipeline._root_dir / "output")
        _write_final(pipeline, "ch01_chunk000", "第二版")
        (tmp_path / "output" / "translated_full.md").write_text("第二版\n", encoding="utf-8")
        (tmp_path / "input").mkdir(exist_ok=True)
        (tmp_path / "input" / "_conversion_audit.md").write_text(
            "- 全书覆盖率: 99.900%\n- 结论: ✅ 通过\n", encoding="utf-8")

        result = run_final_gate(project_root=str(tmp_path))

        assert result["ok"] is False
        assert "③ QA_REPORT 与当前 final 不一致" in "\n".join(result["failures"])


class TestLedger:
    def test_ledger_is_append_only_and_cohorts_are_counted(self, tmp_path):
        append_rescue_event(tmp_path, {
            "chunk_id": "ch01_chunk000", "chapter": "ch01",
            "failure_class": "api_or_no_content", "result": "queued",
            "action": "targeted_rescue", "model": "primary",
        })
        append_rescue_event(tmp_path, {
            "chunk_id": "ch01_chunk000", "chapter": "ch01",
            "failure_class": "api_or_no_content", "result": "quarantined",
            "action": "quarantine", "model": "primary",
        })
        append_rescue_event(tmp_path, {
            "chunk_id": "ch01_chunk001", "chapter": "ch01",
            "failure_class": "quality", "result": "completed",
            "action": "rescue_result", "model": "secondary",
        })

        latest = latest_events_by_chunk(tmp_path)
        assert latest["ch01_chunk000"]["result"] == "quarantined"
        assert ledger_requeue_candidates(tmp_path, "ch01") == ["ch01_chunk000"]
        cohorts = failure_cohorts(list(latest.values()))
        assert sum(x["count"] for x in cohorts) == 2


class TestMonolithMirror:
    def test_generated_monolith_contains_rescue_pipeline(self):
        from src.translator_agent import TaskScheduler as MonoScheduler
        from src.translator_agent import TranslationPipeline as MonoPipeline

        for name in (
            "_auto_rescue_existing_failures",
            "_enter_rescue_or_pf",
            "_process_dirty_batch",
            "_record_rescue_completions",
            "_reset_dirty_runtime_state",
        ):
            assert hasattr(MonoPipeline, name), name
        for name in ("queue_rescue", "stage_rescue_task"):
            assert hasattr(MonoScheduler, name), name
