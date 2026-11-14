"""Round 148: content-policy (HTTP 422) failures switch roles immediately.

The production incident: DeepSeek rejected a literary prompt with
``input new_sensitive (1026)`` while Qwen had already translated the same
source chunk successfully.  Repeating the same provider+prompt is a permanent
client rejection, so the pipeline must persist a role switch on the first
rejection and quarantine only after every configured fallback role refuses.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from src.core.scheduler import TaskScheduler, TaskState
from src.pipeline import TranslationPipeline
from src.rescue import classify_failure, recommended_resume_stage


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


class _Resp:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": "正文"}}], "usage": {}}


class TestAdapterClassification:
    def _adapter(self):
        from src.utils.llm_adapter import OpenAICompatibleAdapter
        adapter = OpenAICompatibleAdapter(
            api_base="https://token.ai-hf.cn/v1", api_key="test-key",
            max_retries=3, retry_delay=0.01,
        )
        adapter.min_request_interval = 0.0
        OpenAICompatibleAdapter._rate_limiters[
            adapter._get_limiter_key(adapter.api_base)]["interval"] = 0.0
        return adapter

    def test_sensitive_422_raises_dedicated_error_without_retry(self):
        from src.utils.llm_adapter import ContentSensitiveError
        adapter = self._adapter()
        calls = {"n": 0}

        def _post(url, **kwargs):
            calls["n"] += 1
            return _Resp(422, '{"message":"input new_sensitive (1026)"}')

        adapter._requests = types.SimpleNamespace(post=_post)
        with pytest.raises(ContentSensitiveError) as exc:
            adapter.generate("prompt", model_name="Deepseek-V4.1-Flash-1017-SS")
        assert exc.value.status_code == 422
        assert calls["n"] == 1

    def test_other_422_is_not_misclassified_as_sensitive(self):
        from src.utils.llm_adapter import ContentSensitiveError
        adapter = self._adapter()
        adapter._requests = types.SimpleNamespace(
            post=lambda url, **kwargs: _Resp(422, '{"message":"invalid field schema"}'))
        with pytest.raises(RuntimeError) as exc:
            adapter.generate("prompt", model_name="Deepseek-V4.1-Flash-1017-SS")
        assert not isinstance(exc.value, ContentSensitiveError)


class TestFailureClassAndStage:
    def test_sensitive_is_a_distinct_failure_class(self):
        reason = 'API 客户端错误 422: {"message":"input new_sensitive (1026)"}'
        assert classify_failure(reason, "REWRITING_LITERARY") == "content_sensitive"
        assert classify_failure(reason, "REWRITING_LITERARY") != "api_or_no_content"

    def test_resume_stage_skips_audit_when_raw_exists(self, tmp_path):
        out = tmp_path / "output"
        out.mkdir()
        (out / "ch01_chunk000_raw.json").write_text("{}", encoding="utf-8")
        task = {"chunk_id": "ch01_chunk000",
                "last_error": "[REWRITING_LITERARY] HTTP 422 new_sensitive"}
        assert recommended_resume_stage(task, "content_sensitive", out) == "REWRITING_LITERARY"
        (out / "ch01_chunk000_raw.json").unlink()
        assert recommended_resume_stage(task, "content_sensitive", out) == "TRANSLATING_RAW"


class TestSensitiveRoleSwitch:
    def test_first_422_switches_to_primary_then_qwen(self, pipeline):
        _insert_task(pipeline, "ch01_chunk000", "FAILED",
                     "[REWRITING_LITERARY] 422 input new_sensitive", retries=1)
        assert pipeline._record_content_sensitive_signal(
            "ch01_chunk000", "REWRITING_LITERARY", "secondary", "xhigh") is True
        ov = pipeline.scheduler.get_retry_override("ch01_chunk000")
        assert ov["sig"] == "content_sensitive"
        assert ov["role"] == "primary"
        assert ov["sensitive_tried_roles"] == ["secondary"]

        assert pipeline._record_content_sensitive_signal(
            "ch01_chunk000", "REWRITING_LITERARY", "primary", "max") is True
        ov = pipeline.scheduler.get_retry_override("ch01_chunk000")
        assert ov["role"] == "literal_translator"
        assert ov["effort"] == "xhigh"  # max 不在 Qwen 档位表，回落到任务默认档
        assert ov["sensitive_tried_roles"] == ["secondary", "primary"]

        assert pipeline._record_content_sensitive_signal(
            "ch01_chunk000", "REWRITING_LITERARY", "literal_translator", "xhigh") is False
        ov = pipeline.scheduler.get_retry_override("ch01_chunk000")
        assert ov["sensitive_exhausted"] is True
        assert pipeline._content_sensitive_exhausted("ch01_chunk000") is True

    def test_exhausted_sensitive_chunk_is_quarantined_without_rescue_loop(self, pipeline):
        _insert_task(pipeline, "ch01_chunk000", "FAILED",
                     "[REWRITING_LITERARY] 422 input new_sensitive", retries=3)
        pipeline._record_content_sensitive_signal(
            "ch01_chunk000", "REWRITING_LITERARY", "secondary", "xhigh")
        pipeline._record_content_sensitive_signal(
            "ch01_chunk000", "REWRITING_LITERARY", "primary", "xhigh")
        pipeline._record_content_sensitive_signal(
            "ch01_chunk000", "REWRITING_LITERARY", "literal_translator", "xhigh")
        task = pipeline.scheduler.get_task("ch01_chunk000")
        pipeline._process_failed_batch([task])
        task = pipeline.scheduler.get_task("ch01_chunk000")
        assert task["state"] == TaskState.PERMANENTLY_FAILED.value
        ledger = pipeline._root_dir / "output" / "rescue" / "failure_ledger.jsonl"
        events = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
        assert events[-1]["action"] == "quarantine"
        assert events[-1]["failure_class"] == "content_sensitive"


class TestBudgetReset:
    def test_reset_only_clears_selected_unfinished_chunks(self, pipeline):
        _insert_task(pipeline, "ch01_chunk000", "PERMANENTLY_FAILED", "422 sensitive",
                     retries=4, rescue_attempts=2, pf_requeues=1)
        _insert_task(pipeline, "ch01_chunk001", "COMPLETED", "", retries=0,
                     rescue_attempts=1, pf_requeues=0)
        pipeline.scheduler.set_retry_override("ch01_chunk000", {"sig": "content_sensitive"})
        reset = pipeline.scheduler.reset_rescue_budget("ch01")
        assert reset == ["ch01_chunk000"]
        task = pipeline.scheduler.get_task("ch01_chunk000")
        assert task["retries"] == 0
        assert task["rescue_attempts"] == 0
        assert task["pf_requeues"] == 0
        assert task["retry_effort_overrides"] is None
        done = pipeline.scheduler.get_task("ch01_chunk001")
        assert done["rescue_attempts"] == 1


class TestMonolithMirror:
    def test_generated_monolith_contains_sensitive_rescue(self):
        import src.translator_agent as mono

        assert issubclass(mono.ContentSensitiveError, RuntimeError)
        for name in (
            "_is_content_sensitive_signature",
            "_record_content_sensitive_signal",
            "_content_sensitive_exhausted",
        ):
            assert hasattr(mono.TranslationPipeline, name), name
        assert hasattr(mono.TaskScheduler, "reset_rescue_budget")
