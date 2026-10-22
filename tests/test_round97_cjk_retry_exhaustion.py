"""Round 97: CJK 重试耗尽早停 — rewriter 反复输出英文时, 提前 fallback 到 raw.

设计动机:
  Round 96 P3 在 critic 阶段加 CJK 硬闸门, 拦截低 CJK 输出触发 rewriter 重试.
  但如果 rewriter 反复输出英文 (LLM 模型选择错误或 prompt 失效), P3 会让
  quality_retries 增长到 max_quality_retries (默认 4) 才走 fallback.
  Round 97 加 CJK 专用重试上限 cjk_max_retries (默认 2), 比通用上限更激进:
    1. 节省 2-4 次 rewriter+critic LLM 调用 (单 chunk 节省 ~$0.01 + ~10s)
    2. raw 翻译稿经 P2 闸门保证是中文, fallback 后译文至少是中文 (文学性差但可读)
    3. final.json 标记 CJK_RETRY_EXHAUSTED 让下游可识别

测试覆盖:
  - cjk_max_retries=0 禁用早停 (向后兼容)
  - CJK-only reject + retries >= cjk_max_retries → 早停 fallback 到 raw
  - 其他 reject 原因 (非 CJK) → 不触发早停 (走原流程)
  - 早停后 state = PERMANENTLY_FAILED + error_msg 含 [CJK_RETRY_EXHAUSTED]
  - 节省 LLM 调用次数: quality_retries < max_quality_retries
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_pipeline_stub(output_dir, *, cjk_max_retries=2, raw_text="这是直译稿的中文内容"):
    """构造 TranslationPipeline stub, _process_judging_batch 可调用."""
    import src.config
    from src.core.scheduler import TaskState
    from src.translator_agent import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock()
    pipeline._llm_call_counter = {
        "reference_extraction": 0,
        "literal_translation": 0,
        "literary_rewriting": 0,
        "critic_scoring": 0,
        "judge_decision": 0,
    }
    pipeline._score_history = {}

    # 写 raw.json + literary.json (文学稿是英文, 触发 P3)
    (output_dir / "ch_test_chunk000_raw.json").write_text(
        json.dumps({"text": raw_text}, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ch_test_chunk000_literary.json").write_text(
        json.dumps({"text": "This is English literary translation only."}, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "ch_test_chunk000_critic_report.json").write_text(
        json.dumps({"scores": {"voice_consistency": 5.0, "literal_residue": 6.0}}, ensure_ascii=False),
        encoding="utf-8",
    )

    # pipeline 方法桩
    pipeline._load_intermediate = lambda cid, step: json.loads(
        (output_dir / f"{cid}_{step}.json").read_text(encoding="utf-8")
    ).get("text", "")
    pipeline._save_intermediate = lambda cid, step, data: (
        (output_dir / f"{cid}_{step}.json").write_text(
            json.dumps({"text": data} if isinstance(data, str) else data,
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )
    pipeline._compute_avg_score = lambda report: 5.5
    pipeline._is_declining_streak = lambda cid, score, consecutive=3: False
    pipeline._get_best_lit = lambda cid, fallback: fallback

    # config
    src.config.config.pipeline["cjk_max_retries"] = cjk_max_retries
    src.config.config.pipeline["max_quality_retries"] = 4

    return pipeline, TaskState


class TestCjkRetryExhaustion:
    """Round 97: _process_judging_batch CJK 重试耗尽早停."""

    def test_cjk_reject_with_max_retries_triggers_early_stop(self, tmp_path):
        """CJK 拒绝 + quality_retries >= cjk_max_retries → 早停 fallback."""
        pipeline, TaskState = _make_pipeline_stub(tmp_path, cjk_max_retries=2)

        tasks = [{
            "chunk_id": "ch_test_chunk000",
            "text_content": "English source",
            "quality_retries": 2,  # == cjk_max_retries
        }]
        judge_result = {
            "decision": "REJECT",
            "reject_reason": "译文 CJK 比例过低: cjk_ratio=0.05 < 0.50 (lit=42 chars)",
        }
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(tasks)

        # 应标 PERMANENTLY_FAILED + error_msg 含 [CJK_RETRY_EXHAUSTED]
        pipeline.scheduler.update_task_state.assert_called_once()
        call_args = pipeline.scheduler.update_task_state.call_args
        assert call_args[0][1].value == TaskState.PERMANENTLY_FAILED.value
        assert "[CJK_RETRY_EXHAUSTED]" in call_args[1]["error_msg"]

        # final.json 应被保存 (用 raw 内容 fallback)
        final = json.loads((tmp_path / "ch_test_chunk000_final.json").read_text(encoding="utf-8"))
        assert final["text"] == "这是直译稿的中文内容"
        assert final["metadata"]["fallback"] is True
        assert final["metadata"]["reason"] == "CJK_RETRY_EXHAUSTED"
        assert final["metadata"]["cjk_max_retries"] == 2
        assert final["metadata"]["raw_cjk_ratio"] == pytest.approx(1.0, abs=1e-3)

    def test_cjk_reject_below_max_retries_does_not_trigger(self, tmp_path):
        """CJK 拒绝但 retries < cjk_max_retries → 走原流程 (不早停)."""
        pipeline, TaskState = _make_pipeline_stub(tmp_path, cjk_max_retries=2)

        tasks = [{
            "chunk_id": "ch_test_chunk000",
            "text_content": "English source",
            "quality_retries": 1,  # < cjk_max_retries=2
        }]
        judge_result = {
            "decision": "REJECT",
            "reject_reason": "译文 CJK 比例过低: cjk_ratio=0.05 < 0.50",
        }
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(tasks)

        # 应继续 retry (更新 state 为 REWRITING_LITERARY, 而非 PERMANENTLY_FAILED)
        # 实际调用: update_task_state(...) for task to REWRITING_LITERARY (after judge)
        # 应至少有 1 次 update_task_state 但不是 PERMANENTLY_FAILED via CJK
        call_args_list = pipeline.scheduler.update_task_state.call_args_list
        for call in call_args_list:
            err = call[1].get("error_msg", "")
            assert "[CJK_RETRY_EXHAUSTED]" not in err

    def test_non_cjk_reject_does_not_trigger_early_stop(self, tmp_path):
        """非 CJK 拒绝 → 不触发 CJK 早停."""
        pipeline, TaskState = _make_pipeline_stub(tmp_path, cjk_max_retries=2)

        tasks = [{
            "chunk_id": "ch_test_chunk000",
            "text_content": "English source",
            "quality_retries": 3,  # >= cjk_max_retries
        }]
        # 拒绝原因是 "实体一致性违反", 不是 CJK
        judge_result = {
            "decision": "REJECT",
            "reject_reason": "实体一致性违反: noul=0.85 eff_conf=0.85",
        }
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(tasks)

        # 应走 max_retries 流程 (不是 CJK 早停)
        call_args_list = pipeline.scheduler.update_task_state.call_args_list
        for call in call_args_list:
            err = call[1].get("error_msg", "")
            assert "[CJK_RETRY_EXHAUSTED]" not in err

    def test_cjk_max_retries_zero_disables_early_stop(self, tmp_path):
        """cjk_max_retries=0 → 禁用早停."""
        pipeline, TaskState = _make_pipeline_stub(tmp_path, cjk_max_retries=0)

        tasks = [{
            "chunk_id": "ch_test_chunk000",
            "text_content": "English source",
            "quality_retries": 3,  # 即使 retries 很高也不触发
        }]
        judge_result = {
            "decision": "REJECT",
            "reject_reason": "译文 CJK 比例过低: cjk_ratio=0.05",
        }
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(tasks)

        # 不应触发 CJK 早停
        call_args_list = pipeline.scheduler.update_task_state.call_args_list
        for call in call_args_list:
            err = call[1].get("error_msg", "")
            assert "[CJK_RETRY_EXHAUSTED]" not in err

    def test_early_stop_includes_llm_savings_metadata(self, tmp_path):
        """早停时 metadata 应包含节省的 LLM 调用次数信息."""
        pipeline, TaskState = _make_pipeline_stub(tmp_path, cjk_max_retries=2)

        tasks = [{
            "chunk_id": "ch_test_chunk000",
            "text_content": "English source",
            "quality_retries": 2,
        }]
        judge_result = {
            "decision": "REJECT",
            "reject_reason": "译文 CJK 比例过低",
        }
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(tasks)

        final = json.loads((tmp_path / "ch_test_chunk000_final.json").read_text(encoding="utf-8"))
        metadata = final["metadata"]
        # 关键字段
        assert metadata["quality_retries"] == 2
        assert metadata["cjk_max_retries"] == 2
        # 计算节省: max_quality_retries(4) - quality_retries(2) = 2
        # 每 retry 节省 1 次 rewriter LLM 调用
        assert "节省" in pipeline.scheduler.update_task_state.call_args[1]["error_msg"]


class TestCjkMaxRetriesConfig:
    """config.yaml cjk_max_retries 配置."""

    def test_config_has_cjk_max_retries(self):
        """config.yaml 包含 cjk_max_retries."""
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "cjk_max_retries" in cfg["pipeline"]
        assert isinstance(cfg["pipeline"]["cjk_max_retries"], int)
        assert cfg["pipeline"]["cjk_max_retries"] >= 1


class TestIntegratedScriptSyncR97:
    """整合脚本同步 Round 97."""

    def test_cjk_retry_exhausted_marker_in_integrated(self):
        """整合脚本 _process_judging_batch 含 CJK_RETRY_EXHAUSTED 标记."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._process_judging_batch)
        assert "CJK_RETRY_EXHAUSTED" in source
        assert "cjk_max_retries" in source
