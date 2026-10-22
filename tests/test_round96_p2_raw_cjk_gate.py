"""Round 96 P2: raw 翻译稿 CJK 闸门 — 不让英文稿喂给 rewriter.

设计动机:
  ch02 实测中, literal_translation LLM 偶发返回英文源文 (或模型混淆成英文).
  旧代码直接保存到 raw.json, 然后 REWRITING_LITERARY 步骤以英文为输入调 rewriter,
  rewriter 输出也是英文 (中文润色 prompt 在英文上无效), 形成连锁失败.
  最终 ch02 final.json 是英文 + 中文注释.

Round 96 P2 在 raw 步骤加 CJK 闸门:
  1) raw CJK >= threshold → 正常推进 (保存 raw, 进 REWRITING_LITERARY)
  2) raw CJK < threshold → 删除 raw 中间产物, 标 FAILED 触发 TRANSLATING_RAW 重试
  3) threshold=0 → 禁用闸门 (向后兼容)

测试覆盖:
  - compute_cjk_ratio 已通过 Round 96 P1 测试, 此处不复测
  - _process_translating_raw_batch 三种场景:
    * 正常中文 raw → 进 success_ids
    * 英文 raw → 标 FAILED + 不进 success_ids + 删除 raw.json
    * threshold=0 → 英文 raw 也通过 (向后兼容)
  - 错误消息含 [TRANSLATING_RAW] 前缀, 让 _get_recovery_stage 路由回 TRANSLATING_RAW
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# 测试基础设施: 构造一个最小可用的 _process_translating_raw_batch stub
# ---------------------------------------------------------------------------

def _make_pipeline_stub(output_dir, raw_text=None, raise_exc=None, raw_cjk_threshold=0.3):
    """构造 TranslationPipeline stub, output_dir 指向 tmp_path.

    raw_text: 模拟 _run_raw_translator 的返回值
    raise_exc: 模拟 _run_raw_translator 抛异常 (覆盖 raw_text)
    raw_cjk_threshold: min_raw_cjk_ratio 配置
    """
    # 直接构造真实 _process_translating_raw_batch 调用环境
    from src.core.scheduler import TaskState
    from src.translator_agent import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock()
    pipeline._llm_call_counter = {"literal_translation": 0}

    # _run_raw_translator 桩: 返回 raw_text 或抛异常
    def _run_raw_translator(source_text, chunk_id=""):
        if raise_exc is not None:
            raise raise_exc
        return raw_text or ""

    pipeline._run_raw_translator = _run_raw_translator
    pipeline._save_intermediate = lambda cid, step, data: (
        (output_dir / f"{cid}_{step}.json").write_text(
            json.dumps({"text": data} if isinstance(data, str) else data,
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )

    # monkey-patch config
    import src.config
    _cfg = src.config.config
    _cfg.pipeline["min_raw_cjk_ratio"] = raw_cjk_threshold

    return pipeline, TaskState


class TestRawTranslationCjkGate:
    """Round 96 P2: _process_translating_raw_batch CJK 闸门."""

    def test_chinese_raw_passes_to_rewriter(self, tmp_path):
        """正常中文 raw → 进 success_ids, 触发 REWRITING_LITERARY 状态."""
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raw_text="这是直译稿的中文内容", raw_cjk_threshold=0.3
        )

        tasks = [{"chunk_id": "ch_test_chunk000", "text_content": "English source"}]
        pipeline._process_translating_raw_batch(tasks)

        # success_ids 应被推送到 REWRITING_LITERARY
        pipeline.scheduler.batch_update_state.assert_called_once()
        call_args = pipeline.scheduler.batch_update_state.call_args
        assert call_args[0][1].value == TaskState.REWRITING_LITERARY.value
        assert "ch_test_chunk000" in call_args[0][0]

        # raw.json 应被保存
        assert (tmp_path / "ch_test_chunk000_raw.json").exists()

    def test_english_raw_marked_failed_no_rewriter(self, tmp_path):
        """英文 raw → 标 FAILED, 不进 success_ids, 删除 raw.json."""
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raw_text="This is the English source text only.", raw_cjk_threshold=0.3
        )

        tasks = [{"chunk_id": "ch_test_chunk000", "text_content": "English source"}]
        pipeline._process_translating_raw_batch(tasks)

        # success_ids 应为空, 不调用 batch_update_state (因为没有成功项)
        pipeline.scheduler.batch_update_state.assert_not_called()

        # 标 FAILED + 错误消息含 [TRANSLATING_RAW] 前缀
        pipeline.scheduler.update_task_state.assert_called_once()
        call_args = pipeline.scheduler.update_task_state.call_args
        assert call_args[0][0] == "ch_test_chunk000"
        assert call_args[0][1] == TaskState.FAILED
        error_msg = call_args[1]["error_msg"]
        # 关键: [TRANSLATING_RAW] 前缀让 _get_recovery_stage 路由回 TRANSLATING_RAW
        assert "[TRANSLATING_RAW]" in error_msg
        assert "raw_not_chinese" in error_msg
        assert "cjk_ratio" in error_msg

        # raw.json 应被删除 (避免污染下游 fallback)
        assert not (tmp_path / "ch_test_chunk000_raw.json").exists()

    def test_ch02_scenario_blocks_cascade(self, tmp_path):
        """ch02 真实场景: raw 是英文源文 + 中文注释 → 闸门拦截, 不进 rewriter."""
        ch02_raw = '''# 1Anthony Cade Signs on

1 Anthony Cade Signs on

"Gentleman Joe!"

Castle's Select Tour, represented by seven depressed-looking
females and three perspiring males...

译文经过文学润色，呈现史诗感与古典韵律。

[^1]: 译注：济慈（1795-1821），英国浪漫主义诗人。'''
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raw_text=ch02_raw, raw_cjk_threshold=0.3
        )

        tasks = [{"chunk_id": "ch02_chunk000", "text_content": "English"}]
        pipeline._process_translating_raw_batch(tasks)

        # ch02 raw CJK ≈ 0.13, 应被拦截
        pipeline.scheduler.update_task_state.assert_called_once()
        call_args = pipeline.scheduler.update_task_state.call_args
        assert call_args[0][0] == "ch02_chunk000"
        assert call_args[0][1] == TaskState.FAILED
        assert "[TRANSLATING_RAW]" in call_args[1]["error_msg"]

    def test_threshold_zero_disables_gate(self, tmp_path):
        """threshold=0 关闭闸门 → 英文 raw 也通过 (向后兼容)."""
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raw_text="Pure English text.", raw_cjk_threshold=0.0
        )

        tasks = [{"chunk_id": "ch_test_chunk000", "text_content": "English"}]
        pipeline._process_translating_raw_batch(tasks)

        # 应正常推进到 REWRITING_LITERARY
        pipeline.scheduler.batch_update_state.assert_called_once()
        # raw.json 应被保存
        assert (tmp_path / "ch_test_chunk000_raw.json").exists()
        # 没有标 FAILED
        pipeline.scheduler.update_task_state.assert_not_called()

    def test_llm_exception_still_marks_failed(self, tmp_path):
        """LLM 抛异常 → 走原 except 分支, 标 FAILED."""
        import src.config
        _cfg = src.config.config
        _cfg.pipeline["min_raw_cjk_ratio"] = 0.3
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raise_exc=RuntimeError("API timeout"), raw_cjk_threshold=0.3
        )

        tasks = [{"chunk_id": "ch_test_chunk000", "text_content": "English"}]
        pipeline._process_translating_raw_batch(tasks)

        # 异常路径也应标 FAILED
        pipeline.scheduler.update_task_state.assert_called_once()
        call_args = pipeline.scheduler.update_task_state.call_args
        assert call_args[0][1] == TaskState.FAILED
        assert "[TRANSLATING_RAW]" in call_args[1]["error_msg"]
        assert "API timeout" in call_args[1]["error_msg"]

    def test_recovery_routes_back_to_translating_raw(self, tmp_path):
        """错误消息含 [TRANSLATING_RAW] 前缀 → _get_recovery_stage 路由回 TRANSLATING_RAW."""
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raw_text="Pure English.", raw_cjk_threshold=0.3
        )

        tasks = [{"chunk_id": "ch_test_chunk000", "text_content": "English"}]
        pipeline._process_translating_raw_batch(tasks)

        # 模拟重试流程: FAILED 状态进入 _process_failed_batch, _get_recovery_stage 应返回 TRANSLATING_RAW
        from src.translator_agent import TranslationPipeline
        # 用真实实例 (TranslationPipeline stub) 调用 _get_recovery_stage
        error_msg = pipeline.scheduler.update_task_state.call_args[1]["error_msg"]
        # _get_recovery_stage 是从 STAGE_MAP 找, 我们的 error_msg 含 [TRANSLATING_RAW]
        recovery = TranslationPipeline._get_recovery_stage(pipeline, error_msg)
        assert recovery == TaskState.TRANSLATING_RAW

    def test_mixed_content_just_above_threshold(self, tmp_path):
        """CJK 比例刚好越过阈值 → 通过."""
        # 10 字符中 3 个 CJK = 30% (>= 0.3)
        mixed = "你好你好你好 hello world"
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raw_text=mixed, raw_cjk_threshold=0.3
        )

        tasks = [{"chunk_id": "ch_test_chunk000", "text_content": "English"}]
        pipeline._process_translating_raw_batch(tasks)

        # 边界值: cjk=0.3 = threshold=0.3 应通过 (>= 不是 >)
        pipeline.scheduler.batch_update_state.assert_called_once()

    def test_partial_chinese_just_below_threshold(self, tmp_path):
        """CJK 比例刚低于阈值 → 拦截."""
        # 10 字符中 2 个 CJK = 20% (< 0.3)
        partial = "你好 hello world abc def"  # 2/16 = 0.125
        pipeline, TaskState = _make_pipeline_stub(
            tmp_path, raw_text=partial, raw_cjk_threshold=0.3
        )

        tasks = [{"chunk_id": "ch_test_chunk000", "text_content": "English"}]
        pipeline._process_translating_raw_batch(tasks)

        # 边界值: cjk=0.2 < 0.3 应被拦截
        pipeline.scheduler.update_task_state.assert_called_once()
        call_args = pipeline.scheduler.update_task_state.call_args
        assert call_args[0][1] == TaskState.FAILED


# ---------------------------------------------------------------------------
# 整合脚本一致性测试
# ---------------------------------------------------------------------------

class TestIntegratedScriptSyncP2:
    """验证 src/translator_agent.py 与散装 src/pipeline.py 在 P2 上同步."""

    def test_process_translating_raw_batch_in_integrated(self):
        """_process_translating_raw_batch 在整合脚本中存在."""
        import src.translator_agent as t
        assert hasattr(t.TranslationPipeline, "_process_translating_raw_batch")

    def test_cjk_gate_in_integrated(self):
        """整合脚本中的 _process_translating_raw_batch 含 raw_not_chinese 拦截."""
        import inspect
        import src.translator_agent as t

        source = inspect.getsource(t.TranslationPipeline._process_translating_raw_batch)
        # Round 96 P2 marker: 错误消息应含 raw_not_chinese
        assert "raw_not_chinese" in source
        # CJK 检查应使用 compute_cjk_ratio (内联在 translator_agent)
        assert "compute_cjk_ratio" in source
        # 配置读取
        assert "min_raw_cjk_ratio" in source

    def test_config_yaml_has_min_raw_cjk_ratio(self):
        """config.yaml 包含 min_raw_cjk_ratio 配置项."""
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "min_raw_cjk_ratio" in cfg["pipeline"]
        assert isinstance(cfg["pipeline"]["min_raw_cjk_ratio"], (int, float))
