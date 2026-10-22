"""Round 99 P3: audit 阶段 early_stop — 首轮评分低直接 fallback, 跳过 judge.

设计动机:
  原 early_stop 仅在 _process_judging_batch 中触发, 即先 judge REJECT 后才检查.
  但若首轮 critic 评分已明显很差 (very_low_count >= max_low_dims), judge LLM 必 REJECT,
  rewriter 也不可能在 1 retry 内修复极差译文. Round 99 P3 把 early_stop 提前到
  _process_auditing_batch, 在 quality_retries == 0 时若评分过差, 直接 fallback 跳过
  judge + rewriter cycle, 节省 1 次 judge LLM 调用.

行为:
  - quality_retries == 0 (首轮)
  - very_low_count (评分 < threshold 的维度数) >= max_low_dims
  - → 保存 final.json (用 _get_best_lit 当前最佳)
  - state = PERMANENTLY_FAILED
  - error_msg 含 [AUDIT_EARLY_STOP] 或 stop_reason 描述

测试覆盖:
  - quality_retries=0 + 多维度低分 → 触发
  - quality_retries=0 + 仅少数低分 → 不触发
  - quality_retries>=1 + 多维度低分 → 不触发 (让 judge 处理)
  - early_stop.enabled=false → 永不触发
  - final.json metadata 含 reason=AUDIT_EARLY_STOP
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_auditing_stub(output_dir, *, early_stop_enabled=True, early_stop_threshold=3.0, early_stop_max_low=3, scores=None):
    """构造 _process_auditing_batch stub.

    scores: dict, 模拟 critic 返回的 scores (默认 None 会自动生成测试数据).
    """
    import src.config
    from src.translator_agent import TranslationPipeline

    if scores is None:
        scores = {"voice_consistency": 2.0, "literal_residue": 2.5, "style_compliance": 2.0}

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock()
    pipeline._entity_violation_history = {}
    pipeline._entity_registry_cache = {}
    pipeline._entity_variants_cache = {}
    pipeline._intermediate_cache = {}
    pipeline._score_history = {}
    pipeline._best_lit = {}
    pipeline.style_guide = "中文文学风格"
    pipeline._llm_call_counter = {
        "reference_extraction": 0,
        "literal_translation": 0,
        "literary_rewriting": 0,
        "critic_scoring": 0,
        "judge_decision": 0,
    }

    pipeline.decision_engine = MagicMock()
    pipeline.decision_engine.conn = MagicMock()
    pipeline._build_entity_registry_for_rewriter = lambda src_text: {}
    pipeline._build_entity_variants_for_critic = lambda src_text: {}
    pipeline._check_entity_consistency = lambda cid, lit, src: []  # 无违规

    pipeline.critic_agent = MagicMock()
    pipeline.critic_agent.process_chunk = MagicMock(
        return_value={"scores": scores, "entity_consistency_notes": []}
    )

    # 写 raw.json + literary.json
    (output_dir / "ch_test_chunk000_raw.json").write_text(
        json.dumps({"text": "raw 中文"}, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ch_test_chunk000_literary.json").write_text(
        json.dumps({"text": "这是润色稿中文内容"}, ensure_ascii=False), encoding="utf-8"
    )

    pipeline._save_intermediate = lambda cid, step, payload: (
        (output_dir / f"{cid}_{step}.json").write_text(
            json.dumps(payload if isinstance(payload, dict) else {"text": payload},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )
    pipeline._load_intermediate = lambda cid, step: json.loads(
        (output_dir / f"{cid}_{step}.json").read_text(encoding="utf-8")
    ).get("text", "")
    pipeline._update_best_lit = lambda cid, lit, rep: pipeline._best_lit.update({cid: (5.0, lit)})
    pipeline._compute_avg_score = lambda rep: sum(rep["scores"].values()) / len(rep["scores"]) if rep.get("scores") else 0.0
    pipeline._get_best_lit = lambda cid, fallback: pipeline._best_lit.get(cid, (0, fallback))[1]
    pipeline._save_final_with_sanity = lambda cid, payload, source_step="literary": (
        (output_dir / f"{cid}_final.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )

    src.config.config.pipeline["early_stop"] = {
        "enabled": early_stop_enabled,
        "low_score_threshold": early_stop_threshold,
        "max_low_dims": early_stop_max_low,
        "apply_only_first_round": True,
    }

    return pipeline


def _build_task(quality_retries=0):
    return [{
        "chunk_id": "ch_test_chunk000",
        "text_content": "English source",
        "quality_retries": quality_retries,
    }]


class TestAuditPhaseEarlyStop:
    """Round 99 P3: _process_auditing_batch early_stop 触发."""

    def test_first_audit_low_scores_triggers_fallback(self, tmp_path):
        """quality_retries=0 + 3+ 个维度 < 3.0 → 触发 AUDIT_EARLY_STOP."""
        scores = {"voice_consistency": 2.0, "literal_residue": 2.5, "style_compliance": 2.0}
        pipeline = _make_auditing_stub(
            tmp_path,
            early_stop_threshold=3.0,
            early_stop_max_low=3,
            scores=scores,
        )

        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        # 触发后应标 PERMANENTLY_FAILED
        all_calls = pipeline.scheduler.update_task_state.call_args_list
        es_calls = [
            c for c in all_calls
            if c[0][1].value == "PERMANENTLY_FAILED"
            and "AUDIT_EARLY_STOP" in c[1].get("error_msg", "")
        ]
        assert len(es_calls) >= 1, f"Expected AUDIT_EARLY_STOP, got {all_calls}"

        # final.json 应保存
        final = json.loads((tmp_path / "ch_test_chunk000_final.json").read_text(encoding="utf-8"))
        assert final["metadata"]["reason"] == "AUDIT_EARLY_STOP"
        assert final["metadata"]["fallback"] is True

    def test_only_few_low_scores_no_trigger(self, tmp_path):
        """仅少数维度低分 → 不触发 (e.g., 1 个 < 3.0, 其他 ≥ 5.0)."""
        scores = {"voice_consistency": 2.0, "literal_residue": 7.0, "style_compliance": 8.0}
        pipeline = _make_auditing_stub(
            tmp_path,
            early_stop_threshold=3.0,
            early_stop_max_low=3,
            scores=scores,
        )

        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_EARLY_STOP" not in err

    def test_quality_retries_gt_zero_no_trigger(self, tmp_path):
        """quality_retries>=1 不触发 (后续 audit 由 NO_PROGRESS / score decline 等处理)."""
        scores = {"voice_consistency": 2.0, "literal_residue": 2.5, "style_compliance": 2.0}
        pipeline = _make_auditing_stub(
            tmp_path,
            early_stop_threshold=3.0,
            early_stop_max_low=3,
            scores=scores,
        )

        pipeline._process_auditing_batch(_build_task(quality_retries=2))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_EARLY_STOP" not in err

    def test_early_stop_disabled_no_trigger(self, tmp_path):
        """early_stop.enabled=false → 不触发."""
        scores = {"voice_consistency": 2.0, "literal_residue": 2.5, "style_compliance": 2.0}
        pipeline = _make_auditing_stub(
            tmp_path,
            early_stop_enabled=False,
            early_stop_threshold=3.0,
            early_stop_max_low=3,
            scores=scores,
        )

        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_EARLY_STOP" not in err


class TestIntegratedScriptSyncR99P3:
    """整合脚本同步."""

    def test_audit_early_stop_marker_in_integrated(self):
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._process_auditing_batch)
        assert "AUDIT_EARLY_STOP" in source
