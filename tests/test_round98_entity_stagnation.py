"""Round 98 P1: 实体违规连续 N 次未修复早停.

设计动机:
  ch32 实测: 实体违规在每次 retry 仍有 1-2 个 serious (variant_instead_of_canonical),
  rewriter 反复修复但失败. 在 quality_retries=2 时已有足够信号表明 rewriter
  无法修复 → 早停 fallback, 节省 2-4 次 LLM 调用.

触发条件:
  - entity_stagnation_threshold > 0 (config 配置, 默认 2)
  - 最近 N 次 audit 每次都有 ≥1 个 serious 实体违规
  - quality_retries >= entity_stagnation_threshold

行为:
  - 保存 final.json (用 _get_best_lit 当前最佳版)
  - state = PERMANENTLY_FAILED
  - error_msg 含 [ENTITY_VIOLATION_STREAK] + counts
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_auditing_stub(output_dir, *, entity_stagnation_threshold=2, serious_per_round=None):
    """构造 _process_auditing_batch 可调用的 stub.

    serious_per_round: list of int, 每次 audit 调用注入的 serious 实体违规数.
    """
    import src.config
    from src.translator_agent import TranslationPipeline

    if serious_per_round is None:
        serious_per_round = [1, 1]

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock()
    pipeline._entity_violation_history = {}
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

    # decision_engine stub
    pipeline.decision_engine = MagicMock()
    pipeline.decision_engine.conn = MagicMock()
    pipeline._build_entity_registry_for_rewriter = lambda src_text: {}
    pipeline._build_entity_variants_for_critic = lambda src_text: {}

    # critic_agent stub
    pipeline.critic_agent = MagicMock()
    pipeline.critic_agent.process_chunk = MagicMock(
        return_value={
            "scores": {"voice_consistency": 6.0, "literal_residue": 6.0},
            "entity_consistency_notes": [],
        }
    )

    # 写 raw.json + literary.json
    (output_dir / "ch_test_chunk000_raw.json").write_text(
        json.dumps({"text": "这是直译稿中文内容"}, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ch_test_chunk000_literary.json").write_text(
        json.dumps({"text": "这是润色稿中文内容"}, ensure_ascii=False), encoding="utf-8"
    )

    # entity_notes 构造: 每次 audit 返回对应数量的 serious 违规 (按调用顺序消费)
    entity_notes_list = []
    for serious_n in serious_per_round:
        notes = [
            {
                "entity": "测试实体",
                "expected": "规范主名",
                "found": f"违规{i}",
                "source_name": f"src{i}",
                "type": "variant_instead_of_canonical",
                "chunk_id": "ch_test_chunk000",
            }
            for i in range(serious_n)
        ]
        entity_notes_list.append(notes)

    pipeline._check_entity_consistency = lambda cid, lit, src: (
        entity_notes_list.pop(0) if entity_notes_list else []
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
    pipeline._update_best_lit = lambda cid, lit, rep: None
    pipeline._compute_avg_score = lambda rep: 6.0
    pipeline._get_best_lit = lambda cid, fallback: fallback
    pipeline._save_final_with_sanity = lambda cid, payload, source_step="literary": (
        (output_dir / f"{cid}_final.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )

    src.config.config.pipeline["entity_stagnation_threshold"] = entity_stagnation_threshold
    src.config.config.pipeline["entity_hard_fail"] = True
    src.config.config.pipeline["entity_hard_fail_degradation_threshold"] = 3

    return pipeline


def _build_task(quality_retries=0):
    return [{
        "chunk_id": "ch_test_chunk000",
        "text_content": "English source",
        "quality_retries": quality_retries,
    }]


class TestEntityStagnation:
    """Round 98 P1: _process_auditing_batch 实体违规连续早停."""

    def test_entity_stagnation_triggers_at_threshold(self, tmp_path):
        """连续 2 次 serious=1 + quality_retries>=2 → 触发早停."""
        # serious_per_round 提供 3 份 notes (对应 3 次 audit 调用), 前 2 份都是 1 个违规
        pipeline = _make_auditing_stub(tmp_path, entity_stagnation_threshold=2, serious_per_round=[1, 1, 1])

        # 模拟 3 次 audit: quality_retries 0 → 1 → 2
        pipeline._process_auditing_batch(_build_task(quality_retries=0))  # audit 1, hist=[1]
        pipeline._process_auditing_batch(_build_task(quality_retries=1))  # audit 2, hist=[1,1]
        pipeline._process_auditing_batch(_build_task(quality_retries=2))  # audit 3: 触发早停

        # 触发后 scheduler.update_task_state 应被调用至少一次 with PERMANENTLY_FAILED
        all_calls = pipeline.scheduler.update_task_state.call_args_list
        early_stop_calls = [
            c for c in all_calls
            if c[0][1].value == "PERMANENTLY_FAILED"
            and "[ENTITY_VIOLATION_STREAK]" in c[1].get("error_msg", "")
        ]
        assert len(early_stop_calls) >= 1, f"Expected ENTITY_VIOLATION_STREAK early-stop call, got {all_calls}"

        call_args = early_stop_calls[-1]
        assert call_args[0][1].value == "PERMANENTLY_FAILED"
        assert "[ENTITY_VIOLATION_STREAK]" in call_args[1]["error_msg"]
        assert "counts=[1, 1]" in call_args[1]["error_msg"]

        # final.json 应被保存
        final = json.loads((tmp_path / "ch_test_chunk000_final.json").read_text(encoding="utf-8"))
        assert final["metadata"]["fallback"] is True
        assert final["metadata"]["reason"] == "ENTITY_VIOLATION_STREAK"
        assert final["metadata"]["entity_stagnation_threshold"] == 2
        assert final["metadata"]["serious_history"] == [1, 1]

    def test_entity_stagnation_zero_disables(self, tmp_path):
        """entity_stagnation_threshold=0 → 禁用早停."""
        pipeline = _make_auditing_stub(tmp_path, entity_stagnation_threshold=0, serious_per_round=[1, 1, 1])

        # 3 次 audit
        pipeline._process_auditing_batch(_build_task(quality_retries=0))
        pipeline._process_auditing_batch(_build_task(quality_retries=1))
        pipeline._process_auditing_batch(_build_task(quality_retries=2))

        # 不应触发 ENTITY_VIOLATION_STREAK
        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for call in all_calls:
            err = call[1].get("error_msg", "")
            assert "[ENTITY_VIOLATION_STREAK]" not in err

    def test_entity_stagnation_zero_serious_breaks_streak(self, tmp_path):
        """连续 audit 但中间夹 0 → 不触发 (上一轮已修复)."""
        # serious_per_round = [1, 0, 1] → 第 2 次 audit serious=0 (修复了), 不触发
        pipeline = _make_auditing_stub(tmp_path, entity_stagnation_threshold=2, serious_per_round=[1, 0, 1])

        pipeline._process_auditing_batch(_build_task(quality_retries=0))
        pipeline._process_auditing_batch(_build_task(quality_retries=1))
        pipeline._process_auditing_batch(_build_task(quality_retries=2))

        # 不应触发 ENTITY_VIOLATION_STREAK (hist=[1, 0, 1], recent=[0, 1] 不全 > 0)
        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for call in all_calls:
            err = call[1].get("error_msg", "")
            assert "[ENTITY_VIOLATION_STREAK]" not in err

    def test_entity_stagnation_requires_quality_retries(self, tmp_path):
        """quality_retries < threshold → 不触发 (第一次违规需给 rewriter 一次机会)."""
        # quality_retries=0 时 audit 1 次, hist=[1], 但 quality_retries=0 < threshold=2, 不触发
        pipeline = _make_auditing_stub(tmp_path, entity_stagnation_threshold=2, serious_per_round=[1])

        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for call in all_calls:
            err = call[1].get("error_msg", "")
            assert "[ENTITY_VIOLATION_STREAK]" not in err


class TestEntityStagnationConfig:
    """config.yaml entity_stagnation_threshold 配置."""

    def test_config_has_entity_stagnation_threshold(self):
        """config.yaml 包含 entity_stagnation_threshold."""
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "entity_stagnation_threshold" in cfg["pipeline"]
        assert isinstance(cfg["pipeline"]["entity_stagnation_threshold"], int)
        assert cfg["pipeline"]["entity_stagnation_threshold"] >= 1


class TestIntegratedScriptSyncR98:
    """整合脚本同步 Round 98."""

    def test_entity_violation_streak_marker_in_integrated(self):
        """整合脚本 _process_auditing_batch 含 ENTITY_VIOLATION_STREAK 标记."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._process_auditing_batch)
        assert "ENTITY_VIOLATION_STREAK" in source
        assert "entity_stagnation_threshold" in source

    def test_entity_violation_history_field_in_init(self):
        """TranslationPipeline.__init__ 初始化 _entity_violation_history 字段."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline.__init__)
        assert "_entity_violation_history" in source
