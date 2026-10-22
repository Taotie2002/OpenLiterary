"""Round 101 P1: audit 高分直通 — 跳过 judge LLM.

设计动机:
  与 R99 P3 AUDIT_EARLY_STOP 互补. R99 P3 抓评分差 → PERMANENTLY_FAILED; R101 P1 抓
  评分好 → COMPLETED. 触发条件: quality_retries == 0 (首轮) AND all_scores ≥
  high_threshold AND 无严重实体违规 AND lit_cjk ≥ final_cjk_threshold.

行为:
  - 保存 final.json (用 lit_text)
  - state = COMPLETED
  - metadata.reason = "AUDIT_HIGH_PASS"
  - 跳过 judge LLM 调用

测试覆盖:
  - 高分+无违规 → 触发
  - 评分有一个低分 → 不触发
  - 有严重实体违规 → 不触发 (即使评分高)
  - quality_retries >= 1 → 不触发 (避免误判 retry 中可能恶化的 chunk)
  - audit_high_pass.enabled=false → 不触发
  - lit_text CJK 不达标 → 不触发
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_audit_stub(output_dir, *, high_pass_enabled=True, high_score_threshold=8.0,
                    scores=None, serious_count=0, lit_text="这是润色稿中文内容。完全中文", quality_retries=0):
    import src.config
    from src.translator_agent import TranslationPipeline

    if scores is None:
        scores = {"voice_consistency": 8.5, "literal_residue": 8.5, "style_compliance": 9.0}

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock()
    pipeline._entity_violation_history = {}
    pipeline._entity_registry_cache = {}
    pipeline._entity_variants_cache = {}
    pipeline._intermediate_cache = {}
    pipeline._narrative_memory_cache = {}
    pipeline._avg_score_cache = {}
    pipeline._score_history = {}
    pipeline._best_lit = {"ch_test_chunk000": (sum(scores.values()) / len(scores), lit_text)}
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

    pipeline.critic_agent = MagicMock()
    pipeline.critic_agent.process_chunk = MagicMock(
        return_value={"scores": scores, "entity_consistency_notes": []}
    )

    # Entity notes stub
    notes_list = [
        {
            "entity": "测试实体",
            "expected": "规范主名",
            "found": f"违规{i}",
            "source_name": f"src{i}",
            "type": "variant_instead_of_canonical",
            "chunk_id": "ch_test_chunk000",
        }
        for i in range(serious_count)
    ]
    pipeline._check_entity_consistency = lambda cid, lit, src: notes_list

    (output_dir / "ch_test_chunk000_raw.json").write_text(
        json.dumps({"text": "raw 中文"}, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ch_test_chunk000_literary.json").write_text(
        json.dumps({"text": lit_text}, ensure_ascii=False), encoding="utf-8"
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
    pipeline._compute_avg_score = lambda rep: sum(rep["scores"].values()) / len(rep["scores"]) if rep.get("scores") else 0.0
    pipeline._get_best_lit = lambda cid, fallback: pipeline._best_lit.get(cid, (0, fallback))[1]
    pipeline._save_final_with_sanity = lambda cid, payload, source_step="literary": (
        (output_dir / f"{cid}_final.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )

    # Configure
    src.config.config.pipeline["audit_high_pass"] = {
        "enabled": high_pass_enabled,
        "high_score_threshold": high_score_threshold,
    }
    src.config.config.pipeline["entity_hard_fail"] = True
    src.config.config.pipeline["final_cjk_threshold"] = 0.5
    src.config.config.pipeline["entity_stagnation_threshold"] = 0  # disable for test isolation
    src.config.config.pipeline["cjk_max_retries"] = 0  # disable
    src.config.config.pipeline["no_progress_window"] = 1  # disable

    return pipeline


def _build_task(quality_retries=0):
    return [{
        "chunk_id": "ch_test_chunk000",
        "text_content": "English source",
        "quality_retries": quality_retries,
    }]


class TestAuditHighPass:
    """Round 101 P1: audit 高分直通."""

    def test_high_scores_no_violation_triggers(self, tmp_path):
        """所有评分高 + 无违规 → 触发."""
        pipeline = _make_audit_stub(
            tmp_path,
            high_score_threshold=8.0,
            scores={"voice_consistency": 8.5, "literal_residue": 8.5, "style_compliance": 9.0},
            serious_count=0,
        )
        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        # 期望: 1 次 update_task_state 标 COMPLETED
        completed_calls = [c for c in all_calls if c[0][1].value == "COMPLETED"]
        assert len(completed_calls) >= 1
        assert "AUDIT_HIGH_PASS" in completed_calls[0][1].get("error_msg", "")

        # final.json 应保存
        final = json.loads((tmp_path / "ch_test_chunk000_final.json").read_text(encoding="utf-8"))
        assert final["metadata"]["reason"] == "AUDIT_HIGH_PASS"
        assert final["metadata"]["audit_short_circuit"] is True

    def test_one_low_score_no_trigger(self, tmp_path):
        """任一维度评分低 → 不触发."""
        pipeline = _make_audit_stub(
            tmp_path,
            high_score_threshold=8.0,
            scores={"voice_consistency": 7.0, "literal_residue": 8.5, "style_compliance": 9.0},  # voice=7.0 < 8.0
            serious_count=0,
        )
        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_HIGH_PASS" not in err

    def test_serious_entity_violation_no_trigger(self, tmp_path):
        """严重实体违规 → 不触发 (即使评分高)."""
        pipeline = _make_audit_stub(
            tmp_path,
            high_score_threshold=8.0,
            scores={"voice_consistency": 9.0, "literal_residue": 9.0, "style_compliance": 9.0},
            serious_count=1,
        )
        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_HIGH_PASS" not in err

    def test_quality_retries_gt_zero_no_trigger(self, tmp_path):
        """quality_retries>=1 → 不触发 (避免误判 retry 中可能恶化的 chunk)."""
        pipeline = _make_audit_stub(
            tmp_path,
            high_score_threshold=8.0,
            scores={"voice_consistency": 9.0, "literal_residue": 9.0, "style_compliance": 9.0},
            serious_count=0,
        )
        pipeline._process_auditing_batch(_build_task(quality_retries=2))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_HIGH_PASS" not in err

    def test_disabled_no_trigger(self, tmp_path):
        """audit_high_pass.enabled=false → 不触发."""
        pipeline = _make_audit_stub(
            tmp_path,
            high_pass_enabled=False,
            high_score_threshold=8.0,
            scores={"voice_consistency": 9.0, "literal_residue": 9.0, "style_compliance": 9.0},
            serious_count=0,
        )
        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_HIGH_PASS" not in err

    def test_low_cjk_no_trigger(self, tmp_path):
        """lit_text CJK 不达标 → 不触发."""
        pipeline = _make_audit_stub(
            tmp_path,
            high_score_threshold=8.0,
            scores={"voice_consistency": 9.0, "literal_residue": 9.0, "style_compliance": 9.0},
            serious_count=0,
            lit_text="This is English text only",  # CJK = 0
        )
        pipeline._process_auditing_batch(_build_task(quality_retries=0))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        for c in all_calls:
            err = c[1].get("error_msg", "")
            assert "AUDIT_HIGH_PASS" not in err


class TestAuditHighPassConfig:
    """config.yaml audit_high_pass 配置."""

    def test_config_has_audit_high_pass(self):
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "audit_high_pass" in cfg["pipeline"]
        assert "enabled" in cfg["pipeline"]["audit_high_pass"]
        assert "high_score_threshold" in cfg["pipeline"]["audit_high_pass"]


class TestIntegratedScriptSyncR101:
    """整合脚本同步."""

    def test_audit_high_pass_marker_in_integrated(self):
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._process_auditing_batch)
        assert "AUDIT_HIGH_PASS" in source
        assert "high_score_threshold" in source
