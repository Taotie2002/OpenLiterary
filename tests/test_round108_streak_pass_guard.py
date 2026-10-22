"""Round 108 P2: streak 早停只适用于 REJECT 路径（ch08 chunk006 根因）。

ch08 chunk006: 首跑 COMPLETED（分数历史 8.157→7.943）→ 回溯重跑 (COMPLETED→DIRTY)
→ 重跑 retry=0 Jev 全部维度通过 (reject=[]) → 但旧分数历史 [8.157, 7.943] + 当前
7.94 拼成「单调不增」→ NO_PROGRESS_STREAK 把通过轮强杀 → PERMANENTLY_FAILED。

修复:
  1. SCORE_DECLINE/NO_PROGRESS streak 检查加 judge_is_pass 守卫 — judge 已 PASS
     时直接定稿，分数趋势不覆盖通过裁决（代码注释本意就是「judge REJECT 后检查」）。
  2. _process_dirty_batch 清空 _score_history/_best_lit — 回溯重跑是新 episode，
     旧历史不应污染 streak 早停（否则重跑仅 1 次 retry 就被旧历史提前 fallback）。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_judging_stub(output_dir, *, no_progress_window=3, score_history=None, current_avg=7.94):
    """构造 _process_judging_batch 可调用的 stub（与 test_round98 同模式）。"""
    import src.config
    from src.translator_agent import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock()
    pipeline._entity_violation_history = {}
    pipeline._score_history = {"ch_test_chunk000": list(score_history or [])}
    pipeline._best_lit = {"ch_test_chunk000": (current_avg, "这是最佳版润色稿")}
    pipeline._llm_call_counter = {
        "reference_extraction": 0,
        "literal_translation": 0,
        "literary_rewriting": 0,
        "critic_scoring": 0,
        "judge_decision": 0,
    }

    (output_dir / "ch_test_chunk000_literary.json").write_text(
        json.dumps({"text": "这是润色稿中文内容"}, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "ch_test_chunk000_critic_report.json").write_text(
        json.dumps({"scores": {"voice_consistency": current_avg}}, ensure_ascii=False), encoding="utf-8"
    )
    pipeline._load_intermediate = lambda cid, step: json.loads(
        (output_dir / f"{cid}_{step}.json").read_text(encoding="utf-8")
    ).get("text", "")
    pipeline._compute_avg_score = lambda rep: current_avg
    pipeline._is_declining_streak = TranslationPipeline._is_declining_streak.__get__(pipeline)
    pipeline._is_no_progress_streak = TranslationPipeline._is_no_progress_streak.__get__(pipeline)
    pipeline._get_best_lit = lambda cid, fallback: pipeline._best_lit.get(cid, (0, fallback))[1]
    pipeline._save_final_with_sanity = lambda cid, payload, source_step="literary": (
        (output_dir / f"{cid}_final.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )
    pipeline._append_chunk_memory = lambda *a, **k: None

    src.config.config.pipeline["no_progress_window"] = no_progress_window
    src.config.config.pipeline["cjk_max_retries"] = 2

    return pipeline


def _build_task(chunk_id="ch_test_chunk000", quality_retries=0):
    return [{
        "chunk_id": chunk_id,
        "text_content": "English source",
        "quality_retries": quality_retries,
    }]


def _state_calls(pipeline):
    return pipeline.scheduler.update_task_state.call_args_list


class TestPassNotKilledByStreak:
    """ch08 chunk006 根因：judge PASS 时 streak 早停不得覆盖通过裁决。"""

    def test_pass_not_killed_by_no_progress_streak(self, tmp_path):
        """历史 [8.157, 7.943] + current 7.94 单调不增，但 judge PASS → 定稿 COMPLETED。"""
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=3,
            score_history=[8.157, 7.943], current_avg=7.94,
        )
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value={
            "decision": "PASS",
            "reject_reason": "",
        })

        pipeline._process_judging_batch(_build_task(quality_retries=0))

        calls = _state_calls(pipeline)
        from src.translator_agent import TaskState
        assert any(c[0][1] == TaskState.COMPLETED for c in calls), f"应 COMPLETED: {calls}"
        assert not any("[NO_PROGRESS_STREAK]" in (c[1].get("error_msg") or "") for c in calls)
        assert not any("[SCORE_DECLINE]" in (c[1].get("error_msg") or "") for c in calls)

    def test_pass_not_killed_by_declining_streak(self, tmp_path):
        """历史 [8.5, 8.2] + current 7.9 严格递减，但 judge PASS → 定稿 COMPLETED。"""
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=3,
            score_history=[8.5, 8.2], current_avg=7.9,
        )
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value={
            "decision": "PASS",
            "reject_reason": "",
        })

        pipeline._process_judging_batch(_build_task(quality_retries=0))

        from src.translator_agent import TaskState
        calls = _state_calls(pipeline)
        assert any(c[0][1] == TaskState.COMPLETED for c in calls), f"应 COMPLETED: {calls}"
        assert not any("[SCORE_DECLINE]" in (c[1].get("error_msg") or "") for c in calls)


class TestRejectStillKilledByStreak:
    """回归守卫：REJECT 路径的 streak 早停行为不变。"""

    def test_reject_still_triggers_no_progress(self, tmp_path):
        # 含持平的历史: 单调不增但非严格递减 → 只触发 NO_PROGRESS (不触发 SCORE_DECLINE)
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=3,
            score_history=[8.157, 7.943, 7.943], current_avg=7.943,
        )
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value={
            "decision": "REJECT",
            "reject_reason": "测试拒因",
        })

        pipeline._process_judging_batch(_build_task(quality_retries=2))

        calls = _state_calls(pipeline)
        assert any("[NO_PROGRESS_STREAK]" in (c[1].get("error_msg") or "") for c in calls), calls

    def test_reject_still_triggers_decline(self, tmp_path):
        # Round 118 P1-4: score_decline_streak 3→4 → 需 4 个严格递减样本才触发.
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=3,
            score_history=[8.6, 8.5, 8.2], current_avg=7.9,
        )
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value={
            "decision": "REJECT",
            "reject_reason": "测试拒因",
        })

        pipeline._process_judging_batch(_build_task(quality_retries=2))

        calls = _state_calls(pipeline)
        assert any("[SCORE_DECLINE]" in (c[1].get("error_msg") or "") for c in calls), calls


class TestDirtyBatchClearsHistory:
    """回溯重跑清空旧分数历史/最佳版本 — 重跑是新 episode。"""

    def test_dirty_batch_clears_score_history_and_best_lit(self, tmp_path):
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.output_dir = tmp_path
        pipeline.scheduler = MagicMock()
        pipeline._score_history = {"ch_test_chunk000": [8.157, 7.943]}
        pipeline._best_lit = {"ch_test_chunk000": (8.157, "旧最佳版")}
        # final.json 存在 → 走备份分支
        (tmp_path / "ch_test_chunk000_final.json").write_text("{}", encoding="utf-8")

        pipeline._process_dirty_batch(_build_task())

        assert pipeline._score_history.get("ch_test_chunk000") in (None, []), pipeline._score_history
        assert "ch_test_chunk000" not in pipeline._best_lit
        # 备份行为不回归
        assert (tmp_path / "ch_test_chunk000_final_backup.json").exists()
        assert not (tmp_path / "ch_test_chunk000_final.json").exists()
