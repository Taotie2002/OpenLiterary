"""Round 98 P2: 非进展 streak — avg score 连续 N 次单调不增早停.

设计动机:
  _is_declining_streak 只抓"严格递减" (s[i] > s[i+1]), 但 ch32 真实场景中更多是
  "持平+抖动" — rewriter 修复了一个问题但又引入另一个, 综合 avg score 不增不减.
  Round 98 P2 加 _is_no_progress_streak, 单调不增 (s[i] >= s[i+1], 含相等) 即触发,
  比 SCORE_DECLINE 更激进.

触发条件:
  - no_progress_window > 1 (config 配置, 默认 3)
  - 最近 window 个样本 (history 末尾 window-1 + current) 任意相邻两次间满足 next <= prev
  - quality_retries >= no_progress_window

行为:
  - 保存 final.json (用 _get_best_lit 当前最佳版)
  - state = PERMANENTLY_FAILED
  - error_msg 含 [NO_PROGRESS_STREAK]
  - metadata.reason = "NO_PROGRESS_STREAK"
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_judging_stub(output_dir, *, no_progress_window=3, score_history=None, current_avg=5.0):
    """构造 _process_judging_batch 可调用的 stub.

    score_history: list of float, 已有的 _score_history (本调用前的累积分数).
    current_avg: 当前 critic_report 的 avg_score.
    """
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

    # stub
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
    pipeline._is_declining_streak = lambda cid, score, consecutive=3: False  # 关闭 SCORE_DECLINE, 只测 NO_PROGRESS
    pipeline._is_no_progress_streak = lambda cid, score, window=3: pipeline._is_no_progress_streak_impl(cid, score, window)
    pipeline._is_no_progress_streak_impl = TranslationPipeline._is_no_progress_streak.__get__(pipeline)
    pipeline._get_best_lit = lambda cid, fallback: pipeline._best_lit.get(cid, (0, fallback))[1]
    pipeline._save_final_with_sanity = lambda cid, payload, source_step="literary": (
        (output_dir / f"{cid}_final.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    )

    src.config.config.pipeline["no_progress_window"] = no_progress_window
    src.config.config.pipeline["cjk_max_retries"] = 2

    return pipeline


def _build_task(chunk_id="ch_test_chunk000", quality_retries=3, current_avg=5.0):
    return [{
        "chunk_id": chunk_id,
        "text_content": "English source",
        "quality_retries": quality_retries,
    }]


class TestNoProgressStreak:
    """Round 98 P2: _process_judging_batch 非进展 streak 早停."""

    def test_no_progress_triggers_on_flat_scores(self, tmp_path):
        """连续 3 次 score 持平 (5.0, 5.0, 5.0) → 触发 NO_PROGRESS (SCORE_DECLINE 不会触发)."""
        # 历史 [5.0, 5.0], current=5.0 → 末尾 3 个样本 [5.0, 5.0, 5.0] 全相等, 单调不增 ✓
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=3,
            score_history=[5.0, 5.0], current_avg=5.0,
        )
        judge_result = {
            "decision": "REJECT",
            "reject_reason": "平均分 5.0 低于通过阈值",
        }
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(_build_task(quality_retries=3))

        # 触发 NO_PROGRESS_STREAK 早停
        all_calls = pipeline.scheduler.update_task_state.call_args_list
        np_calls = [
            c for c in all_calls
            if "[NO_PROGRESS_STREAK]" in c[1].get("error_msg", "")
        ]
        assert len(np_calls) >= 1, f"Expected NO_PROGRESS_STREAK call, got {all_calls}"

        # final.json metadata
        final = json.loads((tmp_path / "ch_test_chunk000_final.json").read_text(encoding="utf-8"))
        assert final["metadata"]["reason"] == "NO_PROGRESS_STREAK"
        assert final["metadata"]["no_progress_window"] == 3

    def test_no_progress_triggers_on_decline_then_flat(self, tmp_path):
        """先递减后持平 → 也触发 (含衰减进入持平)."""
        # 历史 [7.0, 6.0, 5.5], current=5.5 → 末尾 3 个样本 [6.0, 5.5, 5.5], 单调不增 ✓
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=3,
            score_history=[7.0, 6.0, 5.5], current_avg=5.5,
        )
        judge_result = {"decision": "REJECT", "reject_reason": "测试"}
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(_build_task(quality_retries=3))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        assert any("[NO_PROGRESS_STREAK]" in c[1].get("error_msg", "") for c in all_calls)

    def test_no_progress_does_not_trigger_on_improvement(self, tmp_path):
        """score 上升 → 不触发."""
        # 历史 [5.0, 5.5], current=6.0 → 末尾 3 个 [5.5, 6.0] (window=3 但样本不足 → False)
        # 改成 window=2: [5.0, 6.0] → 5.0 <= 6.0? 6.0 > 5.0 → 不单调不增
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=2,
            score_history=[5.0, 5.5], current_avg=6.0,
        )
        judge_result = {"decision": "REJECT", "reject_reason": "测试"}
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(_build_task(quality_retries=2))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        assert not any("[NO_PROGRESS_STREAK]" in c[1].get("error_msg", "") for c in all_calls)

    def test_no_progress_window_zero_disables(self, tmp_path):
        """no_progress_window <= 1 → 禁用."""
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=1,  # 禁用
            score_history=[5.0, 5.0, 5.0], current_avg=5.0,
        )
        judge_result = {"decision": "REJECT", "reject_reason": "测试"}
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(_build_task(quality_retries=3))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        assert not any("[NO_PROGRESS_STREAK]" in c[1].get("error_msg", "") for c in all_calls)

    def test_no_progress_zero_score_filtered(self, tmp_path):
        """历史中有 0.0 (critic 空评分) → 过滤, 不计入窗口."""
        # 历史 [0.0, 5.0, 5.0], current=5.0 → 过滤 0.0 后样本 [5.0, 5.0, 5.0] 全相等 ✓
        pipeline = _make_judging_stub(
            tmp_path, no_progress_window=3,
            score_history=[0.0, 5.0, 5.0], current_avg=5.0,
        )
        judge_result = {"decision": "REJECT", "reject_reason": "测试"}
        pipeline.judge_agent = MagicMock()
        pipeline.judge_agent.process_chunk = MagicMock(return_value=judge_result)

        pipeline._process_judging_batch(_build_task(quality_retries=3))

        all_calls = pipeline.scheduler.update_task_state.call_args_list
        assert any("[NO_PROGRESS_STREAK]" in c[1].get("error_msg", "") for c in all_calls)


class TestNoProgressConfig:
    """config.yaml no_progress_window 配置."""

    def test_config_has_no_progress_window(self):
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "no_progress_window" in cfg["pipeline"]
        assert isinstance(cfg["pipeline"]["no_progress_window"], int)
        assert cfg["pipeline"]["no_progress_window"] >= 1


class TestIntegratedScriptSyncR98P2:
    """整合脚本同步 Round 98 P2."""

    def test_no_progress_streak_marker_in_integrated(self):
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._process_judging_batch)
        assert "NO_PROGRESS_STREAK" in source
        assert "no_progress_window" in source

    def test_is_no_progress_streak_helper_in_integrated(self):
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline)
        assert "_is_no_progress_streak" in source
