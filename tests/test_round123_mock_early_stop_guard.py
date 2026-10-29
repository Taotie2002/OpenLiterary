"""Round 123: mock_heuristic 假分数不得触发 judging 阶段 EARLY_STOP（ch09 chunk013/044 根因）。

ch09 重跑: chunk013/044 的 critic LLM 连续 429 失败 → critic_report 回退
mock_heuristic, 兜底假分数 readability=2.5/fluency=2.5/literal_residue=0.0
(3 维 < 4.0) → judging 阶段 EARLY_STOP 秒杀成 PERMANENTLY_FAILED; 而同一份
mock 分数在 audit 阶段已被 R106 护栏放行 → 两处门控语义不一致。

修复: judging EARLY_STOP 沿用 audit 同款护栏 —— trust_mock_heuristic=false 且
critic 后端为 mock_heuristic 时不早停, 落回 rewriter 正常轮次 / safety-net。
本测试锁定: mock 不早停 / 真实后端不回归 / 显式 trust 时保持旧行为 / 双落点同步。
"""
from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import pytest


LOW_MOCK_SCORES = {
    "voice_consistency": 9.0, "style_compliance": 5.0,
    "literal_residue": 0.0, "semantic_preservation": 10.0,
    "readability": 2.5, "fluency": 2.5, "entity_consistency": 10.0,
}


def _make_judging_stub(output_dir, *, backend_path, scores, judge_decision, trust_mock=False):
    """构造 _process_judging_batch 可调用的 stub（与 test_round108 同模式）。"""
    import src.config
    from src.translator_agent import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline.scheduler = MagicMock()
    pipeline._entity_violation_history = {}
    pipeline._score_history = {}
    pipeline._best_lit = {}
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
        json.dumps({"scores": scores, "_backend_path": backend_path}, ensure_ascii=False),
        encoding="utf-8",
    )

    def _load(cid, step):
        path = output_dir / f"{cid}_{step}.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if step == "literary":
            return data.get("text", "")
        return data

    pipeline._load_intermediate = _load
    pipeline._compute_avg_score = lambda rep: (
        sum(v for v in rep.get("scores", {}).values() if isinstance(v, (int, float)))
        / max(1, len(rep.get("scores", {})))
    )
    pipeline._is_declining_streak = TranslationPipeline._is_declining_streak.__get__(pipeline)
    pipeline._is_no_progress_streak = TranslationPipeline._is_no_progress_streak.__get__(pipeline)
    pipeline._get_best_lit = lambda cid, fallback: pipeline._best_lit.get(cid, (0, fallback))[1]
    pipeline._save_final_with_sanity = lambda cid, payload, source_step="literary": (
        (output_dir / f"{cid}_final.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    )
    pipeline._append_chunk_memory = lambda *a, **k: None
    pipeline._fallback_to_rewriter = MagicMock()
    pipeline.judge_agent = MagicMock()
    pipeline.judge_agent.process_chunk = MagicMock(return_value={
        "decision": judge_decision,
        "reject_reason": "测试拒因" if judge_decision != "PASS" else "",
    })

    src.config.config.pipeline["no_progress_window"] = 3
    src.config.config.pipeline["cjk_max_retries"] = 2
    src.config.config.pipeline["trust_mock_heuristic"] = trust_mock
    return pipeline


def _build_task(quality_retries=0):
    return [{
        "chunk_id": "ch_test_chunk000",
        "text_content": "English source",
        "quality_retries": quality_retries,
    }]


def _early_stop_calls(pipeline):
    from src.translator_agent import TaskState
    return [
        c for c in pipeline.scheduler.update_task_state.call_args_list
        if c[0][1] == TaskState.PERMANENTLY_FAILED
        and "EARLY_STOP" in (c[1].get("error_msg") or "")
    ]


@pytest.fixture
def restore_flags():
    import src.config
    original = src.config.config.pipeline.get("trust_mock_heuristic", False)
    yield
    src.config.config.pipeline["trust_mock_heuristic"] = original


class TestMockScoresDoNotEarlyStop:
    """ch09 chunk013/044 根因: mock 假分数不得秒杀 chunk。"""

    def test_mock_backend_low_scores_no_early_stop(self, tmp_path, restore_flags):
        pipeline = _make_judging_stub(
            tmp_path, backend_path="mock_heuristic",
            scores=LOW_MOCK_SCORES, judge_decision="REJECT",
        )

        pipeline._process_judging_batch(_build_task(quality_retries=0))

        assert not _early_stop_calls(pipeline), "mock 假分数不应触发 EARLY_STOP"
        assert pipeline._fallback_to_rewriter.called, "应落回 rewriter 正常轮次"
        final_path = tmp_path / "ch_test_chunk000_final.json"
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            assert final.get("metadata", {}).get("reason") != "EARLY_STOP"

    def test_real_backend_low_scores_still_early_stop(self, tmp_path, restore_flags):
        """回归守卫: 真实 critic 后端的低分早停行为不变。"""
        pipeline = _make_judging_stub(
            tmp_path, backend_path="openai_compat",
            scores=LOW_MOCK_SCORES, judge_decision="REJECT",
        )

        pipeline._process_judging_batch(_build_task(quality_retries=0))

        assert _early_stop_calls(pipeline), "真实后端低分仍应触发 EARLY_STOP"
        final = json.loads(
            (tmp_path / "ch_test_chunk000_final.json").read_text(encoding="utf-8")
        )
        assert final["metadata"]["reason"] == "EARLY_STOP"
        assert not pipeline._fallback_to_rewriter.called

    def test_trust_mock_opt_in_keeps_old_behavior(self, tmp_path, restore_flags):
        """显式 trust_mock_heuristic=true 时, mock 分数视作真实 → 保持旧早停行为。"""
        pipeline = _make_judging_stub(
            tmp_path, backend_path="mock_heuristic",
            scores=LOW_MOCK_SCORES, judge_decision="REJECT", trust_mock=True,
        )

        pipeline._process_judging_batch(_build_task(quality_retries=0))

        assert _early_stop_calls(pipeline), "trust_mock 时早停行为应保持"


class TestDualSiteSync:
    """双落点纪律: 模块版与单体版都带 mock 护栏。"""

    def test_both_modules_have_guard(self):
        from src.translator_agent import TranslationPipeline as Monolith
        from src.pipeline import TranslationPipeline as Module

        for cls in (Module, Monolith):
            source = inspect.getsource(cls._process_judging_batch)
            assert "mock_untrusted_judging" in source, cls.__module__
            assert "EARLY_STOP 拦截" in source, cls.__module__

    def test_default_config_untrusted_mock(self):
        from src.config import config as _config
        assert _config.pipeline.get("trust_mock_heuristic", False) is False
