"""Round 116: 停滞换模型 — 生成端多样性破停滞。

现象（1013 全量运行，6 chunk PERMANENTLY_FAILED）：
  rewriter 在 retry 时输出与 prev_lit_text 逐字节完全相同（REWRITER_STAGNATED），
  同一模型对同一 prompt 打转，继续重试只是烧 token。
修复：首次字节停滞不直接 fallback，而是切换到 alternate_role（DeepSeek ↔ MiniMax-M3）
  再试一轮；上限 max_switches 次后仍走原 fallback 路径。换模型不占 quality_retries 计数。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def switch_cfg():
    """启用换模型 + 关闭 oscillation 早停；测试后恢复全局 config。"""
    import src.config
    old_osc = src.config.config.pipeline.get("oscillation_detection")
    old_sw = src.config.config.pipeline.get("stagnation_model_switch")
    src.config.config.pipeline["oscillation_detection"] = {
        "enabled": False, "stagnation_detection": {"enabled": False},
    }
    src.config.config.pipeline["stagnation_model_switch"] = {
        "enabled": True, "alternate_role": "primary",
        "alternate_effort": "max", "max_switches": 1,
    }
    yield
    if old_osc is not None:
        src.config.config.pipeline["oscillation_detection"] = old_osc
    if old_sw is not None:
        src.config.config.pipeline["stagnation_model_switch"] = old_sw


@pytest.fixture
def no_switch_cfg():
    import src.config
    old_sw = src.config.config.pipeline.get("stagnation_model_switch")
    src.config.config.pipeline["stagnation_model_switch"] = {"enabled": False}
    yield
    if old_sw is not None:
        src.config.config.pipeline["stagnation_model_switch"] = old_sw


@pytest.fixture
def switch_and_stagnation_cfg():
    """启用换模型 + 启用 reject-category 停滞检测（触发 should_force_fallback 分支）。"""
    import src.config
    old_osc = src.config.config.pipeline.get("oscillation_detection")
    old_sw = src.config.config.pipeline.get("stagnation_model_switch")
    src.config.config.pipeline["oscillation_detection"] = {
        "enabled": False,
        "stagnation_detection": {"enabled": True, "consecutive_required": 1, "match_mode": "category"},
    }
    src.config.config.pipeline["stagnation_model_switch"] = {
        "enabled": True, "alternate_role": "primary", "max_switches": 1,
    }
    yield
    if old_osc is not None:
        src.config.config.pipeline["oscillation_detection"] = old_osc
    if old_sw is not None:
        src.config.config.pipeline["stagnation_model_switch"] = old_sw


def _make_pipeline(output_dir, lit_text, rewriter_outputs):
    """构建 __new__ 实例 + 内存中间文件存储；rewriter 按顺序返回预设输出。"""
    from src.translator_agent import TranslationPipeline

    p = TranslationPipeline.__new__(TranslationPipeline)
    p.output_dir = output_dir
    p.chapter_id = "ch_test"
    p.scheduler = MagicMock()
    p.rewriter_agent = MagicMock()
    p.rewriter_agent.process_chunk = MagicMock(side_effect=[(t, None) for t in rewriter_outputs])
    p._llm_call_counter = {
        "reference_extraction": 0, "literal_translation": 0,
        "literary_rewriting": 0, "critic_scoring": 0, "judge_decision": 0,
    }
    p._score_history = {}
    p._entity_violation_history = {}
    p._best_lit = {}
    p.style_guide = ""

    store = {}
    p._load_intermediate = lambda cid, step: store[(cid, step)]
    p._save_intermediate = lambda cid, step, data: store.__setitem__((cid, step), data)
    p._save_final_with_sanity = MagicMock()
    p._build_entity_registry_for_rewriter = lambda src: {}
    p._load_prior_memory = lambda cid: ""
    p._compute_text_similarity = lambda a, b: 1.0
    p._get_best_lit = lambda cid, fallback: fallback

    store[("ch_test_chunk000", "literary")] = lit_text
    store[("ch_test_chunk000", "raw")] = "直译底稿中文内容"
    return p, store


def _tasks():
    return [{
        "chunk_id": "ch_test_chunk000",
        "text_content": "George Lomax arrived.",
        "quality_retries": 1,
        "last_error": json.dumps({"judge_reason": "语义丢失: 1 处"}, ensure_ascii=False),
    }]


class TestStagnationModelSwitch:
    def test_first_stagnation_switches_instead_of_failing(self, tmp_path, switch_cfg):
        lit = "这是上一版文学润色稿的完整内容。"
        p, store = _make_pipeline(tmp_path, lit, [lit, lit])

        p._process_rewriting_batch(_tasks())

        sidecar = store.get(("ch_test_chunk000", "rewrite_switch"))
        assert sidecar is not None, "首次字节停滞应写 rewrite_switch sidecar"
        sw = sidecar["switches"]
        assert len(sw) == 1 and sw[0]["to"] == "primary"
        p.scheduler.update_task_state.assert_not_called()
        import src.config as _c
        base_role = _c.config.get_task_role("literary_rewrite")  # Round 129: secondary(DeepSeek)
        meta = store.get(("ch_test_chunk000", "rewrite_meta_v1"))
        assert meta["model_role"] == base_role
        assert meta["model_switch_to"] == "primary"
        assert p.rewriter_agent.process_chunk.call_args.kwargs["model_role"] == base_role
        assert p.rewriter_agent.process_chunk.call_args.kwargs.get("model_effort") is None, \
            "未换模型时不加档（沿用任务默认 effort）"

    def test_second_attempt_uses_alternate_then_fails(self, tmp_path, switch_cfg):
        lit = "这是上一版文学润色稿的完整内容。"
        p, store = _make_pipeline(tmp_path, lit, [lit, lit])

        p._process_rewriting_batch(_tasks())
        p._process_rewriting_batch(_tasks())

        assert p.rewriter_agent.process_chunk.call_args.kwargs["model_role"] == "primary"
        assert p.rewriter_agent.process_chunk.call_args.kwargs["model_effort"] == "max", \
            "Round 129: 换到 MiniMax 救稿时加到最高档"
        from src.core.scheduler import TaskState
        p.scheduler.update_task_state.assert_any_call(
            "ch_test_chunk000", TaskState.PERMANENTLY_FAILED,
            error_msg="rewriter 字节停滞 retry=1",
        )

    def test_switch_disabled_keeps_old_behavior(self, tmp_path, no_switch_cfg):
        lit = "这是上一版文学润色稿的完整内容。"
        p, store = _make_pipeline(tmp_path, lit, [lit])

        p._process_rewriting_batch(_tasks())

        assert store.get(("ch_test_chunk000", "rewrite_switch")) is None
        from src.core.scheduler import TaskState
        p.scheduler.update_task_state.assert_any_call(
            "ch_test_chunk000", TaskState.PERMANENTLY_FAILED,
            error_msg="rewriter 字节停滞 retry=1",
        )

    def test_force_fallback_branch_switches_first(self, tmp_path, switch_and_stagnation_cfg):
        """连续 reject-category 停滞（should_force_fallback）也应先换模型。"""
        lit = "这是上一版文学润色稿的完整内容。"
        p, store = _make_pipeline(tmp_path, lit, ["换模型候选稿：内容不同但仍有问题。"])
        # 上一轮 meta：与当前 reject category 相同 → 触发 is_stagnation
        store[("ch_test_chunk000", "rewrite_meta_v1")] = {
            "reject_reason_summary": "实体一致性硬失败: 1 项译名不一致",
        }
        tasks = _tasks()
        tasks[0]["quality_retries"] = 2
        tasks[0]["last_error"] = json.dumps(
            {"judge_reason": "实体一致性硬失败: 2 项译名不一致"}, ensure_ascii=False
        )

        p._process_rewriting_batch(tasks)

        sidecar = store.get(("ch_test_chunk000", "rewrite_switch"))
        assert sidecar is not None and sidecar["switches"][0]["to"] == "primary"
        p.scheduler.update_task_state.assert_not_called()
        p._save_final_with_sanity.assert_not_called()


class TestSwitchStateReader:
    def _reader_pipeline(self, tmp_path):
        from src.translator_agent import TranslationPipeline
        p = TranslationPipeline.__new__(TranslationPipeline)
        p.output_dir = tmp_path
        store = {}
        p._load_intermediate = lambda cid, step: store[(cid, step)]
        p._save_intermediate = lambda cid, step, data: store.__setitem__((cid, step), data)
        return p, store

    def test_alternates_and_caps(self, tmp_path):
        p, _ = self._reader_pipeline(tmp_path)
        st = p._read_rewriter_switch_state("c1", "literal_translator", "primary", 2)
        assert st == {"current_role": "literal_translator", "switches": 0, "can_switch": True, "next_role": "primary"}

        p._record_rewriter_model_switch("c1", "literal_translator", "primary", 1, "stagnation")
        st = p._read_rewriter_switch_state("c1", "literal_translator", "primary", 2)
        assert st["current_role"] == "primary" and st["can_switch"] is True
        assert st["next_role"] == "literal_translator"

        p._record_rewriter_model_switch("c1", "primary", "literal_translator", 2, "stagnation")
        st = p._read_rewriter_switch_state("c1", "literal_translator", "primary", 2)
        assert st["current_role"] == "literal_translator" and st["switches"] == 2 and st["can_switch"] is False

    def test_max_switches_one_caps_after_first(self, tmp_path):
        p, _ = self._reader_pipeline(tmp_path)
        p._record_rewriter_model_switch("c2", "literal_translator", "primary", 1, "r")
        st = p._read_rewriter_switch_state("c2", "literal_translator", "primary", 1)
        assert st["can_switch"] is False
