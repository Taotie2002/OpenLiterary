"""Round 105 P1-3: REWRITER_STAGNATED 早停前保存 rewrite_meta。

ch32 倒查时 chunk000/001 的 retry=1 触发 REWRITER_STAGNATED 后直接 continue，
未保存 rewrite_meta_v1.json，导致无法确认 rewriter 当时收到了什么反馈/拒因。
修复：停滞早停分支在 continue 前先保存 rewrite_meta_v{N}。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


def _make_pipeline(output_dir, lit_text):
    import src.config
    from src.translator_agent import TranslationPipeline

    p = TranslationPipeline.__new__(TranslationPipeline)
    p.output_dir = output_dir
    p.chapter_id = "ch_test"
    p.scheduler = MagicMock()
    p.rewriter_agent = MagicMock()
    p._llm_call_counter = {
        "reference_extraction": 0, "literal_translation": 0,
        "literary_rewriting": 0, "critic_scoring": 0, "judge_decision": 0,
    }
    p._score_history = {}
    p._entity_violation_history = {}
    p.style_guide = ""

    # 中间文件存储（内存 dict）
    store = {}
    p._load_intermediate = lambda cid, step: store[(cid, step)]
    def _save(cid, step, data):
        store[(cid, step)] = data
    p._save_intermediate = _save
    p._save_final_with_sanity = MagicMock()
    p._build_entity_registry_for_rewriter = lambda src: {}
    p._load_prior_memory = lambda cid: ""
    p._compute_text_similarity = lambda a, b: 1.0

    # 预置 literary（prev_lit_text）与 raw
    store[("ch_test_chunk000", "literary")] = lit_text
    store[("ch_test_chunk000", "raw")] = "直译底稿中文内容"

    # rewriter 返回与 prev_lit_text 完全相同 → 触发字节停滞
    p.rewriter_agent.process_chunk = MagicMock(return_value=(lit_text, None))

    # config：关闭 oscillation/stagnation 的额外分支，聚焦字节停滞
    src.config.config.pipeline["oscillation_detection"] = {
        "enabled": False, "stagnation_detection": {"enabled": False},
    }
    # Round 116: 本测试聚焦停滞 fallback 的 meta 保存；换模型路径由 test_round116 覆盖
    src.config.config.pipeline["stagnation_model_switch"] = {"enabled": False}
    return p, store


class TestStagnationMetaSave:
    def test_meta_saved_on_stagnation(self, tmp_path):
        lit = "这是上一版文学润色稿的完整内容。"
        p, store = _make_pipeline(tmp_path, lit)

        tasks = [{
            "chunk_id": "ch_test_chunk000",
            "text_content": "George Lomax arrived.",
            "quality_retries": 1,
            "last_error": json.dumps(
                {"judge_reason": "实体一致性硬失败: 1 项译名不一致"},
                ensure_ascii=False,
            ),
        }]
        p._process_rewriting_batch(tasks)

        # 关键断言：停滞轮次的 rewrite_meta 被保存
        meta = store.get(("ch_test_chunk000", "rewrite_meta_v1"))
        assert meta is not None, "REWRITER_STAGNATED 早停前应保存 rewrite_meta_v1"
        assert meta.get("stagnation") == "REWRITER_STAGNATED"
        assert meta.get("quality_retries") == 1
        assert "实体一致性硬失败" in meta.get("reject_reason_summary", "")
        # 且任务被标为 PERMANENTLY_FAILED
        from src.core.scheduler import TaskState
        p.scheduler.update_task_state.assert_any_call(
            "ch_test_chunk000", TaskState.PERMANENTLY_FAILED,
            error_msg="rewriter 字节停滞 retry=1",
        )
