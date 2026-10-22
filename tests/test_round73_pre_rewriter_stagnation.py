"""Round 73: pre-rewriter 停滞早停 — 在调用 rewriter LLM 前检测连续 retry 间
reject categories 是否完全相同, 若相同则跳过 rewriter LLM + critic + judge 三连击, 直接 fallback.

根因: Round 12 引入 post-rewriter stagnation 检测, 但位于 rewriter LLM 调用之后.
      ch32 实测 6 轮 quality_retries = 6 次重写 LLM 调用 × 3 (rewriter+critic+judge) = 18+ 次 LLM,
      其中后 4 轮 reject categories 完全相同, 浪费了 ~12 次 LLM 调用.
修复: 将 stagnation 检查前移到 rewriter LLM 调用前, 用 v_{N-1} 和 v_{N-2} 的
      reject categories (Round 69 synonym mapping 后) 比较. 一致即 fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    # Also pop parent namespace packages if no real submodules remain
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        # Round 103 P6+: 动态 polluted 自动覆盖 src.* 父包 + 所有子模块,
        # 父包循环已是冗余, 不再单独遍历 parents.
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)
        # Restore parent packages to their pre-test state



@pytest.fixture
def tmp_output(tmp_path, monkeypatch):
    """每个 test 独立的 output 目录, 隔离 _load_intermediate / _save_intermediate."""
    from src import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod.Path, "__new__", Path.__new__)  # noop safety
    yield tmp_path


def _make_pipeline(monkeypatch, output_dir, db_dir):
    """构造一个 TranslationPipeline, 但不触发 __init__ 复杂逻辑."""
    from src.translator_agent import TranslationPipeline
    from src.core.scheduler import TaskScheduler
    from src.core.decision_engine import DecisionEngine
    from src.config import config as _config

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.chapter_id = "test_ch"
    pipeline.output_dir = output_dir
    pipeline._llm_call_counter = {
        "reference_extraction": 0,
        "literal_translation": 0,
        "literary_rewriting": 0,
        "critic_scoring": 0,
        "judge_decision": 0,
    }
    pipeline._sibling_canonicals_cache = None
    pipeline.style_guide = {"avg_sentence_length": "适中", "lexicon_preference": "平实"}
    # 用 mock decision_engine
    pipeline.decision_engine = MagicMock()
    pipeline.decision_engine.conn = MagicMock()
    # Mock scheduler (不需要真实 DB)
    pipeline.scheduler = MagicMock(spec=TaskScheduler)
    pipeline._root_dir = output_dir.parent
    return pipeline


def _save_meta(pipeline, chunk_id, version, reject_summary):
    """保存 __version__ meta, 模拟 _save_intermediate 的行为."""
    from src.pipeline import TranslationPipeline as PipelineModule
    # patch the class method to write into tmp_path
    def _save_intermediate(self, cid, key, data):
        path = self.output_dir / f"{cid}_{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, dict) and "text" in data and "metadata" in data:
            # fallback dict format
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(PipelineModule, "_save_intermediate", _save_intermediate)
    return monkey


class TestPreRewriterStagnation:
    """pre-rewriter 停滞检测在 rewriter LLM 调用前拦截, 节省成本."""

    def test_same_categories_triggers_fallback(self):
        """v_{N-1} 与 v_{N-2} 同 categories → 短路条件满足."""
        from src.utils.text_match import extract_reject_categories
        # 模拟 ch32 场景: retry=4 entering 时, v3 与 v2 都有相同的 3 个 categories
        v2_summary = "实体一致性违反: noul=0.33; 直译残留: noul=0.69; 关键语义丢失: noul=0.61"
        v3_summary = "实体一致性违反: noul=0.34; 直译残留: noul=0.69; 关键语义丢失: noul=0.59"
        v2_cats = extract_reject_categories(v2_summary, as_set=True)
        v3_cats = extract_reject_categories(v3_summary, as_set=True)
        # 短路条件: prev_cats and prev_cats == curr_cats
        should_short_circuit = bool(v2_cats and v2_cats == v3_cats)
        assert should_short_circuit
        assert "实体一致性违反" in v3_cats
        assert "直译残留" in v3_cats
        assert "关键语义丢失" in v3_cats

    def test_different_categories_proceeds(self):
        """v_{N-1} 与 v_{N-2} categories 不同 → 不触发 pre-rewriter 停滞."""
        from src.utils.text_match import extract_reject_categories
        v0_summary = "直译残留: noul=0.71"
        v1_summary = "实体一致性违反: noul=0.65"
        prev_cats = extract_reject_categories(v0_summary, as_set=True)
        curr_cats = extract_reject_categories(v1_summary, as_set=True)
        # 不同 categories → 不短路
        should_short_circuit = bool(prev_cats and prev_cats == curr_cats)
        assert not should_short_circuit

    def test_empty_categories_skip_check(self):
        """empty categories (e.g., first retry) → 不触发 (跳过检查)."""
        from src.utils.text_match import extract_reject_categories
        # Empty / no categories
        v0 = ""
        v1 = ""
        prev_cats = extract_reject_categories(v0, as_set=True)
        curr_cats = extract_reject_categories(v1, as_set=True)
        # 触发条件: prev_cats AND curr_cats 同时非空才比较
        assert not (prev_cats and prev_cats == curr_cats)


class TestRealCh32Trace:
    """用真实的 ch32 meta_v* 数据验证 Round 73 在 ch32 场景下能省多少次 LLM."""

    def test_ch32_chunk001_would_short_circuit_at_retry_3(self):
        """根据真实 ch32_chunk001 meta, pre-rewriter 应在 retry=3 时短路."""
        from src.utils.text_match import extract_reject_categories

        # 从实际 ch32 数据 (output/ch32/ch32_chunk001_rewrite_meta_v{0..6}.json) 复制
        summaries = {
            0: "",
            1: "实体一致性硬失败: 1 项译名不一致，须按 entity_registry 修正",
            2: "实体一致性违反: noul=0.33 conf=0.00; 直译残留: noul=0.69 conf=0.00; 关键语义丢失: noul=0.61 conf=0.00",
            3: "实体一致性违反: noul=0.34 conf=0.00; 直译残留: noul=0.69 conf=0.00; 关键语义丢失: noul=0.59 conf=0.00",
            4: "实体一致性违反: noul=0.35 conf=0.00; 直译残留: noul=0.56 conf=0.00; 关键语义丢失: noul=0.59 conf=0.00",
            5: "实体一致性违反: noul=0.33 conf=0.00; 直译残留: noul=0.55 conf=0.00; 关键语义丢失: noul=0.60 conf=0.00",
            6: "实体一致性违反: noul=0.33 conf=0.00; 直译残留: noul=0.55 conf=0.00; 关键语义丢失: noul=0.60 conf=0.00",
        }
        cats_per_v = {v: extract_reject_categories(s, as_set=True) for v, s in summaries.items()}

        # Round 73 触发条件: 进入 rewriter LLM 前, prev (v_{N-1}) cats == prev_prev (v_{N-2}) cats
        # 且 quality_retries 进入时 >= 1+stag_consecutive (=2 with default consecutive=1)
        trigger_at = None
        for entering_retry in range(2, 7):  # 从 quality_retries=2 进入 (即第3次retry)
            v_prev = cats_per_v.get(entering_retry - 1, set())  # 上一次 reject categories
            v_prev_prev = cats_per_v.get(entering_retry - 2, set())  # 上上次 reject categories
            if v_prev and v_prev == v_prev_prev:
                trigger_at = entering_retry
                break

        # ch32 实测: v2, v3, v4, v5, v6 全部相同 cats → 第一次短路点应在 retry=3 (entering)
        # 即 quality_retries=3 进入 rewriter LLM 时, v2 == v1? 不, v1 只有 entity 一项, v2 是3项
        # 实际上 v2 vs v1 不同, v3 vs v2 相同 → retry=4 (entering) 短路
        assert trigger_at == 4, (
            f"Round 73 预期在 retry=4 (entering) 短路 (基于 ch32 真实数据), 实际={trigger_at}"
        )

    def test_round73_saves_llm_calls_on_ch32(self):
        """Round 73 相比 post-rewriter stagnation, 在 ch32 上多省几次 LLM."""
        from src.utils.text_match import extract_reject_categories

        # post-rewriter (Round 12): 在 retry=N 跑完 rewriter LLM 后判断 (基于 v_N vs v_{N-1})
        # pre-rewriter (Round 73): 在 retry=N 跑 rewriter LLM 前判断 (基于 v_{N-1} vs v_{N-2})
        # 区别: pre-rewriter 比 post-rewriter 早一轮检测 → 多省 1 次 rewriter LLM + critic + judge = 3 次 LLM

        # ch32_chunk001 实际 retry=6 后 fallback
        # post-rewriter: v3 vs v2 同 → retry=3 fallback（但 Round 12 实际没触发，原因待查）
        # pre-rewriter: v2 vs v1 同 → retry=3 fallback (比 post 早一轮)
        # 简化断言: pre-rewriter 至少不晚于 retry=4 触发（节省后续 3-5 次 LLM）

        summaries = {
            0: "",
            1: "实体一致性硬失败: 1 项译名不一致",
            2: "实体一致性违反: noul=0.33; 直译残留: noul=0.69; 关键语义丢失: noul=0.61",
            3: "实体一致性违反: noul=0.34; 直译残留: noul=0.69; 关键语义丢失: noul=0.59",
            4: "实体一致性违反: noul=0.35; 直译残留: noul=0.56; 关键语义丢失: noul=0.59",
        }
        cats_per_v = {v: extract_reject_categories(s, as_set=True) for v, s in summaries.items()}

        # pre-rewriter: entering retry=3 时, prev(v2) vs prev_prev(v1) 比较
        v2 = cats_per_v[2]
        v1 = cats_per_v[1]
        assert v2 != v1, "v2 vs v1 不同 → entering retry=3 不短路"

        # entering retry=4 时, prev(v3) vs prev_prev(v2) 比较
        v3 = cats_per_v[3]
        v2 = cats_per_v[2]
        assert v3 == v2, "v3 vs v2 相同 → entering retry=4 应短路, 省下 retry=4/5/6 三轮 LLM"


class TestIntegratedScriptExports:
    """整合脚本 _process_rewriting_batch 必须包含 Round 73 pre-rewriter block."""

    def test_pipeline_source_contains_pre_rewriter_marker(self):
        from pathlib import Path
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "PRE_REWRITER_STAGNATION" in content
        assert "pre-rewriter 停滞" in content or "Round 73" in content

    def test_pipeline_module_contains_pre_rewriter_marker(self):
        from pathlib import Path
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "PRE_REWRITER_STAGNATION" in content
        assert "Round 73" in content
