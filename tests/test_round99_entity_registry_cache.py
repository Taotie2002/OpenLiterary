"""Round 99 P1: per-chunk entity_registry / variants 缓存.

设计动机:
  _build_entity_registry_for_rewriter 和 _build_entity_variants_for_critic 每次
  audit/rewriting 都被调用. 同一 chunk 的所有 retry 用同一份 source_text,
  而 entity_registry 在 chapter 内稳定. 缓存避免每次 retry 跑
  query_reviewable_entities SQL + JSON parse.

节省:
  - DB 查询次数: 6 retries → 1 query per chunk (省 ~83%)
  - JSON parse 次数: 同上
  - wall-time: 实测单次 query ~5ms, 6 retries × 2 helpers = 60ms → 10ms (省 50ms/chunk)

测试覆盖:
  - 第一次调用 query DB, 第二次命中缓存
  - 不同 source_text 用不同 cache entries
  - _reset_per_run_cache 清空缓存
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def patched_query(monkeypatch):
    """Patch query_reviewable_entities.

    关键点: src.translator_agent 会在加载时把 src.core.decision_engine 替换为 shim 模块,
    详见 src/translator_agent.py 第 9110-9138 行的 _SRC_SUBMODULES 注入逻辑. 因此必须
    patch shim (即 sys.modules['src.core.decision_engine']) 而不是原始模块.
    """
    import sys
    import src.translator_agent  # 触发 shim 注入

    call_log = []

    def fake_query(conn):
        call_log.append("called")
        return [
            ("规范主名", json.dumps(["Alice"]), json.dumps(["艾丽", "爱丽丝"])),
        ]

    # 关键: patch sys.modules 里的 shim, 不是真实模块
    shim = sys.modules["src.core.decision_engine"]
    monkeypatch.setattr(shim, "query_reviewable_entities", fake_query)
    return call_log


def _make_pipeline_with_cache(output_dir):
    from src.translator_agent import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = output_dir
    pipeline.chapter_id = "ch_test"
    pipeline._entity_registry_cache = {}
    pipeline._entity_variants_cache = {}
    pipeline._entity_violation_history = {}
    pipeline._score_history = {}

    pipeline.decision_engine = MagicMock()
    pipeline.decision_engine.conn = MagicMock()

    return pipeline


class TestEntityRegistryCache:
    """Round 99 P1: per-chunk entity_registry 缓存."""

    def test_first_call_queries_db(self, tmp_path, patched_query):
        """首次调用应查询 DB."""
        pipeline = _make_pipeline_with_cache(tmp_path)
        result = pipeline._build_entity_registry_for_rewriter("Alice went to the market")
        assert len(patched_query) == 1
        assert result == {"Alice": "规范主名"}

    def test_second_call_hits_cache(self, tmp_path, patched_query):
        """第二次同 source_text 应命中缓存."""
        pipeline = _make_pipeline_with_cache(tmp_path)
        pipeline._build_entity_registry_for_rewriter("Alice went to the market")
        assert len(patched_query) == 1

        # 缓存命中: 不再 query
        result2 = pipeline._build_entity_registry_for_rewriter("Alice went to the market")
        assert len(patched_query) == 1
        assert result2 == {"Alice": "规范主名"}

    def test_different_source_text_different_cache(self, tmp_path, patched_query):
        """不同 source_text 用不同 cache entries."""
        pipeline = _make_pipeline_with_cache(tmp_path)
        pipeline._build_entity_registry_for_rewriter("Alice")
        pipeline._build_entity_registry_for_rewriter("Bob")
        assert len(patched_query) == 2

        pipeline._build_entity_registry_for_rewriter("Alice")
        pipeline._build_entity_registry_for_rewriter("Bob")
        assert len(patched_query) == 2  # 都命中, 没新增

    def test_entity_variants_also_cached(self, tmp_path, patched_query):
        """_build_entity_variants_for_critic 同样缓存."""
        pipeline = _make_pipeline_with_cache(tmp_path)
        v1 = pipeline._build_entity_variants_for_critic("Alice went to the market")
        assert len(patched_query) == 1
        assert v1 == {"Alice": ["艾丽", "爱丽丝"]}

        v2 = pipeline._build_entity_variants_for_critic("Alice went to the market")
        assert len(patched_query) == 1  # 缓存命中
        assert v2 == {"Alice": ["艾丽", "爱丽丝"]}

    def test_reset_clears_caches(self, tmp_path, patched_query):
        """_reset_per_run_cache 应清空两个缓存."""
        pipeline = _make_pipeline_with_cache(tmp_path)
        pipeline._build_entity_registry_for_rewriter("Alice")
        pipeline._build_entity_variants_for_critic("Alice")
        assert len(pipeline._entity_registry_cache) == 1
        assert len(pipeline._entity_variants_cache) == 1

        pipeline._reset_per_run_cache()
        assert len(pipeline._entity_registry_cache) == 0
        assert len(pipeline._entity_variants_cache) == 0


class TestEntityCacheInit:
    """TranslationPipeline.__init__ 初始化 cache 字段."""

    def test_init_has_cache_fields(self):
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline.__init__)
        assert "_entity_registry_cache" in source
        assert "_entity_variants_cache" in source
        assert "_reset_per_run_cache" in source

    def test_run_resets_caches(self):
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline.run)
        assert "_reset_per_run_cache" in source


class TestIntegratedScriptSyncR99:
    """整合脚本同步 Round 99."""

    def test_entity_cache_in_integrated(self):
        import src.translator_agent as t
        assert hasattr(t.TranslationPipeline, "_build_entity_registry_for_rewriter")
        assert hasattr(t.TranslationPipeline, "_build_entity_variants_for_critic")
        assert hasattr(t.TranslationPipeline, "_reset_per_run_cache")
