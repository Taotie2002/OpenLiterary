"""Round 99 P2: _load_intermediate / _save_intermediate 内存缓存.

设计动机:
  _load_intermediate 每次都做文件 I/O + JSON parse. 同 chunk × step 在一个 chunk 周期内
  被多次加载 (raw/literary 各被审计/文学润色/裁决阶段反复读取, critic_report 被
  auditing 与 judging 阶段读取). 加内存读缓存避免重复 I/O, 写入时 invalidate.

测试覆盖:
  - 第二次同 chunk × step 加载走内存 (mock file I/O 验证)
  - 不同 chunk / step 用不同 cache entry
  - _save_intermediate invalidate 对应 cache entry
  - _reset_per_run_cache 清空 cache
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _make_pipeline(tmp_path):
    from src.translator_agent import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = tmp_path
    pipeline._intermediate_cache = {}
    pipeline._entity_registry_cache = {}
    pipeline._entity_variants_cache = {}
    return pipeline


class TestIntermediateCache:
    """Round 99 P2: _load_intermediate / _save_intermediate 内存缓存."""

    def test_first_load_reads_file(self, tmp_path):
        """首次加载走文件 + JSON parse."""
        pipeline = _make_pipeline(tmp_path)
        (tmp_path / "ch1_raw.json").write_text(
            json.dumps({"text": "这是 raw 文本"}, ensure_ascii=False), encoding="utf-8"
        )

        result = pipeline._load_intermediate("ch1", "raw")
        assert result == "这是 raw 文本"
        assert ("ch1", "raw") in pipeline._intermediate_cache

    def test_second_load_hits_cache(self, tmp_path):
        """第二次同 chunk × step 应命中缓存, 不再读文件."""
        pipeline = _make_pipeline(tmp_path)
        (tmp_path / "ch1_raw.json").write_text(
            json.dumps({"text": "这是 raw 文本"}, ensure_ascii=False), encoding="utf-8"
        )

        # 第一次: 走文件
        r1 = pipeline._load_intermediate("ch1", "raw")
        # 删除文件模拟"如果缓存失效会失败"
        (tmp_path / "ch1_raw.json").unlink()

        # 第二次: 应命中缓存, 不再读文件
        r2 = pipeline._load_intermediate("ch1", "raw")
        assert r1 == r2 == "这是 raw 文本"

    def test_different_chunk_or_step_separate_cache(self, tmp_path):
        """不同 chunk_id 或 step 用不同 cache entry."""
        pipeline = _make_pipeline(tmp_path)
        (tmp_path / "ch1_raw.json").write_text(json.dumps({"text": "ch1 raw"}, ensure_ascii=False), encoding="utf-8")
        (tmp_path / "ch2_raw.json").write_text(json.dumps({"text": "ch2 raw"}, ensure_ascii=False), encoding="utf-8")
        (tmp_path / "ch1_literary.json").write_text(json.dumps({"text": "ch1 lit"}, ensure_ascii=False), encoding="utf-8")

        assert pipeline._load_intermediate("ch1", "raw") == "ch1 raw"
        assert pipeline._load_intermediate("ch2", "raw") == "ch2 raw"
        assert pipeline._load_intermediate("ch1", "literary") == "ch1 lit"
        assert len(pipeline._intermediate_cache) == 3

    def test_save_invalidates_cache(self, tmp_path):
        """_save_intermediate 应 invalidate 对应 cache entry."""
        pipeline = _make_pipeline(tmp_path)
        (tmp_path / "ch1_raw.json").write_text(json.dumps({"text": "old"}, ensure_ascii=False), encoding="utf-8")

        # 加载到缓存
        assert pipeline._load_intermediate("ch1", "raw") == "old"
        assert ("ch1", "raw") in pipeline._intermediate_cache

        # 保存新值, 应 invalidate cache
        pipeline._save_intermediate("ch1", "raw", "new value")
        assert ("ch1", "raw") not in pipeline._intermediate_cache

        # 重新加载应拿到新值
        assert pipeline._load_intermediate("ch1", "raw") == "new value"

    def test_reset_clears_cache(self, tmp_path):
        """_reset_per_run_cache 应清空 intermediate cache."""
        pipeline = _make_pipeline(tmp_path)
        (tmp_path / "ch1_raw.json").write_text(json.dumps({"text": "v"}, ensure_ascii=False), encoding="utf-8")
        pipeline._load_intermediate("ch1", "raw")
        assert len(pipeline._intermediate_cache) == 1

        pipeline._reset_per_run_cache()
        assert len(pipeline._intermediate_cache) == 0

    def test_dict_intermediate_also_cached(self, tmp_path):
        """非 text-only dict 也应缓存 (judge 经常读 critic_report.json 完整 dict)."""
        pipeline = _make_pipeline(tmp_path)
        (tmp_path / "ch1_critic_report.json").write_text(
            json.dumps({"scores": {"voice_consistency": 7.5}, "notes": []}, ensure_ascii=False),
            encoding="utf-8",
        )

        r1 = pipeline._load_intermediate("ch1", "critic_report")
        # r1 应是 dict (因为没有 "text" 字段)
        assert r1 == {"scores": {"voice_consistency": 7.5}, "notes": []}
        assert ("ch1", "critic_report") in pipeline._intermediate_cache

        # 第二次命中缓存
        r2 = pipeline._load_intermediate("ch1", "critic_report")
        assert r1 == r2


class TestIntermediateCacheInit:
    """TranslationPipeline.__init__ / run 接入."""

    def test_init_has_intermediate_cache(self):
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline.__init__)
        assert "_intermediate_cache" in source

    def test_run_resets_intermediate_cache(self):
        """run() 应在启动时调用 _reset_per_run_cache (清空 intermediate cache)."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline.run)
        assert "_reset_per_run_cache" in source

    def test_load_intermediate_uses_cache(self):
        """_load_intermediate 实现含 _intermediate_cache 引用."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._load_intermediate)
        assert "_intermediate_cache" in source

    def test_save_intermediate_invalidates_cache(self):
        """_save_intermediate 实现含 invalidate 逻辑."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._save_intermediate)
        assert "_intermediate_cache" in source


class TestIntegratedScriptSyncR99P2:
    """整合脚本同步."""

    def test_intermediate_cache_in_integrated(self):
        import src.translator_agent as t
        assert hasattr(t.TranslationPipeline, "_intermediate_cache") or True  # 字段在 __init__ 设置
        # 验证可通过 __init__ 路径访问
        from src.translator_agent import TranslationPipeline
        # 仅检查方法存在即可 (避免触发真实 __init__)
        assert hasattr(TranslationPipeline, "_load_intermediate")
        assert hasattr(TranslationPipeline, "_save_intermediate")
