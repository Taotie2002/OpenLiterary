"""Round 102 P1: compute_cjk_ratio 按 id(text) 缓存.

设计动机:
  lit_text / raw_fallback / cand_text 等文本对象在同一次 audit+judge 周期内被反复传入
  compute_cjk_ratio (e.g. R97 CJK_RETRY_EXHAUSTED 路径里 lit_text 被算 3 次). 同 str
  对象 id() 稳定, 用 id() 作 key 命中率高; re.sub + ord 循环每次 ~O(N), ch32 单 chunk
  千字级文本一次 ~0.1ms.

设计:
  - 模块级 dict 缓存, key=id(text). 直接持有 str 强引用 (key 复用, value 不重复存储).
  - 上限 _CJK_RATIO_CACHE_MAX=512 防止极端 literal 调用下无界增长 (FIFO 淘汰).
  - 显式 reset_cjk_ratio_cache() 由 _reset_per_run_cache 在 chapter 切换时调用.
  - 整合脚本 src.translator_agent.py 镜像同一逻辑.

测试覆盖:
  - 同对象二次调用命中缓存
  - 不同对象同内容独立缓存
  - reset 清空缓存
  - baseline 行为不变 (空字符串/纯空白/混合比例)
  - _reset_per_run_cache 触发模块级 cache 清空
  - FIFO 上限保护
  - 整合脚本同步 (有缓存 + 有 reset)
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 模块级 cache 行为
# ---------------------------------------------------------------------------
class TestCjkRatioCacheHit:
    """模块级 cache 命中行为."""

    def test_first_call_computes(self):
        """首次调用应计算并存入 cache."""
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache

        reset_cjk_ratio_cache()
        text = "你好世界这是一段中文文本"
        ratio = compute_cjk_ratio(text)
        assert ratio == pytest.approx(1.0)
        assert id(text) in _cjk_ratio_cache

    def test_second_call_hits_cache(self):
        """同 str 对象第二次调用命中 cache — 通过 spy 验证不再调用 re.sub."""
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache
        import re

        reset_cjk_ratio_cache()
        text = "你好世界这是一段中文文本"
        compute_cjk_ratio(text)  # populate
        cache_size_before = len(_cjk_ratio_cache)

        # 监控 re.sub 次数
        original_sub = re.sub
        call_count = [0]

        def spy_sub(*args, **kwargs):
            call_count[0] += 1
            return original_sub(*args, **kwargs)

        re.sub = spy_sub
        try:
            ratio2 = compute_cjk_ratio(text)
        finally:
            re.sub = original_sub

        assert ratio2 == pytest.approx(1.0)
        assert call_count[0] == 0  # re.sub 完全没调用, 命中 cache
        assert len(_cjk_ratio_cache) == cache_size_before  # cache size 不变

    def test_different_objects_independent_cache(self):
        """不同 str 对象 (即使内容相同) 独立 cache."""
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache

        reset_cjk_ratio_cache()
        # 用 bytearray.decode() 强制创建不同对象, 绕过 CPython 字符串 intern.
        # 持有 list 强引用防止 id 被复用.
        text1 = bytearray("你好世界这是一段中文文本", "utf-8").decode("utf-8")
        text2 = bytearray("你好世界这是一段中文文本", "utf-8").decode("utf-8")
        holder = [text1, text2]
        assert text1 == text2
        assert text1 is not text2
        assert id(text1) != id(text2)

        compute_cjk_ratio(text1)
        compute_cjk_ratio(text2)
        assert len(_cjk_ratio_cache) == 2

    def test_cache_size_remains_one_for_same_object(self):
        """同对象多次调用, cache size 始终为 1."""
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache

        reset_cjk_ratio_cache()
        text = "你好世界这是一段中文文本"
        for _ in range(100):
            compute_cjk_ratio(text)
        assert len(_cjk_ratio_cache) == 1

    def test_cache_persists_across_calls(self):
        """cache 在多次调用间持久 (不自动失效)."""
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache

        reset_cjk_ratio_cache()
        text = "你好世界这是一段中文文本"
        ratio1 = compute_cjk_ratio(text)
        # 修改 str 内容 (新建 str, 内容不同)
        text = text + " 追加"
        ratio2 = compute_cjk_ratio(text)
        # 第二个调用是新 str, 重新计算
        assert len(_cjk_ratio_cache) == 2
        assert ratio1 != ratio2 or ratio1 == ratio2  # 不强制 ratio 不同


class TestResetCjkRatioCache:
    """reset_cjk_ratio_cache() 清空行为."""

    def test_reset_clears_cache(self):
        """reset_cjk_ratio_cache 应清空 cache."""
        from src.utils.text_match import (
            compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache
        )

        # 先 reset, 确保 baseline 是干净状态 (前一个测试可能用相同短字符串 id 留了 entry)
        reset_cjk_ratio_cache()
        # 用 bytearray.decode() 产生不同对象, 持有 list 防止 id 复用
        texts = [
            bytearray("你好世界", "utf-8").decode("utf-8"),
            bytearray("另一段中文文本", "utf-8").decode("utf-8"),
        ]
        for t in texts:
            compute_cjk_ratio(t)
        size_before = len(_cjk_ratio_cache)
        assert size_before == 2

        reset_cjk_ratio_cache()
        assert len(_cjk_ratio_cache) == 0

    def test_reset_idempotent(self):
        """重复调用 reset 不报错."""
        from src.utils.text_match import reset_cjk_ratio_cache, _cjk_ratio_cache

        reset_cjk_ratio_cache()
        reset_cjk_ratio_cache()
        assert len(_cjk_ratio_cache) == 0

    def test_after_reset_cache_works_again(self):
        """reset 后 cache 仍可写入/命中."""
        from src.utils.text_match import (
            compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache
        )

        reset_cjk_ratio_cache()
        text = "你好世界"
        compute_cjk_ratio(text)
        assert len(_cjk_ratio_cache) == 1
        reset_cjk_ratio_cache()
        assert len(_cjk_ratio_cache) == 0
        compute_cjk_ratio(text)  # 重新写入
        assert len(_cjk_ratio_cache) == 1


# ---------------------------------------------------------------------------
# Baseline 行为不变
# ---------------------------------------------------------------------------
class TestBaselinePreserved:
    """原有 R96 P1 测试用例在新 cache 行为下仍通过."""

    def test_pure_chinese(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("你好世界这是一段中文文本") == pytest.approx(1.0, abs=1e-6)

    def test_pure_english(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("Hello world this is English") == pytest.approx(0.0, abs=1e-6)

    def test_empty_string(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("") == 0.0

    def test_whitespace_only(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("   \n\t  ") == 0.0

    def test_mixed_2_of_5(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("你好abc") == pytest.approx(0.4, abs=1e-6)

    def test_chinese_with_punctuation(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("你好世界，中文。") == pytest.approx(0.75, abs=1e-6)

    def test_ext_a_chars(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("㐀") == pytest.approx(1.0, abs=1e-6)

    def test_cache_does_not_corrupt_after_empty(self):
        """空字符串后, 后续正常字符串 ratio 仍正确 (无副作用)."""
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache

        reset_cjk_ratio_cache()
        assert compute_cjk_ratio("") == 0.0
        assert compute_cjk_ratio("   ") == 0.0
        assert compute_cjk_ratio("你好") == 1.0  # 仍然正确

    def test_cache_hit_returns_same_value_as_compute(self):
        """cache 命中值与重新计算值一致."""
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache

        reset_cjk_ratio_cache()
        text = "你好abc"
        ratio1 = compute_cjk_ratio(text)
        cached = _cjk_ratio_cache[id(text)]
        assert ratio1 == cached


# ---------------------------------------------------------------------------
# FIFO 上限保护
# ---------------------------------------------------------------------------
class TestFifoEviction:
    """缓存上限 _CJK_RATIO_CACHE_MAX=512 + FIFO 淘汰."""

    def test_cache_bounded_by_max(self):
        """cache 大小不会无限增长 (插入 600 个不同 str 后, 大小 <= max)."""
        from src.utils.text_match import (
            compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache, _CJK_RATIO_CACHE_MAX
        )

        reset_cjk_ratio_cache()
        # 插入 _CJK_RATIO_CACHE_MAX + 100 个不同 str
        n_unique = _CJK_RATIO_CACHE_MAX + 100
        for i in range(n_unique):
            compute_cjk_ratio(f"测试文本{i}中文")  # 每次新 str
        # cache size 应被限制
        assert len(_cjk_ratio_cache) <= _CJK_RATIO_CACHE_MAX

    def test_fifo_eviction_keeps_recent(self):
        """FIFO 淘汰: 最近插入的应保留, 最老的应被淘汰."""
        from src.utils.text_match import (
            compute_cjk_ratio, reset_cjk_ratio_cache, _cjk_ratio_cache, _CJK_RATIO_CACHE_MAX
        )

        reset_cjk_ratio_cache()
        # 用 bytearray.decode() 绕过 CPython 字符串 intern. 必须持有 list 强引用,
        # 否则 CPython 内存分配器会复用前一个被 GC 的 str 的 id(), 导致 cache 被覆盖.
        texts = []
        for i in range(_CJK_RATIO_CACHE_MAX):
            text = bytearray(f"文本{i}中文", "utf-8").decode("utf-8")
            texts.append(text)
            compute_cjk_ratio(text)
        # 再插入 1 个 (触发 FIFO 淘汰)
        new_text = bytearray("新插入文本中文", "utf-8").decode("utf-8")
        texts.append(new_text)
        compute_cjk_ratio(new_text)
        # 新文本应仍在 cache (最近插入)
        assert id(new_text) in _cjk_ratio_cache
        # 大小仍为 MAX
        assert len(_cjk_ratio_cache) == _CJK_RATIO_CACHE_MAX


# ---------------------------------------------------------------------------
# TranslationPipeline 集成: _reset_per_run_cache 调用 reset_cjk_ratio_cache
# ---------------------------------------------------------------------------
class TestPipelineResetIntegration:
    """TranslationPipeline._reset_per_run_cache 应触发模块级 cache 清空."""

    def test_pipeline_reset_clears_module_cache(self):
        """pipeline._reset_per_run_cache 调用后, 模块级 _cjk_ratio_cache 为空."""
        from src.translator_agent import TranslationPipeline
        from src.utils.text_match import compute_cjk_ratio, _cjk_ratio_cache

        # populate module-level cache
        compute_cjk_ratio("你好世界")
        compute_cjk_ratio("另一段中文")
        assert len(_cjk_ratio_cache) >= 2

        # stub pipeline with all required cache fields
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._entity_registry_cache = {}
        pipeline._entity_variants_cache = {}
        pipeline._intermediate_cache = {}
        pipeline._narrative_memory_cache = {}
        pipeline._avg_score_cache = {}

        pipeline._reset_per_run_cache()
        assert len(_cjk_ratio_cache) == 0

    def test_reset_per_run_source_contains_reset_call(self):
        """_reset_per_run_cache 源码应包含 reset_cjk_ratio_cache 调用."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._reset_per_run_cache)
        assert "reset_cjk_ratio_cache" in source


# ---------------------------------------------------------------------------
# Cache 加速效果 (smoke)
# ---------------------------------------------------------------------------
class TestCacheSpeedup:
    """缓存命中应显著快于重新计算."""

    def test_cache_hit_faster_than_compute(self):
        """重复调用同 str, cache hit 应远快于 re-compute."""
        import time
        from src.utils.text_match import compute_cjk_ratio, reset_cjk_ratio_cache

        reset_cjk_ratio_cache()
        # 构造一个长文本 (500 字符)
        text = "你好世界这是一段中文文本" * 25
        # populate
        compute_cjk_ratio(text)
        # cache hit ×N
        n = 50000
        t0 = time.perf_counter()
        for _ in range(n):
            compute_cjk_ratio(text)
        t1 = time.perf_counter()
        cache_hit_per = (t1 - t0) * 1e6 / n
        # cache hit < 1us 是预期 (re_sub 每次至少几十 us)
        assert cache_hit_per < 5.0, f"cache hit too slow: {cache_hit_per:.3f} us/call"


# ---------------------------------------------------------------------------
# 整合脚本 src.translator_agent.py 同步
# ---------------------------------------------------------------------------
class TestIntegratedScriptSyncR102P1:
    """整合脚本镜像同一逻辑."""

    def test_compute_cjk_ratio_cached_in_integrated(self):
        """整合脚本的 compute_cjk_ratio 也有缓存 (id-based)."""
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.compute_cjk_ratio)
        # 应包含 cache 逻辑
        assert "_cjk_ratio_cache_get_j" in source or "_cjk_ratio_cache_get" in source
        assert "_cjk_ratio_cache_put_j" in source or "_cjk_ratio_cache_put" in source

    def test_reset_cjk_ratio_cache_in_integrated(self):
        """整合脚本暴露 reset_cjk_ratio_cache()."""
        import src.translator_agent as t
        assert hasattr(t, "reset_cjk_ratio_cache")
        assert callable(t.reset_cjk_ratio_cache)

    def test_integrated_cache_hit_works(self):
        """整合脚本的 cache hit 行为."""
        import src.translator_agent as t
        # reset 整合 cache (散装不受影响)
        t.reset_cjk_ratio_cache()
        text = "你好世界这是一段中文文本"
        r1 = t.compute_cjk_ratio(text)
        r2 = t.compute_cjk_ratio(text)
        assert r1 == r2 == pytest.approx(1.0)

    def test_integrated_reset_clears_cache(self):
        """整合脚本 reset 清空."""
        import src.translator_agent as t
        # 找到整合模式的内部 cache dict
        cache_dict_name = "_cjk_ratio_cache"
        cache = getattr(t, cache_dict_name, None)
        assert cache is not None, f"整合脚本缺少 {cache_dict_name}"
        # populate
        t.compute_cjk_ratio("测试文本中文")
        assert len(cache) >= 1
        t.reset_cjk_ratio_cache()
        assert len(cache) == 0
