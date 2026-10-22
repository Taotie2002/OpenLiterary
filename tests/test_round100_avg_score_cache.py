"""Round 100 P2: _compute_avg_score 契约 + Round 103 P3 修正.

Round 100 P2 引入按 id(critic_report) 的内存缓存, 减少 audit 与 judge 阶段对同一
critic_report 重复 list build + sum/len.

Round 103 P3 修正: CPython dict 被 GC 后 id() 可复用, 实测导致
test_round93_score_decline_early_termination::test_zero_filtered_at_update_time
出现 [7.0, 7.0] 而非 [7.0, 6.0] 的 avg 误读. avg 计算本身 O(6) sum/len 微秒级,
不值得 correctness 风险, 移除 id() 缓存. R103 P7 同步到 monolith (散装早已修).

本测试现在锁定 R103 P3 后的契约:
  - _compute_avg_score 每次重新计算 (无缓存, 保证 correctness)
  - _avg_score_cache 字段保留为 backward-compat 占位 (测试桩初始化用),
    但 _compute_avg_score 不再读写它
  - 行为不变: 空 scores / 非 dict / 部分非数值 / 多次调用一致
"""
from __future__ import annotations

import pytest


class TestAvgScoreCache:
    """Round 100 P2: _compute_avg_score 缓存."""

    def test_first_call_computes(self, tmp_path):
        """首次调用正确计算 (Round 103 P3 后无 id() 缓存)."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}  # backward-compat 字段, 现 _compute_avg_score 不再写入

        cr = {"scores": {"voice_consistency": 7.5, "literal_residue": 4.0}}
        result = pipeline._compute_avg_score(cr)
        assert result == pytest.approx(5.75)
        # R103 P3: 不再缓存, 缓存 dict 保持空
        assert pipeline._avg_score_cache == {}

    def test_second_call_recomputes(self, tmp_path):
        """Round 103 P3 后: 第二次调用重新计算 (无缓存), 反映当前 dict 真实状态.
        这正是删除 id() 缓存的原因 — 否则 CPython dict GC 后 id 复用会读到陈旧值.
        """
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}

        cr = {"scores": {"voice_consistency": 7.0}}
        r1 = pipeline._compute_avg_score(cr)

        # 修改 dict: 第二次调用必须读到新值 (R103 P3 修正的核心)
        cr["scores"]["voice_consistency"] = 99.0

        r2 = pipeline._compute_avg_score(cr)
        assert r1 == 7.0
        assert r2 == pytest.approx(99.0)  # 不缓存, 重算

    def test_different_objects_independent(self, tmp_path):
        """不同 dict 对象各自计算正确."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}

        cr1 = {"scores": {"x": 5.0}}
        cr2 = {"scores": {"x": 8.0}}

        r1 = pipeline._compute_avg_score(cr1)
        r2 = pipeline._compute_avg_score(cr2)
        assert r1 == 5.0
        assert r2 == 8.0
        # R103 P3: 不再缓存, _avg_score_cache 保持空 (backward-compat 字段)
        assert pipeline._avg_score_cache == {}

    def test_empty_scores_returns_zero(self, tmp_path):
        """空 scores 应返回 0.0 (并缓存 0.0)."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}

        cr = {"scores": {}}
        assert pipeline._compute_avg_score(cr) == 0.0

    def test_non_dict_returns_zero(self, tmp_path):
        """非 dict 应返回 0.0 (不缓存)."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}

        assert pipeline._compute_avg_score(None) == 0.0
        assert pipeline._compute_avg_score("not a dict") == 0.0
        assert pipeline._compute_avg_score(42) == 0.0

    def test_partial_numeric_only(self, tmp_path):
        """部分非数值: 仅算数值项."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {}

        cr = {"scores": {"x": 6.0, "y": "not a number", "z": 4.0}}
        assert pipeline._compute_avg_score(cr) == pytest.approx(5.0)

    def test_reset_clears_cache_field(self, tmp_path):
        """_reset_per_run_cache 应清空 _avg_score_cache 字段 (backward-compat).

        Round 103 P3 后 _compute_avg_score 不再写入 _avg_score_cache, 但字段
        仍存在 (供旧测试桩与外部调用方), _reset_per_run_cache 仍应清空它,
        保持向后兼容契约.
        """
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._avg_score_cache = {"some_id": 5.0}  # 模拟外部写入 (旧行为残留)
        pipeline._entity_registry_cache = {}
        pipeline._entity_variants_cache = {}
        pipeline._intermediate_cache = {}
        pipeline._narrative_memory_cache = {}

        pipeline._reset_per_run_cache()
        assert pipeline._avg_score_cache == {}


class TestAvgScoreCacheInit:
    """TranslationPipeline 字段接入 (Round 103 P3 后 _avg_score_cache 为 backward-compat 占位)."""

    def test_init_has_avg_score_cache_field(self):
        """__init__ 仍声明 _avg_score_cache 字段 (向后兼容旧测试桩与外部调用方).

        Round 103 P3 删除 id() 缓存逻辑后, 字段本身保留 — 旧测试用
        `pipeline._avg_score_cache = {}` 初始化时仍能正常工作.
        """
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline.__init__)
        assert "_avg_score_cache" in source

    def test_compute_avg_score_does_not_use_cache(self):
        """Round 103 P3 后 _compute_avg_score 函数体不应再引用 _avg_score_cache.

        这是 R103 P3 修正的核心保证: 不缓存, 每次重算, 避免 id() 复用读到陈旧 avg.
        """
        import inspect
        import src.translator_agent as t
        source = inspect.getsource(t.TranslationPipeline._compute_avg_score)
        assert "_avg_score_cache" not in source, (
            "Round 103 P3: _compute_avg_score 不应再访问 _avg_score_cache, "
            "CPython id() 复用会导致读到陈旧 avg."
        )


class TestIntegratedScriptSyncR103P7:
    """整合脚本同步 Round 103 P7 (P3 同步到 monolith)."""

    def test_avg_score_cache_field_in_integrated(self):
        """整合脚本 __init__ 仍保留 _avg_score_cache 字段 (向后兼容)."""
        import inspect
        import src.translator_agent as t
        init_src = inspect.getsource(t.TranslationPipeline.__init__)
        compute_src = inspect.getsource(t.TranslationPipeline._compute_avg_score)
        assert "_avg_score_cache" in init_src, (
            "整合脚本 __init__ 缺失 _avg_score_cache 字段, 破坏 backward-compat"
        )
        assert "_avg_score_cache" not in compute_src, (
            "整合脚本 _compute_avg_score 不应再引用 _avg_score_cache (R103 P3/P7)"
        )
