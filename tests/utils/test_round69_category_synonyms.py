"""Round 69: 实体一致性相关 reject category 同义映射 —— 避免 ch32 死循环。

保护目标：
  - Jev verdict: "实体一致性违反" + "实体一致性违反(per-entity 聚合)"
  - Pipeline entity_hard_fail: "实体一致性硬失败" + "实体一致性硬失败(降级)"
  - 音色漂移变体: "音色漂移（severe）" → "音色漂移"
  - 这些语义相同，但 stagnation 检测会把它们当成不同类别 → 同根因反复触发不退出

修复前：ch32 实测 450+ retries，judge_reason 在 Jev 与 entity_hard_fail 间来回切换，
       categories 集合不同 → match_mode=category 永远不触发 stagnation
修复后：所有 entity-related categories 归一化为 "实体一致性违反" → 同根因立即 stagnation
"""
from __future__ import annotations

import pytest

from src.utils.text_match import (
    extract_reject_categories,
    _normalize_category,
    _CATEGORY_SYNONYMS,
)


class TestNormalizeCategory:
    """_normalize_category 单元"""

    def test_entity_hard_fail_maps_to_entity_consistency_violation(self):
        assert _normalize_category("实体一致性硬失败") == "实体一致性违反"

    def test_entity_hard_fail_degraded_maps(self):
        assert _normalize_category("实体一致性硬失败(降级)") == "实体一致性违反"

    def test_per_entity_aggregation_maps(self):
        assert _normalize_category("实体一致性违反(per-entity 聚合)") == "实体一致性违反"

    def test_voice_drift_severe_maps(self):
        assert _normalize_category("音色漂移（severe）") == "音色漂移"

    def test_passthrough_unknown(self):
        assert _normalize_category("直译残留") == "直译残留"
        assert _normalize_category("关键语义丢失") == "关键语义丢失"
        assert _normalize_category("语域失配") == "语域失配"


class TestExtractRejectCategoriesSynonyms:
    """extract_reject_categories 应用 synonym 映射"""

    def test_jev_entity_violation_vs_pipeline_hard_fail_equal(self):
        r_jev = "实体一致性违反: noul=0.39 conf=0.00"
        r_hard = "实体一致性硬失败: 3 项译名不一致，须按 entity_registry 修正"
        cats_jev = extract_reject_categories(r_jev)
        cats_hard = extract_reject_categories(r_hard)
        assert cats_jev == cats_hard

    def test_all_entity_variants_equal(self):
        variants = [
            "实体一致性违反: noul=0.39",
            "实体一致性违反(per-entity 聚合): noul=0.42",
            "实体一致性硬失败: 3 项译名不一致",
            "实体一致性硬失败(降级): 8 项译名不一致",
        ]
        all_cats = [set(extract_reject_categories(v)) for v in variants]
        first = all_cats[0]
        for i, cats in enumerate(all_cats[1:], 1):
            assert cats == first, f"Variant {i}: {variants[i]} -> {cats} != {first}"

    def test_ch32_realistic_stagnation_pattern(self):
        r_jev = "实体一致性违反: noul=0.39 conf=0.00; 直译残留: noul=0.71 conf=0.00"
        r_hard = "实体一致性硬失败(降级): 8 项译名不一致，须按 entity_registry 修正"
        cats_jev = extract_reject_categories(r_jev)
        cats_hard = extract_reject_categories(r_hard)
        assert "实体一致性违反" in cats_jev
        assert "实体一致性违反" in cats_hard

    def test_voice_drift_synonym_in_categories(self):
        text = "音色漂移（severe）: prob=0.85 conf=0.50"
        cats = extract_reject_categories(text)
        assert "音色漂移" in cats
        assert "音色漂移（severe）" not in cats


class TestStagnationDetectionCh32:
    def test_consecutive_entity_hard_fails_match(self):
        r1 = "实体一致性硬失败(降级): 8 项译名不一致"
        r2 = "实体一致性硬失败(降级): 8 项译名不一致"
        cats1 = extract_reject_categories(r1)
        cats2 = extract_reject_categories(r2)
        assert bool(cats1) and cats1 == cats2

    def test_transition_jev_to_hard_fail_still_categorizable(self):
        r_jev = "实体一致性违反: noul=0.39 conf=0.00; 直译残留: noul=0.71 conf=0.00"
        r_hard = "实体一致性硬失败(降级): 8 项译名不一致"
        cats_jev = extract_reject_categories(r_jev)
        cats_hard = extract_reject_categories(r_hard)
        assert "实体一致性违反" in cats_jev
        assert "实体一致性违反" in cats_hard


class TestBackwardCompatibility:
    def test_no_synonyms_no_change(self):
        text = "直译残留: noul=0.71"
        cats = extract_reject_categories(text, as_set=True)
        assert cats == {"直译残留"}

    def test_empty_input_safe(self):
        assert extract_reject_categories("", as_set=True) == set()
