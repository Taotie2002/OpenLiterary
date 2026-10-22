"""Round 89: extract_reject_categories 策略查找时禁用 category 同义归一化.

背景:
  Round 69 为修复 stagnation 检测 (prev_cats == curr_cats 时强制 fallback),
  引入了 _CATEGORY_SYNONYMS 把 `音色漂移（severe）` 归一为 `音色漂移`.
  但 `extract_reject_categories` 的两个调用方语义不同:
    1. 策略查找 (rewriter_agent.py):
       - 期望原始类别形式, 以便 lookup _JEV_CATEGORY_STRATEGY / GLOBAL_REWRITE_CATEGORIES
       - 这些 dicts 的 keys 是 `音色漂移（severe）`, `音色漂移（subtle）` 等具体形式
       - 同义归一化导致 lookup miss, 落到 generic fallback "请针对此维度整体审视并修订。"
    2. Stagnation 检测 (pipeline.py:1506, 1624):
       - 期望归一化形式, 把 severe / subtle 视为同一类 (都属于 voice drift)
       - 比较 prev_cats == curr_cats

修复:
  extract_reject_categories 增加 normalize 参数 (默认 True, 保持 Round 69 行为),
  策略查找路径显式传 normalize=False.
  
  rewriter_agent.py 的两个调用点:
    - _build_jev_category_section (line 108): 改用 normalize=False
    - _build_global_strategy (line 389): 改用 normalize=False
  pipeline.py 的 stagnation 检测 (line 1506, 1624): 保持 normalize=True (默认)

验证:
  - `_build_jev_category_section('音色漂移（severe）: ...')` 必须命中 severe 策略
  - `_build_global_strategy` 检测 severe voice drift 时落到 GLOBAL_REWRITE_CATEGORIES
    之外的分支 (因为 GLOBAL_REWRITE_CATEGORIES 没有 voice drift, 应按 fallback)
  - stagnation 检测仍把 severe / subtle 视为同 category
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_modules():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
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



def test_extract_reject_categories_default_normalizes():
    """默认行为 (Round 69 兼容): `音色漂移（severe）` → `音色漂移`."""
    from src.utils.text_match import extract_reject_categories

    text = "音色漂移（severe）: prob=0.60 conf=0.55"
    cats = extract_reject_categories(text)
    assert "音色漂移" in cats
    assert "音色漂移（severe）" not in cats


def test_extract_reject_categories_no_normalize_preserves_specific():
    """normalize=False: 保留 `音色漂移（severe）` 等具体形式, 用于策略 lookup."""
    from src.utils.text_match import extract_reject_categories

    text = "音色漂移（severe）: prob=0.60 conf=0.55; 直译残留: noul=0.71"
    cats = extract_reject_categories(text, normalize=False)
    assert "音色漂移（severe）" in cats
    assert "音色漂移" not in cats
    assert "直译残留" in cats


def test_build_jev_category_section_hits_severe_strategy():
    """rewriter_agent._build_jev_category_section 对 severe voice drift 必须命中 severe 策略.

    Round 89 修复前: 同义归一化把 `音色漂移（severe）` 转为 `音色漂移`,
    _JEV_CATEGORY_STRATEGY lookup miss, 落到 generic fallback.
    """
    from src.agents.rewriter_agent import LiteraryRewriterAgent

    # _build_jev_category_section 是 instance method, 用 stub instance
    agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)

    reject_reason = "音色漂移（severe）: prob=0.60 conf=0.55"
    section = agent._build_jev_category_section(reject_reason)

    # 修复前: section 只包含 generic "请针对此维度整体审视并修订。"
    # 修复后: section 必须包含 severe-specific 策略文本
    severe_specific = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY["音色漂移（severe）"]
    assert severe_specific in section, (
        f"severe voice drift 策略未命中. 期望包含: {severe_specific!r}\n"
        f"实际 section:\n{section}"
    )


def test_build_jev_category_section_hits_subtle_strategy():
    """subtle voice drift 必须命中 subtle 策略 (不是 severe)."""
    from src.agents.rewriter_agent import LiteraryRewriterAgent

    agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
    reject_reason = "音色漂移（subtle）: subtle=0.55 severe=0.10 conf=0.55"
    section = agent._build_jev_category_section(reject_reason)

    subtle_specific = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY["音色漂移（subtle）"]
    assert subtle_specific in section


def test_build_jev_category_section_hits_subtle_plus_other_strategy():
    """Round 85 新增的 compound category 必须命中 compound 策略."""
    from src.agents.rewriter_agent import LiteraryRewriterAgent

    agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
    reject_reason = "音色漂移（subtle+其它）: subtle=0.55 severe=0.10 conf=0.55; 直译残留: noul=0.71"
    section = agent._build_jev_category_section(reject_reason)

    compound_specific = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY["音色漂移（subtle+其它）"]
    assert compound_specific in section


def test_stagnation_detection_normalizes_severe():
    """Stagnation 检测 (默认 normalize=True): severe 归一为 `音色漂移`.

    注: 当前 _CATEGORY_SYNONYMS 仅映射 `音色漂移（severe）` → `音色漂移`,
    不映射 `音色漂移（subtle）`. 真实 stagnation 风险主要在 severe 路径
    (severe → 音色漂移 后, 与不带括号的"音色漂移"视为同 category, 触发 stagnation fallback).
    subtle 路径则保持具体形式 (策略查找精确). 这是 Round 69 的设计选择,
    Round 89 不改变 stagnation 语义, 只修复策略查找路径.
    """
    from src.utils.text_match import extract_reject_categories

    severe_cats = extract_reject_categories("音色漂移（severe）: prob=0.60 conf=0.55")
    bare_cats = extract_reject_categories("音色漂移: prob=0.60 conf=0.55")
    assert severe_cats == bare_cats, (
        "severe 必须归一为 音色漂移, 以与不带括号的同 category 视为同一类 (stagnation). "
        f"severe={severe_cats}, bare={bare_cats}"
    )
    assert "音色漂移" in severe_cats
    assert "音色漂移（severe）" not in severe_cats


def test_strategy_lookup_separates_severe_from_subtle():
    """策略 lookup (normalize=False): severe / subtle 视为不同类 (策略不同)."""
    from src.utils.text_match import extract_reject_categories

    severe_cats = extract_reject_categories(
        "音色漂移（severe）: prob=0.60 conf=0.55", normalize=False
    )
    subtle_cats = extract_reject_categories(
        "音色漂移（subtle）: subtle=0.55 conf=0.55", normalize=False
    )
    assert severe_cats != subtle_cats, (
        "策略 lookup 必须保留 severe/subtle 区分. "
        f"severe={severe_cats}, subtle={subtle_cats}"
    )


def test_entity_synonyms_still_normalize():
    """实体一致性 synonyms (Round 69) 默认仍归一化, 不破坏 stagnation 检测."""
    from src.utils.text_match import extract_reject_categories

    text1 = "实体一致性硬失败: noul=0.39 conf=0.61"
    text2 = "实体一致性违反(per-entity 聚合): noul=0.39 conf=0.61"
    cats1 = extract_reject_categories(text1)
    cats2 = extract_reject_categories(text2)
    assert cats1 == cats2
    assert "实体一致性违反" in cats1
