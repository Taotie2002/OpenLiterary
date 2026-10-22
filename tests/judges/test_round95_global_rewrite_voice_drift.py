"""Round 95: GLOBAL_REWRITE_CATEGORIES 补齐 voice_drift 类别 — 端到端验证

修复前: 音色漂移（severe）/ 音色漂移（subtle+其它）触发后只能拿到具体策略,
        走不到"整体重润色"分支 (GLOBAL_REWRITE_CATEGORIES lookup miss → 局部词换).

修复后: 这两个类别纳入 GLOBAL_REWRITE_CATEGORIES, 触发整体重写策略.

注: 音色漂移（subtle）单独不 reject (Round 82 设计), 故不纳入.
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



def _extract_global_strategy_dict(src_text: str) -> dict:
    """从 rewriter_agent 源码中提取 GLOBAL_REWRITE_CATEGORIES dict (源码级静态检查)."""
    import re
    # Match GLOBAL_REWRITE_CATEGORIES = { ... } with multi-line content
    m = re.search(
        r'GLOBAL_REWRITE_CATEGORIES[^{]*\{(.+?)\n        \}',
        src_text,
        re.DOTALL,
    )
    if not m:
        return {}
    block = m.group(0)
    keys = re.findall(r'"([^"]+)":', block)
    return {k: True for k in keys}


class TestGlobalRewriteCategoriesComplete:
    """GLOBAL_REWRITE_CATEGORIES 应覆盖所有 voice_drift 升级类别."""

    def test_severe_voice_drift_in_global(self):
        """音色漂移（severe）必须能在 GLOBAL_REWRITE_CATEGORIES 命中."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent

        # 通过 mock 方式读取 GLOBAL_REWRITE_CATEGORIES 内容 (它在函数内)
        # 改用静态源码检查确保含该 key
        with open(Path(LiteraryRewriterAgent.__module__.replace(".", "/")).with_suffix(".py"), "r") as f:
            src = f.read()
        keys = _extract_global_strategy_dict(src)
        assert "音色漂移（severe）" in keys, (
            f"GLOBAL_REWRITE_CATEGORIES 应含 音色漂移（severe）. 当前 keys: {list(keys.keys())}"
        )

    def test_subtle_plus_other_in_global(self):
        """音色漂移（subtle+其它）必须能在 GLOBAL_REWRITE_CATEGORIES 命中."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent

        with open(Path(LiteraryRewriterAgent.__module__.replace(".", "/")).with_suffix(".py"), "r") as f:
            src = f.read()
        keys = _extract_global_strategy_dict(src)
        assert "音色漂移（subtle+其它）" in keys, (
            f"GLOBAL_REWRITE_CATEGORIES 应含 音色漂移（subtle+其它）. 当前 keys: {list(keys.keys())}"
        )

    def test_subtle_alone_not_in_global(self):
        """音色漂移（subtle）单独不 reject (Round 82 设计), 不应放入 GLOBAL."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent

        with open(Path(LiteraryRewriterAgent.__module__.replace(".", "/")).with_suffix(".py"), "r") as f:
            src = f.read()
        keys = _extract_global_strategy_dict(src)
        assert "音色漂移（subtle）" not in keys, (
            "音色漂移（subtle）单独不 reject, 不应放入 GLOBAL_REWRITE_CATEGORIES"
        )


class TestGlobalStrategyAlignedWithJevStrategy:
    """GLOBAL_REWRITE_CATEGORIES 的 keys 必须都能在 _JEV_CATEGORY_STRATEGY 中找到 (兜底)."""

    def test_all_global_keys_in_jev_strategy(self):
        from src.agents.rewriter_agent import LiteraryRewriterAgent

        jev_keys = set(LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY.keys())
        # 提取 GLOBAL 里的所有 key (跨模块都行, 这里只看散装)
        with open(Path(LiteraryRewriterAgent.__module__.replace(".", "/")).with_suffix(".py"), "r") as f:
            src = f.read()
        global_keys = set(_extract_global_strategy_dict(src).keys())

        missing = global_keys - jev_keys
        assert not missing, (
            f"GLOBAL_REWRITE_CATEGORIES 中以下 key 缺少 _JEV_CATEGORY_STRATEGY 兜底: {missing}"
        )
