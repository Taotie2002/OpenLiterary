"""Round 85: 补全 _JEV_CATEGORY_STRATEGY, 让 Round 82 新增的 subtle+其它 escalation 有专用策略.

背景:
  Round 82 在 verdict.py::synthesize_verdict 中实现了 subtle voice drift 的组合升级:
  - subtle 单独 → 不 reject, 记入 skipped_dimensions
  - subtle + 其它 reject → 加入 reject_reasons (格式: "音色漂移（subtle+其它）: ...")

  但 rewriter_agent.py::_JEV_CATEGORY_STRATEGY 没有这个类别的策略, 导致:
  - _build_jev_category_section 在遍历到该类别时找不到策略
  - 走通用 fallback: "请针对此维度整体审视并修订" (无具体指导)
  - Rewriter 没有针对 subtle+其它 模式的专门 prompt, 重写效果打折

修复:
  在 _JEV_CATEGORY_STRATEGY 增加 "音色漂移（subtle+其它）" 键值:
  - 强调: 此模式意味着 subtle 单独不够, 其它维度也已 reject
  - 指导: 在解决其它核心问题的同时把 voice drift 拉回正轨
  - 注意: 不要过度风格化 (否则反向触发 severe voice drift)

验证:
  - _JEV_CATEGORY_STRATEGY 含 "音色漂移（subtle+其它）" 键
  - 该键对应的策略不是通用 fallback 文本
  - _build_jev_category_section 能渲染该类别
  - 散装 + 整合脚本双向同步
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



# ---------------------------------------------------------------------------
# Round 85: _JEV_CATEGORY_STRATEGY 必须包含 Round 82 新类别
# ---------------------------------------------------------------------------

class TestStrategyComplete:
    """_JEV_CATEGORY_STRATEGY 应包含所有 verdict 生成的 reject 类别."""

    def test_subtle_other_in_strategy(self):
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        assert "音色漂移（subtle+其它）" in strategy, (
            "Round 85: _JEV_CATEGORY_STRATEGY 应包含 '音色漂移（subtle+其它）' 类别"
        )

    def test_subtle_other_strategy_is_substantive(self):
        """策略不应是空字符串或过短 (>= 30 chars 表示有具体内容)."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        s = strategy["音色漂移（subtle+其它）"]
        assert isinstance(s, str)
        assert len(s) >= 30, (
            f"策略过短 ({len(s)} chars): {s!r}. 应>= 30 chars 提供具体指导"
        )

    def test_subtle_other_strategy_mentions_combination(self):
        """策略应明确说明 'subtle 单独不 reject, 与其它叠加' 的语义."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        s = strategy["音色漂移（subtle+其它）"]
        # 至少应包含 "其它" 字样 (说明是组合升级)
        assert "其它" in s, (
            f"策略应说明 '其它' 维度: {s!r}"
        )

    def test_subtle_other_strategy_warns_about_over_correction(self):
        """策略应警告不要过度风格化 (避免反向 severe)."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        s = strategy["音色漂移（subtle+其它）"]
        # 应包含 severe 警告或类似保护说明
        has_warning = any(kw in s for kw in [
            "severe", "过度风格化", "反向", "避免被抹平"
        ])
        assert has_warning, (
            f"策略应警告不要过度风格化 (避免反向 severe): {s!r}"
        )


# ---------------------------------------------------------------------------
# Round 85: _build_jev_category_section 能正确渲染该类别
# ---------------------------------------------------------------------------

class TestBuildJevCategorySection:
    """_build_jev_category_section 应使用专用策略而非通用 fallback."""

    def test_section_includes_subtle_other_strategy(self):
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        # 构造 stub (避免完整 __init__)
        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        reject_reason = "音色漂移（subtle+其它）: subtle=0.70 severe=0.10 conf=0.80"
        section = agent._build_jev_category_section(reject_reason)
        # 应包含专用策略内容
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY[
            "音色漂移（subtle+其它）"
        ]
        assert strategy in section, (
            "Round 85: _build_jev_category_section 应渲染专用策略, "
            f"section={section!r}, strategy={strategy!r}"
        )

    def test_section_not_using_generic_fallback(self):
        """section 不应回退到通用 '请针对此维度整体审视并修订'."""
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        reject_reason = "音色漂移（subtle+其它）: subtle=0.70 severe=0.10 conf=0.80"
        section = agent._build_jev_category_section(reject_reason)
        # 通用 fallback 不应出现
        assert "请针对此维度整体审视并修订" not in section, (
            "Round 85: _build_jev_category_section 不应使用通用 fallback"
        )


# ---------------------------------------------------------------------------
# Round 85: Round 82/82 已修复的其它类别策略不被破坏 (回归)
# ---------------------------------------------------------------------------

class TestRegressionRound82Categories:
    """_JEV_CATEGORY_STRATEGY 仍含 Round 82 之前已有的关键类别."""

    def test_severe_strategy_present(self):
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        assert "音色漂移（severe）" in strategy

    def test_subtle_alone_strategy_present(self):
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        assert "音色漂移（subtle）" in strategy

    def test_literal_residue_strategy_present(self):
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        assert "直译残留" in strategy

    def test_semantic_loss_strategy_present(self):
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        strategy = LiteraryRewriterAgent._JEV_CATEGORY_STRATEGY
        assert "关键语义丢失" in strategy


# ---------------------------------------------------------------------------
# Round 85: 整合脚本 src/translator_agent.py 同步
# ---------------------------------------------------------------------------

class TestIntegratedScriptSynced:
    """散装 src/agents/rewriter_agent.py 修复必须同步到整合脚本."""

    def test_round_85_marker_in_scattered(self):
        content = Path("src/agents/rewriter_agent.py").read_text(encoding="utf-8")
        assert "Round 85" in content, "散装 rewriter_agent.py 未标注 Round 85"

    def test_round_85_marker_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 85" in content, "整合脚本未同步 Round 85"

    def test_subtle_other_strategy_in_scattered(self):
        content = Path("src/agents/rewriter_agent.py").read_text(encoding="utf-8")
        assert '"音色漂移（subtle+其它）"' in content, (
            "散装 rewriter_agent.py 应有 '音色漂移（subtle+其它）' 策略"
        )

    def test_subtle_other_strategy_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert '"音色漂移（subtle+其它）"' in content, (
            "整合脚本 src/translator_agent.py 应有 '音色漂移（subtle+其它）' 策略"
        )
