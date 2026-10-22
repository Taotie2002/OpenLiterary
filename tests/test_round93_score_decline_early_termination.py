"""Round 93: 连续分数下降检测 → 提前 fallback 节省 LLM 调用.

背景:
  Round 6 / 12 / 16 / 73 已有 stagnation/oscillation/pre-rewriter 早停链, 但仅检测
  "类别重复" 或 "文本几乎不变". 现实中 Jev noul 在中文上 noise 较大, 同一 retry 反复触发
  相同类别但分数缓慢下降 (e.g., 7.0 → 6.5 → 6.0), 不被现有 stagnation 触发, 持续浪费 LLM 调用.

修复:
  pipeline.TranslationPipeline 增加 _score_history 状态, 每次 _update_best_lit 同步追加
  当前 avg_score. judge REJECT 路径前检查 _is_declining_streak(chunk_id, current, n=3),
  若末尾 3 个分数严格递减 (s[i-1] > s[i]), 跳过 rewriter fallback, 直接用 best_lit 写入 final,
  并标 PERMANENTLY_FAILED (reason=SCORE_DECLINE_STREAK).

保护目标:
  - _score_history 仅追加 avg > 0.0 的样本 (过滤 critic 空评分)
  - _is_declining_streak(len=consecutive) 需要至少 consecutive 个样本才下结论
  - 严格递减: s[i-1] > s[i], 不含相等
  - 不影响 Round 92 的 best-lit fallback 逻辑
  - judge PASS 路径不走 streak 检查 (PASS 就定稿)
  - Round 6/12/16/73 stagnation 检测保持不变
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



def _new_pipeline():
    from src.pipeline import TranslationPipeline
    p = TranslationPipeline.__new__(TranslationPipeline)
    p._best_lit = {}
    p._score_history = {}
    return p


# ---------------------------------------------------------------------------
# (1) _score_history 维护 — _update_best_lit 同步追加
# ---------------------------------------------------------------------------

class TestScoreHistoryUpdate:
    def test_first_update_appends(self):
        p = _new_pipeline()
        p._update_best_lit("ch1", "v0", {"scores": {"voice": 6.5}})
        assert p._score_history.get("ch1") == [6.5]

    def test_multiple_updates_accumulate(self):
        p = _new_pipeline()
        p._update_best_lit("ch1", "v0", {"scores": {"voice": 7.0}})
        p._update_best_lit("ch1", "v1", {"scores": {"voice": 6.5}})
        p._update_best_lit("ch1", "v2", {"scores": {"voice": 6.0}})
        assert p._score_history["ch1"] == [7.0, 6.5, 6.0]

    def test_zero_score_not_appended(self):
        """critic 返回空 scores → avg=0.0 → 不污染历史."""
        p = _new_pipeline()
        p._update_best_lit("ch1", "v0", {"scores": {}})  # avg=0.0
        p._update_best_lit("ch1", "v1", {"scores": {"voice": 6.0}})
        p._update_best_lit("ch1", "v2", {"scores": {}})  # avg=0.0 again
        # 0.0 样本被过滤, history 仅含 6.0
        assert p._score_history["ch1"] == [6.0]

    def test_history_isolated_per_chunk(self):
        p = _new_pipeline()
        p._update_best_lit("ch1", "v", {"scores": {"voice": 7.0}})
        p._update_best_lit("ch2", "v", {"scores": {"voice": 6.0}})
        assert p._score_history["ch1"] == [7.0]
        assert p._score_history["ch2"] == [6.0]


# ---------------------------------------------------------------------------
# (2) _is_declining_streak — 核心检测逻辑
# ---------------------------------------------------------------------------

class TestIsDecliningStreak:
    def test_no_history_returns_false(self):
        p = _new_pipeline()
        assert p._is_declining_streak("ch1", 5.0) is False

    def test_insufficient_samples_returns_false(self):
        """需要 consecutive 个样本 (含当前), 不足不下结论."""
        p = _new_pipeline()
        p._score_history["ch1"] = [7.0]  # 只有 1 个
        assert p._is_declining_streak("ch1", 6.0, consecutive=3) is False

    def test_strictly_declining_returns_true(self):
        p = _new_pipeline()
        p._score_history["ch1"] = [7.0, 6.5]  # 2 个历史 + 当前 = 3 个
        assert p._is_declining_streak("ch1", 6.0, consecutive=3) is True

    def test_ascending_returns_false(self):
        p = _new_pipeline()
        p._score_history["ch1"] = [5.0, 6.0]
        assert p._is_declining_streak("ch1", 7.0, consecutive=3) is False

    def test_equal_not_declining(self):
        """相等不算严格递减 (允许偶然相等, 避免误判)."""
        p = _new_pipeline()
        p._score_history["ch1"] = [6.0, 6.0]
        assert p._is_declining_streak("ch1", 6.0, consecutive=3) is False

    def test_zero_filtered(self):
        """0.0 样本被过滤, 不计入 consecutive."""
        p = _new_pipeline()
        p._score_history["ch1"] = [7.0, 0.0]  # 0.0 算无效
        # 有效样本仅 7.0 + 6.0 = 2 个, 不够 3
        assert p._is_declining_streak("ch1", 6.0, consecutive=3) is False

    def test_only_last_3_matter(self):
        """历史长度 > consecutive 时, 仅末尾 consecutive 个生效."""
        p = _new_pipeline()
        p._score_history["ch1"] = [9.0, 8.0, 7.0, 6.5]  # 前 2 个高分被忽略
        # 末尾 2 个 [7.0, 6.5] + 当前 6.0 = [7.0, 6.5, 6.0] 严格递减 → True
        assert p._is_declining_streak("ch1", 6.0, consecutive=3) is True

    def test_consecutive_4_requires_4_samples(self):
        """consecutive=4 时需要 history 提供 3 个 + 当前 = 4 个样本."""
        p = _new_pipeline()
        p._score_history["ch1"] = [7.0, 6.5, 6.0]  # 3 个
        assert p._is_declining_streak("ch1", 5.5, consecutive=4) is True
        p._score_history["ch1"] = [7.0, 6.5]  # 2 个, 不够
        assert p._is_declining_streak("ch1", 5.5, consecutive=4) is False


# ---------------------------------------------------------------------------
# (3) 端到端: 重试循环中的 score-decline 触发
# ---------------------------------------------------------------------------

class TestEndToEndScoreDecline:
    def test_three_consecutive_declines_trigger(self):
        """模拟 retry 0=7.0, retry 1=6.5, retry 2=6.0 → 第 3 次 judge 时触发 fallback."""
        p = _new_pipeline()
        # retry 0
        p._update_best_lit("ch1", "v0", {"scores": {"voice": 7.0, "literal": 7.0}})
        # retry 1
        p._update_best_lit("ch1", "v1", {"scores": {"voice": 6.5, "literal": 6.5}})
        # retry 2 (judge 时检查 current_score)
        current_avg = p._compute_avg_score({"scores": {"voice": 6.0, "literal": 6.0}})
        assert p._is_declining_streak("ch1", current_avg, consecutive=3) is True

    def test_no_declining_means_no_trigger(self):
        """分数波动但未持续下降 → 不触发."""
        p = _new_pipeline()
        p._update_best_lit("ch1", "v0", {"scores": {"voice": 6.0}})
        p._update_best_lit("ch1", "v1", {"scores": {"voice": 6.5}})  # 上升
        current_avg = p._compute_avg_score({"scores": {"voice": 6.2}})  # 微降
        assert p._is_declining_streak("ch1", current_avg, consecutive=3) is False

    def test_zero_filtered_at_update_time(self):
        """空 scores (avg=0.0) 在 _update_best_lit 阶段被过滤, 不会进入 history.
        后续 streak 检测基于过滤后的 history 正常运行."""
        p = _new_pipeline()
        p._update_best_lit("ch1", "v0", {"scores": {"voice": 7.0}})
        p._update_best_lit("ch1", "v1", {"scores": {}})  # 0.0 不写入 history
        p._update_best_lit("ch1", "v2", {"scores": {"voice": 6.0}})
        # 实际 history = [7.0, 6.0] (2 个有效样本)
        assert p._score_history["ch1"] == [7.0, 6.0]
        # hist[-2:] = [7.0, 6.0] + [current 5.0] = [7.0, 6.0, 5.0] 严格递减 → True
        current_avg = p._compute_avg_score({"scores": {"voice": 5.0}})
        assert p._is_declining_streak("ch1", current_avg, consecutive=3) is True
        # 但若用 consecutive=4, 样本不足 → False
        assert p._is_declining_streak("ch1", current_avg, consecutive=4) is False


# ---------------------------------------------------------------------------
# (4) 集成回归: Round 93 集成到 _process_judging_batch
# ---------------------------------------------------------------------------

class TestIntegrationRegression:
    def test_pipeline_has_is_declining_streak_method(self):
        from src.pipeline import TranslationPipeline
        assert hasattr(TranslationPipeline, "_is_declining_streak")

    def test_pipeline_has_score_history_state(self):
        """__init__ 中必须初始化 _score_history."""
        from src.pipeline import TranslationPipeline
        import inspect
        src = inspect.getsource(TranslationPipeline.__init__)
        assert "_score_history" in src, (
            "Round 93 要求 __init__ 初始化 _score_history, 实际 init 源码中未找到"
        )

    def test_judge_path_calls_is_declining_streak(self):
        """_process_judging_batch REJECT 路径前必须调 _is_declining_streak."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "_is_declining_streak(" in content, (
            "Round 93 要求 judge REJECT 路径前调用 _is_declining_streak"
        )
        # 检查 SCORE_DECLINE_STREAK 标识字符串
        assert "SCORE_DECLINE_STREAK" in content, (
            "Round 93 要求 fallback 标识用 SCORE_DECLINE_STREAK"
        )

    def test_no_breaking_round_92_best_lit(self):
        """Round 93 不得破坏 Round 92 best_lit 行为."""
        from src.pipeline import TranslationPipeline
        import inspect
        src = inspect.getsource(TranslationPipeline)
        assert "_get_best_lit" in src, "Round 92 helper 必须保留"
        assert "_best_lit" in src, "Round 92 状态必须保留"
