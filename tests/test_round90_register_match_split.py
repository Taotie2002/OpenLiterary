"""Round 90: register_match 拆分 under / over 为具体类别名, 让 _JEV_CATEGORY_STRATEGY 命中.

背景:
  旧版 verdict.py 第 170 行把 register_match 输出合并为一条 reject:
    `语域失配: under+over=0.50 conf=0.95`
  但 `_JEV_CATEGORY_STRATEGY` / `GLOBAL_REWRITE_CATEGORIES` 的 keys 是:
    `语域偏离（under）` / `语域偏离（over）` (具体形式)
  extract_reject_categories 抽出 `语域失配`, lookup miss → 落 generic fallback
  "请针对此维度整体审视并修订。" → 失去 specific 策略 (提升语域 / 降低语域).

修复 (Round 90):
  保持原合并阈值语义 (combined badness >= register_mismatch_prob 仍触发 reject);
  触发后按方向拆分:
    - 任一方向 prob >= threshold/2 (= 0.2) → 输出对应类别
    - 两方向都显著 → 输出两条 reject
    - 仅一方向显著 → 仅输出显著方向 (避免 noise 维度污染 rewriter 策略)

保护目标:
  - 合并阈值不变 (under=0.2 over=0.25 仍 REJECT, 兼容 test_verdict.py 已存测试)
  - 拆出来的两条类别名能命中 _JEV_CATEGORY_STRATEGY 具体策略
  - 拆出来的两条类别名在 stagnation 检测中仍视为不同 category
    (否则 over→over 会被误判为 stagnation 触发强制 fallback, 反向破坏)
  - 与 voice_drift severe 共存 (多 reject 维度独立)
  - 单方向极低 (< 0.2) 时不输出该方向 (避免 noise)
"""
from __future__ import annotations

import sys

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
# (1) verdict.py 输出拆分
# ---------------------------------------------------------------------------

class TestVerdictRegisterMatchSplit:
    def _run(self, probs):
        from src.judges.verdict import synthesize_verdict
        from src.judges.typesafe_spec import ChoiceAnswer
        ans = ChoiceAnswer(choice="match", probabilities=probs, confidence=0.95)
        return synthesize_verdict({"register_match": ans}).reject_reasons

    def test_over_dominant_emits_only_over(self):
        # combined=0.50 >= 0.4, under=0.05 < 0.2 → 仅 over 显著 → 仅输出 over
        reasons = self._run({"under": 0.05, "over": 0.45, "match": 0.50})
        assert reasons == ["语域偏离（over）: over=0.45 conf=0.95"]

    def test_under_dominant_emits_only_under(self):
        reasons = self._run({"under": 0.45, "over": 0.05, "match": 0.50})
        assert reasons == ["语域偏离（under）: under=0.45 conf=0.95"]

    def test_both_significant_emits_two(self):
        # combined=0.95, both >= 0.2 → 两条 reject
        reasons = self._run({"under": 0.50, "over": 0.45, "match": 0.05})
        assert reasons == [
            "语域偏离（under）: under=0.50 conf=0.95",
            "语域偏离（over）: over=0.45 conf=0.95",
        ]

    def test_combined_at_threshold_both_at_dir_threshold(self):
        # combined=0.40 (=threshold), under=0.20 (=dir_thr), over=0.20 (=dir_thr) → 两条
        reasons = self._run({"under": 0.20, "over": 0.20, "match": 0.60})
        assert reasons == [
            "语域偏离（under）: under=0.20 conf=0.95",
            "语域偏离（over）: over=0.20 conf=0.95",
        ]

    def test_combined_below_threshold_passes(self):
        # combined=0.30 < 0.4 → pass (即使各方向 prob 都 >= 0.15)
        reasons = self._run({"under": 0.15, "over": 0.15, "match": 0.70})
        assert reasons == []

    def test_combined_above_one_direction_below_dir_threshold(self):
        # combined=0.45 >= 0.4, over=0.42 >= 0.2, under=0.03 < 0.2 → 仅 over 触发
        reasons = self._run({"under": 0.03, "over": 0.42, "match": 0.55})
        assert reasons == ["语域偏离（over）: over=0.42 conf=0.95"]

    def test_clean_passes(self):
        reasons = self._run({"under": 0.05, "over": 0.10, "match": 0.85})
        assert reasons == []

    def test_old_combined_label_gone(self):
        """旧版合并输出 `语域失配` 必须不再出现."""
        reasons = self._run({"under": 0.45, "over": 0.45, "match": 0.10})
        joined = "; ".join(reasons)
        assert "语域失配" not in joined, (
            f"Round 90 修复后不应再出现 '语域失配' 合并类别. 实际: {joined!r}"
        )

    def test_compat_with_existing_register_match_combined_rejects(self):
        """兼容 tests/judges/test_verdict.py 已存 test_register_match_combined_rejects:
        under=0.20, over=0.25, combined=0.45 >= 0.4 → 必须 REJECT (不破坏合并语义).
        """
        reasons = self._run({"under": 0.20, "over": 0.25, "match": 0.55})
        assert len(reasons) > 0, "旧版 combined>=0.4 应 REJECT, 新版不能破坏此语义"
        # 旧版断言 `语域失配 in reasons` — 新版 `语域失配` 已被替换, 但应该 REJECT.
        joined = "; ".join(reasons)
        assert "语域偏离" in joined, (
            f"拆分后类别应含 `语域偏离`, 实际: {joined!r}"
        )

    def test_custom_threshold_changes_behavior(self):
        """调用方传入 register_mismatch_prob=0.3 时, combined>=0.3 触发."""
        from src.judges.verdict import synthesize_verdict
        from src.judges.typesafe_spec import ChoiceAnswer
        ans = ChoiceAnswer(
            choice="over",
            probabilities={"under": 0.05, "over": 0.32, "match": 0.63},
            confidence=0.95,
        )
        v = synthesize_verdict(
            {"register_match": ans},
            thresholds={"register_mismatch_prob": 0.3},
        )
        # combined=0.37 >= 0.3, dir_thr=0.15, under=0.05<0.15, over=0.32>=0.15 → 仅 over
        assert v.reject_reasons == ["语域偏离（over）: over=0.32 conf=0.95"]

    def test_split_coexists_with_voice_drift_severe(self):
        """register_match 拆分不破坏 voice_drift severe reject."""
        from src.judges.verdict import synthesize_verdict
        from src.judges.typesafe_spec import ChoiceAnswer
        answers = {
            "voice_drift": ChoiceAnswer(
                choice="severe",
                probabilities={"none": 0.10, "subtle": 0.20, "severe": 0.70},
                confidence=0.95,
            ),
            "register_match": ChoiceAnswer(
                choice="over",
                probabilities={"under": 0.05, "over": 0.50, "match": 0.45},
                confidence=0.95,
            ),
        }
        v = synthesize_verdict(answers)
        joined = "; ".join(v.reject_reasons)
        assert "音色漂移（severe）" in joined
        assert "语域偏离（over）" in joined

    def test_noisy_under_filtered_out(self):
        """当 under 方向 prob 极低 (< dir_thr) 时, 不污染策略 (避免 noise 维度)."""
        # combined=0.45 (>= 0.4) 但 under=0.03 (< 0.2) → 仅 over
        reasons = self._run({"under": 0.03, "over": 0.42, "match": 0.55})
        assert "语域偏离（under）" not in "; ".join(reasons), (
            "极低 under prob 不应作为显著方向输出 (噪声维度, 干扰 rewriter)"
        )


# ---------------------------------------------------------------------------
# (2) _JEV_CATEGORY_STRATEGY 命中具体策略
# ---------------------------------------------------------------------------

class TestStrategyLookupHitsSpecific:
    def _build_section(self, reject_reason):
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        return agent._build_jev_category_section(reject_reason)

    def test_over_reason_hits_over_strategy(self):
        section = self._build_section("语域偏离（over）: over=0.45 conf=0.95")
        assert "降低语域" in section
        assert "整体审视并修订" not in section, (
            f"over register strategy miss. section={section!r}"
        )

    def test_under_reason_hits_under_strategy(self):
        section = self._build_section("语域偏离（under）: under=0.45 conf=0.95")
        assert "提升语域" in section
        assert "整体审视并修订" not in section

    def test_both_reasons_hit_both_strategies(self):
        section = self._build_section(
            "语域偏离（under）: under=0.50 conf=0.95; 语域偏离（over）: over=0.45 conf=0.95"
        )
        assert "提升语域" in section
        assert "降低语域" in section

    def test_old_label_falls_back(self):
        """回归证据: 旧版 '语域失配' 仍落 fallback (Round 90 修复是改 verdict 输出)."""
        section = self._build_section("语域失配: under+over=0.50 conf=0.95")
        assert "整体审视并修订" in section, (
            "旧类别 '语域失配' 应继续落 fallback (本测试仅作回归证据, "
            "证明 verdict.py 拆分后旧 label 不可达)"
        )


# ---------------------------------------------------------------------------
# (3) stagnation 检测仍区分 under / over (避免过拟合 stagnation)
# ---------------------------------------------------------------------------

class TestStagnationKeepsUnderOverDistinct:
    def test_under_and_over_are_distinct(self):
        from src.utils.text_match import extract_reject_categories
        under_cats = extract_reject_categories("语域偏离（under）: under=0.45 conf=0.95")
        over_cats = extract_reject_categories("语域偏离（over）: over=0.45 conf=0.95")
        assert under_cats != over_cats, (
            "Round 69 设计: under / over 视为不同 category "
            "(否则 under→under 误判 stagnation, 触发 Round 6 oscillation 强制 fallback)."
        )
        assert "语域偏离（under）" in under_cats
        assert "语域偏离（over）" in over_cats

    def test_under_and_over_serialize_distinctly_in_global_rewrite(self):
        """GLOBAL_REWRITE_CATEGORIES 路径同样要求 under / over 区分."""
        from src.utils.text_match import extract_reject_categories
        under_cats = extract_reject_categories(
            "语域偏离（under）: under=0.45 conf=0.95", normalize=False
        )
        over_cats = extract_reject_categories(
            "语域偏离（over）: over=0.45 conf=0.95", normalize=False
        )
        assert under_cats != over_cats
