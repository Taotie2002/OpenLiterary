"""Round 92: 最佳文学稿 fallback — 跨 retry 选最高分版本.

背景:
  当 rewriter 重试到 max_quality_retries (默认 6) 仍被 critic/judge 拒, 或者 stagnation/
  oscillation / safety_net 触发 fallback 时, pipeline.py 写入 final 的是当前被拒的
  `lit_text` (即最后一版润色稿). 问题是:
    - 最后一版可能是最差一版 (rewriter 已经"忘了"前面的好版本)
    - 早期 retry 可能 avg_score 更高但被拒, 用户却看不到

修复:
  pipeline.TranslationPipeline 增加实例状态 `self._best_lit: Dict[chunk_id, (avg, text)]`,
  每次 critic pass 后调用 `_update_best_lit(chunk_id, lit_text, critic_report)`.
  5 个 fallback / 1 个 pass 路径都改为 `_get_best_lit(chunk_id, lit_text)` 拿最佳版.

保护目标:
  - _update_best_lit 在 avg_score > 当前 best 时覆盖
  - _update_best_lit 在首次添加时无条件写入
  - _get_best_lit 无 best 时返回 fallback_text
  - 跨多次 retry 后, _best_lit 保留最高分版 (而非最后版)
  - _compute_avg_score: scores 空时返回 0.0; 部分非数值时只算数值项
  - 不会因 best_text 是 dict 而报错
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
# (1) _compute_avg_score
# ---------------------------------------------------------------------------

class TestComputeAvgScore:
    def test_empty_returns_zero(self):
        from src.pipeline import TranslationPipeline
        assert TranslationPipeline._compute_avg_score({}) == 0.0
        assert TranslationPipeline._compute_avg_score({"scores": {}}) == 0.0

    def test_non_dict_returns_zero(self):
        from src.pipeline import TranslationPipeline
        assert TranslationPipeline._compute_avg_score(None) == 0.0
        assert TranslationPipeline._compute_avg_score("not a dict") == 0.0
        assert TranslationPipeline._compute_avg_score(42) == 0.0

    def test_all_numeric_averages_correctly(self):
        from src.pipeline import TranslationPipeline
        cr = {"scores": {"voice": 7.5, "literal_residue": 4.0, "semantic": 6.0}}
        assert TranslationPipeline._compute_avg_score(cr) == pytest.approx((7.5 + 4.0 + 6.0) / 3)

    def test_mixed_numeric_and_string_skips_non_numeric(self):
        from src.pipeline import TranslationPipeline
        cr = {"scores": {"voice": 7.5, "literal_residue": 4.0, "comments": "N/A"}}
        assert TranslationPipeline._compute_avg_score(cr) == pytest.approx((7.5 + 4.0) / 2)

    def test_zero_scores_returns_zero(self):
        from src.pipeline import TranslationPipeline
        cr = {"scores": {"voice": 0.0, "literal_residue": 0.0}}
        assert TranslationPipeline._compute_avg_score(cr) == 0.0


# ---------------------------------------------------------------------------
# (2) _update_best_lit
# ---------------------------------------------------------------------------

class TestUpdateBestLit:
    def _new_pipeline(self):
        """构造一个 TranslationPipeline 实例但跳过 __init__ (避免 DB)."""
        from src.pipeline import TranslationPipeline
        p = TranslationPipeline.__new__(TranslationPipeline)
        p._best_lit = {}
        # Round 93: 同步初始化 score 历史 (避免 _update_best_lit 触发 AttributeError)
        p._score_history = {}
        return p

    def test_first_add_writes_unconditionally(self):
        p = self._new_pipeline()
        p._update_best_lit("ch1", "text1", {"scores": {"voice": 7.0}})
        assert "ch1" in p._best_lit
        avg, text = p._best_lit["ch1"]
        assert avg == pytest.approx(7.0)
        assert text == "text1"

    def test_higher_score_overwrites(self):
        p = self._new_pipeline()
        p._update_best_lit("ch1", "text1", {"scores": {"voice": 5.0}})
        p._update_best_lit("ch1", "text2", {"scores": {"voice": 8.0}})
        avg, text = p._best_lit["ch1"]
        assert avg == pytest.approx(8.0)
        assert text == "text2"

    def test_lower_score_preserves_existing(self):
        p = self._new_pipeline()
        p._update_best_lit("ch1", "text1", {"scores": {"voice": 8.0}})
        p._update_best_lit("ch1", "text2", {"scores": {"voice": 5.0}})
        avg, text = p._best_lit["ch1"]
        # 旧版 (8.0) 保留, 不会因新 retry 分低而覆盖
        assert avg == pytest.approx(8.0)
        assert text == "text1"

    def test_equal_score_keeps_existing(self):
        """相等 score 时不覆盖 (避免无意义写入)."""
        p = self._new_pipeline()
        p._update_best_lit("ch1", "text1", {"scores": {"voice": 7.0}})
        p._update_best_lit("ch1", "text2", {"scores": {"voice": 7.0}})
        _, text = p._best_lit["ch1"]
        assert text == "text1"

    def test_separate_chunks_independent(self):
        p = self._new_pipeline()
        p._update_best_lit("ch1", "text1", {"scores": {"voice": 8.0}})
        p._update_best_lit("ch2", "text2", {"scores": {"voice": 6.0}})
        assert p._best_lit["ch1"][1] == "text1"
        assert p._best_lit["ch2"][1] == "text2"


# ---------------------------------------------------------------------------
# (3) _get_best_lit
# ---------------------------------------------------------------------------

class TestGetBestLit:
    def _new_pipeline(self):
        from src.pipeline import TranslationPipeline
        p = TranslationPipeline.__new__(TranslationPipeline)
        p._best_lit = {}
        # Round 93: 同步初始化 score 历史 (避免 _update_best_lit 触发 AttributeError)
        p._score_history = {}
        return p

    def test_no_best_returns_fallback(self):
        p = self._new_pipeline()
        result = p._get_best_lit("ch1", "current_lit")
        assert result == "current_lit"

    def test_with_best_returns_best(self):
        p = self._new_pipeline()
        p._best_lit["ch1"] = (8.5, "best_text")
        result = p._get_best_lit("ch1", "current_lit")
        assert result == "best_text"

    def test_empty_text_in_best_falls_back(self):
        """best entry 存在但 text 为空 → 退回 fallback (防御 None/空字符串)."""
        p = self._new_pipeline()
        p._best_lit["ch1"] = (8.5, "")
        result = p._get_best_lit("ch1", "current_lit")
        assert result == "current_lit"


# ---------------------------------------------------------------------------
# (4) 集成: 跨 retry 后选最高分版
# ---------------------------------------------------------------------------

class TestBestLitAcrossRetries:
    def _new_pipeline(self):
        from src.pipeline import TranslationPipeline
        p = TranslationPipeline.__new__(TranslationPipeline)
        p._best_lit = {}
        # Round 93: 同步初始化 score 历史 (避免 _update_best_lit 触发 AttributeError)
        p._score_history = {}
        return p

    def test_max_score_wins_among_oscillating_retries(self):
        """模拟: retry 0 = 6.5, retry 1 = 8.2, retry 2 = 7.0 → fallback 应选 retry 1."""
        p = self._new_pipeline()
        p._update_best_lit("ch1", "v0", {"scores": {"voice": 6.5}})
        p._update_best_lit("ch1", "v1", {"scores": {"voice": 8.2}})
        p._update_best_lit("ch1", "v2", {"scores": {"voice": 7.0}})

        # 此时 rewriter 已被拒, lit_text="v2", 触发 fallback
        result = p._get_best_lit("ch1", "v2")
        assert result == "v1", (
            f"Round 92 修复目标: fallback 用最高分版 v1 (8.2), 而非当前被拒 v2 (7.0). "
            f"实际返回: {result!r}"
        )

    def test_first_retry_score_wins_if_subsequent_lower(self):
        """retry 0 高分, retry 1-2 越改越差 → fallback 用 retry 0."""
        p = self._new_pipeline()
        p._update_best_lit("ch1", "v0_high", {"scores": {"voice": 9.0}})
        p._update_best_lit("ch1", "v1_med", {"scores": {"voice": 6.0}})
        p._update_best_lit("ch1", "v2_low", {"scores": {"voice": 4.0}})

        result = p._get_best_lit("ch1", "v2_low")
        assert result == "v0_high"

    def test_empty_scores_handled(self):
        """critic 返回空 scores → avg=0.0 → 仍写入 best, 但分数低可能被后续覆盖."""
        p = self._new_pipeline()
        p._update_best_lit("ch1", "v0_no_score", {"scores": {}})  # avg=0.0
        p._update_best_lit("ch1", "v1", {"scores": {"voice": 7.0}})
        result = p._get_best_lit("ch1", "v1")
        assert result == "v1"


# ---------------------------------------------------------------------------
# (5) 回归保护: Round 92 集成到 5 个 fallback / 1 个 pass 路径
# ---------------------------------------------------------------------------

class TestFallbackPathsIntegration:
    """验证 pipeline.py 中 5 个 fallback + 1 个 pass 路径都经最佳稿选择定稿.

    Round 118 P2-8: 定稿调用点统一由 _get_best_lit 升级为 _pairwise_pick_best
    （内部先 _get_best_lit 取最高分版本, Jev 仲裁不可用/平票时原样回退）。
    """

    def test_all_fallback_paths_use_best_lit_selection(self):
        """grep pipeline.py, 数定稿选稿调用点 ≥ 6 (5 fallback + 1 pass)."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        count = content.count("self._pairwise_pick_best(")
        assert count >= 6, (
            f"Round 118 要求至少 6 个 _pairwise_pick_best 调用点 (5 fallback + 1 pass). "
            f"实际: {count}. 漏掉某些 fallback 路径会导致 chunk 用低分版本定稿."
        )
        # _pairwise_pick_best 内部必须回退 _get_best_lit — 保证 Jev 不可用时 Round 92 行为不变
        assert "best = self._get_best_lit(chunk_id, fallback_text)" in content, (
            "_pairwise_pick_best 内部未回退 _get_best_lit, 仲裁不可用时会破坏 Round 92 行为"
        )

    def test_update_best_lit_called_after_critic_save(self):
        """critic save 之后必须有 _update_best_lit 调用 (否则 best 永远空)."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "self._update_best_lit(" in content, (
            "_update_best_lit 未在 pipeline.py 调用, fallback 永远用 fallback_text 而非 best"
        )
        # 调用点必须在 critic save 之后 (粗略检查: 出现在 _save_intermediate("critic_report" 之后)
        save_idx = content.find('self._save_intermediate(chunk_id, "critic_report"')
        update_idx = content.find("self._update_best_lit(")
        assert save_idx < update_idx, (
            f"_update_best_lit 必须在 critic_report save 之后调用, "
            f"save_idx={save_idx} update_idx={update_idx}"
        )
