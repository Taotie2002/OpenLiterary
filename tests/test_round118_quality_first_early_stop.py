"""Round 118 P1-4: 质量优先早停阈值重调测试.

token 成本非首要: 早停阈值宽容一档, 给 rewriter / 停滞换模型更多机会;
同时把严格递减 streak 窗口从硬编码 3 改为 config `score_decline_streak`(默认 4).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from src.pipeline import TranslationPipeline


ROOT = Path(__file__).resolve().parent.parent


class TestConfigQualityFirst:
    def test_config_values_relaxed(self):
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        pipe = cfg["pipeline"]
        assert pipe["max_quality_retries"] >= 5
        assert pipe["cjk_max_retries"] >= 3
        assert pipe["entity_stagnation_threshold"] >= 3
        assert pipe["no_progress_window"] >= 4
        assert pipe["score_decline_streak"] >= 4


class TestDeclineStreakConfig:
    def _pipeline_with_history(self):
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline._score_history = {"ch01_chunk001": [9.0, 8.0, 7.0]}
        return pipeline

    def test_streak_window_from_config(self):
        pipeline = self._pipeline_with_history()
        # consecutive=4: [9,8,7] + current 6 → 严格递减 → True
        assert pipeline._is_declining_streak("ch01_chunk001", 6.0, consecutive=4) is True
        # consecutive=5: 样本不足 → False (旧硬编码 3 会误触发)
        assert pipeline._is_declining_streak("ch01_chunk001", 6.0, consecutive=5) is False

    def test_source_wires_config_value(self):
        src = (ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
        assert 'score_decline_streak' in src
        assert 'consecutive=decline_streak' in src
