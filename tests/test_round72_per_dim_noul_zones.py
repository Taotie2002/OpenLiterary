"""Round 72: per-dimension Noul uncertain zone, 让 literal_residue / semantic_loss
在边界值被跳过, 抑制 Jev/Mock 启发式边缘噪声引起的 rewriter 死循环.

根因: Round 60 引入了全局 noul_uncertain zone [0.3, 0.7], 仅 entity_consistency 受益.
      literal_residue / semantic_loss 在 MockBackend 启发式下经常产出 0.71-0.74 边缘值,
      刚好越过 0.7 边界触发 reject, 触发 rewriter 反复重写却无法改进 (ch32 实测 6 次全失败).

修复: 在 typesafe_thresholds 配置中按维度设置 noul_uncertain_zones, verdict 合成时优先采用
      per-dim zone, 未配置走全局 fallback. 默认推荐:
        entity_consistency : [0.3, 0.7] (硬约束, 严格区间)
        literal_residue    : [0.3, 0.8] (放宽, CJK 噪声较大)
        semantic_loss      : [0.3, 0.75] (适度放宽)
"""
from __future__ import annotations

import sys

import pytest

from src.judges.verdict import synthesize_verdict
from src.judges.typesafe_spec import NoulAnswer


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
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


@pytest.fixture
def per_dim_thresholds():
    return {
        "noul_uncertain_zones": {
            "entity_consistency": [0.3, 0.7],
            "literal_residue": [0.3, 0.8],
            "semantic_loss": [0.3, 0.75],
        }
    }


class TestPerDimZones:
    """per-dimension 区间覆盖全局区间, 各自独立判定 uncertain."""

    def test_literal_residue_relaxed_zone(self, per_dim_thresholds):
        """literal_residue 0.71 应在放宽区间 [0.3, 0.8] 内被跳过."""
        v = synthesize_verdict(
            {"literal_residue": NoulAnswer(noul=0.71, confidence=0.0)},
            thresholds=per_dim_thresholds,
        )
        assert v.decision == "PASS"
        assert "literal_residue" in v.skipped_dimensions
        # 跳过原因标注 per-dim
        assert "(per-dim)" in v.skipped_reasons["literal_residue"]

    def test_literal_residue_outside_relaxed_zone_still_rejects(self, per_dim_thresholds):
        """literal_residue 0.81 > 0.8, 仍应 reject."""
        v = synthesize_verdict(
            {"literal_residue": NoulAnswer(noul=0.81, confidence=0.0)},
            thresholds=per_dim_thresholds,
        )
        assert v.decision == "REJECT"
        assert any("直译残留" in r for r in v.reject_reasons)

    def test_entity_consistency_keeps_strict_zone(self, per_dim_thresholds):
        """entity_consistency 0.71 > 0.7, 严格区间下应 reject (硬约束)."""
        v = synthesize_verdict(
            {"entity_consistency": NoulAnswer(noul=0.71, confidence=0.0)},
            thresholds=per_dim_thresholds,
        )
        assert v.decision == "REJECT"
        assert any("实体一致性" in r for r in v.reject_reasons)

    def test_entity_consistency_in_zone_skipped(self, per_dim_thresholds):
        """entity_consistency 0.39 < 0.7 应跳过."""
        v = synthesize_verdict(
            {"entity_consistency": NoulAnswer(noul=0.39, confidence=0.0)},
            thresholds=per_dim_thresholds,
        )
        assert v.decision == "PASS"
        assert "entity_consistency" in v.skipped_dimensions

    def test_semantic_loss_moderate_zone(self, per_dim_thresholds):
        """semantic_loss 0.74 应在 [0.3, 0.75] 内被跳过."""
        v = synthesize_verdict(
            {"semantic_loss": NoulAnswer(noul=0.74, confidence=0.0)},
            thresholds=per_dim_thresholds,
        )
        assert v.decision == "PASS"
        assert "semantic_loss" in v.skipped_dimensions

    def test_semantic_loss_above_zone_rejects(self, per_dim_thresholds):
        """semantic_loss 0.76 > 0.75, 应 reject."""
        v = synthesize_verdict(
            {"semantic_loss": NoulAnswer(noul=0.76, confidence=0.0)},
            thresholds=per_dim_thresholds,
        )
        assert v.decision == "REJECT"


class TestUnconfiguredDimensionsFallback:
    """未配置 per-dim zone 的维度走全局 [0.3, 0.7]."""

    def test_voice_drift_falls_back_to_global(self):
        """voice_drift 是 ChoiceAnswer, 不参与 noul uncertain zone. 仅验证不爆."""
        from src.judges.typesafe_spec import ChoiceAnswer
        v = synthesize_verdict({
            "voice_drift": ChoiceAnswer(
                choice="subtle",
                probabilities={"none": 0.5, "subtle": 0.4, "severe": 0.1},
                confidence=0.7,
            )
        })
        assert v.decision in ("PASS", "REJECT")

    def test_unknown_noul_dimension_uses_global_zone(self):
        """未在 noul_uncertain_zones 配的 noul 维度走全局 [0.3, 0.7].

        当前 verdict 只对 entity_consistency / literal_residue / semantic_loss 三个
        Noul 维度做 reject 判定, 其它未知 Noul 维度默认 PASS (无对应 reject branch).
        此测试验证: unknown noul qid 不被错误归到 (per-dim), 走全局 zone 判定.
        """
        v = synthesize_verdict(
            {"some_other_noul": NoulAnswer(noul=0.71, confidence=0.0)},
            thresholds={"noul_uncertain_zones": {"literal_residue": [0.3, 0.8]}},
        )
        # 未知 qid 不会触发 reject 分支, 但也不会被错误归为 per-dim skip
        assert v.decision == "PASS"
        assert v.skipped_dimensions == []  # 不应被错误跳过
        assert v.reject_reasons == []  # 不应被 reject


class TestBackwardCompatibility:
    """未配置 noul_uncertain_zones 时, 行为与 Round 60 一致 (全局区间)."""

    def test_no_per_dim_config_uses_global(self):
        """未配 noul_uncertain_zones, 走全局 [0.3, 0.7]."""
        v = synthesize_verdict(
            {"literal_residue": NoulAnswer(noul=0.71, confidence=0.0)},
            thresholds={},  # 无 per-dim 配置
        )
        # literal_residue 0.71 > 全局 0.7, 应 reject (与 Round 60 行为一致)
        assert v.decision == "REJECT"

    def test_empty_per_dim_uses_global(self):
        """noul_uncertain_zones 配空 dict, 走全局."""
        v = synthesize_verdict(
            {"literal_residue": NoulAnswer(noul=0.71, confidence=0.0)},
            thresholds={"noul_uncertain_zones": {}},
        )
        assert v.decision == "REJECT"


class TestDefaultThresholdsIncludeZone:
    """DEFAULT_THRESHOLDS 必须包含 noul_uncertain_zones key (即使空 dict)."""

    def test_default_has_zone_key(self):
        from src.judges.verdict import DEFAULT_THRESHOLDS
        assert "noul_uncertain_zones" in DEFAULT_THRESHOLDS
        assert isinstance(DEFAULT_THRESHOLDS["noul_uncertain_zones"], dict)


class TestConfigYamlWiring:
    """config.yaml typesafe_thresholds 必须包含 per-dim zone 推荐配置."""

    def test_config_has_per_dim_zones(self):
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        zones = cfg["typesafe_thresholds"].get("noul_uncertain_zones")
        assert zones is not None
        assert zones["entity_consistency"] == [0.3, 0.7]
        assert zones["literal_residue"] == [0.3, 0.8]
        assert zones["semantic_loss"] == [0.3, 0.75]


class TestIntegratedScriptExports:
    """整合脚本导出仍包含 synthesize_verdict (per-dim zone 通过 thresholds 传入)."""

    def test_synthesize_verdict_importable(self):
        from src.translator_agent import synthesize_verdict
        assert callable(synthesize_verdict)

    def test_per_dim_zone_works_via_integrated(self):
        """整合脚本的 synthesize_verdict 也支持 per-dim zone."""
        from src.translator_agent import synthesize_verdict
        from src.judges.typesafe_spec import NoulAnswer
        v = synthesize_verdict(
            {"literal_residue": NoulAnswer(noul=0.71, confidence=0.0)},
            thresholds={"noul_uncertain_zones": {"literal_residue": [0.3, 0.8]}},
        )
        assert v.decision == "PASS"
