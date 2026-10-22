"""Round 88: per-dim uncertain zone 冗余检查修复.

背景:
  Round 72 在 verdict.py 加了 first block: NoulAnswer 在 per-dim zone (若无配
  置走全局) 内 → SKIP (continue), 否则 fall through.
  但 entity_consistency 分支的 second block 仍冗余地用 GLOBAL zone 重新检查.
  
  Bug: 当用户配置 NARROWER per-dim zone (如 entity_consistency: [0.4, 0.6])
  全局 zone (默认 [0.3, 0.7]), noul 落在 per-dim 外但在全局内 (例 noul=0.35)
  时:
    - First block: 0.35 NOT in [0.4, 0.6] → don't skip
    - Second block (entity_consistency 分支): 0.35 IS in [0.3, 0.7] → pass (skip)
  
  per-dim zone 的语义是"用此维度的 zone 完全替代全局", 但 second block 用全局
  zone 覆盖了 per-dim 配置, 导致 shadow zone 内该 reject 的没 reject.

当前 config.yaml 中所有 per-dim zone 都比全局宽 (literal_residue [0.3, 0.8],
semantic_loss [0.3, 0.75]), 所以 bug 潜伏. 但用户任意把 per-dim 配窄就触发.

修复:
  - 移除 entity_consistency second block 中的全局 zone 冗余检查
  - first block 已经处理"in zone = skip", second block 只处理"in zone 之外"
  - 对 literal_residue / semantic_loss: 没有 per-zone 二次检查 (Round 72 已经按设计
    处理, 不会有此 bug)
  - 行为: per-dim zone 配置后, 该维度严格按 per-dim 走, 不被全局 zone 影响

验证:
  - per-dim zone [0.4, 0.6] (比全局 [0.3, 0.7] 窄), noul=0.35 → 应 reject
  - per-dim zone [0.4, 0.6] (比全局 [0.3, 0.7] 窄), noul=0.65 → 应 reject
  - per-dim zone [0.4, 0.6] (比全局 [0.3, 0.7] 窄), noul=0.50 → 应 skip (在 zone 内)
  - per-dim zone [0.4, 0.6] (比全局 [0.3, 0.7] 窄), noul=0.30 → 应 reject (低于 zone)
  - per-dim zone [0.3, 0.8] (比全局 [0.3, 0.7] 宽, 当前配置), 保持 Round 72 行为
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



def _make_answers(noul_entity: float):
    """构造包含 entity_consistency noul 的标准 answers dict."""
    from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer
    return {
        "voice_drift": ChoiceAnswer(
            choice="none",
            probabilities={"none": 0.7, "subtle": 0.2, "severe": 0.1},
            confidence=0.7,
        ),
        "register_match": ChoiceAnswer(
            choice="match",
            probabilities={"match": 0.8, "under": 0.1, "over": 0.1},
            confidence=0.8,
        ),
        "entity_consistency": NoulAnswer(noul=noul_entity),
        "literal_residue": NoulAnswer(noul=0.0),
        "semantic_loss": NoulAnswer(noul=0.0),
        "style_compliance_score": ScoreAnswer(
            score=2.0,
            probabilities={0: 0.0, 1: 0.0, 2: 1.0},
            confidence=1.0,
        ),
    }


def _thresholds(per_dim_zones=None):
    return {
        "noul_uncertain_low": 0.3,
        "noul_uncertain_high": 0.7,
        "noul_uncertain_zones": per_dim_zones or {},
    }


def test_per_dim_zone_narrow_below_zone_should_reject():
    """per-dim zone [0.4, 0.6] 比全局 [0.3, 0.7] 窄, noul=0.35 应 reject.

    Round 88 修复前: second block 的全局 zone check 把 noul=0.35 当作 "in
    zone", 跳过 reject, 用户配置 per-dim 完全失效.
    Round 88 修复后: per-dim zone 完全替代全局, 0.35 < 0.4 是 zone 下方 →
    不在 zone 内 → 检查 entity_consistency_noul=0.3 → 0.35 >= 0.3 → REJECT.
    """
    from src.judges.verdict import synthesize_verdict

    v = synthesize_verdict(
        _make_answers(0.35),
        thresholds=_thresholds({"entity_consistency": [0.4, 0.6]}),
    )
    assert v.decision == "REJECT", (
        f"per-dim zone [0.4, 0.6] + noul=0.35: 应 reject (0.35 < 0.4 不是 zone "
        f"内, 且 0.35 >= entity_consistency_noul=0.3). Got decision={v.decision}, "
        f"reasons={v.reject_reasons}"
    )
    assert any("实体一致性" in r for r in v.reject_reasons), (
        f"应包含实体一致性 reject reason. Got: {v.reject_reasons}"
    )
    assert "entity_consistency" not in v.skipped_dimensions


def test_per_dim_zone_narrow_above_zone_should_reject():
    """per-dim zone [0.4, 0.6] 窄, noul=0.65 应 reject.

    Round 88 修复前: 0.65 NOT in [0.4, 0.6] (first block don't skip) → second
    block 检查全局 [0.3, 0.7] → 0.65 IN global → pass (no reject).
    Round 88 修复后: 0.65 > 0.6 不在 per-dim zone 内 → 检查阈值 → REJECT.
    """
    from src.judges.verdict import synthesize_verdict

    v = synthesize_verdict(
        _make_answers(0.65),
        thresholds=_thresholds({"entity_consistency": [0.4, 0.6]}),
    )
    assert v.decision == "REJECT"
    assert any("实体一致性" in r for r in v.reject_reasons)
    assert "entity_consistency" not in v.skipped_dimensions


def test_per_dim_zone_narrow_in_zone_should_skip():
    """per-dim zone [0.4, 0.6] 窄, noul=0.50 应 skip (在 zone 内).

    这是 Round 72 设计本意: zone 内的 noul 视为不确定, 不触发 reject.
    Round 88 修复不应影响此行为.
    """
    from src.judges.verdict import synthesize_verdict

    v = synthesize_verdict(
        _make_answers(0.50),
        thresholds=_thresholds({"entity_consistency": [0.4, 0.6]}),
    )
    assert v.decision == "PASS"
    assert "entity_consistency" in v.skipped_dimensions
    assert "(per-dim)" in v.skipped_reasons["entity_consistency"]


def test_per_dim_zone_wide_keeps_round72_behavior():
    """per-dim zone [0.3, 0.8] (比全局 [0.3, 0.7] 宽, 当前 config), noul=0.75 应 skip.

    Round 72 设计: per-dim zone 完全替代全局. 修复后行为不变.
    """
    from src.judges.verdict import synthesize_verdict

    v = synthesize_verdict(
        _make_answers(0.75),
        thresholds=_thresholds({"entity_consistency": [0.3, 0.8]}),
    )
    assert v.decision == "PASS"
    assert "entity_consistency" in v.skipped_dimensions


def test_per_dim_zone_narrow_zone_plus_thr_means_strict():
    """per-dim zone [0.4, 0.6] 配 entity_consistency_noul=0.5, noul=0.62 应 reject.

    noul=0.62 NOT in [0.4, 0.6] (above zone, 0.62 > 0.6). 检查 0.62 >= 0.5 → REJECT.
    这证明 per-dim zone 严格控制 uncertainty 范围 (不会因为全局 zone 而扩张).
    """
    from src.judges.verdict import synthesize_verdict

    v = synthesize_verdict(
        _make_answers(0.62),
        thresholds={
            "entity_consistency_noul": 0.5,
            "noul_uncertain_low": 0.3,
            "noul_uncertain_high": 0.7,
            "noul_uncertain_zones": {"entity_consistency": [0.4, 0.6]},
        },
    )
    assert v.decision == "REJECT"
    assert any("实体一致性" in r for r in v.reject_reasons)


def test_no_per_dim_zone_falls_back_to_global():
    """未配置 per-dim zone → 走全局. noul=0.35 ∈ [0.3, 0.7] → skip."""
    from src.judges.verdict import synthesize_verdict

    v = synthesize_verdict(
        _make_answers(0.35),
        thresholds=_thresholds(),  # 无 per-dim zones
    )
    assert v.decision == "PASS"
    assert "entity_consistency" in v.skipped_dimensions
    assert "(global)" in v.skipped_reasons["entity_consistency"]


def test_first_block_skip_uses_per_dim_zone_for_skipped_reason():
    """First block 的 skipped_reason 必须标明是 per-dim 还是 global, 不是 hard-code."""
    from src.judges.verdict import synthesize_verdict

    v = synthesize_verdict(
        _make_answers(0.5),
        thresholds=_thresholds({"entity_consistency": [0.4, 0.6]}),
    )
    assert v.skipped_reasons["entity_consistency"].endswith("(per-dim)")


def test_other_dimensions_unaffected_by_per_dim_zone_fix():
    """per-dim zone 只影响 entity_consistency, 不影响 literal_residue / semantic_loss.

    Round 88 修复不涉及 literal_residue / semantic_loss 的 second block (它们
    没有冗余 zone 检查). 回归测试确认它们行为不变.
    """
    from src.judges.typesafe_spec import NoulAnswer
    from src.judges.verdict import synthesize_verdict

    answers = {
        "voice_drift": __import__("src.judges.typesafe_spec", fromlist=["ChoiceAnswer"]).ChoiceAnswer(
            choice="none",
            probabilities={"none": 0.7, "subtle": 0.2, "severe": 0.1},
            confidence=0.7,
        ),
        "register_match": __import__("src.judges.typesafe_spec", fromlist=["ChoiceAnswer"]).ChoiceAnswer(
            choice="match",
            probabilities={"match": 0.8, "under": 0.1, "over": 0.1},
            confidence=0.8,
        ),
        "entity_consistency": NoulAnswer(noul=0.0),
        "literal_residue": NoulAnswer(noul=0.75),  # in per-dim [0.3, 0.8] → skip
        "semantic_loss": NoulAnswer(noul=0.72),  # in per-dim [0.3, 0.75] → skip
        "style_compliance_score": __import__("src.judges.typesafe_spec", fromlist=["ScoreAnswer"]).ScoreAnswer(
            score=2.0,
            probabilities={0: 0.0, 1: 0.0, 2: 1.0},
            confidence=1.0,
        ),
    }
    v = synthesize_verdict(answers, thresholds=_thresholds({
        "literal_residue": [0.3, 0.8],
        "semantic_loss": [0.3, 0.75],
    }))
    assert "literal_residue" in v.skipped_dimensions
    assert "semantic_loss" in v.skipped_dimensions
    assert v.decision == "PASS"
