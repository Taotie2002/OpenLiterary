"""Round 83: NoulAnswer.confidence 自动从 noul 派生, 消除与 to_dict() 的语义不一致.

修复前 BUG:
  - NoulAnswer(noul=0.39) → ans.confidence == 0.0 (dataclass 默认)
  - 但 ans.to_dict()["confidence"] == 0.61 (max(0.39, 1-0.39))
  - 下游代码若读 ans.confidence 直接 (绕过 to_dict), 会得到 conf=0.00 误导
    (让人误以为 Jev "毫无把握", 实际是 CJK 边界概率的正常表达)
  - 真实 run log (ch32) 显示 cost_report reject 字段: "直译残留: noul=0.53 conf=0.00"
    —— 该 conf 字段本应是 0.53 (max(0.53, 0.47))

修复:
  - NoulAnswer.__post_init__ 自动用 max(noul, 1-noul) 填充默认 confidence
  - 显式传 confidence > 0 时不被覆盖 (向后兼容 Round 66/67 mock 测试)
  - to_dict() 直接用 self.confidence (消除重复计算)

验证:
  - NoulAnswer(noul=0.39).confidence == 0.61 (eff_conf, 自动)
  - NoulAnswer(noul=0.5).confidence == 0.5 (eff_conf, 自动)
  - NoulAnswer(noul=0.7, confidence=0.3).confidence == 0.3 (显式, 不覆盖)
  - to_dict()["confidence"] == self.confidence (consistency)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.judges.typesafe_spec import NoulAnswer


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
# Round 83: __post_init__ 自动派生 confidence
# ---------------------------------------------------------------------------

class TestConfidenceAutoDerive:
    """NoulAnswer.confidence 默认从 noul 派生 (max(noul, 1-noul))."""

    def test_low_noul_eff_conf_is_complement(self):
        # noul=0.39 → eff_conf = max(0.39, 0.61) = 0.61
        ans = NoulAnswer(noul=0.39)
        assert ans.confidence == pytest.approx(0.61, abs=1e-6)

    def test_high_noul_eff_conf_is_noul(self):
        # noul=0.8 → eff_conf = max(0.8, 0.2) = 0.8
        ans = NoulAnswer(noul=0.8)
        assert ans.confidence == pytest.approx(0.8, abs=1e-6)

    def test_noul_half_eff_conf_is_half(self):
        # noul=0.5 → eff_conf = max(0.5, 0.5) = 0.5 (最不确定)
        ans = NoulAnswer(noul=0.5)
        assert ans.confidence == pytest.approx(0.5, abs=1e-6)

    def test_noul_zero_eff_conf_is_one(self):
        # noul=0.0 → eff_conf = max(0.0, 1.0) = 1.0
        ans = NoulAnswer(noul=0.0)
        assert ans.confidence == pytest.approx(1.0, abs=1e-6)

    def test_noul_one_eff_conf_is_one(self):
        # noul=1.0 → eff_conf = max(1.0, 0.0) = 1.0
        ans = NoulAnswer(noul=1.0)
        assert ans.confidence == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Round 83: 显式传 confidence > 0 不被覆盖 (向后兼容)
# ---------------------------------------------------------------------------

class TestExplicitConfidencePreserved:
    """显式传 confidence > 0 时, __post_init__ 不覆盖."""

    def test_explicit_confidence_preserved_low(self):
        # Round 66 mock 测试场景: confidence=0.3 (显式)
        ans = NoulAnswer(noul=0.7, confidence=0.3)
        assert ans.confidence == pytest.approx(0.3, abs=1e-6)
        assert ans.noul == pytest.approx(0.7, abs=1e-6)

    def test_explicit_confidence_preserved_high(self):
        # 测试场景: confidence=0.95 (显式)
        ans = NoulAnswer(noul=0.5, confidence=0.95)
        assert ans.confidence == pytest.approx(0.95, abs=1e-6)


# ---------------------------------------------------------------------------
# Round 83: to_dict() 与 ans.confidence 一致性
# ---------------------------------------------------------------------------

class TestToDictConsistency:
    """to_dict()["confidence"] == ans.confidence (无歧义)."""

    def test_to_dict_confidence_matches_field(self):
        # 关键修复: 之前 to_dict 计算 eff_conf 而 ans.confidence = 0.0
        # 现在两者必须一致
        ans = NoulAnswer(noul=0.53)
        d = ans.to_dict()
        assert d["confidence"] == ans.confidence

    def test_to_dict_returns_correct_eff_conf(self):
        ans = NoulAnswer(noul=0.53)
        d = ans.to_dict()
        assert d["confidence"] == pytest.approx(0.53, abs=1e-6)  # max(0.53, 0.47)

    def test_to_dict_for_zero_noul(self):
        ans = NoulAnswer(noul=0.0)
        d = ans.to_dict()
        assert d["confidence"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Round 83: 集成测试 — verdict 路径不受影响 (verdict 已直接用 eff_conf, 不读 confidence)
# ---------------------------------------------------------------------------

class TestVerdictPathUnaffected:
    """verdict.py 仍正确: Noul 答案的 eff_conf = max(noul, 1-noul)."""

    def test_verdict_overall_conf_uses_noul_distance(self):
        from src.judges.verdict import synthesize_verdict
        ans = NoulAnswer(noul=0.3)
        v = synthesize_verdict({"literal_residue": ans})
        # verdict 用 max(0.3, 0.7) = 0.7 聚合 overall_conf
        assert v.confidence == pytest.approx(0.7, abs=1e-2)


# ---------------------------------------------------------------------------
# Round 83: 修复前真实 run log 场景 (回归测试)
# ---------------------------------------------------------------------------

class TestRealticketBugScenario:
    """ch32 实测: reject 字段显示 conf=0.00, 应修复为 conf=max(noul, 1-noul)."""

    def test_user_cost_report_will_show_correct_conf(self):
        # 模拟 reject 原因格式: f"...noul={noul} conf={confidence}"
        ans = NoulAnswer(noul=0.53)  # Jev 返回的 NoulAnswer
        # 修复后: 直接读 ans.confidence 得到 0.53 (而非 0.0)
        assert ans.confidence == pytest.approx(0.53, abs=1e-6)
        # cost_report reject 行不再显示 conf=0.00
        reject_line = f"直译残留: noul={ans.noul:.2f} conf={ans.confidence:.2f}"
        assert "conf=0.53" in reject_line
        assert "conf=0.00" not in reject_line

    def test_semantic_loss_noul_060(self):
        ans = NoulAnswer(noul=0.60)
        assert ans.confidence == pytest.approx(0.60, abs=1e-6)
        reject_line = f"关键语义丢失: noul={ans.noul:.2f} conf={ans.confidence:.2f}"
        assert "conf=0.60" in reject_line


# ---------------------------------------------------------------------------
# Round 83: 整合脚本 src/translator_agent.py 同步修复
# ---------------------------------------------------------------------------

class TestIntegratedScriptSynced:
    """散装 src/judges/typesafe_spec.py 修复必须同步到整合脚本 src/translator_agent.py."""

    def test_round_83_marker_in_scattered(self):
        content = Path("src/judges/typesafe_spec.py").read_text(encoding="utf-8")
        assert "Round 83" in content, "散装 typesafe_spec.py 未标注 Round 83 修复"

    def test_round_83_marker_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 83" in content, "整合脚本未同步 Round 83 修复"

    def test_post_init_present_in_scattered(self):
        content = Path("src/judges/typesafe_spec.py").read_text(encoding="utf-8")
        assert "def __post_init__" in content, (
            "散装 typesafe_spec.py 应有 __post_init__ 方法"
        )

    def test_post_init_present_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "def __post_init__" in content, (
            "整合脚本 src/translator_agent.py 应有 __post_init__ 方法"
        )
