"""Round 80: 修复 style_compliance_min_confidence 字段未接线的 bug.

根因: src/judges/verdict.py:64 定义了 style_compliance_min_confidence: 0.6,
      注释 (line 10) 明确写 "style_compliance.score < 1.0 且 confidence >= 0.6 → reject",
      但 synthesize_verdict (line 192) 只检查 score < min_score, 完全忽略 confidence 门控.
      config.yaml typesafe_thresholds.style_compliance_min_confidence 也被静默忽略.

修复: 在 style_compliance_score 判断时, 增加 confidence >= min_confidence 门控.
      低 confidence 的 ScoreAnswer 视为 Jev 边界判断, 不参与 hard reject
      (与 Noul uncertain zone 一致语义, 写入 skipped_dimensions).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _load_module():
    """Round 80: 用 importlib 直加载 src.judges.verdict, 绕过 sys.modules 父包状态.

    verdict.py 依赖 primitives 和 typesafe_spec, 同样需要直加载.
    """
    import importlib.util
    import sys
    # 直加载 primitives
    spec_p = importlib.util.spec_from_file_location(
        "src.judges.primitives", "src/judges/primitives.py"
    )
    mod_p = importlib.util.module_from_spec(spec_p)
    sys.modules["src.judges.primitives"] = mod_p
    spec_p.loader.exec_module(mod_p)
    # 直加载 typesafe_spec
    spec_s = importlib.util.spec_from_file_location(
        "src.judges.typesafe_spec", "src/judges/typesafe_spec.py"
    )
    mod_s = importlib.util.module_from_spec(spec_s)
    sys.modules["src.judges.typesafe_spec"] = mod_s
    spec_s.loader.exec_module(mod_s)
    # 直加载 verdict
    spec = importlib.util.spec_from_file_location(
        "_verdict_isolated", "src/judges/verdict.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolate_modules():
    """Round 103 P7: 不再 pre-pop sys.modules.

    旧版 fixture 在 setup 时 pop("src.config", None) 强制后续 test 重新 import,
    但若 translator_agent 已被其他 test 加载并保留, 其模块级 _config 仍指向旧
    src.config.config 对象 (Python 模块 globals 在首次 import 时绑定), 新 import
    拿到的 src.config.config 是不同实例, 双方 pipeline dict id 不同, 导致
    test_round98 等后续 test mutate `src.config.config.pipeline["X"] = Y` 不影响
    _config.pipeline, 产生跨测试污染.
    Round 103 P6 的 cleanup 逻辑 (动态 polluted + 只 pop fake) 已能正确恢复,
    无需 pre-pop. 此处直接 yield 即可.
    """
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        # 主动 import 让子模块正确注册到 sys.modules
        try:
            import src.judges.typesafe_backend
        except Exception:
            pass
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



class TestSourcePatch:
    """散装 src/judges/verdict.py 必须应用 Round 80 修复."""

    def test_round_80_marker_present(self):
        content = Path("src/judges/verdict.py").read_text(encoding="utf-8")
        assert "Round 80" in content

    def test_min_confidence_gate_present(self):
        """核心修复: 必须在 style_compliance_score 判断时增加 confidence 门控."""
        content = Path("src/judges/verdict.py").read_text(encoding="utf-8")
        # 找 ScoreAnswer 处理段
        idx = content.find("isinstance(ans, ScoreAnswer)")
        assert idx > 0
        block = content[idx:idx + 1500]
        # 必须出现 style_compliance_min_confidence 的读取
        assert "style_compliance_min_confidence" in block, (
            "Round 80: style_compliance_min_confidence 字段必须在 ScoreAnswer 处理段被读取"
        )
        # 必须有低 conf 跳过逻辑 (skipped_dimensions.append)
        assert "skipped_dimensions.append(qid)" in block, (
            "Round 80: 低 confidence 的 ScoreAnswer 必须加入 skipped_dimensions"
        )

    def test_syntax_check(self):
        import ast
        ast.parse(Path("src/judges/verdict.py").read_text(encoding="utf-8"))


class TestIntegratedScript:
    """整合脚本 src/translator_agent.py 必须同步应用 Round 80 修复."""

    def test_synthesize_verdict_importable(self):
        from src.translator_agent import synthesize_verdict
        assert synthesize_verdict is not None

    def test_round_80_present_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 80" in content, "整合脚本未同步 Round 80 修复"


class TestSynthesizeVerdictBehavior:
    """synthesize_verdict 行为测试 — 覆盖 Round 80 关键场景."""

    def _load_module(self):
        """用 importlib 直加载 src.judges.verdict, 绕过 sys.modules 父包状态."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_verdict_isolated", "src/judges/verdict.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_low_confidence_skipped(self):
        """场景 1: score < min_score 但 confidence < min_confidence → 跳过 (skipped_dimensions)."""
        verdict_mod = _load_module()
        # 构造一个 style_compliance_score: score=0.5 (< min_score 1.0), confidence=0.3 (< min_conf 0.6)
        answers = {
            "style_compliance_score": verdict_mod.ScoreAnswer(
                score=0.5, probabilities={0: 0.5, 1: 0.3, 2: 0.0}, confidence=0.3
            ),
        }
        v = verdict_mod.synthesize_verdict(answers)
        # 旧版本: 会 reject (score < 1.0)
        # Round 80: confidence=0.3 < 0.6 → 跳过, 不 reject
        assert "style_compliance_score" not in v.reject_reasons
        assert v.decision == "PASS", (
            f"Round 80: 低 confidence 的 score 应被跳过, 不参与 reject. "
            f"实际 decision={v.decision}, reject_reasons={v.reject_reasons}"
        )
        # 必须显式记录到 skipped_dimensions (与 Noul uncertain zone 一致)
        assert "style_compliance_score" in v.skipped_dimensions
        assert "low_conf" in v.skipped_reasons.get("style_compliance_score", "")

    def test_high_confidence_low_score_rejects(self):
        """场景 2: score < min_score 且 confidence >= min_confidence → reject (旧行为)."""
        verdict_mod = _load_module()
        answers = {
            "style_compliance_score": verdict_mod.ScoreAnswer(
                score=0.5, probabilities={0: 0.5, 1: 0.3, 2: 0.0}, confidence=0.8
            ),
        }
        v = verdict_mod.synthesize_verdict(answers)
        assert v.decision == "REJECT"
        assert any("风格契合度低" in r for r in v.reject_reasons)
        assert "style_compliance_score" not in v.skipped_dimensions

    def test_high_score_passes(self):
        """场景 3: score >= min_score 且 high confidence → PASS (baseline)."""
        verdict_mod = _load_module()
        answers = {
            "style_compliance_score": verdict_mod.ScoreAnswer(
                score=1.0, probabilities={0: 0, 1: 0.3, 2: 0.7}, confidence=0.8
            ),
        }
        v = verdict_mod.synthesize_verdict(answers)
        assert v.decision == "PASS"
        assert "style_compliance_score" not in v.skipped_dimensions

    def test_custom_min_confidence_threshold(self):
        """场景 4: 允许通过 thresholds dict 自定义 min_confidence."""
        verdict_mod = _load_module()
        # 设 min_confidence = 0.9, score=0.5, confidence=0.85 → 应跳过 (0.85 < 0.9)
        answers = {
            "style_compliance_score": verdict_mod.ScoreAnswer(
                score=0.5, probabilities={0: 0.5, 1: 0.3, 2: 0.0}, confidence=0.85
            ),
        }
        v = verdict_mod.synthesize_verdict(
            answers, thresholds={"style_compliance_min_confidence": 0.9}
        )
        assert v.decision == "PASS", "自定义 min_confidence=0.9 时 confidence=0.85 应被跳过"
        assert "style_compliance_score" in v.skipped_dimensions

    def test_min_confidence_config_wiring(self):
        """场景 5: config.yaml typesafe_thresholds.style_compliance_min_confidence 应被读取."""
        verdict_mod = _load_module()
        from src.config import config as _config
        ts_thresholds = _config.typesafe_thresholds
        assert "style_compliance_min_confidence" in ts_thresholds, (
            "config.yaml typesafe_thresholds 必须含 style_compliance_min_confidence"
        )
        # 用真实 config 值调用, 应能通过门控 (因为 score=1.0, high conf)
        answers = {
            "style_compliance_score": verdict_mod.ScoreAnswer(
                score=0.5, probabilities={0: 0.5, 1: 0.3, 2: 0.0},
                confidence=ts_thresholds["style_compliance_min_confidence"] + 0.01,
            ),
        }
        v = verdict_mod.synthesize_verdict(answers, thresholds=ts_thresholds)
        assert v.decision == "REJECT", "刚好高于 min_confidence 的 score 应触发 reject"


class TestBackwardCompat:
    """回归: 没有 style_compliance_score 时不影响其他维度."""

    def test_no_style_compliance_score_dim(self):
        verdict_mod = _load_module()
        # 只有 voice_drift 维度, 不应有 style_compliance_score
        answers = {
            "voice_drift": verdict_mod.ChoiceAnswer(
                choice="none", probabilities={"none": 0.9, "subtle": 0.05, "severe": 0.05},
                confidence=0.9,
            ),
        }
        v = verdict_mod.synthesize_verdict(answers)
        assert v.decision == "PASS"
        # 不应有 style_compliance_score 相关 skip/reject
        assert "style_compliance_score" not in v.skipped_dimensions
        assert not any("风格契合度" in r for r in v.reject_reasons)
