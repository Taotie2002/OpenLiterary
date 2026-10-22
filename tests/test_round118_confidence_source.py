"""Round 118 P1-1: confidence_source 分档测试.

设计原则:
  - Jev (System One) 是校准概率 -> 阈值原样, 绝不弱化 (默认路径逐字节不变);
  - openai_compat (LLM 自报伪概率) / mock_heuristic (确定性启发式) 属未校准
    来源 -> reject 门槛 +margin, 置信门控 +margin, Noul 不确定区间上界 +margin;
  - verdict 携带 confidence_source / confidence_calibrated 供下游审计区分
    "Jev 认证" 与 "降级模式".
"""
from __future__ import annotations

import sys

import pytest

from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer
from src.judges.verdict import normalize_confidence_source, synthesize_verdict


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)


CONFIG_ZONES = {
    "noul_uncertain_zones": {
        "entity_consistency": [0.3, 0.7],
        "literal_residue": [0.3, 0.8],
        "semantic_loss": [0.3, 0.75],
    }
}


def _register_combined(total: float, conf: float = 0.9) -> ChoiceAnswer:
    """register_match 走 under+over 合计阈值: total 即失配合计概率."""
    return ChoiceAnswer(
        "over",
        {"match": 1.0 - total, "under": total * 0.4, "over": total * 0.6},
        conf,
    )


class TestNormalize:
    @pytest.mark.parametrize("raw,expected", [
        ("jev", "jev"),
        ("JevBackend", "jev"),
        ("typesafe", "jev"),
        ("openai_compat", "openai_compat"),
        ("OpenAICompatBackend", "openai_compat"),
        ("mock_heuristic", "mock_heuristic"),
        ("MockBackend", "mock_heuristic"),
        ("SomeWeirdBackend", "_uncalibrated"),
        (None, "_uncalibrated"),
    ])
    def test_mapping(self, raw, expected):
        assert normalize_confidence_source(raw) == expected


class TestJevUnchanged:
    """Jev 属校准概率: 同样答案下默认路径的裁决不得因本轮改动而改变."""

    def test_default_source_keeps_original_decision(self):
        answers = {"register_match": _register_combined(0.45)}
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"  # 0.45 >= 0.4 (原阈值)
        assert v.confidence_source == "jev"
        assert v.confidence_calibrated is True

    def test_jev_source_identical_to_default(self):
        answers = {"register_match": _register_combined(0.45)}
        v_default = synthesize_verdict(answers)
        v_jev = synthesize_verdict(answers, confidence_source="jev")
        assert v_default.decision == v_jev.decision == "REJECT"
        assert v_default.confidence == v_jev.confidence

    def test_to_dict_carries_source_fields(self):
        v = synthesize_verdict({})
        d = v.to_dict()
        assert d["confidence_source"] == "jev"
        assert d["confidence_calibrated"] is True


class TestUncalibratedMargins:
    def test_openai_compat_choice_requires_stronger_evidence(self):
        """0.45 在 Jev 下 reject; openai_compat margin 0.15 -> 门槛 0.55 -> PASS."""
        answers = {"register_match": _register_combined(0.45)}
        v = synthesize_verdict(answers, confidence_source="openai_compat")
        assert v.decision == "PASS"
        assert v.confidence_source == "openai_compat"
        assert v.confidence_calibrated is False

    def test_openai_compat_strong_signal_still_rejects(self):
        answers = {"register_match": _register_combined(0.7)}
        v = synthesize_verdict(answers, confidence_source="openai_compat")
        assert v.decision == "REJECT"

    def test_mock_heuristic_margin_larger_than_openai(self):
        """合计 0.60: openai_compat (门槛 0.55) reject; mock (门槛 0.65) PASS."""
        answers = {"register_match": _register_combined(0.6)}
        v_openai = synthesize_verdict(answers, confidence_source="openai_compat")
        v_mock = synthesize_verdict(answers, confidence_source="mock_heuristic")
        assert v_openai.decision == "REJECT"
        assert v_mock.decision == "PASS"

    def test_noul_zone_widened_for_uncalibrated(self):
        """semantic_loss noul=0.85: Jev (zone 上界 0.75) reject;
        openai_compat 上界 0.75+0.15=0.90 -> 跳过."""
        answers = {"semantic_loss": NoulAnswer(noul=0.85)}
        v_jev = synthesize_verdict(answers, thresholds=CONFIG_ZONES)
        v_fb = synthesize_verdict(
            answers, thresholds=CONFIG_ZONES, confidence_source="openai_compat"
        )
        assert v_jev.decision == "REJECT"
        assert v_fb.decision == "PASS"
        assert "semantic_loss" in v_fb.skipped_dimensions

    def test_extreme_noul_still_rejects_under_mock(self):
        """widening 封顶 0.98: noul=1.0 的极端信号在 mock 下仍触发 reject."""
        answers = {"semantic_loss": NoulAnswer(noul=1.0)}
        v = synthesize_verdict(
            answers, thresholds=CONFIG_ZONES, confidence_source="mock_heuristic"
        )
        assert v.decision == "REJECT"

    def test_style_confidence_margin(self):
        """style score=0 conf=0.65: Jev min_conf 0.6 -> reject;
        mock min_conf 0.6+0.20=0.80 -> 保守跳过."""
        answers = {
            "style_compliance_score": ScoreAnswer(0.0, {0: 0.7, 1: 0.2, 2: 0.1}, 0.65)
        }
        v_jev = synthesize_verdict(answers)
        v_mock = synthesize_verdict(answers, confidence_source="mock_heuristic")
        assert v_jev.decision == "REJECT"
        assert v_mock.decision == "PASS"
        assert "style_compliance_score" in v_mock.skipped_dimensions

    def test_config_profile_override(self):
        """config 可覆盖默认 margin: 0.3 -> openai 门槛 0.7."""
        thresholds = {
            "confidence_source_profiles": {
                "openai_compat": {"reject_margin": 0.3, "confidence_margin": 0.3},
            }
        }
        answers = {"register_match": _register_combined(0.65)}
        v = synthesize_verdict(
            answers, thresholds=thresholds, confidence_source="openai_compat"
        )
        assert v.decision == "PASS"
        answers2 = {"register_match": _register_combined(0.75)}
        v2 = synthesize_verdict(
            answers2, thresholds=thresholds, confidence_source="openai_compat"
        )
        assert v2.decision == "REJECT"


class TestBackendSourcePropagation:
    def test_fallback_backend_records_mock_path(self):
        from src.judges.typesafe_backend import FallbackBackend

        backend = FallbackBackend(allow_openai_fallback=False, allow_mock=True)
        backend.primary.api_key = ""  # 强制无 Jev
        backend.evaluate(
            {"source_text": "a", "raw_translation": "b", "literary_translation": "c"},
            {},
        )
        assert backend.last_backend == "mock_heuristic"

    def test_fallback_backend_records_jev_path(self):
        from src.judges.typesafe_backend import FallbackBackend

        backend = FallbackBackend(allow_openai_fallback=False, allow_mock=True)

        class _StubJev:
            api_key = "stub"

            def evaluate(self, state, questions, **kw):
                return {"voice_drift": ChoiceAnswer("none", {"none": 0.9}, 0.9)}

        backend.primary = _StubJev()
        answers = backend.evaluate({"source_text": "a"}, {})
        assert backend.last_backend == "jev"
        assert "voice_drift" in answers

    def test_judge_evaluate_chunk_passes_source(self, monkeypatch):
        import src.judges.judge as judge_mod

        class _FakeBackend:
            backend_kind = "openai_compat"
            last_backend = "openai_compat"

            def evaluate(self, state, questions, **kw):
                return {"register_match": _register_combined(0.45, 0.9)}

        monkeypatch.setattr(judge_mod, "get_backend", lambda: _FakeBackend())
        verdict = judge_mod.evaluate_chunk("src", "这是一段中文直译底稿。", "这是一段中文润色译文。", {})
        assert verdict.confidence_source == "openai_compat"
        assert verdict.confidence_calibrated is False
        # 0.45 未达未校准门槛 0.55 -> PASS (Jev 路径下本应 REJECT)
        assert verdict.decision == "PASS"
