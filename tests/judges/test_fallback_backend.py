"""Round 51: 单元测试 - judges/typesafe_backend.py (FallbackBackend).

保护目标：
  - enabled=False 直降 mock（typesafe 路径旁路）
  - enabled=False + allow_mock=False → RuntimeError
  - Jev 成功 → 返回 Jev 结果
  - Jev 失败 + allow_openai_fallback=False → mock
  - Jev 失败 + allow_openai_fallback=True + OpenAICompat 成功 → OpenAICompat
  - Jev 失败 + allow_openai_fallback=True + OpenAICompat 失败 → mock
  - 无 api_key 时同样处理
  - 所有 backend 失败 + allow_mock=False → RuntimeError
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.judges.typesafe_backend import FallbackBackend, JevBackend, OpenAICompatBackend, MockBackend
from src.judges.typesafe_spec import NoulAnswer, ChoiceAnswer, ScoreAnswer


# ────────────────────────────────────────────────────────────────────────
# 测试工具
# ────────────────────────────────────────────────────────────────────────

def _stub_jev(api_key="test-key"):
    """构造一个会被 Jev 视为有 api_key 的 JevBackend mock"""
    jev = MagicMock(spec=JevBackend)
    jev.api_key = api_key
    return jev


def _stub_openai():
    """构造 OpenAICompatBackend mock"""
    return MagicMock(spec=OpenAICompatBackend)


def _stub_mock():
    """构造 MockBackend 实例（真实逻辑，用于验证启发式）"""
    return MockBackend()


def _sample_questions():
    return {
        "voice_drift": {"type": "choice"},
        "literal_residue": {"type": "noul"},
    }


# ────────────────────────────────────────────────────────────────────────
# enabled=False 路径
# ────────────────────────────────────────────────────────────────────────

class TestFallbackDisabled:
    def test_disabled_routes_to_mock(self):
        b = FallbackBackend(enabled=False)
        b._mock = _stub_mock()
        out = b.evaluate("state", _sample_questions())
        # mock 返回 6 个维度答案
        assert "voice_drift" in out

    def test_disabled_without_mock_raises(self):
        b = FallbackBackend(enabled=False, allow_mock=False)
        with pytest.raises(RuntimeError, match="已禁用"):
            b.evaluate("state", _sample_questions())


# ────────────────────────────────────────────────────────────────────────
# Jev 成功路径
# ────────────────────────────────────────────────────────────────────────

class TestFallbackJevSuccess:
    def test_jev_success_skips_mock(self):
        b = FallbackBackend(allow_mock=True)
        b.primary = _stub_jev(api_key="test-key")
        jev_result = {"voice_drift": ChoiceAnswer(choice="none", probabilities={"none": 1.0}, confidence=0.9)}
        b.primary.evaluate.return_value = jev_result
        out = b.evaluate("state", _sample_questions())
        assert out is jev_result
        b.primary.evaluate.assert_called_once()


# ────────────────────────────────────────────────────────────────────────
# Jev 失败 + 不允许 OpenAICompat → Mock
# ────────────────────────────────────────────────────────────────────────

class TestFallbackJevFailNoOpenAICompat:
    def test_jev_fail_routes_to_mock(self):
        b = FallbackBackend(allow_openai_fallback=False)
        b.primary = _stub_jev(api_key="test-key")
        b.primary.evaluate.side_effect = ConnectionError("Jev down")
        b._mock = _stub_mock()
        out = b.evaluate("state", _sample_questions())
        # 应该是 mock 返回值（启发式 6 维度）
        assert "literal_residue" in out
        b.primary.evaluate.assert_called_once()


# ────────────────────────────────────────────────────────────────────────
# Jev 失败 + 允许 OpenAICompat → OpenAICompat
# ────────────────────────────────────────────────────────────────────────

class TestFallbackJevFailOpenAICompat:
    def test_jev_fail_openai_success_returns_openai(self):
        b = FallbackBackend(allow_openai_fallback=True)
        b.primary = _stub_jev(api_key="test-key")
        b.primary.evaluate.side_effect = ConnectionError("Jev down")
        b.fallback = _stub_openai()
        openai_result = {"voice_drift": ChoiceAnswer(choice="subtle", probabilities={"subtle": 1.0}, confidence=0.7)}
        b.fallback.evaluate.return_value = openai_result
        out = b.evaluate("state", _sample_questions())
        assert out is openai_result
        b.fallback.evaluate.assert_called_once()

    def test_jev_fail_openai_fail_routes_to_mock(self):
        b = FallbackBackend(allow_openai_fallback=True, allow_mock=True)
        b.primary = _stub_jev(api_key="test-key")
        b.primary.evaluate.side_effect = ConnectionError("Jev down")
        b.fallback = _stub_openai()
        b.fallback.evaluate.side_effect = RuntimeError("OpenAI down")
        b._mock = _stub_mock()
        out = b.evaluate("state", _sample_questions())
        # mock 返回
        assert "voice_drift" in out
        b.fallback.evaluate.assert_called_once()


# ────────────────────────────────────────────────────────────────────────
# 无 api_key 路径
# ────────────────────────────────────────────────────────────────────────

class TestFallbackNoApiKey:
    def test_no_key_routes_to_mock(self):
        b = FallbackBackend(allow_openai_fallback=False)
        b.primary = _stub_jev(api_key="")  # 空 key
        b._mock = _stub_mock()
        out = b.evaluate("state", _sample_questions())
        # primary.evaluate 不应被调用（因为 api_key 为空）
        b.primary.evaluate.assert_not_called()
        assert "voice_drift" in out

    def test_no_key_openai_fallback_enabled_uses_openai(self):
        b = FallbackBackend(allow_openai_fallback=True)
        b.primary = _stub_jev(api_key="")
        b.fallback = _stub_openai()
        openai_result = {"voice_drift": ChoiceAnswer(choice="severe", probabilities={"severe": 1.0}, confidence=0.8)}
        b.fallback.evaluate.return_value = openai_result
        out = b.evaluate("state", _sample_questions())
        assert out is openai_result

    def test_no_key_openai_fails_routes_to_mock(self):
        b = FallbackBackend(allow_openai_fallback=True, allow_mock=True)
        b.primary = _stub_jev(api_key="")
        b.fallback = _stub_openai()
        b.fallback.evaluate.side_effect = ConnectionError("OpenAI down")
        b._mock = _stub_mock()
        out = b.evaluate("state", _sample_questions())
        assert "voice_drift" in out


# ────────────────────────────────────────────────────────────────────────
# 全失败 + 无 mock
# ────────────────────────────────────────────────────────────────────────

class TestFallbackAllFail:
    def test_jev_fail_no_mock_raises(self):
        b = FallbackBackend(allow_openai_fallback=False, allow_mock=False)
        b.primary = _stub_jev(api_key="test-key")
        b.primary.evaluate.side_effect = ConnectionError("Jev down")
        with pytest.raises(RuntimeError, match="不可用"):
            b.evaluate("state", _sample_questions())

    def test_jev_fail_openai_fail_no_mock_raises(self):
        b = FallbackBackend(allow_openai_fallback=True, allow_mock=False)
        b.primary = _stub_jev(api_key="test-key")
        b.primary.evaluate.side_effect = ConnectionError("Jev down")
        b.fallback = _stub_openai()
        b.fallback.evaluate.side_effect = RuntimeError("OpenAI down")
        with pytest.raises(RuntimeError, match="不可用"):
            b.evaluate("state", _sample_questions())


# ────────────────────────────────────────────────────────────────────────
# timeout 行为
# ────────────────────────────────────────────────────────────────────────

class TestFallbackTimeout:
    def test_caller_timeout_overrides_default(self):
        b = FallbackBackend(timeout=10.0)
        b.primary = _stub_jev(api_key="test-key")
        b.primary.evaluate.return_value = {"voice_drift": ChoiceAnswer(choice="none", probabilities={}, confidence=0.5)}
        b.evaluate("state", _sample_questions(), timeout=2.0)
        # 验证 timeout 参数被传递给 primary.evaluate
        _, kwargs = b.primary.evaluate.call_args
        assert kwargs.get("timeout") == 2.0

    def test_default_timeout_used_when_no_override(self):
        b = FallbackBackend(timeout=15.0)
        b.primary = _stub_jev(api_key="test-key")
        b.primary.evaluate.return_value = {"voice_drift": ChoiceAnswer(choice="none", probabilities={}, confidence=0.5)}
        b.evaluate("state", _sample_questions())
        _, kwargs = b.primary.evaluate.call_args
        assert kwargs.get("timeout") == 15.0
