"""Round 140: Jev 降级路由与 403 诊断回归。

覆盖事故链：
  1. typesafe_sdk 缺失时 Jev 主路径应在进程内熔断，不每 chunk 重复探测；
  2. OpenAICompat 兜底的客户端与 model_name 必须来自同一 critic_scoring 路由；
  3. 中文 403「无权访问模型」必须归类为 auth，而不是 unknown。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agents.critic_agent import _classify_backend_error
from src.config import config as _cfg
from src.judges.typesafe_backend import (
    FallbackBackend,
    OpenAICompatBackend,
    _is_auth_failure,
)


class _FakeLLM:
    max_retries = 3

    def __init__(self):
        self.calls = []

    def generate(self, prompt, model_name, **kwargs):
        self.calls.append({"prompt": prompt, "model_name": model_name, **kwargs})
        return json.dumps({
            "answers": {
                "voice_drift": {
                    "type": "choice",
                    "choice": "none",
                    "confidence": 0.9,
                    "reasoning": "ok",
                }
            }
        }, ensure_ascii=False)


def _questions():
    return {
        "voice_drift": {
            "type": "choice",
            "instructions": "判断语气漂移",
            "criteria": {"none": "无", "subtle": "轻微", "severe": "严重"},
        }
    }


class TestOpenAICompatTaskBinding:
    def test_default_uses_critic_route(self):
        fake = _FakeLLM()
        backend = OpenAICompatBackend(llm_client=fake)

        route_role = _cfg.get_task_role("critic_scoring")
        _, route_params = _cfg.resolve_task_model("critic_scoring")
        expected_model = _cfg.get_role_config(route_role)["model_name"]

        out = backend.evaluate("state", _questions())

        assert out["voice_drift"].choice == "none"
        assert backend.model_role == route_role
        assert fake.calls[0]["model_name"] == expected_model
        assert fake.calls[0]["model_name"] != _cfg.get_role_config("primary")["model_name"]
        assert fake.calls[0]["max_tokens"] == route_params["max_tokens"]
        assert fake.calls[0]["max_tokens"] != 2048

    def test_explicit_model_role_keeps_old_semantics(self):
        fake = _FakeLLM()
        backend = OpenAICompatBackend(model_role="primary", llm_client=fake)

        backend.evaluate("state", _questions())

        assert backend.model_role == "primary"
        assert fake.calls[0]["model_name"] == _cfg.get_role_config("primary")["model_name"]
        assert "effort" not in fake.calls[0]

    def test_fallback_backend_does_not_use_legacy_fallback_role(self):
        backend = FallbackBackend(fallback_role="primary", allow_openai_fallback=True)
        route_role = _cfg.get_task_role("critic_scoring")

        assert backend.fallback.model_role == route_role
        assert backend.fallback.model_name == _cfg.get_role_config(route_role)["model_name"]
        assert backend.fallback.model_name != _cfg.get_role_config("primary")["model_name"]


class TestBackendErrorClassification:
    def test_chinese_403_model_permission_is_auth(self):
        exc = RuntimeError(
            'API 客户端错误 403: {"error":{"message":'
            '"该令牌无权访问模型 MiniMax-M3.1-Flash"}}'
        )
        out = _classify_backend_error(exc)
        assert out["category"] == "auth"
        assert out["retryable"] is False

    def test_missing_typesafe_sdk_is_dependency(self):
        out = _classify_backend_error(ModuleNotFoundError("No module named 'typesafe_sdk'"))
        assert out["category"] == "dependency"
        assert out["retryable"] is False

    def test_400_max_tokens_is_not_auth(self):
        assert not _is_auth_failure(
            RuntimeError("API 客户端错误 400: max_tokens_exceeded")
        )
        assert _is_auth_failure(
            RuntimeError("API 客户端错误 403: 该令牌无权访问模型 MiniMax-M3.1-Flash")
        )


class TestJevCircuitBreaker:
    def _fallback_result(self):
        return {"voice_drift": MagicMock(choice="none")}

    def test_missing_sdk_is_probed_once_then_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "src.judges.typesafe_backend._typesafe_sdk_available", lambda: False
        )
        backend = FallbackBackend(allow_openai_fallback=True)
        backend.primary.evaluate = MagicMock(wraps=backend.primary.evaluate)
        backend.fallback = MagicMock()
        backend.fallback.evaluate.return_value = self._fallback_result()

        backend.evaluate("state", _questions())
        backend.evaluate("state", _questions())

        assert backend._jev_auth_dead is True
        backend.primary.evaluate.assert_called_once()
        assert backend.fallback.evaluate.call_count == 2

    def test_missing_sdk_during_call_is_cached(self, monkeypatch):
        monkeypatch.setattr(
            "src.judges.typesafe_backend._typesafe_sdk_available", lambda: True
        )
        backend = FallbackBackend(allow_openai_fallback=True)
        backend.primary = MagicMock()
        backend.primary.api_key = "test-key"
        backend.primary.evaluate.side_effect = ModuleNotFoundError(
            "No module named 'typesafe_sdk'"
        )
        backend.fallback = MagicMock()
        backend.fallback.evaluate.return_value = self._fallback_result()

        backend.evaluate("state", _questions())
        backend.primary.evaluate.reset_mock()
        backend.evaluate("state", _questions())

        assert backend._jev_auth_dead is True
        backend.primary.evaluate.assert_not_called()
        assert backend.fallback.evaluate.call_count == 2
