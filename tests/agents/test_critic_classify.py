"""Round 53: 单元测试 - agents/critic_agent.py::_classify_backend_error.

保护目标（A6 错误分类）：
  - 7 类 exception → 正确 category 映射
  - 未知 exception → category="unknown"
  - 返回 dict 必含 category/msg/action 三键
  - _TYPESAFE_EXC_AVAILABLE=False 退化路径
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from src.agents.critic_agent import _classify_backend_error, _TYPESAFE_EXC_AVAILABLE
# Round 130: typesafe_sdk 是可选依赖（缺失时生产自动降级）；本文件测其异常分类，
# 无依赖环境应跳过而非收集失败。
pytest.importorskip("typesafe_sdk", reason="可选依赖 typesafe_sdk 未安装（Jev 后端）")
from typesafe_sdk import (
    TypeSafeAuthenticationError, TypeSafeRateLimitError,
    TypeSafeAPITimeoutError, TypeSafeAPIConnectionError,
    TypeSafeBadRequestError, TypeSafeAPIResponseValidationError,
    TypeSafeAPIError,
)
import httpx


def _headers():
    return httpx.Headers({})


# ---------------------------------------------------------------------------
# 真实异常类（typesafe_sdk 已安装）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TYPESAFE_EXC_AVAILABLE, reason="typesafe_sdk not available")
class TestClassifyWithRealExceptions:
    def test_authentication_error_maps_to_auth(self):
        from typesafe_sdk import TypeSafeAuthenticationError
        exc = TypeSafeAuthenticationError(401, {"error": "invalid api key"}, _headers(), "invalid api key")
        out = _classify_backend_error(exc)
        assert out["category"] == "auth"
        assert "check TYPESAFE_API_KEY" in out["action"]
        assert "invalid api key" in out["msg"]

    def test_rate_limit_error_maps_to_rate_limit(self):
        from typesafe_sdk import TypeSafeRateLimitError
        exc = TypeSafeRateLimitError(429, {"error": "quota exceeded"}, _headers(), "quota exceeded")
        out = _classify_backend_error(exc)
        assert out["category"] == "rate_limit"
        assert "wait or upgrade" in out["action"]

    def test_timeout_error_maps_to_timeout(self):
        from typesafe_sdk import TypeSafeAPITimeoutError
        exc = TypeSafeAPITimeoutError(30.0)
        out = _classify_backend_error(exc)
        assert out["category"] == "timeout"
        assert "timeout" in out["action"].lower()

    def test_connection_error_maps_to_network(self):
        from typesafe_sdk import TypeSafeAPIConnectionError
        exc = TypeSafeAPIConnectionError("DNS resolution failed")
        out = _classify_backend_error(exc)
        assert out["category"] == "network"
        assert "DNS" in out["action"] or "firewall" in out["action"]

    def test_bad_request_error_maps_to_bad_request(self):
        from typesafe_sdk import TypeSafeBadRequestError
        exc = TypeSafeBadRequestError(400, {"error": "invalid question schema"}, _headers(), "invalid question schema")
        out = _classify_backend_error(exc)
        assert out["category"] == "bad_request"
        assert "schema" in out["action"].lower()

    def test_response_validation_error_maps_to_schema_mismatch(self):
        from typesafe_sdk import TypeSafeAPIResponseValidationError
        exc = TypeSafeAPIResponseValidationError(500, {"error": "response did not match schema"}, _headers(), "$.answers")
        out = _classify_backend_error(exc)
        assert out["category"] == "schema_mismatch"
        assert "verdict parser" in out["action"]

    def test_generic_api_error_maps_to_api_error(self):
        from typesafe_sdk import TypeSafeAPIError
        exc = TypeSafeAPIError(500, {"error": "internal server error"}, _headers(), "internal server error")
        out = _classify_backend_error(exc)
        assert out["category"] == "api_error"
        assert "status page" in out["action"]


# ---------------------------------------------------------------------------
# 通用 fallback（_TYPESAFE_EXC_AVAILABLE=False 退化路径）
# ---------------------------------------------------------------------------

class TestClassifyUnknownException:
    def test_plain_exception_maps_to_unknown(self):
        out = _classify_backend_error(ValueError("some error"))
        assert out["category"] == "unknown"
        assert "ValueError" in out["msg"]
        assert "inspect stack trace" in out["action"]

    def test_runtime_error_maps_to_unknown(self):
        out = _classify_backend_error(RuntimeError("boom"))
        assert out["category"] == "unknown"

    def test_unknown_exception_with_degraded_path(self):
        """模拟 typesafe_sdk 不可用时的退化路径 — 任何异常都映射到 unknown"""
        # 通过 patch 让 _classify_backend_error 走退化分支
        with patch("src.agents.critic_agent._TYPESAFE_EXC_AVAILABLE", False):
            from typesafe_sdk import TypeSafeAuthenticationError  # noqa
            exc = TypeSafeAuthenticationError(401, {}, _headers(), "won't match in degraded mode")
            out = _classify_backend_error(exc)
            # 在退化模式下，真实异常类被 except 吞掉，落到 unknown
            assert out["category"] == "unknown"
            assert "won't match" in out["msg"]


# ---------------------------------------------------------------------------
# 返回结构契约
# ---------------------------------------------------------------------------

class TestClassifyReturnStructure:
    def test_returns_dict_with_required_keys(self):
        out = _classify_backend_error(ValueError("test"))
        assert isinstance(out, dict)
        assert "category" in out
        assert "msg" in out
        assert "action" in out

    def test_all_real_categories_have_required_keys(self):
        # 收集所有真实 exception type
        from typesafe_sdk import (
            TypeSafeAuthenticationError, TypeSafeRateLimitError,
            TypeSafeAPITimeoutError, TypeSafeAPIConnectionError,
            TypeSafeBadRequestError, TypeSafeAPIResponseValidationError,
            TypeSafeAPIError,
        )
        exceptions = [
            TypeSafeAuthenticationError(401, {}, _headers(), "a"),
            TypeSafeRateLimitError(429, {}, _headers(), "b"),
            TypeSafeAPITimeoutError(30.0),
            TypeSafeAPIConnectionError("d"),
            TypeSafeBadRequestError(400, {}, _headers(), "e"),
            TypeSafeAPIResponseValidationError(500, {}, _headers(), "$.answers"),
            TypeSafeAPIError(500, {}, _headers(), "g"),
        ]
        for exc in exceptions:
            out = _classify_backend_error(exc)
            assert "category" in out
            assert "msg" in out
            assert "action" in out

    def test_category_is_one_of_known_values(self):
        valid = {
            "auth", "rate_limit", "timeout", "network",
            "bad_request", "schema_mismatch", "api_error", "unknown",
        }
        out = _classify_backend_error(ValueError("test"))
        assert out["category"] in valid

    def test_msg_includes_exception_message(self):
        out = _classify_backend_error(ValueError("specific message here"))
        assert "specific message here" in out["msg"]

    def test_action_is_non_empty_string(self):
        out = _classify_backend_error(ValueError("x"))
        assert isinstance(out["action"], str)
        assert len(out["action"]) > 0
