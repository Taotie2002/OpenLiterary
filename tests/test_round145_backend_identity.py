"""Round 145: provider/model audit identity, error taxonomy, fail-closed fallback."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.judges.typesafe_backend import (
    FallbackBackend,
    JevBackend,
    classify_backend_error,
)
from src.judges.verdict import normalize_confidence_source
from src.agents.critic_agent import _classify_backend_error as classify_critic_error
from tests._round145_sdk_stub import install as install_sdk_stub


@pytest.fixture(autouse=True)
def _typesafe_sdk_stub(monkeypatch):
    install_sdk_stub(monkeypatch)


class _FakePrimary:
    backend_kind = "jev"
    provider = "moark"
    api_key = "k"
    model = "d3-flash"

    def __init__(self, result=None, exc=None):
        self.result = result or {}
        self.exc = exc
        self.calls = 0
        self.last_diagnostics = {
            "provider": "moark",
            "model": "d3-flash",
            "base_url_host": "api.moark.com",
            "question_count": 6,
            "batch_count": 1,
            "batch_question_counts": [6],
            "output_tokens": 0,
        }

    def evaluate(self, state, questions, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


class TestErrorTaxonomy:
    def test_model_access_403(self):
        out = classify_backend_error(
            RuntimeError('API 客户端错误 403: 该令牌无权访问模型 d3-flash')
        )
        assert out["category"] == "model_access"

    def test_auth_401(self):
        out = classify_backend_error(RuntimeError("HTTP 401 Unauthorized: invalid api key"))
        assert out["category"] == "auth"

    def test_request_schema_400(self):
        out = classify_backend_error(RuntimeError("HTTP 400: too many questions"))
        assert out["category"] == "request_schema"

    def test_model_name_400_is_model_access(self):
        out = classify_backend_error(
            RuntimeError('400 模型名称"no-such-d3-model"有误，请检查拼写')
        )
        assert out["category"] == "model_access"

    def test_network(self):
        out = classify_backend_error(RuntimeError("DNS resolution failed"))
        assert out["category"] == "network"

    def test_missing_httpx2_is_dependency(self):
        out = classify_backend_error(ModuleNotFoundError("No module named 'httpx2'"))
        assert out["category"] == "dependency"
        assert classify_critic_error(
            ModuleNotFoundError("No module named 'httpx2'"))["category"] == "dependency"


class TestBackendIdentity:
    def test_fallback_records_provider_model_and_batches(self):
        fake = _FakePrimary(result={"voice_drift": object()})
        backend = FallbackBackend(allow_openai_fallback=False, allow_mock=False)
        backend.primary = fake
        out = backend.evaluate("state", {"voice_drift": {"type": "choice"}})
        assert "voice_drift" in out
        assert backend.last_backend == "jev"
        assert backend.last_diagnostics["provider"] == "moark"
        assert backend.last_diagnostics["model"] == "d3-flash"
        assert backend.last_diagnostics["batch_count"] == 1

    def test_backend_kind_stays_jev_for_calibrated_thresholds(self):
        assert JevBackend.backend_kind == "jev"
        assert normalize_confidence_source("jev") == "jev"

    def test_jev_diagnostics_shape(self):
        backend = JevBackend(
            api_key="k", model="d3-flash", api_base="https://api.moark.com",
            max_questions_per_batch=64,
        )
        backend._client = SimpleNamespace(
            system_one=lambda **kw: SimpleNamespace(
                answers={qid: qid for qid in kw["questions"]},
                usage=SimpleNamespace(input_tokens=3, output_tokens=0),
            )
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.judges.typesafe_backend._typesafe_sdk_available", lambda: True)
            mp.setattr(JevBackend, "_convert_answer", staticmethod(lambda ans: ans))
            out = backend.evaluate(
                "state",
                {"q1": {"type": "noul", "instructions": "x", "criteria": None}},
            )
        assert out == {"q1": "q1"}
        diag = backend.last_diagnostics
        assert diag["provider"] == "moark"
        assert diag["model"] == "d3-flash"
        assert diag["base_url_host"] == "api.moark.com"
        assert diag["question_count"] == 1
        assert diag["batch_count"] == 1
        assert diag["output_tokens"] == 0


class TestCircuitAndFailClosed:
    def test_model_access_does_not_trip_auth_breaker(self):
        fake = _FakePrimary(
            exc=RuntimeError("API 客户端错误 403: 该令牌无权访问模型 d3-flash")
        )
        backend = FallbackBackend(allow_openai_fallback=False, allow_mock=False)
        backend.primary = fake
        with pytest.raises(RuntimeError):
            backend.evaluate("state", {"q": {"type": "noul"}})
        assert backend._jev_model_dead is True
        assert backend._jev_auth_dead is False
        with pytest.raises(RuntimeError):
            backend.evaluate("state", {"q": {"type": "noul"}})
        assert fake.calls == 1

    def test_auth_trips_auth_breaker(self):
        fake = _FakePrimary(exc=RuntimeError("HTTP 401 Unauthorized: invalid api key"))
        backend = FallbackBackend(allow_openai_fallback=False, allow_mock=False)
        backend.primary = fake
        with pytest.raises(RuntimeError):
            backend.evaluate("state", {"q": {"type": "noul"}})
        assert backend._jev_auth_dead is True
        assert backend._jev_model_dead is False

    def test_generic_dependency_trips_breaker_to_avoid_per_chunk_retry(self):
        fake = _FakePrimary(exc=ModuleNotFoundError("No module named 'httpx2'"))
        backend = FallbackBackend(allow_openai_fallback=False, allow_mock=False)
        backend.primary = fake
        with pytest.raises(RuntimeError, match="所有 backend 都不可用"):
            backend.evaluate("state", {"q": {"type": "noul"}})
        assert backend._jev_auth_dead is True
        with pytest.raises(RuntimeError, match="所有 backend 都不可用"):
            backend.evaluate("state", {"q": {"type": "noul"}})
        assert fake.calls == 1

    def test_fallback_false_does_not_mask_network_failure(self):
        fake = _FakePrimary(exc=RuntimeError("DNS resolution failed"))
        backend = FallbackBackend(allow_openai_fallback=False, allow_mock=False)
        backend.primary = fake
        with pytest.raises(RuntimeError, match="所有 backend 都不可用"):
            backend.evaluate("state", {"q": {"type": "noul"}})
