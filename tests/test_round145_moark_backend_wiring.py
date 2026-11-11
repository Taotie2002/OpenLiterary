"""Round 145: MoArk d3 backend wiring (api_base / key / model priority)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.judges.typesafe_backend import (
    DEFAULT_API_BASE,
    DEFAULT_MAX_QUESTIONS_PER_BATCH,
    JevBackend,
    normalize_api_root,
)
from tests._round145_sdk_stub import install as install_sdk_stub


@pytest.fixture(autouse=True)
def _typesafe_sdk_stub(monkeypatch):
    install_sdk_stub(monkeypatch)


def _cfg(**over):
    base = {
        "primary_model": "d3-flash",
        "api_base": "https://api.moark.com",
        "api_key": "cfg-key",
        "max_questions_per_batch": DEFAULT_MAX_QUESTIONS_PER_BATCH,
    }
    base.update(over)
    return base


class TestApiRootNormalization:
    def test_strips_v1_systemone_suffix(self):
        assert normalize_api_root("https://api.moark.com/v1/systemone") == "https://api.moark.com"

    def test_strips_v1_suffix(self):
        assert normalize_api_root("https://api.moark.com/v1/") == "https://api.moark.com"

    def test_keeps_root(self):
        assert normalize_api_root("https://api.moark.com/") == "https://api.moark.com"


class TestJevBackendWiring:
    def test_config_api_base_and_key_enter_client(self):
        with patch("src.judges.typesafe_backend._load_typesafe_config", return_value=_cfg()), \
             patch.dict(os.environ, {}, clear=True), \
             patch("typesafe_sdk.TypeSafeClient") as mock_client:
            backend = JevBackend()
            backend._ensure_client()
            kwargs = mock_client.call_args.kwargs
            assert kwargs["api_key"] == "cfg-key"
            assert kwargs["base_url"] == "https://api.moark.com"
            assert "http_client" not in kwargs

    def test_explicit_values_win(self):
        with patch("src.judges.typesafe_backend._load_typesafe_config", return_value=_cfg()), \
             patch.dict(os.environ, {}, clear=True):
            backend = JevBackend(
                api_key="explicit-key",
                model="d3-turbo",
                api_base="https://api.moark.com/v1/systemone",
                max_questions_per_batch=32,
            )
            assert backend.api_key == "explicit-key"
            assert backend.model == "d3-turbo"
            assert backend.api_base == "https://api.moark.com"
            assert backend.max_questions_per_batch == 32

    def test_config_wins_over_legacy_typesafe_env(self):
        # The machine may still export the expired Jev key as TYPESAFE_API_KEY.
        with patch("src.judges.typesafe_backend._load_typesafe_config", return_value=_cfg(api_key="moark-config")), \
             patch.dict(os.environ, {"TYPESAFE_API_KEY": "expired-jev", "MOARK_API_KEY": "moark-env"}, clear=True):
            backend = JevBackend()
            assert backend.api_key == "moark-config"

    def test_moark_env_fallback_when_config_key_empty(self):
        with patch("src.judges.typesafe_backend._load_typesafe_config", return_value=_cfg(api_key="")), \
             patch.dict(os.environ, {"TYPESAFE_API_KEY": "expired-jev", "MOARK_API_KEY": "moark-env"}, clear=True):
            backend = JevBackend()
            assert backend.api_key == "moark-env"

    def test_unresolved_placeholder_is_not_a_key(self):
        with patch("src.judges.typesafe_backend._load_typesafe_config", return_value=_cfg(api_key="${MOARK_API_KEY}")), \
             patch.dict(os.environ, {}, clear=True):
            backend = JevBackend()
            assert backend.api_key == ""

    def test_diagnostics_do_not_include_key(self):
        with patch("src.judges.typesafe_backend._load_typesafe_config", return_value=_cfg(api_key="secret-key")), \
             patch.dict(os.environ, {}, clear=True):
            backend = JevBackend(api_base=DEFAULT_API_BASE)
            backend.last_diagnostics = {"provider": "moark", "model": backend.model}
            assert "secret-key" not in repr(backend.last_diagnostics)
