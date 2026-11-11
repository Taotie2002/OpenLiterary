"""Round 146: .env is the local secret source; shell env is fallback only."""
from __future__ import annotations

import os

from src.config import Config, _read_dotenv


def test_read_dotenv_does_not_export_values(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGATRANS_ROUND146_SOURCE_TEST", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "export SAGATRANS_ROUND146_SOURCE_TEST='from-dotenv'\n"
        "SAGATRANS_ROUND146_OTHER=\"other\"\n",
        encoding="utf-8",
    )

    values = _read_dotenv(env_file)

    assert values["SAGATRANS_ROUND146_SOURCE_TEST"] == "from-dotenv"
    assert values["SAGATRANS_ROUND146_OTHER"] == "other"
    assert "SAGATRANS_ROUND146_SOURCE_TEST" not in os.environ


def test_dotenv_value_wins_over_shell_for_placeholder(monkeypatch):
    monkeypatch.setenv("SAGATRANS_ROUND146_PRECEDENCE", "shell")

    assert Config._expand_env_vars(
        "${SAGATRANS_ROUND146_PRECEDENCE}",
        {"SAGATRANS_ROUND146_PRECEDENCE": "dotenv"},
    ) == "dotenv"


def test_shell_is_fallback_when_dotenv_has_no_value(monkeypatch):
    monkeypatch.setenv("SAGATRANS_ROUND146_FALLBACK", "shell")

    assert Config._expand_env_vars(
        "${SAGATRANS_ROUND146_FALLBACK}", {}
    ) == "shell"


def test_missing_placeholder_stays_visible():
    assert Config._expand_env_vars("${SAGATRANS_ROUND146_MISSING}", {}) == \
        "${SAGATRANS_ROUND146_MISSING}"
