"""Round 146: deployment contract for one-key translation."""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _script() -> str:
    return (REPO / "scripts" / "translate_book.sh").read_text(encoding="utf-8")


def test_translate_book_has_single_python_contract():
    text = _script()
    assert 'PYTHON="${SAGATRANS_PYTHON:-}"' in text
    assert '"$PYTHON" -m src.translator_agent' in text
    assert '"$PYTHON" -c' in text
    assert "python3 -m src.translator_agent" not in text
    assert "python3 -c " not in text


def test_preflight_checks_system_one_dependency_and_key():
    text = _script()
    assert "import typesafe_sdk, httpx2" in text
    assert "MOARK_API_KEY" in text
    assert "typesafe.api_key unresolved" in text


def test_shell_and_python_lock_path_can_be_aligned():
    text = _script()
    assert 'SAGATRANS_LOCK_PATH:-$PROJECT_ROOT/db/.sagatrans.lock' in text


def test_gate_runs_before_clean_intermediate():
    text = _script()
    gate_idx = text.index("# 4.8 T4: 交付门禁 (gate)")
    clean_idx = text.index("# 5. 清理中间产物")
    assert gate_idx < clean_idx


def test_clean_runs_only_after_green_gate():
    text = _script()
    assert 'if [[ "$CLEAN_INTERMEDIATE" == "true" && "$gate_ok" == "true" ]]' in text
    assert "gate 未通过：保留 raw/literary/critic" in text
