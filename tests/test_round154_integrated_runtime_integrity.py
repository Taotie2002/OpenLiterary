"""Round 154: integrated snapshot must be runtime-complete, not merely equal.

``build_integrated.py`` assembles the single-file snapshot from hand-maintained
symbol lists.  ``--check`` used to compare bytes only, so a missing helper
inside a function body stayed invisible until production hit that branch.
These tests lock the free-name scan and the three production-critical symbols
that were absent from the snapshot.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
    polluted = tuple(k for k in list(sys.modules) if k == "src" or k.startswith("src."))
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        for key in polluted:
            if snapshot.get(key) is not None:
                sys.modules[key] = snapshot[key]
            else:
                current = sys.modules.get(key)
                if current is not None and getattr(current, "__file__", None) is None:
                    sys.modules.pop(key, None)


def _load_builder():
    path = ROOT / "scripts" / "build_integrated.py"
    spec = importlib.util.spec_from_file_location("_round154_build_integrated", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestIntegratedFreeNames:
    def test_generated_snapshot_has_no_undefined_globals(self):
        source = (ROOT / "src" / "translator_agent.py").read_text(encoding="utf-8")
        missing = _load_builder().find_undefined_globals(source)
        assert missing == [], f"integrated snapshot has undefined globals: {missing}"

    def test_scan_detects_a_deliberate_omission(self):
        builder = _load_builder()
        missing = builder.find_undefined_globals(
            "def f():\n    return _round154_missing_helper()\n"
        )
        assert any(name == "_round154_missing_helper" for name, _ctx in missing)

    def test_output_mode_keeps_stdout_machine_clean(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_integrated.py"), "--output"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-1000:]
        assert proc.stdout.startswith("#!/usr/bin/env python3")
        assert "自由名完整性" not in proc.stdout


class TestIntegratedRuntimeSymbols:
    def test_production_helpers_exist_and_are_callable(self):
        import src.translator_agent as ta

        for name in (
            "_typesafe_sdk_available",
            "_snapshot_sqlite_database",
            "backend_confidence_source",
            "run_injection_ab",
        ):
            assert callable(getattr(ta, name, None)), f"{name} missing from snapshot"

    def test_sdk_probe_does_not_raise_nameerror(self):
        import src.translator_agent as ta

        assert isinstance(ta._typesafe_sdk_available(), bool)

    def test_sqlite_snapshot_is_independent(self, tmp_path):
        import src.translator_agent as ta

        source = tmp_path / "source.sqlite"
        dest = tmp_path / "dest.sqlite"
        conn = sqlite3.connect(source)
        conn.execute("CREATE TABLE t (value TEXT)")
        conn.execute("INSERT INTO t VALUES ('round154')")
        conn.commit()
        conn.close()

        ta._snapshot_sqlite_database(source, dest)
        copy = sqlite3.connect(dest)
        try:
            assert copy.execute("SELECT value FROM t").fetchone() == ("round154",)
        finally:
            copy.close()

    def test_backend_confidence_source_prefers_last_backend(self):
        import src.translator_agent as ta

        class Backend:
            backend_kind = "jev"

        backend = Backend()
        backend.last_backend = "jev"
        assert ta.backend_confidence_source(backend) == "jev"
        del backend.last_backend
        assert ta.backend_confidence_source(backend) == "jev"
