"""Round 158: lock-path replacement must not create a second writer."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "translate_book.sh"


def _hold_lock(path: Path) -> int:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _close_instance_fds() -> None:
    import src.main as main

    for attr in ("_INSTANCE_LOCK_FD", "_INSTANCE_GUARD_FD"):
        fd = getattr(main, attr, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            setattr(main, attr, None)


def _shell_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("SAGATRANS_LOCK_HELD", None)
    env.pop("SAGATRANS_ALLOW_CONCURRENT", None)
    env["SAGATRANS_LOCK_PATH"] = str(tmp_path / "db" / ".sagatrans.lock")
    env["SAGATRANS_GUARD_LOCK_PATH"] = str(tmp_path / "runtime" / "guard.lock")
    env["SAGATRANS_LOCK_RUNTIME_DIR"] = str(tmp_path / "runtime")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    return env


class TestPythonRuntimeGuard:
    def test_guard_path_is_deterministic_and_outside_repo(self, tmp_path, monkeypatch):
        from src.main import _guard_lock_path

        monkeypatch.setenv("SAGATRANS_LOCK_RUNTIME_DIR", str(tmp_path / "runtime"))
        primary = tmp_path / "db" / ".sagatrans.lock"
        first = _guard_lock_path(primary)
        second = _guard_lock_path(primary)
        assert first == second
        assert first.parent == tmp_path / "runtime"
        assert REPO not in first.parents

    def test_guard_contention_rejects_even_when_primary_is_free(self, tmp_path, monkeypatch):
        from src.main import acquire_instance_lock

        primary = tmp_path / "db" / ".sagatrans.lock"
        guard = tmp_path / "runtime" / "guard.lock"
        monkeypatch.setenv("SAGATRANS_GUARD_LOCK_PATH", str(guard))
        fd = _hold_lock(guard)
        try:
            assert acquire_instance_lock(lock_path=primary) is False
        finally:
            os.close(fd)
            _close_instance_fds()


class TestShellLockLifecycle:
    def test_replaced_primary_path_fails_closed(self, tmp_path):
        env = _shell_env(tmp_path)
        script = r'''
source "$1"
acquire_instance_lock probe
rm -f "$SAGATRANS_LOCK_PATH"
: > "$SAGATRANS_LOCK_PATH"
assert_instance_lock_intact
echo SHOULD_NOT_REACH
'''
        proc = subprocess.run(
            ["bash", "-c", script, "bash", str(SCRIPT)], env=env,
            capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "inode 已被替换" in proc.stderr
        assert "SHOULD_NOT_REACH" not in proc.stdout

    def test_runtime_guard_blocks_replacement_writer(self, tmp_path):
        env = _shell_env(tmp_path)
        ready = tmp_path / "ready"
        first_script = r'''
source "$1"
acquire_instance_lock first
rm -f "$SAGATRANS_LOCK_PATH"
: > "$SAGATRANS_LOCK_PATH"
: > "$2"
sleep 20
'''
        first = subprocess.Popen(
            ["bash", "-c", first_script, "bash", str(SCRIPT), str(ready)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 8
            while not ready.exists() and first.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert ready.exists(), first.communicate(timeout=2)

            second_script = r'''
source "$1"
acquire_instance_lock second
echo SECOND_ACQUIRED
sleep 1
'''
            second = subprocess.run(
                ["bash", "-c", second_script, "bash", str(SCRIPT)], env=env,
                capture_output=True, text=True, timeout=20,
            )
            assert second.returncode == 1, second.stdout + second.stderr
            assert "运行期副锁已被持有" in second.stderr
            assert "SECOND_ACQUIRED" not in second.stdout
        finally:
            try:
                os.killpg(first.pid, 15)
            except ProcessLookupError:
                pass
            try:
                first.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(first.pid, 9)
                except ProcessLookupError:
                    pass
                first.communicate(timeout=3)


def test_run_cmd_checks_lock_before_and_after_each_child():
    text = SCRIPT.read_text(encoding="utf-8")
    start = text.index("run_cmd() {")
    end = text.index("\n}\n", start) + 3
    body = text[start:end]
    assert body.count("assert_instance_lock_intact") >= 2
    assert body.index("assert_instance_lock_intact") < body.index("PY_CHILD_ACTIVE=1")

