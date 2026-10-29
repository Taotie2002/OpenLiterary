"""Round 130: 外部审计四项修复的回归测试.

审计① 单实例约束只在文档 → flock 守卫（src/main.py；translate_book.sh 同源实现）
审计② gate ⑦ 顺序耦合   → run_final_gate 门禁内联自动补跑 pun_audit
审计③ 非法转移软拒绝     → update_task_state 返回 bool / batch_update_state 返回 valid_ids
审计④ 作品先验零残留漏网 → Config.critic_thresholds_for_book() 按书覆盖
                             （input/book_profile.json 的 "critic_thresholds" 块）
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _insert_task(sched, chunk_id, state, chapter_id="ch_test"):
    cursor = sched.conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO chunk_tasks
        (chunk_id, chapter_id, text_content, state, retries, quality_retries, last_error)
        VALUES (?, ?, ?, ?, 0, 0, NULL)
    ''', (chunk_id, chapter_id, "测试文本", state))
    sched.conn.commit()


def _get_state(sched, chunk_id):
    row = sched.conn.execute(
        'SELECT state FROM chunk_tasks WHERE chunk_id = ?', (chunk_id,)).fetchone()
    return row[0] if row else None


@pytest.fixture
def scheduler(tmp_path):
    from src.core.scheduler import TaskScheduler
    sched = TaskScheduler(db_path=str(tmp_path / "test_workflow.db"))
    yield sched
    try:
        sched.conn.close()
    except Exception:
        pass


def _hold_lock(path: Path):
    """在测试进程内独占持有 flock（不同 fd 与 acquire_instance_lock 冲突）。"""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


# ────────────────────────────────────────────────────────────────────
# 审计① 单实例 flock 守卫
# ────────────────────────────────────────────────────────────────────
class TestInstanceLockGuard:
    ENV_KEYS = ("SAGATRANS_LOCK_HELD", "SAGATRANS_ALLOW_CONCURRENT")

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        for k in self.ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        yield
        import src.main as m
        if m._INSTANCE_LOCK_FD is not None:
            try:
                os.close(m._INSTANCE_LOCK_FD)
            except OSError:
                pass
            m._INSTANCE_LOCK_FD = None

    def test_acquire_free_lock_and_write_holder(self, tmp_path):
        from src.main import acquire_instance_lock
        lockp = tmp_path / "db" / ".sagatrans.lock"
        assert acquire_instance_lock(lock_path=lockp) is True
        content = lockp.read_text(encoding="utf-8")
        assert f"pid={os.getpid()}" in content

    def test_contended_lock_rejected(self, tmp_path, capsys):
        from src.main import acquire_instance_lock
        lockp = tmp_path / "db" / ".sagatrans.lock"
        fd = _hold_lock(lockp)
        try:
            assert acquire_instance_lock(lock_path=lockp) is False
            assert "⛔" in capsys.readouterr().err
        finally:
            os.close(fd)

    def test_allow_concurrent_env_bypasses(self, tmp_path, monkeypatch, capsys):
        from src.main import acquire_instance_lock
        monkeypatch.setenv("SAGATRANS_ALLOW_CONCURRENT", "1")
        lockp = tmp_path / "db" / ".sagatrans.lock"
        fd = _hold_lock(lockp)
        try:
            assert acquire_instance_lock(lock_path=lockp) is True
            assert "放行" in capsys.readouterr().err
        finally:
            os.close(fd)

    def test_lock_held_env_skips_probe(self, tmp_path, monkeypatch):
        """脚本持锁后 export SAGATRANS_LOCK_HELD=1，子进程 CLI 直接放行且不触碰锁文件。"""
        from src.main import acquire_instance_lock
        monkeypatch.setenv("SAGATRANS_LOCK_HELD", "1")
        lockp = tmp_path / "db" / ".sagatrans.lock"
        lockp.parent.mkdir(parents=True, exist_ok=True)
        lockp.write_text("pid=99999 sentinel\n", encoding="utf-8")
        assert acquire_instance_lock(lock_path=lockp) is True
        assert "sentinel" in lockp.read_text(encoding="utf-8")

    def test_mutating_command_wiring(self):
        from src.main import _MUTATING_COMMANDS
        assert {"pipeline", "init", "split", "style-profile",
                "consolidate", "finalize", "notes-normalize"} <= _MUTATING_COMMANDS
        for readonly in ("gate", "lint", "debug", "golden", "golden-gate",
                         "memory", "drift-audit", "consistency"):
            assert readonly not in _MUTATING_COMMANDS

    def test_shell_script_has_same_lock_guard(self):
        text = (REPO / "scripts" / "translate_book.sh").read_text(encoding="utf-8")
        assert "flock -n 9" in text
        assert "SAGATRANS_LOCK_HELD=1" in text

    @pytest.mark.skipif(sys.platform == "win32", reason="fcntl/flock Linux-only")
    def test_cli_rejects_mutating_command_under_lock(self, tmp_path):
        lockp = tmp_path / "db" / ".sagatrans.lock"
        fd = _hold_lock(lockp)
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, "-m", "src.translator_agent", "init", "--chapter", "ch01"],
                cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
            assert proc.returncode == 1, proc.stdout + proc.stderr
            assert "已有 SagaTrans 实例" in (proc.stdout + proc.stderr)
            # 守卫在 init 之前生效：未创建 workflow.db
            assert not (tmp_path / "db" / "workflow.db").exists()
        finally:
            os.close(fd)


# ────────────────────────────────────────────────────────────────────
# 审计② gate ⑦ 内联自动补跑 pun_audit
# ────────────────────────────────────────────────────────────────────
def _build_gate_project(root: Path, with_pun_watchlist: bool = True) -> None:
    from src.core.scheduler import TaskScheduler, TaskState
    (root / "db").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(exist_ok=True)
    (root / "input").mkdir(exist_ok=True)
    sched = TaskScheduler(db_path=str(root / "db" / "workflow.db"))
    _insert_task(sched, "ch01_chunk000", TaskState.COMPLETED.value, chapter_id="ch01")
    sched.conn.close()
    outch = root / "output" / "ch01"
    outch.mkdir()
    (outch / "ch01_chunk000_final.json").write_text(json.dumps(
        {"text": "他说了一句双关语：猫头鹰与夜莺。", "metadata": {}},
        ensure_ascii=False), encoding="utf-8")
    (root / "output" / "QA_REPORT.md").write_text("# QA\n\n全部通过\n", encoding="utf-8")
    (root / "input" / "_conversion_audit.md").write_text(
        "全书覆盖率: 100.0%\n\n✅ 通过\n", encoding="utf-8")
    if with_pun_watchlist:
        profile = {"pun_watchlist": [{
            "source": "owl/nightingale pun",
            "anchors": ["猫头鹰"],
            "strategy": "直译+注释",
        }]}
        (root / "input" / "book_profile.json").write_text(
            json.dumps(profile, ensure_ascii=False), encoding="utf-8")
        con = sqlite3.connect(str(root / "db" / "decision_db.sqlite"))
        con.execute("CREATE TABLE IF NOT EXISTS decision_db "
                    "(source_key TEXT, translation TEXT, reason TEXT)")
        con.execute("INSERT INTO decision_db VALUES (?, ?, ?)",
                    ("owl/nightingale pun", "猫头鹰与夜莺", "双关保留"))
        con.commit()
        con.close()


class TestGateInlinePunAudit:
    @pytest.fixture(autouse=True)
    def _py_path(self, monkeypatch):
        monkeypatch.setenv(
            "PYTHONPATH", str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def test_gate_auto_runs_pun_audit_and_goes_green(self, tmp_path):
        from src.pipeline import run_final_gate
        _build_gate_project(tmp_path, with_pun_watchlist=True)
        # 第一轮：无脚本 → ⑦ 必须红（缺 pun_audit.json 且无法自动补跑）
        res1 = run_final_gate(project_root=tmp_path)
        assert any("⑦" in f and "未运行" in f for f in res1["failures"]), res1["failures"]
        assert not (tmp_path / "output" / "pun_audit.json").exists()
        # 补上脚本（模拟 repo 正常态）→ 门禁自动补跑 pun_audit，转绿
        (tmp_path / "scripts").mkdir()
        shutil.copy(REPO / "scripts" / "pun_audit.py", tmp_path / "scripts" / "pun_audit.py")
        res2 = run_final_gate(project_root=tmp_path)
        assert (tmp_path / "output" / "pun_audit.json").exists()
        assert not any("⑦" in f for f in res2["failures"]), res2["failures"]
        assert res2["ok"] is True, res2["failures"]
        assert res2["stats"]["pun"]["total"] == 1
        assert res2["stats"]["pun"]["uncovered_nonwaived"] == 0

    def test_gate_skips_pun_without_watchlist(self, tmp_path):
        from src.pipeline import run_final_gate
        _build_gate_project(tmp_path, with_pun_watchlist=False)
        res = run_final_gate(project_root=tmp_path)
        assert not any("⑦" in f for f in res["failures"]), res["failures"]
        assert res["ok"] is True, res["failures"]
        assert not (tmp_path / "output" / "pun_audit.json").exists()


# ────────────────────────────────────────────────────────────────────
# 审计③ 状态转移返回语义
# ────────────────────────────────────────────────────────────────────
class TestSchedulerTransitionReturnSemantics:
    def test_legal_transition_returns_true(self, scheduler):
        from src.core.scheduler import TaskState
        _insert_task(scheduler, "ch_test_chunk000", TaskState.PENDING.value)
        assert scheduler.update_task_state(
            "ch_test_chunk000", TaskState.EXTRACTING_TERMS) is True
        assert _get_state(scheduler, "ch_test_chunk000") == TaskState.EXTRACTING_TERMS.value

    def test_illegal_transition_returns_false_and_state_unchanged(self, scheduler):
        from src.core.scheduler import TaskState
        _insert_task(scheduler, "ch_test_chunk000", TaskState.COMPLETED.value)
        assert scheduler.update_task_state(
            "ch_test_chunk000", TaskState.COMPLETED) is False
        assert _get_state(scheduler, "ch_test_chunk000") == TaskState.COMPLETED.value

    def test_terminal_restore_flag_allows_pf_to_completed(self, scheduler):
        from src.core.scheduler import TaskState
        _insert_task(scheduler, "ch_test_chunk000", TaskState.PERMANENTLY_FAILED.value)
        # 无旗标：拒绝（返回 False，状态不变）
        assert scheduler.update_task_state(
            "ch_test_chunk000", TaskState.COMPLETED) is False
        assert _get_state(scheduler, "ch_test_chunk000") == TaskState.PERMANENTLY_FAILED.value
        # 显式旗标（_restore_dirty_backups 专用）：放行
        assert scheduler.update_task_state(
            "ch_test_chunk000", TaskState.COMPLETED, allow_terminal_restore=True) is True
        assert _get_state(scheduler, "ch_test_chunk000") == TaskState.COMPLETED.value

    def test_batch_update_returns_valid_ids_only(self, scheduler):
        from src.core.scheduler import TaskState
        _insert_task(scheduler, "ch_test_chunk000", TaskState.PENDING.value)
        _insert_task(scheduler, "ch_test_chunk001", TaskState.AUDITING.value)
        updated = scheduler.batch_update_state(
            ["ch_test_chunk000", "ch_test_chunk001", "ch_test_chunk999"],
            TaskState.EXTRACTING_TERMS)
        assert updated == ["ch_test_chunk000"]
        assert _get_state(scheduler, "ch_test_chunk000") == TaskState.EXTRACTING_TERMS.value
        assert _get_state(scheduler, "ch_test_chunk001") == TaskState.AUDITING.value
        assert _get_state(scheduler, "ch_test_chunk999") is None

    def test_batch_update_empty_returns_empty_list(self, scheduler):
        from src.core.scheduler import TaskState
        assert scheduler.batch_update_state([], TaskState.EXTRACTING_TERMS) == []

    def test_pipeline_restore_paths_perceive_return_value(self):
        """两处历史上会假报状态的恢复路径必须感知返回值（Round 130 收紧）。"""
        text = (REPO / "src" / "pipeline.py").read_text(encoding="utf-8")
        assert "if self.scheduler.update_task_state(chunk_id, TaskState.DIRTY):" in text
        assert "_state_ok = self.scheduler.update_task_state(" in text

    def test_monolith_contains_round130_symbols(self):
        text = (REPO / "src" / "translator_agent.py").read_text(encoding="utf-8")
        assert "def acquire_instance_lock" in text
        assert "_MUTATING_COMMANDS" in text
        assert "def critic_thresholds_for_book" in text
        assert "allow_terminal_restore" in text


class TestMonolithSplitterPackaging:
    """Round 130 事故: monolith 缺 _input_prep_cfg 等打包符号 → split NameError。

    build_integrated 的 Section 13 spec 曾漏掉 input_prep 助手与
    _CHAPTER_TITLE_PATTERNS；translate_book.sh 的 split 步骤直接崩溃。
    """

    def test_input_prep_helpers_present_in_monolith(self):
        text = (REPO / "src" / "translator_agent.py").read_text(encoding="utf-8")
        for frag in ("def _input_prep_cfg", "def _prep_key", "def _chapter_h1",
                     "def _apply_input_repairs", "_CHAPTER_TITLE_PATTERNS = ["):
            assert frag in text, f"monolith 缺少打包符号: {frag}"

    def test_monolith_split_end_to_end(self, tmp_path):
        book = tmp_path / "mini.txt"
        book.write_text(
            "Chapter 1\n" + ("the sleeper must awaken " * 120) + "\n\n"
            "Chapter 2\n" + ("a beginning is a very delicate time " * 100) + "\n",
            encoding="utf-8")
        outdir = tmp_path / "out"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "src.translator_agent", "split",
             "--input", str(book), "--input-dir", str(outdir)],
            cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "NameError" not in (proc.stdout + proc.stderr)
        assert list(outdir.glob("ch*.md")), proc.stdout


class TestPython310SyntaxCompat:
    """README 声明支持 >=3.10。

    事故（外部评审）: critic_agent f-string 内嵌同型双引号需 PEP 701（3.12+），
    3.10 生产解释器下 import 全瘫（散装+单体 SyntaxError）。
    守卫: 有 python3.10 解释器则全库 compile 扫描；无则跳过。
    """

    CHECKER = textwrap.dedent('''
        import sys
        from pathlib import Path
        bad, n = [], 0
        for p in sorted(Path(sys.argv[1]).rglob("*.py")):
            s = str(p)
            if any(seg in s for seg in ("/.venv", "/venv/", "node_modules", "/.git/")):
                continue
            n += 1
            try:
                compile(p.read_text(encoding="utf-8"), s, "exec")
            except SyntaxError as e:
                bad.append(f"{s}: {e.msg} (line {e.lineno})")
        print(f"scanned={n} syntax_errors={len(bad)}")
        for b in bad:
            print(" ", b)
        sys.exit(1 if bad else 0)
    ''')

    def test_repo_compiles_under_python310(self, tmp_path):
        py310 = shutil.which("python3.10")
        if not py310:
            pytest.skip("本机无 python3.10 解释器（可选守卫）")
        checker = tmp_path / "check_py310.py"
        checker.write_text(self.CHECKER, encoding="utf-8")
        proc = subprocess.run([py310, str(checker), str(REPO)],
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, (
            f"3.10 语法不兼容:\n{proc.stdout}\n{proc.stderr}")


# ────────────────────────────────────────────────────────────────────
# 审计④ 按书 critic 阈值覆盖
# ────────────────────────────────────────────────────────────────────
class TestCriticThresholdsForBook:
    DEFAULTS = {"style_compliance": 6.0, "voice_drift": 7.0, "register_match": 6.5}

    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        from src.config import Config
        monkeypatch.setattr(Config, "_book_thresh_cache", {}, raising=False)
        cfg = Config()
        monkeypatch.setitem(cfg._data, "critic_thresholds", dict(self.DEFAULTS))

    def test_defaults_without_profile(self, tmp_path):
        from src.config import Config
        merged = Config().critic_thresholds_for_book(project_root=tmp_path)
        assert merged == self.DEFAULTS

    def test_profile_override_merges_and_filters_junk(self, tmp_path):
        from src.config import Config
        prof = {"critic_thresholds": {"style_compliance": 8.5, "bad": "x",
                                      "neg": -1, "big": 11.0, "flag": True}}
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "book_profile.json").write_text(
            json.dumps(prof), encoding="utf-8")
        merged = Config().critic_thresholds_for_book(project_root=tmp_path)
        assert merged["style_compliance"] == 8.5
        assert merged["voice_drift"] == 7.0
        for junk in ("bad", "neg", "big", "flag"):
            assert junk not in merged

    def test_profile_rewrite_invalidates_cache(self, tmp_path):
        from src.config import Config
        d = tmp_path / "input"
        d.mkdir()
        f = d / "book_profile.json"
        f.write_text(json.dumps({"critic_thresholds": {"style_compliance": 8.5}}),
                     encoding="utf-8")
        assert Config().critic_thresholds_for_book(
            project_root=tmp_path)["style_compliance"] == 8.5
        f.write_text(json.dumps({"critic_thresholds": {"style_compliance": 9.9}}),
                     encoding="utf-8")
        st = f.stat()
        os.utime(f, ns=(st.st_mtime_ns + 10 ** 9, st.st_mtime_ns + 10 ** 9))
        assert Config().critic_thresholds_for_book(
            project_root=tmp_path)["style_compliance"] == 9.9

    def test_corrupt_profile_falls_back(self, tmp_path):
        from src.config import Config
        (tmp_path / "input").mkdir()
        (tmp_path / "input" / "book_profile.json").write_text("{ not json", encoding="utf-8")
        assert Config().critic_thresholds_for_book(project_root=tmp_path) == self.DEFAULTS

    def test_profile_normalize_keeps_threshold_block(self):
        """审计④机制前提：剖面归一化不得吞掉 critic_thresholds（载入往返保留）。"""
        from src.style_profiler import normalize_profile
        prof = normalize_profile({"critic_thresholds": {"style_compliance": 8.5, "junk": "x"}})
        assert prof["critic_thresholds"] == {"style_compliance": 8.5}

    def test_heuristic_profile_has_empty_threshold_block(self):
        from src.style_profiler import heuristic_profile
        assert heuristic_profile()["critic_thresholds"] == {}
