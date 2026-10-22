"""Round 78: 修复 src/main.py 散装入口缺失的 import, 让 `python3 -m src.main` 真正可用.

根因: src/main.py 是散装 CLI 入口, 但顶部 import 仅 run_consistency_check.
      函数体内引用了 11+ 个散装符号:
        - TaskScheduler (src.core.scheduler)
        - SmartChunker (src.utils.chunker)
        - _config (src.config)
        - TranslationPipeline (src.pipeline)
        - run_golden_test (src.test_golden_set)
        - run_golden_gate (src.test_memory_pressure) [历史遗留命名]
        - test_memory_pressure (src.test_memory_pressure)
        - EpubSplitter, TextSplitter, MdSplitter (src.splitters)
        - _EPUB_DEPS_OK, _EPUB_DEPS_ERROR (src.splitters, conditional)
        - _splitter_log_ok, _splitter_log_info (src.splitters)
      这些符号从未在顶部 import — 任何 `python3 -m src.main <cmd>` 都会 NameError.

修复: 在 src/main.py 顶部补全所有 import. 让散装入口与整合脚本行为对齐,
      兑现"散装 = 事实源"的设计原则.

影响: 开发者/CI 可以直接用散装模式跑 CLI, 无需依赖整合脚本.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_src_main_module():
    """每次测试前清空 src.main 模块缓存, 确保我们 import 的是当前磁盘版本.

    隔离 src.* 命名空间避免污染父包.
    """
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        for k in polluted:
            sys.modules.pop(k, None)
        yield
    finally:
        # Round 103 P6+: 动态 polluted 自动覆盖 src.* 父包 + 所有子模块,
        # 父包循环已是冗余, 不再单独遍历 parents.
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)



class TestSourcePatch:
    """散装 src/main.py 必须补全顶部 import."""

    def test_round_78_marker_present(self):
        content = Path("src/main.py").read_text(encoding="utf-8")
        assert "Round 78" in content

    def test_required_imports_present(self):
        """所有缺失的 import 必须出现在 src/main.py 顶部."""
        content = Path("src/main.py").read_text(encoding="utf-8")
        # 取顶部 import 段 (从 src.consistency 到第一个非 import 行)
        # 我们只检查关键 import 是否被引用, 不强制顶部位置 (允许 Round 78 加注释)
        required = [
            "from src.config import",
            "from src.utils.chunker import",
            "from src.core.scheduler import",
            "from src.pipeline import",
            "from src.test_golden_set import",
            "from src.test_memory_pressure import",
            "from src.splitters import",
        ]
        for imp in required:
            assert imp in content, f"Round 78: src/main.py 缺少 {imp!r}"

    def test_syntax_check(self):
        import ast
        ast.parse(Path("src/main.py").read_text(encoding="utf-8"))


class TestImportBehavior:
    """实际 import 行为: src.main 必须能加载且不抛 NameError."""

    def test_src_main_module_loads(self):
        """核心修复: from src.main import ... 不能触发 NameError."""
        # 不污染 sys.modules — 我们 import 后立刻清掉
        import src.main as m
        # 关键符号应该能在 src.main 命名空间中找到 (Round 78 import 引入)
        for name in ("_config", "SmartChunker", "TaskScheduler", "TranslationPipeline",
                     "run_golden_test", "run_golden_gate", "test_memory_pressure",
                     "EpubSplitter", "TextSplitter", "MdSplitter"):
            assert hasattr(m, name), f"src.main 应暴露 {name}, 但未找到"

    def test_init_project_function_callable(self):
        """init_project 函数必须可调用 (不能因 NameError 加载失败)."""
        import src.main as m
        assert callable(m.init_project)

    def test_main_function_callable(self):
        """main() 函数必须可调用."""
        import src.main as m
        assert callable(m.main)


class TestCliExecution:
    """实际跑 `python3 -m src.main --help` 必须成功."""

    def test_help_command_runs(self):
        """Round 78 核心: --help 必须能跑通 (旧版本会 NameError)."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "--help"],
            capture_output=True, text=True, timeout=15,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0, (
            f"`python3 -m src.main --help` 失败: returncode={result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )
        # 必须显示 CLI 选项 (Round 78 前会因为 NameError 提早退出)
        assert "command" in result.stdout.lower() or "chapter" in result.stdout.lower()

    def test_debug_command_runs(self):
        """debug 子命令必须能跑通 (旧版本会 NameError on TaskScheduler)."""
        result = subprocess.run(
            [sys.executable, "-m", "src.main", "debug"],
            capture_output=True, text=True, timeout=15,
            cwd=Path(__file__).resolve().parent.parent,
        )
        # debug 应该不报 NameError (它读 DB 输出 task 列表)
        # 即使 DB 是空的也应正常退出
        assert "NameError" not in result.stderr, (
            f"`python3 -m src.main debug` 触发 NameError: {result.stderr[:500]}"
        )
        assert "TaskScheduler" not in result.stderr or "未定义" not in result.stderr


class TestNoNameErrorOnFunctionBodies:
    """函数体内的 NameError 必须被 Round 78 import 修复消除."""

    def test_init_project_body_has_all_deps(self):
        """init_project 函数体内引用的符号都必须在模块顶层 import."""
        import src.main as m
        # 强制访问 init_project 源码 (静态分析其闭包依赖)
        import inspect
        src_text = inspect.getsource(m.init_project)
        # 这些符号如果不在顶层 import, 函数被调用时会 NameError
        for name in ("TaskScheduler", "_config", "SmartChunker"):
            assert name in src_text, f"init_project 引用了 {name}"

        # 关键: 这些符号必须能在模块命名空间中找到
        for name in ("TaskScheduler", "_config", "SmartChunker"):
            assert hasattr(m, name), (
                f"Round 78: {name} 必须在 src.main 顶层 import, "
                f"否则 init_project 调用时会 NameError"
            )

    def test_main_body_has_all_deps(self):
        """main() 函数体内引用的符号都必须在模块顶层 import."""
        import src.main as m
        for name in ("TranslationPipeline", "run_golden_test", "run_golden_gate",
                     "test_memory_pressure", "split_input_to_chapters"):
            assert hasattr(m, name), (
                f"Round 78: {name} 必须在 src.main 顶层 import"
            )

    def test_split_input_to_chapters_body_has_all_deps(self):
        """split_input_to_chapters 函数体内引用的符号都必须在模块顶层 import."""
        import src.main as m
        for name in ("EpubSplitter", "TextSplitter", "MdSplitter",
                     "_EPUB_DEPS_OK", "_splitter_log_ok", "_splitter_log_info"):
            assert hasattr(m, name), (
                f"Round 78: {name} 必须在 src.main 顶层 import"
            )
