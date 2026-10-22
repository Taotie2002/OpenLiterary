"""Round 75: 修复 cost_report 的 chapter_dirs NameError, 让 Jev 边界判断 / 不确定跳过 统计表
   真正渲染.

根因: Round 62 引入 Jev skipped_dimensions 统计, 但其内层 for 循环引用 `chapter_dirs`
      变量, 该变量从未在 _print_llm_cost_report 函数体内定义.
      外层 try/except Exception 静默吞掉 NameError, 导致整段统计从未渲染.
修复: 在内层 for 之前显式定义 `chapter_dirs = [self.output_dir]`
      (本章节所有 critic_report.json 都在 self.output_dir 目录下, 单层目录).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
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



class TestSourceHasChapterDirsDefinition:
    """散装 src/pipeline.py 必须显式定义 chapter_dirs."""

    def test_chapter_dirs_defined_before_loop(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        func_start = content.find("def _print_llm_cost_report")
        func_end = content.find("\n    def ", func_start + 1)
        func_body = content[func_start:func_end]
        # chapter_dirs 必须有定义
        assert "chapter_dirs = " in func_body, (
            "Round 75: chapter_dirs 必须显式定义, 否则 skipped_dimensions 统计静默失败"
        )
        # 定义必须在 for 循环之前
        def_idx = func_body.find("chapter_dirs = ")
        for_idx = func_body.find("for chunk_dir in chapter_dirs:")
        assert def_idx > 0 and for_idx > 0, "chapter_dirs 定义或 for 循环缺失"
        assert def_idx < for_idx, f"chapter_dirs 定义 ({def_idx}) 必须在 for 循环 ({for_idx}) 之前"

    def test_round_75_marker_present(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "Round 75" in content or "chapter_dirs" in content

    def test_syntax_check(self):
        """防止 Round 75 引入 syntax error (历史教训: Round 75 第一次实现时把 chapter_dirs
           定义放在了 for 之后, 导致 SyntaxError)."""
        import ast
        ast.parse(Path("src/pipeline.py").read_text(encoding="utf-8"))


class TestIntegratedScript:
    """整合脚本同步导出 Round 75 修复."""

    def test_translation_pipeline_importable(self):
        """Round 75 fix 第一次破坏了 TranslationPipeline 的导入 (因 syntax error),
        测试保证修复后能正常导入."""
        from src.translator_agent import TranslationPipeline
        assert TranslationPipeline is not None

    def test_integrated_source_has_chapter_dirs(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "chapter_dirs" in content


class TestCostReportBehavior:
    """模拟 critic_report.json 含 skipped_dimensions, 验证 Jev 边界判断 / 不确定跳过 表渲染."""

    def _make_pipeline(self, output_dir):
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_test"
        pipeline.output_dir = output_dir
        pipeline._llm_call_counter = {
            "reference_extraction": 0,
            "literal_translation": 0,
            "literary_rewriting": 0,
            "critic_scoring": 0,
            "judge_decision": 0,
        }
        pipeline.decision_engine = None  # 跳过 entity registry 快照
        return pipeline

    def test_jev_skipped_dimensions_table_renders(self, tmp_path):
        """Round 75 fix: Jev skipped_dimensions 表必须真正渲染 (不再被 NameError 吞掉)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)

        # 写一个含 skipped_dimensions 的 critic_report
        critic_report = {
            "scores": {"voice_consistency": 7.0, "literal_residue": 5.0},
            "_backend_path": "mock_heuristic",
            "_typesafe_verdict": {
                "decision": "PASS",
                "skipped_dimensions": ["entity_consistency", "literal_residue"],
            },
        }
        with open(output_dir / "ch_test_chunk000_critic_report.json", "w", encoding="utf-8") as f:
            json.dump(critic_report, f, ensure_ascii=False)
        # 写一个空 final.json 让 quality_summary 段渲染
        with open(output_dir / "ch_test_chunk000_final.json", "w", encoding="utf-8") as f:
            json.dump({"text": "...", "metadata": {}}, f, ensure_ascii=False)

        pipeline = self._make_pipeline(output_dir)
        try:
            pipeline._print_llm_cost_report()
        except Exception as e:
            pytest.fail(f"_print_llm_cost_report raised: {e}")

        report_text = (output_dir / "cost_report.md").read_text(encoding="utf-8")
        # Jev 边界判断 / 不确定跳过 表必须渲染
        assert "Jev 边界判断" in report_text, (
            "Round 75 修复后 Jev 边界判断 表必须渲染. 之前因 chapter_dirs NameError "
            "被静默吞掉, 用户从未看到这个统计"
        )
        assert "entity_consistency" in report_text
        assert "literal_residue" in report_text

    def test_no_skipped_dimensions_table_omitted(self, tmp_path):
        """没有 skipped_dimensions 数据时, 表应省略 (不显示空表)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)

        # critic_report 没有 skipped_dimensions
        critic_report = {
            "scores": {"voice_consistency": 7.0},
            "_backend_path": "mock_heuristic",
            "_typesafe_verdict": {"decision": "PASS"},
        }
        with open(output_dir / "ch_test_chunk000_critic_report.json", "w", encoding="utf-8") as f:
            json.dump(critic_report, f, ensure_ascii=False)
        with open(output_dir / "ch_test_chunk000_final.json", "w", encoding="utf-8") as f:
            json.dump({"text": "...", "metadata": {}}, f, ensure_ascii=False)

        pipeline = self._make_pipeline(output_dir)
        pipeline._print_llm_cost_report()

        report_text = (output_dir / "cost_report.md").read_text(encoding="utf-8")
        # 没有 skipped_dimensions → 不应有 "跳过" 字样的统计
        # (但可以有其他表格的 "跳过" 字样, 如 backend diagnostics)
        # 这里只检查 cost_report 中没有 "Jev 边界判断" 表头
        assert "## Jev 边界判断" not in report_text
