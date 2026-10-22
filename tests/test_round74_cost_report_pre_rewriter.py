"""Round 74: cost_report 必须识别 Round 73 新增的 PRE_REWRITER_STAGNATION 状态.

背景: Round 73 在 _process_rewriting_batch 头部引入 pre-rewriter 短路, 节省 LLM 调用.
      但 _print_llm_cost_report 的 quality_counter / status_emoji / early_stop_count
      / per-chunk emoji 都还停留在 4 个 fallback 状态, 没把 Round 73 的新状态纳入统计.
      运维无法量化 Round 73 收益.
修复: 把 PRE_REWRITER_STAGNATION 加入所有 cost_report 字典 + emoji 映射.
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



class TestCostReportSourceIncludesNewStatus:
    """散装 src/pipeline.py 的 _print_llm_cost_report 必须包含 PRE_REWRITER_STAGNATION."""

    def test_quality_counter_includes_pre_rewriter(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        # quality_counter 字典初始化处 (在 _print_llm_cost_report 内)
        # 找到第一次出现 quality_counter = { 后, 寻找是否含 PRE_REWRITER_STAGNATION
        idx = content.find('"PRE_REWRITER_STAGNATION": 0')
        assert idx > 0, "quality_counter 未包含 PRE_REWRITER_STAGNATION"
        # 该字典应在 _print_llm_cost_report 函数体内 (Round 32 引入)
        # 检查字典位置在 _print_llm_cost_report 函数内
        func_start = content.find("def _print_llm_cost_report")
        func_end = content.find("\n    def ", func_start + 1)
        func_body = content[func_start:func_end]
        assert '"PRE_REWRITER_STAGNATION": 0' in func_body

    def test_status_emoji_includes_pre_rewriter(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        func_start = content.find("def _print_llm_cost_report")
        func_end = content.find("\n    def ", func_start + 1)
        func_body = content[func_start:func_end]
        # 至少一处 status_emoji 字典包含 PRE_REWRITER_STAGNATION
        assert '"PRE_REWRITER_STAGNATION":' in func_body

    def test_per_chunk_emoji_map_includes_pre_rewriter(self):
        """Per-Chunk 摘要表 (Round 37) 的 emoji 映射也要包含 PRE_REWRITER_STAGNATION."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        func_start = content.find("def _print_llm_cost_report")
        func_end = content.find("\n    def ", func_start + 1)
        func_body = content[func_start:func_end]
        # Per-Chunk emoji 字典 (单行 .get())
        # 形如 emoji = {"SAFETY_NET_PASSED": "🟡", ..., "PRE_REWRITER_STAGNATION": "⏭️"}.get(...)
        assert "PRE_REWRITER_STAGNATION" in func_body

    def test_retry_efficiency_counts_pre_rewriter(self):
        """Round 74: early_stop_count 应包含 PRE_REWRITER_STAGNATION."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        func_start = content.find("def _print_llm_cost_report")
        func_end = content.find("\n    def ", func_start + 1)
        func_body = content[func_start:func_end]
        # 检查 reason in (...) 元组包含 PRE_REWRITER_STAGNATION
        assert '("EARLY_STOP", "REWRITER_STAGNATED", "PRE_REWRITER_STAGNATION")' in func_body


class TestIntegratedScriptExports:
    """整合脚本同步导出 Round 74 改动."""

    def test_integrated_source_includes_pre_rewriter_counters(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        # 整合脚本包含 Round 73 PRE_REWRITER_STAGNATION 字面量 (multiple occurrences)
        n = content.count("PRE_REWRITER_STAGNATION")
        assert n >= 3, f"整合脚本应至少 3 处 PRE_REWRITER_STAGNATION (counter + emoji + reason), 实际 {n}"


class TestCostReportBehavior:
    """用模拟 final.json 验证 cost_report 正确分类 PRE_REWRITER_STAGNATION.

    Round 13 起, cost_report 写入 self.output_dir / "cost_report.md" (Round 41 还写 cost_summary.json),
    测试直接读这两个文件.
    """

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
        # Round 40: decision_engine 是 entity registry 快照必需, 用 None 让 _print_llm_cost_report 跳过
        pipeline.decision_engine = None
        return pipeline

    def test_pre_rewriter_status_renders_in_table(self, tmp_path):
        """_print_llm_cost_report 写入 cost_report.md, 应在 翻译质量摘要 表中显示 PRE_REWRITER_STAGNATION 行."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)

        final_data = {
            "text": "译文文本",
            "metadata": {
                "fallback": True,
                "reason": "PRE_REWRITER_STAGNATION",
                "judge_reason": "stagnant categories: 实体一致性违反, 直译残留",
                "quality_retries": 4,
                "stagnant_categories": ["实体一致性违反", "直译残留"],
            }
        }
        with open(output_dir / "ch_test_chunk000_final.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False)

        pipeline = self._make_pipeline(output_dir)
        try:
            pipeline._print_llm_cost_report()
        except Exception as e:
            pytest.fail(f"_print_llm_cost_report raised: {e}")

        report_path = output_dir / "cost_report.md"
        assert report_path.exists(), "cost_report.md 必须生成"
        report_text = report_path.read_text(encoding="utf-8")

        assert "PRE_REWRITER_STAGNATION" in report_text
        assert "翻译质量摘要" in report_text
        assert "⏭️" in report_text

    def test_early_stop_label_mentions_round_74(self, tmp_path):
        """cost_report.md 的 重试效率 表的 "早停命中" 标签应说明含 PRE_REWRITER_STAGNATION."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)

        # 写一个 PRE_REWRITER_STAGNATION final, 让 重试效率 表被渲染
        final_data = {
            "text": "译文",
            "metadata": {
                "fallback": True,
                "reason": "PRE_REWRITER_STAGNATION",
                "quality_retries": 4,
            }
        }
        with open(output_dir / "ch_test_chunk000_final.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False)

        pipeline = self._make_pipeline(output_dir)
        pipeline._print_llm_cost_report()

        report_text = (output_dir / "cost_report.md").read_text(encoding="utf-8")

        # 重试效率表应出现, 且说明文字含 PRE_REWRITER_STAGNATION
        assert "早停命中" in report_text
        assert "PRE_REWRITER_STAGNATION" in report_text

    def test_cost_summary_json_includes_pre_rewriter_status(self, tmp_path):
        """Round 41 cost_summary.json 应在 retry_efficiency.early_stop_count 中算入 PRE_REWRITER_STAGNATION."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)

        # 一个 PRE_REWRITER_STAGNATION + 一个 PASS
        for cid, reason, fallback in [
            ("ch_test_chunk000", "PRE_REWRITER_STAGNATION", True),
            ("ch_test_chunk001", None, False),
        ]:
            md = {"quality_retries": 4}
            if fallback:
                md["fallback"] = True
                md["reason"] = reason
            with open(output_dir / f"{cid}_final.json", "w", encoding="utf-8") as f:
                json.dump({"text": "...", "metadata": md}, f, ensure_ascii=False)

        pipeline = self._make_pipeline(output_dir)
        pipeline._print_llm_cost_report()

        summary_path = output_dir / "cost_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Round 74: PRE_REWRITER_STAGNATION 计入 early_stop_count
        assert summary["retry_efficiency"]["early_stop_count"] == 1
