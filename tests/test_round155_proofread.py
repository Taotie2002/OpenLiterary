"""Round 155: deterministic post-translation proofreading and gate ⑩."""
from __future__ import annotations

import json
from pathlib import Path

from src.book_lint import lint_book
from src.core.scheduler import TaskScheduler
from src.main import _MUTATING_COMMANDS
from src.pipeline import run_final_gate
from src.proofread import _scan_text, proofread_book
from src.rescue import write_qa_report


ROOT = Path(__file__).resolve().parent.parent
CLEAN_BOOK = "# 第一章 起点\n\n他说：“你好。”然后[^1]。\n\n[^1]: 说明。\n"
AUDIT_PASS = (
    "# 转换完整性审计报告 (T1)\n\n"
    "- 全书覆盖率: 99.900%\n"
    "- 结论: ✅ 通过\n"
)


def _write_book(tmp_path: Path, text: str) -> tuple[Path, Path, Path]:
    out = tmp_path / "output"
    out.mkdir(exist_ok=True)
    merged = out / "translated_full.md"
    merged.write_text(text, encoding="utf-8")
    return tmp_path, out, merged


def _mk_gate_project(tmp_path: Path, merged_text: str) -> None:
    (tmp_path / "db").mkdir()
    (tmp_path / "input").mkdir()
    out = tmp_path / "output"
    out.mkdir()
    sched = TaskScheduler(db_path=str(tmp_path / "db" / "workflow.db"))
    sched.conn.execute(
        "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
        "VALUES (?, ?, ?, ?)",
        ("ch01_chunk000", "ch01", "source", "COMPLETED"),
    )
    sched.conn.commit()
    sched.conn.close()
    chapter = out / "ch01"
    chapter.mkdir()
    (chapter / "ch01_chunk000_final.json").write_text(
        json.dumps({"text": CLEAN_BOOK}, ensure_ascii=False), encoding="utf-8")
    (chapter / "ch01_chunk000_critic_report.json").write_text(
        json.dumps({"_backend_path": "jev"}), encoding="utf-8")
    (tmp_path / "input" / "_conversion_audit.md").write_text(
        AUDIT_PASS, encoding="utf-8")
    (out / "translated_full.md").write_text(merged_text, encoding="utf-8")
    write_qa_report(tmp_path, out)


class TestProofreadRules:
    def test_dry_run_reports_without_writing_merged(self, tmp_path):
        text = "# 第一章 起点\n\n正文\u200b。\n\n## 脚注\n\n注。\n"
        root, out, merged = _write_book(tmp_path, text)

        result = proofread_book(project_root=root, output_dir=out,
                                merged_path=merged, apply=False)

        assert result["ok"] is False
        assert merged.read_text(encoding="utf-8") == text
        rules = {issue["rule"] for issue in result["issues"]}
        assert {"invisible_char", "artifact_heading"} <= rules
        assert Path(result["report_path"]).exists()
        assert Path(result["report_path"]).with_suffix(".md").exists()

    def test_apply_only_removes_safe_artifacts(self, tmp_path):
        text = "# 第一章 起点\n\n正文\u200b。\n\n## 译文\n\n下一段（未闭合。\n"
        root, out, merged = _write_book(tmp_path, text)

        result = proofread_book(project_root=root, output_dir=out,
                                merged_path=merged, apply=True)
        repaired = merged.read_text(encoding="utf-8")

        assert "\u200b" not in repaired
        assert "## 译文" not in repaired
        assert "下一段（未闭合。" in repaired
        assert result["ok"] is False
        assert any(issue["rule"] == "unbalanced_bracket"
                   and issue["severity"] == "blocking"
                   for issue in result["issues"])

    def test_bracket_is_blocking_but_cross_paragraph_quote_is_warning(self):
        issues = _scan_text("# 第一章 起点\n\n她说：“你好。\n\n她来过（真的。\n")
        severities = {(issue["rule"], issue["severity"]) for issue in issues}
        assert ("unbalanced_bracket", "blocking") in severities
        assert ("unbalanced_quote", "warning") in severities

    def test_repeated_hard_punctuation_blocks_emphasis_only_warns(self):
        issues = _scan_text("# 第一章 起点\n\n他说，，好。\n\n真的？？\n")
        by_rule = {issue["rule"]: issue["severity"] for issue in issues}
        assert by_rule["duplicate_punctuation"] == "blocking"
        assert by_rule["repeated_emphasis"] == "warning"

    def test_markdown_hard_break_is_not_trailing_whitespace(self):
        hard_break = _scan_text("# 第一章 起点\n\n诗行甲  \n诗行乙。\n")
        assert not any(issue["rule"] == "trailing_whitespace"
                       for issue in hard_break)
        abnormal = _scan_text("# 第一章 起点\n\n诗行甲   \n诗行乙。\n")
        assert any(issue["rule"] == "trailing_whitespace"
                   for issue in abnormal)

    def test_clean_book_passes(self, tmp_path):
        root, out, merged = _write_book(tmp_path, CLEAN_BOOK)
        result = proofread_book(project_root=root, output_dir=out,
                                merged_path=merged)
        assert result["ok"] is True, result["issues"]


class TestProofreadWiring:
    def test_gate_red_and_green_on_proofread_rule(self, tmp_path):
        _mk_gate_project(tmp_path, CLEAN_BOOK)
        assert lint_book(project_root=str(tmp_path))["ok"] is True
        stage_report = tmp_path / "output" / "proofread_report.json"
        stage_report.write_text('{"stage":"keep"}', encoding="utf-8")
        green = run_final_gate(project_root=str(tmp_path))
        assert green["ok"] is True, green["failures"]
        assert green["stats"]["proofread"]["blocking"] == 0
        assert stage_report.read_text(encoding="utf-8") == '{"stage":"keep"}'
        assert (tmp_path / "output" / "proofread_gate_report.json").exists()

        bad = CLEAN_BOOK + "\n## 译文\n\n另一个完整段落。\n"
        (tmp_path / "output" / "translated_full.md").write_text(
            bad, encoding="utf-8")
        red = run_final_gate(project_root=str(tmp_path))
        assert red["ok"] is False
        joined = "\n".join(red["failures"])
        assert "⑩ 机械排错阻断" in joined
        assert "artifact_heading" in joined

    def test_cli_lock_and_shell_order(self):
        assert "proofread" in _MUTATING_COMMANDS
        shell = (ROOT / "scripts" / "translate_book.sh").read_text(encoding="utf-8")
        normalize_at = shell.index("notes-normalize --apply")
        proofread_at = shell.index("translator_agent proofread --apply")
        qa_at = shell.index("translator_agent qa-report")
        assert normalize_at < proofread_at < qa_at

    def test_integrated_snapshot_exposes_proofread(self):
        text = (ROOT / "src" / "translator_agent.py").read_text(encoding="utf-8")
        assert "def proofread_book" in text
        assert '"src.proofread"' in text
