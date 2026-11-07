"""Round 141: 中文成品文件名 + Ctrl+C 两段式防误触。"""
from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest

from src.book_lint import (
    book_title_zh,
    lint_book,
    notes_normalize_outputs,
    resolve_book_output_path,
)


def _write_final(root: Path, text: str) -> None:
    out = root / "output" / "ch01"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ch01_chunk000_final.json").write_text(
        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")


def test_config_title_zh_wins_and_filename_is_sanitized(tmp_path):
    (tmp_path / "config.yaml").write_text(
        'book:\n  title_zh: "沙丘 / 第二部: 终章?"\n', encoding="utf-8")
    path = resolve_book_output_path(tmp_path, tmp_path / "output", prefer_existing=False)
    assert path == tmp_path / "output" / "沙丘 第二部 终章.md"


def test_profile_title_zh_is_preferred(tmp_path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "book_profile.json").write_text(
        json.dumps({"work_guess": {"title": "Dune", "title_zh": "沙丘"}},
                   ensure_ascii=False), encoding="utf-8")
    assert book_title_zh(tmp_path) == "沙丘"


def test_title_inferred_from_first_translated_h1(tmp_path):
    _write_final(tmp_path, "# 第一部\n# 沙丘\n\n正文。\n")
    path = resolve_book_output_path(tmp_path, tmp_path / "output", prefer_existing=False)
    assert path.name == "沙丘.md"


def test_generic_chapter_heading_is_not_used_as_book_title(tmp_path):
    _write_final(tmp_path, "# 第一章 起点\n\n正文。\n")
    path = resolve_book_output_path(tmp_path, tmp_path / "output", prefer_existing=False)
    assert path.name == "translated_full.md"


def test_legacy_file_remains_readable_until_chinese_file_exists(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    legacy = out / "translated_full.md"
    legacy.write_text("# 第一章 起点\n\n正文。\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text('book:\n  title_zh: "沙丘"\n', encoding="utf-8")
    assert resolve_book_output_path(tmp_path, out) == legacy


def test_notes_and_lint_use_chinese_output_path(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (tmp_path / "config.yaml").write_text('book:\n  title_zh: "沙丘"\n', encoding="utf-8")
    merged = "# 第一部（Book 1）\n\n正文。\n"
    (out / "沙丘.md").write_text(merged, encoding="utf-8")

    res = notes_normalize_outputs(project_root=tmp_path, apply=False)
    assert res["stats"]["files"] == 1
    lint_res = lint_book(project_root=tmp_path)
    assert lint_res["ok"] is True, lint_res["failures"]


def test_style_profile_normalize_preserves_title_zh():
    from src.style_profiler import normalize_profile
    prof = normalize_profile({"work_guess": {"title_zh": " 沙丘 "}})
    assert prof["work_guess"]["title_zh"] == "沙丘"


def test_sigint_guard_warns_once_then_aborts(monkeypatch, capsys):
    import src.main as main_mod

    old_handler = signal.getsignal(signal.SIGINT)
    try:
        handler = main_mod._install_sigint_guard()
        assert handler is not None
        handler(signal.SIGINT, None)
        errtxt = capsys.readouterr().err
        assert "检测到 Ctrl+C" in errtxt
        assert "断点续跑" in errtxt
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGINT, old_handler)


def test_translate_book_shell_has_sigint_guard_and_chinese_output_resolver():
    script = Path(__file__).resolve().parent.parent / "scripts" / "translate_book.sh"
    text = script.read_text(encoding="utf-8")
    assert "trap 'handle_sigint' INT" in text
    assert "PY_CHILD_ACTIVE" in text
    assert "Ctrl+C x2" in text
    assert "resolve_book_output_path" in text
    assert "run_cmd python3 -m src.translator_agent pipeline" in text
    check = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_shell_sigint_handler_warns_then_exits_130(tmp_path):
    script = Path(__file__).resolve().parent.parent / "scripts" / "translate_book.sh"
    text = script.read_text(encoding="utf-8")
    prefix = text.rsplit('\nmain "$@"', 1)[0]
    harness = tmp_path / "guard_harness.sh"
    harness.write_text(prefix, encoding="utf-8")
    cmd = (
        'source "$1"; '
        'handle_sigint; echo FIRST_OK; '
        'handle_sigint; echo SHOULD_NOT_REACH'
    )
    res = subprocess.run(["bash", "-c", cmd, "bash", str(harness)],
                         capture_output=True, text=True)
    assert res.returncode == 130
    assert "FIRST_OK" in res.stdout
    assert "SHOULD_NOT_REACH" not in res.stdout
    assert "Ctrl+C x2" in res.stderr
