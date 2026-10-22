"""Round 120 (L5): editor_review — 逐章成书级编辑审读（fake LLM, 离线）。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "editor_review", _ROOT / "scripts" / "editor_review.py")
er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(er)


def _seed_project(tmp, chapters=("ch01",)):
    (tmp / "input").mkdir(parents=True, exist_ok=True)
    for ch in chapters:
        d = tmp / "output" / ch
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{ch}_chunk000_final.json").write_text(
            json.dumps({"text": f"{ch} 译文段落。"}, ensure_ascii=False), encoding="utf-8")
        (tmp / "input" / f"{ch}.md").write_text("Source text.", encoding="utf-8")


VALID = json.dumps({"issues": [{
    "severity": "p0", "category": "DANGLING", "quote": "就成了下面这个样子：——",
    "note": "后无内容", "suggestion": "补译缺失段",
}]}, ensure_ascii=False)


class TestPrompt:
    def test_prompt_contains_texts_and_schema(self):
        p = er.build_review_prompt("ch04", "Alice.", "爱丽丝。")
        assert "ch04" in p and "Alice." in p and "爱丽丝。" in p
        assert '"issues"' in p and "P0" in p and "dangling" in p

    def test_prompt_truncates_oversize(self):
        p = er.build_review_prompt("ch04", "A" * 200, "译" * 200, max_chars=50)
        assert "截断" in p


class TestReviewChapter:
    def test_parses_and_normalizes(self):
        r = er.review_chapter("ch04", "src", "tgt", lambda _p: VALID)
        assert r["error"] is None
        it = r["issues"][0]
        assert it["severity"] == "P0" and it["category"] == "dangling"

    def test_non_dict_items_skipped_and_defaults(self):
        raw = json.dumps({"issues": ["x", {"category": "voice"}]}, ensure_ascii=False)
        r = er.review_chapter("ch04", "src", "tgt", lambda _p: raw)
        assert len(r["issues"]) == 1
        assert r["issues"][0]["category"] == "voice"
        assert r["issues"][0]["severity"] == "P2"

    def test_llm_error_recorded(self):
        def boom(_p):
            raise RuntimeError("no key")
        r = er.review_chapter("ch04", "src", "tgt", boom)
        assert r["error"].startswith("llm_error") and r["issues"] == []

    def test_parse_error_recorded(self):
        r = er.review_chapter("ch04", "src", "tgt", lambda _p: "not json at all")
        assert r["error"] == "parse_error"

    def test_empty_translated_text(self):
        r = er.review_chapter("ch04", "src", "", lambda _p: VALID)
        assert r["error"] == "no_final_texts"


class TestMain:
    def test_end_to_end_and_fail_on_p0(self, tmp_path):
        _seed_project(tmp_path, chapters=("ch01", "ch02"))
        gen = lambda _p: VALID  # noqa: E731
        rc = er.main(["--project-root", str(tmp_path), "--md", "output/editor_review.md",
                      "--json", "output/editor_review.json"], generate=gen)
        assert rc == 0
        payload = json.loads(
            (tmp_path / "output/editor_review.json").read_text(encoding="utf-8"))
        assert payload["summary"]["p0"] == 2 and payload["summary"]["chapters"] == 2
        md = (tmp_path / "output/editor_review.md").read_text(encoding="utf-8")
        assert "P0 汇总" in md and "ch01" in md
        rc = er.main(["--project-root", str(tmp_path), "--fail-on-p0",
                      "--md", "output/editor_review2.md",
                      "--json", "output/editor_review2.json"], generate=gen)
        assert rc == 1

    def test_chapters_filter(self, tmp_path):
        _seed_project(tmp_path, chapters=("ch01", "ch02"))
        seen = []
        rc = er.main(["--project-root", str(tmp_path), "--chapters", "ch02"],
                     generate=lambda p: seen.append(p) or json.dumps({"issues": []}))
        assert rc == 0 and len(seen) == 1 and "ch02" in seen[0]
