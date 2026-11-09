"""Round 142 Phase D: optional cross-chapter retrieval for editor_review."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load_editor_review():
    path = _ROOT / "scripts" / "editor_review.py"
    spec = importlib.util.spec_from_file_location("round143_editor_review", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeIndex:
    def __init__(self, error: bool = False):
        self.error = error
        self.queries = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        if self.error:
            raise RuntimeError("graph offline")
        return {"results": [
            {"chapter": "ch01", "scene_id": "scene:ch01:0001", "score": 0.9,
             "anchor": {"quote": "same chapter must be filtered"}},
            {"chapter": "ch02", "scene_id": "scene:ch02:0001", "score": 0.8,
             "anchor": {"quote": "later chapter evidence one"}},
            {"chapter": "ch03", "scene_id": "scene:ch03:0001", "score": 0.7,
             "anchor": {"quote": "later chapter evidence two"}},
            {"chapter": "ch04", "scene_id": "scene:ch04:0001", "score": 0.6,
             "anchor": {"quote": "later chapter evidence three"}},
        ]}


def test_retrieval_context_filters_current_chapter_and_bounds_budget():
    er = _load_editor_review()
    ctx = er.build_retrieval_context("query", FakeIndex(), "ch01", top_k=2,
                                     max_chars=600)
    assert "same chapter" not in ctx
    assert "ch02" in ctx and "ch03" in ctx
    assert "ch04" not in ctx
    assert len(ctx) <= 600


def test_retrieval_context_is_fail_open():
    er = _load_editor_review()
    assert er.build_retrieval_context("query", FakeIndex(error=True), "ch01") == ""


def test_review_prompt_injects_retrieval_block_only_when_present():
    er = _load_editor_review()
    base = er.build_review_prompt("ch01", "src", "tgt", global_mode=True)
    with_ctx = er.build_review_prompt(
        "ch01", "src", "tgt", global_mode=True,
        retrieval_context="【跨章检索证据】\n- ch02 quote")
    assert "跨章检索证据" not in base
    assert "跨章检索证据" in with_ctx
    assert "ch02 quote" in with_ctx


def test_review_chapter_uses_retrieval_and_records_meta():
    er = _load_editor_review()
    prompts = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return '{"issues": []}'

    result = er.review_chapter(
        "ch01", "source text", "translated text", generate,
        global_mode=True, retrieval_index=FakeIndex(), retrieval_top_k=2,
        retrieval_max_chars=600)
    assert result["issues"] == []
    assert result["retrieval"][0]["injected"] is True
    assert result["retrieval"][0]["chars"] > 0
    assert "跨章检索证据" in prompts[0]
    assert "same chapter" not in prompts[0]


def test_retrieval_requires_global_mode():
    er = _load_editor_review()
    assert er.main(["--project-root", str(_ROOT), "--retrieval",
                    "--chapters", "ch01"], generate=lambda _p: '{"issues":[]}') == 2
