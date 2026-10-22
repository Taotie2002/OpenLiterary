"""Round 118 P2-6: 章节漂移审计测试.

  - 统计层: 指标计算 / 窗口趋势 / profile 基线偏差（离线确定性）;
  - Jev 层: 相邻窗口配对检查（backend 可 mock; 无 key 时安全降级）;
  - 报告渲染与 CLI 入口参数。
"""
from __future__ import annotations

import json
import sys

import pytest

from src.config import config as _config
from src.drift_audit import (
    audit_chapter,
    chunk_style_metrics,
    collect_chunk_texts,
    jev_window_check,
    summarize_report,
)
from src.judges.typesafe_spec import ChoiceAnswer


@pytest.fixture(autouse=True)
def _isolate_modules():
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)


def _write_finals(root, chapter, texts):
    out = root / "output" / chapter
    out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        payload = {"text": text, "metadata": {}}
        (out / f"{chapter}_chunk{i:03d}_final.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
    # 干扰文件: 畸形 JSON 应被跳过
    (out / f"{chapter}_chunk999_final.json").write_text("{broken", encoding="utf-8")


SHORT = "他说。她笑。走了。"
LONG = (
    "他缓缓地说道，那是一个漫长而复杂的夜晚，风从山谷深处吹来，带着潮湿的草木气息；"
    "而她只是静静地站在那里，望着远处层层叠叠的灯火，心里想着那些再也回不去的日子，"
    "以及那些曾经以为永远不会改变的事情。"
)


class TestConfig:
    def test_drift_audit_config(self):
        cfg = _config.pipeline["drift_audit"]
        assert int(cfg["window"]) >= 1
        assert 0.0 < float(cfg["trend_threshold"]) < 1.0
        assert 0.0 < float(cfg["sentence_len_tolerance"]) < 1.0


class TestMetrics:
    def test_metrics_computed(self):
        m = chunk_style_metrics(LONG)
        assert m["avg_sentence_len"] > 20
        assert m["cjk_ratio"] > 0.9
        m2 = chunk_style_metrics(SHORT)
        assert m2["avg_sentence_len"] < m["avg_sentence_len"]

    def test_collect_sorted_and_skips_broken(self, tmp_path):
        _write_finals(tmp_path, "ch01", [SHORT, LONG])
        chunks = collect_chunk_texts(tmp_path / "output" / "ch01")
        assert [cid for cid, _t in chunks] == ["ch01_chunk000", "ch01_chunk001"]


class TestAudit:
    def test_window_trend_detected(self, tmp_path):
        _write_finals(tmp_path, "ch01", [SHORT] * 5 + [LONG] * 5)
        report = audit_chapter(tmp_path, chapter="ch01", window=5)
        assert report["chunk_count"] == 10
        metrics = {f["metric"] for f in report["drift_flags"]}
        assert "avg_sentence_len" in metrics
        text = summarize_report(report)
        assert "漂移警告" in text

    def test_no_drift_for_stable_text(self, tmp_path):
        _write_finals(tmp_path, "ch01", [LONG] * 10)
        report = audit_chapter(tmp_path, chapter="ch01", window=5)
        trend_flags = [f for f in report["drift_flags"] if f["type"] == "window_trend"]
        assert trend_flags == []

    def test_profile_deviation_flagged(self, tmp_path):
        _write_finals(tmp_path, "ch01", [SHORT] * 4)
        guide = {
            "avg_sentence_length": 200.0,
            "register_default": "庄重书面",
            "_profile_source": "test",
        }
        report = audit_chapter(tmp_path, chapter="ch01", window=5, profile_guide=guide)
        types = {f["type"] for f in report["drift_flags"]}
        assert "profile_deviation" in types

    def test_missing_output_dir_is_safe(self, tmp_path):
        report = audit_chapter(tmp_path, chapter="ch99", window=5)
        assert report["chunk_count"] == 0
        assert report["drift_flags"] == []


class TestJevLayer:
    def test_unavailable_without_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        chunks = [("ch01_chunk000", SHORT), ("ch01_chunk001", LONG)]
        out = jev_window_check(chunks, window=1)
        assert out["available"] is False

    def test_drifted_pairs_flagged_with_stub_backend(self, monkeypatch):
        import src.judges.typesafe_backend as tb

        class _Stub:
            last_backend = "jev"

            class primary:  # noqa: N801
                api_key = "stub"

            def evaluate(self, state, questions, **kw):
                return {
                    "voice_drift": ChoiceAnswer(
                        "severe", {"none": 0.1, "subtle": 0.2, "severe": 0.7}, 0.8
                    ),
                    "register_match": ChoiceAnswer(
                        "over", {"match": 0.5, "under": 0.1, "over": 0.4}, 0.8
                    ),
                }

        monkeypatch.setattr(tb, "get_backend", lambda: _Stub())
        chunks = [(f"ch01_chunk{i:03d}", SHORT if i % 2 == 0 else LONG) for i in range(4)]
        out = jev_window_check(chunks, window=2)
        assert out["available"] is True
        assert out["drifted_pairs"]

    def test_clean_windows_not_flagged(self, monkeypatch):
        import src.judges.typesafe_backend as tb

        class _Stub:
            last_backend = "jev"

            class primary:  # noqa: N801
                api_key = "stub"

            def evaluate(self, state, questions, **kw):
                return {
                    "voice_drift": ChoiceAnswer(
                        "none", {"none": 0.8, "subtle": 0.15, "severe": 0.05}, 0.8
                    ),
                    "register_match": ChoiceAnswer(
                        "match", {"match": 0.8, "under": 0.1, "over": 0.1}, 0.8
                    ),
                }

        monkeypatch.setattr(tb, "get_backend", lambda: _Stub())
        chunks = [(f"ch01_chunk{i:03d}", LONG) for i in range(4)]
        out = jev_window_check(chunks, window=2)
        assert out["available"] is True
        assert out["drifted_pairs"] == []
