"""Round 118 P2-8: 主流程接入 Jev 配对仲裁测试.

  - 候选池: 每次 critic 后记录 (avg, lit)，去重、按分排序、截断;
  - 定稿路径: top-2 候选过同一组 primitives，Jev 配对投票选稿;
  - 平票/不可用/异常 -> 回退最高 critic 分版本（原行为不变）。
"""
from __future__ import annotations

import sys

import pytest

from src.config import config as _config
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


class _StubBackend:
    last_backend = "jev"
    timeout = 30.0

    def __init__(self, exc=None):
        self.exc = exc
        self.calls = 0

    def evaluate(self, state, questions, timeout=None):
        self.calls += 1
        if self.exc:
            raise self.exc
        return {"voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.1}, 0.9)}


def _pipeline(cls, candidates, best):
    p = cls.__new__(cls)
    p._best_lit = {"ch01_chunk001": best}
    p._lit_candidates = {"ch01_chunk001": list(candidates)}
    p._pairwise_info = {}
    p.style_guide = {}
    p._build_entity_registry_for_rewriter = lambda src: {}
    return p


def _patch_backend(monkeypatch, backend, winner="B"):
    import src.judges.judge as judge_mod
    import src.judges.primitives as prims_mod
    import src.judges.typesafe_backend as tb
    import src.judges.pairwise as pw

    monkeypatch.setattr(judge_mod, "build_state", lambda **kw: {"lit": kw.get("literary_translation")})
    monkeypatch.setattr(prims_mod, "build_question_set", lambda: {"voice_drift": {}})
    monkeypatch.setattr(tb, "get_backend", lambda: backend)
    monkeypatch.setattr(
        pw, "arbitrate_answers",
        lambda a, b, min_confidence=0.5: {"winner": winner, "votes": {"A": 1, "B": 2}},
    )


class TestConfig:
    def test_pairwise_config(self):
        cfg = _config.pipeline["pairwise_arbitration"]
        assert cfg["enabled"] is True
        assert 0.0 < float(cfg["min_confidence"]) <= 1.0
        assert int(cfg["max_candidates"]) >= 2


class TestCandidatePool:
    def test_pool_dedupes_and_sorts(self):
        from src.pipeline import TranslationPipeline

        p = TranslationPipeline.__new__(TranslationPipeline)
        p._best_lit = {}
        p._lit_candidates = {}
        p._score_history = {}
        p._update_best_lit("c1", "text-A", {"scores": {"fluency": 7.0}})
        p._update_best_lit("c1", "text-B", {"scores": {"fluency": 6.0}})
        p._update_best_lit("c1", "text-A", {"scores": {"fluency": 7.0}})  # 重复
        cands = p._lit_candidates["c1"]
        assert [t for _s, t in cands] == ["text-A", "text-B"]
        assert p._best_lit["c1"][1] == "text-A"


class TestPickBest:
    def test_winner_b_returned(self, monkeypatch):
        from src.pipeline import TranslationPipeline

        backend = _StubBackend()
        _patch_backend(monkeypatch, backend, winner="B")
        p = _pipeline(TranslationPipeline, [(7.0, "text-A"), (6.8, "text-B")], (7.0, "text-A"))
        out = p._pairwise_pick_best("ch01_chunk001", "fallback", source_text="src")
        assert out == "text-B"
        assert p._pairwise_info["ch01_chunk001"]["winner"] == "B"
        assert backend.calls == 2

    def test_winner_a_returned(self, monkeypatch):
        from src.pipeline import TranslationPipeline

        _patch_backend(monkeypatch, _StubBackend(), winner="A")
        p = _pipeline(TranslationPipeline, [(7.0, "text-A"), (6.8, "text-B")], (7.0, "text-A"))
        assert p._pairwise_pick_best("ch01_chunk001", "fallback", source_text="src") == "text-A"

    def test_tie_falls_back_to_best_score(self, monkeypatch):
        from src.pipeline import TranslationPipeline

        _patch_backend(monkeypatch, _StubBackend(), winner="tie")
        p = _pipeline(TranslationPipeline, [(7.0, "text-A"), (6.8, "text-B")], (7.0, "text-A"))
        assert p._pairwise_pick_best("ch01_chunk001", "fallback", source_text="src") == "text-A"

    def test_no_candidates_returns_best(self, monkeypatch):
        from src.pipeline import TranslationPipeline

        backend = _StubBackend()
        _patch_backend(monkeypatch, backend)
        p = _pipeline(TranslationPipeline, [(7.0, "text-A")], (7.0, "text-A"))
        assert p._pairwise_pick_best("ch01_chunk001", "fallback", source_text="src") == "text-A"
        assert backend.calls == 0

    def test_backend_failure_returns_best(self, monkeypatch):
        from src.pipeline import TranslationPipeline

        _patch_backend(monkeypatch, _StubBackend(exc=RuntimeError("jev down")))
        p = _pipeline(TranslationPipeline, [(7.0, "text-A"), (6.8, "text-B")], (7.0, "text-A"))
        assert p._pairwise_pick_best("ch01_chunk001", "fallback", source_text="src") == "text-A"

    def test_disabled_returns_best(self, monkeypatch):
        from src.pipeline import TranslationPipeline

        monkeypatch.setitem(_config.pipeline["pairwise_arbitration"], "enabled", False)
        backend = _StubBackend()
        _patch_backend(monkeypatch, backend)
        p = _pipeline(TranslationPipeline, [(7.0, "text-A"), (6.8, "text-B")], (7.0, "text-A"))
        assert p._pairwise_pick_best("ch01_chunk001", "fallback", source_text="src") == "text-A"
        assert backend.calls == 0

    def test_monolith_mirror(self, monkeypatch):
        import src.translator_agent as ta

        _patch_backend(monkeypatch, _StubBackend(), winner="B")
        p = _pipeline(ta.TranslationPipeline, [(7.0, "text-A"), (6.8, "text-B")], (7.0, "text-A"))
        assert p._pairwise_pick_best("ch01_chunk001", "fallback", source_text="src") == "text-B"
