"""Round 120 L3: 双稿仲裁 — verse / pun-dense 片段生成端多样性择优。

verse / pun_watchlist 命中的高价值片段，除首稿（literary_rewrite → DeepSeek）外，
用 alternate_role（MiniMax-M3）再生成一份文学稿，按 pairwise（Jev 配对）→
critic 均分 → 首稿 三级择优；结果写 <chunk>_dual_draft_meta.json sidecar。

默认关闭（config.pipeline.dual_draft.enabled=false），Phase 1 小场景 A/B 后定夺。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
_DRAFT_A = "首稿：爱丽丝顺着兔子洞往下掉。"
_DRAFT_B = "第二稿：另一版润色文本。"


@pytest.fixture
def dual_cfg():
    import src.config
    old = src.config.config.pipeline.get("dual_draft")
    src.config.config.pipeline["dual_draft"] = {
        "enabled": True, "alternate_role": "primary", "max_chunks": 25,
    }
    yield
    if old is not None:
        src.config.config.pipeline["dual_draft"] = old
    else:
        src.config.config.pipeline.pop("dual_draft", None)


@pytest.fixture
def dual_disabled_cfg():
    import src.config
    old = src.config.config.pipeline.get("dual_draft")
    src.config.config.pipeline["dual_draft"] = {"enabled": False}
    yield
    if old is not None:
        src.config.config.pipeline["dual_draft"] = old
    else:
        src.config.config.pipeline.pop("dual_draft", None)


def _make_pipeline(tmp_path, *, b_text=_DRAFT_B, style_guide="",
                   verse_fragments=None, rewriter_side_effect=None, stub_pair=True):
    from src.translator_agent import TranslationPipeline

    p = TranslationPipeline.__new__(TranslationPipeline)
    p.output_dir = tmp_path
    p.chapter_id = "ch_test"
    p.style_guide = style_guide
    p._dual_draft_count = 0
    p.rewriter_agent = MagicMock()
    if rewriter_side_effect is not None:
        p.rewriter_agent.process_chunk = MagicMock(side_effect=rewriter_side_effect)
    else:
        p.rewriter_agent.process_chunk = MagicMock(return_value=(b_text, None))
    p.critic_agent = MagicMock()

    store = {}
    p._load_intermediate = lambda cid, step: store[(cid, step)]
    p._save_intermediate = lambda cid, step, data: store.__setitem__((cid, step), data)
    p._verse_fragments_for = lambda cid, src: list(verse_fragments or [])
    p._verse_note_for = lambda cid, src: ""
    p._build_entity_registry_for_rewriter = lambda src: {}
    p._compute_avg_score = lambda report: (report or {}).get("avg")
    if stub_pair:
        p._dual_pairwise_winner = MagicMock(return_value=(None, {"winner": None}))
        p._dual_critic_avg = MagicMock(return_value=None)
    return p, store


def _run(p, draft_a=_DRAFT_A, source="Alice went down the rabbit hole.",
         raw="爱丽丝顺着兔子洞往下走。"):
    return p._dual_draft_run(
        "ch_test_chunk000", source, raw, None, None, None, None,
        [], [], None, [], [], draft_a,
    )


def _meta(store, cid="ch_test_chunk000"):
    return store[(cid, "dual_draft_meta")]


class TestGating:
    def test_disabled_returns_draft_a(self, tmp_path, dual_disabled_cfg):
        p, store = _make_pipeline(tmp_path)
        assert _run(p) == _DRAFT_A
        p.rewriter_agent.process_chunk.assert_not_called()
        assert ("ch_test_chunk000", "dual_draft_meta") not in store

    def test_plain_chunk_not_routed(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path)
        assert _run(p) == _DRAFT_A
        p.rewriter_agent.process_chunk.assert_not_called()

    def test_max_chunks_cap(self, tmp_path, dual_cfg):
        p, _ = _make_pipeline(tmp_path, verse_fragments=["line1"])
        p._dual_draft_count = 25
        assert _run(p) == _DRAFT_A
        p.rewriter_agent.process_chunk.assert_not_called()

    def test_existing_meta_skips(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, verse_fragments=["line1"])
        store[("ch_test_chunk000", "dual_draft_meta")] = {"chosen": "b"}
        assert _run(p) == _DRAFT_A
        p.rewriter_agent.process_chunk.assert_not_called()


class TestRoutingAndSelection:
    def test_verse_route_and_fallback_keeps_a(self, tmp_path, dual_cfg):
        import src.config as _c
        p, store = _make_pipeline(tmp_path, verse_fragments=["l1"])
        assert _run(p) == _DRAFT_A
        meta = _meta(store)
        assert meta["kind"] == "verse" and meta["chosen"] == "a"
        assert meta["reason"] == "fallback"
        assert meta["roles"]["b"] == "primary"
        assert meta["roles"]["a"] == _c.config.get_task_role("literary_rewrite")
        kwargs = p.rewriter_agent.process_chunk.call_args.kwargs
        assert kwargs["model_role"] == "primary" and kwargs["retry_count"] == 0

    def test_verse_takes_priority_over_pun(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(
            tmp_path, verse_fragments=["l1"],
            style_guide={"pun_watchlist": [{"anchors": ["Caucus-race"]}]})
        _run(p, source="The Caucus-race went down the rabbit hole.")
        assert _meta(store)["kind"] == "verse"

    def test_pun_kind_from_watchlist_anchor(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(
            tmp_path, style_guide={"pun_watchlist": [{"anchors": ["Caucus-race"]}]})
        _run(p, source="They ran the Caucus-race in a circle.")
        meta = _meta(store)
        assert meta["kind"] == "pun" and meta["chosen"] == "a"

    def test_pun_kind_from_src_anchors(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(
            tmp_path,
            style_guide={"pun_watchlist": [
                {"source": "whiting (fish vs. shoe-blacking)",
                 "src_anchors": ["whiting"], "anchors": ["牙鳕", "白垩粉"]}]})
        _run(p, source="‘The whiting,’ said the Gryphon.")
        assert _meta(store)["kind"] == "pun"

    def test_pairwise_winner_b_returns_draft_b(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, verse_fragments=["l1"])
        p._dual_pairwise_winner = MagicMock(
            return_value=("b", {"winner": "B", "backend": "stub", "votes": {"prim1": "B"}}))
        p._dual_critic_avg = MagicMock(return_value=None)
        assert _run(p) == _DRAFT_B
        meta = _meta(store)
        assert meta["chosen"] == "b" and meta["reason"] == "pairwise"
        assert meta["pairwise"]["winner"] == "B"

    def test_pairwise_winner_a_keeps_draft_a(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, verse_fragments=["l1"])
        p._dual_pairwise_winner = MagicMock(
            return_value=("a", {"winner": "A", "backend": "stub", "votes": {}}))
        p._dual_critic_avg = MagicMock(return_value=0.9)
        assert _run(p) == _DRAFT_A
        meta = _meta(store)
        assert meta["chosen"] == "a" and meta["reason"] == "pairwise"

    def test_pairwise_unavailable_critic_picks_b(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, verse_fragments=["l1"])
        p._dual_critic_avg = MagicMock(side_effect=[0.4, 0.83])
        assert _run(p) == _DRAFT_B
        meta = _meta(store)
        assert meta["chosen"] == "b" and meta["reason"] == "critic"
        assert meta["critic_avg"] == {"a": 0.4, "b": 0.83}

    def test_critic_tie_keeps_draft_a(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, verse_fragments=["l1"])
        p._dual_critic_avg = MagicMock(side_effect=[0.7, 0.7])
        assert _run(p) == _DRAFT_A
        meta = _meta(store)
        assert meta["chosen"] == "a" and meta["reason"] == "critic"

    def test_count_increments_once(self, tmp_path, dual_cfg):
        p, _ = _make_pipeline(tmp_path, verse_fragments=["l1"])
        _run(p)
        assert p._dual_draft_count == 1

    def test_pairwise_wiring_winner_b(self, tmp_path, dual_cfg, monkeypatch):
        p, _ = _make_pipeline(tmp_path, stub_pair=False)
        fake_backend = MagicMock()
        fake_backend.timeout = 5.0
        fake_backend.last_backend = "stub-backend"
        fake_backend.evaluate = MagicMock(
            return_value={"prim1": {"answer": "yes", "confidence": 0.9}})
        monkeypatch.setattr("src.judges.typesafe_backend.get_backend", lambda: fake_backend)
        monkeypatch.setattr(
            "src.judges.pairwise.arbitrate_answers",
            lambda a, b, min_confidence=0.6: {"winner": "B", "votes": {"prim1": "B"}})
        winner, info = p._dual_pairwise_winner("c1", "Alice.", "爱丽丝。", "稿A", "稿B")
        assert winner == "b"
        assert info == {"winner": "B", "backend": "stub-backend", "votes": {"prim1": "B"}}


class TestFallbacks:
    def test_typeerror_signature_mismatch_keeps_draft_a(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, verse_fragments=["l1"],
                                  rewriter_side_effect=TypeError("no model_role kwarg"))
        assert _run(p) == _DRAFT_A
        assert ("ch_test_chunk000", "dual_draft_meta") not in store

    def test_generic_exception_keeps_draft_a(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, verse_fragments=["l1"],
                                  rewriter_side_effect=RuntimeError("boom"))
        assert _run(p) == _DRAFT_A
        assert ("ch_test_chunk000", "dual_draft_meta") not in store

    def test_identical_second_draft_keeps_draft_a(self, tmp_path, dual_cfg):
        p, store = _make_pipeline(tmp_path, b_text=_DRAFT_A, verse_fragments=["l1"])
        assert _run(p) == _DRAFT_A
        assert ("ch_test_chunk000", "dual_draft_meta") not in store


class TestHookWiring:
    def test_hook_present_in_both_copies(self):
        needle = "lit_text = self._dual_draft_run("
        for path in ("src/pipeline.py", "src/translator_agent.py"):
            src = (ROOT / path).read_text(encoding="utf-8")
            assert src.count(needle) == 1, f"{path} 缺少 dual-draft 钩子调用"
            idx = src.index(needle)
            window = src[max(0, idx - 260):idx]
            assert 'self._save_intermediate(chunk_id, "literary", lit_text)' in window
            assert "quality_retries == 0 and lit_text" in window
