"""Round 118 P2-5: 直译难度路由测试.

  - 难度启发式: 典故密度 + 长句占比 + 引文特征;
  - difficulty >= hard_threshold -> primary（MiniMax）+ thinking 开;
  - 普通片段 -> literal_translator（DeepSeek）+ thinking 关（配置原样）.
"""
from __future__ import annotations

import sys

import pytest

from src.config import config as _config


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


class _StubLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return "译"


class _StubCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows


class _StubDecisionEngine:
    def __init__(self, refs):
        self.conn = type("Conn", (), {"cursor": lambda self2: _StubCursor(refs)})()

    def format_prompt_context(self, source_text="", **kwargs):
        return ""


def _make_pipeline(cls, refs=()):
    p = cls.__new__(cls)
    p.literal_role = _config.get_task_role("literal_translation")
    p.llm = _StubLLM()
    p.decision_engine = _StubDecisionEngine([(r,) for r in refs])
    p.style_guide = {}
    return p


HARD_TEXT = (
    "The wild-cat venture and the morganatic alliance were both discussed at length, "
    "and everyone agreed that the matter was far more complicated than it appeared at first sight. "
    "He quoted from a famous poem: \"The year's at the spring and day's at the morn, morning's at seven, "
    "the hill-side's dew-pearled, the lark's on the wing, the snail's on the thorn.\" "
    "It was, in short, a very long and winding sentence that went on and on without pause or relief, "
    "which made the translator's task considerably harder than usual for everyone involved."
)

EASY_TEXT = "He said hello. She smiled. They went home."


class TestConfig:
    def test_difficulty_routing_config(self):
        cfg = _config.pipeline["difficulty_routing"]
        assert cfg["enabled"] is True
        assert cfg["hard_role"] == "primary"
        assert cfg["hard_enable_thinking"] is True
        assert float(cfg["hard_threshold"]) <= 2.0


class TestDifficultyScore:
    def test_reference_hits_add_score(self):
        from src.pipeline import TranslationPipeline

        p = _make_pipeline(TranslationPipeline, refs=["wild-cat", "morganatic"])
        assert p._count_reference_hits(HARD_TEXT) == 2
        assert p._chunk_difficulty(HARD_TEXT) >= 2.0

    def test_long_sentences_and_quotes(self):
        from src.pipeline import TranslationPipeline

        p = _make_pipeline(TranslationPipeline, refs=[])
        score = p._chunk_difficulty(HARD_TEXT)
        assert score >= 1.0

    def test_easy_text_low_score(self):
        from src.pipeline import TranslationPipeline

        p = _make_pipeline(TranslationPipeline, refs=[])
        assert p._chunk_difficulty(EASY_TEXT) < 1.0


class TestRouting:
    def test_hard_chunk_uses_primary_and_thinking(self, monkeypatch):
        import src.utils.llm_adapter as la
        from src.pipeline import TranslationPipeline

        hard_client = _StubLLM()
        monkeypatch.setattr(la, "get_llm_client", lambda role: hard_client)
        p = _make_pipeline(TranslationPipeline, refs=["wild-cat", "morganatic"])
        p._run_raw_translator(HARD_TEXT)
        assert len(hard_client.calls) == 1
        assert p.llm.calls == []
        call = hard_client.calls[0]
        assert call["enable_thinking"] is True
        assert call["model_name"]  # primary 角色模型（MiniMax-M3）

    def test_easy_chunk_uses_literal_and_no_thinking(self, monkeypatch):
        import src.utils.llm_adapter as la
        from src.pipeline import TranslationPipeline

        hard_client = _StubLLM()
        monkeypatch.setattr(la, "get_llm_client", lambda role: hard_client)
        p = _make_pipeline(TranslationPipeline, refs=[])
        p._run_raw_translator(EASY_TEXT)
        assert len(p.llm.calls) == 1
        assert hard_client.calls == []
        assert p.llm.calls[0].get("enable_thinking") is False

    def test_monolith_mirror(self, monkeypatch):
        import src.utils.llm_adapter as la
        import src.translator_agent as ta

        hard_client = _StubLLM()
        monkeypatch.setattr(la, "get_llm_client", lambda role: hard_client)
        p = _make_pipeline(ta.TranslationPipeline, refs=["wild-cat", "morganatic"])
        p._run_raw_translator(HARD_TEXT)
        assert len(hard_client.calls) == 1
        assert hard_client.calls[0]["enable_thinking"] is True
