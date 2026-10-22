"""Round 105 P2-1: missing 核查的括注剥离 + 前缀/后缀容差。

ch32 倒查根因：`_check_entity_consistency` 的 missing 分支要求规范名整串出现，
导致两类系统性假阳性 → 实体硬失败 → rewriter 互斥指令死循环：
  1. 规范名带括注（镇（town）/科德斯（Codders）/卡特拉姆勋爵（第九代…））
     — 译文永远不会含括注，必然 missing。
  2. 前缀/后缀写法（译文 乔治·洛马克斯 vs 规范 乔治·洛马克斯先生）。
"""
from __future__ import annotations

import json
import sqlite3

import pytest


def _make_pipeline(conn):
    from src.translator_agent import TranslationPipeline
    p = TranslationPipeline.__new__(TranslationPipeline)

    class _DE:
        def __init__(self):
            self.conn = conn
    p.decision_engine = _DE()
    from src.core.decision_engine import get_sibling_canonicals_by_source
    p._sibling_canonicals_cache = get_sibling_canonicals_by_source(conn)
    return p


@pytest.fixture
def registry():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            confidence TEXT DEFAULT 'high',
            pending_review INTEGER DEFAULT 0
        );
    """)
    rows = [
        ("乔治·洛马克斯先生", ["Mr. Lomax", "Mr. George Lomax", "George", "sir"]),
        ("镇（town）", ["town"]),
        ("科德斯（Codders）", ["Codders"]),
        ("黛西", ["Daisy"]),
    ]
    for canon, srcs in rows:
        conn.execute(
            "INSERT INTO entity_registry (canonical_name, source_names) VALUES (?, ?)",
            (canon, json.dumps(srcs)),
        )
    conn.commit()
    yield conn
    conn.close()


class TestStripParenthetical:
    def test_fullwidth(self):
        from src.translator_agent import TranslationPipeline as TP
        assert TP._strip_parenthetical("镇（town）") == "镇"
        assert TP._strip_parenthetical("科德斯（Codders）") == "科德斯"
        assert TP._strip_parenthetical("卡特拉姆勋爵（第九代卡特拉姆侯爵）") == "卡特拉姆勋爵"

    def test_halfwidth(self):
        from src.translator_agent import TranslationPipeline as TP
        assert TP._strip_parenthetical("London (UK)") == "London"

    def test_no_paren(self):
        from src.translator_agent import TranslationPipeline as TP
        assert TP._strip_parenthetical("乔治·洛马克斯先生") == "乔治·洛马克斯先生"


class TestCanonicalPresentInText:
    def setup_method(self):
        from src.translator_agent import TranslationPipeline as TP
        self.p = TP.__new__(TP)

    def test_prefix_match(self):
        assert self.p._canonical_present_in_text("乔治·洛马克斯先生", "乔治·洛马克斯摆出神气。") is True

    def test_suffix_match(self):
        assert self.p._canonical_present_in_text("乔治·洛马克斯先生", "洛马克斯先生抵达。") is True

    def test_absent(self):
        assert self.p._canonical_present_in_text("乔治·洛马克斯先生", "某人抵达庄园。") is False

    def test_parenthetical_single_char_base_suppressed(self):
        # 镇（town）→ base 镇（单字）→ 抑制 missing
        assert self.p._canonical_present_in_text("镇（town）", "小镇上很安静。") is True

    def test_parenthetical_base_match(self):
        assert self.p._canonical_present_in_text("科德斯（Codders）", "科德斯庄园。") is True
        assert self.p._canonical_present_in_text("科德斯（Codders）", "某人来了。") is False

    def test_short_name_no_tolerance(self):
        # 2 字名不做前缀容差（避免 黛 这种单字前缀误判）
        assert self.p._canonical_present_in_text("黛西", "黛西在花园。") is True
        assert self.p._canonical_present_in_text("黛西", "花园里。") is False


class TestMissingCheckIntegration:
    def test_prefix_form_not_missing(self, registry):
        """译文用前缀形式（乔治·洛马克斯）不应报 missing。"""
        p = _make_pipeline(registry)
        source_text = "George Lomax arrived at the manor."
        lit_text = "乔治·洛马克斯抵达庄园。"
        notes = p._check_entity_consistency("chXX_chunk000", lit_text, source_text)
        missing = [n for n in notes if n.get("type") == "missing"]
        assert missing == [], f"前缀形式不应报 missing: {missing}"

    def test_parenthetical_canonical_not_missing(self, registry):
        """带括注规范名（科德斯（Codders））译文用 base（科德斯）不应报 missing。"""
        p = _make_pipeline(registry)
        source_text = "Codders is a village."
        lit_text = "科德斯是个村庄。"
        notes = p._check_entity_consistency("chXX_chunk000", lit_text, source_text)
        missing = [n for n in notes if n.get("type") == "missing"]
        assert missing == [], f"括注规范名 base 命中不应报 missing: {missing}"

    def test_genuinely_absent_still_missing(self, registry):
        """完全未出现仍应报 missing（保留原能力）。"""
        p = _make_pipeline(registry)
        source_text = "Daisy was in the garden."
        lit_text = "花园里很安静。"
        notes = p._check_entity_consistency("chXX_chunk000", lit_text, source_text)
        missing = [n for n in notes if n.get("type") == "missing"]
        assert any(n["entity"] == "黛西" for n in missing), f"真 missing 应报告: {notes}"
