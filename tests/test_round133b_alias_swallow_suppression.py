"""Round 133b: 别名吞并抑制 — 短别名被其它实体更长别名完全包裹时视为伪命中.

chunk071 根因: entity_registry 把 "Mother" 登记给 杰西卡/杰西卡夫人, 而 chunk071
原文中 "Mother" 唯一出现于 "Reverend Mother"(盖乌斯·海伦·莫希阿姆 的别名) 内部 →
词边界防护(Round 114)放行 → 杰西卡夫人被误判"在场但缺失" → rewriter 收到不可执行
指令(要求把 "Reverend Mother" 那处译为「杰西卡夫人」) → 改坏正文 → NO_PROGRESS_STREAK
→ PERMANENTLY_FAILED.

修复语义:
  - _partition_swallowed_source_names: 别名 a 的每个命中 span 都被某"更长注册别名 b"
    (len(b)>len(a), a 是 b 子串, b 归属某 canonical ≠ 当前实体) 的命中 span 包含 →
    a 剔除; 一处未被吞并即保留.
  - _check_entity_consistency / _build_entity_registry_for_rewriter /
    _build_entity_variants_for_critic / _audit_chapter_entities 四处共享此过滤器.
"""
from __future__ import annotations

import json
import sqlite3


def _make_registry(rows):
    """rows: [(canonical, [source_names...], pending_review)]"""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            confidence TEXT DEFAULT 'low',
            first_seen_chunk TEXT,
            pending_review INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pronoun_gender TEXT,
            entity_type TEXT,
            review_reason TEXT,
            translation_strategy TEXT DEFAULT 'free'
        );
    """)
    for canon, srcs, pending in rows:
        conn.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, confidence, pending_review) "
            "VALUES (?, ?, 'high', ?)",
            (canon, json.dumps(srcs), pending),
        )
    conn.commit()
    return conn


DUNE_ROWS = [
    ("杰西卡夫人", ["Mother", "my mother", "his mother"], 0),
    ("盖乌斯·海伦·莫希阿姆", ["the Reverend Mother", "Reverend Mother", "old woman"], 0),
]

DUNE_SRC = '“Did the Reverend Mother warn you?” Paul blurted.'
DUNE_LIT = '“盖乌斯·海伦·莫希阿姆——那位老妇人——可曾警告过你？”保罗脱口而出。'


def _pipeline(conn):
    from src.translator_agent import TranslationPipeline
    p = TranslationPipeline.__new__(TranslationPipeline)

    class _DE:
        def __init__(self, c):
            self.conn = c

    p.decision_engine = _DE(conn)
    p._sibling_canonicals_cache = None
    p._entity_registry_cache = {}
    p._entity_variants_cache = {}
    return p


class TestAliasSwallowSuppression:
    def test_reverend_mother_no_jessica_missing(self):
        """chunk071 场景: missing 假阳性被吞并抑制清除."""
        p = _pipeline(_make_registry(DUNE_ROWS))
        notes = p._check_entity_consistency("chXX_chunk000", DUNE_LIT, DUNE_SRC)
        missing = [n for n in notes if n.get("type") == "missing"]
        assert missing == [], f"吞并伪命中不应再报 missing, 但 notes={notes}"

    def test_standalone_mother_still_reported(self):
        """真在场("Mother" 独立出现)时 missing 仍须报告 — 防过度抑制."""
        p = _pipeline(_make_registry(DUNE_ROWS))
        notes = p._check_entity_consistency(
            "chXX_chunk000", "有人开口。", "Mother said: be careful, Paul.",
        )
        assert any(n.get("type") == "missing" and n["entity"] == "杰西卡夫人" for n in notes), (
            f"独立出现的 Mother 应报 missing, 但 notes={notes}"
        )

    def test_partial_occurrence_not_swallowed(self):
        """多出现场景: 一处被吞并、一处独立 → 保留（真在场）."""
        p = _pipeline(_make_registry(DUNE_ROWS))
        notes = p._check_entity_consistency(
            "chXX_chunk000", "有人开口。",
            "The Reverend Mother spoke. Mother smiled at Paul.",
        )
        assert any(n.get("type") == "missing" and n["entity"] == "杰西卡夫人" for n in notes), (
            f"存在未吞并出现时应报 missing, 但 notes={notes}"
        )

    def test_registry_excludes_swallowed_alias(self):
        """rewriter registry 不得注入 "Mother"→杰西卡夫人 的伪映射."""
        p = _pipeline(_make_registry(DUNE_ROWS))
        reg = p._build_entity_registry_for_rewriter(DUNE_SRC)
        assert "Mother" not in reg, f"被吞并别名不应注入 registry: {reg}"
        assert reg.get("the Reverend Mother") == "盖乌斯·海伦·莫希阿姆"
        assert all(v != "杰西卡夫人" for v in reg.values()), f"杰西卡夫人 本章无真在场别名: {reg}"

    def test_variants_excludes_swallowed_alias(self):
        """critic variants 表同语义过滤."""
        p = _pipeline(_make_registry(DUNE_ROWS))
        variants = p._build_entity_variants_for_critic(DUNE_SRC)
        assert "Mother" not in variants, f"被吞并别名不应进入 variants: {variants}"

    def test_same_canonical_coverer_does_not_suppress(self):
        """覆盖别名与候选别名同属一个 canonical 时不抑制（防误杀同人长名）."""
        conn = _make_registry([("保罗·阿特雷德斯", ["Paul", "Paul Atreides"], 0)])
        p = _pipeline(conn)
        reg = p._build_entity_registry_for_rewriter("Paul Atreides walked.")
        assert reg.get("Paul") == "保罗·阿特雷德斯", f"同人覆盖不应抑制: {reg}"

    def test_uncovered_registered_alias_keeps_candidate(self):
        """更长别名已注册但原文未出现（不构成覆盖）→ 短别名保留."""
        conn = _make_registry([
            ("杰西卡夫人", ["Mother"], 0),
            ("盖乌斯·海伦·莫希阿姆", ["Reverend Mother"], 0),
        ])
        p = _pipeline(conn)
        reg = p._build_entity_registry_for_rewriter("Mother smiled.")
        assert reg.get("Mother") == "杰西卡夫人", f"未出现的长别名不应造成抑制: {reg}"


class TestSpansPrimitive:
    def test_spans_match_boundary_semantics(self):
        from src.translator_agent import entity_source_name_spans
        assert entity_source_name_spans(
            "Mother", "Reverend Mother and Mother",
        ) == [(9, 15), (20, 26)]
        assert entity_source_name_spans("Mother", "Motherhood") == []
        assert entity_source_name_spans("Mother", "Reverend Mother") == [(9, 15)]
