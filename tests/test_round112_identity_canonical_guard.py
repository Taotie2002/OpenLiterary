"""Round 112: 身份映射 canonical（未翻译英文主名）不进入实体核查（ch10 chunk007 根因）。

ch10 chunk007: 登记表 canonical=" Cloak Room"（ASCII + 前导空格），source_name
"Cloak Room" 与其规范化后恒等。核查要求中文译文出现英文串「 Cloak Room」，
而正常译法是「行李寄存票/衣帽间」→ 不可满足 → 实体硬失败退回重写 → rewriter
字节停滞（retry=2 仍与上一版相同）→ REWRITER_STAGNATED → PERMANENTLY_FAILED。

修复（Round 112）：canonical 无 CJK 且与某个 source_name 规范化（折叠空白+小写）
后相等 → 身份映射，不含翻译信息，不进入 _check_entity_consistency /
_build_entity_registry_for_rewriter / _build_entity_variants_for_critic / 实体审计。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.pipeline import TranslationPipeline as P


# ---------------------------------------------------------------------------
# 谓词：_is_enforceable_canonical
# ---------------------------------------------------------------------------

class TestEnforceableCanonical:
    @pytest.mark.parametrize("canonical,source_names", [
        (" Cloak Room", ["Cloak Room"]),          # ch10 chunk007 根因（前导空格）
        ("Cloak Room", ["Cloak Room"]),           # 去空格后仍恒等
        ("Richmond", ["Richmond"]),
        ("South America", ["South America", "those little republics"]),
        ("Father", ["poor old Father", "Father"]),
        ("H. I.", ["H. I.", "the initials H. I."]),
        ("Morris Cowley", ["Morris Cowley"]),
    ])
    def test_identity_mapping_not_enforceable(self, canonical, source_names):
        assert P._is_enforceable_canonical(canonical, source_names) is False

    @pytest.mark.parametrize("canonical,source_names", [
        ("帕丁顿行李寄存处", ["Left Luggage Office"]),
        ("布伦特福德", ["Brentford"]),
        ("豪恩斯洛", ["Hounslow"]),
        ("戈尔兹路", ["Goldhawk Road"]),
        ("Sûreté", ["the Sûreté"]),               # 英文但与 source_name 不同 → 可核查
    ])
    def test_translated_or_distinct_canonical_enforceable(self, canonical, source_names):
        assert P._is_enforceable_canonical(canonical, source_names) is True

    def test_empty_canonical_not_enforceable(self):
        assert P._is_enforceable_canonical("", ["x"]) is False
        assert P._is_enforceable_canonical("   ", ["x"]) is False

    def test_monolith_class_has_new_rule(self):
        from src.translator_agent import TranslationPipeline as TP
        assert TP._is_enforceable_canonical(" Cloak Room", ["Cloak Room"]) is False
        assert TP._is_enforceable_canonical("帕丁顿行李寄存处", ["Left Luggage Office"]) is True


# ---------------------------------------------------------------------------
# 端到端：核查 / 注册表 / variant 表
# ---------------------------------------------------------------------------

def _new_registry_conn(rows):
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
    for canon, srcs in rows:
        conn.execute(
            "INSERT INTO entity_registry (canonical_name, source_names) VALUES (?, ?)",
            (canon, json.dumps(srcs)),
        )
    conn.commit()
    return conn


def _make_pipeline(conn, module_name):
    mod = __import__(module_name, fromlist=["TranslationPipeline"])
    TP = mod.TranslationPipeline
    p = TP.__new__(TP)

    class _DE:
        def __init__(self):
            self.conn = conn

        def is_canonical_name(self, name):
            return False

        def add_entity_variant(self, canonical, variant):
            pass

    p.decision_engine = _DE()
    from src.core.decision_engine import get_sibling_canonicals_by_source
    p._sibling_canonicals_cache = get_sibling_canonicals_by_source(conn)
    p._entity_registry_cache = {}
    p._entity_variants_cache = {}
    return p


@pytest.fixture(params=["src.pipeline", "src.translator_agent"])
def pipeline_factory(request):
    return lambda conn: _make_pipeline(conn, request.param)


class TestCheckEntityConsistency:
    def test_ch10_chunk007_identity_canonical_suppressed(self, pipeline_factory):
        """ch10 chunk007 复现：canonical=" Cloak Room" 不再误报 missing。"""
        conn = _new_registry_conn([(" Cloak Room", ["Cloak Room"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch10_chunk007",
            "把行李寄存票丢在站台上。",                    # 正常中文译法，无英文串
            "drop the Cloak Room ticket on the platform.",  # 源文出现 Cloak Room
        )
        assert notes == [], notes
        conn.close()

    def test_genuine_translated_canonical_missing_reported(self, pipeline_factory):
        """常规实体（中文 canonical）缺席仍报 missing（原行为不回归）。"""
        conn = _new_registry_conn([("帕丁顿行李寄存处", ["Left Luggage Office"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch10_chunk007",
            "他把包放在了站台上。",
            "He left it at the Left Luggage Office.",
        )
        assert any(n["type"] == "missing" for n in notes), notes
        conn.close()


class TestRegistryBuilders:
    def test_identity_canonical_excluded_from_rewriter_registry(self, pipeline_factory):
        conn = _new_registry_conn([
            (" Cloak Room", ["Cloak Room"]),
            ("帕丁顿行李寄存处", ["Left Luggage Office"]),
        ])
        p = pipeline_factory(conn)
        reg = p._build_entity_registry_for_rewriter(
            "drop the Cloak Room ticket at the Left Luggage Office"
        )
        assert reg == {"Left Luggage Office": "帕丁顿行李寄存处"}, reg
        conn.close()

    def test_identity_canonical_excluded_from_critic_variants(self, pipeline_factory):
        conn = _new_registry_conn([
            (" Cloak Room", ["Cloak Room"]),
            ("帕丁顿行李寄存处", ["Left Luggage Office"]),
        ])
        p = pipeline_factory(conn)
        vs = p._build_entity_variants_for_critic(
            "drop the Cloak Room ticket at the Left Luggage Office"
        )
        assert list(vs.keys()) == ["Left Luggage Office"], vs
        conn.close()
