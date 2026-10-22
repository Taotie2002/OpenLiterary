"""Round 111: 限定词 + 全小写普通名词短语不再算实体（ch09 chunk004 根因）。

ch09 chunk004: reference agent 把 "the stairs" → 楼梯 登记为实体。实体核查按
"规范主名必须字面出现" 判 missing → 实体硬失败退回重写；而中文习惯动词化表达
（上楼/下楼），rewriter 无法两全（自评明确说"上楼即 the stairs 的动作表达"）
→ 输出字节级停滞 → REWRITER_STAGNATED → PERMANENTLY_FAILED。

修复（Round 111）：
  1. _is_valid_entity_source_name 新增规则：限定词（the/a/an/my/...）+ 其后所有词
     全小写 → 普通名词短语（非专名）。大小写敏感，真专名（the Comrades / the King /
     the Tower）不受影响。
  2. _build_entity_registry_for_rewriter / _build_entity_variants_for_critic 同步
     过滤无效 source_name，避免 rewriter/critic 再收到不可执行的"统一为规范主名"指令。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.pipeline import TranslationPipeline as P


# ---------------------------------------------------------------------------
# 谓词：限定词 + 全小写名词短语
# ---------------------------------------------------------------------------

class TestArticleNounPattern:
    @pytest.mark.parametrize("sn", [
        "the stairs", "the chauffeur", "my maid", "the unemployed young man",
        "The manager", "the financier", "the drawing-room", "your house",
        "the desk clerk", "a young man", "the room", "the yew hedge",
    ])
    def test_common_noun_phrase_rejected(self, sn):
        assert P._is_valid_entity_source_name(sn) is False, f"{sn!r} 是普通名词短语，应判无效"

    @pytest.mark.parametrize("sn", [
        # 大写专名不受影响（限定词后含大写词）
        "the Comrades", "the Tower", "the King", "the City", "the Abbey",
        "the Sûreté", "the Bois", "the Wallace Collection", "the Obolovitch family",
        # 无限定词的专名
        "Walton", "Mr. Lomax", "Count Stylptitch", "Virginia Revel", "Ranelagh",
        "Anthony Cade", "Gentleman Joe",
    ])
    def test_proper_names_accepted(self, sn):
        assert P._is_valid_entity_source_name(sn) is True, f"{sn!r} 是专名，不应被拒"

    def test_monolith_class_has_new_rule(self):
        from src.translator_agent import TranslationPipeline as TP
        assert TP._is_valid_entity_source_name("the stairs") is False
        assert TP._is_valid_entity_source_name("the Comrades") is True


# ---------------------------------------------------------------------------
# 注册表构建过滤（rewriter / critic）
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


class TestRegistryBuildersFilter:
    def test_junk_alias_excluded_rewriter_registry(self, pipeline_factory):
        conn = _new_registry_conn([
            ("楼梯", ["the stairs"]),
            ("沃尔顿", ["Walton", "the chauffeur"]),
        ])
        p = pipeline_factory(conn)
        src = "He went up the stairs, and Walton, the chauffeur, waited below."
        reg = p._build_entity_registry_for_rewriter(src)
        # 普通名词别名被过滤；真专名保留（chauffeur 别名被滤掉，Walton 保留）
        assert reg == {"Walton": "沃尔顿"}, reg
        conn.close()

    def test_junk_entity_variants_excluded(self, pipeline_factory):
        conn = _new_registry_conn([
            ("楼梯", ["the stairs"]),
            ("沃尔顿", ["Walton", "the chauffeur"]),
        ])
        p = pipeline_factory(conn)
        src = "He went up the stairs, and Walton, the chauffeur, waited below."
        vs = p._build_entity_variants_for_critic(src)
        assert list(vs.keys()) == ["Walton"], vs
        conn.close()


# ---------------------------------------------------------------------------
# 实体核查：ch09 chunk004 根因不复现，真违规不回归
# ---------------------------------------------------------------------------

class TestCheckEntityConsistency:
    def test_common_noun_missing_suppressed(self, pipeline_factory):
        """ch09 chunk004: "the stairs"→楼梯 不再误报 missing。"""
        conn = _new_registry_conn([("楼梯", ["the stairs"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch09_chunk004",
            "他一声不响地上楼去了。",          # 动词化表达，无「楼梯」字样
            "He went up the stairs.",          # 源文出现 the stairs
        )
        assert notes == [], notes
        conn.close()

    def test_proper_name_missing_still_reported(self, pipeline_factory):
        """真专名缺席仍报 missing（原行为不回归）。"""
        conn = _new_registry_conn([("沃尔顿", ["Walton"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch09_chunk000",
            "有人等在楼下。",
            "Walton waited below.",
        )
        assert any(n["type"] == "missing" for n in notes), notes
        conn.close()

    def test_proper_name_rendered_no_report(self, pipeline_factory):
        conn = _new_registry_conn([("沃尔顿", ["Walton"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch09_chunk000",
            "沃尔顿等在楼下。",
            "Walton waited below.",
        )
        assert notes == [], notes
        conn.close()
