"""Round 113: 叙事从句 source_name / 描述性占位 canonical 不再触发实体硬失败（ch16 chunk006 根因）。

ch16 chunk006: reference agent 把第一人称叙事从句 "I wanted to know" / "I haven't
written" 登记为实体 "女主人公（未具名叙述者）" 的 source_name。实体核查按
"规范主名必须字面出现" 判 missing —— 而规范主名是描述性占位串（译文用第一人称
"我"，永远不可能字面出现）→ 实体硬失败退回重写 → rewriter 无事可做
（oscillation_similarity=0.9987）→ STAGNATION → PERMANENTLY_FAILED。

修复（Round 113）：
  1. _is_valid_entity_source_name 新增从句规则：主语代词（I/you/he/she/it/we/they）
     开头且带后续内容 → 叙事从句，非实体指称（泛化 Round 106 的有限动词白名单，
     "wanted/haven't written" 不在白名单内被击穿）。
  2. _is_enforceable_canonical 剔除描述性占位 canonical（未具名/无名/匿名/隐含…）：
     无名角色没有可字面核查的专名，字面要求必然不可满足。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.pipeline import TranslationPipeline as P


class TestClauseSourceName:
    @pytest.mark.parametrize("sn", [
        "I wanted to know", "I haven't written", "I've written",
        "i wanted to know", "You know nothing", "you know",
        "he (指代 Professor Wynward)", "we of the Sûreté",
        "They were standing there", "It was raining",
    ])
    def test_clause_rejected(self, sn):
        assert P._is_valid_entity_source_name(sn) is False, f"{sn!r} 是叙事从句，应判无效"

    @pytest.mark.parametrize("sn", [
        # 单 token 专名不受影响（不能因字母组合前缀被误杀）
        "Ivan", "Shelley", "Walton", "Daisy", "Herzoslovakia",
        "Sherlock Holmes", "Anthony Cade", "Mr. Lomax", "Baron Lollipop",
        "the Comrades", "the King",
    ])
    def test_proper_names_accepted(self, sn):
        assert P._is_valid_entity_source_name(sn) is True, f"{sn!r} 是专名，不应被拒"

    def test_monolith_class_has_new_rule(self):
        from src.translator_agent import TranslationPipeline as TP
        assert TP._is_valid_entity_source_name("I wanted to know") is False
        assert TP._is_valid_entity_source_name("Ivan") is True


class TestPlaceholderCanonical:
    @pytest.mark.parametrize("canonical", [
        "女主人公（未具名叙述者）",
        "美国绅士（未具名）",
        "年轻陆军军官（未具名）",
        "（未具名的死者）",
        "（未具名的男性对话者）",
        "女主人（无名叙事者）",
        "叙述中的主人公（无名男子）",
        "管家（无名）",
        "被敲诈的英国女人（匿名受害者）",
        "法国人（无具名配角）",
    ])
    def test_placeholder_canonical_not_enforceable(self, canonical):
        assert P._is_enforceable_canonical(canonical, ["I wanted to know"]) is False

    @pytest.mark.parametrize("canonical", [
        "安东尼·凯德",
        "卡特赖特医生",
        "多卡斯·杜莫弗依（推测）",
        "卡特拉姆勋爵（第九代卡特拉姆侯爵）",
    ])
    def test_real_canonical_still_enforceable(self, canonical):
        assert P._is_enforceable_canonical(canonical, ["Anthony Cade"]) is True

    def test_identity_mapping_regression_round112(self):
        assert P._is_enforceable_canonical("Cloak Room", ["Cloak Room"]) is False
        assert P._is_enforceable_canonical("Morris Cowley", ["Morris Cowley"]) is False


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
    def test_narrator_clause_entity_excluded(self, pipeline_factory):
        conn = _new_registry_conn([
            ("女主人公（未具名叙述者）", ["I wanted to know", "I haven't written"]),
            ("夏洛克·福尔摩斯", ["Sherlock Holmes"]),
        ])
        p = pipeline_factory(conn)
        src = "I wanted to know. I haven't written anything. Sherlock Holmes nodded."
        reg = p._build_entity_registry_for_rewriter(src)
        assert reg == {"Sherlock Holmes": "夏洛克·福尔摩斯"}, reg
        vs = p._build_entity_variants_for_critic(src)
        assert list(vs.keys()) == ["Sherlock Holmes"], vs
        conn.close()


class TestCheckEntityConsistency:
    def test_clause_source_no_missing(self, pipeline_factory):
        """ch16 chunk006: "I wanted to know"→女主人公 不再误报 missing。"""
        conn = _new_registry_conn([
            ("女主人公（未具名叙述者）", ["I wanted to know", "I haven't written"]),
        ])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch16_chunk006",
            "然后我就想知道得太多了。我还没写下来。",
            "Then I wanted to know too much. I haven't written.",
        )
        assert notes == [], notes
        conn.close()

    def test_placeholder_canonical_missing_suppressed(self, pipeline_factory):
        conn = _new_registry_conn([("美国绅士（未具名）", ["an American gent"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch16_chunk000",
            "一位美国绅士坐在那里。",
            "An American gent sat there.",
        )
        assert notes == [], notes
        conn.close()

    def test_proper_name_missing_still_reported(self, pipeline_factory):
        """真专名缺席仍报 missing（原行为不回归）。"""
        conn = _new_registry_conn([("沃尔顿", ["Walton"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch16_chunk000",
            "有人等在楼下。",
            "Walton waited below.",
        )
        assert any(n["type"] == "missing" for n in notes), notes
        conn.close()
