"""Round 114: 实体"在场"判定改按词/短语边界（ch30 chunk000 根因）。

ch30 chunk000: 原文是 "The Bertillon measurements that that scoundrel talked
about?"（贝蒂荣测量法），registry 实体 "伯特"←"Bert" 因裸子串 "Bert" ⊂ "Bertillon"
被误判"实体在场" → 实体核查要求译文出现「伯特」→ 不可满足 → 实体硬失败 →
rewriter 自评"源文没有独立的 Bert"一字未改 → REWRITER_STAGNATED →
PERMANENTLY_FAILED。同类隐患: "Europe"⊂"European"、"Herzoslovakia"⊂
"Herzoslovakian"、"drawing-room"⊂"drawing-rooms"。

修复（Round 114）：新增共享工具 entity_source_name_in_text
（src/utils/text_match.py + 单文件镜像），全部 source_name 在场判定
（pipeline 核查/审计/registry 构建 + judge build_state 过滤）改用它。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.pipeline import TranslationPipeline as P
from src.utils.text_match import entity_source_name_in_text


class TestHelperFunction:
    @pytest.mark.parametrize("sn,text", [
        ("Bert", "The Bertillon measurements that that scoundrel talked about?"),
        ("Bert", "Albert was here."),
        ("Bert", "Bertrand spoke quietly."),
        ("Europe", "European customs vary."),
        ("Herzoslovakia", "a Herzoslovakian nobleman"),
        ("drawing-room", "two drawing-rooms upstairs"),
        ("Lomax", "Lomaxwell was not there."),
    ])
    def test_derivative_substring_rejected(self, sn, text):
        assert entity_source_name_in_text(sn, text) is False

    @pytest.mark.parametrize("sn,text", [
        ("Bert", "Bert waited."),
        ("Bert", "Bert's hat"),
        ("Bert", "—Bert—"),
        ("Europe", "a trip to Europe."),
        ("Herzoslovakia", "Herzoslovakia,"),
        ("drawing-room", "the drawing-room door"),
        ("Mr. Lomax", "Mr. Lomax arrived."),
        ("Mr.", "Mr. Lomax"),
        ("the Comrades", "the Comrades met"),
    ])
    def test_boundary_forms_accepted(self, sn, text):
        assert entity_source_name_in_text(sn, text) is True

    def test_none_and_empty_safe(self):
        assert entity_source_name_in_text(None, "Bert") is False
        assert entity_source_name_in_text("Bert", None) is False
        assert entity_source_name_in_text("", "Bert") is False
        assert entity_source_name_in_text("Bert", "") is False

    def test_monolith_mirror(self):
        from src.translator_agent import entity_source_name_in_text as M
        assert M("Bert", "The Bertillon measurements") is False
        assert M("Bert", "Bert's hat") is True


class TestBuildStateFilter:
    def _call(self, build_state, source):
        return build_state(
            source_text=source,
            raw_translation="直译",
            literary_translation="润色",
            style_guide_stats={},
            entity_registry={"Bert": "伯特"},
            entity_variants={"Bert": ["伯儿提"]},
        )

    @pytest.mark.parametrize("mod", ["src.judges.judge", "src.translator_agent"])
    def test_derivative_only_excluded(self, mod):
        mod_obj = __import__(mod, fromlist=["build_state"])
        st = self._call(mod_obj.build_state, "The Bertillon measurements that that scoundrel talked about?")
        assert st["entity_registry"] == {}, st["entity_registry"]
        assert st["entity_variants"] == {}, st["entity_variants"]

    @pytest.mark.parametrize("mod", ["src.judges.judge", "src.translator_agent"])
    def test_standalone_included(self, mod):
        mod_obj = __import__(mod, fromlist=["build_state"])
        st = self._call(mod_obj.build_state, "Bert waited below.")
        assert st["entity_registry"] == {"Bert": "伯特"}
        assert st["entity_variants"] == {"Bert": ["伯儿提"]}


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


class TestPipelineIntegration:
    def test_ch30_bertillon_no_missing(self, pipeline_factory):
        """ch30 chunk000: "Bert" 仅在 "Bertillon" 内出现 → 不再强制「伯特」。"""
        conn = _new_registry_conn([("伯特", ["Bert"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch30_chunk000",
            "“指纹。那个恶棍谈到过的贝蒂荣测量法？”",
            "The Bertillon measurements that that scoundrel talked about?",
        )
        assert notes == [], notes
        conn.close()

    def test_standalone_bert_missing_still_reported(self, pipeline_factory):
        conn = _new_registry_conn([("伯特", ["Bert"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch30_chunk000",
            "那人等着。",
            "Bert waited below.",
        )
        assert any(n["type"] == "missing" for n in notes), notes
        conn.close()

    def test_registry_builder_excludes_derivative_only(self, pipeline_factory):
        conn = _new_registry_conn([("伯特", ["Bert"])])
        p = pipeline_factory(conn)
        assert p._build_entity_registry_for_rewriter("The Bertillon measurements.") == {}
        assert p._build_entity_registry_for_rewriter("Bert waited.") == {"Bert": "伯特"}
        conn.close()
