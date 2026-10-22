"""Round 115: 斜杠备选 canonical 在场判定 + AUDIT_EARLY_STOP 与 verdict 语义对齐（ch06 chunk012 根因）。

ch06 chunk012: 译文质量实际合格（无英文残留、语义完整），但两个系统性误差叠加：
  1. canonical "国民党/民族主义者"（斜杠双备选）— 译文用「民族主义者」，整串/前后缀
     容差都差在斜杠上 → 实体核查误判 missing + Jev per-entity noul 0.24 未被护栏覆盖
     → entity_consistency 2.4 分。
  2. AUDIT_EARLY_STOP (R99 P3) 按原始 legacy 分数把三个"被 verdict 明确跳过
     （uncertain）"的维度计入低于 4.0 分 → 首轮直接 fallback，跳过 judge + rewriter。
R8 起 synthesize_verdict 对 uncertain 维度本就不判 REJECT（ch06 chunk012 报告
decision=PASS），early-stop 的前提"judge 必然 REJECT"不成立 → 合格译文被误杀。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.judges.primitives import (
    apply_entity_presence_override,
    entity_present_in_text,
)
from src.judges.typesafe_spec import NoulAnswer
from src.pipeline import TranslationPipeline as P


# ---------------------------------------------------------------------------
# 斜杠备选 canonical 在场判定
# ---------------------------------------------------------------------------

class TestSlashCanonicalPresence:
    def test_alternative_hit(self):
        assert entity_present_in_text("国民党/民族主义者", [], "民族主义者也会派人来。") == "民族主义者"
        assert entity_present_in_text("国民党/民族主义者", [], "国民党方面尚无回应。") == "国民党"

    def test_no_alternative_miss(self):
        assert entity_present_in_text("国民党/民族主义者", [], "完全无关的文本。") is None

    def test_fullwidth_slash(self):
        assert entity_present_in_text("国民党／民族主义者", [], "民族主义者来了") == "民族主义者"

    def test_parenthesized_slash_still_uses_base(self):
        # 刚果（河/地区）→ 括注剥除后 base=刚果（斜杠已随括注消失）
        assert entity_present_in_text("刚果（河/地区）", [], "他去了刚果。") == "刚果"

    def test_canonical_present_in_text_pipeline(self):
        p = P.__new__(P)
        assert p._canonical_present_in_text("国民党/民族主义者", "民族主义者也会派人来。") is True
        assert p._canonical_present_in_text("国民党/民族主义者", "完全无关。") is False

    def test_monolith_mirror(self):
        from src.translator_agent import TranslationPipeline as TP
        from src.translator_agent import entity_present_in_text as M
        assert M("国民党/民族主义者", [], "民族主义者来了") == "民族主义者"
        tp = TP.__new__(TP)
        assert tp._canonical_present_in_text("国民党/民族主义者", "民族主义者来了") is True


class TestOverrideSlashCanonical:
    def _registry(self):
        return {"the Nationalists": {"canonical": "国民党/民族主义者", "variants": []}}

    def test_jazzy_false_positive_overridden(self):
        ans = NoulAnswer(noul=0.24)
        answers = {"entity_per_the_Nationalists": ans}
        overrides = apply_entity_presence_override(answers, self._registry(), "……民族主义者也会派人来。")
        assert ans.noul == 1.0
        assert [o["matched"] for o in overrides] == ["民族主义者"]

    def test_real_missing_not_overridden(self):
        ans = NoulAnswer(noul=0.24)
        answers = {"entity_per_the_Nationalists": ans}
        overrides = apply_entity_presence_override(answers, self._registry(), "完全无关的文本。")
        assert ans.noul == 0.24
        assert overrides == []


# ---------------------------------------------------------------------------
# AUDIT_EARLY_STOP 只统计结论性低分
# ---------------------------------------------------------------------------

CH06_SCORES = {
    "voice_consistency": 9.0, "style_compliance": 8.5,
    "literal_residue": 3.1, "semantic_preservation": 3.9,
    "readability": 7.4, "fluency": 7.4, "entity_consistency": 2.4,
}
CH06_SKIPPED = ["voice_drift", "entity_consistency", "literal_residue",
                "semantic_loss", "style_compliance_score"]


class TestAuditEarlyStopCount:
    @pytest.mark.parametrize("module", ["src.pipeline", "src.translator_agent"])
    def test_ch06_uncertain_dims_not_counted(self, module):
        TP = __import__(module, fromlist=["TranslationPipeline"]).TranslationPipeline
        report = {"scores": dict(CH06_SCORES),
                  "_typesafe_verdict": {"skipped_dimensions": CH06_SKIPPED}}
        assert TP._audit_early_stop_count(report, 4.0) == 0

    @pytest.mark.parametrize("module", ["src.pipeline", "src.translator_agent"])
    def test_decisive_low_still_counted(self, module):
        TP = __import__(module, fromlist=["TranslationPipeline"]).TranslationPipeline
        report = {"scores": {"literal_residue": 1.0, "semantic_preservation": 2.0,
                             "entity_consistency": 2.0},
                  "_typesafe_verdict": {"skipped_dimensions": []}}
        assert TP._audit_early_stop_count(report, 4.0) == 3

    def test_partial_skip_counted_only_unskipped(self):
        report = {"scores": {"literal_residue": 1.0, "semantic_preservation": 2.0,
                             "entity_consistency": 2.0},
                  "_typesafe_verdict": {"skipped_dimensions": ["semantic_preservation"]}}
        assert P._audit_early_stop_count(report, 4.0) == 2

    def test_old_report_without_verdict_falls_back_to_old_behavior(self):
        report = {"scores": {"literal_residue": 1.0, "semantic_preservation": 2.0,
                             "entity_consistency": 2.0}}
        assert P._audit_early_stop_count(report, 4.0) == 3


# ---------------------------------------------------------------------------
# 实体核查集成：ch06 chunk012 根因不复现
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
    def test_slash_canonical_alternative_no_missing(self, pipeline_factory):
        conn = _new_registry_conn([("国民党/民族主义者", ["the Nationalists"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch06_chunk012",
            "很快，我猜，民族主义者和独立自由党也会派个代表团来。",
            "Soon, I suppose, the Nationalists and the Independent Liberals will be sending a delegation too.",
        )
        assert notes == [], notes
        conn.close()

    def test_slash_canonical_real_missing_still_reported(self, pipeline_factory):
        conn = _new_registry_conn([("国民党/民族主义者", ["the Nationalists"])])
        p = pipeline_factory(conn)
        notes = p._check_entity_consistency(
            "ch06_chunk012",
            "有人等在楼下。",
            "I have made enemies of the Nationalists.",
        )
        assert any(n["type"] == "missing" for n in notes), notes
        conn.close()
