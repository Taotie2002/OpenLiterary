"""Round 144: entity alias hygiene regression tests.

这些测试固定住四条纪律：
  1. 泛称/关系/描述不能进入硬专名；
  2. 共享 alias 不能单独把人名串成一个传递闭包；
  3. 跨人物 variant_names 不能继续充当“合法异写”；
  4. decision DB 能唯一裁决时，不因另一个污染 owner 把正确 alias 一并隔离。
"""
from __future__ import annotations

import json
import sqlite3


def _mk_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            context_names TEXT,
            quarantined_names TEXT,
            variant_names TEXT,
            pronoun_gender TEXT,
            entity_type TEXT,
            first_seen_chunk TEXT,
            confidence TEXT DEFAULT 'high',
            translation_strategy TEXT DEFAULT 'free',
            pending_review INTEGER DEFAULT 0,
            review_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        );
        CREATE TABLE decision_db (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER NOT NULL,
            source_key TEXT NOT NULL,
            translation TEXT NOT NULL,
            reason TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            locked INTEGER DEFAULT 0
        );
    """)
    return conn


def _row(conn, canonical, sources, variants=None, pending=0, confidence="high",
         etype="person", contexts=None, quarantine=None):
    conn.execute(
        "INSERT INTO entity_registry "
        "(canonical_name, source_names, context_names, quarantined_names, "
        "variant_names, entity_type, confidence, pending_review) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (canonical, json.dumps(sources, ensure_ascii=False),
         json.dumps(contexts or [], ensure_ascii=False),
         json.dumps(quarantine or [], ensure_ascii=False),
         json.dumps(variants or [], ensure_ascii=False),
         etype, confidence, pending),
    )


def _decision(conn, source, translation):
    conn.execute(
        "INSERT INTO decision_db (level, source_key, translation) VALUES (2, ?, ?)",
        (source, translation),
    )


class TestAliasClassification:
    def test_context_forms_are_not_hard_names(self):
        from src.core.decision_engine import classify_entity_source_name

        for text in ("the Emperor", "my Lord", "her Duke", "Emperors"):
            assert classify_entity_source_name(text) == "title", text
        for text in ("fifteen years", "the man", "Emperor's ears", "the cub"):
            assert classify_entity_source_name(text) == "description", text
        for text in ("Grandfather", "his nephew"):
            assert classify_entity_source_name(text) == "relation", text
        for text in ("Duke Leto", "Paul Atreides", "Baron Harkonnen"):
            assert classify_entity_source_name(text) == "proper", text
        for text in ("Judge of the Change", "Truthsayer", "Princess Royal",
                     "the old woman"):
            assert classify_entity_source_name(text) in ("title", "relation", "description"), text

    def test_canonical_identity_is_not_shared_title(self):
        from src.core.decision_engine import (
            canonical_names_related, canonical_variant_forms_related,
        )

        assert canonical_names_related("雷托公爵", "雷托")
        assert canonical_names_related("哈克南男爵", "弗拉基米尔·哈克南男爵")
        assert not canonical_names_related("保罗·厄崔迪", "弗拉基米尔·哈克南男爵")
        assert not canonical_names_related("皇帝", "弗拉基米尔·哈克南男爵")
        assert canonical_variant_forms_related("伊鲁兰公主", "伊如兰公主")
        assert canonical_variant_forms_related("阿拉基斯", "阿拉肯")
        assert not canonical_variant_forms_related("保罗·厄崔迪", "雷托公爵")

    def test_alias_attestation_handles_case_apostrophe_and_boundaries(self):
        from src.core.decision_engine import alias_attested_in_text

        text = "Duke Leto’s son, Paul Atreides, met Baron Harkonnen."
        assert alias_attested_in_text("Duke Leto", text)
        assert alias_attested_in_text("Paul Atreides", text)
        assert alias_attested_in_text("Baron Harkonnen", text)
        assert not alias_attested_in_text("Shaddam IV", text)
        assert not alias_attested_in_text("Paul", "Paula arrived.")


class TestAliasSanitize:
    def test_decision_db_keeps_correct_owner_and_quarantines_others(self):
        from src.core.decision_engine import sanitize_entity_registry_aliases

        conn = _mk_db()
        _decision(conn, "Paul Atreides", "保罗·厄崔迪")
        _decision(conn, "Duke Leto", "雷托公爵")
        _decision(conn, "Duke Paul Atreides", "保罗·厄崔迪")
        _row(conn, "保罗·厄崔迪",
             ["Paul Atreides", "Duke Leto", "Duke Paul", "an Atreides", "my Lord"])
        _row(conn, "弗拉基米尔·哈克南男爵",
             ["Paul Atreides", "Duke Leto", "Baron Harkonnen", "the Emperor"])
        _row(conn, "雷托公爵", ["Duke Leto", "Leto the Just"])
        _row(conn, "雷托", ["Leto", "Duke Leto"])
        _row(conn, "沙达姆四世", ["Shaddam IV", "the Emperor"])
        _row(conn, "皇帝", ["the Emperor"])
        conn.commit()

        before = conn.execute(
            "SELECT source_names FROM entity_registry WHERE canonical_name='保罗·厄崔迪'"
        ).fetchone()[0]
        res = sanitize_entity_registry_aliases(conn, dry_run=True)
        after = conn.execute(
            "SELECT source_names FROM entity_registry WHERE canonical_name='保罗·厄崔迪'"
        ).fetchone()[0]
        assert after == before, "dry-run 不得写库"
        assert res["decision_preferred_aliases"] >= 2

        paul = next(c for c in res["changes"] if c["canonical"] == "保罗·厄崔迪")
        baron = next(c for c in res["changes"] if c["canonical"] == "弗拉基米尔·哈克南男爵")
        assert "Paul Atreides" in paul["source_names"]
        assert "Duke Leto" not in paul["source_names"]
        assert "an Atreides" not in paul["source_names"]
        assert "my Lord" in paul["context_names"]
        assert "Paul Atreides" not in baron["source_names"]
        assert "Paul Atreides" in baron["quarantined_names"]
        assert "Baron Harkonnen" in baron["source_names"]
        conn.close()

    def test_single_owner_foreign_alias_quarantined_by_decision_db(self):
        from src.core.decision_engine import sanitize_entity_registry_aliases

        conn = _mk_db()
        _decision(conn, "Duke Leto", "雷托公爵")
        _row(conn, "保罗·厄崔迪", ["Paul Atreides", "Duke Leto"])
        _row(conn, "雷托公爵", ["Leto"])
        conn.commit()

        res = sanitize_entity_registry_aliases(conn, dry_run=True)
        assert res["decision_single_owner_quarantined"] >= 1
        paul = next(c for c in res["changes"] if c["canonical"] == "保罗·厄崔迪")
        assert "Duke Leto" not in paul["source_names"]
        assert "Duke Leto" in paul["quarantined_names"]
        assert "Paul Atreides" in paul["source_names"]
        conn.close()

    def test_decision_ownership_is_conservative_for_titles_and_role_names(self):
        from src.core.decision_engine import decision_alias_conflicts_with_owner

        targets = {
            "duke leto": {"雷托公爵"},
            "the emperor": {"皇帝"},
            "sayyadina": {"赛亚迪娜"},
        }
        names = ["保罗·厄崔迪", "雷托", "雷托公爵", "沙达姆四世",
                 "皇帝", "杰西卡", "赛亚迪娜"]
        types = {n: "person" for n in names}
        assert decision_alias_conflicts_with_owner(
            "Duke Leto", "保罗·厄崔迪", targets, names, types)
        assert not decision_alias_conflicts_with_owner(
            "Duke Leto", "雷托", targets, names, types)
        assert decision_alias_conflicts_with_owner(
            "Sayyadina", "杰西卡", targets, names, types)
        assert not decision_alias_conflicts_with_owner(
            "Sayyadina", "赛亚迪娜", targets, names, types)
        assert not decision_alias_conflicts_with_owner(
            "the Emperor", "沙达姆四世", targets, names, types)

    def test_decision_alias_matching_does_not_use_loose_character_similarity(self):
        from src.core.decision_engine import sanitize_entity_registry_aliases

        conn = _mk_db()
        _decision(conn, "Chalice", "查利斯")
        _row(conn, "查莉丝", ["Chalice"])
        _row(conn, "伊布利斯", ["Iblis"])
        conn.commit()

        res = sanitize_entity_registry_aliases(conn, dry_run=True)
        assert res["decision_single_owner_quarantined"] == 0
        src = conn.execute(
            "SELECT source_names FROM entity_registry WHERE canonical_name='查莉丝'"
        ).fetchone()[0]
        assert json.loads(src) == ["Chalice"]
        conn.close()

    def test_report_separates_active_and_pending_owners(self):
        from src.core.decision_engine import entity_registry_alias_report

        conn = _mk_db()
        _row(conn, "保罗·厄崔迪", ["Paul Atreides"], pending=0)
        _row(conn, "弗拉基米尔·哈克南男爵", ["Paul Atreides"], pending=1)
        conn.commit()
        report = entity_registry_alias_report(conn)
        assert report["active_proper_conflicts"] == 0
        assert report["all_proper_conflicts"] == 1
        assert report["pending_rows"] == 1
        conn.close()

    def test_report_counts_single_owner_foreign_alias(self):
        from src.core.decision_engine import entity_registry_alias_report

        conn = _mk_db()
        _decision(conn, "Duke Leto", "雷托公爵")
        _row(conn, "保罗·厄崔迪", ["Paul Atreides", "Duke Leto"])
        _row(conn, "雷托公爵", ["Leto"])
        conn.commit()
        report = entity_registry_alias_report(conn)
        assert report["active_foreign_aliases"] == 1
        assert report["foreign_alias_examples"][0]["canonical"] == "保罗·厄崔迪"
        conn.close()


class TestVariantSanitize:
    def test_cross_person_variants_removed_but_legitimate_variants_kept(self):
        from src.core.decision_engine import sanitize_entity_registry_variants

        conn = _mk_db()
        _row(conn, "保罗·厄崔迪", ["Paul Atreides"],
             variants=["雷托公爵", "雷托", "保罗·厄崔迪"])
        _row(conn, "弗拉基米尔·哈克南男爵", ["Baron Harkonnen"],
             variants=["哈克南男爵", "保罗·厄崔迪", "雷托公爵",
                       "雷托", "沙达姆四世", "皇帝陛下", "皇帝"])
        _row(conn, "沙达姆四世", ["Shaddam IV"])
        _row(conn, "皇帝", ["the Emperor"])
        _row(conn, "雷托公爵", ["Duke Leto"])
        _row(conn, "雷托", ["Leto"])
        _row(conn, "伊鲁兰公主", ["Irulan"], variants=["伊如兰公主", "皇家公主"])
        _row(conn, "伊如兰公主", ["Irulan"])
        _row(conn, "阿拉基斯", ["Arrakis"], variants=["阿拉肯"], etype="place")
        _row(conn, "阿拉肯", ["Arrakeen"], etype="place")
        _row(conn, "卡拉丹城堡", ["Castle Caladan"], variants=["卡拉丹"], etype="place")
        _row(conn, "卡拉丹", ["Caladan"], etype="place")
        _row(conn, "贝尼·杰瑟里特姐妹会", ["Bene Gesserit"], variants=["姐妹会"],
             etype="organization")
        _row(conn, "岳医生", ["Wellington Yueh"], variants=["韦林顿·岳"])
        _row(conn, "韦林顿·岳", ["Wellington Yueh"])
        _row(conn, "沙胡鲁德", ["Shai-Hulud"], variants=["沙虫"], etype="other")
        _row(conn, "沙虫", ["sandworm"], etype="other")
        _row(conn, "阿特雷德家族", ["House Atreides"], variants=["厄崔迪家族", "亚崔迪家族"],
             etype="organization")
        _row(conn, "厄崔迪家族", ["House Atreides"], etype="organization")
        _row(conn, "亚崔迪家族", ["House Atreides"], etype="organization")
        _row(conn, "皇家公主", ["Princess Royal"])
        _row(conn, "凯恩斯", ["Kynes"], variants=["交接法官"])
        _row(conn, "交接法官", ["Judge of the Change"])
        _row(conn, "盖乌斯·海伦·莫希阿姆", ["Gaius Helen Mohiam"],
             variants=["杰西卡", "真言者", "老妇人"])
        _row(conn, "杰西卡", ["Jessica"])
        _row(conn, "真言者", ["Truthsayer"])
        _row(conn, "老妇人", ["the old woman"])
        conn.commit()

        res = sanitize_entity_registry_variants(conn, dry_run=True)
        assert res["removed_instances"] >= 5
        baron = next(c for c in res["changes"]
                     if c["canonical"] == "弗拉基米尔·哈克南男爵")
        assert "哈克南男爵" in baron["kept"]
        for bad in ("保罗·厄崔迪", "雷托公爵", "雷托", "沙达姆四世"):
            assert bad in baron["removed"]
        for canonical in ("伊鲁兰公主", "阿拉基斯", "卡拉丹城堡", "贝尼·杰瑟里特姐妹会",
                          "岳医生", "沙胡鲁德", "阿特雷德家族", "凯恩斯"):
            assert not any(c["canonical"] == canonical for c in res["changes"])
        gaius = next(c for c in res["changes"]
                     if c["canonical"] == "盖乌斯·海伦·莫希阿姆")
        assert "杰西卡" in gaius["removed"]
        assert "真言者" not in gaius["removed"]
        assert "老妇人" not in gaius["removed"]
        conn.close()


class TestQueryConsumers:
    def test_consistency_glossary_filters_context_and_cross_person_variants(self, tmp_path):
        from src.consistency import _load_glossary

        conn = _mk_db()
        _row(conn, "阿尔法", ["Test Alpha", "the Emperor"], variants=["贝塔"])
        _row(conn, "贝塔", ["Test Beta"])
        conn.commit()
        db_path = tmp_path / "decision_db.sqlite"
        dst = sqlite3.connect(str(db_path))
        conn.backup(dst)
        conn.close()
        dst.close()

        entities = _load_glossary(str(db_path))
        alpha = entities["阿尔法"]
        assert alpha["source_names"] == ["Test Alpha"]
        assert "贝塔" not in alpha["variant_names"]

    def test_story_graph_aliases_filter_context_quarantine_and_foreign_canonical(self):
        from src.story_graph import _entity_aliases

        row = {
            "canonical_name": "阿尔法",
            "source_names": json.dumps(
                ["Test Alpha", "the Emperor", "Duke Beta"], ensure_ascii=False),
            "variant_names": json.dumps(["贝塔"], ensure_ascii=False),
        }
        aliases = _entity_aliases(
            row,
            canonical_names=["阿尔法", "贝塔"],
            quarantined_names=["Duke Beta"],
            entity_types={"阿尔法": "person", "贝塔": "person"},
            entity_sources={"阿尔法": ["Test Alpha"], "贝塔": ["Test Beta"]},
        )
        assert "阿尔法" in aliases
        assert "Test Alpha" in aliases
        assert "the Emperor" not in aliases
        assert "Duke Beta" not in aliases
        assert "贝塔" not in aliases

    def test_cross_person_helper_is_conservative_for_nonperson_and_unknown_names(self):
        from src.core.decision_engine import is_cross_person_canonical_variant

        names = ["保罗·厄崔迪", "弗拉基米尔·哈克南男爵", "哈克南男爵", "雷托公爵", "皇帝",
                 "岳医生", "韦林顿·岳", "沙胡鲁德", "沙虫",
                 "阿特雷德家族", "厄崔迪家族", "伊鲁兰公主", "伊如兰公主"]
        types = {
            "保罗·厄崔迪": "person", "弗拉基米尔·哈克南男爵": "person",
            "哈克南男爵": "person",
            "雷托公爵": "person", "皇帝": "person",
            "岳医生": "person", "韦林顿·岳": "person",
            "沙胡鲁德": "other", "沙虫": "other",
            "阿特雷德家族": "organization", "厄崔迪家族": "organization",
            "伊鲁兰公主": "person", "伊如兰公主": "person",
        }
        assert is_cross_person_canonical_variant(
            "弗拉基米尔·哈克南男爵", "保罗·厄崔迪", names, types)
        assert not is_cross_person_canonical_variant(
            "弗拉基米尔·哈克南男爵", "哈克南男爵", names, types)
        assert not is_cross_person_canonical_variant(
            "岳医生", "韦林顿·岳", names, types)
        assert not is_cross_person_canonical_variant(
            "沙胡鲁德", "沙虫", names, types)
        assert not is_cross_person_canonical_variant(
            "阿特雷德家族", "厄崔迪家族", names, types)
        assert not is_cross_person_canonical_variant(
            "伊鲁兰公主", "伊如兰公主", names, types)


class TestUnifyGuard:
    def test_shared_alias_does_not_create_giant_person_cluster(self):
        from src.core.decision_engine import unify_entity_registry

        conn = _mk_db()
        _row(conn, "保罗·厄崔迪", ["Paul Atreides", "Duke Leto", "an Atreides"])
        _row(conn, "弗拉基米尔·哈克南男爵",
             ["Baron Harkonnen", "Duke Leto", "the Emperor"])
        _row(conn, "雷托公爵", ["Duke Leto"])
        _row(conn, "雷托", ["Leto", "Duke Leto"])
        _row(conn, "沙达姆四世", ["Shaddam IV", "the Emperor"])
        _row(conn, "皇帝", ["The Emperor"])
        conn.commit()

        plan = unify_entity_registry(conn, dry_run=True)
        for info in plan.values():
            members = set([info["winner"]] + list(info["demoted"]))
            assert not ({"保罗·厄崔迪", "弗拉基米尔·哈克南男爵"} <= members)
            assert not ({"保罗·厄崔迪", "沙达姆四世"} <= members)
        conn.close()

    def test_confidence_beats_alias_count_for_winner(self):
        from src.core.decision_engine import unify_entity_registry

        conn = _mk_db()
        _row(conn, "Zorn", ["Zorn", "Zorn1", "Zorn2", "Zorn3", "Zorn4"],
             confidence="medium")
        _row(conn, "Zorn·Long", ["Zorn"], confidence="high")
        conn.commit()
        plan = unify_entity_registry(conn, dry_run=True)
        hit = next(v for v in plan.values()
                   if "Zorn" in (v["winner"], *v["demoted"]))
        assert hit["winner"] == "Zorn·Long", hit
        conn.close()


class TestReferencePromptAndRegistration:
    def test_prompt_separates_hard_names_from_context(self):
        from src.agents.reference_agent import ReferenceAgent

        agent = ReferenceAgent.__new__(ReferenceAgent)
        prompt = agent._build_prompt("Duke Leto looked at Paul.")
        assert "context_names" in prompt
        assert "the Duke / my Lord" in prompt
        assert "source_names: 本片段中能**唯一指向该实体**的专名" in prompt

    def test_registration_quarantines_cross_owner_proper_alias(self):
        from src.agents.reference_agent import ReferenceAgent

        conn = _mk_db()
        db = type("DB", (), {"conn": conn})()
        agent = ReferenceAgent.__new__(ReferenceAgent)
        agent.db = db
        agent._register_entities("ch01_chunk000", [
            {"canonical_name": "保罗·厄崔迪",
             "source_names": ["Paul Atreides"], "context_names": ["my Lord"],
             "pronoun_gender": "he", "entity_type": "person",
             "confidence": "high", "translation_strategy": "transliteration"},
            {"canonical_name": "弗拉基米尔·哈克南男爵",
             "source_names": ["Paul Atreides", "Baron Harkonnen"],
             "context_names": ["my Lord"], "pronoun_gender": "he",
             "entity_type": "person", "confidence": "high",
             "translation_strategy": "transliteration"},
        ])
        rows = dict(conn.execute(
            "SELECT canonical_name, source_names FROM entity_registry"
        ).fetchall())
        assert "Paul Atreides" not in json.loads(rows["保罗·厄崔迪"])
        assert "Paul Atreides" not in json.loads(rows["弗拉基米尔·哈克南男爵"])
        assert "Baron Harkonnen" in json.loads(rows["弗拉基米尔·哈克南男爵"])
        qt = dict(conn.execute(
            "SELECT canonical_name, quarantined_names FROM entity_registry"
        ).fetchall())
        assert "Paul Atreides" in json.loads(qt["保罗·厄崔迪"])
        assert "Paul Atreides" in json.loads(qt["弗拉基米尔·哈克南男爵"])
        conn.close()

    def test_registration_quarantines_single_owner_foreign_alias_and_repairs_owner(self):
        from src.agents.reference_agent import ReferenceAgent

        conn = _mk_db()
        _decision(conn, "Duke Leto", "雷托公爵")
        _row(conn, "保罗·厄崔迪", ["Paul Atreides"])
        _row(conn, "雷托公爵", ["Leto"], quarantine=["Duke Leto"])
        conn.commit()
        db = type("DB", (), {"conn": conn})()
        agent = ReferenceAgent.__new__(ReferenceAgent)
        agent.db = db

        agent._register_entities("ch01_chunk000", [
            {"canonical_name": "保罗·厄崔迪",
             "source_names": ["Paul Atreides", "Duke Leto"],
             "context_names": [], "pronoun_gender": "he",
             "entity_type": "person", "confidence": "high",
             "translation_strategy": "transliteration"},
        ])
        paul = conn.execute(
            "SELECT source_names, quarantined_names FROM entity_registry "
            "WHERE canonical_name='保罗·厄崔迪'"
        ).fetchone()
        assert "Duke Leto" not in json.loads(paul[0])
        assert "Duke Leto" in json.loads(paul[1])

        agent._register_entities("ch01_chunk001", [
            {"canonical_name": "雷托公爵",
             "source_names": ["Leto", "Duke Leto"],
             "context_names": [], "pronoun_gender": "he",
             "entity_type": "person", "confidence": "high",
             "translation_strategy": "transliteration"},
        ])
        leto = conn.execute(
            "SELECT source_names, quarantined_names FROM entity_registry "
            "WHERE canonical_name='雷托公爵'"
        ).fetchone()
        assert "Duke Leto" in json.loads(leto[0])
        assert "Duke Leto" not in json.loads(leto[1])
        conn.close()

    def test_registration_quarantines_unattested_alias(self):
        from src.agents.reference_agent import ReferenceAgent

        conn = _mk_db()
        db = type("DB", (), {"conn": conn})()
        agent = ReferenceAgent.__new__(ReferenceAgent)
        agent.db = db
        agent._register_entities(
            "ch01_chunk000",
            [{"canonical_name": "保罗·厄崔迪",
              "source_names": ["Paul Atreides", "Shaddam IV"],
              "context_names": [], "pronoun_gender": "he",
              "entity_type": "person", "confidence": "high",
              "translation_strategy": "transliteration"}],
            source_text="Paul Atreides met Baron Harkonnen.",
        )
        row = conn.execute(
            "SELECT source_names, quarantined_names FROM entity_registry "
            "WHERE canonical_name='保罗·厄崔迪'"
        ).fetchone()
        assert "Paul Atreides" in json.loads(row[0])
        assert "Shaddam IV" not in json.loads(row[0])
        assert "Shaddam IV" in json.loads(row[1])
        conn.close()
