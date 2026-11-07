"""Round 131: deterministic story graph coverage, collisions, and versioning."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.story_graph import build_story_graph


def _make_project(tmp_path: Path) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "db").mkdir()
    (tmp_path / "input" / "ch01.md").write_text(
        "Paul met Jessica. Paul left.", encoding="utf-8")
    (tmp_path / "input" / "ch02.md").write_text("Jessica stayed alone.", encoding="utf-8")
    con = sqlite3.connect(tmp_path / "db" / "decision_db.sqlite")
    con.execute("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY, canonical_name TEXT, source_names TEXT,
            variant_names TEXT, entity_type TEXT, pronoun_gender TEXT,
            confidence REAL, pending_review INTEGER, translation_strategy TEXT,
            first_seen_chunk TEXT
        )
    """)
    con.execute("INSERT INTO entity_registry VALUES "
                "(1,'保罗','[\"Paul\"]','[]','person','male',0.9,0,'literal','ch01')")
    con.execute("INSERT INTO entity_registry VALUES "
                "(2,'杰西卡','[\"Jessica\"]','[]','person','female',0.9,0,'literal','ch01')")
    con.execute("INSERT INTO entity_registry VALUES "
                "(3,'缺席者','[\"Absent\"]','[]','person','',0.9,0,'literal','ch03')")
    con.commit()
    con.close()
    return tmp_path


def test_story_graph_reports_uncovered_entities_and_chapters(tmp_path: Path):
    root = _make_project(tmp_path)
    graph = build_story_graph(root)
    cov = graph["coverage"]
    assert cov["entities_total"] == 3
    assert cov["uncovered_entities"] == ["缺席者"]
    assert cov["uncovered_chapters"] == []
    assert cov["answerable"] is True
    assert cov["partial_coverage"] is True
    strict = build_story_graph(root, entity_coverage_min=0.9)["coverage"]
    assert strict["answerable"] is False
    assert "trusted_entity_coverage_below_threshold" in strict["answerable_reasons"]
    gaps = {g["code"]: g for g in graph["knowledge_gaps"]}
    assert gaps["uncovered_entities"]["count"] == 1
    assert gaps["uncovered_chapters"]["count"] == 0
    assert gaps["uncertified_entities"]["count"] == 0


def test_story_graph_partial_coverage_includes_uncovered_chapters(tmp_path: Path):
    root = _make_project(tmp_path)
    # Make ch02 contain no registered entity; the graph still covers ch01.
    (root / "input" / "ch02.md").write_text("A quiet room with no names.", encoding="utf-8")
    graph = build_story_graph(root)
    cov = graph["coverage"]
    assert cov["uncovered_chapters"] == ["ch02"]
    assert cov["chapter_coverage"] < 1.0
    assert cov["partial_coverage"] is True


def test_story_graph_fingerprint_tracks_registry_content_not_time(tmp_path: Path):
    root = _make_project(tmp_path)
    first = build_story_graph(root)
    second = build_story_graph(root)
    assert first["fingerprint"] == second["fingerprint"]

    con = sqlite3.connect(root / "db" / "decision_db.sqlite")
    con.execute("INSERT INTO entity_registry VALUES "
                "(4,'新实体','[\"New One\"]','[]','person','',0.9,0,'literal','ch02')")
    con.commit()
    con.close()
    changed = build_story_graph(root)
    assert changed["fingerprint"] != first["fingerprint"]
    assert changed["generation"]["inputs"]["decision_db_hash"] != \
        first["generation"]["inputs"]["decision_db_hash"]


def test_story_graph_never_guesses_ambiguous_alias(tmp_path: Path):
    root = _make_project(tmp_path)
    con = sqlite3.connect(root / "db" / "decision_db.sqlite")
    con.execute("INSERT INTO entity_registry VALUES "
                "(4,'另一人','[\"Paul\"]','[]','person','male',0.9,0,'literal','ch02')")
    con.commit()
    con.close()
    graph = build_story_graph(root)
    diag = graph["diagnostics"]
    assert diag["unresolved_alias_count"] == 1
    assert diag["unresolved_aliases"][0]["alias"] == "paul"
    # Neither entity may claim the ambiguous surface form.
    appears = [e for e in graph["edges"] if e["type"] == "appears_in"]
    assert all(e["source"] not in ("entity:entity-1", "entity:entity-4") for e in appears)
    alias_edges = [e for e in graph["edges"]
                   if e["type"] == "alias_of" and e.get("ambiguous")]
    assert len(alias_edges) == 2
    assert all(e["certification"] == "pending" for e in alias_edges)


def test_story_graph_rolls_entity_certification_into_alias_and_fact_edges(tmp_path: Path):
    root = _make_project(tmp_path)
    con = sqlite3.connect(root / "db" / "decision_db.sqlite")
    # Canonical owner wins the shared surface form; the other owner keeps an
    # audit edge but it must not remain auto_verified.
    con.execute("INSERT INTO entity_registry VALUES "
                "(4,'另一人','[\"Paul\"]','[]','person','male',0.9,0,'literal','ch02')")
    # A low-confidence entity still appears in the text; the deterministic
    # occurrence fact exists, but it is pending until the identity is trusted.
    con.execute("INSERT INTO entity_registry VALUES "
                "(5,'斯提尔格','[\"Stilgar\"]','[]','person','male',0.4,1,'literal','ch02')")
    con.execute("INSERT INTO entity_registry VALUES "
                "(6,'Paul','[\"Paul\"]','[]','person','male',0.9,0,'literal','ch01')")
    con.commit()
    con.close()
    (root / "input" / "ch02.md").write_text(
        "Jessica stayed alone. Stilgar watched Jessica.", encoding="utf-8")

    graph = build_story_graph(root)
    alias_edges = [e for e in graph["edges"] if e["type"] == "alias_of"
                   and e.get("target") in ("entity:entity-2", "entity:entity-4")]
    by_target = {e["target"]: e for e in alias_edges}
    # A unique alias inherits the trusted entity's certification.
    assert by_target["entity:entity-2"]["certification"] == "auto_verified"
    # The losing side of a canonical-owner collision remains visible as an audit
    # edge, but it cannot claim its rejected mapping was verified.
    assert by_target["entity:entity-4"]["certification"] == "pending"
    assert by_target["entity:entity-4"]["ambiguous"] is True
    assert by_target["entity:entity-4"]["collision_owner"] == "entity:entity-6"

    stilgar = next(e for e in graph["edges"]
                   if e["type"] == "appears_in" and e["source"] == "entity:entity-5")
    assert stilgar["certification"] == "pending"
    assert stilgar["entity_certification"] == "pending"

    cooccurs = [e for e in graph["edges"] if e["type"] == "co_occurs_with"
                and "entity:entity-5" in (e["source"], e["target"])]
    assert cooccurs
    assert all(e["certification"] == "pending" for e in cooccurs)


def test_story_graph_reads_text_confidence_and_live_chapter_context(tmp_path: Path):
    """Production registry stores high/medium/low, not floats.

    The first implementation called float("high") and silently marked every
    entity pending; ``story_graph_context_for_chapter`` then filtered every
    occurrence edge out.  This test uses the real database vocabulary and the
    same trusted-entity rule as ``query_high_confidence_entities``.
    """
    root = _make_project(tmp_path)
    con = sqlite3.connect(root / "db" / "decision_db.sqlite")
    con.execute("UPDATE entity_registry SET confidence='high', pending_review=0 WHERE id=1")
    con.execute("UPDATE entity_registry SET confidence='medium', pending_review=1 WHERE id=2")
    con.execute("UPDATE entity_registry SET confidence='high', pending_review=0 WHERE id=3")
    con.commit()
    con.close()

    graph = build_story_graph(root)
    nodes = {n["id"]: n for n in graph["nodes"] if n["type"] == "entity"}
    assert nodes["entity:entity-1"]["attributes"]["confidence"] == 0.9
    assert nodes["entity:entity-1"]["attributes"]["confidence_label"] == "high"
    assert nodes["entity:entity-1"]["attributes"]["certification"] == "auto_verified"
    assert nodes["entity:entity-2"]["attributes"]["confidence"] == 0.6
    assert nodes["entity:entity-2"]["attributes"]["confidence_label"] == "medium"
    assert nodes["entity:entity-2"]["attributes"]["certification"] == "pending"

    cov = graph["coverage"]
    assert cov["certified_entities_total"] == 2
    assert cov["certified_entities_with_appearance"] == 1
    assert cov["certified_entity_coverage"] > 0
    assert cov["certified_entity_coverage_of_trusted"] == 0.5
    assert cov["answerable"] is True
    assert cov["uncertified_entities"] == ["杰西卡"]
    assert cov["chapter_certified_entity_coverage"]["ch01"] > 0

    from src.story_graph import story_graph_context_for_chapter
    context = story_graph_context_for_chapter(graph, "ch01")
    assert "保罗" in context
    assert "杰西卡" not in context


def test_story_graph_answerable_uses_trusted_coverage_not_noisy_rows(tmp_path: Path):
    """Pending/duplicate registry rows must not block a healthy trusted index."""
    root = _make_project(tmp_path)
    con = sqlite3.connect(root / "db" / "decision_db.sqlite")
    for row_id in range(10, 20):
        con.execute(
            "INSERT INTO entity_registry VALUES (?,?,?,?,?,?,?,?,?,?)",
            (row_id, f"噪声实体{row_id}", "[]", "[]", "noise", "", "medium", 1,
             "literal", "ch03"),
        )
    con.commit()
    con.close()

    graph = build_story_graph(root)
    cov = graph["coverage"]
    assert cov["entities_total"] == 13
    assert cov["entity_coverage"] < 0.5
    assert cov["certified_entity_coverage_of_trusted"] > 0.5
    assert cov["answerable"] is True
    assert cov["answerable_threshold_basis"] == "certified_entity_coverage_of_trusted"
    assert len(cov["uncertified_entity_ids"]) == 10
    gaps = {g["code"]: g for g in graph["knowledge_gaps"]}
    assert gaps["uncertified_entities"]["count"] == 10
    assert gaps["uncertified_entities"]["sample"]


def test_story_graph_narrative_edges_keep_evidence_and_chapter_coverage(tmp_path: Path):
    root = _make_project(tmp_path)
    profile = {
        "schema_version": 2,
        "setup_payoff": [{
            "id": "payoff-1",
            "type": "setup_payoff",
            "chapters": ["ch01", "ch02"],
            "claim": "开场的见面在尾章留下回声。",
            "confidence": 0.9,
            "status": "active",
            "l2_verdict": "confirmed",
            "evidence": [
                {"chapter": "ch01", "quote": "Paul met Jessica.",
                 "offset": 0, "match_kind": "exact", "match_score": 1.0},
                {"chapter": "ch02", "quote": "Jessica stayed alone.",
                 "offset": 0, "match_kind": "exact", "match_score": 1.0},
            ],
        }],
    }

    graph = build_story_graph(root, profile=profile)
    note = next(n for n in graph["nodes"] if n["id"] == "note:payoff-1")
    setup = next(e for e in graph["edges"] if e["type"] == "sets_up")

    assert note["certification"] == "l1_l2_verified"
    assert setup["certification"] == "l1_l2_verified"
    evidence = setup["provenance"][0]["evidence"]
    assert [e["chapter"] for e in evidence] == ["ch01", "ch02"]
    assert evidence[0]["quote"] == "Paul met Jessica."
    assert evidence[1]["offset"] == 0

    cov = graph["coverage"]
    assert cov["narrative_chapter_coverage"] == 1.0
    assert cov["uncovered_narrative_chapters"] == []
    assert cov["narrative_answerable_for"] == "validated_note_index"
    assert cov["narrative_partial_coverage"] is False


def test_story_graph_narrative_partial_coverage_uses_chapter_holes(tmp_path: Path):
    root = _make_project(tmp_path)
    profile = {
        "schema_version": 2,
        "voice_notes": [{
            "id": "voice-1", "type": "voice", "chapters": ["ch01"],
            "claim": "开篇语气保持冷峻。", "confidence": 0.9,
            "status": "active", "l2_verdict": "confirmed",
            "evidence": [{"chapter": "ch01", "quote": "Paul met Jessica.",
                          "offset": 0, "match_kind": "exact"}],
        }],
    }

    cov = build_story_graph(root, profile=profile)["coverage"]
    assert cov["narrative_note_coverage"] == 1.0
    assert cov["narrative_chapter_coverage"] == 0.5
    assert cov["uncovered_narrative_chapters"] == ["ch02"]
    assert cov["narrative_partial_coverage"] is True


def test_monolith_contains_round131_runtime_modules():
    root = Path(__file__).resolve().parent.parent
    text = (root / "src" / "translator_agent.py").read_text(encoding="utf-8")
    for marker in ("class JudgmentContext", "def build_story_graph",
                   "def build_deep_profile", "def apply_review_fixes",
                   "REVIEW_FIX_SCHEMA_VERSION"):
        assert marker in text, f"monolith 缺少 Round 131 符号: {marker}"


def test_monolith_keeps_real_round131_modules_in_repo_mode():
    """The shim must not replace real modules and hide private helpers.

    Regression: pun_audit imports style_profiler._split_anchors.  If importing
    the monolith first left a fake src.style_profiler in sys.modules, a later
    import in the same interpreter lost that helper.
    """
    import src.translator_agent  # noqa: F401 - installs repository-mode shims
    import src.style_profiler as sp

    assert getattr(sp, "__file__", None), "repo mode must keep the real module"
    assert hasattr(sp, "_split_anchors")
    assert sp._split_anchors("draw = 拉/画，抽签") == ["draw", "拉", "画", "抽签"]
