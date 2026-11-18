"""Round 158: read-only evidence audit for outstanding term forms."""
from __future__ import annotations

import json
import sqlite3
from argparse import Namespace
from pathlib import Path

from src.term_form_audit import _completed_corpus, audit_term_forms
from src.main import _MUTATING_COMMANDS, _command_needs_lock
from src.term_form_contracts import (
    _corpus_consensus_forms,
    _embedded_source_gloss_forms,
    _gloss_extension_forms,
    _gloss_modifier_elision_forms,
    _gloss_surface_form,
    _registry_word_order_forms,
    _sibling_decision_forms,
    _single_context_common_noun_forms,
    _short_forms,
    _source_gloss_forms,
    build_term_form_contracts,
)
from src.pending_backtrack_reconcile import body_before_footnotes


def _make_fixture(root: Path) -> None:
    db = root / "db"
    db.mkdir(parents=True)
    workflow = sqlite3.connect(str(db / "workflow.db"))
    try:
        workflow.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, "
            "chapter_id TEXT, text_content TEXT, state TEXT, last_error TEXT, "
            "backtrack_rounds INTEGER, rescue_stage TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        workflow.executemany(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, "
            "state, backtrack_rounds) VALUES (?, 'ch01', ?, 'COMPLETED', 0)",
            [
                ("ch01_chunk000", "Yueh waited."),
                ("ch01_chunk001", "Yueh spoke."),
                ("ch01_chunk002", "Yueh left."),
                ("ch01_chunk003", "The enemy watched."),
                ("ch01_chunk004", "Yueh watched."),
                ("ch01_chunk010", "A Harkonnen guard waited."),
                ("ch01_chunk011", "The Harkonnen house answered."),
                ("ch01_chunk020", "Sire watched."),
            ],
        )
        workflow.commit()
    finally:
        workflow.close()

    decisions = sqlite3.connect(str(db / "decision_db.sqlite"))
    try:
        decisions.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, reason TEXT, updated_at TEXT, "
            "locked INTEGER DEFAULT 0)"
        )
        decisions.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, "
            "canonical_name TEXT, source_names TEXT, context_names TEXT, "
            "variant_names TEXT, quarantined_names TEXT, entity_type TEXT, "
            "confidence TEXT, pending_review INTEGER, review_reason TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        decisions.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)"
        )
        decisions.executemany(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "updated_at, locked) VALUES (?, 2, ?, ?, "
            "'2026-11-17 00:00:00', 0)",
            [
                (1, "Yueh", "岳医生"),
                (2, "Harkonnen", "哈克南家族"),
                (3, "Sire", "主公"),
            ],
        )
        decisions.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, ?, ?, 1, "
            "'2026-11-17 00:00:00', '2026-11-17 00:00:00')",
            [
                ("ch01_chunk002", 1, "Yueh"),
                ("ch01_chunk004", 1, "Yueh"),
                ("ch01_chunk010", 2, "Harkonnen"),
                ("ch01_chunk020", 3, "Sire"),
            ],
        )
        decisions.commit()
    finally:
        decisions.close()

    texts = {
        "ch01_chunk000": "岳医生等待着。",
        "ch01_chunk001": "岳医生开口。",
        "ch01_chunk002": "尤厄医生离开了。",
        "ch01_chunk003": "敌人注视着。",
        "ch01_chunk004": "岳注视着。",
        "ch01_chunk010": "哈克南卫兵等待着。",
        "ch01_chunk011": "哈克南家族回应了。",
        "ch01_chunk020": "阁下注视着。",
    }
    out = root / "output" / "ch01"
    out.mkdir(parents=True)
    for chunk_id, text in texts.items():
        (out / (chunk_id + "_final.json")).write_text(
            json.dumps({"text": text}, ensure_ascii=False),
            encoding="utf-8")
    (root / "input").mkdir()
    (root / "input" / "ch01.md").write_text("# One\n", encoding="utf-8")


def test_term_form_audit_classifies_stem_body_drift_and_target_missing(
        tmp_path: Path):
    _make_fixture(tmp_path)
    workflow_before = (tmp_path / "db" / "workflow.db").read_bytes()
    decisions_before = (tmp_path / "db" / "decision_db.sqlite").read_bytes()

    result = audit_term_forms(tmp_path)

    assert result["ok"] is True
    assert result["mode"] == "read_only"
    assert result["write_attempted"] is False
    assert result["counts"]["outstanding_rows"] == 3
    assert result["counts"]["groups"] == 2
    by_source = {group["source_key"]: group for group in result["groups"]}
    assert by_source["Yueh"]["classification"] == "body_drift_candidate"
    yueh_forms = {
        row["form"]
        for row in by_source["Yueh"]["pending_rows"][0]["near_target_forms"]
    }
    assert "尤厄医生" in yueh_forms
    assert by_source["Harkonnen"]["classification"] == "target_stem_contract"
    assert "Sire" not in by_source
    assert (tmp_path / "db" / "workflow.db").read_bytes() == workflow_before
    assert (tmp_path / "db" / "decision_db.sqlite").read_bytes() == decisions_before


def test_term_form_audit_cli_is_read_only():
    assert _command_needs_lock("term-form-audit", Namespace()) is False
    assert "term-form-audit" not in _MUTATING_COMMANDS
    assert _command_needs_lock("term-form-contract", Namespace()) is True
    assert "term-form-contract" in _MUTATING_COMMANDS


def test_term_form_contract_uses_profile_form_and_short_address(tmp_path: Path):
    _make_fixture(tmp_path)
    (tmp_path / "input" / "book_profile.json").write_text(json.dumps({
        "translation_hints": [
            "Harkonnen = 哈克南家族／哈克南：家族名。",
            "Yueh = 岳医生：医师。",
        ],
    }, ensure_ascii=False), encoding="utf-8")

    result = build_term_form_contracts(tmp_path)

    assert result["ok"] is True
    assert result["mode"] == "read_only"
    by_source = {row["source_key"]: row for row in result["contracts"]}
    harkonnen = by_source["Harkonnen"]
    assert "哈克南" in {form["form"] for form in harkonnen["forms"]}
    assert harkonnen["matched_chunks"] == [{"chunk_id": "ch01_chunk010",
                                            "form": "哈克南"}]
    assert harkonnen["status"] == "complete"
    assert "Sire" not in by_source
    # The Decision remains 岳医生, but the single-character surname address is
    # legal only for this role pattern and is attested across positives.
    yueh = by_source["Yueh"]
    assert "岳" in {form["form"] for form in yueh["forms"]}
    assert yueh["matched_chunks"] == [{"chunk_id": "ch01_chunk004",
                                       "form": "岳"}]


def test_corpus_consensus_rejects_self_attestation():
    corpus = {
        "ch01_chunk047": {
            "source": "Why is there no weather control?",
            "final": "为什么没有天气控制？",
        },
    }

    forms = _corpus_consensus_forms(
        ["为什么没有天气控制？"],
        "weather control",
        "气候控制",
        corpus,
        pending_chunk_ids=["ch01_chunk047"],
    )

    assert forms == []


def test_corpus_consensus_ignores_numbered_footnote_explanations():
    prose = "他望着奎萨茨·哈德拉赫。"
    footnote = "\n\n---\n\n[1] 名称含有“缩短距离”的双关。"
    text = prose + footnote
    corpus = {
        "ch01_chunk015": {"source": "Perhaps you are the Kwisatz Haderach.",
                          "final": text},
        "ch01_chunk016": {"source": "The Kwisatz Haderach comes.",
                          "final": text},
    }

    forms = _corpus_consensus_forms(
        [text], "Kwisatz Haderach", "短距者", corpus,
        pending_chunk_ids=["ch01_chunk015"],
    )

    assert body_before_footnotes(text).strip() == prose
    assert forms == []


def test_term_form_audit_corpus_strips_numbered_footnotes(tmp_path: Path):
    out = tmp_path / "output" / "ch01"
    out.mkdir(parents=True)
    (out / "ch01_chunk900_final.json").write_text(
        json.dumps({
            "text": "他望着奎萨茨·哈德拉赫。\n\n---\n\n[1] 短距者是名称双关。",
        }, ensure_ascii=False),
        encoding="utf-8")
    context = {
        "output_dir": tmp_path / "output",
        "tasks": {
            "ch01_chunk900": {
                "state": "COMPLETED",
                "chapter_id": "ch01",
                "text_content": "He looked at the Kwisatz Haderach.",
            },
        },
    }

    corpus = _completed_corpus(context)

    assert corpus["ch01_chunk900"]["final"].strip() == "他望着奎萨茨·哈德拉赫。"
    assert "短距者" not in corpus["ch01_chunk900"]["final"]


def test_corpus_consensus_prefers_independent_closest_form_over_sentence_fragment():
    corpus = {
        "ch01_chunk047": {
            "source": "Why is there no weather control?",
            "final": "为什么没有天气控制？宇航公会还会为卫星控制开价。",
        },
        "ch01_chunk048": {
            "source": "Weather control is expensive.",
            "final": "天气控制很昂贵。",
        },
    }

    forms = _corpus_consensus_forms(
        ["为什么没有天气控制？宇航公会还会为卫星控制开价。"],
        "weather control",
        "气候控制",
        corpus,
        pending_chunk_ids=["ch01_chunk047"],
    )

    assert [item["form"] for item in forms] == ["天气控制"]


def test_corpus_consensus_keeps_compound_suffix_instead_of_truncating_it():
    corpus = {
        "ch01_chunk023": {
            "source": "Duncan drew a kindjal.",
            "final": "邓肯拔出一把金贾尔匕首。",
        },
        "ch01_chunk024": {
            "source": "The kindjal was sharp.",
            "final": "贾尔匕首很锋利。",
        },
    }

    forms = _corpus_consensus_forms(
        ["邓肯拔出一把金贾尔匕首。"],
        "kindjal",
        "坎贾尔",
        corpus,
        pending_chunk_ids=["ch01_chunk023"],
    )

    assert [item["form"] for item in forms] == ["贾尔匕首"]


def test_single_context_variant_rejects_half_consensus_semantic_substitution():
    corpus = {
        "ch01_chunk001": {
            "source": "Desert power is the real power.",
            "final": "空中力量才是真正的力量。",
        },
        "ch01_chunk002": {
            "source": "Desert power rules here.",
            "final": "空中力量统治这里。",
        },
        "ch01_chunk003": {
            "source": "The desert power remains.",
            "final": "沙漠力量仍在。",
        },
        "ch01_chunk004": {
            "source": "We need desert power.",
            "final": "我们需要沙漠力量。",
        },
    }

    forms = _single_context_common_noun_forms(
        ["空中力量才是真正的力量。"],
        "desert power",
        "沙漠力量",
        corpus,
        pending_chunk_ids=["ch01_chunk001"],
    )

    assert forms == []


def test_multi_part_short_head_rejects_faction_title_fragment():
    corpus = {
        "ch01_chunk001": {"source": "Bene Gesserit witch waited.",
                          "final": "贝尼·杰瑟里特妖妇等待着。"},
        "ch01_chunk002": {"source": "The Bene Gesserit witch spoke.",
                          "final": "贝尼·杰瑟里特妖妇开口了。"},
        "ch01_chunk003": {"source": "A Bene Gesserit witch left.",
                          "final": "贝尼·杰瑟里特妖妇离开了。"},
    }

    forms = _short_forms(
        "贝尼·杰瑟里特妖妇",
        "Bene Gesserit witch",
        corpus,
    )

    assert "贝尼" not in {item["form"] for item in forms}


def test_sibling_decision_is_explicit_form_authority():
    context = {
        "decisions": {
            506: {"id": 506, "source_key": "crysknife",
                  "translation": "晶牙刀"},
            1277: {"id": 1277, "source_key": "Crysknife",
                   "translation": "晶牙匕"},
        },
    }

    forms = _sibling_decision_forms(context, "crysknife", 506)

    assert [item["form"] for item in forms] == ["晶牙匕"]
    assert forms[0]["authority"] == "sibling_decision_alias"


def test_single_context_common_noun_variant_is_lowercase_only():
    corpus = {
        "ch01_chunk047": {
            "source": "Why is there no weather control?",
            "final": "为什么没有天气控制？宇航公会还会为卫星控制开价。",
        },
    }

    common = _single_context_common_noun_forms(
        ["为什么没有天气控制？宇航公会还会为卫星控制开价。"],
        "weather control",
        "气候控制",
        corpus,
        pending_chunk_ids=["ch01_chunk047"],
    )
    proper = _single_context_common_noun_forms(
        ["马普斯仍然清醒。"],
        "Mapes",
        "马佩斯",
        {"ch01_chunk001": {
            "source": "Mapes remained awake.",
            "final": "马普斯仍然清醒。",
        }},
        pending_chunk_ids=["ch01_chunk001"],
    )

    assert [item["form"] for item in common] == ["天气控制"]
    assert common[0]["authority"] == "single_context_common_noun_variant"
    assert proper == []


def test_source_gloss_forms_are_independent_contract_authority():
    corpus = {
        "ch01_chunk000": {
            "source": "Perhaps he is the Kwisatz Haderach.",
            "final": "也许他就是短距者（Kwisatz Haderach）。",
        },
        "ch03_chunk047": {
            "source": "You believe I am the Kwisatz Haderach.",
            "final": "你以为我是奎萨茨·哈德拉赫（Kwisatz Haderach）。",
        },
        "ch03_chunk099": {
            "source": "The Kwisatz Haderach comes.",
            "final": "他从不使用其他称号。",
        },
    }

    forms = _source_gloss_forms(corpus, "Kwisatz Haderach", "短距者")

    by_form = {item["form"]: item for item in forms}
    assert "短距者" in by_form
    assert "奎萨茨·哈德拉赫" in by_form
    assert by_form["奎萨茨·哈德拉赫"]["authority"] == "source_gloss"
    assert "explicit gloss" in by_form["奎萨茨·哈德拉赫"]["evidence"]


def test_source_gloss_accepts_footnote_marker_before_closing_parenthesis():
    corpus = {
        "ch04_chunk036": {
            "source": "You have a legend of the Voice from the Outer World.",
            "final": "外世界之声（Voice from the Outer World[^7]）。",
        },
    }

    forms = _source_gloss_forms(
        corpus, "Voice from the Outer World", "外世界之声")

    assert [item["form"] for item in forms] == ["外世界之声"]
    assert forms[0]["authority"] == "source_gloss"


def test_gloss_extractor_strips_shared_verb_boundary():
    assert _gloss_surface_form("会用杂酚灌木", "木馏油灌木") == "杂酚灌木"
    assert _gloss_surface_form("如果我在沙虫活动区", "沙虫活动区") == "沙虫活动区"
    assert _gloss_surface_form("如果运载飞行器", "运载飞行器") == "运载飞行器"
    assert _gloss_surface_form("宾杜悬浮", "明点悬止") == "宾杜悬浮"
    assert _gloss_surface_form("掀起他斗篷", "阿拉伯斗篷") == "斗篷"


def test_embedded_source_gloss_accepts_phrase_ending_in_decision_key():
    corpus = {
        "ch03_chunk009": {
            "source": "Even if you are caught in the Great Erg.",
            "final": "即使被困在大沙海（Great Erg）之中。",
        },
        "ch04_chunk049": {
            "source": "She recited the Bene Gesserit litany.",
            "final": "她默念着贝尼·杰瑟里特祷文（Bene Gesserit litany）。",
        },
    }
    decisions = {
        3370: {"source_key": "erg", "translation": "厄格"},
        3531: {"source_key": "litany", "translation": "连祷文"},
    }

    erg_forms = _embedded_source_gloss_forms(
        corpus, "erg", "厄格", decisions)
    litany_forms = _embedded_source_gloss_forms(
        corpus, "litany", "连祷文", decisions)

    assert [item["form"] for item in erg_forms] == ["大沙海"]
    assert [item["form"] for item in litany_forms] == ["贝尼·杰瑟里特祷文"]
    assert erg_forms[0]["authority"] == "source_gloss_embedded_term"
    assert litany_forms[0]["authority"] == "source_gloss_embedded_term"


def test_embedded_source_gloss_rejects_non_tail_or_competing_decision():
    corpus = {
        "ch03_chunk025": {
            "source": "The carryall wing tilted.",
            "final": "运载飞行器的机翼（carryall wing）倾斜了。",
        },
        "ch04_chunk059": {
            "source": "He raised a body shield.",
            "final": "他举起人体护盾（body shield）。",
        },
    }

    assert _embedded_source_gloss_forms(
        corpus, "carryall", "运载飞行器", {}) == []
    assert _embedded_source_gloss_forms(
        corpus, "shield", "护盾",
        {3497: {"source_key": "body shield", "translation": "人体护盾"}},
    ) == []


def test_embedded_source_gloss_rejects_phrase_owned_by_specific_decision():
    corpus = {
        "ch03_chunk009": {
            "source": "Even if you are caught in the Great Erg.",
            "final": "即使被困在大沙海（Great Erg）之中。",
        },
        "ch04_chunk049": {
            "source": "She recited the Bene Gesserit litany.",
            "final": "她默念着贝尼·杰瑟里特祷文（Bene Gesserit litany）。",
        },
    }
    decisions = {
        1828: {"source_key": "Great Erg", "translation": "大厄格（Great Erg）"},
        3370: {"source_key": "erg", "translation": "厄格"},
        3463: {
            "source_key": "Bene Gesserit litany",
            "translation": "贝尼·杰瑟里特祷文",
        },
        3531: {"source_key": "litany", "translation": "连祷文"},
    }

    assert _embedded_source_gloss_forms(
        corpus, "erg", "厄格", decisions) == []
    assert _embedded_source_gloss_forms(
        corpus, "litany", "连祷文", decisions) == []


def test_source_gloss_allows_one_character_body_extension():
    corpus = {
        "ch03_chunk012": {
            "source": "Most Fremen use creosote bush juice.",
            "final": "会用杂酚灌木（creosote bush）叶片的汁液。",
        },
    }

    forms = _gloss_extension_forms(
        ["烟树、杂酚油灌木……"], "creosote bush", "木馏油灌木", corpus)

    assert [item["form"] for item in forms] == ["杂酚油灌木"]
    assert forms[0]["authority"] == "source_gloss_one_char_extension"


def test_source_gloss_allows_bounded_modifier_elision():
    corpus = {
        "ch03_chunk021": {
            "source": "If I am outside the worm zone, I stay with the ship.",
            "final": "如果我在沙虫活动区（worm zone）之外，我就留在飞船上。",
        },
    }

    forms = _gloss_modifier_elision_forms(
        ["在沙虫区启动护盾，你就死定了。"], "worm zone", "沙虫活动区", corpus)

    assert [item["form"] for item in forms] == ["沙虫区"]
    assert forms[0]["authority"] == "source_gloss_modifier_elision"


def test_registry_word_order_variant_requires_active_owner():
    key = "reverend mother gaius helen mohiam"
    context = {
        "registry_form_owners": {
            key: [{
                "canonical": "盖乌斯·海伦·莫希阿姆",
                "variants": ["圣母盖乌斯·海伦·莫希阿姆"],
                "pending_review": 0,
            }],
        },
    }

    forms = _registry_word_order_forms(
        context, "Reverend Mother Gaius Helen Mohiam",
        "盖乌斯·海伦·莫希阿姆圣母")

    assert [item["form"] for item in forms] == ["圣母盖乌斯·海伦·莫希阿姆"]
    assert forms[0]["authority"] == "registry_active_word_order_variant"
