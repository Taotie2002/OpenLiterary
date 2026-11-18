"""Round 158: locked-target authority audit is evidence-only."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from argparse import Namespace
from pathlib import Path

from src.locked_target_authority_audit import (
    audit_locked_target_authorities,
    evaluate_decision_lock_authority,
)
from src.main import _MUTATING_COMMANDS, _command_needs_lock
from src.term_form_audit import _cjk_ngrams


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fixture(root: Path) -> Path:
    workflow = root / "db" / "workflow.db"
    workflow.parent.mkdir(parents=True)
    con = sqlite3.connect(str(workflow))
    try:
        con.execute(
            "CREATE TABLE chunk_tasks (chunk_id TEXT PRIMARY KEY, "
            "chapter_id TEXT, text_content TEXT, state TEXT, last_error TEXT, "
            "backtrack_rounds INTEGER, rescue_stage TEXT, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        rows = [
            ("ch01_chunk001", "ch01", "The shield held.", "COMPLETED"),
            ("ch01_chunk002", "ch01", "Another shield rose.", "COMPLETED"),
            ("ch01_chunk003", "ch01", "A third shield fell.", "COMPLETED"),
            ("ch01_chunk004", "ch01", "Nothing here.", "COMPLETED"),
            ("ch01_chunk010", "ch01", "Beyond the Shield Wall.", "COMPLETED"),
            ("ch01_chunk011", "ch01", "Beyond the Shield Wall.", "COMPLETED"),
            ("ch01_chunk012", "ch01", "Beyond the Shield Wall.", "COMPLETED"),
            ("ch02_chunk020", "ch02",
             "Reverend Mother Gaius Helen Mohiam spoke.", "COMPLETED"),
            ("ch02_chunk021", "ch02",
             "Reverend Mother Gaius Helen Mohiam spoke.", "COMPLETED"),
            ("ch02_chunk022", "ch02",
             "Reverend Mother Gaius Helen Mohiam spoke.", "COMPLETED"),
            ("ch03_chunk030", "ch03", "The Judge of the Change spoke.", "COMPLETED"),
            ("ch03_chunk031", "ch03", "The Judge of the Change spoke.", "COMPLETED"),
            ("ch03_chunk032", "ch03", "The Judge of the Change spoke.", "COMPLETED"),
            ("ch03_chunk033", "ch03", "The Judge of the Change spoke.", "COMPLETED"),
            ("ch03_chunk040", "ch03", "A carryall landed.", "COMPLETED"),
            ("ch03_chunk041", "ch03", "A carryall landed.", "COMPLETED"),
            ("ch03_chunk042", "ch03", "A carryall landed.", "COMPLETED"),
            ("ch03_chunk043", "ch03", "A carryall landed.", "COMPLETED"),
            ("ch04_chunk050", "ch04", "A thumper sounded.", "COMPLETED"),
            ("ch04_chunk051", "ch04", "The stillsuit held.", "COMPLETED"),
        ]
        con.executemany(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state, "
            "backtrack_rounds) VALUES (?, ?, ?, ?, 0)", rows)
        con.commit()
    finally:
        con.close()

    decision = root / "db" / "decision_db.sqlite"
    con = sqlite3.connect(str(decision))
    try:
        con.execute(
            "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER, "
            "source_key TEXT, translation TEXT, reason TEXT, updated_at TEXT, "
            "locked INTEGER DEFAULT 0)")
        con.execute(
            "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, "
            "canonical_name TEXT, source_names TEXT, confidence TEXT, "
            "pending_review INTEGER DEFAULT 0)")
        con.execute(
            "CREATE TABLE pending_backtrack (chunk_id TEXT PRIMARY KEY, "
            "decision_id INTEGER, source_key TEXT, queue_rounds INTEGER, "
            "created_at TEXT, updated_at TEXT)")
        decisions = [
            (315, "shield", "屏蔽场", "", 1),
            (1103, "Shield Wall", "屏蔽墙",
             "Round 158 confirmed term drift lock: 屏蔽墙", 1),
            (478, "Judge of the Change", "变革仲裁官", "", 1),
            (1235, "carryall", "运载飞行器", "", 1),
            (700, "Reverend Mother Gaius Helen Mohiam",
             "盖乌斯·海伦·莫希阿姆圣母", "", 1),
            (1300, "thumper", "捶击器",
             "Round 158 confirmed term drift lock: 捶击器", 1),
            (1301, "stillsuit", "蒸馏服", "", 1),
        ]
        con.executemany(
            "INSERT INTO decision_db (id, level, source_key, translation, "
            "reason, updated_at, locked) VALUES (?, 2, ?, ?, ?, "
            "'2026-11-18 00:00:00', ?)", decisions)
        con.executemany(
            "INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, "
            "queue_rounds, created_at, updated_at) VALUES (?, ?, ?, 1, "
            "'2026-11-18 00:00:00', '2026-11-18 00:00:00')",
            [
                ("ch01_chunk001", 315, "shield"),
                ("ch01_chunk002", 315, "shield"),
                ("ch01_chunk003", 315, "shield"),
                ("ch01_chunk010", 315, "shield"),
                ("ch02_chunk020", 700, "Reverend Mother Gaius Helen Mohiam"),
                ("ch02_chunk021", 700, "Reverend Mother Gaius Helen Mohiam"),
                ("ch02_chunk022", 700, "Reverend Mother Gaius Helen Mohiam"),
                ("ch03_chunk030", 478, "Judge of the Change"),
                ("ch03_chunk040", 1235, "carryall"),
            ])
        con.commit()
    finally:
        con.close()

    finals = {
        "ch01_chunk001": "护盾撑住了。",
        "ch01_chunk002": "护盾升了起来。",
        "ch01_chunk003": "护盾落了下去。",
        "ch01_chunk004": "这里什么也没有。",
        "ch01_chunk010": "盾墙之外。",
        "ch01_chunk011": "屏蔽墙之外。",
        "ch01_chunk012": "屏蔽墙之外。",
        "ch02_chunk020": "圣母盖乌斯·海伦·莫希阿姆开口。",
        "ch02_chunk021": "圣母盖乌斯·海伦·莫希阿姆说道。",
        "ch02_chunk022": "盖乌斯·海伦·莫希阿姆开口。",
        "ch03_chunk030": "法官开口。",
        "ch03_chunk031": "变革法官开口。",
        "ch03_chunk032": "变革仲裁官开口。",
        "ch03_chunk033": "变革法官开口。",
        "ch03_chunk040": "运输机降落了。",
        "ch03_chunk041": "运输机降落了。",
        "ch03_chunk042": "运输机降落了。",
        "ch03_chunk043": "运载飞行器降落了。",
        "ch04_chunk050": "发声器响了。",
        "ch04_chunk051": "防护服撑住了。",
    }
    for chunk_id, text in finals.items():
        chapter = chunk_id[:4]
        path = root / "output" / chapter / f"{chunk_id}_final.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}, ensure_ascii=False),
                        encoding="utf-8")
    input_dir = root / "input"
    input_dir.mkdir()
    for chapter in ("ch01", "ch02", "ch03", "ch04"):
        (input_dir / f"{chapter}.md").write_text("# Chapter\n", encoding="utf-8")
    return decision


def test_locked_target_audit_classifies_corpus_authority(tmp_path: Path):
    decision_db = _make_fixture(tmp_path)
    before = _digest(decision_db)

    result = audit_locked_target_authorities(tmp_path, decision_db=decision_db)

    assert _digest(decision_db) == before
    assert result["mode"] == "read_only"
    assert result["write_attempted"] is False
    assert result["ok"] is True
    assert result["counts"]["pending_rows"] == 9
    assert result["counts"]["groups"] == 5

    groups = {
        (group["decision_id"], group["authority_kind"]): group
        for group in result["groups"]
    }
    shield = groups[(315, "generic")]
    assert shield["classification"] == "contradicted"
    assert shield["target_hits"] == 0
    assert shield["authority_chunks"] == 3
    assert shield["candidates"][0]["form"] == "护盾"
    assert shield["candidates"][0]["positive_chunks"] == 3

    shield_wall = groups[(1103, "specific")]
    assert shield_wall["classification"] == "confirmed"
    assert shield_wall["pending_decision_id"] == 315
    assert shield_wall["target_hits"] == 2
    assert shield_wall["authority_chunks"] == 3

    reverend = groups[(700, "generic")]
    assert reverend["classification"] == "legitimate_variant"
    assert reverend["target_hits"] == 0

    assert groups[(478, "generic")]["classification"] == "ambiguous"
    assert groups[(1235, "generic")]["classification"] == "ambiguous"
    assert Path(result["json_path"]).exists()
    assert Path(result["md_path"]).exists()


def test_locked_target_audit_is_not_a_mutating_command():
    assert "locked-target-authority-audit" not in _MUTATING_COMMANDS
    assert _command_needs_lock(
        "locked-target-authority-audit", Namespace()) is False


def test_repair_sweep_only_reads_marked_decisions(tmp_path: Path):
    decision_db = _make_fixture(tmp_path)

    result = audit_locked_target_authorities(
        tmp_path, decision_db=decision_db,
        include_marked_repair_sweep=True,
    )

    swept_sources = {
        str(group.get("source_key")) for group in result["groups"]
        if group.get("sweep")
    }
    assert swept_sources == {"thumper"}


def _evaluate_fixture(
    finals: dict,
    *,
    target: str,
    source_key: str = "shield",
    proposed_chunk_id: str = "",
    proposed_text: str = "",
    negative_chunk_ids=(),
):
    decisions = {
        315: {
            "id": 315,
            "level": 2,
            "source_key": source_key,
            "translation": target,
        },
    }
    source_positive = (
        "The shield held." if source_key == "shield"
        else "Reverend Mother Gaius Helen Mohiam spoke.")
    context = {"tasks": {}, "decisions": decisions}
    for chunk_id in finals:
        context["tasks"][chunk_id] = {
            "chunk_id": chunk_id,
            "chapter_id": chunk_id[:4],
            "text_content": (
                "Nothing happened." if chunk_id in negative_chunk_ids
                else source_positive),
            "state": "COMPLETED",
        }
    if proposed_chunk_id and proposed_chunk_id not in context["tasks"]:
        context["tasks"][proposed_chunk_id] = {
            "chunk_id": proposed_chunk_id,
            "chapter_id": proposed_chunk_id[:4],
            "text_content": source_positive,
            "state": "DIRTY",
        }
    corpus = {
        chunk_id: {
            "chapter": chunk_id[:4],
            "source": context["tasks"][chunk_id]["text_content"],
            "final": text,
        }
        for chunk_id, text in finals.items()
    }
    grams = {chunk_id: _cjk_ngrams(text) for chunk_id, text in finals.items()}
    return evaluate_decision_lock_authority(
        decision_id=315,
        source_key=source_key,
        target=target,
        context=context,
        corpus=corpus,
        gram_cache=grams,
        proposed_chunk_id=proposed_chunk_id,
        proposed_text=proposed_text,
    )


def test_lock_authority_contradicted_generic_target_blocks():
    result = _evaluate_fixture(
        {
            "ch01_chunk001": "护盾撑住了。",
            "ch01_chunk002": "护盾升起来。",
            "ch01_chunk003": "护盾落下去。",
            "ch01_chunk009": "风声掠过了。",
        },
        target="屏蔽力场",
        proposed_chunk_id="ch01_chunk000",
        proposed_text="屏蔽力场在挡光。",
        negative_chunk_ids=("ch01_chunk009",),
    )

    assert result["status"] == "contradicted"
    assert result["lock_allowed"] is False
    assert result["target_hits"] == 1
    assert result["book_target_hits"] == 0
    assert result["proposed_target_hit"] is True
    assert result["candidates"][0]["form"] == "护盾"
    assert result["candidates"][0]["positive_chunks"] == 3


def test_lock_authority_book_majority_confirms_target():
    result = _evaluate_fixture(
        {
            "ch01_chunk001": "屏蔽力场撑住了。",
            "ch01_chunk002": "屏蔽力场升起来。",
            "ch01_chunk003": "屏蔽力场落下去。",
            "ch01_chunk009": "风声掠过了。",
        },
        target="屏蔽力场",
        proposed_chunk_id="ch01_chunk000",
        proposed_text="屏蔽力场在挡光。",
        negative_chunk_ids=("ch01_chunk009",),
    )

    assert result["status"] == "confirmed"
    assert result["lock_allowed"] is True
    assert result["target_hits"] == 4
    assert result["book_target_hits"] == 3


def test_lock_authority_low_corpus_allows_proposed_authority():
    result = _evaluate_fixture(
        {},
        target="屏蔽力场",
        proposed_chunk_id="ch01_chunk000",
        proposed_text="屏蔽力场在挡光。",
    )

    assert result["status"] == "confirmed"
    assert result["reason"] == "low_corpus_target_confirmed_by_proposed_final"
    assert result["lock_allowed"] is True
    assert result["book_authority_chunks"] == 0


def test_lock_authority_ambiguous_split_blocks():
    result = _evaluate_fixture(
        {
            "ch01_chunk001": "屏蔽力场撑住了。",
            "ch01_chunk002": "屏蔽力场升起来。",
            "ch01_chunk003": "屏蔽力墙落下去。",
            "ch01_chunk004": "屏蔽力墙挡住了光。",
            "ch01_chunk005": "屏蔽力墙在震动。",
            "ch01_chunk009": "风声掠过了。",
        },
        target="屏蔽力场",
        negative_chunk_ids=("ch01_chunk009",),
    )

    assert result["status"] == "ambiguous"
    assert result["lock_allowed"] is False


def test_lock_authority_legitimate_word_order_variant_requires_contract():
    result = _evaluate_fixture(
        {
            "ch02_chunk020": "圣母盖乌斯·海伦·莫希阿姆开口。",
            "ch02_chunk021": "圣母盖乌斯·海伦·莫希阿姆说道。",
            "ch02_chunk022": "圣母盖乌斯·海伦·莫希阿姆沉默。",
            "ch02_chunk029": "无人开口。",
        },
        target="盖乌斯·海伦·莫希阿姆圣母",
        source_key="Reverend Mother Gaius Helen Mohiam",
        negative_chunk_ids=("ch02_chunk029",),
    )

    assert result["status"] == "legitimate_variant"
    assert result["lock_allowed"] is False


def test_lock_authority_counts_proposed_final_in_candidate_grams():
    finals = {
        "ch01_chunk001": "屏蔽墙。",
        "ch01_chunk002": "风声掠过了。",
    }
    source_key = "shield"
    target = "屏蔽力场"
    tasks = {
        chunk_id: {
            "chunk_id": chunk_id,
            "chapter_id": chunk_id[:4],
            "text_content": (
                "Nothing happened." if chunk_id == "ch01_chunk002"
                else "The shield held."),
            "state": "COMPLETED",
        }
        for chunk_id in finals
    }
    tasks["ch01_chunk000"] = {
        "chunk_id": "ch01_chunk000",
        "chapter_id": "ch01",
        "text_content": "The shield held.",
        "state": "DIRTY",
    }
    decisions = {
        315: {"id": 315, "source_key": source_key, "translation": target},
    }
    corpus = {
        chunk_id: {
            "chapter": chunk_id[:4],
            "source": tasks[chunk_id]["text_content"],
            "final": text,
        }
        for chunk_id, text in finals.items()
    }
    original_grams = {
        chunk_id: _cjk_ngrams(text) for chunk_id, text in finals.items()
    }

    result = evaluate_decision_lock_authority(
        decision_id=315,
        source_key=source_key,
        target=target,
        context={"tasks": tasks, "decisions": decisions},
        corpus=corpus,
        gram_cache=original_grams,
        proposed_chunk_id="ch01_chunk000",
        proposed_text="屏蔽墙。",
    )

    assert "ch01_chunk000" not in original_grams
    assert result["status"] == "contradicted"
    assert result["book_target_hits"] == 0
    assert result["candidates"][0]["form"] == "屏蔽墙"
    assert result["candidates"][0]["positive_chunks"] == 2
