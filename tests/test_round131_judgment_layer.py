"""Round 131: judgment artifact provenance, caps, and deep-merge fallbacks."""
from __future__ import annotations

import json
from pathlib import Path

from src import style_profiler as sp
from src.judgment_layer import (
    active_notes,
    build_judgment_context,
    enforce_note_limits,
    map_quote_to_chunks,
    match_quote,
    normalize_note,
    restrict_notes_to_candidate_quotes,
    validate_note_evidence,
    verify_notes_l2,
)
from src.style_profiler import load_profile, save_profile


def _note(nid: str, ntype: str, chapters, claim: str, quote: str,
          confidence: float = 0.8, status: str = "active") -> dict:
    return {
        "id": nid, "type": ntype, "chapters": list(chapters), "claim": claim,
        "evidence": [{"chapter": chapters[0], "quote": quote}],
        "confidence": confidence, "status": status,
    }


def test_normalize_note_uses_chapter_span_for_evidence_without_chapter():
    note = normalize_note({
        "type": "voice", "chapters": ["ch01"],
        "claim": "叙事声音保持克制。",
        "evidence": [{"quote": "The sleeper must awaken."}],
    })
    assert note is not None
    assert note["evidence"][0]["chapter"] == "ch01"


def test_enforce_note_limits_does_not_count_audit_only_notes():
    notes = [_note(f"voice-{i}", "voice", ["ch01"], f"claim {i}", "quote",
                   confidence=0.9 - i * 0.01) for i in range(2)]
    notes.append(_note("dropped-prescriptive", "voice", ["ch01"], "claim d",
                       "quote", status="dropped"))
    out = enforce_note_limits(notes)["notes"]
    active = [n for n in out if n["status"] == "active"]
    assert len(active) == 2
    assert any(n["id"] == "dropped-prescriptive" for n in out)


def test_merge_quotes_must_come_from_extraction_candidates():
    candidates = [_note("voice-c", "voice", ["ch01"], "candidate", "Real source quote.")]
    merged = [_note("voice-m", "voice", ["ch01"], "invented", "A real quote elsewhere.")]
    out = restrict_notes_to_candidate_quotes(merged, candidates)
    assert out[0]["status"] == "dropped"
    assert out[0]["schema_reject_reason"] == "merge_quote_not_from_extraction_candidates"


def test_l1_matching_uses_boundaries_and_normalized_quotes():
    text = "The  “hero” said run. The sleeper must awaken."

    # ASCII substring collisions must not count as evidence.
    assert match_quote(text, "he")["ok"] is False

    normalized = match_quote(text, 'The "hero" said run.')
    assert normalized["ok"] is True
    assert normalized["match_kind"] == "normalized_exact"

    fuzzy = match_quote(text, "The sleeper must awaken!")
    assert fuzzy["ok"] is True
    assert fuzzy["match_kind"] == "fuzzy"


def test_l1_keeps_unpaired_setup_dormant_and_drops_active_single_chapter():
    quote = "The sleeper must awaken."
    text = f"A beginning. {quote}"
    dormant = _note("setup-unpaired", "setup_payoff", ["ch01"],
                    "开篇埋线，尚未找到回收。", quote, status="dormant")
    dormant["pairing_status"] = "unpaired"
    active = _note("setup-active-single", "setup_payoff", ["ch01"],
                   "自称已成立的伏笔。", quote, status="active")

    out = validate_note_evidence([dormant, active], {"ch01": text})
    by_id = {n["id"]: n for n in out["notes"]}

    assert by_id["setup-unpaired"]["status"] == "dormant"
    assert by_id["setup-unpaired"]["l1_note"] == "l1:setup_payoff_unpaired"
    assert by_id["setup-unpaired"]["evidence"][0]["match_kind"] in (
        "exact", "normalized_exact", "fuzzy")
    assert by_id["setup-active-single"]["status"] == "dropped"
    assert by_id["setup-active-single"]["l1_reject_reason"] == \
        "l1:setup_payoff_requires_two_chapters"


def test_unpaired_setup_l1_note_survives_profile_round_trip(tmp_path: Path):
    quote = "The sleeper must awaken."
    dormant = _note("setup-unpaired-persist", "setup_payoff", ["ch01"],
                    "开篇埋线，尚未找到回收。", quote, status="dormant")
    dormant["pairing_status"] = "unpaired"
    out = validate_note_evidence([dormant], {"ch01": f"A beginning. {quote}"})
    profile = {"schema_version": 2, "setup_payoff": out["notes"]}
    path = tmp_path / "book_profile.json"

    save_profile(profile, path)
    loaded = load_profile(path)

    note = loaded["setup_payoff"][0]
    assert note["status"] == "dormant"
    assert note["pairing_status"] == "unpaired"
    assert note["l1_note"] == "l1:setup_payoff_unpaired"


def test_review_quote_mapping_rejects_cross_chunk_ambiguity():
    chunks = {"ch01_c1": "同一锚句。", "ch01_c2": "同一锚句。"}
    result = map_quote_to_chunks(chunks, "同一锚句。")
    assert result["ok"] is False
    assert result["reason"] == "ambiguous_across_chunks"
    assert {h["chunk_id"] for h in result["hits"]} == {"ch01_c1", "ch01_c2"}


def test_l2_rejects_prescriptive_terminology_even_if_confirmed():
    notes = [
        _note("voice-ok", "voice", ["ch01"], "叙述保持冷峻。", "cold text"),
        _note("term-bad", "voice", ["ch01"], "术语必须统一为冷峻。", "cold text"),
    ]

    def gen(prompt):
        return json.dumps({"decisions": [
            {"id": "voice-ok", "verdict": "confirmed", "reason": "supported"},
            {"id": "term-bad", "verdict": "confirmed", "reason": "supported"},
        ]})

    out = verify_notes_l2(notes, {"ch01": "cold text"}, llm_generate=gen)
    by_id = {n["id"]: n for n in out["notes"]}
    assert by_id["voice-ok"]["status"] == "active"
    assert by_id["voice-ok"]["l2_verdict"] == "confirmed"
    assert by_id["term-bad"]["status"] == "dormant"
    assert by_id["term-bad"]["l2_verdict"] == "rejected"
    assert by_id["term-bad"]["l2_reject_reason"] == "prescriptive_terminology_claim"


def test_active_status_without_l1_l2_provenance_is_not_injected():
    prof = {"schema_version": 2, "voice_notes": [
        _note("voice-unvalidated", "voice", ["ch01"], "未经校验的注记", "quote")
    ]}
    assert active_notes(prof) == []
    ctx = build_judgment_context(prof, "ch01", chunk_text="quote", budget_chars=300)
    assert ctx.block == ""


def test_chunk_anchor_outranks_unanchored_higher_confidence_note():
    voice = _note("voice-high", "voice", ["ch01"], "全书音色判断。",
                  "quote elsewhere", confidence=0.99)
    voice.update({"l2_verdict": "confirmed",
                  "evidence": [{"chapter": "ch01", "quote": "quote elsewhere",
                                "match_kind": "exact", "unique": True}]})
    setup = _note("setup-anchor", "setup_payoff", ["ch01", "ch04"],
                  "本块埋线将在后文回收。", "payoff anchor", confidence=0.55)
    setup.update({"l2_verdict": "confirmed", "evidence": [
        {"chapter": "ch01", "quote": "payoff anchor",
         "match_kind": "exact", "unique": True},
        {"chapter": "ch04", "quote": "late payoff",
         "match_kind": "exact", "unique": True},
    ]})
    prof = {"schema_version": 2, "voice_notes": [voice],
            "setup_payoff": [setup], "narrative_arc": []}
    ctx = build_judgment_context(
        prof, "ch01", chunk_text="context payoff anchor context",
        budget_chars=300, min_confidence=0.55, max_notes=1)
    assert ctx.selected[0]["id"] == "setup-anchor"


def test_chunk_anchor_uses_l1_matched_quote_for_fuzzy_evidence():
    """A fuzzy L1 hit must still count as anchored in the source chunk."""
    voice = _note("voice-high", "voice", ["ch01"], "全书音色判断。",
                  "quote elsewhere", confidence=0.99)
    voice.update({"l2_verdict": "confirmed",
                  "evidence": [{"chapter": "ch01", "quote": "quote elsewhere",
                                "match_kind": "exact", "unique": True}]})
    setup = _note("setup-fuzzy-anchor", "setup_payoff", ["ch01", "ch04"],
                  "本块埋线将在后文回收。", "payoff ancor", confidence=0.55)
    setup.update({"l2_verdict": "confirmed", "evidence": [
        {"chapter": "ch01", "quote": "payoff ancor",
         "matched_quote": "payoff anchor",
         "match_kind": "fuzzy", "match_score": 0.95, "unique": True},
        {"chapter": "ch04", "quote": "late payoff",
         "match_kind": "exact", "unique": True},
    ]})
    prof = {"schema_version": 2, "voice_notes": [voice],
            "setup_payoff": [setup], "narrative_arc": []}
    ctx = build_judgment_context(
        prof, "ch01", chunk_text="context payoff anchor context",
        budget_chars=300, min_confidence=0.55, max_notes=1)
    assert ctx.selected[0]["id"] == "setup-fuzzy-anchor"


def test_profile_round_trip_preserves_l1_l2_provenance_and_deep_read(tmp_path: Path):
    note = _note("voice-persisted", "voice", ["ch01"], "冷峻叙述", "quote")
    note.update({
        "l2_verdict": "confirmed",
        "l2_reason": "证据直接支撑",
        "evidence": [{"chapter": "ch01", "quote": "quote",
                      "match_kind": "exact", "match_score": 1.0,
                      "matched_quote": "quote", "unique": True}],
    })
    profile = {
        "schema_version": 2, "source": "sampled", "confidence": 0.8,
        "voice_notes": [note],
        "deep_read": {"merge_mode": "merged", "coverage": {
            "chapter_coverage": 1.0, "active_notes": 1,
        }},
    }
    path = tmp_path / "book_profile.json"
    save_profile(profile, path)

    loaded = load_profile(path)

    assert loaded["deep_read"]["merge_mode"] == "merged"
    assert loaded["deep_read"]["coverage"]["active_notes"] == 1
    assert active_notes(loaded, require_validated=True)[0]["id"] == "voice-persisted"
    assert loaded["voice_notes"][0]["l2_reason"] == "证据直接支撑"


def test_deep_merge_fallback_and_coverage(tmp_path: Path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text(
        "A beginning is a very delicate time.\n\nThe sleeper must awaken.",
        encoding="utf-8")
    (tmp_path / "input" / "ch02.md").write_text(
        "The sleeper has awakened.\n\nThe spice must flow.",
        encoding="utf-8")

    merge_calls = []

    def gen(prompt: str) -> str:
        if "合并编辑" in prompt:
            merge_calls.append(prompt)
            return json.dumps({
                "chapter_register_map": {"ch01": "冷峻叙述", "ch02": "收束与回响"},
                "voice_notes": [
                    {"id": "voice-m1", "type": "voice", "chapters": ["ch01", "ch02"],
                     "claim": "叙述声音由冷峻转向收束。",
                     "evidence": [
                         {"chapter": "ch01", "quote": "A beginning is a very delicate time."},
                         {"chapter": "ch02", "quote": "The sleeper has awakened."},
                     ], "confidence": 0.9}
                ],
                "setup_payoff": [
                    {"id": "payoff-m1", "type": "setup_payoff", "chapters": ["ch01", "ch02"],
                     "claim": "沉睡者必须觉醒的伏笔在尾章得到回收。",
                     "evidence": [
                         {"chapter": "ch01", "quote": "The sleeper must awaken."},
                         {"chapter": "ch02", "quote": "The sleeper has awakened."},
                     ], "confidence": 0.9}
                ],
                "narrative_arc": [
                    {"id": "arc-invented", "type": "arc", "chapters": ["ch01", "ch02"],
                     "claim": "这个判断使用了合并层自己编造的引文。",
                     "evidence": [{"chapter": "ch01", "quote": "The spice must flow."}],
                     "confidence": 0.8}
                ],
            }, ensure_ascii=False)
        cid = "ch01" if "第 ch01 章" in prompt else "ch02"
        if cid == "ch01":
            notes = [
                {"id": "voice-ch01-p1-001", "type": "voice", "chapters": ["ch01"],
                 "claim": "开篇语气冷峻。", "confidence": 0.8,
                 "evidence": [{"chapter": "ch01",
                               "quote": "A beginning is a very delicate time."}]},
                {"id": "setup-ch01-p1-001", "type": "setup_payoff", "chapters": ["ch01"],
                 "claim": "沉睡者必须觉醒是开篇伏笔。", "confidence": 0.8,
                 "evidence": [{"chapter": "ch01",
                               "quote": "The sleeper must awaken."}]},
            ]
        else:
            notes = [
                {"id": "voice-ch02-p1-001", "type": "voice", "chapters": ["ch02"],
                 "claim": "尾章语气转向收束。", "confidence": 0.8,
                 "evidence": [{"chapter": "ch02",
                               "quote": "The sleeper has awakened."}]},
                {"id": "setup-ch02-p1-001", "type": "setup_payoff", "chapters": ["ch02"],
                 "claim": "沉睡者已经觉醒，回应开篇伏笔。", "confidence": 0.8,
                 "evidence": [{"chapter": "ch02",
                               "quote": "The sleeper has awakened."}]},
            ]
        return json.dumps({
            "register_summary": f"{cid} register",
            "synopsis": f"{cid} synopsis",
            "notes": notes,
        }, ensure_ascii=False)

    def l2(prompt: str) -> str:
        ids = []
        marker = '【待复核注记】'
        payload = prompt.split(marker, 1)[-1].strip()
        try:
            rows = json.loads(payload)
            ids = [str(x.get("id")) for x in rows if isinstance(x, dict)]
        except Exception:
            pass
        return json.dumps({"decisions": [
            {"id": nid, "verdict": "confirmed", "reason": "证据充分"} for nid in ids
        ]}, ensure_ascii=False)

    res = sp.build_deep_profile(
        tmp_path, sp.heuristic_profile(note="test"), llm_generate=gen,
        l2_generate=l2, max_chars=10000, overlap_chars=0,
    )
    assert res["ok"] is True
    assert res["merge_mode"] == "merged"
    cov = res["coverage"]
    assert cov["chapters_with_active_notes"] == 2
    assert cov["chapter_coverage"] == 1.0
    assert cov["segments_ok"] == 2
    assert cov["segment_coverage"] == 1.0
    assert merge_calls

    prof = res["profile"]
    active = [n for n in prof["voice_notes"] + prof["setup_payoff"] + prof["narrative_arc"]
              if n.get("status") == "active"]
    ids = {n["id"] for n in active}
    assert "voice-m1" in ids
    assert "payoff-m1" in ids
    assert "arc-invented" not in ids
    invented = [n for n in prof["narrative_arc"] if n["id"] == "arc-invented"]
    assert invented and invented[0]["status"] == "dropped"
    assert invented[0]["schema_reject_reason"] == "merge_quote_not_from_extraction_candidates"
    assert prof["deep_read"]["merge_mode"] == "merged"
    assert prof["deep_read"]["coverage"]["active_notes"] == len(active)


def test_deep_merge_invalid_json_falls_back_to_candidates(tmp_path: Path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text(
        "A beginning is a very delicate time.",
        encoding="utf-8")

    def gen(prompt: str) -> str:
        if "合并编辑" in prompt:
            return "not json"
        return json.dumps({
            "register_summary": "冷峻叙述",
            "synopsis": "开篇",
            "notes": [{
                "id": "voice-ch01-p1-001", "type": "voice", "chapters": ["ch01"],
                "claim": "开篇语气冷峻。", "confidence": 0.8,
                "evidence": [{"chapter": "ch01",
                              "quote": "A beginning is a very delicate time."}],
            }],
        }, ensure_ascii=False)

    def l2(prompt: str) -> str:
        return json.dumps({"decisions": [
            {"id": "voice-ch01-p1-001", "verdict": "confirmed", "reason": "ok"}
        ]})

    res = sp.build_deep_profile(
        tmp_path, sp.heuristic_profile(note="test"), llm_generate=gen,
        l2_generate=l2, max_chars=10000, overlap_chars=0,
    )
    assert res["merge_mode"] == "candidate_fallback"
    assert res["active_notes"] == 1
    assert res["profile"]["chapter_register_map"]["ch01"] == "冷峻叙述"


def test_deep_read_skips_merge_when_no_candidates(tmp_path: Path):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("Plain source text.", encoding="utf-8")
    merge_calls = []

    def gen(prompt: str) -> str:
        if "合并编辑" in prompt:
            merge_calls.append(prompt)
            return "{}"
        return json.dumps({"register_summary": "平实", "synopsis": "叙述", "notes": []})

    res = sp.build_deep_profile(
        tmp_path, sp.heuristic_profile(note="test"), llm_generate=gen,
        l2_generate=lambda prompt: "{}", max_chars=10000, overlap_chars=0,
    )
    assert res["merge_mode"] == "no_candidate_notes"
    assert merge_calls == []
    assert res["profile"]["deep_read"]["merge_called"] is False


def _candidate(nid: str, ntype: str, chapter: str, claim: str, quote: str) -> dict:
    return {
        "id": nid, "type": ntype, "chapters": [chapter], "claim": claim,
        "confidence": 0.8,
        "evidence": [{"chapter": chapter, "quote": quote}],
        "status": "active",
    }


def _merge_candidates(prompt: str) -> list:
    marker = ("【候选注记（quote 逐字复用）】" if "【候选注记（quote 逐字复用）】" in prompt
              else "【setup_payoff 候选】")
    payload = prompt.split(marker, 1)[-1].strip()
    rows = json.loads(payload)
    return [row for row in rows if isinstance(row, dict)]


def test_hierarchical_merge_pairs_setup_across_batches():
    chapters = ["ch01", "ch02", "ch03", "ch04"]
    candidates = []
    payloads = []
    for idx, cid in enumerate(chapters, 1):
        candidates.append(_candidate(
            f"voice-{cid}", "voice", cid,
            f"{cid} voice " + ("长" * 500), f"Voice quote {cid}"))
        candidates.append(_candidate(
            f"setup-{cid}", "setup_payoff", cid,
            f"{cid} setup " + ("线" * 300), f"Setup quote {cid}"))
        payloads.append({
            "chapter_id": cid, "part_id": f"{cid}:1",
            "register_summary": f"{cid} register", "synopsis": f"{cid} synopsis",
            "notes": [],
        })

    def gen(prompt: str) -> str:
        rows = _merge_candidates(prompt)
        partial = "本批章节 id" in prompt
        notes = []
        if partial:
            for ntype in ("voice", "setup_payoff"):
                match = next((row for row in rows if row.get("type") == ntype), None)
                if match:
                    notes.append(match)
        else:
            voices = []
            seen_voice = set()
            for row in rows:
                if row.get("type") == "voice" and row.get("id") not in seen_voice:
                    seen_voice.add(row.get("id"))
                    voices.append(row)
                if len(voices) >= 4:
                    break
            setups = [row for row in rows if row.get("type") == "setup_payoff"]
            notes.extend(voices)
            if len(setups) >= 2:
                early, late = setups[0], setups[-1]
                notes.append({
                    "id": "setup-pair-m", "type": "setup_payoff",
                    "chapters": [early["evidence"][0]["chapter"],
                                 late["evidence"][0]["chapter"]],
                    "claim": "早期线索在尾章得到回收。", "confidence": 0.9,
                    "evidence": [early["evidence"][0], late["evidence"][0]],
                    "status": "active",
                })
        cmap = {cid: f"{cid} merged" for cid in
                {ev.get("chapter") for row in rows for ev in row.get("evidence") or []}}
        return json.dumps({"chapter_register_map": cmap, "voice_notes": [
            n for n in notes if n.get("type") == "voice"],
            "setup_payoff": [n for n in notes if n.get("type") == "setup_payoff"],
            "narrative_arc": []}, ensure_ascii=False)

    merge_max_chars = 5000
    res = sp.merge_deep_candidates(
        chapters, payloads, candidates, sp.heuristic_profile(note="test"), gen,
        merge_max_chars=merge_max_chars,
    )
    assert res["mode"] == "hierarchical_merged"
    assert res["batch_count"] > 1
    assert res["merge_calls"] == res["batch_count"] + 1
    assert res["prompt_chars"][0] > merge_max_chars
    assert max(res["prompt_chars"][1:]) <= merge_max_chars
    pair = next(n for n in res["notes"] if n["id"] == "setup-pair-m")
    assert pair["status"] == "active"
    assert {ev["chapter"] for ev in pair["evidence"]} == {"ch01", "ch04"}


def test_hierarchical_merge_keeps_voice_notes_when_only_setup_pairing_fits():
    chapters = ["ch01", "ch02", "ch03", "ch04"]
    candidates = []
    payloads = []
    for idx, cid in enumerate(chapters):
        candidates.append(_candidate(
            f"setup-{cid}", "setup_payoff", cid,
            f"{cid} setup " + ("线" * 300), f"Setup quote {cid}"))
        payloads.append({
            "chapter_id": cid, "part_id": f"{cid}:1",
            "register_summary": f"{cid} register", "synopsis": f"{cid} synopsis",
            "notes": [],
        })
        for j in range(8):
            candidates.append(_candidate(
                f"voice-{cid}-{j}", "voice", cid,
                f"{cid} voice {j} " + ("长" * 500), f"Voice quote {cid} {j}"))

    def gen(prompt: str) -> str:
        rows = _merge_candidates(prompt)
        only_setup = rows and all(row.get("type") == "setup_payoff" for row in rows)
        if only_setup:
            early, late = rows[0], rows[-1]
            return json.dumps({"pairs": [{
                "setup_id": early["id"], "payoff_id": late["id"],
                "reason": "同一叙事线",
            }]}, ensure_ascii=False)
        notes = [row for row in rows if row.get("type") == "voice"]
        notes += [row for row in rows if row.get("type") == "setup_payoff"]
        return json.dumps({"chapter_register_map": {}, "voice_notes": [
            n for n in notes if n.get("type") == "voice"],
            "setup_payoff": [n for n in notes if n.get("type") == "setup_payoff"],
            "narrative_arc": []}, ensure_ascii=False)

    merge_max_chars = 8000
    res = sp.merge_deep_candidates(
        chapters, payloads, candidates, sp.heuristic_profile(note="test"), gen,
        merge_max_chars=merge_max_chars,
    )
    assert res["mode"] == "hierarchical_setup_pairing"
    pair = next(n for n in res["notes"] if n["type"] == "setup_payoff")
    assert pair["meta"]["setup_id"] == "setup-ch01"
    assert pair["meta"]["payoff_id"] == "setup-ch04"
    assert {ev["chapter"] for ev in pair["evidence"]} == {"ch01", "ch04"}
    assert len([n for n in res["notes"] if n["type"] == "voice"]) == 32


def test_partial_merge_prompt_allows_unresolved_setup_but_final_forbids_it():
    candidate = _candidate(
        "setup-ch01", "setup_payoff", "ch01", "开篇埋线。", "Setup quote ch01")
    partial_prompt = sp.build_deep_merge_prompt(
        ["ch01"], [], [candidate], {}, partial=True)
    final_prompt = sp.build_deep_merge_prompt(
        ["ch01"], [], [candidate], {}, partial=False)
    assert "仍保留为 setup_payoff" in partial_prompt
    assert "无法确定回收时不要输出 setup_payoff" in final_prompt
    pairing_prompt = sp.build_setup_pairing_prompt(["ch01"], [candidate])
    assert "只能从候选 id 中选择" in pairing_prompt
    assert "不要复制 evidence" in pairing_prompt


def test_setup_pairing_commits_evidence_from_candidate_ids_not_model_text():
    early = _candidate(
        "setup-early", "setup_payoff", "ch01", "开篇埋线。", "Early quote.")
    late = _candidate(
        "setup-late", "setup_payoff", "ch04", "尾章回收。", "Late quote.")
    seen_prompt = []

    def gen(prompt: str) -> str:
        seen_prompt.append(prompt)
        return json.dumps({"pairs": [{
            "setup_id": "setup-early", "payoff_id": "setup-late",
            "reason": "同一伏笔",
        }]})

    res = sp.pair_setup_candidates(
        ["ch01", "ch02", "ch03", "ch04"], [early, late], gen)
    assert res["error"] == ""
    assert len(res["notes"]) == 1
    pair = res["notes"][0]
    assert pair["type"] == "setup_payoff"
    assert pair["chapters"] == ["ch01", "ch04"]
    assert [ev["quote"] for ev in pair["evidence"]] == ["Early quote.", "Late quote."]
    assert pair["status"] == "active"
    assert pair["meta"]["setup_id"] == "setup-early"
    assert pair["meta"]["payoff_id"] == "setup-late"
    # The model never sees a request to echo quotes; code owns envelope assembly.
    assert "Early quote." in seen_prompt[0]


def test_setup_pairing_rejects_invented_reversed_and_reused_ids():
    early = _candidate(
        "setup-early", "setup_payoff", "ch01", "开篇埋线。", "Early quote.")
    late = _candidate(
        "setup-late", "setup_payoff", "ch04", "尾章回收。", "Late quote.")
    pairs = [
        {"setup_id": "invented", "payoff_id": "setup-late"},
        {"setup_id": "setup-late", "payoff_id": "setup-early"},
        {"setup_id": "setup-early", "payoff_id": "setup-late"},
        {"setup_id": "setup-early", "payoff_id": "setup-late"},
    ]
    notes = sp.apply_setup_pairs(
        pairs, [early, late], ["ch01", "ch02", "ch03", "ch04"])
    assert len(notes) == 1
    assert notes[0]["meta"]["setup_id"] == "setup-early"
    assert notes[0]["meta"]["payoff_id"] == "setup-late"


def test_deep_read_uses_separate_merge_task(tmp_path: Path, monkeypatch):
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text("Alpha beta.", encoding="utf-8")
    tasks = []

    def fake_default(prompt: str, task_name: str = "style_profiling") -> str:
        tasks.append(task_name)
        note = {
            "id": "voice-m", "type": "voice", "chapters": ["ch01"],
            "claim": "语气克制。", "confidence": 0.8,
            "evidence": [{"chapter": "ch01", "quote": "Alpha beta."}],
            "status": "active",
        }
        if task_name == "style_profiling_merge":
            return json.dumps({"chapter_register_map": {"ch01": "克制"},
                               "voice_notes": [note], "setup_payoff": [],
                               "narrative_arc": []}, ensure_ascii=False)
        return json.dumps({"register_summary": "克制", "synopsis": "开篇",
                           "notes": [note]}, ensure_ascii=False)

    monkeypatch.setattr(sp, "default_style_profiler_generate", fake_default)

    def l2(prompt: str) -> str:
        return json.dumps({"decisions": [
            {"id": "voice-m", "verdict": "confirmed", "reason": "ok"}
        ]})

    res = sp.build_deep_profile(
        tmp_path, sp.heuristic_profile(note="test"),
        l2_generate=l2, max_chars=10000, overlap_chars=0,
    )
    assert res["complete"] is True
    assert tasks == ["style_profiling_deep", "style_profiling_merge"]
