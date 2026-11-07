"""Round 131: deep-profile upgrade, story-graph context, and injection A/B."""
from __future__ import annotations

import json
import sqlite3
import types
from pathlib import Path

import src.test_golden_set as g
from src import style_profiler as sp


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, model_name=None, extra_body=None, **params):
        self.calls.append({"prompt": prompt, "model_name": model_name,
                           "params": params})
        return "直译稿"


class _FakeAgent:
    def __init__(self, out):
        self.out = out
        self.calls = []

    def process_chunk(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.out


class _FakeEvaluator:
    instance = None

    def __init__(self, db_path=None, literal_role=None):
        type(self).instance = self
        self.literal_role = literal_role
        self.llm = _FakeLLM()
        self.ref_agent = _FakeAgent("refs")
        self.db = types.SimpleNamespace(
            format_prompt_context=lambda source_text="": "gloss")
        self.rewriter_agent = _FakeAgent("润色稿")
        self.critic_agent = _FakeAgent({"scores": {"fluency": 8.0}})
        self.judge_agent = _FakeAgent({
            "decision": "PASS",
            "_backend_path": "jev",
            "_typesafe_verdict": {"confidence_source": "jev"},
        })

    def evaluate_style_collapse(self, source_text, translated_text):
        return {
            "style_collapse_rate": 0.1,
            "average_preservation": 0.9,
            "preservation_per_dimension": {"avg_sentence_length": 0.9},
        }


class _FakeChunker:
    def __init__(self, soft_limit=1000, hard_limit=2500):
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    def split_markdown(self, text):
        return ["chunk one", "chunk two"]


class _FakeConfig:
    chunker = {"soft_limit": 1000, "hard_limit": 2500}
    style_guide = {"genre": "测试"}
    pipeline = {"golden_ab": {"min_calibrated_ratio": 0.5}}

    def get(self, key, default=None):
        if key == "judgment_injection":
            return {"max_chars": 300, "min_confidence": 0.55,
                    "max_notes": 6, "prompt_version": "r131"}
        return default

    def get_task_role(self, task_name):
        return "literal_translator"

    def resolve_task_model(self, task_name):
        return "literal_translator", {"temperature": 0.1}


def _prepare_golden_env(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    monkeypatch.setattr(g, "__file__", str(src_dir / "test_golden_set.py"))
    monkeypatch.setattr(g, "_config", _FakeConfig())
    monkeypatch.setattr(g, "GoldenSetEvaluator", _FakeEvaluator)
    monkeypatch.setattr(g, "SmartChunker", _FakeChunker)
    monkeypatch.setattr(g, "get_role_model_name", lambda role: f"model:{role}")


def _row(decision: str, score: float, *, conf: str = "jev",
         injected: int = 0) -> dict:
    return {
        "chunk_id": f"{decision}-{score}-{conf}",
        "critic_scores": {"fluency": score},
        "judge_decision": decision,
        "judge_backend": conf,
        "confidence_source": conf,
        "style_preservation": 0.9,
        "judgment_block_chars": injected,
        "judgment_note_ids": ["voice-1"] if injected else [],
    }


def test_source_text_does_not_require_golden_file(tmp_path, monkeypatch):
    _prepare_golden_env(tmp_path, monkeypatch)
    assert not (tmp_path / "input" / "golden" / "hyperion_5k.md").exists()
    report = tmp_path / "golden.json"

    results = g.run_golden_test(
        source_text="A short source passage.",
        source_id="inline",
        report_path=report,
    )

    assert results
    assert results[0]["chunk_id"].startswith("inline_chunk")
    assert report.exists()


def test_compare_injection_arms_insufficient_data():
    off = [_row("PASS", 8.0)]
    on = [{"chunk_id": "broken", "error": "boom"}]
    out = g.compare_injection_arms(off, on)
    assert out["decision"] == "insufficient_data"
    assert "no_injectable_chunks" in out["reasons"]
    assert "no_valid_samples" in out["reasons"]


def test_compare_injection_arms_regression_keeps_disabled():
    off = [_row("PASS", 8.0), _row("PASS", 8.0)]
    on = [_row("REJECT", 7.5, injected=120), _row("REJECT", 7.5, injected=120)]
    out = g.compare_injection_arms(off, on)
    assert out["decision"] == "keep_disabled"
    assert any(r.startswith("regression:") for r in out["reasons"])
    assert out["injected_note_ids"] == ["voice-1"]


def test_compare_injection_arms_enable_on_measured_gain():
    off = [_row("REJECT", 7.0), _row("REJECT", 7.0)]
    on = [_row("PASS", 8.0, injected=120), _row("PASS", 8.0, injected=120)]
    out = g.compare_injection_arms(off, on)
    assert out["decision"] == "enable"
    assert out["reasons"] == []
    assert out["delta"]["pass_rate"] == 1.0


def test_run_injection_ab_without_notes_never_calls_models(tmp_path, monkeypatch):
    _prepare_golden_env(tmp_path, monkeypatch)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text(
        "A quiet room with no names.", encoding="utf-8")

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("no injectable notes must short-circuit before model calls")

    monkeypatch.setattr(g, "run_golden_test", _must_not_run)
    report_path = tmp_path / "out" / "ab.json"
    res = g.run_injection_ab(
        "ch01", {"schema_version": 2, "source": "sampled"}, report_path=report_path)

    assert res["ok"] is False
    assert res["decision"] == "keep_disabled"
    assert res["reasons"] == ["no_injectable_notes"]
    assert report_path.exists()


def test_judgment_ab_uses_one_private_db_snapshot_per_arm(tmp_path, monkeypatch):
    _prepare_golden_env(tmp_path, monkeypatch)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "ch01.md").write_text(
        "A quiet room with Paul.", encoding="utf-8")
    profile = {
        "schema_version": 2,
        "source": "sampled",
        "voice_notes": [{
            "id": "voice-ch01-001",
            "type": "voice",
            "chapters": ["ch01"],
            "claim": "本章叙述保持克制。",
            "evidence": [{
                "chapter": "ch01",
                "quote": "A quiet room",
                "match_kind": "exact",
            }],
            "confidence": 0.9,
            "status": "active",
            "l2_verdict": "confirmed",
        }],
    }
    db_paths = []

    def fake_run_golden_test(*, source_text=None, source_id=None, report_path=None,
                             judgment_context_provider=None, db_path=None, **_kwargs):
        db_paths.append(str(db_path))
        if judgment_context_provider:
            return [_row("PASS", 8.2, injected=120)]
        return [_row("REJECT", 7.8)]

    monkeypatch.setattr(g, "run_golden_test", fake_run_golden_test)
    report_path = tmp_path / "out" / "ab.json"
    res = g.run_injection_ab("ch01", profile, report_path=report_path)

    assert res["ok"] is True
    assert res["decision_db_isolation"] == "snapshot_per_arm"
    assert len(db_paths) == 2 and len(set(db_paths)) == 2
    assert all("sagatrans_judgment_ab_" in p for p in db_paths)
    assert all(Path(p).name in {"off_decision_db.sqlite", "on_decision_db.sqlite"}
               for p in db_paths)


def test_sqlite_snapshot_isolated_from_source_and_other_arm(tmp_path):
    source = tmp_path / "production.sqlite"
    dest = tmp_path / "arm.sqlite"
    con = sqlite3.connect(source)
    con.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO sample VALUES (1, 'production')")
    con.commit()
    con.close()

    g._snapshot_sqlite_database(source, dest)
    con = sqlite3.connect(dest)
    con.execute("INSERT INTO sample VALUES (2, 'arm-only')")
    con.commit()
    con.close()

    source_con = sqlite3.connect(source)
    assert source_con.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 1
    source_con.close()
    dest_con = sqlite3.connect(dest)
    assert dest_con.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 2
    dest_con.close()


def _write_profile(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_profile_artifact_is_published_without_temp_tail(tmp_path):
    path = tmp_path / "input" / "book_profile.json"
    sp.save_profile({"schema_version": 2, "source": "sampled"}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert list(path.parent.glob("*.tmp")) == []


def test_deep_completion_requires_success_but_partial_artifact_is_not_done():
    assert sp.profile_is_deep({"deep_read": {"complete": True}}) is True
    assert sp.profile_is_deep({"deep_read": {"complete": False,
                                             "status": "partial"}}) is False
    assert sp.profile_is_deep({"deep_read": {"complete": False,
                                             "status": "empty"}}) is False
    # Early Round-131 artifacts predate the completion bit.
    assert sp.profile_is_deep({"deep_read": {"merge_mode": "merged"}}) is True


def _install_fake_deep(monkeypatch):
    calls = []

    def fake_deep(root, base_profile, **kwargs):
        calls.append(dict(base_profile))
        prof = dict(base_profile)
        prof["schema_version"] = sp.PROFILE_VERSION
        prof["deep_read"] = {"complete": True, "merge_mode": "merged",
                             "coverage": {"active_notes": 0}}
        return {
            "ok": True,
            "complete": True,
            "profile": prof,
            "audit": {},
            "segments": 1,
            "candidate_notes": 0,
            "notes_after_limits": 0,
            "active_notes": 0,
            "merge_mode": "merged",
            "coverage": {"active_notes": 0},
        }

    monkeypatch.setattr(sp, "build_deep_profile", fake_deep)
    monkeypatch.setattr(sp, "_try_build_story_graph", lambda root, profile: ("", {}))
    return calls


def test_deep_request_upgrades_v1_and_shallow_v2_profiles(tmp_path, monkeypatch):
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    (in_dir / "ch01.md").write_text("Chapter one text.", encoding="utf-8")
    profile_path = in_dir / "book_profile.json"
    calls = _install_fake_deep(monkeypatch)

    _write_profile(profile_path, {
        "schema_version": 1,
        "source": "sampled",
        "confidence": 0.7,
        "genre": "科幻小说",
    })
    v1_result = sp.generate_book_profile(tmp_path, deep=True)
    assert v1_result.get("skipped") is not True
    assert len(calls) == 1
    assert v1_result["profile"]["deep_read"]["merge_mode"] == "merged"

    # A shallow profile written by current code is schema v2 but still lacks
    # deep_read; it must not masquerade as a finished deep artifact.
    sp.save_profile(sp.normalize_profile({"source": "sampled", "confidence": 0.7}),
                    profile_path)
    shallow_result = sp.generate_book_profile(tmp_path, deep=True)
    assert shallow_result.get("skipped") is not True
    assert len(calls) == 2

    done_result = sp.generate_book_profile(tmp_path, deep=True)
    assert done_result.get("skipped") is True
    assert len(calls) == 2


def test_story_graph_context_skips_chapter_without_entities():
    from src.story_graph import story_graph_context_for_chapter

    graph = {
        "coverage": {
            "answerable": True,
            "chapters_with_entities": 1,
            "chapter_entity_coverage": {"ch01": 0.5, "ch02": 0.0},
            "uncovered_chapters": ["ch02"],
        },
        "nodes": [
            {"id": "entity:entity-1", "type": "entity", "label": "保罗"},
        ],
        "edges": [
            {"type": "appears_in", "source": "entity:entity-1",
             "target": "chapter:ch01", "count": 2,
             "certification": "auto_verified"},
        ],
    }

    assert "保罗" in story_graph_context_for_chapter(graph, "ch01")
    assert story_graph_context_for_chapter(graph, "ch02") == ""
    assert story_graph_context_for_chapter(graph, "ch99") == ""

    graph["edges"][0]["certification"] = "pending"
    assert story_graph_context_for_chapter(graph, "ch01") == ""


def test_cli_and_one_click_wiring_for_deep_graph_and_ab():
    root = Path(__file__).resolve().parent.parent
    main_src = (root / "src" / "main.py").read_text(encoding="utf-8")
    mono_src = (root / "src" / "translator_agent.py").read_text(encoding="utf-8")
    script = (root / "scripts" / "translate_book.sh").read_text(encoding="utf-8")

    for text in (main_src, mono_src):
        assert 'parser.add_argument("--judgment-ab"' in text
        assert "run_injection_ab(" in text
    assert "--skip-deep-profile" in script
    assert "profile_is_deep()" in script
    assert "style-profile --deep" in script
    assert "story-graph" in script

    from src.main import _MUTATING_COMMANDS
    assert "story-graph" in _MUTATING_COMMANDS
