"""Round 131 Phase 3: verified structural P0 repair sidecars and governance."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

from src.config import config as _cfg
from src.core.scheduler import TaskScheduler
from src.judgment_layer import sha256_text
from src.pipeline import TranslationPipeline


def _load_editor_review():
    path = Path(__file__).resolve().parent.parent / "scripts" / "editor_review.py"
    spec = importlib.util.spec_from_file_location("round131_editor_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub(tmp_path: Path) -> TranslationPipeline:
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = tmp_path
    pipeline.chapter_id = "ch01"
    pipeline.scheduler = MagicMock(spec=TaskScheduler)
    pipeline.decision_engine = MagicMock()
    pipeline.decision_engine.format_prompt_context.return_value = ""
    pipeline._intermediate_cache = {}
    return pipeline


def _final(tmp_path: Path, chunk_id: str, text: str, suffix: str = "final") -> Path:
    path = tmp_path / f"{chunk_id}_{suffix}.json"
    path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    return path


def _artifact(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "review_dirty.json"
    path.write_text(json.dumps({"entries": entries}, ensure_ascii=False), encoding="utf-8")
    return path


def _entry(chunk_id: str, quote: str, category: str = "dangling") -> dict:
    return {
        "chapter": "ch01", "chunk_id": chunk_id, "category": category,
        "quote": quote, "note": "结构问题", "suggestion": "补齐/去重",
    }


def test_apply_review_fixes_remeasures_current_final_and_skips_ambiguous(tmp_path):
    pipeline = _stub(tmp_path)
    _final(tmp_path, "ch01_c1", "开头。这是唯一锚句。结尾。内部重复。内部重复。")
    _final(tmp_path, "ch01_c2", "另一条唯一。重复锚句。")
    _final(tmp_path, "ch01_c3", "重复锚句。")
    pipeline.scheduler.trigger_backtrack.return_value = {
        "dirty": ["ch01_c1"], "over_cap": [], "skipped_state": [],
        "pf_requeued": [], "pf_over_cap": [],
    }
    path = _artifact(tmp_path, [
        _entry("ch01_c1", "这是唯一锚句。"),
        _entry("ch01_c2", "另一条唯一。", "voice"),
        _entry("ch01_c1", "重复锚句。"),
        _entry("ch01_c1", "内部重复。"),
        _entry("ch01_c1", "已经消失的锚句。"),
        _entry("ch01_c1", "另一条唯一。"),
    ])

    result = pipeline.apply_review_fixes(path)

    assert result["ok"] is True
    assert result["mapped_issues"] == 1
    assert result["scheduled_chunks"] == ["ch01_c1"]
    reasons = {x.get("reason") for x in result["skipped"]}
    assert {"not_structural_p0", "ambiguous_across_chunks",
            "quote_not_unique_in_chunk", "quote_not_in_final_chunks",
            "chunk_mapping_changed"} <= reasons
    pipeline.scheduler.trigger_backtrack.assert_called_once_with(["ch01_c1"])

    sidecar_path = tmp_path / "ch01_c1_review_fix.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["chunk_id"] == "ch01_c1"
    assert sidecar["original_final_sha256"] == sha256_text(
        "开头。这是唯一锚句。结尾。内部重复。内部重复。")
    assert len(sidecar["issues"]) == 1
    assert sidecar["issues"][0]["category"] == "dangling"
    assert sidecar["issues"][0]["unique_in_chunk"] is True


def test_review_dirty_artifact_does_not_offer_nonunique_anchor(tmp_path):
    editor_review = _load_editor_review()
    chapter_dir = tmp_path / "output" / "ch01"
    chapter_dir.mkdir(parents=True)
    _final(chapter_dir, "ch01_c1", "重复锚句。重复锚句。")
    results = [{
        "chapter": "ch01",
        "issues": [{
            "severity": "P0", "category": "dup_note", "quote": "重复锚句。",
            "note": "重复", "suggestion": "去重", "anchor_verified": True,
        }],
    }]

    dirty = editor_review.build_review_dirty(tmp_path, results)

    assert dirty["summary"] == {"entries": 0, "skipped": 1}
    assert dirty["skipped"][0]["reason"] == "quote_not_unique_in_chunk"


def test_default_generate_factory_binds_prompt_at_call_time(monkeypatch):
    editor_review = _load_editor_review()
    import src.utils.llm_adapter as llm_adapter

    fake_cfg = MagicMock()
    fake_cfg.task_routing = {"editor_review_global": {}}
    fake_cfg.resolve_task_model.return_value = ("secondary", {"temperature": 0.2})
    fake_cfg.get_task_role.return_value = "secondary"
    fake_llm = MagicMock()
    fake_llm.generate.return_value = '{"issues": []}'

    monkeypatch.setattr("src.config.get_config", lambda: fake_cfg)
    monkeypatch.setattr(llm_adapter, "get_llm_client_for_task", lambda _task: fake_llm)
    monkeypatch.setattr(llm_adapter, "get_role_model_name", lambda _role: "model-x")

    generate = editor_review._make_default_generate("editor_review_global")

    assert callable(generate)
    assert generate("PROMPT") == '{"issues": []}'
    fake_llm.generate.assert_called_once_with(
        "PROMPT", model_name="model-x", temperature=0.2
    )


def test_review_fix_context_requires_exact_backup_hash(tmp_path):
    pipeline = _stub(tmp_path)
    current_text = "新版包含其他内容。"
    base_text = "开头。缺失的段落。结尾。"
    final = _final(tmp_path, "ch01_c1", current_text)
    pipeline._save_intermediate("ch01_c1", "review_fix", {
        "schema_version": 1, "chunk_id": "ch01_c1",
        "original_final_sha256": sha256_text(base_text),
        "issues": [{"severity": "P0", "category": "dangling",
                    "quote": "缺失的段落。", "note": "缺段"}],
    })

    # The sidecar was written against another final text and there is no
    # pre-rerun backup: never inject it.
    assert pipeline._review_fix_context_for("ch01_c1") == ""

    backup = tmp_path / "ch01_c1_final_backup.json"
    final.rename(backup)
    assert pipeline._review_fix_context_for("ch01_c1") == ""
    backup.write_text(json.dumps({"text": base_text}), encoding="utf-8")
    context = pipeline._review_fix_context_for("ch01_c1")
    assert "成书终审结构修复" in context
    assert "缺失的段落。" in context

    backup.write_text(json.dumps({"text": "篡改后的备份"}), encoding="utf-8")
    assert pipeline._review_fix_context_for("ch01_c1") == ""


def test_structural_fix_sidecar_bypasses_version_merit_and_is_cleaned(tmp_path):
    pipeline = _stub(tmp_path)
    cid = "ch01_c1"
    old_text = "旧版含结构缺陷。"
    new_text = "新版已补齐结构缺陷。"
    _final(tmp_path, cid, new_text)
    _final(tmp_path, cid, old_text, suffix="final_backup")
    (tmp_path / f"{cid}_critic_report.json").write_text(
        json.dumps({"scores": {"fluency": 1.0}}), encoding="utf-8")
    (tmp_path / f"{cid}_critic_report_backup.json").write_text(
        json.dumps({"scores": {"fluency": 9.9}}), encoding="utf-8")
    pipeline._save_intermediate(cid, "review_fix", {
        "schema_version": 1, "chunk_id": cid,
        "original_final_sha256": sha256_text(old_text),
        "issues": [{"severity": "P0", "category": "dangling",
                    "quote": "结构缺陷。", "note": "补段"}],
    })

    restored = pipeline._restore_dirty_backups()

    assert restored == []
    assert json.loads((tmp_path / f"{cid}_final.json").read_text())["text"] == new_text
    assert not (tmp_path / f"{cid}_final_backup.json").exists()
    assert not (tmp_path / f"{cid}_review_fix.json").exists()


def test_apply_review_fixes_cleans_sidecars_frozen_by_budget(tmp_path):
    pipeline = _stub(tmp_path)
    _final(tmp_path, "ch01_c1", "只此一处。")
    pipeline.scheduler.trigger_backtrack.return_value = {
        "dirty": [], "over_cap": ["ch01_c1"], "skipped_state": [],
        "pf_requeued": [], "pf_over_cap": [],
    }
    result = pipeline.apply_review_fixes(_artifact(tmp_path, [_entry("ch01_c1", "只此一处。")]))

    assert result["ok"] is True
    assert result["scheduled_chunks"] == []
    assert any(x.get("reason") == "backtrack_budget_exhausted"
               for x in result["skipped"])
    assert not (tmp_path / "ch01_c1_review_fix.json").exists()


def test_pipeline_literal_prompt_carries_judgment_and_rewriter_carries_fix(tmp_path):
    pipeline = _stub(tmp_path)
    pipeline.book_profile = {
        "schema_version": 2,
        "voice_notes": [{
            "id": "voice-1", "type": "voice", "chapters": ["ch01"],
            "claim": "叙述保持冷峻。", "confidence": 0.9, "status": "active",
            "l2_verdict": "confirmed",
            "evidence": [{"chapter": "ch01", "quote": "cold text",
                          "match_kind": "exact"}],
        }],
    }
    pipeline.judgment_cfg = {"enabled": True, "max_chars": 300,
                             "min_confidence": 0.55, "max_notes": 6}
    prompt = pipeline._build_literal_prompt("cold text", "ch01_c1")
    assert "叙述保持冷峻。" in prompt

    from src.agents.rewriter_agent import LiteraryRewriterAgent
    agent = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
    agent._few_shot_examples = ""
    rw_prompt = agent._build_prompt(
        "底稿", "【术语】", {"genre": "小说"},
        review_fix_context="【成书终审结构修复（P0，必须逐条落实）】\n1. [dangling] 锚句: 缺段",
    )
    assert "成书终审结构修复" in rw_prompt
    assert "缺段" in rw_prompt


def test_review_fix_cfg_present_for_builtin_and_repo_config():
    cfg = _cfg.get("review_fix", {}) or {}
    assert "structural_p0_bypass_merit" in cfg


def test_pipeline_fix_from_review_cli_wires_fix_before_run(tmp_path, monkeypatch):
    """The CLI must apply verified fixes before entering the normal pipeline run."""
    import sys
    import src.main as main_mod

    calls = []

    class FakePipeline:
        def __init__(self, chapter_id):
            calls.append(("init", chapter_id))

        def apply_review_fixes(self, path):
            calls.append(("fix", Path(path)))
            return {"ok": True, "skipped": []}

        def run(self):
            calls.append(("run",))
            return True

    monkeypatch.setattr(main_mod, "TranslationPipeline", FakePipeline)
    monkeypatch.setattr(main_mod, "acquire_instance_lock", lambda: True)
    monkeypatch.setattr(main_mod, "print_banner", lambda: None)
    monkeypatch.setattr(sys, "argv", [
        "sagatrans", "pipeline", "--chapter", "ch01", "--fix-from-review",
        "--review-dirty-path", str(tmp_path / "review_dirty.json"),
    ])

    main_mod.main()

    assert [c[0] for c in calls] == ["init", "fix", "run"]
    assert calls[0][1] == "ch01"
    assert calls[1][1] == tmp_path / "review_dirty.json"
