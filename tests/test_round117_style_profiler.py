"""Round 117 P0: StyleProfiler 单元测试（抽样/归一/回退/覆盖层/落盘/顶层入口）。"""
from __future__ import annotations

import json
from pathlib import Path

from src import style_profiler as sp

SAMPLE_LLM_JSON = json.dumps({
    "work_guess": {"recognized": True, "title": "测试作品", "author": "测试作者", "year": "1922"},
    "source": "known_work",
    "confidence": 0.8,
    "genre": "侦探小说",
    "era": "1920s",
    "region": "英国",
    "persona": "擅长讽刺对话的译者",
    "register_default": "克制、口语化、带讽刺",
    "footnote_policy": "inline",
    "forbidden_modernisms": ["老铁"],
    "translation_hints": ["wild-cat = 投机的（抽样中出现）"],
    "chapter_register_map": {"ch01": "讽刺而克制", "ch05": "动作场景，短句"},
    "evidence_quotes": ["'That is the stuff,' he said."],
}, ensure_ascii=False)


def test_spread_sample_chapters_covers_head_tail():
    files = [Path(f"ch{i:02d}.md") for i in range(1, 33)]
    picked = sp.spread_sample_chapters(files, 5)
    names = [p.name for p in picked]
    assert len(names) == 5
    assert names[0] == "ch01.md" and names[-1] == "ch32.md"
    assert names == sorted(names)


def test_load_chapter_samples_explicit_and_missing(tmp_path):
    (tmp_path / "ch01.md").write_text("A" * 100, encoding="utf-8")
    (tmp_path / "ch02.md").write_text("B" * 100, encoding="utf-8")
    samples = sp.load_chapter_samples(tmp_path, chapters=["ch02", "ch99"])
    assert [s["chapter_id"] for s in samples] == ["ch02"]
    assert samples[0]["text"] == "B" * 100


def test_normalize_profile_fills_defaults_and_coerces():
    prof = sp.normalize_profile({
        "genre": " 科幻小说 ",
        "confidence": 1.5,
        "footnote_policy": "bogus",
        "source": "nonsense",
        "chapter_register_map": {"ch01": " 冷静克制 "},
        "translation_hints": [{"term": "foo", "note": "示例"}],
        "forbidden_modernisms": "not-a-list",
        "work_guess": {"recognized": True, "title": " X "},
    })
    assert prof["genre"] == "科幻小说"
    assert prof["confidence"] == 1.0
    assert prof["footnote_policy"] == "inline"
    assert prof["source"] == "sampled"
    assert prof["chapter_register_map"]["ch01"] == "冷静克制"
    assert prof["translation_hints"] == ["foo = 示例"]
    assert prof["forbidden_modernisms"] == []
    assert prof["work_guess"]["title"] == "X"
    assert prof["schema_version"] == sp.PROFILE_VERSION


def test_parse_profile_response_roundtrip_and_fallbacks():
    prof = sp.parse_profile_response(SAMPLE_LLM_JSON, samples=[{"chapter_id": "ch01", "text": "x"}])
    assert prof["source"] == "known_work"
    assert prof["samples_used"] == ["ch01"]

    # 非 JSON → 中性回退（不注入作品特定先验）
    fb = sp.parse_profile_response("这不是 JSON", samples=[{"chapter_id": "ch01", "text": "x"}])
    assert fb["source"] == "fallback"
    assert fb["confidence"] == 0.0
    assert fb["samples_used"] == ["ch01"]
    assert fb["translation_hints"] == []


def test_build_profile_llm_error_falls_back():
    def _boom(prompt):
        raise RuntimeError("no network")
    prof = sp.build_profile([{"chapter_id": "ch01", "text": "x"}], llm_generate=_boom)
    assert prof["source"] == "fallback"
    assert "llm_error" in prof["note"]


def test_profile_to_style_guide_overlay_and_chapter_map():
    prof = sp.parse_profile_response(SAMPLE_LLM_JSON)
    base = {"avg_sentence_length": "基线", "work_type": "长篇小说"}
    guide = sp.profile_to_style_guide(prof, chapter_id="ch05", base=base)
    assert guide["avg_sentence_length"] == "基线"
    assert guide["genre"] == "侦探小说"
    assert guide["persona"] == "擅长讽刺对话的译者"
    assert guide["register_default"] == "动作场景，短句"  # 章节映射优先
    assert guide["translation_hints"] == ["wild-cat = 投机的（抽样中出现）"]
    assert guide["work_era_region"] == "1920s，英国"
    assert guide["work_title"] == "测试作品"

    guide_other = sp.profile_to_style_guide(prof, chapter_id="ch03", base=base)
    assert guide_other["register_default"] == "克制、口语化、带讽刺"  # 未命中章节 → 全局


def test_save_load_roundtrip(tmp_path):
    prof = sp.parse_profile_response(SAMPLE_LLM_JSON)
    p = tmp_path / "book_profile.json"
    sp.save_profile(prof, p)
    loaded = sp.load_profile(p)
    assert loaded["genre"] == "侦探小说"
    assert loaded["chapter_register_map"]["ch01"] == "讽刺而克制"

    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    assert sp.load_profile(tmp_path / "bad.json") is None
    assert sp.load_profile(tmp_path / "missing.json") is None


def test_generate_book_profile_dry_run_and_llm_write(tmp_path):
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    (in_dir / "ch01.md").write_text("Chapter one text.", encoding="utf-8")
    (in_dir / "ch02.md").write_text("Chapter two text.", encoding="utf-8")

    res_dry = sp.generate_book_profile(tmp_path, dry_run=True)
    assert res_dry["ok"] and res_dry["dry_run"]
    assert not Path(res_dry["path"]).exists()

    res = sp.generate_book_profile(tmp_path, llm_generate=lambda prompt: SAMPLE_LLM_JSON)
    assert res["ok"] and not res.get("skipped")
    assert res.get("saved") is True
    assert Path(res["path"]).exists()
    assert res["source"] == "known_work"
    assert "ch01" in res["samples"]

    res_skip = sp.generate_book_profile(tmp_path, llm_generate=lambda prompt: (_ for _ in ()).throw(AssertionError("must not call")))
    assert res_skip["skipped"] is True

    res_force = sp.generate_book_profile(tmp_path, force=True, llm_generate=lambda prompt: SAMPLE_LLM_JSON)
    assert res_force["ok"] and not res_force.get("skipped")


def test_generate_book_profile_llm_failure_does_not_save(tmp_path):
    """LLM 不可用 → 中性回退且不落盘（避免空剖面遮蔽后续真实生成）。"""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    (in_dir / "ch01.md").write_text("Chapter one text.", encoding="utf-8")

    def _boom(prompt):
        raise RuntimeError("network down")

    res = sp.generate_book_profile(tmp_path, llm_generate=_boom)
    assert res["ok"] is True
    assert res["source"] == "fallback"
    assert res["saved"] is False
    assert not (in_dir / "book_profile.json").exists()


def test_generate_book_profile_no_samples(tmp_path):
    (tmp_path / "input").mkdir()
    res = sp.generate_book_profile(tmp_path)
    assert res["ok"] is False
    assert "未在" in res["error"]


def test_config_style_profiling_routing():
    from src.config import get_config
    cfg = get_config()
    role, params = cfg.resolve_task_model("style_profiling")
    assert role == "primary"
    assert params.get("effort") == "max"    # Round 129: 需要文学史判定 → 开到最高
    assert "entity_variants" in dir(cfg) or hasattr(type(cfg), "entity_variants")
