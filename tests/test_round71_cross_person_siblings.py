"""Round 71: 跨人物冲突抑制 — 同一英文 source_name 被登记到多个 canonical 时,
若 lit_text 出现任一 sibling canonical, 不报 "missing" 假阳性.

根因: ReferenceAgent._register_entities 用 canonical_name 做唯一键, 不同章节注册
      同一英文 source_name 到不同中文译名 → entity_registry 跨人物冲突.
      修前: rewriter 用别名译名时, 其它 sibling canonical 报 "missing" → 自相矛盾反馈
            → ch32 实测 6 轮 quality_retries 全失败.
      修后: _check_entity_consistency 用 get_sibling_canonicals_by_source 查全 sibling,
            抑制 sibling 命中时的 "missing" 报告. variant_instead_of_canonical 不受影响.
"""
from __future__ import annotations

import sqlite3
import sys

import pytest


@pytest.fixture
def in_memory_registry():
    """构造一个 in-memory entity_registry, 复刻 ch32 的跨人物冲突场景."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            confidence TEXT DEFAULT 'low',
            first_seen_chunk TEXT,
            pending_review INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pronoun_gender TEXT,
            entity_type TEXT,
            review_reason TEXT,
            translation_strategy TEXT DEFAULT 'free'
        );
    """)
    # Mr. Lomax 冲突 (ch32 根因):
    rows = [
        ("乔治·洛马克斯阁下", ["the Hon. George Lomax"], "high", 1),
        ("洛马克斯先生", ["Mr. Lomax", "George", "Lomax"], "high", 1),
        ("乔治·洛马克斯先生", ["Mr. Lomax", "Mr. George Lomax", "George", "sir"], "high", 0),
        # 普通实体无冲突:
        ("伦敦", ["London"], "high", 0),
        # 完全独立:
        ("巴特尔警督", ["the superintendent", "Buttle"], "high", 0),
    ]
    for canon, srcs, conf, pending in rows:
        import json
        conn.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, confidence, pending_review) "
            "VALUES (?, ?, ?, ?)",
            (canon, json.dumps(srcs), conf, pending),
        )
    conn.commit()
    yield conn
    conn.close()


class TestSiblingLookup:
    """get_sibling_canonicals_by_source 应返回含 pending_review=1 行的 sibling 集合"""

    def test_returns_conflict_siblings(self, in_memory_registry):
        from src.core.decision_engine import get_sibling_canonicals_by_source
        conn = in_memory_registry
        sib = get_sibling_canonicals_by_source(conn)
        # Mr. Lomax 跨 2 个 canonical (第三方的 George 跨 2 个, but "the Hon. George Lomax" only 1)
        assert sib["Mr. Lomax"] == {"洛马克斯先生", "乔治·洛马克斯先生"}
        assert sib["George"] == {"洛马克斯先生", "乔治·洛马克斯先生"}

    def test_unique_source_returns_single_canonical(self, in_memory_registry):
        from src.core.decision_engine import get_sibling_canonicals_by_source
        sib = get_sibling_canonicals_by_source(in_memory_registry)
        assert sib["London"] == {"伦敦"}
        assert sib["Buttle"] == {"巴特尔警督"}

    def test_includes_pending_review_rows(self, in_memory_registry):
        """关键: sibling 查找必须包含 pending_review=1 的行, 否则运行时查不到冲突."""
        from src.core.decision_engine import get_sibling_canonicals_by_source
        sib = get_sibling_canonicals_by_source(in_memory_registry)
        # the Hon. George Lomax 仅在 pending=1 行注册
        assert sib["the Hon. George Lomax"] == {"乔治·洛马克斯阁下"}

    def test_empty_or_invalid_source_names_ignored(self):
        """robust 处理: source_names 为空/非列表/JSON 损坏的 row 不爆."""
        from src.core.decision_engine import get_sibling_canonicals_by_source
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE entity_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                source_names TEXT NOT NULL,
                pending_review INTEGER DEFAULT 0
            );
        """)
        # 空 / None / 坏 JSON / 非列表 都应被忽略
        conn.execute("INSERT INTO entity_registry VALUES (1, 'a', '[]', 0)")
        conn.execute("INSERT INTO entity_registry VALUES (3, 'c', 'not-json', 0)")
        conn.execute("INSERT INTO entity_registry VALUES (4, 'd', '\"a string not list\"', 0)")
        conn.execute("INSERT INTO entity_registry VALUES (5, 'e', '[\"valid\"]', 0)")
        conn.commit()
        sib = get_sibling_canonicals_by_source(conn)
        assert sib == {"valid": {"e"}}


class TestCrossPersonConflicts:
    """detect_cross_person_conflicts 应找出所有 source_name -> ≥2 canonicals 的冲突."""

    def test_finds_known_conflicts(self, in_memory_registry):
        from src.core.decision_engine import detect_cross_person_conflicts
        conflicts = detect_cross_person_conflicts(in_memory_registry)
        assert "Mr. Lomax" in conflicts
        assert set(conflicts["Mr. Lomax"]) == {"洛马克斯先生", "乔治·洛马克斯先生"}

    def test_unique_sources_not_listed(self, in_memory_registry):
        from src.core.decision_engine import detect_cross_person_conflicts
        conflicts = detect_cross_person_conflicts(in_memory_registry)
        assert "London" not in conflicts
        assert "Buttle" not in conflicts


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        # Round 103 P6+: 动态 polluted 自动覆盖 src.* 父包 + 所有子模块,
        # 父包循环已是冗余, 不再单独遍历 parents.
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)


class TestPipelineSiblingSuppression:
    """Pipeline._check_entity_consistency 在 sibling 命中时抑制 'missing' 报告."""

    def _make_pipeline_with_registry(self, conn):
        """绕过 __init__ 注入 decision_engine 与 sibling cache."""
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        # 用 mock 替换 decision_engine
        class _DE:
            def __init__(self):
                self.conn = conn
        pipeline.decision_engine = _DE()
        # 预填 cache, 跳过 lazy init
        from src.core.decision_engine import get_sibling_canonicals_by_source
        pipeline._sibling_canonicals_cache = get_sibling_canonicals_by_source(conn)
        return pipeline

    def test_sibling_in_lit_suppresses_missing(self, in_memory_registry):
        """ch32 场景: rewriter 用 '洛马克斯先生' (sibling), canonical=乔治·洛马克斯先生
        应被 sibling 命中抑制, 不报 'missing'."""
        pipeline = self._make_pipeline_with_registry(in_memory_registry)
        source_text = "Mr. Lomax arrived at the manor."
        # rewriter 实际产出 (使用 sibling 译名, 不是 "winner" canonical)
        lit_text = "洛马克斯先生抵达庄园。"
        notes = pipeline._check_entity_consistency("chXX_chunk000", lit_text, source_text)
        # sibling 命中: 不应有 missing 类型
        missing = [n for n in notes if n.get("type") == "missing"]
        assert missing == [], (
            f"sibling 命中应抑制 missing, 但仍报: {missing}. "
            f"实际 notes: {notes}"
        )

    def test_no_sibling_still_reports_missing(self, in_memory_registry):
        """无 sibling 命中时, 'missing' 应正常报告."""
        pipeline = self._make_pipeline_with_registry(in_memory_registry)
        source_text = "Mr. Lomax arrived at the manor."
        # rewriter 产出中没有任何 sibling canonical
        lit_text = "某人抵达庄园。"
        notes = pipeline._check_entity_consistency("chXX_chunk000", lit_text, source_text)
        missing = [n for n in notes if n.get("type") == "missing"]
        # canonical=乔治·洛马克斯先生 应被报 missing
        assert any(n["entity"] == "乔治·洛马克斯先生" for n in missing), (
            f"无 sibling 时 missing 应正常报告, 但 notes={notes}"
        )

    def test_variant_path_unaffected(self, in_memory_registry):
        """variant_instead_of_canonical 路径不受 sibling 抑制影响 (variant 是真问题)."""
        pipeline = self._make_pipeline_with_registry(in_memory_registry)
        source_text = "London was foggy."
        # 伦敦 是无冲突实体, lit_text 残留英文 "London"
        lit_text = "London 雾气弥漫。"
        notes = pipeline._check_entity_consistency("chXX_chunk000", lit_text, source_text)
        # 应报 variant_instead_of_canonical (英文残留)
        variants = [n for n in notes if n.get("type") == "variant_instead_of_canonical"]
        assert any(n["entity"] == "伦敦" for n in variants), (
            f"variant 路径不受影响, 应报英文残留, 但 notes={notes}"
        )

    def test_unrelated_entity_normal_flow(self, in_memory_registry):
        """完全无冲突的实体走原流程."""
        pipeline = self._make_pipeline_with_registry(in_memory_registry)
        source_text = "Buttle was on duty."
        lit_text = "巴特尔警督执勤中。"
        notes = pipeline._check_entity_consistency("chXX_chunk000", lit_text, source_text)
        # 无 missing 无 variant
        assert notes == [], f"完全合规的翻译不应触发任何 notes, 但 notes={notes}"

    def test_cache_reset_between_audits(self, in_memory_registry):
        """_process_auditing_batch 应重置 _sibling_canonicals_cache."""
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        class _DE:
            def __init__(self):
                self.conn = in_memory_registry
        pipeline.decision_engine = _DE()
        # 模拟上一次审计遗留的 cache
        pipeline._sibling_canonicals_cache = {"stale": {"x"}}
        # 调用 _process_auditing_batch 应清掉 cache
        pipeline._process_auditing_batch([])
        assert pipeline._sibling_canonicals_cache is None, (
            "每次审计批次开始应重置 cache, 防止跨章节脏数据"
        )

    def test_lazy_init_when_cache_missing(self, in_memory_registry):
        """首次调用 _check_entity_consistency 应 lazy-init cache."""
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        class _DE:
            def __init__(self):
                self.conn = in_memory_registry
        pipeline.decision_engine = _DE()
        # 没有 _sibling_canonicals_cache
        assert not hasattr(pipeline, "_sibling_canonicals_cache") or \
               pipeline._sibling_canonicals_cache is None
        pipeline._check_entity_consistency(
            "chXX_chunk000", "伦敦在下雨", "London was raining.",
        )
        # 第一次调用后 cache 应被设置
        assert pipeline._sibling_canonicals_cache is not None
        assert "London" in pipeline._sibling_canonicals_cache


class TestIntegratedScriptExports:
    """整合脚本 src/translator_agent.py 必须导出 Round 70+71 helpers."""

    def test_get_sibling_canonicals_by_source_importable(self):
        from src.translator_agent import get_sibling_canonicals_by_source
        assert callable(get_sibling_canonicals_by_source)

    def test_detect_cross_person_conflicts_importable(self):
        from src.translator_agent import detect_cross_person_conflicts
        assert callable(detect_cross_person_conflicts)

    def test_synthetic_submodule_exports_match(self):
        """_SRC_SUBMODULES 注册的 src.core.decision_engine 必须包含 Round 71 names."""
        from src.translator_agent import _SRC_SUBMODULES
        attrs = _SRC_SUBMODULES["src.core.decision_engine"]
        assert "get_sibling_canonicals_by_source" in attrs
        assert "detect_cross_person_conflicts" in attrs

    def test_case_insensitive_find_exportable(self):
        """Round 71 fix: TranslationPipeline._case_insensitive_find 引用 _text_match_case_insensitive_find,
        整合脚本必须导出 case_insensitive_find 并通过别名兼容."""
        from src.translator_agent import (
            case_insensitive_find, _text_match_case_insensitive_find,
        )
        assert callable(case_insensitive_find)
        # pipeline.py:30 写的是 `case_insensitive_find as _text_match_case_insensitive_find`
        # 整合脚本顶部必须有相同别名, 否则 _case_insensitive_find 触发 NameError
        assert _text_match_case_insensitive_find is case_insensitive_find

    def test_text_match_helpers_full_set_exported(self):
        """build_section_5b 应同时导出 Round 70 + Round 71 helpers."""
        from src.translator_agent import (
            extract_reject_categories, _normalize_category, _CATEGORY_SYNONYMS,
            case_insensitive_find,
        )
        assert callable(extract_reject_categories)
        assert callable(_normalize_category)
        assert isinstance(_CATEGORY_SYNONYMS, dict)
        assert callable(case_insensitive_find)


class TestBuildScriptWiring:
    """scripts/build_integrated.py 必须打包 Round 71 helpers."""

    def test_build_section_4_includes_sibling_helpers(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_integrated", "scripts/build_integrated.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.build_section_4()
        assert "get_sibling_canonicals_by_source" in out
        assert "detect_cross_person_conflicts" in out
