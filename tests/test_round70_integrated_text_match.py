"""Round 70: 回归测试 — 整合脚本 src/translator_agent.py 必须从 utils/text_match 导出
reject category helpers, 避免 _extract_reject_categories 触发 NameError.

保护目标：
  - 整合脚本未打包 utils/text_match.py 时,LiteraryRewriterAgent._extract_reject_categories
    和 TranslationPipeline._extract_reject_categories 会触发 NameError:
      name 'extract_reject_categories' is not defined
  - 修复: scripts/build_integrated.py 添加 build_section_5b(), 在整合脚本顶层导出
    _CATEGORY_SYNONYMS / _normalize_category / extract_reject_categories
  - 测试锁定 (1) 这些符号可从 src.translator_agent 导入,
                 (2) thin-shim 方法不再 NameError,
                 (3) 同义映射行为正常.
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
    """src.translator_agent 在 import 时会向 sys.modules 注入合成子模块
    (src.pipeline / src.splitters / ...), 破坏后续测试对真实散装模块的 import.

    本 fixture 把受影响的 key 隔离, 测试结束后恢复, 避免污染其他测试.
    """
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


class TestIntegratedTextMatchExports:
    """utils/text_match.py 中的关键符号必须存在于 src.translator_agent 全局命名空间"""

    def test_extract_reject_categories_importable(self):
        from src.translator_agent import extract_reject_categories
        assert callable(extract_reject_categories)

    def test_normalize_category_importable(self):
        from src.translator_agent import _normalize_category
        assert callable(_normalize_category)

    def test_category_synonyms_dict_importable(self):
        from src.translator_agent import _CATEGORY_SYNONYMS
        assert isinstance(_CATEGORY_SYNONYMS, dict)
        # 必须包含 Round 69 的全部同义映射条目
        assert "实体一致性硬失败" in _CATEGORY_SYNONYMS
        assert "实体一致性硬失败(降级)" in _CATEGORY_SYNONYMS
        assert "实体一致性违反(per-entity 聚合)" in _CATEGORY_SYNONYMS
        assert "音色漂移（severe）" in _CATEGORY_SYNONYMS
        # 同义映射的目标必须是规范名称
        assert _CATEGORY_SYNONYMS["实体一致性硬失败"] == "实体一致性违反"
        assert _CATEGORY_SYNONYMS["实体一致性硬失败(降级)"] == "实体一致性违反"


class TestIntegratedTextMatchBehavior:
    """导入后行为必须与散装源 src/utils/text_match.py 一致"""

    def test_normalize_category_collapses_hard_fail_variants(self):
        from src.translator_agent import _normalize_category
        assert _normalize_category("实体一致性硬失败") == "实体一致性违反"
        assert _normalize_category("实体一致性硬失败(降级)") == "实体一致性违反"
        assert _normalize_category("实体一致性违反(per-entity 聚合)") == "实体一致性违反"
        # 未知类别原样返回
        assert _normalize_category("直译残留") == "直译残留"

    def test_extract_reject_categories_normalizes_synonyms(self):
        from src.translator_agent import extract_reject_categories
        # as_set=False -> list (LiteraryRewriterAgent 路径)
        ordered = extract_reject_categories(
            "实体一致性硬失败(降级): foo; 音色漂移: bar",
            as_set=False,
        )
        assert ordered == ["实体一致性违反", "音色漂移"]
        # as_set=True -> set (TranslationPipeline 路径)
        as_set = extract_reject_categories(
            "实体一致性硬失败(降级): foo; 音色漂移: bar",
            as_set=True,
        )
        assert as_set == {"实体一致性违反", "音色漂移"}

    def test_extract_reject_categories_dedups_via_synonym(self):
        """Round 69: 同一规范类别即使出现多种 variant label 也只计 1 个"""
        from src.translator_agent import extract_reject_categories
        result = extract_reject_categories(
            "实体一致性硬失败: a; 实体一致性硬失败(降级): b; 实体一致性违反: c",
            as_set=True,
        )
        assert result == {"实体一致性违反"}

    def test_extract_reject_categories_handles_empty(self):
        from src.translator_agent import extract_reject_categories
        assert extract_reject_categories("", as_set=False) == []
        assert extract_reject_categories("", as_set=True) == set()


class TestIntegratedThinShimMethods:
    """整合脚本中 LiteraryRewriterAgent / TranslationPipeline 的 thin-shim 不再 NameError"""

    def test_rewriter_thin_shim_no_nameerror(self):
        from src.translator_agent import LiteraryRewriterAgent
        # 绕过 __init__，直接构造裸实例，只为触发 bound method
        rewriter = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
        out = rewriter._extract_reject_categories("实体一致性硬失败(降级): x; 直译残留: y")
        assert out == ["实体一致性违反", "直译残留"]

    def test_pipeline_thin_shim_no_nameerror(self):
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        out = pipeline._extract_reject_categories(
            "实体一致性硬失败: x; 音色漂移（severe）: y",
        )
        assert out == {"实体一致性违反", "音色漂移"}


class TestBuildScriptWiring:
    """scripts/build_integrated.py 必须注册 build_section_5b 并将其加入 sections"""

    def test_build_section_5b_defined(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_integrated", "scripts/build_integrated.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "build_section_5b")
        assert callable(mod.build_section_5b)

    def test_build_section_5b_appended_in_build(self):
        """build() 必须按顺序 append build_section_5() -> build_section_5b() -> build_section_6()"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_integrated", "scripts/build_integrated.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # 用 inspect 拿到 build() 源码
        import inspect
        build_src = inspect.getsource(mod.build)
        # 检查顺序: section_5 之后必须有 section_5b, 之后才是 section_6
        i_5 = build_src.find("build_section_5()")
        i_5b = build_src.find("build_section_5b()")
        i_6 = build_src.find("build_section_6()")
        assert i_5 != -1, "build_section_5() not in build()"
        assert i_5b != -1, "build_section_5b() not appended in build()"
        assert i_6 != -1, "build_section_6() not in build()"
        assert i_5 < i_5b < i_6, (
            f"build_section_5b must be between section_5 ({i_5}) and section_6 ({i_6}); "
            f"found at {i_5b}"
        )

    def test_build_section_5b_includes_synonyms_dict(self):
        """build_section_5b() 输出必须包含 _CATEGORY_SYNONYMS 字面量"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_integrated", "scripts/build_integrated.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.build_section_5b()
        assert "_CATEGORY_SYNONYMS" in out
        assert "_normalize_category" in out
        assert "extract_reject_categories" in out
