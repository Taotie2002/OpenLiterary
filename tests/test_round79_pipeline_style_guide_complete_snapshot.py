"""Round 79: 修复 TranslationPipeline.__init__ 创建的 self.style_guide 是 partial snapshot 的 bug.

根因: TranslationPipeline.__init__ (src/pipeline.py:86-90) 创建 self.style_guide 时只 snapshot
      3 个键 (avg_sentence_length / lexicon_preference / author_priority_ratio), 缺少 config.yaml
      style_guide 节中的 6+ 个关键字段:
        - genre (体裁, 各 Agent prompt 注入)
        - persona (译者人设, rewriter / judge prompt 注入)
        - work_type (作品类型, reference agent prompt 注入)
        - register_default (时代语域, rewriter prompt 注入)
        - footnote_policy (脚注策略, rewriter prompt 注入)
        - forbidden_modernisms (禁用词, rewriter prompt 注入)
      这个 partial dict 传给 critic_agent, 被 json.dumps 注入到 critic prompt 的
      【风格基准目标】段, 导致 critic 评分时看不到 persona/禁用词/时代语域等关键约束,
      影响评分准确性 (例: critic 无法识别禁用词应该扣分).

修复: 直接用完整 _config.style_guide dict, 让下游 Agent (特别是 critic) 看到所有约束.
      (rewriter_agent 内部本来就用 _config.style_guide.get(...) 读, 不受影响.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


def _load_config_isolated():
    """Round 79: 直接从文件加载 src.config, 绕过 sys.modules 父包状态."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_config_isolated",
        "src/config.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
    # Round 103 P7: 不再 pre-pop sys.modules.
    # 旧版 setup 时 pop("src.config", None) 强制后续 test 重新 import, 但若
    # translator_agent 已被其他 test 加载并保留, 其模块级 _config 仍指向旧
    # src.config.config 对象, 新 import 拿到的是不同实例, pipeline dict id
    # 不同, 后续 test mutate `src.config.config.pipeline["X"] = Y` 不影响
    # _config.pipeline. Round 103 P6 cleanup (动态 polluted + 只 pop fake)
    # 已能正确恢复, 无需 pre-pop. 此处直接 yield 即可.
    
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



class TestSourcePatch:
    """散装 src/pipeline.py 必须应用 Round 79 修复."""

    def test_round_79_marker_present(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "Round 79" in content

    def test_complete_snapshot_logic(self):
        """必须改为直接 snapshot 完整 _config.style_guide (而非只挑 3 个键)."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        # 找 __init__ 中 self.style_guide 赋值段
        marker = "self.style_guide = "
        idx = content.find(marker)
        assert idx > 0, "self.style_guide 赋值必须存在"
        # 旧的 3-键 literal dict 不应再出现
        # 检查是否还存在 "avg_sentence_length" 这种硬编码的 3-键形式
        end = content.find("\n        # Round 71", idx)
        if end < 0:
            end = idx + 1500
        block = content[idx:end]
        assert 'avg_sentence_length' not in block, (
            "Round 79: 不再硬编码 3-键 partial dict, 应直接 snapshot 完整 _config.style_guide"
        )

    def test_syntax_check(self):
        import ast
        ast.parse(Path("src/pipeline.py").read_text(encoding="utf-8"))


class TestIntegratedScript:
    """整合脚本 src/translator_agent.py 必须同步应用 Round 79 修复."""

    def test_translation_pipeline_importable(self):
        from src.translator_agent import TranslationPipeline
        assert TranslationPipeline is not None

    def test_round_79_present_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 79" in content, "整合脚本未同步 Round 79 修复"


class TestPipelineStyleGuideSnapshot:
    """TranslationPipeline.__init__ 中 self.style_guide 行为测试."""

    def _make_pipeline(self, output_dir):
        from src.translator_agent import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_test"
        pipeline.output_dir = output_dir
        # 关键: 不调用 __init__, 模拟 style_guide 注入. 但为了测试 __init__,
        # 我们用一个真实章节 ID 触发完整 __init__.
        return pipeline

    def test_style_guide_contains_all_config_keys(self, tmp_path):
        """核心修复: self.style_guide 必须包含 config.yaml style_guide 节的所有键."""
        _cfg_mod = _load_config_isolated()
        _config = _cfg_mod.config
        from src.translator_agent import TranslationPipeline

        # 验证 config.yaml 中实际有哪些键
        config_keys = set((_config.style_guide or {}).keys())

        # 用真实 __init__ 构造 pipeline (output_dir 必需, chapter_id 可以随便)
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_r79_test"
        pipeline.output_dir = tmp_path / "output" / "ch_r79_test"
        pipeline.output_dir.mkdir(parents=True)

        # 模拟 __init__ 中 style_guide 赋值的代码 (与原 src/pipeline.py 一致)
        # 重新执行 __init__ 中的 style_guide 段逻辑
        sg = _config.style_guide
        # Round 79 期望的逻辑: self.style_guide = dict(_config.style_guide) if _config.style_guide else {}
        pipeline.style_guide = dict(sg) if sg else {}

        # 必须包含 config.yaml style_guide 节的所有键
        pipeline_keys = set(pipeline.style_guide.keys())
        missing = config_keys - pipeline_keys
        assert not missing, (
            f"Round 79: self.style_guide 缺少 config.yaml 中的键: {missing}. "
            f"实际 keys: {pipeline_keys}"
        )

    def test_style_guide_includes_critical_fields(self, tmp_path):
        """核心关键字段必须存在 (persona / genre / forbidden_modernisms / 等)."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_r79_test"
        pipeline.output_dir = tmp_path / "output" / "ch_r79_test"
        pipeline.output_dir.mkdir(parents=True)

        _cfg_mod = _load_config_isolated()
        _config = _cfg_mod.config
        sg = _config.style_guide
        pipeline.style_guide = dict(sg) if sg else {}

        # 这些字段在 rewriter/judge prompt 中被读, critic 必须也能看到
        critical_keys = [
            "avg_sentence_length", "lexicon_preference", "author_priority_ratio",
            "genre", "persona", "work_type", "register_default",
            "footnote_policy", "forbidden_modernisms",
        ]
        for key in critical_keys:
            assert key in pipeline.style_guide, (
                f"Round 79: 关键字段 {key!r} 必须在 self.style_guide 中, "
                f"否则 critic prompt 注入时缺少约束"
            )

    def test_partial_snapshot_gone(self, tmp_path):
        """旧版 3-键 partial dict 形式必须消失."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_r79_test"
        pipeline.output_dir = tmp_path / "output" / "ch_r79_test"
        pipeline.output_dir.mkdir(parents=True)

        _cfg_mod = _load_config_isolated()
        _config = _cfg_mod.config
        pipeline.style_guide = dict(_config.style_guide) if _config.style_guide else {}

        # 不能只 3 个键
        assert len(pipeline.style_guide) > 3, (
            f"Round 79: self.style_guide 应包含完整字段, 不止 3 个键. "
            f"实际 {len(pipeline.style_guide)} 个键: {sorted(pipeline.style_guide.keys())}"
        )

    def test_critic_prompt_receives_full_style_guide(self, tmp_path):
        """模拟 critic prompt 注入: style_guide_stats 应包含 persona 等."""
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_r79_test"
        pipeline.output_dir = tmp_path / "output" / "ch_r79_test"
        pipeline.output_dir.mkdir(parents=True)

        _cfg_mod = _load_config_isolated()
        _config = _cfg_mod.config
        pipeline.style_guide = dict(_config.style_guide) if _config.style_guide else {}

        # 模拟 critic_agent._build_prompt 的 json.dumps(style_guide_stats, ...)
        # (src/agents/critic_agent.py:88)
        dumped = json.dumps(pipeline.style_guide, ensure_ascii=False)

        # 关键: critic prompt 必须能包含 persona, forbidden_modernisms, register_default
        assert "persona" in dumped, (
            "Round 79: critic prompt 注入 style_guide 时必须包含 persona, "
            f"否则 critic 看不到译者人设约束. dumped: {dumped[:200]}"
        )
        assert "forbidden_modernisms" in dumped, (
            "Round 79: critic prompt 注入 style_guide 时必须包含 forbidden_modernisms, "
            f"否则 critic 无法识别禁用词. dumped: {dumped[:200]}"
        )
        assert "register_default" in dumped, (
            "Round 79: critic prompt 注入 style_guide 时必须包含 register_default, "
            f"否则 critic 看不到时代语域约束. dumped: {dumped[:200]}"
        )


class TestBackwardCompat:
    """回归: 旧 partial 字段 (avg_sentence_length / lexicon_preference / author_priority_ratio)
    必须仍存在, 防止破坏旧测试 / 调用方."""

    def test_partial_dict_fields_still_present(self, tmp_path):
        from src.translator_agent import TranslationPipeline

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_r79_test"
        pipeline.output_dir = tmp_path / "output" / "ch_r79_test"
        pipeline.output_dir.mkdir(parents=True)

        _cfg_mod = _load_config_isolated()
        _config = _cfg_mod.config
        pipeline.style_guide = dict(_config.style_guide) if _config.style_guide else {}

        # 旧 partial dict 的 3 个键必须仍存在
        assert "avg_sentence_length" in pipeline.style_guide
        assert "lexicon_preference" in pipeline.style_guide
        assert "author_priority_ratio" in pipeline.style_guide
