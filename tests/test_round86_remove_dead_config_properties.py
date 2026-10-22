"""Round 86: 移除 Config.paths 和 Config.logging 死 property.

背景:
  src/config.py 中的 @property Config.paths 和 @property Config.logging 返回
  `self.get("paths", {})` 和 `self.get("logging", {})`. 这两个 property 在整
  个 codebase (src/ + tests/) 0 引用, 是纯粹的 dead code.

  历史: config.yaml 仍含 paths: 和 logging: 节, 但代码全部用硬编码路径
  (Path(__file__).resolve().parent.parent) 或默认 logging.INFO, 从未通过
  `config.paths` / `config.logging` 访问.

修复:
  - 移除 Config.paths @property
  - 移除 Config.logging @property
  - yaml paths: 和 logging: 节保留 (向后兼容运维 .yaml 配置)
  - 不影响任何现有行为 (0 引用)

验证:
  - Config 实例不再有 .paths 和 .logging 属性
  - 其它属性 (style_guide / openai_api / role_models 等) 不受影响
  - 散装 + 整合脚本双向同步
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_modules():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
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



# ---------------------------------------------------------------------------
# Round 86: Config 类不再有 paths 和 logging 属性
# ---------------------------------------------------------------------------

class TestDeadPropertiesGone:
    """Config 类不应再有 paths 和 logging 死 property."""

    def test_paths_attribute_removed(self):
        from src.config import Config
        cfg = Config()
        # 不再支持 .paths 属性 (AttributeError)
        with pytest.raises(AttributeError):
            _ = cfg.paths

    def test_logging_attribute_removed(self):
        from src.config import Config
        cfg = Config()
        # 不再支持 .logging 属性 (AttributeError)
        with pytest.raises(AttributeError):
            _ = cfg.logging

    def test_config_class_has_no_paths_in_dir(self):
        """Config 类的属性列表不应含 paths."""
        from src.config import Config
        # Check if 'paths' is in dir(Config) — should NOT be
        assert 'paths' not in dir(Config), (
            "Round 86: Config.paths 仍是属性, 应被移除"
        )

    def test_config_class_has_no_logging_in_dir(self):
        """Config 类的属性列表不应含 logging (避免与 stdlib logging 冲突)."""
        from src.config import Config
        assert 'logging' not in dir(Config), (
            "Round 86: Config.logging 仍是属性, 应被移除 (避免与 stdlib logging 冲突)"
        )


# ---------------------------------------------------------------------------
# Round 86: 其它 active properties 不受影响 (回归测试)
# ---------------------------------------------------------------------------

class TestActivePropertiesIntact:
    """其它 active properties 仍可访问, 不受 Round 86 影响."""

    def test_style_guide_still_works(self):
        from src.config import Config
        cfg = Config()
        # style_guide 是 active property, 应返回 dict
        result = cfg.style_guide
        assert isinstance(result, dict)

    def test_openai_api_still_works(self):
        from src.config import Config
        cfg = Config()
        result = cfg.openai_api
        assert isinstance(result, dict)

    def test_task_routing_still_works(self):
        from src.config import Config
        cfg = Config()
        result = cfg.task_routing
        assert isinstance(result, dict)

    def test_llm_backend_still_works(self):
        from src.config import Config
        cfg = Config()
        result = cfg.llm_backend
        assert isinstance(result, str)

    def test_critic_thresholds_still_works(self):
        from src.config import Config
        cfg = Config()
        result = cfg.critic_thresholds
        assert isinstance(result, dict)

    def test_typesafe_thresholds_still_works(self):
        from src.config import Config
        cfg = Config()
        result = cfg.typesafe_thresholds
        assert isinstance(result, dict)

    def test_role_models_still_works(self):
        from src.config import Config
        cfg = Config()
        # has_role_models() 不应因 Round 86 而坏
        result = cfg.has_role_models()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Round 86: 整合脚本 src/translator_agent.py 同步
# ---------------------------------------------------------------------------

class TestIntegratedScriptSynced:
    """散装 src/config.py 修复必须同步到整合脚本."""

    def test_round_86_marker_in_scattered(self):
        content = Path("src/config.py").read_text(encoding="utf-8")
        assert "Round 86" in content, "散装 config.py 未标注 Round 86 修复"

    def test_round_86_no_paths_in_builtin_config(self):
        """整合脚本 _BuiltinConfig 类 (单文件部署 fallback) 也不应有 paths/logging.

        历史: _BuiltinConfig 一直是 stub class, 没有 @property decorator,
        也没有 paths / logging 类属性. 它从未用过这两个字段 (0 引用).
        Round 86 同步语义: 与 src.config.Config 一致 —— 都不再有 paths/logging.
        """
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        # 找 _BuiltinConfig 类, 验证类内无 paths 和 logging 属性
        idx = content.find("class _BuiltinConfig")
        assert idx > 0, "_BuiltinConfig 类应存在"
        # 找下一个类开始位置
        NL = chr(10)
        next_idx = content.find(NL + "class ", idx + 10)
        if next_idx < 0:
            next_idx = content.find(NL + "# Section", idx + 10)
        if next_idx < 0:
            next_idx = idx + 5000  # fallback
        builtin_block = content[idx:next_idx]
        # 检查不应有 paths = 或 logging = (作为类属性, 8 空格缩进)
        assert "        paths =" not in builtin_block, (
            "Round 86: _BuiltinConfig 不应有 paths 类属性"
        )
        assert "        logging =" not in builtin_block, (
            "Round 86: _BuiltinConfig 不应有 logging 类属性"
        )

    def test_paths_property_def_not_in_source(self):
        """散装 src/config.py 不应有 'def paths(self)'."""
        content = Path("src/config.py").read_text(encoding="utf-8")
        assert "    def paths(self)" not in content, (
            "Round 86: src/config.py 不应再有 'def paths(self)' 方法"
        )

    def test_logging_property_def_not_in_source(self):
        """散装 src/config.py 不应有 'def logging(self)'."""
        content = Path("src/config.py").read_text(encoding="utf-8")
        assert "    def logging(self)" not in content, (
            "Round 86: src/config.py 不应再有 'def logging(self)' 方法"
        )

    def test_paths_property_def_not_in_integrated(self):
        """整合脚本不应有 'def paths(self)' (因为已同步)."""
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        # 整合脚本 class _BuiltinConfig (header 内) 不应有 paths property
        # 注: 其它 class 可能有 paths 字段 (e.g., Path) 但不是 Config paths
        assert "    def paths(self)" not in content, (
            "Round 86: 整合脚本不应再有 'def paths(self)'"
        )

    def test_logging_property_def_not_in_integrated(self):
        """整合脚本不应有 'def logging(self)'."""
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "    def logging(self)" not in content, (
            "Round 86: 整合脚本不应再有 'def logging(self)'"
        )


# ---------------------------------------------------------------------------
# Round 86: config.yaml 保留向后兼容 (paths: 和 logging: 节仍存在)
# ---------------------------------------------------------------------------

class TestYamlStillBackwardCompatible:
    """yaml paths: 和 logging: 节仍保留, 运维部署不破坏."""

    def test_yaml_paths_section_present(self):
        content = Path("config.yaml").read_text(encoding="utf-8")
        assert "paths:" in content, (
            "Round 86: config.yaml 不应删除 paths: 节 (向后兼容)"
        )

    def test_yaml_logging_section_present(self):
        content = Path("config.yaml").read_text(encoding="utf-8")
        assert "logging:" in content, (
            "Round 86: config.yaml 不应删除 logging: 节 (向后兼容)"
        )

    def test_yaml_paths_db_dir_defined(self):
        """config.yaml paths 节仍含 db_dir 配置 (运维可覆盖)."""
        content = Path("config.yaml").read_text(encoding="utf-8")
        assert "db_dir:" in content, (
            "Round 86: config.yaml paths 节应保留 db_dir 配置"
        )

    def test_yaml_logging_level_defined(self):
        """config.yaml logging 节仍含 level 配置."""
        content = Path("config.yaml").read_text(encoding="utf-8")
        # 找 logging: 之后的内容
        idx = content.find("logging:")
        assert idx > 0
        after = content[idx:idx+200]
        assert "level:" in after, (
            "Round 86: config.yaml logging 节应保留 level 配置"
        )
