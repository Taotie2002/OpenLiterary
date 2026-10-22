"""Round 77: 修复 _load_typesafe_config 未读取 typesafe.model_name 配置的 bug.

根因: config.yaml 中 typesafe 节用 `model_name` 字段 (与全局 llm_backend 节命名一致),
      但 src/judges/typesafe_backend.py::_load_typesafe_config 在最后一段
        for k in ("primary_model", "fallback_role"):
            if k in section and section[k]:
                defaults[k] = str(section[k])
      读的是 `primary_model`, 不是 `model_name`. 结果: 用户改了 config.yaml
      的 `typesafe.model_name` 永远被静默忽略, 一直用硬编码默认 `jev-latest`.

修复: 优先读 `model_name`, 兼容旧 yaml 用 `primary_model`. 内部 key 仍叫
      `primary_model` (供 FallbackBackend.__init__ 消费, 不改接口).

影响: 让 typesafe 后端的模型选择真正可配置 (与全局 llm_backend/llm_openai_api
      命名约定统一, 降低用户配置困惑).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
    # Round 77: setup 时主动清理 Round 75/76 cleanup 留下的孤儿模块.
    # Round 75/76 fixture 把 `src.judges` 父包从 sys.modules pop 掉, 但
    # `src.judges.typesafe_backend` 仍在 sys.modules 中残留, 变成
    # __file__=None 的孤儿对象 — 下次 `from src.judges.typesafe_backend import _X`
    # 会报 ImportError "cannot import name ... (unknown location)" 或
    # ModuleNotFoundError "'src.judges' is not a package".
    # 解决: 在 setup 时主动 pop 这个孤儿模块 + 重建 src.judges 父包.
    sys.modules.pop("src.judges.typesafe_backend", None)
    try:
        import src.judges  # noqa: F401  强制父包注册到 sys.modules
    except ImportError:
        pass
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    # 不要 pop `src.judges` 父包 — Round 75 fixture 的 cleanup 会重建它,
    # 但如果其他测试后续清理时再次干掉, 下一个测试的 import 会失败.
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



def _load_module_isolated():
    """Round 77: 直接从文件加载 typesafe_backend, 绕过 sys.modules 父包状态.

    背景: Round 75/76 fixture cleanup 把 `src.judges` 父包从 sys.modules pop 掉,
    但 `src.judges.typesafe_backend` 子模块可能残留, 导致后续 import 报
    ImportError "cannot import name ... (unknown location)" 或
    ModuleNotFoundError "'src.judges' is not a package".
    这里用 importlib.util.spec_from_file_location 完全跳过 sys.modules.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_typesafe_backend_isolated",
        "src/judges/typesafe_backend.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSourcePatch:
    """散装 src/judges/typesafe_backend.py 必须应用 Round 77 修复."""

    def test_round_77_marker_present(self):
        content = Path("src/judges/typesafe_backend.py").read_text(encoding="utf-8")
        assert "Round 77" in content

    def test_model_name_field_is_read(self):
        """核心修复: 必须读 `model_name`, 不能只读 `primary_model`."""
        content = Path("src/judges/typesafe_backend.py").read_text(encoding="utf-8")
        func_start = content.find("def _load_typesafe_config")
        assert func_start > 0, "_load_typesafe_config 必须存在"
        func_body = content[func_start:func_start + 2500]
        # 必须出现 `model_name` 字面量
        assert '"model_name"' in func_body or "'model_name'" in func_body, (
            "Round 77: _load_typesafe_config 必须读取 model_name 字段"
        )
        # 不能只 read `primary_model` (round 77 之前的行为)
        # 检查 for k in ("primary_model", "fallback_role") 这种旧模式不存在
        assert 'for k in ("primary_model", "fallback_role")' not in func_body, (
            "Round 77: 旧的 for k 循环必须删除, 改为读 model_name 字段"
        )

    def test_syntax_check(self):
        """防止 syntax error (历史教训)."""
        import ast
        ast.parse(Path("src/judges/typesafe_backend.py").read_text(encoding="utf-8"))


class TestIntegratedScript:
    """整合脚本 src/translator_agent.py 必须同步应用 Round 77 修复."""

    def test_typesafe_backend_importable(self):
        from src.translator_agent import _load_typesafe_config
        assert _load_typesafe_config is not None

    def test_round_77_present_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 77" in content, "整合脚本未同步 Round 77 修复"


class TestLoadTypesafeConfigBehavior:
    """_load_typesafe_config 行为测试 - 覆盖关键场景."""

    def test_model_name_in_config_wins(self):
        """场景 1: config.yaml 中 model_name 配置生效."""
        _mod = _load_module_isolated()
        _load_typesafe_config = _mod._load_typesafe_config

        fake_section = {
            "model_name": "jev-2.0-flash",
            "fallback_role": "primary",
            "enabled": True,
        }
        with patch("src.config.Config") as MockConfig:
            MockConfig.return_value.get_section.return_value = fake_section
            cfg = _load_typesafe_config()

        # 内部 key 仍是 `primary_model`, 但值应来自 yaml 的 `model_name`
        assert cfg["primary_model"] == "jev-2.0-flash", (
            f"Round 77: config.yaml 的 model_name 必须生效, 实际={cfg['primary_model']!r}"
        )

    def test_primary_model_legacy_fallback(self):
        """场景 2 (向后兼容): 旧 yaml 用 `primary_model` 时仍能读取."""
        _mod = _load_module_isolated()
        _load_typesafe_config = _mod._load_typesafe_config

        fake_section = {
            # 没有 model_name, 只有 legacy primary_model
            "primary_model": "jev-legacy",
            "fallback_role": "primary",
        }
        with patch("src.config.Config") as MockConfig:
            MockConfig.return_value.get_section.return_value = fake_section
            cfg = _load_typesafe_config()

        assert cfg["primary_model"] == "jev-legacy", (
            "向后兼容: 旧 yaml 用 primary_model 时仍能读取"
        )

    def test_model_name_overrides_primary_model(self):
        """场景 3: 同时配置两者时, model_name 优先 (与 config.yaml 当前约定一致)."""
        _mod = _load_module_isolated()
        _load_typesafe_config = _mod._load_typesafe_config

        fake_section = {
            "model_name": "jev-new",
            "primary_model": "jev-old-legacy",  # 应该被忽略
        }
        with patch("src.config.Config") as MockConfig:
            MockConfig.return_value.get_section.return_value = fake_section
            cfg = _load_typesafe_config()

        assert cfg["primary_model"] == "jev-new", (
            "model_name 必须优先于 primary_model (config.yaml 当前约定)"
        )

    def test_no_model_field_uses_default(self):
        """场景 4: 都没配时, 使用硬编码默认 `jev-latest`."""
        _mod = _load_module_isolated()
        _load_typesafe_config = _mod._load_typesafe_config

        fake_section = {"fallback_role": "primary"}
        with patch("src.config.Config") as MockConfig:
            MockConfig.return_value.get_section.return_value = fake_section
            cfg = _load_typesafe_config()

        assert cfg["primary_model"] == "jev-latest", (
            "未配置 model 字段时应保持向后兼容默认 jev-latest"
        )

    def test_fallback_role_still_works(self):
        """回归: fallback_role 字段读取未被破坏."""
        _mod = _load_module_isolated()
        _load_typesafe_config = _mod._load_typesafe_config

        fake_section = {
            "model_name": "jev-latest",
            "fallback_role": "backup_role",
        }
        with patch("src.config.Config") as MockConfig:
            MockConfig.return_value.get_section.return_value = fake_section
            cfg = _load_typesafe_config()

        assert cfg["fallback_role"] == "backup_role"

    def test_other_fields_still_work(self):
        """回归: enabled / allow_mock / allow_openai_fallback / timeout 字段未被破坏."""
        _mod = _load_module_isolated()
        _load_typesafe_config = _mod._load_typesafe_config

        fake_section = {
            "model_name": "jev-latest",
            "enabled": False,
            "allow_mock": False,
            "allow_openai_fallback": True,
            "timeout": 60.0,
        }
        with patch("src.config.Config") as MockConfig:
            MockConfig.return_value.get_section.return_value = fake_section
            cfg = _load_typesafe_config()

        assert cfg["enabled"] is False
        assert cfg["allow_mock"] is False
        assert cfg["allow_openai_fallback"] is True
        assert cfg["timeout"] == 60.0

    def test_config_section_missing(self):
        """场景 5: typesafe 节完全缺失 (None) → 使用全部默认."""
        _mod = _load_module_isolated()
        _load_typesafe_config = _mod._load_typesafe_config

        with patch("src.config.Config") as MockConfig:
            MockConfig.return_value.get_section.return_value = None
            cfg = _load_typesafe_config()

        # 应使用 defaults
        assert cfg["primary_model"] == "jev-latest"
        assert cfg["fallback_role"] == "primary"
        assert cfg["enabled"] is True
        assert cfg["timeout"] == 30.0
