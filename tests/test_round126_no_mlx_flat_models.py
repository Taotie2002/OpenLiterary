"""Round 126: 配置结构收敛（models 单一事实源）+ 移除本地 MLX 推理支持。

背景:
  - 同一模型原先写两处（main.models + openai_api.models），且含 mlx 本地推理段。
  - 用户决议: 今后不再使用本地模型；每个模型的 BASE_URL / API_KEY /
    model_name / Effort(thinking) 必须集中在一处。

修复:
  - config.yaml: 顶层 `models:` 为唯一模型定义处（含 thinking）；删除 `mlx:` 段与
    `openai_api.models` 重复段；task_routing 只在偏离模型默认时写 thinking。
  - src/config.py: resolve_task_model 把 thinking 归一为 enable_thinking；
    （Round 128 起 thinking 退役 → 任务级 effort，见 test_round128_effort_levels.py）
    get_role_config 注入 backend 默认值；删除 mlx_models/mlx_memory 属性。
  - src/utils/llm_adapter.py: 删除 MLXNativeAdapter / auto_unload_if_needed；
    get_role_model_name 只读 model_name。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolate_modules():
    polluted = tuple(k for k in list(sys.modules) if k == "src" or k.startswith("src."))
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)


@pytest.fixture
def fresh_config():
    from src.config import config
    config.reload()
    yield config
    config.reload()


class TestConfigSingleSource:
    def test_no_mlx_section(self, fresh_config):
        cfg = fresh_config
        assert "mlx" not in (cfg._data or {}), "config.yaml 不得再有 mlx: 段"
        assert not hasattr(cfg, "mlx_models"), "Config.mlx_models 应已删除"
        assert not hasattr(cfg, "mlx_memory"), "Config.mlx_memory 应已删除"

    def test_no_duplicate_model_pool(self, fresh_config):
        cfg = fresh_config
        assert cfg.get("main.models") is None, "旧 main.models 双落点应废弃"
        assert cfg.get("openai_api.models") is None, "openai_api.models 重复段应删除"

    def test_models_section_shape(self, fresh_config):
        cfg = fresh_config
        models = cfg.models
        assert set(models) >= {"literal_translator", "primary", "secondary"}
        for role, block in models.items():
            assert block.get("api_base", "").startswith("http"), f"{role}.api_base"
            assert block.get("api_key"), f"{role}.api_key"
            assert block.get("model_name"), f"{role}.model_name"
            assert "thinking" not in block, f"{role}.thinking 已废弃（Round 128 → 任务级 effort）"

    def test_role_config_backend_default(self, fresh_config):
        cfg = fresh_config
        for role in ("literal_translator", "primary", "secondary"):
            rc = cfg.get_role_config(role)
            assert rc.get("backend") == "openai_api", "backend 省略时由 config 归一"
            assert "extra_body" not in rc, "模型级不再承载 extra_body（Round 127 已迁到任务级）"


class TestEffortNormalization:
    """Round 128: effort 任务级归一（默认不下发；配置项原样进入 params，
    合法性校验与编码详见 tests/test_round128_effort_levels.py）。"""

    def test_literal_translation_explicit_none(self, fresh_config):
        """Round 129: 直译显式关闭思考（none），不再依赖服务端默认。"""
        cfg = fresh_config
        _, params = cfg.resolve_task_model("literal_translation")
        assert params.get("effort") == "none"

    def test_task_effort_applies(self, fresh_config):
        cfg = fresh_config
        _, params = cfg.resolve_task_model("judge_decision")
        assert params.get("effort") == "max"
        _, params = cfg.resolve_task_model("critic_scoring")
        assert params.get("effort") == "none"
        _, params = cfg.resolve_task_model("literary_rewrite")
        assert params.get("effort") == "max"

    def test_legacy_thinking_raises(self, fresh_config):
        cfg = fresh_config
        cfg.task_routing["judge_decision"]["params_override"]["thinking"] = True
        with pytest.raises(ValueError, match="thinking"):
            cfg.resolve_task_model("judge_decision")

    def test_no_raw_thinking_key_in_params(self, fresh_config):
        cfg = fresh_config
        for task in ("reference_extraction", "literal_translation", "literary_rewrite",
                     "critic_scoring", "judge_decision", "style_profiling", "editor_review"):
            _, params = cfg.resolve_task_model(task)
            assert "thinking" not in params
            assert "enable_thinking" not in params


class TestNoMLXInSource:
    FORBIDDEN = ("MLXNativeAdapter", "mlx_lm", "import mlx", "mlx.core")

    def test_no_mlx_code_references(self):
        roots = [PROJECT / "src", PROJECT / "scripts"]
        offenders = []
        for root in roots:
            for f in root.rglob("*.py"):
                if "__pycache__" in f.parts:
                    continue
                text = f.read_text(encoding="utf-8")
                for token in self.FORBIDDEN:
                    if token in text:
                        offenders.append(f"{f.relative_to(PROJECT)}:{token}")
        assert not offenders, f"仍存在 MLX 代码引用: {offenders}"


class TestAdapterAPI:
    def test_auto_unload_removed(self):
        from src.utils.llm_adapter import LLMAdapter
        assert not hasattr(LLMAdapter, "auto_unload_if_needed")

    def test_role_model_name_uses_model_name(self, fresh_config):
        from src.utils.llm_adapter import get_role_model_name
        assert get_role_model_name("secondary") == fresh_config.models["secondary"]["model_name"]
