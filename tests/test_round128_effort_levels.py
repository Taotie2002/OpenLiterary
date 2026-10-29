"""Round 128/129: thinking 布尔开关退役 —— 统一为任务级 effort 档位。

决策（用户）:
  - 档位即开关：不写 effort = 不下发参数；合法档位逐家
    （src/config.EFFORT_LEVELS_BY_PROVIDER）：
      MiniMax low/medium/high/xhigh/max；DeepSeek none/low/high/xhigh/max；
      Qwen none/low/medium/xhigh（none = 显式关闭思考；Round 129 实测）。
  - 编码（src.utils.llm_adapter.build_effort_params）：reasoning_effort: <level>；
    MiniMax 另附 thinking=adaptive（服务端强制，R121 实测）。
  - 旧 `thinking`/`enable_thinking` 键出现在配置（models 段或 params_override）→ resolve 报错。

覆盖:
  - 任务路由档位表 / 非法档位 fail fast / 旧键 fail fast / 模型级 effort 拒绝；
  - 兼容位置 params_override.effort 与规范位置优先级；
  - build_effort_params 编码矩阵（散装 + 单体镜像）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent

TASK_EFFORT_EXPECTED = {
    # Round 132 调整：reference_extraction→none（Qwen low 推理失控撞 300s 网关）；
    # reference_confirmation→low（max 档实测超 300s 被掐断）。此处与 config.yaml 同步。
    "reference_extraction": "none",
    "reference_confirmation": "low",
    "literary_rewrite": "max",
    "literal_translation": "none",
    "critic_scoring": "none",
    "judge_decision": "max",
    "style_profiling": "max",
    "editor_review": "max",
}


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


class TestResolveEffort:
    def test_task_effort_table(self, fresh_config):
        for task, expected in TASK_EFFORT_EXPECTED.items():
            _, params = fresh_config.resolve_task_model(task)
            assert params.get("effort") == expected, task

    def test_literal_translation_effort_none(self, fresh_config):
        """直译不需要思考 → Round 129 起显式 none（不再依赖服务端默认）。"""
        _, params = fresh_config.resolve_task_model("literal_translation")
        assert params.get("effort") == "none"

    def test_illegal_level_fails_fast(self, fresh_config):
        cfg = fresh_config
        cfg.task_routing["critic_scoring"]["effort"] = "medium"   # deepseek 无 medium
        with pytest.raises(ValueError, match="非法"):
            cfg.resolve_task_model("critic_scoring")
        cfg.task_routing["literary_rewrite"]["effort"] = "off"    # qwen 无 off（已退役档位名）
        with pytest.raises(ValueError, match="非法"):
            cfg.resolve_task_model("literary_rewrite")
        cfg.task_routing["judge_decision"]["effort"] = "none"     # minimax 无 none 档
        with pytest.raises(ValueError, match="非法"):
            cfg.resolve_task_model("judge_decision")

    def test_legal_edges(self, fresh_config):
        cfg = fresh_config
        for task, level in (("judge_decision", "low"),
                            ("reference_confirmation", "none"),
                            ("reference_extraction", "xhigh")):
            cfg.task_routing[task]["effort"] = level
            _, params = cfg.resolve_task_model(task)
            assert params["effort"] == level

    def test_bare_off_yaml_bool_coerced(self, fresh_config):
        cfg = fresh_config
        cfg.task_routing["critic_scoring"]["effort"] = False  # YAML 裸 off 的解析结果
        _, params = cfg.resolve_task_model("critic_scoring")
        assert params["effort"] == "none", "裸 off（YAML bool false）归一为 none"

    def test_bare_on_yaml_bool_rejected(self, fresh_config):
        cfg = fresh_config
        cfg.task_routing["critic_scoring"]["effort"] = True  # YAML 裸 on 的解析结果
        with pytest.raises(ValueError, match="布尔 true"):
            cfg.resolve_task_model("critic_scoring")

    def test_legacy_thinking_in_overrides_fails(self, fresh_config):
        cfg = fresh_config
        cfg.task_routing["judge_decision"]["params_override"]["thinking"] = True
        with pytest.raises(ValueError, match="thinking.*废弃"):
            cfg.resolve_task_model("judge_decision")

    def test_model_level_effort_rejected(self, fresh_config):
        cfg = fresh_config
        cfg.role_models["primary"]["effort"] = "max"
        with pytest.raises(ValueError, match="任务级"):
            cfg.resolve_task_model("judge_decision")

    def test_params_override_fallback_and_precedence(self, fresh_config):
        cfg = fresh_config
        route = cfg.task_routing["critic_scoring"]
        route.pop("effort", None)
        route["params_override"]["effort"] = "low"
        _, params = cfg.resolve_task_model("critic_scoring")
        assert params["effort"] == "low"
        route["effort"] = "high"
        _, params = cfg.resolve_task_model("critic_scoring")
        assert params["effort"] == "high", "规范位置（task 下 effort:）优先于 params_override"


class TestBuildEffortParams:
    @pytest.mark.parametrize("api_base,model_name,effort,expected", [
        ("https://api.minimax.cn/v1", "MiniMax-M3.1-Flash", "max",
         {"reasoning_effort": "max", "thinking": {"type": "adaptive"}}),
        ("https://api.minimax.cn/v1", "MiniMax-M3.1-Flash", None, {}),
        ("https://token.ai-hf.cn/v1", "Deepseek-V4.1-Flash-1017-SS", "none",
         {"reasoning_effort": "none"}),
        ("https://token.ai-hf.cn/v1", "Qwen3.8-Flash-Next-SS", "medium",
         {"reasoning_effort": "medium"}),
        ("https://example.com/v1", "unknown-model", "high", {"reasoning_effort": "high"}),
    ])
    def test_encoding_matrix(self, api_base, model_name, effort, expected):
        from src.utils.llm_adapter import build_effort_params
        assert build_effort_params(api_base, model_name, effort) == expected

    def test_normalizes_case_and_whitespace(self):
        from src.utils.llm_adapter import build_effort_params
        out = build_effort_params("https://token.ai-hf.cn/v1", "Qwen3.8-Flash-Next-SS", "  XHigh ")
        assert out == {"reasoning_effort": "xhigh"}


class TestMonolithMirror:
    def test_symbols(self):
        src = (PROJECT / "src" / "translator_agent.py").read_text(encoding="utf-8")
        assert "def build_effort_params(" in src, "单体须镜像 effort 编码函数"
        assert "build_thinking_params" not in src, "旧 thinking 归一函数不应残留"
        assert "enable_thinking" not in src, "enable_thinking 布尔开关应彻底退役"
