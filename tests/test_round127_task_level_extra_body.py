"""Round 127: extra_body 配置从模型级迁到任务级。

决策（用户）:
  - extra_body 只允许配置在 task_routing.<task> 下，不再从 models 段读取；
  - 规范位置 task_routing.<task>.extra_body；兼容 params_override.extra_body
    （两者都写时规范位置优先）；
  - resolve_task_model 返回值统一携带 extra_body dict，调用方直接 **params 透传，
    由适配层在按 effort 归一出的参数之后合并（同名覆盖）。

验证:
  - 默认所有任务 extra_body={}；
  - 任务级注入/兼容位置/优先级；模型级 extra_body 被忽略；
  - get_task_extra_body 读取语义与容错；
  - 单体（src/translator_agent.py）镜像：新符号存在、旧符号消失。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent

ALL_TASKS = [
    "reference_extraction", "reference_confirmation", "literal_translation",
    "literary_rewrite", "critic_scoring", "judge_decision", "style_profiling",
    "editor_review",
]


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


class TestTaskLevelExtraBody:
    def test_default_empty_for_all_tasks(self, fresh_config):
        for task in ALL_TASKS:
            _, params = fresh_config.resolve_task_model(task)
            assert params.get("extra_body") == {}, task

    def test_task_level_injection(self, fresh_config):
        cfg = fresh_config
        cfg.task_routing["literal_translation"]["extra_body"] = {"reasoning_effort": "low"}
        _, params = cfg.resolve_task_model("literal_translation")
        assert params["extra_body"] == {"reasoning_effort": "low"}
        _, other = cfg.resolve_task_model("critic_scoring")
        assert other["extra_body"] == {}, "未配置的任务不受影响"

    def test_params_override_fallback_position(self, fresh_config):
        cfg = fresh_config
        cfg.task_routing["critic_scoring"]["params_override"]["extra_body"] = {"thinking_budget": 4096}
        _, params = cfg.resolve_task_model("critic_scoring")
        assert params["extra_body"] == {"thinking_budget": 4096}

    def test_sibling_position_wins_over_nested(self, fresh_config):
        cfg = fresh_config
        route = cfg.task_routing["judge_decision"]
        route["extra_body"] = {"a": 1}
        route["params_override"]["extra_body"] = {"b": 2}
        _, params = cfg.resolve_task_model("judge_decision")
        assert params["extra_body"] == {"a": 1}

    def test_model_level_extra_body_is_ignored(self, fresh_config):
        cfg = fresh_config
        cfg.role_models["primary"]["extra_body"] = {"should": "be ignored"}
        _, params = cfg.resolve_task_model("judge_decision")
        assert params["extra_body"] == {}, "模型级 extra_body 不再被读取"


class TestGetTaskExtraBody:
    def test_reads_task_routing(self, fresh_config):
        from src.utils.llm_adapter import get_task_extra_body
        fresh_config.task_routing["editor_review"]["extra_body"] = {"reasoning_effort": "high"}
        assert get_task_extra_body("editor_review") == {"reasoning_effort": "high"}

    def test_missing_or_none_task_returns_empty(self, fresh_config):
        from src.utils.llm_adapter import get_task_extra_body
        assert get_task_extra_body("no_such_task") == {}
        assert get_task_extra_body(None) == {}


class TestMonolithMirror:
    def test_task_level_symbol_present_role_level_absent(self):
        src = (PROJECT / "src" / "translator_agent.py").read_text(encoding="utf-8")
        assert "def get_task_extra_body(" in src, "单体须镜像任务级 extra_body getter"
        assert "get_role_extra_body" not in src, "旧模型级 getter 不应残留于单体"
