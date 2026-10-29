"""Round 125: 新增 secondary(Qwen) 角色分流 MiniMax 用量。

背景: ch09 双进程事故证明 MiniMax 配额是单点瓶颈。经统计, critic_scoring
（审计评分, 含 typesafe openai_compat 降级与 legacy 双路径共用同一 client）
是 MiniMax 最大单一负载; 任务形态属"大批量、模式化、结构化评分", 适合
Flash 档模型; 且与 judge(MiniMax) 异家族可降低同源盲区。

布局（Round 125；具体角色↔模型绑定以 config.yaml 为准，Round 127 用户已重排）:
  - critic_scoring        → secondary (Qwen；重排后当前为 DeepSeek 网关)
  - reference_confirmation→ secondary (Qwen；同上)
  - judge_decision / style_profiling / editor_review / 难度直译 → 仍 primary (MiniMax)
本测试锁定路由与角色定义（含双落点 PRICING 同步）。
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def fresh_config():
    """支持 env 注入后 reload，并在结束后恢复真实配置。"""
    from src.config import config

    def _reload():
        config.reload()
        return config

    yield _reload
    config.reload()


class TestSecondaryRole:
    def test_role_defined(self, fresh_config):
        cfg = fresh_config()
        role = cfg.get_role_config("secondary")
        assert role, "main.models.secondary 必须存在"
        assert role.get("backend") == "openai_api"
        assert role.get("model_name")
        assert role.get("api_base", "").startswith("http")

    def test_api_key_env_wired(self, fresh_config, monkeypatch):
        """角色 api_key 的 ${ENV} 占位符须与 shell/.env 注入联动。

        不锁定"角色↔具体环境变量名"的绑定：读取 config.yaml 声明的占位符，
        逐一声明注入再断言解析结果（Round 127 起绑定可随用户重排）。
        """
        import yaml as _yaml
        from pathlib import Path as _Path

        cfg = fresh_config()
        raw_models = (_yaml.safe_load(
            (_Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
        ) or {}).get("models", {}) or {}
        checked = 0
        for role_key, block in raw_models.items():
            raw = str((block or {}).get("api_key", ""))
            if not (raw.startswith("${") and raw.endswith("}")):
                continue
            env_name = raw[2:-1]
            sentinel = f"test-{role_key}-key"
            monkeypatch.setenv(env_name, sentinel)
            cfg.reload()
            assert cfg.get_role_config(role_key).get("api_key") == sentinel, role_key
            checked += 1
        assert checked >= 3, "models 段各角色的 api_key 都应是 ${ENV} 占位符"

    def test_model_name_resolution(self, fresh_config):
        from src.utils.llm_adapter import get_role_model_name
        cfg = fresh_config()
        resolved = get_role_model_name("secondary")
        # 模型名由用户按自建网关命名填写，可随时替换；此处锁定"配置 -> 解析链"一致且非占位。
        assert resolved == cfg.get_role_config("secondary").get("model_name")
        assert resolved and "TODO" not in resolved and "placeholder" not in resolved.lower()


class TestRoutingAfterRound125:
    def test_critic_routes_to_secondary(self, fresh_config):
        cfg = fresh_config()
        assert cfg.get_task_role("critic_scoring") == "secondary"
        model_key, params = cfg.resolve_task_model("critic_scoring")
        assert model_key == "secondary"
        assert params.get("temperature") == 0.1
        assert params.get("effort") == "none"   # DeepSeek 档位：结构化评分不需要推理（Round 129 改名）

    def test_reference_confirmation_routes_to_secondary(self, fresh_config):
        assert fresh_config().get_task_role("reference_confirmation") == "secondary"

    def test_quality_gates_stay_on_primary(self, fresh_config):
        cfg = fresh_config()
        assert cfg.get_task_role("judge_decision") == "primary", "最终裁决必须留在最强模型"
        assert cfg.get_task_role("style_profiling") == "primary"
        assert cfg.get_task_role("editor_review") == "primary"
        hard_role = (cfg.pipeline.get("difficulty_routing") or {}).get("hard_role", "primary")
        assert hard_role == "primary", "困难片段直译仍用主力模型"

    def test_literal_unchanged_rewrite_moved_r129(self, fresh_config):
        cfg = fresh_config()
        assert cfg.get_task_role("literal_translation") == "literal_translator"
        # Round 129: 润色迁至 secondary（DS 大任务实测稳定 + 思考 max）
        assert cfg.get_task_role("literary_rewrite") == "secondary"


class TestPricingSync:
    def test_pricing_includes_secondary(self):
        import inspect
        from src.pipeline import TranslationPipeline as Module
        from src.translator_agent import TranslationPipeline as Monolith

        for cls in (Module, Monolith):
            source = inspect.getsource(cls._print_llm_cost_report)
            assert '"secondary"' in source, f"{cls.__module__} PRICING 缺 secondary"


class TestEnvHygiene:
    """防 .env 粘贴事故: 值首尾残留引号会把 key 弄脏 → 401（Round 125 实际发生过）。"""

    def test_resolved_keys_have_no_stray_quotes(self, fresh_config):
        cfg = fresh_config()
        for role in ("literal_translator", "secondary"):
            key = str(cfg.get_role_config(role).get("api_key", ""))
            if not key or key.startswith("${"):
                pytest.skip(f"{role} key 未解析（无 .env 环境）")
            assert key == key.strip(), f"{role} key 含首尾空白"
            assert key[0] not in "\"'" and key[-1] not in "\"'", \
                f"{role} key 疑似残留引号（.env 引号未配对）"
