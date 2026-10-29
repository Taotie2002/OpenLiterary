"""Round 129: MiniMax 配额韧性 + effort 档位更新（按 2026-10 实测口径）。

覆盖：
  - OpenAICompatibleAdapter: 429 单次重试 / 连续 429 → QuotaExhaustedError；
    402/403 带配额关键字 → QuotaExhaustedError；成功响应清零计数器；
  - FallbackRoleAdapter: 配额错误触发永久降级（effort 跨家映射 medium→high），
    其它错误不降级；降级后不再触碰主模型；
  - _build_role_client: models.primary.fallback_role=secondary 自动套包装器；
  - config: 档位/路由/fallback_role 的配置事实源校验；monolith 镜像符号。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent


def _mk_adapter(api_base: str):
    from src.utils.llm_adapter import OpenAICompatibleAdapter
    adapter = OpenAICompatibleAdapter(api_base=api_base, api_key="test-key", max_retries=3, retry_delay=0.01)
    adapter.min_request_interval = 0.0
    OpenAICompatibleAdapter._rate_limiters[adapter._get_limiter_key(api_base)]["interval"] = 0.0
    return adapter


class _Resp:
    def __init__(self, status_code: int, text: str = "", content: str = "正文"):
        self.status_code = status_code
        self.text = text
        self._content = content

    def raise_for_status(self):
        if self.status_code != 200:
            raise _FakeHTTPError(f"status {self.status_code}")

    def json(self):
        return {
            "choices": [{"message": {"content": self._content, "reasoning_content": ""}}],
            "usage": {},
        }


class _FakeHTTPError(Exception):
    """近似 requests.HTTPError：非 RuntimeError，走适配层通用重试路径。"""


class Test429Handling:
    def test_single_429_then_success(self):
        """首次 429 属瞬时限流 → 退避重试后成功，不判配额耗尽。"""
        from src.utils.llm_adapter import OpenAICompatibleAdapter
        adapter = _mk_adapter("https://api.minimax.cn/v1")
        state = {"calls": 0}

        def _post(url, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                return _Resp(429, '{"error":"too many requests"}')
            return _Resp(200)

        adapter._requests = types.SimpleNamespace(post=_post)
        out = adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="max")
        assert out == "正文"
        assert state["calls"] == 2
        assert adapter._consecutive_429 == 0

    def test_consecutive_429_raises_quota(self):
        from src.utils.llm_adapter import OpenAICompatibleAdapter, QuotaExhaustedError
        adapter = _mk_adapter("https://api.minimax.cn/v1")

        def _post(url, **kwargs):
            return _Resp(429, '{"error":"rate limited"}')

        adapter._requests = types.SimpleNamespace(post=_post)
        with pytest.raises(QuotaExhaustedError):
            adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="max")
        assert adapter._consecutive_429 == 2, "第二次 429 即判定配额耗尽（不再空转退避）"

    def test_402_with_quota_keyword_raises(self):
        from src.utils.llm_adapter import QuotaExhaustedError
        adapter = _mk_adapter("https://api.minimax.cn/v1")

        def _post(url, **kwargs):
            return _Resp(402, '{"error":"insufficient balance: quota exceeded"}')

        adapter._requests = types.SimpleNamespace(post=_post)
        with pytest.raises(QuotaExhaustedError):
            adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="max")


class _StubClient:
    def __init__(self, out=None, exc=None):
        self.out = out
        self.exc = exc
        self.calls = []

    def generate(self, prompt, model_name, **kwargs):
        self.calls.append({"prompt": prompt, "model_name": model_name, "kwargs": kwargs})
        if self.exc is not None:
            raise self.exc
        return self.out

    def get_cumulative_usage(self):
        return {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "calls": len(self.calls)}

    def unload_model(self):
        pass


DS_LEVELS = ("none", "low", "high", "xhigh", "max")


class TestFallbackRoleAdapter:
    def _mk(self, primary, fallback):
        from src.utils.llm_adapter import FallbackRoleAdapter
        return FallbackRoleAdapter(
            primary=primary, fallback=fallback,
            fallback_model_name="Deepseek-V4.1-Flash-1017-SS",
            primary_model_name="MiniMax-M3.1-Flash",
            fallback_role="secondary", fallback_levels=DS_LEVELS,
        )

    def test_degrades_on_quota_and_maps_effort(self):
        from src.utils.llm_adapter import QuotaExhaustedError
        primary = _StubClient(exc=QuotaExhaustedError("quota"))
        fallback = _StubClient(out="兜底译文")
        adapter = self._mk(primary, fallback)
        out = adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="medium")
        assert out == "兜底译文"
        assert adapter.degraded is True
        assert fallback.calls[0]["model_name"] == "Deepseek-V4.1-Flash-1017-SS"
        assert fallback.calls[0]["kwargs"]["effort"] == "high", "minimax medium → deepseek high"
        # 降级后第二次调用不再触碰主模型
        out2 = adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="max")
        assert out2 == "兜底译文"
        assert len(primary.calls) == 1
        assert fallback.calls[1]["kwargs"]["effort"] == "max", "max 在两家均合法，原样透传"

    def test_non_quota_error_not_degraded(self):
        primary = _StubClient(exc=RuntimeError("网络超时"))
        fallback = _StubClient(out="不应被调用")
        adapter = self._mk(primary, fallback)
        with pytest.raises(RuntimeError, match="网络超时"):
            adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="max")
        assert adapter.degraded is False
        assert fallback.calls == []

    def test_unmappable_effort_dropped(self):
        from src.utils.llm_adapter import QuotaExhaustedError
        primary = _StubClient(exc=QuotaExhaustedError("quota"))
        fallback = _StubClient(out="兜底译文")
        adapter = self._mk(primary, fallback)
        adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="banana")
        assert "effort" not in fallback.calls[0]["kwargs"], "无法安全映射 → 不下发档位"

    def test_usage_aggregates(self):
        from src.utils.llm_adapter import QuotaExhaustedError
        primary = _StubClient(exc=QuotaExhaustedError("quota"))
        fallback = _StubClient(out="兜底译文")
        adapter = self._mk(primary, fallback)
        adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="max")
        adapter.generate("p", model_name="MiniMax-M3.1-Flash", effort="max")
        u = adapter.get_cumulative_usage()
        assert u["calls"] == 3, "聚合 = 主模型 1 次 + 兜底 2 次"

    def test_effort_mapper_direct(self):
        from src.utils.llm_adapter import _map_effort_for_fallback
        assert _map_effort_for_fallback("medium", DS_LEVELS) == "high"
        assert _map_effort_for_fallback("xhigh", DS_LEVELS) == "xhigh"
        assert _map_effort_for_fallback("low", DS_LEVELS) == "low"
        assert _map_effort_for_fallback("nothing", DS_LEVELS) is None


class TestBuildRoleClientWiring:
    def test_primary_wrapped_with_fallback(self, monkeypatch):
        from src.utils import llm_adapter as m
        cfgs = {
            "primary": {
                "backend": "openai_api", "api_base": "https://api.minimax.cn/v1",
                "api_key": "k", "model_name": "MiniMax-M3.1-Flash", "fallback_role": "secondary",
            },
            "secondary": {
                "backend": "openai_api", "api_base": "https://token.ai-hf.cn/v1",
                "api_key": "k", "model_name": "Deepseek-V4.1-Flash-1017-SS",
            },
        }
        monkeypatch.setattr(m._config, "get_role_config", lambda role: dict(cfgs[role]))
        client = m._build_role_client(dict(cfgs["primary"]))
        assert isinstance(client, m.FallbackRoleAdapter)
        assert client.degraded is False

    def test_role_without_fallback_stays_plain(self, monkeypatch):
        from src.utils import llm_adapter as m
        monkeypatch.setattr(m._config, "get_role_config", lambda role: {
            "backend": "openai_api", "api_base": "https://token.ai-hf.cn/v1",
            "api_key": "k", "model_name": "Qwen3.8-Flash-Next-SS",
        })
        client = m._build_role_client({
            "backend": "openai_api", "api_base": "https://token.ai-hf.cn/v1",
            "api_key": "k", "model_name": "Qwen3.8-Flash-Next-SS",
        })
        assert not isinstance(client, m.FallbackRoleAdapter)

    def test_fallback_chain_not_recursive(self, monkeypatch):
        """fallback 目标自身带 fallback_role 时停止包装（避免链式递归）。"""
        from src.utils import llm_adapter as m
        cfgs = {
            "primary": {"backend": "openai_api", "api_base": "https://api.minimax.cn/v1",
                         "api_key": "k", "model_name": "MiniMax-M3.1-Flash", "fallback_role": "secondary"},
            "secondary": {"backend": "openai_api", "api_base": "https://token.ai-hf.cn/v1",
                          "api_key": "k", "model_name": "Deepseek-V4.1-Flash-1017-SS",
                          "fallback_role": "primary"},
        }
        monkeypatch.setattr(m._config, "get_role_config", lambda role: dict(cfgs[role]))
        client = m._build_role_client(dict(cfgs["primary"]))
        assert not isinstance(client, m.FallbackRoleAdapter), "链式 fallback 应停止包装"


class TestConfigRound129:
    @pytest.fixture()
    def cfg(self):
        from src.config import config
        config.reload()
        yield config
        config.reload()

    def test_primary_fallback_role_configured(self, cfg):
        assert cfg.models["primary"].get("fallback_role") == "secondary"

    def test_rewrite_moved_to_secondary_max(self, cfg):
        assert cfg.get_task_role("literary_rewrite") == "secondary"
        _, params = cfg.resolve_task_model("literary_rewrite")
        assert params.get("effort") == "max", "DS 大任务实测 max 稳定 → 思考开到最高"

    def test_effort_tables_measured_levels(self, cfg):
        from src.config import EFFORT_LEVELS_BY_PROVIDER
        assert EFFORT_LEVELS_BY_PROVIDER["deepseek"] == ("none", "low", "high", "xhigh", "max")
        assert EFFORT_LEVELS_BY_PROVIDER["qwen"] == ("none", "low", "medium", "xhigh")
        assert EFFORT_LEVELS_BY_PROVIDER["minimax"] == ("low", "medium", "high", "xhigh", "max")

    def test_monolith_mirrors(self):
        src = (PROJECT / "src" / "translator_agent.py").read_text(encoding="utf-8")
        for sym in ("class QuotaExhaustedError", "class FallbackRoleAdapter",
                    "def _build_single_client", "def _map_effort_for_fallback",
                    "_QUOTA_HINT_RE"):
            assert sym in src, f"单体须镜像 {sym}（需重建 build_integrated）"
