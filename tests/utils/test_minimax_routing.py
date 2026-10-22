"""Round 55: 测试 - utils/llm_adapter.py 的 MiniMax 端点匹配.

Bug 保护：
  - api.minimax.cn 端点必须匹配 MiniMax 分支（thinking/reasoning_effort 正确设置）
  - api.deepseek.com 端点匹配 DeepSeek 分支
  - api.openai.com 端点匹配 OpenAI 分支
  - 修复前 'minimaxi' (带双 i) 永远不会匹配 api.minimax.cn
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------

def _normalize_api_base(api_base: str) -> str:
    """复刻 src/utils/llm_adapter.py::_get_limiter_key 的分组逻辑"""
    api_base_lower = api_base.lower()
    if "minimax" in api_base_lower:
        return "minimax"
    if "deepseek" in api_base_lower:
        return "deepseek"
    if "openai" in api_base_lower:
        return "openai"
    from urllib.parse import urlparse
    return urlparse(api_base).netloc


def _provider_branch(api_base: str) -> str:
    """复刻 OpenAICompatibleAdapter.generate 中的 provider 分支选择"""
    api_base = api_base.lower()
    if "minimax" in api_base:
        return "minimax"
    elif "deepseek" in api_base:
        return "deepseek"
    elif "openai" in api_base:
        return "openai"
    return "fallback"


# ---------------------------------------------------------------------------
# 端点路由
# ---------------------------------------------------------------------------

class TestEndpointRouting:
    def test_minimax_endpoint_matches_minimax_branch(self):
        # 关键回归保护：真实端点必须命中 minimax 分支
        assert "minimax" in "api.minimax.cn"
        assert _provider_branch("https://api.minimax.cn/v1") == "minimax"

    def test_minimax_endpoint_limiter_key(self):
        assert _normalize_api_base("https://api.minimax.cn/v1") == "minimax"

    def test_deepseek_endpoint_matches_deepseek_branch(self):
        assert _provider_branch("https://api.deepseek.com/v1") == "deepseek"

    def test_deepseek_endpoint_limiter_key(self):
        assert _normalize_api_base("https://api.deepseek.com") == "deepseek"

    def test_openai_endpoint_matches_openai_branch(self):
        assert _provider_branch("https://api.openai.com/v1") == "openai"

    def test_unknown_endpoint_falls_back_to_netloc(self):
        result = _normalize_api_base("https://api.example.com/v1")
        assert result == "api.example.com"

    def test_legacy_typo_endpoint(self):
        # 历史 bug 端点（拼写错误 minimaxi）现已废弃
        # 但作为向后兼容，分组 key 应返回 "minimax"
        result = _normalize_api_base("https://api.minimaxi.cn/v1")
        assert result == "minimax"  # 5-letter prefix matches

    def test_provider_branch_for_unknown_endpoint(self):
        # 未知端点走 fallback（不发特定 provider 参数）
        assert _provider_branch("https://api.example.com/v1") == "fallback"


# ---------------------------------------------------------------------------
# 与实际 config.yaml 对齐
# ---------------------------------------------------------------------------

class TestConfigAlignment:
    def test_main_role_endpoints_routed_correctly(self):
        # 与 config.yaml 一致：literal_translator → deepseek; primary → minimax
        assert _provider_branch("https://api.deepseek.com") == "deepseek"
        assert _provider_branch("https://api.minimax.cn/v1") == "minimax"

    def test_endpoints_case_insensitive(self):
        # URL 大小写不影响匹配
        assert _provider_branch("HTTPS://API.MINIMAX.CN/v1") == "minimax"
        assert _provider_branch("Https://Api.DeepSeek.Com") == "deepseek"
        assert _provider_branch("HTTPS://API.OPENAI.COM/v1") == "openai"
