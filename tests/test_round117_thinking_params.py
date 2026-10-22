"""Round 117: thinking 参数归一 + reasoning_content 审计轨迹 + monolith shim 回归。

覆盖：
  - build_thinking_params 按 provider 归一（minimax/deepseek/openai/DeepSeek 代理宿主/未知）
    —— 散装 (src.utils.llm_adapter) 与 monolith (src.translator_agent) 两份实现一致
  - OpenAICompatibleAdapter.generate: reasoning_content 不混入正文；仅推理无正文时兜底 + 标记
  - DeepSeek 代理宿主（token.ai-hf.cn）仅在显式开启时下发 thinking 字段（防止 400）
  - monolith shim 回归: `import src.translator_agent; import src.judges.pairwise` 子进程可导入
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from src.utils.llm_adapter import (
    OpenAICompatibleAdapter,
    build_thinking_params as build_params,
)


def _mono_build(api_base: str, enable: bool):
    import src.translator_agent as ta
    return ta.build_thinking_params(api_base, enable)


@pytest.mark.parametrize("api_base,enable,expected", [
    ("https://api.minimax.cn/v1", True, {"thinking": {"type": "adaptive"}}),
    ("https://api.minimax.cn/v1", False, {"reasoning_effort": "none", "thinking": {"type": "disabled"}}),
    ("https://api.deepseek.com/v1", True, {"thinking": {"type": "enabled"}}),
    ("https://api.deepseek.com/v1", False, {"thinking": {"type": "disabled"}}),
    ("https://api.openai.com/v1", True, {"reasoning_effort": "high"}),
    ("https://api.openai.com/v1", False, {}),
    ("https://token.ai-hf.cn/v1", True, {"thinking": {"type": "enabled"}}),
    ("https://token.ai-hf.cn/v1", False, {}),
    ("https://example.com/v1", True, {}),
    ("https://example.com/v1", False, {}),
])
def test_build_thinking_params_parts_and_monolith(api_base, enable, expected):
    assert build_params(api_base, enable) == expected
    assert _mono_build(api_base, enable) == expected


def _mk_adapter(api_base: str) -> OpenAICompatibleAdapter:
    adapter = OpenAICompatibleAdapter(api_base=api_base, api_key="test-key")
    adapter.min_request_interval = 0.0
    OpenAICompatibleAdapter._rate_limiters[adapter._get_limiter_key(api_base)]["interval"] = 0.0
    return adapter


def _mk_post(captured: dict, content: str, reasoning: str):
    class _Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": content, "reasoning_content": reasoning}}],
                "usage": {},
            }

    def _post(url, headers=None, json=None, timeout=None, verify=None):
        captured.update(json or {})
        return _Resp()

    return _post


def test_reasoning_content_not_merged_into_output():
    adapter = _mk_adapter("https://api.minimax.cn/v1")
    captured: dict = {}
    adapter._requests = types.SimpleNamespace(post=_mk_post(captured, "正式译文", "推理轨迹"))
    out = adapter.generate("p", model_name="MiniMax-M3", enable_thinking=True)
    assert out == "正式译文"
    assert adapter.last_reasoning == "推理轨迹"
    assert adapter.last_reasoning_only is False
    assert captured.get("thinking") == {"type": "adaptive"}


def test_reasoning_only_fallback_is_flagged():
    adapter = _mk_adapter("https://api.minimax.cn/v1")
    captured: dict = {}
    adapter._requests = types.SimpleNamespace(post=_mk_post(captured, "", "只有推理没有正文"))
    out = adapter.generate("p", model_name="MiniMax-M3", enable_thinking=True)
    assert out == "只有推理没有正文"
    assert adapter.last_reasoning_only is True


def test_inline_think_block_extracted_from_content():
    """MiniMax adaptive thinking 内联 think 块 → 抽为审计轨迹，不混入正文。"""
    adapter = _mk_adapter("https://api.minimax.cn/v1")
    captured: dict = {}
    adapter._requests = types.SimpleNamespace(
        post=_mk_post(captured, "<think需要先推理再输出。</think>正式译文", ""))
    out = adapter.generate("p", model_name="MiniMax-M3", enable_thinking=True)
    assert out == "正式译文"
    assert adapter.last_reasoning == "需要先推理再输出。"
    assert adapter.last_reasoning_only is False


def test_deepseek_proxy_sends_thinking_only_when_enabled():
    adapter = _mk_adapter("https://token.ai-hf.cn/v1")
    captured_off: dict = {}
    adapter._requests = types.SimpleNamespace(post=_mk_post(captured_off, "译", ""))
    adapter.generate("p", model_name="Deepseek-V4.1-Flash-SS", enable_thinking=False)
    assert "thinking" not in captured_off
    assert "reasoning_effort" not in captured_off

    captured_on: dict = {}
    adapter._requests = types.SimpleNamespace(post=_mk_post(captured_on, "译", ""))
    adapter.generate("p", model_name="Deepseek-V4.1-Flash-SS", enable_thinking=True)
    assert captured_on.get("thinking") == {"type": "enabled"}


def test_400_thinking_param_stripped_and_retried():
    """服务端 400 不识别 thinking 字段 → 剥离后重试一次（可用性优先）。"""
    adapter = _mk_adapter("https://token.ai-hf.cn/v1")
    state = {"calls": 0, "payloads": []}

    class _Resp400:
        status_code = 400
        text = '{"error":"unknown parameter: thinking"}'

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    class _Resp200:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "OK", "reasoning_content": ""}}], "usage": {}}

    def _post(url, headers=None, json=None, timeout=None, verify=None):
        state["calls"] += 1
        state["payloads"].append(dict(json or {}))
        return _Resp400() if state["calls"] == 1 else _Resp200()

    adapter._requests = types.SimpleNamespace(post=_post)
    out = adapter.generate("p", model_name="m", enable_thinking=True)
    assert out == "OK"
    assert state["calls"] == 2
    assert "thinking" in state["payloads"][0]
    assert "thinking" not in state["payloads"][1]
    assert "reasoning_effort" not in state["payloads"][1]


def test_monolith_shim_allows_judges_submodule_import():
    code = (
        "import src.translator_agent; "
        "import src.judges.pairwise as p; "
        "print('SHIM_OK', hasattr(p, 'arbitrate_answers'))"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "SHIM_OK True" in proc.stdout


def test_monolith_inline_think_regex_available():
    """monolith 侧 _INLINE_THINK_RE 可用（与散装对称）。"""
    LT, GT = chr(60), chr(62)
    import src.translator_agent as ta
    assert ta._INLINE_THINK_RE.search(LT + "thinkX" + LT + "/think" + GT)
