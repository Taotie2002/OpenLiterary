"""Round 117/128/129: effort 参数编码 + reasoning 审计轨迹 + monolith shim 回归。

覆盖：
  - build_effort_params 按 provider 编码（minimax 档位附加 adaptive thinking；网关/未知仅
    reasoning_effort: <level>）—— 散装 (src.utils.llm_adapter) 与 monolith 两份实现一致
  - OpenAICompatibleAdapter.generate: reasoning 不混入正文（reasoning_content 与网关
    `reasoning` 字段都读取）；仅推理无正文判定调用失败（Round 129：绝不把 CoT 当译文）
  - 无 effort 时不下发任何 thinking/reasoning_effort 参数（防止上游 400）
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
    build_effort_params as build_params,
)


def _mono_build(api_base: str, model_name: str, effort):
    import src.translator_agent as ta
    return ta.build_effort_params(api_base, model_name, effort)


@pytest.mark.parametrize("api_base,model_name,effort,expected", [
    ("https://api.minimax.cn/v1", "MiniMax-M3.1-Flash", "high",
     {"reasoning_effort": "high", "thinking": {"type": "adaptive"}}),
    ("https://api.minimax.cn/v1", "MiniMax-M3.1-Flash", None, {}),
    ("https://token.ai-hf.cn/v1", "Deepseek-V4.1-Flash-1017-SS", "low",
     {"reasoning_effort": "low"}),
    ("https://token.ai-hf.cn/v1", "Deepseek-V4.1-Flash-1017-SS", None, {}),
    ("https://token.ai-hf.cn/v1", "Qwen3.8-Flash-Next-SS", "xhigh",
     {"reasoning_effort": "xhigh"}),
    ("https://example.com/v1", "unknown-model", "high", {"reasoning_effort": "high"}),
    ("https://example.com/v1", "unknown-model", None, {}),
])
def test_build_effort_params_parts_and_monolith(api_base, model_name, effort, expected):
    assert build_params(api_base, model_name, effort) == expected
    assert _mono_build(api_base, model_name, effort) == expected


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
    out = adapter.generate("p", model_name="MiniMax-M3", effort="high")
    assert out == "正式译文"
    assert adapter.last_reasoning == "推理轨迹"
    assert adapter.last_reasoning_only is False
    assert captured.get("thinking") == {"type": "adaptive"}
    assert captured.get("reasoning_effort") == "high"


def test_reasoning_only_is_hard_failure():
    """Round 129: 仅推理无正文 = thinking 吃光预算 → 判定失败（不再返回推理兜底）。"""
    adapter = _mk_adapter("https://api.minimax.cn/v1")
    captured: dict = {}
    adapter._requests = types.SimpleNamespace(post=_mk_post(captured, "", "只有推理没有正文"))
    with pytest.raises(RuntimeError, match="仅输出推理"):
        adapter.generate("p", model_name="MiniMax-M3", effort="high")
    assert adapter.last_reasoning_only is True


def test_gateway_reasoning_field_is_audited_not_merged():
    """Round 129: 网关字段名 reasoning（非 reasoning_content）同样只作审计轨迹。"""
    adapter = _mk_adapter("https://token.ai-hf.cn/v1")
    captured: dict = {}

    class _Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "正式译文", "reasoning": "网关推理轨迹"}}],
                "usage": {},
            }

    def _post(url, headers=None, json=None, timeout=None, verify=None):
        captured.update(json or {})
        return _Resp()

    adapter._requests = types.SimpleNamespace(post=_post)
    out = adapter.generate("p", model_name="Deepseek-V4.1-Flash-1017-SS", effort="low")
    assert out == "正式译文"
    assert adapter.last_reasoning == "网关推理轨迹"
    assert adapter.last_reasoning_only is False


def test_gateway_reasoning_only_raises():
    """网关 reasoning 有值、content 空 → 判定失败（不得把推理当正文返回）。"""
    adapter = _mk_adapter("https://token.ai-hf.cn/v1")
    captured: dict = {}

    class _Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "", "reasoning": "只有推理没有正文"}}],
                "usage": {},
            }

    def _post(url, headers=None, json=None, timeout=None, verify=None):
        captured.update(json or {})
        return _Resp()

    adapter._requests = types.SimpleNamespace(post=_post)
    with pytest.raises(RuntimeError, match="仅输出推理"):
        adapter.generate("p", model_name="Deepseek-V4.1-Flash-1017-SS", effort="max")
    assert adapter.last_reasoning_only is True
def test_inline_think_block_extracted_from_content():
    """MiniMax adaptive thinking 内联 think 块 → 抽为审计轨迹，不混入正文。"""
    adapter = _mk_adapter("https://api.minimax.cn/v1")
    captured: dict = {}
    adapter._requests = types.SimpleNamespace(
        post=_mk_post(captured, "<think需要先推理再输出。</think>正式译文", ""))
    out = adapter.generate("p", model_name="MiniMax-M3", effort="high")
    assert out == "正式译文"
    assert adapter.last_reasoning == "需要先推理再输出。"
    assert adapter.last_reasoning_only is False


def test_gateway_effort_sent_verbatim_no_extra_thinking():
    adapter = _mk_adapter("https://token.ai-hf.cn/v1")
    captured_off: dict = {}
    adapter._requests = types.SimpleNamespace(post=_mk_post(captured_off, "译", ""))
    adapter.generate("p", model_name="Qwen3.8-Flash-Next-SS", effort=None)
    assert "thinking" not in captured_off
    assert "reasoning_effort" not in captured_off

    captured_on: dict = {}
    adapter._requests = types.SimpleNamespace(post=_mk_post(captured_on, "译", ""))
    adapter.generate("p", model_name="Deepseek-V4.1-Flash-SS", effort="low")
    assert captured_on.get("reasoning_effort") == "low"
    assert "thinking" not in captured_on


def test_400_effort_param_stripped_and_retried():
    """服务端 400 不识别 thinking/reasoning_effort 字段 → 剥离后重试一次（可用性优先）。"""
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
    out = adapter.generate("p", model_name="m", effort="high")
    assert out == "OK"
    assert state["calls"] == 2
    assert state["payloads"][0].get("reasoning_effort") == "high"
    assert "reasoning_effort" not in state["payloads"][1]
    assert "thinking" not in state["payloads"][1]


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
