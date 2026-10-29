"""Round 122: monolith `src/translator_agent.py::_classify_backend_error` 防回退测试。

现场（ch09_chunk037，2026-10-24）：该函数曾被定义在
`if backend.primary.api_key and not getattr(backend, "_jev_auth_dead", False):`
分支内，却在 except 中无条件引用。Jev 无 key / 已熔断时（本 run 实测），异常
处理器自身崩成 "cannot access local variable '_classify_backend_error'"，真实
错误被吞掉，该块审计直接失败 → PERMANENTLY_FAILED。

修复：函数提升到模块级（与 src/agents/critic_agent.py 的模块级实现对齐）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

# 先导入真实子包，避免 translator_agent Round 103 shim 把尚未导入的 src.judges
# 替换为假 ModuleType（见 scripts/jev_pairwise.py 同款注意事项）。
import src.judges.judge  # noqa: F401
import src.judges.primitives  # noqa: F401
import src.agents.critic_agent  # noqa: F401
import src.translator_agent as ta


def test_source_def_is_module_level():
    src_path = Path(__file__).resolve().parents[1] / "src" / "translator_agent.py"
    lines = src_path.read_text(encoding="utf-8").splitlines()
    matches = [line for line in lines if line.strip().startswith("def _classify_backend_error")]
    assert matches, "monolith 中找不到 _classify_backend_error 定义"
    assert matches[0].startswith("def "), (
        "回归：_classify_backend_error 又被缩进进了条件分支，"
        "Jev 熔断时 except 处理器将再次 UnboundLocalError"
    )


def test_module_attribute_resolvable_without_jev_path():
    assert callable(getattr(ta, "_classify_backend_error", None))


@pytest.mark.parametrize(
    "message, category, retryable",
    [
        ("429 Client Error: Too Many Requests for url: https://api.minimax.cn/v1/chat/completions", "rate_limit", True),
        ("POST https://api.typesafe.ai/v1/systemone: 401 Cannot authenticate with the server.", "auth", False),
        ("POST https://api.typesafe.ai/v1/systemone: 451 region blocked", "unknown", True),
    ],
)
def test_classification_core_cases(message, category, retryable):
    out = ta._classify_backend_error(RuntimeError(message))
    assert out["category"] == category
    assert out["retryable"] is retryable
    assert {"category", "msg", "action"} <= set(out)
