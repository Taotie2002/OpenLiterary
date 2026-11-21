"""Round 158 regression: modular JudgeAgent must resolve DecisionLevel."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

from src.agents.judge_agent import JudgeAgent
from src.core.decision_engine import DecisionLevel


def test_modular_judge_style_rule_records_decision(monkeypatch):
    judge = JudgeAgent.__new__(JudgeAgent)
    judge.db = MagicMock()
    judge.llm = MagicMock()
    judge.role = "judge_decision"
    judge.llm.generate.return_value = (
        '{"decision":"PASS","reject_reason":"",'
        '"new_style_rule":{"rule_description":"使用短句。","reason":"节奏"},'
        '"unresolved_edits":[]}'
    )

    # Other suites temporarily pop ``src.agents`` from sys.modules while
    # rebuilding the package.  Patch the module object that actually defines
    # JudgeAgent instead of resolving a dotted path through the package; the
    # latter is order-dependent and fails when the parent attribute is stale.
    judge_module = sys.modules[JudgeAgent.__module__]
    monkeypatch.setattr(
        judge_module._config, "resolve_task_model",
        lambda _task: ("judge-model", {}),
    )
    monkeypatch.setattr(
        judge_module, "get_role_model_name",
        lambda _role: "judge-model",
    )
    out = judge.process_chunk(
        "ch01_chunk000", "source", "译文",
        critic_report={"scores": {}, "specific_edits": []},
    )

    assert out["decision"] == "PASS"
    judge.db.add_decision.assert_called_once()
    assert judge.db.add_decision.call_args.kwargs["level"] == DecisionLevel.STYLE
