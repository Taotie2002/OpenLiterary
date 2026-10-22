"""Round 118 P1-2: REJECT 第二意见测试.

设计原则:
  - typesafe verdict REJECT 后, judge 触发一次 LLM 复核, 把概率拒因翻译成
    rewriter 可执行的编辑指令 (specific_edits / unresolved_edits);
  - 第二意见绝不改写 decision (不弱化 Jev 仲裁); 输出不可用时不阻断流程;
  - pipeline 在 critic_report.specific_edits 为空时并入 judge.unresolved_edits.
"""
from __future__ import annotations

import json
import sys

import pytest

from src.config import config as _config


@pytest.fixture(autouse=True)
def _isolate_modules():
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
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


class _StubLLM:
    def __init__(self, output: str = None, exc: Exception = None):
        self.output = output
        self.exc = exc
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.output


def _make_judge(agent_cls, verdict_cls, llm):
    agent = agent_cls.__new__(agent_cls)
    agent.llm = llm
    agent.role = _config.get_task_role("judge_decision")
    agent.db = None
    return agent


def _verdict_dict(decision: str = "REJECT") -> dict:
    return {
        "decision": decision,
        "confidence": 0.72,
        "reject_reasons": (["音色漂移（severe）: prob=0.62 conf=0.80"]
                           if decision == "REJECT" else []),
        "primitive_summary": {
            "voice_drift": {"type": "choice", "choice": "severe",
                            "probabilities": {"none": 0.1, "subtle": 0.28, "severe": 0.62},
                            "confidence": 0.8},
        },
        "skipped_dimensions": [],
        "skipped_reasons": {},
        "confidence_source": "jev",
        "confidence_calibrated": True,
    }


EDITS_JSON = json.dumps({
    "edits": [
        {"original": "他说道", "problem": "叙述口吻平淡",
         "suggested_direction": "改为更克制的间接叙述，保留讽刺语气"},
    ],
    "confirmed": True,
}, ensure_ascii=False)


class TestJudgeAgentSecondOpinion:
    def test_reject_triggers_second_opinion(self):
        from src.agents.judge_agent import JudgeAgent
        from src.judges.typesafe_spec import CompositeVerdict

        llm = _StubLLM(output=EDITS_JSON)
        judge = _make_judge(JudgeAgent, CompositeVerdict, llm)
        report = {"_typesafe_verdict": _verdict_dict(), "_backend_path": "jev", "scores": {}}
        result = judge.process_chunk("ch01_chunk001", "原文", "润色稿", report)
        assert result["decision"] == "REJECT"
        assert len(result["unresolved_edits"]) == 1
        assert result["unresolved_edits"][0]["suggested_direction"].startswith("改为更克制")
        assert result["_second_opinion"]["confirmed_reject"] is True
        assert len(llm.calls) == 1

    def test_pass_does_not_call_second_opinion(self):
        from src.agents.judge_agent import JudgeAgent
        from src.judges.typesafe_spec import CompositeVerdict

        llm = _StubLLM(output=EDITS_JSON)
        judge = _make_judge(JudgeAgent, CompositeVerdict, llm)
        report = {"_typesafe_verdict": _verdict_dict("PASS"), "_backend_path": "jev", "scores": {}}
        result = judge.process_chunk("ch01_chunk002", "原文", "润色稿", report)
        assert result["decision"] == "PASS"
        assert result["unresolved_edits"] == []
        assert llm.calls == []

    def test_second_opinion_failure_keeps_reject(self):
        from src.agents.judge_agent import JudgeAgent
        from src.judges.typesafe_spec import CompositeVerdict

        llm = _StubLLM(exc=RuntimeError("api down"))
        judge = _make_judge(JudgeAgent, CompositeVerdict, llm)
        report = {"_typesafe_verdict": _verdict_dict(), "_backend_path": "jev", "scores": {}}
        result = judge.process_chunk("ch01_chunk003", "原文", "润色稿", report)
        assert result["decision"] == "REJECT"
        assert result["unresolved_edits"] == []
        assert "api down" in result["_second_opinion"]["error"]

    def test_empty_edits_confirmed_false_no_downgrade(self):
        from src.agents.judge_agent import JudgeAgent
        from src.judges.typesafe_spec import CompositeVerdict

        llm = _StubLLM(output=json.dumps({"edits": [], "confirmed": False}))
        judge = _make_judge(JudgeAgent, CompositeVerdict, llm)
        report = {"_typesafe_verdict": _verdict_dict(), "_backend_path": "jev", "scores": {}}
        result = judge.process_chunk("ch01_chunk004", "原文", "润色稿", report)
        assert result["decision"] == "REJECT"  # 不翻案
        assert result["unresolved_edits"] == []
        assert result["_second_opinion"]["confirmed_reject"] is False

    def test_monolith_mirror(self):
        import src.translator_agent as ta

        llm = _StubLLM(output=EDITS_JSON)
        judge = _make_judge(ta.JudgeAgent, ta.CompositeVerdict, llm)
        report = {"_typesafe_verdict": _verdict_dict(), "_backend_path": "jev", "scores": {}}
        result = judge.process_chunk("ch01_chunk005", "原文", "润色稿", report)
        assert result["decision"] == "REJECT"
        assert len(result["unresolved_edits"]) == 1


class TestPipelineFeedbackMerge:
    def test_unresolved_edits_merged_into_feedback(self):
        from src.pipeline import TranslationPipeline

        captured = {}

        class _StubScheduler:
            def update_task_state(self, chunk_id, state, error_msg=None, quality_retry=False):
                captured["chunk_id"] = chunk_id
                captured["error_msg"] = error_msg

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.scheduler = _StubScheduler()
        critic_report = {"scores": {}, "specific_edits": []}
        judge_result = {
            "reject_reason": "音色漂移 severe",
            "unresolved_edits": [{
                "index": 1, "original": "他说道", "problem": "平淡",
                "suggested_direction": "改为间接叙述",
            }],
            "_second_opinion": {"confirmed_reject": True, "edits": 1},
        }
        pipeline._fallback_to_rewriter("ch01_chunk010", "润色稿", critic_report, judge_result)
        feedback = json.loads(captured["error_msg"])
        assert feedback["specific_edits"][0]["suggested_direction"] == "改为间接叙述"
        assert feedback["_second_opinion"]["confirmed_reject"] is True

    def test_critic_specific_edits_take_precedence(self):
        from src.pipeline import TranslationPipeline

        captured = {}

        class _StubScheduler:
            def update_task_state(self, chunk_id, state, error_msg=None, quality_retry=False):
                captured["error_msg"] = error_msg

        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.scheduler = _StubScheduler()
        critic_report = {"scores": {}, "specific_edits": [
            {"index": 1, "original": "A", "suggested_direction": "critic 指令"},
        ]}
        judge_result = {"reject_reason": "r", "unresolved_edits": [
            {"index": 1, "original": "B", "suggested_direction": "judge 指令"},
        ]}
        pipeline._fallback_to_rewriter("ch01_chunk011", "润色稿", critic_report, judge_result)
        feedback = json.loads(captured["error_msg"])
        assert feedback["specific_edits"][0]["suggested_direction"] == "critic 指令"
