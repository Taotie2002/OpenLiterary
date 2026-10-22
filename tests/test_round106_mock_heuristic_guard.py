"""Round 106: mock_heuristic 后端不可信护栏回归测试。

ch05 倒查根因：Jev API 被 451 区域封锁 → critic 回退到 mock_heuristic，其
_typesafe_verdict 是确定性假分数（entity_consistency=0 / literal_residue 极低），
judge 又直接复用该 verdict（从不调 LLM 独立裁决），导致整章 7/7 chunk
PERMANENTLY_FAILED。

修复：trust_mock_heuristic 默认 false 时——
  - judge 不复用 mock 的 _typesafe_verdict，强制走 LLM 独立裁决；
  - mock 分数不触发 AUDIT_EARLY_STOP / 高分直通（pipeline 侧，本测试覆盖 judge 侧）。
本测试锁定 judge 侧行为。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.agents.judge_agent import JudgeAgent
from src.config import config as _config


def _make_judge():
    judge = JudgeAgent(decision_engine=MagicMock())
    return judge


def _mock_verdict_report(backend: str, decision: str = "REJECT") -> dict:
    """构造带 _typesafe_verdict 的 critic_report（模拟 mock/jev 输出）。"""
    return {
        "scores": {
            "voice_consistency": 9.0, "style_compliance": 5.0,
            "literal_residue": 0.2, "semantic_preservation": 10.0,
            "readability": 2.5, "fluency": 2.5, "entity_consistency": 0.0,
        },
        "_backend_path": backend,
        "_typesafe_verdict": {
            "decision": decision,
            "confidence": 0.5,
            "reject_reasons": ["直译残留: noul=0.98 eff_conf=0.98"],
            "primitive_summary": {},
        },
    }


@pytest.fixture
def restore_trust_flag():
    original = _config.pipeline.get("trust_mock_heuristic", False)
    original_second = dict(_config.pipeline.get("judge_second_opinion", {}) or {})
    yield
    _config.pipeline["trust_mock_heuristic"] = original
    _config.pipeline["judge_second_opinion"] = original_second


class TestJudgeMockHeuristicGuard:
    def test_mock_verdict_not_reused_by_default(self, restore_trust_flag):
        """默认（trust_mock_heuristic=false）：mock 的 REJECT verdict 不被复用，走 LLM。"""
        _config.pipeline["trust_mock_heuristic"] = False
        judge = _make_judge()
        # mock LLM 返回 PASS（证明 judge 真的走了 LLM 路径，而非复用 mock 的 REJECT）
        judge.llm.generate = MagicMock(
            return_value=json.dumps({"decision": "PASS", "reject_reason": ""})
        )
        report = _mock_verdict_report("mock_heuristic", decision="REJECT")
        result = judge.process_chunk("c000", "src", "lit", report)
        # 关键断言：走了 LLM 路径（generate 被调用），且结果是 LLM 的 PASS 而非 mock 的 REJECT
        assert judge.llm.generate.called, "mock_heuristic 时应强制走 LLM 独立裁决"
        assert result["decision"] == "PASS"
        assert result.get("_backend_path") == "legacy"

    def test_jev_verdict_reused_no_llm(self, restore_trust_flag):
        """jev 后端：verdict 被直接复用，不调 LLM（无 LLM 成本）。"""
        _config.pipeline["trust_mock_heuristic"] = False
        # Round 118 P1-2: typesafe REJECT 会额外触发一次"第二意见"LLM 调用
        # （不改 decision，仅为 rewriter 生成可执行编辑；见
        # test_round118_reject_second_opinion.py）。本测试锁定"verdict 复用不产生
        # 裁决类 LLM 调用"的原契约，故暂时关闭第二意见。
        _config.pipeline["judge_second_opinion"] = {"enabled": False}
        judge = _make_judge()
        judge.llm.generate = MagicMock()
        report = _mock_verdict_report("jev", decision="REJECT")
        result = judge.process_chunk("c000", "src", "lit", report)
        assert not judge.llm.generate.called, "jev 后端应复用 verdict，不调 LLM"
        assert result["decision"] == "REJECT"
        assert result.get("_backend_path") == "jev"

    def test_trust_mock_opt_in_reuses_verdict(self, restore_trust_flag):
        """显式 trust_mock_heuristic=true：mock verdict 被复用（离线环境可选）。"""
        _config.pipeline["trust_mock_heuristic"] = True
        _config.pipeline["judge_second_opinion"] = {"enabled": False}
        judge = _make_judge()
        judge.llm.generate = MagicMock()
        report = _mock_verdict_report("mock_heuristic", decision="REJECT")
        result = judge.process_chunk("c000", "src", "lit", report)
        assert not judge.llm.generate.called, "trust_mock_heuristic=true 时应复用 mock verdict"
        assert result["decision"] == "REJECT"
        assert result.get("_backend_path") == "mock_heuristic"

    def test_default_config_is_untrusted(self):
        """config.yaml 默认 trust_mock_heuristic 为 false（不信任 mock）。"""
        assert _config.pipeline.get("trust_mock_heuristic", False) is False


class TestJudgePromptSanitization:
    """Round 106 P2: mock 后端时清洗 critic_report，避免假分数带偏 LLM 裁决。"""

    def _mock_report(self):
        return {
            "scores": {"readability": 2.5, "fluency": 2.5, "literal_residue": 0.5},
            "_backend_path": "mock_heuristic",
            "_typesafe_verdict": {"decision": "REJECT", "reject_reasons": ["直译残留"]},
            "critique": "直译残留: noul=0.98",
            "improvement_suggestions": "直译残留: noul=0.98",
            "entity_consistency_notes": [{"entity": "X", "type": "missing"}],
            "specific_edits": [],
        }

    def test_sanitize_removes_mock_fields(self):
        from src.agents.judge_agent import JudgeAgent
        s = JudgeAgent._sanitize_critic_report_for_llm(self._mock_report())
        assert "scores" not in s
        assert "_typesafe_verdict" not in s
        assert "critique" not in s
        assert "improvement_suggestions" not in s
        # 保留可靠字段
        assert "entity_consistency_notes" in s
        assert "specific_edits" in s
        assert "_note" in s

    def test_prompt_without_scores_uses_quality_instruction(self):
        from src.agents.judge_agent import JudgeAgent
        judge = JudgeAgent(decision_engine=MagicMock())
        s = JudgeAgent._sanitize_critic_report_for_llm(self._mock_report())
        prompt = judge._build_prompt("src text", "lit text", s)
        # 无 scores → 不应出现"评分参考阈值"，应出现"判断依据"
        assert "评分参考阈值" not in prompt
        assert "判断依据" in prompt
        # 假分数不应出现在 prompt 里
        assert "2.5" not in prompt
        assert "mock_heuristic" in prompt  # _note 标注

    def test_prompt_with_scores_keeps_thresholds(self):
        from src.agents.judge_agent import JudgeAgent
        judge = JudgeAgent(decision_engine=MagicMock())
        report = {
            "scores": {"readability": 8.0, "fluency": 8.0},
            "_backend_path": "jev",
        }
        prompt = judge._build_prompt("src text", "lit text", report)
        assert "评分参考阈值" in prompt
        assert "判断依据" not in prompt
