"""Round 118 P1-3: 参考提取两阶段测试.

  - 阶段 1 召回（DeepSeek）: 高召回初筛;
  - 阶段 2 确认（MiniMax, task=reference_confirmation）: 逐条核验/删误报/补漏/
    统一实体, 输出定稿;
  - 确认阶段任何失败/禁用 -> None -> 回退召回结果（绝不丢数据）;
  - decision (Jev 之外的 LLM 路线) 不改变 pipeline 结构, 仅替换 parsed_data.
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


RECALL = {
    "references": [
        {"source_text": "wild-cat", "allusion_target": "金融术语", "strategy": "LITERAL_TRANSLATION",
         "translated_term": "野猫", "reason": "字面直译"},
        {"source_text": "morganatic", "allusion_target": "贵族通婚制度", "strategy": "RETAIN_AND_ANNOTATE",
         "translated_term": "贵贱通婚", "reason": "法律专名"},
    ],
    "entities": [{"canonical_name": "吉姆", "source_names": ["Jim"]}],
}

CONFIRMED = {
    "references": [
        {"source_text": "morganatic", "allusion_target": "贵族与非贵族通婚制度", "strategy": "RETAIN_AND_ANNOTATE",
         "translated_term": "贵贱通婚", "reason": "法律专名，保留并加注"},
    ],
    "entities": [{"canonical_name": "吉姆", "source_names": ["Jim", "James"]}],
}


class TestConfig:
    def test_routes_and_flags_exist(self):
        # Round 125: 确认阶段由 MiniMax 换为 secondary(Qwen) 分流；仍在强模型上（非直译 flash）
        assert _config.get_task_role("reference_confirmation") == "secondary"
        assert _config.pipeline["reference_two_stage"]["enabled"] is True
        assert _config.pipeline["reference_two_stage"]["max_recall_chars"] >= 4000


class TestConfirmStage:
    def _agent(self):
        from src.agents.reference_agent import ReferenceAgent
        agent = ReferenceAgent.__new__(ReferenceAgent)
        agent.chapter_id = "ch01"
        agent.db = None
        agent._scheduler = None
        agent.role = _config.get_task_role("reference_extraction")
        return agent

    def test_confirm_returns_filtered_data(self, monkeypatch):
        import src.agents.reference_agent as ra

        stub = _StubLLM(output=json.dumps(CONFIRMED, ensure_ascii=False))
        monkeypatch.setattr(ra, "get_llm_client_for_task", lambda task: stub)
        agent = self._agent()
        out = agent._run_confirm_stage("ch01_chunk001", "原文 morganatic wild-cat", RECALL)
        assert out is not None
        assert len(out["references"]) == 1
        assert out["references"][0]["source_text"] == "morganatic"
        assert out["entities"][0]["source_names"] == ["Jim", "James"]
        assert len(stub.calls) == 1

    def test_confirm_json_failure_falls_back(self, monkeypatch):
        import src.agents.reference_agent as ra

        stub = _StubLLM(output="这不是 JSON")
        monkeypatch.setattr(ra, "get_llm_client_for_task", lambda task: stub)
        assert self._agent()._run_confirm_stage("ch01_chunk002", "x", RECALL) is None

    def test_confirm_exception_falls_back(self, monkeypatch):
        import src.agents.reference_agent as ra

        stub = _StubLLM(exc=ConnectionError("network down"))
        monkeypatch.setattr(ra, "get_llm_client_for_task", lambda task: stub)
        assert self._agent()._run_confirm_stage("ch01_chunk003", "x", RECALL) is None

    def test_confirm_disabled_returns_none(self, monkeypatch):
        monkeypatch.setitem(_config.pipeline["reference_two_stage"], "enabled", False)
        assert self._agent()._run_confirm_stage("ch01_chunk004", "x", RECALL) is None

    def test_process_chunk_uses_confirmed_refs(self, monkeypatch):
        agent = self._agent()
        recall_llm = _StubLLM(output=json.dumps(RECALL, ensure_ascii=False))
        agent.llm = recall_llm
        monkeypatch.setattr(agent, "_find_chunks_containing", lambda source: [])
        monkeypatch.setattr(agent, "_run_confirm_stage",
                            lambda cid, text, data: dict(CONFIRMED))
        recorded = []

        class _StubDB:
            def add_decision(self, **kw):
                recorded.append(kw)

        agent.db = _StubDB()
        monkeypatch.setattr(agent, "_register_entities", lambda cid, ents: recorded.append(
            {"entities": ents}))
        agent.process_chunk("ch01_chunk005", "原文 morganatic")
        ref_sources = [r["source"] for r in recorded if "source" in r]
        assert ref_sources == ["morganatic"]  # wild-cat 已被确认阶段剔除

    def test_monolith_mirror(self, monkeypatch):
        import src.translator_agent as ta

        stub = _StubLLM(output=json.dumps(CONFIRMED, ensure_ascii=False))
        monkeypatch.setattr(ta, "get_llm_client_for_task", lambda task: stub)
        agent = ta.ReferenceAgent.__new__(ta.ReferenceAgent)
        out = agent._run_confirm_stage("ch01_chunk006", "x", RECALL)
        assert out is not None and len(out["references"]) == 1
