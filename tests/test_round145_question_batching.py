"""Round 145: d3 64-question batching and atomic merge semantics."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.judges.typesafe_backend import JevBackend, batch_questions
from tests._round145_sdk_stub import install as install_sdk_stub


@pytest.fixture(autouse=True)
def _typesafe_sdk_stub(monkeypatch):
    install_sdk_stub(monkeypatch)


def _questions(n: int):
    return {
        f"q{i:03d}": {"type": "noul", "instructions": "判断该实体是否合规", "criteria": None}
        for i in range(n)
    }


class _FakeClient:
    def __init__(self, fail_on_call=None, extra_qid=None, omit_qid=None):
        self.calls = []
        self.timeouts = []
        self.fail_on_call = fail_on_call
        self.extra_qid = extra_qid
        self.omit_qid = omit_qid
        self.model = None

    def system_one(self, state, questions, model=None, timeout=None):
        self.calls.append(list(questions))
        self.timeouts.append(timeout)
        self.model = model
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("batch failed")
        answers = {}
        for qid in questions:
            if qid == self.omit_qid:
                continue
            answers[qid] = qid
        if self.extra_qid:
            answers[self.extra_qid] = self.extra_qid
        return SimpleNamespace(
            answers=answers,
            usage=SimpleNamespace(input_tokens=1, output_tokens=0),
        )


class TestBatchSplit:
    def test_64_is_one_batch(self):
        batches = batch_questions(_questions(64), 64)
        assert len(batches) == 1
        assert len(batches[0]) == 64

    def test_65_is_two_batches(self):
        batches = batch_questions(_questions(65), 64)
        assert [len(b) for b in batches] == [64, 1]

    def test_order_is_stable(self):
        qs = _questions(70)
        batches = batch_questions(qs, 64)
        merged = list(batches[0]) + list(batches[1])
        assert merged == list(qs)


class TestJevEvaluateBatches:
    def _backend(self, client, batch=64):
        backend = JevBackend(api_key="k", model="d3-flash", max_questions_per_batch=batch)
        backend._client = client
        return backend

    def test_65_questions_merge_in_original_order(self):
        client = _FakeClient()
        backend = self._backend(client)
        questions = _questions(65)
        with patch("src.judges.typesafe_backend._typesafe_sdk_available", return_value=True), \
             patch.object(JevBackend, "_convert_answer", staticmethod(lambda ans: ans)):
            out = backend.evaluate("state", questions, timeout=12)
        assert [len(c) for c in client.calls] == [64, 1]
        assert client.timeouts == [12, 12]
        assert list(out) == list(questions)
        assert len(out) == 65
        assert backend.last_diagnostics["batch_count"] == 2
        assert backend.last_diagnostics["batch_question_counts"] == [64, 1]
        assert backend.last_diagnostics["question_count"] == 65
        assert backend.last_diagnostics["output_tokens"] == 0
        assert client.model == "d3-flash"

    def test_batch_failure_has_no_partial_result(self):
        client = _FakeClient(fail_on_call=2)
        backend = self._backend(client)
        with patch("src.judges.typesafe_backend._typesafe_sdk_available", return_value=True), \
             patch.object(JevBackend, "_convert_answer", staticmethod(lambda ans: ans)):
            with pytest.raises(RuntimeError, match="batch failed"):
                backend.evaluate("state", _questions(65))

    def test_missing_answer_in_batch_raises(self):
        client = _FakeClient(omit_qid="q000")
        backend = self._backend(client)
        with patch("src.judges.typesafe_backend._typesafe_sdk_available", return_value=True), \
             patch.object(JevBackend, "_convert_answer", staticmethod(lambda ans: ans)):
            with pytest.raises(RuntimeError, match="缺少 qid"):
                backend.evaluate("state", _questions(3))

    def test_unexpected_qid_raises(self):
        client = _FakeClient(extra_qid="q999")
        backend = self._backend(client)
        with patch("src.judges.typesafe_backend._typesafe_sdk_available", return_value=True), \
             patch.object(JevBackend, "_convert_answer", staticmethod(lambda ans: ans)):
            with pytest.raises(RuntimeError, match="未请求"):
                backend.evaluate("state", _questions(2))
