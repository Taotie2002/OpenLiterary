"""Round 151: source-grounded mechanical gate for LLM name diagnosis."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repair_names_with_llm.py"
_SPEC = importlib.util.spec_from_file_location("repair_names_with_llm", _SCRIPT)
repair = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(repair)


def _issue(wrong: str, right: str, context: str, reason: str):
    return {
        "wrong": wrong,
        "right": right,
        "context": context,
        "reason": reason,
    }


def test_accepts_name_when_english_target_is_in_source():
    final = "盖乌斯看着保罗说：“走。”"
    issue = _issue("盖乌斯", "杰西卡", "盖乌斯看着保罗", "Jessica looked at Paul.")
    validated, reason = repair._validate_issue(
        issue, final, "Jessica looked at Paul. “Go,” she said.")
    assert reason == "ok"
    assert validated is not None
    assert final[validated["start"]:validated["end"]] == "盖乌斯"


def test_rejects_background_inference_when_name_is_absent_from_source():
    final = "盖乌斯夫人看着保罗。"
    issue = _issue("盖乌斯夫人", "杰西卡夫人", "盖乌斯夫人看着保罗",
                   "The source says Jessica is Paul's mother.")
    validated, reason = repair._validate_issue(
        issue, final, "She looked at Paul. Her mother stayed silent.")
    assert validated is None
    assert reason == "source_evidence_missing:杰西卡夫人"


def test_role_target_requires_truthsayer_in_source():
    final = "盖乌斯说：“他知道。”"
    issue = _issue("盖乌斯", "真言者", "盖乌斯说", "He fears the Truthsayer.")
    validated, reason = repair._validate_issue(
        issue, final, "He, too, fears the Truthsayer.")
    assert reason == "ok"
    assert validated is not None


def test_rejects_same_person_even_with_source_evidence():
    final = "保罗·厄崔迪看着雷托。"
    issue = _issue("保罗·厄崔迪", "保罗", "保罗·厄崔迪看着雷托",
                   "Paul looked at Leto.")
    validated, reason = repair._validate_issue(
        issue, final, "Paul looked at Leto.")
    assert validated is None
    assert reason == "same_person:保罗·厄崔迪->保罗"


def test_rejects_repeated_context_to_avoid_wrong_occurrence():
    final = "盖乌斯走了。后来，盖乌斯又回来了。"
    issue = _issue("盖乌斯", "杰西卡", "盖乌斯", "Jessica left.")
    validated, reason = repair._validate_issue(
        issue, final, "Jessica left. Later Jessica returned.")
    assert validated is None
    assert reason == "ambiguous_context"


def test_rejects_english_substring_without_word_boundary():
    final = "盖乌斯说：“走。”"
    issue = _issue("盖乌斯", "杰西卡", "盖乌斯说", "Jessica said.")
    validated, reason = repair._validate_issue(
        issue, final, "Jessicalike text without the actual name.")
    assert validated is None
    assert reason == "source_evidence_missing:杰西卡"


def test_rejects_context_with_repeated_wrong_token():
    final = "盖乌斯夫人说：他也怕盖乌斯。"
    issue = _issue(
        "盖乌斯",
        "真言者",
        "盖乌斯夫人说：他也怕盖乌斯。",
        "He too fears the Truthsayer.",
    )
    validated, reason = repair._validate_issue(
        issue, final, "Jessica said: He too fears the Truthsayer.")
    assert validated is None
    assert reason == "ambiguous_wrong_in_context"
