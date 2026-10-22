"""Round 48: 单元测试 - judges/primitives.py (build_question_set + ALL_PRIMITIVES).

保护目标（typesafe-ai 6 atomic primitives）：
  - 6 个 primitives 全在；与 verify_typesafe_refactor 第 2 项契约一致
  - 每个 primitive 有正确的 type 字段（choice/score/noul）
  - Choice criteria 必须为 dict（选项 → 描述）
  - Score criteria 必须为 list（有序 rubric）
  - Noul criteria 必须含 true/false 键（A5 Round 2 修复后契约）
  - instructions 字符串或 {question, context} 结构合法
"""
from __future__ import annotations

import pytest

from src.judges.primitives import (
    VOICE_DRIFT,
    REGISTER_MATCH,
    ENTITY_CONSISTENCY,
    LITERAL_RESIDUE,
    SEMANTIC_LOSS,
    STYLE_COMPLIANCE_SCORE,
    ALL_PRIMITIVES,
    build_question_set,
)


# ---------------------------------------------------------------------------
# 完整性
# ---------------------------------------------------------------------------

EXPECTED_PRIMITIVE_IDS = {
    "voice_drift",
    "register_match",
    "entity_consistency",
    "literal_residue",
    "semantic_loss",
    "style_compliance_score",
}


class TestPrimitivesCompleteness:
    def test_all_primitives_count_is_6(self):
        assert len(ALL_PRIMITIVES) == 6

    def test_all_expected_ids_present(self):
        assert set(ALL_PRIMITIVES.keys()) == EXPECTED_PRIMITIVE_IDS

    def test_build_question_set_returns_6_questions(self):
        qs = build_question_set()
        assert len(qs) == 6
        assert set(qs.keys()) == EXPECTED_PRIMITIVE_IDS

    def test_build_question_set_returns_independent_dict(self):
        qs1 = build_question_set()
        qs2 = build_question_set()
        assert qs1 is not qs2  # 返回独立副本（防外部 mutate 全局）
        qs1["voice_drift"]["type"] = "mutated"
        assert qs2["voice_drift"]["type"] != "mutated"

    def test_build_question_set_matches_all_primitives(self):
        assert build_question_set() == ALL_PRIMITIVES


# ---------------------------------------------------------------------------
# 各 primitive 的 type 字段
# ---------------------------------------------------------------------------

class TestPrimitiveTypes:
    def test_voice_drift_is_choice(self):
        assert VOICE_DRIFT["type"] == "choice"

    def test_register_match_is_choice(self):
        assert REGISTER_MATCH["type"] == "choice"

    def test_entity_consistency_is_noul(self):
        assert ENTITY_CONSISTENCY["type"] == "noul"

    def test_literal_residue_is_noul(self):
        assert LITERAL_RESIDUE["type"] == "noul"

    def test_semantic_loss_is_noul(self):
        assert SEMANTIC_LOSS["type"] == "noul"

    def test_style_compliance_score_is_score(self):
        assert STYLE_COMPLIANCE_SCORE["type"] == "score"

    def test_all_types_are_valid(self):
        valid_types = {"choice", "score", "noul"}
        for qid, q in ALL_PRIMITIVES.items():
            assert q["type"] in valid_types, f"{qid} has invalid type: {q['type']}"


# ---------------------------------------------------------------------------
# Choice primitive — criteria 必须是 dict
# ---------------------------------------------------------------------------

class TestChoicePrimitiveCriteria:
    def test_voice_drift_criteria_is_dict(self):
        assert isinstance(VOICE_DRIFT["criteria"], dict)

    def test_voice_drift_criteria_options(self):
        criteria = VOICE_DRIFT["criteria"]
        assert "none" in criteria
        assert "subtle" in criteria
        assert "severe" in criteria

    def test_register_match_criteria_is_dict(self):
        assert isinstance(REGISTER_MATCH["criteria"], dict)

    def test_register_match_criteria_options(self):
        criteria = REGISTER_MATCH["criteria"]
        assert "under" in criteria
        assert "match" in criteria
        assert "over" in criteria


# ---------------------------------------------------------------------------
# Score primitive — criteria 必须是 list（有序）
# ---------------------------------------------------------------------------

class TestScorePrimitiveCriteria:
    def test_style_compliance_criteria_is_list(self):
        assert isinstance(STYLE_COMPLIANCE_SCORE["criteria"], list)

    def test_style_compliance_criteria_non_empty(self):
        assert len(STYLE_COMPLIANCE_SCORE["criteria"]) >= 2

    def test_style_compliance_criteria_entries_are_strings(self):
        for entry in STYLE_COMPLIANCE_SCORE["criteria"]:
            assert isinstance(entry, str)


# ---------------------------------------------------------------------------
# Noul primitive — A5 Round 2 修复契约
# ---------------------------------------------------------------------------

class TestNoulPrimitiveCriteria:
    def test_entity_consistency_criteria_has_true_false(self):
        criteria = ENTITY_CONSISTENCY["criteria"]
        assert "true" in criteria, "ENTITY_CONSISTENCY 缺 criteria.true"
        assert "false" in criteria, "ENTITY_CONSISTENCY 缺 criteria.false"

    def test_literal_residue_criteria_has_true_false(self):
        criteria = LITERAL_RESIDUE["criteria"]
        assert "true" in criteria
        assert "false" in criteria

    def test_semantic_loss_criteria_has_true_false(self):
        criteria = SEMANTIC_LOSS["criteria"]
        assert "true" in criteria
        assert "false" in criteria

    def test_all_noul_primitives_have_true_false(self):
        noul_ids = [qid for qid, q in ALL_PRIMITIVES.items() if q["type"] == "noul"]
        assert len(noul_ids) == 3  # entity/literal/semantic
        for qid in noul_ids:
            q = ALL_PRIMITIVES[qid]
            assert "true" in q["criteria"], f"{qid} 缺 criteria.true"
            assert "false" in q["criteria"], f"{qid} 缺 criteria.false"

    def test_noul_instructions_structured(self):
        # A5 修复后 instructions 是 {question, context} 结构
        for qid in ["entity_consistency", "literal_residue", "semantic_loss"]:
            instr = ALL_PRIMITIVES[qid]["instructions"]
            assert isinstance(instr, dict), f"{qid} instructions 应为 dict"
            assert "question" in instr, f"{qid} instructions 缺 question"
            assert "context" in instr, f"{qid} instructions 缺 context"


# ---------------------------------------------------------------------------
# Instructions 字段类型
# ---------------------------------------------------------------------------

class TestPrimitiveInstructions:
    def test_choice_instructions_is_string(self):
        # Choice 用 string instructions（旧式）
        for qid in ["voice_drift", "register_match"]:
            instr = ALL_PRIMITIVES[qid]["instructions"]
            assert isinstance(instr, str), f"{qid} instructions 应为 str"
            assert len(instr) > 0

    def test_score_instructions_is_string(self):
        instr = STYLE_COMPLIANCE_SCORE["instructions"]
        assert isinstance(instr, str)
        assert len(instr) > 0


# ---------------------------------------------------------------------------
# 与 verify_typesafe_refactor 契约一致
# ---------------------------------------------------------------------------

class TestContractWithVerifyScript:
    """这些 primitive 名与 tools/verify_typesafe_refactor.py 第 2 项检查一致"""

    def test_primitive_names_match_typesafe_contract(self):
        # verify_typesafe_refactor.py 第 2 项（test_02_verdict_synthesis）依赖这些名字
        required_names = {
            "voice_drift",
            "register_match",
            "entity_consistency",
            "literal_residue",
            "semantic_loss",
            "style_compliance_score",
        }
        assert required_names.issubset(set(ALL_PRIMITIVES.keys()))

    def test_three_noul_two_choice_one_score(self):
        # 6 个 primitives 的类型分布契约：3 noul + 2 choice + 1 score
        type_count = {"noul": 0, "choice": 0, "score": 0}
        for q in ALL_PRIMITIVES.values():
            type_count[q["type"]] += 1
        assert type_count == {"noul": 3, "choice": 2, "score": 1}
