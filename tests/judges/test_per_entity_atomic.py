"""Round 61: 单元测试 - primitives.per-entity atomic Noul + verdict 聚合.

保护目标：
  - build_per_entity_questions 为每个实体生成 atomic Noul（不是复合 Noul）
  - 每个 question 自带 canonical + variants 描述（Jev 在小 context 上判断）
  - key 合法化（特殊字符、空白、纯数字首字符）→ 全部变成合法 Python identifier
  - key 冲突处理（两个 source_name 净化后相同）→ 追加 _2, _3
  - aggregate_per_entity_nouls 用 max 聚合（OR 语义）
  - synthesize_verdict 在 per-entity Nouls 存在时覆盖 entity_consistency 维度
  - per-entity Nouls 仍参与 overall_conf 聚合
"""
from __future__ import annotations

import pytest

from src.judges.primitives import (
    build_per_entity_questions,
    build_question_set,
    aggregate_per_entity_nouls,
    PER_ENTITY_KEY_PREFIX,
    _sanitize_key,
)
from src.judges.typesafe_spec import NoulAnswer, ChoiceAnswer, ScoreAnswer
from src.judges.verdict import synthesize_verdict


# ---------------------------------------------------------------------------
# _sanitize_key：key 合法化
# ---------------------------------------------------------------------------

class TestSanitizeKey:
    def test_simple_alphanumeric(self):
        used = set()
        k = _sanitize_key("George", used)
        assert k == "entity_per_George"
        assert k in used

    def test_space_replaced_with_underscore(self):
        used = set()
        k = _sanitize_key("Mr. Lomax", used)
        # "." 和 " " 都替换为 "_"
        assert k == "entity_per_Mr__Lomax"

    def test_unicode_chinese_kept(self):
        # 中文 entity key 不常见但应保留（CJK 在 identifier 中合法）
        used = set()
        k = _sanitize_key("洛马克斯", used)
        assert "洛马克斯" in k

    def test_pure_digit_prefix_gets_prefix(self):
        # 首字符是数字 → 加 e_ 前缀
        used = set()
        k = _sanitize_key("3rdPerson", used)
        assert k[0].isalpha()
        assert "3rdPerson" in k

    def test_empty_string_handled(self):
        used = set()
        k = _sanitize_key("", used)
        # 空字符串净化后为空，加 e_ 前缀
        assert k.startswith(PER_ENTITY_KEY_PREFIX)

    def test_collision_appends_suffix(self):
        used = set()
        # 两个 source_name 净化后相同
        k1 = _sanitize_key("Mr. Lomax", used)
        k2 = _sanitize_key("Mr__Lomax", used)  # 不同输入但净化后相同
        # 第二 key 应追加 _2
        assert k1 != k2
        assert k2.endswith("_2")


# ---------------------------------------------------------------------------
# build_per_entity_questions
# ---------------------------------------------------------------------------

class TestBuildPerEntityQuestions:
    def test_empty_registry_returns_empty_dict(self):
        assert build_per_entity_questions({}) == {}

    def test_basic_entity_with_canonical(self):
        registry = {"George": {"canonical": "乔治", "variants": []}}
        qs = build_per_entity_questions(registry)
        assert len(qs) == 1
        key = list(qs.keys())[0]
        assert key.startswith(PER_ENTITY_KEY_PREFIX)
        q = qs[key]
        assert q["type"] == "noul"
        assert "乔治" in q["instructions"]["question"]
        assert "George" in q["instructions"]["question"]

    def test_entity_with_variants_in_question(self):
        registry = {
            "Mr. Lomax": {
                "canonical": "乔治·洛马克斯先生",
                "variants": ["洛马克斯先生", "洛马克斯"],
            }
        }
        qs = build_per_entity_questions(registry)
        key = list(qs.keys())[0]
        q = qs[key]
        # canonical + variants 都应出现在 instructions
        assert "乔治·洛马克斯先生" in q["instructions"]["question"]
        assert "洛马克斯先生" in q["instructions"]["question"]
        assert "洛马克斯" in q["instructions"]["question"]
        # criteria.true 描述合规（含 canonical 或 variants）
        assert "乔治·洛马克斯先生" in q["criteria"]["true"]
        assert "洛马克斯先生" in q["criteria"]["true"]

    def test_entity_string_format_backward_compat(self):
        # entity_registry 旧格式：value 直接是 str（无 variants）
        registry = {"George": "乔治"}
        qs = build_per_entity_questions(registry)
        assert len(qs) == 1

    def test_multiple_entities(self):
        registry = {
            "George": {"canonical": "乔治", "variants": []},
            "Mr. Lomax": {"canonical": "洛马克斯先生", "variants": []},
            "Codders": {"canonical": "科德斯", "variants": ["科德斯（Codders）"]},
        }
        qs = build_per_entity_questions(registry)
        assert len(qs) == 3
        for q in qs.values():
            assert q["type"] == "noul"

    def test_skip_empty_canonical(self):
        # canonical 为空 → 跳过
        registry = {"X": {"canonical": "", "variants": []}}
        assert build_per_entity_questions(registry) == {}

    def test_skip_empty_source_name(self):
        registry = {"": {"canonical": "乔治", "variants": []}}
        assert build_per_entity_questions(registry) == {}

    def test_criteria_have_true_false(self):
        # A5 修复契约：所有 Noul criteria 必须含 true/false
        registry = {"George": {"canonical": "乔治", "variants": ["乔"]}}
        qs = build_per_entity_questions(registry)
        q = list(qs.values())[0]
        assert "true" in q["criteria"]
        assert "false" in q["criteria"]


# ---------------------------------------------------------------------------
# aggregate_per_entity_nouls
# ---------------------------------------------------------------------------

class TestAggregatePerEntityNouls:
    def test_empty_returns_none(self):
        assert aggregate_per_entity_nouls({}) is None

    def test_ignores_non_per_entity_keys(self):
        answers = {
            "entity_consistency": NoulAnswer(noul=0.5),
            "literal_residue": NoulAnswer(noul=0.8),
        }
        assert aggregate_per_entity_nouls(answers) is None

    def test_or_violation_aggregation(self):
        # Round 95: 聚合语义 max(1 - noul) = "OR 违规" 概率.
        # per-entity noul = P(合规)，此聚合等价于 1 - min(noul).
        # George 0.3 (强违规) → 1-0.3=0.7; Mr_Lomax 0.8 → 1-0.8=0.2; Codders 0.4 → 1-0.4=0.6.
        # max(0.7, 0.2, 0.6) = 0.7.
        answers = {
            "entity_per_George": NoulAnswer(noul=0.3),
            "entity_per_Mr_Lomax": NoulAnswer(noul=0.8),
            "entity_per_Codders": NoulAnswer(noul=0.4),
        }
        result = aggregate_per_entity_nouls(answers)
        assert result == pytest.approx(0.7, abs=0.01)

    def test_all_compliant_aggregation(self):
        # 全部合规 (noul=0.95) → 聚合 = max(1-0.95) = 0.05 (近 0, 无违规).
        answers = {
            "entity_per_George": NoulAnswer(noul=0.95),
            "entity_consistency": NoulAnswer(noul=0.99),  # 不是 per-entity，应忽略
            "literal_residue": NoulAnswer(noul=0.99),     # 不是 per-entity，应忽略
        }
        result = aggregate_per_entity_nouls(answers)
        assert result == pytest.approx(0.05, abs=0.01)


# ---------------------------------------------------------------------------
# synthesize_verdict 集成：per-entity 覆盖 entity_consistency
# ---------------------------------------------------------------------------

class TestSynthesizeVerdictPerEntityAggregation:
    def test_per_entity_max_overrides_composite(self):
        # Round 95: 新语义 — per-entity noul = P(合规); 低 noul = 违规.
        # composite ENTITY_CONSISTENCY noul=0.1（PASS，因为 composite 的 noul 是违规概率，0.1 低 = 合规）
        # per-entity Mr_Lomax noul=0.2（违规） → 1-0.2=0.8 → max 聚合 = 0.8 → REJECT.
        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
            "register_match": ChoiceAnswer("match", {"match": 0.9, "under": 0.05, "over": 0.05}, 0.8),
            "entity_consistency": NoulAnswer(noul=0.1),  # composite 没违规
            "literal_residue": NoulAnswer(noul=0.05),
            "semantic_loss": NoulAnswer(noul=0.05),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.8),
            "entity_per_George": NoulAnswer(noul=0.9),
            "entity_per_Mr_Lomax": NoulAnswer(noul=0.2),  # 违规 → 1-0.2=0.8 → 应覆盖 composite
            "entity_per_Codders": NoulAnswer(noul=0.9),
        }
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"
        assert any("实体一致性违反" in r and "per-entity" in r for r in v.reject_reasons)

    def test_per_entity_all_low_no_reject(self):
        # Round 95: per-entity 全部高 noul（合规） → 1-noul 都小 → 聚合 ≤ 0.1 → PASS.
        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
            "register_match": ChoiceAnswer("match", {"match": 0.9, "under": 0.05, "over": 0.05}, 0.8),
            "entity_consistency": NoulAnswer(noul=0.1),
            "literal_residue": NoulAnswer(noul=0.05),
            "semantic_loss": NoulAnswer(noul=0.05),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.8),
            "entity_per_George": NoulAnswer(noul=0.9),
            "entity_per_Mr_Lomax": NoulAnswer(noul=0.9),
            "entity_per_Codders": NoulAnswer(noul=0.95),
        }
        v = synthesize_verdict(answers)
        assert v.decision == "PASS"

    def test_composite_used_when_no_per_entity(self):
        # 没有 per-entity Noul → 使用 composite
        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
            "register_match": ChoiceAnswer("match", {"match": 0.9, "under": 0.05, "over": 0.05}, 0.8),
            "entity_consistency": NoulAnswer(noul=0.8),  # composite 违规
            "literal_residue": NoulAnswer(noul=0.05),
            "semantic_loss": NoulAnswer(noul=0.05),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.8),
        }
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"
        # 没有 per-entity 标记
        assert not any("per-entity" in r for r in v.reject_reasons)

    def test_per_entity_in_uncertain_zone_skipped(self):
        # Round 95: 新语义 — per-entity 全部 0.5（不确定区） → 1-0.5=0.5 也在不确定区 → skip.
        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
            "register_match": ChoiceAnswer("match", {"match": 0.9, "under": 0.05, "over": 0.05}, 0.8),
            "entity_consistency": NoulAnswer(noul=0.1),
            "literal_residue": NoulAnswer(noul=0.05),
            "semantic_loss": NoulAnswer(noul=0.05),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.8),
            "entity_per_George": NoulAnswer(noul=0.5),  # 1-0.5=0.5 → 在 uncertain zone
            "entity_per_Mr_Lomax": NoulAnswer(noul=0.5),
        }
        v = synthesize_verdict(answers)
        assert v.decision == "PASS"

    def test_per_entity_above_uncertain_high_triggers_reject(self):
        # Round 95: 新语义 — per-entity noul=0.2 (违规) → 1-0.2=0.8 → 区间外上侧 → REJECT.
        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
            "register_match": ChoiceAnswer("match", {"match": 0.9, "under": 0.05, "over": 0.05}, 0.8),
            "entity_consistency": NoulAnswer(noul=0.1),
            "literal_residue": NoulAnswer(noul=0.05),
            "semantic_loss": NoulAnswer(noul=0.05),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.8),
            "entity_per_George": NoulAnswer(noul=0.2),  # 1-0.2=0.8 → 区间外上侧
        }
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"

    def test_per_entity_summary_marked_as_aggregated(self):
        # Round 95: per-entity 存在时，summary 中的 entity_consistency 应有聚合标记.
        # entity_per_George noul=0.2 (违规) → 聚合 max(1-0.2, 1-0.9) = max(0.8, 0.1) = 0.8.
        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
            "register_match": ChoiceAnswer("match", {"match": 0.9, "under": 0.05, "over": 0.05}, 0.8),
            "entity_consistency": NoulAnswer(noul=0.5),
            "literal_residue": NoulAnswer(noul=0.05),
            "semantic_loss": NoulAnswer(noul=0.05),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.8),
            "entity_per_George": NoulAnswer(noul=0.2),  # 违规 → 1-0.2=0.8
            "entity_per_Mr_Lomax": NoulAnswer(noul=0.9),  # 合规 → 1-0.9=0.1
        }
        v = synthesize_verdict(answers)
        ec = v.primitive_summary.get("entity_consistency", {})
        assert ec.get("_aggregated_from_per_entity") is True
        assert ec.get("noul") == pytest.approx(0.8, abs=0.01)

    def test_per_entity_in_overall_conf(self):
        # per-entity Nouls 应参与 overall_conf 聚合
        answers = {
            "voice_drift": ChoiceAnswer("none", {"none": 1.0}, 0.5),  # conf=0.5 → overall conf 上限 0.5
            "register_match": ChoiceAnswer("match", {"match": 1.0}, 0.5),
            "entity_consistency": NoulAnswer(noul=0.1),
            "literal_residue": NoulAnswer(noul=0.05),
            "semantic_loss": NoulAnswer(noul=0.05),
            "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.5),
            "entity_per_George": NoulAnswer(noul=0.5),  # max(0.5, 0.5) = 0.5
            "entity_per_Mr_Lomax": NoulAnswer(noul=0.5),
        }
        v = synthesize_verdict(answers)
        # overall_conf 应是 min(0.5, 0.5, 0.9, 0.95, 0.95, 0.5, 0.5, 0.5) = 0.5
        assert v.confidence == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# end-to-end: build_question_set + per-entity 合并（evaluate_chunk 内部用）
# ---------------------------------------------------------------------------

class TestBuildQuestionSetPlusPerEntity:
    def test_base_questions_have_6_keys(self):
        qs = build_question_set()
        assert len(qs) == 6

    def test_merge_base_plus_per_entity(self):
        base = build_question_set()
        per_entity = build_per_entity_questions({
            "George": {"canonical": "乔治", "variants": []},
            "Mr. Lomax": {"canonical": "洛马克斯先生", "variants": ["洛马克斯"]},
        })
        merged = {**base, **per_entity}
        assert len(merged) == 8
        # base keys 全部保留
        for k in ("voice_drift", "register_match", "entity_consistency",
                  "literal_residue", "semantic_loss", "style_compliance_score"):
            assert k in merged
        # per-entity keys 都加进来
        assert any(k.startswith(PER_ENTITY_KEY_PREFIX) for k in merged.keys())
