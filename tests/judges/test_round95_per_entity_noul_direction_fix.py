"""Round 95: per-entity Noul 方向反转修复 — 端到端验证

修复前: per-entity noul=0.95 (合规) → max(0.95)=0.95 → ≥ 0.3 → reject (false positive)
修复后: per-entity noul=0.95 (合规) → max(1-0.95)=0.05 → < 0.3 → pass (correct)

也覆盖 Mock 行为对齐 Jev 方向（合规=高 noul）。
"""
import pytest

from src.judges.primitives import (
    PER_ENTITY_KEY_PREFIX,
    aggregate_per_entity_nouls,
    build_per_entity_questions,
    build_question_set,
)
from src.judges.typesafe_backend import MockBackend
from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer
from src.judges.verdict import synthesize_verdict


# ---------------------------------------------------------------------------
# 聚合方向
# ---------------------------------------------------------------------------

class TestPerEntityAggregationDirection:
    def test_all_compliant_low_violation_prob(self):
        # 全部高 noul (合规) → 聚合 max(1-noul) 接近 0
        answers = {f"{PER_ENTITY_KEY_PREFIX}A": NoulAnswer(noul=0.95) for A in "ABCDE"}
        assert aggregate_per_entity_nouls(answers) == pytest.approx(0.05, abs=0.01)

    def test_all_violation_high_violation_prob(self):
        # 全部低 noul (违规) → 聚合 max(1-noul) 接近 1
        answers = {f"{PER_ENTITY_KEY_PREFIX}A": NoulAnswer(noul=0.05) for A in "ABCDE"}
        assert aggregate_per_entity_nouls(answers) == pytest.approx(0.95, abs=0.01)

    def test_one_violation_in_many(self):
        # 9 个合规 + 1 个违规 → 聚合 = 1-0.05 = 0.95
        answers = {f"{PER_ENTITY_KEY_PREFIX}A": NoulAnswer(noul=0.95) for A in range(9)}
        answers[f"{PER_ENTITY_KEY_PREFIX}bad"] = NoulAnswer(noul=0.05)
        assert aggregate_per_entity_nouls(answers) == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------------------
# synthesize_verdict 端到端：Jev 风格真实场景
# ---------------------------------------------------------------------------

def _pass_baseline_answers():
    """返回所有非 entity 维度都 PASS 的 baseline answers dict"""
    return {
        "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
        "register_match": ChoiceAnswer("match", {"match": 0.9, "under": 0.05, "over": 0.05}, 0.8),
        "literal_residue": NoulAnswer(noul=0.05),
        "semantic_loss": NoulAnswer(noul=0.05),
        "style_compliance_score": ScoreAnswer(2.0, {0: 0.05, 1: 0.15, 2: 0.8}, 0.8),
    }


class TestVerdictWithJevStylePerEntity:
    def test_real_jev_compliant_pass(self):
        """模拟真实 Jev：合规 entity 给出 noul≈0.95 → 应 PASS"""
        answers = _pass_baseline_answers()
        answers["entity_consistency"] = NoulAnswer(noul=0.05)
        for ent in ("A", "B", "C"):
            answers[f"{PER_ENTITY_KEY_PREFIX}{ent}"] = NoulAnswer(noul=0.95)
        v = synthesize_verdict(answers)
        assert v.decision == "PASS", f"Expected PASS, got {v.decision}: {v.reject_reasons}"

    def test_real_jev_one_violation_rejects(self):
        """模拟真实 Jev：1 个 entity 违规 → 应 REJECT"""
        answers = _pass_baseline_answers()
        answers["entity_consistency"] = NoulAnswer(noul=0.05)
        answers[f"{PER_ENTITY_KEY_PREFIX}A"] = NoulAnswer(noul=0.95)
        answers[f"{PER_ENTITY_KEY_PREFIX}B"] = NoulAnswer(noul=0.95)
        answers[f"{PER_ENTITY_KEY_PREFIX}C"] = NoulAnswer(noul=0.05)  # 违规
        v = synthesize_verdict(answers)
        assert v.decision == "REJECT"
        assert any("实体一致性违反" in r and "per-entity" in r for r in v.reject_reasons)


# ---------------------------------------------------------------------------
# MockBackend 行为对齐
# ---------------------------------------------------------------------------

class TestMockBackendAlignedDirection:
    def test_mock_canonical_in_lit_compliant(self):
        """Mock 对齐 Jev: canonical 在 lit → noul=1.0 (合规)"""
        b = MockBackend()
        qs = build_per_entity_questions({
            "George": {"canonical": "乔治", "variants": []},
        })
        out = b.evaluate(
            state={
                "source_text": "x", "raw_translation": "x",
                "literary_translation": "乔治来到房间。",
                "entity_registry": {"George": "乔治"},
                "entity_variants": {},
            },
            questions=qs,
        )
        per = next(v for k, v in out.items() if k.startswith(PER_ENTITY_KEY_PREFIX))
        assert per.noul == pytest.approx(1.0, abs=0.01)

    def test_mock_canonical_missing_violation(self):
        """Mock 对齐 Jev: canonical 缺失 → noul=0.0 (违规)"""
        b = MockBackend()
        qs = build_per_entity_questions({
            "Lomax": {"canonical": "洛马克斯先生", "variants": []},
        })
        out = b.evaluate(
            state={
                "source_text": "x", "raw_translation": "x",
                "literary_translation": "另一个人在房间。",
                "entity_registry": {"Lomax": "洛马克斯先生"},
                "entity_variants": {},
            },
            questions=qs,
        )
        per = next(v for k, v in out.items() if k.startswith(PER_ENTITY_KEY_PREFIX))
        assert per.noul == pytest.approx(0.0, abs=0.01)

    def test_mock_variant_in_lit_compliant(self):
        """Mock 对齐 Jev: variant 在 lit → noul=1.0 (合规)"""
        b = MockBackend()
        qs = build_per_entity_questions({
            "George": {"canonical": "乔治·洛马克斯先生", "variants": ["乔治·洛马克斯"]},
        })
        out = b.evaluate(
            state={
                "source_text": "x", "raw_translation": "x",
                "literary_translation": "乔治·洛马克斯走进房间。",
                "entity_registry": {"George": "乔治·洛马克斯先生"},
                "entity_variants": {"George": ["乔治·洛马克斯"]},
            },
            questions=qs,
        )
        per = next(v for k, v in out.items() if k.startswith(PER_ENTITY_KEY_PREFIX))
        assert per.noul == pytest.approx(1.0, abs=0.01)

    def test_mock_aggregate_all_compliant(self):
        """Mock 端到端: 全部 entity 合规 → 聚合 max(1-noul)≈0 (无违规概率)"""
        from src.judges.primitives import aggregate_per_entity_nouls
        b = MockBackend()
        qs = build_per_entity_questions({
            "George": {"canonical": "乔治", "variants": []},
            "Lomax": {"canonical": "洛马克斯先生", "variants": []},
        })
        out = b.evaluate(
            state={
                "source_text": "x", "raw_translation": "x",
                "literary_translation": "乔治与洛马克斯先生交谈。",
                "entity_registry": {"George": "乔治", "Lomax": "洛马克斯先生"},
                "entity_variants": {},
            },
            questions=qs,
        )
        agg = aggregate_per_entity_nouls(out)
        # 全部合规 noul=1.0 → 聚合 max(1-1.0)=0
        assert agg == pytest.approx(0.0, abs=0.01)
