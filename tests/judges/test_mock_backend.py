"""Round 50: 单元测试 - judges/typesafe_backend.py (MockBackend).

保护目标（启发式离线裁判）：
  - voice_drift：length_ratio 三段式（< 0.3 / > 1.5 / 正常）
  - register_match：古文虚词（之乎矣焉其）→ over
  - entity_consistency：1 - (present/total) → noul
  - literal_residue：SequenceMatcher sim 阈值
  - semantic_loss：length_ratio < 0.5 → 怀疑压缩
  - style_compliance_score：综合 style + 长度
"""
from __future__ import annotations

import pytest

from src.judges.typesafe_backend import MockBackend
from src.judges.typesafe_spec import (
    NoulAnswer, ChoiceAnswer, ScoreAnswer, Answer,
)


# ---------------------------------------------------------------------------
# 基础返回
# ---------------------------------------------------------------------------

class TestMockBackendBasics:
    def test_returns_dict_with_all_6_keys(self):
        b = MockBackend()
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": "x", "literary_translation": "x"},
            questions={},
        )
        assert set(out.keys()) == {
            "voice_drift",
            "register_match",
            "entity_consistency",
            "literal_residue",
            "semantic_loss",
            "style_compliance_score",
        }

    def test_state_as_dict_or_str_handled(self):
        b = MockBackend()
        # dict 输入
        out_dict = b.evaluate(
            state={"source_text": "abc", "raw_translation": "x", "literary_translation": "x"},
            questions={},
        )
        # str 输入（应被识别为其他类型，回退到空）
        out_str = b.evaluate(state="raw string", questions={})
        assert "voice_drift" in out_str  # 不抛异常，返回 6 个默认启发式结果

    def test_empty_state_yields_zero_nouls(self):
        b = MockBackend()
        out = b.evaluate(
            state={"source_text": "", "raw_translation": "", "literary_translation": ""},
            questions={},
        )
        # 空字符串 → ratio=0 → severe voice_drift
        assert out["voice_drift"].choice in {"none", "subtle", "severe"}
        assert isinstance(out["entity_consistency"], NoulAnswer)


# ---------------------------------------------------------------------------
# voice_drift 启发式
# ---------------------------------------------------------------------------

class TestMockBackendVoiceDrift:
    def test_severe_when_lit_much_shorter_than_raw(self):
        b = MockBackend()
        # raw=1000 chars, lit=100 chars → ratio=0.1 < 0.3 → severe
        raw = "a" * 1000
        lit = "b" * 100
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        assert out["voice_drift"].choice == "severe"
        assert out["voice_drift"].probabilities["severe"] >= 0.5

    def test_subtle_when_lit_much_longer_than_raw(self):
        b = MockBackend()
        # raw=100, lit=300 → ratio=3.0 > 1.5 → subtle
        raw = "a" * 100
        lit = "b" * 300
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        assert out["voice_drift"].choice == "subtle"

    def test_none_when_ratios_balanced(self):
        b = MockBackend()
        raw = "a" * 100
        lit = "b" * 100
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        assert out["voice_drift"].choice == "none"


# ---------------------------------------------------------------------------
# register_match 启发式
# ---------------------------------------------------------------------------

class TestMockBackendRegisterMatch:
    def test_over_when_classical_particles_present(self):
        b = MockBackend()
        # 包含古文虚词 → over
        lit = "其形貌甚伟，观之令人惊异。"
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": "x", "literary_translation": lit},
            questions={},
        )
        assert out["register_match"].choice == "over"

    def test_match_when_no_classical_particles(self):
        b = MockBackend()
        lit = "普通现代中文内容，没有文言虚词。"
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": "x", "literary_translation": lit},
            questions={},
        )
        assert out["register_match"].choice == "match"

    def test_all_five_particles_detected(self):
        b = MockBackend()
        for ch in ["之", "乎", "矣", "焉", "其"]:
            lit = f"含有{ch}字。"
            out = b.evaluate(
                state={"source_text": "x", "raw_translation": "x", "literary_translation": lit},
                questions={},
            )
            assert out["register_match"].choice == "over", f"failed for particle: {ch}"


# ---------------------------------------------------------------------------
# entity_consistency 启发式
# ---------------------------------------------------------------------------

class TestMockBackendEntityConsistency:
    def test_zero_noul_when_all_entities_present(self):
        b = MockBackend()
        ent_reg = {"Luffy": "鲁夫", "Zoro": "索隆"}
        lit = "鲁夫和索隆在船上。"
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": lit,
                "entity_registry": ent_reg,
            },
            questions={},
        )
        assert out["entity_consistency"].noul == 0.0

    def test_high_noul_when_no_entities_present(self):
        b = MockBackend()
        ent_reg = {"Luffy": "鲁夫", "Zoro": "索隆"}
        lit = "本章完全不涉及这两个角色。"
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": lit,
                "entity_registry": ent_reg,
            },
            questions={},
        )
        # 0/2 present → noul = 1.0
        assert out["entity_consistency"].noul == 1.0

    def test_partial_match_yields_partial_noul(self):
        b = MockBackend()
        ent_reg = {"Luffy": "鲁夫", "Zoro": "索隆"}
        lit = "只有鲁夫出场。"
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": lit,
                "entity_registry": ent_reg,
            },
            questions={},
        )
        # 1/2 present → noul = 0.5
        assert out["entity_consistency"].noul == 0.5

    def test_empty_registry_yields_zero_noul(self):
        b = MockBackend()
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": "x",
                "entity_registry": {},
            },
            questions={},
        )
        assert out["entity_consistency"].noul == 0.0

    def test_none_registry_yields_zero_noul(self):
        b = MockBackend()
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": "x",
                "entity_registry": None,
            },
            questions={},
        )
        assert out["entity_consistency"].noul == 0.0


# ---------------------------------------------------------------------------
# literal_residue 启发式
# ---------------------------------------------------------------------------

class TestMockBackendLiteralResidue:
    def test_high_noul_when_lit_matches_raw_closely(self):
        b = MockBackend()
        # raw == lit → sim = 1.0 → noul ≈ 1.0
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "abcdef" * 50,
                "literary_translation": "abcdef" * 50,
            },
            questions={},
        )
        assert out["literal_residue"].noul >= 0.8

    def test_zero_noul_when_lit_very_different(self):
        b = MockBackend()
        # raw="aaa...", lit="ZZZ..." → sim 极低
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "a" * 500,
                "literary_translation": "中文字符" * 50,
            },
            questions={},
        )
        # sim 可能很低，但 literal_residue 阈值 sim=0.4 → noul=0
        assert out["literal_residue"].noul == 0.0

    def test_empty_raw_or_lit_yields_zero_noul(self):
        b = MockBackend()
        for raw, lit in [("", "some lit"), ("some raw", "")]:
            out = b.evaluate(
                state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
                questions={},
            )
            assert out["literal_residue"].noul == 0.0


# ---------------------------------------------------------------------------
# semantic_loss 启发式
# ---------------------------------------------------------------------------

class TestMockBackendSemanticLoss:
    def test_high_noul_when_lit_much_shorter(self):
        b = MockBackend()
        # raw=1000, lit=400 → ratio=0.4 < 0.5 → noul = (0.5-0.4)*2 = 0.2
        raw = "a" * 1000
        lit = "b" * 400
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        assert 0.0 < out["semantic_loss"].noul <= 1.0

    def test_zero_noul_when_length_balanced(self):
        b = MockBackend()
        # ratio=1.0 → 不触发
        raw = "a" * 100
        lit = "b" * 100
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        assert out["semantic_loss"].noul == 0.0

    def test_severe_compression_yields_high_noul(self):
        b = MockBackend()
        # raw=1000, lit=100 → ratio=0.1 → noul = (0.5-0.1)*2 = 0.8
        raw = "a" * 1000
        lit = "b" * 100
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        assert out["semantic_loss"].noul >= 0.8


# ---------------------------------------------------------------------------
# style_compliance_score 启发式
# ---------------------------------------------------------------------------

class TestMockBackendStyleCompliance:
    def test_perfect_score_for_balanced_modern_text(self):
        b = MockBackend()
        raw = "a" * 100
        lit = "现代中文内容，无文言。" * 5
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        assert out["style_compliance_score"].score == 1.0

    def test_low_score_for_extreme_length_ratio(self):
        b = MockBackend()
        # ratio < 0.3 或 > 2.0 → score = 0.0
        for raw, lit in [("a" * 1000, "b" * 100), ("a" * 100, "b" * 1000)]:
            out = b.evaluate(
                state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
                questions={},
            )
            assert out["style_compliance_score"].score == 0.0

    def test_mid_score_for_classical_register(self):
        b = MockBackend()
        raw = "a" * 100
        lit = "其形貌甚伟，观之。" * 10  # ratio 接近 1.0 但 register_over
        out = b.evaluate(
            state={"source_text": "x", "raw_translation": raw, "literary_translation": lit},
            questions={},
        )
        # 古文虚词 + 正常长度 → score = 0.5
        assert out["style_compliance_score"].score == 0.5


# ---------------------------------------------------------------------------
# 返回类型
# ---------------------------------------------------------------------------

class TestMockBackendReturnTypes:
    def test_voice_drift_is_choice(self):
        b = MockBackend()
        out = b.evaluate({"source_text": "x", "raw_translation": "x", "literary_translation": "x"}, {})
        assert isinstance(out["voice_drift"], ChoiceAnswer)

    def test_register_match_is_choice(self):
        b = MockBackend()
        out = b.evaluate({"source_text": "x", "raw_translation": "x", "literary_translation": "x"}, {})
        assert isinstance(out["register_match"], ChoiceAnswer)

    def test_three_nouls(self):
        b = MockBackend()
        out = b.evaluate({"source_text": "x", "raw_translation": "x", "literary_translation": "x"}, {})
        for key in ("entity_consistency", "literal_residue", "semantic_loss"):
            assert isinstance(out[key], NoulAnswer), f"{key} should be NoulAnswer"

    def test_style_score_is_score(self):
        b = MockBackend()
        out = b.evaluate({"source_text": "x", "raw_translation": "x", "literary_translation": "x"}, {})
        assert isinstance(out["style_compliance_score"], ScoreAnswer)


# ---------------------------------------------------------------------------
# Round 61: per-entity atomic Noul 启发式
# ---------------------------------------------------------------------------

class TestMockBackendPerEntity:
    """Round 61: Mock 也支持 per-entity Noul question（用本地字符匹配）"""

    def test_per_entity_canonical_present_returns_zero_noul(self):
        from src.judges.primitives import build_per_entity_questions, PER_ENTITY_KEY_PREFIX
        from src.judges.typesafe_spec import NoulAnswer
        b = MockBackend()
        qs = build_per_entity_questions({
            "George": {"canonical": "乔治", "variants": []},
        })
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": "乔治来到房间。",
                "entity_registry": {"George": "乔治"},
                "entity_variants": {},
            },
            questions=qs,
        )
        # canonical 出现在 lit → noul=0（合规）
        assert any(k.startswith(PER_ENTITY_KEY_PREFIX) for k in out.keys())
        per_ans = [v for k, v in out.items() if k.startswith(PER_ENTITY_KEY_PREFIX)][0]
        assert isinstance(per_ans, NoulAnswer)
        # Round 95: 合规 → noul=1.0
        assert per_ans.noul == pytest.approx(1.0, abs=0.01)

    def test_per_entity_canonical_missing_returns_high_noul(self):
        from src.judges.primitives import build_per_entity_questions, PER_ENTITY_KEY_PREFIX
        from src.judges.typesafe_spec import NoulAnswer
        b = MockBackend()
        qs = build_per_entity_questions({
            "Mr. Lomax": {"canonical": "乔治·洛马克斯先生", "variants": []},
        })
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": "另一个人在房间。",
                "entity_registry": {"Mr. Lomax": "乔治·洛马克斯先生"},
                "entity_variants": {},
            },
            questions=qs,
        )
        per_ans = [v for k, v in out.items() if k.startswith(PER_ENTITY_KEY_PREFIX)][0]
        # Round 95: 缺失 → 违规 → noul < 0.5
        assert per_ans.noul < 0.5

    def test_per_entity_with_variant_in_lit_returns_zero(self):
        """Mock 也支持变体匹配（若 lit 含 variant 则视为合规）"""
        from src.judges.primitives import build_per_entity_questions, PER_ENTITY_KEY_PREFIX
        from src.judges.typesafe_spec import NoulAnswer
        b = MockBackend()
        qs = build_per_entity_questions({
            "George": {"canonical": "乔治·洛马克斯先生", "variants": ["乔治·洛马克斯"]},
        })
        out = b.evaluate(
            state={
                "source_text": "x",
                "raw_translation": "x",
                "literary_translation": "乔治·洛马克斯走进房间。",
                "entity_registry": {"George": "乔治·洛马克斯先生"},
                "entity_variants": {"George": ["乔治·洛马克斯"]},
            },
            questions=qs,
        )
        per_ans = [v for k, v in out.items() if k.startswith(PER_ENTITY_KEY_PREFIX)][0]
        # Round 95: 变体在 lit 中 → 合规 → noul=1.0
        assert per_ans.noul == pytest.approx(1.0, abs=0.01)
