"""Round 105 P2-3: Jev 实体违规拒因可执行化。

ch32 倒查根因：entity_consistency REJECT 拒因只有 "noul=0.82 eff_conf=0.82"，
没列出具体该统一哪些实体 → rewriter 无从下手 → 逐字节停滞 → PERMANENTLY_FAILED。
修复：拒因追加「违规实体(须统一为规范主名): George→乔治·洛马克斯先生(noul=0.20), ...」。
"""
from __future__ import annotations

from src.judges.typesafe_spec import NoulAnswer
from src.judges.verdict import synthesize_verdict, flagged_entity_labels


def _answers_with_entities():
    return {
        # 复合 entity_consistency 答案（真实运行同时有复合 + per-entity）
        "entity_consistency": NoulAnswer(noul=0.8, confidence=0.8),
        "entity_per_George": NoulAnswer(noul=0.2, confidence=0.8),
        "entity_per_Mr_Lomax": NoulAnswer(noul=0.3, confidence=0.7),
        "entity_per_Daisy": NoulAnswer(noul=0.9, confidence=0.9),
        # 非 per-entity 维度给合规值，避免其它维度干扰
        "voice_drift": NoulAnswer(noul=0.1, confidence=0.9),
        "literal_residue": NoulAnswer(noul=0.2, confidence=0.8),
        "semantic_loss": NoulAnswer(noul=0.2, confidence=0.8),
    }


REG = {
    "George": {"canonical": "乔治·洛马克斯先生", "variants": []},
    "Mr. Lomax": {"canonical": "洛马克斯先生", "variants": []},
    "Daisy": {"canonical": "黛西", "variants": []},
}


class TestFlaggedEntityLabels:
    def test_labels_include_canonical(self):
        labels = flagged_entity_labels(_answers_with_entities(), 0.3, REG)
        assert any("George→乔治·洛马克斯先生" in l for l in labels), labels
        assert any("Mr_Lomax" in l or "Mr. Lomax" in l for l in labels), labels
        # 合规实体（Daisy noul=0.9）不应出现
        assert not any("Daisy" in l or "黛西" in l for l in labels), labels

    def test_labels_without_registry(self):
        labels = flagged_entity_labels(_answers_with_entities(), 0.3)
        assert any("George" in l for l in labels), labels
        assert not any("Daisy" in l for l in labels), labels

    def test_no_flagged_when_all_compliant(self):
        answers = {
            "entity_per_George": NoulAnswer(noul=0.9, confidence=0.9),
        }
        assert flagged_entity_labels(answers, 0.3, REG) == []


class TestSynthesizeVerdictActionable:
    def test_reject_reason_lists_flagged_entities(self):
        answers = _answers_with_entities()
        verdict = synthesize_verdict(
            answers, lit_text="乔治·洛马克斯摆出神气。这是一段中文译文。",
            entity_registry_full=REG,
        )
        assert verdict.decision == "REJECT", verdict.reject_reasons
        entity_reasons = [r for r in verdict.reject_reasons if "实体一致性" in r]
        assert entity_reasons, verdict.reject_reasons
        # 关键：拒因含具体违规实体 + 规范主名
        assert any("违规实体" in r for r in entity_reasons), entity_reasons
        assert any("乔治·洛马克斯先生" in r for r in entity_reasons), entity_reasons

    def test_backward_compat_no_registry(self):
        """不传 entity_registry_full 也能工作（用 sanitized source_name）。"""
        answers = _answers_with_entities()
        verdict = synthesize_verdict(
            answers, lit_text="这是一段足够长的中文译文内容。"
        )
        # 不应抛异常
        assert verdict.decision in ("PASS", "REJECT")
