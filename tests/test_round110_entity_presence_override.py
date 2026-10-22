"""Round 110: per-entity 确定性护栏 — 规范名已出现时覆盖 LLM 假阳性。

保护目标（ch08 chunk001 根因）：
  Jev 主路径超时 → openai_compat 兜底把 "The manager/the manager" 判为违规
  （noul=0.00），而润色稿含规范名「经理」且确定性 entity audit 确认 present
  → entity_consistency 被 max(1-noul) 聚合打到 0 → judge REJECT → rewriter 收到
  "统一为规范主名"的不可执行指令 → 输出字节停滞 → REWRITER_STAGNATED →
  PERMANENTLY_FAILED。

修复：apply_entity_presence_override 在 synthesize_verdict 之前用确定性子串
比对复核 per-entity 判定 — 命中规范名/变体 → 强制 noul=1.0（合规）；未命中 →
保留原判定，不掩盖真实缺失。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.judges.primitives import (
    apply_entity_presence_override,
    aggregate_per_entity_nouls,
    entity_present_in_text,
)
from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer
from src.judges.verdict import synthesize_verdict


PROJECT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# entity_present_in_text
# ---------------------------------------------------------------------------

class TestEntityPresentInText:
    def test_exact_canonical_hit(self):
        assert entity_present_in_text("经理", [], "他要求见经理。") == "经理"

    def test_variant_hit(self):
        assert entity_present_in_text("卡特拉姆侯爵", ["卡特拉姆勋爵"], "卡特拉姆勋爵说道") == "卡特拉姆勋爵"

    def test_parenthetical_stripped_base_hit(self):
        # 规范名带括注 → 去括注 base 命中（Round 105/108 同语义）
        assert entity_present_in_text("镇（town）", [], "小镇上的人") == "镇"

    def test_long_name_prefix_tolerance(self):
        # 译文用 乔治·洛马克斯，规范名 乔治·洛马克斯先生 → 前缀容差命中
        assert entity_present_in_text("乔治·洛马克斯先生", [], "乔治·洛马克斯来了") is not None

    def test_miss_returns_none(self):
        assert entity_present_in_text("阿尔弗雷德", [], "完全无关的文本") is None

    def test_empty_text_returns_none(self):
        assert entity_present_in_text("经理", [], "") is None


# ---------------------------------------------------------------------------
# apply_entity_presence_override
# ---------------------------------------------------------------------------

def _full_answers(entity_noul: float = 0.0):
    """构造完整 primitive 答案集（其余维度通过，仅 entity 假阳性）。"""
    return {
        "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.9),
        "register_match": ChoiceAnswer("match", {"under": 0.08, "match": 0.85, "over": 0.07}, 0.85),
        "entity_consistency": NoulAnswer(noul=1.0),
        "literal_residue": NoulAnswer(0.0),
        "semantic_loss": NoulAnswer(0.0),
        "style_compliance_score": ScoreAnswer(
            score=4.0, probabilities={0: 0.05, 1: 0.05, 2: 0.1, 3: 0.1, 4: 0.7}, confidence=0.7
        ),
        "entity_per_The_manager": NoulAnswer(noul=entity_noul),
        "entity_per_the_manager": NoulAnswer(noul=entity_noul),
    }


class TestApplyEntityPresenceOverride:
    def test_ch08_false_positive_overridden_to_pass(self):
        """ch08 chunk001 复现：假阳性 REJECT → 护栏后 PASS。"""
        lit = "然后他要求见经理。经理是个法国人。"
        registry = {
            "The manager": {"canonical": "经理", "variants": []},
            "the manager": {"canonical": "经理", "variants": []},
        }
        answers = _full_answers(entity_noul=0.0)
        # 修前：per-entity 聚合 1.0 → REJECT
        before = synthesize_verdict(
            answers, lit_text=lit, raw_text=lit, entity_registry_full=registry
        )
        assert before.decision == "REJECT"
        assert before.primitive_summary["entity_consistency"]["noul"] == 1.0

        overrides = apply_entity_presence_override(answers, registry, lit)
        assert len(overrides) == 2
        assert all(o["matched"] == "经理" for o in overrides)
        assert aggregate_per_entity_nouls(answers) == 0.0

        after = synthesize_verdict(
            answers, lit_text=lit, raw_text=lit, entity_registry_full=registry
        )
        assert after.decision == "PASS"
        assert after.reject_reasons == []
        assert after.primitive_summary["entity_consistency"]["noul"] == 0.0

    def test_genuine_violation_kept(self):
        """真违规（规范名缺失）不被覆盖，仍 REJECT。"""
        lit = "文本里完全没有那个名字。"
        registry = {"Alfred": {"canonical": "阿尔弗雷德", "variants": []}}
        answers = {"entity_per_Alfred": NoulAnswer(noul=0.0)}
        overrides = apply_entity_presence_override(answers, registry, lit)
        assert overrides == []
        assert answers["entity_per_Alfred"].noul == 0.0

    def test_key_collision_replay_matches_build_order(self):
        """净化后 key 冲突（_2 后缀）时回放仍能命中正确 qid。"""
        registry = {
            "Mr. Lomax": {"canonical": "洛马克斯先生", "variants": []},
            "Mr__Lomax": {"canonical": "洛马克斯先生", "variants": []},
        }
        answers = {
            "entity_per_Mr__Lomax": NoulAnswer(noul=0.0),
            "entity_per_Mr__Lomax_2": NoulAnswer(noul=0.0),
        }
        overrides = apply_entity_presence_override(answers, registry, "洛马克斯先生到场。")
        assert {o["qid"] for o in overrides} == {"entity_per_Mr__Lomax", "entity_per_Mr__Lomax_2"}

    def test_missing_answer_and_empty_inputs_noop(self):
        registry = {"Alfred": {"canonical": "阿尔弗雷德", "variants": []}}
        assert apply_entity_presence_override({}, registry, "阿尔弗雷德到场") == []
        assert apply_entity_presence_override({"entity_per_Alfred": NoulAnswer(0.0)}, None, "x") == []
        assert apply_entity_presence_override({"entity_per_Alfred": NoulAnswer(0.0)}, registry, "") == []

    def test_already_compliant_answer_untouched(self):
        registry = {"Alfred": {"canonical": "阿尔弗雷德", "variants": []}}
        answers = {"entity_per_Alfred": NoulAnswer(noul=1.0)}
        assert apply_entity_presence_override(answers, registry, "阿尔弗雷德到场") == []


# ---------------------------------------------------------------------------
# 集成：CriticAgent.process_chunk / judge.evaluate_chunk
# ---------------------------------------------------------------------------

class _StubBackend:
    """返回预设答案的 backend stub（绕过网络）。"""

    timeout = 30.0

    def __init__(self, answers):
        self._answers = answers

    def evaluate(self, state, questions, timeout=None, model=None):
        return dict(self._answers)


class TestCriticIntegration:
    def _run_critic(self):
        from src.agents.critic_agent import CriticAgent

        lit = "然后他要求见经理。经理是个法国人。"
        critic = CriticAgent.__new__(CriticAgent)
        critic.llm = None
        critic.role = "critic_scoring"
        answers = _full_answers(entity_noul=0.0)
        registry = {"The manager": "经理", "the manager": "经理"}
        with patch("src.judges.get_backend", return_value=_StubBackend(answers)):
            report = critic.process_chunk(
                "ch_test_chunk000",
                source_text="He asked to see The manager. The manager was a Frenchman.",
                raw_trans=lit,
                lit_trans=lit,
                style_guide={},
                entity_registry=registry,
                # entity_variants 非空才生成 per-entity questions（Round 61 契约）
                entity_variants={"The manager": [], "the manager": []},
            )
        return report

    def test_critic_report_entity_score_recovered(self):
        report = self._run_critic()
        assert report["scores"]["entity_consistency"] == 10.0
        assert report["_typesafe_verdict"]["decision"] == "PASS"
        summary = report["_typesafe_verdict"]["primitive_summary"]
        assert summary["entity_per_The_manager"]["noul"] == 1.0
        assert summary["entity_per_the_manager"]["noul"] == 1.0

    def test_critic_redeclares_false_positive_reason(self):
        # 修前 critique 会含"实体一致性违反（per-entity 聚合）"，修后必须消失
        report = self._run_critic()
        assert "实体一致性违反" not in report["critique"]


class TestJudgeEvaluateChunkIntegration:
    def test_evaluate_chunk_passes_after_override(self):
        from src.judges import judge as judge_mod

        lit = "然后他要求见经理。经理是个法国人。"
        answers = _full_answers(entity_noul=0.0)
        with patch.object(judge_mod, "get_backend", return_value=_StubBackend(answers)):
            verdict = judge_mod.evaluate_chunk(
                source_text="He asked to see The manager. The manager was a Frenchman.",
                raw_translation=lit,
                literary_translation=lit,
                style_guide_stats={},
                entity_registry={"The manager": "经理", "the manager": "经理"},
                entity_variants={"The manager": [], "the manager": []},
            )
        assert verdict.decision == "PASS"
        assert verdict.primitive_summary["entity_consistency"]["noul"] == 0.0


# ---------------------------------------------------------------------------
# 散装 ↔ 整合快照同步
# ---------------------------------------------------------------------------

class TestMonolithParity:
    def test_monolith_contains_guard_and_shim_exposure(self):
        src_text = (PROJECT / "src" / "translator_agent.py").read_text(encoding="utf-8")
        assert "def entity_present_in_text(" in src_text
        assert "def apply_entity_presence_override(" in src_text
        # shim 必须暴露新符号，否则 monolith 内 `from src.judges.primitives import ...` 会 ImportError
        assert '"apply_entity_presence_override"' in src_text
