"""Round 108: 实体违规可执行化（ch07 chunk007/016 根因）。

chunk007: 规范名「敲诈者（临时访客）」带括注，译文永远不会含括注 → Jev 整串检查
  必然判违规 (noul=0.06) → rewriter 无法修复（括注不可能写进正文）→ 振荡 fallback。
  修复: critic 侧规范名去括注（与本地 _canonical_present_in_text 同语义）
  + DB 规范名清理为「敲诈者」。
chunk016: flagged_entity_labels 多词 source_name 的规范名映射永远失效
  （key_to_src 的 key 带 entity_per_ 前缀，查表侧用去前缀 base_key）
  → 拒因只剩不可执行的 "Lady_Eileen_Brent(noul=0.11)"
  → rewriter 不知道规范主名是「艾琳·布伦特夫人」→ 字节级停滞。
  修复: 构造 key_to_src 时剥掉 entity_per_ 前缀。
chunk016 (robustness): 本地核查的后缀容差（「特夫人」）抑制 missing，已登记变体表
  又为空 → 一字之差译名漂移（伊琳 vs 艾琳）本地零报告，只有 Jev 能发现。
  修复: _find_near_variant 一字之差近变体检测 → 报 variant_instead_of_canonical（可执行）。
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.judges.primitives import PER_ENTITY_KEY_PREFIX
from src.judges.typesafe_spec import ChoiceAnswer, NoulAnswer, ScoreAnswer
from src.judges.verdict import flagged_entity_labels, synthesize_verdict


# ---------------------------------------------------------------------------
# Fix 3: flagged_entity_labels 多词 source_name 规范名映射
# ---------------------------------------------------------------------------

REG_MULTIWORD = {
    "Lady Eileen Brent": {"canonical": "艾琳·布伦特夫人", "variants": []},
    "the blackmailer": {"canonical": "敲诈者", "variants": []},
}


def _answers_multiword():
    return {
        "entity_consistency": NoulAnswer(noul=0.89, confidence=0.79),
        PER_ENTITY_KEY_PREFIX + "Lady_Eileen_Brent": NoulAnswer(noul=0.11, confidence=0.89),
        PER_ENTITY_KEY_PREFIX + "the_blackmailer": NoulAnswer(noul=0.95, confidence=0.95),
        "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.05, "severe": 0.05}, 0.8),
        "register_match": ChoiceAnswer("match", {"under": 0.05, "match": 0.9, "over": 0.05}, 0.8),
        "literal_residue": NoulAnswer(noul=0.2, confidence=0.8),
        "semantic_loss": NoulAnswer(noul=0.2, confidence=0.8),
        "style_compliance_score": ScoreAnswer(score=8.5, probabilities={8: 0.9}, confidence=0.8),
    }


class TestFlaggedEntityLabelsMultiWord:
    def test_multiword_source_name_maps_canonical(self):
        """ch07 chunk016 根因：多词 source_name 必须映射到规范主名。"""
        labels = flagged_entity_labels(_answers_multiword(), 0.3, REG_MULTIWORD)
        assert any("Lady Eileen Brent→艾琳·布伦特夫人" in l for l in labels), labels

    def test_compliant_entity_not_flagged(self):
        labels = flagged_entity_labels(_answers_multiword(), 0.3, REG_MULTIWORD)
        assert not any("blackmailer" in l or "敲诈者" in l for l in labels), labels

    def test_reject_reason_contains_canonical(self):
        verdict = synthesize_verdict(
            _answers_multiword(),
            lit_text="她要求与伊琳·布伦特夫人通话。这是一段足够长的中文译文。",
            entity_registry_full=REG_MULTIWORD,
        )
        assert verdict.decision == "REJECT", verdict.reject_reasons
        entity_reasons = [r for r in verdict.reject_reasons if "实体一致性" in r]
        assert any("艾琳·布伦特夫人" in r for r in entity_reasons), entity_reasons


class TestMonolithFlaggedEntityLabels:
    def test_monolith_multiword_maps_canonical(self):
        from src.translator_agent import flagged_entity_labels as mono_labels
        labels = mono_labels(_answers_multiword(), 0.3, REG_MULTIWORD)
        assert any("Lady Eileen Brent→艾琳·布伦特夫人" in l for l in labels), labels


# ---------------------------------------------------------------------------
# Fix 2: critic 侧规范名去括注
# ---------------------------------------------------------------------------

def _instr_text(instr) -> str:
    """per-entity question 的 instructions 可能是 dict（context+question）或 str。"""
    if isinstance(instr, dict):
        return " ".join(str(v) for v in instr.values())
    return str(instr)


def _compliant_answers(questions):
    """为所有 question 生成合规答案（fake backend 用）。"""
    answers = {}
    for qid, q in questions.items():
        qtype = q.get("type")
        if qid.startswith(PER_ENTITY_KEY_PREFIX):
            answers[qid] = NoulAnswer(noul=0.95)
        elif qtype == "noul":
            answers[qid] = NoulAnswer(noul=0.2)
        elif qtype == "choice":
            options = list(q.get("criteria", {}).keys()) or ["none"]
            probs = {o: (0.9 if i == 0 else 0.05) for i, o in enumerate(options)}
            answers[qid] = ChoiceAnswer(options[0], probs, 0.8)
        elif qtype == "score":
            answers[qid] = ScoreAnswer(score=8.5, probabilities={8: 0.9}, confidence=0.8)
    return answers


class TestCriticRegistryParentheticalStripped:
    def test_monolith_evaluate_chunk_strips_parenthetical(self, monkeypatch):
        # 测试环境下 ta.evaluate_chunk 实际是散装 src.judges.judge.evaluate_chunk
        # （单体第 10254 行 from src.judges.judge import evaluate_chunk 覆盖），
        # 故 patch 散装模块的 get_backend。
        import src.judges.judge as judge_mod
        import src.translator_agent as ta
        captured = {}

        class _FakeBackend:
            timeout = 30.0

            def evaluate(self, state, questions):
                captured["questions"] = questions
                return _compliant_answers(questions)

        monkeypatch.setattr(judge_mod, "get_backend", lambda: _FakeBackend())
        ta.evaluate_chunk(
            source_text="Did you fall over the blackmailer in the hall?",
            raw_translation="你在门厅里撞倒那个敲诈者了吗？",
            literary_translation="你在门厅里撞倒那个敲诈者了吗？这是一段足够长的中文译文。",
            style_guide_stats={},
            entity_registry={"the blackmailer": "敲诈者（临时访客）"},
            entity_variants={"the blackmailer": []},
        )
        per_q = {k: v for k, v in captured["questions"].items() if k.startswith(PER_ENTITY_KEY_PREFIX)}
        assert per_q, "per-entity question not built"
        instr = _instr_text(per_q[list(per_q)[0]]["instructions"])
        assert "敲诈者" in instr
        assert "临时访客" not in instr

    def test_loose_critic_agent_strips_parenthetical(self, monkeypatch):
        import src.judges as judges_pkg
        from src.judges.typesafe_backend import FallbackBackend
        captured = {}

        class _FakeJev:
            api_key = "fake"

            def evaluate(self, state, questions, timeout=None):
                captured["questions"] = questions
                return _compliant_answers(questions)

        fb = FallbackBackend(
            primary_model="jev-latest", allow_mock=False,
            allow_openai_fallback=False, enabled=True, timeout=30.0,
        )
        fb.primary = _FakeJev()
        monkeypatch.setattr(judges_pkg, "get_backend", lambda: fb)
        from src.agents.critic_agent import CriticAgent
        agent = CriticAgent()
        agent.process_chunk(
            "test_chunk",
            "Did you fall over the blackmailer in the hall?",
            "你在门厅里撞倒那个敲诈者了吗？",
            "你在门厅里撞倒那个敲诈者了吗？这是一段足够长的中文译文。",
            {},
            entity_registry={"the blackmailer": "敲诈者（临时访客）"},
            entity_variants={"the blackmailer": []},
        )
        per_q = {k: v for k, v in captured["questions"].items() if k.startswith(PER_ENTITY_KEY_PREFIX)}
        assert per_q, "per-entity question not built"
        instr = _instr_text(per_q[list(per_q)[0]]["instructions"])
        assert "敲诈者" in instr
        assert "临时访客" not in instr


# ---------------------------------------------------------------------------
# Fix 4: _find_near_variant 一字之差近变体检测
# ---------------------------------------------------------------------------

def _make_pipeline(conn, module_name):
    mod = __import__(module_name, fromlist=["TranslationPipeline"])
    TP = mod.TranslationPipeline
    p = TP.__new__(TP)

    class _DE:
        def __init__(self):
            self.conn = conn

        def is_canonical_name(self, name):
            return False

        def add_entity_variant(self, canonical, variant):
            pass

    p.decision_engine = _DE()
    from src.core.decision_engine import get_sibling_canonicals_by_source
    p._sibling_canonicals_cache = get_sibling_canonicals_by_source(conn)
    return p


@pytest.fixture(params=["src.pipeline", "src.translator_agent"])
def pipeline_factory(request):
    return lambda conn: _make_pipeline(conn, request.param)


def _new_registry_conn(rows):
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            confidence TEXT DEFAULT 'high',
            pending_review INTEGER DEFAULT 0
        );
    """)
    for canon, srcs in rows:
        conn.execute(
            "INSERT INTO entity_registry (canonical_name, source_names) VALUES (?, ?)",
            (canon, json.dumps(srcs)),
        )
    conn.commit()
    return conn


class TestFindNearVariant:
    def test_separator_name_one_char_diff(self, pipeline_factory):
        """ch07 chunk016: 艾琳·布伦特夫人 vs 伊琳·布伦特夫人。"""
        conn = _new_registry_conn([])
        p = pipeline_factory(conn)
        lit = "她拿到了号码，要求与伊琳·布伦特夫人（“束束”）通话。"
        assert p._find_near_variant("艾琳·布伦特夫人", lit) == "伊琳·布伦特夫人"
        conn.close()

    def test_pure_cjk_not_detected(self, pipeline_factory):
        """纯 CJK 规范名不做近变体检测（ch07 chunk007 实测：「敲诈者」与动词短语
        「敲诈谁/敲诈勒」一字之差碰撞，CJK 无空格无法可靠分词，误报会改坏正文）。"""
        conn = _new_registry_conn([])
        p = pipeline_factory(conn)
        assert p._find_near_variant("烟囱宅", "烟囱庄的灯还亮着。") is None
        assert p._find_near_variant("敲诈者", "但他来这里敲诈谁？") is None
        assert p._find_near_variant("敲诈者", "一个干敲诈勒索的人") is None
        conn.close()

    def test_no_variant_when_only_canonical(self, pipeline_factory):
        conn = _new_registry_conn([])
        p = pipeline_factory(conn)
        lit = "艾琳·布伦特夫人走了。"
        assert p._find_near_variant("艾琳·布伦特夫人", lit) is None
        conn.close()

    def test_short_canonical_skipped(self, pipeline_factory):
        conn = _new_registry_conn([])
        p = pipeline_factory(conn)
        lit = "伊琳走了。"
        assert p._find_near_variant("艾琳", lit) is None
        conn.close()

    def test_parenthetical_stripped_first(self, pipeline_factory):
        conn = _new_registry_conn([])
        p = pipeline_factory(conn)
        lit = "那个敲诈者又来了。"
        assert p._find_near_variant("敲诈者（临时访客）", lit) is None
        conn.close()

    def test_common_word_guard(self, pipeline_factory):
        """一字之差候选是普通词（COMMON_WORDS 黑名单）→ 不报。"""
        conn = _new_registry_conn([])
        p = pipeline_factory(conn)
        # 段「警察」的一字之差候选「警方」在 COMMON_WORDS 黑名单 → 过滤
        assert p._find_near_variant("艾琳·警察", "她叫艾琳·警方。") is None
        conn.close()


class TestCheckEntityConsistencyNearVariant:
    def test_one_char_diff_reported_actionable(self, pipeline_factory):
        """ch07 chunk016 端到端：伊琳·布伦特夫人 → variant_instead_of_canonical。"""
        conn = _new_registry_conn([("艾琳·布伦特夫人", ["Lady Eileen Brent"])])
        p = pipeline_factory(conn)
        source_text = "She asked to speak to Lady Eileen Brent."
        lit_text = "她拿到了号码，要求与伊琳·布伦特夫人通话。"
        notes = p._check_entity_consistency("ch07_chunk016", lit_text, source_text)
        assert notes, "near-variant should be reported"
        n = notes[0]
        assert n["type"] == "variant_instead_of_canonical"
        assert n["found"] == "伊琳·布伦特夫人"
        assert n["expected"] == "艾琳·布伦特夫人"
        conn.close()

    def test_canonical_usage_no_false_positive(self, pipeline_factory):
        """规范名正常使用 → 零报告（不误伤）。"""
        conn = _new_registry_conn([("艾琳·布伦特夫人", ["Lady Eileen Brent"])])
        p = pipeline_factory(conn)
        source_text = "She asked to speak to Lady Eileen Brent."
        lit_text = "她拿到了号码，要求与艾琳·布伦特夫人通话。"
        notes = p._check_entity_consistency("ch07_chunk016", lit_text, source_text)
        assert notes == [], notes
        conn.close()

    def test_missing_still_reported_when_no_near_variant(self, pipeline_factory):
        """无近变体且规范名缺席 → 仍报 missing（原行为不回归）。"""
        conn = _new_registry_conn([("乔治·洛马克斯先生", ["Mr. Lomax"])])
        p = pipeline_factory(conn)
        source_text = "Mr. Lomax arrived."
        lit_text = "一位绅士抵达了庄园。"
        notes = p._check_entity_consistency("ch07_chunk001", lit_text, source_text)
        assert any(n["type"] == "missing" for n in notes), notes
        conn.close()
