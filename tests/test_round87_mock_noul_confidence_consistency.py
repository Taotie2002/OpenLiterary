"""Round 87: MockBackend NoulAnswer.confidence 一致性修复.

背景:
  Round 83 引入 NoulAnswer.__post_init__ 自动派生 confidence:
    NoulAnswer(noul=0.39).confidence == 0.61 (max(0.39, 1-0.39))
  Jev 路径 (JevBackend.evaluate) 调用 NoulAnswer(noul=ans.noul) 不传 confidence,
  __post_init__ 自动派生, 与 typesafe-ai SDK 行为一致.

  但 MockBackend (typesafe_backend.py:417, 428, 432, 469) 仍硬塞:
    NoulAnswer(noul=0.71, confidence=0.4)   ← 硬编码 0.4
    NoulAnswer(noul=0.71, confidence=0.3)   ← 硬编码 0.3
    NoulAnswer(noul=0.0, confidence=0.3)    ← 硬编码 0.3
  这导致 Mock 路径的 confidence 与 noul 脱节:
    - noul=0.71 (强 yes) → confidence=0.4 (实际应该是 0.71)
    - noul=0.0  (强 no)  → confidence=0.3 (实际应该是 1.0)
  下游 verdict 合成用 eff_conf = max(noul, 1-noul) 不受硬编码影响, 但:
    - primitive_summary.to_dict()["confidence"] 字段错误写入 critic_report.json
    - 任何下游 consumer 直接读 ans.confidence 会被误导
    - Mock 路径与 Jev 路径行为分裂, 违反 Round 66 的"透明 confidence"目标

修复:
  - 移除 MockBackend 中所有 NoulAnswer 的硬塞 confidence=0.3/0.4
  - 让 __post_init__ 自动从 noul 派生 (与 Jev 路径一致)
  - per-entity atomic Noul 同样修复
  - 显式 confidence > 0 仍可保留 (向后兼容, MockBackend 不主动用此模式)

验证:
  - NoulAnswer(noul=0.71) 不传 confidence 时, __post_init__ 派生 0.71
  - MockBackend 返回的 Noul 答案 confidence 必须等于 max(noul, 1-noul)
  - to_dict()["confidence"] 字段对 Mock/Jev 路径完全一致
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_modules():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        # Round 103 P6+: 动态 polluted 自动覆盖 src.* 父包 + 所有子模块,
        # 父包循环已是冗余, 不再单独遍历 parents.
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)



def _make_state():
    return {
        "source_text": "Mr. Lomax tipped his hat politely to the gentleman.",
        "raw_translation": "洛马克斯先生礼貌地向那位先生脱帽.",
        "literary_translation": "洛马克斯先生礼貌地向那位绅士脱帽.",
        "style_guide": {"avg_sentence_length": "适中"},
        "entity_registry": {"MrLomax": "洛马克斯先生", "gentleman": "绅士"},
        "entity_variants": {},
    }


def test_noul_post_init_autoderive_basic():
    """__post_init__ 自动派生: noul=0.71 → confidence=0.71."""
    from src.judges.typesafe_spec import NoulAnswer
    a = NoulAnswer(noul=0.71)
    assert a.confidence == pytest.approx(0.71)
    assert a.to_dict()["confidence"] == pytest.approx(0.71)


def test_mock_entity_consistency_noul_uses_eff_conf():
    """MockBackend.entity_consistency 不再硬塞 confidence=0.4.

    修复前: NoulAnswer(noul=1.0, confidence=0.4)  → conf=0.4
    修复后: NoulAnswer(noul=1.0)                  → conf=1.0 (eff_conf)
    """
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set

    state = _make_state()
    questions = build_question_set()
    answers = MockBackend().evaluate(state, questions)

    ans = answers["entity_consistency"]
    eff_conf = max(ans.noul, 1.0 - ans.noul)
    assert ans.confidence == pytest.approx(eff_conf), (
        f"Mock NoulAnswer.confidence must equal max(noul, 1-noul); "
        f"got conf={ans.confidence}, noul={ans.noul}, eff_conf={eff_conf}"
    )


def test_mock_literal_residue_noul_uses_eff_conf():
    """MockBackend.literal_residue 不再硬塞 confidence=0.3."""
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set

    state = _make_state()
    questions = build_question_set()
    answers = MockBackend().evaluate(state, questions)

    ans = answers["literal_residue"]
    eff_conf = max(ans.noul, 1.0 - ans.noul)
    assert ans.confidence == pytest.approx(eff_conf)


def test_mock_semantic_loss_noul_uses_eff_conf():
    """MockBackend.semantic_loss 不再硬塞 confidence=0.3."""
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set

    state = _make_state()
    questions = build_question_set()
    answers = MockBackend().evaluate(state, questions)

    ans = answers["semantic_loss"]
    eff_conf = max(ans.noul, 1.0 - ans.noul)
    assert ans.confidence == pytest.approx(eff_conf)


def test_mock_per_entity_noul_uses_eff_conf():
    """per-entity atomic Noul 不再硬塞 confidence=0.3."""
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set, build_per_entity_questions

    state = _make_state()
    questions = build_question_set()
    per_entity_q = build_per_entity_questions({
        "MrLomax": {"canonical": "洛马克斯先生", "variants": []},
        "gentleman": {"canonical": "绅士", "variants": []},
    })
    questions.update(per_entity_q)

    answers = MockBackend().evaluate(state, questions)
    for qid, ans in answers.items():
        if not qid.startswith("entity_per_"):
            continue
        eff_conf = max(ans.noul, 1.0 - ans.noul)
        assert ans.confidence == pytest.approx(eff_conf), (
            f"{qid}: conf={ans.confidence} != eff_conf={eff_conf} (noul={ans.noul})"
        )


def test_mock_noul_to_dict_matches_jev_semantics():
    """Mock NoulAnswer.to_dict() 必须与 Jev 路径返回的 to_dict() 等价.

    这是 Round 87 修复的核心契约: 不论 backend 来源, NoulAnswer 序列化结果
    必须按 typesafe-ai SDK 一致语义 (Round 66 透明 confidence 目标的扩展).
    """
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set
    from src.judges.typesafe_spec import NoulAnswer

    state = _make_state()
    questions = build_question_set()
    mock_answers = MockBackend().evaluate(state, questions)

    # 模拟 Jev 路径的最小化返回: 只传 noul, 让 __post_init__ 派生
    jev_answers = {
        "entity_consistency": NoulAnswer(noul=mock_answers["entity_consistency"].noul),
        "literal_residue": NoulAnswer(noul=mock_answers["literal_residue"].noul),
        "semantic_loss": NoulAnswer(noul=mock_answers["semantic_loss"].noul),
    }

    for qid in ("entity_consistency", "literal_residue", "semantic_loss"):
        mock_d = mock_answers[qid].to_dict()
        jev_d = jev_answers[qid].to_dict()
        # noul 必须一致
        assert mock_d["noul"] == pytest.approx(jev_d["noul"])
        # confidence 必须一致 (这是 Round 87 修复的核心)
        assert mock_d["confidence"] == pytest.approx(jev_d["confidence"]), (
            f"{qid}: mock conf={mock_d['confidence']} != jev conf={jev_d['confidence']}"
        )


def test_mock_noul_unknown_entity_low_noul_high_conf():
    """per-entity Noul: 未知 entity → noul=0 (no violation) → 派生 confidence=1.0.

    修复前: NoulAnswer(noul=0.0, confidence=0.3)  → conf=0.3 (错误: 表示不确定)
    修复后: NoulAnswer(noul=0.0)                  → conf=1.0 (eff_conf, 表示强 no)
    """
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set, build_per_entity_questions

    state = _make_state()
    questions = build_question_set()
    per_entity_q = build_per_entity_questions({
        "UnknownEntity": {"canonical": "未知实体", "variants": []},
    })
    questions.update(per_entity_q)

    answers = MockBackend().evaluate(state, questions)
    a = answers["entity_per_UnknownEntity"]
    # Round 95 修正: 未知 entity → 视为合规 (noul=1.0), 不再返回 noul=0.0.
    # 原 "0.0 表示 no violation" 的注释是错语义; 现已对齐 Jev 方向 (合规=高 noul).
    assert a.noul == 1.0
    assert a.confidence == pytest.approx(1.0)  # 强 yes = 强 confidence


def test_mock_noul_present_entity_zero_noul_high_conf():
    """per-entity Noul: 实体在润色稿中 → noul=0 → 派生 confidence=1.0."""
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set, build_per_entity_questions

    state = _make_state()  # "洛马克斯先生" 在 literary_translation 中
    questions = build_question_set()
    per_entity_q = build_per_entity_questions({
        "MrLomax": {"canonical": "洛马克斯先生", "variants": []},
    })
    questions.update(per_entity_q)

    answers = MockBackend().evaluate(state, questions)
    a = answers["entity_per_MrLomax"]
    # Round 95: 合规 → noul=1.0
    assert a.noul == 1.0
    assert a.confidence == pytest.approx(1.0)


def test_mock_noul_missing_entity_high_noul_high_conf():
    """per-entity Noul: 实体不在润色稿中 → noul=0.6 → 派生 confidence=0.6."""
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set, build_per_entity_questions

    state = _make_state()
    # 把 Mr. Lomax 的 canonical 改成润色稿中不存在的形式
    state["entity_registry"] = {"MrLomax": "洛马克斯老爷", "gentleman": "绅士"}
    questions = build_question_set()
    per_entity_q = build_per_entity_questions({
        "MrLomax": {"canonical": "洛马克斯老爷", "variants": []},
    })
    questions.update(per_entity_q)

    answers = MockBackend().evaluate(state, questions)
    a = answers["entity_per_MrLomax"]
    # Round 95: canonical 不在 lit → 违规 → noul=0.0
    assert a.noul == pytest.approx(0.0)
    assert a.confidence == pytest.approx(1.0)  # 强 no = 强 confidence


def test_mock_noul_high_noul_confidence_is_not_low_confidence():
    """回归测试: Mock Noul 答案 confidence 必须 >= 0.5 当 noul 远离 0.5 时.

    修复前: noul=0.71, confidence=0.4 → < 0.5 → 被任何 confidence gate 视为不确定
    修复后: noul=0.71, confidence=0.71 → >= 0.5 → 视为强 yes
    """
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set, build_per_entity_questions

    state = _make_state()
    state["entity_registry"] = {"MrLomax": "不存在的译名", "gentleman": "绅士"}
    questions = build_question_set()
    per_entity_q = build_per_entity_questions({
        "MrLomax": {"canonical": "不存在的译名", "variants": []},
    })
    questions.update(per_entity_q)

    answers = MockBackend().evaluate(state, questions)
    a = answers["entity_per_MrLomax"]
    eff_conf = max(a.noul, 1.0 - a.noul)
    # 修复后的硬性约束
    assert a.confidence == eff_conf
    if eff_conf >= 0.5:
        assert a.confidence >= 0.5, (
            f"eff_conf={eff_conf} >= 0.5 but Mock answer reports {a.confidence} < 0.5"
        )


def test_mock_noul_confidence_in_primitive_summary():
    """verdict 合成后 primitive_summary.confidence 字段必须等于 eff_conf.

    这是 critic_report.json 里 _typesafe_verdict.primitive_summary 的内容,
    修复前会显示 confidence=0.3/0.4 (与 noul 脱节).
    """
    from src.judges.typesafe_backend import MockBackend
    from src.judges.primitives import build_question_set
    from src.judges.verdict import synthesize_verdict

    state = _make_state()
    questions = build_question_set()
    answers = MockBackend().evaluate(state, questions)
    verdict = synthesize_verdict(answers)

    for qid in ("entity_consistency", "literal_residue", "semantic_loss"):
        s = verdict.primitive_summary[qid]
        noul = s["noul"]
        eff_conf = max(noul, 1.0 - noul)
        assert s["confidence"] == pytest.approx(eff_conf), (
            f"{qid}: primitive_summary.confidence={s['confidence']} != "
            f"max(noul={noul}, 1-noul)={eff_conf}"
        )
