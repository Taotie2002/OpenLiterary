"""Round 91: OpenAICompatBackend Noul 路径 confidence 与其他 Noul 创建点一致.

背景:
  Round 83 在 NoulAnswer.__post_init__ 引入 `confidence == 0.0` sentinel 派生:
    NoulAnswer(noul=0.39).confidence → 0.61 (eff_conf)
    NoulAnswer(noul=0.39, confidence=0.3).confidence → 0.3 (显式保留)

  Round 87 修复了 MockBackend 的 5 个 Noul 创建点, 让其使用 __post_init__ 派生.
  但 OpenAICompatBackend._parse_answers Noul 分支 (typesafe_backend.py:331)
  仍是唯一硬塞 `confidence=conf` 的地方, 与 Jev / Mock 都不一致:

    OLD: out[qid] = NoulAnswer(noul=noul_val, confidence=conf)
    NEW: out[qid] = NoulAnswer(noul=noul_val)  (Round 91)

  后果:
    - LLM 响应缺 confidence 字段 → conf 默认 0.5 → primitive_summary 显示 0.5
      (应显示 max(noul, 1-noul))
    - LLM 自报 confidence=0.5 → 同样问题 (任意 [0,1] 内的 LLM 自报均不校准)
    - LLM 自报越界 confidence=1.5 / -0.3 → 直接污染 cost_report / critic_report

修复:
  1. OpenAICompat Noul 分支改为 `NoulAnswer(noul=noul_val)` — 与 Jev / Mock 对齐
  2. __post_init__ 增加越界 sanity 校验: conf 不在 [0,1] 时强制派生

保护目标:
  - OpenAICompat _parse_answers Noul 分支: 任何 LLM 自报 confidence (含缺失/越界)
    都派生自 noul, 与 Jev/Mock 输出 eff_conf 字段一致
  - __post_init__ 越界校验: confidence > 1 或 < 0 时不污染字段
  - 显式 confidence=0.0 / 0.3 / 1.0 仍按 Round 83 语义保留 (向后兼容)
"""
from __future__ import annotations

import sys

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



# ---------------------------------------------------------------------------
# (1) NoulAnswer.__post_init__ 越界校验 (Round 91 新增)
# ---------------------------------------------------------------------------

class TestPostInitOutOfRangeValidation:
    """confidence 不在 [0,1] 时, __post_init__ 强制派生 (sanity 校验)."""

    def test_confidence_above_one_derives(self):
        # 1.5 越界 → 派生 (noul=0.4 → eff_conf=0.6)
        ans = __import__("src.judges.typesafe_spec", fromlist=["NoulAnswer"]).NoulAnswer(
            noul=0.4, confidence=1.5
        )
        assert ans.confidence == pytest.approx(0.6, abs=1e-6)

    def test_confidence_negative_derives(self):
        # -0.3 越界 → 派生 (noul=0.4 → eff_conf=0.6)
        from src.judges.typesafe_spec import NoulAnswer
        ans = NoulAnswer(noul=0.4, confidence=-0.3)
        assert ans.confidence == pytest.approx(0.6, abs=1e-6)

    def test_confidence_just_above_one_derives(self):
        """1.0001 (LLM 数值噪声) → 派生, 不让 1.0001 进入 primitive_summary."""
        from src.judges.typesafe_spec import NoulAnswer
        ans = NoulAnswer(noul=0.5, confidence=1.0001)
        assert ans.confidence == pytest.approx(0.5, abs=1e-6)

    def test_confidence_just_below_zero_derives(self):
        from src.judges.typesafe_spec import NoulAnswer
        ans = NoulAnswer(noul=0.5, confidence=-1e-9)
        assert ans.confidence == pytest.approx(0.5, abs=1e-6)

    def test_explicit_confidence_in_range_preserved(self):
        """[0,1] 内的显式 confidence 不被覆盖 (Round 83 向后兼容)."""
        from src.judges.typesafe_spec import NoulAnswer
        # 0.3 (Round 83 兼容路径)
        a = NoulAnswer(noul=0.7, confidence=0.3)
        assert a.confidence == pytest.approx(0.3, abs=1e-6)
        # 0.5 (边界, 应保留)
        b = NoulAnswer(noul=0.5, confidence=0.5)
        assert b.confidence == pytest.approx(0.5, abs=1e-6)
        # 1.0 (边界, 应保留)
        c = NoulAnswer(noul=0.5, confidence=1.0)
        assert c.confidence == pytest.approx(1.0, abs=1e-6)
        # 0.0 (Round 83 sentinel, 派生)
        d = NoulAnswer(noul=0.4, confidence=0.0)
        assert d.confidence == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# (2) OpenAICompatBackend._parse_answers Noul 分支一致性
# ---------------------------------------------------------------------------

class TestOpenAICompatNoulAlignment:
    """OpenAICompat Noul 分支与 Jev / Mock 5 个点一致: 不硬塞 confidence."""

    def _build_backend(self):
        from unittest.mock import MagicMock
        from src.judges.typesafe_backend import OpenAICompatBackend
        backend = OpenAICompatBackend.__new__(OpenAICompatBackend)
        backend.llm = MagicMock()
        backend.model_role = "primary"
        return backend

    def test_missing_confidence_field_derives_from_noul(self):
        """LLM 响应缺 confidence → 派生自 noul (与 Jev 一致)."""
        backend = self._build_backend()
        questions = {"literal_residue": {"type": "noul", "instructions": "is literal?", "criteria": {}}}
        raw = '{"answers": {"literal_residue": {"type": "noul", "noul": 0.6, "reasoning": "looks literal"}}}'
        out = backend._parse_answers(raw, questions)
        ans = out["literal_residue"]
        # noul=0.6 → eff_conf = max(0.6, 0.4) = 0.6
        assert ans.confidence == pytest.approx(0.6, abs=1e-6)
        assert ans.noul == pytest.approx(0.6, abs=1e-6)

    def test_llm_self_reported_confidence_overridden_by_derivation(self):
        """LLM 自报 confidence=0.5 → Round 91 派生覆盖 (Noul 不携带 confidence per typesafe SDK)."""
        backend = self._build_backend()
        questions = {"literal_residue": {"type": "noul", "instructions": "is literal?", "criteria": {}}}
        raw = '{"answers": {"literal_residue": {"type": "noul", "noul": 0.3, "confidence": 0.5}}}'
        out = backend._parse_answers(raw, questions)
        ans = out["literal_residue"]
        # 修复后: 派生覆盖 LLM 自报. noul=0.3 → eff_conf=0.7
        assert ans.confidence == pytest.approx(0.7, abs=1e-6)

    def test_llm_self_reported_confidence_zero_derives(self):
        """LLM 自报 confidence=0 → __post_init__ sentinel 派生."""
        backend = self._build_backend()
        questions = {"literal_residue": {"type": "noul", "instructions": "is literal?", "criteria": {}}}
        raw = '{"answers": {"literal_residue": {"type": "noul", "noul": 0.7, "confidence": 0.0}}}'
        out = backend._parse_answers(raw, questions)
        ans = out["literal_residue"]
        assert ans.confidence == pytest.approx(0.7, abs=1e-6)

    def test_llm_self_reported_out_of_range_confidence_derives(self):
        """LLM 自报越界 confidence=1.5 → Round 91 派生覆盖."""
        backend = self._build_backend()
        questions = {"literal_residue": {"type": "noul", "instructions": "is literal?", "criteria": {}}}
        raw = '{"answers": {"literal_residue": {"type": "noul", "noul": 0.5, "confidence": 1.5}}}'
        out = backend._parse_answers(raw, questions)
        ans = out["literal_residue"]
        # 越界 1.5 → __post_init__ 派生: max(0.5, 0.5) = 0.5
        assert ans.confidence == pytest.approx(0.5, abs=1e-6)

    def test_to_dict_confidence_matches_eff_conf(self):
        """OpenAICompat 生成的 Noul to_dict() 必须含派生后的 confidence (与 Jev 一致)."""
        backend = self._build_backend()
        questions = {"literal_residue": {"type": "noul", "instructions": "is literal?", "criteria": {}}}
        raw = '{"answers": {"literal_residue": {"type": "noul", "noul": 0.4, "confidence": 0.0}}}'
        out = backend._parse_answers(raw, questions)
        ans = out["literal_residue"]
        d = ans.to_dict()
        assert d["confidence"] == pytest.approx(0.6, abs=1e-6)
        assert d["confidence"] == ans.confidence

    def test_openai_compat_choice_path_unaffected(self):
        """修复只动 Noul 分支, Choice 仍用 conf_aligned (Round 67 不破坏)."""
        backend = self._build_backend()
        questions = {
            "voice_drift": {"type": "choice", "instructions": "drift?",
                            "criteria": {"none": "无", "subtle": "轻微", "severe": "严重"}},
        }
        raw = '{"answers": {"voice_drift": {"type": "choice", "choice": "severe", "confidence": 0.9}}}'
        out = backend._parse_answers(raw, questions)
        ans = out["voice_drift"]
        # Choice: confidence = max(probs) (Round 67 派生)
        assert ans.choice == "severe"
        # severe 概率 0.9 → confidence 应该 ≈ 0.9 (因为其他概率都被压缩到 0.1/n)
        assert ans.confidence == pytest.approx(0.9, abs=0.05)


# ---------------------------------------------------------------------------
# (3) 集成: OpenAICompat → verdict → rewriter 路径
# ---------------------------------------------------------------------------

class TestOpenAICompatIntegrationPath:
    """端到端: OpenAICompat Noul 输出进入 verdict 路径, 不污染 downstream."""

    def test_openai_noul_synthesize_verdict_uses_eff_conf(self):
        """verdict 聚合 overall_conf 时用 max(noul, 1-noul), 与 primitive_summary 一致."""
        from src.judges.verdict import synthesize_verdict
        from src.judges.typesafe_spec import NoulAnswer
        # 模拟 OpenAICompat 派生后的 Noul (noul=0.4, confidence=0.6)
        ans = NoulAnswer(noul=0.4)  # __post_init__ 派生
        v = synthesize_verdict({"literal_residue": ans})
        # verdict.overall_conf = max(noul, 1-noul) = 0.6
        # reject 因为 noul=0.4 < uncertain_low=0.3 + noul < literal_residue_noul=0.5 → pass
        # 但 overall_conf 应来自 ans.confidence=0.6 (Round 91 一致性)
        assert v.confidence == pytest.approx(0.6, abs=1e-2)

    def test_openai_noul_primitive_summary_confidence_not_zero(self):
        """primitive_summary 中的 Noul confidence 字段不再为 0.0 (Round 91 修复目标)."""
        from src.judges.verdict import synthesize_verdict
        from src.judges.typesafe_spec import NoulAnswer
        ans = NoulAnswer(noul=0.4)
        v = synthesize_verdict({"literal_residue": ans})
        ps = v.primitive_summary["literal_residue"]
        assert ps["confidence"] != 0.0, (
            "primitive_summary.confidence 必须派生自 noul (Round 91 修复目标, "
            "之前为 0.0 误导 QA / cost_report)"
        )
        assert ps["confidence"] == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# (4) 回归保护: Round 83 已有测试不能破坏
# ---------------------------------------------------------------------------

class TestRound83ContractPreserved:
    """Round 91 不破坏 Round 83 的核心契约."""

    def test_default_derivation_still_works(self):
        from src.judges.typesafe_spec import NoulAnswer
        # Round 83: NoulAnswer(noul=0.39).confidence == 0.61
        ans = NoulAnswer(noul=0.39)
        assert ans.confidence == pytest.approx(0.61, abs=1e-6)

    def test_explicit_zero_derives(self):
        from src.judges.typesafe_spec import NoulAnswer
        # Round 83: NoulAnswer(noul=0.7, confidence=0.0).confidence == 0.7
        ans = NoulAnswer(noul=0.7, confidence=0.0)
        assert ans.confidence == pytest.approx(0.7, abs=1e-6)

    def test_explicit_in_range_preserved(self):
        from src.judges.typesafe_spec import NoulAnswer
        # Round 83: 显式 0.3 不被覆盖
        ans = NoulAnswer(noul=0.5, confidence=0.3)
        assert ans.confidence == pytest.approx(0.3, abs=1e-6)
