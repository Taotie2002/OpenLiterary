"""
typesafe-spec 兼容的判断原语 schema（typesafe-ai 模式）

源：https://docs.typesafe.ai/primitives.md

设计目标：
  - 在翻译质量评估中，把"单一 LLM-as-judge"拆成多个 atomic primitive
  - 每个 primitive 返回结构化结果（score/choice/noul）+ 概率 + confidence
  - code 端用确定性逻辑组合 verdict，而不是再次调 LLM
  - 实际后端可路由：Jev（typesafe 原生） / MiniMax（fallback） / Mock

原语三态（与 typesafe.ai 完全对齐）：
  Choice  从枚举中选一项，返回 probabilities + confidence
  Score   按有序 rubric 打分（0..N-1），可落在级别之间
  Noul    0..1 真值分数（"是 / 否 / 多大程度是"）

性能收益（vs 单一 critic LLM）：
  - primitives 可并行（一次 system_one 调用全部完成）
  - verdict 合成纯 code，无 LLM 成本
  - confidence 低的维度自动升级（路由到更强模型 / 触发人工）
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


# ────────────────────────────────────────────────────────────────────────
# 原语结果（与 typesafe_sdk.SystemOneResponse.answers 对齐）
# ────────────────────────────────────────────────────────────────────────

@dataclass
class NoulAnswer:
    """Noul 答案：0..1 真值分数（SDK 无 confidence 字段）

    Round 60：typesafe_sdk 0.7.1 的 NoulAnswer schema 只有 type + noul，没有 confidence。
    Round 83：自动派生 confidence 字段，消除 ans.confidence 与 to_dict()["confidence"]
    的语义不一致 —— 之前 ans.confidence 返回 0.0 (默认), 但 to_dict() 已按
    typesafe-ai 一致性原则计算 max(noul, 1-noul). 现在 __post_init__ 会把
    显式传 confidence==0.0 (sentinel) 的实例自动用 max(noul, 1-noul) 填充,
    让 ans.confidence 与 to_dict() 返回一致.

    示例:
      NoulAnswer(noul=0.39).confidence   == 0.61   (eff_conf, 自动派生)
      NoulAnswer(noul=0.5).confidence    == 0.5    (eff_conf, 自动派生)
      NoulAnswer(noul=0.7, confidence=0.3).confidence == 0.3   (显式传, 不覆盖)
    """
    noul: float
    confidence: float = 0.0

    def __post_init__(self):
        # Round 83: confidence == 0.0 (sentinel for "未设置") 时自动从 noul 派生
        # 显式传 confidence > 0 时不会被覆盖 (向后兼容)
        #
        # Round 91: 防御性 sanity 校验 — 若显式传的 confidence 越界 (<0 或 >1),
        # 视为无效值, 强制派生. 防止 LLM 返回 confidence=1.5 / -0.5 等异常
        # 直接污染 primitive_summary, 下游 verdict 会用 max(noul, 1-noul) 聚合
        # 但 critic_report.json 里出现 conf=1.5 会让 cost_report / QA 报告误读.
        # 派生策略: 越界 → 与 0.0 sentinel 同样处理 (用 eff_conf).
        if self.confidence == 0.0 or not (0.0 <= self.confidence <= 1.0):
            self.confidence = max(self.noul, 1.0 - self.noul)

    def to_dict(self) -> Dict[str, Any]:
        # Round 83: 直接用 self.confidence (已自动派生), 避免重复计算
        return {"type": "noul", "noul": self.noul, "confidence": self.confidence}


@dataclass
class ChoiceAnswer:
    """Choice 答案：选中项 + 各选项概率 + confidence"""
    choice: str
    probabilities: Dict[str, float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


@dataclass
class ScoreAnswer:
    """Score 答案：rubric 位置 + 各位置概率 + confidence"""
    score: float
    probabilities: Dict[int, float]  # level_idx -> probability
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "probabilities": {str(k): v for k, v in self.probabilities.items()},
            "confidence": self.confidence,
        }


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass
class CompositeVerdict:
    """Code 合成的最终裁决（基于多个 primitive 答案）

    Round 62 新增字段：
      - skipped_dimensions: 被合成器跳过的 dimension 列表
        （Noul 在 uncertain zone / Choice 低 confidence / Score 低 confidence）。
        下游 consumer 可以据此区分"明确 PASS"（assert 0 violation）与
        "不确定跳过"（Jev 给出边界值，未达到 reject 门槛）。
      - skipped_reasons: 跳过原因明细（dict：dimension → reason 字符串）

    Round 118 P1-1 新增字段：
      - confidence_source: 概率来源（"jev"=System One 校准概率 /
        "openai_compat"=LLM 自报伪概率 / "mock_heuristic"=确定性启发式 /
        "_uncalibrated"=未知来源按未校准处理）。
      - confidence_calibrated: False 表示阈值已按未校准来源收紧，
        PASS 未经 Jev 认证（降级模式审计标记）。
    """
    decision: str  # "PASS" | "REJECT"
    confidence: float  # 综合置信度 0..1
    reject_reasons: List[str] = field(default_factory=list)
    primitive_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skipped_dimensions: List[str] = field(default_factory=list)
    skipped_reasons: Dict[str, str] = field(default_factory=dict)
    confidence_source: str = "jev"
    confidence_calibrated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
