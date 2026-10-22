"""
SagaTrans 类型化翻译质量裁判（typesafe-ai 重构）

源：typesafe.ai System One 编程模型
重构：原 critic + judge 双 LLM 调用 → 多 atomic primitives 并行评估 + code 端 verdict 合成

模块：
  typesafe_spec      typesafe 兼容的答案 schema（Noul/Choice/Score/CompositeVerdict）
  primitives         6 个原子判断原语（voice_drift/register_match/entity_consistency/...）
  typesafe_backend   Jev 优先 + MiniMax/DeepSeek fallback 后端
  verdict            基于概率与阈值的 code-side 合成器
  judge              对外统一入口（evaluate_chunk / to_legacy_scores）

公开 API：
  evaluate_chunk(src, raw, lit, style, entity_reg) → CompositeVerdict
  to_legacy_scores(verdict) → Dict[str, float]   # 旧 critic 兼容
"""
from src.judges.typesafe_spec import (
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    CompositeVerdict,
)
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
from src.judges.typesafe_backend import (
    SystemOneBackend,
    JevBackend,
    OpenAICompatBackend,
    FallbackBackend,
    get_backend,
)
from src.judges.verdict import synthesize_verdict, DEFAULT_THRESHOLDS
from src.judges.judge import evaluate_chunk, to_legacy_scores, build_state

__all__ = [
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "CompositeVerdict",
    "VOICE_DRIFT",
    "REGISTER_MATCH",
    "ENTITY_CONSISTENCY",
    "LITERAL_RESIDUE",
    "SEMANTIC_LOSS",
    "STYLE_COMPLIANCE_SCORE",
    "ALL_PRIMITIVES",
    "build_question_set",
    "SystemOneBackend",
    "JevBackend",
    "OpenAICompatBackend",
    "FallbackBackend",
    "get_backend",
    "synthesize_verdict",
    "DEFAULT_THRESHOLDS",
    "evaluate_chunk",
    "to_legacy_scores",
    "build_state",
]
