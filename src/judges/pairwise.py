"""Round 116: Jev 配对仲裁原型（bench 工具，尚未接线主流水线）。

用途：同一 source/raw 下，把两个候选润色稿（例如 DeepSeek 版 vs MiniMax 版）
分别过同一组 typesafe primitives，用校准概率做**配对比较**，输出每维胜负、
分歧清单与总体建议。

设计原则：
  - 只消费 answers（dataclass 或已序列化 dict 均可），后端调用由调用方注入 → 可 mock 单测
  - 低置信维度跳过（与 verdict.py 的 uncertain zone 语义一致）
  - 每维一票，不做加权玄学；margin 过小计平票
  - 这是"Jev=仲裁核心"的最小验证件：先证明配对比较可跑通，再谈接线
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# 维度 → (取值方式, 是否越低越好)
#   noul          : 直接取 noul（越低越好）
#   choice_severity: voice_drift 概率 → severity = P(subtle)*1 + P(severe)*2（越低越好）
#   choice_mismatch: register_match 概率 → mismatch = P(under)+P(over)（越低越好）
#   score_expected : 有序 rubric 概率 → 期望档位（越高越好）
DIM_POLICY: Dict[str, Tuple[str, bool]] = {
    "entity_consistency": ("noul", True),
    "semantic_loss": ("noul", True),
    "literal_residue": ("noul", True),
    "cjk_compliance": ("noul", True),
    "voice_drift": ("choice_severity", True),
    "register_match": ("choice_mismatch", True),
    "style_compliance_score": ("score_expected", False),
}


def _as_fields(answer: Any) -> Optional[Tuple[str, Any, float]]:
    """归一化 answer → (kind, payload, confidence)；无法解析返回 None。

    支持 dataclass（NoulAnswer/ChoiceAnswer/ScoreAnswer）与 to_dict() 序列化形态。
    """
    if answer is None:
        return None
    if isinstance(answer, dict):
        kind = answer.get("type")
        conf = answer.get("confidence")
        if kind == "noul":
            noul = float(answer.get("noul", 0.0) or 0.0)
            conf = float(conf) if conf is not None else max(noul, 1.0 - noul)
            return ("noul", noul, conf)
        if kind == "choice":
            probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            return ("choice", probs, float(conf or 0.0))
        if kind == "score":
            probs = {int(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
            return ("score", probs, float(conf or 0.0))
        return None
    # dataclass 形态
    kind = getattr(answer, "type", None)
    if kind == "noul":
        return ("noul", float(getattr(answer, "noul", 0.0)), float(getattr(answer, "confidence", 0.0)))
    if kind == "choice":
        probs = {str(k): float(v) for k, v in (getattr(answer, "probabilities", {}) or {}).items()}
        return ("choice", probs, float(getattr(answer, "confidence", 0.0)))
    if kind == "score":
        probs = {int(k): float(v) for k, v in (getattr(answer, "probabilities", {}) or {}).items()}
        return ("score", probs, float(getattr(answer, "confidence", 0.0)))
    # 部分 dataclass 无 type 属性时按字段名兜底
    if hasattr(answer, "noul"):
        return ("noul", float(getattr(answer, "noul")), float(getattr(answer, "confidence", 0.0)))
    if hasattr(answer, "probabilities"):
        probs_raw = getattr(answer, "probabilities", {}) or {}
        conf = float(getattr(answer, "confidence", 0.0))
        if hasattr(answer, "choice"):
            return ("choice", {str(k): float(v) for k, v in probs_raw.items()}, conf)
        return ("score", {int(k): float(v) for k, v in probs_raw.items()}, conf)
    return None


def _dim_value(kind: str, payload: Any, method: str) -> Optional[float]:
    """按 method 把 payload 折算成可比较的数值。"""
    if method == "noul":
        return float(payload)
    if kind != "choice" and method in ("choice_severity", "choice_mismatch"):
        return None
    if kind != "score" and method == "score_expected":
        return None
    if method == "choice_severity":
        return float(payload.get("subtle", 0.0)) * 1.0 + float(payload.get("severe", 0.0)) * 2.0
    if method == "choice_mismatch":
        return float(payload.get("under", 0.0)) + float(payload.get("over", 0.0))
    if method == "score_expected":
        return sum(float(level) * p for level, p in payload.items())
    return None


def arbitrate_answers(
    answers_a: Dict[str, Any],
    answers_b: Dict[str, Any],
    *,
    min_confidence: float = 0.5,
    tie_epsilon: float = 0.05,
    close_margin: float = 0.15,
) -> Dict[str, Any]:
    """配对比较两组 answers → 每维胜负 + 总体建议。

    Args:
        answers_a/answers_b: {dim_name: Answer}（dataclass 或 to_dict() 形态）
        min_confidence: 任一候选该维 confidence 低于此值 → skip（不投票）
        tie_epsilon: |value_a - value_b| ≤ 该值 → 该维计平票
        close_margin: 胜出 margin ≤ 该值 → 记入 close_dimensions（人工关注）
    """
    dims: Dict[str, Any] = {}
    votes = {"A": 0, "B": 0, "tie": 0, "skipped": 0}
    close_dims = []
    skipped_reasons: Dict[str, str] = {}

    for dim, (method, lower_is_better) in DIM_POLICY.items():
        fa = _as_fields(answers_a.get(dim)) if isinstance(answers_a, dict) else None
        fb = _as_fields(answers_b.get(dim)) if isinstance(answers_b, dict) else None
        if fa is None or fb is None:
            votes["skipped"] += 1
            skipped_reasons[dim] = "missing"
            continue
        kind_a, payload_a, conf_a = fa
        kind_b, payload_b, conf_b = fb
        if conf_a < min_confidence or conf_b < min_confidence:
            votes["skipped"] += 1
            skipped_reasons[dim] = f"low_confidence(a={conf_a:.2f}, b={conf_b:.2f})"
            dims[dim] = {"status": "skipped_low_confidence", "conf_a": conf_a, "conf_b": conf_b}
            continue
        value_a = _dim_value(kind_a, payload_a, method)
        value_b = _dim_value(kind_b, payload_b, method)
        if value_a is None or value_b is None:
            votes["skipped"] += 1
            skipped_reasons[dim] = "unsupported_payload"
            continue

        margin = value_a - value_b
        if abs(margin) <= tie_epsilon:
            winner = "tie"
            votes["tie"] += 1
        else:
            a_better = (margin < 0) if lower_is_better else (margin > 0)
            winner = "A" if a_better else "B"
            votes[winner] += 1
            if abs(margin) <= close_margin:
                close_dims.append(dim)
        dims[dim] = {
            "status": "voted",
            "winner": winner,
            "value_a": round(value_a, 4),
            "value_b": round(value_b, 4),
            "margin_a_minus_b": round(margin, 4),
            "conf_a": round(conf_a, 3),
            "conf_b": round(conf_b, 3),
            "lower_is_better": lower_is_better,
        }

    if votes["A"] > votes["B"]:
        winner = "A"
    elif votes["B"] > votes["A"]:
        winner = "B"
    else:
        winner = "tie"

    return {
        "winner": winner,
        "votes": votes,
        "dimensions": dims,
        "close_dimensions": close_dims,
        "skipped_reasons": skipped_reasons,
    }


def evaluate_with_backend(backend: Any, state_a: Any, state_b: Any, questions: Dict[str, Dict[str, Any]]):
    """对两个候选 state 各跑一遍同一组 questions（同一次 bench 用同一 backend）。"""
    answers_a = backend.evaluate(state_a, questions)
    answers_b = backend.evaluate(state_b, questions)
    return answers_a, answers_b


def summarize(result: Dict[str, Any]) -> str:
    """把 arbitrate_answers 结果渲染成可读文本（给 CLI/日志用）。"""
    lines = []
    votes = result.get("votes", {})
    lines.append(
        f"配对仲裁: winner={result.get('winner')} "
        f"(A={votes.get('A', 0)}, B={votes.get('B', 0)}, tie={votes.get('tie', 0)}, "
        f"skipped={votes.get('skipped', 0)})"
    )
    for dim, info in (result.get("dimensions") or {}).items():
        if info.get("status") == "skipped_low_confidence":
            lines.append(f"  - {dim}: skipped (low conf a={info.get('conf_a')}, b={info.get('conf_b')})")
            continue
        lines.append(
            f"  - {dim}: {info.get('winner')} "
            f"(a={info.get('value_a')} vs b={info.get('value_b')}, "
            f"margin={info.get('margin_a_minus_b')})"
        )
    if result.get("close_dimensions"):
        lines.append(f"  ⚠️ close dimensions: {', '.join(result['close_dimensions'])}")
    if result.get("skipped_reasons"):
        lines.append(f"  skipped: {result['skipped_reasons']}")
    return "\n".join(lines)
