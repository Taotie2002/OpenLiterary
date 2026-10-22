"""
类型化翻译质量裁判（typesafe-ai 重构核心）

对外接口：
    evaluate_chunk(...) → CompositeVerdict

工作流（typesafe.ai System One 模式）：
    1. 构造 state（原文 + 直译 + 润色稿 + 实体登记表 + 风格基线）
    2. 用 primitives.build_question_set() 构造 questions 字典（6 个 atomic primitives）
    3. 【Round 61】如提供 entity_variants，附加 per-entity atomic Nouls
    4. 调 get_backend().evaluate(state, questions) → 所有 primitive 答案并行
    5. 用 verdict.synthesize_verdict(answers) 合成 PASS/REJECT（纯 code）
    6. 返回 CompositeVerdict

与旧 critic/judge 对比：
    - 旧：单一 critic LLM 一次性评 6 维度整数分 + judge LLM 总结
    - 新：6 个 atomic primitives 并行评估 + code-side verdict 合成
    - 优势：调试粒度细、概率可见、可针对弱项升级模型、可路由到 typesafe Jev
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from src.judges.typesafe_spec import CompositeVerdict
from src.judges.primitives import (
    build_question_set,
    build_per_entity_questions,
    apply_entity_presence_override,
)
from src.judges.typesafe_backend import get_backend
from src.judges.verdict import synthesize_verdict
from src.utils.text_match import entity_source_name_in_text


def backend_confidence_source(backend) -> str:
    """Round 118 P1-1: 取 backend 的概率来源标识（供 verdict 分档）。

    FallbackBackend 在 evaluate() 成功后写 last_backend；非 fallback 的
    单后端用类属性 backend_kind（JevBackend/OpenAICompatBackend/MockBackend）。
    """
    last = getattr(backend, "last_backend", None)
    if last:
        return str(last)
    return str(getattr(backend, "backend_kind", type(backend).__name__))


def build_state(
    source_text: str,
    raw_translation: str,
    literary_translation: str,
    style_guide_stats: Dict[str, Any],
    entity_registry: Optional[Dict[str, Any]] = None,
    entity_variants: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """构造 typesafe System One 的 state（与 typesafe State 规范对齐）

    Args:
        source_text: 原文
        raw_translation: 直译底稿
        literary_translation: 润色稿
        style_guide_stats: 风格基线统计
        entity_registry: 已登记的实体译名（key=source_name, value=canonical_zh）
        entity_variants: 【Round 61】已登记实体的中文变体
            （key=source_name, value=List[variant_zh]）

    Returns:
        state dict（typesafe SDK 接受 str/dict/list）
    """
    # A4 修复：按 source_text 过滤 entity_registry（与 pipeline._check_entity_consistency 同语义）
    relevant_entities: Dict[str, str] = {}
    if entity_registry:
        for source_name, canonical in entity_registry.items():
            if not source_name:
                continue
            if entity_source_name_in_text(source_name, source_text):
                relevant_entities[source_name] = canonical

    # Round 61: 同步过滤 entity_variants（与 entity_registry 同语义）
    relevant_variants: Dict[str, List[str]] = {}
    if entity_variants:
        for source_name, variants in entity_variants.items():
            if not source_name:
                continue
            if entity_source_name_in_text(source_name, source_text):
                relevant_variants[source_name] = list(variants) if variants else []

    return {
        "source_text": source_text,
        "raw_translation": raw_translation,
        "literary_translation": literary_translation,
        "style_guide": style_guide_stats,
        "entity_registry": relevant_entities,
        "entity_variants": relevant_variants,
    }


def evaluate_chunk(
    source_text: str,
    raw_translation: str,
    literary_translation: str,
    style_guide_stats: Dict[str, Any],
    entity_registry: Optional[Dict[str, Any]] = None,
    *,
    entity_variants: Optional[Dict[str, List[str]]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> CompositeVerdict:
    """对单个 chunk 做完整 typesafe 评估

    Args:
        source_text: 原文
        raw_translation: 直译底稿
        literary_translation: 待评估润色稿
        style_guide_stats: 风格基线（句长/词频等）
        entity_registry: 实体译名登记表（可选）
        entity_variants: 【Round 61】实体中文变体登记表（可选）
        thresholds: 自定义阈值（可选，覆盖默认）

    Returns:
        CompositeVerdict（decision/confidence/reasons/primitive_summary）
    """
    state = build_state(
        source_text=source_text,
        raw_translation=raw_translation,
        literary_translation=literary_translation,
        style_guide_stats=style_guide_stats,
        entity_registry=entity_registry,
        entity_variants=entity_variants,
    )
    questions = build_question_set()

    # Round 61: 构造 per-entity atomic Nouls 并合并（一次 system_one 调用全部完成）
    # Round 105 P2-3: entity_registry_full 始终构造（即使无 variants），供
    # synthesize_verdict 把违规实体映射到规范主名（可执行拒因）。
    entity_registry_full: Dict[str, Dict[str, Any]] = {}
    for source_name, canonical in (entity_registry or {}).items():
        variants = (entity_variants or {}).get(source_name, []) or []
        # Round 108: 规范名去括注（如 敲诈者（临时访客）→敲诈者）。译文永远不会
        # 含括注，Jev 按整串检查必然判违规（ch07 chunk007 根因）。与本地
        # _canonical_present_in_text 同语义（同样剥括注）。
        _canon_critic = re.sub(r"（[^）]*）|\([^)]*\)", "", str(canonical)).strip() or canonical
        entity_registry_full[source_name] = {
            "canonical": _canon_critic,
            "variants": variants,
        }
    if entity_variants:
        per_entity_q = build_per_entity_questions(entity_registry_full)
        if per_entity_q:
            questions.update(per_entity_q)

    backend = get_backend()

    # 调 backend（Jev → OpenAICompat fallback）— 一次调用评估所有 questions
    answers = backend.evaluate(state, questions)
    confidence_source = backend_confidence_source(backend)

    # Round 110: per-entity 确定性护栏 — 规范名/变体确已出现时覆盖 LLM 假阳性
    # （ch08 chunk001: Jev 超时回落 openai_compat 后把 "The manager" 判为违规
    #  noul=0.00，而润色稿含「经理」→ entity_consistency=0 → 不可执行的 rewriter
    #  指令 → REWRITER_STAGNATED。）
    apply_entity_presence_override(answers, entity_registry_full, literary_translation)

    # code 端合成 verdict（自动处理 per-entity 聚合）
    # Round 96 P3: 传 lit_text 让 synthesize_verdict 派生 cjk_compliance 硬闸门
    # Round 105 P2-3: 传 entity_registry_full 让违规实体拒因含规范主名
    return synthesize_verdict(
        answers, thresholds=thresholds or {}, lit_text=literary_translation,
        raw_text=raw_translation,
        entity_registry_full=entity_registry_full,
        confidence_source=confidence_source,
    )


# ────────────────────────────────────────────────────────────────────────
# 兼容层：返回旧 critic 风格的 scores dict（用于过渡期）
# ────────────────────────────────────────────────────────────────────────

def to_legacy_scores(verdict: CompositeVerdict) -> Dict[str, Any]:
    """把 CompositeVerdict 翻译成旧 critic 的 scores dict（向后兼容）

    映射：
      voice_drift             →  voice_consistency
      register_match          →  style_compliance
      entity_consistency      →  entity_consistency_notes
      literal_residue         →  literal_residue
      semantic_loss           →  semantic_preservation
      style_compliance_score  →  readability / fluency 平均
    """
    summary = verdict.primitive_summary
    scores: Dict[str, float] = {}

    vd = summary.get("voice_drift", {})
    if vd.get("choice") == "none":
        scores["voice_consistency"] = 9.0
    elif vd.get("choice") == "subtle":
        scores["voice_consistency"] = 7.5
    elif vd.get("choice") == "severe":
        scores["voice_consistency"] = 4.0
    else:
        scores["voice_consistency"] = 6.0

    rm = summary.get("register_match", {})
    if rm.get("choice") == "match":
        scores["style_compliance"] = 8.5
    else:
        scores["style_compliance"] = 5.0

    # 反向映射：noul 越高越差
    lr = summary.get("literal_residue", {})
    scores["literal_residue"] = round(10.0 - lr.get("noul", 0.0) * 10.0, 1)

    sl = summary.get("semantic_loss", {})
    scores["semantic_preservation"] = round(10.0 - sl.get("noul", 0.0) * 10.0, 1)

    # Round 61：style_compliance_score 同时映射 readability 与 fluency
    sc = summary.get("style_compliance_score", {})
    style_score = sc.get("score", 1.0)
    readability = round(style_score * 5.0, 1)
    scores["readability"] = readability
    scores["fluency"] = readability

    # entity_consistency 维度：把 noul 倒数映射成 0-10 分（保留旧接口）
    ec = summary.get("entity_consistency", {})
    if ec:
        n = ec.get("noul", 0.0)
        scores["entity_consistency_notes"] = []  # 占位（真实 notes 由本地 _check_entity_consistency 写入）
        scores["entity_consistency"] = round(10.0 - n * 10.0, 1)

    # 缺省维度：兜底为 6.0
    for dim in (
        "voice_consistency",
        "style_compliance",
        "literal_residue",
        "semantic_preservation",
        "readability",
        "fluency",
    ):
        if dim not in scores:
            scores[dim] = 6.0

    return scores

