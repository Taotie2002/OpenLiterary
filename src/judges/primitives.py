"""
翻译质量判断原语（typesafe-ai Choice / Score / Noul 风格的 atomic questions）

设计原则（typesafe.ai/docs/concepts/system-one）：
  每个 question 是"一个高度专注的判断，资深从业者几秒钟能下结论"
  避免把"整体翻译质量"作为一个模糊问题——分解为可独立判断的 atomic 单元

原语列表（覆盖翻译质量六大维度）：
  voice_drift            Choice    音色漂移（none / subtle / severe）
  entity_consistency     Noul      实体译名一致性（noul = "存在不一致" 概率）
  literal_residue        Noul      直译残留（noul = "存在直译痕迹" 概率）
  semantic_loss          Noul      关键意象/动作丢失（noul = "存在丢失" 概率）
  register_match          Choice    语域契合（under / match / over）
  style_compliance_score Score     综合风格契合度（与基准指纹距离）

每个 primitive 都被 fast_pair 装入 system_one 的 questions 字典，
一次调用返回所有答案，code 端做概率合成。

【A5 Round 2 修复】
所有 Noul primitive 已重构为 yes/no 语义：
  - instructions 改为结构化对象 {question, context} 而非单字符串
  - 新增 criteria.{true, false} 显式定义 yes/no 的含义
  - noul 值 = "yes" 概率；verdict 阈值检查使用此语义

【Round 61 重构】
entity_consistency 由单一大 Noul（复合判断）拆为 per-entity atomic Noul：
  - 每个 (source_name, canonical, variants) 对应一个独立 Noul question
  - Jev 在更小的 context 上做更精确判断（绕过 CJK 复合判断不准确问题）
  - code 端通过 max 聚合回 entity_consistency 维度（OR 语义）
  - 不提供 entity_variants 时回退到原复合 ENTITY_CONSISTENCY 大 Noul

"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional


# ────────────────────────────────────────────────────────────────────────
# 6 个 atomic primitives（命名空间 — 与 critic 6 维度对齐）
# ────────────────────────────────────────────────────────────────────────

# Choice 1: 音色漂移
VOICE_DRIFT = {
    "type": "choice",
    "instructions": (
        "对比原文与润色稿的叙事口吻/角色音色。原文中角色口吻或旁白视角的特征（如讽刺、克制、幽默、紧张等），"
        "在润色稿中是否被保留？还是被抹平为平淡的中性叙述？"
    ),
    "criteria": {
        "none": "完全保留原著口吻，音色一致",
        "subtle": "轻微漂移但不影响整体",
        "severe": "明显抹平/替换为不同口吻",
    },
}

# Choice 2: 语域契合
REGISTER_MATCH = {
    "type": "choice",
    "instructions": (
        "润色稿的语域（口语/书面/方言色彩）与目标读者期望是否匹配？"
        "过度口语化会让文学文本显得轻浮，过度文学会让儿童文学显得距离感太强。"
    ),
    "criteria": {
        "under": "语域过低（太口语/方言化，与目标风格不符）",
        "match": "语域合适",
        "over": "语域过高（过于书面/文言，与目标不符)",
    },
}

# Noul 1: 实体一致性（yes/no 语义：noul = "存在不一致"的概率）
# Round 61 弃用此作为主路径 — 改用 build_per_entity_questions 拆为 atomic Nouls
# 此处保留仅作 fallback：当 entity_variants 未提供时仍然可用
ENTITY_CONSISTENCY = {
    "type": "noul",
    "instructions": {
        "question": "对照 state.entity_registry 中登记的实体译名，state.literary_translation 中的人名/地名/组织名是否与登记表一致？",
        "context": (
            "翻译方向：English → Chinese（文学翻译）。"
            "state.source_text 为英文原文，state.literary_translation 为中文润色稿，"
            "state.entity_registry 形如 {english_source_name: chinese_canonical}（key=英文指称形式，value=规范中文主名）。"
            "判定规则：(1) 若 entity_registry 为空或没有任何 entry 适用于本章，则视为无登记约束 → 不存在不一致（answer=false）。"
            "(2) 若 entity_registry 有 N 个 entry，逐一检查每个 value（规范中文主名）是否在 literary_translation 中以原样或近形出现；"
            "若 source_text 中能找到对应 key，但润色稿未使用该 value（即译名漏译或换成其他译法），则算 true。"
            "(3) 同一中文译名可以有多种合理变体（如「乔治·洛马克斯先生」/「洛马克斯先生」/「洛马克斯」），"
            "只要润色稿使用的是已登记的同义变体，不算违规。"
        ),
    },
    "criteria": {
        "true": "润色稿中至少存在一处：source_text 中出现的实体（registry 的某个 key 在 source_text 里可找到）在 literary_translation 中未使用对应的规范中文主名或已登记的同义变体",
        "false": "润色稿中所有出现的实体译名都与登记表一致；或 entity_registry 为空无登记约束",
    },
}

# Noul 2: 直译残留（yes/no 语义）
LITERAL_RESIDUE = {
    "type": "noul",
    "instructions": {
        "question": "state.literary_translation 中是否仍有未消化的直译痕迹？",
        "context": "直译痕迹示例：'可悲的缺钱需要'（逐词堆砌）、'他的脸是红色的因为他生气'（英文句式残留）。翻译方向：English → Chinese。",
    },
    "criteria": {
        "true": "润色稿中存在明显的直译残留（中文表达生硬、英文句式套用、未消化的逐词翻译）",
        "false": "润色稿已完全润色为自然中文表达",
    },
}

# Noul 3: 关键语义丢失（yes/no 语义）
SEMANTIC_LOSS = {
    "type": "noul",
    "instructions": {
        "question": "对比 state.source_text 与 state.literary_translation，原文的关键意象、动作细节、隐喻、反讽是否在润色稿中被保留？",
        "context": "翻译方向：English → Chinese。重要信息丢失的示例：典故被淡化、动作细节被省略、反讽语气被删去、关键意象被改写为弱化版本。",
    },
    "criteria": {
        "true": "存在重要信息丢失（典故/动作/反讽/关键意象中至少一项被削弱或省略）",
        "false": "原文的所有关键意象、动作细节、隐喻、反讽都已完整保留",
    },
}

# Score: 综合风格契合度（0=完全不匹配目标风格基线, 2=完全匹配）
STYLE_COMPLIANCE_SCORE = {
    "type": "score",
    "instructions": (
        "综合评估润色稿与目标风格基线的契合度。考虑句长分布、词汇密度、"
        "修辞风格（克制vs铺陈）、口吻词、人称距离等。"
    ),
    "criteria": [
        "明显偏离目标风格（如史诗目标出现口语化，或儿童文学目标出现文言）",
        "部分匹配但有偏差",
        "完全契合目标风格基线",
    ],
}


ALL_PRIMITIVES: Dict[str, Dict[str, Any]] = {
    "voice_drift": VOICE_DRIFT,
    "register_match": REGISTER_MATCH,
    "entity_consistency": ENTITY_CONSISTENCY,
    "literal_residue": LITERAL_RESIDUE,
    "semantic_loss": SEMANTIC_LOSS,
    "style_compliance_score": STYLE_COMPLIANCE_SCORE,
}


def build_question_set() -> Dict[str, Dict[str, Any]]:
    """返回完整的 questions 字典（可一次性提交给 system_one）

    Round 48: 用 deepcopy 隔离嵌套 dict，防止外部 mutate 污染全局 ALL_PRIMITIVES。
    """
    return copy.deepcopy(ALL_PRIMITIVES)


# ────────────────────────────────────────────────────────────────────────
# Round 61: Per-entity atomic Noul 拆分
# ────────────────────────────────────────────────────────────────────────

# Per-entity Noul question 的 key 前缀（verdict 用它识别需要聚合的 Noul）
PER_ENTITY_KEY_PREFIX = "entity_per_"


def _sanitize_key(source_name: str, used: set) -> str:
    """把 source_name 转成合法 Python identifier（用于 question key）。

    Args:
        source_name: 英文 source_name（如 "Mr. Lomax"）
        used: 已使用的 keys（用于避免冲突）

    Returns:
        合法 key，格式：entity_per_<sanitized>
        重复时追加 _2, _3, ...
    """
    # 替换非标识符字符为 _
    sanitized = "".join(c if (c.isalnum() or c == "_") else "_" for c in source_name)
    sanitized = sanitized.strip("_")
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "e_" + sanitized
    # 截断到合理长度
    sanitized = sanitized[:48]
    base = f"{PER_ENTITY_KEY_PREFIX}{sanitized}"
    candidate = base
    n = 2
    while candidate in used:
        candidate = f"{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def build_per_entity_questions(
    entity_registry_full: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """为每个登记实体生成独立的 atomic Noul question。

    Round 61: 把复合 ENTITY_CONSISTENCY 拆为 per-entity atomic Nouls。
    每个问题只问一个 entity 的翻译是否正确（包含变体容认）—— Jev 在更小
    context 上做更精确判断，避免单一大 Noul 在 CJK state 上的准确率问题
    （typesafe.ai 官方 models.md 警告 CJK 准确率较低）。

    Args:
        entity_registry_full: {source_name: {"canonical": str, "variants": List[str]}}
            source_name: 英文 key（来自原文 source_text）
            canonical: 中文规范主名（如 "乔治·洛马克斯先生"）
            variants: 已登记的中文同义变体（可为空列表）

    Returns:
        Dict[str, Dict[str, Any]]: question_id → noul question dict
        question_id 形如 "entity_per_George"、"entity_per_Mr_Lomax" 等

    设计要点：
      - 每个 question 的 instructions 注入 canonical + variants → Jev 知道什么是"合规"
      - instructions.question 简洁明确 — Jev 只判断一个 entity
      - criteria.{true, false} 显式定义 yes/no：true=canonical/variant 出现（合规）
    """
    if not entity_registry_full:
        return {}

    questions: Dict[str, Dict[str, Any]] = {}
    used_keys: set = set()
    for source_name, info in entity_registry_full.items():
        if not source_name:
            continue
        if not isinstance(info, dict):
            # 向后兼容：info 可能是 str（仅 canonical）
            canonical = info
            variants: List[str] = []
        else:
            canonical = info.get("canonical", "") or ""
            variants = info.get("variants", []) or []

        if not canonical:
            continue

        key = _sanitize_key(source_name, used_keys)

        # 构造 variants 描述
        # 构造 variants 描述（始终定义，避免作用域问题）
        variants_text = "、".join(f"「{v}」" for v in variants if v)
        variant_clause = (
            f"或其任意已登记变体：{variants_text}"
            if variants_text
            else ""
        )

        # 构造 criteria
        compliant_desc = (
            f"润色稿中包含规范中文主名「{canonical}」{variant_clause}"
            if variant_clause
            else f"润色稿中包含规范中文主名「{canonical}」"
        )

        questions[key] = {
            "type": "noul",
            "instructions": {
                "question": (
                    f"对于英文 source_name `{source_name}`，state.literary_translation 中"
                    f"是否包含规范中文主名 `{canonical}`{variant_clause}？"
                ),
                "context": (
                    f"翻译方向：English → Chinese。本问题只判断此单一实体的翻译一致性。"
                    f"规范主名 = `{canonical}`。"
                    + (f"已登记变体：{variants_text}。" if variants_text else "")
                    + "state.literary_translation 是中文润色稿。"
                ),
            },
            "criteria": {
                "true": compliant_desc + "（合规）",
                "false": (
                    f"润色稿中既未出现规范中文主名「{canonical}」"
                    + (f"，也未出现任何已登记变体（{variant_clause}）" if variant_clause else "")
                    + "（违规：缺失或使用了未登记译法）"
                ),
            },
        }

    return questions


def aggregate_per_entity_nouls(
    per_entity_answers: Dict[str, Any],
) -> Optional[float]:
    """把 per-entity Noul 答案聚合成单一 entity_consistency noul 值。

    聚合语义：max(1 - noul)（OR 违规）。任一实体违规 → 整体视为违规。
    与原复合 ENTITY_CONSISTENCY 一致："至少存在一处 ... → true"。
    per-entity 问题 criteria.true="合规"，noul = P(合规)；
    此处返回 1 - min(noul) = max(1 - noul) = "存在任一违规"的概率，
    与下游 verdict.py "noul >= threshold → reject" 的语义匹配。

    Args:
        per_entity_answers: {question_id: NoulAnswer}（key 必须以 PER_ENTITY_KEY_PREFIX 开头）

    Returns:
        max(1 - noul) 值（0..1，表示 "任一实体违规" 的概率），若无 per-entity 答案则返回 None
    """
    nouls: List[float] = []
    for qid, ans in per_entity_answers.items():
        if not qid.startswith(PER_ENTITY_KEY_PREFIX):
            continue
        # NoulAnswer 有 noul 字段
        if hasattr(ans, "noul"):
            nouls.append(float(ans.noul))
    if not nouls:
        return None
    return max(1.0 - n for n in nouls)


# ────────────────────────────────────────────────────────────────────────
# Round 110: per-entity 确定性护栏（抵消 LLM/Jev 对 CJK 的假阳性）
# ────────────────────────────────────────────────────────────────────────

def entity_present_in_text(
    canonical: str,
    variants: Optional[List[str]],
    lit_text: str,
) -> Optional[str]:
    """确定性判定规范名（或已登记变体）是否出现在润色稿中，命中返回匹配形式。

    与 pipeline / translator_agent 的 _canonical_present_in_text（Round 105）同语义，
    消除两类系统性假阳性：
      1. 规范名带括注（镇（town）/科德斯（Codders））— 译文永远不会含括注；
      2. 前缀/后缀写法（译文用 乔治·洛马克斯，规范名 乔治·洛马克斯先生）。

    判定顺序：
      - 完整规范名 / 任一已登记变体直接命中；
      - 去括注后的 base 命中；
      - base 长度≥4 时，显著前缀/后缀（长度 ≥ max(3, 2/3·len)）命中。

    Args:
        canonical: 中文规范主名
        variants: 已登记的中文变体列表（可为 None/空）
        lit_text: 待核验的润色稿

    Returns:
        命中的形式（子串），全不命中返回 None
    """
    if not lit_text:
        return None
    cands: List[str] = []
    if canonical:
        cands.append(str(canonical))
    cands.extend(str(v) for v in (variants or []) if v)
    # 1) 完整规范名 / 变体直接命中
    for form in cands:
        if form in lit_text:
            return form
    # 1.5) Round 115: 斜杠备选 canonical（如 "国民党/民族主义者"）— 译文通常只用其中
    # 一个备选（ch06 chunk012 用「民族主义者」），整串/前后缀容差都会差在斜杠上 →
    # 误判 missing → 实体硬失败死循环。任一备选字面命中即算在场。
    for form in cands:
        base = re.sub(r"（[^）]*）|\([^)]*\)", "", form).strip()
        if "/" in base or "／" in base:
            for alt in re.split(r"[/／]", base):
                alt = alt.strip()
                if len(alt) >= 2 and alt in lit_text:
                    return alt
    # 2) 去括注 base 命中
    bases: List[str] = []
    for form in cands:
        base = re.sub(r"（[^）]*）|\([^)]*\)", "", form).strip()
        if base:
            bases.append(base)
        if base and base != form and base in lit_text:
            return base
    # 3) 长名前/后缀容差（与 Round 105 _canonical_present_in_text 同语义）
    for base in bases:
        if len(base) < 4:
            continue
        threshold = max(3, (len(base) * 2) // 3)
        if threshold >= len(base):
            continue
        if base[:threshold] in lit_text:
            return base[:threshold]
        if base[-threshold:] in lit_text:
            return base[-threshold:]
    return None


def apply_entity_presence_override(
    answers: Dict[str, Any],
    entity_registry_full: Optional[Dict[str, Dict[str, Any]]],
    lit_text: str,
) -> List[Dict[str, Any]]:
    """把"规范名确已出现"的 per-entity 违规判定覆盖为合规（noul=1.0）。

    Round 110 背景（ch08 chunk001 根因）: Jev 超时回落 openai_compat 后，把
    "The manager/the manager" 判成违规（noul=0.00），而润色稿含规范名「经理」
    （确定性 entity audit 也确认 present）→ entity_consistency 被 max 聚合打到 0
    → judge REJECT → rewriter 收到"统一为规范主名"的不可执行指令 → 输出字节停滞
    → REWRITER_STAGNATED → PERMANENTLY_FAILED。

    per-entity 问题的 criteria 本身是确定性的（"润色稿包含规范主名或其变体"），
    因此可用子串比对直接复核：命中 → 强制 noul=1.0，否决 LLM 判定；未命中 →
    保留原判定（真违规不受影响，避免掩盖真实缺失）。

    聚合语义（max(1-noul)）在 synthesize_verdict 内执行，本函数需在其之前调用。

    Args:
        answers: backend 返回的 {question_id: Answer}（原地修改）
        entity_registry_full: {source_name: {"canonical": str, "variants": List[str]}}
        lit_text: 润色稿

    Returns:
        被覆盖的实体列表 [{qid, source_name, canonical, old_noul, matched}, ...]
    """
    if not answers or not entity_registry_full or not lit_text:
        return []
    overrides: List[Dict[str, Any]] = []
    # key 回放必须与 build_per_entity_questions 完全同序（同样的 skip 条件 +
    # _sanitize_key 调用顺序），否则多实体 key 冲突（_2/_3 后缀）会错位。
    used_keys: set = set()
    for source_name, info in entity_registry_full.items():
        if not source_name:
            continue
        if isinstance(info, dict):
            canonical = info.get("canonical", "") or ""
            variants = info.get("variants", []) or []
        else:
            canonical = str(info or "")
            variants = []
        if not canonical:
            continue
        qid = _sanitize_key(source_name, used_keys)
        ans = answers.get(qid)
        if ans is None or not hasattr(ans, "noul"):
            continue
        old_noul = float(ans.noul)
        if old_noul >= 1.0:
            continue
        matched = entity_present_in_text(canonical, variants, lit_text)
        if not matched:
            continue
        ans.noul = 1.0
        ans.confidence = 1.0
        overrides.append({
            "qid": qid,
            "source_name": source_name,
            "canonical": canonical,
            "old_noul": old_noul,
            "matched": matched,
        })
    return overrides
