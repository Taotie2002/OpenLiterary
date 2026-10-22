"""
Verdict 合成器 — 把多个 primitive 答案合成为 PASS/REJECT

设计原则（typesafe.ai/docs/patterns/composite-scoring）：
  - verdict 是 code-side 决策，不调 LLM
  - 每个 primitive 提供概率 + confidence
  - confidence 低的维度要么触发升级（更强 LLM / 人工），要么采用保守判断

阈值常量（与 config.critic_thresholds 语义对齐）：
  - voice_drift.severe prob >= 0.5 → reject
  - voice_drift.subtle prob >= 0.7 → 单独不 reject, 叠加其它 reject 时升级 (Round 82)
  - register_match.{under|over} prob >= 0.4 → reject
  - entity_consistency.noul >= 0.3 → reject（至少一处违反即触发）
  - literal_residue.noul >= 0.5 → reject
  - semantic_loss.noul >= 0.4 → reject
  - style_compliance.score < 1.0 且 confidence >= 0.6 → reject

综合 confidence：
  min(confidences) — 最低把握维度主导
  confidences 对 Choice/Score = ans.confidence；对 Noul = max(noul, 1-noul)
  仅 Choice/Score 参与 low_conf_threshold 升级门控；Noul 由阈值判断决定 reject

Round 60：引入 Noul uncertain zone（typesafe-ai 官方 cookbook
  consistency_noul_cookbook.md 推荐模式）：
  - 当 Noul 概率落在 [noul_uncertain_low, noul_uncertain_high] 时
    不触发 hard reject — 视为 Jev 边界判断，不足以作为拒稿依据。
  - 防止 Jev 在 CJK 评估（state.literary_translation 是中文）准确率较低
    时，noul=0.4~0.6 的噪声概率触发 rewriter 死循环震荡（ch32 实测 113 次）。
  - 区间外（noul < low 或 noul > high）按原 threshold 触发 reject。

Round 61：per-entity atomic Noul 聚合
  - 当 answers 中存在以 PER_ENTITY_KEY_PREFIX 开头的 Noul 时，verdict 优先使用
    这些 atomic Nouls 的 max（OR 聚合）覆盖 entity_consistency 维度。
  - 不存在 per-entity 答案时，回退到原 entity_consistency 单 Noul。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.judges.typesafe_spec import (
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    Answer,
    CompositeVerdict,
)
from src.judges.primitives import (
    PER_ENTITY_KEY_PREFIX,
    aggregate_per_entity_nouls,
)
from src.utils.text_match import compute_cjk_ratio  # Round 96 P3: 派生 CJK 硬闸门


# ────────────────────────────────────────────────────────────────────────
# Round 105 P2-3: 实体违规可执行化
# ────────────────────────────────────────────────────────────────────────

def flagged_entity_labels(
    answers: Dict[str, "Answer"],
    noul_threshold: float,
    entity_registry_full: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[str]:
    """列出 per-entity 判定为「违规」的实体标签，供 reject reason 使用。

    ch32 倒查根因：entity_consistency REJECT 的拒因只有 "noul=0.82 eff_conf=0.82"，
    没列出具体该统一哪些实体 → rewriter 无从下手 → 逐字节停滞。

    违规判定：per-entity noul = P(合规)，(1 - noul) >= max(noul_threshold, 0.5)
    视为该实体违规（Jev 认为更可能不合规）。

    Args:
        answers: {question_id: Answer}，含 entity_per_* 的 per-entity 答案
        noul_threshold: entity_consistency 的 reject 阈值
        entity_registry_full: 可选 {source_name: {"canonical":..., "variants":[...]}}，
            提供时标签含规范主名（更可直接执行）；否则仅用 sanitized source_name。

    Returns:
        ["George→乔治·洛马克斯先生(noul=0.20)", "Mr_Lomax(noul=0.30)", ...]
    """
    labels: List[str] = []
    bar = max(float(noul_threshold), 0.5)
    # 建立 sanitized_key -> source_name 反查（若提供 registry）
    key_to_src: Dict[str, str] = {}
    if entity_registry_full:
        from src.judges.primitives import _sanitize_key, PER_ENTITY_KEY_PREFIX as _PEP
        used: set = set()
        for sn in entity_registry_full:
            if sn:
                # Round 108: _sanitize_key 返回带 entity_per_ 前缀的 key，而查表侧用
                # 去前缀的 base_key — 修前多词 source_name（如 "Lady Eileen Brent"）
                # 的规范名映射永远失效，拒因只剩不可执行的 sanitized key
                # （ch07 chunk016 根因：rewriter 不知道规范主名是「艾琳·布伦特夫人」）。
                key_to_src[_sanitize_key(sn, used)[len(_PEP):]] = sn
    for qid, ans in answers.items():
        if not qid.startswith(PER_ENTITY_KEY_PREFIX):
            continue
        if not hasattr(ans, "noul"):
            continue
        noul = float(ans.noul)
        if (1.0 - noul) < bar:
            continue
        sanitized = qid[len(PER_ENTITY_KEY_PREFIX):]
        # 去掉可能的 _2/_3 去重后缀
        base_key = sanitized.rsplit("_", 1)[0] if sanitized.rsplit("_", 1)[-1].isdigit() else sanitized
        src_name = key_to_src.get(base_key, sanitized)
        if entity_registry_full and src_name in entity_registry_full:
            info = entity_registry_full[src_name]
            canonical = info.get("canonical", "") if isinstance(info, dict) else str(info)
            if canonical:
                labels.append(f"{src_name}→{canonical}(noul={noul:.2f})")
                continue
        labels.append(f"{src_name}(noul={noul:.2f})")
    # 按 noul 升序（最违规的在前），最多 8 个
    labels.sort(key=lambda s: float(s.rsplit("noul=", 1)[-1].rstrip(")")))
    return labels[:8]


# ────────────────────────────────────────────────────────────────────────
# 阈值配置（可由外部覆盖）
# ────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "voice_drift_severe_prob": 0.5,
    "voice_drift_subtle_prob": 0.7,  # subtle 单独不触发，但叠加其它会
    "register_mismatch_prob": 0.4,
    "entity_consistency_noul": 0.3,
    "literal_residue_noul": 0.5,
    "semantic_loss_noul": 0.4,
    # Round 109: semantic_loss 长度交叉校验 — 抑制 Jev 在 CJK 上的误判.
    # ch08_chunk009: Jev 返回 semantic_loss noul=1.0 (100% 丢失), 但译文忠实
    # (len(lit)/len(raw)=0.994, 未压缩). 当润色稿相对直译稿未明显压缩时,
    # 高 semantic_loss noul 视为 Jev 校准噪声, cap 到 reject 阈值以下.
    "semantic_loss_length_crosscheck": True,
    "semantic_loss_length_ratio_min": 0.8,
    "semantic_loss_noul_cap": 0.2,
    "style_compliance_min_score": 1.0,
    "style_compliance_min_confidence": 0.6,
    "low_confidence_threshold": 0.5,
    # Round 60: Noul uncertain zone — typesafe-ai consistency_noul_cookbook 推荐区间
    # 落在区间内的 Noul 概率视为边界判断，不参与 hard reject 门控。
    "noul_uncertain_low": 0.3,
    "noul_uncertain_high": 0.7,
    # Round 72: Per-dimension Noul uncertain zones (默认 = 全局区间).
    # 某些维度（如 literal_residue / semantic_loss）的 Jev 信号在 CJK 上校准噪声较大，
    # 可放宽区间到 [0.3, 0.8] 以抑制边缘值误判. entity_consistency 仍走严格区间 (0.3-0.7).
    # 调用方在 thresholds dict 中覆盖; 未配置走全局 fallback.
    "noul_uncertain_zones": {},  # 例: {"literal_residue": [0.3, 0.8], "semantic_loss": [0.3, 0.75]}
}


# ────────────────────────────────────────────────────────────────────────
# Round 118 P1-1: confidence_source 分档
# ────────────────────────────────────────────────────────────────────────
# Jev (System One) 输出的是校准概率 — 按 config 阈值原样裁决，绝不削弱。
# 降级来源不是校准概率：openai_compat 的 confidence 是 LLM 自报值、概率是
# 构造出的伪分布；mock 是确定性启发式。对未校准来源，上调 hard-reject 门槛
# 与不确定区间上界 — 只有更强信号才允许 REJECT，边界噪声走
# skipped_dimensions（保守跳过），并由 verdict 元数据标注
# confidence_calibrated=False，供下游区分"Jev 认证"与"降级模式"。

DEFAULT_SOURCE_PROFILES: Dict[str, Dict[str, Any]] = {
    # Jev：校准概率，阈值按配置原样（margin=0 保证不弱化）
    "jev": {"calibrated": True, "reject_margin": 0.0, "confidence_margin": 0.0},
    # LLM-as-judge：自报 confidence + 伪概率分布，reject 需要更强证据
    "openai_compat": {
        "calibrated": False, "reject_margin": 0.15, "confidence_margin": 0.15,
        "widen_noul_uncertain_zone": True,
    },
    # Mock：确定性启发式（字符匹配/长度比），精度最低 → 最收紧
    "mock_heuristic": {
        "calibrated": False, "reject_margin": 0.25, "confidence_margin": 0.20,
        "widen_noul_uncertain_zone": True,
    },
    # 未知 backend 一律按未校准处理
    "_uncalibrated": {
        "calibrated": False, "reject_margin": 0.20, "confidence_margin": 0.15,
        "widen_noul_uncertain_zone": True,
    },
}

# reject_margin 附加的维度阈值（Noul 与 Choice 的 hard-reject 门槛）
_SOURCE_MARGIN_THRESHOLD_KEYS = (
    "voice_drift_severe_prob",
    "register_mismatch_prob",
    "entity_consistency_noul",
    "literal_residue_noul",
    "semantic_loss_noul",
)
# confidence_margin 附加的置信门控阈值（Score/Choice 升级门控）
_SOURCE_MARGIN_CONFIDENCE_KEYS = (
    "style_compliance_min_confidence",
    "low_confidence_threshold",
    "cjk_low_confidence_threshold",
)
# widening 后仍不允许吞掉 noul≈1.0 的极端信号
_MAX_WIDENED_ZONE_HIGH = 0.98


def normalize_confidence_source(source: Optional[str]) -> str:
    """把 backend 标识归一化到 profile key。

    "jev"/"JevBackend"/"typesafe"/"system_one" → "jev"
    "openai_compat"/"OpenAICompatBackend"/"llm_self_report" → "openai_compat"
    "mock_heuristic"/"mock"/"MockBackend" → "mock_heuristic"
    其余 → "_uncalibrated"
    """
    s = str(source or "").strip().lower()
    if s in ("jev", "jevbackend", "typesafe", "system_one", "calibrated"):
        return "jev"
    if s in ("openai_compat", "openaicompatbackend", "llm_self_report", "llm"):
        return "openai_compat"
    if s in ("mock_heuristic", "mockbackend", "mock"):
        return "mock_heuristic"
    return "_uncalibrated"


def _apply_source_profile(thr: Dict[str, Any], confidence_source: str) -> tuple:
    """按 confidence_source 生成有效阈值副本，返回 (thr, calibrated)。

    Jev 路径：原阈值原样返回（不弱化）。
    未校准来源：上调 reject 门槛 / 置信门槛，并拓宽 Noul 不确定区间上界，
    让边界噪声走保守跳过而不是 hard REJECT。
    """
    profiles: Dict[str, Dict[str, Any]] = {
        k: dict(v) for k, v in DEFAULT_SOURCE_PROFILES.items()
    }
    user_profiles = thr.get("confidence_source_profiles")
    if isinstance(user_profiles, dict):
        for key, override in user_profiles.items():
            if isinstance(override, dict):
                profiles.setdefault(str(key), {}).update(override)
    source_key = normalize_confidence_source(confidence_source)
    profile = profiles.get(source_key) or profiles["_uncalibrated"]
    calibrated = bool(profile.get("calibrated", source_key == "jev"))
    reject_margin = float(profile.get("reject_margin", 0.0) or 0.0)
    confidence_margin = float(profile.get("confidence_margin", 0.0) or 0.0)
    if reject_margin > 0.0:
        for key in _SOURCE_MARGIN_THRESHOLD_KEYS:
            base = float(thr.get(key, DEFAULT_THRESHOLDS.get(key, 0.0)) or 0.0)
            thr[key] = round(min(0.95, base + reject_margin), 4)
    if confidence_margin > 0.0:
        for key in _SOURCE_MARGIN_CONFIDENCE_KEYS:
            raw = thr.get(key, DEFAULT_THRESHOLDS.get(key))
            if raw is None:
                continue
            thr[key] = round(min(0.95, float(raw) + confidence_margin), 4)
    if reject_margin > 0.0 and profile.get("widen_noul_uncertain_zone", False):
        zones = thr.get("noul_uncertain_zones")
        if isinstance(zones, dict) and zones:
            widened: Dict[str, Any] = {}
            for dim, zone in zones.items():
                if isinstance(zone, (list, tuple)) and len(zone) == 2:
                    widened[dim] = [
                        zone[0],
                        round(min(_MAX_WIDENED_ZONE_HIGH, float(zone[1]) + reject_margin), 4),
                    ]
                else:
                    widened[dim] = zone
            thr["noul_uncertain_zones"] = widened
        high = float(thr.get(
            "noul_uncertain_high", DEFAULT_THRESHOLDS.get("noul_uncertain_high", 0.7)
        ) or 0.0)
        thr["noul_uncertain_high"] = round(
            min(_MAX_WIDENED_ZONE_HIGH, high + reject_margin), 4
        )
    return thr, calibrated


def synthesize_verdict(
    answers: Dict[str, Answer],
    thresholds: Dict[str, float] = None,
    *,
    lit_text: Optional[str] = None,
    raw_text: Optional[str] = None,
    entity_registry_full: Optional[Dict[str, Dict[str, Any]]] = None,
    confidence_source: str = "jev",
) -> CompositeVerdict:
    """合成最终裁决.

    Round 96 P3: 可选 lit_text 参数用于派生 cjk_compliance 硬性闸门.
    当 lit_text CJK 比例 < cjk_hard_threshold (默认 0.5) 时, 直接加 REJECT
    reason (不走 Noul uncertain zone 跳过逻辑), 让 critic / judge 在 Jev 给高分
    情况下仍能拦截低 CJK 输出, 触发 rewriter 重试.
    Round 105 P2-3: entity_registry_full 供 flagged_entity_labels 把违规实体
    映射到规范主名, 生成可执行拒因.
    """
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)
    # Round 118 P1-1: 按概率来源分档（Jev 原样；未校准来源收紧）
    thr, source_calibrated = _apply_source_profile(thr, confidence_source)

    reasons: List[str] = []
    confidences: List[float] = []
    summary: Dict[str, Dict] = {}
    pending_subtle_voice_drift: List = []  # Round 82: 暂存 subtle voice_drift, 二遍决定升级/跳过
    # Round 62: 跟踪被跳过的 dimension（让下游区分"明确 PASS"vs"不确定跳过"）
    skipped_dimensions: List[str] = []
    skipped_reasons: Dict[str, str] = {}

    # Round 96 P3: CJK 硬性闸门 — 在主循环前派生 cjk_compliance 维度
    # 阈值走 config 中的 `cjk_hard_threshold` (默认 0.5), 0 表示禁用.
    # 区别于 Noul uncertain zone: 这是硬性 FAIL, 不走 skipped_dimensions 跳过逻辑.
    if lit_text is not None:
        cjk_hard_threshold = float(thr.get("cjk_hard_threshold", 0.5))
        if cjk_hard_threshold > 0:
            lit_cjk = compute_cjk_ratio(lit_text)
            # Round 107 P6: 短文本/前言 CJK 阈值放宽.
            # ch01 倒查: 前言/目录 (65 字符, 含英文公司名/版权符号/年份) CJK 比例天然低 (0.40),
            # 0.50 阈值误判. 对短文本 (<100 字符) 或前言特征文本, 放宽到 0.30.
            _is_short_or_frontmatter = (
                len(lit_text) < 100
                or any(kw in lit_text for kw in ("目录", "版权", "献给", "版权所有", "题献", "Dedication", "Contents"))
            )
            # 阈值是 cjk_noul (=1-cjk_ratio) 的门槛: cjk_noul >= threshold → reject.
            # 原 0.50: cjk_ratio <= 0.50 → reject. 放宽到 0.70: cjk_ratio <= 0.30 → reject.
            if _is_short_or_frontmatter and cjk_hard_threshold < 0.70:
                cjk_hard_threshold = 0.70
            # noul = P(不是中文) = 1 - cjk_ratio. 比例越低, noul 越高 → reject.
            cjk_noul = 1.0 - lit_cjk
            cjk_eff_conf = max(cjk_noul, 1.0 - cjk_noul)
            summary["cjk_compliance"] = {
                "type": "noul",
                "noul": cjk_noul,
                "confidence": cjk_eff_conf,
                "_hard_gate": True,
            }
            if cjk_noul >= cjk_hard_threshold:
                reasons.append(
                    f"译文 CJK 比例过低: cjk_ratio={lit_cjk:.2f} "
                    f"< {cjk_hard_threshold:.2f} (lit={len(lit_text)} chars)"
                )

    low_conf_threshold = thr["low_confidence_threshold"]
    noul_uncertain_low = thr["noul_uncertain_low"]
    noul_uncertain_high = thr["noul_uncertain_high"]

    # Round 95: per-dim zones 在循环外只查一次, 避免 entity_consistency 第二 block 重新查表.
    per_dim_zones_global = thr.get("noul_uncertain_zones") or {}

    # Round 95: CJK confidence gate.
    # typesafe.ai 官方 models.md / state.md 警告: Jev 在 CJK 上准确率较低,
    # 校准概率的边界值噪声大. 对 Choice/SScore 答案, 应使用比默认 (0.5) 更高的置信门槛,
    # 让边界判断走 skipped_dimensions 而非触发 hard reject, 避免 noise-driven 死循环.
    # 阈值走 config 中的 `cjk_low_confidence_threshold` (默认 0.6), 关闭可用 `cjk_confidence_gating: false`.
    cjk_gating_enabled = thr.get("cjk_confidence_gating", True)
    cjk_low_conf_threshold = thr.get(
        "cjk_low_confidence_threshold",
        max(low_conf_threshold, 0.6),  # 默认 ≥ 0.6 (高于通用 0.5, 收紧 CJK 路径)
    )
    effective_low_conf_threshold = cjk_low_conf_threshold if cjk_gating_enabled else low_conf_threshold

    # Round 61: 如果 answers 中存在 per-entity Nouls，提取出来覆盖 entity_consistency 维度
    per_entity_noul = aggregate_per_entity_nouls(answers)

    for qid, ans in answers.items():
        # 跳过 per-entity Nouls（在聚合阶段处理，不进入主合成逻辑）
        if qid.startswith(PER_ENTITY_KEY_PREFIX):
            summary[qid] = ans.to_dict()
            # 也参与 overall_conf 聚合（max(noul, 1-noul)）
            if isinstance(ans, NoulAnswer):
                confidences.append(max(ans.noul, 1.0 - ans.noul))
            continue
        summary[qid] = ans.to_dict()
        # Noul 答案：SDK 不提供 confidence 信号（实测验证 typesafe_sdk 0.7.1）。
        # Noul 的 noul 值本身就是校准概率，"低置信度"已由 noul 距离 0/1 的远近
        # 表达（noul=0.5 即完全不确定），不需要额外的 confidence 门控。
        # Choice / Score 答案：SDK 提供 confidence，可作为升级门控使用。
        if isinstance(ans, NoulAnswer):
            eff_conf = max(ans.noul, 1.0 - ans.noul)
            confidences.append(eff_conf)
        else:
            confidences.append(ans.confidence)
        # 低置信度（<0.5）跳过强制 reject —— 应升级到更强模型/人工，
        # 当前架构不强制 reject（避免启发式噪声污染决策）。
        # 注意：Noul 不参与此门控（见上），noul 阈值判断仍照常进行。
        if not isinstance(ans, NoulAnswer) and ans.confidence < effective_low_conf_threshold:
            # Round 62: 显式记录 Choice/Score 低置信度跳过
            skipped_dimensions.append(qid)
            skipped_reasons[qid] = (
                f"low_conf={ans.confidence:.2f} < {effective_low_conf_threshold}"
                + (" (cjk)" if cjk_gating_enabled else "")
            )
            continue
        # Round 60: Noul uncertain zone — 边界判断不触发 hard reject。
        # ch32 实测：noul=0.39 (entity_consistency)、noul=0.59 (semantic_loss)
        # 因落在不确定区间内被强 reject，触发 rewriter 113 次震荡。
        # 区间外（noul < low 或 noul > high）按原 threshold 走 reject 逻辑。
        # Round 72: 支持 per-dimension 不确定区间. 在 thresholds 中配置
        #   noul_uncertain_zones = {"literal_residue": [0.3, 0.8], ...}
        # 可放宽特定维度的上界 (literal_residue / semantic_loss 在 CJK 上噪声较大).
        # 未配置走全局 fallback [noul_uncertain_low, noul_uncertain_high].
        if isinstance(ans, NoulAnswer):
            # Round 72: 取 per-dim zone (若无配置走全局)
            zone = per_dim_zones_global.get(qid)
            if zone is None:
                zone = (noul_uncertain_low, noul_uncertain_high)
            zone_low, zone_high = zone
            if zone_low <= ans.noul <= zone_high:
                # Round 62: 显式记录"不确定跳过"而非"明确 PASS"
                skipped_dimensions.append(qid)
                skipped_reasons[qid] = (
                    f"uncertain_noul={ans.noul:.2f} ∈ "
                    f"[{zone_low}, {zone_high}]"
                    + (" (per-dim)" if qid in per_dim_zones_global else " (global)")
                )
                continue
        if isinstance(ans, ChoiceAnswer):
            if qid == "voice_drift":
                if ans.probabilities.get("severe", 0.0) >= thr["voice_drift_severe_prob"]:
                    reasons.append(
                        f"音色漂移（severe）: prob={ans.probabilities['severe']:.2f} conf={ans.confidence:.2f}"
                    )
                elif ans.probabilities.get("subtle", 0.0) >= thr["voice_drift_subtle_prob"]:
                    # Round 82: subtle 单独不触发, 但叠加其它 reject 时升级.
                    # 设计原则: severe 单维度足够 → 立即 reject;
                    #          subtle 不够 → 必须与其他维度共振才 reject (避免轻微漂移误触发 rewriter).
                    # 因为维度按 dict 插入顺序遍历, 单独检查 len(reasons) 会因顺序漏判.
                    # 改为: 先标记为待升级, 所有维度遍历完后, 若 reasons>0 则升级, 否则跳过.
                    pending_subtle_voice_drift.append((qid, ans))
            elif qid == "register_match":
                # Round 90: 保持原合并阈值语义, 仅拆分输出类别名, 让策略 lookup 命中.
                #
                # 旧版: `bad = under + over`, if bad >= register_mismatch_prob (0.4):
                #   reasons.append("语域失配: under+over=...")
                #   → _JEV_CATEGORY_STRATEGY lookup miss → generic fallback.
                #
                # 新版: 合并阈值不变 (避免回归). 触发 reject 后按方向拆分:
                #   - 任一方向 prob >= threshold/2 = 0.2 → 输出对应类别 (说明该方向显著贡献)
                #   - 两方向都显著 → 输出两条 reject
                #   - 一方向显著另一方向不显著 → 仅输出显著方向 (避免 noise 维度污染策略)
                #
                # 这样保持原 register_match_combined_rejects 测试期望 (under=0.2 over=0.25
                # combined=0.45 >= 0.4 → REJECT) 不破坏, 同时让 rewriter 收到具体策略.
                under_p = ans.probabilities.get("under", 0.0)
                over_p = ans.probabilities.get("over", 0.0)
                combined = under_p + over_p
                thr_prob = thr["register_mismatch_prob"]
                if combined >= thr_prob:
                    dir_thr = thr_prob / 2.0
                    if under_p >= dir_thr:
                        reasons.append(
                            f"语域偏离（under）: under={under_p:.2f} conf={ans.confidence:.2f}"
                        )
                    if over_p >= dir_thr:
                        reasons.append(
                            f"语域偏离（over）: over={over_p:.2f} conf={ans.confidence:.2f}"
                        )
        elif isinstance(ans, NoulAnswer):
            eff_conf = max(ans.noul, 1.0 - ans.noul)
            if qid == "entity_consistency":
                # Round 61: per-entity Noul 存在时，优先用聚合值（更准确）
                effective_noul = (
                    per_entity_noul if per_entity_noul is not None else ans.noul
                )
                # Round 88 修复: 第二 block 的 zone 检查必须用 per-dim zone (与第一 block 一致),
                # 不能用全局 zone. Round 72 设计本意: per-dim zone 完全替代该维度的全局 zone.
                # 旧代码 [BUG]: 用全局 zone [noul_uncertain_low, noul_uncertain_high] 重新检查,
                # 当用户配置更窄的 per-dim zone (如 [0.4, 0.6]) 时形成 shadow zone:
                # per-dim 外但全局内的 noul (如 0.35) 被错误跳过 reject, per-dim 配置失效.
                # 修复: 与第一 block 共用 per_dim_zones, 保持 per-dim zone 单一来源.
                # Round 95: 使用循环外已查的 per_dim_zones_global, 避免重复 dict 查找.
                entity_zone = per_dim_zones_global.get("entity_consistency")
                if entity_zone is None:
                    entity_zone = (noul_uncertain_low, noul_uncertain_high)
                entity_zone_low, entity_zone_high = entity_zone
                if entity_zone_low <= effective_noul <= entity_zone_high:
                    pass  # 在该维度 zone 内, 跳过 reject
                elif effective_noul >= thr["entity_consistency_noul"]:
                    # 标明聚合来源
                    source_tag = "（per-entity 聚合）" if per_entity_noul is not None else ""
                    reason = (
                        f"实体一致性违反{source_tag}: noul={effective_noul:.2f} "
                        f"eff_conf={eff_conf:.2f}"
                    )
                    # Round 105 P2-3: 列出具体违规实体，让 rewriter 可执行
                    flagged = flagged_entity_labels(
                        answers, thr["entity_consistency_noul"],
                        entity_registry_full,
                    )
                    if flagged:
                        reason += " | 违规实体(须统一为规范主名): " + ", ".join(flagged)
                    reasons.append(reason)
            elif qid == "literal_residue":
                if ans.noul >= thr["literal_residue_noul"]:
                    reasons.append(
                        f"直译残留: noul={ans.noul:.2f} eff_conf={eff_conf:.2f}"
                    )
            elif qid == "semantic_loss":
                # Round 109: 长度交叉校验 — 抑制 Jev semantic_loss 误判.
                # 若润色稿相对直译稿未明显压缩 (len(lit)/len(raw) >= ratio_min),
                # 高 semantic_loss noul 视为 Jev 校准噪声, cap 到阈值以下.
                eff_noul = ans.noul
                if (
                    thr.get("semantic_loss_length_crosscheck", True)
                    and raw_text and lit_text
                ):
                    ratio_min = float(thr.get("semantic_loss_length_ratio_min", 0.8))
                    cap = float(thr.get("semantic_loss_noul_cap", 0.2))
                    ratio = len(lit_text) / max(1, len(raw_text))
                    if ratio >= ratio_min and eff_noul > cap:
                        eff_noul = cap
                        summary[qid]["noul"] = eff_noul
                        summary[qid]["_length_capped"] = True
                if eff_noul >= thr["semantic_loss_noul"]:
                    reasons.append(
                        f"关键语义丢失: noul={eff_noul:.2f} eff_conf={eff_conf:.2f}"
                    )
        elif isinstance(ans, ScoreAnswer):
            if qid == "style_compliance_score":
                # Round 80: 修复 style_compliance_min_confidence 字段未接线 bug.
                # 旧版只检查 score < min_score, 完全忽略 confidence 门控,
                # 导致低 confidence 的 score 触发 reject (即使 Jev 自己都不确信).
                # 设计原则: confidence < min_confidence 时, 视为 Jev 边界判断,
                # 不参与 hard reject 门控 (与 Noul uncertain zone 一致语义).
                style_min_conf = thr.get("style_compliance_min_confidence", 0.6)
                if ans.confidence < style_min_conf:
                    # 与 Noul uncertain zone 一致: 显式记录"不确定跳过"而非"明确 PASS"
                    skipped_dimensions.append(qid)
                    skipped_reasons[qid] = (
                        f"low_conf={ans.confidence:.2f} < {style_min_conf:.2f}"
                    )
                    continue
                if ans.score < thr["style_compliance_min_score"]:
                    reasons.append(
                        f"风格契合度低: score={ans.score:.2f} conf={ans.confidence:.2f}"
                    )

    # Round 61: 如果用了 per-entity 聚合，把聚合值写到 primitive_summary.entity_consistency.noul
    # 让下游消费者看到一致视图
    if per_entity_noul is not None and "entity_consistency" in summary:
        ec_summary = summary["entity_consistency"]
        # 保留原 eff_conf，但用聚合值替换 noul
        old_eff_conf = ec_summary.get("confidence", max(per_entity_noul, 1 - per_entity_noul))
        summary["entity_consistency"] = {
            "type": "noul",
            "noul": per_entity_noul,
            "confidence": old_eff_conf,
            "_aggregated_from_per_entity": True,
        }

    # Round 82: 第二遍处理 subtle voice_drift 升级 (必须在所有 reasons 收集完成后才知是否升级)
    for qid, ans in pending_subtle_voice_drift:
        if len(reasons) > 0:
            reasons.append(
                f"音色漂移（subtle+其它）: subtle={ans.probabilities['subtle']:.2f} "
                f"severe={ans.probabilities.get('severe', 0.0):.2f} "
                f"conf={ans.confidence:.2f}"
            )
        else:
            # 单独 subtle: 不 reject, 记录为软跳过供下游参考
            skipped_dimensions.append(qid)
            skipped_reasons[qid] = (
                f"subtle={ans.probabilities['subtle']:.2f} < "
                f"{DEFAULT_THRESHOLDS['voice_drift_subtle_prob']:.2f} (alone, no escalation)"
            )

    overall_conf = min(confidences) if confidences else 0.0
    decision = "REJECT" if reasons else "PASS"

    return CompositeVerdict(
        decision=decision,
        confidence=overall_conf,
        reject_reasons=reasons,
        primitive_summary=summary,
        skipped_dimensions=skipped_dimensions,
        skipped_reasons=skipped_reasons,
        confidence_source=normalize_confidence_source(confidence_source),
        confidence_calibrated=source_calibrated,
    )
