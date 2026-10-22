"""
CriticAgent — 多维文学审计 Agent

源：原整合脚本 §6 中的 CriticAgent 类。
职责：对润色稿进行 6 维评分（fluency / readability / style_compliance /
       voice_consistency / semantic_preservation / literal_residue）。
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional

from src.config import config as _config
from src.utils.llm_adapter import (
    get_llm_client_for_task,
    get_role_model_name,
    get_role_extra_body,
    robust_json_loads,
)
from src.judges import evaluate_chunk as ts_evaluate_chunk
from src.judges.typesafe_spec import CompositeVerdict
from src.judges.judge import to_legacy_scores

# A6 Round 2 修复：导入 typesafe_sdk 具体异常类型用于分类诊断
try:
    from typesafe_sdk import (
        TypeSafeError,
        TypeSafeAPIConnectionError,
        TypeSafeAPITimeoutError,
        TypeSafeAuthenticationError,
        TypeSafeRateLimitError,
        TypeSafeBadRequestError,
        TypeSafeAPIResponseValidationError,  # 注：实际 SDK 类名带 "API" 前缀
        TypeSafeAPIError,
    )
    _TYPESAFE_EXC_AVAILABLE = True
except ImportError:
    _TYPESAFE_EXC_AVAILABLE = False
    TypeSafeError = TypeSafeAPIConnectionError = TypeSafeAPITimeoutError = Exception  # type: ignore
    TypeSafeAuthenticationError = TypeSafeRateLimitError = TypeSafeBadRequestError = Exception  # type: ignore
    TypeSafeAPIResponseValidationError = TypeSafeAPIError = Exception  # type: ignore


def _classify_backend_error(exc: Exception) -> Dict[str, Any]:
    """A6: 把异常分类为可读的诊断 dict（写入 critic_report._backend_diagnostics）

    分类目标：让用户在 critic_report 中看到 Jev 失败的根因（auth/network/schema/...），
    而不是只能看到一个通用回退路径。
    """
    if _TYPESAFE_EXC_AVAILABLE:
        if isinstance(exc, TypeSafeAuthenticationError):
            return {"category": "auth", "msg": str(exc), "action": "check TYPESAFE_API_KEY"}
        if isinstance(exc, TypeSafeRateLimitError):
            return {"category": "rate_limit", "msg": str(exc), "action": "wait or upgrade quota"}
        if isinstance(exc, TypeSafeAPITimeoutError):
            return {"category": "timeout", "msg": str(exc), "action": "increase timeout or retry"}
        if isinstance(exc, TypeSafeAPIConnectionError):
            return {"category": "network", "msg": str(exc), "action": "check DNS/firewall/cert"}
        if isinstance(exc, TypeSafeBadRequestError):
            return {"category": "bad_request", "msg": str(exc), "action": "check question schema"}
        if isinstance(exc, TypeSafeAPIResponseValidationError):
            return {"category": "schema_mismatch", "msg": str(exc), "action": "update verdict parser"}
        if isinstance(exc, TypeSafeAPIError):
            return {"category": "api_error", "msg": str(exc), "action": "check typesafe status page"}
    # 兜底：未分类异常
    return {"category": "unknown", "msg": f"{type(exc).__name__}: {exc}", "action": "inspect stack trace"}


class CriticAgent:
    def __init__(self):
        self.llm = get_llm_client_for_task("critic_scoring")
        self.role = _config.get_task_role("critic_scoring")

    def _build_prompt(self, source_text: str, raw_translation: str, literary_translation: str, style_guide_stats: dict) -> str:
        return f"""你是一位极其严苛的文学编辑与翻译评论家。
你的任务是对提供的【文学润色稿】进行深度审计，并依据【风格基准】进行打分。

【评分维度 (0-10分)】
1. Fluency (流畅度): 中文表达是否自然，是否摆脱了"翻译腔"。
2. Readability (可读性): 段落结构、标点使用、句式变化是否符合中文阅读习惯。
3. Style_Compliance (风格契合度): 是否符合要求的平均句长、词汇密度、修辞风格等统计学要求。
4. Voice_Consistency (音色一致性): 角色口吻或旁白视角是否与原著语境保持一致。
5. Semantic_Preservation (语义保留度): 对比【直译底稿】，润色稿是否丢失核心意象、动作细节或隐喻。
6. Literal_Residue (直译残留): 润色稿中是否仍有未打磨的生硬直译残留（如"可悲的缺钱需要"这种逐词翻译）。

【风格基准目标】
{json.dumps(style_guide_stats, ensure_ascii=False)}

【原文】
{source_text}

【直译底稿 (Reference)】
{raw_translation}

【待审计的文学润色稿】
{literary_translation}

【JSON 输出规则 — 必须严格遵守】
1. 输出纯 JSON，禁止包含任何 Markdown 标记、代码块、或额外说明文字
2. 字符串值中的双引号 " 必须转义为 \"
3. 不得在数组/对象的最后一个元素后加逗号
4. 布尔值用 true/false（不加引号），数字用纯数字（不加引号）
5. ⚠️ 如果 JSON 格式错误，你的整份评语将被丢弃，本次审计作废

Schema 要求：
{{
  "scores": {{
    "fluency": 8,
    "readability": 8,
    "style_compliance": 7,
    "voice_consistency": 9,
    "semantic_preservation": 8,
    "literal_residue": 8
  }},
  "critique": "一段尖锐的综合评价",
  "improvement_suggestions": "针对低分项给出具体的修改建议（若无则留空）",
  "specific_edits": [
    {{
      "location": "第X段第Y句（精确定位，必须是润色稿中实际出现的句子）",
      "original": "从润色稿中摘录的原句（≤30字）",
      "issue": "问题类型：passive_voice | word_choice | rhythm | register_mismatch | pronoun_gender | repetition | awkward_metaphor",
      "suggested_direction": "修改方向的具体描述（不超过50字，不直接给替换文本，避免越权替 Rewriter 做决定）"
    }}
  ]
}}

约束：
1. specific_edits 数组可为空（如果整体质量达标无需逐条修改）
2. original 必须是润色稿中实际出现的句子（不要 LLM 编造）
3. suggested_direction 不超过 50 字，只给方向不代写
4. issue 必须是上述封闭枚举值之一
5. 如果 specific_edits 超过 20 条，只输出最关键的 20 条

请严格按上述 Schema 输出唯一一个 JSON 对象。
"""

    def process_chunk(
        self,
        chunk_id: str,
        source_text: str,
        raw_trans: str,
        lit_trans: str,
        style_guide: dict,
        *,
        entity_registry: Optional[Dict] = None,
        entity_variants: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """多维度文学审计

        T2 重构：优先走 typesafe-ai System One 路径（多 atomic primitives 并行评估 +
        code-side verdict 合成），失败/无网络时 fallback 到旧 LLM 6 维评分。

        Args:
            entity_registry: 可选，Decision DB 的实体译名登记表（提升 entity_consistency 维度精度）
            entity_variants: 【Round 61】可选，{source_name: [variant_zh, ...]}。提供时，
                critic 会附加 per-entity atomic Noul questions（Jev 在更小 context 上
                单独评估每个实体一致性，结果通过 max 聚合覆盖 entity_consistency 维度）
        """
        # ───── T2 typesafe-ai 路径 ─────
        from src.judges.typesafe_backend import FallbackBackend, JevBackend, OpenAICompatBackend, MockBackend
        from src.judges.judge import build_state
        from src.judges.primitives import build_question_set, apply_entity_presence_override
        from src.judges.verdict import synthesize_verdict
        from src.judges import get_backend

        try:
            backend = get_backend()
            state = build_state(
                source_text=source_text,
                raw_translation=raw_trans,
                literary_translation=lit_trans,
                style_guide_stats=style_guide,
                entity_registry=entity_registry,
                entity_variants=entity_variants,
            )
            questions = build_question_set()

            # Round 61: per-entity atomic Noul 拆分 — Jev 在小 context 上单独判断每个实体
            # Round 105 P2-3: entity_registry_full 始终构造，供 synthesize_verdict 映射违规实体
            from src.judges.primitives import build_per_entity_questions
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

            # 区分实际命中的 backend 类型（便于 _backend_path 标注）
            backend_path = "unknown"
            answers = {}
            # 非 FallbackBackend 路径（如注入的 stub/单 backend）不会填充 diagnostics，
            # 预先初始化为空 dict，避免末尾 `if backend_diagnostics:` UnboundLocalError
            backend_diagnostics: Dict[str, Any] = {}
            try:
                if isinstance(backend, FallbackBackend):
                    # 探测：哪个 backend 能用（A6 修复：异常不再静默吞，分类记录）
                    if backend.primary.api_key:
                        try:
                            answers = backend.primary.evaluate(state, questions, timeout=backend.timeout)
                            backend_path = "jev"
                        except Exception as e_jev:
                            backend_diagnostics["jev"] = _classify_backend_error(e_jev)
                    if not answers and backend.allow_openai_fallback:
                        try:
                            answers = backend.fallback.evaluate(state, questions, timeout=backend.timeout)
                            backend_path = "openai_compat"
                        except (ConnectionError, OSError) as e_conn:
                            backend_diagnostics["openai_compat"] = {
                                "category": "network",
                                "msg": str(e_conn),
                                "action": "check api.minimax.cn / api.deepseek.com reachability",
                            }
                            raise ConnectionError("OpenAICompat 网络层失败") from e_conn
                else:
                    answers = backend.evaluate(state, questions, timeout=backend.timeout)
                    backend_path = type(backend).__name__
            except (ConnectionError, OSError) as e_conn_outer:
                # 兜底走 Mock 时记录 connection 级诊断
                backend_diagnostics_outer = {
                    "network": {
                        "category": "network",
                        "msg": str(e_conn_outer),
                        "action": "check sandbox/container network policy",
                    }
                }
                if backend.allow_mock and backend._mock is not None:
                    answers = backend._mock.evaluate(state, questions)
                    backend_path = "mock_heuristic"

            # 兜底：OpenAICompat 返回 0 个答案（说明 _config.llm_backend="mock" 走 mock backend，
            #       mock 输出无法解析成 typesafe schema），降级到 MockBackend 启发式
            if not answers:
                if isinstance(backend, FallbackBackend) and backend.allow_mock and backend._mock is not None:
                    answers = backend._mock.evaluate(state, questions)
                    backend_path = "mock_heuristic"
                else:
                    raise RuntimeError("所有 backend 都未产出答案")

            # Round 110: per-entity 确定性护栏 — 规范名/变体确已出现时覆盖 LLM 假阳性
            # （ch08 chunk001: Jev 超时回落 openai_compat 后把 "The manager" 判为违规
            #  noul=0.00，而润色稿含「经理」→ entity_consistency 被 max 聚合打到 0
            #  → judge REJECT → rewriter 无可改 → REWRITER_STAGNATED。）
            presence_overrides = apply_entity_presence_override(
                answers, entity_registry_full, lit_trans
            )
            if presence_overrides:
                detail = ", ".join(
                    f"{o['source_name']}→{o['canonical']}"
                    f"(noul {o['old_noul']:.2f}→1.00, 命中「{o['matched']}」)"
                    for o in presence_overrides[:5]
                )
                print(f"🛡️ [Critic Agent] {chunk_id} per-entity 假阳性护栏: {detail}")

            # Round 8: 阈值从 config.typesafe_thresholds 读取（默认用 verdict.DEFAULT_THRESHOLDS）
            # 让运维可调 Jev reject 门槛，避免 literal_residue/semantic_loss 在边界值（0.5±）误触发 retry
            custom_thresholds = _config.typesafe_thresholds or None
            if custom_thresholds:
                verdict: CompositeVerdict = synthesize_verdict(
                    answers, thresholds=custom_thresholds, lit_text=lit_trans,
                    raw_text=raw_trans,
                    entity_registry_full=entity_registry_full,
                    confidence_source=backend_path,
                )
            else:
                verdict: CompositeVerdict = synthesize_verdict(
                    answers, lit_text=lit_trans,
                    raw_text=raw_trans,
                    entity_registry_full=entity_registry_full,
                    confidence_source=backend_path,
                )
            scores = to_legacy_scores(verdict)
            critique = "; ".join(verdict.reject_reasons) or "全部维度通过"
            result = {
                "scores": scores,
                "critique": critique,
                "improvement_suggestions": critique if verdict.decision == "REJECT" else "",
                "specific_edits": [],
                "_typesafe_verdict": verdict.to_dict(),
                "_backend_path": backend_path,
            }
            # A6: 写入 backend_diagnostics（如有），便于事后排查
            if backend_diagnostics:
                result["_backend_diagnostics"] = backend_diagnostics
            return result
        except Exception as e:
            # A6: 把异常分类后写入 stderr + 不再静默吞
            diag = _classify_backend_error(e)
            print(f"⚠️ [Jev] typesafe 路径失败 [{diag["category"]}]: {diag["msg"]} → {diag["action"]}", file=sys.stderr)

        # ───── 旧 LLM 路径（fallback） ─────
        print(f"🧐 [Critic Agent] 正在对 {chunk_id} 进行多维度文学审计...")
        prompt = self._build_prompt(source_text, raw_trans, lit_trans, style_guide)

        model_key, params = _config.resolve_task_model("critic_scoring")
        model_name = get_role_model_name(self.role) or model_key
        extra_body = get_role_extra_body(self.role)

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
            extra_body=extra_body,
        )

        result = robust_json_loads(raw_output)
        if not result:
            print(f"⚠️ Critic Agent 输出异常")
            return {"critique": "JSON解析失败", "scores": {}, "specific_edits": []}

        if "specific_edits" not in result:
            result["specific_edits"] = []
        elif len(result["specific_edits"]) > 20:
            result["specific_edits"] = result["specific_edits"][:20]

        result["_backend_path"] = "legacy"
        # A6: 若有来自 typesafe 路径的 diagnostics，透传到 legacy fallback 报告
        try:
            if "backend_diagnostics" in dir() and backend_diagnostics:
                result["_backend_diagnostics"] = backend_diagnostics
        except (NameError, UnboundLocalError):
            pass
        return result


# ----- 6.4 JudgeAgent -----
