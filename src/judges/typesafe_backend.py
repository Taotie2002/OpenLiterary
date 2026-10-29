"""
typesafe.ai System One 后端 — Jev 优先，MiniMax/DeepSeek fallback

设计：
  - 优先用 typesafe_sdk.TypeSafeClient（调 Jev，typesafe-ai 原生）
  - 不可达/失败时降级到现有 OpenAICompatibleAdapter（MiniMax/DeepSeek）
  - 降级路径同样按 Choice/Score/Noul 三种 schema 提问
  - 返回值结构与 typesafe_sdk.SystemOneResponse 对齐

为什么需要 fallback：
  - typesafe_sdk 已安装（0.7.1），但 api.typesafe.ai 当前 DNS 失败
  - 用户指示："如果执行中发现LLM调用超时，那就是本时段的token配额用完了，请每隔5个小时重试"
  - 因此必须支持优雅降级，让 typesafe-spec 重构即使无 Jev 也可用
"""
from __future__ import annotations

import json
import os
import time
import requests
from typing import Any, Dict, Optional

from src.judges.typesafe_spec import (
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    Answer,
)

logger = __import__("logging").getLogger("sagatrans.judges")


def _confidence_from_probs(probs: Dict[Any, float]) -> float:
    """Round 66: 从概率分布导出 confidence —— 取最大概率值（与 typesafe-ai SDK 行为一致）

    旧实现：MockBackend 硬编码 confidence=0.3/0.4/0.5，与概率分布脱节。
    新实现：confidence = max(probabilities.values()) —— 单峰分布高 confidence，多峰/均匀低 confidence。

    这样下游 low_conf_threshold=0.5 的"跳过"语义更诚实：
      - 概率尖锐（如 severe=0.7, subtle=0.2, none=0.1）→ conf=0.7（明确）→ 不跳过
      - 概率均匀（如 {a:0.33, b:0.33, c:0.33}）→ conf=0.33（模糊）→ 跳过
    """
    if not probs:
        return 0.0
    try:
        return float(max(probs.values()))
    except (TypeError, ValueError):
        return 0.0


# ────────────────────────────────────────────────────────────────────────
# Backend 抽象
# ────────────────────────────────────────────────────────────────────────

class SystemOneBackend:
    """typesafe.ai System One 后端抽象（接口与 typesafe_sdk 对齐）"""

    # Round 118 P1-1: 概率来源标识（供 verdict 按来源分档；子类覆盖）
    backend_kind: str = "unknown"

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        *,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Answer]:
        """对 state 评估所有 questions，返回 answers 字典

        Args:
            state: 字符串 / dict / list（与 typesafe_sdk 一致）
            questions: {q_id: question_dict} （type/instructions/criteria）
            model: 模型覆盖（None 走 backend 默认）
        """
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────────
# Backend 1: Jev 原生（typesafe_sdk）
# ────────────────────────────────────────────────────────────────────────

class JevBackend(SystemOneBackend):
    """typesafe.ai Jev 1.13 后端（首选）"""

    backend_kind = "jev"

    def __init__(self, api_key: Optional[str] = None, model: str = "jev-latest"):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from typesafe_sdk import TypeSafeClient
            if not self.api_key:
                raise RuntimeError(
                    "JevBackend 需要 TYPESAFE_API_KEY 环境变量或显式传入"
                )
            # Round 107 修正: 恢复默认行为 (trust_env=True), 让 httpx2 读代理环境变量.
            # 背景: 用户区域被 Typesafe 451 封锁, 靠 SOCKS 代理路由到允许区域.
            # curl 内置 SOCKS 支持所以能通; httpx2 需要 socksio 包 + socks5:// 方案.
            # 前置条件: pip install socksio.
            # Round 107 P2: httpx2 只支持 socks5/socks5h 方案, 不支持 socks.
            # 若环境变量用 socks://, 归一化为 socks5:// (curl 两者等价).
            import os
            import httpx2
            proxy_url = None
            for var in ("ALL_PROXY", "HTTPS_PROXY", "https_proxy", "all_proxy"):
                val = os.environ.get(var, "")
                if val:
                    proxy_url = val
                    break
            if proxy_url and proxy_url.startswith("socks://"):
                proxy_url = proxy_url.replace("socks://", "socks5://", 1)
            if proxy_url:
                http_client = httpx2.Client(proxy=proxy_url)
                self._client = TypeSafeClient(api_key=self.api_key, http_client=http_client)
            else:
                self._client = TypeSafeClient(api_key=self.api_key)
        return self._client

    def evaluate(self, state, questions, *, model=None, timeout=30.0):
        client = self._ensure_client()
        # typesafe_sdk 期望 Choice/Score/Noul 对象；我们 dict 入参则需要构造
        from typesafe_sdk import Choice, Score, Noul
        q_objects: Dict[str, Any] = {}
        for qid, q in questions.items():
            t = q.get("type")
            if t == "choice":
                q_objects[qid] = Choice(
                    instructions=q.get("instructions"),
                    criteria=q.get("criteria") or {},
                )
            elif t == "score":
                q_objects[qid] = Score(
                    instructions=q.get("instructions"),
                    criteria=q.get("criteria") or [],
                )
            elif t == "noul":
                q_objects[qid] = Noul(
                    instructions=q.get("instructions"),
                    criteria=q.get("criteria"),
                )
            else:
                raise ValueError(f"未知 primitive type: {t}")

        resp = client.system_one(
            state=state,
            questions=q_objects,
            model=model or self.model,
            timeout=timeout,
        )

        # 转换为统一 Answer 格式
        out: Dict[str, Answer] = {}
        for qid, ans in resp.answers.items():
            from typesafe_sdk import NoulAnswer as _NoulA, ChoiceAnswer as _ChoiceA, ScoreAnswer as _ScoreA
            if isinstance(ans, _NoulA):
                # SDK NoulAnswer 没有 confidence 字段（实测验证）
                # SDK schema 文档（api.md）也明确："Choice and Score answers also carry a confidence"
                # Noul 答案的"模型不确定度"由 noul 值本身隐含（noul=0.5 即不确定），
                # 不再尝试访问 ans.confidence（AttributeError 的根因）。
                out[qid] = NoulAnswer(noul=ans.noul)
            elif isinstance(ans, _ChoiceA):
                out[qid] = ChoiceAnswer(
                    choice=ans.choice,
                    probabilities=dict(ans.probabilities),
                    confidence=ans.confidence,
                )
            elif isinstance(ans, _ScoreA):
                out[qid] = ScoreAnswer(
                    score=ans.score,
                    probabilities={int(k): float(v) for k, v in ans.probabilities.items()},
                    confidence=ans.confidence,
                )
        return out


# ────────────────────────────────────────────────────────────────────────
# Backend 2: MiniMax/DeepSeek 降级（用现有 OpenAICompatibleAdapter）
# ────────────────────────────────────────────────────────────────────────

class OpenAICompatBackend(SystemOneBackend):
    """OpenAI 兼容后端降级（MiniMax M3 / DeepSeek V4-flash）

    实现：
      - 把 questions 序列化为 JSON Schema，让 LLM 按 schema 输出
      - 解析 LLM 输出为 Choice/Score/Noul 答案
      - 不返回真实概率（confidence 由 LLM 自报，不严要求）
    """

    backend_kind = "openai_compat"

    def __init__(
        self,
        model_role: str = "primary",  # primary = MiniMax M3
        llm_client=None,
    ):
        from src.utils.llm_adapter import get_llm_client_for_task
        self.model_role = model_role
        self.llm = llm_client or get_llm_client_for_task("critic_scoring")

    def evaluate(self, state, questions, *, model=None, timeout=30.0):
        # 把 state 与 questions 拼成一个 prompt
        if isinstance(state, dict):
            state_text = json.dumps(state, ensure_ascii=False, indent=2)
        elif isinstance(state, list):
            state_text = "\n".join(str(s) for s in state)
        else:
            state_text = str(state)

        # 构建 JSON Schema 指令
        schema = {}
        for qid, q in questions.items():
            t = q.get("type")
            inst = q.get("instructions", "")
            if t == "choice":
                options = list((q.get("criteria") or {}).keys())
                schema[qid] = {
                    "type": "choice",
                    "instruction": inst,
                    "options": options,
                }
            elif t == "score":
                levels = q.get("criteria") or []
                schema[qid] = {
                    "type": "score",
                    "instruction": inst,
                    "levels": levels,
                }
            elif t == "noul":
                schema[qid] = {
                    "type": "noul",
                    "instruction": inst,
                }

        prompt = f"""你是文学翻译质量的原子裁判助手。每次同时判断多个独立的、原子化的判断问题，每个问题都要给出**结构化 JSON 答案**。

【状态（待评估对象）】
{state_text}

【待评估问题集】
{json.dumps(schema, ensure_ascii=False, indent=2)}

【输出 Schema — 严格遵守】
{{
  "answers": {{
    "<question_id>": {{
      "type": "choice" | "score" | "noul",
      "choice": "<选中选项>",            // 仅 choice
      "score": 0..N-1,                    // 仅 score
      "noul": 0..1,                       // 仅 noul
      "confidence": 0..1,                 // 你对这个答案的把握度
      "reasoning": "<不超过 50 字的简短依据>"
    }}
  }}
}}

【硬性约束】
1. 一次请求同时回答所有 questions，不要拆分多次
2. 输出纯 JSON，无 markdown 代码块，无解释文字
3. confidence 必须填写（0=完全猜，1=非常确定）
4. reasoning 不要超过 50 字（避免消耗 token）
5. 答案 ID 必须与上面 schema 中的 key 完全一致

请输出 JSON："""

        # 走完整 OpenAICompatibleAdapter.generate 签名（需要 model_name）
        # 但在网络层错误时快速失败，不消耗 3 次重试 budget
        from src.utils.llm_adapter import get_role_model_name
        model_name = get_role_model_name(self.model_role) or self.model_role
        # 临时把 OpenAI 客户端的重试次数调成 1（默认 3 次 ×5s/10s 重试间隔太慢）
        original_max_retries = getattr(self.llm, "max_retries", 3)
        try:
            if hasattr(self.llm, "max_retries"):
                self.llm.max_retries = 1
            raw = self.llm.generate(
                prompt=prompt,
                model_name=model_name,
                max_tokens=2048,
                temperature=0.1,
            )
            return self._parse_answers(raw, questions)
        except (ConnectionError, OSError, requests.exceptions.ConnectionError) as e:
            # 网络层错误快速失败，让 fallback chain 路由到下一级
            logger.info(f"OpenAICompat 网络层错误，快速降级: {type(e).__name__}: {e}")
            raise ConnectionError(str(e)) from e
        except RuntimeError as e:
            # OpenAICompatibleAdapter 在所有重试耗尽后会抛 RuntimeError
            # 若异常信息是网络层错误，转换成 ConnectionError 让 fallback chain 接管
            err_msg = str(e)
            if any(k in err_msg for k in ("HTTPSConnection", "ConnectionError", "DNS", "NameResolution", "Connection refused")):
                logger.info(f"OpenAICompat 内部网络错误，快速降级: {e}")
                raise ConnectionError(err_msg) from e
            raise
        finally:
            if hasattr(self.llm, "max_retries"):
                self.llm.max_retries = original_max_retries

    def _parse_answers(self, raw: str, questions) -> Dict[str, Answer]:
        from src.utils.llm_adapter import robust_json_loads
        try:
            data = robust_json_loads(raw)
        except Exception as e:
            logger.warning(f"OpenAICompatBackend 解析失败: {e}")
            return {}
        if not isinstance(data, dict):
            return {}
        raw_answers = data.get("answers", {})

        out: Dict[str, Answer] = {}
        for qid, q in questions.items():
            ans_raw = raw_answers.get(qid)
            if not ans_raw:
                continue
            t = q.get("type")
            conf = float(ans_raw.get("confidence", 0.5))
            try:
                if t == "choice":
                    options = list((q.get("criteria") or {}).keys())
                    chosen = ans_raw.get("choice", options[0] if options else "")
                    # Round 67: 伪概率构造 — chosen 默认归入 options（避免 "chosen 不在 probs" 导致 max(probs) ≠ conf）
                    # 旧逻辑: 若 LLM 返回 chosen 不在 criteria，probs 没 chosen → max(probs) = (1-conf)/(n-1) << conf
                    #         self-contradiction: confidence 字段显示高，probability 显示低
                    # 新逻辑: 若 chosen 不在 options，把 chosen 临时加入 probs（自洽优先于 strict criteria 校验）
                    if chosen and chosen not in options and options:
                        options = options + [chosen]
                    rest = (1.0 - conf) / max(1, len(options) - 1)
                    probs = {opt: round(rest, 2) for opt in options}
                    if chosen and chosen in probs:
                        probs[chosen] = round(conf, 2)
                    # 边界：options 为空 → probs 用单点分布，避免下游 DictionaryKeyError
                    if not probs and chosen:
                        probs = {chosen: 1.0}
                    # 使用 Round 66 helper：confidence 与 max(probs) 严格一致
                    conf_aligned = _confidence_from_probs(probs) if probs else conf
                    out[qid] = ChoiceAnswer(
                        choice=chosen,
                        probabilities=probs,
                        confidence=conf_aligned,
                    )
                elif t == "score":
                    levels = q.get("criteria") or []
                    n = len(levels)
                    score_val = float(ans_raw.get("score", 0))
                    # Round 67: 均匀分布 score 的 confidence 应该是 max(probs) = 1/n
                    # 旧逻辑：硬塞 conf（与均匀分布不一致）→ 下游 low_conf_threshold 跳过判定错误
                    probs = {i: round(1.0 / n, 2) for i in range(n)}
                    conf_aligned = _confidence_from_probs(probs) if probs else conf
                    out[qid] = ScoreAnswer(
                        score=score_val,
                        probabilities=probs,
                        confidence=conf_aligned,
                    )
                elif t == "noul":
                    noul_val = float(ans_raw.get("noul", 0.0))
                    # Round 91: 不传 confidence, 让 __post_init__ 从 noul 派生 (与 Jev 路径 line 139
                    # 一致). 旧版硬塞 `confidence=conf`, 而 conf 默认 0.5 — 若 LLM 缺失 confidence
                    # 字段, 直接得到 0.5 (与 eff_conf 不一致); 若 LLM 返回 confidence=0,
                    # __post_init__ 仍按 0.0 sentinel 触发派生, 但任何 (0, 0.5] 的弱 confidence
                    # 都会卡住不再派生 (Round 83 sentinel 只识别 ==0.0).
                    # 设计原则: Noul 答案不携带 confidence (typesafe.ai 官方 SDK schema),
                    # MiniMax/DeepSeek fallback 的 "confidence" 是 LLM 自报, 不可靠, 不应
                    # 覆盖 max(noul, 1-noul) 的距离校准.
                    out[qid] = NoulAnswer(noul=noul_val)
            except Exception as e:
                logger.warning(f"Primitive {qid} 解析失败: {e}")
                continue
        return out


# ────────────────────────────────────────────────────────────────────────
# Backend 3: Mock（启发式离线评估 — 不依赖网络 / LLM API）
# ────────────────────────────────────────────────────────────────────────

class MockBackend(SystemOneBackend):
    """基于规则的离线裁判（用于无网络环境验证 typesafe-ai 架构）

    启发式（最简实现，待后续接入真实模型校准）：
      voice_drift          : 检测直译稿与润色稿的句式相似度（低→疑似漂移）
      register_match       : 检测口语化词密度（过高→under, 文学化词→over）
      entity_consistency   : 检测 entity_registry 中译名是否在润色稿中出现
      literal_residue      : 检测直译底稿特有的英文句式残留
      semantic_loss        : 启发式：润色稿长度 < 直译稿 50% → 怀疑语义压缩
      style_compliance_score: 综合 style_guide 与字数比
    """

    backend_kind = "mock_heuristic"

    def evaluate(self, state, questions, *, model=None, timeout=30.0):
        # 解析 state
        if isinstance(state, dict):
            src = state.get("source_text", "")
            raw = state.get("raw_translation", "")
            lit = state.get("literary_translation", "")
            ent_reg = state.get("entity_registry", {}) or {}
        else:
            src = raw = lit = ""
            ent_reg = {}

        out: Dict[str, Answer] = {}

        # voice_drift
        lit_len, raw_len = len(lit), max(1, len(raw))
        length_ratio = lit_len / raw_len
        if length_ratio < 0.3:
            # 润色稿严重缩水 → 怀疑"假装润色其实删减"
            v_probs = {"none": 0.1, "subtle": 0.2, "severe": 0.7}
            out["voice_drift"] = ChoiceAnswer(
                choice="severe",
                probabilities=v_probs,
                confidence=_confidence_from_probs(v_probs),
            )
        elif length_ratio > 1.5:
            v_probs = {"none": 0.4, "subtle": 0.5, "severe": 0.1}
            out["voice_drift"] = ChoiceAnswer(
                choice="subtle",
                probabilities=v_probs,
                confidence=_confidence_from_probs(v_probs),
            )
        else:
            v_probs = {"none": 0.6, "subtle": 0.3, "severe": 0.1}
            out["voice_drift"] = ChoiceAnswer(
                choice="none",
                probabilities=v_probs,
                confidence=_confidence_from_probs(v_probs),
            )

        # register_match（启发式：检测文学化词密度）
        lit_register_over = any(w in lit for w in ["之", "乎", "矣", "焉", "其"])
        if lit_register_over:
            r_probs = {"match": 0.3, "under": 0.2, "over": 0.5}
            out["register_match"] = ChoiceAnswer(
                choice="over",
                probabilities=r_probs,
                confidence=_confidence_from_probs(r_probs),
            )
        else:
            r_probs = {"match": 0.7, "under": 0.2, "over": 0.1}
            out["register_match"] = ChoiceAnswer(
                choice="match",
                probabilities=r_probs,
                confidence=_confidence_from_probs(r_probs),
            )

        # entity_consistency（看 entity_reg 中的中文译名是否在润色稿中出现）
        if ent_reg:
            present = sum(1 for v in ent_reg.values() if v in lit)
            consistency_ratio = present / len(ent_reg)
            noul_val = max(0.0, 1.0 - consistency_ratio)
        else:
            noul_val = 0.0
        out["entity_consistency"] = NoulAnswer(noul=round(noul_val, 2))  # Round 87: 不硬塞 confidence, 让 __post_init__ 自动派生 (与 Jev 路径一致)

        # literal_residue（直译底稿是否被大量原样保留？启发式：相似度比例）
        # 用简单的字符级比较
        if raw and lit:
            from difflib import SequenceMatcher
            sim = SequenceMatcher(None, raw[:500], lit[:500]).ratio()
            # 相似度 > 0.7 → 高度直译残留
            lit_residue_noul = max(0.0, min(1.0, (sim - 0.4) / 0.6))
        else:
            lit_residue_noul = 0.0
        out["literal_residue"] = NoulAnswer(noul=round(lit_residue_noul, 2))  # Round 87: __post_init__ 派生

        # semantic_loss
        sem_loss_noul = max(0.0, min(1.0, (0.5 - length_ratio) * 2.0)) if length_ratio < 0.5 else 0.0
        out["semantic_loss"] = NoulAnswer(noul=round(sem_loss_noul, 2))  # Round 87: __post_init__ 派生

        # style_compliance_score（综合启发式）
        style_score = 1.0
        if length_ratio < 0.3 or length_ratio > 2.0:
            style_score = 0.0
        elif lit_register_over:
            style_score = 0.5
        s_probs = {0: max(0, 1 - style_score), 1: 0.3, 2: max(0, style_score - 0.3)}
        out["style_compliance_score"] = ScoreAnswer(
            score=style_score,
            probabilities=s_probs,
            confidence=_confidence_from_probs(s_probs),
        )

        # Round 61: per-entity atomic Nouls — Mock 用本地启发式判断每个 entity canonical 是否在润色稿中
        # 与 _check_entity_consistency 同语义（确定性字符匹配，无 LLM）
        from src.judges.primitives import PER_ENTITY_KEY_PREFIX
        state_variants = (state.get("entity_variants", {}) if isinstance(state, dict) else {})
        for qid, q in questions.items():
            if not qid.startswith(PER_ENTITY_KEY_PREFIX):
                continue
            if not isinstance(q, dict):
                continue
            # 反向查 entity_registry 找 canonical（按 source_name 模糊匹配）
            ent_canonical = None
            for sn, cn in ent_reg.items():
                sn_normalized = sn.replace(".", "").replace(" ", "").lower()
                qid_normalized = qid.replace(PER_ENTITY_KEY_PREFIX, "").replace("_", "").lower()
                if sn_normalized == qid_normalized:
                    ent_canonical = cn
                    break
            if not ent_canonical:
                # 兜底：用 qid 末段当作 source_name 查表
                src_name = qid.replace(PER_ENTITY_KEY_PREFIX, "").replace("_", " ").strip()
                ent_canonical = ent_reg.get(src_name, "")
            if not ent_canonical:
               # Mock 不知道该 entity：返回合规（noul=0 表示 no violation）
                out[qid] = NoulAnswer(noul=1.0)  # Round 95: Mock 对齐 Jev 方向 (合规=高 noul)
                continue
            # 检查 canonical 或变体是否在润色稿中
            # Round 95 修复：方向反转 bug。
            # 旧逻辑 "0.0 if in_lit else 0.6" 是为了配合 verdict.py 错误方向故意反着赋值，
            # 与真实 Jev 语义相反，导致 Mock 表面"正常"而真实 Jev 必 reject。
            # 现在与 Jev 一致：合规 = 高 noul = 1.0，不合规 = 低 noul = 0.0。
            matched = ent_canonical in lit
            if not matched:
                for sn, vs in state_variants.items():
                    if vs and ent_canonical in (ent_reg.get(sn, "") or ""):
                        for v in vs:
                            if v and v in lit:
                                matched = True
                                break
                    if matched:
                        break
            noul = 1.0 if matched else 0.0
            out[qid] = NoulAnswer(noul=noul)  # Round 87: __post_init__ 派生

        return out


# ────────────────────────────────────────────────────────────────────────
# Fallback 链
# ────────────────────────────────────────────────────────────────────────

class FallbackBackend(SystemOneBackend):
    """三级 fallback 链：Jev → OpenAICompat → Mock（启发式）

    适用场景：
      - 有 TYPESAFE_API_KEY + 网络可达：Jev 原生
      - 无 typesafe API 但有 OpenAI 兼容 LLM：MiniMax / DeepSeek
      - 完全离线或网络中断：Mock 启发式（仅用于架构验证，不保证精度）
    """

    def __init__(
        self,
        *,
        primary_model: str = "jev-latest",
        fallback_role: str = "primary",
        allow_mock: bool = True,
        allow_openai_fallback: bool = False,
        enabled: bool = True,
        timeout: float = 30.0,
    ):
        self.primary = JevBackend(model=primary_model)
        # self.fallback 保留为 OpenAICompatBackend 实例（向后兼容 verify_typesafe_refactor.py
        # test_05 静态探测），但默认不参与自动链（Jev → Mock）。
        # OpenAICompat 在 LLM-as-judge 模式下不返回校准概率，与 Jev 语义不等价，
        # 强行 fallback 会让 verdict 失去概率含义。详见 audit A3。
        self.fallback = OpenAICompatBackend(model_role=fallback_role)
        self.allow_openai_fallback = allow_openai_fallback
        self.allow_mock = allow_mock
        self.enabled = enabled
        self.timeout = timeout
        self._mock = MockBackend() if allow_mock else None
        # Round 118 P1-1: 记录本次实际命中的后端（供 verdict confidence_source 分档）
        self.last_backend: str = "unknown"
        # Round 121: 401 鉴权失败缓存 —— key 修复须重启进程，本轮内不再反复试探 Jev
        self._jev_auth_dead = False

    def evaluate(self, state, questions, *, model=None, timeout=None):
        # 有效 timeout：调用方传入优先，否则用 self.timeout（来自 config.yaml typesafe.timeout）
        eff_timeout = timeout if timeout is not None else self.timeout

        # 0. enabled=false → 直接走 mock（typesafe 路径完全旁路）
        if not self.enabled:
            if self._mock is not None:
                self.last_backend = "mock_heuristic"
                return self._mock.evaluate(state, questions, model=model, timeout=eff_timeout)
            raise RuntimeError("typesafe 已禁用 (enabled=false)，且 Mock 未启用")

        # 1. 尝试 Jev（typesafe.ai 原生）；401 后本轮不再试探（见下方 except）
        if self.primary.api_key and not self._jev_auth_dead:
            try:
                answers = self.primary.evaluate(state, questions, model=model, timeout=eff_timeout)
                self.last_backend = "jev"
                return answers
            except Exception as e:
                if type(e).__name__ == "TypeSafeAuthenticationError":
                    # 401 鉴权失败：key 修复须重启进程；本进程内不再试探 Jev（避免每 chunk 白跑+刷屏）
                    if not self._jev_auth_dead:
                        logger.warning(f"Jev 鉴权失败（{e}）；本进程内不再尝试 Jev，改走降级链")
                    self._jev_auth_dead = True
                else:
                    logger.info(f"Jev 不可用: {type(e).__name__}: {e}")
                if not self.allow_openai_fallback:
                    # 默认跳过 OpenAICompat（设计原则：Jev → Mock，不模拟 Jev）
                    # 详见 audit A3；OpenAICompat 不返回校准概率，与 Jev 语义不等价。
                    logger.info("allow_openai_fallback=false，跳过 OpenAICompat 直降 Mock")
                else:
                    # 显式开启时才走 OpenAICompat（用于 A/B 调试或本地无 Jev 场景）
                    try:
                        answers = self.fallback.evaluate(state, questions, model=model, timeout=eff_timeout)
                        self.last_backend = "openai_compat"
                        return answers
                    except Exception as e2:
                        logger.info(f"OpenAICompat 不可用，降级到 Mock: {e2}")
        else:
            logger.debug("Jev 跳过：无 TYPESAFE_API_KEY")
            if self.allow_openai_fallback:
                # 无 Jev key 时仍可走 OpenAICompat（显式开启）
                try:
                    answers = self.fallback.evaluate(state, questions, model=model, timeout=eff_timeout)
                    self.last_backend = "openai_compat"
                    return answers
                except Exception as e:
                    logger.info(f"OpenAICompat 不可用，降级到 Mock: {e}")

        # 3. 降级到 Mock（启发式离线）
        if self._mock is not None:
            self.last_backend = "mock_heuristic"
            return self._mock.evaluate(state, questions, model=model, timeout=eff_timeout)
        raise RuntimeError("所有 backend 都不可用，且 Mock 未启用")


# 默认 backend（全局单例）
_DEFAULT_BACKEND: Optional[SystemOneBackend] = None


def _load_typesafe_config() -> Dict[str, Any]:
    """读取 config.yaml 的 typesafe 节（缺字段时用硬编码默认值）

    不在 typesafe_backend.py 顶部 from src.config import Config 是为了避免
    循环 import；此处惰性导入。

    Round 77: 修复 `model_name` 字段未接线的 bug.
      旧版本读 `primary_model`, 但 config.yaml 中 typesafe 节实际配置字段是
      `model_name` (与全局 llm_backend/llm_openai_api 节命名一致).
      旧代码导致 config.yaml 中 `typesafe.model_name` 被静默忽略,
      永远用硬编码默认 `jev-latest`. 现在改读 `model_name`; 同时保留
      `primary_model` 作为 legacy fallback (若用户旧版 yaml 还在用).
    """
    defaults: Dict[str, Any] = {
        "primary_model": "jev-latest",  # 内部 key, 由 FallbackBackend.__init__ 消费
        "fallback_role": "primary",
        "allow_mock": True,
        "allow_openai_fallback": False,  # 默认关闭，详见 audit A3
        "enabled": True,
        "timeout": 30.0,
    }
    try:
        from src.config import Config
        section = Config().get_section("typesafe") or {}
        if "enabled" in section:
            defaults["enabled"] = bool(section["enabled"])
        if "allow_mock" in section:
            defaults["allow_mock"] = bool(section["allow_mock"])
        if "allow_openai_fallback" in section:
            defaults["allow_openai_fallback"] = bool(section["allow_openai_fallback"])
        if "timeout" in section:
            try:
                defaults["timeout"] = float(section["timeout"])
            except (TypeError, ValueError):
                pass
        # Round 77: config.yaml 用 `model_name` 字段 (与全局 llm_backend/llm_openai_api 节命名一致),
        # 旧代码读 `primary_model` 导致配置静默失效. 现在优先 `model_name`,
        # 兼容旧 yaml 用 `primary_model`.
        model_name = section.get("model_name") or section.get("primary_model")
        if model_name:
            defaults["primary_model"] = str(model_name)
        fallback_role = section.get("fallback_role")
        if fallback_role:
            defaults["fallback_role"] = str(fallback_role)
    except Exception as e:
        logger.debug(f"typesafe config 读取失败，使用全部默认值: {e}")
    return defaults


def get_backend() -> SystemOneBackend:
    """获取默认 System One 后端（懒初始化，从 config.yaml:typesafe 读参数）"""
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        cfg = _load_typesafe_config()
        _DEFAULT_BACKEND = FallbackBackend(
            primary_model=cfg["primary_model"],
            fallback_role=cfg["fallback_role"],
            allow_mock=cfg["allow_mock"],
            allow_openai_fallback=cfg["allow_openai_fallback"],
            enabled=cfg["enabled"],
            timeout=cfg["timeout"],
        )
    return _DEFAULT_BACKEND
