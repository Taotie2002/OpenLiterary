"""
System One 判断后端 — Round 145 起默认 MoArk d3-flash。

设计：
  - 优先用 typesafe_sdk.TypeSafeClient 调 MoArk ``/v1/systemone``
  - ``JevBackend`` / ``backend_kind="jev"`` 仅保留历史兼容与已校准阈值档
  - 单请求按 ``max_questions_per_batch``（d3 默认 64）稳定分批
  - 失败时按配置降级到 OpenAICompat / Mock；生产配置默认关闭这两条掩蔽路径
  - 返回值结构与 typesafe_sdk.SystemOneResponse 对齐
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
import requests
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.judges.typesafe_spec import (
    NoulAnswer,
    ChoiceAnswer,
    ScoreAnswer,
    Answer,
)

logger = __import__("logging").getLogger("sagatrans.judges")

# Round 145: MoArk d3 decision backend defaults.  The legacy class/env names
# ("JevBackend", TYPESAFE_*) are kept for compatibility, but the actual
# provider/model are MoArk d3-flash.  ``backend_kind`` intentionally stays
# "jev" so verdict.py keeps the calibrated-probability threshold profile until
# a separate A/B proves d3 is interchangeable.
DEFAULT_API_BASE = "https://api.moark.com"
DEFAULT_MODEL = "d3-flash"
DEFAULT_MAX_QUESTIONS_PER_BATCH = 64


def _config_text(value: Any) -> str:
    """Return a usable config string, treating unresolved ``${ENV}`` as empty."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return ""
    return text


def normalize_api_root(value: Any) -> str:
    """Normalize the API root without duplicating the SDK's ``/v1/systemone``."""
    root = _config_text(value).rstrip("/")
    if not root:
        return ""
    lowered = root.lower()
    for suffix in ("/v1/systemone", "/v1"):
        if lowered.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            lowered = root.lower()
    return root


def _api_base_host(value: str) -> str:
    try:
        return urlparse(value or "").netloc
    except Exception:
        return ""


def _typesafe_sdk_module():
    try:
        import typesafe_sdk as _sdk
        return _sdk
    except ImportError:
        return None


def classify_backend_error(
    exc: Exception,
    *,
    sdk_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Classify a decision-backend failure for diagnostics/circuit breaking.

    Categories used by Round 145:
      auth          — 401/403 explicitly about an invalid API key/token
      quota_exhausted— terminal account balance/quota exhaustion (no retry)
      model_access  — 403 explicitly about permission to use a model
      request_schema— 400 (question count, instructions type, etc.)
      rate_limit / timeout / network / dependency / parse / server / unknown

    The distinction matters: a model-access 403 must NOT set the process-wide
    auth circuit breaker, otherwise a single wrong model name silently disables
    the primary backend for the rest of the run.
    """
    text = str(exc).lower()
    sdk = _typesafe_sdk_module() if sdk_available is not False else None

    # Status-code-specific rules must run before SDK class mapping: MoArk may
    # surface a model-permission 403 as TypeSafeAuthenticationError.
    has_403 = bool(("403" in text) or ("forbidden" in text))
    has_401 = bool(("401" in text) or ("unauthorized" in text))
    has_400 = bool(("400" in text) or ("bad request" in text))
    model_words = any(k in text for k in ("model", "模型"))
    permission_words = any(k in text for k in (
        "无权访问", "无权使用", "无权限", "没有权限", "访问被拒",
        "not allowed", "no access", "access denied", "permission",
        "not authorized", "forbidden",
    ))
    key_words = any(k in text for k in (
        "api key", "api_key", "invalid token", "invalid api",
        "令牌无效", "密钥无效", "鉴权失败", "认证失败", "authentication",
    ))
    quota_exhausted_words = any(k in text for k in (
        "余额已耗尽", "资源包余额", "配额已耗尽", "额度已用尽",
        "insufficient balance", "quota exhausted", "not enough credit",
        "credit balance", "payment required",
    ))
    model_name_error = any(k in text for k in (
        "模型名称", "模型不存在", "model name", "model not found",
        "unknown model", "invalid model",
    ))

    if quota_exhausted_words:
        # A degraded-balance 400 is terminal: retrying every chunk only burns
        # time and produces the same misleading schema diagnosis.
        return {
            "category": "quota_exhausted",
            "retryable": False,
            "msg": str(exc)[:200],
            "action": "充值/切换 MoArk 资源包后重启；本轮应 fail-closed 使用旧校准证据",
        }
    if has_403 and model_words and permission_words and not key_words:
        return {
            "category": "model_access",
            "retryable": False,
            "msg": str(exc)[:200],
            "action": "检查该 API key 是否有权访问当前模型（403 model_access）",
        }
    if model_name_error and (has_400 or has_403):
        # MoArk returns 400 "模型名称有误" for a typo'd model.  Treat it as a
        # model-access/config error, not a question-schema error.
        return {
            "category": "model_access",
            "retryable": False,
            "msg": str(exc)[:200],
            "action": "检查 config typesafe.model_name 是否为 d3-flash / d3-turbo",
        }
    if has_401 or (has_403 and key_words):
        return {
            "category": "auth",
            "retryable": False,
            "msg": str(exc)[:200],
            "action": "check MOARK_API_KEY / legacy TYPESAFE_API_KEY",
        }
    if has_400:
        return {
            "category": "request_schema",
            "retryable": False,
            "msg": str(exc)[:200],
            "action": "检查 max_questions_per_batch / instructions schema / 模型问题上限",
        }

    if sdk is not None:
        if isinstance(exc, getattr(sdk, "TypeSafeAuthenticationError", ())):
            return {
                "category": "auth", "retryable": False,
                "msg": str(exc)[:200], "action": "check MOARK_API_KEY / legacy TYPESAFE_API_KEY",
            }
        if isinstance(exc, getattr(sdk, "TypeSafeRateLimitError", ())):
            return {
                "category": "rate_limit", "retryable": True,
                "msg": str(exc)[:200], "action": "wait or upgrade quota",
            }
        if isinstance(exc, getattr(sdk, "TypeSafeAPITimeoutError", ())):
            return {
                "category": "timeout", "retryable": True,
                "msg": str(exc)[:200], "action": "increase timeout or retry",
            }
        if isinstance(exc, getattr(sdk, "TypeSafeAPIConnectionError", ())):
            return {
                "category": "network", "retryable": True,
                "msg": str(exc)[:200], "action": "check DNS/firewall/cert/proxy",
            }
        if isinstance(exc, getattr(sdk, "TypeSafeBadRequestError", ())):
            return {
                "category": "request_schema", "retryable": False,
                "msg": str(exc)[:200], "action": "检查 max_questions_per_batch / instructions schema",
            }
        if isinstance(exc, getattr(sdk, "TypeSafeAPIResponseValidationError", ())):
            return {
                "category": "schema_mismatch", "retryable": False,
                "msg": str(exc)[:200], "action": "update verdict parser",
            }
        if isinstance(exc, getattr(sdk, "TypeSafeAPIError", ())):
            return {
                "category": "api_error", "retryable": True,
                "msg": str(exc)[:200], "action": "check MoArk status page",
            }

    missing_module = isinstance(exc, ModuleNotFoundError) or any(
        k in text for k in ("no module named", "modulenotfound", "not installed")
    )
    if missing_module:
        module_name = getattr(exc, "name", None)
        target = f" `{module_name}`" if module_name else ""
        return {
            "category": "dependency", "retryable": False,
            "msg": str(exc)[:200],
            "action": f"安装缺失依赖{target}（见 requirements-core.txt），或显式启用 OpenAICompat 降级",
        }
    if any(k in text for k in ("429", "rate limit", "rate_limit", "quota", "too many requests")):
        return {"category": "rate_limit", "retryable": True, "msg": str(exc)[:200],
                "action": "wait or upgrade quota"}
    if any(k in text for k in ("timeout", "timed out", "connection reset", "connection aborted")):
        return {"category": "timeout", "retryable": True, "msg": str(exc)[:200],
                "action": "increase timeout or retry"}
    if any(k in text for k in ("json", "decode", "parse", "invalid response", "malformed")):
        return {"category": "parse", "retryable": False, "msg": str(exc)[:200],
                "action": "check response format"}
    if any(k in text for k in ("500", "502", "503", "504", "internal server", "bad gateway", "service unavailable")):
        return {"category": "server", "retryable": True, "msg": str(exc)[:200],
                "action": "check MoArk status page"}
    if any(k in text for k in ("connection", "dns", "network", "unreachable", "ssl", "certificate", "proxy")):
        return {"category": "network", "retryable": True, "msg": str(exc)[:200],
                "action": "check DNS/firewall/cert/proxy"}
    return {
        "category": "unknown", "retryable": True,
        "msg": f"{type(exc).__name__}: {exc}"[:200],
        "action": "inspect stack trace",
    }


def _typesafe_sdk_available() -> bool:
    """类型化后端依赖探测（不触发 import，避免每 chunk 重复失败）。"""
    try:
        return importlib.util.find_spec("typesafe_sdk") is not None
    except (ImportError, ValueError):
        return False


def _is_auth_failure(exc: Exception) -> bool:
    """Only true auth failures should trip the process-wide circuit breaker."""
    return classify_backend_error(exc).get("category") == "auth"


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


def batch_questions(
    questions: Dict[str, Any],
    max_per_batch: int = DEFAULT_MAX_QUESTIONS_PER_BATCH,
) -> list:
    """Split questions into stable, insertion-order-preserving batches.

    d3 currently accepts at most 64 questions per system_one request.  The
    caller's dict order is meaningful: base primitives are inserted before
    per-entity questions, so keeping order also keeps the base questions in
    the first batch.
    """
    try:
        limit = max(1, int(max_per_batch))
    except (TypeError, ValueError):
        limit = DEFAULT_MAX_QUESTIONS_PER_BATCH
    items = list(questions.items())
    if not items:
        return []
    return [dict(items[i:i + limit]) for i in range(0, len(items), limit)]


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
    """MoArk d3 System One 后端（保留 JevBackend 类名以兼容旧调用方）。

    Round 145: Jev 配额已停用，实际 provider/model 改为 MoArk
    ``d3-flash``。``backend_kind`` 仍为 ``"jev"`` —— 这不是伪装迁移完成，
    而是让 verdict 继续使用 Jev 的已校准阈值档，直到 d3-flash / d3-turbo
    的 A/B 证明概率语义可互换。真实 provider/model/endpoint 会写入
    ``last_diagnostics`` 和 critic report 的 ``_backend_diagnostics``。
    """

    backend_kind = "jev"  # calibrated-probability compatibility label
    provider = "moark"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        max_questions_per_batch: Optional[int] = None,
    ):
        cfg = _load_typesafe_config()

        # Priority: explicit > config (resolved ${MOARK_API_KEY}) > MOARK env >
        # legacy TYPESAFE_API_KEY.  Config/MOARK must win over the legacy env
        # name: on this machine TYPESAFE_API_KEY still holds the expired Jev
        # key, while MOARK_API_KEY is the working d3 credential.
        if api_key is None:
            api_key = (
                _config_text(cfg.get("api_key"))
                or _config_text(os.environ.get("MOARK_API_KEY"))
                or _config_text(os.environ.get("TYPESAFE_API_KEY"))
            )
        self.api_key = _config_text(api_key)

        if model is None:
            model = _config_text(cfg.get("primary_model")) or DEFAULT_MODEL
        self.model = str(model or DEFAULT_MODEL)

        if api_base is None:
            api_base = (
                _config_text(cfg.get("api_base"))
                or _config_text(os.environ.get("TYPESAFE_BASE_URL"))
                or DEFAULT_API_BASE
            )
        self.api_base = normalize_api_root(api_base) or DEFAULT_API_BASE

        if max_questions_per_batch is None:
            max_questions_per_batch = cfg.get("max_questions_per_batch")
        try:
            self.max_questions_per_batch = max(1, int(max_questions_per_batch))
        except (TypeError, ValueError):
            self.max_questions_per_batch = DEFAULT_MAX_QUESTIONS_PER_BATCH

        self._client = None
        self.last_diagnostics: Dict[str, Any] = {}

    def _ensure_client(self):
        if self._client is None:
            from typesafe_sdk import TypeSafeClient
            if not self.api_key:
                raise RuntimeError(
                    "System One 后端需要 MOARK_API_KEY（兼容 TYPESAFE_API_KEY）环境变量、"
                    "config typesafe.api_key，或显式传入"
                )
            # Round 107: keep proxy support.  httpx2 only accepts
            # socks5/socks5h, so normalize the common socks:// spelling.
            # Import lazily only when a proxy is configured: the SDK already
            # depends on httpx2, but direct no-proxy calls must not fail on a
            # partially installed/optional environment.
            proxy_url = None
            for var in ("ALL_PROXY", "HTTPS_PROXY", "https_proxy", "all_proxy"):
                val = os.environ.get(var, "")
                if val:
                    proxy_url = val
                    break
            if proxy_url and proxy_url.startswith("socks://"):
                proxy_url = proxy_url.replace("socks://", "socks5://", 1)
            kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }
            if proxy_url:
                import httpx2
                kwargs["http_client"] = httpx2.Client(proxy=proxy_url)
            self._client = TypeSafeClient(**kwargs)
        return self._client

    @staticmethod
    def _convert_answer(ans: Any) -> Answer:
        from typesafe_sdk import (
            NoulAnswer as _NoulA,
            ChoiceAnswer as _ChoiceA,
            ScoreAnswer as _ScoreA,
        )
        if isinstance(ans, _NoulA):
            # Noul carries no confidence field; noul itself encodes uncertainty.
            return NoulAnswer(noul=ans.noul)
        if isinstance(ans, _ChoiceA):
            return ChoiceAnswer(
                choice=ans.choice,
                probabilities=dict(ans.probabilities),
                confidence=ans.confidence,
            )
        if isinstance(ans, _ScoreA):
            return ScoreAnswer(
                score=ans.score,
                probabilities={int(k): float(v) for k, v in ans.probabilities.items()},
                confidence=ans.confidence,
            )
        raise TypeError(f"未知 typesafe answer 类型: {type(ans).__name__}")

    def evaluate(self, state, questions, *, model=None, timeout=30.0):
        if not _typesafe_sdk_available():
            raise ModuleNotFoundError("No module named 'typesafe_sdk'")
        client = self._ensure_client()
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

        used_model = model or self.model
        batches = batch_questions(q_objects, self.max_questions_per_batch)
        merged: Dict[str, Answer] = {}
        batch_counts = []
        input_tokens = 0
        output_tokens = 0

        for batch in batches:
            resp = client.system_one(
                state=state,
                questions=batch,
                model=used_model,
                timeout=timeout,
            )
            batch_counts.append(len(batch))
            answers_obj = getattr(resp, "answers", None) or {}
            missing = [qid for qid in batch if qid not in answers_obj]
            if missing:
                raise RuntimeError(
                    "system_one 返回缺少 qid: " + ", ".join(str(q) for q in missing[:8])
                )
            for qid, ans in answers_obj.items():
                if qid not in batch:
                    raise RuntimeError(f"system_one 返回了未请求的 qid: {qid}")
                if qid in merged:
                    raise ValueError(f"重复 qid: {qid}")
                merged[qid] = self._convert_answer(ans)
            usage = getattr(resp, "usage", None)
            try:
                input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass

        missing_all = [qid for qid in q_objects if qid not in merged]
        if missing_all:
            raise RuntimeError(
                "system_one 合并后缺少 qid: " + ", ".join(str(q) for q in missing_all[:8])
            )

        self.last_diagnostics = {
            "provider": self.provider,
            "model": used_model,
            "base_url_host": _api_base_host(self.api_base),
            "question_count": len(q_objects),
            "batch_count": len(batches),
            "batch_question_counts": batch_counts,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        # Preserve the caller's original question order for deterministic
        # aggregation downstream.
        return {qid: merged[qid] for qid in q_objects}


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
        model_role: Optional[str] = None,
        llm_client=None,
        task_name: str = "critic_scoring",
    ):
        """绑定与客户端一致的任务路由（Round 140）。

        旧实现从 `critic_scoring` 取 DeepSeek 客户端，却从
        `typesafe.fallback_role: primary` 取 MiniMax 模型名，导致请求
        `DeepSeek 客户端 + MiniMax-M3.1-Flash` 被网关 403。模型名必须和
        客户端来自同一个 task 路由；`fallback_role` 仅保留为历史配置兼容。
        """
        from src.config import config as _cfg
        from src.utils.llm_adapter import (
            get_llm_client,
            get_llm_client_for_task,
            get_role_model_name,
        )

        self.task_name = task_name
        # 显式 model_role 仍按旧语义工作；只有“不指定角色”的生产兜底才使用
        # critic_scoring 任务路由，避免客户端与 model_name 来自不同角色。
        self.model_role = model_role or _cfg.get_task_role(task_name)
        _, params = _cfg.resolve_task_model(task_name)
        self.params = dict(params)
        if model_role and model_role != _cfg.get_task_role(task_name):
            # task effort 是按 task 路由角色校验的；显式换角色时不能把
            # 该角色的档位原样下发（例如 DeepSeek 的 none 发给 MiniMax）。
            self.params.pop("effort", None)
        self.model_name = get_role_model_name(self.model_role) or self.model_role
        if llm_client is not None:
            self.llm = llm_client
        elif model_role:
            self.llm = get_llm_client(model_role)
        else:
            self.llm = get_llm_client_for_task(task_name)

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
        model_name = model or self.model_name
        call_params = dict(self.params)
        call_params.setdefault("temperature", 0.1)
        call_params.setdefault("max_tokens", 2048)
        # 临时把 OpenAI 客户端的重试次数调成 1（默认 3 次 ×5s/10s 重试间隔太慢）
        original_max_retries = getattr(self.llm, "max_retries", 3)
        try:
            if hasattr(self.llm, "max_retries"):
                self.llm.max_retries = 1
            raw = self.llm.generate(
                prompt=prompt,
                model_name=model_name,
                **call_params,
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
    """三级 fallback 链：MoArk d3 → OpenAICompat → Mock（启发式）

    适用场景：
      - 有 MOARK_API_KEY + 网络可达：MoArk d3 原生
      - 无 typesafe API 但有 OpenAI 兼容 LLM：MiniMax / DeepSeek
      - 完全离线或网络中断：Mock 启发式（仅用于架构验证，不保证精度）
    """

    def __init__(
        self,
        *,
        primary_model: str = DEFAULT_MODEL,
        api_base: Optional[str] = None,
        max_questions_per_batch: Optional[int] = None,
        fallback_role: str = "primary",
        allow_mock: bool = True,
        allow_openai_fallback: bool = False,
        enabled: bool = True,
        timeout: float = 30.0,
    ):
        self.primary = JevBackend(
            model=primary_model,
            api_base=api_base,
            max_questions_per_batch=max_questions_per_batch,
        )
        # self.fallback 保留为 OpenAICompatBackend 实例（向后兼容 verify_typesafe_refactor.py
        # test_05 静态探测），但默认不参与自动链（Jev → Mock）。
        # OpenAICompat 在 LLM-as-judge 模式下不返回校准概率，与 Jev 语义不等价，
        # 强行 fallback 会让 verdict 失去概率含义。详见 audit A3。
        # 兜底的 LLM 客户端与模型名统一由 critic_scoring 任务路由解析。
        # fallback_role 参数保留仅为旧配置/旧调用方兼容，不再参与模型绑定。
        self.fallback_role = fallback_role
        self.fallback = OpenAICompatBackend(task_name="critic_scoring")
        self.allow_openai_fallback = allow_openai_fallback
        self.allow_mock = allow_mock
        self.enabled = enabled
        self.timeout = timeout
        self._mock = MockBackend() if allow_mock else None
        # Round 118 P1-1: 记录本次实际命中的后端（供 verdict confidence_source 分档）
        self.last_backend: str = "unknown"
        self.last_diagnostics: Dict[str, Any] = {}
        # Round 121/145: auth and model-access failures are process-lifetime
        # circuit breakers (a restart is required after fixing key/model access).
        self._jev_auth_dead = False
        self._jev_model_dead = False
        self._jev_quota_dead = False
        self._jev_disabled_reason = ""

    def primary_available(self) -> bool:
        """True when the primary backend should still be attempted."""
        return bool(
            self.primary.api_key
            and not self._jev_auth_dead
            and not self._jev_model_dead
            and not self._jev_quota_dead
        )

    def evaluate(self, state, questions, *, model=None, timeout=None):
        # 有效 timeout：调用方传入优先，否则用 self.timeout（来自 config.yaml typesafe.timeout）
        eff_timeout = timeout if timeout is not None else self.timeout

        # 0. enabled=false → 直接走 mock（typesafe 路径完全旁路）
        if not self.enabled:
            if self._mock is not None:
                self.last_backend = "mock_heuristic"
                return self._mock.evaluate(state, questions, model=model, timeout=eff_timeout)
            raise RuntimeError("typesafe 已禁用 (enabled=false)，且 Mock 未启用")

        # 1. 尝试 MoArk d3（保留 last_backend="jev" 校准标签）
        if self.primary_available():
            try:
                answers = self.primary.evaluate(state, questions, model=model, timeout=eff_timeout)
                self.last_backend = "jev"
                self.last_diagnostics = dict(
                    getattr(self.primary, "last_diagnostics", {}) or {}
                )
                return answers
            except Exception as e:
                missing_sdk = isinstance(e, (ImportError, ModuleNotFoundError)) or (
                    "typesafe_sdk" in str(e)
                )
                diag = classify_backend_error(e)
                self.last_diagnostics = {
                    "provider": getattr(self.primary, "provider", "moark"),
                    "model": getattr(self.primary, "model", None),
                    "error": diag,
                }
                dependency_failure = missing_sdk or diag["category"] == "dependency"
                if dependency_failure or diag["category"] == "auth":
                    # Auth/dependency failure: restart required after fixing.
                    if not self._jev_auth_dead:
                        reason = ("System One 判断后端依赖缺失" if dependency_failure
                                  else "MoArk 鉴权失败")
                        logger.warning(f"{reason}（{e}）；本进程内不再尝试主后端，改走降级链")
                    self._jev_auth_dead = True
                    self._jev_disabled_reason = (reason if dependency_failure
                                                 else "MoArk 鉴权失败")
                elif diag["category"] == "model_access":
                    if not self._jev_model_dead:
                        logger.warning(
                            "MoArk 无权访问模型 %s（%s）；本进程内不再尝试主后端",
                            getattr(self.primary, "model", "?"), e,
                        )
                    self._jev_model_dead = True
                    self._jev_disabled_reason = "MoArk 模型访问被拒 (model_access)"
                elif diag["category"] == "quota_exhausted":
                    if not self._jev_quota_dead:
                        logger.warning(
                            "MoArk 资源包余额/quota 已耗尽（%s）；本进程内不再尝试主后端",
                            e,
                        )
                    self._jev_quota_dead = True
                    self._jev_disabled_reason = "MoArk 资源包余额已耗尽 (quota_exhausted)"
                else:
                    logger.info(f"MoArk 不可用: {type(e).__name__}: {e}")
                if not self.allow_openai_fallback:
                    # 默认跳过 OpenAICompat（设计原则：Jev → Mock，不模拟 Jev）
                    # 详见 audit A3；OpenAICompat 不返回校准概率，与 Jev 语义不等价。
                    logger.info("allow_openai_fallback=false，跳过 OpenAICompat 直降 Mock")
                else:
                    # 显式开启时才走 OpenAICompat（用于 A/B 调试或本地无 Jev 场景）
                    try:
                        answers = self.fallback.evaluate(state, questions, model=model, timeout=eff_timeout)
                        self.last_backend = "openai_compat"
                        self.last_diagnostics = {
                            "provider": "openai_compat",
                            "model": getattr(self.fallback, "model_name", None),
                        }
                        return answers
                    except Exception as e2:
                        logger.info(f"OpenAICompat 不可用，降级到 Mock: {e2}")
        else:
            if self._jev_auth_dead or self._jev_model_dead or self._jev_quota_dead:
                logger.debug("Jev 跳过：%s", self._jev_disabled_reason or "鉴权/依赖不可用")
            else:
                logger.debug("Jev 跳过：无 MOARK_API_KEY / TYPESAFE_API_KEY")
            if self.allow_openai_fallback:
                # 无 Jev key 时仍可走 OpenAICompat（显式开启）
                try:
                    answers = self.fallback.evaluate(state, questions, model=model, timeout=eff_timeout)
                    self.last_backend = "openai_compat"
                    self.last_diagnostics = {
                        "provider": "openai_compat",
                        "model": getattr(self.fallback, "model_name", None),
                    }
                    return answers
                except Exception as e:
                    logger.info(f"OpenAICompat 不可用，降级到 Mock: {e}")

        # 3. 降级到 Mock（启发式离线）
        if self._mock is not None:
            self.last_backend = "mock_heuristic"
            self.last_diagnostics = {"provider": "mock_heuristic", "model": None}
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
        "primary_model": DEFAULT_MODEL,  # 内部 key, 由 FallbackBackend.__init__ 消费
        "api_base": DEFAULT_API_BASE,
        "api_key": "",
        "max_questions_per_batch": DEFAULT_MAX_QUESTIONS_PER_BATCH,
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
        api_base = section.get("api_base")
        if _config_text(api_base):
            defaults["api_base"] = normalize_api_root(api_base)
        api_key = section.get("api_key")
        if _config_text(api_key):
            defaults["api_key"] = str(api_key).strip()
        try:
            defaults["max_questions_per_batch"] = max(
                1, int(section.get("max_questions_per_batch", DEFAULT_MAX_QUESTIONS_PER_BATCH))
            )
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
            api_base=cfg.get("api_base"),
            max_questions_per_batch=cfg.get("max_questions_per_batch"),
            fallback_role=cfg["fallback_role"],
            allow_mock=cfg["allow_mock"],
            allow_openai_fallback=cfg["allow_openai_fallback"],
            enabled=cfg["enabled"],
            timeout=cfg["timeout"],
        )
    return _DEFAULT_BACKEND
