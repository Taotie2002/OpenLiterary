"""
SagaTrans LLM 适配层（2 个后端实现 + JSON 解析 + 客户端管理）

源：原整合脚本 §2（LLM 适配层）+ 文件头中的 robust_json_loads，拆分到独立模块。

后端支持：
  - MockLLMAdapter          离线测试 / dry-run
  - OpenAICompatibleAdapter 通用 OpenAI 兼容 HTTP（DeepSeek / MiniMax / vLLM / LM Studio 等）

公共 API：
  get_llm_client(role)            按角色获取 LLM 客户端
  get_llm_client_for_task(task)   按任务名路由到角色
  get_role_model_name(role)       获取角色配置中的模型名
  get_task_extra_body(task)       获取任务级 extra_body 透传参数
  robust_json_loads(raw)          防御性 JSON 解析
"""
from __future__ import annotations

import json
import os
import re
import time
import sys
import threading
import hashlib
import gc
import random
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.config import config as _config

logger = logging.getLogger("sagatrans")


def robust_json_loads(raw_response: str, expected_type: type = dict):
    """Robust JSON parser for LLM outputs.

    Handles: <think> blocks, markdown fences, preamble text,
    trailing text (via raw_decode), trailing commas (via cleanup),
    and embedded JSON extraction as last resort.

    Args:
        raw_response: 待解析字符串
        expected_type: 期望返回类型（dict 或 list）。解析结果类型不符时按策略降级。

    Returns:
        解析后的 dict/list；解析失败返回 expected_type() 的空实例（dict→{}, list→[]）。
    """
    empty = [] if expected_type is list else {}

    if not raw_response or not raw_response.strip():
        return empty

    cleaned = raw_response
    # 1. Strip <think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    # 2. Strip preamble before first { or [
    cleaned = re.sub(r'^[^{[]*', '', cleaned)
    # 3. Strip markdown code fences
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return empty

    # Strategy 1: raw_decode extracts first JSON value, ignores trailing text
    try:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(cleaned)
        if isinstance(result, expected_type):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: clean trailing commas and retry（保留原始类型：list 或 dict）
    try:
        fallback = re.sub(r',\s*}', '}', cleaned)
        fallback = re.sub(r',\s*]', ']', fallback)
        result = json.loads(fallback)
        if isinstance(result, expected_type):
            return result
        # 类型不符（如 Judge 返回数组/字符串而期望 dict）：返回空实例，
        # 维持"解析失败返回 expected_type() 空实例"契约，避免调用方 .get 崩溃
        return empty
    except json.JSONDecodeError:
        pass

    # Strategy 3: json_repair handles unescaped quotes, missing brackets, single quotes, etc.
    try:
        from json_repair import repair_json as _rj
        repaired = _rj(cleaned)
        result = json.loads(repaired)
        if isinstance(result, expected_type):
            return result
        return empty
    # P2-2: 收窄兜底异常，保留 MemoryError/KeyboardInterrupt 上抛
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        pass

    # Strategy 4: extract outermost { ... } block
    try:
        brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if brace_match:
            result = json.loads(brace_match.group())
            if isinstance(result, expected_type):
                return result
            return empty
    except json.JSONDecodeError:
        pass

    return empty


# 配置加载器（支持单文件模式降级）
# P0-1: 多候选探测 —— 既支持 `python -m src.translator_agent`（找到 src.config），
# 也支持 `python translator_agent.py`（单文件部署，无 src 包），降级到内建默认。
# 同时支持 SAGATRANS__CONFIG_MODULE 环境变量显式指定外部配置模块。



# Section 2 — LLM 适配层
# ────────────────────────────────────────────────────────────────────


class QuotaExhaustedError(RuntimeError):
    """Round 129: 配额/限流持续耗尽（区别于可重试的瞬时网络/服务端错误）。

    由 OpenAICompatibleAdapter 在 429 连续出现（或 402/403 带配额关键字）时抛出；
    FallbackRoleAdapter 捕获后把该角色永久降级到 fallback 角色，避免整章翻译
    卡在单一模型的退避重试里（MiniMax 配额按 5 小时窗重置）。
    """


class LLMAdapter(ABC):
    """大语言模型调用基类"""
    def __init__(self):
        # Round 27: 累积 token 用量（每次 generate 成功后递增）
        self._cumulative_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        self._usage_lock = threading.Lock()

    @abstractmethod
    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        """生成接口；**kwargs 用于透传子类特有参数（如 extra_body / effort）。"""
        pass

    @abstractmethod
    def unload_model(self):
        """强制释放显存的统一接口"""
        pass

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Round 27: 子类在 generate 末尾调用此方法累加真实 token 用量。

        OpenAI-compatible API 标准 usage 字段：prompt_tokens / completion_tokens / total_tokens。
        Mock 后端按字符估算；HTTP 后端取服务端返回的真实 usage。
        Round 30: 累加后检查阈值，触发一次性 stderr 告警（idempotent via _warned_thresholds）。
        """
        try:
            pt = max(0, int(prompt_tokens))
            ct = max(0, int(completion_tokens))
        except (TypeError, ValueError):
            return
        with self._usage_lock:
            self._cumulative_usage["prompt_tokens"] += pt
            self._cumulative_usage["completion_tokens"] += ct
            self._cumulative_usage["total_tokens"] += pt + ct
            self._cumulative_usage["calls"] += 1
            current_total = self._cumulative_usage["total_tokens"]

        # Round 30: 阈值告警（一次性，跨进程 reset 时清空）
        self._maybe_warn_cost_threshold(current_total)

    def _maybe_warn_cost_threshold(self, current_total: int) -> None:
        """Round 30: 当 cumulative total_tokens 跨过阈值时输出一次性 stderr 告警。

        阈值从 _config.cost_warnings.{role_name}_token_cap 读取；缺省值见 _default_warn_thresholds。
        阈值跨越使用 self._warned_thresholds set 去重，避免每次调用都打印。
        """
        if not hasattr(self, "_warned_thresholds"):
            self._warned_thresholds = set()
        # 按 50k / 100k / 200k / 500k 阶梯告警（避免阈值单点过度告警）
        warn_steps = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
        for step in warn_steps:
            if current_total >= step and step not in self._warned_thresholds:
                self._warned_thresholds.add(step)
                role = getattr(self, "_role_label", type(self).__name__)
                print(
                    f"⚠️ [Cost Monitor] {role} 累计 token 已达 {step:,}（current={current_total:,}）。"
                    "若非预期，请检查 retry 循环是否失控。",
                    file=sys.stderr,
                )

    def get_cumulative_usage(self) -> Dict[str, int]:
        """Round 27: 返回 adapter 自创建以来累计 token 用量（线程安全快照）。"""
        with self._usage_lock:
            return dict(self._cumulative_usage)

    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前进程内存使用情况"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_gb": mem_info.rss / (1024**3),
            "vms_gb": mem_info.vms / (1024**3),
            "percent": process.memory_percent()
        }

    def check_memory_pressure(self) -> bool:
        """检查进程内存压力（通用诊断；原 MLX 本地推理专有逻辑已移除）。"""
        usage = self.get_memory_usage()
        threshold = 0.8
        return usage["percent"] / 100 > threshold


class MockLLMAdapter(LLMAdapter):
    """Mock LLM 适配器：用于开发测试，无需真实模型服务"""

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self._count_lock = threading.Lock()

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        with self._count_lock:
            self.call_count += 1

        # Round 27: 先算 content 再统一 return，便于末尾记录 mock token 估算
        if "考据专家" in prompt and "典故来源" in prompt:
            content = self._mock_reference_response()
        elif "直译" in prompt and "字面忠实" in prompt and "不进行文学润色" in prompt:
            content = self._mock_raw_translation(prompt)
        # P1-5: 使用稳定的结构标识路由 Mock（不再是已废弃的"荣获过星云奖"人设句）
        elif "排版与脚注协议" in prompt and ("文学润色" in prompt or "直译底稿" in prompt):
            content = self._mock_literary_rewrite(prompt)
        elif "极其严苛的文学编辑与翻译评论家" in prompt and "评分维度" in prompt:
            content = self._mock_critic_report()
        elif "终审" in prompt and ("决定" in prompt or "定稿" in prompt):
            content = self._mock_judge_decision()
        else:
            content = f"[Mock Response #{self.call_count}] 模拟输出"

        # Round 27: mock 后端按 len//4 估算（与 Round 13 一致；mock 仅为本地测试）
        self.record_usage(len(prompt) // 4, len(content) // 4)
        return content

    def _mock_reference_response(self) -> str:
        return json.dumps({
            "references": [
                {
                    "source_text": "Keats",
                    "allusion_target": "英国浪漫主义诗人约翰·济慈",
                    "strategy": "RETAIN_AND_ANNOTATE",
                    "translated_term": "济慈",
                    "reason": "知名历史人物，保留音译并在脚注中说明其文学地位"
                },
                {
                    "source_text": "Beauty is truth, truth beauty",
                    "allusion_target": "济慈《希腊古瓮颂》名句",
                    "strategy": "RETAIN_AND_ANNOTATE",
                    "translated_term": "美即是真，真即是美",
                    "reason": "经典诗句，需保留原意并加注出处"
                }
            ]
        }, ensure_ascii=False)

    def _mock_raw_translation(self, prompt: str) -> str:
        if "【原文】" in prompt:
            source = prompt.split("【原文】")[-1].strip()
        else:
            source = "未知原文"
        return f"[直译] {source}"

    def _mock_literary_rewrite(self, prompt: str) -> str:
        markers = ["【直译底稿（语义基准，不可偏离）】", "【直译底稿】"]
        marker = next((m for m in markers if m in prompt), None)
        if marker:
            after_marker = prompt.split(marker)[-1]
            raw = after_marker.split("请直接输出")[0].strip()
        else:
            raw = ""
        raw = raw.replace("[直译] ", "")
        return f"""{raw}

译文经过文学润色，保持了原著的风格与节奏。

[^1]: 译注：济慈（1795-1821），英国浪漫主义诗人。
[^2]: 译注：引自《希腊古瓮颂》。"""

    def _mock_critic_report(self) -> str:
        return json.dumps({
            "scores": {
                "fluency": 8,
                "style_compliance": 7,
                "voice_consistency": 8,
                "semantic_preservation": 8,
                "readability": 8,
                "literal_residue": 8
            },
            "critique": "译文流畅自然，风格契合度良好，语义保留完整。",
            "improvement_suggestions": ""
        }, ensure_ascii=False)

    def _mock_judge_decision(self) -> str:
        return json.dumps({
            "decision": "PASS",
            "reject_reason": "",
            "new_style_rule": {
                "rule_description": "处理诗歌引用时保持原韵律感",
                "reason": "提升文学翻译的美感还原度"
            }
        }, ensure_ascii=False)

    def unload_model(self):
        pass


# Round 128: effort 档位 → 供应商原生参数（取代 Round 117-126 的 thinking 布尔归一）。
# - 三家均为 OpenAI 兼容接口，档位统一编码为 `reasoning_effort: <level>`；
# - minimax：服务端强制 adaptive thinking（R121 实测），档位附加在其上；
# - effort 为空 = 不下发任何参数（由上游默认决定）；
# - 合法档位表与校验在 config.resolve_task_model（本函数只负责编码）。
def build_effort_params(api_base: str, model_name: str, effort: Optional[str]) -> Dict[str, Any]:
    if not effort:
        return {}
    eff = str(effort).strip().lower()
    payload: Dict[str, Any] = {"reasoning_effort": eff}
    target = f"{api_base or ''} {model_name or ''}".lower()
    if "minimax" in target:
        payload["thinking"] = {"type": "adaptive"}
    return payload


# Round 117b: 内联 think 块提取（用 . 匹配换行；MiniMax adaptive 模式实测会把
# 推理内联在 content 首位，如 "<think…</think>正文"）
_INLINE_THINK_RE = re.compile(r'<think(.*?)</think>', re.DOTALL)

# Round 129: 配额耗尽关键字（仅配合 402/403 判定；429 走连续计数）
_QUOTA_HINT_RE = re.compile(r"quota|insufficient|欠费|额度|余额", re.IGNORECASE)


class OpenAICompatibleAdapter(LLMAdapter):
    """基于 HTTP 的 OpenAI 兼容 API 适配器 (如 LM Studio / vLLM)"""
    # 类级：按 api_base 分组的限流器，同一服务商共享限流
    # P4-6: 用 OrderedDict 实现 LRU，上限 _MAX_LIMITERS (默认 32)；超出时清理最久未用
    from collections import OrderedDict as _OD_p46
    _rate_limiters: "_OD_p46[str, dict]" = _OD_p46()  # api_base_key -> {"last": float, "lock": Lock, "interval": float}
    _MAX_LIMITERS = 32

    def __init__(self, api_base: str, api_key: str, max_retries: int = 3, retry_delay: float = 2.0, request_timeout: int = 300):
        # Round 32 修：调 super().__init__() 初始化 _cumulative_usage + _usage_lock
        # （Round 27 改动时遗漏，导致 cost_report 触发时 AttributeError）
        super().__init__()
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise RuntimeError("缺少依赖 requests，请运行: pip install requests")
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.min_request_interval = 1.5
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # SSL 校验：默认严格。历史上 minimaxi 拼写错误曾触发 verify=False workaround，
        # 正确端点是 api.minimax.cn（少一个 i），无 SSL 问题，故此处保持严格校验。
        self._verify_ssl = True
        # Round 117: thinking 模式审计轨迹（reasoning_content 不混入主输出）
        self.last_reasoning = ""
        self.last_reasoning_only = False
        # Round 129: 连续 429 计数（配额耗尽判定；成功响应即清零）
        self._consecutive_429 = 0
        # 初始化该 api_base 的限流器
        limiter_key = self._get_limiter_key(api_base)
        if limiter_key not in self.__class__._rate_limiters:
            # P4-6: LRU 上限保护 — 超过上限时清理最久未用
            while len(self.__class__._rate_limiters) >= self.__class__._MAX_LIMITERS:
                evicted_key, _ = self.__class__._rate_limiters.popitem(last=False)
                # 仅 stderr 提示（限流器清理不应影响 stdout）
                print(f"🗑️ [OpenAI Adapter] 限流器表已满，清理最久未用: {evicted_key}", file=sys.stderr)
            self.__class__._rate_limiters[limiter_key] = {
                "last": 0.0,
                "lock": threading.Lock(),
                "interval": self.min_request_interval,
            }
        else:
            # P4-6: 复用时刷新 LRU 顺序（移到末尾），并允许 interval 更新
            self.__class__._rate_limiters.move_to_end(limiter_key)
            self.__class__._rate_limiters[limiter_key]["interval"] = self.min_request_interval

    def _get_limiter_key(self, api_base: str) -> str:
        """提取服务商标识作为限流分组键"""
        api_base_lower = api_base.lower()
        if "minimax" in api_base_lower:
            return "minimax"
        if "deepseek" in api_base_lower:
            return "deepseek"
        if "openai" in api_base_lower:
            return "openai"
        # 其它按 host 分组
        from urllib.parse import urlparse
        return urlparse(api_base).netloc

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        effort = kwargs.pop("effort", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Round 128: effort 档位按 provider 编码（无档位=不下发；extra_body 仍最后合并）
        payload.update(build_effort_params(self.api_base, model_name, effort))
        # 合并剩余自定义 extra_body（用户覆盖）
        if extra_body:
            payload.update(extra_body)

        # 频率限制：按服务商分组，各自独立限流
        limiter_key = self._get_limiter_key(self.api_base)
        limiter = self.__class__._rate_limiters[limiter_key]
        with limiter["lock"]:
            now = time.time()
            elapsed = now - limiter["last"]
            if elapsed < limiter["interval"]:
                time.sleep(limiter["interval"] - elapsed)
            limiter["last"] = time.time()
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                    verify=self._verify_ssl,
                )
                # Round 117/R121: 服务端 400 不识别 thinking 字段 → 剥离后立即重发一次。
                # 旧实现用 continue 依赖剩余 attempt 预算：max_retries=1（judge 兜底客户端）
                # 时会直接耗尽循环，以 "API 调用失败,已重试 1 次: None" 形式误报。
                if (response.status_code == 400
                        and any(k in payload for k in ("thinking", "reasoning_effort"))):
                    _rejected = [k for k in ("thinking", "reasoning_effort") if k in payload]
                    payload.pop("thinking", None)
                    payload.pop("reasoning_effort", None)
                    print(f"⚠️ [OpenAI Adapter] 400: 服务端不识别 {_rejected}"
                          f"（model={payload.get('model')}）→ 剥离后重试"
                          f" | body: {response.text[:160]}",
                          file=sys.stderr, flush=True)
                    response = self._requests.post(
                        f"{self.api_base}/chat/completions",
                        headers=self.headers,
                        json=payload,
                        timeout=self.request_timeout,
                        verify=self._verify_ssl,
                    )
                # Round 129: 配额守卫 —— 429 首次按常规重试（瞬时限流），连续两次
                # （或 402/403 带配额关键字）判定为配额耗尽 → QuotaExhaustedError
                # （不继续退避空转；由 FallbackRoleAdapter 降级到兜底角色）。
                if response.status_code == 429 or (
                    response.status_code in (402, 403)
                    and _QUOTA_HINT_RE.search(response.text[:400])
                ):
                    if response.status_code == 429:
                        self._consecutive_429 += 1
                    if response.status_code != 429 or self._consecutive_429 >= 2:
                        raise QuotaExhaustedError(
                            f"HTTP {response.status_code} 配额/限流持续耗尽"
                            f"（model={payload.get('model')}）: {response.text[:160]}"
                        )
                # 429 是限流（属于 4xx 但应重试），其他 4xx 是客户端错误；5xx 由 raise_for_status 抛出后被外层重试逻辑捕获
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise RuntimeError(f"API 客户端错误 {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                self._consecutive_429 = 0
                resp_json = response.json()
                message = resp_json["choices"][0]["message"]
                # Round 27: 解析真实 token usage（OpenAI 兼容 API 标准字段）
                usage = resp_json.get("usage") or {}
                if usage:
                    self.record_usage(
                        int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)),
                    )
                # Round 117: thinking 模式下 reasoning_content 只作审计轨迹，
                # 不得混入 content（旧行为把推理拼进译文会污染下游产物）。
                # Round 129: token.ai-hf.cn 网关的字段名是 reasoning（非
                # reasoning_content），两者都读取，避免推理被落在盲区。
                content = message.get("content") or ""
                reasoning = (message.get("reasoning_content")
                             or message.get("reasoning") or "")
                # Round 117b: 部分 provider（如 MiniMax adaptive thinking）把推理内联在
                # content 里（"<think…</think>正文"）。若 reasoning_content 为空，
                # 把闭合的 think 块抽出为审计轨迹，正文只保留 think 之后的内容。
                if (not reasoning) and ("<think" in content):
                    m = _INLINE_THINK_RE.search(content)
                    if m:
                        reasoning = m.group(1).strip()
                        content = (content[:m.start()] + content[m.end():]).strip()
                    else:
                        # 未闭合 think 块（推理被 max_tokens 截断）→ 判定失败，
                        # 避免把半截推理当正文写入成品。
                        self.last_reasoning = content.strip()
                        self.last_reasoning_only = True
                        raise RuntimeError(
                            "模型输出为未闭合 think 块（疑推理截断），判定调用失败"
                        )
                self.last_reasoning = reasoning
                if not content.strip() and reasoning:
                    # Round 129（质量优先）：仅有推理无正文 = 思考吃光 max_tokens。
                    # 旧行为把推理当译文返回，会把思维链污染进成品；现判定调用失败，
                    # 触发上层重试/兜底，绝不让 CoT 混入译文。
                    self.last_reasoning_only = True
                    raise RuntimeError(
                        f"模型仅输出推理、无正文（reasoning={len(reasoning)} chars）——"
                        f"疑似 thinking 耗尽 max_tokens，判定调用失败"
                    )
                return content
            except RuntimeError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️ API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}, {wait_time:.1f}s 后重试...")
                    time.sleep(wait_time)

        raise RuntimeError(f"API 调用失败，已重试 {self.max_retries} 次: {last_error}")

    def unload_model(self):
        pass


def _map_effort_for_fallback(effort: str, levels: tuple) -> Optional[str]:
    """Round 129: 降级时的 effort 跨家映射（就近取更强档）；无法安全映射则不下发。"""
    eff = str(effort or "").strip().lower()
    if not eff:
        return None
    if not levels:
        return eff
    if eff in levels:
        return eff
    alias = {"medium": "high", "minimal": "low", "off": "none"}
    cand = alias.get(eff, eff)
    return cand if cand in levels else None


def _fallback_levels_for(api_base: str, model_name: str) -> tuple:
    """推断 fallback 目标的合法档位集合（供降级映射用；失败时返回空=原样透传）。"""
    try:
        from src.config import EFFORT_LEVELS_BY_PROVIDER, effort_provider_of
        provider = effort_provider_of(model_name, api_base)
        return EFFORT_LEVELS_BY_PROVIDER.get(provider, ()) if provider else ()
    except Exception:
        return ()


class FallbackRoleAdapter(LLMAdapter):
    """Round 129: 角色级配额降级包装器（透明替代普通适配器）。

    主角色（如 primary/MiniMax）调用因配额/限流持续耗尽（QuotaExhaustedError）时，
    进程内永久切换到 fallback 角色（如 secondary/DeepSeek），保证流水线不因单模型
    配额中断或长时间阻塞；其余错误不触发降级，按原路径处理。

    降级单向（避免失败-回切抖动）；MiniMax 配额按 5 小时窗重置，重启进程即恢复。
    """

    def __init__(
        self,
        primary: LLMAdapter,
        fallback: LLMAdapter,
        fallback_model_name: str,
        primary_model_name: str = "",
        fallback_role: str = "",
        fallback_levels: tuple = (),
    ):
        super().__init__()
        self._primary = primary
        self._fallback = fallback
        self._fallback_model_name = fallback_model_name
        self._primary_model_name = primary_model_name or "primary"
        self._fallback_role = fallback_role or "fallback"
        self._fallback_levels = tuple(fallback_levels or ())
        self._degraded = False
        self._role_label = f"quota-fallback({self._primary_model_name})"
        self.last_reasoning = ""
        self.last_reasoning_only = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _sync_audit(self, adapter: LLMAdapter) -> None:
        self.last_reasoning = getattr(adapter, "last_reasoning", "")
        self.last_reasoning_only = getattr(adapter, "last_reasoning_only", False)

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048,
                 temperature: float = 0.3, **kwargs) -> str:
        if not self._degraded:
            try:
                out = self._primary.generate(
                    prompt=prompt, model_name=model_name,
                    max_tokens=max_tokens, temperature=temperature, **kwargs,
                )
                self._sync_audit(self._primary)
                return out
            except QuotaExhaustedError as e:
                self._degraded = True
                print(
                    f"⛔ [Quota Fallback] {self._primary_model_name} 配额/限流耗尽 → "
                    f"永久降级到 {self._fallback_model_name}（role={self._fallback_role}；"
                    f"配额重置后重启进程恢复主力） | {e}",
                    file=sys.stderr, flush=True,
                )
        kwargs = dict(kwargs)
        if kwargs.get("effort") and self._fallback_levels:
            kwargs["effort"] = _map_effort_for_fallback(str(kwargs["effort"]), self._fallback_levels)
            if kwargs["effort"] is None:
                kwargs.pop("effort", None)
        out = self._fallback.generate(
            prompt=prompt, model_name=self._fallback_model_name,
            max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        self._sync_audit(self._fallback)
        return out

    def get_cumulative_usage(self) -> Dict[str, int]:
        a = self._primary.get_cumulative_usage()
        b = self._fallback.get_cumulative_usage()
        return {k: int(a.get(k, 0)) + int(b.get(k, 0))
                for k in ("prompt_tokens", "completion_tokens", "total_tokens", "calls")}

    def unload_model(self):
        try:
            self._primary.unload_model()
        finally:
            self._fallback.unload_model()


# Per-role 客户端缓存：role_key -> LLMAdapter
_client_instances: dict = {}


def _build_single_client(role_cfg: Dict[str, Any]) -> LLMAdapter:
    """构建单一角色的底层适配器（不含 fallback 包装）。"""
    backend = role_cfg.get("backend", "mock")
    if backend == "mock":
        return MockLLMAdapter()
    if backend in ("openai_api", "ollama", "nim", "mistral", "custom"):
        section = _config.openai_api if hasattr(_config, "openai_api") else {}
        # P4-8: 优先读 role_cfg 内的 per-role 超时配置，回退全局 openai_api
        return OpenAICompatibleAdapter(
            api_base=role_cfg.get("api_base") or section.get("api_base", "http://127.0.0.1:1234/v1"),
            api_key=role_cfg.get("api_key") or section.get("api_key", "lm-studio"),
            max_retries=role_cfg.get("max_retries", section.get("max_retries", 3)),
            retry_delay=role_cfg.get("retry_delay", section.get("retry_delay", 2.0)),
            request_timeout=role_cfg.get("request_timeout", section.get("request_timeout", 300)),
        )
    raise ValueError(f"不支持的 backend: {backend}")


def _build_role_client(role_cfg: Dict[str, Any]) -> LLMAdapter:
    """构建角色客户端；配置了 `fallback_role` 时套配额降级包装器（Round 129）。

    fallback 目标须为普通角色（自身无 fallback_role，避免链式递归）；mock 后端跳过。
    """
    client = _build_single_client(role_cfg)
    fb_role = str(role_cfg.get("fallback_role") or "").strip()
    if not fb_role:
        return client
    try:
        fb_cfg = _config.get_role_config(fb_role) or {}
    except Exception as e:
        print(f"⚠️ [Adapter] fallback 角色 {fb_role} 配置读取失败: {e}", file=sys.stderr)
        return client
    if fb_cfg.get("backend", "mock") == "mock" or fb_cfg.get("fallback_role"):
        return client
    fb_client = _build_single_client(fb_cfg)
    return FallbackRoleAdapter(
        primary=client,
        fallback=fb_client,
        fallback_model_name=fb_cfg.get("model_name", ""),
        primary_model_name=role_cfg.get("model_name", ""),
        fallback_role=fb_role,
        fallback_levels=_fallback_levels_for(
            fb_cfg.get("api_base", ""), fb_cfg.get("model_name", "")
        ),
    )


def get_llm_client(role: str) -> LLMAdapter:
    """按角色返回 LLM 客户端实例（首次按需构建并缓存）。

    当全局 llm_backend=mock 时强制 mock，并清空可能残留的非 mock 缓存。
    """
    if getattr(_config, "llm_backend", None) == "mock":
        # 避免此前 openai 客户端残留导致 dry-run 仍打真 API
        stale = [k for k, v in _client_instances.items() if not isinstance(v, MockLLMAdapter)]
        for k in stale:
            del _client_instances[k]
        role_cfg = {"backend": "mock", "default_params": {}}
        if role not in _client_instances:
            _client_instances[role] = _build_role_client(role_cfg)
        return _client_instances[role]
    if role not in _client_instances:
        role_cfg = _config.get_role_config(role)
        _client_instances[role] = _build_role_client(role_cfg)
    return _client_instances[role]


def get_llm_client_for_task(task_name: str) -> LLMAdapter:
    """根据任务名解析出角色并返回客户端。"""
    return get_llm_client(_config.get_task_role(task_name))


def get_role_model_name(role: str) -> str:
    cfg = _config.get_role_config(role)
    return cfg.get("model_name", "")


def get_task_extra_body(task_name: str) -> Dict[str, Any]:
    """读取任务级 extra_body（供应商特有透传参数）；无则返回空 dict。

    Round 127: 配置位置从模型级迁到任务级 —— 规范位置
    `task_routing.<task>.extra_body`；兼容 `params_override.extra_body`。
    """
    try:
        routing = (_config.task_routing or {}).get(task_name, {}) or {}
        body = routing.get("extra_body")
        if body is None:
            body = (routing.get("params_override") or {}).get("extra_body")
        return dict(body) if isinstance(body, dict) else {}
    except Exception:
        return {}



# ────────────────────────────────────────────────────────────────────
