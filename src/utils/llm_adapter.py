"""
SagaTrans LLM 适配层（4 个后端实现 + JSON 解析 + 客户端管理）

源：原整合脚本 §2（LLM 适配层）+ 文件头中的 robust_json_loads，拆分到独立模块。

后端支持：
  - MockLLMAdapter          离线测试 / dry-run
  - OpenAICompatibleAdapter 通用 OpenAI 兼容 HTTP（DeepSeek / MiniMax / vLLM / LM Studio 等）
  - MLXNativeAdapter        Apple Silicon 本地推理（mlx-lm）

公共 API：
  get_llm_client(role)            按角色获取 LLM 客户端
  get_llm_client_for_task(task)   按任务名路由到角色
  get_role_model_name(role)       获取角色配置中的模型名
  get_role_extra_body(role)       获取角色配置中的 extra_body
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
        """生成接口；**kwargs 用于透传子类特有参数（如 extra_body / enable_thinking）。"""
        pass

    @abstractmethod
    def unload_model(self):
        """强制释放显存的统一接口"""
        pass

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Round 27: 子类在 generate 末尾调用此方法累加真实 token 用量。

        OpenAI-compatible API 标准 usage 字段：prompt_tokens / completion_tokens / total_tokens。
        Mock / MLX 后端按各自实现做估算或 tokenizer encode 真实值。
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
        """检查内存压力"""
        usage = self.get_memory_usage()
        threshold = _config.mlx_memory.get("warning_threshold", 0.8)
        return usage["percent"] / 100 > threshold

    def auto_unload_if_needed(self) -> bool:
        """内存压力时自动卸载模型，返回是否触发了卸载"""
        auto_unload = _config.mlx_memory.get("auto_unload_on_pressure", True)
        if auto_unload and self.check_memory_pressure():
            print(f"⚠️ 内存压力过大 ({self.get_memory_usage()['percent']:.1f}%)，自动卸载模型")
            self.unload_model()
            return True
        return False


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


# Round 117: thinking 参数按 provider 归一。
# - minimax: enabled→adaptive；disabled→reasoning_effort=none + thinking=disabled
# - deepseek: enabled/disabled 显式下发
# - openai: enabled→reasoning_effort=high
# - DeepSeek 代理宿主（自建 OpenAI 兼容代理）: 仅显式开启时下发 thinking=enabled，
#   关闭时不发参数（保持既有可用行为不变，避免代理不识别字段导致 400）
_DEEPSEEK_PROXY_HOSTS = ("token.ai-hf.cn",)


def build_thinking_params(api_base: str, enable_thinking: bool) -> Dict[str, Any]:
    base = (api_base or "").lower()
    if "minimax" in base:
        if enable_thinking:
            return {"thinking": {"type": "adaptive"}}
        return {"reasoning_effort": "none", "thinking": {"type": "disabled"}}
    if "deepseek" in base:
        return {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}
    if "openai" in base:
        return {"reasoning_effort": "high"} if enable_thinking else {}
    from urllib.parse import urlparse
    host = urlparse(api_base).netloc.lower() if api_base else ""
    if host in _DEEPSEEK_PROXY_HOSTS:
        return {"thinking": {"type": "enabled"}} if enable_thinking else {}
    return {}


# Round 117b: 内联 think 块提取（用 . 匹配换行；MiniMax adaptive 模式实测会把
# 推理内联在 content 首位，如 "<think…</think>正文"）
_INLINE_THINK_RE = re.compile(r'<think(.*?)</think>', re.DOTALL)


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
        enable_thinking = kwargs.pop("enable_thinking", False)
        extra_body = kwargs.pop("extra_body", None) or {}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        # Round 117: thinking 参数按 provider 归一（含 DeepSeek 代理宿主 opt-in）
        payload.update(build_thinking_params(self.api_base, enable_thinking))
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
        thinking_stripped = False  # Round 117: 服务端 400 不识别 thinking 字段时剥离重试一次
        for attempt in range(self.max_retries):
            try:
                response = self._requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout,
                    verify=self._verify_ssl,
                )
                # 429 是限流（属于 4xx 但应重试），其他 4xx 是客户端错误；5xx 由 raise_for_status 抛出后被外层重试逻辑捕获
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    if (response.status_code == 400 and not thinking_stripped
                            and any(k in payload for k in ("thinking", "reasoning_effort"))):
                        # 代理/服务端不识别 thinking 参数：剥离后立即重试一次（保证可用性优先）
                        thinking_stripped = True
                        payload.pop("thinking", None)
                        payload.pop("reasoning_effort", None)
                        print("⚠️ [OpenAI Adapter] 400: 服务端可能不识别 thinking 参数，剥离后重试",
                              file=sys.stderr)
                        continue
                    raise RuntimeError(f"API 客户端错误 {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
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
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                # Round 117b: 部分 provider（如 MiniMax adaptive thinking）把推理内联在
                # content 里（"<think…</think>正文"）。若 reasoning_content 为空，
                # 把闭合的 think 块抽出为审计轨迹，正文只保留 think 之后的内容。
                if (not reasoning) and ("<think" in content):
                    m = _INLINE_THINK_RE.search(content)
                    if m:
                        reasoning = m.group(1).strip()
                        content = (content[:m.start()] + content[m.end():]).strip()
                self.last_reasoning = reasoning
                if not content.strip() and reasoning:
                    # 仅有推理无回复：返回推理兜底（标记以便上游觉察）
                    self.last_reasoning_only = True
                    return reasoning
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


class MLXNativeAdapter(LLMAdapter):
    """基于 Apple MLX 的原生本地推理适配器 (专为 M1 统一内存优化)"""

    # 类级共享缓存：LRU 顺序字典，最近使用移到末尾
    # 避免每个角色加载一份同模型副本，节省 2x~3x 内存
    # 限制最大缓存模型数，防止 16GB 统一内存 OOM
    _shared_models: "OrderedDict[str, tuple]" = OrderedDict()
    _cache_lock = threading.Lock()
    _MAX_CACHED_MODELS = 2  # 可通过配置覆盖

    def __init__(self):
        try:
            from mlx_lm import load as _mlx_load
            from mlx_lm import generate as _mlx_generate
            from mlx_lm.sample_utils import make_sampler as _make_sampler
            import mlx.core as _mx
            self.mlx_load = _mlx_load
            self.mlx_generate = _mlx_generate
            self.make_sampler = _make_sampler
            self.mx = _mx
        except ImportError:
            raise RuntimeError("⚠️ 环境中未安装 mlx-lm。请在虚拟环境中执行 `pip install mlx-lm`")

        self.current_model_name = None
        self.model = None
        self.tokenizer = None

    def _unload_single_model(self, model, tokenizer, model_name: str):
        """真正卸载单个模型：解引用 + Metal 缓存清理 + GC"""
        # 1. 解引用
        del model
        del tokenizer
        # 2. Python GC
        gc.collect()
        # 3. Metal 缓存清理（KV Cache、计算图）
        if hasattr(self.mx.metal, "clear_cache"):
            self.mx.metal.clear_cache()
        print(f"🧹 [MLX] 已彻底卸载模型: {model_name}，显存已释放")

    def _load_model_if_needed(self, model_name: str):
        """按需加载，LRU 缓存 + 线程安全 + 自动驱逐

        设计：
          - 多个 MLXNativeAdapter 实例（每个角色一个）共用同一份 (model, tokenizer)
          - 首次加载 → 实际 mlx_load；缓存命中 → 直接复用并更新 LRU
          - 超过 _MAX_CACHED_MODELS 时驱逐最久未使用的模型（真正释放显存）
        """
        with self._cache_lock:
            # 缓存命中：更新 LRU 顺序（移到末尾）
            if model_name in self._shared_models:
                self._shared_models.move_to_end(model_name)
                print(f"♻️  [MLX] 复用共享缓存: {model_name}（LRU 刷新）")
            else:
                # 驱逐最旧模型（如果超过限制）
                while len(self._shared_models) >= self._MAX_CACHED_MODELS:
                    oldest_name, (old_model, old_tok) = self._shared_models.popitem(last=False)
                    print(f"🗑️ [MLX] 缓存满，驱逐最旧模型: {oldest_name}")
                    self._unload_single_model(old_model, old_tok, oldest_name)

                # 加载新模型
                print(f"🚀 [MLX] 正在将模型加载至 Apple Silicon 统一内存: {model_name}...")
                self._shared_models[model_name] = self.mlx_load(model_name)

            # 绑定到当前实例
            self.model, self.tokenizer = self._shared_models[model_name]
            self.current_model_name = model_name

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        # 线程安全的加载/驱逐
        self._load_model_if_needed(model_name)

        # 组装聊天模板
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            enable_thinking = kwargs.get("enable_thinking", False)
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                # tokenizer 不支持 enable_thinking（如标准 Qwen/Gemma），回退
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            formatted_prompt = prompt

        # 性能计时
        start_time = time.time()
        mem_before = self.get_memory_usage()

        # MLX 生成调用
        sampler = self.make_sampler(temp=temperature)
        response = self.mlx_generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
            sampler=sampler,
        )

        # 性能统计
        elapsed_ms = (time.time() - start_time) * 1000
        mem_after = self.get_memory_usage()

        # Round 27: tokenizer 真实 token 数（prompt + completion）
        try:
            prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
            completion_tokens = len(self.tokenizer.encode(response))
        except Exception:
            prompt_tokens = len(formatted_prompt) // 4
            completion_tokens = len(response) // 4

        print(f"📊 [MLX Perf] 耗时: {elapsed_ms:.0f}ms, "
              f"内存: {mem_before['rss_gb']:.2f}GB -> {mem_after['rss_gb']:.2f}GB, "
              f"约 {completion_tokens} tokens, "
              f"{completion_tokens / (elapsed_ms/1000):.1f} tok/s")

        # Round 27: 累积 token 用量（与 OpenAI adapter 统一口径）
        self.record_usage(prompt_tokens, completion_tokens)

        return response

    def unload_model(self):
        """M1 环境下至关重要的显存清理操作

        从共享缓存中真正移除当前模型（若无其他实例引用），释放 Metal 显存。
        """
        with self._cache_lock:
            if self.current_model_name and self.current_model_name in self._shared_models:
                model, tokenizer = self._shared_models.pop(self.current_model_name)
                print(f"🧹 [MLX] 从共享缓存移除: {self.current_model_name}")
                self._unload_single_model(model, tokenizer, self.current_model_name)
            else:
                print(f"ℹ️ [MLX] 当前模型 {self.current_model_name} 不在共享缓存中或已被驱逐")

        self.model = None
        self.tokenizer = None
        self.current_model_name = None

        # 兜底：再次 GC + Metal 清理
        gc.collect()
        if hasattr(self.mx.metal, "clear_cache"):
            self.mx.metal.clear_cache()

        print("✅ [MLX] Metal 缓存已清空，显存已释放。")

    @classmethod
    def set_max_cached_models(cls, max_models: int):
        """运行时调整最大缓存模型数（配置热加载用）。

        用于在内存受限环境（如 16GB M1）下动态收缩 LRU 缓存上限。修改后会
        立即驱逐多余的最久未使用模型（解引用 + GC + Metal cache 清理）。

        Args:
            max_models: 新的最大缓存模型数（会被钳制到 ≥ 1）。

        Example:
            >>> MLXNativeAdapter.set_max_cached_models(1)  # 强制单模型缓存
        """
        with cls._cache_lock:
            cls._MAX_CACHED_MODELS = max(1, max_models)
            # 如果当前缓存超过新限制，驱逐多余
            while len(cls._shared_models) > cls._MAX_CACHED_MODELS:
                oldest_name, (old_model, old_tok) = cls._shared_models.popitem(last=False)
                print(f"🗑️ [MLX] 配置变更驱逐: {oldest_name}")
                # 静态清理：del + gc + Metal cache
                del old_model, old_tok
                gc.collect()
                # Metal cache 清理需要 mlx.core 单例，用 try 兜底
                try:
                    import mlx.core as mx
                    if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                        mx.metal.clear_cache()
                except (ImportError, AttributeError):
                    pass


# Per-role 客户端缓存：role_key -> LLMAdapter
_client_instances: dict = {}


def _build_role_client(role_cfg: Dict[str, Any]) -> LLMAdapter:
    backend = role_cfg.get("backend", "mock")
    if backend == "mock":
        return MockLLMAdapter()
    if backend == "mlx":
        return MLXNativeAdapter()
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


def get_llm_client(role: str) -> LLMAdapter:
    """按角色返回 LLM 客户端实例（首次按需构建并缓存）。

    当全局 llm_backend=mock 时强制 mock，并清空可能残留的非 mock 缓存。
    """
    if getattr(_config, "llm_backend", None) == "mock":
        # 避免此前 openai 客户端残留导致 dry-run 仍打真 API
        stale = [k for k, v in _client_instances.items() if not isinstance(v, MockLLMAdapter)]
        for k in stale:
            del _client_instances[k]
        role_cfg = {"backend": "mock", "default_params": {}, "extra_body": {}}
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
    return cfg.get("model_id") or cfg.get("model_name", "")


def get_role_extra_body(role: str) -> Dict[str, Any]:
    """读取角色配置中的 extra_body（供应商特有参数）；无则返回空 dict。"""
    try:
        cfg = _config.get_role_config(role) or {}
        body = cfg.get("extra_body") or {}
        return dict(body) if isinstance(body, dict) else {}
    except Exception:
        return {}



# ────────────────────────────────────────────────────────────────────
