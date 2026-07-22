#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║        OpenLiterary — AI 文学语义编译系统 (单体聚合版)           ║
║                                                                ║
║  此文件由 src/ 下所有模块拼接生成，未改动任何业务逻辑。            ║
║  用途：方便整体上传至会话式 AI（ChatGPT、Gemini 等）进行全量审计  ║
║                                                                ║
║  原始项目：translator-agent/                                     ║
║  生成日期：2026-06-24                                           ║
╚══════════════════════════════════════════════════════════════════╝

目录：
  Section 1 — 枚举与类型定义        (TaskState, DecisionLevel)
  Section 2 — LLM 适配层            (LLMAdapter, MockLLMAdapter, OpenAICompatibleAdapter, MLXNativeAdapter, get_llm_client)
  Section 3 — Scheduler 任务调度器   (TaskScheduler)
  Section 4 — Decision Engine 决策引擎 (DecisionEngine)
  Section 5 — SmartChunker 文本切分器 (SmartChunker)
  Section 6 — Agents                (ReferenceAgent, LiteraryRewriterAgent, CriticAgent, JudgeAgent)
  Section 7 — Pipeline 主流程        (TranslationPipeline)
  Section 8 — 测试工具               (GoldenSetEvaluator, MemoryPressureTests)
  Section 9 — 辅助入口               (init_project, debug_db)
  Section 10 — __main__ 统一入口
  Section 11 — 输入格式适配器         (EpubSplitter, TextSplitter, MdSplitter)
  Section 11.5 — 译后命名一致性校正   (run_consistency_check, 子命令 consistency)
  Section 12 — 统一入口点            (split_input_to_chapters, 暴露 CLI 参数)
"""

import os
import sys
import re
import json
import time
import random
import gc
import hashlib
import logging  # P4-1: 引入 logging 框架
from datetime import datetime
import sqlite3
import psutil
import threading
from collections import OrderedDict

# P4-1: 模块级 logger；通过 OPENLITERARY__LOG_LEVEL 环境变量控制
logger = logging.getLogger("openlit")
_log_level = os.environ.get("OPENLITERARY__LOG_LEVEL", "INFO").upper()
if _log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    logger.setLevel(getattr(logging, _log_level))
else:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)


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
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Tuple  # P3-1: 引入 Tuple

# 配置加载器（支持单文件模式降级）
# P0-1: 多候选探测 —— 既支持 `python -m src.translator_agent`（找到 src.config），
# 也支持 `python translator_agent.py`（单文件部署，无 src 包），降级到内建默认。
# 同时支持 OPENLITERARY__CONFIG_MODULE 环境变量显式指定外部配置模块。
_config = None
_config_source = "builtin"
try:
    import os as _os_cfg
    _ext_mod = _os_cfg.environ.get("OPENLITERARY__CONFIG_MODULE", "").strip()
    if _ext_mod:
        # 1. 显式环境变量指定的外部配置（最优先）
        import importlib as _il
        _ext = _il.import_module(_ext_mod)
        if hasattr(_ext, "get_config"):
            _config = _ext.get_config()
        elif hasattr(_ext, "config"):
            _config = _ext.config
        _config_source = f"env:{_ext_mod}"
    else:
        # 2. 探测 src 包上下文（python -m src.translator_agent）
        try:
            from src.config import get_config as _get_config_external
            _config = _get_config_external()
            _config_source = "src.config"
        except ImportError:
            # 3. 单文件部署模式：无 src 包时降级到内建默认
            raise
except ImportError:
    pass  # 走下方 _BuiltinConfig 兜底
except Exception as _cfg_err:
    print(f"[WARN] 加载外部配置失败 ({_cfg_err})，降级到内建默认配置", file=sys.stderr)

if _config is None:
    # 单文件部署模式：外部配置不可用，使用内建默认配置
    class _BuiltinConfig:
        """单体聚合版内建默认配置（仅在 src.config 不可用时启用）"""
        llm_backend = "mock"
        task_routing = {
            "reference_extraction": {"model": "primary", "params_override": {}},
            "literal_translation": {"model": "literal_translator", "params_override": {}},
            "literary_rewrite": {"model": "primary", "params_override": {}},
            "critic_scoring": {"model": "primary", "params_override": {"temperature": 0.1}},
            "judge_decision": {"model": "primary", "params_override": {}},
        }
        mlx_models = {
            "literal_translator": {"model_id": "google/gemma-2-9b-it-mlx-4bit", "default_params": {}},
            "primary": {"model_id": "qwen/Qwen2.5-7B-Instruct-MLX-4bit", "default_params": {}},
        }
        openai_models = {
            "literal_translator": {"model_name": "gpt-4o-mini", "default_params": {}},
            "primary": {"model_name": "gpt-4o-mini", "default_params": {}},
        }
        mlx_memory = {"warning_threshold": 0.8, "auto_unload_on_pressure": True}
        openai_api = {"api_base": "http://127.0.0.1:1234/v1", "api_key": "lm-studio",
                      "max_retries": 3, "retry_delay": 2.0, "request_timeout": 300,
                      "max_concurrent_requests": 4}
        pipeline = {"batch_size": 50, "max_retries": 3, "max_quality_retries": 6, "judge_safety_net_avg_min": 5.5,
                    "poll_interval": 0.5, "entity_hard_fail": True,
                    "early_stop": {"enabled": True, "max_low_dims": 3, "low_score_threshold": 3.0, "apply_only_first_round": True}}
        chunker = {"soft_limit": 1000, "hard_limit": 2500}  # P1-3: respect_scene_breaks 死配置删除
        decision_engine = {"terminology_triggers_backtrack": True,
                           "reference_triggers_backtrack": True,
                           "style_triggers_backtrack": False,
                           "max_affected_chunks_per_decision": 50,
                           "backtrack_scope": "book",
                           "context_max_terminology": 40,
                           "context_max_reference": 20,
                           "context_max_style": 10,
                           "context_max_entities": 30}
        critic_thresholds = {"fluency": 7.0, "readability": 7.0, "style_compliance": 6.5,
                            "voice_consistency": 7.0, "semantic_preservation": 7.0,
                            "average_score_min": 7.0}
        style_guide = {"avg_sentence_length": "适中、口语化",
                       "lexicon_preference": "平实、克制、英式幽默",
                       "author_priority_ratio": 0.7,
                       "genre": "儿童文学",
                       "work_type": "长篇童话"}
        # P1-1: paths 死配置已删除（grep 全文件 0 引用，配置项不生效）

        def resolve_task_model(self, task_name: str):
            routing = self.task_routing.get(task_name, {})
            model_key = routing.get("model", "primary")
            return model_key, routing.get("params_override", {})

        def _get_model_config(self, model_key: str):
            if self.llm_backend == "mlx":
                return self.mlx_models.get(model_key, {})
            if self.llm_backend == "openai_api":
                # 优先读新格式：openai_api.models.<role>
                nested = self.openai_api.get("models", {})
                if nested and model_key in nested:
                    return nested[model_key]
                # 兼容旧格式：openai_models.<role>
            return self.openai_models.get(model_key, {})

        def get_task_role(self, task_name: str):
            return self.task_routing.get(task_name, {}).get("model", "primary")

        def has_role_models(self):
            # 单文件模式不支持 per-role default_params；请用 task_routing.*.params_override 覆盖
            return False

        def role_models(self):
            return {}

        def get_role_config(self, role: str):
            legacy = self._get_model_config(role)
            if self.llm_backend == "mock":
                return {"backend": "mock", "default_params": {}}
            if self.llm_backend == "mlx":
                return {
                    "backend": "mlx",
                    "model_id": legacy.get("model_id", "qwen/Qwen2.5-7B-Instruct-MLX-4bit"),
                    "default_params": legacy.get("default_params", {}),
                }
            section = self.openai_api
            return {
                "backend": "openai_api",
                "api_base": section.get("api_base", "http://127.0.0.1:1234/v1"),
                "api_key": section.get("api_key", "lm-studio"),
                "model_name": legacy.get("model_name", "deepseek-v4-flash"),
                "default_params": legacy.get("default_params", {}),
            }

    _config = _BuiltinConfig()
    # P0-2: 降级行为显式化 —— 给出可操作指引；STRICT 模式直接退出便于 CI 捕获
    print(
        "[WARN] 未找到外部配置（src.config 或 OPENLITERARY__CONFIG_MODULE 指定模块），"
        "已降级到内建 mock 默认配置。\n"
        "      单文件部署模式下 pipeline 会使用 mock 后端跑完全程，"
        "若需真实 LLM 后端请：\n"
        "        1) 使用 `python -m src.translator_agent` 启用 src 包内的 config.py；或\n"
        "        2) 设置 OPENLITERARY__CONFIG_MODULE=<your_config_module>；或\n"
        "        3) 配置环境变量 OPENLITERARY__LLM_BACKEND 等覆盖项。",
        file=sys.stderr,
    )
    if os.environ.get("OPENLITERARY__STRICT") == "1":
        print("[ERROR] OPENLITERARY__STRICT=1 模式下拒绝降级，退出。", file=sys.stderr)
        sys.exit(2)

# ────────────────────────────────────────────────────────────────────
# Section 1 — 枚举与类型定义
# ────────────────────────────────────────────────────────────────────

class TaskState(Enum):
    """任务生命周期状态"""
    # 基础流程状态
    PENDING = "PENDING"
    EXTRACTING_TERMS = "EXTRACTING_TERMS"
    TRANSLATING_RAW = "TRANSLATING_RAW"
    REWRITING_LITERARY = "REWRITING_LITERARY"
    AUDITING = "AUDITING"
    JUDGING = "JUDGING"

    # 终态与异常态
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"           # 失败，等待重试 (retries < 3)
    PERMANENTLY_FAILED = "PERMANENTLY_FAILED"  # 终态：重试耗尽，需人工介入

    # 回溯专属状态
    DIRTY = "DIRTY"             # 因 Decision DB 变更被标记为已污染，需局部重跑


VALID_TRANSITIONS = {
    TaskState.PENDING:          [TaskState.EXTRACTING_TERMS],
    TaskState.DIRTY:            [TaskState.EXTRACTING_TERMS],
    TaskState.FAILED:           [TaskState.EXTRACTING_TERMS, TaskState.TRANSLATING_RAW,
                                 TaskState.REWRITING_LITERARY, TaskState.AUDITING,
                                 TaskState.JUDGING, TaskState.PERMANENTLY_FAILED],
    TaskState.PERMANENTLY_FAILED: [],  # 终态：不可转移
    TaskState.EXTRACTING_TERMS: [TaskState.TRANSLATING_RAW, TaskState.FAILED],
    TaskState.TRANSLATING_RAW:  [TaskState.REWRITING_LITERARY, TaskState.FAILED],
    TaskState.REWRITING_LITERARY: [TaskState.AUDITING, TaskState.FAILED],
    # AUDITING 可退回润色：实体硬失败等质量问题无需再走 Judge
    TaskState.AUDITING:         [TaskState.JUDGING, TaskState.REWRITING_LITERARY, TaskState.FAILED],
    TaskState.JUDGING:          [TaskState.COMPLETED, TaskState.REWRITING_LITERARY, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    TaskState.COMPLETED:        [TaskState.DIRTY],
}


class DecisionLevel(IntEnum):
    """决策等级"""
    TERMINOLOGY = 1  # 术语级，全局必须一致
    REFERENCE = 2    # 典故级，影响重写策略
    STYLE = 3        # 风格约束级，限制重写器词汇与句式


# ────────────────────────────────────────────────────────────────────
# Section 2 — LLM 适配层
# ────────────────────────────────────────────────────────────────────


class LLMAdapter(ABC):
    """大语言模型调用基类"""
    @abstractmethod
    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        """生成接口；**kwargs 用于透传子类特有参数（如 extra_body / enable_thinking）。"""
        pass

    @abstractmethod
    def unload_model(self):
        """强制释放显存的统一接口"""
        pass

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
        self.call_count = 0
        self._count_lock = threading.Lock()

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        with self._count_lock:
            self.call_count += 1

        # 根据 prompt 的独特标识符返回对应模拟响应（优先级：最独特的优先）
        if "考据专家" in prompt and "典故来源" in prompt:
            return self._mock_reference_response()
        elif "直译" in prompt and "字面忠实" in prompt and "不进行文学润色" in prompt:
            return self._mock_raw_translation(prompt)
        elif "荣获过星云奖和雨果奖" in prompt and "排版与脚注协议" in prompt:
            return self._mock_literary_rewrite(prompt)
        elif "极其严苛的文学编辑与翻译评论家" in prompt and "评分维度" in prompt:
            return self._mock_critic_report()
        elif "星云奖级别的终审译者" in prompt and ("决定" in prompt or "定稿" in prompt):
            return self._mock_judge_decision()
        else:
            return f"[Mock Response #{self.call_count}] 模拟输出"

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

译文经过文学润色，呈现史诗感与古典韵律。

[^1]: 译注：济慈（1795-1821），英国浪漫主义诗人。
[^2]: 译注：引自《希腊古瓮颂》。"""

    def _mock_critic_report(self) -> str:
        return json.dumps({
            "scores": {
                "fluency": 8,
                "style_compliance": 7,
                "voice_consistency": 8,
                "semantic_preservation": 8,
                "readability": 8
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


class OpenAICompatibleAdapter(LLMAdapter):
    """基于 HTTP 的 OpenAI 兼容 API 适配器 (如 LM Studio / vLLM)"""
    # 类级：按 api_base 分组的限流器，同一服务商共享限流
    # P4-6: 用 OrderedDict 实现 LRU，上限 _MAX_LIMITERS (默认 32)；超出时清理最久未用
    from collections import OrderedDict as _OD_p46
    _rate_limiters: "_OD_p46[str, dict]" = _OD_p46()  # api_base_key -> {"last": float, "lock": Lock, "interval": float}
    _MAX_LIMITERS = 32

    def __init__(self, api_base: str, api_key: str, max_retries: int = 3, retry_delay: float = 2.0, request_timeout: int = 300):
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
        if "minimaxi" in api_base_lower:
            return "minimaxi"
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
        # 根据 enable_thinking + provider 生成对应 API 参数
        api_base = self.api_base.lower()
        if "minimaxi" in api_base:
            if enable_thinking:
                payload["thinking"] = {"type": "adaptive"}
            else:
                payload["reasoning_effort"] = "none"
                payload["thinking"] = {"type": "disabled"}
        elif "deepseek" in api_base:
            payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
        elif "openai" in api_base:
            if enable_thinking:
                payload["reasoning_effort"] = "high"
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
                    timeout=self.request_timeout
                )
                # 429 是限流（属于 4xx 但应重试），其他 4xx 是客户端错误；5xx 由 raise_for_status 抛出后被外层重试逻辑捕获
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise RuntimeError(f"API 客户端错误 {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                # 兼容 think-mode provider：reasoning_content + content 双字段
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or ""
                if not content.strip() and reasoning:
                    # 仅有推理无回复：把推理作为输出返回（保留完整生成产物）
                    return reasoning
                if reasoning and content:
                    # 同时存在：用换行拼接，保留推理轨迹便于审计
                    return f"{reasoning}\n\n{content}"
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

        # 估算 token 数（粗略）
        try:
            completion_tokens = len(self.tokenizer.encode(response))
        except Exception:
            completion_tokens = len(response)

        print(f"📊 [MLX Perf] 耗时: {elapsed_ms:.0f}ms, "
              f"内存: {mem_before['rss_gb']:.2f}GB -> {mem_after['rss_gb']:.2f}GB, "
              f"约 {completion_tokens} tokens, "
              f"{completion_tokens / (elapsed_ms/1000):.1f} tok/s")

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
# Section 3 — Scheduler 任务调度器
# ────────────────────────────────────────────────────────────────────

class TaskScheduler:
    def __init__(self, db_path="db/workflow.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._db_path_str = str(self.db_path)
        self._init_db()

    @property
    def conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path_str, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    def _init_db(self):
        """初始化任务流转表"""
        cursor = self.conn.cursor()

        # 任务序列表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_tasks (
                chunk_id TEXT PRIMARY KEY,
                chapter_id TEXT,
                text_content TEXT,
                state TEXT NOT NULL,
                retries INTEGER DEFAULT 0,
                quality_retries INTEGER DEFAULT 0,
                last_error TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 存量数据库兼容迁移
        try:
            cursor.execute('ALTER TABLE chunk_tasks ADD COLUMN quality_retries INTEGER DEFAULT 0')
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        
        # 批处理支持索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_state ON chunk_tasks(state)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chapter ON chunk_tasks(chapter_id)')
        self.conn.commit()

    def init_chapter_tasks(self, chapter_id: str, chunks: list[str], force: bool = False):
        """批量注入章节任务

        Args:
            chapter_id: 章节 ID
            chunks: 切分后的文本块列表
            force: 是否覆盖已存在的 chunk（同时重置 text_content 和 retries）
        """
        cursor = self.conn.cursor()
        # 记录执行前的行数，用于准确统计 overwritten（SQLite ON CONFLICT DO UPDATE 的 rowcount 总是 1）
        if force:
            cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            count_before = cursor.fetchone()[0]
        skipped = 0
        for i, text in enumerate(chunks):
            chunk_id = f"{chapter_id}_chunk{i:03d}"
            try:
                if force:
                    # force 模式：覆盖 text_content 并完全重置（含 retries=0, quality_retries=0 让 PERMANENTLY_FAILED 任务可重新执行）
                    cursor.execute('''
                        INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            text_content = excluded.text_content,
                            state = ?,
                            last_error = NULL,
                            retries = 0,
                            quality_retries = 0
                    ''', (chunk_id, chapter_id, text, TaskState.PENDING.value, TaskState.PENDING.value))
                else:
                    cursor.execute('''
                        INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)
                        VALUES (?, ?, ?, ?)
                    ''', (chunk_id, chapter_id, text, TaskState.PENDING.value))
            except sqlite3.IntegrityError:
                skipped += 1
        # --force 时删除不在新 chunk 列表中的旧 DB 行（避免 stale 行骗过产出校验）
        if force:
            new_ids = {f"{chapter_id}_chunk{i:03d}" for i in range(len(chunks))}
            cursor.execute('SELECT chunk_id FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            old_ids = {row[0] for row in cursor.fetchall()}
            orphan_ids = old_ids - new_ids
            if orphan_ids:
                placeholders = ','.join(['?'] * len(orphan_ids))
                cursor.execute(f'DELETE FROM chunk_tasks WHERE chunk_id IN ({placeholders})', tuple(orphan_ids))
                print(f"  🧹 清理 {len(orphan_ids)} 个旧 chunk DB 行: {orphan_ids}")
        self.conn.commit()
        # 用前后 COUNT 差值计算真正被覆盖的 chunk 数
        overwritten = 0
        if force:
            cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            count_after = cursor.fetchone()[0]
            inserted = max(count_after - count_before, 0)
            overwritten = len(chunks) - inserted
        if skipped > 0:
            print(f"⚠️ 跳过 {skipped} 个已存在的 chunk（若原文已变更请使用 --force 重新初始化）")
        if force and overwritten > 0:
            print(f"🔄 --force 模式：覆盖更新 {overwritten} 个已有 chunk")
        print(f"✅ 章节 {chapter_id} 初始化完成，共 {len(chunks)} 个执行块。")

    def get_tasks_by_state(self, state: TaskState, batch_size: int = 10, chapter_id: str | None = None) -> List[Dict]:
        cursor = self.conn.cursor()
        if chapter_id:
            cursor.execute('''
                SELECT * FROM chunk_tasks
                WHERE state = ? AND chapter_id = ?
                ORDER BY chunk_id ASC LIMIT ?
            ''', (state.value, chapter_id, batch_size))
        else:
            cursor.execute('''
                SELECT * FROM chunk_tasks
                WHERE state = ?
                ORDER BY chunk_id ASC LIMIT ?
            ''', (state.value, batch_size))
        return [dict(row) for row in cursor.fetchall()]

    def update_task_state(self, chunk_id: str, new_state: TaskState, error_msg: str = None, quality_retry: bool = False):
        """更新任务状态（含转移合法性校验）"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT state FROM chunk_tasks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        if row:
            from_state = TaskState(row[0])
            allowed = VALID_TRANSITIONS.get(from_state, [])
            if new_state not in allowed:
                print(f"⚠️ [Scheduler] 非法转移 {chunk_id}: {row[0]} -> {new_state.value}")
                return
        if error_msg and quality_retry:
            cursor.execute('''
                UPDATE chunk_tasks
                SET state = ?, quality_retries = quality_retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            ''', (new_state.value, error_msg, chunk_id))
        elif error_msg:
            cursor.execute('''
                UPDATE chunk_tasks
                SET state = ?, retries = retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            ''', (new_state.value, error_msg, chunk_id))
        else:
            cursor.execute('''
                UPDATE chunk_tasks
                SET state = ?, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            ''', (new_state.value, chunk_id))
        self.conn.commit()

    def batch_update_state(self, chunk_ids: List[str], new_state: TaskState, error_msgs: Dict[str, str] = None, quality_retry: bool = False, reset_counters: bool = False):
        """批量更新任务状态 - 减少数据库往返开销

        Args:
            chunk_ids: 任务 ID 列表
            new_state: 目标状态
            error_msgs: 可选，按 chunk_id 分错误信息 {chunk_id: error_msg}
            quality_retry: 是否为质量重试（True=增 quality_retries，False=增 retries）
            reset_counters: 是否重置 retries 和 quality_retries 为 0（用于 DIRTY 回溯后重新计数）
        """
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)

        # 获取当前状态并校验转移合法性
        cursor.execute(f'''
            SELECT chunk_id, state FROM chunk_tasks WHERE chunk_id IN ({placeholders})
        ''', tuple(chunk_ids))
        current_states = {row[0]: row[1] for row in cursor.fetchall()}

        valid_ids = []
        for cid in chunk_ids:
            cur = current_states.get(cid)
            if cur is None:
                print(f"⚠️ [Scheduler] chunk_id {cid} 不存在，跳过")
                continue
            from_state = TaskState(cur)
            allowed = VALID_TRANSITIONS.get(from_state, [])
            if new_state not in allowed:
                print(f"⚠️ [Scheduler] 非法转移 {cid}: {cur} -> {new_state.value} (允许: {[s.value for s in allowed]})")
                continue
            valid_ids.append(cid)

        if not valid_ids:
            return

        # 逐条执行以保留各自的 error_msg
        for cid in valid_ids:
            msg = error_msgs.get(cid) if error_msgs else None
            if msg and quality_retry:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, quality_retries = quality_retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, msg, cid))
            elif msg:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, retries = retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, msg, cid))
            elif reset_counters:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, retries = 0, quality_retries = 0, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, cid))
            else:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, cid))
        self.conn.commit()

    def get_all_tasks_by_chapter(self, chapter_id: str) -> List[Dict]:
        """获取章节所有任务"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM chunk_tasks WHERE chapter_id = ? ORDER BY chunk_id', (chapter_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_task(self, chunk_id: str) -> Optional[Dict]:
        """获取单个任务"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM chunk_tasks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def trigger_backtrack(self, chunk_ids: List[str]):
        """回溯引擎：仅将已定稿(COMPLETED)的数据块标记为 DIRTY 重跑。

        与 VALID_TRANSITIONS 对齐：只允许 COMPLETED -> DIRTY。
        禁止对 JUDGING/AUDITING 等中间态打 DIRTY（否则会阻断 COMPLETED 并可能死循环）。
        尚未产出译文的 PENDING / EXTRACTING_TERMS 等也不回溯——后续阶段会自然读到新决策。
        """
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        # 仅 COMPLETED 可进入 DIRTY，与 VALID_TRANSITIONS[COMPLETED] 一致
        cursor.execute(f'''
            UPDATE chunk_tasks
            SET state = ?, updated_at = CURRENT_TIMESTAMP
            WHERE chunk_id IN ({placeholders})
            AND state = ?
        ''', (TaskState.DIRTY.value,) + tuple(chunk_ids) + (TaskState.COMPLETED.value,))
        self.conn.commit()
        updated = cursor.rowcount
        skipped = len(chunk_ids) - updated
        msg = f"🔄 已触发 {updated} 个数据块的回溯重构（标记 DIRTY）"
        if skipped > 0:
            msg += f"（跳过 {skipped} 个非 COMPLETED 状态）"
        print(msg + "。")

    def find_chunks_containing(
        self,
        source_text: str,
        chapter_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """全文搜索包含 source_text 的 chunk_id。

        Args:
            source_text: 要匹配的原文片段
            chapter_id: 若给定则仅扫该章；None 表示全书
            limit: 最多返回条数（None=不截断）
        """
        if not source_text:
            return []
        esc = source_text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        cursor = self.conn.cursor()
        like_pat = f"%{esc}%"
        if chapter_id:
            sql = (
                "SELECT chunk_id FROM chunk_tasks "
                "WHERE chapter_id = ? AND text_content LIKE ? ESCAPE '\\' "
                "ORDER BY chunk_id ASC"
            )
            params: list = [chapter_id, like_pat]
        else:
            sql = (
                "SELECT chunk_id FROM chunk_tasks "
                "WHERE text_content LIKE ? ESCAPE '\\' "
                "ORDER BY chunk_id ASC"
            )
            params = [like_pat]
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        cursor.execute(sql, tuple(params))
        return [row[0] for row in cursor.fetchall()]

    def get_completed_chunks(self, chapter_id: str) -> List[str]:
        """获取已完成的 chunk_id 列表"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT chunk_id FROM chunk_tasks
            WHERE chapter_id = ? AND state = ?
            ORDER BY chunk_id
        ''', (chapter_id, TaskState.COMPLETED.value))
        return [row['chunk_id'] for row in cursor.fetchall()]

    def delete_tasks(self, chunk_ids: List[str]):
        """永久删除指定任务（用于清理永久失败的任务）"""
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor.execute(f'DELETE FROM chunk_tasks WHERE chunk_id IN ({placeholders})', tuple(chunk_ids))
        self.conn.commit()


# ────────────────────────────────────────────────────────────────────
# Section 4 — Decision Engine 决策引擎
# ────────────────────────────────────────────────────────────────────

class DecisionEngine:
    def __init__(self, db_path="db/decision_db.sqlite", scheduler_factory: Optional[Callable] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._db_path_str = str(self.db_path)
        self._scheduler_factory = scheduler_factory
        self._init_tables()

    @property
    def conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path_str, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    def _init_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_db (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level INTEGER NOT NULL,
                source_key TEXT NOT NULL,
                translation TEXT NOT NULL,
                reason TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_source ON decision_db(source_key)')

        # 记录决策影响的 chunk 映射表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_impact (
                decision_id INTEGER,
                chunk_id TEXT,
                FOREIGN KEY(decision_id) REFERENCES decision_db(id),
                PRIMARY KEY (decision_id, chunk_id)
            )
        ''')

        # 实体注册表：增量构建的全书实体知识库
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,          -- 主名（中文译名）
                source_names TEXT NOT NULL,            -- JSON数组：原文出现的所有指称形式
                pronoun_gender TEXT,                   -- he/she/it/unknown
                entity_type TEXT,                      -- person/place/organization/other
                first_seen_chunk TEXT,                 -- 首次出现的 chunk_id
                confidence TEXT,                       -- high/medium/low
                pending_review INTEGER DEFAULT 0,      -- 0/1：是否有待人工/后续核查的歧义
                review_reason TEXT,                    -- 待核查原因说明
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_canonical ON entity_registry(canonical_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_first_seen ON entity_registry(first_seen_chunk)')

        # 中文异体写法列（如 爱丽丝/艾丽丝）；旧库幂等迁移
        try:
            self.conn.execute("SELECT variant_names FROM entity_registry LIMIT 1")
        except Exception:
            self.conn.execute("ALTER TABLE entity_registry ADD COLUMN variant_names TEXT")

        self.conn.commit()

        # 种子：将已知中文异体登记进 entity_registry（幂等，仅合并不覆盖）
        try:
            seed_entity_variants(self.conn)
        except Exception as e:
            print(f"  ⚠️ 实体异体种子失败: {e}", file=sys.stderr)

    def is_canonical_name(self, name: str) -> bool:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1 FROM entity_registry WHERE canonical_name = ? LIMIT 1", (name,))
            return cur.fetchone() is not None
        except Exception:
            return False

    def add_entity_variant(self, canonical: str, variant: str):
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT id, variant_names FROM entity_registry WHERE canonical_name = ?", (canonical,))
            row = cur.fetchone()
            if not row:
                return
            vid, vn_json = row
            variants = robust_json_loads(vn_json, expected_type=list) if vn_json else []
            if variant in variants:
                return
            variants.append(variant)
            cur.execute(
                "UPDATE entity_registry SET variant_names = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(variants, ensure_ascii=False), vid),
            )
            self.conn.commit()
        except Exception as e:
            print(f"  ⚠️ 登记实体异体失败 ({canonical}<-{variant}): {e}", file=sys.stderr)

    def add_decision(self, level: DecisionLevel, source: str, translation: str, reason: str, affected_chunks: List[str] = None):
        """插入或更新决策，并记录影响的 chunk。

        幂等：同一 source_key 且 translation 未变时不触发 backtrack。
        STYLE 默认不回溯（见 style_triggers_backtrack）；即使开启也不得依赖「当前 JUDGING chunk」。
        """
        cursor = self.conn.cursor()
        try:
            # 幂等检测：translation 未变则跳过 backtrack（仍可合并 impact）
            cursor.execute(
                'SELECT id, translation FROM decision_db WHERE source_key = ?',
                (source,),
            )
            existing = cursor.fetchone()
            translation_unchanged = bool(
                existing and existing[1] == translation
            )

            # 原子化 UPSERT：INSERT ... ON CONFLICT DO UPDATE
            cursor.execute('''
                INSERT INTO decision_db (level, source_key, translation, reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    level = excluded.level,
                    translation = excluded.translation,
                    reason = excluded.reason,
                    updated_at = CURRENT_TIMESTAMP
            ''', (level.value, source, translation, reason))
            decision_id = cursor.lastrowid
            
            # 获取真实的决策 ID (UPSERT 后 lastrowid 在冲突时可能为 0)
            if not decision_id:
                cursor.execute('SELECT id FROM decision_db WHERE source_key = ?', (source,))
                row = cursor.fetchone()
                decision_id = row[0] if row else None
            if existing:
                decision_id = existing[0]
            
            # 记录影响的 chunk：追加合并（不删除历史 impact）
            if affected_chunks and decision_id:
                cursor.executemany(
                    'INSERT OR IGNORE INTO decision_impact (decision_id, chunk_id) VALUES (?, ?)',
                    [(decision_id, cid) for cid in affected_chunks]
                )

            self.conn.commit()
            print(f"✅ [Decision Engine] 记录 {level.name}: {source} -> {translation}")

            if translation_unchanged:
                print(f"  ℹ️ 决策未变更 (source_key={source})，跳过回溯")
                return

            # 触发回溯：必须在 commit 之后调用，避免跨库事务不一致
            de_cfg = _config.decision_engine
            should_backtrack = False
            if level == DecisionLevel.TERMINOLOGY and de_cfg.get("terminology_triggers_backtrack", True):
                should_backtrack = True
            elif level == DecisionLevel.REFERENCE and de_cfg.get("reference_triggers_backtrack", True):
                should_backtrack = True
            elif level == DecisionLevel.STYLE and de_cfg.get("style_triggers_backtrack", False):
                # 默认 False：风格规则只累积进后续 prompt，不重跑已完成 chunk
                should_backtrack = True

            if should_backtrack and affected_chunks and decision_id:
                # 截断受影响 chunk 数量，避免单次决策雪崩式重译
                max_affected = int(de_cfg.get("max_affected_chunks_per_decision", 50) or 50)
                chunks_to_dirty = list(dict.fromkeys(affected_chunks))  # 保序去重
                if max_affected > 0 and len(chunks_to_dirty) > max_affected:
                    print(
                        f"  ⚠️ 受影响 chunk {len(chunks_to_dirty)} 个，"
                        f"截断为 max_affected_chunks_per_decision={max_affected}"
                    )
                    chunks_to_dirty = sorted(chunks_to_dirty)[:max_affected]
                self._trigger_backtrack(chunks_to_dirty)
        except Exception as e:
            print(f"❌ 决策写入失败: {e}")
            raise

    def _trigger_backtrack(self, chunk_ids: List[str]):
        """触发回溯：通过工厂函数获取 scheduler 并标记 DIRTY"""
        if not self._scheduler_factory:
            return
        try:
            scheduler = self._scheduler_factory()
            scheduler.trigger_backtrack(chunk_ids)
        except Exception as e:
            # 回溯失败是跨库一致性事件，不应静默吞掉
            raise RuntimeError(f"回溯触发失败，Decision DB 与 Workflow DB 可能不一致: {e}") from e

    def get_all_decisions(self):
        """为 Agent 提供 Prompt 上下文（全量；优先用 format_prompt_context）"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT level, source_key, translation FROM decision_db ORDER BY level ASC')
        return cursor.fetchall()

    def get_decisions_for_chunk(self, chunk_id: str):
        """获取影响特定 chunk 的所有决策"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT d.level, d.source_key, d.translation, d.reason
            FROM decision_db d
            JOIN decision_impact i ON d.id = i.decision_id
            WHERE i.chunk_id = ?
            ORDER BY d.level ASC
        ''', (chunk_id,))
        return cursor.fetchall()

    def format_prompt_context(
        self,
        source_text: str = "",
        max_terminology: Optional[int] = None,
        max_reference: Optional[int] = None,
        max_style: Optional[int] = None,
        max_entities: Optional[int] = None,
    ) -> str:
        """有界决策/实体上下文：优先收录与 source_text 命中的条目，硬上限由配置控制。

        不再把 decision_db 全量无过滤灌进 prompt。
        """
        de_cfg = _config.decision_engine if hasattr(_config, "decision_engine") else {}
        max_terminology = max_terminology if max_terminology is not None else int(de_cfg.get("context_max_terminology", 40) or 40)
        max_reference = max_reference if max_reference is not None else int(de_cfg.get("context_max_reference", 20) or 20)
        max_style = max_style if max_style is not None else int(de_cfg.get("context_max_style", 10) or 10)
        max_entities = max_entities if max_entities is not None else int(de_cfg.get("context_max_entities", 30) or 30)

        src = source_text or ""
        src_lower = src.lower()

        cursor = self.conn.cursor()
        cursor.execute("SELECT level, source_key, translation FROM decision_db ORDER BY level ASC, id DESC")
        rows = cursor.fetchall()

        terms: List[str] = []
        refs: List[str] = []
        styles: List[str] = []
        # 命中优先：先扫 hit，再扫非 hit 填满配额
        term_hits, term_rest = [], []
        ref_hits, ref_rest = [], []
        style_all = []

        for level, source, trans in rows:
            line_t = f"- 术语: '{source}' -> 必须译为 '{trans}'"
            line_r = f"- 典故: '{source}' -> 必须译为 '{trans}' (若策略为保留并加注，请生成脚注)"
            line_s = f"- 风格: {trans}"
            hit = bool(source) and (source in src or (len(source) >= 3 and source.lower() in src_lower))
            if level == DecisionLevel.TERMINOLOGY.value:
                (term_hits if hit else term_rest).append(line_t)
            elif level == DecisionLevel.REFERENCE.value:
                (ref_hits if hit else ref_rest).append(line_r)
            elif level == DecisionLevel.STYLE.value:
                style_all.append(line_s)

        terms = (term_hits + term_rest)[:max_terminology]
        refs = (ref_hits + ref_rest)[:max_reference]
        styles = style_all[:max_style]

        entity_terms: List[str] = []
        try:
            rows = query_high_confidence_entities(self.conn)
            ent_hits, ent_rest = [], []
            for row in rows:
                canonical, source_names_json, variant_names_json = row
                if not canonical:
                    continue
                clean = re.sub(r"[^\u4e00-\u9fff\w\s]", "", canonical)[:30]
                if not clean or len(clean) < 2:
                    continue
                sn_list = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
                hit = any(
                    (sn and (sn in src or sn.lower() in src_lower))
                    for sn in sn_list
                ) or (clean in src)
                line = f"- 实体: '{clean}' (源语指称: {source_names_json}) -> 必须译为 '{clean}'"
                _vn = robust_json_loads(variant_names_json, expected_type=list) if variant_names_json else []
                if _vn:
                    line += f"；禁止写法（异体）: {', '.join(_vn)}"
                (ent_hits if hit else ent_rest).append(line)
            entity_terms = (ent_hits + ent_rest)[:max_entities]
        except Exception:
            pass

        if not terms and not refs and not styles and not entity_terms:
            return "无特殊词汇约束。"

        parts = ["【全局翻译决策（必须严格遵守；已按相关度截断）】"]
        if terms:
            parts.append("\n".join(terms))
        if refs:
            parts.append("\n".join(refs))
        if styles:
            parts.append("\n".join(styles))
        if entity_terms:
            parts.append("\n【实体译名约束（高置信度，必须严格遵守）】")
            parts.append("\n".join(entity_terms))
        truncated = (
            len(term_hits) + len(term_rest) > max_terminology
            or len(ref_hits) + len(ref_rest) > max_reference
            or len(style_all) > max_style
        )
        if truncated:
            parts.append(
                f"\n（上下文已截断：terminology≤{max_terminology}, "
                f"reference≤{max_reference}, style≤{max_style}, entities≤{max_entities}）"
            )
        return "\n".join(parts)

    def set_scheduler_factory(self, factory: Callable):
        """设置调度器工厂函数（用于延迟绑定，避免循环导入）"""
        self._scheduler_factory = factory


# P3-1: entity_registry 公共查询 helpers（双变体——confidence 过滤语义不等价）
def query_high_confidence_entities(conn) -> List[Tuple[str, str]]:
    """返回高置信度且无待核查的实体 (canonical_name, source_names_json)。

    用途：prompt 上下文注入、译后一致性报告。
    等价 SQL：`WHERE confidence = 'high' AND pending_review = 0`
    """
    cursor = conn.cursor()
    cursor.execute(
        'SELECT canonical_name, source_names, variant_names FROM entity_registry '
        'WHERE confidence = "high" AND pending_review = 0'
    )
    return cursor.fetchall()


def query_reviewable_entities(conn) -> List[Tuple[str, str]]:
    """返回所有非待核查实体 (canonical_name, source_names_json, variant_names_json)。

    用途：译文核查 (Pipeline._check_entity_consistency)。
    **不过滤 confidence** —— low/medium 也要查，因为不一致检测需要更宽覆盖。
    等价 SQL：`WHERE pending_review = 0`
    """
    cursor = conn.cursor()
    cursor.execute(
        'SELECT canonical_name, source_names, variant_names FROM entity_registry '
        'WHERE pending_review = 0'
    )
    return cursor.fetchall()


# ────────────────────────────────────────────────────────────────────
# Section 5 — SmartChunker 文本切分器
# ────────────────────────────────────────────────────────────────────

# P3-2: CJK 引号字符集常量（SmartChunker / 各 Splitter 共用）
QUOTE_CHARS_OPEN = "\u201c\u2018\u300c\u300e"   # “ ‘ 「 『
QUOTE_CHARS_CLOSE = "\u201d\u2019\u300d\u300f"  # ” ’ 」 』


class SmartChunker:
    def __init__(self, soft_limit: int = None, hard_limit: int = None):
        """
        :param soft_limit: 软切分阈值（字符数）。达到此值且对话闭合时切分。
        :param hard_limit: 强制切分阈值。极端防错机制，超过此值即使对话未闭合也强制切分。
        """
        chunker_cfg = _config.chunker
        self.soft_limit = soft_limit if soft_limit is not None else chunker_cfg.get("soft_limit", 1000)
        self.hard_limit = hard_limit if hard_limit is not None else chunker_cfg.get("hard_limit", 2500)

        # 匹配 Markdown 一级边界：标题 (#) 或场景转场 (***, ---)
        self.scene_break_pattern = re.compile(r'^(#+|\*\*\*|---)\s*')

    def split_markdown(self, markdown_text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', markdown_text) if p.strip()]

        chunks = []
        current_chunk = []
        current_len = 0
        # 引号状态：>0 表示有未闭合引号（弯引号计数 + ASCII toggle 贡献）
        open_quotes = 0
        ascii_double_open = False  # ASCII " 奇偶切换
        ascii_single_open = False  # ASCII ' 奇偶切换（缩写 don't 可能误计，硬切分兜底）

        def _reset_quote_state():
            nonlocal open_quotes, ascii_double_open, ascii_single_open
            open_quotes = 0
            ascii_double_open = False
            ascii_single_open = False

        for p in paragraphs:
            # 1. 物理边界探测 (Scene-Aware)
            is_scene_break = bool(self.scene_break_pattern.match(p))

            # 如果遇到新场景或标题，且当前缓冲区有内容，立刻打包上一个 Chunk
            if is_scene_break and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
                _reset_quote_state()

            # 将当前段落加入缓冲区
            current_chunk.append(p)
            current_len += len(p)

            # 2. 对话状态探针更新
            # - ASCII " / ' ：toggle（英文直引号无开闭字形）
            # - 弯引号 / 日式引号：开 +1 / 闭 -1
            for ch in p:
                if ch == '"':
                    if ascii_double_open:
                        open_quotes = max(0, open_quotes - 1)
                        ascii_double_open = False
                    else:
                        open_quotes += 1
                        ascii_double_open = True
                elif ch == "'":
                    if ascii_single_open:
                        open_quotes = max(0, open_quotes - 1)
                        ascii_single_open = False
                    else:
                        open_quotes += 1
                        ascii_single_open = True
                elif ch in QUOTE_CHARS_OPEN:
                    open_quotes += 1
                elif ch in QUOTE_CHARS_CLOSE:
                    open_quotes = max(0, open_quotes - 1)

            # 3. 软硬边界触发逻辑
            if not is_scene_break:
                # 软边界：达到字数且无未闭合引号
                if current_len >= self.soft_limit and open_quotes == 0:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    _reset_quote_state()
                # 硬边界：字数超限
                elif current_len >= self.hard_limit:
                    print(f"⚠️ 触发硬切分保护 (长度: {current_len})，可能存在未闭合引号。")
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    _reset_quote_state()

        # 收尾
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


# ────────────────────────────────────────────────────────────────────
# Section 6 — Agents
# ────────────────────────────────────────────────────────────────────

# ----- 6.1 ReferenceAgent -----

class ReferenceAgent:
    def __init__(self, decision_engine: DecisionEngine, chapter_id: str = None):
        self.llm = get_llm_client_for_task("reference_extraction")
        self.role = _config.get_task_role("reference_extraction")
        self.db = decision_engine
        self.chapter_id = chapter_id
        # 获取 scheduler 用于全库扫描
        self._scheduler = decision_engine._scheduler_factory() if decision_engine._scheduler_factory else None

    def _find_chunks_containing(self, source_text: str) -> List[str]:
        """从 workflow DB 全文搜索包含 source_text 的 chunk_id。

        范围由 decision_engine.backtrack_scope 控制：
          - book（默认）：全书扫描
          - chapter：仅当前章节
        """
        if not self._scheduler:
            return []
        try:
            de_cfg = _config.decision_engine if hasattr(_config, "decision_engine") else {}
            scope = (de_cfg.get("backtrack_scope") or "book").lower()
            max_aff = int(de_cfg.get("max_affected_chunks_per_decision", 50) or 50)
            chapter_id = self.chapter_id if scope == "chapter" else None
            if scope == "chapter" and not self.chapter_id:
                return []
            return self._scheduler.find_chunks_containing(
                source_text, chapter_id=chapter_id, limit=max_aff if max_aff > 0 else None
            )
        except Exception as e:
            print(f"⚠️ [ReferenceAgent] 扫描受影响 chunk 失败: {e}")
            return []

    def _build_prompt(self, text_chunk: str) -> str:
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        work_type = (_config.style_guide or {}).get("work_type") or "长篇小说"
        return f"""你是一位精通西方文学、历史、神话和宗教的资深翻译考据专家。
你的任务是扫描给定的{genre}（{work_type}）片段，完成两项并行任务：
1. 识别其中的文学典故、宗教隐喻、神话引用或历史名词，并制定翻译策略。
2. 识别本片段中出现的人物/地点实体，建立实体档案。

【严格定义】
不要提取普通的角色名字（如 Paul）或普通地名，除非它们具有明显的象征意义或典故来源。

【翻译策略池】
对于识别出的典故，你必须从以下策略中选择一种：
1. "RETAIN_AND_ANNOTATE" (保留音译/直译，并在后续生成脚注)
2. "CULTURAL_EQUIVALENT" (寻找目标语言中的等效文化意象)
3. "LITERAL_TRANSLATION" (仅作字面翻译，放弃深层隐喻)

【实体识别要求】
对于识别出的人物/地点实体，必须提供以下字段：
- canonical_name: 主名（中文译名，统一规范）
- source_names: 原文中出现的所有指称形式（本名、昵称、"the captain"这类称谓、代词轨迹），JSON数组
- pronoun_gender: 代词性别
- entity_type: 实体类型
- confidence: 把握程度

【输出格式】
必须输出纯 JSON 格式，不要包含任何 Markdown 代码块标记（如 ```json）。
Schema 如下：
{{
  "references": [
    {{
      "source_text": "原文中的词汇或短语",
      "allusion_target": "该词汇背后的真实典故来源（如：济慈的长诗《拉米亚》）",
      "strategy": "翻译策略池中的一项",
      "translated_term": "你建议的最终译法",
      "reason": "为什么采用此种译法？（必须详细说明考据理由）"
    }}
  ],
  "entities": [
    {{
      "canonical_name": "主名（中文译名，统一规范）",
      "source_names": ["原文中的指称形式1", "指称形式2"],
      "pronoun_gender": "he/she/it/unknown",
      "entity_type": "person/place/organization/other",
      "confidence": "high/medium/low"
    }}
  ]
}}

【待分析文本】
{text_chunk}
"""

    def _parse_and_clean_json(self, raw_response: str) -> Dict[str, Any]:
        """防御性 JSON 解析器：处理 LLM 可能输出的多种格式违规"""
        result = robust_json_loads(raw_response)
        if not result:
            print(f"⚠️ JSON 解析失败，模型输出格式违规")
            print(f"原始内容预览: {raw_response[:200]}...")
        return result if result else {"references": []}

    def process_chunk(self, chunk_id: str, text_chunk: str, affected_chunks: List[str] = None):
        """处理单个文本块，识别典故并自动写入决策引擎，同时登记实体到注册表"""
        print(f"🔍 [Reference Agent] 正在考据数据块: {chunk_id}...")

        # 1. 调用 LLM (使用配置路由)
        prompt = self._build_prompt(text_chunk)
        model_key, params = _config.resolve_task_model("reference_extraction")
        model_name = get_role_model_name(self.role) or model_key
        extra_body = get_role_extra_body(self.role)

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
            extra_body=extra_body,
        )

        # 2. 解析 JSON
        parsed_data = self._parse_and_clean_json(raw_output)
        references = parsed_data.get("references", [])
        entities = parsed_data.get("entities", [])

        if not references and not entities:
            print("  未发现深度典故或实体。")
            return

        # 3. 将决策写入 Decision DB (Level 2: 典故)
        for ref in references:
            source = ref.get("source_text")
            translation = ref.get("translated_term")
            strategy = ref.get("strategy")
            reason = f"【来源】{ref.get('allusion_target')} | 【策略】{strategy} | 【依据】{ref.get('reason')}"

            if source and translation:
                # 扫描全章所有包含该 source_text 的 chunk
                all_affected = self._find_chunks_containing(source)
                if affected_chunks:
                    all_affected = list(set(all_affected) | set(affected_chunks))

                self.db.add_decision(
                    level=DecisionLevel.REFERENCE,
                    source=source,
                    translation=translation,
                    reason=reason,
                    affected_chunks=all_affected or [chunk_id]
                )

        # 4. 登记实体到 entity_registry (增量构建全书实体知识库)
        if entities:
            self._register_entities(chunk_id, entities)

    def _register_entities(self, chunk_id: str, entities: List[Dict]):
        """将识别到的实体增量登记到 entity_registry 表"""
        if not entities:
            return
        
        cursor = self.db.conn.cursor()
        
        for ent in entities:
            canonical = ent.get("canonical_name")
            source_names = ent.get("source_names", [])
            pronoun_gender = ent.get("pronoun_gender", "unknown")
            entity_type = ent.get("entity_type", "other")
            confidence = ent.get("confidence", "low")
            
            if not canonical or not source_names:
                continue
            
            # 确保 source_names 是列表
            if isinstance(source_names, str):
                source_names = [source_names]
            
            # 过滤代词/停用词；Round 18 审计修复：移除 `len(sn) >= 4` 长度阈值
            # 原因：
            #   1. 长度阈值导致所有 <4 字符的英文原名（如 3 字符人名 "Joe"/"Kim"/"Ana"）
            #      永远无法进入 entity_registry.source_names
            #   2. Section 11.5 的 english_residue 检测依赖 source_names 列表，
            #      短英文原名的残留检测因此"天生失效"（Round 18 问题 1）
            #   3. 短名字的"子串误匹配"风险由调用方处理：
            #      - _detect_issues 已用 `\b` 词边界匹配（L4191-4193）
            #      - format_prompt_context 已用 `(sn in src or sn.lower() in src_lower)`
            #      停用词表本身已能排除代词/冠词/介词的误判
            _STOPWORDS = {
                "a", "an", "the", "and", "or", "but", "in", "on", "at",
                "to", "for", "of", "by", "with", "from", "as", "is", "was",
                "he", "she", "it", "they", "we", "you", "i", "me", "him",
                "her", "his", "my", "your", "its", "our", "their", "them",
                "us", "this", "that", "these", "those", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "can", "could", "shall", "should", "may", "might", "not",
            }
            source_names = [sn for sn in source_names if sn.lower() not in _STOPWORDS]
            if not source_names:
                continue
            
            source_names_json = json.dumps(source_names, ensure_ascii=False)
            
            # 查找是否已存在同一 canonical_name
            cursor.execute('SELECT id, source_names, pending_review FROM entity_registry WHERE canonical_name = ?', (canonical,))
            row = cursor.fetchone()
            
            if row:
                # 已存在：合并 source_names
                existing_id, existing_names_json, pending_review = row
                # P2-1: 统一走 robust_json_loads
                existing_names = robust_json_loads(existing_names_json, expected_type=list) if existing_names_json else []
                
                # 合并新旧 source_names（去重）
                merged_names = list(set(existing_names + source_names))
                merged_names_json = json.dumps(merged_names, ensure_ascii=False)
                
                # 合并置信度：取较高者
                confidence_order = {"high": 3, "medium": 2, "low": 1}
                new_conf_level = confidence_order.get(confidence, 1)
                cursor.execute('SELECT confidence FROM entity_registry WHERE id = ?', (existing_id,))
                current_row = cursor.fetchone()
                if current_row:
                    current_conf = confidence_order.get(current_row[0], 1)
                    final_conf = confidence if new_conf_level > current_conf else current_row[0]
                else:
                    final_conf = confidence
                
                cursor.execute('''
                    UPDATE entity_registry 
                    SET source_names = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (merged_names_json, final_conf, existing_id))
                print(f"  📝 实体已存在，合并指称: {canonical} <- {source_names}")
            else:
                # 检查是否有语义相近的已有实体（简单字符串包含/编辑距离初筛）
                cursor.execute('SELECT id, canonical_name, source_names FROM entity_registry')
                similar_found = False
                for existing_row in cursor.fetchall():
                    eid, ecanon, enames_json = existing_row
                    # P2-1: 统一走 robust_json_loads
                    enames = robust_json_loads(enames_json, expected_type=list) if enames_json else []
                    
                    # 双向子串包含 + 编辑距离(≤2) 初筛
                    # 注：source_names 已在 _register_entities 入口过滤停用词
                    substr_match = any(
                        sn in existing_name or existing_name in sn
                        for existing_name in enames
                        for sn in source_names
                    )
                    fuzzy_match = any(
                        _levenshtein_distance(sn, existing_name) <= 2
                        for existing_name in enames
                        for sn in source_names
                    )
                    if substr_match or fuzzy_match:
                        # 发现潜在同一实体的不同译名
                        # 不自动合并，标记待核查
                        cursor.execute('''
                            UPDATE entity_registry 
                            SET pending_review = 1, review_reason = ?
                            WHERE id = ?
                        ''', (f"称谓差异大: 新实体 {canonical} 与现有 {ecanon} 可能指向同一对象", eid))
                        similar_found = True
                        print(f"  ⚠️ 发现潜在同一实体不同译名: {canonical} vs {ecanon}，标记待核查")
                        break
                
                if not similar_found:
                    # 新建条目
                    cursor.execute('''
                        INSERT INTO entity_registry (canonical_name, source_names, pronoun_gender, entity_type, first_seen_chunk, confidence, pending_review)
                        VALUES (?, ?, ?, ?, ?, ?, 0)
                    ''', (canonical, source_names_json, pronoun_gender, entity_type, chunk_id, confidence))
                    print(f"  ✅ 新实体登记: {canonical} ({entity_type}, {confidence})")
        
        self.db.conn.commit()

class LiteraryRewriterAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client_for_task("literary_rewrite")
        self.role = _config.get_task_role("literary_rewrite")
        self.db = decision_engine
        # 加载 Few-shot 示例
        self._few_shot_examples = self._load_few_shot_examples()

    def _build_decision_context(self, source_text: str = "") -> str:
        """有界决策上下文（委托 DecisionEngine.format_prompt_context）。"""
        return self.db.format_prompt_context(source_text=source_text)

    def _load_few_shot_examples(self) -> str:
        """加载 Few-shot 示例，注入 Prompt 作为参考"""
        try:
            root_dir = Path(__file__).resolve().parent.parent
            few_shot_file = root_dir / "docs" / "few_shot_examples.json"
            with open(few_shot_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            examples = data.get("few_shot_examples", [])
            if not examples:
                return ""
            
            parts = ["【Few-shot 示例：以下为优秀润色范例，请模仿其风格、脚注格式、代词处理】\n"]
            for ex in examples[:4]:  # 最多注入 4 个示例
                parts.append(f"【示例 {ex['id']}】")
                parts.append(f"【直译底稿】\n{ex['raw_translation']}\n")
                parts.append(f"【优秀润色】\n{ex['literary_translation']}\n")
                if "edits" in ex and ex["edits"]:
                    parts.append("【修改要点】")
                    for edit in ex["edits"]:
                        parts.append(f"  - {edit['location']}: \"{edit['before']}\" -> \"{edit['after']}\" ({edit['reason']})")
                parts.append("")  # 空行分隔
            
            parts.append("———\n参考上述示例的脚注格式、术语一致性与人称代词处理即可；文风须以本任务的【风格指南】为准，不得套用示例的文言/史诗体。\n")
            return "\n".join(parts)
        except Exception:
            return ""

    def _build_prompt(
        self,
        raw_translation: str,
        decisions_context: str,
        style_guide: str,
        narrative_memory: str = "",
    ) -> str:
        few_shot = self._few_shot_examples
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        mem_block = ""
        if narrative_memory:
            mem_block = f"\n【前文摘要（保持声口与指代连贯）】\n{narrative_memory}\n"
        return f"""你是一位荣获过星云奖和雨果奖的资深{genre}文学译者。
你的任务是对提供的【直译底稿】进行最高水准的文学润色。

{decisions_context}
{mem_block}
{few_shot}

【风格基准 (Style Guide)】
{style_guide}

【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. P4 优化：对于卡罗尔《爱丽丝梦游仙境》这类儿童文学，**禁止在润色阶段添加脚注**。
3. 如遇需要注释的典故，应在首次出现时用文内括号简注（如：渡渡鸟（一种已灭绝的鸟）），而非使用脚注。
4. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。

【人称代词规则】
1. 严格遵循原文人称代词性别：he/him/his → "他"，she/her → "她"，it/its → "它"。
2. 注意原文中人物对话和叙述视角的代词指代关系，不得混淆角色性别。

【直译底稿】
{raw_translation}

请直接输出润色后的 Markdown 文本，不要包含任何多余的开头问候或解释：
"""

    def _build_retry_prompt(
        self,
        raw_translation: str,
        prev_lit_text: str,
        decisions_context: str,
        style_guide: str,
        reject_reason: str,
        critic_feedback: Optional[str],
        retry_count: int,
        all_feedback: Optional[List[Dict]] = None,
        specific_edits: Optional[List[Dict]] = None,
        register_target: str = "",
        low_dims: Optional[List[str]] = None,
    ) -> str:
        critic_section = ""
        if critic_feedback:
            # L 方案：将改进建议提升为"必执行顶层指令"，避免 rewriter 只盯 specific_edits 忽略结构性抱怨
            critic_section = (
                "\n【Critic 总体改进建议（必须执行，非可选参考）】\n"
                f"{critic_feedback}\n"
                "以上建议为终审拒绝的核心原因，必须在本次修订中**整体落实**，"
                "尤其涉及语域/文言/四字格/脚注密度的全局性问题须全篇处理。\n"
            )

        specific_edits_section = ""
        if specific_edits:
            edits_lines = []
            for i, edit in enumerate(specific_edits, 1):
                loc = edit.get("location", "?")
                orig = (edit.get("original", "") or "")[:30]
                issue = edit.get("issue", "")
                direction = (edit.get("suggested_direction", "") or "")[:50]
                edits_lines.append(f"  {i}. [{loc}] \"{orig}\" → {issue}: {direction}")
            specific_edits_section = (
                "\n【Critic 结构化修改指令（必须逐条处理，不得只改一处）】\n"
                + "\n".join(edits_lines) + "\n\n"
                # M 方案：强制逐条回应全部 specific_edits，禁止"只改一处"的打勾式空转
                "【强制要求】你必须在这一轮中**逐条处理以上全部 "
                f"{len(specific_edits)} 条修改指令**，不得只挑选其中 1 条。"
                "输出润色稿之后，另起一行用 ```json 包裹输出每条的处理结果：\n"
                '```json\n'
                '[{"index": 1, "status": "resolved|partial|skipped", "reason": "..."}, '
                '{"index": 2, "status": "resolved|partial|skipped", "reason": "..."}]\n'
                '```\n'
                f"共 {len(specific_edits)} 条，每条对应一个 index，缺一不可。\n"
            )

        history_section = ""
        if all_feedback and len(all_feedback) > 1:
            lines = []
            for i, fb in enumerate(all_feedback[:-1]):
                reason_text = fb.get("reason", "")[:80]
                tag = "已修复" if fb.get("resolved", False) else "需持续关注"
                if reason_text.strip():
                    lines.append(f"  第 {i+1} 轮 ({tag}): {reason_text}")
            if lines:
                history_section = (
                    "\n【过去所有被拒原因 — 需确认全部已解决】\n"
                    + "\n".join(lines) + "\n"
                )

        # L 方案：当风格/音色维度持续不达标，强制"全局重润色"而非局部词换
        style_dims = {"style_compliance", "voice_consistency"}
        low_style = [d for d in (low_dims or []) if d in style_dims]
        if low_style and retry_count >= 2:
            strategy = (
                "本轮回退到**基于直译底稿的整体重润色**：必须以【本章目标语域】为锚点，"
                "重新组织全篇的断句、节奏、语域与脚注，而非仅替换个别词语。"
                "上一版被拒的根因是全局风格/音色偏离，单点词换无法解决，必须结构性重做。"
            )
        elif retry_count <= 2:
            strategy = (
                "请在保留上一版优点的基础上**针对性修改**，只动被拒原因相关的段落，"
                "不要推倒重来。"
            )
        elif retry_count <= 4:
            strategy = (
                "你可以**重写 1-2 个段落**来系统性修复被拒原因，"
                "其他部分保持稳定。注意保持整体风格的统一。"
            )
        else:
            strategy = (
                "本轮允许**基于直译底稿重新润色全部内容**，"
                "但必须避免自第一轮以来所有被拒原因列出的问题。"
            )

        register_block = ""
        if register_target:
            register_block = (
                f"【本章目标语域（必须遵循，重写时以此为准）】\n{register_target}\n"
            )

        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        return f"""你是一位荣获过星云奖和雨果奖的资深{genre}文学译者。
这是第 {retry_count + 1} 次修订。上一版译文因以下原因被终审拒绝。

{decisions_context}

{strategy}

{register_block}

【上一版被拒原因（必须解决）】
{reject_reason}{critic_section}
{history_section}
{specific_edits_section}

【风格基准 (Style Guide)】
{style_guide}

【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. P4 优化：对于卡罗尔《爱丽丝梦游仙境》这类儿童文学，**禁止在润色阶段添加脚注**。
3. 如遇需要注释的典故，应在首次出现时用文内括号简注（如：渡渡鸟（一种已灭绝的鸟）），而非使用脚注。
4. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。

【人称代词规则】
1. 严格遵循原文人称代词性别：he/him/his → "他"，she/her → "她"，it/its → "它"。
2. 注意原文中人物对话和叙述视角的代词指代关系，不得混淆角色性别。

【直译底稿（语义基准，不可偏离）】
{raw_translation}

【上一版译文（在此基础上精修，只动被拒原因相关段落）】
{prev_lit_text}

请直接输出修订后的 Markdown 文本，重点解决被拒原因，其他部分保持稳定：
"""

    def _calculate_retry_temperature(self, base_temperature: float, retry_count: int) -> float:
        if retry_count <= 2:
            return min(base_temperature + retry_count * 0.1, 0.8)
        elif retry_count <= 4:
            return min(base_temperature + 0.3, 0.9)
        else:
            return min(base_temperature + 0.5, 1.0)

    def _infer_author_priority_ratio(self, source_text: str) -> float:
        """
        动态推断 Author_Priority_Ratio (0-1)

        基于原文特征决定翻译时对原著风格的保留程度：
        - 1.0 = 完全保留作者风格（直译倾向）
        - 0.0 = 完全按译者风格重写（意译倾向）

        影响因子：
        1. 典故/引用密度：越高越倾向保留原文风格
        2. 诗歌/韵文比例：越高越倾向保留
        3. 专有名词密度：越高越倾向保留
        4. 叙事视角稳定性：不稳定时降低优先级
        5. 情感强度：高情感段落保留作者音色
        """
        base_ratio = 0.7

        # 1. 典故/引用检测（英文原文使用英文作者名）
        allusion_markers = ['Keats', 'Shakespeare', 'Milton', 'Dante', 'Homer', 'Shelley', 'Byron']
        allusion_count = sum(source_text.count(m) for m in allusion_markers)
        allusion_density = min(allusion_count / (len(source_text) / 1000), 1.0)

        # 2. 诗歌/韵文特征（P3 优化：为卡罗尔诗歌添加更多标记）
        poetry_markers = ['\n\n', '——', '...', 'beauty is truth', 'truth beauty', 
                         'How doth', 'How cheerful', 'I am older', 'Who am I',
                         'Curiouser', 'said the', 'replied the']
        poetry_score = sum(1 for m in poetry_markers if m in source_text)
        # 额外检测：换行符密度（诗歌通常有更多换行）
        newline_density = source_text.count('\n') / max(len(source_text) / 100, 1)
        if newline_density > 0.5:
            poetry_score += 1

        # 3. 专有名词密度
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', source_text)
        proper_noun_density = min(len(proper_nouns) / (len(source_text) / 1000), 2.0)

        # 4. 叙事视角标记（英文原文使用英文代词）
        pov_markers = [' I ', ' my ', ' we ', ' he ', ' she ', ' it ']
        pov_changes = sum(source_text.count(m) for m in pov_markers)

        # 5. 情感词汇
        emotion_words = [' pain', ' sorrow', ' anger', ' joy', ' love', ' hate', ' fear', ' despair', ' hope', ' dream', ' soul', ' grief', ' rage']
        emotion_count = sum(source_text.count(w) for w in emotion_words)
        emotion_density = min(emotion_count / (len(source_text) / 1000), 1.0)

        # 动态调整
        if allusion_density > 0.5:
            base_ratio += 0.15
        elif allusion_density > 0.2:
            base_ratio += 0.05

        if poetry_score > 0:
            base_ratio += 0.1

        if proper_noun_density > 1.0:
            base_ratio += 0.05

        if pov_changes > 20:
            base_ratio -= 0.1

        if emotion_density > 0.5:
            base_ratio += 0.1

        return max(0.3, min(0.9, base_ratio))

    def _infer_register_target(self, source_text: str) -> str:
        """
        推断本章目标语域方向（N 方案：章节级 register 信号）。

        用于同时指导首轮润色与 retry：明确告诉 Rewriter 本章应"加文学度"
        还是"减文学度"，避免 ch02/13（欠润色）与 ch04 等（过润色/文言）拿
        到同质指令后随机摆动、永不收敛。

        判定逻辑：检测原文中"需要克制平实"的信号（科学论述、对话自述、
        维多利亚口语标记）与"可适度文学化"的信号（强意象/诗意描写）。
        二者平衡时给出中性指令。
        """
        if not source_text:
            return "保持克制、平实、贴近原文的维多利亚叙事语域；避免过度意译、文言虚词与四字格堆砌。"
        low = 0.0
        high = 0.0
        # 克制信号：科学/论述/自述语气
        plain_markers = ['I suppose', 'I think', 'you see', 'of course', 'rather',
                         'however', 'perhaps', 'I mean', 'you know', 'the fact is',
                         'it seems', 'I confess', 'we were', 'I was']
        low += sum(source_text.count(m) for m in plain_markers)
        # 过文学风险信号：密集形容词/诗化意象（原文侧近似）
        literary_markers = ['glowing', 'shimmering', 'gleamed', 'silence', 'vast',
                            'shadow', 'wonder', 'marvellous', 'strange beauty',
                            'soft', 'dim', 'ghost']
        high += sum(source_text.count(m) for m in literary_markers)
        # 对话密集 → 偏口语克制
        if source_text.count('"') > 6:
            low += 1.5
        if high > low + 1.5:
            return ("本章原文意象/描写密度高，可适度文学化，但仍须以维多利亚克制语域为底；"
                    "禁止文言虚词（之/乃/遂/盖）、四字格堆砌与超过 3 条脚注。")
        if low >= high + 1.0:
            return ("本章为叙述者口语化自述/科学论述，目标语域=克制平实、贴近原文的维多利亚闲谈；"
                    "在直译底稿基础上做文学润色（断句、节奏、语气词），但不得退化为直译稿，"
                    "也不得文言化；脚注不超过 3 条。")
        return ("保持克制、平实、贴近原文的维多利亚叙事语域；避免过度意译、文言虚词与四字格堆砌；脚注不超过 3 条。")

    def _build_style_guide(self, style_guide_stats: dict, source_text: str = "") -> str:
        """构建风格指南，包含动态 Author_Priority_Ratio 与章节级 register 目标（N 方案）"""
        if source_text:
            author_priority = self._infer_author_priority_ratio(source_text)
        else:
            author_priority = style_guide_stats.get('author_priority_ratio', 0.7)
        register_target = self._infer_register_target(source_text) if source_text else \
            "保持克制、平实、贴近原文的维多利亚叙事语域；避免过度意译、文言虚词与四字格堆砌。"

        if author_priority >= 0.8:
            priority_instruction = (
                "【高作者优先级模式】严格保留原著句式节奏、修辞手法、词汇选择。"
                "即使中文表达略显生硬，也要优先还原作者的原意与音色。"
            )
        elif author_priority >= 0.6:
            priority_instruction = (
                "【平衡模式】在保持作者核心风格的前提下，适度调整句式使其符合中文阅读习惯。"
                "保留关键修辞、典故处理，非核心处可本地化。"
            )
        else:
            priority_instruction = (
                "【译者主导模式】以目标语言的自然流畅为首要目标。"
                "大胆重组句式、替换词汇，只保留核心语义与关键意象。"
            )

        style_guide = (
            f"请模仿以下风格特征：平均句长偏向 {style_guide_stats.get('avg_sentence_length', '中等')}，"
            f"词汇倾向 {style_guide_stats.get('lexicon_preference', '文学化')}。\n"
            f"作者优先级比率: {author_priority:.2f} (0=译者主导, 1=作者主导)\n"
            f"{priority_instruction}\n"
            f"【本章目标语域（必须遵循）】{register_target}"
        )
        return style_guide

    def process_chunk(
        self,
        chunk_id: str,
        raw_translation: str,
        style_guide_stats: dict,
        source_text: str = "",
        prev_lit_text: Optional[str] = None,
        reject_reason: Optional[str] = None,
        critic_feedback: Optional[str] = None,
        retry_count: int = 0,
        all_feedback: Optional[List[Dict]] = None,
        specific_edits: Optional[List[Dict]] = None,
        narrative_memory: str = "",
        low_dims: Optional[List[str]] = None,
    ):
        print(f"✍️ [Rewriter Agent] 正在进行文学润色: {chunk_id} (retry={retry_count})...")

        style_guide = self._build_style_guide(style_guide_stats, source_text)
        decisions_context = self._build_decision_context(source_text)
        # N 方案：章节级 register 目标，作为 retry 的方向锚点
        register_target = self._infer_register_target(source_text) if source_text else ""

        # 使用配置路由：literary_rewrite（先于 temperature 计算，确保 base 从 config 读取）
        model_key, params = _config.resolve_task_model("literary_rewrite")
        model_name = get_role_model_name(self.role) or model_key
        base_temp = params.get("temperature", 0.3)

        if retry_count > 0 and prev_lit_text:
            if not reject_reason:
                reject_reason = "终审未提供具体拒因，请结合历史反馈与风格基准进行全面审视与精修。"
            prompt = self._build_retry_prompt(
                raw_translation=raw_translation,
                prev_lit_text=prev_lit_text,
                decisions_context=decisions_context,
                style_guide=style_guide,
                reject_reason=reject_reason,
                critic_feedback=critic_feedback,
                retry_count=retry_count,
                all_feedback=all_feedback,
                specific_edits=specific_edits,
                register_target=register_target,
                low_dims=low_dims,
            )
            temperature = self._calculate_retry_temperature(base_temp, retry_count)
        else:
            prompt = self._build_prompt(
                raw_translation, decisions_context, style_guide, narrative_memory=narrative_memory
            )
            temperature = base_temp

        # 覆盖 temperature（用计算后的值替换 params 中的配置值）
        params = {**params, "temperature": temperature}
        extra_body = get_role_extra_body(self.role)

        final_markdown = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
            extra_body=extra_body,
        )

        # 解析自评 JSON（仅 retry 轮）
        self_review = None
        clean_text = final_markdown
        if retry_count > 0 and specific_edits:
            # P3-3: 使用顶层 re（删除冗余 `import re as _re`）
            # 匹配末尾的 ```json ... ``` 块
            m = re.search(r'```json\s*(.+?)\s*```', final_markdown, re.DOTALL)
            if m:
                raw_block = m.group(1).strip()
                # P2-1: 统一走 robust_json_loads，先按 list 解析（自评是数组），失败回退 dict
                parsed = robust_json_loads(raw_block, expected_type=list)
                if not parsed:
                    parsed_dict = robust_json_loads(raw_block, expected_type=dict)
                    parsed = [parsed_dict] if parsed_dict else None
                if parsed:
                    self_review = parsed
                    clean_text = final_markdown[:m.start()].rstrip()
                else:
                    # NDJSON 兜底
                    try:
                        ndjson_items = []
                        for line in raw_block.splitlines():
                            line = line.strip()
                            if line:
                                # P2-1: NDJSON 行用 robust_json_loads 抗畸形
                                obj = robust_json_loads(line, expected_type=dict)
                                if obj:
                                    ndjson_items.append(obj)
                        if ndjson_items:
                            self_review = ndjson_items
                            clean_text = final_markdown[:m.start()].rstrip()
                    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
                        pass
            if self_review is None:
                print(f"  ⚠️ [Rewriter] {chunk_id} 自评 JSON 解析失败，降级为无自评模式")

        # 返回 (润色正文, 自评) 元组，由调用方保存到 metadata
        return clean_text, self_review


# ----- 6.3 CriticAgent -----

# 注意：阈值配置从 _config.critic_thresholds 读取，而非此处常量
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
    "semantic_preservation": 8
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

    def process_chunk(self, chunk_id: str, source_text: str, raw_trans: str, lit_trans: str, style_guide: dict) -> Dict[str, Any]:
        print(f"🧐 [Critic Agent] 正在对 {chunk_id} 进行多维度文学审计...")
        prompt = self._build_prompt(source_text, raw_trans, lit_trans, style_guide)

        # 使用配置路由：critic_scoring
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

        # 向后兼容：specific_edits 缺失时默认 []
        if "specific_edits" not in result:
            result["specific_edits"] = []
        elif len(result["specific_edits"]) > 20:
            # 防止 enriched_feedback 膨胀
            result["specific_edits"] = result["specific_edits"][:20]

        return result


# ----- 6.4 JudgeAgent -----

class JudgeAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client_for_task("judge_decision")
        self.role = _config.get_task_role("judge_decision")
        self.db = decision_engine

    def _build_prompt(self, source_text: str, lit_trans: str, critic_report: dict, rewriter_self_review: Optional[list] = None) -> str:
        # P0: 将 critic_thresholds 注入为参考阈值（非硬性否决条件）
        thresholds = _config.critic_thresholds
        thresholds_lines = "\n".join(
            f"  - {dim}: 参考阈值 {v}"
            for dim, v in thresholds.items() if dim != "average_score_min"
        )
        thresholds_block = f"""
【评分参考阈值（供综合判断，不是硬性否决条件）】
审辩者各维度的参考阈值为：
{thresholds_lines}

如果审辩者报告的 scores 中某维度略低于阈值，但你认为整体质量达标，可以判 PASS。
如果多个维度显著低于阈值，且润色稿确实问题明显，才判 REJECT。
"""

        self_review_block = ""
        specific_edits = critic_report.get("specific_edits", [])
        if specific_edits:
            self_review_block = "\n【Rewriter 自评报告】\n"
            if rewriter_self_review:
                for item in rewriter_self_review:
                    idx = item.get("index", "?")
                    status = item.get("status", "?")
                    reason = (item.get("reason", "") or "")[:80]
                    self_review_block += f"  Edit #{idx}: status={status}, reason={reason}\n"
            else:
                self_review_block += "  （Rewriter 未提供自评，以下由你独立判断）\n"

        return f"""你是一位星云奖级别的终审译者。
你需要综合【审辩者报告】，决定当前的【文学润色稿】是否可以直接定稿。

【审辩者报告】
{json.dumps(critic_report, ensure_ascii=False)}
{self_review_block}
【原文】
{source_text}
【当前润色稿】
{lit_trans}
{thresholds_block}
【JSON 输出规则 — 必须严格遵守】
1. 输出纯 JSON，禁止包含任何 Markdown 标记、代码块、或额外说明文字
2. 字符串值中的双引号 " 必须转义为 \"
3. 不得在数组/对象的最后一个元素后加逗号
4. ⚠️ 如果 JSON 格式错误，裁决将直接判定为 REJECT

输出 JSON Schema：
{{
  "decision": "PASS" | "REJECT",
  "reject_reason": "如果 REJECT，告诉上游的 Rewriter 必须重点修改哪里",
  "new_style_rule": {{
    "rule_description": "例如：处理独白时必须使用短促、断裂的句式。",
    "reason": "为什么这条规则对整本书很重要？"
  }},
  "unresolved_edits": [
    {{
      "index": 1,
      "location": "...",
      "issue": "...",
      "reason": "为什么这条没解决"
    }}
  ]
}}

约束：
1. unresolved_edits 是可选字段，仅在有 specific_edits 且 Rewriter 未完全解决时才输出
2. 如果全部解决或没有 specific_edits，unresolved_edits 应为空数组 []
3. unresolved_edits 不改变 decision 主逻辑——即使整体 PASS 也可以有 unresolved_edits 作为参考

注意："final_text" 不需要你输出，系统将自动使用 Rewriter 的润色稿作为最终译文。
请严格按上述 Schema 输出唯一一个 JSON 对象。
"""

    def process_chunk(self, chunk_id: str, source_text: str, lit_trans: str, critic_report: dict, affected_chunks: List[str] = None, rewriter_self_review: Optional[list] = None) -> Dict[str, Any]:
        print(f"⚖️ [Judge Agent] 正在对 {chunk_id} 进行最终裁决...")
        prompt = self._build_prompt(source_text, lit_trans, critic_report, rewriter_self_review=rewriter_self_review)

        # 使用配置路由：judge_decision
        model_key, params = _config.resolve_task_model("judge_decision")
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
            # H7（v2）：Judge JSON 损坏 → 抛异常 → pipeline 置 FAILED → H6 恢复 JUDGING
            raise RuntimeError("Judge 输出 JSON 损坏，重新裁决。")

        scores = critic_report.get("scores", {})

        # P0: 安全网——仅在 LLM 判 PASS 但评分极低时覆盖，防止 LLM 幻觉放行劣质译文
        # 不同于旧版无条件覆盖（avg<7.5→REJECT），新版尊重 LLM 独立判断
        if scores:
            avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / len(scores)
            safety_min = _config.pipeline.get("judge_safety_net_avg_min", 5.5)
            if result.get("decision") == "PASS" and avg_score < safety_min:
                result["decision"] = "REJECT"
                result["reject_reason"] = (f"安全网: Critic 平均分 {avg_score:.1f} 低于安全阈值 {safety_min}，"
                                           f"但 Judge 判定通过，疑似 LLM 幻觉。")

        # 如果 PASS 并且提炼出了新的高级规则，写入 Decision DB (Level 3)
        # STYLE 只入库、不把当前 chunk 列入 affected；source_key 用规则指纹去重，避免每 chunk 一条无限膨胀
        if result.get("decision") == "PASS" and "new_style_rule" in result and result["new_style_rule"]:
            rule = result["new_style_rule"]
            desc = (rule.get("rule_description") or "").strip()
            if desc:
                fp = hashlib.sha1(desc.encode("utf-8")).hexdigest()[:12]
                self.db.add_decision(
                    level=DecisionLevel.STYLE,
                    source=f"style_{fp}",
                    translation=desc,
                    reason=rule.get("reason", "Judge Agent 动态提炼"),
                    affected_chunks=None,  # 禁止自脏；STYLE 默认不 backtrack
                )

        return result


# ────────────────────────────────────────────────────────────────────
# Section 7 — Pipeline 主流程
# ────────────────────────────────────────────────────────────────────

class TranslationPipeline:
    def __init__(self, chapter_id: str):
        self.chapter_id = chapter_id
        # 使用绝对路径定位数据库（相对于此文件的上级目录）
        self._root_dir = Path(__file__).resolve().parent.parent
        self.scheduler = TaskScheduler(db_path=str(self._root_dir / "db" / "workflow.db"))
        self.decision_engine = DecisionEngine(
            db_path=str(self._root_dir / "db" / "decision_db.sqlite"),
            scheduler_factory=lambda: self.scheduler
        )

        self.llm = get_llm_client_for_task("literal_translation")
        self.literal_role = _config.get_task_role("literal_translation")

        # 实例化 Agents
        self.ref_agent = ReferenceAgent(self.decision_engine, chapter_id)
        self.rewriter_agent = LiteraryRewriterAgent(self.decision_engine)
        self.critic_agent = CriticAgent()
        self.judge_agent = JudgeAgent(self.decision_engine)

        # 绝对路径 output
        self.output_dir = self._root_dir / "output" / chapter_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 从配置加载风格指南
        sg = _config.style_guide
        self.style_guide = {
            "avg_sentence_length": sg.get("avg_sentence_length", "较长且富有韵律"),
            "lexicon_preference": sg.get("lexicon_preference", "古典、史诗感、冷硬"),
            "author_priority_ratio": sg.get("author_priority_ratio", 0.7)
        }

    def _save_intermediate(self, chunk_id: str, step: str, data: str | dict):
        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            if isinstance(data, str):
                json.dump({"text": data}, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_intermediate(self, chunk_id: str, step: str):
        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        return data

    def _build_literal_prompt(self, source_text: str) -> str:
        """组装直译 prompt（含有界术语表）；供生产路径与单测共用。"""
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        glossary = self.decision_engine.format_prompt_context(source_text=source_text)
        return (
            f"请将以下{genre}片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n"
            f"{glossary}\n\n"
            f"【原文】\n{source_text}"
        )

    def _run_raw_translator(self, source_text: str) -> str:
        prompt = self._build_literal_prompt(source_text)
        # 使用配置路由：literal_translation
        model_key, params = _config.resolve_task_model("literal_translation")
        model_name = get_role_model_name(self.literal_role) or model_key
        extra_body = get_role_extra_body(self.literal_role)
        return self.llm.generate(prompt, model_name=model_name, extra_body=extra_body, **params)

    def _memory_path(self) -> Path:
        return self.output_dir / "_memory.jsonl"

    def _append_chunk_memory(self, chunk_id: str, lit_text: str, source_text: str = "") -> None:
        """定稿后写入轻量章内记忆，供后续 chunk 润色注入。"""
        text = lit_text if isinstance(lit_text, str) else str(lit_text)
        # 规则摘要：取前 2–3 句或截断 240 字
        parts = re.split(r"[。！？\.\!\?]\s*", text.strip())
        sentences = [p.strip() for p in parts if p and p.strip()]
        summary = "。".join(sentences[:3])
        if summary and not summary.endswith("。"):
            summary += "。"
        if len(summary) > 240:
            summary = summary[:240] + "…"
        rec = {
            "chunk_id": chunk_id,
            "summary": summary or text[:200],
            "source_preview": (source_text or "")[:80],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with open(self._memory_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"  ⚠️ 写入 chapter memory 失败: {e}")

    def _load_prior_memory(self, current_chunk_id: str, max_entries: int = 3) -> str:
        """读取本章已完成 chunk 的摘要（不含当前 chunk）。"""
        path = self._memory_path()
        if not path.exists():
            return ""
        lines = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # P2-1: memory jsonl 用 robust_json_loads 抗损坏行
                    rec = robust_json_loads(line, expected_type=dict)
                    if not rec:
                        continue
                    if rec.get("chunk_id") == current_chunk_id:
                        continue
                    lines.append(rec)
        except OSError:
            return ""
        recent = lines[-max_entries:]
        if not recent:
            return ""
        return "\n".join(
            f"- [{r.get('chunk_id', '?')}] {r.get('summary', '')}" for r in recent
        )

    def run(self) -> bool:
        print(f"🚀 [Pipeline] 启动批处理模式处理章节: {self.chapter_id}")

        # 检查章节是否有任务，避免在空 DB 上误报"完成"
        existing_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
        if not existing_tasks:
            # 区分两种空：真未 init vs init 被跳过但 DB 无行/无产出
            print(f"❌ [Pipeline] 章节 {self.chapter_id} 无可执行任务（可能 init 被跳过或 DB 无记录），"
                  f"请用 --force 重新初始化。")
            return False

        # 从配置读取 pipeline 参数
        pipe_cfg = _config.pipeline
        batch_size = pipe_cfg.get("batch_size", 50)
        poll_interval = pipe_cfg.get("poll_interval", 0.5)

        # 阶段顺序定义：按流水线顺序处理，每阶段批量处理
        pipeline_stages = [
            (TaskState.DIRTY, self._process_dirty_batch, "回溯重跑"),
            (TaskState.FAILED, self._process_failed_batch, "失败重试"),
            (TaskState.PENDING, self._process_pending_batch, "初始化"),
            (TaskState.EXTRACTING_TERMS, self._process_extracting_terms_batch, "术语提取"),
            (TaskState.TRANSLATING_RAW, self._process_translating_raw_batch, "直译"),
            (TaskState.REWRITING_LITERARY, self._process_rewriting_batch, "文学润色"),
            (TaskState.AUDITING, self._process_auditing_batch, "审计评分"),
            (TaskState.JUDGING, self._process_judging_batch, "最终裁决"),
        ]

        while True:
            any_progress = False

            for state, handler, stage_name in pipeline_stages:
                tasks = self.scheduler.get_tasks_by_state(state, batch_size=batch_size, chapter_id=self.chapter_id)
                if not tasks:
                    continue

                print(f"📦 [Batch] {stage_name} 阶段: 处理 {len(tasks)} 个任务")
                handler(tasks)
                any_progress = True

            if not any_progress:
                # 仅表示队列已清空，不代表产出完整；最终是否成功由 _verify_final_outputs 决定
                print(f"⏹ [Pipeline] 章节 {self.chapter_id} 任务队列已清空，进入产出校验…")
                break

            time.sleep(poll_interval)

        # 完成后交叉验证：final.json 落盘数应与任务数一致，否则视为假阳性完成
        return self._verify_final_outputs(len(existing_tasks))

    def _verify_final_outputs(self, expected_count: int, strict: bool = True) -> bool:
        """校验本章 final.json 实际落盘是否匹配任务。

        防"任务队列空即报成功"的假阳性完成。除比数外，还比对 chunk_id
        集合：残留旧 final（上一轮多余 chunk）或缺失都判残缺，避免 stale 假通过。
        strict=True（默认）时额外要求所有任务 state==COMPLETED（排除 PERMANENTLY_FAILED/fallback 章）。
        返回 True 表示产出完整，False 表示残缺。
        """
        if expected_count <= 0:
            return True
        actual_files = list(self.output_dir.glob("*_final.json"))
        actual_ids = {f.stem.rsplit("_final", 1)[0] for f in actual_files}
        expected_ids = {t["chunk_id"] for t in self.scheduler.get_all_tasks_by_chapter(self.chapter_id)}

        missing = expected_ids - actual_ids
        stale = actual_ids - expected_ids
        if missing or stale:
            print(f"❌ [Pipeline] 章节 {self.chapter_id} 产出残缺/残留："
                  f"期望 {len(expected_ids)} 个，缺失 {len(missing)}，残留旧文件 {len(stale)}。"
                  f"请用 --force 重新初始化后重跑。")
            if missing:
                print(f"   缺失 chunk: {sorted(missing)}")
            if stale:
                print(f"   残留 chunk: {sorted(stale)}")
            return False
        if strict:
            bad = [t["chunk_id"] for t in self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
                    if t["state"] != TaskState.COMPLETED.value]
            if bad:
                fallback_chunks = []
                missing_chunks = []
                for cid in bad:
                    if (self.output_dir / f"{cid}_final.json").exists():
                        fallback_chunks.append(cid)
                    else:
                        missing_chunks.append(cid)
                if fallback_chunks:
                    print(f"⚠️ [Pipeline] 章节 {self.chapter_id} 有 {len(fallback_chunks)} 个 chunk 质量未达标(fallback)，"
                          f"译文已写入但状态为 PERMANENTLY_FAILED：{fallback_chunks}")
                if missing_chunks:
                    print(f"❌ [Pipeline] 章节 {self.chapter_id} 有 {len(missing_chunks)} 个 chunk 完全无产出：{missing_chunks}")
                return False
        return True

    def _process_pending_batch(self, tasks):
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS)

    def _process_dirty_batch(self, tasks):
        for task in tasks:
            chunk_id = task['chunk_id']
            print(f"🔄 [Pipeline] {chunk_id} 标记为 DIRTY，重置到术语提取")
            for step in ["raw", "literary", "critic_report", "final"]:
                intermediate = self.output_dir / f"{chunk_id}_{step}.json"
                if intermediate.exists():
                    try:
                        intermediate.unlink()
                    except OSError as e:
                        print(f"  ⚠️ 清理失败 {intermediate}: {e}")
            # 清理版本历史文件
            for pattern in [f"{chunk_id}_literary_v*.json", f"{chunk_id}_rewrite_meta_v*.json"]:
                for f in self.output_dir.glob(pattern):
                    try:
                        f.unlink()
                    except OSError as e:
                        print(f"  ⚠️ 清理失败 {f}: {e}")
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS, reset_counters=True)

    def _get_recovery_stage(self, last_error: str) -> TaskState:
        if not last_error:
            return TaskState.EXTRACTING_TERMS
        # 兼容两种格式：
        # 1) [STAGE_NAME] 错误描述
        # 2) JSON {"judge_reason": ...}（quality_retry 路径）
        stage_match = re.search(r'^\[(\w+)\]', last_error)
        if stage_match:
            stage = stage_match.group(1)
        elif '"judge_reason"' in last_error:
            stage = "REWRITING_LITERARY"
        else:
            return TaskState.EXTRACTING_TERMS
        STAGE_MAP = {
            "EXTRACTING_TERMS": TaskState.EXTRACTING_TERMS,
            "TRANSLATING_RAW": TaskState.TRANSLATING_RAW,
            "REWRITING_LITERARY": TaskState.REWRITING_LITERARY,
            "AUDITING": TaskState.AUDITING,
            "JUDGING": TaskState.JUDGING,
        }
        return STAGE_MAP.get(stage, TaskState.EXTRACTING_TERMS)

    def _try_fallback_final(self, chunk_id: str) -> bool:
        """API 级重试耗尽时，从已有中间产物抢救 fallback final.json

        优先用文学润色稿，回退到直译稿。若两者都不存在则不保存。
        Returns True 如果成功保存了 fallback。
        """
        for step, label in [("literary", "文学润色稿"), ("raw", "直译稿")]:
            try:
                data = self._load_intermediate(chunk_id, step)
                text = data if isinstance(data, str) else data.get("text", str(data))
                if text and text.strip():
                    self._save_intermediate(chunk_id, "final", {
                        "text": text,
                        "metadata": {
                            "fallback": True,
                            "reason": f"API_FAILURE_FALLBACK_FROM_{step.upper()}",
                            "source_step": step,
                        }
                    })
                    print(f"  ↪ 已从{label}抢救 fallback final: {chunk_id}")
                    return True
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
        return False

    def _process_failed_batch(self, tasks):
        retry_by_stage = {}
        permanent_fail_ids = []
        max_retries = _config.pipeline.get("max_retries", 3)
        for task in tasks:
            chunk_id = task['chunk_id']
            retries = task.get('retries', 0)
            if retries >= max_retries:
                print(f"❌ [Pipeline] {chunk_id} 重试次数过多，转入 PERMANENTLY_FAILED 终态")
                self._try_fallback_final(chunk_id)
                permanent_fail_ids.append(chunk_id)
            else:
                recovery_stage = self._get_recovery_stage(task.get('last_error', ''))
                retry_by_stage.setdefault(recovery_stage, []).append(chunk_id)
                print(f"🔁 [Pipeline] {chunk_id} 重试中 (第 {retries + 1} 次) → {recovery_stage.value}")
        if permanent_fail_ids:
            self.scheduler.batch_update_state(
                permanent_fail_ids,
                TaskState.PERMANENTLY_FAILED,
                error_msgs={cid: f"超过重试上限 (retries>={max_retries})，需人工介入" for cid in permanent_fail_ids}
            )
        for stage, ids in retry_by_stage.items():
            if ids:
                print(f"🔁 [Pipeline] {len(ids)} 个任务恢复到 {stage.value}")
                self.scheduler.batch_update_state(ids, stage)

    def _process_extracting_terms_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                self.ref_agent.process_chunk(chunk_id, source_text, affected_chunks=[chunk_id])
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 术语提取异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[EXTRACTING_TERMS] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.TRANSLATING_RAW)

    def _process_translating_raw_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                raw_text = self._run_raw_translator(source_text)
                self._save_intermediate(chunk_id, "raw", raw_text)
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 直译异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[TRANSLATING_RAW] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.REWRITING_LITERARY)

    def _process_rewriting_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                quality_retries = task.get('quality_retries', 0)
                low_dims = []
                dim_scores = {}

                reject_reason_raw = task.get('last_error', '')
                reject_reason = reject_reason_raw
                critic_feedback = None
                specific_edits = []
                if reject_reason_raw:
                    # P2-1: 用 robust_json_loads 抗 critic 报告畸形；返回 {} 走原 raw 字符串兜底
                    feedback = robust_json_loads(reject_reason_raw, expected_type=dict)
                    if feedback:
                        reject_reason = feedback.get('judge_reason', '')
                        critic_feedback = feedback.get('critic_suggestions', '')
                        specific_edits = feedback.get("specific_edits", [])
                        low_dims = feedback.get('low_dims', [])
                        dim_scores = feedback.get('scores', {})
                        if low_dims:
                            low_dim_detail = ", ".join(
                                f"{d}({dim_scores.get(d, '?')}分)" for d in low_dims
                            )
                            critic_feedback = f"低分维度：{low_dim_detail}\n" + (critic_feedback or '')

                all_feedback = []
                if reject_reason_raw and quality_retries > 0:
                    for i in range(quality_retries):
                        try:
                            meta = self._load_intermediate(chunk_id, f"rewrite_meta_v{i}")
                            all_feedback.append({
                                "reason": (meta.get("reject_reason_summary") or "")[:80],
                                "resolved": False,
                            })
                        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                            pass

                prev_lit_text = None
                try:
                    prev_lit_text = self._load_intermediate(chunk_id, "literary")
                except (FileNotFoundError, KeyError):
                    pass

                try:
                    raw_text = self._load_intermediate(chunk_id, "raw")
                except (FileNotFoundError, KeyError):
                    print(f"⚠️ [Pipeline] {chunk_id} 缺失 raw 底稿，转入 FAILED 等待恢复")
                    self.scheduler.update_task_state(chunk_id, TaskState.FAILED, error_msg="[TRANSLATING_RAW] raw file missing, needs re-translation")
                    continue

                narrative_memory = self._load_prior_memory(chunk_id)
                raw_result = self.rewriter_agent.process_chunk(
                    chunk_id, raw_text, self.style_guide, source_text,
                    prev_lit_text=prev_lit_text,
                    reject_reason=reject_reason,
                    critic_feedback=critic_feedback,
                    retry_count=quality_retries,
                    all_feedback=all_feedback,
                    specific_edits=specific_edits,
                    narrative_memory=narrative_memory,
                    low_dims=low_dims,
                )
                # process_chunk 返回 (lit_text, self_review) 元组
                if isinstance(raw_result, tuple):
                    lit_text, self_review = raw_result
                else:
                    lit_text, self_review = raw_result, None  # 向后兼容旧返回值
                self._save_intermediate(chunk_id, "literary", lit_text)

                meta = {
                    "retry_count": quality_retries,
                    "quality_retries": quality_retries,
                    "reject_reason_summary": (reject_reason or "")[:200],
                    "critic_feedback_summary": (critic_feedback or "")[:200],
                    "all_feedback_summaries": all_feedback,
                    "self_review": self_review,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._save_intermediate(chunk_id, f"literary_v{quality_retries}", lit_text)
                self._save_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}", meta)

                success_ids.append(chunk_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ [Pipeline] {task['chunk_id']} 润色异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[REWRITING_LITERARY] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.AUDITING)

    def _check_entity_consistency(self, chunk_id: str, lit_text: str, source_text: str) -> List[Dict]:
        """实体一致性核查：检查译文中实体译名是否与 entity_registry 一致
        
        Returns:
            List[Dict]: 不一致项列表，每项包含 entity, expected, found, chunk_id
        """
        if not self.decision_engine:
            return []

        # P3-1: 改用公共 helper（语义：所有非待核查实体，不过滤 confidence）
        entities = query_reviewable_entities(self.decision_engine.conn)

        if not entities:
            return []
        
        inconsistencies = []
        for row in entities:
            canonical = row[0]
            source_names_json = row[1]
            source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
            variant_names = robust_json_loads(row[2], expected_type=list) if len(row) > 2 and row[2] else []

            if len(canonical) < 2:
                continue

            _cjk_vf = None
            for vn in variant_names:
                if vn and vn in lit_text:
                    _cjk_vf = vn
                    break
            if _cjk_vf:
                inconsistencies.append({
                    "entity": canonical,
                    "expected": canonical,
                    "found": _cjk_vf,
                    "type": "variant_instead_of_canonical",
                    "chunk_id": chunk_id,
                })
                continue

            if canonical not in lit_text:
                found_variant = None
                for sn in source_names:
                    if sn in lit_text:
                        found_variant = sn
                        break
                
                if found_variant:
                    inconsistencies.append({
                        "entity": canonical,
                        "expected": canonical,
                        "found": found_variant,
                        "type": "variant_instead_of_canonical",
                        "chunk_id": chunk_id
                    })
                else:
                    # 只有原文中确实出现了该实体时才报 missing
                    # 避免"全书角色本章未出场"的假阳性
                    appears_in_source = any(sn in source_text for sn in source_names)
                    if appears_in_source:
                        inconsistencies.append({
                            "entity": canonical,
                            "expected": canonical,
                            "found": None,
                            "type": "missing",
                            "chunk_id": chunk_id
                        })

            self._maybe_discover_entity_variant(canonical, lit_text, variant_names)

        return inconsistencies

    def _maybe_discover_entity_variant(self, canonical: str, lit_text: str, existing_variants: list):
        try:
            if not canonical or not re.fullmatch(r'[一-鿿]{2,}', canonical):
                return
            length = len(canonical)
            candidates = set(re.findall(r'[一-鿿]{%d}' % length, lit_text))
            for cand in candidates:
                if cand == canonical or cand in existing_variants:
                    continue
                if self.decision_engine.is_canonical_name(cand):
                    continue
                diff = sum(1 for a, b in zip(canonical, cand) if a != b)
                if diff == 1:
                    self.decision_engine.add_entity_variant(canonical, cand)
        except Exception:
            pass

    def _process_auditing_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                raw_text = self._load_intermediate(chunk_id, "raw")
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self.critic_agent.process_chunk(chunk_id, source_text, raw_text, lit_text, self.style_guide)

                # Q3 修复：Critic LLM 偶发拼错评分维度名（如 voice_consistence），
                # 归一化后再入库，避免脏维度名在 Judge 阈值查表时被静默忽略。
                if isinstance(critic_report, dict) and "scores" in critic_report:
                    critic_report["scores"] = _normalize_scores(critic_report.get("scores"))

                # 实体一致性核查：检查译文中实体译名是否与 entity_registry 一致
                entity_notes = self._check_entity_consistency(chunk_id, lit_text, source_text)
                if entity_notes:
                    critic_report["entity_consistency_notes"] = entity_notes
                    print(f"  ⚠️ 实体一致性核查: {len(entity_notes)} 条不一致项")

                self._save_intermediate(chunk_id, "critic_report", critic_report)
                if not critic_report.get("scores"):
                    print(f"⚠️ [Pipeline] {chunk_id} Critic 评分为空（JSON 解析失败），转入 FAILED 等待重试")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.FAILED,
                        error_msg="[AUDITING] Critic 返回空 scores"
                    )
                    continue

                # 实体硬失败：严重不一致时强制一轮质量退回（notes-only 不够）
                # P0 修复：增加降级机制，避免死锁
                entity_hard_fail = bool(_config.pipeline.get("entity_hard_fail", True))
                degradation_threshold = int(_config.pipeline.get("entity_hard_fail_degradation_threshold", 3))
                quality_retries = task.get("quality_retries", 0) or 0
                serious = [
                    n for n in (entity_notes or [])
                    if n.get("type") in ("variant_instead_of_canonical", "missing")
                ]
                
                # 降级逻辑：质量重试 > 0 时，仅在严重不一致项 > 阈值时才强制退回
                # 避免因少量实体问题导致死锁
                if entity_hard_fail and serious and quality_retries == 0:
                    feedback = {
                        "judge_reason": f"实体一致性硬失败: {len(serious)} 项译名不一致，须按 entity_registry 修正",
                        "critic_suggestions": json.dumps(serious, ensure_ascii=False)[:500],
                        "low_dims": ["voice_consistency"],
                        "scores": critic_report.get("scores", {}),
                        "specific_edits": [
                            {
                                "location": s.get("entity", "?"),
                                "original": s.get("found") or "",
                                "issue": s.get("type"),
                                "suggested_direction": f"统一为 {s.get('expected')}",
                            }
                            for s in serious[:10]
                        ],
                    }
                    self.scheduler.update_task_state(
                        chunk_id,
                        TaskState.REWRITING_LITERARY,
                        error_msg=json.dumps(feedback, ensure_ascii=False),
                        quality_retry=True,
                    )
                    print(f"🔁 [Pipeline] {chunk_id} 实体硬失败，退回文学润色")
                    continue
                elif entity_hard_fail and serious and quality_retries > 0 and len(serious) > degradation_threshold:
                    # 降级逻辑：质量重试 > 0 且严重不一致项 > 阈值时，仍强制退回
                    feedback = {
                        "judge_reason": f"实体一致性硬失败(降级): {len(serious)} 项译名不一致，须按 entity_registry 修正",
                        "critic_suggestions": json.dumps(serious, ensure_ascii=False)[:500],
                        "low_dims": ["voice_consistency"],
                        "scores": critic_report.get("scores", {}),
                        "specific_edits": [
                            {
                                "location": s.get("entity", "?"),
                                "original": s.get("found") or "",
                                "issue": s.get("type"),
                                "suggested_direction": f"统一为 {s.get('expected')}",
                            }
                            for s in serious[:10]
                        ],
                    }
                    self.scheduler.update_task_state(
                        chunk_id,
                        TaskState.REWRITING_LITERARY,
                        error_msg=json.dumps(feedback, ensure_ascii=False),
                        quality_retry=True,
                    )
                    print(f"🔁 [Pipeline] {chunk_id} 实体硬失败(降级)，退回文学润色 (quality_retries={quality_retries})")
                    continue
                elif entity_hard_fail and serious and quality_retries > 0 and len(serious) <= degradation_threshold:
                    # 降级逻辑：质量重试 > 0 且严重不一致项 ≤ 阈值时，允许继续进入裁决阶段
                    print(f"🟡 [Pipeline] {chunk_id} 实体一致性降级放行 (quality_retries={quality_retries}, serious={len(serious)})")
                    # 不 continue，继续进入 success_ids

                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 审计异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[AUDITING] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.JUDGING)

    def _fallback_to_rewriter(self, chunk_id, lit_text, critic_report, judge_result):
        scores = critic_report.get("scores", {})
        thresholds = _config.critic_thresholds
        low_dims = [k for k, v in scores.items()
                    if isinstance(v, (int, float)) and v < thresholds.get(k, 7.0)]
        specific_edits = critic_report.get("specific_edits", [])[:20]
        if specific_edits:
            for e in specific_edits:
                e["suggested_direction"] = (e.get("suggested_direction", "") or "")[:200]
        # 实体一致性：将 registry 主名反馈给 Rewriter，避免重试仍用错译名（修复 ch02 实体死锁）
        entity_notes = critic_report.get("entity_consistency_notes", []) or []
        entity_block = ""
        if entity_notes:
            lines = []
            for n in entity_notes[:30]:
                ent = n.get("entity", "")
                exp = n.get("expected", "")
                found = n.get("found", "")
                ntype = n.get("type", "")
                if exp and found and found != exp:
                    lines.append(f"- 「{found}」应统一为「{exp}」（{ntype}）")
                elif exp:
                    lines.append(f"- 实体「{ent}」须使用主名「{exp}」（{ntype}）")
            if lines:
                entity_block = "实体一致性要求（必须遵循）：\n" + "\n".join(lines)
        enriched_feedback = json.dumps({
            "judge_reason": judge_result.get("reject_reason", ""),
            "critic_suggestions": critic_report.get("improvement_suggestions", ""),
            "entity_consistency_notes": entity_block,
            "low_dims": low_dims,
            "scores": scores,
            "specific_edits": specific_edits,
        }, ensure_ascii=False)
        self.scheduler.update_task_state(
            chunk_id, TaskState.REWRITING_LITERARY,
            error_msg=enriched_feedback, quality_retry=True
        )
        print(f"🔁 [Pipeline] {chunk_id} 裁决未通过，退回文学润色")

    def _process_judging_batch(self, tasks):
        max_quality_retries = _config.pipeline.get("max_quality_retries", 6)
        early_stop_cfg = _config.pipeline.get("early_stop", {})
        early_stop_enabled = early_stop_cfg.get("enabled", True)
        early_stop_max_low = early_stop_cfg.get("max_low_dims", 3)
        early_stop_threshold = early_stop_cfg.get("low_score_threshold", 3.0)
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                quality_retries = task.get('quality_retries', 0)
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self._load_intermediate(chunk_id, "critic_report")

                # 加载 Rewriter 自评（如果有）
                self_review = None
                if quality_retries > 0:
                    try:
                        meta = self._load_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}")
                        if isinstance(meta, dict):
                            self_review = meta.get("self_review")
                    except Exception:
                        pass

                judge_result = self.judge_agent.process_chunk(
                    chunk_id, source_text, lit_text, critic_report,
                    affected_chunks=[chunk_id],
                    rewriter_self_review=self_review,
                )

                if judge_result.get("decision") == "PASS":
                    self._save_intermediate(chunk_id, "final", lit_text)
                    self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED)
                    self._append_chunk_memory(chunk_id, lit_text if isinstance(lit_text, str) else lit_text.get("text", str(lit_text)) if isinstance(lit_text, dict) else str(lit_text), source_text)
                    print(f"✅ [Pipeline] {chunk_id} 定稿完成。")
                elif quality_retries >= max_quality_retries:
                    # O 方案：安全网放行——若各维度已接近阈值（仅差 1-2 分），
                    # 不再整章 PERMANENTLY_FAILED 丢失译文，改为带告警定稿。
                    scores = critic_report.get("scores", {}) if isinstance(critic_report, dict) else {}
                    thresholds = _config.critic_thresholds or {}
                    numeric = [v for v in scores.values() if isinstance(v, (int, float))]
                    avg_score = (sum(numeric) / len(numeric)) if numeric else 0.0
                    safety_net_min = float(_config.pipeline.get("judge_safety_net_avg_min", 5.5))
                    # 是否有维度低于阈值超过 1 分（即严重偏离）
                    severe_count = sum(
                        1 for k, v in scores.items()
                        if isinstance(v, (int, float))
                        and v < thresholds.get(k, 7.0) - 1.0
                    )
                    if avg_score >= safety_net_min and severe_count == 0:
                        self._save_intermediate(chunk_id, "final", {
                            "text": lit_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "SAFETY_NET_PASSED",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                                "avg_score": round(avg_score, 2),
                            }
                        })
                        self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED, error_msg=f"[SAFETY_NET] 平均分{avg_score:.2f}≥{safety_net_min}，接近阈值放行")
                        print(f"🟡 [Pipeline] {chunk_id} 安全网放行（平均分{avg_score:.2f}，未达PERMANENTLY_FAILED）")
                    else:
                        self._save_intermediate(chunk_id, "final", {
                            "text": lit_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "PERMANENTLY_FAILED_AFTER_MAX_RETRIES",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        })
                        self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, error_msg=judge_result.get("reject_reason") or "质量重试耗尽")
                        print(f"⚠️ [Pipeline] {chunk_id} 质量重试耗尽，已 fallback 写入 final（最后一版润色稿）")
                elif early_stop_enabled and quality_retries == 0:
                    # early_stop 仅在首轮裁决生效：避免 rework 一两次就放弃可恢复的 chunk
                    # 第二轮起若仍不达标，正常走 fallback 完成 max_quality_retries 轮 rework
                    # 详见 config.yaml pipeline.early_stop.apply_only_first_round
                    scores = critic_report.get("scores", {})
                    very_low_count = sum(
                        1 for v in scores.values()
                        if isinstance(v, (int, float)) and v < early_stop_threshold
                    )
                    if very_low_count >= early_stop_max_low:
                        stop_reason = (
                            f"EARLY_STOP: {very_low_count}项低于{early_stop_threshold}分"
                            f"（{scores}），放弃重试"
                        )
                        print(f"⏭️ [Pipeline] {chunk_id} {stop_reason}")
                        self._save_intermediate(chunk_id, "final", {
                            "text": lit_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "EARLY_STOP",
                                "early_stop_detail": stop_reason,
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        })
                        self.scheduler.update_task_state(
                            chunk_id, TaskState.PERMANENTLY_FAILED,
                            error_msg=stop_reason
                        )
                        continue
                    self._fallback_to_rewriter(chunk_id, lit_text, critic_report, judge_result)
                else:
                    self._fallback_to_rewriter(chunk_id, lit_text, critic_report, judge_result)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 裁决异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[JUDGING] {e}")


# ────────────────────────────────────────────────────────────────────
# Section 8 — 测试工具
# ────────────────────────────────────────────────────────────────────

# ----- 8.1 Golden Set Evaluator -----

class GoldenSetEvaluator:
    """黄金测试集评估器 - 风格坍缩率量化"""

    def __init__(self, db_path: Optional[str] = None):
        self.llm = get_llm_client_for_task("literal_translation")
        self.literal_role = _config.get_task_role("literal_translation")
        shared_db = DecisionEngine(db_path=db_path) if db_path else DecisionEngine()
        self.db = shared_db
        self.ref_agent = ReferenceAgent(shared_db, chapter_id="golden_test")
        self.rewriter_agent = LiteraryRewriterAgent(shared_db)
        self.critic_agent = CriticAgent()
        self.judge_agent = JudgeAgent(shared_db)

    def evaluate_style_collapse(self, source_text: str, translated_text: str) -> dict:
        """评估风格坍缩率：对比原文与译文的风格特征"""
        source_stats = self._analyze_style(source_text)
        trans_stats = self._analyze_style(translated_text)

        preservation = {}
        for key in source_stats:
            if source_stats[key] > 0:
                ratio = min(trans_stats[key] / source_stats[key], 1.0)
                preservation[key] = ratio

        avg_preservation = sum(preservation.values()) / len(preservation) if preservation else 0
        collapse_rate = 1.0 - avg_preservation

        return {
            "source_style": source_stats,
            "translated_style": trans_stats,
            "preservation_per_dimension": preservation,
            "average_preservation": avg_preservation,
            "style_collapse_rate": collapse_rate
        }

    def _analyze_style(self, text: str) -> dict:
        """提取文本风格统计特征"""
        # 语言检测：是否包含中文字符
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))
        sent_split = '。' if has_cjk else r'[.!?]+'
        sentences = [s.strip() for s in re.split(sent_split, text) if s.strip()]

        avg_sent_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0

        words = text.split()
        _stop_words = {'的', '了', '是', '我', '在', '有', '和', '为', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        content_words = [w for w in words if len(w) > 1 and w.lower() not in _stop_words]
        vocab_density = len(content_words) / len(words) if words else 0

        rhetorical_markers = ['如', '似', '仿佛', '好像', '像', '犹如', '宛若', 'like', 'as if', 'as though', 'seem', 'appear', 'resemble']
        rhetorical_count = sum(text.lower().count(m) for m in rhetorical_markers)
        rhetorical_density = rhetorical_count / len(sentences) if sentences else 0

        punctuation = ['，', '。', '；', '：', '——', '…', '\u201c', '\u201d', '\u2018', '\u2019', ',', ';', ':', '—', '...', '"', "'", '?', '!']
        punct_count = sum(text.count(p) for p in punctuation)
        punct_density = punct_count / len(text) * 1000 if text else 0

        return {
            "avg_sentence_length": avg_sent_len,
            "vocabulary_density": vocab_density,
            "rhetorical_density": rhetorical_density,
            "punctuation_density": punct_density
        }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的 Levenshtein 编辑距离（手写，无外部依赖）"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def run_golden_test():
    """运行黄金测试集（需要 input/golden/hyperion_5k.md）"""
    print("=" * 60)
    print("🧪 OpenLiterary 黄金测试集跑分")
    print("=" * 60)

    root_dir = Path(__file__).resolve().parent.parent
    test_file = root_dir / "input" / "golden" / "hyperion_5k.md"
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return []

    with open(test_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    print(f"📖 测试文本长度: {len(source_text)} 字符")

    # 使用配置的 chunker 参数
    chunker_cfg = _config.chunker
    chunker = SmartChunker(
        soft_limit=chunker_cfg.get("soft_limit", 1000),
        hard_limit=chunker_cfg.get("hard_limit", 2500)
    )
    chunks = chunker.split_markdown(source_text)
    print(f"✂️ 切分为 {len(chunks)} 个块")

    evaluator = GoldenSetEvaluator(db_path=str(root_dir / "db" / "decision_db.sqlite"))
    all_results = []

    # 从配置获取风格指南
    style_guide = _config.style_guide

    for i, chunk in enumerate(chunks):
        chunk_id = f"golden_chunk{i:03d}"
        try:
            print(f"\n📦 处理第 {i+1}/{len(chunks)} 块 ({len(chunk)} 字符)...")

            evaluator.ref_agent.process_chunk(chunk_id, chunk, affected_chunks=[chunk_id])

            genre = (_config.style_guide or {}).get("genre") or "文学小说"
            gloss = evaluator.db.format_prompt_context(source_text=chunk)
            raw_prompt = (
                f"请将以下{genre}片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n"
                f"{gloss}\n\n【原文】\n{chunk}"
            )
            # 使用配置路由 literal_translation
            model_key, params = _config.resolve_task_model("literal_translation")
            model_name = get_role_model_name(evaluator.literal_role) or model_key
            extra_body = get_role_extra_body(evaluator.literal_role)
            raw_text = evaluator.llm.generate(raw_prompt, model_name=model_name, extra_body=extra_body, **params)

            raw_result = evaluator.rewriter_agent.process_chunk(chunk_id, raw_text, style_guide, chunk)
            lit_text, _ = raw_result if isinstance(raw_result, tuple) else (raw_result, None)

            critic_report = evaluator.critic_agent.process_chunk(chunk_id, chunk, raw_text, lit_text, style_guide)

            judge_result = evaluator.judge_agent.process_chunk(chunk_id, chunk, lit_text, critic_report, affected_chunks=[chunk_id])

            final_text = lit_text
            style_eval = evaluator.evaluate_style_collapse(chunk, final_text)

            result = {
                "chunk_id": chunk_id,
                "source_length": len(chunk),
                "critic_scores": critic_report.get("scores", {}),
                "judge_decision": judge_result.get("decision"),
                "style_collapse_rate": style_eval["style_collapse_rate"],
                "style_preservation": style_eval["average_preservation"],
                "preservation_details": style_eval["preservation_per_dimension"]
            }
            all_results.append(result)

            print(f"  Critic: scores={critic_report.get('scores')}")
            print(f"  Judge: {judge_result.get('decision')}")
            print(f"  风格坍缩率: {style_eval['style_collapse_rate']:.2%}")
            print(f"  风格保持度: {style_eval['average_preservation']:.2%}")
        except Exception as e:
            print(f"⚠️ 第 {i+1} 块处理失败，跳过: {e}")
            all_results.append({"chunk_id": chunk_id, "error": str(e)})
        done = i + 1
        print(f"  📊 [{done}/{len(chunks)}] 完成 ({done/max(len(chunks),1)*100:.0f}%)")

    # 汇总报告
    print("\n" + "=" * 60)
    print("📊 黄金测试集汇总报告")
    print("=" * 60)

    total_chunks = len(all_results)
    error_count = sum(1 for r in all_results if "error" in r)
    # 仅对成功处理的块计算聚合指标，跳过错误条目避免 KeyError
    valid_results = [r for r in all_results if "error" not in r]
    pass_count = sum(1 for r in valid_results if r["judge_decision"] == "PASS")
    avg_collapse = sum(r["style_collapse_rate"] for r in valid_results) / max(len(valid_results), 1)
    avg_preservation = sum(r["style_preservation"] for r in valid_results) / max(len(valid_results), 1)

    print(f"总块数: {total_chunks}")
    print(f"错误数: {error_count}")
    print(f"通过数: {pass_count} ({pass_count/max(total_chunks,1)*100:.1f}%)")
    print(f"平均风格坍缩率: {avg_collapse:.2%}")
    print(f"平均风格保持度: {avg_preservation:.2%}")

    dims = ["avg_sentence_length", "vocabulary_density", "rhetorical_density", "punctuation_density"]
    for dim in dims:
        dim_avg = sum(r.get("preservation_details", {}).get(dim, 0) for r in valid_results) / max(len(valid_results), 1)
        print(f"  {dim}: {dim_avg:.2%}")

    output_file = root_dir / "output" / "golden_test_report.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_chunks": total_chunks,
                "error_count": error_count,
                "pass_rate": pass_count / max(total_chunks, 1),
                "avg_style_collapse_rate": avg_collapse,
                "avg_style_preservation": avg_preservation
            },
            "details": all_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细报告已保存: {output_file}")
    return all_results


# ----- 8.2 Memory Pressure Tests -----

def get_system_memory() -> dict:
    """获取系统内存信息"""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent": mem.percent
    }


def get_process_memory() -> dict:
    """获取当前进程内存信息"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_gb": mem_info.rss / (1024**3),
        "vms_gb": mem_info.vms / (1024**3),
        "percent": process.memory_percent()
    }


def simulate_memory_load(target_gb: float):
    """模拟内存负载（分配大量内存）"""
    print(f"📦 分配 {target_gb:.1f}GB 内存模拟负载...")
    data = bytearray(int(target_gb * 1024 * 1024 * 1024))
    return data


def run_golden_gate():
    """Golden Gate：基于人工评分基线的动态阈值质量门禁
    
    1. 加载人工评分基线（如果存在）
    2. 运行 golden_test 获取 pipeline 实际 pass_rate
    3. 动态阈值 = max(0.7, human_baseline_pass_rate * 0.9) 或 fallback 0.8
    4. 比较 pipeline pass_rate 与动态阈值
    
    Returns:
        bool: True=通过质量门禁, False=未通过
    """
    import math
    
    root_dir = Path(__file__).resolve().parent.parent
    baseline_file = root_dir / "docs" / "golden_human_baseline.json"
    
    # 加载人工评分基线
    human_scores = {}
    meta = {}
    if baseline_file.exists():
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        baseline_records = baseline_data.get("baseline", [])
        if baseline_records:
            human_scores = {rec["chunk_id"]: rec["human_score"] for rec in baseline_records}
        meta = baseline_data.get("metadata", {})
        print(f"📋 加载人工评分基线: {len(human_scores)} 条记录")
    else:
        print(f"⚠️ 未找到人工评分基线文件: {baseline_file}")
    
    # 运行 golden_test 获取 pipeline 实际结果
    print("\n🧪 运行 Golden Test 获取 pipeline 质量数据...")
    results = run_golden_test()
    if not results:
        print("❌ Golden Test 无法运行（测试文件可能不存在），门禁无法判定")
        return False
    valid = [r for r in results if "error" not in r]
    total_valid = len(valid)
    
    if total_valid == 0:
        print("❌ Golden Test 无有效结果，门禁无法判定")
        return False
    
    # 计算 pipeline pass_rate
    pass_count = sum(1 for r in valid if r["judge_decision"] == "PASS")
    pipeline_pass_rate = pass_count / total_valid
    
    # 计算各 chunk 的 pipeline 平均分（critic_scores 的五维均值）
    pipeline_avg_scores = {}
    for r in valid:
        scores = r.get("critic_scores", {})
        if scores:
            vals = [v for v in scores.values() if isinstance(v, (int, float))]
            pipeline_avg_scores[r["chunk_id"]] = sum(vals) / len(vals) if vals else 0.0
    
    # 计算 Pearson 相关系数（人工评分 vs pipeline 平均分）
    common_chunks = [cid for cid in human_scores if cid in pipeline_avg_scores]
    pearson_r = 0.0
    if len(common_chunks) >= 3:
        n = len(common_chunks)
        h_vals = [human_scores[cid] for cid in common_chunks]
        p_vals = [pipeline_avg_scores[cid] for cid in common_chunks]
        sum_h = sum(h_vals)
        sum_p = sum(p_vals)
        sum_hp = sum(h * p for h, p in zip(h_vals, p_vals))
        sum_h2 = sum(h * h for h in h_vals)
        sum_p2 = sum(p * p for p in p_vals)
        denom = math.sqrt((n * sum_h2 - sum_h * sum_h) * (n * sum_p2 - sum_p * sum_p))
        if denom > 0:
            pearson_r = (n * sum_hp - sum_h * sum_p) / denom
        print(f"📊 Pearson 相关系数 (人工 vs Pipeline): {pearson_r:.3f} ({n} 个共同样本)")
    else:
        print(f"📊 共同样本不足 ({len(common_chunks)} < 3)，跳过相关系数计算")
    
    # 确定动态阈值
    if human_scores:
        acceptance_threshold = meta.get("acceptance_threshold", 7)
        human_acceptable = sum(1 for s in human_scores.values() if s >= acceptance_threshold)
        total_human = len(human_scores)
        human_pass_rate = human_acceptable / total_human if total_human > 0 else 0
        dynamic_threshold = max(0.7, human_pass_rate * 0.9)
        print(f"📊 人工基线通过率: {human_pass_rate:.1%} ({human_acceptable}/{total_human} ≥ {acceptance_threshold})")
        print(f"🎯 动态阈值: {dynamic_threshold:.1%} = max(0.7, {human_pass_rate:.1%} * 0.9)")
    else:
        dynamic_threshold = 0.8
        print(f"⚠️ 无人工基线，使用默认阈值 {dynamic_threshold:.0%}（仅供参考）")
    
    print(f"\n📊 Pipeline 实际 pass_rate: {pipeline_pass_rate:.1%} ({pass_count}/{total_valid})")
    
    if pipeline_pass_rate >= dynamic_threshold:
        print(f"✅ Golden Gate PASSED: pipeline {pipeline_pass_rate:.1%} ≥ 阈值 {dynamic_threshold:.1%}")
        return True
    else:
        print(f"❌ Golden Gate FAILED: pipeline {pipeline_pass_rate:.1%} < 阈值 {dynamic_threshold:.1%}")
        return False


def test_memory_pressure():
    """运行 5 项内存压力测试"""
    print("=" * 60)
    print("🧪 OpenLiterary 内存压力测试 (16GB 限制)")
    print("=" * 60)

    sys_mem = get_system_memory()
    print(f"💻 系统内存: {sys_mem['total_gb']:.1f}GB 总计, {sys_mem['available_gb']:.1f}GB 可用")

    client = get_llm_client_for_task("reference_extraction")
    print(f"🤖 当前后端: {_config.llm_backend}")

    print("\n📊 测试 1: 基础内存监控")
    proc_mem = get_process_memory()
    print(f"  进程内存: RSS={proc_mem['rss_gb']:.2f}GB, VMS={proc_mem['vms_gb']:.2f}GB")

    print("\n⚡ 测试 2: 生成性能基线")
    test_prompt = "请将以下文本翻译成中文：The Shrike is not a god, nor a demon, nor even a machine."
    start = time.time()
    result = client.generate(test_prompt, model_name="test-model", max_tokens=100, temperature=0.3)
    elapsed = (time.time() - start) * 1000
    print(f"  生成耗时: {elapsed:.0f}ms")
    print(f"  结果长度: {len(result)} 字符")
    proc_mem = get_process_memory()
    print(f"  生成后内存: RSS={proc_mem['rss_gb']:.2f}GB")

    if sys_mem['available_gb'] > 4:
        print("\n🔥 测试 3: 内存压力模拟")
        load_data = simulate_memory_load(2.0)
        proc_mem = get_process_memory()
        print(f"  负载后进程内存: RSS={proc_mem['rss_gb']:.2f}GB ({proc_mem['percent']:.1f}%)")
        if hasattr(client, 'check_memory_pressure'):
            pressure = client.check_memory_pressure()
            print(f"  内存压力检测: {'触发' if pressure else '正常'}")
        del load_data
        gc.collect()
        time.sleep(1)
        proc_mem = get_process_memory()
        print(f"  释放后进程内存: RSS={proc_mem['rss_gb']:.2f}GB")
    else:
        print("\n⚠️ 系统可用内存不足，跳过内存压力模拟")

    print("\n🔄 测试 4: 连续生成压力测试 (10次)")
    total_tokens = 0
    total_time = 0
    for i in range(10):
        prompt = f"测试 {i+1}: 翻译科幻片段。The Shrike waited in the shadows."
        start = time.time()
        result = client.generate(prompt, model_name="test-model", max_tokens=50, temperature=0.3)
        elapsed = (time.time() - start) * 1000
        tokens = len(result.split())
        total_tokens += tokens
        total_time += elapsed
        if i % 3 == 0:
            proc_mem = get_process_memory()
            print(f"  第 {i+1} 次: {elapsed:.0f}ms, {tokens} tokens, 内存={proc_mem['rss_gb']:.2f}GB")
    avg_time = total_time / 10
    avg_tok_s = total_tokens / (total_time / 1000)
    print(f"  平均耗时: {avg_time:.0f}ms")
    print(f"  平均吞吐: {avg_tok_s:.1f} tok/s")
    print(f"  总生成: {total_tokens} tokens")

    print("\n🧹 测试 5: 模型卸载机制")
    if hasattr(client, 'unload_model'):
        mem_before = get_process_memory()
        print(f"  卸载前: RSS={mem_before['rss_gb']:.2f}GB")
        client.unload_model()
        gc.collect()
        time.sleep(0.5)
        mem_after = get_process_memory()
        print(f"  卸载后: RSS={mem_after['rss_gb']:.2f}GB")
        print(f"  释放内存: {mem_before['rss_gb'] - mem_after['rss_gb']:.2f}GB")
    else:
        print("  当前后端不支持模型卸载 (Mock/OpenAI API 模式)")

    print("\n" + "=" * 60)
    print("✅ 内存压力测试完成")
    print("=" * 60)


# ────────────────────────────────────────────────────────────────────
# Section 9 — 辅助入口
# ────────────────────────────────────────────────────────────────────

def init_project(chapter_id: str = "ch01", force: bool = False):
    """初始化翻译项目：切分文本并写入任务数据库

    Args:
        chapter_id: 章节 ID
        force: 是否覆盖已存在的 chunk（同时重置 text_content 和 retries）
    """
    root_dir = Path(__file__).resolve().parent.parent
    db_dir = root_dir / "db"
    db_file = db_dir / "workflow.db"
    input_file = root_dir / "input" / f"{chapter_id}.md"

    if not input_file.exists():
        print(f"❌ 找不到文件: {input_file}")
        return

    db_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    scheduler = TaskScheduler(db_path=str(db_file))

    # 使用配置的 chunker 参数
    chunker_cfg = _config.chunker
    chunks = SmartChunker(
        soft_limit=chunker_cfg.get("soft_limit", 1000),
        hard_limit=chunker_cfg.get("hard_limit", 2500)
    ).split_markdown(markdown_text)
    if not chunks:
        print("❌ 警告：没有切分出任何 chunk！")
        return

    scheduler.init_chapter_tasks(chapter_id=chapter_id, chunks=chunks, force=force)

    # --force：清理上一轮残留的 orphan final.json（避免 stale 文件骗过产出校验）
    if force:
        out_dir = root_dir / "output" / chapter_id
        if out_dir.is_dir():
            valid_ids = {f"{chapter_id}_chunk{i:03d}" for i in range(len(chunks))}
            for f in out_dir.glob("*_final.json"):
                fid = f.stem.rsplit("_final", 1)[0]
                if fid not in valid_ids:
                    f.unlink()
                    print(f"  🧹 清理残留 final: {f.name}")

    print(f"\n✅ 初始化成功，数据已写入: {db_file}")


def debug_db():
    """调试工具：查看数据库中的任务状态"""
    root_dir = Path(__file__).resolve().parent.parent
    db_path = root_dir / "db" / "workflow.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, state, retries, quality_retries, last_error FROM chunk_tasks")
    tasks = cursor.fetchall()

    print(f"数据库中任务总数: {len(tasks)}")
    for chunk_id, state, retries, qr, last_error in tasks:
        print(f"  {chunk_id} | {state} | retries={retries} | quality_retries={qr}")
        if last_error:
            print(f"    last_error: {last_error[:120]}")
    conn.close()


# ────────────────────────────────────────────────────────────────────
# Section 10 — 统一入口
# ────────────────────────────────────────────────────────────────────

def print_banner():
    """横幅输出到 stderr，避免污染 stdout（stdout 留给机器可解析的输出，如章节路径）"""
    # P4-2: 动态统计 Section 数（1, 2, ..., 11, 11.5, 12 = 12 个含子节）
    _src_path = Path(__file__).resolve()
    try:
        _src_text = _src_path.read_text(encoding="utf-8")
        _section_count = len(re.findall(r'^# Section \d+(?:\.\d+)? —', _src_text, re.MULTILINE))
    except Exception:
        _section_count = 12  # fallback
    print(rf"""
   ____                   __    _ __
  / __ \____  ___  ____  / /   (_) /____  _________ ________  __
 / / / / __ \/ _ \/ __ \/ /   / / __/ _ \/ ___/ __ `/ ___/ / / /
/ /_/ / /_/ /  __/ / / / /___/ / /_/  __/ /  / /_/ / /  / /_/ /
\____/ .___/\___/_/ /_/_____/_/\__/\___/_/   \__,_/_/   \__, /
    /_/                                                /____/

  OpenLiterary — AI 文学语义编译系统  (单体聚合版)
  共 {_section_count} 个模块 | 用于全量代码审计 / 单文件部署
""", file=sys.stderr)

def main():
    print_banner()

    import argparse
    parser = argparse.ArgumentParser(description="OpenLiterary — AI 文学语义编译系统")
    parser.add_argument("command", nargs="?", default="pipeline",
                        choices=["pipeline", "init", "golden", "golden-gate", "memory", "debug", "split", "consistency"],
                        help="执行命令: pipeline (默认), init, golden, golden-gate, memory, debug, split, consistency")
    parser.add_argument("--chapter", default="ch01", help="章节 ID (默认: ch01)")
    parser.add_argument("--force", action="store_true", help="强制重新初始化（覆盖已存在的 chunk，重置状态和重试计数）")
    # 新增：split 命令专用参数
    parser.add_argument("--input", help="split 命令：输入文件路径（EPUB/TXT/MD）")
    parser.add_argument("--input-dir", default="input", help="split 命令：输出目录（默认 input/）")
    parser.add_argument("--input-format", choices=["auto", "epub", "txt", "md"], default="auto",
                        help="split 命令：输入格式（默认 auto=按扩展名自动判断）")
    parser.add_argument("--chapter-size", type=int, default=5000,
                        help="split 命令：TXT/MD 均分模式下的目标章节字数（默认 5000）")
    parser.add_argument("--min-chapter-size", type=int, default=1000,
                        help="split 命令：短于此字数的章节会被合并到下一章（默认 1000）")
    parser.add_argument("--apply", action="store_true",
                        help="consistency 命令：执行自动修正（默认仅生成差异报告）")

    args = parser.parse_args()

    # P0-4: 章节 ID 合法性校验（路径遍历防御）
    if args.chapter and not re.match(r'^[a-zA-Z0-9_-]{1,32}$', args.chapter):
        print(f"❌ 非法 chapter_id: '{args.chapter}' (仅允许字母/数字/下划线/连字符，1-32 字符)", file=sys.stderr)
        sys.exit(1)

    # P4-9: 顶层异常包装（除已规范的 consistency/golden-gate 外，统一捕获并退出）
    try:
        if args.command == "pipeline":
            pipeline = TranslationPipeline(chapter_id=args.chapter)
            if not pipeline.run():
                sys.exit(1)
        elif args.command == "init":
            init_project(chapter_id=args.chapter, force=args.force)
        elif args.command == "golden":
            run_golden_test()
        elif args.command == "golden-gate":
            sys.exit(0 if run_golden_gate() else 1)
        elif args.command == "memory":
            test_memory_pressure()
        elif args.command == "debug":
            debug_db()
        elif args.command == "consistency":
            n_issues = run_consistency_check(dry_run=not args.apply)
            sys.exit(0 if n_issues == 0 else 2)
        elif args.command == "split":
            if not args.input:
                # P0-3: split 提示命令按 __package__ 自适应
                if __package__:
                    cmd_hint = f"python -m {__package__} split --input <file.epub|txt|md>"
                else:
                    cmd_hint = "python translator_agent.py split --input <file.epub|txt|md>"
                print(f"❌ split 命令需要 --input 参数", file=sys.stderr)
                print(f"   用法: {cmd_hint}", file=sys.stderr)
                sys.exit(1)
            split_input_to_chapters(
                input_path=args.input,
                output_dir=args.input_dir,
                input_format=args.input_format,
                target_chars=args.chapter_size,
                min_chars=args.min_chapter_size,
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        # P4-9: 顶层异常统一退出（保留 KeyboardInterrupt/SystemExit 上抛）
        print(f"❌ 命令 {args.command} 异常终止: {e}", file=sys.stderr)
        import traceback as _tb_p49
        _tb_p49.print_exc()
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────
# Section 11 — 输入格式适配器 (EpubSplitter / TextSplitter / MdSplitter)
# ─────────────────────────────────────────────────────────────────────────────
# 职责：将原始输入（EPUB / TXT / MD）切分为多个章节 Markdown 文件，
#       输出到 input/chXX.md，供 SmartChunker 二次切分。
# 注意：本节为输入适配层，与翻译管线业务逻辑完全解耦。

def _splitter_log_info(msg: str):
    print(f"[INFO] {msg}", file=sys.stderr)


def _splitter_log_ok(msg: str):
    print(f"[OK] {msg}", file=sys.stderr)


def _splitter_log_warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)


def _splitter_log_err(msg: str):
    print(f"[ERR] {msg}", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────
# 11.1 EpubSplitter — EPUB → 章节 Markdown
# ─────────────────────────────────────────────────────────────────────────────

# EPUB 解析依赖（懒加载，缺失时给出友好提示）
try:
    import ebooklib
    from ebooklib import epub as _epub_lib
    from bs4 import BeautifulSoup
    _EPUB_DEPS_OK = True
except ImportError as _e_dep:
    _EPUB_DEPS_OK = False
    _EPUB_DEPS_ERROR = str(_e_dep)


def _normalize_chapter_title(title: str) -> str:
    """规范章节标题：在「序号.」与后续正文之间补空格。

    源 EPUB 常见 `II.The Machine` / `1.Introduction` 这类序号与英文紧接、
    无空格的形式，拼成 Markdown 标题后观感差。统一规范为
    `II. The Machine` / `1. Introduction`。同时处理「第N章xxx」保持原样。
    """
    if not title:
        return title
    # 形如 "II.The" / "1.Introduction" / "XV.The Time Traveller's Return"
    # 在「字母数字序号 + 英文句点」后、紧接大写字母处插入空格
    fixed = re.sub(r'^([A-Za-z0-9]+)\.([A-Z])', r'\1. \2', title.strip())
    return fixed


# Critic LLM 偶发把维度名拼错（如 voice_consistency → voice_consistence），
# 导致下游 Judge 阈值查表时该脏维度被静默忽略。此处用白名单归一化，
# 把已知拼写变体映射回规范维度名，未命中白名单的键原样保留。
_SCORE_KEY_NORMALIZATION = {
    "voice_consistence": "voice_consistency",
    "voice_consistancy": "voice_consistency",
    "voice consistence": "voice_consistency",
    "style_complience": "style_compliance",
    "style_compliance_": "style_compliance",
    "semantic_preservation": "semantic_preservation",
    "semantic_preserv": "semantic_preservation",
    "fluency": "fluency",
    "readability": "readability",
}


def _normalize_scores(scores: Optional[dict]) -> dict:
    """归一化 Critic 评分维度名，剔除脏维度名对 Judge 阈值逻辑的污染。

    只处理 dict 且值为数值（int/float）的项；非数值或空值原样跳过，
    避免把 critique 文本当分数累加。
    """
    if not isinstance(scores, dict):
        return scores if isinstance(scores, dict) else {}
    normalized: dict = {}
    for k, v in scores.items():
        key = k
        if isinstance(k, str):
            key = _SCORE_KEY_NORMALIZATION.get(k.strip().lower(), k.strip())
        if isinstance(v, (int, float)):
            normalized[key] = v
    return normalized


def _epub_clean_html_text(html_content: str) -> str:
    """清洗 HTML，提取纯文本（保留段落结构）。

    P 方案修复：原先对每个块级标签 insert_after('\\n') 再用
    get_text(separator='\\n')，因源 XML 段内自带换行，最终整章塌成 1 段。
    改为逐个提取块级元素文本、以 '\\n\\n' 拼接，确定性还原段落边界，
    且对段内换行免疫。
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    for br in soup.find_all('br'):
        br.replace_with('\n')

    # 仅取 p 与标题标签作为段落边界；不含 div（div 常为容器，
    # 其 get_text 会把整章压成一段，反而丢失段落结构）
    blocks = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    if not blocks:
        # 无块级标签时退化为整体文本（保留段内换行）
        text = soup.get_text('\n')
    else:
        parts = [b.get_text(' ', strip=True) for b in blocks]
        parts = [p for p in parts if p]
        text = '\n\n'.join(parts)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 注意：不能用 \s（含 \n）配合 MULTILINE 去修整行首尾，否则会把段落间的
    # \n\n 也吃掉、重新塌成 1 段。仅修剪行首尾的空格/制表符。
    text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', text)
    return text.strip()


# Q2 修复：古腾堡计划 EPUB 头部含许可证公告 + 目录（"Contents"/"目录"），
# 这些非正文文本若随章节被翻译，会污染最终译文。此处按常见特征裁剪。
_GUTENBERG_HEADER_PATTERNS = [
    re.compile(r'Project Gutenberg', re.IGNORECASE),
    re.compile(r'www\.gutenberg\.org', re.IGNORECASE),
    re.compile(r'This eBook is for the use of anyone'),
    re.compile(r'古腾堡计划', re.IGNORECASE),
]
# 目录标题行：单独出现即为目录标志（需配合紧随其后的编号条目块才裁剪）
_GUTENBERG_TOC_TITLE_PATTERN = re.compile(
    r'^\s*(contents|table of contents|目录)\s*$', re.IGNORECASE | re.MULTILINE)
# 编号目录项：罗马数字（I./II.）或阿拉伯数字（1./Chapter 1）开头的行
_GUTENBERG_TOC_ITEM_PATTERN = re.compile(r'^\s*(?:[IVXLC]+\.?|\d+\.?|chapter\s+\d+)\s', re.IGNORECASE | re.MULTILINE)


def _strip_epub_front_matter(text: str) -> str:
    """裁剪古腾堡计划 EPUB 头部的许可证公告与目录块，仅保留正文。

    安全策略（不误删正文）：
    - 仅跳过含「古腾堡版权/许可证」特征的块；
    - 仅当某个块是「目录标题（Contents/目录）」且紧随其后的块含 ≥2 个
      编号目录项时，才把该目录块一并跳过。
    单处出现的 "I. The Time Machine" 等正文段落不会被误判为目录。
    """
    if not text:
        return text
    blocks = re.split(r'\n\s*\n', text)

    start_idx = 0
    # 1) 跳过连续的首部版权/许可证块
    for i, block in enumerate(blocks):
        if any(p.search(block) for p in _GUTENBERG_HEADER_PATTERNS):
            start_idx = i + 1
        else:
            break
    if start_idx >= len(blocks):
        return ""

    # 2) 若下一块是目录标题 + 再下一块含 ≥2 编号项 → 跳过目录块
    toc_title_idx = start_idx
    if toc_title_idx + 1 < len(blocks) and _GUTENBERG_TOC_TITLE_PATTERN.search(blocks[toc_title_idx]):
        item_block = blocks[toc_title_idx + 1]
        item_count = len(_GUTENBERG_TOC_ITEM_PATTERN.findall(item_block))
        if item_count >= 2:
            start_idx = toc_title_idx + 2

    return '\n\n'.join(blocks[start_idx:]).strip()


def _epub_split_large_doc_by_toc(html_content: str, toc_entries: list, chapter_idx_start: int) -> list:
    """
    将单个超大 HTML 文档按 TOC 锚点切分。
    返回 [(chapter_id, markdown), ...]
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    toc_anchors = []
    for entry in toc_entries:
        href = getattr(entry, 'href', '') or ''
        if isinstance(href, str) and '#' in href:
            anchor_id = href.split('#', 1)[1]
            title = getattr(entry, 'title', '') or ''
            toc_anchors.append((title, anchor_id))

    if not toc_anchors:
        return []

    lines = html_content.split('\n')
    anchor_lines = {}
    for i, line in enumerate(lines):
        for title, anchor_id in toc_anchors:
            if f'id="{anchor_id}"' in line or f"id='{anchor_id}'" in line or f'name="{anchor_id}"' in line or f"name='{anchor_id}'" in line:
                if anchor_id not in anchor_lines:
                    anchor_lines[anchor_id] = (i, title)

    sorted_anchors = sorted(anchor_lines.items(), key=lambda x: x[1][0])
    chapters = []
    for i, (anchor_id, (line_no, toc_title)) in enumerate(sorted_anchors):
        start_line = line_no
        end_line = sorted_anchors[i + 1][1][0] if i + 1 < len(sorted_anchors) else len(lines)
        chunk_html = '\n'.join(lines[start_line:end_line])

        title = toc_title or f"Section {i+1}"

        chunk_soup = BeautifulSoup(chunk_html, 'html.parser')
        # P 方案：与 _epub_clean_html_text 一致，逐 p/标题元素以 \n\n 拼接还原段落（不含 div 容器）
        cblocks = chunk_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if cblocks:
            cparts = [b.get_text(' ', strip=True) for b in cblocks]
            cparts = [p for p in cparts if p]
            text = '\n\n'.join(cparts)
        else:
            text = chunk_soup.get_text('\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', text)
        text = text.strip()
        # Q2：剔除古腾堡头部许可证/目录，避免非正文进入译文
        text = _strip_epub_front_matter(text)

        if not text or len(text) < 50:
            continue

        chapter_idx = chapter_idx_start + len(chapters) + 1
        ch_id = f"ch{chapter_idx:02d}"
        chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))

    return chapters


def _epub_extract_chapters(book) -> list:
    """从 EPUB 提取章节，返回 [(chapter_id, markdown), ...]"""
    chapters = []
    chapter_idx = 0

    toc_entries = []
    try:
        for item in book.toc:
            toc_entries.append(item)
            if isinstance(item, tuple) and len(item) == 2:
                section, sub_items = item
                toc_entries.append(section)
                if isinstance(sub_items, list):
                    toc_entries.extend(sub_items)
    except Exception:
        pass

    LARGE_DOC_THRESHOLD = 50000

    for item_id in book.spine:
        if isinstance(item_id, tuple):
            item_id = item_id[0]
        item = book.get_item_with_id(item_id)
        if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        html_content = item.get_content().decode('utf-8', errors='ignore')
        text = _epub_clean_html_text(html_content)
        # Q2：剔除古腾堡头部许可证/目录，避免非正文进入译文
        text = _strip_epub_front_matter(text)

        if not text or len(text) < 50:
            continue

        if len(text) <= LARGE_DOC_THRESHOLD:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
            title = title_tag.get_text(strip=True) if title_tag else f"Chapter {chapter_idx}"
            chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))
            continue

        sub_chapters = _epub_split_large_doc_by_toc(html_content, toc_entries, chapter_idx)
        if sub_chapters:
            chapters.extend(sub_chapters)
            chapter_idx += len(sub_chapters)
        else:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
            title = title_tag.get_text(strip=True) if title_tag else f"Chapter {chapter_idx}"
            chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))

    return chapters


class EpubSplitter:
    """EPUB → 章节 Markdown 切分器

    用法:
        splitter = EpubSplitter("book.epub", "input/")
        chapters = splitter.split()  # 返回 [(ch_id, md_text), ...]
        splitter.write_files()     # 写入磁盘
    """

    def __init__(self, epub_path: str, output_dir: str):
        if not _EPUB_DEPS_OK:
            raise RuntimeError(
                f"❌ 缺少依赖: {_EPUB_DEPS_ERROR}\n"
                "请先安装: pip install ebooklib beautifulsoup4 lxml"
            )
        self.epub_path = Path(epub_path)
        self.output_dir = Path(output_dir)
        self.book = None
        self.chapters: list = []

    def load(self):
        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB 文件不存在: {self.epub_path}")
        _splitter_log_info(f"读取 EPUB: {self.epub_path}")
        self.book = _epub_lib.read_epub(str(self.epub_path))

        title = self.book.get_metadata('DC', 'title')
        author = self.book.get_metadata('DC', 'creator')
        if title:
            _splitter_log_info(f"书名: {title[0][0]}")
        if author:
            _splitter_log_info(f"作者: {author[0][0]}")
        return self

    def split(self) -> list:
        """切分章节，返回 [(ch_id, markdown), ...]"""
        if self.book is None:
            self.load()
        self.chapters = _epub_extract_chapters(self.book)
        if not self.chapters:
            _splitter_log_warn("未提取到任何章节内容")
        else:
            _splitter_log_ok(f"共提取 {len(self.chapters)} 章")
            self._validate_split()
        return self.chapters

    def _validate_split(self):
        """P4：切分完整性 + 段落结构回归校验。

        - 完整性：提取纯文本总字符应≈EPUB 文档总字符（±5% 空格归一化容差），
          防止内容丢失。
        - 段落结构：每章按空行拆出的段落数应>1（除非原文确实极短），
          防止再次出现"整章塌成 1 段"的回归。
        """
        try:
            import ebooklib
            # 基线：对每一个被提取器实际处理的 spine 文档跑同样的清洗，
            # 累加字符数（与 _epub_extract_chapters 口径一致，跳过 <50 字符文档）。
            epub_total = 0
            for item_id in self.book.spine:
                iid = item_id[0] if isinstance(item_id, tuple) else item_id
                it = self.book.get_item_with_id(iid)
                if not it or it.get_type() != ebooklib.ITEM_DOCUMENT:
                    continue
                html = it.get_content().decode('utf-8', errors='ignore')
                cleaned = _epub_clean_html_text(html)
                if len(cleaned) >= 50:
                    epub_total += len(cleaned)
            extracted_total = sum(len(md) for _, md in self.chapters)
            if epub_total > 0:
                ratio = extracted_total / epub_total
                if ratio < 0.95:
                    _splitter_log_err(
                        f"完整性异常：提取字符 {extracted_total} 仅占 EPUB {epub_total} 的 {ratio*100:.1f}%，疑似内容丢失"
                    )
                else:
                    _splitter_log_ok(f"完整性校验通过：提取 {extracted_total}/{epub_total} 字符 ({ratio*100:.1f}%)")

            single_para = 0
            for ch_id, md in self.chapters:
                paras = [p for p in re.split(r'\n{2,}', md) if p.strip()]
                if len(paras) <= 1 and len(md) > 500:
                    single_para += 1
            if single_para > 0:
                _splitter_log_warn(
                    f"段落结构预警：{single_para} 章疑似仍塌缩为单段落（>500字符却无空行分段），请检查切分逻辑"
                )
            else:
                _splitter_log_ok("段落结构校验通过：各章均含多段落边界")
        except Exception as e:
            _splitter_log_warn(f"校验跳过（非致命）：{e}")

    def write_files(self) -> list:
        """将切分结果写入 output_dir/chXX.md，返回生成的文件路径列表"""
        if not self.chapters:
            self.split()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for ch_id, markdown in self.chapters:
            out_file = self.output_dir / f"{ch_id}.md"
            out_file.write_text(markdown, encoding='utf-8')
            generated.append(str(out_file))
            _splitter_log_ok(f"{ch_id}.md ({len(markdown)} chars)")

        summary = self.output_dir / "_chapters_summary.txt"
        summary.write_text(
            "\n".join(
                f"{ch_id}\t{Path(f).stat().st_size} bytes"
                for ch_id, f in zip([c[0] for c in self.chapters], generated)
            ),
            encoding='utf-8'
        )
        _splitter_log_info(f"汇总: {summary}")
        return generated


# ────────────────────────────────────────────────────────────────────
# 11.2 TextSplitter — 纯文本 → 章节 Markdown
# ─────────────────────────────────────────────────────────────────────────────

# 章节标题正则（按优先级匹配）
# 支持中英文常见模式
_CHAPTER_TITLE_PATTERNS = [
    re.compile(r'^\s*(?:Chapter|CHAPTER|第[一二三四五六七八九十百千零\d]+章|序章|楔子|引子|前言|后记|尾声)\s*[\s\d.、:：\-—]*[^\n]*$', re.IGNORECASE),
    re.compile(r'^\s*\[(?:CHAPTER|Chapter|第[一二三四五六七八九十百千零\d]+章)[^\]]*\]\s*$', re.IGNORECASE),
    re.compile(r'^\s*#{1,2}\s*[^\n]+$'),  # Markdown H1/H2
]


def _text_detect_chapter_title(line: str) -> bool:
    """判断一行是否为章节标题"""
    if len(line) > 80:
        return False
    for pat in _CHAPTER_TITLE_PATTERNS:
        if pat.match(line):
            return True
    return False


class TextSplitter:
    """纯文本 → 章节 Markdown 切分器

    策略（按优先级）:
      1. 检测显式章节标记（"Chapter N" / "第N章" / Markdown H1/H2）
      2. 若未找到任何标记，按 target_chars 均分
      3. 段落内连续空行作为段落边界

    参数:
      text_path: 输入 TXT 文件路径
      output_dir: 输出目录
      target_chars: 均分模式下的目标章节字数（默认 5000）
      min_chars: 短于该字数的章节会被合并到下一章（默认 1000）
    """

    def __init__(self, text_path: str, output_dir: str,
                 target_chars: int = 5000, min_chars: int = 1000):
        self.text_path = Path(text_path)
        self.output_dir = Path(output_dir)
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.chapters: list = []

    def load_text(self) -> str:
        if not self.text_path.exists():
            raise FileNotFoundError(f"文本文件不存在: {self.text_path}")
        _splitter_log_info(f"读取文本: {self.text_path}")
        text = self.text_path.read_text(encoding='utf-8', errors='ignore')
        _splitter_log_info(f"总字符数: {len(text)}")
        return text

    def _find_title_lines(self, text: str) -> list:
        """扫描全文，定位所有章节标题行号"""
        title_lines = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if _text_detect_chapter_title(line):
                title_lines.append((i, line.strip()))
        return title_lines, lines

    def split(self) -> list:
        """切分章节，返回 [(ch_id, markdown), ...]"""
        text = self.load_text()

        title_lines, lines = self._find_title_lines(text)

        if title_lines:
            return self._split_by_titles(text, title_lines, lines)
        else:
            _splitter_log_warn("未发现显式章节标记，按字数均分")
            return self._split_by_size(text)

    def _split_by_titles(self, text: str, title_lines: list, lines: list) -> list:
        """按显式章节标记切分"""
        chapters = []
        for i, (line_no, title) in enumerate(title_lines):
            start_line = line_no
            end_line = title_lines[i + 1][0] if i + 1 < len(title_lines) else len(lines)
            body = '\n'.join(lines[start_line:end_line]).strip()

            if not body or len(body) < 50:
                continue

            chapter_idx = len(chapters) + 1
            ch_id = f"ch{chapter_idx:02d}"
            markdown = f"# {_normalize_chapter_title(title)}\n\n{body}"
            chapters.append((ch_id, markdown))

        # 若显式切分后章节太少（或每章过大），再尝试二次均分
        if len(chapters) < 2 and len(text) > self.target_chars * 2:
            _splitter_log_warn("显式标记过少，启用均分兜底")
            return self._split_by_size(text)

        return chapters

    def _split_by_size(self, text: str) -> list:
        """按目标字数均分"""
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if not paragraphs:
            return []

        chapters = []
        current_paragraphs = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)

            # 若当前章节非空、且加入新段会超 target_chars 且当前已达 min_chars，则打包
            if current_paragraphs and current_len + p_len > self.target_chars and current_len >= self.min_chars:
                chapter_text = '\n\n'.join(current_paragraphs)
                chapter_idx = len(chapters) + 1
                ch_id = f"ch{chapter_idx:02d}"
                chapters.append((ch_id, f"# Part {chapter_idx}\n\n{chapter_text}"))
                current_paragraphs = []
                current_len = 0

            current_paragraphs.append(p)
            current_len += p_len

        # 收尾
        if current_paragraphs:
            chapter_text = '\n\n'.join(current_paragraphs)
            chapter_idx = len(chapters) + 1
            ch_id = f"ch{chapter_idx:02d}"
            chapters.append((ch_id, f"# Part {chapter_idx}\n\n{chapter_text}"))

        return chapters

    def write_files(self) -> list:
        if not self.chapters:
            self.chapters = self.split()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for ch_id, markdown in self.chapters:
            out_file = self.output_dir / f"{ch_id}.md"
            out_file.write_text(markdown, encoding='utf-8')
            generated.append(str(out_file))
            _splitter_log_ok(f"{ch_id}.md ({len(markdown)} chars)")

        summary = self.output_dir / "_chapters_summary.txt"
        summary.write_text(
            "\n".join(
                f"{ch_id}\t{Path(f).stat().st_size} bytes"
                for ch_id, f in zip([c[0] for c in self.chapters], generated)
            ),
            encoding='utf-8'
        )
        _splitter_log_info(f"汇总: {summary}")
        return generated


# ────────────────────────────────────────────────────────────────────
# 11.3 MdSplitter — 整本 Markdown → 多章节（均分 or 按 H1）
# ─────────────────────────────────────────────────────────────────────────────

class MdSplitter:
    """整本 Markdown → 多章节切分器

    策略:
      1. 按 H1 标题切分
      2. 若无 H1，则按 H2 切分
      3. 若无任何标题，按 target_chars 均分

    用于：用户已有整本 Markdown（如从 EPUB/TXT 转出来但没分章节）
    """

    def __init__(self, md_path: str, output_dir: str,
                 target_chars: int = 5000, min_chars: int = 1000):
        self.md_path = Path(md_path)
        self.output_dir = Path(output_dir)
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.chapters: list = []

    def load(self) -> str:
        if not self.md_path.exists():
            raise FileNotFoundError(f"MD 文件不存在: {self.md_path}")
        _splitter_log_info(f"读取 MD: {self.md_path}")
        text = self.md_path.read_text(encoding='utf-8', errors='ignore')
        _splitter_log_info(f"总字符数: {len(text)}")
        return text

    def split(self) -> list:
        text = self.load()

        # 按 H1 切分
        parts = re.split(r'(?m)^# .+$', text)
        headings = re.findall(r'(?m)^# (.+)$', text)

        if len(parts) > 1 and headings:
            chapters = []
            for i, (heading, body) in enumerate(zip(headings, parts[1:])):
                body = body.strip()
                if not body or len(body) < 50:
                    continue
                chapter_idx = len(chapters) + 1
                ch_id = f"ch{chapter_idx:02d}"
                chapters.append((ch_id, f"# {heading.strip()}\n\n{body}"))
            if chapters:
                return chapters

        # 兜底：复用 TextSplitter 的均分逻辑
        _splitter_log_warn("未发现 H1 标题，启用均分兜底")
        fake_path = self.md_path.with_suffix('.txt')
        fake_text = TextSplitter(
            str(fake_path), str(self.output_dir),
            target_chars=self.target_chars, min_chars=self.min_chars
        )
        # 直接借用均分方法
        return fake_text._split_by_size(text)

    def write_files(self) -> list:
        if not self.chapters:
            self.chapters = self.split()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for ch_id, markdown in self.chapters:
            out_file = self.output_dir / f"{ch_id}.md"
            out_file.write_text(markdown, encoding='utf-8')
            generated.append(str(out_file))
            _splitter_log_ok(f"{ch_id}.md ({len(markdown)} chars)")

        summary = self.output_dir / "_chapters_summary.txt"
        summary.write_text(
            "\n".join(
                f"{ch_id}\t{Path(f).stat().st_size} bytes"
                for ch_id, f in zip([c[0] for c in self.chapters], generated)
            ),
            encoding='utf-8'
        )
        _splitter_log_info(f"汇总: {summary}")
        return generated


# ────────────────────────────────────────────────────────────────────
# Section 11.5 — 译后命名一致性校正
# ─────────────────────────────────────────────────────────────────────────────

def _load_glossary(db_path: str) -> dict:
    db = sqlite3.connect(db_path)
    entities = {}
    try:
        try:
            db.execute("SELECT variant_names FROM entity_registry LIMIT 1")
        except Exception:
            db.execute("ALTER TABLE entity_registry ADD COLUMN variant_names TEXT")
            db.commit()
        try:
            seed_entity_variants(db)
        except Exception:
            pass
        cursor = db.execute(
            "SELECT canonical_name, source_names, entity_type, confidence, variant_names "
            "FROM entity_registry WHERE confidence = 'high' AND pending_review = 0"
        )
        for row in cursor:
            entities[row[0]] = {
                "source_names": robust_json_loads(row[1], expected_type=list) if row[1] else [],
                "type": row[2],
                "confidence": row[3],
                "variant_names": robust_json_loads(row[4], expected_type=list) if row[4] else [],
            }
    except Exception as e:
        print(f"  ⚠️ 读取 entity_registry 失败: {e}", file=sys.stderr)
    db.close()
    return entities


def _load_decisions(db_path: str) -> dict:
    """从 decision_db 加载术语/典故决策"""
    db = sqlite3.connect(db_path)
    decisions = {}
    try:
        cursor = db.execute(
            "SELECT source_key, translation, level FROM decision_db WHERE level IN (1, 2)"
        )
        for row in cursor:
            decisions[row[0]] = {"translation": row[1], "level": row[2]}
    except Exception as e:
        print(f"  ⚠️ 读取 decision_db 失败: {e}", file=sys.stderr)
    db.close()
    return decisions


def _scan_finals(output_dir: str) -> list:
    """扫描 output/ 下所有 final.json，返回 [{chapter, file, text, path}]"""
    finals = []
    p = Path(output_dir)
    if not p.exists():
        return finals
    for ch_dir in sorted(p.iterdir()):
        if not ch_dir.is_dir():
            continue
        for f in sorted(ch_dir.iterdir()):
            if not f.name.endswith("_final.json"):
                continue
            # P2-1: 统一走 robust_json_loads 抗损坏 final.json
            data = robust_json_loads(f.read_text(encoding="utf-8"), expected_type=dict)
            if not data:
                print(f"  ⚠️ 跳过 {f}: JSON 解析失败", file=sys.stderr)
                continue
            text = data.get("text", "") if isinstance(data, dict) else str(data)
            finals.append({
                "chapter": ch_dir.name,
                "file": f.name,
                "text": text,
                "path": str(f),
            })
    return finals


def _find_slash_aliases(canonical: str) -> list:
    """'省长/地方市长' → ['省长', '地方市长']"""
    parts = [p.strip() for p in canonical.split("/")]
    return parts if len(parts) > 1 else []


# 中文异体漂移种子（可扩展）：canonical -> 易混用异体写法
# 通过 seed_entity_variants() 注入 entity_registry.variant_names。
SEED_ENTITY_VARIANTS = {
    "爱丽丝": ["艾丽丝"],
}


def seed_entity_variants(conn):
    """将 SEED_ENTITY_VARIANTS 合并进 entity_registry.variant_names（幂等，不覆盖既有异体）。"""
    cur = conn.cursor()
    for canonical, variants in SEED_ENTITY_VARIANTS.items():
        cur.execute("SELECT id, variant_names FROM entity_registry WHERE canonical_name = ?", (canonical,))
        row = cur.fetchone()
        if not row:
            continue
        vid, vn_json = row
        existing = robust_json_loads(vn_json, expected_type=list) if vn_json else []
        merged = list(dict.fromkeys(existing + variants))
        if merged == existing:
            continue
        cur.execute(
            "UPDATE entity_registry SET variant_names = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(merged, ensure_ascii=False), vid),
        )
    conn.commit()


def _in_quote_or_note(text: str, pos: int) -> bool:
    """判断 pos 是否位于引号内或译注/脚注区块内（英文残留误报豁免）。"""
    q = 0
    for ch in text[:pos]:
        if ch in '“”"\'‘’':
            q += 1
    if q % 2 == 1:
        return True
    pre = text[max(0, pos - 200):pos]
    if "译注" in pre or "脚注" in pre or "此处保留" in pre:
        return True
    # 脚注定义行（以 [^ 起头）内的英文源词附注明豁免
    line_start = text.rfind("\n", 0, pos)
    line_end = text.find("\n", pos)
    line = text[line_start + 1: line_end if line_end != -1 else None]
    if line.lstrip().startswith("[^"):
        return True
    return False


def _detect_issues(finals: list, entities: dict, decisions: dict) -> list:
    """
    检测三类问题：
    3a. 别名混用 —— 同一实体在同章节内混用多个译名
    3b. 英源残留 —— 英文名意外出现在中文译文中（脚注原文附注除外）
    """
    issues = []

    for fin in finals:
        text = fin["text"]

        # 3a. 实体名一致性（仅检测含 / 的多译名实体）
        for canonical, info in entities.items():
            aliases = _find_slash_aliases(canonical)
            if not aliases:
                continue  # 无 slash 别名，跳过
            all_names = [canonical] + aliases
            primary = canonical.split("/")[0].strip()  # 取 / 前作为主名
            counts = {n: text.count(n) for n in all_names}

            if counts.get(primary, 0) > 0:
                for n in aliases:
                    if counts.get(n, 0) > 0:
                        has_cjk_alias = bool(re.search(r'[\u4e00-\u9fff]', n))
                        pat_alias = re.escape(n) if has_cjk_alias else r'\b' + re.escape(n) + r'\b'
                        for m in re.finditer(pat_alias, text):
                            ctx = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
                            issues.append({
                                "chapter": fin["chapter"],
                                "file": fin["file"],
                                "path": fin["path"],
                                "position": m.start(),
                                "variant": n,
                                "canonical": primary,
                                "context": ctx,
                                "issue_type": "alias_mixed",
                            })

        # 3b. 英源词残留检测
        for canonical, info in entities.items():
            primary = canonical.split("/")[0].strip()
            for sn in info.get("source_names", []):
                # 先用 in 快速预检（无边界，仅做性能优化），再用 regex 精确匹配
                if sn.lower() in text.lower():
                    # CJK 源名用子串匹配（中文无词边界），非 CJK 用 \b 边界
                    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', sn))
                    pat = re.escape(sn) if has_cjk else r'(?<!\w)' + re.escape(sn) + r'(?!\w)'
                    for m in re.finditer(pat, text, re.IGNORECASE):
                        before = text[max(0, m.start() - 3): m.start()]
                        if "（" in before or "(" in before:
                            continue  # 脚注中的原文附注不算残留
                        if _in_quote_or_note(text, m.start()):
                            continue  # 引号内引文 / 译注中的英文源词不算残留
                        ctx = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
                        issues.append({
                            "chapter": fin["chapter"],
                            "file": fin["file"],
                            "path": fin["path"],
                            "position": m.start(),
                            "variant": sn,
                            "canonical": primary,
                            "context": ctx,
                            "issue_type": "english_residue",
                        })

        # 3c. 中文异体漂移检测（数据源：entity_registry.variant_names）
        for canonical, info in entities.items():
            variants = info.get("variant_names", [])
            if not variants:
                continue
            for v in variants:
                if v not in text:
                    continue
                for m in re.finditer(re.escape(v), text):
                    if _in_quote_or_note(text, m.start()):
                        continue
                    ctx = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
                    issues.append({
                        "chapter": fin["chapter"],
                        "file": fin["file"],
                        "path": fin["path"],
                        "position": m.start(),
                        "variant": v,
                        "canonical": canonical,
                        "context": ctx,
                        "issue_type": "cjk_variant",
                    })

    return issues


def _generate_report(finals: list, issues: list, dry_run: bool) -> str:
    """生成差异报告 Markdown"""
    lines = []
    now = datetime.now().isoformat(timespec="seconds")

    lines.append("# 译后术语一致性校正报告")
    lines.append("")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 扫描章节: {len(finals)} 个 chunk")
    lines.append(f"- 发现不一致: {len(issues)} 处")
    lines.append(f"- 模式: {'DRY RUN (仅报告)' if dry_run else '已自动修正'}")
    lines.append("")

    if issues:
        by_type: dict = {}
        for iss in issues:
            by_type.setdefault(iss["issue_type"], []).append(iss)

        type_labels = {
            "alias_mixed": "同实体混用多个别名",
            "english_residue": "英文源词残留于中文译文中",
            "cjk_variant": "中文译名异体漂移",
        }

        for t, items in by_type.items():
            label = type_labels.get(t, t)
            lines.append(f"## {label} ({len(items)} 处)")
            lines.append("")
            for item in items:
                lines.append(f"### [{item['chapter']}/{item['file']}] @ {item['position']}")
                lines.append(f"- **变体**: `{item['variant']}` → **规范**: `{item['canonical']}`")
                lines.append(f"- 上下文: ...{item['context']}...")
                lines.append("")
    else:
        lines.append("未发现任何名称不一致。")
        lines.append("")

    lines.append("---")
    lines.append("*由 consistency 子命令自动生成*")
    lines.append("")

    return "\n".join(lines)


def _apply_issues(finals: list, issues: list) -> int:
    """应用修正到 final.json，返回实际修改的文件数"""
    by_file: dict = {}
    for iss in issues:
        if iss["issue_type"] in ("alias_mixed", "english_residue", "cjk_variant"):
            by_file.setdefault(iss["path"], []).append(iss)

    patched_files = 0
    for filepath, reps in by_file.items():
        p = Path(filepath)
        # P2-1: 统一走 robust_json_loads 抗损坏 final.json
        data = robust_json_loads(p.read_text(encoding="utf-8"), expected_type=dict)
        if not data:
            continue
        text = data.get("text", "") if isinstance(data, dict) else str(data)

        reps_sorted = sorted(reps, key=lambda x: x["position"], reverse=True)
        changes = 0
        for r in reps_sorted:
            if r["variant"] != r["canonical"]:
                old = text[r["position"]:r["position"] + len(r["variant"])]
                if old == r["variant"]:
                    text = text[:r["position"]] + r["canonical"] + text[r["position"] + len(r["variant"]):]
                    changes += 1

        if changes > 0:
            if isinstance(data, dict):
                data["text"] = text
                meta = data.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                meta["consistency_patched"] = True
                meta["consistency_patches"] = meta.get("consistency_patches", 0) + changes
                meta["consistency_patched_at"] = datetime.now().isoformat(timespec="seconds")
                data["metadata"] = meta
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            patched_files += 1
            print(f"  ✅ {p.name}: {changes} 处修正")

    return patched_files


def run_consistency_check(dry_run: bool = True, check_only: bool = False,
                          db_path: str = "", output_dir: str = "") -> int:
    """
    译后一致性校正入口。
    返回发现的 issues 数量。
    """
    _root = Path(__file__).resolve().parent.parent
    if not db_path:
        db_path = str(_root / "db" / "decision_db.sqlite")
    if not output_dir:
        output_dir = str(_root / "output")

    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}", file=sys.stderr)
        print("   请先运行 init 初始化项目。", file=sys.stderr)
        return -1

    print("📖 加载术语对照表...")
    entities = _load_glossary(db_path)
    decisions = _load_decisions(db_path)
    print(f"    entity_registry: {len(entities)} 个 high-confidence 实体")
    print(f"    decision_db:     {len(decisions)} 条决策")

    print("\n🔍 扫描 final.json...")
    finals = _scan_finals(output_dir)
    if not finals:
        print("    ⚠️ 没有找到 final.json（output/ 目录为空？）")
        return 0
    print(f"    找到 {len(finals)} 个文件")

    print("\n🔎 检测名称不一致...")
    issues = _detect_issues(finals, entities, decisions)
    print(f"    发现 {len(issues)} 处")

    if check_only:
        return len(issues)

    print("\n📝 生成报告...")
    report = _generate_report(finals, issues, dry_run=dry_run)
    report_path = Path(output_dir) / "consistency_diff.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"    → {report_path}")

    if not dry_run and issues:
        print("\n✏️ 执行修正...")
        patched = _apply_issues(finals, issues)
        print(f"    修改了 {patched} 个文件")
    elif dry_run and issues:
        print("\n⏸️  DRY RUN 模式：未修改任何文件（--apply 以执行修正）")

    print("\n🏁 完成")
    return len(issues)


# ────────────────────────────────────────────────────────────────────
# Section 12 — 统一入口点 (split_input_to_chapters)
# ─────────────────────────────────────────────────────────────────────────────

# 输入格式检测表
_INPUT_FORMAT_TABLE = {
    '.epub': 'epub',
    '.txt': 'txt',
    '.md': 'md',
    '.markdown': 'md',
}


def detect_input_format(file_path: str) -> str:
    """根据文件扩展名自动检测输入格式"""
    suffix = Path(file_path).suffix.lower()
    fmt = _INPUT_FORMAT_TABLE.get(suffix)
    if fmt is None:
        raise ValueError(
            f"无法识别输入格式: {file_path}\n"
            f"支持的扩展名: {list(_INPUT_FORMAT_TABLE.keys())}"
        )
    return fmt


def split_input_to_chapters(
    input_path: str,
    output_dir: str,
    input_format: str = 'auto',
    target_chars: int = 5000,
    min_chars: int = 1000,
) -> list:
    """
    统一入口：根据输入格式自动选择 Splitter 并切分章节。

    参数:
      input_path: 输入文件路径（EPUB / TXT / MD）
      output_dir: 输出目录（将生成 chXX.md）
      input_format: 'auto' | 'epub' | 'txt' | 'md'
      target_chars: TXT/MD 均分模式下的目标章节字数
      min_chars: 短于此字数的章节会被合并

    返回:
      生成的章节文件路径列表（仅 stdout，方便 mapfile 捕获）

    仅输出文件路径到 stdout，日志走 stderr。
    """
    if input_format == 'auto':
        input_format = detect_input_format(input_path)

    # P4-10: 依赖预检 — EPUB 格式在构造 splitter 前检查依赖，避免 RuntimeError 滞后
    if input_format == 'epub':
        if not _EPUB_DEPS_OK:
            raise RuntimeError(
                f"❌ 缺少 EPUB 解析依赖: {_EPUB_DEPS_ERROR}\n"
                "   请先安装: pip install ebooklib beautifulsoup4 lxml"
            )
        splitter = EpubSplitter(input_path, output_dir)
    elif input_format == 'txt':
        splitter = TextSplitter(input_path, output_dir, target_chars, min_chars)
    elif input_format == 'md':
        splitter = MdSplitter(input_path, output_dir, target_chars, min_chars)
    else:
        raise ValueError(f"不支持的格式: {input_format}")

    splitter.chapters = splitter.split()
    generated = splitter.write_files()

    # 汇总信息（stderr）
    _splitter_log_ok(f"完成！共生成 {len(generated)} 个章节文件")
    _splitter_log_info("下一步：")
    for f in generated[:5]:
        ch_id = Path(f).stem
        # P0-3: 根据 __package__ 自适应提示命令（单文件部署 vs python -m）
        if __package__:
            _cmd_prefix = f"python -m {__package__}"
        else:
            _cmd_prefix = "python translator_agent.py"
        _splitter_log_info(f"     {_cmd_prefix} init --chapter {ch_id} --force")
        _splitter_log_info(f"     {_cmd_prefix} pipeline --chapter {ch_id}")
    if len(generated) > 5:
        _splitter_log_info(f"     ... (共 {len(generated)} 章)")

    # 章节路径输出到 stdout（供 mapfile / shell 脚本捕获）
    for f in generated:
        print(f)
    return generated


if __name__ == "__main__":
    main()
