#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║        SagaTrans — AI 文学语义编译系统 (单体聚合版)           ║
║                                                                ║
║  此文件由 src/ 下所有模块拼接生成，未改动任何业务逻辑。            ║
║  用途：方便整体上传至会话式 AI（ChatGPT、Gemini 等）进行全量审计  ║
║                                                                ║
║  生成方式：python scripts/build_integrated.py                  ║
║  事实源：  src/ 下散装模块（17 个 Python 文件）                  ║
╚══════════════════════════════════════════════════════════════════╝

目录：
  Section 1 — 枚举与类型定义        (TaskState, DecisionLevel)
  Section 2 — LLM 适配层            (LLMAdapter, MockLLMAdapter, OpenAICompatibleAdapter, get_llm_client)
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
  Section 11.6 — typesafe-ai judges   (多原子原语 + verdict 合成 + 三级 fallback)
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
from pathlib import Path
import sqlite3
import psutil
import threading
from collections import OrderedDict
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from urllib.parse import urlparse
import argparse
import traceback
import math

# P4-1: 模块级 logger；通过 SAGATRANS__LOG_LEVEL 环境变量控制
logger = logging.getLogger("sagatrans")
_log_level = os.environ.get("SAGATRANS__LOG_LEVEL", "INFO").upper()
if _log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    logger.setLevel(getattr(logging, _log_level))
else:
    logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)


# 第三方依赖（用 try/except 包装，避免硬依赖）
# ────────────────────────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    yaml = None

try:
    from json_repair import repair_json as _rj
except ImportError:
    _rj = None

# EPUB 工具（可选）
try:
    import ebooklib
    from ebooklib import epub as _epub_lib
    from bs4 import BeautifulSoup
except ImportError:
    ebooklib = None
    _epub_lib = None
    BeautifulSoup = None

def print_banner():
    """横幅输出到 stderr，避免污染 stdout（stdout 留给机器可解析的输出，如章节路径）"""
    print(r"""
   _____                  ______
  / ___/____ _____ _____ /_  __/________ _____  _____
  \__ \/ __ `/ __ `/ __ `// / / ___/ __ `/ __ \/ ___/
 ___/ / /_/ / /_/ / /_/ // / / /  / /_/ / / / (__  )
/____/\__,_/\__, /\__,_//_/ /_/   \__,_/_/ /_/____/
           /____/

  SagaTrans — AI 文学语义编译系统  (单体聚合版)
  共 13 个模块 | 用于全量代码审计 / 单文件部署
""", file=sys.stderr)


# P4-9: 配置加载适配层（向后兼容旧版 _config 全局）
_config = None
_config_source = "builtin"
try:
    import os as _os_cfg
    _ext_mod = _os_cfg.environ.get("SAGATRANS__CONFIG_MODULE", "").strip()
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
            # Round 25: reference_extraction 默认改为 literal_translator (deepseek-flash) — 与字面直译同模型
            "reference_extraction": {"model": "literal_translator", "params_override": {}},
            "literal_translation": {"model": "literal_translator", "params_override": {}},
            "literary_rewrite": {"model": "primary", "params_override": {}},
            "critic_scoring": {"model": "primary", "params_override": {"temperature": 0.1}},
            "judge_decision": {"model": "primary", "params_override": {}},
        }
        # Round 126: 模型定义集中于此（唯一事实源）；本地 mlx 支持已移除。
        models = {
            "literal_translator": {"model_name": "deepseek-v4-flash", "thinking": False},
            "primary": {"model_name": "MiniMax-M3.1-Flash", "thinking": True},
            "secondary": {"model_name": "Qwen3.8-Flash-Next-SS", "thinking": False},
        }
        openai_api = {"api_base": "http://127.0.0.1:1234/v1", "api_key": "lm-studio",
                      "max_retries": 3, "retry_delay": 2.0, "request_timeout": 300,
                      "max_concurrent_requests": 4}
        pipeline = {"batch_size": 50, "max_retries": 3, "max_quality_retries": 6, "judge_safety_net_avg_min": 5.5,
                    "poll_interval": 0.5, "entity_hard_fail": True,
                    "version_merit": {"enabled": True, "tolerance": 0.0},
                    "early_stop": {"enabled": True, "max_low_dims": 3, "low_score_threshold": 3.0, "apply_only_first_round": True}}
        chunker = {"soft_limit": 1000, "hard_limit": 2500}
        decision_engine = {"terminology_triggers_backtrack": True,
                           "reference_triggers_backtrack": True,
                           "style_triggers_backtrack": False,
                           "max_affected_chunks_per_decision": 50,
                           "backtrack_scope": "book",
                           "backtrack_mode": "deferred",
                           "max_backtrack_sweeps_per_chapter": 1,
                           "max_backtrack_rounds_per_chunk": 2,
                           "context_max_terminology": 40,
                           "context_max_reference": 20,
                           "context_max_style": 10,
                           "context_max_entities": 30}
        critic_thresholds = {"fluency": 7.0, "readability": 7.0, "style_compliance": 6.5,
                            "voice_consistency": 7.0, "semantic_preservation": 7.0,
                            "literal_residue": 7.0,
                            "average_score_min": 7.0}

        def critic_thresholds_for_book(self, project_root=None):
            """通用默认 + book_profile 可选覆盖（与 src/config.py::Config 方法同语义；
            单文件部署兜底不读缓存，直接合并）。"""
            merged = dict(self.critic_thresholds or {})
            try:
                root = Path(project_root) if project_root else Path.cwd()
                pp = root / "input" / "book_profile.json"
                if pp.exists():
                    data = json.loads(pp.read_text(encoding="utf-8"))
                    blk = data.get("critic_thresholds") if isinstance(data, dict) else None
                    if isinstance(blk, dict):
                        for k, v in blk.items():
                            if (isinstance(v, (int, float)) and not isinstance(v, bool)
                                    and 0.0 <= float(v) <= 10.0):
                                merged[str(k)] = float(v)
            except Exception:
                pass
            return merged
        style_guide = {"avg_sentence_length": "适中、口语化",
                       "lexicon_preference": "平实、克制、英式幽默",
                       "author_priority_ratio": 0.7,
                       "genre": "儿童文学",
                       "work_type": "长篇童话"}

        def resolve_task_model(self, task_name):
            EFFORT_LEVELS_BY_PROVIDER = {
                "minimax": ("low", "medium", "high", "xhigh", "max"),
                "deepseek": ("none", "low", "high", "xhigh", "max"),
                "qwen": ("none", "low", "medium", "xhigh"),
            }
            model_key = self.get_task_role(task_name)
            role_cfg = self.models.get(model_key, {}) or {}
            params = dict(role_cfg.get("default_params", {}) or {})
            if "thinking" in role_cfg or "thinking" in params:
                raise ValueError(f"{task_name}: `thinking` 已废弃（Round 128）——改用任务级 effort")
            routing = dict(self.task_routing.get(task_name, {}) or {})
            override = dict(routing.get("params_override", {}) or {})
            if "thinking" in override:
                raise ValueError(f"{task_name}: `thinking` 已废弃（Round 128）——改用任务级 effort")
            nested_effort = override.pop("effort", None)
            merged = {**params, **override}
            merged.pop("effort", None)
            effort = routing.get("effort")
            if effort is None:
                effort = nested_effort
            if effort is not None:
                if isinstance(effort, bool):  # YAML 1.1: 裸 off/on/no/yes 会被解析为布尔
                    if effort is True:
                        raise ValueError(
                            f"{task_name}: effort 为布尔 true（应出自裸 on/yes）——"
                            f"需要思考请写具体档位；关闭思考写 none"
                        )
                    effort = "none"
                effort = str(effort).strip().lower()
                target = f"{role_cfg.get('model_name', '')} {role_cfg.get('api_base', '')}".lower()
                for provider, levels in EFFORT_LEVELS_BY_PROVIDER.items():
                    if provider in target:
                        if effort not in levels:
                            raise ValueError(
                                f"{task_name}: effort '{effort}' 对 {provider} 非法；"
                                f"合法档位: {'/'.join(levels)}"
                            )
                        break
                merged["effort"] = effort
            nested_extra = merged.pop("extra_body", None)
            task_extra = routing.get("extra_body")
            if task_extra is None:
                task_extra = nested_extra
            merged["extra_body"] = dict(task_extra) if isinstance(task_extra, dict) else {}
            return model_key, merged

        def get_task_role(self, task_name):
            return self.task_routing.get(task_name, {}).get("model", "primary")

        def has_role_models(self):
            return bool(self.models)

        def role_models(self):
            return self.models

        def get_role_config(self, role):
            if self.llm_backend == "mock":
                return {"backend": "mock", "default_params": {}}
            cfg = dict(self.models.get(role, {}) or {})
            section = self.openai_api
            cfg.setdefault("backend", "openai_api")
            cfg.setdefault("api_base", section.get("api_base", "http://127.0.0.1:1234/v1"))
            cfg.setdefault("api_key", section.get("api_key", "lm-studio"))
            cfg.setdefault("model_name", "deepseek-v4-flash")
            return cfg

    _config = _BuiltinConfig()
    if os.environ.get("SAGATRANS__STRICT") == "1":
        print("[ERROR] SAGATRANS__STRICT=1 模式下拒绝降级，退出。", file=sys.stderr)
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



class DecisionLevel(IntEnum):
    """决策等级"""
    TERMINOLOGY = 1  # 术语级，全局必须一致
    REFERENCE = 2    # 典故级，影响重写策略
    STYLE = 3        # 风格约束级，限制重写器词汇与句式



# Section 2 — LLM 适配层
# ────────────────────────────────────────────────────────────────────

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


_client_instances: dict = {}


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


# Section 3 — Scheduler 任务调度器
# ────────────────────────────────────────────────────────────────────

VALID_TRANSITIONS = {
    TaskState.PENDING:          [TaskState.EXTRACTING_TERMS],
    TaskState.DIRTY:            [TaskState.EXTRACTING_TERMS],
    TaskState.FAILED:           [TaskState.EXTRACTING_TERMS, TaskState.TRANSLATING_RAW,
                                 TaskState.REWRITING_LITERARY, TaskState.AUDITING,
                                 TaskState.JUDGING, TaskState.PERMANENTLY_FAILED],
    TaskState.PERMANENTLY_FAILED: [],  # 终态：不可转移
    TaskState.EXTRACTING_TERMS: [TaskState.TRANSLATING_RAW, TaskState.FAILED],
    TaskState.TRANSLATING_RAW:  [TaskState.REWRITING_LITERARY, TaskState.FAILED],
    TaskState.REWRITING_LITERARY: [TaskState.AUDITING, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    # AUDITING 可退回润色：实体硬失败等质量问题无需再走 Judge
    # Round 107: 高分直通 (R101 P1) 从 AUDITING 直接 COMPLETED, 跳过 judge.
    # 之前缺 COMPLETED 导致转移被拒, chunk 卡 AUDITING 死循环, 最终被 NO_PROGRESS_STREAK 误杀.
    TaskState.AUDITING:         [TaskState.JUDGING, TaskState.COMPLETED, TaskState.REWRITING_LITERARY, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    TaskState.JUDGING:          [TaskState.COMPLETED, TaskState.REWRITING_LITERARY, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    TaskState.COMPLETED:        [TaskState.DIRTY],
}


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

        # Round 130: 治理计数列（跨 episode 累计回溯轮数，不随质量计数清零）
        try:
            cursor.execute('ALTER TABLE chunk_tasks ADD COLUMN backtrack_rounds INTEGER DEFAULT 0')
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

        # Round 133: 重试适配 override 列（no_content 签名 → 降档/换角救稿，按块持久化 JSON）
        try:
            cursor.execute('ALTER TABLE chunk_tasks ADD COLUMN retry_effort_overrides TEXT')
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
            # Round 11: smart-force 模式——保留已 COMPLETED 的 chunk，避免重译浪费 LLM 调用
            cursor.execute(
                'SELECT chunk_id, state FROM chunk_tasks WHERE chapter_id = ?',
                (chapter_id,)
            )
            existing_states = {row[0]: row[1] for row in cursor.fetchall()}
        skipped = 0
        preserved_completed = 0
        for i, text in enumerate(chunks):
            chunk_id = f"{chapter_id}_chunk{i:03d}"
            try:
                if force:
                    # Round 11: smart-force——若 chunk 已是 COMPLETED 状态，跳过重置（保留 final.json）
                    if existing_states.get(chunk_id) == TaskState.COMPLETED.value:
                        preserved_completed += 1
                        continue
                    # force 模式（但仅对非 COMPLETED）：覆盖 text_content 并完全重置
                    # 让 PERMANENTLY_FAILED/FAILED/PENDING 任务可重新执行
                    cursor.execute('''
                        INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            text_content = excluded.text_content,
                            state = ?,
                            last_error = NULL,
                            retries = 0,
                            quality_retries = 0,
                            retry_effort_overrides = NULL
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
        # Round 11: smart-force 报告保留的 COMPLETED chunk 数
        if force and preserved_completed:
            print(f"  🔒 smart-force：保留 {preserved_completed} 个 COMPLETED chunk（不重译）")

        # 用前后 COUNT 差值计算真正被覆盖的 chunk 数
        # Round 11: 减去 preserved_completed（那些没动 DB 的），否则会显示"覆盖更新"全部 chunk
        overwritten = 0
        if force:
            cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            count_after = cursor.fetchone()[0]
            inserted = max(count_after - count_before, 0)
            overwritten = len(chunks) - inserted - preserved_completed
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

    def update_task_state(self, chunk_id: str, new_state: TaskState, error_msg: str = None,
                          quality_retry: bool = False, allow_terminal_restore: bool = False):
        """更新任务状态（含转移合法性校验）。

        Round 124: allow_terminal_restore — 仅 _restore_dirty_backups 使用的例外：
        回溯重跑失败、恢复旧 final.json 时, PERMANENTLY_FAILED → COMPLETED 是合法
        语义（保留之前已合格的译文）。此前该转移被合法性校验静默拒绝 → 状态残留 PF,
        日志却假报 "已恢复…(COMPLETED)"（ch09 run2 曾出现 8 条）。

        Round 130（审计③）: 返回 bool —— True=已按预期更新；False=被拒绝（非法转移，
        状态未变），调用方可感知，避免日志与状态脱节。
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT state FROM chunk_tasks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        if row:
            from_state = TaskState(row[0])
            allowed = VALID_TRANSITIONS.get(from_state, [])
            terminal_restore = (
                allow_terminal_restore
                and from_state == TaskState.PERMANENTLY_FAILED
                and new_state == TaskState.COMPLETED
            )
            if new_state not in allowed and not terminal_restore:
                print(f"⚠️ [Scheduler] 非法转移 {chunk_id}: {row[0]} -> {new_state.value}"
                      f" (允许: {[s.value for s in allowed]})")
                return False
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
        return True

    # ────────────────────────────────────────────────────────────────
    # Round 133: 块级重试适配 override（no_content 签名 → 降档/换角救稿）
    # payload 例: {"sig":"no_content","stage":"TRANSLATING_RAW","role":"primary",
    #             "effort":"xhigh","used_role":"primary","used_effort":"max",
    #             "down_delta":1,"rescue_used":false,"escalated":false,"applied_at":...}
    # 语义: role/effort = 下一次尝试应使用的角色与档位（apply 侧直接读取）。
    # ────────────────────────────────────────────────────────────────
    def get_retry_override(self, chunk_id: str) -> Optional[Dict]:
        """读取块级重试适配 override（JSON；无记录/列缺失/解析失败 → None）。"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'SELECT retry_effort_overrides FROM chunk_tasks WHERE chunk_id = ?',
                (chunk_id,))
            row = cursor.fetchone()
        except sqlite3.OperationalError:
            return None
        if not row or not row[0]:
            return None
        try:
            data = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def set_retry_override(self, chunk_id: str, payload: Dict) -> bool:
        """写块级重试适配 override（JSON 序列化）；返回 True=已更新。"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'UPDATE chunk_tasks SET retry_effort_overrides = ?, '
                'updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?',
                (json.dumps(payload, ensure_ascii=False), chunk_id))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.OperationalError:
            return False

    def clear_retry_override(self, chunk_id: str) -> bool:
        """清空块级重试适配 override。"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                'UPDATE chunk_tasks SET retry_effort_overrides = NULL, '
                'updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?',
                (chunk_id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.OperationalError:
            return False

    def batch_update_state(self, chunk_ids: List[str], new_state: TaskState, error_msgs: Dict[str, str] = None, quality_retry: bool = False, reset_counters: bool = False):
        """批量更新任务状态 - 减少数据库往返开销

        Args:
            chunk_ids: 任务 ID 列表
            new_state: 目标状态
            error_msgs: 可选，按 chunk_id 分错误信息 {chunk_id: error_msg}
            quality_retry: 是否为质量重试（True=增 quality_retries，False=增 retries）
            reset_counters: 是否重置 retries 和 quality_retries 为 0（用于 DIRTY 回溯后重新计数）

        Returns (Round 130, 审计③):
            List[str] — 实际被更新的 chunk_id 列表（非法转移/不存在的 chunk 会被跳过，
            不再无声无息；调用方可据此感知拒绝情况）。
        """
        if not chunk_ids:
            return []
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
            return []

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
        return valid_ids

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

    def trigger_backtrack(self, chunk_ids: List[str]) -> Dict[str, List[str]]:
        """回溯引擎：仅将已定稿(COMPLETED)的数据块标记为 DIRTY 重跑。

        与 VALID_TRANSITIONS 对齐：只允许 COMPLETED -> DIRTY。
        禁止对 JUDGING/AUDITING 等中间态打 DIRTY（否则会阻断 COMPLETED 并可能死循环）。
        尚未产出译文的 PENDING / EXTRACTING_TERMS 等也不回溯——后续阶段会自然读到新决策。

        Round 130 治理预算（防"回溯→重跑→再回溯"自激长跑）：
        单块累计回溯轮数 backtrack_rounds（跨 episode 持久，不随质量计数清零）达到
        decision_engine.max_backtrack_rounds_per_chunk 即冻结，不再标记 DIRTY。
        0 = 不限制（旧行为，向后兼容）。

        Returns:
            {"dirty": [...], "over_cap": [...], "skipped_state": [...]}
        """
        if not chunk_ids:
            return {"dirty": [], "over_cap": [], "skipped_state": []}
        de_cfg = getattr(_config, "decision_engine", {}) or {}
        try:
            cap = int(de_cfg.get("max_backtrack_rounds_per_chunk", 2) or 0)
        except (TypeError, ValueError):
            cap = 2
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor.execute(f'''
            SELECT chunk_id, state, COALESCE(backtrack_rounds, 0)
            FROM chunk_tasks WHERE chunk_id IN ({placeholders})
        ''', tuple(chunk_ids))
        dirty: List[str] = []
        over_cap: List[str] = []
        skipped_state: List[str] = []
        for cid, state, rounds in cursor.fetchall():
            if state != TaskState.COMPLETED.value:
                skipped_state.append(cid)
                continue
            if cap > 0 and int(rounds or 0) >= cap:
                over_cap.append(cid)
                continue
            dirty.append(cid)
        if dirty:
            ph2 = ','.join('?' for _ in dirty)
            cursor.execute(f'''
                UPDATE chunk_tasks
                SET state = ?, backtrack_rounds = COALESCE(backtrack_rounds, 0) + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id IN ({ph2})
            ''', (TaskState.DIRTY.value,) + tuple(dirty))
        self.conn.commit()
        msg = f"🔄 已触发 {len(dirty)} 个数据块的回溯重构（标记 DIRTY）"
        if skipped_state:
            msg += f"（跳过 {len(skipped_state)} 个非 COMPLETED 状态）"
        if over_cap:
            msg += f"（治理冻结 {len(over_cap)} 个超单块预算）"
        print(msg + "。")
        return {"dirty": dirty, "over_cap": over_cap, "skipped_state": skipped_state}

    def find_chunks_containing(
        self,
        source_text: str,
        chapter_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """全文搜索包含 source_text 的 chunk_id（T3：两级筛选）。

        一级：SQL LIKE 粗筛（原样 + 空白归一两种形态，后者用 SQLite
        REPLACE 把块内 \\n/\\r/\\t 折成空格，覆盖跨行折行短语）；二级：
        Python 词边界精筛（见 _source_term_pattern）。limit 在精筛后截断。

        Args:
            source_text: 要匹配的原文片段
            chapter_id: 若给定则仅扫该章；None 表示全书
            limit: 最多返回条数（None=不截断）
        """
        if not source_text:
            return []
        pattern = _source_term_pattern(source_text)
        if pattern is None:
            return []
        norm_term = re.sub(r'\s+', ' ', source_text).strip()

        def _like_esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        raw_pat = f"%{_like_esc(source_text)}%"
        norm_pat = f"%{_like_esc(norm_term)}%"
        norm_sql = ("REPLACE(REPLACE(REPLACE(text_content, char(10), ' '), "
                    "char(13), ' '), char(9), ' ')")
        cond = f"text_content LIKE ? ESCAPE '\\' OR {norm_sql} LIKE ? ESCAPE '\\'"
        cursor = self.conn.cursor()
        if chapter_id:
            sql = (
                "SELECT chunk_id, text_content FROM chunk_tasks "
                f"WHERE chapter_id = ? AND ({cond}) "
                "ORDER BY chunk_id ASC"
            )
            params: list = [chapter_id, raw_pat, norm_pat]
        else:
            sql = (
                "SELECT chunk_id, text_content FROM chunk_tasks "
                f"WHERE {cond} "
                "ORDER BY chunk_id ASC"
            )
            params = [raw_pat, norm_pat]
        cursor.execute(sql, tuple(params))
        matched = []
        for chunk_id, text_content in cursor.fetchall():
            norm = re.sub(r'\s+', ' ', text_content or '')
            if pattern.search(norm):
                matched.append(chunk_id)
        if limit is not None and limit > 0:
            matched = matched[:int(limit)]
        return matched

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
                translation_strategy TEXT DEFAULT 'free', -- literal/transliteration/free：翻译策略，首次登记后固定
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

        try:
            self.conn.execute("SELECT translation_strategy FROM entity_registry LIMIT 1")
        except Exception:
            self.conn.execute("ALTER TABLE entity_registry ADD COLUMN translation_strategy TEXT DEFAULT 'free'")

        # 决策锁定列（Round 121）：防跨章译文翻转 → 回溯乒乓。
        # locked=1 的决策拒绝覆盖性更新且不触发 backtrack；旧库幂等迁移
        try:
            self.conn.execute("SELECT locked FROM decision_db LIMIT 1")
        except Exception:
            self.conn.execute("ALTER TABLE decision_db ADD COLUMN locked INTEGER DEFAULT 0")

        # Round 130: deferred 回溯队列 — 决策变更先登记，章末合并应用一次
        # （防"决策翻转→回溯已完成块→重跑再翻转"自激长跑；同一 chunk 幂等合并）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_backtrack (
                chunk_id TEXT PRIMARY KEY,
                decision_id INTEGER,
                source_key TEXT,
                queue_rounds INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

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

        Round 130: backtrack_mode=deferred（默认）时，变更不再即时标记 DIRTY，
        而是登记进 pending_backtrack 队列，由 pipeline 在章末（队列清空点）合并应用一次。
        """
        cursor = self.conn.cursor()
        try:
            # 幂等检测：translation 未变则跳过 backtrack（仍可合并 impact）
            cursor.execute(
                'SELECT id, translation, locked FROM decision_db WHERE source_key = ?',
                (source,),
            )
            existing = cursor.fetchone()
            translation_unchanged = bool(
                existing and existing[1] == translation
            )

            # Round 121 锁定决策：拒绝覆盖性变更，保住跨章一致译名
            # （翻转会被 _find_chunks_containing 放大成全书回溯乒乓）
            if existing and int(existing[2] or 0) == 1 and not translation_unchanged:
                print(
                    f"🔒 [Decision Lock] {source}: 保留锁定译文「{existing[1]}」，"
                    f"忽略新译文「{translation}」（不触发回溯）"
                )
                return

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
                backtrack_mode = str(de_cfg.get("backtrack_mode", "deferred") or "deferred").lower()
                if backtrack_mode == "immediate":
                    self._trigger_backtrack(chunks_to_dirty)
                else:
                    self._queue_backtrack(chunks_to_dirty, decision_id, source)
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

    def _queue_backtrack(self, chunk_ids: List[str], decision_id: Optional[int], source_key: str):
        """Round 130 deferred 模式：把受影响 chunk 登记进待回溯队列（幂等合并）。

        队列由 pipeline 在章末合并点经 take_pending_backtracks 取出后统一应用；
        同一 chunk 多次命中只保留一行并累计 queue_rounds，避免重复重跑。
        """
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, queue_rounds, updated_at)
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                decision_id = excluded.decision_id,
                source_key = excluded.source_key,
                queue_rounds = COALESCE(queue_rounds, 0) + 1,
                updated_at = CURRENT_TIMESTAMP
        ''', [(cid, decision_id, source_key) for cid in chunk_ids])
        self.conn.commit()
        print(f"⏳ [Backtrack Queue] 登记 {len(chunk_ids)} 个待回溯 chunk（章末合并应用）: {source_key[:40]}")

    def take_pending_backtracks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """取出并删除指定 chunk 的待回溯登记（章末合并应用点调用）。

        Returns: [{"chunk_id", "decision_id", "source_key", "queue_rounds"}, ...]
        """
        if not chunk_ids:
            return []
        cur = self.conn.cursor()
        ph = ','.join('?' for _ in chunk_ids)
        rows = cur.execute(
            f'SELECT chunk_id, decision_id, source_key, COALESCE(queue_rounds, 0) AS queue_rounds '
            f'FROM pending_backtrack WHERE chunk_id IN ({ph})',
            tuple(chunk_ids)).fetchall()
        if rows:
            cur.execute(f'DELETE FROM pending_backtrack WHERE chunk_id IN ({ph})', tuple(chunk_ids))
            self.conn.commit()
        return [dict(r) for r in rows]

    def count_pending_backtracks(self, chunk_ids: Optional[List[str]] = None) -> int:
        """统计待回溯登记数（可限定 chunk 列表；None=全库）。"""
        cur = self.conn.cursor()
        if chunk_ids:
            ph = ','.join('?' for _ in chunk_ids)
            return int(cur.execute(
                f'SELECT COUNT(*) FROM pending_backtrack WHERE chunk_id IN ({ph})',
                tuple(chunk_ids)).fetchone()[0])
        return int(cur.execute('SELECT COUNT(*) FROM pending_backtrack').fetchone()[0])

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
                canonical, source_names_json, variant_names_json, strategy = row[0], row[1], row[2], (row[3] if len(row) > 3 else 'free')
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
                strategy_label = {"literal": "直译", "transliteration": "音译", "free": "自由意译"}.get(strategy, "自由意译")
                line = f"- 实体: '{clean}' (源语指称: {source_names_json}) -> 必须译为 '{clean}' [策略: {strategy_label}]"
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



def query_high_confidence_entities(conn) -> List[Tuple[str, str, str, str]]:
    """返回高置信度且无待核查的实体 (canonical_name, source_names_json, variant_names_json, translation_strategy)。

    用途：prompt 上下文注入、译后一致性报告。
    等价 SQL：`WHERE confidence = 'high' AND pending_review = 0`
    """
    cursor = conn.cursor()
    cursor.execute(
        'SELECT canonical_name, source_names, variant_names, translation_strategy FROM entity_registry '
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


def get_sibling_canonicals_by_source(conn) -> Dict[str, Set[str]]:
    """Round 71: 对 entity_registry 中每个 source_name, 返回它登记到的所有 canonical 集合.

    包含 pending_review = 0 与 = 1 的全部行 (与运行时 audit 仅看 0 不同).
    原因: ReferenceAgent 标记 pending_review=1 的行同样是有效 source_name 注册,
          如果不纳入, 跨任务冲突会被静默忽略 → rewriter 收到自相矛盾的反馈
          → 死循环 (ch32 实测 6 轮 quality_retries 全失败).

    返回: {source_name: {canonical, ...}}. 即使无冲突的 source_name 也保留
           (调用方可直接 set lookup, 不必先判 None).
    """
    cursor = conn.cursor()
    cursor.execute(
        'SELECT canonical_name, source_names FROM entity_registry'
    )
    rows = cursor.fetchall()
    out: Dict[str, Set[str]] = {}
    for canonical, source_names_json in rows:
        if not source_names_json:
            continue
        try:
            import json as _json
            srcs = _json.loads(source_names_json)
        except (ValueError, TypeError):
            continue
        if not isinstance(srcs, list):
            continue
        for sn in srcs:
            if isinstance(sn, str) and sn:
                out.setdefault(sn, set()).add(canonical)
    return out


def detect_cross_person_conflicts(conn) -> Dict[str, List[str]]:
    """Round 71: 检测 entity_registry 中 source_name 跨人物冲突 (含 pending_review).

    冲突定义: 同一英文 source_name 被登记到 ≥2 个 canonical 时 (不论 pending_review 状态).
    返回: {source_name: [canonical, ...]} 仅包含冲突条目.

    Note: 与 src/pipeline.py:_entity_registry_conflict_audit 同语义, 但本函数
          包含 pending_review = 1 的行, 用于运行时 sibling 查找.
    """
    siblings = get_sibling_canonicals_by_source(conn)
    return {sn: sorted(cs) for sn, cs in siblings.items() if len(cs) >= 2}


def seed_entity_variants(conn, variants: Optional[Dict[str, List[str]]] = None):
    """将 per-work 实体异体种子合并进 entity_registry.variant_names（幂等，不覆盖既有异体）。

    P1-4: 种子不再硬编码在通用代码里 — 默认读取配置 entity_variants 段，
    为空时回退 per-work 资源 input/entity_variants.yaml（均可选；缺失 = 不注入）。
    """
    if variants is None:
        variants = load_entity_variant_seeds()
    cur = conn.cursor()
    for canonical, variant_list in (variants or {}).items():
        variant_list = list(variant_list or [])
        if not canonical or not variant_list:
            continue
        cur.execute("SELECT id, variant_names FROM entity_registry WHERE canonical_name = ?", (canonical,))
        row = cur.fetchone()
        if not row:
            continue
        vid, vn_json = row
        existing = robust_json_loads(vn_json, expected_type=list) if vn_json else []
        merged = list(dict.fromkeys(existing + variant_list))
        if merged == existing:
            continue
        cur.execute(
            "UPDATE entity_registry SET variant_names = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(merged, ensure_ascii=False), vid),
        )
    conn.commit()


# Section 5 — SmartChunker 文本切分器
# ────────────────────────────────────────────────────────────────────

QUOTE_CHARS_OPEN = "\u201c\u2018\u300c\u300e"
QUOTE_CHARS_CLOSE = "\u201d\u2019\u300d\u300f"

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



# Section 5b — Reject Category 归一化与提取 (Round 70) + 文本匹配 helpers (Round 71)
# ────────────────────────────────────────────────────────────────────

_CATEGORY_SYNONYMS: dict = {
    "实体一致性硬失败": "实体一致性违反",
    "实体一致性硬失败(降级)": "实体一致性违反",
    "实体一致性违反(per-entity 聚合)": "实体一致性违反",
    "音色漂移（severe）": "音色漂移",
}


def _normalize_category(category: str) -> str:
    """Round 69: 把同义 category 名称归一化到一个规范形式."""
    return _CATEGORY_SYNONYMS.get(category, category)


def extract_reject_categories(
    reason_text: str,
    as_set: bool = False,
    normalize: bool = True,
):
    """从 reject_reason 文本里抽出 category 列表 (按出现顺序去重).

    Args:
        reason_text: critic / judge 输出的 reject 原因字符串, 多个 category 用 `;` 分隔
        as_set:     True 返回 set (无序), False 返回 list (有序)
        normalize:  是否应用 _CATEGORY_SYNONYMS 归一化.
                    - True (默认): stagnation 检测用, 把 `音色漂移（severe）` 和
                      `音色漂移（subtle）` 视为同一类 (都属于 voice drift)
                    - False: 策略查找用, 保留具体形式以便 lookup
                      `_JEV_CATEGORY_STRATEGY` / `GLOBAL_REWRITE_CATEGORIES`
                      (这些 dicts 的 keys 用具体形式)
    Round 89: 增加 normalize 参数, 让 stagnation 检测和策略查找可独立控制.
    """
    if not reason_text:
        return set() if as_set else []
    seen: Set[str] = set()
    ordered: List[str] = []
    for seg in reason_text.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        idx = seg.find(":")
        if idx > 0:
            raw = seg[:idx].strip()
            c = _normalize_category(raw) if normalize else raw
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
    return seen if as_set else ordered


def case_insensitive_find(name: str, haystack: Optional[str]) -> Optional[str]:
    if not name or not haystack:
        return None
    if not name.isascii():
        return name if name in haystack else None
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    m = pat.search(haystack)
    return m.group(0) if m else (name if name in haystack else None)


# Round 71: alias 与 pipeline.py:30 import 保持一致


_text_match_case_insensitive_find = case_insensitive_find





# Section 6 — Agents
# ────────────────────────────────────────────────────────────────────

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
        hints = (_config.style_guide or {}).get("translation_hints", []) or []
        hints_block = ""
        if hints:
            hints_block = (
                "   （本书已登记考据条目，优先核对：\n"
                + "\n".join(f"      - {h}" for h in hints[:30])
                + "\n   ）"
            )
        return f"""你是一位精通西方文学、历史、神话和宗教的资深翻译考据专家。
你的任务是扫描给定的{genre}（{work_type}）片段，完成两项并行任务：
1. 识别其中的文学典故、宗教隐喻、神话引用或历史名词，并制定翻译策略。
2. 识别本片段中出现的人物/地点实体，建立实体档案。

【硬约束】
- 思考过程必须用中文。最终输出必须是严格的 JSON，禁止任何 JSON 之外的字符。
- 不要输出 "思考过程"，最终只保留 JSON。

【严格定义】
不要提取普通的角色名字（如 Paul）或普通地名，除非它们具有明显的象征意义或典故来源。

【典故识别 — 两类独立检查】
1. 专有名词/术语: 具有特定法律/社会/宗教含义的专名（直译会让中文读者误解的）
2. 文学引用/诗句: 对知名诗句/名言的引用或谐仿（判据：押韵、格律、典故化用等引文特征）
{hints_block}

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
- translation_strategy: 翻译策略（literal=直译, transliteration=音译, free=自由意译）；首次登记后固定，后续必须沿用同一策略

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
      "confidence": "high/medium/low",
      "translation_strategy": "literal/transliteration/free"
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

    def _run_confirm_stage(
        self, chunk_id: str, text_chunk: str, recall_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Round 118 P1-3: 第二阶段确认（MiniMax 复核 DeepSeek 高召回初筛）.

        Round 25 曾为降本把考据收敛到单模型；质量优先恢复两阶段分工：
          - 阶段 1（DeepSeek, 低温）：高召回，宁多勿漏；
          - 阶段 2（MiniMax, thinking）：逐条核验典故来源、删误报、补漏、
            统一实体 canonical/策略，输出定稿 JSON。

        任何失败（配置缺失/网络/JSON 损坏）都返回 None，调用方回退召回结果，
        绝不因确认阶段把已有召回丢掉。
        """
        cfg = _config.pipeline.get("reference_two_stage", {}) or {}
        if not cfg.get("enabled", True):
            return None
        try:
            role = _config.get_task_role("reference_confirmation")
            confirm_llm = get_llm_client_for_task("reference_confirmation")
            model_key, params = _config.resolve_task_model("reference_confirmation")
        except Exception as e:
            print(f"⚠️ [Reference Agent] 两阶段确认不可用（保留召回结果）: {e}", file=sys.stderr)
            return None
        model_name = get_role_model_name(role) or model_key
        max_chars = int(cfg.get("max_recall_chars", 12000))
        recall_json = json.dumps(recall_data, ensure_ascii=False, indent=2)[:max_chars]
        prompt = f"""你是资深文学考据复核员（第二意见）。另一位考据员已对片段做「高召回」初筛，你的任务是复核并定稿。

【原文片段】
{text_chunk[:8000]}

【初筛候选（JSON）】
{recall_json}

【复核规则】
1. 逐条核验 allusion_target 是否被原文支撑；无依据、张冠李戴或明显误判 → 删除。
2. 补漏：初筛漏掉且确有出处的典故/隐喻/历史名词 → 追加（宁精勿滥）。
3. 实体：合并重复指称；canonical_name 统一为规范中文译名；source_names 必须都是原文实际出现的形式。
4. 策略只能是 RETAIN_AND_ANNOTATE / CULTURAL_EQUIVALENT / LITERAL_TRANSLATION 三选一。
5. 若初筛正确，原样保留其条目（可修正字段格式）；不要为凑数新增无依据条目。

【输出格式】
输出与初筛完全相同的纯 JSON（references / entities 两个数组），不要输出 JSON 以外的任何字符。
"""
        try:
            raw = confirm_llm.generate(
                prompt=prompt,
                model_name=model_name,
                **params,
            )
        except Exception as e:
            print(
                f"⚠️ [Reference Agent] 两阶段确认调用失败（保留召回结果）: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return None
        data = robust_json_loads(raw)
        # robust_json_loads 对垃圾输入可能返回 {}；确认结果必须至少含
        # references/entities 之一，否则视为解析失败回退召回。
        if not isinstance(data, dict) or not data:
            print("⚠️ [Reference Agent] 两阶段确认 JSON 解析失败（保留召回结果）", file=sys.stderr)
            return None
        if "references" not in data and "entities" not in data:
            print("⚠️ [Reference Agent] 两阶段确认 JSON 解析失败（保留召回结果）", file=sys.stderr)
            return None
        refs = data.get("references")
        ents = data.get("entities")
        return {
            "references": refs if isinstance(refs, list) else [],
            "entities": ents if isinstance(ents, list) else [],
        }

    def process_chunk(self, chunk_id: str, text_chunk: str, affected_chunks: List[str] = None):
        """处理单个文本块，识别典故并自动写入决策引擎，同时登记实体到注册表"""
        print(f"🔍 [Reference Agent] 正在考据数据块: {chunk_id}...")

        # 1. 调用 LLM (使用配置路由)
        prompt = self._build_prompt(text_chunk)
        model_key, params = _config.resolve_task_model("reference_extraction")
        model_name = get_role_model_name(self.role) or model_key

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
        )

        # 2. 解析 JSON
        parsed_data = self._parse_and_clean_json(raw_output)
        references = parsed_data.get("references", [])
        entities = parsed_data.get("entities", [])

        # Round 118 P1-3: 两阶段 — DeepSeek 高召回初筛 + MiniMax 逐条复核定稿。
        # 确认失败/禁用时静默回退召回结果（绝不丢数据）。
        confirmed = self._run_confirm_stage(chunk_id, text_chunk, parsed_data)
        if confirmed is not None:
            print(
                f"🔬 [Reference Agent] 两阶段确认: references "
                f"{len(references)}→{len(confirmed['references'])}, entities "
                f"{len(entities)}→{len(confirmed['entities'])}"
            )
            parsed_data = confirmed
            references = confirmed["references"]
            entities = confirmed["entities"]

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
            translation_strategy = ent.get("translation_strategy", "free")
            
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
                # Round 121: 疑问词/连词泄漏护栏（ch06 "who"→帽匠等 6 实体污染，
                # 造成"在场却漏译"不可满足核查 → 实体硬失败循环）
                "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
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
                cursor.execute('SELECT id, canonical_name, source_names, variant_names FROM entity_registry')
                similar_found = False
                for existing_row in cursor.fetchall():
                    eid, ecanon, enames_json, evariants_json = existing_row
                    # P2-1: 统一走 robust_json_loads
                    enames = robust_json_loads(enames_json, expected_type=list) if enames_json else []
                    evariants = robust_json_loads(evariants_json, expected_type=list) if evariants_json else []
                    
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
                        # Round 121: 新中文名已是该实体的登记异体 → 已知别名，
                        # 仅并入 source_names，不再标记待核查（防归一后反复重标）
                        if canonical in evariants:
                            merged_names = list(set(enames + source_names))
                            cursor.execute('''
                                UPDATE entity_registry
                                SET source_names = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            ''', (json.dumps(merged_names, ensure_ascii=False), eid))
                            similar_found = True
                            print(f"  📝 已知异体合并: {canonical} <- {source_names} (并入 {ecanon})")
                            break
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
                        INSERT INTO entity_registry (canonical_name, source_names, pronoun_gender, entity_type, first_seen_chunk, confidence, pending_review, translation_strategy)
                        VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    ''', (canonical, source_names_json, pronoun_gender, entity_type, chunk_id, confidence, translation_strategy))
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

    # Round 38: 一次性告警 set（避免每次 retry 都打同样的 few-shot 缺失警告）
    _few_shot_warned_paths: set = set()

    def _load_few_shot_examples(self) -> str:
        """加载 Few-shot 示例，注入 Prompt 作为参考"""
        try:
            root_dir = Path(__file__).resolve().parent.parent
            # depth-independent：整合脚本在 src/ 下两层就到根，散装 src/agents/*.py 要三层
            # 探测 config.yaml 向上找
            if not (root_dir / "config.yaml").exists():
                root_dir = root_dir.parent
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
            # Round 38: 一次性告警（同一个 path 只打一次），减少跑批噪音
            path_key = str(few_shot_file)
            if path_key not in self._few_shot_warned_paths:
                self._few_shot_warned_paths.add(path_key)
                print(f"  ⚠️ [Rewriter] 加载 few-shot 示例失败: {few_shot_file} 不存在或格式错误（后续同类警告将省略）", file=sys.stderr)
            return ""

    # Round 34: thin shim — delegates to src.utils.text_match.extract_reject_categories
    # Round 89: 增加 normalize 参数 (默认 True 保持向后兼容). 策略查找路径显式传 False.
    @staticmethod
    def _extract_reject_categories(reason_text: str, normalize: bool = True) -> list:
        return extract_reject_categories(reason_text, as_set=False, normalize=normalize)

    _JEV_CATEGORY_STRATEGY: dict = {
        "直译残留": "本轮必须消除译稿中的英文句式套用与逐词堆砌，以自然中文表达为目标整体重写。",
        "关键语义丢失": "保留原文所有关键意象、动作细节、隐喻、反讽，不得概括或弱化。",
        "音色漂移（severe）": "本轮保留原文的叙事口吻/角色音色，包括讽刺、克制、幽默等特征，避免被抹平为平淡中性叙述。",
        "音色漂移（subtle）": "轻微调整使叙事口吻更接近原文，避免过度风格化。",
        # Round 85: Round 82 新增 subtle+其它 escalation 类别专用策略.
        # 单独 subtle 不 reject (Round 82); 仅在叠加其它 reject 时升级.
        # Rewriter 看到此类别时, 已知 subtle 单独不够, 其它维度也已 reject,
        # 需要在解决其它核心问题的同时一并把 voice drift 拉回正轨.
        "音色漂移（subtle+其它）": (
            "本轮 Jev 判定 subtle voice drift 与其它问题叠加 (例如 实体一致性 / 直译残留 / "
            "关键语义丢失) 共同触发 reject. 请在解决其它维度的同时, 把叙事口吻向原文基线 "
            "靠拢 (保留讽刺/克制/幽默等特征), 避免被抹平为平淡中性叙述 —— 但不要过度风格化 "
            "(否则会反向触发 severe voice drift)."
        ),
        "语域偏离（under）": "提升语域，减少口语/方言化表达，匹配目标读者期望。",
        "语域偏离（over）": "降低语域，减少文言/书面化表达，避免距离感过强。",
        "整体风格偏离": "全篇风格重润色，匹配目标风格基线。",
        # Round 107 P5: 风格契合度低 — Jev Score primitive, 全局风格问题
        "风格契合度低": "全篇风格重润色：以风格基准为锚点，重新组织断句节奏、词汇密度、修辞与语域，使译文风格与目标基线对齐。局部词换无法解决全局风格偏离。",
    }

    def _build_jev_category_section(self, reject_reason: str) -> str:
        """Round 23: 在 retry prompt 顶部渲染 Jev 判定的核心问题维度 + 对应策略。

        Round 89: 策略查找路径必须用 normalize=False, 否则 `_CATEGORY_SYNONYMS`
        把 `音色漂移（severe）` 归一为 `音色漂移`, 导致 lookup miss, 落到 generic
        fallback "请针对此维度整体审视并修订。" (失去 specific 策略).
        """
        cats = self._extract_reject_categories(reject_reason, normalize=False)
        if not cats:
            return ""
        lines = []
        for c in cats:
            strategy = self._JEV_CATEGORY_STRATEGY.get(c)
            if strategy:
                lines.append(f"  - {c}：{strategy}")
            elif c == "实体一致性违反":
                lines.append(f"  - {c}：详见下方【实体一致性反馈】清单。")
            else:
                lines.append(f"  - {c}：请针对此维度整体审视并修订。")
        body = "\n".join(lines)
        return (
            "\n【⚠️ 本轮核心问题维度（Jev/primitives 判定 — 必须整体性处理）】\n"
            "本轮 critic 触发了以下 Jev primitives 维度：\n"
            f"{body}\n"
            "提示：上述每一项都应在**整体重写层面**解决（而非局部词换）。"
            "请把每一维度当作「必须 100% 解决」的硬约束。\n"
        )

    def _build_prompt(
        self,
        raw_translation: str,
        decisions_context: str,
        style_guide: str,
        narrative_memory: str = "",
        entity_registry: Optional[Dict[str, str]] = None,  # 回路修复
        verse_note: Optional[str] = None,  # Round 119 L1 韵文协议
    ) -> str:
        few_shot = self._few_shot_examples
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        mem_block = ""
        if narrative_memory:
            mem_block = f"\n【前文摘要（保持声口与指代连贯）】\n{narrative_memory}\n"
        persona = (_config.style_guide or {}).get("persona", "职业文学译者")
        footnote_policy = (_config.style_guide or {}).get("footnote_policy", "inline")

        footnote_rules = {
            "none": "禁止在润色阶段添加脚注。如遇需要注释的典故，用文内括号简注（如：渡渡鸟（一种已灭绝的鸟））。",
            "inline": "如遇需要注释的典故，应在首次出现时用文内括号简注，而非使用脚注。",
            "markdown_footnote": "如遇需要注释的典故，可使用 Markdown 脚注语法 [^1]。",
        }
        footnote_instruction = footnote_rules.get(footnote_policy, footnote_rules["inline"])

        # 回路修复：构建实体译名登记表 section（按 canonical 分组，源文别名放括号里）
        # 这样 Rewriter 看到 canonical 优先 + 所有英文别名，能推断 "George Lomax" 也属于 "乔治·洛马克斯先生"
        entity_registry_section = ""
        if entity_registry:
            # 按 canonical 分组：{canonical: [source_name1, ...]}
            grouped: Dict[str, List[str]] = {}
            for src_name, zh_canonical in entity_registry.items():
                if src_name and zh_canonical:
                    grouped.setdefault(zh_canonical, []).append(src_name)
            if grouped:
                lines = []
                for canonical, aliases in sorted(grouped.items()):
                    alias_str = " / ".join(aliases)
                    lines.append(f"  - {canonical}（源文中可能的形式：{alias_str}）")
                entity_registry_section = (
                    "\n【实体译名登记表（首次润色参考 — 严格遵循）】\n"
                    "译稿中的人名/地名/组织名必须使用下列规范译名。"
                    "源文中如出现任一英文别名（含部分匹配），都应翻译为对应中文译名。\n"
                    "【优先级规则】实体译名若与上方考据/术语决策的写法不一致，"
                    "**以本登记表规范主名为准**（统一为规范主名，不得保留考据决策中的"
                    "异体/简称/缺后缀形式）。\n"
                    + "\n".join(lines)
                    + "\n"
                )

        # Round 54: 把 config.yaml 的 forbidden_modernisms 注入 Rewriter prompt（修复 dead config）
        forbidden_modernisms = (_config.style_guide or {}).get("forbidden_modernisms", []) or []
        forbidden_section = ""
        if forbidden_modernisms:
            forbidden_str = "、".join(str(w) for w in forbidden_modernisms)
            forbidden_section = (
                "\n【禁用词列表 (CRITICAL — 译文将被自动校验)】\n"
                f"以下表达严禁出现在译文中（含其变体形式）：{forbidden_str}\n"
                "如出现即视为违背风格指南，无论文学质量如何都会被 reject。\n"
            )

        # Round 10: 实体登记表提到首位（最显眼位置）+ MUST 语言 + self-check 指令
        # 原因：Round 9 审计暴露 ch02_chunk000 漏译"绅士乔"等——LLM 即使收到 entity_registry 也常忽略
        # 修复策略：(1) 把登记 section 放在 prompt 顶部，避免被后续段落淹没
        #          (2) 用 MUST/否则被拒 等强约束语言
        #          (3) 在 output 前要求 self-check，确保所有登记实体都已呈现
        # Round 119 L1: 韵文 chunk 注入逐行对应/复沓保留协议
        verse_section = verse_note if verse_note else ""
        return f"""你是一位{persona}。
你的任务是对提供的【直译底稿】进行最高水准的文学润色。

{entity_registry_section}{forbidden_section}
{decisions_context}
{mem_block}
{few_shot}

【风格基准 (Style Guide)】
{style_guide}

【⚠️ 实体译名一致性 (CRITICAL — 译文将被自动校验)】
完成润色后必须 self-check：源文中出现的每个登记实体，译文中都必须能找到对应的规范中文译名。
如有遗漏（如 "Codders" 漏译），即使文学质量再高也会被 reject。


{verse_section}【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. {footnote_instruction}
3. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。
4. 脚注只陈述典故/文化背景/双关机制本身；不得出现任何翻译流程元语（如"保留与加注策略""保留并加注""保留音译""按约束""直译自""策略池"等字样），不得描述译者的处理方式或决策理由。
5. 原文里的方括号编辑标记（如 [later editions continued as follows]）必须保留并译出，用【】包裹（如【后来版本续之如下：】），不得删除或改写成叙述。
6. 注释一律使用 Markdown 脚注 [^n] 并带正文引标；不得新增"【章末注释】""【脚注】"等标题式注释块，不得输出无引标的裸编号注列表（如 [1] xxx 列表）。

【翻译约束 (必须同时遵守)】
1. 不得丢弃原文的比喻/意象 (除非确实不可译)
2. 不得添加原文没有的评价性词语或从句
3. 保持原文信息量，不要"锦上添花"

【语气标签校准】
- 台词前的副词 (coquettishly/angrily/softly) 作为语域校准强约束信号
- 匹配副词的情感色彩到译文语气
- 例如: coquettishly → 娇嗔/撒娇的语气, angrily → 生气的语气, softly → 轻柔的语气

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
        # 回路修复：新增 entity_feedback + entity_registry 参数（带默认值，需放在 retry_count 之后）
        entity_feedback: Optional[str] = None,
        entity_registry: Optional[Dict[str, str]] = None,
        all_feedback: Optional[List[Dict]] = None,
        specific_edits: Optional[List[Dict]] = None,
        register_target: str = "",
        low_dims: Optional[List[str]] = None,
        # Round 64: 上轮被 Jev uncertain zone 跳过的维度，避免对边界判断维度做无效重写
        skipped_dimensions: Optional[List[str]] = None,
        verse_note: Optional[str] = None,  # Round 119 L1 韵文协议
    ) -> str:
        # Round 10: 重试场景下也强化 entity section 的位置 + MUST 语言
        # 重试时 entity_feedback 已具体列出问题项，本轮必须 100% 修复
        # 实体反馈 section（最显眼位置，prompt 顶部）
        entity_feedback_section = ""
        if entity_feedback:
            # Round 29: 在 entity_feedback 顶部插入"输出前全文自检"指令
            # —— 解决"rewriter 改了一处 variant 但另一处 source_name 残留"导致下一轮又失败的循环
            self_check_preamble = (
                "\n【🔍 输出前必做：实体一致性全文自检】\n"
                "在你输出最终润色稿之前，必须按以下步骤全文自检：\n"
                "  1. 把【直译底稿】与下方【实体一致性反馈】清单中的每一项逐条对照；\n"
                "  2. 把你**准备输出**的整篇润色稿全文扫一遍，标记任何残留的英文 source_name 或中文 variant；\n"
                "  3. 把所有残留替换为清单中标注的 expected 规范主名；\n"
                "  4. 若清单中标注「missing」的实体原文中出现，**确保译稿中存在对应规范主名**。\n"
                "任何一处残留都会让本次任务再次被 reject。\n"
            )
            entity_feedback_section = (
                self_check_preamble
                + "\n【⚠️ 实体一致性反馈 (CRITICAL — 上次被拒原因，必须逐项修复)】\n"
                + f"{entity_feedback}\n"
                + "请在本次修订中**严格按上述清单**逐项修正译稿中的实体译名，"
                "尤其是清单中标注「missing」的项，必须在译稿中补回；标注「variant_instead_of_canonical」的项，必须替换为主名。\n"
            )

        # Round 23: 从 reject_reason 中解析 Jev reject categories，渲染顶部"硬约束维度"section
        # —— 让 rewriter 拿到与 entity_feedback 同级的 explicit 维度指引
        jev_categories_section = self._build_jev_category_section(reject_reason)

        # Round 64: 渲染"边界判断跳过"section —— 让 rewriter 知道哪些维度是 Jev 边界判断
        # （Noul ∈ [0.3, 0.7] 或 Choice/Score confidence < 0.5），未达 reject 门槛。
        # 本轮不需要重点修改这些维度；如确实需要修改，请基于整体润色判断而非局部修复。
        skipped_section = ""
        if skipped_dimensions:
            skipped_section = (
                "\n【ℹ️ 上轮 Jev 边界判断跳过（无需重点修改这些维度）】\n"
                "Jev 在以下维度给出边界判断（noul ∈ [0.3, 0.7] 或 confidence < 0.5），"
                "未达 reject 门槛。本轮不需要重点修改这些维度；"
                "如确实需要修改，请基于整体润色判断而非局部修复。\n"
                f"  跳过维度：{', '.join(skipped_dimensions)}\n"
            )

        # 实体登记表 section（紧跟 entity_feedback，方便 Rewriter 查阅）
        entity_registry_section = ""
        if entity_registry:
            # 按 canonical 分组（与 _build_prompt 同语义）
            grouped: Dict[str, List[str]] = {}
            for src_name, zh_canonical in entity_registry.items():
                if src_name and zh_canonical:
                    grouped.setdefault(zh_canonical, []).append(src_name)
            if grouped:
                lines = []
                for canonical, aliases in sorted(grouped.items()):
                    alias_str = " / ".join(aliases)
                    lines.append(f"  - {canonical}（源文中可能的形式：{alias_str}）")
                entity_registry_section = (
                    "\n【实体译名登记表（修订时参考 — 严格遵循）】\n"
                    "译稿中的人名/地名/组织名必须使用下列规范译名。"
                    "源文中如出现任一英文别名（含部分匹配），都应翻译为对应中文译名。\n"
                    "【优先级规则】实体译名若与上方考据/术语决策的写法不一致，"
                    "**以本登记表规范主名为准**（统一为规范主名，不得保留考据决策中的"
                    "异体/简称/缺后缀形式）。\n"
                    + "\n".join(lines)
                    + "\n"
                )

        # Round 56: 把 forbidden_modernisms 注入 retry prompt（Round 54 漏改 — retry 时约束也必须保持）
        forbidden_modernisms = (_config.style_guide or {}).get("forbidden_modernisms", []) or []
        forbidden_section = ""
        if forbidden_modernisms:
            forbidden_str = "、".join(str(w) for w in forbidden_modernisms)
            forbidden_section = (
                "\n【禁用词列表 (CRITICAL — 修订时同样适用)】\n"
                f"以下表达严禁出现在译文中（含其变体形式）：{forbidden_str}\n"
                "修订时若发现上一版有上述任何表达，必须替换为符合风格指南的替代表达。\n"
            )

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
        # Round 24 扩展：Jev reject categories（直译残留 / 关键语义丢失 / 语域偏离 / 整体风格偏离）
        #   也作为整体重润色的早触发条件——这些维度本质是"全篇级问题"，局部词换永远无法解决，
        #   而 max_quality_retries=4 意味着"重写 1-2 段落"分支跑 4 次后会直接 fallback，
        #   等不到 retry_count > 4 的整体重润色分支。
        # Round 28 扩展：entity_consistency 也触发整体重写——实体译名是全篇级约束，
        #   局部词换一个 variant 时可能让另一个 variant 残留，导致下一轮又触发新的不一致。
        style_dims = {"style_compliance", "voice_consistency"}
        low_style = [d for d in (low_dims or []) if d in style_dims]
        entity_low = "entity_consistency" in (low_dims or [])

        # Round 24: Jev primitive categories → 整体重润色策略
        # Round 95: 补齐 3 个 voice_drift 类别中可触发整体重写的两条:
        #   - 音色漂移（severe）: 严重漂移, 单点词换无法解决, 必须整体重写
        #   - 音色漂移（subtle+其它）: 与其它维度叠加, 必须一并整体修正
        # 音色漂移（subtle）单独不 reject (Round 82 设计), 不放入 GLOBAL.
        GLOBAL_REWRITE_CATEGORIES = {
            "直译残留": "本轮必须**基于直译底稿整体重写**：消除译稿中的英文句式套用与逐词堆砌，逐段重组成自然中文表达。局部词换无法解决直译残留问题。",
            "关键语义丢失": "本轮必须**整体重写**以补回原文所有关键意象、动作细节、隐喻、反讽；不得概括或弱化任何一项。局部词换无法挽回已经丢失的语义。",
            "整体风格偏离": "本轮必须**整体重润色**：以目标风格基线为锚点，重新组织全篇的断句、节奏、语域与脚注。",
            "语域偏离（under）": "本轮必须**整体重写**以提升语域：减少口语/方言化表达，逐段匹配目标读者期望的语域。",
            "语域偏离（over）": "本轮必须**整体重写**以降低语域：减少文言/书面化表达，逐段匹配目标读者期望的语域。",
            "音色漂移（severe）": (
                "本轮必须**整体重写**以拉回叙事口吻与角色音色: 保留讽刺/克制/幽默等特征,"
                "避免被抹平为平淡中性叙述. 局部词换无法解决严重音色漂移——"
                "必须从句子节奏、句式选择、叙事视角整体重做."
            ),
            "音色漂移（subtle+其它）": (
                "本轮必须**整体重写**: subtle voice drift 与其它维度 (实体一致性 / 直译残留 / "
                "关键语义丢失) 共同 reject, 需在解决其它核心问题的同时, 把叙事口吻向原文基线 "
                "靠拢 (保留讽刺/克制/幽默), 又不能过度风格化 (避免反向触发 severe)."
            ),
            # Round 107 P5: 风格契合度低 — 全局风格问题, 局部词换无法解决, 必须整体重润色
            "风格契合度低": (
                "本轮必须**整体重润色**以拉高风格契合度: 以【风格基准】为锚点, "
                "重新组织全篇的断句节奏、词汇密度、修辞手法与语域, 使译文风格与目标基线对齐. "
                "局部词换无法解决全局风格偏离——必须从句子结构、段落节奏、用词选择整体重做."
            ),
        }
        # Round 89: GLOBAL_REWRITE_CATEGORIES 的 keys 是 `语域偏离（under）` 等具体形式,
        # 策略查找路径用 normalize=False 保留具体形式.
        jev_cats = self._extract_reject_categories(reject_reason, normalize=False)
        jev_global = [c for c in jev_cats if c in GLOBAL_REWRITE_CATEGORIES]

        # Round 28: entity-specific 整体重写策略
        ENTITY_GLOBAL_STRATEGY = (
            "本轮必须**基于直译底稿整体重写**，且以【实体译名登记表】为最高优先级硬约束。"
            "清单已逐项列出每处违规（含源文形式与应替换为的规范主名）。"
            "**一次重写内**把所有违规实体译名统一为登记表规范主名，**不得**保留任何 variant / source_name 残留形式。"
            "局部词换无法解决实体一致性问题——一处修改可能让另一处 variant 残留。"
        )

        if (low_style and retry_count >= 2) or (entity_low and retry_count >= 1) or (jev_global and retry_count >= 1):
            # Round 24 + Round 28: 选择最具体的策略（优先级：Jev category > entity_consistency > style/voice）
            if jev_global:
                strategy = GLOBAL_REWRITE_CATEGORIES[jev_global[0]]
            elif entity_low:
                strategy = ENTITY_GLOBAL_STRATEGY
            else:
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

        persona = (_config.style_guide or {}).get("persona", "职业文学译者")
        footnote_policy = (_config.style_guide or {}).get("footnote_policy", "inline")

        footnote_rules = {
            "none": "禁止在润色阶段添加脚注。如遇需要注释的典故，用文内括号简注。",
            "inline": "如遇需要注释的典故，应在首次出现时用文内括号简注，而非使用脚注。",
            "markdown_footnote": "如遇需要注释的典故，可使用 Markdown 脚注语法 [^1]。",
        }
        footnote_instruction = footnote_rules.get(footnote_policy, footnote_rules["inline"])

        register_block = ""
        if register_target:
            register_block = (
                f"【本章目标语域（必须遵循，重写时以此为准）】\n{register_target}\n"
            )

        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        # Round 119 L1: 韵文 chunk 注入逐行对应/复沓保留协议（retry 轮同样携带）
        verse_section = verse_note if verse_note else ""
        return f"""你是一位{persona}。
这是第 {retry_count + 1} 次修订。上一版译文因以下原因被终审拒绝。

{decisions_context}

{strategy}

{register_block}

【上一版被拒原因（必须解决）】
{reject_reason}{critic_section}
{jev_categories_section}
{skipped_section}
{entity_feedback_section}
{history_section}
{specific_edits_section}

【实体译名登记表（修订时参考 — 严格遵循）】
{entity_registry_section}{forbidden_section}

【风格基准 (Style Guide)】
{style_guide}

{verse_section}【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. {footnote_instruction}
3. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。
4. 脚注只陈述典故/文化背景/双关机制本身；不得出现任何翻译流程元语（如"保留与加注策略""保留并加注""保留音译""按约束""直译自""策略池"等字样），不得描述译者的处理方式或决策理由。
5. 原文里的方括号编辑标记（如 [later editions continued as follows]）必须保留并译出，用【】包裹（如【后来版本续之如下：】），不得删除或改写成叙述。
6. 注释一律使用 Markdown 脚注 [^n] 并带正文引标；不得新增"【章末注释】""【脚注】"等标题式注释块，不得输出无引标的裸编号注列表（如 [1] xxx 列表）。

【人称代词规则】
1. 严格遵循原文人称代词性别：he/him/his → "他"，she/her → "她"，it/its → "它"。
2. 注意原文中人物对话和叙述视角的代词指代关系，不得混淆角色性别。

【直译底稿（语义基准，不可偏离）】
{raw_translation}

【上一版译文（在此基础上精修，只动被拒原因相关段落）】
{prev_lit_text}

请直接输出修订后的 Markdown 文本，重点解决被拒原因，其他部分保持稳定。

【CRITICAL — 必须修改】
你必须对上一版译文做出**至少一处实质性修改**（改写句子、调整语序、替换用词、补充细节等）。
如果输出与上一版完全相同（字节级一致），本次任务将直接判定失败（REWRITER_STAGNATED）。
请确保你的输出与【上一版译文】有可见差异。
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

        # 2. 诗歌/韵文特征（P1-4: 只保留作品无关的结构性标记；作品特定诗句样例已移除）
        poetry_markers = ['\n\n', '——', '...']
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
        register_default = (_config.style_guide or {}).get("register_default", "保持克制、平实、贴近原文的叙事语域")
        if not source_text:
            return register_default
        low = 0.0
        high = 0.0
        plain_markers = ['I suppose', 'I think', 'you see', 'of course', 'rather',
                         'however', 'perhaps', 'I mean', 'you know', 'the fact is',
                         'it seems', 'I confess', 'we were', 'I was']
        low += sum(source_text.count(m) for m in plain_markers)
        literary_markers = ['glowing', 'shimmering', 'gleamed', 'silence', 'vast',
                            'shadow', 'wonder', 'marvellous', 'strange beauty',
                            'soft', 'dim', 'ghost']
        high += sum(source_text.count(m) for m in literary_markers)
        if source_text.count('"') > 6:
            low += 1.5
        if high > low + 1.5:
            return ("本章原文意象/描写密度高，可适度文学化，但仍须以克制语域为底；"
                    "禁止文言虚词（之/乃/遂/盖）、四字格堆砌与超过 3 条脚注。")
        if low >= high + 1.0:
            return ("本章为叙述者口语化自述/科学论述，目标语域=克制平实、贴近原文的闲谈；"
                    "在直译底稿基础上做文学润色（断句、节奏、语气词），但不得退化为直译稿，"
                    "也不得文言化；脚注不超过 3 条。")
        return register_default

    def _build_style_guide(self, style_guide_stats: dict, source_text: str = "") -> str:
        """构建风格指南，包含动态 Author_Priority_Ratio 与章节级 register 目标（N 方案）"""
        if source_text:
            author_priority = self._infer_author_priority_ratio(source_text)
        else:
            author_priority = style_guide_stats.get('author_priority_ratio', 0.7)
        register_target = self._infer_register_target(source_text) if source_text else \
            (_config.style_guide or {}).get("register_default", "保持克制、平实、贴近原文的叙事语域")

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
        # Rewriter↔Entity 回路修复：新增 2 个参数
        entity_feedback: Optional[str] = None,  # 上轮实体的具体反馈（含「X 应统一为 Y」清单）
        entity_registry: Optional[Dict[str, str]] = None,  # 实体译名登记表 {english_source: chinese_canonical}
        retry_count: int = 0,
        all_feedback: Optional[List[Dict]] = None,
        specific_edits: Optional[List[Dict]] = None,
        narrative_memory: str = "",
        low_dims: Optional[List[str]] = None,
        # Round 64: 让 rewriter 感知被 Jev uncertain zone 跳过的维度
        skipped_dimensions: Optional[List[str]] = None,
        verse_note: Optional[str] = None,  # Round 119 L1 韵文协议
        model_role: Optional[str] = None,  # Round 116: 停滞换模型覆盖角色
        model_effort: Optional[str] = None,  # Round 129: 换模型时覆盖 effort（如 primary→max）
    ):
        print(f"✍️ [Rewriter Agent] 正在进行文学润色: {chunk_id} (retry={retry_count})...")

        style_guide = self._build_style_guide(style_guide_stats, source_text)
        decisions_context = self._build_decision_context(source_text)
        # N 方案：章节级 register 目标，作为 retry 的方向锚点
        register_target = self._infer_register_target(source_text) if source_text else ""

        # Round 116: 停滞换模型 — 本次调用可覆盖角色（客户端/模型名随之切换；
        # Round 127: extra_body 为任务级配置，不随角色变化）
        llm = self.llm
        role = self.role
        if model_role and model_role != role:
            role = model_role
            llm = get_llm_client(role)

        # 使用配置路由：literary_rewrite（先于 temperature 计算，确保 base 从 config 读取）
        model_key, params = _config.resolve_task_model("literary_rewrite")
        # Round 129: 停滞换模型/双稿第二稿切到更强模型时，按调用方要求加档
        if model_effort:
            params["effort"] = str(model_effort).strip().lower()
        model_name = get_role_model_name(role) or model_key
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
                entity_feedback=entity_feedback,           # 回路修复
                entity_registry=entity_registry,           # 回路修复
                retry_count=retry_count,
                all_feedback=all_feedback,
                specific_edits=specific_edits,
                register_target=register_target,
                low_dims=low_dims,
                skipped_dimensions=skipped_dimensions,  # Round 64
                verse_note=verse_note,
            )
            temperature = self._calculate_retry_temperature(base_temp, retry_count)
        else:
            prompt = self._build_prompt(
                raw_translation, decisions_context, style_guide,
                narrative_memory=narrative_memory,
                entity_registry=entity_registry,            # 回路修复（首次润色也有登记表参考）
                verse_note=verse_note,
            )
            temperature = base_temp

        # 覆盖 temperature（用计算后的值替换 params 中的配置值）
        params = {**params, "temperature": temperature}

        final_markdown = llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
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
                    if backend.primary.api_key and not getattr(backend, "_jev_auth_dead", False):
                        try:
                            answers = backend.primary.evaluate(state, questions, timeout=backend.timeout)
                            backend_path = "jev"
                        except Exception as e_jev:
                            backend_diagnostics["jev"] = _classify_backend_error(e_jev)
                            if type(e_jev).__name__ == "TypeSafeAuthenticationError":
                                backend._jev_auth_dead = True
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
            # Round 130: 内层引号用单引号——f-string 内嵌套同类引号需 PEP 701（3.12+），
            # README 声明支持 >=3.10，此处曾致 3.10 生产解释器全库 SyntaxError。
            print(f"⚠️ [Jev] typesafe 路径失败 [{diag['category']}]: {diag['msg']} → {diag['action']}", file=sys.stderr)

        # ───── 旧 LLM 路径（fallback） ─────
        print(f"🧐 [Critic Agent] 正在对 {chunk_id} 进行多维度文学审计...")
        prompt = self._build_prompt(source_text, raw_trans, lit_trans, style_guide)

        model_key, params = _config.resolve_task_model("critic_scoring")
        model_name = get_role_model_name(self.role) or model_key

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
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



class JudgeAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client_for_task("judge_decision")
        self.role = _config.get_task_role("judge_decision")
        self.db = decision_engine

    @staticmethod
    def _sanitize_critic_report_for_llm(critic_report: dict) -> dict:
        """Round 106 P2: 移除 mock_heuristic 的假分数/verdict，避免带偏 LLM 裁决。

        mock 后端（Jev 不可用）生成的 scores/_typesafe_verdict/critique 是确定性
        假数据，喂给 LLM judge 会使其把假分数当真实质量信号（ch05 实测 7/7 chunk
        被 REJECT）。移除这些字段，保留 entity_consistency_notes（本地核查，可靠）
        和 specific_edits，让 LLM 纯凭译文质量独立判断。
        """
        sanitized = dict(critic_report)
        for key in ("scores", "_typesafe_verdict", "critique", "improvement_suggestions"):
            sanitized.pop(key, None)
        sanitized["_note"] = (
            "（critic 后端为 mock_heuristic，Jev 不可用，无可靠评分；"
            "请纯凭【原文】与【当前润色稿】的翻译质量独立判断，不要假设存在低分维度）"
        )
        return sanitized

    def _second_opinion_reject_edits(
        self,
        chunk_id: str,
        source_text: str,
        lit_trans: str,
        verdict: "CompositeVerdict",
        reject_reason: str,
    ) -> tuple:
        """Round 118 P1-2: REJECT 第二意见 → 可执行编辑清单.

        Jev 的 reject_reasons 常只有概率数字（如 "noul=0.82"），rewriter 无法执行，
        导致 REWRITER_STAGNATED（ch32 实测）。本方法在 REJECT 后触发一次独立 LLM
        复核（judge_decision 角色），把拒因翻译成逐条可执行编辑；不改变 Jev 的
        decision（绝不弱化仲裁），只补 rewriter 所需的具体指令。

        Returns:
            (edits, info): edits 兼容 pipeline 的 specific_edits 结构；
            info 为审计元数据（confirmed_reject=false 表示第二意见认为拒因
            无可执行修改点，仅供参考，不用于翻案）。
        """
        cfg = _config.pipeline.get("judge_second_opinion", {}) or {}
        if not cfg.get("enabled", True):
            return [], {"enabled": False}
        max_edits = int(cfg.get("max_edits", 20))
        reasons = (reject_reason or "; ".join(verdict.reject_reasons))[:1500]
        summary_json = json.dumps(verdict.primitive_summary, ensure_ascii=False)[:2500]
        prompt = f"""你是资深文学翻译终审编辑。另一位校准裁判已给出 REJECT，但拒因多为概率信号，润色师无法执行。
你的任务：仅产出**可执行的修改指令**（不要重新评分、不要讨论是否改判）。

【原文】
{source_text[:2500]}

【当前润色稿】
{lit_trans[:2500]}

【裁判拒因】
{reasons}

【原语信号（JSON）】
{summary_json}

【输出 JSON — 严格遵守】
{{"edits": [{{"original": "译文中的问题片段（照抄，便于定位）", "problem": "问题说明（一句话）", "suggested_direction": "具体修改方向或替换建议"}}], "confirmed": true}}

【规则】
1. 只列真实存在、可定位的修改点；每条 original 必须能在润色稿中找到。
2. 若拒因确无可执行修改点（疑似概率噪声），edits 输出空数组，confirmed 置 false。
3. 最多 {max_edits} 条；不要输出 JSON 以外的任何文字。"""
        model_key, params = _config.resolve_task_model("judge_decision")
        model_name = get_role_model_name(self.role) or model_key
        params = dict(params)
        params["temperature"] = float(cfg.get("temperature", 0.2))
        try:
            raw = self.llm.generate(
                prompt=prompt, model_name=model_name, **params,
            )
            data = robust_json_loads(raw)
            if not isinstance(data, dict):
                raise RuntimeError("第二意见输出非 JSON 对象")
            edits: List[Dict[str, Any]] = []
            for i, e in enumerate(data.get("edits") or [], 1):
                if not isinstance(e, dict):
                    continue
                direction = str(e.get("suggested_direction", "")).strip()
                problem = str(e.get("problem", "")).strip()
                if not direction and not problem:
                    continue
                edits.append({
                    "index": i,
                    "original": str(e.get("original", ""))[:400],
                    "problem": problem[:300],
                    "suggested_direction": direction[:300],
                })
            confirmed = bool(data.get("confirmed", True))
            info = {
                "triggered": True,
                "edits": len(edits),
                "confirmed_reject": confirmed,
                "model_role": "judge_decision",
            }
            print(f"🧭 [Judge Agent] {chunk_id} REJECT 第二意见: edits={len(edits)} confirmed={confirmed}")
            return edits[:max_edits], info
        except Exception as e:
            print(
                f"⚠️ [Judge Agent] {chunk_id} REJECT 第二意见失败"
                f"（保留原裁决，不阻断流程）: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return [], {"triggered": True, "error": f"{type(e).__name__}: {e}"}

    def _build_prompt(self, source_text: str, lit_trans: str, critic_report: dict, rewriter_self_review: Optional[list] = None, verse_note: Optional[str] = None) -> str:
        # P0: 将 critic_thresholds 注入为参考阈值（非硬性否决条件）
        thresholds = _config.critic_thresholds_for_book()
        thresholds_lines = "\n".join(
            f"  - {dim}: 参考阈值 {v}"
            for dim, v in thresholds.items() if dim != "average_score_min"
        )
        # Round 106 P2: 无 scores 时（mock 后端已清洗），不引用阈值，纯凭译文质量判断
        if critic_report.get("scores"):
            thresholds_block = f"""
【评分参考阈值（供综合判断，不是硬性否决条件）】
审辩者各维度的参考阈值为：
{thresholds_lines}

如果审辩者报告的 scores 中某维度略低于阈值，但你认为整体质量达标，可以判 PASS。
如果多个维度显著低于阈值，且润色稿确实问题明显，才判 REJECT。
"""
        else:
            thresholds_block = """
【判断依据】
审辩者未提供可靠评分（后端不可用）。请纯凭【原文】与【当前润色稿】的翻译质量
（流畅度、可读性、风格、语义保真、实体译名一致性）独立判断：质量达标判 PASS；
确有明显问题（直译腔、生硬拗口、漏译、实体译名错误）才判 REJECT，并在
reject_reason 中说明具体位置与改法。
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

        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        persona = (_config.style_guide or {}).get("persona", "职业文学译者")
        # Round 119 L1: 韵文对齐检查注入（仅 verse chunk 携带）
        verse_block = ""
        if verse_note:
            verse_block = (
                "\n【韵文对齐检查（本片段含韵文，必须逐行核对）】\n"
                f"{verse_note}"
                "若润色稿把韵文合并成散文行、诗行数明显不足（低于协议下限）"
                "或复沓（refrain）丢失，应判 REJECT，并在 reject_reason 中"
                "要求逐行对齐重排。\n")
        return f"""你是一位{persona}，担任本次{genre}终审。
你需要综合【审辩者报告】，决定当前的【文学润色稿】是否可以直接定稿。

【审辩者报告】
{json.dumps(critic_report, ensure_ascii=False)}
{self_review_block}
【原文】
{source_text}
【当前润色稿】
{lit_trans}
{verse_block}
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

    def process_chunk(self, chunk_id: str, source_text: str, lit_trans: str, critic_report: dict, affected_chunks: List[str] = None, rewriter_self_review: Optional[list] = None, verse_note: Optional[str] = None) -> Dict[str, Any]:
        """最终裁决

        T2 重构：
          1. 优先 typesafe-ai verdict 合成（code-side，无 LLM 成本）
          2. 若 critic_report 携带 _typesafe_verdict，则直接复用其 verdict
          3. 否则 fallback 旧 LLM 路径
        """
        # ───── T2 typesafe-ai 路径：从 critic 传过来的 verdict 直接使用 ─────
        verdict: Optional[CompositeVerdict] = critic_report.get("_typesafe_verdict")
        if isinstance(verdict, dict) and "decision" in verdict:
            verdict = CompositeVerdict(**{k: v for k, v in verdict.items() if k in (
                "decision", "confidence", "reject_reasons", "primitive_summary",
                "confidence_source", "confidence_calibrated",
            )})

        # Round 106: mock_heuristic 的 verdict 不可信。Jev API 不可用（如 451 区域封锁）
        # 时 critic 回退到 mock 启发式，其 _typesafe_verdict 是确定性假分数（ch05 实测
        # entity_consistency=0 / literal_residue 极低 → 整章 7/7 chunk PERMANENTLY_FAILED）。
        # 默认不信任（trust_mock_heuristic=false）：强制走 LLM 独立裁决，让真正的质量门生效。
        if (isinstance(verdict, CompositeVerdict)
                and not _config.pipeline.get("trust_mock_heuristic", False)
                and critic_report.get("_backend_path") == "mock_heuristic"):
            print(f"⚖️ [Judge Agent] {chunk_id} critic 后端为 mock_heuristic（Jev 不可用），"
                  f"跳过 typesafe verdict 复用，改用 LLM 独立裁决")
            verdict = None

        if isinstance(verdict, CompositeVerdict):
            backend_path = critic_report.get("_backend_path", "unknown")
            reject_reason = "; ".join(verdict.reject_reasons) or "全部维度通过"
            result = {
                "decision": verdict.decision,
                "reject_reason": reject_reason,
                "unresolved_edits": [],
                "_typesafe_verdict": verdict.to_dict(),
                "_backend_path": backend_path,
            }
            # 安全网：typesafe 路径也保留（即便 verdict=PASS，confidence 极低时仍拒）
            safety_min = float(_config.pipeline.get("judge_safety_net_avg_min", 5.5))
            scores = critic_report.get("scores", {})
            if scores and result["decision"] == "PASS":
                avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / max(1, len(scores))
                if avg_score < safety_min and verdict.confidence < 0.5:
                    result["decision"] = "REJECT"
                    result["reject_reason"] = (
                        f"安全网(typesafe): 平均分 {avg_score:.1f} < {safety_min} 且"
                        f" verdict 置信度 {verdict.confidence:.2f} 偏低，疑似启发式噪声"
                    )
            # Round 118 P1-2: REJECT 第二意见 — 把概率拒因翻译成可执行编辑。
            # 不改变 decision（绝不弱化 Jev），仅补 rewriter 所需的具体指令。
            if result["decision"] == "REJECT":
                second_edits, second_info = self._second_opinion_reject_edits(
                    chunk_id, source_text, lit_trans, verdict, result["reject_reason"],
                )
                if second_edits:
                    result["unresolved_edits"] = second_edits
                if second_info:
                    result["_second_opinion"] = second_info
            # style rule 提取暂不做（typesafe 不生成新规则，避免误判；保留旧路径能力）
            return result

        # ───── 旧 LLM 路径（fallback） ─────
        # Round 106 P2: mock_heuristic 的分数/verdict 不可信，喂给 LLM 会带偏裁决
        # （ch05 实测 LLM judge 反复引用 mock 假分数 readability=2.5/fluency=2.5 作为拒因，
        #  导致 7/7 chunk 全被 REJECT）。清洗 critic_report：移除 mock 的假分数/verdict，
        #  让 LLM 纯凭译文质量独立判断。
        if (not _config.pipeline.get("trust_mock_heuristic", False)
                and critic_report.get("_backend_path") == "mock_heuristic"):
            critic_report = self._sanitize_critic_report_for_llm(critic_report)
        print(f"⚖️ [Judge Agent] 正在对 {chunk_id} 进行最终裁决...")
        prompt = self._build_prompt(source_text, lit_trans, critic_report, rewriter_self_review=rewriter_self_review, verse_note=verse_note)

        model_key, params = _config.resolve_task_model("judge_decision")
        model_name = get_role_model_name(self.role) or model_key

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
        )

        result = robust_json_loads(raw_output)
        if not result:
            raise RuntimeError("Judge 输出 JSON 损坏，重新裁决。")

        scores = critic_report.get("scores", {})
        # Round 106: mock_heuristic 的分数不可信（Jev 不可用时的兜底假分数），
        # 安全网不用 mock 分数否决 LLM 的 PASS，否则 LLM 判过也会被 mock 拉回 REJECT。
        _mock_untrusted_safety = (
            not _config.pipeline.get("trust_mock_heuristic", False)
            and critic_report.get("_backend_path") == "mock_heuristic"
        )

        if scores and not _mock_untrusted_safety:
            avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / max(1, len(scores))
            safety_min = _config.pipeline.get("judge_safety_net_avg_min", 5.5)
            if result.get("decision") == "PASS" and avg_score < safety_min:
                result["decision"] = "REJECT"
                result["reject_reason"] = (f"安全网: Critic 平均分 {avg_score:.1f} 低于安全阈值 {safety_min}，"
                                           f"但 Judge 判定通过，疑似 LLM 幻觉。")

        # style rule 提取（仅旧 LLM 路径支持；typesafe 不生成新规则，避免误判）
        if result.get("decision") == "PASS" and "new_style_rule" in result and result["new_style_rule"]:
            rule = result["new_style_rule"]
            desc = (rule.get("rule_description") or "").strip()
            if desc:
                import hashlib
                fp = hashlib.sha1(desc.encode("utf-8")).hexdigest()[:12]
                self.db.add_decision(
                    level=DecisionLevel.STYLE,
                    source=f"style_{fp}",
                    translation=desc,
                    reason=rule.get("reason", "Judge Agent 动态提炼"),
                    affected_chunks=None,
                )

        result["_backend_path"] = "legacy"
        return result






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

        # Round 117 P0: 动态风格剖面加载（StyleProfiler 产物）→ 运行时 style_guide 覆盖层。
        # 剖面由 `python3 -m src.translator_agent style-profile` 生成并落盘 input/book_profile.json;
        # 缺失/损坏时安静跳过（保持 config.yaml 基线, 不注入任何作品特定先验）。
        self.book_profile = None
        try:
            from src.style_profiler import load_profile, profile_to_style_guide
            _prof_path = self._root_dir / "input" / "book_profile.json"
            _prof = load_profile(_prof_path)
            if _prof:
                self.book_profile = _prof
                _config.set_runtime_style_guide(profile_to_style_guide(_prof, chapter_id=chapter_id))
                print(f"🎨 [StyleProfiler] 已加载作品风格剖面 ← {_prof_path.name} "
                      f"(source={_prof.get('source')}, confidence={_prof.get('confidence')})")
        except Exception as _prof_err:
            print(f"⚠️ [StyleProfiler] 剖面加载失败，忽略: {_prof_err}", file=sys.stderr)

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

        # Round 13: LLM 调用计数器（按 task_type 分组）
        # Round 65 修订：每次 process_chunk 调用 + quality_retries 次重试 = 实际 LLM 调用次数
        # （之前 +=1 漏算 retry，导致 cost_report 严重低估 literary_rewriting / critic_scoring / judge_decision 成本）
        self._llm_call_counter = {
            "reference_extraction": 0,
            "literal_translation": 0,
            "literary_rewriting": 0,
            "critic_scoring": 0,
            "judge_decision": 0,
        }

        # 从配置加载风格指南
        # Round 79: 之前只 snapshot 3 个键 (avg_sentence_length / lexicon_preference /
        # author_priority_ratio), 缺 genre/persona/work_type/register_default/
        # footnote_policy/forbidden_modernisms 6+ 个键. partial dict 传给 critic_agent
        # 被注入 critic prompt, 导致 critic 看不到 persona/禁用词/时代语域约束, 影响评分准确性.
        # 现在直接用完整 _config.style_guide dict, 让下游 Agent 看到所有约束.
        # (rewriter_agent 内部本来就用 _config.style_guide.get(...) 读, 不受影响.)
        self.style_guide = dict(_config.style_guide) if _config.style_guide else {}

        # Round 71: 跨人物冲突 sibling 缓存. 每次审计调用 _check_entity_consistency 时,
        # 第一次会从 entity_registry 计算 sibling 集合, 后续复用. 在 _process_auditing_batch
        # 入口重置, 避免跨章节脏数据.
        self._sibling_canonicals_cache: Optional[Dict[str, Set[str]]] = None

        # Round 119 L1: 韵文通道运行时缓存（tags/manifest 惰性加载；无韵文时零开销）
        self._verse_tags_cache: Optional[dict] = None
        self._verse_manifest_cache: Optional[dict] = None
        self._verse_note_cache: Dict[str, str] = {}

        # Round 92: 最佳文学稿 fallback 状态. 跨 retry 追踪最高 critic 平均分对应的 lit_text,
        # fallback 时 (stagnation / oscillation / max_retries / safety_net) 用此版本,
        # 而不是当前被 critic 拒掉的版本. 用户看到的是"最高质量的版本"而不是"最差一次的版本".
        # key=chunk_id, value=(avg_score, lit_text)
        self._best_lit: Dict[str, Tuple[float, str]] = {}
        # Round 118 P2-8: 候选稿池（top-N by critic avg），供 Jev 配对仲裁做最终选择。
        # key=chunk_id, value=List[(avg_score, lit_text)] 按 avg 降序、文本去重
        self._lit_candidates: Dict[str, List[Tuple[float, str]]] = {}
        self._pairwise_info: Dict[str, Dict[str, Any]] = {}

        # Round 93: 每 chunk 各 retry 的 critic avg_score 历史. 用于检测"连续下降"模式
        # (e.g., retry 0=6.5, retry 1=6.0, retry 2=5.5), 提前 fallback 节省 LLM 调用.
        # key=chunk_id, value=List[float] 按 retry 顺序追加
        self._score_history: Dict[str, List[float]] = {}
        # Round 98 P1: 实体违规计数历史. 类似 _score_history, 但跟踪每次 audit 后
        # serious 实体违规数. 用于检测"实体违规连续 N 次未修复"模式, 提前 fallback
        # 节省 rewriter+critic LLM 调用. ch32 实测每次 retry 仍 serious=1-2,
        # rewriter 反复失败, 早停可在 quality_retries=2 时止损.
        self._entity_violation_history: Dict[str, List[int]] = {}
        # Round 99 P1: per-chunk entity_registry 缓存. 同一 chunk 的所有 retry 用同一份 source_text,
        # registry 在 chapter 内不变 (除非 ext 模式重载). 缓存避免每次 retry 重跑 query_reviewable_entities
        # SQL + JSON parse. chapter 切换时清空 (在 run() 重置, 见 _reset_per_run_cache).
        self._entity_registry_cache: Dict[str, Dict[str, str]] = {}
        self._entity_variants_cache: Dict[str, Dict[str, List[str]]] = {}
        # Round 99 P2: intermediate 文件读缓存. key=(chunk_id, step), value=loaded data.
        # 同 chunk × step 多次加载走内存, 写时自动 invalidate (见 _save_intermediate).
        self._intermediate_cache: Dict[Tuple[str, str], object] = {}
        # Round 100 P1: narrative_memory 读缓存. _load_prior_memory 每次 rewriter retry
        # 都重读 _memory.jsonl + JSON parse + filter. 同 chunk 跨 retry 内容不变, 缓存命中.
        # _append_chunk_memory 时 invalidate 对应 chunk 的 cache entry (内容变了).
        self._narrative_memory_cache: Dict[str, str] = {}
        # Round 100 P2: _compute_avg_score 缓存 (key=id(critic_report))
        self._avg_score_cache: Dict[int, float] = {}

    # Round 92: 最佳文学稿追踪 + fallback 选择
    # Round 100 P2: 实例方法 (非 staticmethod), 因为需要访问 self._avg_score_cache
    # Round 103 P1: 改为 classmethod, 兼容测试
    # `TranslationPipeline._compute_avg_score(cr)` (无 self) 调用.
    # 实例方法 `self._compute_avg_score(cr)` 也工作 (Python 自动传 cls=self).
    #
    # Round 103 P3: 删除 Round 100 P2 的 id() 缓存. 原设计假设 id() 在
    # critic_report 生命周期内稳定, 但 CPython 在 dict 被 GC 后可复用 id() —
    # 实测导致测试 test_zero_filtered_at_update_time 出现 [7.0, 7.0] 而非
    # [7.0, 6.0] 的 bug. avg 计算本身是 O(6) sum/len, 微秒级, 不值得为它
    # 引入 id() 缓存的 correctness risk.
    #
    # 如未来确有性能需求, 用 weakref.WeakKeyDictionary 替代 (但 dict 不
    # 支持 weakref, 须包装或换用 @dataclass(frozen=True) CriticReport 类).

    @classmethod
    def _compute_avg_score(cls, critic_report: dict) -> float:
        """从 critic_report 提取所有数值维度 score 的算术平均. 无 scores 返回 0.0.

        Round 92: 引入. Round 100 P2: 加 id() 缓存 (Round 103 P3 移除).
        """
        if not isinstance(critic_report, dict):
            return 0.0
        scores = critic_report.get("scores", {}) or {}
        numeric = [v for v in scores.values() if isinstance(v, (int, float))]
        return (sum(numeric) / len(numeric)) if numeric else 0.0

    def _update_best_lit(self, chunk_id: str, lit_text: str, critic_report: dict) -> None:
        """把当前 retry 的 (avg_score, lit_text) 与已存最佳比较, 若更高则更新.

        设计原则:
          - 即使 critic reject (judge=REJECT) 也保存当前版本 — 因为它的 avg_score 可能比之前高
          - 即使 critic pass (judge=PASS) 也保存 — 防御性, 万一后续 retry 不小心覆盖更高分版本
          - 当 chunk 完成 (judge PASS, fallback 触发, 或 chapter end) 时, _best_lit 保留
            直至下一轮 processing 清理
          - Round 93: 同时维护 _score_history[chunk_id], 追加当前 avg_score 用于"连续下降"检测
        """
        avg = self._compute_avg_score(critic_report)
        existing = self._best_lit.get(chunk_id)
        if existing is None or avg > existing[0]:
            self._best_lit[chunk_id] = (avg, lit_text)
        # Round 118 P2-8: 维护候选池（去重，按 avg 降序截断）
        if lit_text:
            self._ensure_candidate_pools()
            cands = self._lit_candidates.setdefault(chunk_id, [])
            if all(t != lit_text for _s, t in cands):
                cands.append((avg, lit_text))
                cands.sort(key=lambda st: st[0], reverse=True)
                max_cands = int(
                    (_config.pipeline.get("pairwise_arbitration", {}) or {})
                    .get("max_candidates", 4)
                )
                del cands[max(1, max_cands):]
        # Round 93: 同步记录 score 历史 (无论是否 best). 仅记录数值 > 0 的样本,
        # critic 评分空时 (avg=0.0) 不污染历史.
        if avg > 0.0:
            self._score_history.setdefault(chunk_id, []).append(avg)

    def _is_declining_streak(self, chunk_id: str, current_score: float, consecutive: int = 3) -> bool:
        """检查 chunk 是否处于"连续下降"模式.

        Returns True 当 _score_history[chunk_id] 末尾 `consecutive` 个分数严格递减.

        设计:
          - 默认 consecutive=3: 连续 3 次 retry 分数递减视为不可能改善, 提前 fallback
          - 边界: history 长度 < consecutive → 返回 False (样本不足, 不下结论)
          - 例外: 任一分数 == 0.0 (无评分) 不计入 (避免 critic 返回空误触发)
          - 严格递减: s[i-1] > s[i] (非 ">=", 允许偶然相等)

        用法: judge 阶段 REJECT 后, 调用此方法判断是否直接 fallback (跳过 rewriter).
        """
        hist = self._score_history.get(chunk_id, [])
        # 末尾 consecutive 个 + 当前 score = consecutive 个样本
        samples = hist[-consecutive + 1:] + [current_score] if consecutive > 1 else [current_score]
        if len(samples) < consecutive:
            return False
        # 仅保留 > 0 的样本 (过滤 critic 空评分)
        samples = [s for s in samples if s > 0.0]
        if len(samples) < consecutive:
            return False
        # 严格递减检查
        return all(samples[i] > samples[i + 1] for i in range(len(samples) - 1))

    # Round 98 P2: 非进展 streak — 比 _is_declining_streak 更激进
    # _is_declining_streak 要求严格递减 (s[i-1] > s[i]), 遇到「持平」场景不触发
    # (rewriter 输出虽然不恶化但也不进步, 浪费 LLM). 本检查: 末尾 N 个样本任意两次间
    # 都满足 next <= prev (即单调不增), 即触发. 实际语义: 连续 N 次 retry 没有净改善.
    def _is_no_progress_streak(self, chunk_id: str, current_score: float, window: int = 3) -> bool:
        """检查 chunk 是否处于「连续不增」模式 (含持平).

        Returns True 当 _score_history[chunk_id] 末尾 window 个样本与当前 score 形成
        window 个连续非增对比 (即: s[0] >= s[1] >= ... >= s[window-1] >= current).

        设计:
          - 默认 window=3: 与 _is_declining_streak 默认值一致 (可独立调)
          - 边界: history 长度 < window → 返回 False (样本不足)
          - 例外: 任一分数 == 0.0 不计入 (过滤 critic 空评分)
          - 单调不增: s[i] <= s[i+1] (允许相等, 与 _is_declining_streak 的严格递减互补)
        """
        if window <= 1:
            return False
        hist = self._score_history.get(chunk_id, [])
        # 末尾 window-1 个历史 + 当前 score = window 个样本
        samples = hist[-(window - 1):] + [current_score] if window > 1 else [current_score]
        if len(samples) < window:
            return False
        # 仅保留 > 0 的样本 (过滤 critic 空评分)
        samples = [s for s in samples if s > 0.0]
        if len(samples) < window:
            return False
        # 单调不增检查 (next <= prev)
        return all(samples[i] >= samples[i + 1] for i in range(len(samples) - 1))

    # Round 99 P1: chapter 切换时清空 per-chunk 缓存, 避免跨 chapter 脏数据.
    def _reset_per_run_cache(self) -> None:
        """chapter 切换时调用 — 清空 entity_registry 缓存和 entity_violation history.

        设计: 与 sibling_canonicals_cache 同样在 _process_auditing_batch 起手清空,
        但 entity_violation_history 与 score_history 是累积统计, 不应在 batch 间清空
        (它们按 chunk_id 索引, 不同 chapter 用不同 chunk_id, 互不干扰).
        这里只清空 _entity_registry_cache 和 _entity_variants_cache, 因为它们
        按 chunk_id 索引但数据可能因 chapter 切换而不准确.
        """
        # 容错: 若字段未初始化 (例如测试桩或外部调用), 跳过清空
        if hasattr(self, '_entity_registry_cache') and self._entity_registry_cache is not None:
            self._entity_registry_cache.clear()
        if hasattr(self, '_entity_variants_cache') and self._entity_variants_cache is not None:
            self._entity_variants_cache.clear()
        if hasattr(self, '_intermediate_cache') and self._intermediate_cache is not None:
            self._intermediate_cache.clear()
        if hasattr(self, '_narrative_memory_cache') and self._narrative_memory_cache is not None:
            self._narrative_memory_cache.clear()
        # Round 103 P3: 删除 _avg_score_cache 实例字段 (Round 100 P2 引入)
        # 在 _reset_per_run_cache 中清理. 现 _compute_avg_score 不再缓存, 无需清理.
        # 保留 hasattr 检查向后兼容 (若未来重新引入缓存).
        if hasattr(self, '_avg_score_cache') and self._avg_score_cache is not None:
            self._avg_score_cache.clear()
        # Round 102 P1: cjk_ratio 缓存是模块级 (非实例字段), 跨 chapter 累积无意义, 同步清空.
        try:
            reset_cjk_ratio_cache()
        except (NameError, AttributeError):
            pass  # 单文件 shim 模式下模块级 cache 名称不同, 容错
        # 注: _intermediate_cache 字段在 __init__ 初始化 (line 3386),
        #     这里只清空, 不重复声明.

    # ── Round 116: 停滞换模型（生成端多样性）────────────────────────────────
    _RW_SWITCH_STEP = "rewrite_switch"

    def _read_rewriter_switch_state(
        self, chunk_id: str, base_role: str, alt_role: str, max_switches: int
    ) -> dict:
        """读取换模型 sidecar，决定本轮润色应使用的角色。

        sidecar 格式: {"switches": [{"from","to","retry","reason","timestamp"}, ...]}
        语义：当前角色 = 最后一次 switch 的 to（无记录则 base_role）；
              next_role = base/alt 交替；can_switch 受 max_switches 上限约束。
        """
        switches: list = []
        try:
            data = self._load_intermediate(chunk_id, self._RW_SWITCH_STEP)
            if isinstance(data, dict):
                switches = list(data.get("switches", []) or [])
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
            switches = []
        current_role = base_role
        for sw in switches:
            to_role = (sw or {}).get("to")
            if to_role:
                current_role = to_role
        next_role = alt_role if current_role != alt_role else base_role
        try:
            max_sw = int(max_switches)
        except (TypeError, ValueError):
            max_sw = 1
        return {
            "current_role": current_role,
            "switches": len(switches),
            "can_switch": len(switches) < max(0, max_sw),
            "next_role": next_role,
        }

    def _record_rewriter_model_switch(
        self, chunk_id: str, from_role: str, to_role: str, retry: int, reason: str
    ) -> None:
        """追加一条换模型记录（sidecar），供下一轮 _read_rewriter_switch_state 读取。"""
        data: dict = {}
        try:
            loaded = self._load_intermediate(chunk_id, self._RW_SWITCH_STEP)
            if isinstance(loaded, dict):
                data = loaded
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
            data = {}
        switches = list(data.get("switches", []) or [])
        switches.append({
            "from": from_role,
            "to": to_role,
            "retry": retry,
            "reason": (reason or "")[:200],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save_intermediate(chunk_id, self._RW_SWITCH_STEP, {"switches": switches})

    def _get_best_lit(self, chunk_id: str, fallback_text: str) -> str:
        """返回该 chunk 跨 retry 的最佳文学稿; 若无最佳 (例如从未跑 critic), 返回 fallback_text."""
        entry = self._best_lit.get(chunk_id)
        if entry is None or not entry[1]:
            return fallback_text
        return entry[1]

    def _ensure_candidate_pools(self) -> None:
        """Round 118 P2-8: 惰性初始化候选池 (兼容半初始化实例, 例如测试用 __new__ 构造)."""
        if getattr(self, "_lit_candidates", None) is None:
            self._lit_candidates = {}
        if getattr(self, "_pairwise_info", None) is None:
            self._pairwise_info = {}

    def _pairwise_pick_best(
        self,
        chunk_id: str,
        fallback_text: str,
        source_text: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> str:
        """Round 118 P2-8: 早停/兜底定稿时, 用 Jev 配对仲裁在 top-2 候选间选更优稿.

        只在定稿路径调用; 任何失败/平票/不可用都回退 _get_best_lit（保持原行为）。
        仲裁只做"选稿", 不改变裁决阈值语义; 结果写入 self._pairwise_info 供审计。
        """
        best = self._get_best_lit(chunk_id, fallback_text)
        cfg = _config.pipeline.get("pairwise_arbitration", {}) or {}
        if not cfg.get("enabled", True) or not source_text:
            return best
        self._ensure_candidate_pools()
        cands = self._lit_candidates.get(chunk_id, [])
        if len(cands) < 2:
            return best
        (score_a, text_a), (score_b, text_b) = cands[0], cands[1]
        if not text_a or not text_b or text_a == text_b:
            return best
        try:
            if raw_text is None:
                try:
                    raw_text = self._load_intermediate(chunk_id, "raw")
                except Exception:
                    raw_text = ""
            from src.judges.judge import build_state
            from src.judges.primitives import build_question_set
            from src.judges.typesafe_backend import get_backend
            from src.judges.pairwise import arbitrate_answers

            style = dict(getattr(self, "style_guide", None) or {})
            entity_registry = self._build_entity_registry_for_rewriter(source_text)
            questions = build_question_set()
            backend = get_backend()
            eff_timeout = getattr(backend, "timeout", 30.0)
            state_a = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_a, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            state_b = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_b, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            answers_a = backend.evaluate(state_a, questions, timeout=eff_timeout)
            answers_b = backend.evaluate(state_b, questions, timeout=eff_timeout)
            result = arbitrate_answers(
                answers_a, answers_b,
                min_confidence=float(cfg.get("min_confidence", 0.6)),
            )
            winner = result.get("winner")
            backend_path = getattr(
                backend, "last_backend",
                getattr(backend, "backend_kind", type(backend).__name__),
            )
            info = {
                "winner": winner,
                "backend": backend_path,
                "votes": result.get("votes", {}),
                "candidate_scores": [round(score_a, 2), round(score_b, 2)],
            }
            self._pairwise_info[chunk_id] = info
            print(
                f"⚖️ [Pipeline] {chunk_id} pairwise 仲裁: winner={winner} "
                f"backend={backend_path} votes={result.get('votes')}"
            )
            if winner == "B":
                return text_b
            if winner == "A":
                return text_a
            return best
        except Exception as e:
            print(
                f"⚠️ [Pipeline] {chunk_id} pairwise 仲裁不可用（回退 best score）: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return best

    def _save_intermediate(self, chunk_id: str, step: str, data: str | dict):
        """保存 chunk intermediate 文件. Round 99 P2: 写入时 invalidate 缓存."""
        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            if isinstance(data, str):
                json.dump({"text": data}, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
        # Invalidate 缓存 — 下次加载走文件, 反映最新内容
        if hasattr(self, "_intermediate_cache") and self._intermediate_cache is not None:
            self._intermediate_cache.pop((chunk_id, step), None)

    def _load_intermediate(self, chunk_id: str, step: str):
        """加载 chunk intermediate 文件 (raw / literary / critic_report 等).

        Round 99 P2: 内存读缓存. 同 chunk × step 在 chapter 内多次加载走内存,
        省文件 I/O + JSON parse. 仅缓存 step='file' (即 chunk_id + '_' + step.json
        文件路径), 缓存粒度 = 单 chunk × 单 step.

        写入 (via _save_intermediate) 会 invalidate 对应 cache entry.
        """
        # 缓存 key
        cache_key = (chunk_id, step)
        if not hasattr(self, "_intermediate_cache") or self._intermediate_cache is None:
            self._intermediate_cache = {}
        if cache_key in self._intermediate_cache:
            return self._intermediate_cache[cache_key]

        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "text" in data:
            result = data["text"]
        else:
            result = data
        self._intermediate_cache[cache_key] = result
        return result

    # Round 96: final.json 写入前的 CJK sanity 闸门.
    #
    # 设计动机: ch02 / ch32 实测中, rewriter LLM 偶发"输出英文+中文注释"而不是
    # 中文译文 (典型 prompt misunderstanding 或模型选择错误). 旧代码直接保存
    # LLM 输出, 下游消费者无从察觉. Round 96 在写入前检查 CJK 比例:
    #   1) primary text 通过 (cjk >= threshold) → 直接保存, 仅追加 cjk_ratio 字段
    #   2) primary 未通过 → 依次尝试 literary/raw intermediate, 选 CJK 最高的合规版本
    #   3) 全部未通过 → 仍保存 primary (保留重试痕迹), 但标记 sanity_failed=true
    #                   + sanity_inspected 列出所有检查过的 step 的 CJK 比例
    #
    # 阈值通过 config.yaml pipeline.final_cjk_threshold 配置 (默认 0.5).
    # 关闭方法: 设 final_cjk_threshold=0 或 0.0, 则闸门失效走原 _save_intermediate.
    def _save_final_with_sanity(
        self,
        chunk_id: str,
        data,
        source_step: str = "literary",
        inspect_steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """final.json 写入前 CJK 闸门 (Round 96).

        Args:
            chunk_id: 章节 chunk id.
            data: 写入数据, 可以是 str (被 _save_intermediate 包裹) 或
                  dict 形如 {"text": ..., "metadata": ...}.
            source_step: data 来自哪个 step (literary / raw / best_lit) — 仅日志使用.
            inspect_steps: fallback 候选 step 列表 (默认 ['literary', 'raw']).

        Returns:
            实际写入 final.json 的 dict (含 text + metadata).
        """
        if inspect_steps is None:
            inspect_steps = ["literary", "raw"]

        # Round 96: 阈值通过 config 取; 0 表示禁用闸门 (向后兼容)
        threshold_cfg = _config.pipeline.get("final_cjk_threshold", 0.5)
        try:
            cjk_threshold = float(threshold_cfg)
        except (TypeError, ValueError):
            cjk_threshold = 0.5
        
        # Round 107 P6: 短文本/前言 CJK 阈值放宽.
        # ch01 倒查: 前言/目录 (65 字符, 含英文公司名/版权符号/年份) CJK 比例天然低 (0.308),
        # 0.50 阈值误判. 对短文本 (<100 字符) 或前言特征文本, 放宽到 0.30.
        if isinstance(data, dict):
            _txt = (data or {}).get("text", "") or ""
        else:
            _txt = data if isinstance(data, str) else ""
        _is_short_or_frontmatter = (
            len(_txt) < 100
            or any(kw in _txt for kw in ("目录", "版权", "献给", "版权所有", "题献", "Dedication", "Contents"))
        )
        # final sanity 的 cjk_threshold 是 cjk_ratio 的门槛: cjk_ratio >= threshold → pass.
        # 原 0.50: cjk_ratio < 0.50 → fail. 放宽到 0.30: cjk_ratio < 0.30 → fail.
        if _is_short_or_frontmatter and cjk_threshold > 0.30:
            cjk_threshold = 0.30

        # 关闭闸门 (threshold <= 0) — 走原 _save_intermediate 路径
        if cjk_threshold <= 0:
            self._save_intermediate(chunk_id, "final", data)
            if isinstance(data, str):
                return {"text": data, "metadata": {}}
            return dict(data)

        # 提取 primary text + metadata
        if isinstance(data, str):
            primary_text = data
            primary_metadata: Dict[str, Any] = {}
        else:
            primary_text = (data or {}).get("text", "") or ""
            primary_metadata = dict((data or {}).get("metadata", {}) or {})

        primary_cjk = compute_cjk_ratio(primary_text)

        # primary 已合规 → 直接保存, 仅追加 cjk_ratio 元数据
        if primary_cjk >= cjk_threshold:
            merged = {
                "text": primary_text,
                "metadata": {**primary_metadata, "cjk_ratio": round(primary_cjk, 3)},
            }
            self._save_intermediate(chunk_id, "final", merged)
            return merged

        # primary 未通过 → 尝试 fallback 候选 (按 cjk 降序, 取合规 + 比 primary 高)
        candidates: List[Tuple[str, str, float]] = []
        for step in inspect_steps:
            if step == source_step:
                continue  # 跳过与 primary 同源的 step (避免重复检查)
            try:
                loaded = self._load_intermediate(chunk_id, step)
                cand_text = loaded if isinstance(loaded, str) else (loaded or {}).get("text", "") or ""
                if cand_text and cand_text.strip():
                    cjk = compute_cjk_ratio(cand_text)
                    candidates.append((step, cand_text, cjk))
            except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                continue

        candidates.sort(key=lambda x: -x[2])

        chosen_step = None
        chosen_text: Optional[str] = None
        chosen_cjk = 0.0
        for step, cand_text, cjk in candidates:
            if cjk >= cjk_threshold and cjk > primary_cjk:
                chosen_step, chosen_text, chosen_cjk = step, cand_text, cjk
                break

        if chosen_step is not None and chosen_text is not None:
            merged = {
                "text": chosen_text,
                "metadata": {
                    **primary_metadata,
                    "cjk_ratio": round(chosen_cjk, 3),
                    "sanity_fallback": True,
                    "sanity_source_step": chosen_step,
                    "sanity_original_step": source_step,
                    "sanity_original_cjk": round(primary_cjk, 3),
                    "sanity_threshold": cjk_threshold,
                },
            }
            self._save_intermediate(chunk_id, "final", merged)
            print(
                f"⚠️ [Pipeline] {chunk_id} CJK sanity fallback: "
                f"{source_step} cjk={primary_cjk:.2f} → {chosen_step} cjk={chosen_cjk:.2f}"
            )
            return merged

        # 全部未通过 → 仍保存 primary (保留重试痕迹), 标记 sanity_failed
        sanity_inspected = [{"step": source_step, "cjk": round(primary_cjk, 3)}]
        sanity_inspected.extend(
            {"step": s, "cjk": round(c, 3)} for s, _, c in candidates
        )
        merged = {
            "text": primary_text,
            "metadata": {
                **primary_metadata,
                "cjk_ratio": round(primary_cjk, 3),
                "sanity_failed": True,
                "sanity_threshold": cjk_threshold,
                "sanity_inspected": sanity_inspected,
            },
        }
        self._save_intermediate(chunk_id, "final", merged)
        print(
            f"❌ [Pipeline] {chunk_id} CJK sanity 失败: primary={primary_cjk:.2f}, "
            f"all fallback < {cjk_threshold:.2f} ({sanity_inspected})"
        )
        return merged

    # Round 119 L1: 韵文通道 helpers（tags/manifest 惰性加载 + 文本包含兜底）。
    # 说明：verse 相关导入全部放在函数内，导入失败时静默降级为空（非韵文场景无影响）。
    def _verse_tags(self) -> dict:
        if getattr(self, '_verse_tags_cache', None) is None:
            try:
                from src.verse import load_chunk_tags
                self._verse_tags_cache = load_chunk_tags(self._root_dir / "db")
            except Exception:
                self._verse_tags_cache = {}
        return self._verse_tags_cache

    def _verse_manifest(self) -> dict:
        if getattr(self, '_verse_manifest_cache', None) is None:
            try:
                from src.verse import load_verse_manifest
                self._verse_manifest_cache = load_verse_manifest(self._root_dir / "input")
            except Exception:
                self._verse_manifest_cache = {}
        return self._verse_manifest_cache

    def _verse_fragments_for(self, chunk_id: str, source_text: str) -> list:
        """chunk 的韵文片段列表（无韵文/未启用时返回 []）。"""
        if not chunk_id:
            return []
        try:
            from src.verse import fragments_for_chunk, verse_config
            if not verse_config().get("enabled", True):
                return []
            return fragments_for_chunk(self.chapter_id, chunk_id, source_text,
                                       self._verse_tags(), self._verse_manifest())
        except Exception:
            return []

    def _verse_note_for(self, chunk_id: str, source_text: str) -> str:
        """韵文协议提示块（按 chunk 缓存；无韵文返回 ""）。"""
        if not chunk_id:
            return ""
        cache = getattr(self, '_verse_note_cache', None)
        if cache is None:
            cache = {}
            self._verse_note_cache = cache
        if chunk_id not in cache:
            try:
                from src.verse import build_verse_note
                cache[chunk_id] = build_verse_note(
                    self._verse_fragments_for(chunk_id, source_text))
            except Exception:
                cache[chunk_id] = ""
        return cache[chunk_id]

    def _verse_lint_gate(self, chunk_id: str, source_text: str, final_text) -> Optional[dict]:
        """PASS 定稿前的韵文对齐检查。None=无韵文/未启用；否则 verse_lint 结果 + 元数据。"""
        try:
            from src.verse import verse_lint, verse_config
        except Exception:
            return None
        cfg = verse_config()
        if not cfg.get("enabled", True):
            return None
        frags = self._verse_fragments_for(chunk_id, source_text)
        if not frags:
            return None
        res = verse_lint(frags, final_text,
                         min_line_ratio=float(cfg.get("min_line_ratio", 0.6)),
                         max_line_chars=int(cfg.get("max_line_chars", 56)),
                         refrain_ratio=float(cfg.get("refrain_ratio", 0.5)))
        res["retried"] = self._verse_retry_used(chunk_id)
        res["retry_on_fail"] = bool(cfg.get("retry_on_fail", True))
        res["chunk_id"] = chunk_id
        return res

    def _mark_verse_retry(self, chunk_id: str, verse_gate: dict) -> None:
        """记录「verse 定向重写已发生」（一次性限额；init --force 时重置）。"""
        try:
            self._save_intermediate(chunk_id, "verse_retry", {
                "reason": verse_gate.get("reason", ""),
                "expected_lines": verse_gate.get("expected_lines"),
                "found_lines": verse_gate.get("found_lines"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        except Exception as exc:
            print(f"⚠️ [Pipeline] {chunk_id} verse_retry 标记写入失败: {exc}", file=sys.stderr)

    def _verse_retry_used(self, chunk_id: str) -> bool:
        return (self.output_dir / f"{chunk_id}_verse_retry.json").exists()

    def _seed_pun_decisions(self, source_text: str) -> int:
        """L2: pun_watchlist 命中 chunk 时强制落 REFERENCE 决策（幂等；无 impact 不回溯）。

        source_key 直接用 anchor 原词，使 format_prompt_context 的命中过滤能在
        含该词的后续 chunk prompt 中带出策略；affected_chunks=None 保证不触发回溯。
        """
        guide = getattr(self, "style_guide", None)
        if not isinstance(guide, dict) or not source_text:
            return 0
        watch = guide.get("pun_watchlist") or []
        if not watch:
            return 0
        engine = getattr(self, "decision_engine", None)
        if engine is None or not hasattr(engine, "add_decision"):
            return 0
        seeded = getattr(self, "_pun_seeded_keys", None)
        if seeded is None:
            seeded = set()
            self._pun_seeded_keys = seeded
        src_lower = source_text.lower()
        n = 0
        for item in watch:
            if not isinstance(item, dict):
                continue
            candidates = [str(a).strip() for a in
                          (list(item.get("src_anchors") or []) + list(item.get("anchors") or []))
                          if str(a).strip()]
            hits = [a for a in candidates if len(a) >= 2 and a.lower() in src_lower]
            if not hits:
                continue
            strategy = str(item.get("strategy") or "").strip() or "双关需在译文保留笑点，必要时章末加注"
            tag = str(item.get("source") or "").strip()
            for anchor in hits[:2]:
                dedupe = "pun:" + anchor.lower()
                if dedupe in seeded:
                    continue
                try:
                    from src.core.decision_engine import DecisionLevel as _DL
                    engine.add_decision(
                        _DL.REFERENCE,
                        anchor,
                        f"[双关·保留笑点] {strategy}",
                        f"pun_watchlist: {tag}",
                        affected_chunks=None,
                    )
                    n += 1
                except Exception as exc:
                    print(f"⚠️ [Pipeline] pun 决策写入失败 {anchor}: {exc}", file=sys.stderr)
                seeded.add(dedupe)
        return n

    def _dual_draft_cfg(self) -> dict:
        """L3: 双稿开关（默认关闭；小场景 A/B 后定夺）。"""
        cfg = _config.pipeline.get("dual_draft", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        try:
            max_chunks = int(cfg.get("max_chunks", 25) or 25)
        except (TypeError, ValueError):
            max_chunks = 25
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "alternate_role": cfg.get("alternate_role") or "primary",
            "alternate_effort": cfg.get("alternate_effort"),
            "base_role": cfg.get("base_role") or "",
            "max_chunks": max_chunks,
        }

    def _dual_draft_kind(self, chunk_id: str, source_text: str) -> str:
        """'' | 'verse' | 'pun' —— 高价值片段路由判定（L3）。"""
        try:
            if self._verse_fragments_for(chunk_id, source_text):
                return "verse"
        except Exception:
            pass
        guide = getattr(self, "style_guide", None)
        if isinstance(guide, dict) and source_text:
            src_lower = source_text.lower()
            for item in (guide.get("pun_watchlist") or []):
                if not isinstance(item, dict):
                    continue
                anchors = list(item.get("src_anchors") or []) + list(item.get("anchors") or [])
                for anchor in anchors:
                    a = str(anchor).strip()
                    if len(a) >= 2 and a.lower() in src_lower:
                        return "pun"
        return ""

    def _dual_critic_avg(self, chunk_id: str, source_text: str, raw_text: str, text: str):
        """L3: critic 单稿均分（失败返回 None）。"""
        if not text:
            return None
        try:
            report = self.critic_agent.process_chunk(
                chunk_id, source_text, raw_text or "", text, self.style_guide,
                entity_registry=self._build_entity_registry_for_rewriter(source_text),
                entity_variants=self._build_entity_variants_for_critic(source_text),
            )
            avg = self._compute_avg_score(report)
            return float(avg) if avg else None
        except Exception:
            return None

    def _dual_pairwise_winner(self, chunk_id: str, source_text: str, raw_text: str,
                              text_a: str, text_b: str):
        """L3: 两份候选稿 Jev 配对仲裁 → (winner 'a'|'b'|None, info dict)。"""
        cfg = _config.pipeline.get("pairwise_arbitration", {}) or {}
        info = {"winner": None, "backend": None, "votes": {}}
        try:
            from src.judges.judge import build_state
            from src.judges.primitives import build_question_set
            from src.judges.typesafe_backend import get_backend
            from src.judges.pairwise import arbitrate_answers

            style = dict(getattr(self, "style_guide", None) or {})
            entity_registry = self._build_entity_registry_for_rewriter(source_text)
            questions = build_question_set()
            backend = get_backend()
            eff_timeout = getattr(backend, "timeout", 30.0)
            state_a = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_a, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            state_b = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_b, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            answers_a = backend.evaluate(state_a, questions, timeout=eff_timeout)
            answers_b = backend.evaluate(state_b, questions, timeout=eff_timeout)
            result = arbitrate_answers(
                answers_a, answers_b,
                min_confidence=float(cfg.get("min_confidence", 0.6)),
            )
            winner_raw = result.get("winner")
            backend_path = getattr(
                backend, "last_backend",
                getattr(backend, "backend_kind", type(backend).__name__),
            )
            info = {"winner": winner_raw, "backend": backend_path,
                    "votes": result.get("votes", {})}
            if winner_raw == "B":
                return "b", info
            if winner_raw == "A":
                return "a", info
            return None, info
        except Exception as e:
            print(f"⚠️ [Pipeline] {chunk_id} 双稿 pairwise 不可用: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return None, info

    def _dual_draft_run(self, chunk_id: str, source_text: str, raw_text: str,
                        prev_lit_text, reject_reason, critic_feedback, entity_feedback,
                        all_feedback, specific_edits, narrative_memory, low_dims,
                        skipped_dimensions, draft_a: str) -> str:
        """L3: 高价值片段双稿仲裁 — 不满足条件时原样返回 draft_a。"""
        cfg = self._dual_draft_cfg()
        if not cfg.get("enabled") or not draft_a or not source_text:
            return draft_a
        count = int(getattr(self, "_dual_draft_count", 0) or 0)
        if count >= int(cfg.get("max_chunks") or 25):
            return draft_a
        try:
            existing = self._load_intermediate(chunk_id, "dual_draft_meta")
            if isinstance(existing, dict) and existing.get("chosen"):
                return draft_a
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
            pass
        kind = self._dual_draft_kind(chunk_id, source_text)
        if not kind:
            return draft_a
        alt_role = cfg.get("alternate_role") or "primary"
        base_role = cfg.get("base_role") or _config.get_task_role("literary_rewrite")
        try:
            draft_b_raw = self.rewriter_agent.process_chunk(
                chunk_id, raw_text, self.style_guide, source_text,
                prev_lit_text=prev_lit_text,
                reject_reason=reject_reason,
                critic_feedback=critic_feedback,
                entity_feedback=entity_feedback,
                entity_registry=self._build_entity_registry_for_rewriter(source_text),
                retry_count=0,
                all_feedback=all_feedback,
                specific_edits=specific_edits,
                narrative_memory=narrative_memory,
                low_dims=low_dims,
                skipped_dimensions=skipped_dimensions,
                verse_note=self._verse_note_for(chunk_id, source_text),
                model_role=alt_role,
                model_effort=cfg.get("alternate_effort"),
            )
        except TypeError:
            return draft_a
        except Exception as e:
            print(f"⚠️ [Pipeline] {chunk_id} 双稿第二稿生成失败, 保留首稿: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return draft_a
        draft_b = draft_b_raw[0] if isinstance(draft_b_raw, tuple) else draft_b_raw
        if not draft_b or draft_b == draft_a:
            return draft_a
        score_a = self._dual_critic_avg(chunk_id, source_text, raw_text, draft_a)
        score_b = self._dual_critic_avg(chunk_id, source_text, raw_text, draft_b)
        winner, pair_info = self._dual_pairwise_winner(
            chunk_id, source_text, raw_text, draft_a, draft_b)
        if winner == "b":
            chosen, chosen_text, reason = "b", draft_b, "pairwise"
        elif winner == "a":
            chosen, chosen_text, reason = "a", draft_a, "pairwise"
        elif score_b is not None and (score_a is None or score_b > score_a):
            chosen, chosen_text, reason = "b", draft_b, "critic"
        else:
            chosen, chosen_text = "a", draft_a
            reason = "critic" if score_a is not None else "fallback"
        self._save_intermediate(chunk_id, "dual_draft_meta", {
            "chosen": chosen,
            "reason": reason,
            "kind": kind,
            "roles": {"a": base_role, "b": alt_role},
            "critic_avg": {"a": score_a, "b": score_b},
            "pairwise": pair_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._dual_draft_count = count + 1
        print(f"⚖️ [Pipeline] {chunk_id} 双稿仲裁({kind}): 选 {chosen}"
              f"（{reason}; critic a={score_a} b={score_b}; pairwise={pair_info.get('winner')}）")
        return chosen_text

    def _build_literal_prompt(self, source_text: str, chunk_id: str = "") -> str:
        """组装直译 prompt（含有界术语表 + 本书用词提示 + 韵文协议）；供生产路径与单测共用。"""
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        glossary = self.decision_engine.format_prompt_context(source_text=source_text)
        verse_note = self._verse_note_for(chunk_id, source_text) if chunk_id else ""
        return (
            f"请将以下{genre}片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n"
            f"{glossary}\n\n"
            f"{self._build_translation_hints_block()}"
            f"{verse_note}"
            f"【原文】\n{source_text}"
        )

    @staticmethod
    def _build_translation_hints_block() -> str:
        """本书用词提示块（P1-4：作品特定提示只允许来自风格剖面 translation_hints）。

        旧版曾在此硬编码「1920年代英式俚语 wild-cat / 丁尼生名句」等作品特定先验，
        会污染其他作品的直译；现在无 profile 时只给作品无关的通用指令。
        """
        hints = (_config.style_guide or {}).get("translation_hints", []) or []
        if hints:
            lines = "\n".join(f"- {h}" for h in hints[:30])
            return (
                "⚠️ 本书已确认的年代/地域用词与引用（先识别含义再翻译，不要按字面意思直译）：\n"
                f"{lines}\n\n"
            )
        return (
            "⚠️ 注意识别时代/地域俚语、行业行话与文学引用：先识别含义再翻译，不要按字面意思直译；"
            "无法确认的引用保留可考据的译法，并在术语表中登记。\n\n"
        )

    def _run_raw_translator(self, source_text: str, chunk_id: str = "") -> str:
        prompt = self._build_literal_prompt(source_text, chunk_id)
        # 使用配置路由：literal_translation
        model_key, params = _config.resolve_task_model("literal_translation")
        role = self.literal_role
        llm = self.llm
        params = dict(params)
        # Round 118 P2-5 / Round 128: 难度路由 — 高难度片段切到更强模型（默认 primary/MiniMax）
        # 并按段加档 effort（hard_effort）；普通片段仍走任务默认（不下发档位）。
        dcfg = _config.pipeline.get("difficulty_routing", {}) or {}
        if dcfg.get("enabled", True):
            difficulty = self._chunk_difficulty(source_text)
            if difficulty >= float(dcfg.get("hard_threshold", 2.0)):
                role = dcfg.get("hard_role", "primary")
                llm = self._get_role_llm(role)
                hard_effort = str(dcfg.get("hard_effort") or "").strip().lower()
                if hard_effort:
                    params["effort"] = hard_effort
                print(
                    f"🧩 [Pipeline] 难度路由: difficulty={difficulty:.2f} ≥ "
                    f"{float(dcfg.get('hard_threshold', 2.0)):.1f} → role={role} "
                    f"(effort={hard_effort or 'default'})"
                )
        # Round 133: no_content 重试适配 override（降档/换角救稿）优先于难度路由结果
        if chunk_id:
            ov = self._get_retry_override(chunk_id)
            if ov and ov.get("sig") == "no_content" and ov.get("stage") == "TRANSLATING_RAW":
                ov_role = str(ov.get("role") or "")
                if ov_role and ov_role != role:
                    role = ov_role
                    llm = self._get_role_llm(role)
                ov_eff = ov.get("effort")
                if ov_eff:
                    params["effort"] = str(ov_eff).strip().lower()
                print(
                    f"🔻 [Pipeline] {chunk_id} 重试适配生效: role={role} "
                    f"effort={params.get('effort') or 'default'}"
                )
        model_name = get_role_model_name(role) or model_key
        self._note_last_call(chunk_id, "TRANSLATING_RAW", role, params.get("effort"), model_name)
        return llm.generate(prompt, model_name=model_name, **params)

    def _get_role_llm(self, role: str):
        """按角色缓存 LLM 客户端（难度路由切模型时复用，避免重复构造）。"""
        cache = getattr(self, "_role_llm_cache", None)
        if cache is None:
            cache = {}
            self._role_llm_cache = cache
        if role not in cache:
            from src.utils.llm_adapter import get_llm_client
            cache[role] = get_llm_client(role)
        return cache[role]

    def _count_reference_hits(self, source_text: str) -> int:
        """统计本片段命中的已登记典故数（DecisionLevel.REFERENCE = 2）。"""
        if not source_text:
            return 0
        src_lower = source_text.lower()
        try:
            cur = self.decision_engine.conn.cursor()
            cur.execute("SELECT source_key FROM decision_db WHERE level = 2")
            hits = 0
            for (source_key,) in cur.fetchall():
                if not source_key:
                    continue
                if source_key in source_text:
                    hits += 1
                elif len(source_key) >= 3 and source_key.lower() in src_lower:
                    hits += 1
            return hits
        except Exception:
            return 0

    def _chunk_difficulty(self, source_text: str) -> float:
        """Round 118 P2-5: 片段难度启发式打分（用于直译阶段模型路由）。

        信号：
          - 已登记典故密度（考据越密越难；每命中 1 条 +1，封顶 3）
          - 长句占比（≥35 词句子占比 ≥0.3 → +1.5，≥0.15 → +0.75）
          - 引文特征（含成对引号且引文长度 ≥6 词 → +1）
        """
        if not source_text:
            return 0.0
        score = float(min(3, self._count_reference_hits(source_text)))
        sentences = [s for s in re.split(r"[.!?]+\s*", source_text) if s.strip()]
        if sentences:
            long_ratio = sum(
                1 for s in sentences if len(s.split()) >= 35
            ) / len(sentences)
            if long_ratio >= 0.3:
                score += 1.5
            elif long_ratio >= 0.15:
                score += 0.75
        for m in re.finditer(r"[\"“]([^\"”]{20,})[\"”]", source_text):
            if len(m.group(1).split()) >= 6:
                score += 1.0
                break
        return score

    @staticmethod
    def _compute_text_similarity(a: str, b: str) -> float:
        """Round 6: 文本相似度（字符级 SequenceMatcher 比值）

        用于 oscillation 检测：连续 retry 译文相似度 ≥ 阈值 → LLM 原地打转 → 早停。

        返回 0.0~1.0；空字符串特殊处理（视为 0.0 避免除零）。
        """
        if not a or not b:
            return 0.0
        if len(a) > 4000 or len(b) > 4000:
            # 长文本截断对比（避免 difflib 平方复杂度爆栈）
            a, b = a[:4000], b[:4000]
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()


    # Round 34: thin shim — delegates to src.utils.text_match.extract_reject_categories
    # （行为完全等价 Round 16 实现；保留类方法签名避免破坏 pipeline 内部调用点）
    @staticmethod
    def _extract_reject_categories(reason_text: str) -> set:
        return extract_reject_categories(reason_text, as_set=True)
    # Round 34: thin shim — delegates to src.utils.text_match.case_insensitive_find
    @staticmethod
    def _case_insensitive_find(name: str, haystack: "str | None") -> "str | None":
        return case_insensitive_find(name, haystack)
    def _memory_path(self) -> Path:
        return self.output_dir / "_memory.jsonl"

    def _append_chunk_memory(self, chunk_id: str, lit_text: str, source_text: str = "") -> None:
        """定稿后写入轻量章内记忆，供后续 chunk 润色注入。

        Round 100 P1: 写入后 invalidate _narrative_memory_cache (新 chunk 加入让
        后续 chunk 的 prior_memory 视图变化).
        """
        if hasattr(self, '_narrative_memory_cache') and self._narrative_memory_cache is not None:
            # 全清: 新写入影响所有"当前 chunk 之后"的视图, 但最简方案是全清
            self._narrative_memory_cache.clear()
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
        """读取本章已完成 chunk 的摘要（不含当前 chunk）。

        Round 100 P1: 内存读缓存. 同 chunk 跨 retry 内容不变, 缓存命中跳过文件 I/O
        和 JSON parse. _append_chunk_memory 写入时会 invalidate 对应 cache entry.
        """
        if not hasattr(self, '_narrative_memory_cache') or self._narrative_memory_cache is None:
            self._narrative_memory_cache = {}
        cache_key = current_chunk_id
        if cache_key in self._narrative_memory_cache:
            return self._narrative_memory_cache[cache_key]

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
        result = "\n".join(
            f"- [{r.get('chunk_id', '?')}] {r.get('summary', '')}" for r in recent
        )
        self._narrative_memory_cache[cache_key] = result
        return result

    def run(self) -> bool:
        print(f"🚀 [Pipeline] 启动批处理模式处理章节: {self.chapter_id}")

        # Round 99 P1: chapter 切换时清空 per-chunk 缓存, 防止跨 chapter 脏数据
        self._reset_per_run_cache()

        # 检查章节是否有任务，避免在空 DB 上误报"完成"
        existing_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
        if not existing_tasks:
            # 区分两种空：真未 init vs init 被跳过但 DB 无行/无产出
            print(f"❌ [Pipeline] 章节 {self.chapter_id} 无可执行任务（可能 init 被跳过或 DB 无记录），"
                  f"请用 --force 重新初始化。")
            return False

        # Round 104: 恢复"COMPLETED 但 final.json 缺失"的 chunk (回溯删 final 后重跑失败的残留)
        try:
            self._recover_completed_without_final()
        except Exception as e:
            print(f"⚠️ [Pipeline] COMPLETED 恢复检查异常（不影响主流程）: {e}")

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

        # Round 130: deferred 回溯的章级合并预算计数（本次运行内累计）
        self._backtrack_sweeps_done = 0

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
                # Round 130: deferred 回溯在队列清空点（章末）合并应用一次；
                # 治理预算（章级轮数 + 单块轮数）保证不收敛循环无法再出现。
                # immediate 模式下此调用为 no-op（决策变更已即时标 DIRTY）。
                try:
                    if self._consolidate_deferred_backtracks():
                        any_progress = True
                except Exception as e:
                    print(f"⚠️ [Pipeline] 章末合并回溯异常（不影响主流程）: {e}")

            if not any_progress:
                # 仅表示队列已清空，不代表产出完整；最终是否成功由 _verify_final_outputs 决定
                print(f"⏹ [Pipeline] 章节 {self.chapter_id} 任务队列已清空，进入产出校验…")
                break

            time.sleep(poll_interval)

        # Round 7: 译后一致性校正（章节级 scope，auto-apply）
        # 修复 alias_mixed / english_residue / cjk_variant 三类命名漂移
        # QA_REPORT 显示 162 处不一致散落在历史章节里，启用后可批量修复
        post_consistency_cfg = _config.pipeline.get("post_consistency", {}) or {}
        if post_consistency_cfg.get("enabled", True):
            try:
                self._run_post_consistency(
                    chapter_scoped=bool(post_consistency_cfg.get("chapter_scoped", True))
                )
            except Exception as e:
                print(f"⚠️ [Pipeline] 译后一致性校正异常（不影响主流程）: {e}")

        # Round 15: entity_registry 冲突检测（生成 entity_registry_conflicts.md）
        reg_audit_cfg = _config.pipeline.get("entity_registry_audit", {}) or {}
        if reg_audit_cfg.get("enabled", True):
            try:
                conflicts = self._entity_registry_conflict_audit()
                if conflicts:
                    cross = sum(1 for c in conflicts if c["conflict_type"] == "cross_entity")
                    variant = sum(1 for c in conflicts if c["conflict_type"] == "variant_for_same_person")
                    lines = []
                    lines.append(f"# entity_registry 冲突检测 - {self.chapter_id}\n\n")
                    lines.append(f"- 总冲突数: {len(conflicts)} (跨人物: {cross}, 同人变体: {variant})\n\n")
                    lines.append("## 跨人物冲突（明确错误，需人工修复）\n\n")
                    for c in [x for x in conflicts if x["conflict_type"] == "cross_entity"]:
                        lines.append("- **\"" + c["source_name"] + "\"** ← " + str(c["canonicals"]) + "\n")
                    lines.append("\n## 同人变体（可接受但建议整合）\n\n")
                    for c in [x for x in conflicts if x["conflict_type"] == "variant_for_same_person"]:
                        lines.append("- **\"" + c["source_name"] + "\"** ← " + str(c["canonicals"]) + "\n")
                    lines.append("\n注：本报告为诊断用，**不会自动修改 entity_registry**。\n")
                    lines.append("修复方式：人工编辑 decision_db.sqlite 的 entity_registry 表 source_names 列。\n")
                    conflict_path = self.output_dir / "entity_registry_conflicts.md"
                    conflict_path.write_text("".join(lines), encoding="utf-8")
                    print(f"⚠️ [Entity Registry] 发现 {cross} 个跨人物冲突 + {variant} 个同人变体（详见 entity_registry_conflicts.md）")
                else:
                    print(f"✅ [Entity Registry] 无冲突")
            except Exception as e:
                print(f"⚠️ [Pipeline] entity_registry 冲突检测异常（不影响主流程）: {e}")

        # Round 9: 实体完整性审计（纯诊断，生成 entity_audit.md）
        entity_audit_cfg = _config.pipeline.get("entity_audit", {}) or {}
        if entity_audit_cfg.get("enabled", True):
            try:
                self._post_pipeline_entity_audit()
            except Exception as e:
                print(f"⚠️ [Pipeline] 实体审计异常（不影响主流程）: {e}")

        # Round 13: LLM 调用成本报告
        cost_cfg = _config.pipeline.get("cost_report", {}) or {}
        if cost_cfg.get("enabled", True):
            try:
                self._print_llm_cost_report()
            except Exception as e:
                print(f"⚠️ [Pipeline] 成本报告异常（不影响主流程）: {e}")

        # Round 14: intermediate 文件清理
        try:
            self._cleanup_intermediates()
        except Exception as e:
            print(f"⚠️ [Pipeline] 中间文件清理异常（不影响主流程）: {e}")

        # Round 107 P7: 恢复回溯重跑失败的 final_backup.json (在验证前)
        try:
            self._restore_dirty_backups()
        except Exception as e:
            print(f"⚠️ [Pipeline] 恢复 dirty backup 异常（不影响主流程）: {e}")

        # 完成后交叉验证：final.json 落盘数应与任务数一致，否则视为假阳性完成
        return self._verify_final_outputs(len(existing_tasks))

    def _run_post_consistency(self, chapter_scoped: bool = True) -> None:
        """Round 7: 章节完成后自动跑 consistency（章节级 scope）

        Round 105: 受 pipeline.consistency_auto_fix 开关控制。
          - false（默认）: 仅生成 consistency_diff.md 报告，不落地修改 final.json
            （ch32 实测自动修正会把普通词当人名异体替换，改坏正文）。
          - true: 直接 apply 修正。
          CLI 子命令 `consistency --apply` 不受此开关影响（显式人工操作）。

        Args:
            chapter_scoped: True=仅修正当前章节的 final.json（生产环境推荐）
                           False=扫描整个 output/（CI/全量审计用，会修改所有章节）
        """
        from src.consistency import run_consistency_check_for_chapter, run_consistency_check
        auto_fix = bool(_config.pipeline.get('consistency_auto_fix', False))
        dry_run = not auto_fix
        if chapter_scoped:
            n = run_consistency_check_for_chapter(self.chapter_id, dry_run=dry_run)
        else:
            # 全局模式：复用 CLI 子命令行为（覆盖整个 output/）
            n = run_consistency_check(dry_run=dry_run)
        if n and n > 0:
            if dry_run:
                print(f"  ⏸️ [Consistency] 发现 {n} 处命名不一致（consistency_auto_fix=false，仅报告未修改；详见 consistency_diff.md）")
            else:
                print(f"  ✅ [Consistency] 本章修正 {n} 处命名不一致")

    def _print_llm_cost_report(self) -> None:
        """Round 13: 输出 LLM 调用成本报告（按 task_type 分组）"""
        counter = self._llm_call_counter
        total = sum(counter.values())

        display_names = {
            "reference_extraction": "典故/实体提取",
            "literal_translation": "字面直译",
            "literary_rewriting": "文学润色",
            "critic_scoring": "Critic 评分",
            "judge_decision": "最终裁决",
        }
        est_tokens_per_call = {
            "reference_extraction": 4000,
            "literal_translation": 2500,
            "literary_rewriting": 5000,
            "critic_scoring": 4500,
            "judge_decision": 3500,
        }

        lines = []
        lines.append("# LLM 成本报告 - " + self.chapter_id + "\n\n")
        lines.append("- 生成时间: " + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        lines.append("- 总 LLM 调用次数: **" + str(total) + "**\n\n")
        lines.append("| Task 类型 | 调用次数 | 估算 tokens | 占比 |\n")
        lines.append("|---|---:|---:|---:|\n")
        sorted_types = sorted(counter.items(), key=lambda kv: -kv[1])
        for task_type, count in sorted_types:
            if count == 0:
                continue
            pct = (count / total * 100) if total > 0 else 0
            est_tokens = count * est_tokens_per_call.get(task_type, 3000)
            lines.append("| " + display_names.get(task_type, task_type) + " | " + str(count) + " | ~" + format(est_tokens, ",") + " | " + format(pct, ".1f") + "% |\n")
        total_tokens = sum(c * est_tokens_per_call.get(t, 3000) for t, c in counter.items())
        lines.append("| **合计** | **" + str(total) + "** | **~" + format(total_tokens, ",") + "** | 100% |\n\n")
        lines.append("- Round 6 oscillation + Round 12 stagnation + Round 36 字节停滞 + Round 73 pre-rewriter 短路 早停链 已显著减少 literary_rewriting / critic_scoring / judge_decision 调用次数\n")

        # Round 27: 真实 token 累计（按 role；OpenAI-compatible API 直接解析 usage；Mock 估算 len//4）。
        #              注意：同 role 多 task 共享 adapter 时的累计会合并显示
        try:
            from src.utils.llm_adapter import get_llm_client
            seen_roles = set()
            role_rows = []
            for task_type in counter.keys():
                role = _config.get_task_role(task_type)
                if role in seen_roles:
                    continue
                seen_roles.add(role)
                client = get_llm_client(role)
                u = client.get_cumulative_usage()
                role_rows.append((role, u))
            if role_rows:
                total_real = sum(u["total_tokens"] for _, u in role_rows)
                lines.append("\n## 真实 token 累计（Round 27，按 role 汇总）\n\n")
                lines.append("| Role (后端) | 调用次数 | Prompt tokens | Completion tokens | 合计 |\n")
                lines.append("|---|---:|---:|---:|---:|\n")
                for role, u in sorted(role_rows, key=lambda kv: -kv[1]["total_tokens"]):
                    if u["calls"] == 0:
                        continue
                    lines.append(
                        "| " + role + " | " + str(u["calls"]) + " | "
                        + format(u["prompt_tokens"], ",") + " | "
                        + format(u["completion_tokens"], ",") + " | "
                        + format(u["total_tokens"], ",") + " |\n"
                    )
                lines.append("| **合计** | | | | **" + format(total_real, ",") + "** |\n\n")
                # 对账：估算 vs 真实（Round 27 + Round 31 加偏差告警）
                if total_tokens > 0 and total_real > 0:
                    ratio = total_real / total_tokens
                    delta_pct = abs(ratio - 1.0) * 100
                    lines.append(
                        "- 对账：估算 " + format(total_tokens, ",")
                        + " → 真实 " + format(total_real, ",")
                        + f"（比值 {ratio:.2f}×，偏差 {delta_pct:.1f}%）\n"
                    )
                    # Round 31: 偏差 > 30% 提示用户更新 mental model
                    if delta_pct > 30:
                        lines.append(
                            "- ⚠️ **估算偏差 > 30%**：实际 token 与 Round 13 估算差距显著，"
                            "建议更新 est_tokens_per_call 表以提升后续章节预测精度。\n"
                        )

            # Round 35: 自动建议 est_tokens_per_call 更新值（基于本章真实 per-call 平均）
            try:
                                # Round 84: 补全 Round 81 fix —— 之前 round 81 只修了 lines.append("\n") 和内层 f-string, 
                # 外层 markdown 结构 (section header / 段落 / 表格 header) 仍含字面 \n,
                # 写入 markdown 后渲染为字面 \n 而非真换行. 现在统一改为 \n.
                lines.append("\n### est_tokens_per_call 建议值（Round 35）\n\n")
                lines.append("基于本章真实 token / 调用次数计算的每调用平均：\n\n")
                lines.append("| Role | 当前每调用真实平均 | 当前估算 | 建议替换值 |\n")
                lines.append("|---|---:|---:|---:|\n")
                seen_roles_suggest = set()
                for task_type in counter.keys():
                    role = _config.get_task_role(task_type)
                    if role in seen_roles_suggest:
                        continue
                    seen_roles_suggest.add(role)
                    client = get_llm_client(role)
                    u = client.get_cumulative_usage()
                    if u["calls"] == 0:
                        continue
                    avg = u["total_tokens"] / u["calls"]
                    current_est = est_tokens_per_call.get(task_type, 3000)
                    delta_pct_est = abs(avg - current_est) / current_est * 100
                    suggested = round(avg / 500) * 500
                    marker = "**" if delta_pct_est > 30 else ""
                    lines.append(
                        f"| {role} | {avg:,.0f} | {current_est:,} | "
                        f"{marker}{suggested:,}{marker}（偏差 {delta_pct_est:.0f}%）|" + "\n"
                    )
                # Round 81: 真换行符 (修复前是字面反斜杠 + n, 不会被 markdown 渲染)
                lines.append("\n")  # Round 81: 真换行符 (修复前字面 \n, 不会被 markdown 渲染)
            except Exception as _sug_err:
                print(f"  ⚠️ [_print_llm_cost_report] est_tokens_per_call 建议生成失败: {_sug_err}", file=sys.stderr)

        except Exception as _real_err:

            print(f"  ⚠️ [_print_llm_cost_report] 真实 token 统计失败: {_real_err}", file=sys.stderr)

        # Round 18: 统计 Jev 后端实际利用率（扫描 critic_report.json 的 _backend_path）
        backend_counter = {"jev": 0, "openai_compat": 0, "mock_heuristic": 0, "legacy": 0, "unknown": 0}
        try:
            for cr_path in self.output_dir.glob("*_critic_report.json"):
                try:
                    with open(cr_path, "r", encoding="utf-8") as f:
                        cr = json.load(f)
                    bp = cr.get("_backend_path", "unknown")
                    backend_counter[bp] = backend_counter.get(bp, 0) + 1
                except (OSError, json.JSONDecodeError):
                    pass
            total_critic = sum(backend_counter.values())
            if total_critic > 0:
                lines.append("\n## Jev 后端利用率（Round 18）\n\n")
                lines.append("| Backend | 调用次数 | 占比 | 说明 |\n")
                lines.append("|---|---:|---:|---|\n")
                bp_desc = {
                    "jev": "✅ Jev 原生（typesafe_sdk 直连）",
                    "openai_compat": "⚠️ OpenAI 兼容 fallback（MiniMax/DeepSeek）",
                    "mock_heuristic": "ℹ️ Mock 启发式（兜底）",
                    "legacy": "ℹ️ 旧 Critic LLM 路径",
                    "unknown": "❓ 未识别",
                }
                for bp in ("jev", "openai_compat", "mock_heuristic", "legacy", "unknown"):
                    n = backend_counter.get(bp, 0)
                    if n == 0:
                        continue
                    pct = n / total_critic * 100
                    lines.append("| " + bp + " | " + str(n) + " | " + f"{pct:.1f}%" + " | " + bp_desc.get(bp, "") + " |\n")
                lines.append("\n")
                jev_pct = backend_counter.get("jev", 0) / total_critic * 100
                if jev_pct >= 95:
                    lines.append("- ✅ Jev 实际使用率 " + f"{jev_pct:.1f}%" + "（接近 100% 说明 fallback 链路健康）\n")
                elif jev_pct >= 50:
                    lines.append("- 🟡 Jev 实际使用率 " + f"{jev_pct:.1f}%" + "（部分 chunk 走了 fallback，检查 _backend_diagnostics）\n")
                else:
                    lines.append("- 🔴 Jev 实际使用率 " + f"{jev_pct:.1f}%" + "（多数 chunk 走了 fallback，调查原因）\n")

                # Round 62: 统计 skipped_dimensions — 让用户看到哪些维度被 Jev 不确定跳过
                # 帮助诊断：是 Jev 真给 clean verdict，还是边界值被 uncertain zone 过滤？
                # Round 75 修复: 原代码引用 chapter_dirs 但从未定义, 整个 try 块静默失败
                # (except Exception 吞掉). 改为 [self.output_dir] (本章节 critic_report 都在该目录下).
                try:
                    from collections import Counter as _SkCounter
                    skipped_counter: _SkCounter = _SkCounter()
                    total_with_skipped = 0
                    # Round 75 修复: 原代码引用 chapter_dirs 但从未定义, 整个 try 块静默失败
                    # (except Exception 吞掉). 改为 [self.output_dir] (本章节 critic_report 都在该目录下).
                    chapter_dirs = [self.output_dir]
                    for chunk_dir in chapter_dirs:
                        for cr_path in chunk_dir.glob("*_critic_report.json"):
                            try:
                                cr = json.loads(cr_path.read_text(encoding="utf-8"))
                            except Exception:
                                continue
                            tv = cr.get("_typesafe_verdict", {}) or {}
                            skipped = tv.get("skipped_dimensions", []) or []
                            if skipped:
                                total_with_skipped += 1
                                for dim in skipped:
                                    skipped_counter[dim] += 1
                    if skipped_counter:
                        lines.append("\n## Jev 边界判断 / 不确定跳过（Round 62）\n\n")
                        lines.append("| 维度 | 跳过次数 | 说明 |\n")
                        lines.append("|---|---:|---|\n")
                        dim_desc = {
                            "voice_drift": "音色漂移（Choice conf < 0.5）",
                            "register_match": "语域契合（Choice conf < 0.5）",
                            "entity_consistency": "实体一致性（Noul ∈ [0.3, 0.7]）",
                            "literal_residue": "直译残留（Noul ∈ [0.3, 0.7]）",
                            "semantic_loss": "关键语义丢失（Noul ∈ [0.3, 0.7]）",
                            "style_compliance_score": "风格契合度（Score conf < 0.6）",
                        }
                        for dim, n in skipped_counter.most_common():
                            lines.append(f"| {dim} | {n} | {dim_desc.get(dim, '—')} |\n")
                        lines.append("\n")
                        lines.append(f"- {total_with_skipped}/{total_critic} chunk 至少有一个维度被跳过\n")
                        lines.append("- 跳过 ≠ 通过：被跳过的维度 Jev 不确定，仍可能有问题但未达 reject 门槛\n")
                except Exception as _sk_err:
                    print(f"  ⚠️ [_print_llm_cost_report] skipped_dimensions 统计失败: {_sk_err}", file=sys.stderr)
        except Exception as _bp_err:
            print(f"  ⚠️ [_print_llm_cost_report] Jev 利用率统计失败: {_bp_err}", file=sys.stderr)

        # Round 32: 章节翻译质量摘要（扫描 final.json 分类 fallback）
        try:
            quality_counter = {
                "PASS": 0,
                "SAFETY_NET_PASSED": 0,
                "PERMANENTLY_FAILED_AFTER_MAX_RETRIES": 0,
                "EARLY_STOP": 0,
                "REWRITER_STAGNATED": 0,  # Round 36 字节停滞
                "PRE_REWRITER_STAGNATION": 0,  # Round 73 pre-rewriter 短路
            }
            total_quality_retries = 0
            max_retries_seen = 0
            for fin_path in self.output_dir.glob("*_final.json"):
                try:
                    with open(fin_path, "r", encoding="utf-8") as f:
                        fin = json.load(f)
                    if not isinstance(fin, dict):
                        continue
                    md = fin.get("metadata", {}) or {}
                    if md.get("fallback"):
                        reason = md.get("reason", "UNKNOWN")
                        quality_counter[reason] = quality_counter.get(reason, 0) + 1
                        qr = int(md.get("quality_retries", 0) or 0)
                        total_quality_retries += qr
                        max_retries_seen = max(max_retries_seen, qr)
                    else:
                        quality_counter["PASS"] += 1
                except (OSError, json.JSONDecodeError):
                    pass
            total_chunks = sum(quality_counter.values())
            if total_chunks > 0:
                pass_pct = quality_counter["PASS"] / total_chunks * 100
                lines.append("\n## 翻译质量摘要（Round 32）\n\n")
                lines.append("| 状态 | chunk 数 | 占比 |\n")
                lines.append("|---|---:|---:|\n")
                status_emoji = {
                    "PASS": "✅",
                    "SAFETY_NET_PASSED": "🟡",
                    "PERMANENTLY_FAILED_AFTER_MAX_RETRIES": "🔴",
                    "EARLY_STOP": "⏭️",
                    "REWRITER_STAGNATED": "⏹️",
                    "PRE_REWRITER_STAGNATION": "⏭️",  # Round 73: pre-rewriter 短路
                }
                for status, n in quality_counter.items():
                    if n == 0:
                        continue
                    pct = n / total_chunks * 100
                    emoji = status_emoji.get(status, "")
                    lines.append(f"| {emoji} {status} | {n} | {pct:.1f}% |\n")
                lines.append(f"| **合计** | **{total_chunks}** | 100% |\n\n")
                lines.append(f"- 一次通过率：**{pass_pct:.1f}%** ({quality_counter['PASS']}/{total_chunks})\n")
                if total_quality_retries > 0:
                    avg_qr = total_quality_retries / max(1, total_chunks - quality_counter["PASS"])
                    lines.append(f"- 失败 chunk 平均重试轮次：{avg_qr:.2f}（最高 {max_retries_seen}）\n")
                # 健康度判断
                if pass_pct >= 90:
                    lines.append("- ✅ 章节翻译质量良好（一次通过率 ≥ 90%）\n")
                elif pass_pct >= 70:
                    lines.append("- 🟡 章节翻译质量中等（一次通过率 70-90%，失败 chunk 已 fallback 定稿）\n")
                else:
                    lines.append(f"- 🔴 章节翻译质量差（一次通过率 {pass_pct:.1f}%），建议人工复核 fallback chunk\n")
        except Exception as _qs_err:
            print(f"  ⚠️ [_print_llm_cost_report] 质量摘要统计失败: {_qs_err}", file=sys.stderr)

        # Round 33: backend diagnostics 摘要（让用户诊断 typesafe 调用失败模式）
        try:
            diag_counter = {"auth": 0, "rate_limit": 0, "timeout": 0, "network": 0,
                            "bad_request": 0, "schema_mismatch": 0, "api_error": 0,
                            "unknown": 0}
            total_diag = 0
            for cr_path in self.output_dir.glob("*_critic_report.json"):
                try:
                    with open(cr_path, "r", encoding="utf-8") as f:
                        cr = json.load(f)
                    diags = cr.get("_backend_diagnostics", {}) or {}
                    if isinstance(diags, dict):
                        for backend_key, diag in diags.items():
                            if isinstance(diag, dict):
                                cat = diag.get("category", "unknown")
                                diag_counter[cat] = diag_counter.get(cat, 0) + 1
                                total_diag += 1
                except (OSError, json.JSONDecodeError):
                    pass
            if total_diag > 0:

                lines.append("\n## Backend Diagnostics 摘要（Round 33）\n\n")
                lines.append("本章累计 backend 失败诊断数：" + str(total_diag) + "\n\n")
                lines.append("| Category | 次数 | 含义 / 建议动作 |\n")
                lines.append("|---|---:|---|\n")
                diag_desc = {
                    "auth": "🔑 认证失败 — 检查 TYPESAFE_API_KEY",
                    "rate_limit": "🚦 速率限制 — 等待或升级配额",
                    "timeout": "⏱️ 超时 — 增加 timeout 或重试",
                    "network": "🌐 网络故障 — 检查 DNS / 防火墙 / 证书",
                    "bad_request": "📝 请求格式错误 — 检查 question schema",
                    "schema_mismatch": "🔧 响应解析失败 — 更新 verdict parser",
                    "api_error": "⚠️ 服务端错误 — 检查 typesafe status page",
                    "unknown": "❓ 未分类",
                }
                for cat, n in sorted(diag_counter.items(), key=lambda kv: -kv[1]):
                    if n == 0:
                        continue
                    desc = diag_desc.get(cat, "❓ 未分类")
                    lines.append(f"| {cat} | {n} | {desc} |\n")
                # 健康度判断
                if diag_counter.get("auth", 0) > 0:
                    lines.append("\n- 🔴 **认证失败次数 > 0**：检查 .env 中的 TYPESAFE_API_KEY 是否有效\n")
                elif diag_counter.get("network", 0) > total_diag * 0.5:
                    lines.append("\n- 🔴 **网络失败占比 > 50%**：检查 DNS / 防火墙 / 证书（参考 Round 18 Jev 利用率）\n")
                elif diag_counter.get("rate_limit", 0) > 0:
                    lines.append("\n- 🟡 出现速率限制：考虑降速或升级 typesafe 配额\n")
        except Exception as _bd_err:
            print(f"  ⚠️ [_print_llm_cost_report] backend diagnostics 统计失败: {_bd_err}", file=sys.stderr)
        # Round 74 修复: 原代码从 summary.get("retry_efficiency") 读, 但 summary 此时尚未填入 retry_efficiency,
        # 导致重试效率表永远不渲染. 改为在表渲染前先计算 re_eff (本地权威), 同时供 cost_summary.json 复用.
        try:
            re_eff_total_retries = 0
            re_eff_wasted = 0
            re_eff_early_stop = 0
            for fin_path in self.output_dir.glob("*_final.json"):
                try:
                    with open(fin_path, "r", encoding="utf-8") as f:
                        fin = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(fin, dict):
                    continue
                md = fin.get("metadata", {}) or {}
                qr = int(md.get("quality_retries", 0) or 0)
                re_eff_total_retries += qr
                if md.get("fallback") and md.get("reason") == "PERMANENTLY_FAILED_AFTER_MAX_RETRIES":
                    re_eff_wasted += qr
                # Round 74: PRE_REWRITER_STAGNATION 也算早停命中
                if md.get("reason") in ("EARLY_STOP", "REWRITER_STAGNATED", "PRE_REWRITER_STAGNATION"):
                    re_eff_early_stop += 1
            re_eff_useful = max(0, re_eff_total_retries - re_eff_wasted)
            re_eff_efficiency_pct = round(re_eff_useful / re_eff_total_retries * 100, 1) if re_eff_total_retries > 0 else None
            if re_eff_total_retries > 0 or re_eff_early_stop > 0:
                eff_str = f"{re_eff_efficiency_pct:.1f}%" if re_eff_efficiency_pct is not None else "-"
                lines.append("\n## 重试效率（Round 44）\n\n")
                lines.append("| 指标 | 值 | 说明 |\n")
                lines.append("|---|---:|---|\n")
                lines.append(f"| 总重试轮次 | {re_eff_total_retries} | 所有 fallback chunk 累计 quality_retries |\n")
                lines.append(f"| 有收益重试 | {re_eff_useful} | 最终通过/安全网放行的 chunk 累计 retry |\n")
                lines.append(f"| 浪费重试 | {re_eff_wasted} | 走到 PERMANENTLY_FAILED_AFTER_MAX_RETRIES 的 retry |\n")
                lines.append(f"| 重试效率 | {eff_str} | useful_retries / total_retries |\n")
                lines.append(f"| 早停命中 | {re_eff_early_stop} | EARLY_STOP / REWRITER_STAGNATED / PRE_REWRITER_STAGNATION 触发的 chunk 数 |\n\n")
        except Exception:
            pass

        # Round 37: per-chunk 摘要表（结合 final.json metadata + critic_report scores + backend path）
        try:
            chunk_rows = []
            for fin_path in sorted(self.output_dir.glob("*_final.json")):
                cid = re.sub(r"_final\.json$", "", fin_path.name)
                try:
                    with open(fin_path, "r", encoding="utf-8") as f:
                        fin = json.load(f)
                    md = fin.get("metadata", {}) or {}
                    if md.get("fallback"):
                        status = md.get("reason", "FALLBACK")
                        emoji = {"SAFETY_NET_PASSED": "🟡", "PERMANENTLY_FAILED_AFTER_MAX_RETRIES": "🔴", "EARLY_STOP": "⏭️", "REWRITER_STAGNATED": "⏹️", "PRE_REWRITER_STAGNATION": "⏭️"}.get(status, "🔴")  # Round 74
                    else:
                        status = "PASS"
                        emoji = "✅"
                    qr = int(md.get("quality_retries", 0) or 0)
                    avg = md.get("avg_score")
                except (OSError, json.JSONDecodeError):
                    status, emoji, qr, avg = "?", "❓", 0, None

                # 配对 critic_report 拿 backend_path
                backend_path = "?"
                cr_path = self.output_dir / f"{cid}_critic_report.json"
                if cr_path.exists():
                    try:
                        with open(cr_path, "r", encoding="utf-8") as f:
                            cr = json.load(f)
                        backend_path = cr.get("_backend_path", "?")
                    except (OSError, json.JSONDecodeError):
                        pass

                chunk_rows.append((cid, emoji, status, qr, avg, backend_path))
            if chunk_rows:
                lines.append("\n## Per-Chunk 摘要（Round 37）\n\n")
                lines.append("| Chunk | 状态 | reason | 重试 | 平均分 | Backend |\n")
                lines.append("|---|---|---|---:|---:|---|\n")
                for cid, emoji, status, qr, avg, bp in chunk_rows:
                    avg_str = f"{avg:.2f}" if isinstance(avg, (int, float)) else "-"
                    lines.append(f"| {cid} | {emoji} | {status} | {qr} | {avg_str} | {bp} |\n")
                # 异常标记
                failed = [r for r in chunk_rows if r[2] not in ("PASS", "SAFETY_NET_PASSED")]
                if failed:
                    lines.append(f"\n- 🔴 {len(failed)} 个 chunk 未通过（PASS/SAFETY 之外）：{[r[0] for r in failed]}\n")
        except Exception as _pc_err:
            print(f"  ⚠️ [_print_llm_cost_report] per-chunk 摘要统计失败: {_pc_err}", file=sys.stderr)

        # Round 39: 估算花费（基于 Round 27 真实 token × 公开单价，仅参考）
        # 单价可在 cost_report 里直接编辑覆盖；缺省按国内常见公开定价
        try:
            PRICING = {
                # role → (input ¥/M tokens, output ¥/M tokens)
                "primary": (0.8, 2.0),                # MiniMax-M3
                "secondary": (0.4, 1.2),              # Round 125: Qwen-flash（估算）
                "literal_translator": (0.1, 0.2),     # deepseek-flash
                "mock": (0.0, 0.0),                   # mock 后端无花费
                "jev": (0.5, 1.5),                    # Jev (typesafe_sdk) — 估算
            }
            lines.append("\n## 估算花费（Round 39，仅供参考）\n\n")
            lines.append("基于 Round 27 真实 per-role token 累计 × 公开单价（单位 ¥，按 1M tokens 折算）：\n\n")
            lines.append("| Role | Prompt tokens | Completion tokens | 估算花费 |\n")
            lines.append("|---|---:|---:|---:|\n")
            total_cost = 0.0
            seen_roles_cost = set()
            for task_type in counter.keys():
                role = _config.get_task_role(task_type)
                if role in seen_roles_cost:
                    continue
                seen_roles_cost.add(role)
                client = get_llm_client(role)
                u = client.get_cumulative_usage()
                if u["calls"] == 0:
                    continue
                p_in, p_out = PRICING.get(role, (0.0, 0.0))
                cost = (u["prompt_tokens"] * p_in + u["completion_tokens"] * p_out) / 1_000_000
                total_cost += cost
                lines.append(
                    f"| {role} | {u['prompt_tokens']:,} | {u['completion_tokens']:,} | ¥{cost:.4f} |\n"
                )
            lines.append(f"| **合计** | | | **¥{total_cost:.4f}** |\n\n")
            lines.append("- 默认单价（¥/M tokens, input/output）：\n")
            for role, (p_in, p_out) in PRICING.items():
                lines.append(f"  - {role}: {p_in} / {p_out}\n")
            lines.append("- ⚠️ 仅为估算，实际花费以 provider 账单为准\n")
        except Exception as _cost_err:
            print(f"  ⚠️ [_print_llm_cost_report] 估算花费失败: {_cost_err}", file=sys.stderr)

        # Round 40: entity registry 章节快照（统计总实体 / 总 variant + 本章活跃实体）
        try:
            if self.decision_engine:
                all_entities = self.decision_engine.conn.execute(
                    "SELECT canonical_name, source_names, variant_names FROM entity_registry"
                ).fetchall()
                total_entities = len(all_entities)
                total_variants = 0
                # 收集本章 final.json 的所有 text 字段（拼成大字符串判断哪些实体出现过）
                chapter_text = ""
                for fin_path in self.output_dir.glob("*_final.json"):
                    try:
                        with open(fin_path, "r", encoding="utf-8") as f:
                            fin = json.load(f)
                        txt = fin.get("text", "") if isinstance(fin, dict) else ""
                        if isinstance(txt, str):
                            chapter_text += txt + "\n"
                    except (OSError, json.JSONDecodeError):
                        pass
                chapter_active = []
                for row in all_entities:
                    canonical = row[0]
                    src_names_json = row[1]
                    var_names_json = row[2] if len(row) > 2 else None
                    var_names = json.loads(var_names_json) if var_names_json else []
                    src_names = json.loads(src_names_json) if src_names_json else []
                    total_variants += len(var_names)
                    # Round 40 fix: 该实体是否在本章出现过（按 canonical / source_name / variant 任一匹配）
                    active = (canonical and canonical in chapter_text) \
                        or any(sn in chapter_text for sn in src_names if sn) \
                        or any(vn in chapter_text for vn in var_names if vn)
                    if active:
                        chapter_active.append((canonical, len(var_names)))
                lines.append("\n## Entity Registry 摘要（Round 40）\n\n")
                lines.append(f"- 全库实体总数：**{total_entities}**\n")
                lines.append(f"- 全库 variant 名总数：**{total_variants}**\n")
                lines.append(f"- 本章活跃实体（出现于 final.json）：**{len(chapter_active)}**\n\n")
                if chapter_active:
                    lines.append("| Canonical | Variant 数 |\n")
                    lines.append("|---|---:|\n")
                    for canonical, vc in sorted(chapter_active, key=lambda kv: -kv[1])[:20]:
                        lines.append(f"| {canonical} | {vc} |\n")
                    if len(chapter_active) > 20:
                        lines.append(f"| ... | (共 {len(chapter_active)} 个) |\n")
        except Exception as _er_err:
            print(f"  ⚠️ [_print_llm_cost_report] entity registry 摘要失败: {_er_err}", file=sys.stderr)

        report_text = "".join(lines)
        report_path = self.output_dir / "cost_report.md"
        report_path.write_text(report_text, encoding="utf-8")

        # Round 41: machine-readable JSON summary alongside cost_report.md
        try:
            summary = {
                "chapter_id": self.chapter_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_llm_calls": int(total),
                "total_tokens_estimated": int(total_tokens),
                "task_breakdown": {t: int(c) for t, c in counter.items() if c > 0},
            }
            # Round 27: real tokens per role
            try:
                role_tokens = {}
                for task_type in counter.keys():
                    role = _config.get_task_role(task_type)
                    if role in role_tokens:
                        continue
                    u = get_llm_client(role).get_cumulative_usage()
                    if u["calls"] > 0:
                        role_tokens[role] = {
                            "calls": u["calls"],
                            "prompt_tokens": u["prompt_tokens"],
                            "completion_tokens": u["completion_tokens"],
                            "total_tokens": u["total_tokens"],
                        }
                summary["real_tokens_by_role"] = role_tokens
            except Exception:
                summary["real_tokens_by_role"] = None
            # Round 39: estimated cost
            try:
                PRICING = {"primary": (0.8, 2.0), "secondary": (0.4, 1.2), "literal_translator": (0.1, 0.2), "mock": (0.0, 0.0), "jev": (0.5, 1.5)}
                total_cost = 0.0
                cost_by_role = {}
                for role, u in (summary.get("real_tokens_by_role") or {}).items():
                    p_in, p_out = PRICING.get(role, (0.0, 0.0))
                    cost = (u["prompt_tokens"] * p_in + u["completion_tokens"] * p_out) / 1_000_000
                    cost_by_role[role] = round(cost, 6)
                    total_cost += cost
                summary["estimated_cost_yuan_by_role"] = cost_by_role
                summary["estimated_cost_yuan_total"] = round(total_cost, 6)
            except Exception:
                summary["estimated_cost_yuan_total"] = None
            # Round 18: Jev utilization
            try:
                backend_counter = {"jev": 0, "openai_compat": 0, "mock_heuristic": 0, "legacy": 0, "unknown": 0}
                for cr_path in self.output_dir.glob("*_critic_report.json"):
                    try:
                        with open(cr_path, "r", encoding="utf-8") as f:
                            cr = json.load(f)
                        bp = cr.get("_backend_path", "unknown")
                        backend_counter[bp] = backend_counter.get(bp, 0) + 1
                    except (OSError, json.JSONDecodeError):
                        pass
                total_critic = sum(backend_counter.values())
                summary["backend_distribution"] = backend_counter
                summary["jev_utilization_pct"] = round(backend_counter["jev"] / total_critic * 100, 1) if total_critic > 0 else None
            except Exception:
                summary["jev_utilization_pct"] = None
            # Round 32: pass rate
            try:
                pass_count = sum(1 for fin_path in self.output_dir.glob("*_final.json")
                                 if not json.loads(fin_path.read_text(encoding="utf-8")).get("metadata", {}).get("fallback"))
                total_finals = sum(1 for _ in self.output_dir.glob("*_final.json"))
                summary["pass_pct"] = round(pass_count / total_finals * 100, 1) if total_finals > 0 else None
                summary["total_chunks"] = total_finals
                summary["passed_chunks"] = pass_count
            except Exception:
                summary["pass_pct"] = None
            # Round 44+74: 重试效率汇总 — 复用本函数顶部已计算的 re_eff_* 变量 (Round 74 修复 pre-render bug)
            try:
                summary["retry_efficiency"] = {
                    "total_quality_retries": re_eff_total_retries,
                    "useful_retries": re_eff_useful,
                    "wasted_retries": re_eff_wasted,
                    "efficiency_pct": re_eff_efficiency_pct,
                    "early_stop_count": re_eff_early_stop,
                }
            except Exception:
                summary["retry_efficiency"] = None
            # Round 45: 持久化 Round 35 建议值到 cost_summary.json（未来可读取）
            try:
                est_suggestions = {}
                seen_roles_r45 = set()
                for task_type in counter.keys():
                    role = _config.get_task_role(task_type)
                    if role in seen_roles_r45:
                        continue
                    seen_roles_r45.add(role)
                    client = get_llm_client(role)
                    u = client.get_cumulative_usage()
                    if u["calls"] == 0:
                        continue
                    avg = u["total_tokens"] / u["calls"]
                    current_est = est_tokens_per_call.get(task_type, 3000)
                    delta_pct = abs(avg - current_est) / current_est * 100 if current_est > 0 else 0
                    est_suggestions[role] = {
                        "task_type": task_type,
                        "real_avg_tokens": round(avg, 1),
                        "current_estimated": current_est,
                        "suggested_estimated": round(avg / 500) * 500,
                        "delta_pct": round(delta_pct, 1),
                        "needs_update": delta_pct > 30,
                    }
                summary["est_tokens_per_call_suggestion"] = est_suggestions
            except Exception:
                summary["est_tokens_per_call_suggestion"] = None
            json_path = self.output_dir / "cost_summary.json"
            json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            # Round 42: 一致性自校验（cost_summary.json vs cost_report.md）
            drift_warnings = []
            try:
                md_text = report_text
                # total_llm_calls: 章节末尾 "合计                N 次"
                import re as _re
                md_total_match = _re.search(r"合计\s+(\d+)\s+次", md_text)
                md_total = int(md_total_match.group(1)) if md_total_match else None
                if md_total is not None and md_total != summary.get("total_llm_calls"):
                    drift_warnings.append(
                        f"total_llm_calls 漂移：json={summary.get('total_llm_calls')} vs md={md_total}"
                    )
                # pass_pct: "一次通过率：**25.0%** (1/4)"
                md_pass_match = _re.search(r"一次通过率：\*\*([\d.]+)%\*\*", md_text)
                md_pass = float(md_pass_match.group(1)) if md_pass_match else None
                if md_pass is not None and summary.get("pass_pct") is not None:
                    if abs(md_pass - summary["pass_pct"]) > 0.1:
                        drift_warnings.append(
                            f"pass_pct 漂移：json={summary['pass_pct']:.1f}% vs md={md_pass:.1f}%"
                        )
                # jev_utilization_pct: "Jev 实际使用率 50.0%"
                md_jev_match = _re.search(r"Jev 实际使用率 ([\d.]+)%", md_text)
                md_jev = float(md_jev_match.group(1)) if md_jev_match else None
                if md_jev is not None and summary.get("jev_utilization_pct") is not None:
                    if abs(md_jev - summary["jev_utilization_pct"]) > 0.1:
                        drift_warnings.append(
                            f"jev_utilization_pct 漂移：json={summary['jev_utilization_pct']:.1f}% vs md={md_jev:.1f}%"
                        )
                # Round 44: retry_efficiency — wasted_retries + useful_retries 数字一致性
                re_eff = summary.get("retry_efficiency") or {}
                md_wasted_match = _re.search(r"浪费重试 \| (\d+)", md_text)
                md_wasted = int(md_wasted_match.group(1)) if md_wasted_match else None
                if md_wasted is not None and re_eff.get("wasted_retries") is not None:
                    if md_wasted != re_eff["wasted_retries"]:
                        drift_warnings.append(
                            f"retry_efficiency.wasted_retries 漂移：json={re_eff['wasted_retries']} vs md={md_wasted}"
                        )
                if drift_warnings:
                    print(
                        f"⚠️ [Cost Report Drift] {len(drift_warnings)} 项 metric 在 cost_report.md 与 cost_summary.json 之间不一致：",
                        file=sys.stderr,
                    )
                    for w in drift_warnings:
                        print(f"  - {w}", file=sys.stderr)
            except Exception as _drift_err:
                pass  # drift check failure is non-fatal
        except Exception as _js_err:
            print(f"  ⚠️ [_print_llm_cost_report] cost_summary.json dump failed: {_js_err}", file=sys.stderr)

        print()
        print("[LLM Cost] " + self.chapter_id + " 调用统计:")
        for task_type, count in sorted_types:
            if count > 0:
                print("    " + display_names.get(task_type, task_type).ljust(20) + " " + str(count).rjust(3) + " 次")
        print("    " + "合计".ljust(20) + " " + str(total).rjust(3) + " 次")
        print("    详细: " + str(report_path))

    def _cleanup_intermediates(self) -> None:
        """Round 14: 章节完成后清理 intermediate 文件

        保留：
          - chunk_xxx_final.json（最终译文）
          - chunk_xxx_literary.json（canonical 润色稿）
          - chunk_xxx_critic_report.json（最后一次 critic 报告）
          - chunk_xxx_raw.json（直译底稿）
          - chunk_xxx_literary_vN.json（N = 该 chunk 最终 quality_retries；保留最后一版）
          - chunk_xxx_rewrite_meta_vN.json（同上，最后一次）

        删除（对于 COMPLETED chunks）：
          - chunk_xxx_literary_v0..vN-1.json（已被 vN 覆盖）
          - chunk_xxx_rewrite_meta_v0..vN-1.json（同上）

        PERMANENTLY_FAILED chunks 默认保留全部中间文件（人工 inspect 需要）。
        可通过 config.pipeline.intermediate_cleanup.preserve_for_failed=false 关闭。
        """
        cfg = _config.pipeline.get("intermediate_cleanup", {}) or {}
        if not cfg.get("enabled", True):
            return
        preserve_for_failed = bool(cfg.get("preserve_for_failed", True))

        cleaned_files = 0
        kept_files = 0
        for chunk_file in sorted(self.output_dir.glob("*_final.json")):
            chunk_id = chunk_file.stem.replace("_final", "")
            # 检查 chunk 状态
            task_state = None
            try:
                tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
                for t in tasks:
                    if t.get("chunk_id") == chunk_id:
                        task_state = t.get("state")
                        break
            except Exception:
                pass

            # PERMANENTLY_FAILED 默认保留
            if preserve_for_failed and task_state == "PERMANENTLY_FAILED":
                kept_files += 1
                continue

            # 找最后一次 retry 的 quality_retries（从 DB 取）
            final_qr = 0
            try:
                tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
                for t in tasks:
                    if t.get("chunk_id") == chunk_id:
                        final_qr = int(t.get("quality_retries", 0) or 0)
                        break
            except Exception:
                pass

            # 删除 v0..vN-1（保留 vN 即最后一次）
            for r in range(final_qr):
                for pattern in [f"{chunk_id}_literary_v{r}.json", f"{chunk_id}_rewrite_meta_v{r}.json"]:
                    f = self.output_dir / pattern
                    if f.exists():
                        try:
                            f.unlink()
                            cleaned_files += 1
                        except OSError:
                            pass

        # 报告清理结果
        total_chunks = sum(1 for _ in self.output_dir.glob("*_final.json"))
        preserved_failed = sum(
            1 for t in self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
            if t.get("state") == "PERMANENTLY_FAILED"
        ) if preserve_for_failed else 0
        if cleaned_files > 0:
            print(
                f"🧹 [Cleanup] 清理 {cleaned_files} 个中间文件"
                f"（{total_chunks - preserved_failed} COMPLETED chunk 去重 + "
                f"{preserved_failed} PERMANENTLY_FAILED 保留 inspect）"
            )
        elif total_chunks > 0:
            print(
                f"🧹 [Cleanup] 无中间文件需清理（{total_chunks} chunks："
                f"{total_chunks - preserved_failed} COMPLETED 无 retry 历史 + "
                f"{preserved_failed} PERMANENTLY_FAILED 保留）"
            )

    def _entity_registry_conflict_audit(self) -> List[Dict]:
        """Round 15: 检测 entity_registry 中 source_name 冲突

        冲突定义：同一英文 source_name 被登记到 ≥2 个 canonical 时
        影响：consistency.py 无法确定替换目标，导致 Round 7/9 行为不可预期
        解决：本方法只生成诊断报告（entity_registry_conflicts.md），运维手动修复

        Returns:
            [{source_name: str, canonicals: [str, ...], conflict_type: str}, ...]
        """
        if not self.decision_engine:
            return []
        # Round 15: 查全部实体（含 pending_review=1）—— 冲突可能就藏在待核查里
        try:
            cursor = self.decision_engine.conn.execute(
                "SELECT canonical_name, source_names FROM entity_registry"
            )
            entities = [(row[0], row[1], "[]") for row in cursor.fetchall()]
        except Exception:
            return []

        # 反向索引：source_name → [canonical, ...]
        from collections import defaultdict
        src_to_canonicals = defaultdict(list)
        for row in entities:
            canonical = row[0]
            sn_json = row[1]
            sn_list = robust_json_loads(sn_json, expected_type=list) if sn_json else []
            for sn in sn_list:
                if sn:
                    src_to_canonicals[sn].append(canonical)

        # 找冲突：source_name → ≥2 canonicals
        conflicts = []
        for src_name, canonicals in src_to_canonicals.items():
            if len(canonicals) < 2:
                continue
            # 区分冲突类型
            # 同名变体（如 "Mr. Lomax" 同时给 洛马克斯先生 + 乔治·洛马克斯先生）：可能是 acceptable（同一人不同译名）
            # 跨人物冲突（如 "Joe" 同时给 安东尼·凯德 + 绅士乔）：明确错误
            conflict_type = "cross_entity"  # 默认
            canon_set = set(canonicals)
            # 简单启发式：如果所有 canonical 都包含相同的人名姓氏，可能是变体
            # 否则视为跨人物冲突
            unique_first_chars = set()
            for c in canonicals:
                # 取中文译名前 2 字作为"姓氏"
                if len(c) >= 2:
                    unique_first_chars.add(c[:2])
            if len(unique_first_chars) == 1:
                conflict_type = "variant_for_same_person"
            else:
                conflict_type = "cross_entity"

            conflicts.append({
                "source_name": src_name,
                "canonicals": canonicals,
                "conflict_type": conflict_type,
            })

        return conflicts

    def _post_pipeline_entity_audit(self) -> dict:
        """Round 9: 实体完整性审计（纯诊断，不改译文）

        对每个 chunk：
          1. 从 entity_registry 找"本章 source_text 出现过"的实体（任一 source_name in source_text）
          2. 检查 final_text 中是否含 canonical 或任一已登记 variant_names
          3. 若均缺失 → 计入 missing（提示 Rewriter 漏译）

        输出：output/{chapter_id}/entity_audit.md（人类可读）+ entity_audit.json（结构化）

        Returns:
            {chunk_id: [missing_canonical_names, ...], ...}
        """
        if not self.decision_engine:
            return {}
        try:
            from src.core.decision_engine import query_reviewable_entities
            entities = query_reviewable_entities(self.decision_engine.conn)
        except Exception:
            return {}

        # 把 chapter 内所有 chunk 的 source_text 预加载
        all_tasks = {t["chunk_id"]: t.get("text_content", "") for t in
                     self.scheduler.get_all_tasks_by_chapter(self.chapter_id)}

        report_lines = []
        report_data = {}
        for chunk_file in sorted(self.output_dir.glob("*_final.json")):
            chunk_id = chunk_file.stem.replace("_final", "")
            try:
                data = json.loads(chunk_file.read_text(encoding="utf-8"))
                text = data.get("text", "") if isinstance(data, dict) else str(data)
            except (json.JSONDecodeError, OSError):
                continue
            source_text = all_tasks.get(chunk_id, "")
            if not source_text:
                continue

            chunk_missing = []
            chunk_present = []
            for row in entities:
                canonical = row[0]
                if not canonical:
                    continue
                source_names_json = row[1]
                variant_names_json = row[2] if len(row) > 2 else None
                source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
                variant_names = robust_json_loads(variant_names_json, expected_type=list) if variant_names_json else []

                # Round 112: 身份映射 canonical（未翻译英文主名）不进入实体审计
                if not self._is_enforceable_canonical(canonical, source_names):
                    continue

                # 哪些 source_name 在本章出现过？
                # Round 106: 过滤无效 source_name（动词短语/关系代词/泛称，如 "I could"/
                # "his wife"/"the fellow"），避免 reference agent 误登记的脏指称导致
                # 审计误报"漏译"（ch05 实测 女主人公 因 "I could" 被误判 missing）。
                # Round 133b: 别名吞并抑制（与 _check_entity_consistency 同语义）—
                # 剔除被其它实体更长别名吞并的伪命中（chunk071 "Mother"⊂"Reverend Mother"）。
                valid_sns = [
                    sn for sn in source_names
                    if sn and self._is_valid_entity_source_name(sn)
                ]
                effective_sns, _swallowed = self._partition_swallowed_source_names(
                    source_text, canonical, valid_sns,
                )
                matched_sources = [
                    sn for sn in effective_sns
                    if entity_source_name_in_text(sn, source_text)
                ]
                if not matched_sources:
                    continue

                # 检查 final_text 是否含 canonical 或任一 variant
                # Round 105: 用 _canonical_present_in_text 容差判定（与一致性检查同语义），
                # 消除括注名（镇（town）/科德斯（Codders））与前缀写法（乔治·洛马克斯 vs 乔治·洛马克斯先生）的假阳性
                present_forms = [canonical] + [v for v in variant_names if v]
                found = any(self._canonical_present_in_text(form, text) for form in present_forms)
                if found:
                    chunk_present.append({
                        "canonical": canonical,
                        "matched_sources": matched_sources,
                    })
                else:
                    chunk_missing.append({
                        "canonical": canonical,
                        "matched_sources": matched_sources,
                    })

            report_data[chunk_id] = {
                "missing": chunk_missing,
                "present": chunk_present,
            }

        # 写 markdown 报告
        cfg = _config.pipeline.get("entity_audit", {}) or {}
        min_missing = int(cfg.get("min_chunk_missing_threshold", 1))
        total_missing = sum(len(d["missing"]) for d in report_data.values())

        report_lines.append(f"# 实体完整性审计 — {self.chapter_id}\n")
        report_lines.append(f"- 生成时间: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        report_lines.append(f"- 扫描 chunk 数: {len(report_data)}\n")
        report_lines.append(f"- 漏译实体总数: {total_missing}\n")
        report_lines.append(f"- 仅供诊断；不会修改 final.json\n\n")

        chunks_with_missing = [(cid, d) for cid, d in report_data.items()
                                if len(d["missing"]) >= min_missing]
        if not chunks_with_missing:
            report_lines.append("✅ 本章所有已登记实体均已在 final.json 中正确呈现。\n")
        else:
            for cid, d in chunks_with_missing:
                report_lines.append(f"## {cid} 漏译 {len(d['missing'])} 个实体\n")
                for ent in d["missing"]:
                    src_form = " / ".join(ent["matched_sources"][:3])
                    report_lines.append(f"- **{ent['canonical']}**（源文: {src_form}）\n")
                report_lines.append("\n")

        report_path = self.output_dir / "entity_audit.md"
        report_path.write_text("".join(report_lines), encoding="utf-8")

        # 结构化 JSON（供后续程序消费）
        json_path = self.output_dir / "entity_audit.json"
        json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2),
                             encoding="utf-8")

        if total_missing > 0:
            print(f"⚠️ [Entity Audit] 本章共漏译 {total_missing} 个实体（详见 entity_audit.md）")
        else:
            print(f"✅ [Entity Audit] 本章所有已登记实体均已翻译")
        return report_data

    def _recover_completed_without_final(self) -> List[str]:
        """Round 104: 恢复"COMPLETED 但 final.json 缺失"的 chunk.

        根因: 回溯 (COMPLETED -> DIRTY) 会删除 final.json (_process_dirty_batch),
        若重跑因 bug 未重新落盘 final.json, chunk 会卡在 COMPLETED 且无产出.
        此时 --force (smart-force) 会保留 COMPLETED chunk, 用户无法恢复.
        本方法在 run() 启动时检测这类 chunk 并重置为 DIRTY, 让其重新走完整流水线.

        Returns:
            被恢复的 chunk_id 列表.
        """
        recovered: List[str] = []
        try:
            tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
        except Exception:
            return recovered
        for t in tasks:
            if t.get("state") != TaskState.COMPLETED.value:
                continue
            chunk_id = t.get("chunk_id")
            if not chunk_id:
                continue
            final_path = self.output_dir / f"{chunk_id}_final.json"
            if final_path.exists():
                continue
            # COMPLETED 但无 final.json → 重置为 DIRTY 重跑 (合法转移, 不增计数)
            try:
                # Round 130（审计③）: 感知返回值 —— 拒绝时不得假报已恢复
                if self.scheduler.update_task_state(chunk_id, TaskState.DIRTY):
                    recovered.append(chunk_id)
                    print(f"♻️ [Pipeline] {chunk_id} 状态为 COMPLETED 但 final.json 缺失，重置为 DIRTY 重跑")
                else:
                    print(f"⚠️ [Pipeline] {chunk_id} 重置 DIRTY 被拒绝（非法转移），跳过（状态未变）")
            except Exception as e:
                print(f"⚠️ [Pipeline] {chunk_id} 恢复失败: {e}")
        return recovered

    def _restore_dirty_backups(self):
        """Round 107 P7: 恢复回溯重跑失败的 final_backup.json.

        回溯重跑 (COMPLETED→DIRTY) 时 final.json 被重命名为 final_backup.json.
        若重跑成功, 新 final.json 会写入; Round 130 版本择优在此裁决:
        新版 critic 均分低于旧版时保留旧版（备份版覆盖回 final.json）,
        新版至少不差时删除无用 backup.
        若重跑失败 (PERMANENTLY_FAILED), final.json 不存在, 恢复 backup:
        - final_backup.json → final.json
        - 状态 PERMANENTLY_FAILED → COMPLETED (保留之前的合格译文)
        """
        restored = []
        for backup in self.output_dir.glob("*_final_backup.json"):
            chunk_id = backup.stem.replace("_final_backup", "")
            final_path = self.output_dir / f"{chunk_id}_final.json"
            if final_path.exists():
                # 重跑成功, 新 final.json 已写入.
                # Round 130: 版本择优 — 新版 critic 均分低于旧版(超出 tolerance)时
                # 用备份版覆盖 final.json（保留更优旧版），返回 True。
                if self._apply_version_merit(chunk_id, backup, final_path):
                    continue
                # 新版至少不差 → 清理无用 backup（含 critic 备份）
                try:
                    backup.unlink()
                except OSError:
                    pass
                self._cleanup_critic_backup(chunk_id)
                continue
            # 重跑失败, 恢复 backup
            try:
                backup.rename(final_path)
                # 状态恢复: PERMANENTLY_FAILED → COMPLETED (Round 124: 显式放行的终态恢复)
                # Round 130（审计③）: 感知返回值，拒绝时不再假报 COMPLETED
                try:
                    _state_ok = self.scheduler.update_task_state(
                        chunk_id, TaskState.COMPLETED, allow_terminal_restore=True)
                except Exception as _st_exc:
                    _state_ok = False
                    print(f"  ⚠️ {chunk_id} 状态恢复异常: {_st_exc}")
                restored.append(chunk_id)
                if _state_ok:
                    print(f"♻️ [Pipeline] {chunk_id} 回溯重跑失败, 已恢复之前的 final.json (COMPLETED)")
                else:
                    print(f"⚠️ [Pipeline] {chunk_id} 已恢复 final.json，但状态恢复被拒绝（状态未变，请人工核查 chunk 状态）")
            except OSError as e:
                print(f"  ⚠️ 恢复 {chunk_id} backup 失败: {e}")
        return restored

    def _cleanup_critic_backup(self, chunk_id: str) -> None:
        """Round 130: 清理已裁决的 *_critic_report_backup.json（防止残留堆积）。"""
        cb = self.output_dir / f"{chunk_id}_critic_report_backup.json"
        if cb.exists():
            try:
                cb.unlink()
            except OSError as e:
                print(f"  ⚠️ 清理 {cb.name} 失败: {e}")

    def _apply_version_merit(self, chunk_id: str, backup: Path, final_path: Path) -> bool:
        """Round 130 版本择优: 对比备份版与新版 critic 均分, 决定是否保留旧版.

        新版均分 < 旧版均分 - tolerance → 用备份版覆盖 final.json（旧版更优），
        并把备份 critic 报告归位为 critic_report.json；返回 True。
        其余情况返回 False（调用方按"无用 backup"清理）。
        旧版/新版评分缺失（无 critic 文件，如 legacy chunk）时不做裁决。
        """
        vm_cfg = _config.pipeline.get("version_merit", {}) or {}
        if not vm_cfg.get("enabled", True):
            return False
        critic_new = self.output_dir / f"{chunk_id}_critic_report.json"
        critic_old = self.output_dir / f"{chunk_id}_critic_report_backup.json"
        old_score = self._critic_avg_from_file(critic_old)
        new_score = self._critic_avg_from_file(critic_new)
        if old_score <= 0 or new_score <= 0:
            return False
        try:
            tolerance = float(vm_cfg.get("tolerance", 0.0) or 0.0)
        except (TypeError, ValueError):
            tolerance = 0.0
        if new_score >= old_score - tolerance:
            return False
        try:
            final_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            backup.unlink()
            if critic_old.exists():
                critic_old.replace(critic_new)  # 旧版评分归位, 保持 final 与评分一致
        except OSError as e:
            print(f"  ⚠️ [Version Merit] {chunk_id} 恢复旧版本失败: {e}")
            return False
        print(f"🛡️ [Version Merit] {chunk_id} 新版均分 {new_score:.2f} < 旧版 {old_score:.2f}，保留旧版")
        return True

    def _critic_avg_from_file(self, path: Path) -> float:
        """读取 critic_report json 的数值维度均分; 缺失/损坏返回 0.0。"""
        try:
            return self._compute_avg_score(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def _verify_final_outputs(self, expected_count: int, strict: bool = True) -> bool:
        """校验本章 final.json 实际落盘是否匹配任务。

        防"任务队列空即报成功"的假阳性完成。除比数外，还比对 chunk_id
        集合：残留旧 final（上一轮多余 chunk）或缺失都判残缺，避免 stale 假通过。

        Round 76: 分离 completeness 与 quality 校验。
          - completeness 失败（缺 final.json / 残留旧文件）→ return False（真残缺）
          - 仅 quality 降级（state=PERMANENTLY_FAILED 但 final.json 已落盘的 fallback 章节）
            → 视为"产出完整但质量降级"，return True，仅 warning 告知。
        旧版本将 fallback 章节也判 False，导致 CLI 退出码 = 1 与磁盘上
        已有 final.json 的事实不一致；调用方可读 cost_report 看到 fallback 分类。

        Round 119 T4: DIRTY 单独处理 —— DIRTY 且带 final 是"回溯标记未整合"的
        陈旧产物，return False（先 consolidate 收敛再合并；DIRTY 不入成书）。

        返回 True 表示产出完整（可能含 fallback 降级），False 表示真残缺。
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
            all_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
            dirty_chunks = [t["chunk_id"] for t in all_tasks
                            if t["state"] == TaskState.DIRTY.value]
            if dirty_chunks:
                print(f"❌ [Pipeline] 章节 {self.chapter_id} 有 {len(dirty_chunks)} 个 chunk 处于 DIRTY（陈旧待整合）："
                      f"{dirty_chunks}")
                print(f"   请运行 consolidate 收敛 DIRTY 后再合并（T4: DIRTY 不入成书）。")
                return False
            bad = [t["chunk_id"] for t in all_tasks
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
                # Round 76: 仅 fallback（无 missing）→ 产出完整但质量降级，warning 已告知。
                return True
        return True

    def _process_pending_batch(self, tasks):
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS)

    def _process_dirty_batch(self, tasks):
        for task in tasks:
            chunk_id = task['chunk_id']
            print(f"🔄 [Pipeline] {chunk_id} 标记为 DIRTY，重置到术语提取")
            # Round 108: 清空上一轮的分数历史/最佳版本 — 回溯重跑是新 episode,
            # 旧历史会污染 streak 早停 (ch08 chunk006: 旧历史 8.157→7.943 与重跑
            # 7.94 拼成「单调不增」假阳性, 重跑仅 1 次 retry 就被提前 fallback)。
            self._score_history.pop(chunk_id, None)
            self._best_lit.pop(chunk_id, None)
            self._ensure_candidate_pools()
            self._lit_candidates.pop(chunk_id, None)
            self._pairwise_info.pop(chunk_id, None)
            # Round 107 P7: final.json 不删除, 重命名为 final_backup.json.
            # ch07 倒查: 回溯重跑 (COMPLETED→DIRTY) 删除 final.json 后, 若重跑失败
            # (NO_PROGRESS_STREAK 等), chunk 从 COMPLETED 降级为 PERMANENTLY_FAILED 且无产出.
            # 保留备份: 重跑成功会覆盖, 重跑失败则 _restore_dirty_backups 恢复.
            final_path = self.output_dir / f"{chunk_id}_final.json"
            backup_path = self.output_dir / f"{chunk_id}_final_backup.json"
            if final_path.exists():
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                    final_path.rename(backup_path)
                    print(f"  💾 已备份 final.json → final_backup.json (重跑失败可恢复)")
                except OSError as e:
                    print(f"  ⚠️ 备份 final.json 失败: {e}")
            # Round 130: critic_report 保留为 *_critic_report_backup.json，
            # 供章末"版本择优"（_apply_version_merit）对比新旧评分。
            critic_path = self.output_dir / f"{chunk_id}_critic_report.json"
            critic_backup = self.output_dir / f"{chunk_id}_critic_report_backup.json"
            if critic_path.exists():
                try:
                    if critic_backup.exists():
                        critic_backup.unlink()
                    critic_path.rename(critic_backup)
                except OSError as e:
                    print(f"  ⚠️ 备份 critic_report 失败: {e}")
            for step in ["raw", "literary"]:
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

    def _consolidate_deferred_backtracks(self) -> bool:
        """Round 130: 章末合并式回溯（decision_engine.backtrack_mode=deferred）。

        在批处理循环"队列清空"点调用：把 decision_engine 登记的本章待回溯 chunk
        合并应用一次（标记 DIRTY → 主循环继续重跑这些块）。

        治理预算（防自激循环）：
          - 章级：每章每次运行最多应用 max_backtrack_sweeps_per_chapter 轮（默认 1）；
            预算用尽后仅冻结（队列登记保留至下一次运行，打印告警）。
          - 单块：累计回溯轮数上限由 scheduler.trigger_backtrack 执行（超限冻结，
            并写出 backtrack_frozen.md 明细）。

        Returns:
            True 表示有 chunk 被标记 DIRTY（调用方应继续主循环）。
        """
        de_cfg = _config.decision_engine or {}
        mode = str(de_cfg.get("backtrack_mode", "deferred") or "deferred").lower()
        if mode == "immediate":
            return False
        if not getattr(self, "decision_engine", None):
            return False
        try:
            max_sweeps = int(de_cfg.get("max_backtrack_sweeps_per_chapter", 1) or 0)
        except (TypeError, ValueError):
            max_sweeps = 1
        sweeps_done = int(getattr(self, "_backtrack_sweeps_done", 0) or 0)
        task_ids = [t["chunk_id"] for t in self.scheduler.get_all_tasks_by_chapter(self.chapter_id)]
        if sweeps_done >= max_sweeps:
            pending = self.decision_engine.count_pending_backtracks(task_ids)
            if pending:
                print(f"🧊 [Backtrack] 本章合并预算已用尽（{max_sweeps} 轮），"
                      f"冻结 {pending} 项待回溯登记（保留至下次运行）")
            return False
        entries = self.decision_engine.take_pending_backtracks(task_ids)
        if not entries:
            return False
        stats = self.scheduler.trigger_backtrack([e["chunk_id"] for e in entries])
        self._backtrack_sweeps_done = sweeps_done + 1
        dirty = stats.get("dirty", []) if isinstance(stats, dict) else []
        over_cap = stats.get("over_cap", []) if isinstance(stats, dict) else []
        print(f"🧩 [Backtrack] 章末合并回溯：登记 {len(entries)} 个 → 标记 DIRTY {len(dirty)} 个"
              + (f"，治理冻结 {len(over_cap)} 个（超单块预算）" if over_cap else ""))
        if over_cap:
            self._write_backtrack_frozen_report(over_cap, entries)
        return bool(dirty)

    def _write_backtrack_frozen_report(self, frozen_ids: List[str], entries: List[Dict]) -> None:
        """把治理冻结的 chunk 写入 output/<chapter>/backtrack_frozen.md（诊断用）。"""
        try:
            meta = {e["chunk_id"]: e for e in entries if isinstance(e, dict)}
            lines = ["# 回溯治理冻结记录 (Round 130)\n\n",
                     "- 冻结原因：单块累计回溯轮数达到 max_backtrack_rounds_per_chunk\n\n"]
            for cid in frozen_ids:
                e = meta.get(cid, {})
                lines.append(f"- `{cid}` ← source_key={e.get('source_key', '?')} "
                             f"(queue_rounds={e.get('queue_rounds', '?')})\n")
            path = self.output_dir / "backtrack_frozen.md"
            path.write_text("".join(lines), encoding="utf-8")
            print(f"  📄 冻结明细: {path}")
        except OSError as e:
            print(f"  ⚠️ 写 backtrack_frozen.md 失败: {e}")

    # ────────────────────────────────────────────────────────────────
    # Round 133: 重试适配 — 「仅推理无正文」(no_content) 签名驱动的
    # 自动降档（max→xhigh→high→…）与换角色救稿（重试预算耗尽后最后 1 次）。
    # 背景：ch01 通宵跑实测 7 块 PERMANENTLY_FAILED 全部为 effort=max 思考链
    # （5-7 万字符）吃光 max_tokens、正文为空；同参数重试必然复现。
    # 只对命中签名的块生效（override 按块持久化），正常块零影响。
    # ────────────────────────────────────────────────────────────────
    _NO_CONTENT_MARKERS = ("无正文", "未闭合 think 块", "reasoning_only")

    @classmethod
    def _is_no_content_signature(cls, err_text) -> bool:
        """判定异常文本是否为「仅推理无正文」签名（thinking 耗尽 max_tokens）。"""
        text = str(err_text or "")
        return any(m in text for m in cls._NO_CONTENT_MARKERS)

    @staticmethod
    def _effort_levels_table() -> Dict[str, tuple]:
        """provider → 合法 effort 档位表（优先 src.config 单一事实源；单体兜底内置副本）。"""
        try:
            from src.config import EFFORT_LEVELS_BY_PROVIDER as _tbl
            return dict(_tbl)
        except Exception:
            return {
                "minimax": ("low", "medium", "high", "xhigh", "max"),
                "deepseek": ("none", "low", "high", "xhigh", "max"),
                "qwen": ("none", "low", "medium", "xhigh"),
            }

    @classmethod
    def _provider_of_role(cls, role: str) -> Optional[str]:
        """按角色配置的 model_name/api_base 推断 effort provider；未知返回 None。"""
        try:
            role_models = getattr(_config, "role_models", None)
            if callable(role_models):
                role_models = role_models()
            if isinstance(role_models, dict) and role_models and role not in role_models:
                return None  # 未定义角色：不用 legacy 兜底模型猜 provider
        except Exception:
            pass
        try:
            role_cfg = _config.get_role_config(role) or {}
        except Exception:
            return None
        target = f"{role_cfg.get('model_name', '')} {role_cfg.get('api_base', '')}".lower()
        for provider in cls._effort_levels_table():
            if provider in target:
                return provider
        return None

    def _retry_adapt_cfg(self) -> Dict[str, Any]:
        cfg = dict(_config.pipeline.get("retry_adaptation", {}) or {})
        cfg.setdefault("enabled", True)
        cfg.setdefault("downgrade_on_signature", ["no_content"])
        cfg.setdefault("fallback_role_switch", True)
        cfg.setdefault("minimax_protect", True)
        cfg.setdefault("quality_escalate_back", True)
        return cfg

    def _load_model_capabilities(self) -> Dict[str, Any]:
        """读取 L1 档位探测结果（db/model_capabilities.json）；缺失/过期 → 空 dict。"""
        try:
            probe_cfg = dict(_config.pipeline.get("model_probe", {}) or {})
        except Exception:
            probe_cfg = {}
        rel = str(probe_cfg.get("path") or "db/model_capabilities.json")
        path = Path(rel)
        if not path.is_absolute():
            root = getattr(self, "_root_dir", None) or Path.cwd()
            path = Path(root) / rel
        try:
            if not path.exists():
                return {}
            ttl_days = float(probe_cfg.get("ttl_days", 7) or 7)
            if ttl_days > 0 and (time.time() - path.stat().st_mtime) > ttl_days * 86400:
                return {}
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _effort_ladder_for_role(self, role: str) -> tuple:
        """角色可用档位阶梯：provider 全档位 → 探测文件 accepted 过滤（有则用）。"""
        provider = self._provider_of_role(role)
        if not provider:
            return ()
        ladder = tuple(self._effort_levels_table().get(provider) or ())
        caps = self._load_model_capabilities()
        rec = dict((caps.get("models") or {}).get(role) or {})
        levels_rec = dict(rec.get("levels") or {})
        if rec.get("provider") == provider and levels_rec:
            accepted = [lv for lv in ladder if (levels_rec.get(lv) or {}).get("accepted")]
            if accepted:
                return tuple(accepted)
        return ladder

    def _downgrade_one_step(self, role: str, effort) -> Optional[str]:
        """按 provider 档位表降一档；降无可降/档位未知 → None。"""
        ladder = self._effort_ladder_for_role(role)
        eff = str(effort or "").strip().lower()
        if not ladder or not eff or eff not in ladder:
            return None
        i = ladder.index(eff)
        return ladder[i - 1] if i > 0 else None

    def _escalate_one_step(self, role: str, effort) -> Optional[str]:
        ladder = self._effort_ladder_for_role(role)
        eff = str(effort or "").strip().lower()
        if not ladder or not eff or eff not in ladder:
            return None
        i = ladder.index(eff)
        return ladder[i + 1] if i + 1 < len(ladder) else None

    def _get_retry_override(self, chunk_id: str) -> Dict[str, Any]:
        try:
            return dict(self.scheduler.get_retry_override(chunk_id) or {})
        except Exception:
            return {}

    def _set_retry_override(self, chunk_id: str, payload: Dict[str, Any]) -> bool:
        try:
            return bool(self.scheduler.set_retry_override(chunk_id, payload))
        except Exception:
            return False

    def _task_default_effort(self, task_name: str) -> Optional[str]:
        try:
            _, params = _config.resolve_task_model(task_name)
            eff = params.get("effort")
            return str(eff).strip().lower() if eff else None
        except Exception:
            return None

    def _note_last_call(self, chunk_id: str, stage: str, role: str, effort, model_name: str = "") -> None:
        """记录块最近一次 LLM 调用的 (stage, role, effort)，供 no_content 签名记录使用。"""
        if not chunk_id:
            return
        cache = getattr(self, "_last_call_info", None)
        if cache is None:
            cache = {}
            self._last_call_info = cache
        cache[chunk_id] = {
            "stage": stage,
            "role": role or "",
            "effort": (str(effort).strip().lower() if effort else None),
            "model_name": model_name or "",
        }

    def _record_no_content_signal(self, chunk_id: str, stage: str) -> None:
        """命中「仅推理无正文」签名 → 持久化降档 override（下次重试自动降一档）。"""
        cfg = self._retry_adapt_cfg()
        if not cfg.get("enabled", True):
            return
        sigs = {str(s).strip().lower() for s in (cfg.get("downgrade_on_signature") or ["no_content"])}
        if "no_content" not in sigs:
            return
        info = (getattr(self, "_last_call_info", {}) or {}).get(chunk_id) or {}
        used_role = info.get("role") or ""
        used_eff = info.get("effort")
        prev = self._get_retry_override(chunk_id)
        next_eff = self._downgrade_one_step(used_role, used_eff) if used_eff else None
        if next_eff is None and prev.get("effort"):
            next_eff = prev.get("effort")  # 已降无可降：保留现有 override
        delta = int(prev.get("down_delta", 0) or 0) + (
            1 if (next_eff and next_eff != prev.get("effort")) else 0)
        payload = {
            "sig": "no_content",
            "stage": stage,
            "role": prev.get("role") or used_role,
            "effort": next_eff,
            "used_role": used_role,
            "used_effort": used_eff,
            "down_delta": delta,
            "rescue_used": bool(prev.get("rescue_used")),
            "escalated": bool(prev.get("escalated")),
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if self._set_retry_override(chunk_id, payload):
            if next_eff:
                print(f"🔻 [Pipeline] {chunk_id} no_content 签名 → 下次重试降档 "
                      f"{used_eff or 'default'}→{next_eff} (role={payload['role'] or '?'})")
            else:
                print(f"🔻 [Pipeline] {chunk_id} no_content 签名（{used_eff or 'default'} 已无更低档）")

    def _rescue_role_for(self, stage: str, failing_role: str) -> Optional[str]:
        """换角色救稿目标选择（fallback_role_switch）。

        RAW 救稿默认 secondary/literal_translator（minimax_protect 时不选 primary）；
        REWRITE 救稿排除失败角色，protect 时优先 secondary。
        """
        cfg = self._retry_adapt_cfg()
        protect = bool(cfg.get("minimax_protect", True))
        if stage == "TRANSLATING_RAW":
            cands = ["secondary", "literal_translator"]
            if not protect:
                cands.append("primary")
        else:
            if protect and failing_role != "secondary":
                cands = ["secondary", "literal_translator", "primary"]
            else:
                cands = ["primary", "secondary", "literal_translator"]
        cands = [r for r in cands if r and r != failing_role]
        return cands[0] if cands else None

    def _rescue_effort_for(self, target_role: str, prev_effort) -> Optional[str]:
        """救稿档位：跨家映射到目标 role 合法档位后，再降一档（避免同档复现 no_content）。"""
        ladder = self._effort_ladder_for_role(target_role)
        eff = str(prev_effort or "").strip().lower()
        if not ladder or not eff:
            return None
        mapped = _map_effort_for_fallback(eff, ladder)
        if mapped is None:
            return None
        i = ladder.index(mapped)
        return ladder[i - 1] if i > 0 else ladder[i]

    def _maybe_rescue_no_content(self, task) -> Optional[str]:
        """重试预算耗尽的救稿决策。

        返回 recovery stage value（已写入换角色 override，调用方把任务放回该阶段再试 1 次）；
        返回 None 表示放弃救稿（照旧转 PERMANENTLY_FAILED）。
        """
        cfg = self._retry_adapt_cfg()
        if not cfg.get("enabled", True) or not cfg.get("fallback_role_switch", True):
            return None
        last_error = task.get("last_error", "") or ""
        if not self._is_no_content_signature(last_error):
            return None
        chunk_id = task.get("chunk_id", "")
        ov = self._get_retry_override(chunk_id)
        if ov.get("rescue_used"):
            return None
        stage = ov.get("stage") or self._get_recovery_stage(last_error).value
        failing_role = ov.get("used_role") or ov.get("role") or ""
        rescue_role = self._rescue_role_for(stage, failing_role)
        if not rescue_role:
            return None
        prev_eff = ov.get("used_effort") or ov.get("effort")
        rescue_eff = self._rescue_effort_for(rescue_role, prev_eff)
        payload = dict(ov)
        payload.update({
            "sig": "no_content",
            "stage": stage,
            "role": rescue_role,
            "effort": rescue_eff,
            "used_role": failing_role,
            "used_effort": prev_eff,
            "rescue_used": True,
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if not self._set_retry_override(chunk_id, payload):
            # fail-safe：override 落盘失败则放弃救稿（否则 rescue_used 无法持久化 → 无限重入）
            print(f"⚠️ [Pipeline] {chunk_id} 救稿 override 落盘失败，放弃换角色（转 PF）")
            return None
        print(f"🆘 [Pipeline] {chunk_id} no_content 重试耗尽 → 换角色救稿: "
              f"{failing_role or '?'}@{prev_eff or 'default'} → {rescue_role}@{rescue_eff or 'default'}")
        return stage

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
                    self._save_final_with_sanity(chunk_id, {
                        "text": text,
                        "metadata": {
                            "fallback": True,
                            "reason": f"API_FAILURE_FALLBACK_FROM_{step.upper()}",
                            "source_step": step,
                        }
                    }, source_step="literary")
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
                # Round 133: no_content 签名 + 预算耗尽 → 换角色救稿一次（默认开启）
                rescue_stage = self._maybe_rescue_no_content(task)
                if rescue_stage:
                    stage_state = self._get_recovery_stage(f"[{rescue_stage}]")
                    retry_by_stage.setdefault(stage_state, []).append(chunk_id)
                    continue
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
                self._llm_call_counter["reference_extraction"] += 1  # Round 13
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 术语提取异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[EXTRACTING_TERMS] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.TRANSLATING_RAW)

    def _process_translating_raw_batch(self, tasks):
        success_ids = []
        # Round 96 P2: raw 翻译必须含中文. ch02 案例中 LLM 返回了英文源文 (或模型混淆),
        # 该英文稿被喂给 rewriter 造成连锁失败. 此处加 CJK 闸门, 不通过则 FAILED 触发 TRANSLATING_RAW 重试.
        try:
            raw_cjk_threshold = float(_config.pipeline.get("min_raw_cjk_ratio", 0.3))
        except (TypeError, ValueError):
            raw_cjk_threshold = 0.3
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                self._seed_pun_decisions(source_text)  # L2: pun_watchlist 命中即强制落 decision_db
                raw_text = self._run_raw_translator(source_text, chunk_id)
                self._llm_call_counter["literal_translation"] += 1  # Round 13
                # Round 96 P2: 检查 raw_text 是否真的是中文 (LLM 偶发返回英文源文)
                raw_cjk = compute_cjk_ratio(raw_text) if raw_text else 0.0
                if raw_cjk_threshold > 0 and raw_cjk < raw_cjk_threshold:
                    print(
                        f"⚠️ [Pipeline] {chunk_id} raw 翻译 CJK={raw_cjk:.3f} < "
                        f"{raw_cjk_threshold:.3f} (LLM 返回英文?), FAILED 重试"
                    )
                    # Round 96 P2: 删除 raw 中间产物, 避免污染下游 fallback
                    try:
                        (self.output_dir / f"{chunk_id}_raw.json").unlink(missing_ok=True)
                    except OSError:
                        pass
                    self.scheduler.update_task_state(
                        chunk_id,
                        TaskState.FAILED,
                        error_msg=(
                            f"[TRANSLATING_RAW] raw_not_chinese "
                            f"(cjk_ratio={raw_cjk:.3f} < {raw_cjk_threshold:.3f}, "
                            f"len={len(raw_text or '')})"
                        ),
                    )
                    continue  # 不进 success_ids
                self._save_intermediate(chunk_id, "raw", raw_text)
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 直译异常: {e}")
                if self._is_no_content_signature(e):
                    self._record_no_content_signal(task['chunk_id'], "TRANSLATING_RAW")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[TRANSLATING_RAW] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.REWRITING_LITERARY)

    def _process_rewriting_batch(self, tasks):
        success_ids = []
        entity_feedback: str = ''
        entity_registry_for_rewriter: Dict[str, str] = {}
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
                # Round 64: 默认空，由 feedback['_typesafe_verdict'] 填充
                skipped_dimensions: List[str] = []
                if reject_reason_raw:
                    # P2-1: 用 robust_json_loads 抗 critic 报告畸形；返回 {} 走原 raw 字符串兜底
                    feedback = robust_json_loads(reject_reason_raw, expected_type=dict)
                    if feedback:
                        reject_reason = feedback.get('judge_reason', '')
                        critic_feedback = feedback.get('critic_suggestions', '')
                        # 回路修复：提取 entity_consistency_notes（之前一直被丢弃，导致 Rewriter 看不到实体反馈）
                        entity_feedback = feedback.get('entity_consistency_notes', '') or ''
                        specific_edits = feedback.get("specific_edits", [])
                        low_dims = feedback.get('low_dims', [])
                        dim_scores = feedback.get('scores', {})
                        # Round 64: 从 _typesafe_verdict 提取 skipped_dimensions（避免对边界判断维度做无效重写）
                        typesafe_verdict = feedback.get('_typesafe_verdict', {}) or {}
                        skipped_dimensions = list(typesafe_verdict.get('skipped_dimensions', []) or [])
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
                # 回路修复：构建 entity_registry 注入 Rewriter（让 Rewriter 看到完整登记表以自我核对）
                entity_registry_for_rewriter = self._build_entity_registry_for_rewriter(source_text)

                # Round 73: pre-rewriter 停滞早停 — 在调用 rewriter LLM 前检查上一次 reject categories
                # 与上上一次是否完全相同 (Round 69 synonym mapping 后). 若相同, rewriter 不会再产生
                # 新效果, 直接 fallback 省一次昂贵的 LLM 调用 + 配套 critic + judge 三连击.
                # 区别于 post-rewriter stagnation (Round 12): 后者跑完一轮 LLM 才判断, 这里在 LLM 调用前拦截.
                stag_cfg_p73 = (_config.pipeline.get("oscillation_detection", {}) or {}).get("stagnation_detection", {}) or {}
                if bool(stag_cfg_p73.get("enabled", True)) and quality_retries >= 1:
                    stag_consecutive_p73 = int(stag_cfg_p73.get("consecutive_required", 1))
                    if quality_retries >= 1 + stag_consecutive_p73:
                        try:
                            prev_meta_obj = self._load_intermediate(
                                chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                            )
                            prev_reject_summary = (prev_meta_obj.get("reject_reason_summary", "") or "") if prev_meta_obj else ""
                            if prev_reject_summary:
                                prev_prev_meta_obj = self._load_intermediate(
                                    chunk_id, f"rewrite_meta_v{quality_retries - 1 - stag_consecutive_p73}"
                                )
                                prev_prev_summary = (prev_prev_meta_obj.get("reject_reason_summary", "") or "") if prev_prev_meta_obj else ""
                                if prev_prev_summary:
                                    prev_cats = self._extract_reject_categories(prev_prev_summary)
                                    curr_cats = self._extract_reject_categories(prev_reject_summary)
                                    if prev_cats and prev_cats == curr_cats:
                                        # Round 92: 用跨 retry 最佳文学稿 (而非当前被拒的 prev_lit_text)
                                        candidate = prev_lit_text or self._load_intermediate(chunk_id, "literary") or ""
                                        final_text = self._get_best_lit(chunk_id, candidate)
                                        self._save_final_with_sanity(chunk_id, {
                                            "text": final_text,
                                            "metadata": {
                                                "fallback": True,
                                                "reason": "PRE_REWRITER_STAGNATION",
                                                "judge_reason": prev_reject_summary[:200],
                                                "quality_retries": quality_retries,
                                                "stagnant_categories": sorted(prev_cats),
                                            }
                                        }, source_step="literary")
                                        self.scheduler.update_task_state(
                                            chunk_id, TaskState.PERMANENTLY_FAILED,
                                            error_msg=f"[PRE_REWRITER_STAGNATION] 跳过 rewriter LLM: cats {sorted(prev_cats)} 连续 {stag_consecutive_p73 + 1} 次不变"
                                        )
                                        print(f"⏹️ [Pipeline] {chunk_id} pre-rewriter 停滞 (retry={quality_retries}, cats={sorted(prev_cats)}), 跳过 rewriter LLM")
                                        continue
                        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                            # meta 缺失 (首次 retry), 正常走 rewriter
                            pass

                # Round 65: 重试计入实际 LLM 调用次数（之前 +=1 漏算 retry，导致 cost_report 严重低估 literary_rewriting 成本）
                self._llm_call_counter["literary_rewriting"] += 1 + quality_retries
                # Round 116: 停滞换模型（生成端多样性）— 决定本 chunk 本轮润色所用角色
                sw_cfg = _config.pipeline.get("stagnation_model_switch", {}) or {}
                sw_enabled = bool(sw_cfg.get("enabled", False))
                sw_base_role = sw_cfg.get("base_role") or _config.get_task_role("literary_rewrite")
                sw_alt_role = sw_cfg.get("alternate_role") or ""
                sw_state = None
                model_role = sw_base_role
                if sw_enabled and sw_alt_role:
                    sw_state = self._read_rewriter_switch_state(
                        chunk_id, sw_base_role, sw_alt_role, sw_cfg.get("max_switches", 1)
                    )
                    model_role = sw_state.get("current_role") or sw_base_role
                # Round 129: 换到 alternate（更强模型）时加档 effort
                model_effort = (sw_cfg.get("alternate_effort") if model_role != sw_base_role else None)
                # Round 133: no_content 重试适配 override（降档/换角救稿）优先于停滞换档；
                # 降档后若因质量重进润色（quality_retries≥1）→ 升档补偿一次（防死循环）。
                ov = self._get_retry_override(chunk_id)
                if ov and ov.get("sig") == "no_content" and ov.get("stage") == "REWRITING_LITERARY":
                    if ov.get("role"):
                        model_role = ov["role"]
                    if ov.get("effort"):
                        model_effort = str(ov["effort"]).strip().lower()
                    raccfg = self._retry_adapt_cfg()
                    if (raccfg.get("quality_escalate_back", True)
                            and not ov.get("escalated")
                            and model_effort
                            and quality_retries >= 1
                            and int(ov.get("down_delta", 0) or 0) > 0):
                        up_eff = self._escalate_one_step(model_role, model_effort)
                        if up_eff and up_eff != model_effort:
                            print(
                                f"🔼 [Pipeline] {chunk_id} 降档后质量重进润色 → 升档补偿一次 "
                                f"{model_effort}→{up_eff} (role={model_role})"
                            )
                            model_effort = up_eff
                            ov["escalated"] = True
                            ov["effort"] = up_eff
                            self._set_retry_override(chunk_id, ov)
                    if model_effort:
                        print(
                            f"🔻 [Pipeline] {chunk_id} 重试适配生效: role={model_role} "
                            f"effort={model_effort}"
                        )
                self._note_last_call(
                    chunk_id, "REWRITING_LITERARY", model_role,
                    model_effort or self._task_default_effort("literary_rewrite"),
                    get_role_model_name(model_role),
                )
                raw_result = self.rewriter_agent.process_chunk(
                    chunk_id, raw_text, self.style_guide, source_text,
                    prev_lit_text=prev_lit_text,
                    reject_reason=reject_reason,
                    critic_feedback=critic_feedback,
                    entity_feedback=entity_feedback,        # 回路修复
                    entity_registry=entity_registry_for_rewriter,  # 回路修复
                    retry_count=quality_retries,
                    all_feedback=all_feedback,
                    specific_edits=specific_edits,
                    narrative_memory=narrative_memory,
                    low_dims=low_dims,
                    skipped_dimensions=skipped_dimensions,  # Round 64
                    verse_note=self._verse_note_for(chunk_id, source_text),  # Round 119 L1
                    model_role=model_role,  # Round 116: 停滞换模型
                    model_effort=model_effort,  # Round 129: 换模型加档
                )
                # process_chunk 返回 (lit_text, self_review) 元组
                if isinstance(raw_result, tuple):
                    lit_text, self_review = raw_result
                else:
                    lit_text, self_review = raw_result, None  # 向后兼容旧返回值
                self._save_intermediate(chunk_id, "literary", lit_text)

                # L3: 高价值片段（verse/pun-dense）双稿仲裁 — 仅首轮，每 chunk 一次
                if quality_retries == 0 and lit_text:
                    try:
                        lit_text = self._dual_draft_run(
                            chunk_id, source_text, raw_text, prev_lit_text,
                            reject_reason, critic_feedback, entity_feedback,
                            all_feedback, specific_edits, narrative_memory,
                            low_dims, skipped_dimensions, lit_text,
                        )
                        self._save_intermediate(chunk_id, "literary", lit_text)
                    except Exception as _dd_exc:
                        print(f"⚠️ [Pipeline] {chunk_id} 双稿仲裁异常，保留首稿: {_dd_exc}", file=sys.stderr)

                # Round 36: rewriter 字节级停滞检测 — 当 retry 输出与 prev_lit_text 完全相同时，
                # 说明 LLM 没消化任何反馈，继续 retry 只是浪费 token。直接走 fallback。
                # （区别于 Round 6 oscillation：基于相似度阈值；此处为精确字节比较，
                #  两者并行运行，任一触发即 fallback。）
                if prev_lit_text and lit_text == prev_lit_text and quality_retries >= 1:
                    print(f"⏹️ [Pipeline] {chunk_id} rewriter 输出与 prev_lit_text 完全相同 (retry={quality_retries})，触发 REWRITER_STAGNATED 早停")
                    stag_meta = {'retry_count': quality_retries, 'quality_retries': quality_retries, 'reject_reason_summary': (reject_reason or '')[:200], 'critic_feedback_summary': (critic_feedback or '')[:200], 'all_feedback_summaries': all_feedback, 'self_review': self_review, 'stagnation': 'REWRITER_STAGNATED', 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
                    # Round 116: 字节停滞优先换模型再试一轮；达上限后走原 fallback
                    if sw_enabled and sw_alt_role and sw_state and sw_state.get("can_switch"):
                        switch_to = sw_state.get("next_role") or sw_alt_role
                        self._record_rewriter_model_switch(
                            chunk_id, model_role, switch_to, quality_retries, "REWRITER_STAGNATED"
                        )
                        stag_meta["model_role"] = model_role
                        stag_meta["model_switch_to"] = switch_to
                        self._save_intermediate(chunk_id, f'rewrite_meta_v{quality_retries}', stag_meta)
                        print(f"🔁 [Pipeline] {chunk_id} 字节停滞 retry={quality_retries}：换模型 {model_role} → {switch_to} 再试一轮")
                        continue
                    self._save_intermediate(chunk_id, f'rewrite_meta_v{quality_retries}', stag_meta)
                    self._save_final_with_sanity(chunk_id, {
                        "text": lit_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "REWRITER_STAGNATED",
                            "judge_reason": f"rewriter 在 retry={quality_retries} 时输出与 prev_lit_text 完全相同，未消化任何反馈",
                            "quality_retries": quality_retries,
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=f"rewriter 字节停滞 retry={quality_retries}"
                    )
                    continue

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

                # Round 6: oscillation 早停检测（similarity-based）
                # Round 12: stagnation 早停检测（reject_reason-based）——并行运行，任一触发即 fallback
                osc_cfg = _config.pipeline.get("oscillation_detection", {}) or {}
                osc_enabled = bool(osc_cfg.get("enabled", True))
                osc_sim_threshold = float(osc_cfg.get("similarity_threshold", 0.95))
                osc_consecutive = int(osc_cfg.get("consecutive_required", 2))
                stag_cfg = osc_cfg.get("stagnation_detection", {}) or {}
                stag_enabled = bool(stag_cfg.get("enabled", True))
                stag_consecutive = int(stag_cfg.get("consecutive_required", 2))
                stag_exact_only = bool(stag_cfg.get("exact_match_only", True))

                is_oscillation = False
                osc_sim = 0.0
                if osc_enabled and prev_lit_text and quality_retries >= 1:
                    osc_sim = self._compute_text_similarity(prev_lit_text, lit_text)
                    if osc_sim >= osc_sim_threshold:
                        is_oscillation = True

                # Round 12 stagnation 检测（reject_reason-based）
                # Round 16: 默认用 primitive-category 集合对比（Jev-aware）
                #   - 旧 exact_match_only=true 永远不触发（Jev noul 每次浮动）
                #   - match_mode=category 比较 primitive 集合（"实体一致性违反" vs "noul=0.33"）
                #   - match_mode=exact 保留原字符串严格匹配
                #   - match_mode=substring 保留旧"包含关系"宽松匹配
                current_reject_summary = (reject_reason or "")[:200]
                is_stagnation = False
                match_mode = stag_cfg.get("match_mode")  # Round 16 新增
                if match_mode is None:
                    # 向后兼容：未指定 match_mode 时按 exact_match_only 推断
                    match_mode = "exact" if stag_exact_only else "substring"
                if stag_enabled and quality_retries >= 1 and current_reject_summary:
                    try:
                        prev_meta_obj = self._load_intermediate(
                            chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                        )
                        prev_reject_summary = prev_meta_obj.get("reject_reason_summary", "") or ""
                        if match_mode == "category":
                            # Round 16: 比较 primitive category 集合（忽略 noul 浮动）
                            prev_cats = self._extract_reject_categories(prev_reject_summary)
                            curr_cats = self._extract_reject_categories(current_reject_summary)
                            is_stagnation = bool(prev_cats) and prev_cats == curr_cats
                        elif match_mode == "exact":
                            is_stagnation = (prev_reject_summary == current_reject_summary)
                        else:  # "substring"
                            is_stagnation = (
                                prev_reject_summary in current_reject_summary
                                or current_reject_summary in prev_reject_summary
                            ) if prev_reject_summary else False
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                        pass

                # 把 oscillation / stagnation 信息写进 meta，便于下一 retry 读取
                meta["oscillation"] = is_oscillation
                meta["oscillation_similarity"] = round(osc_sim, 4) if osc_sim else 0.0
                meta["oscillation_threshold"] = osc_sim_threshold
                meta["stagnation"] = is_stagnation
                meta["reject_reason_summary"] = current_reject_summary
                self._save_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}", meta)

                # 触发条件：连续 oscillation / stagnation 任一满足 → 强制 fallback
                # 语义：consecutive_required=N 表示当前 + 前 N 次都需一致（总共 N+1 个一致信号）
                # Round 12 默认 N=1：当前 + 前 1 次 stagnation = 2 个一致信号即触发（更激进，省 LLM）
                should_force_fallback = False
                force_reason = ""
                if is_oscillation and quality_retries >= osc_consecutive:
                    try:
                        prev_meta_obj = self._load_intermediate(
                            chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                        )
                        # 累计 oscillation 计数：当前 + 前 N-1 次 prev_meta 是否都 oscillation
                        osc_count = 1  # 当前
                        cur_meta = prev_meta_obj
                        for _ in range(osc_consecutive - 1):
                            if not cur_meta or not bool(cur_meta.get("oscillation", False)):
                                osc_count = 0
                                break
                            try:
                                nxt_meta = self._load_intermediate(
                                    chunk_id, f"rewrite_meta_v{quality_retries - 2 - _}"
                                )
                                cur_meta = nxt_meta
                                osc_count += 1
                            except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                                osc_count = 0
                                break
                        if osc_count >= osc_consecutive:
                            should_force_fallback = True
                            force_reason = (
                                f"连续{osc_consecutive + 1}次 retry 相似度"
                                f"≥{osc_sim_threshold:.0%}（sim={osc_sim:.4f}）"
                            )
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                        pass
                if not should_force_fallback and is_stagnation and quality_retries >= stag_consecutive:
                    try:
                        prev_meta_obj = self._load_intermediate(
                            chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                        )
                        stag_count = 1  # 当前
                        cur_meta = prev_meta_obj
                        for _ in range(stag_consecutive - 1):
                            if not cur_meta or not bool(cur_meta.get("stagnation", False)):
                                stag_count = 0
                                break
                            try:
                                nxt_meta = self._load_intermediate(
                                    chunk_id, f"rewrite_meta_v{quality_retries - 2 - _}"
                                )
                                cur_meta = nxt_meta
                                stag_count += 1
                            except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                                stag_count = 0
                                break
                        if stag_count >= stag_consecutive:
                            should_force_fallback = True
                            force_reason = (
                                f"连续{stag_consecutive + 1}次 retry reject_reason 未变（Rewriter 没修）"
                            )
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                        pass

                if should_force_fallback:
                    print(
                        f"⏹️ [Pipeline] {chunk_id} retry={quality_retries} 强制 fallback：{force_reason}"
                    )
                    # Round 116: 强制 fallback 前优先换模型再试一轮
                    if sw_enabled and sw_alt_role and sw_state and sw_state.get("can_switch"):
                        switch_to = sw_state.get("next_role") or sw_alt_role
                        self._record_rewriter_model_switch(
                            chunk_id, model_role, switch_to, quality_retries,
                            f"STAGNATION:{force_reason}"
                        )
                        switch_meta = {
                            "retry_count": quality_retries,
                            "quality_retries": quality_retries,
                            "reject_reason_summary": current_reject_summary,
                            "stagnation": "STAGNATION_MODEL_SWITCH",
                            "model_role": model_role,
                            "model_switch_to": switch_to,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        self._save_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}", switch_meta)
                        print(f"🔁 [Pipeline] {chunk_id} 强制 fallback 前换模型 {model_role} → {switch_to} 再试：{force_reason}")
                        continue
                    # Round 92: fallback 写入跨 retry 最佳版本, 而非当前被拒 lit_text
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text)
                    self._save_final_with_sanity(chunk_id, best_text)
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=f"[STAGNATION] {force_reason}，提前 fallback",
                    )
                    continue  # 跳过 success_ids，不再进 AUDITING

                success_ids.append(chunk_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ [Pipeline] {task['chunk_id']} 润色异常: {e}")
                if self._is_no_content_signature(e):
                    self._record_no_content_signal(task['chunk_id'], "REWRITING_LITERARY")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[REWRITING_LITERARY] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.AUDITING)

    @staticmethod
    def _strip_parenthetical(name: str) -> str:
        """Round 105: 剥离规范名中的括注（全角/半角），返回纯名。

        例: 镇（town）→镇 / 科德斯（Codders）→科德斯 / 卡特拉姆勋爵（第九代…）→卡特拉姆勋爵
        """
        if not name:
            return name
        import re as _re_p
        return _re_p.sub(r"（[^）]*）|\([^)]*\)", "", name).strip()

    def _canonical_present_in_text(self, canonical: str, lit_text: str) -> bool:
        """Round 105: 判定规范名（或其显著部分）是否出现在译文中。

        用于 missing 核查，消除两类系统性假阳性（ch32 倒查根因）：
          1. 规范名带括注（镇（town）/科德斯（Codders）/卡特拉姆勋爵（第九代…））
             — 译文永远不会含括注，整串匹配必然 missing。
          2. 前缀/后缀写法（译文用 乔治·洛马克斯，规范名 乔治·洛马克斯先生）
             — 同一人不同形式，不应判硬 missing（会触发 rewriter 互斥指令死循环）。

        判定为"已出现"（返回 True）的条件（任一）：
          - 完整规范名在 lit_text；
          - 去括注后的 base（len≥2）在 lit_text；
          - base 的显著前缀或后缀（长度 ≥ max(3, 2/3·len)）在 lit_text；
          - base 长度 < 2（如 镇（town）→镇）：单字名无法可靠核验，视为已出现
            （抑制 missing，避免对单字名死循环）。
        """
        if not lit_text or not canonical:
            return False
        if canonical in lit_text:
            return True
        base = self._strip_parenthetical(canonical)
        if not base:
            return False
        if len(base) < 2:
            # 单字 base（镇/城/国…）无法可靠核验，抑制 missing 假阳性
            return True
        if base != canonical and base in lit_text:
            return True
        # Round 115: 斜杠备选 canonical（如 "国民党/民族主义者"）— 译文通常只用其中
        # 一个备选（ch06 chunk012 用「民族主义者」），整串/前后缀容差都会差在斜杠上
        # → 误判 missing → 实体硬失败死循环。任一备选字面命中即算在场。
        if "/" in base or "／" in base:
            for alt in re.split(r"[/／]", base):
                alt = alt.strip()
                if len(alt) >= 2 and alt in lit_text:
                    return True
        ref = base
        if len(ref) < 4:
            # 短名（2-3 字）不做前缀/后缀容差，避免误判（如 黛西 的前缀 黛 太短）
            return ref in lit_text
        threshold = max(3, (len(ref) * 2) // 3)
        if threshold >= len(ref):
            return ref in lit_text
        prefix = ref[:threshold]
        suffix = ref[-threshold:]
        return prefix in lit_text or suffix in lit_text

    # ch06 倒查：reference agent 把非实体短语登记为 source_name（sir/his eyelids/
    # the support/a sharp command），导致实体一致性核查误报 missing → 硬失败死循环。
    _NON_ENTITY_SOURCE_NAMES = {
        "sir", "madam", "mister", "miss", "mr", "mrs", "ms",
        "the man", "the woman", "the fellow", "the other",
        "a man", "a woman", "a gentleman", "a lady",
        # Round 106: ch05 倒查补充的泛称/描述（reference agent 误登记）
        "the lady", "the gentleman", "the young man", "the old man",
        "an englishwoman", "the englishwoman", "the local castle",
        "he (implied)", "she (implied)", "it (implied)",
        # Round 107: ch03 倒查补充 — 代词/反身代词/描述性短语
        "himself", "herself", "itself", "themselves",
        "he/his/him", "she/her/her", "it/its",
        "my boy", "my man", "my friend", "my old friend",
        "an old friend of yours", "a friend of yours", "an old friend",
        "this young fellow", "that young fellow", "this old fellow",
        "this man", "that man", "this woman", "that woman",
    }
    _NON_ENTITY_POSSESSIVE_NOUNS = {
        "eyelids", "eyelid", "impressions", "impression",
        "face", "eyes", "hands", "voice", "head",
        # Round 106: 关系名词（his/her + 关系 = 指代另一人，非本实体专名）
        "wife", "husband", "brother", "sister", "son", "daughter",
        "mother", "father", "parent", "child", "children",
        "friend", "friends", "colleague", "colleagues",
        # Round 107: ch03 倒查补充
        "boy", "man", "fellow", "gentleman", "lady",
    }
    # Round 106: 动词短语（代词主语 + 情态/助动词/常见动词）— 明显非实体指称
    # （ch05 实测 "I could"/"I should"/"I had" 被登记为女主人公 source_name → 误报 missing）
    _NON_ENTITY_VERB_RE = re.compile(
        r"^(i|he|she|it|we|they)\s+"
        r"(could|should|would|might|may|must|can|will|shall|had|has|have|did|do|does|"
        r"is|am|are|was|were|said|told|asked|looked|turned|stood|sat|walked|ran|"
        r"came|went|felt|knew|thought|saw|heard|made|got|took|gave|put|let|set|"
        r"began|started|tried|kept|left|meant|happened|seemed|appeared|proved)\b"
    )
    # Round 107: 代词/反身代词（含斜杠分隔的多代词，如 "he/his/him"）
    _NON_ENTITY_PRONOUN_RE = re.compile(
        r"^(i|you|he|she|it|we|they|me|him|her|us|them|my|your|his|her|its|our|their|"
        r"mine|yours|hers|ours|theirs|myself|yourself|himself|herself|itself|ourselves|"
        r"themselves)(/(\w+))*$"
    )
    # Round 107: 描述性短语（冠词 + 形容词 + 名词 + of + 代词，如 "an old friend of yours"）
    _NON_ENTITY_DESCR_RE = re.compile(
        r"^(a|an|the)\s+\w+\s+\w+\s+of\s+(my|your|his|her|its|our|their|yours|his|hers)$"
    )
    # Round 107: this/that + 名词短语（2-3 词, 如 "this country", "this young fellow"）
    # Round 107 P8: 扩展到 2 词 (this/that + 单个普通名词), ch07 倒查 "this country" 误提取.
    _NON_ENTITY_THIS_THAT_RE = re.compile(
        r"^(this|that)\s+\w+(\s+\w+)?$"
    )
    # Round 107 P3: like + 名词短语（如 "Like all women"）— 比较/类比短语, 非实体指称
    _NON_ENTITY_LIKE_RE = re.compile(
        r"^like\s+\w+(\s+\w+){0,3}$"
    )
    # Round 111: 限定词 + 全小写普通名词短语（如 "the stairs"/"the chauffeur"/"my maid"/
    # "the unemployed young man"）— 叙事中的泛指描述，非专名。
    # ch09 chunk004 根因: reference agent 把 "the stairs"→楼梯 登记为实体后，实体核查按
    # "规范主名必须字面出现" 判 missing → 硬失败退回重写；而中文习惯动词化表达（上楼/下楼），
    # rewriter 无法两全 → 输出字节停滞 → REWRITER_STAGNATED → PERMANENTLY_FAILED。
    # 判定基于原文大小写（大小写敏感）：真专名首字母大写（the Comrades/the King/
    # the Tower），不受影响；仅当限定词后的所有词全小写时视为普通名词短语。
    _NON_ENTITY_ARTICLE_NOUN_RE = re.compile(
        r"^(?:The|the|A|a|An|an|My|my|Your|your|His|his|Her|her|Its|its|Our|our|Their|their)\s+"
        r"[a-z][a-z'\-]*(\s+[a-z][a-z'\-]*){0,3}$"
    )
    # Round 113: 主语代词开头的叙事从句（如 "I wanted to know"/"I haven't written"/
    # "he (指代 Professor Wynward)"）— 是叙事文本，不是实体指称。
    # ch16 chunk006 根因: reference agent 把叙事从句登记为 "女主人公（未具名叙述者）"
    # 的 source_name；Round 106 的动词白名单是有限枚举（"wanted/haven't written"
    # 不在表内）被再次击穿 → 核查要求描述性主名字面出现 → 不可满足 → 实体硬失败 →
    # rewriter 停滞 → PERMANENTLY_FAILED。主语代词 + 任何后续内容一律视为从句。
    _NON_ENTITY_CLAUSE_RE = re.compile(
        r"^(i|you|he|she|it|we|they)(['’][a-z]+)?\s+\S"
    )
    # Round 113: 描述性占位 canonical（无名角色的描述串，如
    # "女主人公（未具名叙述者）"/"美国绅士（未具名）"）— 译文永远不会字面出现整串，
    # 核查必然判 missing → 硬失败死循环；这类实体没有可核查的专名，不进入核查/审计/
    # registry（与 Round 111/112 同语义：防不可满足的硬约束）。
    _NON_ENFORCEABLE_CANONICAL_RE = re.compile(
        r"未具名|无具名|无名|匿名|未给出姓名|隐含"
    )

    @classmethod
    def _is_valid_entity_source_name(cls, sn: str) -> bool:
        """判定 source_name 是否为有效实体指称（而非泛称/身体部位/普通短语）。"""
        if not sn:
            return False
        low = sn.strip().lower()
        if not low:
            return False
        if low in cls._NON_ENTITY_SOURCE_NAMES:
            return False
        m = re.match(r"^(his|her|its|their)\s+(\w+)$", low)
        if m and m.group(2) in cls._NON_ENTITY_POSSESSIVE_NOUNS:
            return False
        # Round 106: 动词短语（代词主语 + 动词）
        if cls._NON_ENTITY_VERB_RE.match(low):
            return False
        # Round 113: 主语代词 + 后续内容的从句（泛化 Round 106 的有限动词白名单，
        # 覆盖 "I wanted to know"/"I haven't written" 等叙事片段）
        if cls._NON_ENTITY_CLAUSE_RE.match(low):
            return False
        # Round 107: 代词/反身代词（含斜杠分隔的多代词）
        if cls._NON_ENTITY_PRONOUN_RE.match(low):
            return False
        # Round 107: 描述性短语（冠词 + 形容词 + 名词 + of + 代词）
        if cls._NON_ENTITY_DESCR_RE.match(low):
            return False
        # Round 107: this/that + 形容词 + 名词
        if cls._NON_ENTITY_THIS_THAT_RE.match(low):
            return False
        # Round 107 P3: like + 名词短语 (类比/比较, 非实体指称)
        if cls._NON_ENTITY_LIKE_RE.match(low):
            return False
        # Round 111: 限定词 + 全小写普通名词短语（用原文大小写判定, 不能走 low）
        if cls._NON_ENTITY_ARTICLE_NOUN_RE.match(sn.strip()):
            return False
        return True

    @classmethod
    def _is_enforceable_canonical(cls, canonical: str, source_names: list) -> bool:
        """Round 112: 判定 canonical 是否可作实体核查基准（防"未翻译 canonical"死循环）。

        ch10 chunk007 根因: 登记表 canonical=" Cloak Room"（ASCII + 前导空格，与
        source_name "Cloak Room" 规范化后恒等）→ 核查要求中文译文出现英文串，
        而正常译法是「行李寄存票/衣帽间」→ 不可满足 → 实体硬失败退回重写 →
        rewriter 字节停滞 → REWRITER_STAGNATED → PERMANENTLY_FAILED。

        判定: canonical 无 CJK 且与某个 source_name 规范化（折叠空白+小写）后相等
        → 身份映射（reference agent 未给出中文译名），不含翻译信息，不进入核查/
        registry（Morris Cowley 等"保留英文原名"场景由译文自然处理，无需强约束）。

        Round 113: 描述性占位 canonical（未具名/无名/匿名 等标记）同样剔除 —
        ch16 chunk006 "女主人公（未具名叙述者）" 根因（字面要求不可满足）。
        """
        if not canonical:
            return False
        cs = re.sub(r"\s+", " ", canonical).strip()
        if not cs:
            return False
        if cls._NON_ENFORCEABLE_CANONICAL_RE.search(cs):
            return False
        if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", cs):
            return True
        norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().lower()
        if any(norm(sn) == norm(cs) for sn in (source_names or []) if sn):
            return False
        return True

    def _find_near_variant(self, canonical: str, lit_text: str) -> Optional[str]:
        """Round 108: 检测 lit_text 中与规范名一字之差的近变体（译名漂移）。

        ch07 chunk016 根因：规范名「艾琳·布伦特夫人」，译文写成「伊琳·布伦特夫人」
        （一字之差）。本地核查的后缀容差（「特夫人」）抑制了 missing，已登记变体表
        又为空 → 本地核查零报告，只有 Jev 能发现，而拒因不可执行 → rewriter 停滞。
        这里主动检测一字之差近变体，报 variant_instead_of_canonical（可执行）。

        只处理含分隔符（·）的规范名（如 艾琳·布伦特夫人）：按 CJK 段扫描一字之差
        串，重构后整串须出现在 lit_text（强精度守卫）。
        纯 CJK 规范名不做检测（ch07 chunk007 实测：「敲诈者」与动词短语「敲诈谁/
        敲诈勒」一字之差碰撞，CJK 无空格无法可靠分词，误报会诱导 rewriter 改坏正文）。

        Returns:
            找到近变体串则返回，否则 None。
        """
        if not lit_text or not canonical:
            return None
        base = self._strip_parenthetical(canonical)
        if not base or len(base) < 3:
            return None
        try:
            from src.utils.entity_stopwords import is_plausible_name_variant
        except Exception:
            is_plausible_name_variant = None

        def _one_char_diff(a: str, b: str) -> bool:
            if len(a) != len(b):
                return False
            return sum(1 for x, y in zip(a, b) if x != y) == 1

        def _plausible(ref: str, cand: str) -> bool:
            if is_plausible_name_variant is None:
                return True
            try:
                ok, _reason = is_plausible_name_variant(ref, cand)
                return bool(ok)
            except Exception:
                return True

        # 只处理含分隔符的规范名（如 艾琳·布伦特夫人）— 分隔符提供天然 token 边界，
        # 重构守卫（整串须出现在 lit_text）保证精度。纯 CJK 规范名不做检测（见 docstring）。
        if re.fullmatch(r'[\u4e00-\u9fff]+', base):
            return None
        segments = [s for s in re.split(r'[^\u4e00-\u9fff]+', base) if len(s) >= 2]
        if not segments:
            return None
        for seg in segments:
            length = len(seg)
            cands = set(re.findall(r'(?=([\u4e00-\u9fff]{%d}))' % length, lit_text))
            cands = {c for c in cands if c != seg and _one_char_diff(seg, c) and _plausible(seg, c)}
            if not cands:
                continue
            # 尝试 base 中 seg 的每个出现位置
            start = 0
            while True:
                idx = base.find(seg, start)
                if idx < 0:
                    break
                for cand in cands:
                    near = base[:idx] + cand + base[idx + length:]
                    if near in lit_text:
                        return near
                start = idx + 1
        return None

    def _get_sibling_canonicals(self) -> Dict[str, Set[str]]:
        """Round 71/133b: 全 registry 的 source_name → {canonical} 映射（含 pending 行）。

        lazy-init + 按审计批次缓存（_process_auditing_batch 入口重置 _sibling_canonicals_cache）。
        兼容测试桩：测试用 __new__ 构造的实例也可用（hasattr 兜底）。
        """
        if not hasattr(self, "_sibling_canonicals_cache") or \
                self._sibling_canonicals_cache is None:
            try:
                self._sibling_canonicals_cache = get_sibling_canonicals_by_source(
                    self.decision_engine.conn
                )
            except Exception:
                self._sibling_canonicals_cache = {}
        return self._sibling_canonicals_cache or {}

    def _partition_swallowed_source_names(
        self, source_text: str, canonical: str, valid_sns: List[str],
    ) -> Tuple[List[str], Set[str]]:
        """Round 133b: 别名吞并抑制 — 剔除"被其它实体更长别名吞并"的伪命中。

        chunk071 根因: entity_registry 里 "Mother" 被同时登记给 杰西卡/杰西卡夫人，
        而 chunk071 原文中 "Mother" 唯一出现于 "Reverend Mother"（盖乌斯·海伦·莫希阿姆
        的别名）内部 → Round 114 词边界防护放行（"Mother" 是合法词）→ 杰西卡夫人被
        判"原文在场但译文缺失" → rewriter 收到不可执行指令（"Reverend Mother" 那处
        必须出现「杰西卡夫人」）→ 改坏正文 → NO_PROGRESS_STREAK → PERMANENTLY_FAILED。

        判定: 别名 a 的所有命中 span 都被某"更长注册别名 b"(len(b) > len(a), a 是 b 的
        子串, b 归属某 canonical ≠ 当前 canonical) 的命中 span 包含 → a 为伪命中, 剔除;
        只要一处未被吞并即保留（真在场）。

        Returns:
            (effective, swallowed): effective=真在场别名（按原顺序）;
            swallowed=出现在原文但被吞并的别名集合。
        """
        if not valid_sns:
            return [], set()
        siblings = self._get_sibling_canonicals()
        own_spans_cache: Dict[str, List[Tuple[int, int]]] = {}
        coverer_spans_cache: Dict[str, List[Tuple[int, int]]] = {}

        effective: List[str] = []
        swallowed: Set[str] = set()
        for sn in valid_sns:
            if not sn:
                continue
            if sn not in own_spans_cache:
                own_spans_cache[sn] = entity_source_name_spans(sn, source_text)
            spans = own_spans_cache[sn]
            if not spans:
                continue
            # 候选覆盖别名: 更长 + 包含 sn + 归属于≥1 个其它 canonical
            coverers = [
                other for other in siblings.keys()
                if len(other) > len(sn) and sn in other
                and any(oc != canonical for oc in (siblings.get(other) or set()))
                and self._is_valid_entity_source_name(other)
            ]
            fully_swallowed = False
            if coverers:
                cov_spans: List[Tuple[int, int]] = []
                for other in coverers:
                    if other not in coverer_spans_cache:
                        coverer_spans_cache[other] = entity_source_name_spans(other, source_text)
                    cov_spans.extend(coverer_spans_cache[other])
                fully_swallowed = all(
                    any(cs <= s and e <= ce for (cs, ce) in cov_spans)
                    for (s, e) in spans
                )
            if fully_swallowed:
                swallowed.add(sn)
            else:
                effective.append(sn)
        return effective, swallowed

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

        # Round 71: 跨人物冲突抑制 — 同一英文 source_name 被登记到多个 canonical 时,
        # 若 lit_text 中出现任何一个 sibling canonical (共享 source_name 的其它译名),
        # 则视为 rewriter 用了别名译名, 不报 "missing". variant_instead_of_canonical
        # 不受影响 (variant 是真问题, sibling 命中不影响 variant 路径).
        # 缓存: 单次审计内多次调用 _check_entity_consistency 时复用同一份.
        sibling_canonicals = self._get_sibling_canonicals()

        inconsistencies = []
        for row in entities:
            canonical = row[0]
            source_names_json = row[1]
            source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
            variant_names = robust_json_loads(row[2], expected_type=list) if len(row) > 2 and row[2] else []

            if len(canonical) < 2:
                continue
            # Round 112: 身份映射（canonical ≡ source_name 且无 CJK）不进入核查 —
            # ch10 chunk007 " Cloak Room" 根因（不可满足的英文主名要求）
            if not self._is_enforceable_canonical(canonical, source_names):
                continue

            # B2 修复：只检查本章原文出现的实体（按 source_text 过滤），
            # 避免全库实体每次 AUDITING 都被遍历、variant 误发现累积、
            # quality_retries 无限增长。修前 ch32 quality_retries=112+ 死循环。
            # ch06: 过滤非实体 source_name（sir/his eyelids 等），避免误判"实体出场"
            valid_source_names = [sn for sn in source_names if self._is_valid_entity_source_name(sn)]
            # Round 133b: 别名吞并抑制 — 剔除被其它实体更长别名吞并的伪命中
            # （chunk071 "Mother"⊂"Reverend Mother" 根因, 详见 _partition_swallowed_source_names）
            effective_source_names, _swallowed = self._partition_swallowed_source_names(
                source_text, canonical, valid_source_names,
            )
            appears_in_source = any(
                (sn and entity_source_name_in_text(sn, source_text))
                for sn in effective_source_names
            )
            if not appears_in_source:
                continue

            # Round 21: 收集所有出现在 lit_text 中的 variant（不只第一个）
            matched_variants = []  # [(variant_found_in_lit, source_name_matched)]
            for vn in variant_names:
                found_form = self._case_insensitive_find(vn, lit_text)
                if found_form:
                    # 反查该 variant 在原文中以哪种 source_name 形式出现
                    matched_src = None
                    for sn in source_names:
                        if self._case_insensitive_find(sn, source_text):
                            matched_src = sn
                            break
                    matched_variants.append((found_form, matched_src))

            # Round 21: 收集所有出现在 lit_text 中的 source_name
            matched_source_names = []
            for sn in source_names:
                found_form = self._case_insensitive_find(sn, lit_text)
                if found_form:
                    matched_source_names.append(found_form)

            # 去重：source_name 命中后也算 variant_instead_of_canonical（英文残留）
            all_violations = list(matched_variants)
            for sn in matched_source_names:
                if not any(sn == src for _, src in matched_variants):
                    all_violations.append((sn, sn))

            near_variant = None
            if all_violations:
                # Round 21: 为每一个违规单独报告一条 inconsistency（rewriter 一次性全部修）
                for found_val, src_name in all_violations:
                    inconsistencies.append({
                        "entity": canonical,
                        "expected": canonical,
                        "found": found_val,
                        "source_name": src_name,
                        "type": "variant_instead_of_canonical",
                        "chunk_id": chunk_id,
                    })
                # 只要有违规就 continue，不再走 missing 分支（variant 已表明该实体被引用）
                # 但仍要 try 发现新变体（_maybe_discover_entity_variant 可能在 variant_names 已包含的 case 误触发）
            else:
                # Round 108: 一字之差近变体检测（译名漂移，如 伊琳·布伦特夫人 vs 艾琳·布伦特夫人）。
                # 比 missing 更可执行：给 rewriter 具体的 found→expected 替换清单。
                near_variant = self._find_near_variant(canonical, lit_text)
                if near_variant:
                    matched_source = None
                    for sn in effective_source_names:
                        if entity_source_name_in_text(sn, source_text):
                            matched_source = sn
                            break
                    inconsistencies.append({
                        "entity": canonical,
                        "expected": canonical,
                        "found": near_variant,
                        "source_name": matched_source,
                        "type": "variant_instead_of_canonical",
                        "chunk_id": chunk_id,
                    })
            if not all_violations and not near_variant and not self._canonical_present_in_text(canonical, lit_text):
                # 只有原文中确实出现了该实体时才报 missing
                # 避免"全书角色本章未出场"的假阳性
                matched_source = None
                for sn in effective_source_names:
                    if entity_source_name_in_text(sn, source_text):
                        matched_source = sn
                        break
                if matched_source:
                    # Round 71: 跨人物冲突抑制 — 若 lit_text 中已经出现共享 source_name 的
                    # 其它 canonical, 说明 rewriter 用了别名译名, 不算 missing.
                    # 例: canonical="乔治·洛马克斯先生", source_name="Mr. Lomax" 在源文,
                    #      lit_text 含 "洛马克斯先生" (同 source_name 的 sibling canonical).
                    siblings = sibling_canonicals.get(matched_source) or set()
                    if siblings and any(sib != canonical and sib in lit_text for sib in siblings):
                        # sibling 命中, 抑制 missing 假阳性
                        pass
                    else:
                        inconsistencies.append({
                            "entity": canonical,
                            "expected": canonical,
                            "found": None,
                            "source_name": matched_source,
                            "type": "missing",
                            "chunk_id": chunk_id,
                        })

            self._maybe_discover_entity_variant(canonical, lit_text, variant_names)

        return inconsistencies

    def _build_entity_registry_for_rewriter(self, source_text: str) -> Dict[str, str]:
        """回路修复 helper：构建 Rewriter 可见的实体译名登记表

        Round 99 P1: per-chunk 缓存. 同一 chunk 的所有 retry 用同一份 source_text,
        registry 在 chapter 内不变. 用 chunk_id 作为缓存 key (隐式从 source_text 提取).

        只注入本章原文出现的实体（与 _check_entity_consistency 同语义），
        避免无关实体污染 prompt 同时节约 token。

        关键：把每个 entity 的所有 source_name 都加进 dict（不只是第一个），
        这样 Rewriter 能在 prompt 里看到所有英文别名 → 中文译名的映射，
        即使 source_text 使用的是 source_names[2] 也能正确翻译。
        修前只取第一个 source_name 时，遇到 "Mr. George Lomax" 在源文里写
        "George Lomax" 时 Rewriter 找不到匹配的指引。

        Returns:
            {english_source_name: chinese_canonical} dict（每个 entity 可能多 key）
        """
        if not self.decision_engine:
            return {}

        # Round 99 P1: per-chunk 缓存. 用 source_text hash 作为 key (chunk_id 不可得,
        # 因为此 helper 是无状态调用). hash 同一 source_text 一致, 缓存命中走短路.
        cache_key = hash(source_text)
        if cache_key in self._entity_registry_cache:
            return self._entity_registry_cache[cache_key]

        try:
            from src.core.decision_engine import query_reviewable_entities
            rows = query_reviewable_entities(self.decision_engine.conn)
        except Exception:
            return {}
        out: Dict[str, str] = {}
        for row in rows:
            canonical = row[0]
            source_names_json = row[1]
            source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
            # Round 111: 过滤无效 source_name（限定词+全小写普通名词短语等，如 "the stairs"/
            # "the chauffeur"/"my maid"）— 与 _check_entity_consistency 同语义，避免
            # rewriter/critic 收到"把描述性短语统一为规范主名"的不可执行指令（ch09 chunk004）。
            # Round 112: 身份映射 canonical（如 " Cloak Room"→"Cloak Room"）同步剔除。
            if not self._is_enforceable_canonical(canonical, source_names):
                continue
            valid_sns = [sn for sn in source_names if self._is_valid_entity_source_name(sn)]
            # Round 133b: 别名吞并抑制 — swallowed 别名不注入/不当在场证据
            # （chunk071 "Mother"→杰西卡夫人 污染 prompt 根因）
            effective_sns, swallowed_sns = self._partition_swallowed_source_names(
                source_text, canonical, valid_sns,
            )
            # 只取本章出现的实体
            if not any((sn and entity_source_name_in_text(sn, source_text)) for sn in effective_sns):
                continue
            # 把每个匹配的 source_name 都映射到 canonical（修前 bug: 只取第一个）
            for sn in valid_sns:
                if sn in swallowed_sns:
                    continue
                if sn and sn not in out:
                    out[sn] = canonical
        # 缓存结果 (同一 source_text 在 chapter 内稳定)
        self._entity_registry_cache[cache_key] = out
        return out

    def _build_entity_variants_for_critic(self, source_text: str) -> Dict[str, List[str]]:
        """Round 61 helper：为 critic 构造 per-entity 变体表（供 typesafe Jev atomic Noul 使用）

        Round 99 P1: per-chunk 缓存 (与 _build_entity_registry_for_rewriter 同样策略).

        与 _build_entity_registry_for_rewriter 同语义：只注入本章原文出现的实体。
        Returns:
            {english_source_name: [variant_zh_1, variant_zh_2, ...]}
        """
        if not self.decision_engine:
            return {}

        cache_key = hash(source_text)
        if cache_key in self._entity_variants_cache:
            return self._entity_variants_cache[cache_key]

        try:
            from src.core.decision_engine import query_reviewable_entities
            rows = query_reviewable_entities(self.decision_engine.conn)
        except Exception:
            return {}
        out: Dict[str, List[str]] = {}
        for row in rows:
            canonical = row[0]
            source_names_json = row[1]
            source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
            variant_names_json = row[2] if len(row) > 2 else None
            variant_names = robust_json_loads(variant_names_json, expected_type=list) if variant_names_json else []
            # Round 111: 与 _build_entity_registry_for_rewriter 同语义过滤无效 source_name
            # Round 112: 身份映射 canonical 同步剔除（无翻译信息）
            if not self._is_enforceable_canonical(canonical, source_names):
                continue
            valid_sns = [sn for sn in source_names if self._is_valid_entity_source_name(sn)]
            # Round 133b: 别名吞并抑制（与 _build_entity_registry_for_rewriter 同语义）
            effective_sns, swallowed_sns = self._partition_swallowed_source_names(
                source_text, canonical, valid_sns,
            )
            if not any((sn and entity_source_name_in_text(sn, source_text)) for sn in effective_sns):
                continue
            for sn in valid_sns:
                if sn in swallowed_sns:
                    continue
                if sn and sn not in out:
                    out[sn] = list(variant_names) if variant_names else []
        # Round 99 P1: 缓存 variants 结果
        self._entity_variants_cache[cache_key] = out
        return out

    def _maybe_discover_entity_variant(self, canonical: str, lit_text: str, existing_variants: list):
        """检测润色稿中可能的新变体（与 canonical 一字之差）

        B3 Round 2 修复：旧实现"只要发现一字之差就入注册表" → 与 canonical 冲突 →
        触发 entity_hard_fail 死循环（每次都查到新"变体"）。

        修复策略：
          - 加 min_occurrence 门槛（默认 2 次）：变体必须在同一 chunk 内出现 ≥N 次才入注册表
          - 加 auto_register 配置开关（Round 105 起默认 false，仅 soft-hint，需显式开启）
          - Round 105: 硬长度护栏（canonical < variant_min_auto_len 默认 3 字绝不自动入表）
          - Round 105: 普通词护栏（src.utils.entity_stopwords）：普通词/高频差异字
            一律不作为人名异体，杜绝 东西/卡特/警察/伦敦金 等污染
          - 低于门槛的候选写入 soft_candidates 字段（不持久化，仅返回供调用方参考）

        Args:
            canonical: 已登记的主名（中文）
            lit_text: 润色稿
            existing_variants: 已登记的 variant 列表
        """
        try:
            if not canonical or not re.fullmatch(r'[一-鿿]{2,}', canonical):
                return [], []
            length = len(canonical)
            # 用 finditer 取所有匹配位置，便于统计出现次数
            all_matches = re.findall(r'(?=([一-鿿]{%d}))' % length, lit_text)
            from collections import Counter
            cand_counts = Counter(all_matches)

            # B3 配置：min_occurrence 门槛 + auto_register 开关
            min_occurrence = int(_config.pipeline.get("variant_min_occurrence", 2))
            # Round 105: 默认改为 False。一字之差自动入表误报率极高（ch32 实测把
            # 东西/卡特/警察/伦敦金 等普通词登记为人名异体，随后被 consistency 3c
            # 盲目替换改坏正文）。默认只记 soft-hint 供人工复核，需显式开启才入表。
            auto_register = bool(_config.pipeline.get("variant_auto_register", False))
            # Round 105: 硬长度护栏。短规范名（<3 字）的一字之差几乎必为普通词
            # （黛西←东西 / 卡洛←卡特），即使 auto_register 开启也绝不自动入表。
            min_auto_len = int(_config.pipeline.get("variant_min_auto_len", 3))

            # Round 105: 普通词护栏（共享 src.utils.entity_stopwords）。
            try:
                from src.utils.entity_stopwords import is_plausible_name_variant
            except Exception:  # 单文件部署兜底
                is_plausible_name_variant = None

            to_register = []
            soft_candidates = []
            for cand, count in cand_counts.items():
                if cand == canonical or cand in existing_variants:
                    continue
                if self.decision_engine.is_canonical_name(cand):
                    continue
                diff = sum(1 for a, b in zip(canonical, cand) if a != b)
                if diff != 1:
                    continue
                # Round 105: 普通词/高频差异字一律不作为人名异体（连 soft-hint 也不记，
                # 避免噪声）。真异体（差异字冷僻）才进入后续门槛判断。
                if is_plausible_name_variant is not None:
                    plausible, _reason = is_plausible_name_variant(canonical, cand)
                    if not plausible:
                        continue
                # 只考虑一字之差的近音/近形变体
                can_auto = auto_register and length >= min_auto_len
                if count >= min_occurrence and can_auto:
                    self.decision_engine.add_entity_variant(canonical, cand)
                    to_register.append((cand, count))
                else:
                    # 低于门槛 / 未开启自动入表：仅记为 soft candidate，不写入 registry
                    soft_candidates.append((cand, count))

            return to_register, soft_candidates
        except Exception:
            return [], []

    @staticmethod
    def _audit_early_stop_count(critic_report: Dict[str, Any], threshold: float) -> int:
        """Round 115: AUDIT_EARLY_STOP 只统计"结论性低分"维度。

        ch06 chunk012 根因: 3 个维度（literal_residue 3.1 / semantic_preservation 3.9 /
        entity_consistency 2.4）实际都被 Jev verdict 标为 uncertain 并列入
        skipped_dimensions（R8 起 synthesize_verdict 据此不判 REJECT），但
        AUDIT_EARLY_STOP（R99 P3）仍按原始 legacy 分数把它们计入"低于阈值" →
        首轮直接 fallback，跳过 judge —— 前提"judge 必然 REJECT"对 uncertain
        维度不成立，导致合格译文被误杀。改为只统计 verdict 未跳过、分数仍低于
        阈值的维度。
        """
        scores = (critic_report or {}).get("scores") or {}
        skipped = set(
            ((critic_report or {}).get("_typesafe_verdict") or {}).get("skipped_dimensions") or []
        )
        # legacy scores 键与 primitive 维度名不同名（semantic_preservation↔semantic_loss 等）
        _alias = {
            "voice_drift": "voice_consistency",
            "style_compliance_score": "style_compliance",
            "semantic_loss": "semantic_preservation",
        }
        skipped_legacy = {_alias.get(d, d) for d in skipped}
        return sum(
            1 for k, v in scores.items()
            if k not in skipped_legacy and isinstance(v, (int, float)) and v < threshold
        )

    def _process_auditing_batch(self, tasks):
        success_ids = []
        # Round 71: 每次审计批次开始重置跨人物冲突 sibling 缓存, 避免跨章节脏数据
        self._sibling_canonicals_cache = None
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                # Round 68: 提前到循环顶部提取 quality_retries（修复 Round 65 引入的 NameError）
                # 旧位置在 entity_hard_fail 之后定义，导致 line 1893 自加时变量未绑定
                quality_retries = task.get('quality_retries', 0) or 0
                source_text = task['text_content']
                raw_text = self._load_intermediate(chunk_id, "raw")
                lit_text = self._load_intermediate(chunk_id, "literary")
                # 回路修复：把 entity_registry 也注入 Critic/Jev
                # （修前 pipeline 只在 Rewriter 调用时构造 registry，Critic 完全拿不到，
                #  Jev 看到的 entity_registry 是空 → entity_consistency 维度永远 Noul~0.5 随机噪声）
                # 用与 Rewriter 完全相同的过滤逻辑，保证 Jev 与本地 _check_entity_consistency 看到同一份登记表。
                entity_registry_for_critic = self._build_entity_registry_for_rewriter(source_text)
                # Round 61: 把 variants 也传给 critic（per-entity atomic Noul 需要）
                entity_variants_for_critic = self._build_entity_variants_for_critic(source_text)
                # Round 65: 每次 rewriter retry 都会触发新一轮 critic_scoring，计入实际调用次数
                self._llm_call_counter["critic_scoring"] += 1 + quality_retries
                critic_report = self.critic_agent.process_chunk(
                    chunk_id, source_text, raw_text, lit_text, self.style_guide,
                    entity_registry=entity_registry_for_critic,
                    entity_variants=entity_variants_for_critic,
                )

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
                # Round 92: 追踪当前 retry 的 lit_text 与 critic score, fallback 时用最佳版
                self._update_best_lit(chunk_id, lit_text, critic_report)
                # Round 99 P3: audit-phase early stop. 旧实现仅在 _process_judging_batch 检查
                # early_stop (judge REJECT 后), 但若首轮 critic 评分已极差, 跑 judge LLM 是浪费
                # (judge 必然 REJECT, rewriter 也不可能在 1 retry 内修复极差译文).
                # 把 early_stop 提前到 audit 阶段: quality_retries == 0 时,
                # 若 very_low_count >= max_low_dims (即多数维度评分 < threshold),
                # 直接 fallback, 跳过 judge + 后续 rewriter cycle.
                early_stop_cfg_p99 = _config.pipeline.get("early_stop", {}) or {}
                early_stop_enabled_p99 = bool(early_stop_cfg_p99.get("enabled", True))
                # Round 106: mock_heuristic 的分数不可信（Jev 不可用时的兜底假分数），
                # 不触发 AUDIT_EARLY_STOP，让 LLM judge 独立裁决。
                _mock_untrusted_p99 = (
                    not _config.pipeline.get("trust_mock_heuristic", False)
                    and critic_report.get("_backend_path") == "mock_heuristic"
                )
                if (early_stop_enabled_p99
                        and quality_retries == 0
                        and not _mock_untrusted_p99
                        and critic_report.get("scores")):
                    es_threshold_p99 = float(early_stop_cfg_p99.get("low_score_threshold", 3.0))
                    es_max_low_p99 = int(early_stop_cfg_p99.get("max_low_dims", 3))
                    # Round 115: 只统计结论性低分 — verdict 明确跳过（uncertain/low_conf）
                    # 的维度不计入，否则与 R8 verdict 合成语义冲突（ch06 chunk012 根因）。
                    very_low_count_p99 = self._audit_early_stop_count(critic_report, es_threshold_p99)
                    if very_low_count_p99 >= es_max_low_p99:
                        stop_reason_p99 = (
                            f"AUDIT_EARLY_STOP: {very_low_count_p99}项低于{es_threshold_p99}分"
                            f"（{critic_report['scores']}），跳过 judge + rewriter 直接 fallback"
                        )
                        print(f"⏭️ [Pipeline] {chunk_id} {stop_reason_p99}")
                        best_text_p99 = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text_p99,
                            "metadata": {
                                "fallback": True,
                                "reason": "AUDIT_EARLY_STOP",
                                "early_stop_detail": stop_reason_p99,
                                "scores": critic_report["scores"],
                                "quality_retries": quality_retries,
                            }
                        }, source_step="literary")
                        self.scheduler.update_task_state(
                            chunk_id, TaskState.PERMANENTLY_FAILED,
                            error_msg=stop_reason_p99
                        )
                        continue
                if not critic_report.get("scores"):
                    print(f"⚠️ [Pipeline] {chunk_id} Critic 评分为空（JSON 解析失败），转入 FAILED 等待重试")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.FAILED,
                        error_msg="[AUDITING] Critic 返回空 scores"
                    )
                    continue

                # 实体硬失败标志 + 降级阈值 + serious 列表 (Round 101 P1 提前到这里, 高分直通检查需要用)
                entity_hard_fail = bool(_config.pipeline.get("entity_hard_fail", True))
                degradation_threshold = int(_config.pipeline.get("entity_hard_fail_degradation_threshold", 3))
                serious = [
                    n for n in (entity_notes or [])
                    if n.get("type") in ("variant_instead_of_canonical", "missing")
                ]
                # Round 101 P1: audit 高分直通 — 所有维度评分 ≥ high_threshold + 无实体违规 + CJK 高,
                # 直接 COMPLETED, 跳过 judge LLM 调用. 与 AUDIT_EARLY_STOP (R99 P3) 互补:
                #   R99 P3: 评分低 → PERMANENTLY_FAILED (跳 judge + rewriter)
                #   R101 P1: 评分高 → COMPLETED (跳 judge)
                # 触发条件: quality_retries == 0 (首轮, 避免误判 retry 中可能恶化的 chunk)
                #           AND all_scores >= high_threshold
                #           AND serious == 0 (无严重实体违规)
                #           AND lit_cjk >= final_cjk_threshold (译文是中文)
                high_pass_cfg = _config.pipeline.get("audit_high_pass", {}) or {}
                high_pass_enabled = bool(high_pass_cfg.get("enabled", True))
                # Round 68: quality_retries 已提升到循环顶部, 此处不再重复定义.
                # Round 106: mock_heuristic 的分数不可信，不触发高分直通，让 LLM judge 裁决。
                _mock_untrusted_hp = (
                    not _config.pipeline.get("trust_mock_heuristic", False)
                    and critic_report.get("_backend_path") == "mock_heuristic"
                )
                if (high_pass_enabled and quality_retries == 0
                        and not _mock_untrusted_hp
                        and not (entity_hard_fail and serious)):
                    high_threshold = float(high_pass_cfg.get("high_score_threshold", 8.0))
                    scores_dict = critic_report.get("scores", {}) or {}
                    all_high = (
                        scores_dict
                        and all(
                            isinstance(v, (int, float)) and v >= high_threshold
                            for v in scores_dict.values()
                        )
                    )
                    lit_cjk_r101 = compute_cjk_ratio(lit_text)
                    final_threshold = float(_config.pipeline.get("final_cjk_threshold", 0.5))
                    cjk_ok = (final_threshold <= 0) or (lit_cjk_r101 >= final_threshold)
                    if all_high and cjk_ok:
                        # 直接 COMPLETED, 跳过 judge LLM
                        print(
                            f"✅ [Pipeline] {chunk_id} audit 高分直通 (all_scores≥{high_threshold}, "
                            f"cjk={lit_cjk_r101:.2f}), 跳过 judge 直接 COMPLETED"
                        )
                        # 保存 final (用 literary 版, 通过 sanity gate)
                        final_payload = {
                            "text": lit_text,
                            "metadata": {
                                "fallback": False,
                                "reason": "AUDIT_HIGH_PASS",
                                "scores": scores_dict,
                                "lit_cjk_ratio": round(lit_cjk_r101, 3),
                                "quality_retries": quality_retries,
                                "audit_short_circuit": True,
                            }
                        }
                        self._save_final_with_sanity(chunk_id, final_payload, source_step="literary")
                        self.scheduler.update_task_state(
                            chunk_id, TaskState.COMPLETED,
                            error_msg=f"[AUDIT_HIGH_PASS] all_scores≥{high_threshold}, cjk={lit_cjk_r101:.2f}, judge skipped"
                        )
                        continue

                # 实体硬失败：严重不一致时强制一轮质量退回（notes-only 不够）
                # P0 修复：增加降级机制，避免死锁
                # 注: entity_hard_fail / degradation_threshold / serious 已在 Round 101 P1 高分直通检查处定义

                # Round 98 P1: 实体违规连续 N 次早停
                # ch32 实测: 每次 retry 仍有 1-2 个 serious 违规 (variant_instead_of_canonical),
                # rewriter 反复修复但失败. 在 entity_stagnation_threshold 次 retry 后 fallback,
                # 节省 rewriter+critic+judge LLM 调用.
                # 触发条件: 最近 N 次 audit 每次都有 ≥1 个 serious 违规 (不管走 entity_hard_fail
                #           还是 degradation pass 路径), 且 quality_retries >= entity_stagnation_threshold.
                # 行为: fallback 到 _get_best_lit (可能仍有违规, 但已是 rewriter 能拿到的最高分);
                #       标 PERMANENTLY_FAILED + error_msg 含 [ENTITY_VIOLATION_STREAK].
                entity_stagnation_threshold = int(_config.pipeline.get("entity_stagnation_threshold", 2))
                entity_hist = self._entity_violation_history.setdefault(chunk_id, [])
                entity_hist.append(len(serious))
                if len(entity_hist) > 5:
                    entity_hist.pop(0)  # 仅保留最近 5 次, 避免内存增长
                recent_serious = entity_hist[-entity_stagnation_threshold:]
                if (entity_stagnation_threshold > 0
                        and len(recent_serious) == entity_stagnation_threshold
                        and all(c > 0 for c in recent_serious)
                        and entity_hard_fail
                        and quality_retries >= entity_stagnation_threshold):
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text)
                    self._save_final_with_sanity(chunk_id, {
                        "text": best_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "ENTITY_VIOLATION_STREAK",
                            "serious_history": recent_serious,
                            "entity_stagnation_threshold": entity_stagnation_threshold,
                            "judge_reason": f"实体违规连续 {entity_stagnation_threshold} 次未修复 (counts={recent_serious})",
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=(
                            f"[ENTITY_VIOLATION_STREAK] 实体违规连续 {entity_stagnation_threshold} 次未修复 "
                            f"(counts={recent_serious}), 提前 fallback"
                        )
                    )
                    print(
                        f"⏹️ [Pipeline] {chunk_id} 实体违规连续 {entity_stagnation_threshold} 次未修复 "
                        f"(counts={recent_serious}), 提前 fallback"
                    )
                    continue

                # 降级逻辑：质量重试 > 0 时，仅在严重不一致项 > 阈值时才强制退回
                # 避免因少量实体问题导致死锁
                if entity_hard_fail and serious and quality_retries == 0:
                    # 回路修复：把 entity_block 也写入 feedback（之前只放在 specific_edits，缺失形式化清单）
                    entity_lines = []
                    for s_e in serious[:30]:
                        ent = s_e.get("entity", "")
                        exp = s_e.get("expected", "")
                        found = s_e.get("found", "")
                        ntype = s_e.get("type", "")
                        # Round 20: 把 source_name 一并展示（rewriter 可定位源文位置）
                        src_name = s_e.get("source_name", "")
                        src_hint = f"，源文形式：「{src_name}」" if src_name else ""
                        if exp and found and found != exp:
                            entity_lines.append(f"- 「{found}」应统一为「{exp}」（{ntype}{src_hint}）")
                        elif exp:
                            entity_lines.append(f"- 实体「{ent}」须使用主名「{exp}」（{ntype}{src_hint}）")
                    entity_block = ("实体一致性要求（必须遵循）：\n" + "\n".join(entity_lines)) if entity_lines else ""

                    feedback = {
                        "judge_reason": f"实体一致性硬失败: {len(serious)} 项译名不一致，须按 entity_registry 修正",
                        "critic_suggestions": json.dumps(serious, ensure_ascii=False)[:500],
                        "entity_consistency_notes": entity_block,  # 回路修复：补这个字段
                        "low_dims": ["entity_consistency"],
                        "scores": critic_report.get("scores", {}),
                        "specific_edits": [
                            {
                                "location": s.get("entity", "?"),
                                "original": s.get("found") or "",
                                "issue": s.get("type"),
                                "suggested_direction": f"统一为 {s.get('expected')}" + (f"（源文:{s.get('source_name', '')}）" if s.get("source_name") else ""),
                            }
                            for s in serious[:10]
                        ],
                        # Round 64: 把 _typesafe_verdict 透传给 rewriter，让它知道哪些维度被 Jev uncertain 跳过
                        "_typesafe_verdict": critic_report.get("_typesafe_verdict", {}) or {},
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
                    # B1 修复：先检查是否达到 max_quality_retries_entity 上限，
                    #         达上限则降级放行，避免死循环（实测 ch32 quality_retries=112+）
                    max_qr_entity = int(_config.pipeline.get("max_quality_retries_entity", 4))
                    if quality_retries >= max_qr_entity:
                        print(f"🟡 [Pipeline] {chunk_id} 实体一致性达上限放行 (qr={quality_retries} >= max_qr_entity={max_qr_entity}, serious={len(serious)})")
                        success_ids.append(chunk_id)
                        continue
                    # 回路修复：同样构造 entity_block
                    entity_lines = []
                    for s_e in serious[:30]:
                        ent = s_e.get("entity", "")
                        exp = s_e.get("expected", "")
                        found = s_e.get("found", "")
                        ntype = s_e.get("type", "")
                        if exp and found and found != exp:
                            entity_lines.append(f"- 「{found}」应统一为「{exp}」（{ntype}）")
                        elif exp:
                            entity_lines.append(f"- 实体「{ent}」须使用主名「{exp}」（{ntype}）")
                    entity_block = ("实体一致性要求（必须遵循）：\n" + "\n".join(entity_lines)) if entity_lines else ""

                    feedback = {
                        "judge_reason": f"实体一致性硬失败(降级): {len(serious)} 项译名不一致，须按 entity_registry 修正",
                        "critic_suggestions": json.dumps(serious, ensure_ascii=False)[:500],
                        "entity_consistency_notes": entity_block,  # 回路修复
                        "low_dims": ["entity_consistency"],
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
                        # Round 64: 把 _typesafe_verdict 透传给 rewriter
                        "_typesafe_verdict": critic_report.get("_typesafe_verdict", {}) or {},
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
        thresholds = _config.critic_thresholds_for_book()
        low_dims = [k for k, v in scores.items()
                    if isinstance(v, (int, float)) and v < thresholds.get(k, 7.0)]
        specific_edits = critic_report.get("specific_edits", [])[:20]
        # Round 118 P1-2: typesafe 路径下 critic_report.specific_edits 恒为空，
        # 可执行编辑由 judge 的 REJECT 第二意见产出（unresolved_edits）并入。
        if not specific_edits and judge_result.get("unresolved_edits"):
            specific_edits = judge_result.get("unresolved_edits", [])[:20]
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
            # Round 64: 把 _typesafe_verdict 透传给 rewriter（让 rewriter 感知 uncertain 跳过维度）
            "_typesafe_verdict": critic_report.get("_typesafe_verdict", {}) or {},
            # Round 118 P1-2: 第二意见审计元数据（confirmed_reject 仅参考，不翻案）
            "_second_opinion": judge_result.get("_second_opinion", {}) or {},
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

                # Round 123: mock_heuristic 是 Jev/critic LLM 不可用时的兜底假分数,
                # 不得触发 judging 阶段 EARLY_STOP — 与 audit 阶段 R106 护栏
                # (_mock_untrusted_p99) 同语义。根因: ch09 chunk013/044 critic
                # 连续 429 回退 mock 后, 假分数 readability=2.5/fluency=2.5/
                # literal_residue=0.0 (3 维<4.0) 被 EARLY_STOP 秒杀成
                # PERMANENTLY_FAILED, 而同一份 mock 分数在 audit 阶段是放行的。
                # 未显式 trust mock 时跳过 EARLY_STOP, 交由 rewriter 正常轮次。
                mock_untrusted_judging = (
                    isinstance(critic_report, dict)
                    and not _config.pipeline.get("trust_mock_heuristic", False)
                    and critic_report.get("_backend_path") == "mock_heuristic"
                )

                # 加载 Rewriter 自评（如果有）
                self_review = None
                if quality_retries > 0:
                    try:
                        meta = self._load_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}")
                        if isinstance(meta, dict):
                            self_review = meta.get("self_review")
                    except Exception:
                        pass

                # Round 65: 每次 rewriter retry 都会触发新一轮 judge_decision，计入实际调用次数
                self._llm_call_counter["judge_decision"] += 1 + quality_retries
                judge_result = self.judge_agent.process_chunk(
                    chunk_id, source_text, lit_text, critic_report,
                    affected_chunks=[chunk_id],
                    rewriter_self_review=self_review,
                    verse_note=self._verse_note_for(chunk_id, source_text),  # Round 119 L1
                )

                # Round 93: judge REJECT 后检查是否"连续下降" — 若连续 3 次 retry 分数递减,
                # 跳过 rewriter fallback, 直接用 best_lit 写入 final. 节省 ~2 次重试 LLM 调用
                # (rewriter + critic + judge 共 3 LLM 调用每次).
                current_avg = self._compute_avg_score(critic_report)
                # Round 108: streak 早停只适用于 REJECT 路径 — judge 已 PASS 时直接定稿。
                # 修前无条件执行: 旧分数历史（如回溯重跑前）会把通过轮误杀
                # (ch08 chunk006: 全部维度通过, 却被 NO_PROGRESS_STREAK 打成 PERMANENTLY_FAILED)。
                judge_is_pass = judge_result.get("decision") == "PASS"
                decline_streak = int(_config.pipeline.get("score_decline_streak", 4))
                if not judge_is_pass and self._is_declining_streak(chunk_id, current_avg, consecutive=decline_streak):
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    self._save_final_with_sanity(chunk_id, {
                        "text": best_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "SCORE_DECLINE_STREAK",
                            "judge_reason": judge_result.get("reject_reason", ""),
                            "quality_retries": quality_retries,
                            "score_history": list(self._score_history.get(chunk_id, [])[-5:]),
                            "current_avg": round(current_avg, 2),
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=f"[SCORE_DECLINE] 连续 {decline_streak} 次 retry 分数递减, 提前 fallback (history={self._score_history.get(chunk_id, [])[-5:]})"
                    )
                    print(f"⏹️ [Pipeline] {chunk_id} 连续 {decline_streak} 次 retry 分数递减, 提前 fallback (current={current_avg:.2f}, history={self._score_history.get(chunk_id, [])[-5:]})")
                    continue
                # Round 98 P2: 非进展 streak — 末尾 N 个 score 与当前 score 单调不增
                # (含持平) 时提前 fallback. 比 _is_declining_streak (严格递减) 更激进,
                # 抓住「rewriter 不恶化但也不进步」场景, 节省 rewriter+critic LLM 调用.
                # 与 SCORE_DECLINE 互补: SCORE_DECLINE 要求 s[i] > s[i+1], NO_PROGRESS 允许 s[i] == s[i+1].
                no_progress_window = int(_config.pipeline.get("no_progress_window", 3))
                if (not judge_is_pass and no_progress_window > 1
                        and self._is_no_progress_streak(chunk_id, current_avg, window=no_progress_window)):
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    self._save_final_with_sanity(chunk_id, {
                        "text": best_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "NO_PROGRESS_STREAK",
                            "judge_reason": judge_result.get("reject_reason", ""),
                            "quality_retries": quality_retries,
                            "score_history": list(self._score_history.get(chunk_id, [])[-5:]),
                            "current_avg": round(current_avg, 2),
                            "no_progress_window": no_progress_window,
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=(
                            f"[NO_PROGRESS_STREAK] 连续 {no_progress_window} 次 retry 单调不增 (含持平), "
                            f"提前 fallback (history={self._score_history.get(chunk_id, [])[-5:]})"
                        )
                    )
                    print(
                        f"⏹️ [Pipeline] {chunk_id} 连续 {no_progress_window} 次 retry 单调不增, "
                        f"提前 fallback (current={current_avg:.2f}, history={self._score_history.get(chunk_id, [])[-5:]})"
                    )
                    continue
                # Round 97: CJK 重试耗尽早停 — rewriter 反复输出英文, 不浪费 LLM 调用
                # 触发条件: REJECT 仅因 CJK 比例过低 (Round 96 P3 闸门), 且已 retry >= cjk_max_retries 次.
                # 行为: fallback 到 raw 翻译稿 (raw 经 P2 闸门保证是中文). 比 max_quality_retries (4) 早停,
                #       节省 2-4 次 rewriter+critic LLM 调用.
                reject_reason_raw = judge_result.get("reject_reason", "") or ""
                cjk_max_retries = int(_config.pipeline.get("cjk_max_retries", 2))
                cjk_only_reject = (
                    "CJK 比例过低" in reject_reason_raw
                    and cjk_max_retries > 0
                    and quality_retries >= cjk_max_retries
                )
                if cjk_only_reject:
                    # 加载 raw 翻译稿作为 fallback (raw 经 P2 闸门保证是中文, 即使文学性差)
                    raw_fallback = ""
                    try:
                        raw_data = self._load_intermediate(chunk_id, "raw")
                        raw_fallback = raw_data if isinstance(raw_data, str) else (
                            raw_data.get("text", "") if isinstance(raw_data, dict) else ""
                        )
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError):
                        pass
                    if not raw_fallback or not raw_fallback.strip():
                        # raw 也不可用 → 用 lit (最佳一版, 即便英文)
                        raw_fallback = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    raw_cjk = compute_cjk_ratio(raw_fallback)
                    self._save_final_with_sanity(chunk_id, {
                        "text": raw_fallback,
                        "metadata": {
                            "fallback": True,
                            "reason": "CJK_RETRY_EXHAUSTED",
                            "judge_reason": reject_reason_raw[:200],
                            "quality_retries": quality_retries,
                            "cjk_max_retries": cjk_max_retries,
                            "raw_cjk_ratio": round(raw_cjk, 3),
                            "lit_cjk_ratio": round(compute_cjk_ratio(lit_text), 3),
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=(
                            f"[CJK_RETRY_EXHAUSTED] rewriter 重试 {quality_retries} 次仍输出英文 "
                            f"(lit cjk={compute_cjk_ratio(lit_text):.2f}), 提前 fallback 到 raw "
                            f"(raw cjk={raw_cjk:.2f}), 节省 {max_quality_retries - quality_retries} 次 LLM 调用"
                        )
                    )
                    print(
                        f"⏹️ [Pipeline] {chunk_id} CJK 重试耗尽 (lit cjk={compute_cjk_ratio(lit_text):.2f}, "
                        f"raw cjk={raw_cjk:.2f}, retries={quality_retries}/{max_quality_retries}), "
                        f"fallback 到 raw, 节省 {max_quality_retries - quality_retries} 次 LLM 调用"
                    )
                    continue
                if judge_result.get("decision") == "PASS":
                    # Round 119 L1: 韵文对齐闸门 — PASS 前先 lint；未达标触发一次定向重写。
                    verse_gate = self._verse_lint_gate(chunk_id, source_text, lit_text)
                    if (verse_gate is not None and not verse_gate["ok"]
                            and not verse_gate.get("retried")
                            and verse_gate.get("retry_on_fail", True)):
                        self._mark_verse_retry(chunk_id, verse_gate)
                        judge_result["reject_reason"] = (
                            "【韵文对齐未达标】" + (verse_gate.get("reason") or "")
                            + "；请严格按韵文协议重排：逐行对应、保留复沓（refrain）、"
                            + "不得把韵文并入散文段落。")
                        self._fallback_to_rewriter(chunk_id, lit_text, critic_report, judge_result)
                        continue
                    # Round 92: judge PASS 时也用最佳版本. 即使当前 retry 通过 judge,
                    # 之前的 retry 可能有更高 critic score (未达 judge pass 阈值但更高分).
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    if verse_gate is not None and not verse_gate["ok"]:
                        # L1: 一次定向重写后仍不达标 — 放行但落档（gate ⑥ 另行核验）
                        base_text = (best_text.get("text", "") if isinstance(best_text, dict)
                                     else str(best_text or ""))
                        best_text = {"text": base_text, "metadata": {"verse_lint": {
                            "ok": False,
                            "reason": verse_gate.get("reason", ""),
                            "expected_lines": verse_gate.get("expected_lines"),
                            "found_lines": verse_gate.get("found_lines"),
                        }}}
                        print(f"⚠️ [Pipeline] {chunk_id} verse_lint 未达标（已用过一次定向重写）: {verse_gate.get('reason', '')}")
                    self._save_final_with_sanity(chunk_id, best_text)
                    self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED)
                    self._append_chunk_memory(chunk_id, best_text if isinstance(best_text, str) else best_text.get("text", str(best_text)) if isinstance(best_text, dict) else str(best_text), source_text)
                    print(f"✅ [Pipeline] {chunk_id} 定稿完成。")
                elif quality_retries >= max_quality_retries:
                    # O 方案：安全网放行——若各维度已接近阈值（仅差 1-2 分），
                    # 不再整章 PERMANENTLY_FAILED 丢失译文，改为带告警定稿。
                    scores = critic_report.get("scores", {}) if isinstance(critic_report, dict) else {}
                    thresholds = _config.critic_thresholds_for_book()
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
                        # Round 92: 安全网放行时也用最佳版本, 而非当前被拒的 lit_text
                        best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "SAFETY_NET_PASSED",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                                "avg_score": round(avg_score, 2),
                            }
                        }, source_step="literary")
                        self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED, error_msg=f"[SAFETY_NET] 平均分{avg_score:.2f}≥{safety_net_min}，接近阈值放行")
                        print(f"🟡 [Pipeline] {chunk_id} 安全网放行（平均分{avg_score:.2f}，未达PERMANENTLY_FAILED）")
                    else:
                        # Round 92: max retries fallback 也用最佳版本
                        best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "PERMANENTLY_FAILED_AFTER_MAX_RETRIES",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        }, source_step="literary")
                        self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, error_msg=judge_result.get("reject_reason") or "质量重试耗尽")
                        print(f"⚠️ [Pipeline] {chunk_id} 质量重试耗尽，已 fallback 写入 final（最佳一版润色稿）")
                elif early_stop_enabled and quality_retries == 0:
                    # early_stop 仅在首轮裁决生效：避免 rework 一两次就放弃可恢复的 chunk
                    # 第二轮起若仍不达标，正常走 fallback 完成 max_quality_retries 轮 rework
                    # 详见 config.yaml pipeline.early_stop.apply_only_first_round
                    scores = critic_report.get("scores", {})
                    very_low_count = sum(
                        1 for v in scores.values()
                        if isinstance(v, (int, float)) and v < early_stop_threshold
                    )
                    if very_low_count >= early_stop_max_low and mock_untrusted_judging:
                        # Round 123: mock 假分数不可信 — 不早停, 落回 rewriter 正常轮次
                        # (与 audit 阶段 R106 护栏同语义, 见上方 mock_untrusted_judging)
                        print(f"🛡️ [Pipeline] {chunk_id} EARLY_STOP 拦截: critic 后端 mock_heuristic（假分数不可信），转 rewriter 正常重试")
                    elif very_low_count >= early_stop_max_low:
                        stop_reason = (
                            f"EARLY_STOP: {very_low_count}项低于{early_stop_threshold}分"
                            f"（{scores}），放弃重试"
                        )
                        print(f"⏭️ [Pipeline] {chunk_id} {stop_reason}")
                        # Round 92: early stop 也用最佳版本 (虽然首轮平均分低, 但其它维度可能还行)
                        best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "EARLY_STOP",
                                "early_stop_detail": stop_reason,
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        }, source_step="literary")
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



# Section 8 — 测试工具
# ────────────────────────────────────────────────────────────────────

GOLDEN_AB_MATRIX: List[Dict[str, Any]] = [
    {"name": "literal_effort_none", "role": "literal_translator", "effort": None},
    {"name": "literal_effort_xhigh", "role": "literal_translator", "effort": "xhigh"},
    {"name": "primary_effort_none", "role": "primary", "effort": None},
    {"name": "primary_effort_max", "role": "primary", "effort": "max"},
]


class GoldenSetEvaluator:
    """黄金测试集评估器 - 风格坍缩率量化"""

    def __init__(self, db_path: Optional[str] = None, literal_role: Optional[str] = None):
        # Round 118 P2-7: literal_role 可被 A/B 组合覆盖（DeepSeek vs MiniMax）
        self.literal_role = literal_role or _config.get_task_role("literal_translation")
        self.llm = get_llm_client(self.literal_role)
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


def run_golden_test(
    ab_override: Optional[Dict[str, Any]] = None,
    report_path: Optional[Any] = None,
):
    """运行黄金测试集（需要 input/golden/hyperion_5k.md）.

    Round 118 P2-7 / Round 128: ab_override={"name","role","effort"} 时按 A/B 组合
    覆盖直译阶段的模型与 effort 档位；report_path 可指定本轮报告输出路径
    （默认 output/golden_test_report.json）。
    """
    ab = dict(ab_override or {})
    print("=" * 60)
    print("🧪 SagaTrans 黄金测试集跑分")
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

    literal_role = ab.get("role") or _config.get_task_role("literal_translation")
    evaluator = GoldenSetEvaluator(
        db_path=str(root_dir / "db" / "decision_db.sqlite"),
        literal_role=literal_role,
    )
    if ab:
        print(
            f"🧪 A/B 组合: {ab.get('name')} "
            f"(role={literal_role}, effort={ab.get('effort') or 'none'})"
        )
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
            # 使用配置路由 literal_translation（A/B 时按组合覆盖 role + effort）
            model_key, params = _config.resolve_task_model("literal_translation")
            params = dict(params)
            if ab:
                ab_effort = ab.get("effort")
                if ab_effort:
                    params["effort"] = str(ab_effort).strip().lower()
                else:
                    params.pop("effort", None)
            model_name = get_role_model_name(literal_role) or model_key
            raw_text = evaluator.llm.generate(raw_prompt, model_name=model_name, **params)

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
                "judge_backend": judge_result.get("_backend_path"),
                "confidence_source": _extract_confidence_source(judge_result),
                "literal_role": literal_role,
                "literal_effort": params.get("effort"),
                "ab": ab.get("name") if ab else None,
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

    output_file = Path(report_path) if report_path else (root_dir / "output" / "golden_test_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_chunks": total_chunks,
                "error_count": error_count,
                "pass_rate": pass_count / max(total_chunks, 1),
                "avg_style_collapse_rate": avg_collapse,
                "avg_style_preservation": avg_preservation,
                "ab_stats": summarize_golden_results(all_results),
            },
            "details": all_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细报告已保存: {output_file}")
    return all_results


def _extract_confidence_source(judge_result: Dict[str, Any]) -> str:
    """Round 118 P1-1/P2-7: 从 judge 结果提取概率来源, 供 A/B 分组统计.

    - typesafe verdict 自带 confidence_source（jev / openai_compat / mock_heuristic / _uncalibrated）
    - 旧 LLM 路径无 verdict → "legacy" 后端映射为 "_uncalibrated"（非 Jev 校准）
    """
    verdict = (judge_result or {}).get("_typesafe_verdict") or {}
    src = verdict.get("confidence_source")
    if src:
        return str(src)
    backend = (judge_result or {}).get("_backend_path") or "unknown"
    if backend == "legacy":
        return "_uncalibrated"
    return str(backend)


def summarize_golden_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Round 118 P2-7: 聚合单轮 golden 结果, 按 verdict 来源分组统计.

    calibrated_ratio = confidence_source=="jev" 的有效样本占比。降级环境
    （无 TYPESAFE_API_KEY / 后端 fallback）的高通过率不得作为 A/B 结论或门禁依据。
    """
    valid = [r for r in results if "error" not in r]
    total = len(results)
    valid_count = len(valid)
    pass_count = sum(1 for r in valid if r.get("judge_decision") == "PASS")

    def _avg(values: List[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    chunk_avgs: List[float] = []
    for r in valid:
        scores = r.get("critic_scores") or {}
        nums = [v for v in scores.values() if isinstance(v, (int, float))]
        if nums:
            chunk_avgs.append(sum(nums) / len(nums))

    backend_dist: Dict[str, int] = {}
    confidence_dist: Dict[str, int] = {}
    for r in valid:
        backend = str(r.get("judge_backend") or "unknown")
        backend_dist[backend] = backend_dist.get(backend, 0) + 1
        conf = str(r.get("confidence_source") or "unknown")
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

    return {
        "total": total,
        "valid_count": valid_count,
        "error_count": total - valid_count,
        "pass_count": pass_count,
        "pass_rate": (pass_count / valid_count) if valid_count else 0.0,
        "avg_critic_score": _avg(chunk_avgs),
        "avg_style_preservation": _avg([float(r.get("style_preservation", 0.0)) for r in valid]),
        "avg_style_collapse_rate": _avg([float(r.get("style_collapse_rate", 0.0)) for r in valid]),
        "judge_backend_dist": backend_dist,
        "confidence_source_dist": confidence_dist,
        "calibrated_ratio": (confidence_dist.get("jev", 0) / valid_count) if valid_count else 0.0,
    }


def recommend_golden_combo(
    combos: Dict[str, Dict[str, Any]],
    min_calibrated: float = 0.5,
) -> Dict[str, Any]:
    """Round 118 P2-7: 在 Jev 校准占比达标的组合中选推荐（质量优先排序）.

    排序键: (pass_rate, avg_critic_score, avg_style_preservation) 依次取高。
    无达标组合时明确返回 name=None — 拒绝用降级数据得出结论。
    """
    calibrated = [
        name for name, c in combos.items()
        if (c.get("summary") or {}).get("valid_count", 0) > 0
        and (c.get("summary") or {}).get("calibrated_ratio", 0.0) >= min_calibrated
    ]
    if not calibrated:
        no_samples = all(
            (c.get("summary") or {}).get("valid_count", 0) == 0 for c in combos.values()
        )
        if no_samples:
            return {
                "name": None,
                "reason": "no_valid_samples",
                "note": "A/B 无有效样本（黄金集缺失或全部处理失败），无法判定",
            }
        return {
            "name": None,
            "reason": "no_calibrated_combo",
            "note": "所有组合的 verdict 均非 Jev 校准（环境降级），A/B 结论与门禁不可用",
        }

    def _key(name: str) -> Tuple[float, float, float]:
        s = combos[name]["summary"]
        return (
            float(s.get("pass_rate", 0.0)),
            float(s.get("avg_critic_score", 0.0)),
            float(s.get("avg_style_preservation", 0.0)),
        )

    best = max(calibrated, key=_key)
    return {
        "name": best,
        "reason": "best_calibrated_pass_rate",
        "calibrated_candidates": calibrated,
    }


def run_golden_ab(
    matrix: Optional[List[Dict[str, Any]]] = None,
    report_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Round 118 P2-7 / Round 128: 黄金集 A/B — 直译模型 × effort 档位对比.

    每个组合跑完整 直译→重写→critic→judge 流程（重写阶段按 task_routing 定，
    A/B 只切直译阶段），统计 pass_rate / critic 均分 / 风格保持度，并按
    judge `_backend_path` 与 `confidence_source` 分组；仅 Jev 校准占比 ≥
    config.pipeline.golden_ab.min_calibrated_ratio 的组合参与推荐。
    """
    matrix = [dict(m) for m in (matrix or GOLDEN_AB_MATRIX)]
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    combos: Dict[str, Dict[str, Any]] = {}
    for idx, combo in enumerate(matrix):
        name = str(combo.get("name") or f"combo_{idx}")
        print("\n" + "=" * 60)
        print(f"🧪 Golden A/B 组合 [{name}]: role={combo.get('role')}, effort={combo.get('effort') or 'none'}")
        print("=" * 60)
        results = run_golden_test(
            ab_override=combo,
            report_path=output_dir / f"golden_ab_{name}.json",
        )
        combos[name] = {
            "config": combo,
            "summary": summarize_golden_results(results),
            "results": results,
        }

    min_calibrated = float((_config.pipeline.get("golden_ab", {}) or {}).get("min_calibrated_ratio", 0.5))
    recommendation = recommend_golden_combo(combos, min_calibrated=min_calibrated)
    report = {
        "matrix": matrix,
        "min_calibrated_ratio": min_calibrated,
        "combos": combos,
        "recommendation": recommendation,
    }

    out = Path(report_path) if report_path else (output_dir / "golden_ab_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("📊 Golden A/B 汇总（仅 Jev 校准组合可作结论）")
    print("=" * 60)
    for name, c in combos.items():
        s = c["summary"]
        print(
            f"  {name:<18} pass={s['pass_count']}/{s['valid_count']} ({s['pass_rate']:.1%}) "
            f"critic={s['avg_critic_score']:.2f} 风格保持={s['avg_style_preservation']:.1%} "
            f"校准占比={s['calibrated_ratio']:.0%} backend={s['judge_backend_dist']}"
        )
    if recommendation.get("name"):
        print(f"✅ 推荐组合: {recommendation['name']} ({recommendation.get('reason')})")
    else:
        print(f"⚠️ 无推荐: {recommendation.get('note')}")
    print(f"📄 A/B 报告已保存: {out}")
    return report


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


def run_golden_gate(use_ab: bool = False):
    """Golden Gate：基于人工评分基线的动态阈值质量门禁
    
    1. 加载人工评分基线（如果存在）
    2. 运行 golden_test 获取 pipeline 实际 pass_rate
    3. 动态阈值 = max(0.7, human_baseline_pass_rate * 0.9) 或 fallback 0.8
    4. 比较 pipeline pass_rate 与动态阈值

    Round 118 P2-7: use_ab=True 时先跑「直译模型×thinking」4 组合 A/B，
    仅在存在 Jev 校准组合时用其推荐组合的结果做门禁（拒绝未校准数据放行）。
    
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
    
    # 运行 golden_test 获取 pipeline 实际结果（--ab 时先跑 4 组合并取 Jev 校准推荐）
    if use_ab:
        print("\n🧪 Golden A/B: 直译模型×thinking 4 组合对比...")
        ab_report = run_golden_ab()
        rec = (ab_report.get("recommendation") or {}).get("name")
        if not rec:
            print("❌ A/B 无 Jev 校准组合（环境降级），门禁拒绝用未校准数据判定")
            return False
        print(f"🎯 A/B 推荐组合: {rec}")
        results = ab_report["combos"][rec]["results"]
    else:
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
    print("🧪 SagaTrans 内存压力测试 (16GB 限制)")
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


# Section 9 — 辅助入口
# ────────────────────────────────────────────────────────────────────

def init_project(chapter_id: str = "ch01", force: bool = False) -> bool:
    """初始化翻译项目：切分文本并写入任务数据库

    Args:
        chapter_id: 章节 ID
        force: 是否覆盖已存在的 chunk（同时重置 text_content 和 retries）

    Returns:
        bool: True 表示初始化成功；False 表示失败（输入文件缺失或切分为空），
              调用方应据此决定是否向 shell 报告非零退出码。
    """
    root_dir = Path(__file__).resolve().parent.parent
    db_dir = root_dir / "db"
    db_file = db_dir / "workflow.db"
    input_file = root_dir / "input" / f"{chapter_id}.md"

    if not input_file.exists():
        print(f"❌ 找不到文件: {input_file}", file=sys.stderr)
        return False

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
        print("❌ 警告：没有切分出任何 chunk！", file=sys.stderr)
        return False

    scheduler.init_chapter_tasks(chapter_id=chapter_id, chunks=chunks, force=force)

    # Round 119 L1：韵文 chunk 标签（旁路 db/chunk_tags.json，不改 workflow.db schema）。
    # manifest 存在则按字符交集打标（含空表清理旧标签）；缺失时同样置空，防跨书残留。
    try:
        from src.verse import (load_verse_manifest, load_chunk_tags,
                               save_chunk_tags, build_chapter_tags)
        _manifest = load_verse_manifest(root_dir / "input")
        _blocks = ((_manifest.get("chapters") or {}).get(chapter_id)) or []
        _tags = load_chunk_tags(db_dir)
        _tags.setdefault("version", 1)
        _tags.setdefault("chapters", {})[chapter_id] = build_chapter_tags(
            chapter_id, markdown_text, chunks, _blocks)
        save_chunk_tags(db_dir, _tags)
        _n_verse = len(_tags["chapters"][chapter_id])
        if _n_verse:
            print(f"  🏮 韵文标签: {_n_verse} 个 verse chunk（db/chunk_tags.json）")
    except Exception as _verse_err:
        print(f"  ⚠️ 韵文标签计算失败（非致命）: {_verse_err}")

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
            # L1：force 视为全新尝试，重置 verse 定向重写的一次性限额标记
            for f in out_dir.glob("*_verse_retry.json"):
                f.unlink(missing_ok=True)

    print(f"\n✅ 初始化成功，数据已写入: {db_file}")
    return True


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


# Section 10 — 统一入口
# ────────────────────────────────────────────────────────────────────

def main():
    print_banner()

    import argparse
    parser = argparse.ArgumentParser(description="SagaTrans — AI 文学语义编译系统")
    parser.add_argument("command", nargs="?", default="pipeline",
                        choices=["pipeline", "init", "golden", "golden-gate", "memory", "debug", "split", "consistency", "consolidate", "finalize", "lint", "gate", "style-profile", "drift-audit", "notes-normalize"],
                        help="执行命令: pipeline (默认), init, golden, golden-gate, memory, debug, split, consistency, consolidate, finalize, lint, gate, style-profile, drift-audit, notes-normalize")
    parser.add_argument("--chapter", default="ch01", help="章节 ID (默认: ch01)")
    parser.add_argument("--chapters", default=None,
                        help="style-profile 命令：抽样章节，逗号分隔（默认均匀抽 5 章）")
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
                        help="consistency / notes-normalize 命令：执行自动修正（notes-normalize 默认仅生成差异报告）")
    parser.add_argument("--window", type=int, default=None,
                        help="drift-audit 命令：滑动窗口大小（默认 config.pipeline.drift_audit.window=5）")
    parser.add_argument("--jev", action="store_true",
                        help="drift-audit 命令：附加 Jev 窗口配对检查（需要 Jev 可用）")
    parser.add_argument("--json-out", default=None,
                        help="drift-audit 命令：报告 JSON 输出路径")
    parser.add_argument("--ab", action="store_true",
                        help="golden/golden-gate 命令：运行 直译模型×thinking 4 组合 A/B 对比")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="consolidate 命令：最大收敛轮数（默认 3）")

    args = parser.parse_args()

    # P0-4: 章节 ID 合法性校验（路径遍历防御）
    # Round 52 修复：re.fullmatch 替代 re.match（避免 $ 在 final \n 前的边界 bug — "ch01\n" 原可通过）
    if args.chapter and not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', args.chapter):
        print(f"❌ 非法 chapter_id: '{args.chapter}' (仅允许字母/数字/下划线/连字符，1-32 字符)", file=sys.stderr)
        sys.exit(1)

    # Round 130: 单实例守卫（审计①）——写共享状态的命令必须独占锁；
    # 只读命令（gate/lint/drift-audit 等）不拦截，便于运行中诊断。
    if args.command in _MUTATING_COMMANDS or (args.command == "consistency" and args.apply):
        if not acquire_instance_lock():
            sys.exit(1)

    # P4-9: 顶层异常包装（除已规范的 consistency/golden-gate 外，统一捕获并退出）
    try:
        if args.command == "pipeline":
            pipeline = TranslationPipeline(chapter_id=args.chapter)
            if not pipeline.run():
                sys.exit(1)
        elif args.command == "init":
            if not init_project(chapter_id=args.chapter, force=args.force):
                sys.exit(1)
        elif args.command == "style-profile":
            from src.style_profiler import generate_book_profile, summary_line
            _chapters = [c.strip() for c in args.chapters.split(",") if c.strip()] if args.chapters else None
            res = generate_book_profile(Path.cwd(), chapters=_chapters, force=args.force)
            if not res.get("ok"):
                print(f"❌ style-profile 失败: {res.get('error')}", file=sys.stderr)
                sys.exit(1)
            if res.get("skipped"):
                print(f"ℹ️ 已存在风格剖面，未重新生成: {res['path']}（--force 可强制重生成）")
            else:
                _prof = res.get("profile") or {}
                if res.get("saved", True):
                    print(f"✅ 风格剖面已写入 {res['path']}（samples: {', '.join(res.get('samples') or [])}）")
                else:
                    print(f"⚠️ 风格剖面未写入（LLM 不可用，未注入作品特定先验）: {_prof.get('note', '')}")
                print(f"   {summary_line(_prof)}")
        elif args.command == "drift-audit":
            from src.drift_audit import (
                audit_chapter,
                collect_chunk_texts,
                jev_window_check,
                summarize_report,
            )
            _report = audit_chapter(Path.cwd(), chapter=args.chapter, window=args.window)
            if args.jev:
                _chunks = collect_chunk_texts(Path.cwd() / "output" / args.chapter)
                _report["jev"] = jev_window_check(
                    _chunks, window=args.window or _report["window"]
                )
                _jv = _report["jev"]
                if _jv.get("available"):
                    print(f"🔬 Jev 窗口检查 (backend={_jv.get('backend')}): "
                          f"drifted_pairs={_jv.get('drifted_pairs')}")
                else:
                    print(f"⚠️ Jev 窗口检查不可用: {_jv.get('error')}", file=sys.stderr)
            print(summarize_report(_report))
            if args.json_out:
                Path(args.json_out).write_text(
                    json.dumps(_report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[out] {args.json_out}")
            sys.exit(0 if not _report.get("drift_flags") else 2)
        elif args.command == "golden":
            if args.ab:
                run_golden_ab()
            else:
                run_golden_test()
        elif args.command == "golden-gate":
            sys.exit(0 if run_golden_gate(use_ab=args.ab) else 1)
        elif args.command == "memory":
            test_memory_pressure()
        elif args.command == "debug":
            debug_db()
        elif args.command == "consistency":
            n_issues = run_consistency_check(dry_run=not args.apply)
            sys.exit(0 if n_issues == 0 else 2)
        elif args.command == "consolidate":
            from src.pipeline import consolidate_dirty_chapters
            _cons_res = consolidate_dirty_chapters(max_rounds=args.max_rounds)
            sys.exit(0 if _cons_res.get("converged") else 1)
        elif args.command == "gate":
            from src.pipeline import run_final_gate
            _gate_res = run_final_gate()
            sys.exit(0 if _gate_res.get("ok") else 1)
        elif args.command == "finalize":
            from src.book_lint import finalize_outputs
            finalize_outputs(apply=args.apply)
            sys.exit(0)
        elif args.command == "notes-normalize":
            from src.book_lint import notes_normalize_outputs
            _nn_res = notes_normalize_outputs(apply=args.apply)
            _nn_stats = _nn_res.get("stats") or {}
            print(f"notes-normalize: files={_nn_stats.get('files', 0)} "
                  f"changed={_nn_stats.get('changed', 0)} → {_nn_res.get('report', '')}")
            sys.exit(0)
        elif args.command == "lint":
            from src.book_lint import lint_book
            sys.exit(0 if lint_book().get("ok") else 1)
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


# Section 11 — 输入格式适配器 (EpubSplitter / TextSplitter / MdSplitter)
# ────────────────────────────────────────────────────────────────────

_SCORE_KEY_NORMALIZATION = {
    "voice_consistence": "voice_consistency",
    "voice_consistancy": "voice_consistency",
    "voice consistence": "voice_consistency",
    "style_complience": "style_compliance",
    "style_compliance_": "style_compliance",
    "semantic_preservation": "semantic_preservation",
    "semantic_preserv": "semantic_preservation",
    "literal_residue": "literal_residue",
    "literal_residues": "literal_residue",
    "literal_residu": "literal_residue",
    "fluency": "fluency",
    "readability": "readability",
}


_GUTENBERG_HEADER_PATTERNS = [
    re.compile(r'Project Gutenberg', re.IGNORECASE),
    re.compile(r'www\.gutenberg\.org', re.IGNORECASE),
    re.compile(r'This eBook is for the use of anyone'),
    re.compile(r'古腾堡计划', re.IGNORECASE),
]


_GUTENBERG_TOC_TITLE_PATTERN = re.compile(
    r'^\s*(contents|table of contents|目录)\s*$', re.IGNORECASE | re.MULTILINE)


_GUTENBERG_TOC_ITEM_PATTERN = re.compile(
    r'^\s*(?:[IVXLC]+\.?|\d+\.?|chapter\s+[IVXLC\d]+\.?)(?:\s|$)',
    re.IGNORECASE | re.MULTILINE)


def _splitter_log_info(msg: str):
    print(f"[INFO] {msg}", file=sys.stderr)


def _splitter_log_ok(msg: str):
    print(f"[OK] {msg}", file=sys.stderr)


def _splitter_log_warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)


def _splitter_log_err(msg: str):
    print(f"[ERR] {msg}", file=sys.stderr)


def _normalize_chapter_title(title: str) -> str:
    """规范章节标题：在「序号.」与后续正文之间补空格。

    源 EPUB 常见 `II.The Machine` / `CHAPTER I.Down the Rabbit-Hole` 这类
    序号与英文紧接、无空格的形式，拼成 Markdown 标题后观感差。统一规范为
    `II. The Machine` / `CHAPTER I. Down the Rabbit-Hole`。同时处理
    「第N章xxx」保持原样。
    """
    if not title:
        return title
    # T1：在任意「字母数字 + 句点 + 紧接大写字母」处插入空格（不再限于
    # 行首，修复 "CHAPTER I.Down…" 型）。两组负向守卫避免误伤缩写链：
    #   - 句点前已是「字母数字+句点」（如 U.S.A.、1.2.）不插入
    #   - 大写字母后紧跟另一个句点（如 U.S.A.、Ph.D.）不插入
    fixed = re.sub(r'(?<![A-Za-z0-9]\.)([A-Za-z0-9])\.(?=[A-Z](?!\.))',
                   r'\1. ', title.strip())
    return fixed


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
    """_epub_clean_html_text_ex 的纯文本包装（不收集韵文块，兼容旧调用）。"""
    return _epub_clean_html_text_ex(html_content)


def _strip_epub_front_matter(text: str) -> str:
    """裁剪古腾堡计划 EPUB 头部的许可证公告与目录块，仅保留正文。

    安全策略（不误删正文）：
    - 仅跳过含「古腾堡版权/许可证」特征的块；
    - 仅当某个块是「目录标题（Contents/目录）」且紧随其后含 ≥2 个编号
      目录项的连续块时，才把该目录区域一并跳过；仅在文本前部 30 块内
      探测（防误删正文中的普通编号列表）。
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

    # 2) 目录块：Contents 标题块 + 紧随其后的编号目录项块（≥2 项才裁剪）。
    #    T1：li/table 兜底提取后目录项会进入文本（此前被静默丢弃），沿用
    #    「≥2 项」安全策略；允许 Contents 不在首块（标题页场景）。
    scan_limit = min(len(blocks), start_idx + 30)
    for j in range(start_idx, scan_limit):
        if not _GUTENBERG_TOC_TITLE_PATTERN.search(blocks[j]):
            continue
        k = j + 1
        item_count = 0
        while k < len(blocks):
            matches = len(_GUTENBERG_TOC_ITEM_PATTERN.findall(blocks[k]))
            if matches == 0:
                break
            item_count += matches
            k += 1
        if item_count >= 2:
            blocks = blocks[:j] + blocks[k:]
            break

    return '\n\n'.join(blocks).strip()


def _epub_split_large_doc_by_toc(html_content: str, toc_entries: list, chapter_idx_start: int,
                                 sources_out: Optional[dict] = None,
                                 verse_out: Optional[dict] = None,
                                 lead_out: Optional[list] = None) -> list:
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

    # 接缝回填（Kindle split 格式）：文档开头到第一个锚点之间的正文是物理
    # 切分的上一章续文（如 ~200KB 分片的 index_split_NNN.html），不处理会被
    # 整段丢弃。把这段交给调用方回填到上一章末尾（实测《Why Nations Fail》
    # 此类分片共丢 ~17 万字符）。lead_out 非 None 时收集。
    if lead_out is not None and sorted_anchors:
        lead_html = '\n'.join(lines[:sorted_anchors[0][1][0]])
        if '<head' in lead_html and '</head>' in lead_html:
            lead_html = lead_html.split('</head>', 1)[1]
        lead_soup = BeautifulSoup(lead_html, 'html.parser')
        _mark_verse_elements(lead_soup)
        for br in lead_soup.find_all('br'):
            br.replace_with('\n')
        lead_blocks: list = []
        lead_text = _extract_block_text(lead_soup, lead_blocks)
        lead_text = re.sub(r'[ \t]+', ' ', lead_text)
        lead_text = re.sub(r'\n{3,}', '\n\n', lead_text)
        lead_text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', lead_text)
        lead_text = _strip_epub_front_matter(lead_text.strip())
        if lead_text:
            lead_out.append({'text': lead_text, 'html': lead_html, 'verse': lead_blocks})

    chapters = []
    for i, (anchor_id, (line_no, toc_title)) in enumerate(sorted_anchors):
        start_line = line_no
        end_line = sorted_anchors[i + 1][1][0] if i + 1 < len(sorted_anchors) else len(lines)
        chunk_html = '\n'.join(lines[start_line:end_line])

        title = toc_title or f"Section {i+1}"

        chunk_soup = BeautifulSoup(chunk_html, 'html.parser')
        _mark_verse_elements(chunk_soup)
        for br in chunk_soup.find_all('br'):
            br.replace_with('\n')
        # T1：与 _epub_clean_html_text 同口径（p/标题/pre + 容器兜底）
        verse_blocks: list = []
        text = _extract_block_text(chunk_soup, verse_blocks)
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
        if sources_out is not None:
            sources_out[ch_id] = [chunk_html]
        if verse_out is not None:
            verse_out[ch_id] = verse_blocks
        chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))

    return chapters


def _epub_extract_chapters(book, sources_out: Optional[dict] = None,
                           skipped_out: Optional[list] = None,
                           verse_out: Optional[dict] = None) -> list:
    """从 EPUB 提取章节，返回 [(chapter_id, markdown), ...]

    sources_out / skipped_out：可选输出（T1 审计用）——每章对应的源
    XHTML 片段 与 被跳过的短文档清单。
    verse_out：可选输出（T-L1）——每章的韵文块记录（供 verse_manifest）。
    """
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
        verse_blocks: list = []
        text = _epub_clean_html_text_ex(html_content, verse_blocks)
        # Q2：剔除古腾堡头部许可证/目录，避免非正文进入译文
        text = _strip_epub_front_matter(text)

        if not text or len(text) < 50:
            if skipped_out is not None:
                skipped_out.append({"id": item.get_name(), "chars": len(text or "")})
            continue

        if len(text) <= LARGE_DOC_THRESHOLD:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title = _epub_extract_title(soup, f"Chapter {chapter_idx}")
            _check_title_present(ch_id, title, text)
            if sources_out is not None:
                sources_out[ch_id] = [html_content]
            if verse_out is not None:
                verse_out[ch_id] = verse_blocks
            chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))
            continue

        lead_box: list = []
        sub_chapters = _epub_split_large_doc_by_toc(html_content, toc_entries, chapter_idx,
                                                    sources_out=sources_out,
                                                    verse_out=verse_out,
                                                    lead_out=lead_box)
        if sub_chapters:
            if lead_box and chapters:
                lead = lead_box[0]
                if len(lead.get('text') or '') >= 40:
                    prev_id, prev_md = chapters[-1]
                    chapters[-1] = (prev_id, prev_md.rstrip() + '\n\n' + lead['text'])
                    if sources_out is not None:
                        sources_out.setdefault(prev_id, []).append(lead['html'])
                    if verse_out is not None and lead.get('verse'):
                        verse_out.setdefault(prev_id, []).extend(lead['verse'])
                    _splitter_log_info(
                        f"接缝回填：{len(lead['text'])} 字符续文 → {prev_id}")
            chapters.extend(sub_chapters)
            chapter_idx += len(sub_chapters)
        else:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title = _epub_extract_title(soup, f"Chapter {chapter_idx}")
            _check_title_present(ch_id, title, text)
            if sources_out is not None:
                sources_out[ch_id] = [html_content]
            if verse_out is not None:
                verse_out[ch_id] = verse_blocks
            chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))

    return chapters


def _text_detect_chapter_title(line: str) -> bool:
    """判断一行是否为章节标题"""
    if len(line) > 80:
        return False
    for pat in _CHAPTER_TITLE_PATTERNS:
        if pat.match(line):
            return True
    return False


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
        self._chapter_sources: dict = {}
        self._skipped_docs: list = []
        self._chapter_verse: dict = {}
        self.verse_manifest: dict = {}
        self.audit_result: dict = {}
        self._excluded_ids: set = set()

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
        self.chapters = _epub_extract_chapters(
            self.book, sources_out=self._chapter_sources, skipped_out=self._skipped_docs,
            verse_out=self._chapter_verse)
        self._apply_input_prep()
        if not self.chapters:
            _splitter_log_warn("未提取到任何章节内容")
        else:
            _splitter_log_ok(f"共提取 {len(self.chapters)} 章")
            self._validate_split()
            _kept = [(cid, md) for cid, md in self.chapters
                     if cid not in self._excluded_ids]
            self.verse_manifest = _build_verse_manifest(_kept, self._chapter_verse)
            n_verse = sum(len(v or []) for v in (self.verse_manifest.get('chapters') or {}).values())
            if n_verse:
                _splitter_log_info(
                    f"韵文通道：检测到 {n_verse} 个韵文块（verse_manifest 随 write_files 落盘）")
        return self.chapters

    def _apply_input_prep(self):
        """input_prep：文本伪影修复 + 非正文标记（写盘时入 skip_dir，不参与翻译）。

        被排除章节仍保留在 self.chapters（完整性审计口径不变）。
        """
        cfg = _input_prep_cfg()
        if not cfg or cfg.get("enabled", True) is False:
            return
        repairs = cfg.get("repairs") or []
        if repairs:
            total = 0
            fixed = []
            for ch_id, md in self.chapters:
                md, n = _apply_input_repairs(md, repairs)
                total += n
                fixed.append((ch_id, md))
            self.chapters = fixed
            if total:
                _splitter_log_info(f"input_prep：文本伪影修复 {total} 处")
        skip_titles = {_prep_key(t) for t in (cfg.get("skip_titles") or []) if t}
        self._excluded_ids = set()
        if not skip_titles:
            return
        labels = []
        for ch_id, md in self.chapters:
            title = _chapter_h1(md)
            if title and _prep_key(title) in skip_titles:
                self._excluded_ids.add(ch_id)
                labels.append(f"{ch_id}（{title}）")
        if self._excluded_ids:
            _splitter_log_info(
                f"input_prep：排除 {len(self._excluded_ids)} 个非正文文档"
                f"（写 {cfg.get('skip_dir') or '_skip'}/，不翻译）: " + "、".join(labels))

    def _validate_split(self):
        """P4：切分完整性 + 段落结构回归校验。

        - T1：完整性改用独立 oracle token 覆盖率审计（run_conversion_audit），
          替换原「同源清洗器字符数」基线（该基线与被测逻辑同源，存在盲区）。
        - 段落结构：每章按空行拆出的段落数应>1（除非原文确实极短），
          防止再次出现"整章塌成 1 段"的回归。
        """
        try:
            self.audit_result = run_conversion_audit(
                self._chapter_sources, self.chapters, self._skipped_docs,
                self.output_dir / "_conversion_audit.md",
            )
            if self.audit_result["ok"]:
                _splitter_log_ok(
                    "转换完整性审计通过：覆盖率 "
                    f"{self.audit_result['overall'] * 100:.2f}% "
                    f"(报告: {self.audit_result['report_path']})")
            else:
                _splitter_log_err(
                    "转换完整性审计未通过：覆盖率 "
                    f"{self.audit_result['overall'] * 100:.3f}% "
                    f"(阈值 {_AUDIT_COVERAGE_THRESHOLD * 100:.1f}%)，"
                    f"详见 {self.audit_result['report_path']}")
        except Exception as e:
            # 审计异常按「未通过」处理（fail-closed，防盲区放行）
            self.audit_result = {"ok": False, "overall": 0.0,
                                 "report_path": "", "error": str(e)}
            _splitter_log_err(f"转换完整性审计异常（按未通过处理）: {e}")

        try:
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
        """将切分结果写入 output_dir/chXX.md，返回「参与翻译」的文件路径列表

        input_prep 标出的非正文文档改写到 output_dir/<skip_dir>/（不计入返回列表，
        CLI/mapfile 与翻译流程天然跳过）；同时清理上次切分遗留的陈旧 chXX.md。
        """
        if not self.chapters:
            self.split()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _cfg = _input_prep_cfg()
        _skip_dir_name = str((_cfg or {}).get("skip_dir") or "_skip")

        generated = []
        pairs = []  # [(ch_id, path)] 参与翻译的章节（供 summary 使用）
        for ch_id, markdown in self.chapters:
            if ch_id in self._excluded_ids:
                skip_dir = self.output_dir / _skip_dir_name
                skip_dir.mkdir(parents=True, exist_ok=True)
                (skip_dir / f"{ch_id}.md").write_text(markdown, encoding='utf-8')
                _splitter_log_info(
                    f"非正文排除: {_skip_dir_name}/{ch_id}.md（{len(markdown)} chars，不参与翻译）")
                continue
            out_file = self.output_dir / f"{ch_id}.md"
            out_file.write_text(markdown, encoding='utf-8')
            generated.append(str(out_file))
            pairs.append((ch_id, str(out_file)))
            _splitter_log_ok(f"{ch_id}.md ({len(markdown)} chars)")

        # 清理陈旧章节文件（同目录重切时的残留，如曾被排除但仍在根目录的 chXX）
        if _cfg and _cfg.get("enabled", True) is not False:
            root_kept = {ch_id for ch_id, _ in pairs}
            for stale in self.output_dir.glob("ch*.md"):
                if re.fullmatch(r'ch\d+', stale.stem) and stale.stem not in root_kept:
                    try:
                        stale.unlink()
                        _splitter_log_info(f"清理陈旧章节: {stale.name}")
                    except OSError:
                        pass
            skip_dir = self.output_dir / _skip_dir_name
            if skip_dir.is_dir():
                for stale in skip_dir.glob("ch*.md"):
                    if re.fullmatch(r'ch\d+', stale.stem) and stale.stem not in self._excluded_ids:
                        try:
                            stale.unlink()
                            _splitter_log_info(f"清理陈旧章节: {_skip_dir_name}/{stale.name}")
                        except OSError:
                            pass

        summary = self.output_dir / "_chapters_summary.txt"
        summary.write_text(
            "\n".join(
                f"{ch_id}\t{Path(f).stat().st_size} bytes"
                for ch_id, f in pairs
            ),
            encoding='utf-8'
        )
        _splitter_log_info(f"汇总: {summary}")

        if self.verse_manifest:
            manifest_file = self.output_dir / "verse_manifest.json"
            manifest_file.write_text(
                json.dumps(self.verse_manifest, ensure_ascii=False, indent=2),
                encoding='utf-8')
            n_blocks = sum(
                len(v or []) for v in (self.verse_manifest.get('chapters') or {}).values())
            _splitter_log_info(f"韵文清单: {manifest_file}（{n_blocks} 个韵文块）")
        return generated



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






# Section 11.5 — 译后命名一致性校正
# ────────────────────────────────────────────────────────────────────

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


def _scan_finals(output_dir: str, flat: bool = False) -> list:
    """扫描 final.json

    Args:
        output_dir: 根目录（默认嵌套模式：output/<chapter>/<final.json>）
        flat: True 时把 output_dir 当作单章目录（output_dir 下直接是 final.json）

    Returns:
        [{chapter, file, text, path}, ...]
    """
    finals = []
    root = Path(output_dir)
    if not root.exists():
        return finals

    if flat:
        # Round 7: 章节级 scope（pipeline 末尾自动校正用）—— 直接把 output_dir 视为单章目录
        chapter_label = root.name
        for f in sorted(root.iterdir()):
            if not f.is_file() or not f.name.endswith("_final.json"):
                continue
            data = robust_json_loads(f.read_text(encoding="utf-8"), expected_type=dict)
            if not data:
                print(f"  ⚠️ 跳过 {f}: JSON 解析失败", file=sys.stderr)
                continue
            text = data.get("text", "") if isinstance(data, dict) else str(data)
            finals.append({
                "chapter": chapter_label,
                "file": f.name,
                "text": text,
                "path": str(f),
            })
        return finals

    # 原嵌套模式：output/<chapter>/*_final.json
    for ch_dir in sorted(root.iterdir()):
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
    # Round 119 (T6): 尾注区豁免 — 段落首行为上标注（¹²³…）/"注："前缀时整段豁免
    # （长注释 200 字符 pre 窗口够不到段首"脚注"字样，D8 ch08 误报即此根因）
    para_start = line_start + 1
    while para_start > 0:
        prev_nl = para_start - 1
        prev_start = text.rfind("\n", 0, prev_nl) + 1
        if not text[prev_start:prev_nl].strip():
            break
        para_start = prev_start
    first_nl = text.find("\n", para_start)
    first_line = text[para_start: first_nl if first_nl != -1 else len(text)]
    if _NOTE_MARKER_RE.match(first_line):
        return True
    # Round 119 (T6): "（原文 …）" 括注内的英文源词附注豁免
    lp = text.rfind("（", 0, pos)
    if lp != -1:
        rp = text.find("）", lp + 1)
        if lp < pos < rp and text[lp + 1:rp].lstrip().startswith("原文"):
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
        # Round 105: 普通词护栏。variant_names 历史上被「一字之差」误发现污染
        # （东西/卡特/警察/伦敦金 等普通词被登记为人名异体）。盲目替换会改坏正文
        # （某样东西→某样黛西 / 卡特姆勋爵→卡洛姆勋爵）。命中普通词黑名单的
        # variant 一律跳过（不报告、不替换）。
        for canonical, info in entities.items():
            variants = info.get("variant_names", [])
            if not variants:
                continue
            for v in variants:
                if v not in text:
                    continue
                if _is_common_word_variant(v):
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


def run_consistency_check_for_chapter(
    chapter_id: str,
    dry_run: bool = False,
    db_path: str = "",
    output_root: str = "",
) -> int:
    """Round 7: 章节级 scope 的 consistency 校正（供 pipeline.run() 末尾调用）

    与 run_consistency_check 的区别：
      - scope 仅限 output/{chapter_id}/*_final.json（无嵌套扫描）
      - 报告写到 {chapter_dir}/consistency_diff.md（而非 output/consistency_diff.md）
      - 默认 dry_run=False（auto-trigger 时应直接 apply，CLI 子命令仍走 dry_run）

    Args:
        chapter_id: 章节 ID（如 'ch32'）
        dry_run: True 仅生成报告，False 直接修改 final.json
        db_path: decision_db.sqlite 路径（默认 _root/db/decision_db.sqlite）
        output_root: output 根目录（默认 _root/output）

    Returns:
        发现的 issues 数量（-1 表示失败）
    """
    _root = Path(__file__).resolve().parent.parent
    if not db_path:
        db_path = str(_root / "db" / "decision_db.sqlite")
    if not output_root:
        output_root = str(_root / "output")

    chapter_dir = Path(output_root) / chapter_id
    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}", file=sys.stderr)
        return -1
    if not chapter_dir.exists():
        print(f"⚠️ 章节目录不存在: {chapter_dir}（跳过 consistency 校正）")
        return 0

    print(f"🔧 [Consistency] 章节级 scope: {chapter_id}")
    entities = _load_glossary(db_path)
    decisions = _load_decisions(db_path)
    finals = _scan_finals(str(chapter_dir), flat=True)
    if not finals:
        print(f"    ⚠️ {chapter_dir} 无 final.json（跳过）")
        return 0
    print(f"    扫描 {len(finals)} 个 final.json")

    issues = _detect_issues(finals, entities, decisions)
    print(f"    发现 {len(issues)} 处不一致")

    # 报告写到 chapter 目录内（避免污染全局报告）
    report = _generate_report(finals, issues, dry_run=dry_run)
    report_path = chapter_dir / "consistency_diff.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"    → 报告: {report_path}")

    if not dry_run and issues:
        patched = _apply_issues(finals, issues)
        print(f"    ✏️ 已修正 {patched} 个文件")
    elif dry_run and issues:
        print(f"    ⏸️ DRY RUN：未修改文件")

    return len(issues)





# Section 13 — 镜像补全区（book_lint / verse / 拆分器与判据辅助）
# ────────────────────────────────────────────────────────────────────

_INLINE_THINK_RE = re.compile(r'<think(.*?)</think>', re.DOTALL)


_QUOTA_HINT_RE = re.compile(r"quota|insufficient|欠费|额度|余额", re.IGNORECASE)


def build_effort_params(api_base: str, model_name: str, effort: Optional[str]) -> Dict[str, Any]:
    if not effort:
        return {}
    eff = str(effort).strip().lower()
    payload: Dict[str, Any] = {"reasoning_effort": eff}
    target = f"{api_base or ''} {model_name or ''}".lower()
    if "minimax" in target:
        payload["thinking"] = {"type": "adaptive"}
    return payload


def entity_source_name_in_text(source_name: Optional[str], source_text: Optional[str]) -> bool:
    """Round 114: source_name 是否以"词/短语边界"形式出现在原文中（而非裸子串）。

    ch30 chunk000 根因: 实体 "伯特"←"Bert" 因原文 "Bertillon measurements" 含子串
    "Bert" 被误判"实体在场" → 实体核查要求译文出现「伯特」→ 不可满足 → 实体硬失败
    → rewriter 停滞 → PERMANENTLY_FAILED。同类: "Europe"⊂"European"、
    "Herzoslovakia"⊂"Herzoslovakian"、"drawing-room"⊂"drawing-rooms"。

    用 Unicode 字母 lookaround（而非 \\b）：兼容以标点结尾的短语（"Mr."/"H. I."）；
    所有格/连字符形式（"Bert's"/"Bert—"）仍视为在场。大小写敏感（英文原文与登记表
    形式一致）。
    """
    if not source_name or not source_text:
        return False
    return re.search(_entity_boundary_pattern(source_name), source_text) is not None


def entity_source_name_spans(
    source_name: Optional[str], source_text: Optional[str],
) -> List[Tuple[int, int]]:
    """Round 133b: entity_source_name_in_text 的 span 版本 — 返回全部命中 [start, end) 区间。

    供"更长别名吞并抑制"使用（chunk071 根因："Mother"(杰西卡夫人 的别名) 在原文中
    唯一出现于 "Reverend Mother"(盖乌斯·海伦·莫希阿姆 的别名) 内部 → 词边界防护
    放行 → 判杰西卡夫人在场 → 译文被判 missing → rewriter 收到不可执行指令 → 死循环）。
    语义与 entity_source_name_in_text 完全一致（大小写敏感、Unicode 字母边界）。
    """
    if not source_name or not source_text:
        return []
    return [
        (m.start(), m.end())
        for m in re.finditer(_entity_boundary_pattern(source_name), source_text)
    ]


def _entity_boundary_pattern(source_name: str) -> str:
    """Round 133b: entity source_name 的"词/短语边界"正则（与 Round 114 语义一致）。"""
    return r"(?<![^\W\d_])" + re.escape(source_name) + r"(?![^\W\d_])"


def _source_term_pattern(source_text: str):
    """构造「词边界」精筛正则（回溯触发精度）。

    - 仅字母参与边界（(?<![A-Za-z])/(?![A-Za-z])），避免 "well" 命中
      "dwelling"/"farewell" 之类子串；数字/CJK 不设边界（保护 "1984"、
      中文短语）。
    - 连字符与空白视作可互换分隔（变形容差：tea-party ≈ tea party）。
    - 大小写不敏感（源文大小写与决策条目常不一致）。
    - 大小写敏感：决策条目按原文摘录，大小写即语义（"Time" 不应命中
      任意小写 time；小写子串型伪匹配已由字母边界拦截）。唯一放宽：
      小写首字母词容忍句首大写（"please your Majesty" ≈ "Please ..."），
      仅放宽首字符、词内其余字符仍严格。
    返回 None 表示归一后为空。
    """
    norm = re.sub(r'\s+', ' ', (source_text or '').strip())
    raw_parts = [p for p in re.split(r'[\s\-]+', norm) if p]
    if not raw_parts:
        return None
    parts = [re.escape(p) for p in raw_parts]
    head = raw_parts[0][:1]
    if head.isalpha() and head.islower():
        parts[0] = (f'[{re.escape(head)}{re.escape(head.upper())}]'
                    f'{re.escape(raw_parts[0][1:])}')
    body = r'[\s\-]+'.join(parts)
    return re.compile(rf'(?<![A-Za-z]){body}(?![A-Za-z])')


SEED_ENTITY_VARIANTS: Dict[str, List[str]] = {}


_CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def load_entity_variant_seeds() -> Dict[str, List[str]]:
    """P1-4 读取器：config 段优先，为空回退 per-work 资源文件（缺失 = 无种子）。"""
    try:
        from src.config import get_config as _get_cfg
        cfg_variants = getattr(_get_cfg(), "entity_variants", None)
        if isinstance(cfg_variants, dict) and cfg_variants:
            return cfg_variants
    except Exception:
        pass
    try:
        from pathlib import Path
        import yaml as _yaml
        seed_path = Path("input/entity_variants.yaml")
        if seed_path.exists():
            data = _yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return {str(k): list(v or []) for k, v in data.items()}
    except Exception:
        pass
    return {}


def unify_entity_registry(conn, dry_run: bool = True) -> Dict[str, Dict]:
    """Round 105 P2-2: 归一 entity_registry 中「同人多规范名」冲突。

    背景（ch32 倒查根因）：同一人物被登记到多个 canonical（George Lomax →
    乔治·洛马克斯阁下 / 洛马克斯先生 / 乔治·洛马克斯先生），运行时 missing 核查
    只认 pending_review=0 的那一个，rewriter 用其它形式即被判 missing → 硬失败死循环。

    策略：
      1. 按共享 source_name 聚类（共享 ≥1 个 source_name 视为同一人）。
      2. 每个 ≥2 成员的簇选一个 winner：
         优先级 confidence(high>medium>low) → 更早 created_at → 更长规范名 → 字典序。
      3. 其余成员降级 pending_review=1（退出运行时核查，仍保留供 sibling 查找）。

    Args:
        conn: sqlite 连接（entity_registry 表）
        dry_run: True 仅返回计划不写库；False 实际 UPDATE。

    Returns:
        {cluster_repr: {"winner": canonical, "demoted": [canonical, ...]}}
        仅包含被处理的（≥2 成员）簇。
    """
    import json as _json

    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, canonical_name, source_names, confidence, created_at, pending_review "
        "FROM entity_registry"
    )
    rows = cursor.fetchall()

    # 聚类：canonical -> set(专名 source_names)
    # Round 105: 只用「含大写字母」的 source_name 聚类（专名/人名），排除
    # "a gentleman"/"the young man"/"the other"/"town" 等通用短语——否则并查集
    # 会经通用指称把不同人物传递性合并（ch32 实测 绅士乔 经 安东尼·凯德 误并入
    # 男爵洛洛普雷特济尔 簇）。
    canon_srcs: Dict[str, set] = {}
    meta: Dict[str, dict] = {}
    for rid, canon, srcs_json, conf, created, pending in rows:
        try:
            srcs = _json.loads(srcs_json) if srcs_json else []
        except (ValueError, TypeError):
            srcs = []
        specific = {
            s for s in srcs
            if isinstance(s, str) and s and any(ch.isupper() for ch in s)
        }
        canon_srcs.setdefault(canon, set()).update(specific)
        meta[canon] = {
            "id": rid, "confidence": conf or "low",
            "created": created or "", "pending": pending or 0,
        }

    # 并查集聚类（共享 source_name 连通）
    parent = {c: c for c in canon_srcs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    canons = list(canon_srcs)
    for i in range(len(canons)):
        for j in range(i + 1, len(canons)):
            if canon_srcs[canons[i]] & canon_srcs[canons[j]]:
                union(canons[i], canons[j])

    clusters: Dict[str, list] = {}
    for c in canons:
        clusters.setdefault(find(c), []).append(c)

    plan: Dict[str, Dict] = {}
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        # 选 winner：confidence → 去括注 base 长度（更长=更具体）→ 更早 created_at → 字典序
        import re as _re_u
        def _base_len(c):
            return len(_re_u.sub(r"（[^）]*）|\([^)]*\)", "", c).strip())
        def _key(c):
            m = meta[c]
            return (
                _CONF_ORDER.get(m["confidence"], 3),
                -_base_len(c),
                m["created"],
                c,
            )
        members_sorted = sorted(members, key=_key)
        winner = members_sorted[0]
        demoted = [c for c in members_sorted[1:] if meta[c]["pending"] == 0]
        if not demoted:
            continue
        plan[" / ".join(sorted(members))] = {"winner": winner, "demoted": demoted}
        if not dry_run:
            for c in demoted:
                cursor.execute(
                    "UPDATE entity_registry SET pending_review = 1, "
                    "review_reason = 'Round105 unify: 同人多规范名降级 (winner=' || ? || ')' "
                    "WHERE id = ? AND pending_review = 0",
                    (winner, meta[c]["id"]),
                )
    if not dry_run:
        conn.commit()
    return plan


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
    # 兜底 1（Round 126）: 非 SDK 异常 / SDK 缺失时按错误消息字符串分类。
    # 与原单体嵌套版（string-matching）逻辑对齐，保留 retryable 语义——
    # Raw HTTP 错误（requests/httpx 抛出的 429/401/超时等）借此仍能正确归类。
    # 只用消息文本匹配（不含 repr/类名）：类名如 TypeSafeAuthenticationError
    # 会让 "authentication" 关键字在 SDK 不可用的退化路径上误命中（Round 53 测试锁定）。
    combined = str(exc).lower()
    if any(k in combined for k in ("429", "rate limit", "rate_limit", "quota", "too many requests")):
        return {"category": "rate_limit", "retryable": True, "msg": str(exc)[:200], "action": "wait or upgrade quota"}
    if any(k in combined for k in ("timeout", "timed out", "connection reset", "connection aborted")):
        return {"category": "timeout", "retryable": True, "msg": str(exc)[:200], "action": "increase timeout or retry"}
    if any(k in combined for k in ("json", "decode", "parse", "invalid response", "malformed")):
        return {"category": "parse", "retryable": False, "msg": str(exc)[:200], "action": "check response format"}
    # auth：短语直配；裸 401/403 仅在与 HTTP/认证上下文共现时才算（真实 httpx/requests
    # 报文必带 URL / "Unauthorized" 字样）。避免退化路径上只含裸状态码的 SDK 异常文本
    # （如 "401 won't match in degraded mode"）误命中 auth（Round 53 测试锁定）。
    _auth_phrase = any(k in combined for k in (
        "unauthorized", "forbidden", "api key", "invalid api",
        "authentication", "authenticate", "authorization",
    ))
    _auth_code_ctx = bool(re.search(r"\b(401|403)\b", combined)) and any(
        k in combined for k in ("client error", "url:", "url ", "http", "status")
    )
    if _auth_phrase or _auth_code_ctx:
        return {"category": "auth", "retryable": False, "msg": str(exc)[:200], "action": "check TYPESAFE_API_KEY"}
    if any(k in combined for k in ("500", "502", "503", "504", "internal server", "bad gateway", "service unavailable")):
        return {"category": "server", "retryable": True, "msg": str(exc)[:200], "action": "check typesafe status page"}
    if any(k in combined for k in ("connection", "dns", "network", "unreachable", "ssl", "certificate")):
        return {"category": "network", "retryable": True, "msg": str(exc)[:200], "action": "check DNS/firewall/cert"}
    # 兜底 2：未分类异常
    return {"category": "unknown", "retryable": True, "msg": f"{type(exc).__name__}: {exc}", "action": "inspect stack trace"}


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


_MAX_WIDENED_ZONE_HIGH = 0.98


_SOURCE_MARGIN_CONFIDENCE_KEYS = (
    "style_compliance_min_confidence",
    "low_confidence_threshold",
    "cjk_low_confidence_threshold",
)


_SOURCE_MARGIN_THRESHOLD_KEYS = (
    "voice_drift_severe_prob",
    "register_mismatch_prob",
    "entity_consistency_noul",
    "literal_residue_noul",
    "semantic_loss_noul",
)


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


_NOTE_MARKER_RE = re.compile(r"^\s*(?:[¹²³⁴⁵⁶⁷⁸⁹]+|注\s*[:：])")


def _is_common_word_variant(variant: str, canonical: str = None) -> bool:
    """Round 105: 判定 variant 是否为普通词/假异体（不应作为人名异体替换）。

    复用 src.utils.entity_stopwords 黑名单 + 一字之差高频差异字护栏。
    导入失败（单文件部署）时退化为仅拦截已知 ch32 污染项，保证安全。

    ch06 倒查：variant_names 被「一字之差」误发现污染（法国→外国/美国/帝国/法庭，
    绅士乔→绅士一/绅士可，楼梯→楼上/楼下/楼去）。这些假异体的共同特征是
    变体侧差异字为高频字（一/外/去/上/下…），而真名异体的差异字偏冷僻。
    故在黑名单之外，追加「一字之差 + 变体侧差异字 ∈ 高频字集」护栏。
    """
    try:
        from src.utils.entity_stopwords import is_likely_common_word
        return is_likely_common_word(variant)
    except Exception:
        # 兜底黑名单（与 entity_stopwords.COMMON_WORDS 的 ch32 污染项一致）
        _FALLBACK = {
            "东西", "卡特", "警察", "警督", "警已", "警方", "警惕", "警司", "警官",
            "材敦", "普敦", "伦次",
            "伦敦金", "伦敦来", "伦敦最", "伦敦高", "伦敦时", "伦敦警", "伦敦可",
        }
        return variant in _FALLBACK


def consolidate_dirty_chapters(
    project_root=None,
    max_rounds: int = 3,
    pipeline_factory=None,
    db_path=None,
    output_dir=None,
) -> dict:
    """T2: 全书决策收敛闭环 — 重跑所有含 DIRTY chunk 的章节直到收敛。

    D2/D3 根因: 章节串行执行时，已结束章节上的回溯标记（DIRTY）没有
    消费点，全书跑完后 merge 会把"回溯前"陈旧 final 收录进成书。
    本函数在全书写完之后统一收口: 扫描 workflow.db 的 DIRTY chunk →
    按章序重跑该章管线（DIRTY 阶段在阶段表首位）→ 重扫；循环到
    DIRTY=0 或 max_rounds。每个重跑 chunk 的旧/新 final 文本差异写入
    output/consolidation_diff.md（旧稿在重跑前采样，不依赖 backup 存活）。

    Returns:
        {
          "converged": bool,     # 最终 DIRTY=0
          "rounds": int,         # 实际执行轮数（无 DIRTY 时为 0）
          "rerun": [...],        # 重跑过的 chunk（按发生顺序）
          "remaining": {...},    # 未收敛残留 {chapter: [chunk_id]}
          "diff_path": str|None,
        }
    """
    import difflib
    from datetime import datetime

    rounds_cap = max(1, int(max_rounds))
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    db_file = Path(db_path) if db_path else root / "db" / "workflow.db"
    out_root = Path(output_dir) if output_dir else root / "output"

    result = {"converged": True, "rounds": 0, "rerun": [], "remaining": {}, "diff_path": None}
    if not db_file.exists():
        print(f"⚠️ [Consolidate] workflow.db 不存在（{db_file}），跳过收敛检查")
        return result

    if pipeline_factory is None:
        pipeline_factory = lambda ch: TranslationPipeline(chapter_id=ch)

    scheduler = TaskScheduler(db_path=str(db_file))

    def _dirty_rows():
        cursor = scheduler.conn.execute(
            "SELECT chapter_id, chunk_id FROM chunk_tasks WHERE state = ? "
            "ORDER BY chapter_id ASC, chunk_id ASC",
            (TaskState.DIRTY.value,),
        )
        return [tuple(row) for row in cursor.fetchall()]

    def _read_final_text(chapter_id, chunk_id):
        path = out_root / chapter_id / f"{chunk_id}_final.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data.get("text", "") if isinstance(data, dict) else str(data)

    diff_records = []
    while result["rounds"] < rounds_cap:
        rows = _dirty_rows()
        if not rows:
            break
        result["rounds"] += 1
        round_no = result["rounds"]
        by_chapter = {}
        for chapter_id, chunk_id in rows:
            by_chapter.setdefault(chapter_id, []).append(chunk_id)
        print(f"🔁 [Consolidate] 第 {round_no}/{rounds_cap} 轮: "
              f"{len(rows)} 个 DIRTY chunk, {len(by_chapter)} 章待收敛")
        for chapter_id in sorted(by_chapter):
            chunk_ids = by_chapter[chapter_id]
            pre_texts = {cid: _read_final_text(chapter_id, cid) for cid in chunk_ids}
            try:
                ok = bool(pipeline_factory(chapter_id).run())
            except Exception as exc:
                ok = False
                print(f"  ⚠️ [Consolidate] 章节 {chapter_id} 重跑异常: {exc}")
            if not ok:
                print(f"  ⚠️ [Consolidate] 章节 {chapter_id} 重跑未完全通过（fallback 已兜底的可能）")
            for cid in chunk_ids:
                diff_records.append({
                    "round": round_no,
                    "chapter": chapter_id,
                    "chunk": cid,
                    "old": pre_texts.get(cid),
                    "new": _read_final_text(chapter_id, cid),
                })
                result["rerun"].append(cid)

    remaining_rows = _dirty_rows()
    result["converged"] = len(remaining_rows) == 0
    for chapter_id, chunk_id in remaining_rows:
        result["remaining"].setdefault(chapter_id, []).append(chunk_id)

    lines = [
        "# 全书决策收敛报告 (consolidation_diff)",
        "",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- 轮次: {result['rounds']} / max_rounds={rounds_cap}",
        f"- 重跑 chunk: {len(result['rerun'])} 个",
        f"- 最终状态: {'✅ 收敛 (DIRTY=0)' if result['converged'] else '❌ 未收敛'}",
        "",
    ]
    if result["remaining"]:
        lines.append("## 未收敛残留")
        lines.append("")
        for chapter_id in sorted(result["remaining"]):
            lines.append(f"- {chapter_id}: {', '.join(result['remaining'][chapter_id])}")
        lines.append("")
    if not diff_records:
        lines.append("（本次扫描 DIRTY=0，无重跑）")
        lines.append("")
    else:
        lines.append("## 重跑明细")
        lines.append("")
        for rec in diff_records:
            lines.append(f"### [{rec['chapter']}/{rec['chunk']}] (轮 {rec['round']})")
            old_text, new_text = rec["old"], rec["new"]
            if old_text is None and new_text is None:
                lines.append("- 旧/新 final 均缺失（重跑失败且无 backup 可恢复）")
            elif old_text is None:
                lines.append(f"- 无旧稿对照；新稿 {len(new_text)} 字")
            elif new_text is None:
                lines.append(f"- 重跑无新稿（已回滚旧稿）；旧稿 {len(old_text)} 字")
            elif old_text == new_text:
                lines.append(f"- 文本无变化（仅决策核销）；{len(new_text)} 字")
            else:
                lines.append(f"- 文本变化: {len(old_text)} 字 → {len(new_text)} 字 "
                             f"(Δ{len(new_text) - len(old_text):+d})")
                old_lines = old_text.splitlines()
                new_lines = new_text.splitlines()
                preview = []
                for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes():
                    if tag == "equal":
                        continue
                    for ln in old_lines[i1:i2][:3]:
                        preview.append("- " + ln[:80])
                    for ln in new_lines[j1:j2][:3]:
                        preview.append("+ " + ln[:80])
                    if len(preview) >= 8:
                        break
                lines.append("- 差异预览:")
                lines.append("")
                lines.append("```diff")
                lines.extend(preview[:8])
                lines.append("```")
            lines.append("")
    lines.append("---")
    lines.append("*由 consolidate 子命令自动生成*")
    lines.append("")

    diff_path = out_root / "consolidation_diff.md"
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        diff_path.write_text("\n".join(lines), encoding="utf-8")
        result["diff_path"] = str(diff_path)
    except OSError as exc:
        print(f"⚠️ [Consolidate] 报告写入失败: {exc}")

    if result["converged"]:
        print(f"✅ [Consolidate] 收敛完成: DIRTY=0 "
              f"(重跑 {len(result['rerun'])} chunk, {result['rounds']} 轮)")
    else:
        total_left = sum(len(v) for v in result["remaining"].values())
        print(f"❌ [Consolidate] 未收敛: 仍余 {total_left} 个 DIRTY chunk"
              f"（max_rounds={rounds_cap} 已用尽）")
        for chapter_id in sorted(result["remaining"]):
            print(f"   {chapter_id}: {', '.join(result['remaining'][chapter_id])}")
    return result


def run_final_gate(
    project_root=None,
    db_path=None,
    output_dir=None,
    qa_report_path=None,
    conversion_audit_path=None,
) -> dict:
    """T4: 交付门禁 — 完成信号不变量化（fail-closed）。

    ① 状态机：全部 chunk COMPLETED 且 DIRTY=0
    ② 产物：每章 final.json 数 == DB 任务数；无 *_final_backup.json 残留
    ③ QA_REPORT：无 fallback 章节条目（## 段）
    ④ T5 lint（finals + merged）全绿
    ⑤ T1 转换审计报告结论为 "✅ 通过"
    ⑥ L1 韵文对齐（verse_lint，门禁内联自动运行）
    ⑦ L2 双关核销（pun_audit，门禁内联自动运行；Round 130 起无需手动补跑）
    任一为红 → ok=False（translate_book.sh 以非零退出中止 / ⛔）。
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    db_file = Path(db_path) if db_path else root / "db" / "workflow.db"
    out_root = Path(output_dir) if output_dir else root / "output"
    qa_file = Path(qa_report_path) if qa_report_path else out_root / "QA_REPORT.md"
    audit_file = (Path(conversion_audit_path) if conversion_audit_path
                  else root / "input" / "_conversion_audit.md")

    failures: list = []
    warnings: list = []
    stats: dict = {"tasks": 0, "states": {}, "chapters": 0, "qa_fallbacks": 0}
    by_state: dict = {}
    expected: dict = {}

    # ① 状态机 + ② 产物
    rows = []
    scheduler = None
    if not db_file.exists():
        failures.append(f"① workflow.db 不存在: {db_file}")
    else:
        scheduler = TaskScheduler(db_path=str(db_file))
        rows = scheduler.conn.execute(
            "SELECT chapter_id, chunk_id, state FROM chunk_tasks").fetchall()
    stats["tasks"] = len(rows)
    for ch, _cid, st in rows:
        by_state[st] = by_state.get(st, 0) + 1
        expected[ch] = expected.get(ch, 0) + 1
    stats["states"] = by_state
    stats["chapters"] = len(expected)

    dirty_n = by_state.get(TaskState.DIRTY.value, 0)
    if dirty_n:
        failures.append(f"① DIRTY={dirty_n}（存在未整合脏块，先跑 consolidate）")
    non_completed = [(cid, st) for _ch, cid, st in rows if st != TaskState.COMPLETED.value]
    if non_completed:
        sample = ", ".join(f"{cid}:{st}" for cid, st in non_completed[:5])
        failures.append(f"① 非 COMPLETED 任务 {len(non_completed)} 个（如 {sample}）")

    for ch, exp in sorted(expected.items()):
        finals = list((out_root / ch).glob("*_final.json"))
        if len(finals) != exp:
            failures.append(f"② {ch}: final.json {len(finals)} 个 != DB 任务 {exp} 个")
    backups = [str(p.relative_to(root)) for p in out_root.glob("ch*/*_final_backup.json")]
    if backups:
        failures.append(f"② 残留 final_backup {len(backups)} 个: {backups[:5]}")

    # ③ QA_REPORT
    if not qa_file.exists():
        failures.append(f"③ QA_REPORT 不存在: {qa_file}")
    else:
        falls = [ln[3:].strip() for ln in qa_file.read_text(encoding="utf-8").split("\n")
                 if ln.startswith("## ")]
        stats["qa_fallbacks"] = len(falls)
        if falls:
            failures.append(f"③ QA_REPORT 记录 {len(falls)} 个 fallback 章节: {falls[:5]}")

    # ④ T5 lint
    try:
        from src.book_lint import lint_book
        lint_res = lint_book(project_root=str(root))
        stats["lint_failures"] = len(lint_res.get("failures", []))
        if not lint_res.get("ok"):
            failures.append(
                f"④ lint 未通过（{stats['lint_failures']} 项，详见 output/book_lint.md）")
    except Exception as exc:
        stats["lint_failures"] = -1
        failures.append(f"④ lint 执行异常: {exc}")

    # ⑤ T1 转换审计
    if not audit_file.exists():
        failures.append(f"⑤ 转换审计报告不存在: {audit_file}")
    else:
        audit_text = audit_file.read_text(encoding="utf-8")
        cov = re.search(r'全书覆盖率:\s*([\d.]+)%', audit_text)
        if cov:
            stats["audit_coverage"] = float(cov.group(1))
        if "✅ 通过" not in audit_text:
            failures.append(f"⑤ 转换审计未通过（详见 {audit_file}）")

    # ⑥ L1 韵文对齐（verse_lint；manifest 无韵文块或功能关闭时跳过）
    try:
        from src.verse import (load_verse_manifest, load_chunk_tags,
                               fragments_for_chunk, verse_lint, verse_config)
        _vcfg = verse_config()
        _vmani = load_verse_manifest(root / "input")
        _vtags = load_chunk_tags(root / "db")
        _vblocks_n = sum(len(v or []) for v in (_vmani.get("chapters") or {}).values())
        verse_stats = {"verse_blocks": _vblocks_n, "checked": 0, "failed": 0}
        stats["verse"] = verse_stats
        if _vcfg.get("enabled", True) and _vblocks_n:
            _vtext_rows = []
            if scheduler is not None:
                _vtext_rows = scheduler.conn.execute(
                    "SELECT chunk_id, chapter_id, text_content FROM chunk_tasks").fetchall()
            _vfail = []
            _vreport = ["# 韵文对齐报告 (verse_lint)", "",
                        f"- 韵文块总数: {_vblocks_n}", ""]
            for _vcid, _vch, _vtext in _vtext_rows:
                _vfinal = out_root / _vch / f"{_vcid}_final.json"
                if not _vfinal.exists():
                    continue
                _vfrags = fragments_for_chunk(_vch, _vcid, _vtext or "", _vtags, _vmani)
                if not _vfrags:
                    continue
                try:
                    _vdata = json.loads(_vfinal.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                _vftext = _vdata.get("text", "") if isinstance(_vdata, dict) else str(_vdata)
                _vres = verse_lint(_vfrags, _vftext,
                                   min_line_ratio=float(_vcfg.get("min_line_ratio", 0.6)),
                                   max_line_chars=int(_vcfg.get("max_line_chars", 56)),
                                   refrain_ratio=float(_vcfg.get("refrain_ratio", 0.5)))
                verse_stats["checked"] += 1
                if _vres["ok"]:
                    _vreport.append(
                        f"- ✅ {_vcid}: lines {_vres['found_lines']}/{_vres['expected_lines']}, "
                        f"refrain {_vres['refrain_target']}/{_vres['refrain_source']}")
                else:
                    verse_stats["failed"] += 1
                    _vreport.append(f"- ❌ {_vcid}: {_vres['reason']}")
                    _vfail.append(f"{_vcid}({_vres['reason']})")
            if verse_stats["checked"] == 0 and _vblocks_n:
                failures.append(
                    "⑥ 韵文清单非空但未匹配到任何 chunk（manifest/tags 可能陈旧，需重新 split+init）")
            if _vfail:
                failures.append(f"⑥ verse_lint 未达标 {len(_vfail)} 个 chunk: {_vfail[:5]}")
            (out_root / "verse_lint.md").write_text(
                "\n".join(_vreport) + "\n", encoding="utf-8")
    except Exception as _v_exc:
        stats["verse"] = {"error": str(_v_exc)}
        failures.append(f"⑥ verse_lint 执行异常: {_v_exc}")
    # ⑦ L2 双关核销（pun_audit；watchlist 为空时跳过）
    try:
        from src.style_profiler import load_profile as _load_profile
        _prof = _load_profile(root / "input" / "book_profile.json")
        _watch = (_prof or {}).get("pun_watchlist") or []
        if _watch:
            _pun_json = out_root / "pun_audit.json"
            # Round 130（外部审计②）: 门禁内联自动运行 pun_audit（纯静态、无 LLM 调用），
            # 消除"必须手动先跑 scripts/pun_audit.py，否则 gate ⑦ 必红"的顺序耦合。
            _pun_script = root / "scripts" / "pun_audit.py"
            if _pun_script.exists():
                import subprocess as _subprocess
                try:
                    _pun_proc = _subprocess.run(
                        [sys.executable, str(_pun_script),
                         "--project-root", str(root), "--no-fail"],
                        cwd=str(root), capture_output=True, text=True, timeout=120)
                    if _pun_proc.returncode != 0:
                        warnings.append(
                            f"⑦ pun_audit 自动执行异常(rc={_pun_proc.returncode}): "
                            f"{(_pun_proc.stderr or _pun_proc.stdout or '').strip()[:200]}")
                except Exception as _pun_run_exc:
                    failures.append(
                        f"⑦ pun_audit 自动执行失败: {type(_pun_run_exc).__name__}: {_pun_run_exc}")
            if not _pun_json.exists():
                failures.append("⑦ pun_audit 未运行（缺 output/pun_audit.json；自动补跑不可用，请手动跑 scripts/pun_audit.py）")
            else:
                _pun_report = json.loads(_pun_json.read_text(encoding="utf-8"))
                _pun_sum = _pun_report.get("summary") or {}
                stats["pun"] = _pun_sum
                _pun_miss = int(_pun_sum.get("uncovered_nonwaived", 0) or 0)
                if _pun_miss:
                    failures.append(f"⑦ pun_watchlist 未核销 {_pun_miss} 项（详见 output/pun_audit.md）")
    except Exception as _p_exc:
        failures.append(f"⑦ pun 审计执行异常: {_p_exc}")

    if stats["tasks"] == 0 and db_file.exists():
        failures.append("① workflow.db 无任务行（未 init？）")

    result = {"ok": not failures, "failures": failures, "warnings": warnings, "stats": stats}
    print("🎯 [Gate] 交付门禁检查：")
    print(f"  ① 状态机: {stats['tasks']} 个任务, states={by_state}")
    print(f"  ② 产物: {stats['chapters']} 章, final_backup={len(backups)}")
    print(f"  ③ QA_REPORT: fallback {stats['qa_fallbacks']} 章")
    print(f"  ④ lint: failures={stats.get('lint_failures')}")
    if stats.get("audit_coverage") is not None:
        print(f"  ⑤ 转换审计: 覆盖率 {stats['audit_coverage']}%")
    _vs = stats.get("verse") or {}
    if _vs.get("error"):
        print(f"  ⑥ 韵文: 异常 {_vs['error']}")
    elif _vs:
        print(f"  ⑥ 韵文: blocks={_vs.get('verse_blocks', 0)}, "
              f"checked={_vs.get('checked', 0)}, failed={_vs.get('failed', 0)}")
    _ps = stats.get("pun") or {}
    if _ps:
        print(f"  ⑦ 双关核销: covered={_ps.get('covered', 0)}/{_ps.get('total', 0)}, "
              f"waived={_ps.get('waived', 0)}, uncovered={_ps.get('uncovered_nonwaived', 0)}")
    for f in failures:
        print(f"  ❌ {f}")
    if result["ok"]:
        print("  ✅ gate 全绿（可交付）")
    return result


_INSTANCE_LOCK_FD: Optional[int] = None


_MUTATING_COMMANDS = {"pipeline", "init", "split", "style-profile",
                      "consolidate", "finalize", "notes-normalize"}


def acquire_instance_lock(lock_path: Optional[Path] = None) -> bool:
    """非阻塞获取单实例锁。

    Returns:
        True  — 已持有（或由祖先进程持有 / 放行开关生效），可继续执行；
        False — 已有另一实例在运行，调用方应中止。
    """
    global _INSTANCE_LOCK_FD
    if os.environ.get("SAGATRANS_LOCK_HELD") == "1":
        return True
    import fcntl
    import time as _time
    path = Path(lock_path) if lock_path else Path.cwd() / "db" / ".sagatrans.lock"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        print(f"⚠️ [Lock] 无法创建锁文件 {path}: {e}（继续执行）", file=sys.stderr)
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            holder = path.read_text(encoding="utf-8").strip().replace("\n", " ") or "(无持有者信息)"
        except OSError:
            holder = "(无持有者信息)"
        try:
            os.close(fd)
        except OSError:
            pass
        if os.environ.get("SAGATRANS_ALLOW_CONCURRENT") == "1":
            print(f"⚠️ [Lock] 已有 SagaTrans 实例在运行（{holder}）——"
                  f"SAGATRANS_ALLOW_CONCURRENT=1 放行", file=sys.stderr)
            return True
        print(f"⛔ [Lock] 已有 SagaTrans 实例在运行：{holder}", file=sys.stderr)
        print(f"   锁文件: {path}；确认无残留进程后可设 SAGATRANS_ALLOW_CONCURRENT=1 强制放行（风险自担）",
              file=sys.stderr)
        return False
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (f"pid={os.getpid()} started={_time.strftime('%F %T')} "
                      f"cmd={' '.join(sys.argv[1:]) or 'pipeline'}\n").encode("utf-8"))
    except OSError:
        pass
    _INSTANCE_LOCK_FD = fd  # 进程存活期间保持打开；fd 随进程退出关闭 → 自动释放锁
    return True


_AUDIT_CHAPTER_FLOOR = 0.90


_AUDIT_COVERAGE_THRESHOLD = 0.995


_AUDIT_MAX_MISSING_SPANS = 20


_AUDIT_TOC_SCAN_LINES = 120


_AUDIT_TOKEN_RE = re.compile(r'[A-Za-z0-9]+|[\u4e00-\u9fff]')


_FALLBACK_BLOCK_TAGS = ['blockquote', 'li', 'dd', 'dt', 'tr', 'table']


_PRIMARY_BLOCK_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre']


_VERSE_CLASS_RE = re.compile(r'poem|stanza|verse|song|lyric|epigraph', re.IGNORECASE)


_CHAPTER_TITLE_PATTERNS = [
    re.compile(r'^\s*(?:Chapter|CHAPTER|第[一二三四五六七八九十百千零\d]+章|序章|楔子|引子|前言|后记|尾声)\s*[\s\d.、:：\-—]*[^\n]*$', re.IGNORECASE),
    re.compile(r'^\s*\[(?:CHAPTER|Chapter|第[一二三四五六七八九十百千零\d]+章)[^\]]*\]\s*$', re.IGNORECASE),
    re.compile(r'^\s*#{1,2}\s*[^\n]+$'),  # Markdown H1/H2
]


def _apply_input_repairs(text: str, repairs) -> tuple:
    """按 input_prep.repairs 表顺序替换文本伪影，返回 (新文本, 替换次数)。"""
    n = 0
    for item in repairs or []:
        try:
            old, new = str(item[0]), str(item[1])
        except (TypeError, IndexError, KeyError):
            continue
        if old and old in text:
            n += text.count(old)
            text = text.replace(old, new)
    return text, n


def _chapter_h1(md: str) -> str:
    """取章节 Markdown 的 H1 标题（前几行内首个 '# ' 行）。"""
    for line in (md or "").splitlines()[:5]:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _input_prep_cfg() -> dict:
    """读取 config.yaml 的 input_prep 配置（不可用时返回 {} = 功能关闭）。"""
    try:
        from src.config import get_config as _gc
        return dict(_gc().input_prep or {})
    except Exception:
        return {}


def _prep_key(s: str) -> str:
    """标题归一（忽略大小写/多余空白）供 skip_titles 匹配。"""
    return " ".join((s or "").strip().casefold().split())


_VERSE_HEURISTIC_MAX_LINE = 40


_VERSE_HEURISTIC_MIN_BR = 2


_VERSE_HEURISTIC_SHORT_RATIO = 0.6


_VERSE_TAG_ATTR = 'data-sagatrans-verse'


def _audit_tokens_with_pos(text: str) -> list:
    return [(m.group(0), m.start()) for m in _AUDIT_TOKEN_RE.finditer(text)]


def _audit_whitelist_spans(text: str) -> list:
    """返回 oracle 文本中「策略豁免」的字符区间（许可证串 + 目录区域）。"""
    spans = []
    for pat in _GUTENBERG_HEADER_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))

    lines = text.split('\n')
    starts = []
    pos = 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1

    for i in range(min(len(lines), _AUDIT_TOC_SCAN_LINES)):
        if not _GUTENBERG_TOC_TITLE_PATTERN.search(lines[i]):
            continue
        j = i + 1
        items = 0
        end = j
        while j < len(lines):
            ln = lines[j]
            if ln.strip() == '':
                # 空行透明：其后仍紧跟目录项行才继续
                k = j + 1
                while k < len(lines) and lines[k].strip() == '':
                    k += 1
                if k < len(lines) and len(lines[k]) < 120 \
                        and _GUTENBERG_TOC_ITEM_PATTERN.search(lines[k]):
                    j = k
                    continue
                break
            if len(ln) >= 120:
                break
            if _GUTENBERG_TOC_ITEM_PATTERN.search(ln):
                items += 1
                j += 1
                end = j
                continue
            # 续行（如表格 <td> 分列后的标题行）：其后仍紧跟目录项行才纳入
            k = j + 1
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            if k < len(lines) and len(lines[k]) < 120 \
                    and _GUTENBERG_TOC_ITEM_PATTERN.search(lines[k]):
                j = k
                end = j
                continue
            break
        if items >= 2:
            # 末尾续行（表格末行标题后无下一项行）：前一行是目录项则一并纳入
            if end < len(lines):
                tail = lines[end]
                if tail.strip() and len(tail) < 120 \
                        and not _GUTENBERG_TOC_ITEM_PATTERN.search(tail) \
                        and end - 1 >= i and _GUTENBERG_TOC_ITEM_PATTERN.search(lines[end - 1]):
                    end += 1
            spans.append((starts[i], starts[end - 1] + len(lines[end - 1])))
            break
    return spans


def _block_element_text(el) -> str:
    """提取单个块级元素文本（pre/韵文保留行结构，其余以空格连接片段）。"""
    if el.name == 'pre' or el.get(_VERSE_TAG_ATTR):
        # 韵文/地址条：逐行 strip 保留行结构；内联标签（<i> 等）不切断行，
        # 故不带分隔符取文本（片段自带空格）。连续空行压成一个（保留诗节）。
        lines: list = []
        for raw in el.get_text('').split('\n'):
            ln = raw.strip()
            if not ln:
                if lines and lines[-1] != '':
                    lines.append('')
                continue
            lines.append(ln)
        while lines and lines[-1] == '':
            lines.pop()
        return '\n'.join(lines)
    return el.get_text(' ', strip=True)


def _build_verse_manifest(chapters: list, chapter_verse: dict) -> dict:
    """T-L1：由各章韵文块记录构建 input/verse_manifest.json 结构。

    逐块在章节 md 中按序定位（字符偏移 → 行号）；定位失败的块跳过并告警。
    """
    chapters_out: dict = {}
    for ch_id, md in chapters:
        blocks = chapter_verse.get(ch_id) or []
        entries: list = []
        cursor = 0
        for b in blocks:
            t = re.sub(r'[ \t]+', ' ', b.get('text') or '')
            if not t.strip():
                continue
            pos = md.find(t, cursor)
            if pos < 0:
                pos = md.find(t)
            if pos < 0:
                _splitter_log_warn(
                    f"{ch_id} 韵文块未在 md 中定位（跳过）: {t[:30]!r}")
                continue
            cursor = pos + len(t)
            start_line = md.count('\n', 0, pos) + 1
            end_line = md.count('\n', 0, pos + len(t)) + 1
            line_count = sum(1 for ln in t.split('\n') if ln.strip())
            entries.append({
                'tag': b.get('tag', 'unknown'),
                'start_line': start_line,
                'end_line': end_line,
                'start_char': pos,
                'end_char': pos + len(t),
                'line_count': line_count,
                'text': t,
            })
        chapters_out[ch_id] = entries
    return {'version': 1, 'chapters': chapters_out}


def _check_title_present(ch_id: str, title: str, text: str) -> None:
    """T1：H1 与正文标题一致性（信息性告警）。

    标题未在正文开头复现时记录 INFO（通常意味着标题来自页眉/元数据，
    属「页眉残留」类问题线索），不阻断切分。
    """
    normalized = _normalize_chapter_title(title)
    if normalized and normalized[:60] not in text[:400]:
        _splitter_log_info(
            f"{ch_id} 标题未在正文再现（可能来自页眉/元数据）: {normalized}")


def _el_verse_tag(el) -> Optional[str]:
    """元素的韵文来源标签：class/启发式 标记 或 pre（多行时）。"""
    tag = el.get(_VERSE_TAG_ATTR)
    if tag:
        return str(tag)
    if el.name == 'pre':
        return 'pre'
    return None


def _epub_clean_html_text_ex(html_content: str, verse_out: Optional[list] = None) -> str:
    """清洗 HTML，提取纯文本（保留段落结构）。

    P 方案修复：原先对每个块级标签 insert_after('\\n') 再用
    get_text(separator='\\n')，因源 XML 段内自带换行，最终整章塌成 1 段。
    改为逐个提取块级元素文本、以 '\\n\\n' 拼接，确定性还原段落边界，
    且对段内换行免疫。
    T1：块级口径扩展为 p/标题/pre（保留韵文行结构）+ 容器类兜底提取。
    T-L1：先标记韵文元素；verse_out 非 None 时收集多行韵文块。
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    _mark_verse_elements(soup)

    for br in soup.find_all('br'):
        br.replace_with('\n')

    text = _extract_block_text(soup, verse_out)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 注意：不能用 \s（含 \n）配合 MULTILINE 去修整行首尾，否则会把段落间的
    # \n\n 也吃掉、重新塌成 1 段。仅修剪行首尾的空格/制表符。
    text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', text)
    return text.strip()


def _epub_extract_title(soup, fallback_title: str) -> str:
    """提取章节标题：优先正文标题标签（h1-h3），退回 <title> 或兜底名。

    T1：<br/> 分隔的标题先在标签内替换为空格；get_text 用空格连接片段，
    避免「CHAPTER I.<br/>Down…」粘连成 "I.Down" 或「…eBook of
    <span>Alice's…」粘连成 "ofAlice's"。
    """
    title_tag = soup.find(['h1', 'h2', 'h3']) or soup.find('title')
    if not title_tag:
        return fallback_title
    for br in title_tag.find_all('br'):
        br.replace_with(' ')
    return title_tag.get_text(' ', strip=True)


def _epub_oracle_text(html_content: str) -> str:
    """独立 oracle 文本：仅去除非正文容器/元数据，其余原样取文本。"""
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'title']):
        tag.decompose()
    return soup.get_text('\n')


def _extract_block_text(root, verse_out: Optional[list] = None) -> str:
    """按块级元素提取纯文本，以 '\\n\\n' 拼接还原段落边界。

    不含 div（div 常为容器，其 get_text 会把整章压成一段，反而丢失段落
    结构）。无任何主路径块级标签时退化为整体文本（历史行为）。
    T-L1：verse_out 非 None 时收集多行韵文块（{"tag", "text"}）。
    """
    primary = root.find_all(_PRIMARY_BLOCK_TAGS)
    if not primary:
        return root.get_text('\n')
    parts = []
    for el in root.find_all(_PRIMARY_BLOCK_TAGS + _FALLBACK_BLOCK_TAGS):
        if el.name in _FALLBACK_BLOCK_TAGS:
            # 兜底仅取「主路径不可见」的元素：含主块级后代（已被主路径提取）
            # 或含更内层兜底元素（避免重复提取）时跳过
            if el.find(_PRIMARY_BLOCK_TAGS) or el.find(_FALLBACK_BLOCK_TAGS):
                continue
        t = _block_element_text(el)
        if t:
            parts.append(t)
            if verse_out is not None and '\n' in t:
                tag = _el_verse_tag(el)
                if tag:
                    verse_out.append({'tag': tag, 'text': t})
    return '\n\n'.join(parts)


def _mark_verse_elements(soup) -> None:
    """T-L1：在 br→'\\n' 替换前标记韵文元素（class 命中或短行 br 启发式）。"""
    for el in soup.find_all(['p', 'pre', 'blockquote', 'div', 'section', 'dd']):
        cls = ' '.join(el.get('class') or [])
        if cls and _VERSE_CLASS_RE.search(cls):
            el[_VERSE_TAG_ATTR] = 'class'
            continue
        if el.name == 'pre':
            continue  # pre 天然保行；tag 在提取时按行数判定
        br_count = len(el.find_all('br'))
        if br_count < _VERSE_HEURISTIC_MIN_BR:
            continue
        lines = [ln.strip() for ln in el.get_text('\n').split('\n') if ln.strip()]
        if len(lines) < 3:
            continue
        short = sum(1 for ln in lines if len(ln) <= _VERSE_HEURISTIC_MAX_LINE)
        if short / len(lines) >= _VERSE_HEURISTIC_SHORT_RATIO:
            el[_VERSE_TAG_ATTR] = 'heuristic'


def run_conversion_audit(
    chapter_sources: dict,
    chapters: list,
    skipped_docs: list,
    output_path,
    threshold: float = _AUDIT_COVERAGE_THRESHOLD,
) -> dict:
    """转换完整性审计：oracle token 多重集 vs md 产出 token 多重集。

    返回 {"ok", "overall", "chapter_results", "missing_spans", "report_path"}。
    低于阈值（或单章低于下限）→ ok=False；报告文件始终写出（可审计）。
    """
    from collections import Counter
    from datetime import datetime

    chapter_results: dict = {}
    missing_spans: list = []
    total_oracle = 0
    total_missing = 0
    total_whitelisted = 0

    for ch_id, md in chapters:
        srcs = chapter_sources.get(ch_id, [])
        oracle_counter: Counter = Counter()
        token_stream: list = []
        whitelisted = 0
        for html in srcs:
            otext = _epub_oracle_text(html)
            wl_spans = _audit_whitelist_spans(otext)
            for tok, pos in _audit_tokens_with_pos(otext):
                if any(s <= pos < e for s, e in wl_spans):
                    whitelisted += 1
                    continue
                oracle_counter[tok] += 1
                token_stream.append((tok, pos, otext))

        md_counter = Counter(_AUDIT_TOKEN_RE.findall(md))
        missing_counter = oracle_counter - md_counter
        missing_n = sum(missing_counter.values())
        total_n = sum(oracle_counter.values())
        coverage = (1.0 - missing_n / total_n) if total_n else 1.0

        # 缺失 span：按 oracle 顺序定位未覆盖 token，合并邻近出现（≤40 字符间隔）
        if missing_n:
            remain = missing_counter.copy()
            runs = []
            cur = None
            prev_text = None
            prev_end = None
            for tok, pos, otext in token_stream:
                if remain.get(tok, 0) <= 0:
                    continue
                remain[tok] -= 1
                end = pos + len(tok)
                if cur is not None and otext is prev_text and pos - prev_end <= 40:
                    cur['end'] = end
                else:
                    if cur is not None:
                        runs.append(cur)
                    cur = {'text': otext, 'start': pos, 'end': end}
                prev_text = otext
                prev_end = end
            if cur is not None:
                runs.append(cur)
            for r in runs:
                s = max(0, r['start'] - 40)
                e = min(len(r['text']), r['end'] + 40)
                excerpt = re.sub(r'\s+', ' ', r['text'][s:e]).strip()
                missing_spans.append({'chapter': ch_id, 'excerpt': excerpt})

        chapter_results[ch_id] = {
            'sources': len(srcs),
            'oracle_tokens': total_n,
            'missing': missing_n,
            'coverage': coverage,
        }
        total_oracle += total_n
        total_missing += missing_n
        total_whitelisted += whitelisted

    overall = (1.0 - total_missing / total_oracle) if total_oracle else 1.0
    min_cov = min((r['coverage'] for r in chapter_results.values()), default=1.0)
    ok = overall >= threshold and min_cov >= _AUDIT_CHAPTER_FLOOR

    report = [
        '# 转换完整性审计报告 (T1)',
        '',
        f'- 生成时间: {datetime.now().isoformat(timespec="seconds")}',
        f'- 阈值: 全书覆盖率 ≥ {threshold * 100:.1f}%；单章下限 {_AUDIT_CHAPTER_FLOOR * 100:.0f}%',
        f'- 全书覆盖率: {overall * 100:.3f}%',
        f'- 策略豁免 token 数（许可证/目录区域）: {total_whitelisted}',
        f'- 结论: {"✅ 通过" if ok else "❌ 未通过"}',
        '',
        '## 逐章覆盖率',
        '',
        '| 章 | 源文档数 | oracle tokens | 缺失 | 覆盖率 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for ch_id, r in chapter_results.items():
        report.append(
            f"| {ch_id} | {r['sources']} | {r['oracle_tokens']} | "
            f"{r['missing']} | {r['coverage'] * 100:.3f}% |")

    report.append('')
    report.append(f'## 未覆盖片段 Top-{_AUDIT_MAX_MISSING_SPANS}')
    report.append('')
    if missing_spans:
        for s in missing_spans[:_AUDIT_MAX_MISSING_SPANS]:
            report.append(f"- [{s['chapter']}] …{s['excerpt']}…")
    else:
        report.append('- （无）')

    report.append('')
    report.append('## 策略豁免说明')
    report.append('')
    report.append('- 古腾堡许可证特征串区域（Project Gutenberg / www.gutenberg.org 等）')
    report.append('- 前部「Contents + ≥2 个编号目录项」区域（Q2 有意裁剪的目录）')
    if skipped_docs:
        report.append('')
        report.append('## 按策略跳过的短文档（<50 字符，不计入审计）')
        report.append('')
        for d in skipped_docs:
            report.append(f"- {d.get('id', '?')} ({d.get('chars', 0)} chars)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(report) + '\n', encoding='utf-8')

    return {
        'ok': ok,
        'overall': overall,
        'chapter_results': chapter_results,
        'missing_spans': missing_spans[:_AUDIT_MAX_MISSING_SPANS],
        'report_path': str(output_path),
    }


CHUNK_TAGS_VERSION = 1


VERSE_MANIFEST_VERSION = 1


_DEFAULT_VERSE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_line_ratio": 0.6,
    "max_line_chars": 56,
    "refrain_ratio": 0.5,
    "retry_on_fail": True,
}


def _block_lines_in_range(block: dict, start: int, end: int) -> List[str]:
    """取韵文块中字符区间 [start, end) 覆盖的非空行。"""
    text = block.get("text") or ""
    base = int(block.get("start_char", 0) or 0)
    lines: List[str] = []
    pos = base
    for ln in text.split("\n"):
        ln_s, ln_e = pos, pos + len(ln)
        if ln.strip() and ln_s < end and ln_e > start:
            lines.append(ln)
        pos = ln_e + 1
    return lines


def _longest_short_line_run(lines: List[str], max_chars: int) -> int:
    """最长「连续短行」段落长度（空行透明，长行重置；容忍诗节空行）。"""
    best = 0
    cur = 0
    for ln in lines:
        if not ln:
            continue
        if len(ln) <= max_chars:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _max_line_repeat(lines: List[str], min_len: int = 4) -> int:
    """归一化后行重复的最大次数（复沓/refrain 探测）。"""
    counts: Dict[str, int] = {}
    for ln in lines:
        key = normalize_whitespace(ln).strip("“”\"'‘’「」")
        if len(key) < min_len:
            continue
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def _normalized_index(text: str) -> Tuple[str, List[int]]:
    """返回 (归一化文本, 每个归一化字符对应的原文本偏移)。"""
    norm_chars: List[str] = []
    orig_idx: List[int] = []
    last_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if not last_space:
                norm_chars.append(" ")
                orig_idx.append(i)
                last_space = True
        else:
            norm_chars.append(ch)
            orig_idx.append(i)
            last_space = False
    return "".join(norm_chars), orig_idx


def blocks_for_span(verse_blocks: List[dict], start: int, end: int) -> List[dict]:
    """与 chunk 字符区间相交的韵文片段：[{"tag", "lines"}]。"""
    frags: List[dict] = []
    for b in verse_blocks or []:
        bs = int(b.get("start_char", -1) or -1)
        be = int(b.get("end_char", -1) or -1)
        if bs < 0 or be <= bs or bs >= end or be <= start:
            continue
        lines = _block_lines_in_range(b, max(bs, start), min(be, end))
        if lines:
            frags.append({"tag": b.get("tag", "unknown"), "lines": lines})
    return frags


def build_chapter_tags(chapter_id: str, markdown_text: str, chunks: List[str],
                       verse_blocks: List[dict]) -> Dict[str, dict]:
    """计算本章 verse chunk 标签：{chunk_id: {"verse": true, "blocks": [{"tag","lines"}]}}。"""
    result: Dict[str, dict] = {}
    if not verse_blocks or not chunks:
        return result
    spans = locate_chunk_spans(markdown_text, chunks)
    for idx, text in enumerate(chunks):
        frags: List[dict] = []
        span = spans[idx] if idx < len(spans) else None
        if span is not None:
            frags = blocks_for_span(verse_blocks, span[0], span[1])
        if not frags:
            frags = fragments_from_manifest(text, verse_blocks)
        if frags:
            result[f"{chapter_id}_chunk{idx:03d}"] = {"verse": True, "blocks": frags}
    return result


def build_verse_note(fragments: List[dict]) -> str:
    """生成注入 literal/rewrite/judge prompt 的韵文处理协议块；无韵文返回 ""。"""
    lines: List[str] = []
    for f in fragments or []:
        lines.extend(f.get("lines") or [])
    lines = [ln.strip() for ln in lines if ln and ln.strip()]
    if not lines:
        return ""
    n = len(lines)
    target = max(2, int(round(n * 0.8)))
    refrain = _max_line_repeat(lines)
    parts = [
        f"【韵文/歌词处理协议（本片段含韵文，源共 {n} 行）】",
        f"1. 逐行对应：每个源行对应一个译行，行与行之间保留换行；译文行数尽量贴近源文"
        f"（源 {n} 行 → 目标 ≥ {target} 行）；严禁把韵文合并进散文段落。",
    ]
    if refrain >= 2:
        parts.append(
            f"2. 复沓（refrain）保留：原文有重复出现的歌词句（最多重复 {refrain} 次），"
            f"译文应按相同的重复节奏重现。")
    parts.append(
        "3. 节奏与尾韵：优先保留童谣/歌谣的节奏感；尾韵尽力而为，不得为押韵牺牲语义。")
    parts.append(
        "4. 韵文正文内不得插入注释或译者注；注释集中放在章末或脚注中。")
    parts.append(
        "5. 本片段为韵文体裁，分行结构是硬性结构要求（不是排版建议），"
        "不得因顺句而合并诗行。")
    return "\n".join(parts) + "\n\n"


def fragments_for_chunk(chapter_id: str, chunk_id: str, chunk_text: str,
                        tags: dict, manifest: dict) -> List[dict]:
    """chunk 的韵文片段：tags 优先（精确交集），miss 时 manifest 包含兜底。"""
    entry = ((tags.get("chapters") or {}).get(chapter_id) or {}).get(chunk_id)
    if isinstance(entry, dict) and entry.get("blocks"):
        return entry["blocks"]
    blocks = ((manifest.get("chapters") or {}).get(chapter_id)) or []
    if not blocks:
        return []
    return fragments_from_manifest(chunk_text, blocks)


def fragments_from_manifest(chunk_text: str, verse_blocks: List[dict]) -> List[dict]:
    """运行时兜底：按归一化「文本包含」匹配（整块连通命中 → 全行；否则按块序连续 ≥2 行命中）。

    Round 120 P1: 单行命中会误报（如 Fury 诗 "would be" 混入散文 chunk），
    故仅接受整块连通命中或连续 ≥2 行的命中段。
    """
    norm_chunk = normalize_whitespace(chunk_text)
    if not norm_chunk:
        return []
    frags: List[dict] = []
    for b in verse_blocks or []:
        lines = [ln for ln in (b.get("text") or "").split("\n") if ln.strip()]
        if not lines:
            continue
        norm_block = normalize_whitespace(b.get("text") or "")
        if norm_block and norm_block in norm_chunk:
            frags.append({"tag": b.get("tag", "unknown"), "lines": lines})
            continue
        hit = [i for i, ln in enumerate(lines)
               if len(ln.strip()) >= 8 and normalize_whitespace(ln) in norm_chunk]
        if not hit:
            continue
        best = cur = [hit[0]]
        for i in hit[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                if len(cur) > len(best):
                    best = cur
                cur = [i]
        if len(cur) > len(best):
            best = cur
        if len(best) >= 2:
            frags.append({"tag": b.get("tag", "unknown"),
                          "lines": [lines[i] for i in best]})
    return frags


def load_chunk_tags(db_dir) -> dict:
    path = Path(db_dir) / "chunk_tags.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}


def load_verse_manifest(input_dir) -> dict:
    path = Path(input_dir) / "verse_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}


def locate_chunk_spans(markdown_text: str, chunks: List[str]) -> List[Optional[Tuple[int, int]]]:
    """按序定位每个 chunk 在 markdown_text 中的 (start_char, end_char)；失败 None。

    归一化空白后查找，再映射回原文本偏移，容忍段落间空白差异。
    """
    norm_text, orig_idx = _normalized_index(markdown_text)
    spans: List[Optional[Tuple[int, int]]] = []
    cursor = 0
    for chunk in chunks:
        nchunk = normalize_whitespace(chunk)
        if not nchunk:
            spans.append(None)
            continue
        pos = norm_text.find(nchunk, cursor)
        if pos < 0:
            pos = norm_text.find(nchunk)
        if pos < 0:
            spans.append(None)
            continue
        end_norm = pos + len(nchunk)
        start_char = orig_idx[pos]
        end_char = orig_idx[end_norm - 1] + 1
        spans.append((start_char, end_char))
        cursor = end_norm
    return spans


def normalize_whitespace(text: str) -> str:
    """归一化空白：全部空白折叠为单个空格并去首尾（用于包含/重复判定）。"""
    return re.sub(r"\s+", " ", text or "").strip()


def save_chunk_tags(db_dir, tags: dict) -> None:
    path = Path(db_dir) / "chunk_tags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tags, ensure_ascii=False, indent=2), encoding="utf-8")


def save_verse_manifest(input_dir, manifest: dict) -> None:
    path = Path(input_dir) / "verse_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def verse_config() -> Dict[str, Any]:
    """读取 config.yaml `verse:` 节（缺失/异常时返回内建默认，向后兼容）。"""
    cfg = dict(_DEFAULT_VERSE_CONFIG)
    try:
        from src.config import config as _config
        section = _config.get_section("verse") or {}
        for k, v in section.items():
            cfg[k] = v
    except Exception:
        pass
    return cfg


def verse_lint(fragments: List[dict], translated_text,
               min_line_ratio: float = 0.6,
               max_line_chars: int = 56,
               refrain_ratio: float = 0.5) -> dict:
    """源行数 vs 译行数偏差 + 复沓存留检查。

    返回 {"ok", "expected_lines", "found_lines", "refrain_source",
          "refrain_target", "failures", "reason"}。
    """
    src_lines: List[str] = []
    for f in fragments or []:
        src_lines.extend(ln.strip() for ln in (f.get("lines") or []) if ln and ln.strip())
    expected = len(src_lines)
    res: Dict[str, Any] = {
        "ok": True, "expected_lines": expected, "found_lines": 0,
        "refrain_source": 0, "refrain_target": 0, "failures": [], "reason": "",
    }
    if expected == 0:
        return res
    text = translated_text
    if isinstance(text, dict):
        text = text.get("text", "")
    tlines = [ln.strip() for ln in str(text or "").split("\n")]
    found = _longest_short_line_run(tlines, max_line_chars)
    res["found_lines"] = found
    if expected <= 2:
        need = expected
    else:
        need = max(2, int(round(expected * min_line_ratio)))
    if found < need:
        res["failures"].append(f"verse_lines:{found}/{expected}(need>={need})")
    r_src = _max_line_repeat(src_lines)
    res["refrain_source"] = r_src
    if r_src >= 3:
        r_tgt = _max_line_repeat([ln for ln in tlines if ln])
        res["refrain_target"] = r_tgt
        need_r = max(2, int(round(r_src * refrain_ratio)))
        if r_tgt < need_r:
            res["failures"].append(f"refrain:{r_tgt}/{r_src}(need>={need_r})")
    res["ok"] = not res["failures"]
    res["reason"] = "；".join(res["failures"])
    return res


_BOLD_LEAD_RE = re.compile(r'^\s*\*\*([^*]{1,40})\*\*\s*[：:]\s*\S')


_BRACKET_DEF_RE = re.compile(r'^\s*[\[［](\d{1,3})[\]］]\s*(\S.*)$')


_CHAPTER_HEAD_RE = re.compile(r'^\s*#?\s*第[一二三四五六七八九十百千零〇\d]+\s*[章.。．]')


_DANGLING_RE = re.compile(r'[：:]\s*——\s*$')


_FOOTNOTE_HEADER_RE = re.compile(
    r'^\s*\*{0,2}\s*【?脚注】?\s*\*{0,2}'
    r'(?:\s*[（(][^）)\n]{0,48}[）)])?'
    r'\s*\*{0,2}\s*[：:]?\s*\*{0,2}\s*$'
)


_FRONTMATTER_LINE_RE = re.compile(
    r'^\s*#?\s*(?:The Project Gutenberg eBook\b.*|THE MILLENNIUM FULCRUM EDITION\b.*'
    r'|古腾堡计划.*|千[禧年]*年支点版.*|目录|Contents)\s*$')


_FRONTMATTER_RESIDUE_RES = [
    ("Gutenberg 英文残留", re.compile(r'The Project Gutenberg eBook')),
    ("Gutenberg 中文残留", re.compile(r'古腾堡')),
    ("版本行残留", re.compile(r'MILLENNIUM FULCRUM|千[禧年]*年支点版')),
    ("标题粘连残留", re.compile(r'ofAlice')),
    ("章节标题缺空格", re.compile(r'CHAPTER\s+[IVXLCDM]+\.(?=[A-Z])')),
]


_MACHINE_FN_ID_RE = re.compile(r'\[\^[^\]]*_c\d+_[^\]]*\]')


_MD_DEF_RE = re.compile(r'^\s*\[\^([^\]]+)\]:\s?(.*)$')


_MD_REF_RE = re.compile(r'\[\^([^\]]+)\](?!:)')


_META_TOKENS = ["保留与加注策略", "保留并加注", "策略池", "RETAIN_AND_ANNOTATE",
                "加注策略", "保留音译", "按约束", "直译自"]


_NOTE_DUP_RATIO = 0.28


_NOTE_DUP_RATIO_HARD = 0.75


_NOTE_LABEL_RE = re.compile(r'^\s*(?:\*+\s*)?(?:译者注|译注|脚注|注释)\s*[:：]?\s*')


_NOTE_TITLE_END_RE = re.compile(
    r'^\s*\*{0,2}\s*【(?:章末注释|本章注释|章末注|尾注|文末注)】\s*\*{0,2}\s*$')


_NOTE_TITLE_RE = re.compile(r'^\s*\*{0,2}\s*【([^】]{1,12})】\s*\*{0,2}\s*$')


_NUM_DEF_RE = re.compile(r'^\s*(\d+)\.\s+(\S.*)$')


_NUM_DEF_START_RE = re.compile(r'^[“”‘’「『（【*\u4e00-\u9fffA-Za-z]')


_PAREN_DEF_RE = re.compile(r'^\s*[（(](\d{1,3})[）)]\s*(\S.*)$')


_QUOTED_TERM_RE = re.compile(r'[“「『]([^”」』]{2,36})[”」』]')


_SEP_LINE_RE = re.compile(r'^\s*-{3,}\s*$')


_SUPER = "¹²³⁴⁵⁶⁷⁸⁹"


_SUP_DEF_RE = re.compile(r'^\s*([¹²³⁴⁵⁶⁷⁸⁹]+)\s+(\S.*)$')


_SUP_REF_RE = re.compile(r'[¹²³⁴⁵⁶⁷⁸⁹]+')


_TR_BLOCK_RE = re.compile(r'^\s*>\s*\*\*译者注[:：]\*\*\s*(.*)$')


def _audit_text(text: str) -> dict:
    """单文本 lint 审计（返回 problems 与统计）。"""
    problems = []
    straight = text.count('"')
    if straight:
        problems.append(f"直引号 {straight} 处")
    cjk_q = sum(text.count(c) for c in '「」『』')
    if cjk_q:
        problems.append(f"「」/『』引号 {cjk_q} 处")
    items, defs, refs = _scan_lines(text)
    sup_defs = [d for d in defs if d['style'] == 'sup']
    if sup_defs:
        problems.append(f"上标脚注定义 {len(sup_defs)} 行（未转 [^n]）")
    def_ids = {d['id'] for d in defs}
    orphans = [d for d in defs if d['id'] not in refs]
    if orphans:
        problems.append(f"悬空脚注定义 {len(orphans)} 条: {[d['id'] for d in orphans]}")
    dangling_refs = sorted(refs - def_ids)
    meta = sum(text.count(t) for t in _META_TOKENS)
    if meta:
        problems.append(f"策略元语 {meta} 处")
    keys = {}
    dups = []
    for d in defs:
        k = _def_key(d['content'])
        if k in keys and keys[k] != d['content']:
            dups.append(k)
        keys[k] = d['content']
    if dups:
        problems.append(f"同键重复注释: {dups}")
    return {"problems": problems, "defs": defs, "orphan_count": len(orphans),
            "dangling_refs": dangling_refs, "meta_count": meta}


def _book_style_config() -> dict:
    try:
        from src.config import config as _cfg
        data = _cfg.get("book", {}) if hasattr(_cfg, "get") else {}
        if data:
            return data
    except Exception:
        pass
    try:
        from src.config import get_config as _get_config
        return _get_config().get("book", {}) or {}
    except Exception:
        return {}


def _chapter_dup_notes(entries) -> list:
    """entries: [(label, text)]（同章）。返回重复注 failures/warnings。"""
    failures, warnings = [], []
    seen = {}
    for label, text in entries:
        _items, defs, _refs = _scan_lines(text)
        for d in defs:
            k = _def_key(d['content'])
            if not k:
                continue
            if k in seen:
                prev_label, prev_content = seen[k]
                if prev_content == d['content']:
                    failures.append(f"重复注（同键同文）: {k} @ {prev_label} / {label}")
                else:
                    import difflib as _dl
                    r = _dl.SequenceMatcher(None, prev_content, d['content']).ratio()
                    if r >= 0.8:
                        failures.append(f"重复注（同键近似 r={r:.2f}）: {k} @ {prev_label} / {label}")
                    else:
                        warnings.append(f"同键不同注: {k} @ {prev_label} / {label}")
            else:
                seen[k] = (label, d['content'])
    return failures, warnings


def _chapter_note_dupes(text: str) -> list:
    """章切片的注释重复检测（供 lint；与 normalize 同判据）。"""
    _blocks, units = _note_candidates(text)
    _group_title_units(units)
    cand = [u for u in units if u['kind'] in ('md', 'num', 'tr', 'para')
            and 'group' not in u]
    out = []
    _stop = _note_anchor_stop([c['content'] for c in cand])
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if _note_is_dup(cand[i]['content'], cand[j]['content'], stop=_stop):
                out.append("重复注: {0} ~ {1}".format(
                    _note_key(cand[i]['content'])[:18],
                    _note_key(cand[j]['content'])[:18]))
    return out


def _def_key(content: str) -> str:
    m = re.match(r'^\s*\*\*(.+?)\*\*', content)
    if m:
        return m.group(1).strip()
    m = re.match(r'^\s*[“「](.+?)[”」]', content)
    if m:
        return m.group(1).strip()
    return content.strip()[:12]


def _find_finals(out_root: Path):
    return sorted(out_root.glob("ch*/*_final.json"))


def _group_title_units(units) -> None:
    """标题块打组：title 及紧随其后的连续 num 单元。"""
    i, n = 0, len(units)
    while i < n:
        if units[i]['kind'] == 'title':
            j = i + 1
            while j < n and units[j]['kind'] == 'num':
                j += 1
            if j - i > 1:
                for k in range(i, j):
                    units[k]['group'] = i
            i = j
        else:
            i += 1


def _note_anchor_stop(contents) -> set:
    """章内频率派生的停用锚点（纯统计，零作品先验，Round 130 审计④）。

    在 ≥60%（且 ≥2 条）注释中都出现的拉丁词（≥4 字母）不构成"两注同源"
    的证据——某词若高频遍布全章注释（典型：作者名/书名/时代词），它对区分
    不同注释没有信息量。初始样本中 Carroll 类词即由此机制自然覆盖，不再
    硬编码任何作品专属停用表。
    """
    import math
    from collections import Counter
    texts = [str(c) for c in contents if c]
    n = len(texts)
    if n < 2:
        return set()
    cnt: Counter = Counter()
    for c in texts:
        cnt.update({w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", c)})
    threshold = max(2, math.ceil(n * 0.6))
    return {w for w, k in cnt.items() if k >= threshold}


def _note_candidates(text: str):
    """注释单元扫描。→ (blocks, units)；blocks 含字符偏移，units 含 kind/id/content。"""
    raw_blocks = [{'start': m.start(), 'end': m.end(), 'seg': m.group(0)}
                  for m in re.finditer(r'(?m)(?:^[ \t]*\S.*(?:\n|$))+', text)]
    blocks = []
    for _b in raw_blocks:
        _ls = _b['seg'].split('\n')
        _body_ls = [l for l in _ls if l.strip()]
        if (len(_body_ls) > 1
                and all(_TR_BLOCK_RE.match(l) for l in _body_ls)):
            _pos = _b['start']
            for _l in _ls:
                if _l.strip():
                    blocks.append({'start': _pos, 'end': _pos + len(_l), 'seg': _l})
                _pos += len(_l) + 1
        else:
            blocks.append(_b)
    units = []
    prev = None
    for bi, b in enumerate(blocks):
        seg = b['seg']
        first = seg.split('\n', 1)[0]
        m_md = _MD_DEF_RE.match(first)
        m_tr = _TR_BLOCK_RE.match(first)
        m_title = _NOTE_TITLE_RE.match(first)
        m_br = _BRACKET_DEF_RE.match(first)
        m_pa = _PAREN_DEF_RE.match(first)
        if m_md:
            u = {'kind': 'md', 'id': m_md.group(1), 'content': m_md.group(2), 'block': bi}
        elif m_tr:
            rest = [re.sub(r'^\s*>\s?', '', l) for l in seg.split('\n')[1:]]
            u = {'kind': 'tr', 'id': None,
                 'content': ' '.join([m_tr.group(1)] + rest).strip(), 'block': bi}
        elif m_title and len(seg.rstrip('\n').split('\n')) == 1:
            u = {'kind': 'title', 'id': None, 'content': m_title.group(1), 'block': bi,
                 'title_end': bool(_NOTE_TITLE_END_RE.match(first))}
        elif (m_br or m_pa) and prev in ('sep', 'num', 'title'):
            g = m_br or m_pa
            u = {'kind': 'num', 'id': g.group(1), 'content': g.group(2), 'block': bi}
        elif _SEP_LINE_RE.match(first):
            u = {'kind': 'sep', 'id': None, 'content': '', 'block': bi}
        elif (_BOLD_LEAD_RE.match(first) and len(seg.strip()) <= 800):
            u = {'kind': 'para', 'id': None, 'content': seg.strip(), 'block': bi}
        else:
            u = {'kind': 'body', 'id': None, 'content': seg, 'block': bi}
        units.append(u)
        prev = u['kind']
    return blocks, units


def _note_is_dup(a: str, b: str, stop=frozenset()) -> bool:
    """内容级重复判定（保守，校准值见上方注释）。

    stop: 章内频率派生的停用锚点（_note_anchor_stop()）；生产调用方须显式传入，
    缺省空集 = 不做停用（仅供单测/直调）。
    """
    import difflib
    na, nb = _note_norm(a), _note_norm(b)
    if not na or not nb or min(len(na), len(nb)) < 30:
        return False
    r = difflib.SequenceMatcher(None, na, nb).ratio()
    if r >= _NOTE_DUP_RATIO_HARD:
        return True
    if r < _NOTE_DUP_RATIO:
        return False
    la = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", a)} - stop
    lb = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", b)} - stop
    if la & lb:
        return True
    ca, cb = set(), set()
    for run in re.findall(r'[\u4e00-\u9fff]{3,}', na):
        ca.update(run[i:i + 4] for i in range(len(run) - 3))
    for run in re.findall(r'[\u4e00-\u9fff]{3,}', nb):
        cb.update(run[i:i + 4] for i in range(len(run) - 3))
    return len(ca & cb) >= 4


def _note_key(content: str) -> str:
    """注释主键：粗体词条 / 引号词条 / 归一化前 12 字。"""
    m = re.match(r'^\s*\*\*(.+?)\*\*', content)
    if m:
        return m.group(1).strip()
    m = re.match(r'^\s*[“「『](.+?)[”」』]', content)
    if m:
        return m.group(1).strip()
    return _note_norm(content)[:12]


def _note_norm(content: str) -> str:
    """注释内容归一化（去重比较用）：剥 markdown/标签/空白。"""
    t = re.sub(r'[*`>]', '', content)
    t = re.sub(r'^\s*\[\^[^\]]+\]:\s*', '', t)
    t = _NOTE_LABEL_RE.sub('', t)
    return re.sub(r'\s+', '', t)


def _note_terms(content: str) -> list:
    """孤儿注改锚候选词（长词优先，只从词条头部提取）。"""
    cands = []
    m = re.match(r'^\s*\*\*([^*]{1,40})\*\*', content)
    if m:
        cands.append(m.group(1))
    for m in _QUOTED_TERM_RE.finditer(content):
        cands.append(m.group(1))
    m = re.match(r'^\s*([^：:]{2,24})[：:]', content)
    if m:
        cands.append(m.group(1))
    seen, out = set(), []
    for t in cands:
        t = re.sub(r'^\s*\[[^\]]*\]\s*', '', t).strip(' *“”"「」『』\u3000')
        pieces = [t, t.split('（')[0], t.split('(')[0]]
        pieces += re.findall(r'[\u4e00-\u9fff]{2,}', t)
        pieces += re.findall(r"[A-Za-z][A-Za-z'-]{3,}", t)
        for p in pieces:
            p = p.strip(' *“”"「」『』：「:」（()）')
            if len(p) >= 2 and p not in seen:
                seen.add(p)
                out.append(p)
    out.sort(key=len, reverse=True)
    return out


def _scan_lines(text: str):
    """一次扫描：行分类 + 注释定义 + 引用集合。

    Returns:
        (items, defs, refs)
        items: [('header', None) | ('def', def_dict) | ('body', line)]
        defs:  [{'style': 'sup'|'md'|'num', 'id': str, 'content': str}]
        refs:  set[str]（正文中的 [^id] 与上标引用，统一为字符串 id）
    """
    items = []
    defs = []
    header_seen = False
    for line in text.split('\n'):
        if _FOOTNOTE_HEADER_RE.match(line):
            header_seen = True
            items.append(('header', None))
            continue
        m = _SUP_DEF_RE.match(line)
        if m:
            d = {'style': 'sup', 'id': str(sup_run_to_int(m.group(1))), 'content': m.group(2)}
            defs.append(d)
            items.append(('def', d))
            continue
        m = _MD_DEF_RE.match(line)
        if m:
            d = {'style': 'md', 'id': m.group(1), 'content': m.group(2)}
            defs.append(d)
            items.append(('def', d))
            continue
        if header_seen:
            m = _NUM_DEF_RE.match(line)
            if m and _NUM_DEF_START_RE.match(m.group(2)):
                d = {'style': 'num', 'id': m.group(1), 'content': m.group(2)}
                defs.append(d)
                items.append(('def', d))
                continue
        items.append(('body', line))
    body_text = '\n'.join(p for k, p in items if k == 'body')
    refs = set()
    for mm in _SUP_REF_RE.finditer(body_text):
        refs.add(str(sup_run_to_int(mm.group(0))))
    for mm in _MD_REF_RE.finditer(body_text):
        refs.add(mm.group(1))
    return items, defs, refs


def _split_h1_sections(text: str):
    """按 H1 切章 → [(章标题, 正文)]（前置段标题为 '(前置)'）。"""
    parts = re.split(r'(?m)^(# .*)$', text)
    out = [('(前置)', parts[0])]
    for k in range(1, len(parts), 2):
        out.append((parts[k].strip(), parts[k + 1] if k + 1 < len(parts) else ''))
    return out


def chapter_heading(project_root, chapter_id: str, style: str = None) -> str:
    """merge 层章节标题。默认 config.book.chapter_titles（bilingual）。

    bilingual: "第一章 掉进兔子洞（Down the Rabbit-Hole）"
    zh:        "第一章 掉进兔子洞"
    keep:      input/<ch>.md 原始 H1（清理 Gutenberg 前缀）
    """
    root = Path(project_root)
    if style is None:
        style = str(_book_style_config().get("chapter_titles", "bilingual"))
    orig = ""
    md = root / "input" / f"{chapter_id}.md"
    if md.exists():
        for line in md.read_text(encoding="utf-8").split('\n')[:10]:
            s = line.strip()
            if s.startswith('# '):
                orig = s[2:].strip()
                break
    orig_clean = re.sub(r'^The Project Gutenberg eBook of\s*', '', orig)
    zh = ""
    out_dir = root / "output" / chapter_id
    firsts = sorted(out_dir.glob("*_final.json"))
    if firsts:
        try:
            data = json.loads(firsts[0].read_text(encoding="utf-8"))
            text = data.get("text", "") if isinstance(data, dict) else str(data)
        except Exception:
            text = ""
        for line in text.split('\n')[:20]:
            m = re.match(r'^#?\s*(第[一二三四五六七八九十百千零〇\d]+\s*[章.。．]\s*\S.*)$', line.strip())
            if m and len(m.group(1)) <= 60:
                zh = m.group(1).strip().rstrip('。．.')
                zh = re.sub(r'\[\^[^\]]+\]', '', zh)
                break
    if style == "keep":
        return orig_clean or chapter_id
    if style == "zh":
        return zh or orig_clean or chapter_id
    orig_short = re.sub(r'^CHAPTER\s+[IVXLCDM]+\.\s*', '', orig_clean)
    if zh and orig_short and orig_short not in zh:
        return f"{zh}（{orig_short}）"
    return zh or orig_short or chapter_id


def finalize_outputs(project_root=None, output_dir=None, apply: bool = False) -> dict:
    """对全部 final.json 执行 normalize（默认 dry-run），写 output/finalize_diff.md。"""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    finals = _find_finals(out_root)
    stats = {"files": len(finals), "changed": 0, "headers_removed": 0,
             "sup_defs_converted": 0, "md_defs_kept": 0, "orphans_relocated": 0,
             "cjk_quotes": 0, "straight_quotes": 0}
    details = []
    for fp in finals:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            print(f"  ⚠️ [Finalize] 跳过损坏 JSON: {fp}")
            continue
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        new_text, ch = normalize_final_style(text)
        if new_text == text:
            continue
        stats["changed"] += 1
        for k in ("headers_removed", "sup_defs_converted", "md_defs_kept",
                  "orphans_relocated", "cjk_quotes", "straight_quotes"):
            stats[k] += ch[k]
        details.append((str(fp.relative_to(root)), ch))
        if apply and isinstance(data, dict):
            data["text"] = new_text
            meta = data.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            meta["style_normalized"] = True
            data["metadata"] = meta
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 成书样式规范化报告 (finalize_diff)", ""]
    lines.append(f"- 模式: {'APPLY (已写入)' if apply else 'DRY RUN (仅报告)'}")
    lines.append(f"- 扫描 final.json: {stats['files']} 个；需要变更: {stats['changed']} 个")
    lines.append(f"- 注释标题行清除: {stats['headers_removed']}；上标定义转 [^n]: {stats['sup_defs_converted']}")
    lines.append(f"- 悬空定义转译者注: {stats['orphans_relocated']}；「」引号: {stats['cjk_quotes']}；直引号: {stats['straight_quotes']}")
    lines.append("")
    if details:
        lines.append("## 明细")
        lines.append("")
        for path, ch in details:
            lines.append(f"- `{path}`: " + ", ".join(f"{k}={v}" for k, v in ch.items() if v))
    else:
        lines.append("（无需变更，样式已统一）")
    lines.append("")
    lines.append("---")
    lines.append("*由 finalize 子命令自动生成*")
    lines.append("")
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "finalize_diff.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"stats": stats, "report": str(report_path), "applied": bool(apply)}


def lint_book(project_root=None, output_dir=None, merged_path=None) -> dict:
    """成书级 lint：finals（逐 chunk）+ merged（整书）双跑；写 output/book_lint.md。"""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    merged = Path(merged_path) if merged_path else out_root / "translated_full.md"
    failures, warnings = [], []
    stats = {"finals": 0}

    by_chapter = {}
    for fp in _find_finals(out_root):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            failures.append(f"{fp.name}: JSON 解析失败")
            continue
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        label = f"{fp.parent.name}/{fp.name}"
        stats["finals"] += 1
        by_chapter.setdefault(fp.parent.name, []).append((label, text))
        a = _audit_text(text)
        for p in a["problems"]:
            failures.append(f"{label}: {p}")
        if a["dangling_refs"]:
            warnings.append(f"{label}: 无定义引用 {a['dangling_refs']}")
    for ch, entries in sorted(by_chapter.items()):
        df, dw = _chapter_dup_notes(entries)
        failures.extend(f"{ch}: {x}" for x in df)
        warnings.extend(f"{ch}: {x}" for x in dw)

    if merged.exists():
        mtext = merged.read_text(encoding="utf-8")
        for name, pat in _FRONTMATTER_RESIDUE_RES:
            n = len(pat.findall(mtext))
            if n:
                failures.append(f"merged: {name} {n} 处")
        style = str(_book_style_config().get("chapter_titles", "bilingual"))
        headings = [ln.strip() for ln in mtext.split('\n') if ln.startswith('# ')]
        stats["headings"] = len(headings)
        if style in ("bilingual", "zh"):
            for h in headings[1:]:
                if not re.match(r'^#\s*第[一二三四五六七八九十百千零〇\d]+\s*章', h):
                    failures.append(f"merged: 章节标题不符合制式({style}): {h[:40]}")
        for h in headings:
            for name, pat in _FRONTMATTER_RESIDUE_RES:
                if pat.search(h):
                    failures.append(f"merged: 标题含残留（{name}）: {h[:40]}")
        dangling = [ln.strip()[:60] for ln in mtext.split('\n') if _DANGLING_RE.search(ln)]
        for d in dangling:
            warnings.append(f"merged: 疑似悬空承诺: {d}")
        a = _audit_text(mtext)
        for p in a["problems"]:
            failures.append(f"merged: {p}")
        machine_ids = _MACHINE_FN_ID_RE.findall(mtext)
        if machine_ids:
            failures.append(
                f"merged: 机器脚注 id 残留 {len(machine_ids)} 处（应先跑 notes-normalize）")
        for label, body in _split_h1_sections(mtext):
            dups = _chapter_note_dupes(body)
            for d in dups[:5]:
                failures.append(f"merged/{label}: {d}")
            if len(dups) > 5:
                failures.append(f"merged/{label}: 另有 {len(dups) - 5} 组重复注未列出")
    else:
        warnings.append("merged 未生成（translated_full.md 缺失），跳过整书级检查")

    result = {"ok": not failures, "failures": failures, "warnings": warnings, "stats": stats}
    lines = ["# 成书 lint 报告 (book_lint)", ""]
    lines.append(f"- 结论: {'✅ 全绿' if result['ok'] else '❌ 未通过'}")
    lines.append(f"- 扫描 final.json: {stats['finals']} 个")
    lines.append("")
    if failures:
        lines.append(f"## 失败项 ({len(failures)})")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")
    if warnings:
        lines.append(f"## 警告项 ({len(warnings)})")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("---")
    lines.append("*由 lint 子命令自动生成*")
    lines.append("")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "book_lint.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def lint_texts(items) -> dict:
    """items: [(label, text), ...] → {'ok': bool, 'failures': [...], 'warnings': [...]}"""
    failures = []
    warnings = []
    for label, text in items:
        a = _audit_text(text)
        for p in a["problems"]:
            failures.append(f"{label}: {p}")
        if a["dangling_refs"]:
            warnings.append(f"{label}: 无定义引用 {a['dangling_refs']}")
    return {"ok": not failures, "failures": failures, "warnings": warnings}


def normalize_chapter_notes(text: str, start_num: int = 1):
    """章级注释统一：去重 → 孤儿改锚/章末归位 → 顺序重编号（幂等）。

    Returns: (new_text, changes: dict, next_num: int)
    """
    blocks, units = _note_candidates(text)
    _group_title_units(units)
    changes = {"merged_dups": 0, "orphans_attached": 0, "md_relocated": 0,
               "num_relocated": 0, "title_blocks_moved": 0, "renumbered": 0,
               "title_shells_dropped": 0,
               "machine_ids": len(_MACHINE_FN_ID_RE.findall(text))}

    # 1) 去重（标题块/分隔行/正文豁免）
    cand = [u for u in units if u['kind'] in ('md', 'num', 'tr', 'para')
            and 'group' not in u]
    parent = list(range(len(cand)))

    def _root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    _stop = _note_anchor_stop([c['content'] for c in cand])
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if _note_is_dup(cand[i]['content'], cand[j]['content'], stop=_stop):
                ri, rj = _root(i), _root(j)
                if ri != rj:
                    parent[rj] = ri
    clusters = {}
    for i in range(len(cand)):
        clusters.setdefault(_root(i), []).append(i)
    dropped = set()
    remap = {}

    def _rank(u):
        score = 2 if u['kind'] == 'md' else (1 if u['kind'] == 'num' else 0)
        return (score, len(_note_norm(u['content'])))

    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda i: _rank(cand[i]), reverse=True)
        surv = cand[members[0]]
        for mi in members[1:]:
            u = cand[mi]
            dropped.add(u['block'])
            changes["merged_dups"] += 1
            if (u['kind'] == 'md' and surv['kind'] == 'md'
                    and u['id'] != surv['id']):
                remap[u['id']] = surv['id']

    # 2) 保留区（去重删除块置空以稳定偏移）内统计引用
    kept = text
    for bi in dropped:
        b = blocks[bi]
        kept = kept[:b['start']] + (' ' * (b['end'] - b['start'])) + kept[b['end']:]
    body_spans = [(u['block'], blocks[u['block']]) for u in units
                  if u['kind'] == 'body' and u['block'] not in dropped]

    def _count_refs(uid, md=True):
        pat = (re.compile(r'\[\^' + re.escape(uid) + r'\](?!:)') if md
               else re.compile(r'[\[［]' + re.escape(uid) + r'[\]］](?!\s*[:：])'))
        cnt, pos = 0, None
        for bi, b in body_spans:
            seg = kept[b['start']:b['end']]
            for m in pat.finditer(seg):
                cnt += 1
                if pos is None:
                    pos = b['start'] + m.start()
        return cnt, pos

    ref_info = {}
    for u in cand:
        if u['kind'] in ('md', 'num') and u['block'] not in dropped:
            ref_info[u['block']] = _count_refs(u['id'], md=(u['kind'] == 'md'))
    for old, new in remap.items():
        for u in cand:
            if u['kind'] == 'md' and u['id'] == new and u['block'] in ref_info:
                c, p = _count_refs(old, md=True)
                cc, pp = ref_info[u['block']]
                ref_info[u['block']] = (cc + c, pp if pp is not None else p)
                break

    # 3) 孤儿处理：改锚（正文词条）或章末归位
    seg_override = {}
    md_fail, num_fail = [], []

    def _block_text(bi):
        return seg_override.get(bi, blocks[bi]['seg'])

    # 编号注列表内交叉引用：同一连续列表内 {编号 → 临时 uid}
    run_map = {}
    runs, cur_run = [], []
    for u in units:
        if u['kind'] == 'num' and 'group' not in u:
            cur_run.append(u)
        else:
            if len(cur_run) > 1:
                runs.append(cur_run)
            cur_run = []
    if len(cur_run) > 1:
        runs.append(cur_run)
    for run in runs:
        m = {}
        for u in run:
            m[str(int(u['id']))] = ('n{0}'.format(u['block'])
                                    if u['kind'] == 'num' else u['id'])
        run_map[id(run)] = m
    uid_run = {}
    for run in runs:
        for u in run:
            uid_run[u['block']] = run_map[id(run)]

    def _rewrite_cross(content, block):
        m = uid_run.get(block)
        if not m:
            return content
        return re.sub(r'\[(\d+)\](?![\]:])',
                      lambda mm: '[^' + m.get(mm.group(1), mm.group(1)) + ']'
                      if mm.group(1) in m else mm.group(0), content)

    for u in cand:
        if u['kind'] not in ('md', 'num', 'tr') or u['block'] in dropped:
            continue
        cnt, _p = ref_info.get(u['block'], (0, None))
        if u['kind'] == 'tr':
            cnt = 0
        if cnt:
            continue
        hit = None
        for term in _note_terms(u['content']):
            for bi, b in body_spans:
                seg = kept[b['start']:b['end']]
                if term in seg:
                    hit = bi
                    break
            if hit is not None:
                break
        uid = u['id'] if u['kind'] == 'md' else 'n{0}'.format(u['block'])
        if hit is not None:
            cur = _block_text(hit)
            cur = cur.rstrip('\n')
            nl = '\n' if blocks[hit]['seg'].endswith('\n') else ''
            seg_override[hit] = cur + '[^' + uid + ']' + nl
            if u['kind'] in ('num', 'tr'):
                seg_override[u['block']] = '[^{0}]: {1}\n'.format(
                    uid, _rewrite_cross(u['content'], u['block']))
            changes["orphans_attached"] += 1
        elif u['kind'] == 'tr':
            continue
        else:
            dropped.add(u['block'])
            if u['kind'] == 'md':
                md_fail.append(u['content'])
                changes["md_relocated"] += 1
            else:
                num_fail.append('[{0}] {1}'.format(
                    u['id'], _rewrite_cross(u['content'], u['block'])))
                changes["num_relocated"] += 1

    # 3b) 章末注释块条目：尽力改锚（编辑器要求"有注无引"→角标；失败者留在块内）
    for tu in units:
        if not (tu['kind'] == 'title' and tu.get('title_end') and 'group' in tu):
            continue
        members = [u for u in units
                   if u.get('group') == tu['group'] and u['kind'] == 'num']
        attached = 0
        for u in members:
            terms = _note_terms(u['content'])
            extra = []
            for _run in re.findall(r'[\u4e00-\u9fff]{4,}', u['content']):
                if _run not in extra:
                    extra.append(_run)
                for _i in range(len(_run) - 3):
                    _w = _run[_i:_i + 4]
                    if _w not in extra:
                        extra.append(_w)
            extra = [r for r in extra if r not in terms]
            terms = terms + extra
            terms.sort(key=len, reverse=True)
            hit = None
            for term in terms:
                if len(term) < 4 and term not in _note_terms(u['content']):
                    continue
                for bi, b in body_spans:
                    seg = kept[b['start']:b['end']]
                    if term in seg:
                        hit = bi
                        break
                if hit is not None:
                    break
            if hit is None:
                continue
            uid = 'n{0}'.format(u['block'])
            cur = _block_text(hit).rstrip('\n')
            nl = '\n' if blocks[hit]['seg'].endswith('\n') else ''
            seg_override[hit] = cur + '[^' + uid + ']' + nl
            seg_override[u['block']] = '[^{0}]: {1}\n'.format(uid, u['content'])
            attached += 1
            changes["orphans_attached"] += 1
        if members and attached == len(members):
            dropped.add(tu['block'])

    # 3c) 空壳注释标题（无任何条目的【注释】块）：删除
    # 仅限"【注释】"制式（结构化编号注的标准标题）；【译注】等可承载
    # 纯段落说明文字，标题存在即合理，不做清理.
    _shell_re = re.compile(r'^\s*\*{0,2}\s*【注释】\s*\*{0,2}\s*$')
    for u in units:
        if u['kind'] != 'title' or 'group' in u or u['block'] in dropped:
            continue
        if _shell_re.match(blocks[u['block']]['seg'].split('\n', 1)[0]):
            dropped.add(u['block'])
            changes["title_shells_dropped"] += 1

    # 4) 章末标题块归位（【章末注释】等；已在章末者不动）
    end_blocks = []
    for u in units:
        if u['kind'] != 'title' or not u.get('title_end') or 'group' not in u:
            continue
        if u['block'] in dropped:
            continue
        grp = [x for x in units if x.get('group') == u['group']]
        last = max(x['block'] for x in grp)
        if not any(x['kind'] == 'body' and x['block'] > last for x in units):
            continue
        movers = [x for x in grp
                  if x['block'] not in dropped and x['block'] not in seg_override]
        movers.sort(key=lambda x: x['block'])
        for x in movers:
            dropped.add(x['block'])
            end_blocks.append(blocks[x['block']]['seg'].rstrip('\n'))
        if movers:
            changes["title_blocks_moved"] += 1

    # 5) 组装
    out_parts, cur = [], 0
    for bi, b in enumerate(blocks):
        out_parts.append(text[cur:b['start']])
        if bi in dropped:
            seg = ''
        elif bi in seg_override:
            seg = seg_override[bi]
        else:
            seg = b['seg']
        out_parts.append(seg)
        cur = b['end']
    out_parts.append(text[cur:])
    assembled = ''.join(out_parts)
    tail = []
    if end_blocks:
        tail.append('---\n\n' + '\n\n'.join(end_blocks))
    if num_fail:
        tail.append('---\n\n**【译注】**\n\n' + '\n\n'.join(num_fail))
    if md_fail:
        tail.append('---\n\n' + '\n\n'.join('> **译者注：** ' + c for c in md_fail))
    if tail:
        assembled = assembled.rstrip('\n') + '\n\n' + '\n\n'.join(tail) + '\n'

    # 6) 顺序重编号（全书连续；被去重旧 id 的引用先并入幸存者）
    if remap:
        assembled = re.sub(
            r'\[\^([^\]]+)\](?!:)',
            lambda m: '[^' + remap.get(m.group(1), m.group(1)) + ']', assembled)
    order = []
    for m in re.finditer(r'\[\^([^\]]+)\](?!:)', assembled):
        if m.group(1) not in order:
            order.append(m.group(1))
    for m in re.finditer(r'\[\^([^\]]+)\]:', assembled):
        if m.group(1) not in order:
            order.append(m.group(1))
    mapping = {old: str(start_num + i) for i, old in enumerate(order)}

    def _sub_ref(m):
        return '[^' + mapping.get(m.group(1), m.group(1)) + ']'

    assembled = re.sub(r'\[\^([^\]]+)\](?!:)', _sub_ref, assembled)
    assembled = re.sub(r'\[\^([^\]]+)\]:',
                       lambda m: '[^' + mapping.get(m.group(1), m.group(1)) + ']:',
                       assembled)
    changes["renumbered"] = len(order)

    # 6.5) 脚注定义章末集中（raw md 可读性；编辑器多次建议"集中至章末"）
    def_lines = {}
    kept_lines = []
    for line in assembled.split('\n'):
        m = re.match(r'^\[\^(\d+)\]:\s(.*)$', line)
        if m:
            def_lines[int(m.group(1))] = line
        else:
            kept_lines.append(line)
    if def_lines:
        while kept_lines and kept_lines[-1].strip() == '':
            kept_lines.pop()
        if kept_lines and kept_lines[-1].strip() == '---':
            kept_lines.pop()
        while kept_lines and kept_lines[-1].strip() == '':
            kept_lines.pop()
        assembled = ('\n'.join(kept_lines).rstrip('\n') + '\n\n---\n\n'
                     + '\n'.join(def_lines[k] for k in sorted(def_lines)) + '\n')
    return assembled, changes, start_num + len(order)


def normalize_final_style(text: str):
    """T5 规范化 pass（幂等）。Returns: (new_text, changes dict)。"""
    changes = {"headers_removed": 0, "sup_defs_converted": 0, "md_defs_kept": 0,
               "orphans_relocated": 0, "cjk_quotes": 0, "straight_quotes": 0}
    items, defs, refs = _scan_lines(text)
    out = []
    relocated = []
    for kind, payload in items:
        if kind == 'header':
            changes["headers_removed"] += 1
            continue
        if kind == 'def':
            d = payload
            if d['style'] == 'num':
                relocated.append(d['content'])
                continue
            if d['id'] in refs:
                out.append("[^" + str(d['id']) + "]: " + d['content'])
                if d['style'] == 'sup':
                    changes["sup_defs_converted"] += 1
                else:
                    changes["md_defs_kept"] += 1
            else:
                relocated.append(d['content'])
            continue
        line = payload
        line = _SUP_REF_RE.sub(lambda m: "[^" + str(sup_run_to_int(m.group(0))) + "]", line)
        out.append(line)
    if relocated:
        changes["orphans_relocated"] = len(relocated)
        while out and out[-1].strip() == '':
            out.pop()
        if out and out[-1].strip() == '---':
            out.pop()
        out.append('')
        for content in relocated:
            out.append('> **译者注：** ' + content)
    t = '\n'.join(out)
    if '「' in t or '」' in t:
        changes["cjk_quotes"] += t.count('「') + t.count('」')
        t = t.replace('「', '“').replace('」', '”')
    if '『' in t or '』' in t:
        changes["cjk_quotes"] += t.count('『') + t.count('』')
        t = t.replace('『', '‘').replace('』', '’')
    if '"' in t:
        changes["straight_quotes"] = t.count('"')
        parts = t.split('"')
        rebuilt = parts[0]
        for k, p in enumerate(parts[1:]):
            rebuilt += ('“' if k % 2 == 0 else '”') + p
        t = rebuilt
    return t, changes


def normalize_notes_document(text: str):
    """全书：按 H1 切章逐章统一（编号全书连续）。→ (new_text, report)"""
    parts = re.split(r'(?m)^(# .*)$', text)
    agg = {"merged_dups": 0, "orphans_attached": 0, "md_relocated": 0,
           "num_relocated": 0, "title_blocks_moved": 0, "renumbered": 0,
           "title_shells_dropped": 0,
           "machine_ids": 0}
    details = []
    counter = 1
    pre_new, ch0, counter = normalize_chapter_notes(parts[0], counter)
    for key in agg:
        agg[key] += ch0.get(key, 0)
    if any(ch0.values()):
        details.append(('(前置)', ch0))
    out_txt = [pre_new]
    for k in range(1, len(parts), 2):
        head = parts[k]
        body = parts[k + 1] if k + 1 < len(parts) else ''
        body_new, ch, counter = normalize_chapter_notes(body, counter)
        for key in agg:
            agg[key] += ch.get(key, 0)
        if any(ch.values()):
            details.append((head.strip(), ch))
        out_txt.append(head)
        out_txt.append(body_new)
    report = {"stats": agg, "chapters": details, "next_num": counter}
    return ''.join(out_txt), report


def notes_normalize_outputs(project_root=None, output_dir=None, apply: bool = False) -> dict:
    """对 translated_full.md 执行章级注释统一；默认 dry-run，写 output/note_policy_diff.md。"""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    merged = out_root / "translated_full.md"
    stats = {"files": 0, "changed": 0, "merged_dups": 0, "orphans_attached": 0,
             "md_relocated": 0, "num_relocated": 0, "title_blocks_moved": 0,
             "renumbered": 0, "title_shells_dropped": 0, "machine_ids": 0}
    details = []
    if merged.exists():
        stats["files"] = 1
        text = merged.read_text(encoding="utf-8")
        new_text, rep = normalize_notes_document(text)
        if new_text != text:
            stats["changed"] = 1
            for key in ("merged_dups", "orphans_attached", "md_relocated",
                        "num_relocated", "title_blocks_moved", "renumbered",
                        "title_shells_dropped", "machine_ids"):
                stats[key] = rep["stats"].get(key, 0)
            details = rep["chapters"]
            if apply:
                merged.write_text(new_text, encoding="utf-8")
    lines = ["# 章级注释统一报告 (note_policy_diff)", "",
             "- 模式: {0}".format('APPLY (已写入)' if apply else 'DRY RUN (仅报告)'),
             "- 处理 file: {0} 个；需要变更: {1} 个".format(stats['files'], stats['changed']),
             "- 去重合并: {0}；孤儿改锚: {1}；md 转译者注: {2}；编号注转译注块: {3}".format(
                 stats['merged_dups'], stats['orphans_attached'],
                 stats['md_relocated'], stats['num_relocated']),
             "- 标题块搬移: {0}；空壳标题清理: {1}；顺编号: {2}；机器 id 清除: {3}".format(
                 stats['title_blocks_moved'], stats['title_shells_dropped'],
                 stats['renumbered'], stats['machine_ids']),
             ""]
    if details:
        lines.append("## 明细")
        lines.append("")
        for label, ch in details:
            lines.append("- `{0}`: {1}".format(
                label, ", ".join("{0}={1}".format(k, v) for k, v in ch.items() if v)))
    else:
        lines.append("（无需变更）")
    lines += ["", "---", "*由 notes-normalize 子命令自动生成*", ""]
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "note_policy_diff.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"stats": stats, "report": str(report_path), "applied": bool(apply)}


def postprocess_chunk_text(text: str, chapter_id: str, chunk_idx: int) -> str:
    """merge 前处理：剥离页眉/版本行残留 + chunk 内嵌章节标题行。"""
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if _FRONTMATTER_LINE_RE.match(s):
            continue
        if _CHAPTER_HEAD_RE.match(s):
            continue
        out.append(line)
    return '\n'.join(out)


def sup_run_to_int(run: str) -> int:
    """上标数字串 → 整数（"¹²" → 12）。"""
    return int(''.join(str(_SUPER.index(c) + 1) for c in run))



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
    TypeSafeAPIResponseValidationError = TypeSafeAPIError = Exception






# Section 11.6 — typesafe-ai judges（多原子原语 + verdict 合成）
# ────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field, asdict


@dataclass
class NoulAnswer:
    """Noul 答案：0..1 真值分数（SDK 无 confidence 字段）

    Round 60：typesafe_sdk 0.7.1 的 NoulAnswer schema 只有 type + noul，没有 confidence。
    Round 83：自动派生 confidence 字段，消除 ans.confidence 与 to_dict()["confidence"]
    的语义不一致 —— 之前 ans.confidence 返回 0.0 (默认), 但 to_dict() 已按
    typesafe-ai 一致性原则计算 max(noul, 1-noul). 现在 __post_init__ 会把
    显式传 confidence==0.0 (sentinel) 的实例自动用 max(noul, 1-noul) 填充,
    让 ans.confidence 与 to_dict() 返回一致.

    示例:
      NoulAnswer(noul=0.39).confidence   == 0.61   (eff_conf, 自动派生)
      NoulAnswer(noul=0.5).confidence    == 0.5    (eff_conf, 自动派生)
      NoulAnswer(noul=0.7, confidence=0.3).confidence == 0.3   (显式传, 不覆盖)
    """
    noul: float
    confidence: float = 0.0

    def __post_init__(self):
        # Round 83: confidence == 0.0 (sentinel for "未设置") 时自动从 noul 派生
        # 显式传 confidence > 0 时不会被覆盖 (向后兼容)
        #
        # Round 91: 防御性 sanity 校验 — 若显式传的 confidence 越界 (<0 或 >1),
        # 视为无效值, 强制派生. 防止 LLM 返回 confidence=1.5 / -0.5 等异常
        # 直接污染 primitive_summary, 下游 verdict 会用 max(noul, 1-noul) 聚合
        # 但 critic_report.json 里出现 conf=1.5 会让 cost_report / QA 报告误读.
        # 派生策略: 越界 → 与 0.0 sentinel 同样处理 (用 eff_conf).
        if self.confidence == 0.0 or not (0.0 <= self.confidence <= 1.0):
            self.confidence = max(self.noul, 1.0 - self.noul)

    def to_dict(self) -> Dict[str, Any]:
        # Round 83: 直接用 self.confidence (已自动派生), 避免重复计算
        return {"type": "noul", "noul": self.noul, "confidence": self.confidence}



@dataclass
class ChoiceAnswer:
    """Choice 答案：选中项 + 各选项概率 + confidence"""
    choice: str
    probabilities: Dict[str, float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }



@dataclass
class ScoreAnswer:
    """Score 答案：rubric 位置 + 各位置概率 + confidence"""
    score: float
    probabilities: Dict[int, float]  # level_idx -> probability
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "probabilities": {str(k): v for k, v in self.probabilities.items()},
            "confidence": self.confidence,
        }



@dataclass
class CompositeVerdict:
    """Code 合成的最终裁决（基于多个 primitive 答案）

    Round 62 新增字段：
      - skipped_dimensions: 被合成器跳过的 dimension 列表
        （Noul 在 uncertain zone / Choice 低 confidence / Score 低 confidence）。
        下游 consumer 可以据此区分"明确 PASS"（assert 0 violation）与
        "不确定跳过"（Jev 给出边界值，未达到 reject 门槛）。
      - skipped_reasons: 跳过原因明细（dict：dimension → reason 字符串）

    Round 118 P1-1 新增字段：
      - confidence_source: 概率来源（"jev"=System One 校准概率 /
        "openai_compat"=LLM 自报伪概率 / "mock_heuristic"=确定性启发式 /
        "_uncalibrated"=未知来源按未校准处理）。
      - confidence_calibrated: False 表示阈值已按未校准来源收紧，
        PASS 未经 Jev 认证（降级模式审计标记）。
    """
    decision: str  # "PASS" | "REJECT"
    confidence: float  # 综合置信度 0..1
    reject_reasons: List[str] = field(default_factory=list)
    primitive_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skipped_dimensions: List[str] = field(default_factory=list)
    skipped_reasons: Dict[str, str] = field(default_factory=dict)
    confidence_source: str = "jev"
    confidence_calibrated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



from typing import Union
Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


import copy


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


from typing import Optional
_DEFAULT_BACKEND: Optional["SystemOneBackend"] = None


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


# Round 72/87: synthesize_verdict / MockBackend 调用的 primitives.py 依赖

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



# ───── src.judges.* 兼容 shim（单文件部署模式） ─────

# Round 102 P1: compute_cjk_ratio 按 id(text) 缓存. 与散装 src/utils/text_match.py 同步.
# 简化版: cache 直接持有 str 强引用 (key=id(text)). 见散装版注释了解 id 复用 trade-off.
_cjk_ratio_cache: "dict[int, float]" = {}
_CJK_RATIO_CACHE_MAX = 512


def _cjk_ratio_cache_get(text):
    return _cjk_ratio_cache.get(id(text))


def _cjk_ratio_cache_put(text, ratio):
    if len(_cjk_ratio_cache) >= _CJK_RATIO_CACHE_MAX:
        try:
            oldest_key = next(iter(_cjk_ratio_cache))
            _cjk_ratio_cache.pop(oldest_key, None)
        except StopIteration:
            pass
    _cjk_ratio_cache[id(text)] = ratio


def reset_cjk_ratio_cache() -> None:
    """Chapter 切换时清空 cjk_ratio 缓存 (整合模式 shim 入口)."""
    _cjk_ratio_cache.clear()


def compute_cjk_ratio(text: str) -> float:
    """计算 CJK 字符占非空白字符的比例.

    Round 96: 复制到 translator_agent.py 内联 (避免虚拟 src.utils.text_match 包找不到).
    Round 102 P1: 加 id(text) 缓存. 散装 src.utils.text_match.compute_cjk_ratio 仍是事实源.
    """
    if not text:
        return 0.0
    cached = _cjk_ratio_cache_get(text)
    if cached is not None:
        return cached
    import re as _re
    stripped = _re.sub(r"\s+", "", text)
    if not stripped:
        _cjk_ratio_cache_put(text, 0.0)
        return 0.0
    cjk_count = 0
    for ch in stripped:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            cjk_count += 1
    ratio = cjk_count / len(stripped)
    _cjk_ratio_cache_put(text, ratio)
    return ratio


# ───── Round 105: 实体变体普通词护栏 (镜像 src/utils/entity_stopwords.py) ─────
# 单文件部署时虚拟 src.utils.entity_stopwords 包找不到, 这里内联定义供 shim 填入.
# 散装 src/utils/entity_stopwords.py 仍是事实源; 修改时两处需同步.
_ENTITY_COMMON_CHARS: frozenset = frozenset(
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
    "尔意条状点再做外孩声西果将头把分还认问由什每没口体全机重明力理信马步听真话别打走叫见远近深浅宽窄厚薄粗细长短高低大小多少新旧好坏美丑真假对错是非黑白红黄蓝绿金银铜铁石木水火土风雨雪云雷电光明暗冷热温凉干湿软硬轻重快慢早晚今明昨昼夜晨昏朝夕春夏秋冬东南西北中央"
    "警察官方司督已去过来回到达至向往朝向对面前后左右上下里外内中间旁边侧邻远近深浅"
    "东西南北上下左右前后里外中间旁边侧近远深浅宽窄厚薄粗细长短高低大小多少新旧好坏美丑真假对错是非"
    "谁啥哪咋吗呢吧啊呀嘛啦哇哟呐哦噢嗯嘿喂哎唉喔"
    "金来最时可可高可时警金来最时"
)
_ENTITY_COMMON_WORDS: frozenset = frozenset({
    "东西", "卡特", "警察", "警督", "警已", "警方", "警惕", "警司", "警官",
    "材敦", "普敦", "伦次",
    "伦敦金", "伦敦来", "伦敦最", "伦敦高", "伦敦时", "伦敦警", "伦敦可",
    "我们", "你们", "他们", "她们", "它们", "自己", "什么", "怎么", "这样",
    "那样", "这个", "那个", "这些", "那些", "没有", "不是", "可以", "已经",
    "因为", "所以", "但是", "如果", "虽然", "然而", "于是", "然后", "接着",
    "突然", "忽然", "渐渐", "慢慢", "轻轻", "重重", "深深", "浅浅",
    "时候", "时间", "地方", "事情", "问题", "办法", "意思", "感觉",
    "声音", "光线", "空气", "天气", "日子", "年月", "世界", "国家", "人民",
    "生活", "工作", "学习", "学校", "家庭", "朋友", "老师", "学生", "孩子",
    "大人", "老人", "青年", "少年", "男人", "女人", "丈夫", "妻子", "父亲",
    "母亲", "哥哥", "姐姐", "弟弟", "妹妹", "叔叔", "阿姨", "舅舅", "姨妈",
    "警卫", "警局", "警长", "警方", "警官", "警司", "警督", "警惕",
})
# 与散装 src/utils/entity_stopwords.py 的公开名对齐 (shim 按此名暴露)
COMMON_CHARS = _ENTITY_COMMON_CHARS
COMMON_WORDS = _ENTITY_COMMON_WORDS


def is_likely_common_word(word: str) -> bool:
    """Round 105: 判定中文词是否为已确认普通词 (仅依赖黑名单, 不用高频字占比)."""
    if not word:
        return True
    return word in _ENTITY_COMMON_WORDS


def differing_char(canonical: str, candidate: str):
    """Round 105: 返回 canonical 与 candidate 唯一不同字符; 非一字之差返回 None."""
    if not canonical or not candidate or len(canonical) != len(candidate):
        return None
    diff_pos = -1
    diff_count = 0
    for i, (a, b) in enumerate(zip(canonical, candidate)):
        if a != b:
            diff_count += 1
            diff_pos = i
    if diff_count != 1:
        return None
    return candidate[diff_pos]


def is_plausible_name_variant(canonical: str, candidate: str):
    """Round 105: 判定 candidate 是否可能是 canonical 的人名/地名异体 (一字之差).

    返回 (是否可信, 原因). 可信: 恰好一字之差 + 不在普通词黑名单 + 差异字非高频字.
    """
    if not canonical or not candidate:
        return False, "empty"
    if candidate == canonical:
        return False, "same_as_canonical"
    dchar = differing_char(canonical, candidate)
    if dchar is None:
        return False, "not_one_char_diff"
    if is_likely_common_word(candidate):
        return False, f"candidate_is_common_word({candidate})"
    if dchar in _ENTITY_COMMON_CHARS:
        return False, f"differing_char_is_common({dchar})"
    return True, "ok"


def flagged_entity_labels(answers, noul_threshold, entity_registry_full=None):
    """Round 105 P2-3: 列出 per-entity 违规实体标签（镜像 src/judges/verdict.py）。

    供 synthesize_verdict 把违规实体映射到规范主名，生成可执行拒因。
    """
    from src.judges.primitives import PER_ENTITY_KEY_PREFIX as _PEP
    labels = []
    bar = max(float(noul_threshold), 0.5)
    key_to_src = {}
    if entity_registry_full:
        from src.judges.primitives import _sanitize_key, PER_ENTITY_KEY_PREFIX as _PEP
        used = set()
        for sn in entity_registry_full:
            if sn:
                # Round 108: _sanitize_key 返回带 entity_per_ 前缀的 key，而查表侧用
                # 去前缀的 base_key — 修前多词 source_name 的规范名映射永远失效
                # （ch07 chunk016 根因：拒因只剩不可执行的 sanitized key）。
                key_to_src[_sanitize_key(sn, used)[len(_PEP):]] = sn
    for qid, ans in answers.items():
        if not qid.startswith(_PEP):
            continue
        if not hasattr(ans, "noul"):
            continue
        noul = float(ans.noul)
        if (1.0 - noul) < bar:
            continue
        sanitized = qid[len(_PEP):]
        base_key = sanitized.rsplit("_", 1)[0] if sanitized.rsplit("_", 1)[-1].isdigit() else sanitized
        src_name = key_to_src.get(base_key, sanitized)
        if entity_registry_full and src_name in entity_registry_full:
            info = entity_registry_full[src_name]
            canonical = info.get("canonical", "") if isinstance(info, dict) else str(info)
            if canonical:
                labels.append(f"{src_name}→{canonical}(noul={noul:.2f})")
                continue
        labels.append(f"{src_name}(noul={noul:.2f})")
    labels.sort(key=lambda s: float(s.rsplit("noul=", 1)[-1].rstrip(")")))
    return labels[:8]


# R104 P6: shim 创建假模块时从 globals() 取属性, 之前这两个 const 未在 monolith
# 内联定义, 导致 fake.src.main 没有 _INPUT_FORMAT_TABLE, fake.src.splitters 没有
# _EPUB_DEPS_OK, CLI split 报 "无法识别输入格式 / 缺少依赖". 这里显式定义,
# 与 src/main.py 和 src/splitters.py 同步, 让 shim block 填入.
_INPUT_FORMAT_TABLE = {'.epub': 'epub', '.txt': 'txt', '.md': 'md', '.markdown': 'md'}
_EPUB_DEPS_OK = True
_EPUB_DEPS_ERROR = ''

import sys as _sys_j
import types as _types_j
# 先注册父包「src」为 package（需要 __path__ 才能 import 子模块）
# Round 103 P4: 只在 sys.modules 完全缺失时创建假 ModuleType. 若已有
# 真实 package 加载（__path__ 含子目录）, 不要再覆盖 — 否则后续
# `from src.core.decision_engine import X` 会返回 __file__=None 的
# 空模块 (Round 71 fixture cleanup 后导致 test_round70→test_round71 连锁失败).
# 单文件部署场景 (standalone 运行本文件, sys.modules 完全干净) 走
# "if not in sys.modules" 分支创建假包, 与旧行为一致.
if "src" not in _sys_j.modules:
    _src_pkg = _types_j.ModuleType("src")
    _src_pkg.__path__ = []
    _sys_j.modules["src"] = _src_pkg
# Round 117 P1: 优先加载真实包/模块，仅在磁盘上确实不存在时才创建假模块。
# 旧行为：只要目标尚未 import，就先塞一个 __path__=[] 的假包进 sys.modules；
# 此后 `from src.judges.pairwise import X`（任何新增子模块）都会报
# "No module named 'src.judges.pairwise'; 'src.judges' is not a package"（新模块地雷）。
# 单文件部署（磁盘无 src 包）时 import 失败 → 仍走假模块兜底，行为不变。
def _ensure_real_or_fake(_name: str) -> None:
    if _name in _sys_j.modules:
        return
    try:
        import importlib as _il_j
        _il_j.import_module(_name)
        return
    except Exception:
        pass
    _fake = _types_j.ModuleType(_name)
    _fake.__path__ = []
    _sys_j.modules[_name] = _fake

_ensure_real_or_fake("src.judges")
# 其他 src.* 子包（如 critic 内部 from src.utils.llm_adapter import）
for _sub in ("src.utils", "src.config", "src.agents", "src.core", "src.consistency", "src.splitters", "src.pipeline", "src.main"):
    _ensure_real_or_fake(_sub)
# Round 103 P2: src.judges.* shim 不覆盖真实模块, 仅补 attr.
# R104 P1 修复: 旧条件 `_existing.__path__` 仅对父 package 有效, 子模块
# (typesafe_spec / primitives / typesafe_backend / verdict / judge) 没有
# __path__, 旧逻辑把它们误判为"未加载", 走 else 创建无 __file__ 的假 ModuleType
# 覆盖 sys.modules, 导致下游 `from src.judges.typesafe_spec import X` 拿到
# 假模块对象 (与真实模块 identity 不一致, isinstance 检查挂掉). 改用 __file__
# 真值 (与 R103 P5 第二个块一致), 真实子模块已加载时只补 attr, 不替换.
#   (a) 真实模块已加载: 只补充暴露整合脚本中的符号到现有模块.
#   (b) 未加载或占位符: 创建假 ModuleType 兜底 (单文件部署).
for _mod_name, _attrs in {
    "src.judges.typesafe_spec": ["NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "CompositeVerdict", "Answer"],
    # Round 87 修复: 旧 shim 漏掉了 primitives.py 的 PER_ENTITY_KEY_PREFIX 常量 /
    # build_per_entity_questions / _sanitize_key 函数, 整合脚本中 MockBackend 等位置
    # 的 `from src.judges.primitives import PER_ENTITY_KEY_PREFIX` 触发 ImportError.
    "src.judges.primitives": ["VOICE_DRIFT", "REGISTER_MATCH", "ENTITY_CONSISTENCY", "LITERAL_RESIDUE", "SEMANTIC_LOSS", "STYLE_COMPLIANCE_SCORE", "ALL_PRIMITIVES", "build_question_set", "PER_ENTITY_KEY_PREFIX", "build_per_entity_questions", "_sanitize_key", "aggregate_per_entity_nouls", "entity_present_in_text", "apply_entity_presence_override"],
    "src.judges.typesafe_backend": ["SystemOneBackend", "JevBackend", "OpenAICompatBackend", "MockBackend", "FallbackBackend", "get_backend", "_DEFAULT_BACKEND"],
    "src.judges.verdict": ["synthesize_verdict", "DEFAULT_THRESHOLDS",
                           "flagged_entity_labels"],
    "src.judges.judge": ["build_state", "evaluate_chunk", "to_legacy_scores"],
    "src.judges": ["NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "CompositeVerdict", "Answer",
                   "VOICE_DRIFT", "REGISTER_MATCH", "ENTITY_CONSISTENCY", "LITERAL_RESIDUE", "SEMANTIC_LOSS", "STYLE_COMPLIANCE_SCORE", "ALL_PRIMITIVES", "build_question_set",
                   "PER_ENTITY_KEY_PREFIX", "build_per_entity_questions", "_sanitize_key", "aggregate_per_entity_nouls",
                   "entity_present_in_text", "apply_entity_presence_override",
                   "SystemOneBackend", "JevBackend", "OpenAICompatBackend", "MockBackend", "FallbackBackend", "get_backend",
                   "synthesize_verdict", "DEFAULT_THRESHOLDS",
                   "build_state", "evaluate_chunk", "to_legacy_scores"],
}.items():
    _existing = _sys_j.modules.get(_mod_name)
    # R104 P1: 用 __file__ 真值判断 (与 R103 P5 第二块对齐).
    # __path__ 仅对父 package 有效, 子模块 (typesafe_spec 等) 无 __path__ 会误判.
    if _existing is not None and hasattr(_existing, '__file__') and _existing.__file__:
        # (a) 真实模块已加载 — 不覆盖, 只补充暴露整合脚本中的符号到现有模块.
        # 这些符号 (如 NoulAnswer) 在真实 src/judges/typesafe_spec.py 里也存在,
        # 不会与真实模块冲突 (函数签名相同).
        for _a in _attrs:
            if _a in globals() and not _a.startswith("_sys_") and not _a.startswith("_types_"):
                if not hasattr(_existing, _a):
                    setattr(_existing, _a, globals()[_a])
    else:
        # (b) 未加载或占位符 — 创建假 ModuleType 兜底 (单文件部署).
        _m = _types_j.ModuleType(_mod_name)
        for _a in _attrs:
            if _a in globals() and not _a.startswith("_sys_") and not _a.startswith("_types_"):
                setattr(_m, _a, globals()[_a])
        _sys_j.modules[_mod_name] = _m

# ───── 把散装模块的内容塞进对应的 src.* 虚拟子包 ─────
# 这样 `from src.utils.llm_adapter import ...` 等也能工作
_SRC_SUBMODULES = {
    "src.utils.llm_adapter": [
        "get_llm_client", "get_llm_client_for_task", "get_role_model_name", "get_task_extra_body",
        "robust_json_loads", "_build_role_client", "_client_instances",
    ],
    # Round 96: 加 text_match shim 让 CJK 比例原语可在整合模式下访问
    # Round 102 P1: 增加 cjk_ratio 缓存相关符号, 让整合模式下也可访问.
    "src.utils.text_match": [
        "compute_cjk_ratio", "extract_reject_categories",
        "_normalize_category", "_CATEGORY_SYNONYMS", "case_insensitive_find",
        "_cjk_ratio_cache", "_CJK_RATIO_CACHE_MAX",
        "reset_cjk_ratio_cache",
        # Round 114: 词边界实体在场判定（ch30 chunk000 "Bert"⊂"Bertillon" 根因）
        "entity_source_name_in_text",
        # Round 133b: span 版原语 + 共享边界正则（别名吞并抑制）
        "entity_source_name_spans", "_entity_boundary_pattern",
    ],

    "src.utils.chunker": ["SmartChunker"],
    # Round 105: 实体变体普通词护栏 shim (镜像 src/utils/entity_stopwords.py)
    "src.utils.entity_stopwords": [
        "COMMON_CHARS", "COMMON_WORDS", "is_likely_common_word",
        "differing_char", "is_plausible_name_variant",
        "_ENTITY_COMMON_CHARS", "_ENTITY_COMMON_WORDS",
    ],
    "src.config": ["Config", "get_config", "config"],
    "src.core.scheduler": ["TaskState", "TaskScheduler", "VALID_TRANSITIONS"],
    "src.core.decision_engine": [
        "DecisionLevel", "DecisionEngine", "query_high_confidence_entities",
        "query_reviewable_entities", "get_sibling_canonicals_by_source",
        "detect_cross_person_conflicts", "seed_entity_variants", "SEED_ENTITY_VARIANTS",
        # Round 105 P2-2: 同人多规范名归一
        "unify_entity_registry", "_CONF_ORDER",
    ],
    "src.agents.reference_agent": ["ReferenceAgent"],
    "src.agents.rewriter_agent": ["LiteraryRewriterAgent"],
    "src.agents.critic_agent": ["CriticAgent"],
    "src.agents.judge_agent": ["JudgeAgent"],
    "src.pipeline": ["TranslationPipeline", "consolidate_dirty_chapters", "run_final_gate"],
    # Round 119 (T5): 成书样式/lint 单源（散装 src/book_lint.py）
    "src.book_lint": ["normalize_final_style", "finalize_outputs", "lint_texts",
                      "lint_book", "postprocess_chunk_text", "chapter_heading",
                      "sup_run_to_int",
                      # L2/L6: pun_audit / literary_metrics 复用的私有 helper
                      "_scan_lines", "_def_key",
                      # Round 121: 注释统一新面（editor_review / 测试复用）
                      "_split_h1_sections", "normalize_chapter_notes",
                      "normalize_notes_document", "notes_normalize_outputs",
                      "_chapter_note_dupes", "_note_is_dup", "_note_anchor_stop"],
    "src.consistency": ["run_consistency_check", "_load_glossary", "_load_decisions",
                        "_scan_finals", "_find_slash_aliases", "_in_quote_or_note",
                        "_NOTE_MARKER_RE",
                        "_detect_issues", "_generate_report", "_apply_issues",
                        "run_consistency_check_for_chapter",
                        # Round 105: 普通词护栏 helper
                        "_is_common_word_variant"],
    "src.splitters": ["EpubSplitter", "TextSplitter", "MdSplitter",
                      "_splitter_log_info", "_splitter_log_ok", "_splitter_log_warn", "_splitter_log_err",
                      "_normalize_chapter_title", "_normalize_scores",
                      "_epub_clean_html_text", "_strip_epub_front_matter",
                      "_epub_split_large_doc_by_toc", "_epub_extract_chapters",
                      "_text_detect_chapter_title",
                      # Round 119 T1: 转换完整性审计与标题提取
                      "_epub_extract_title", "_check_title_present",
                      "run_conversion_audit",
                      # Round 119 L1: 韵文元素标记 / 清洗扩展 / 清单构建
                      "_mark_verse_elements", "_el_verse_tag",
                      "_epub_clean_html_text_ex", "_build_verse_manifest",
                      # R104 P4: split 入口 (translator_agent.py:7064/9469) 依赖此常量
                      "_EPUB_DEPS_OK",
                      # Round 130 修复: input_prep 助手（_apply_input_prep 依赖）
                      "_input_prep_cfg", "_prep_key", "_chapter_h1", "_apply_input_repairs",
                      # TextSplitter 标题正则（_text_detect_chapter_title 依赖）
                      "_CHAPTER_TITLE_PATTERNS"],
    # Round 119 L1: 韵文通道单源（散装 src/verse.py）
    "src.verse": [
        "verse_config", "normalize_whitespace",
        "load_verse_manifest", "save_verse_manifest",
        "load_chunk_tags", "save_chunk_tags",
        "locate_chunk_spans", "blocks_for_span",
        "fragments_from_manifest", "fragments_for_chunk",
        "build_chapter_tags", "build_verse_note", "verse_lint",
        "VERSE_MANIFEST_VERSION", "CHUNK_TAGS_VERSION", "_DEFAULT_VERSE_CONFIG",
        "_normalized_index", "_block_lines_in_range",
        "_max_line_repeat", "_longest_short_line_run",
    ],
    "src.main": ["init_project", "debug_db", "print_banner", "main",
                 "detect_input_format", "split_input_to_chapters",
                 # R104 P4: detect_input_format (translator_agent.py:9433) 依赖此 table
                 "_INPUT_FORMAT_TABLE"],
    "src.test_golden_set": ["GoldenSetEvaluator", "_levenshtein_distance", "run_golden_test",
                            "GOLDEN_AB_MATRIX", "_extract_confidence_source",
                            "summarize_golden_results", "recommend_golden_combo", "run_golden_ab",
                            "run_golden_gate"],
    "src.test_memory_pressure": ["get_system_memory", "get_process_memory", "simulate_memory_load", "test_memory_pressure"],
}
# Round 103 P5: 第二个 shim 块 (源散装模块) 也需要保护真实包.
# 旧代码无条件 `_sys_j.modules[_mod] = _m` 覆盖, 当散装模块已加载时
# (例如先 `import src.core.decision_engine` 后再 `import src.translator_agent`),
# 会丢失原模块的 __file__/__loader__ 等属性, 触发 ImportError.
# 规则与上面 src.judges.* 块一致: 真实包已加载时只补 attr, 不覆盖.
for _mod, _attrs in _SRC_SUBMODULES.items():
    _existing = _sys_j.modules.get(_mod)
    if _existing is not None and hasattr(_existing, '__file__') and _existing.__file__:
        # (a) 真实模块已加载 — 不覆盖, 只补充暴露整合脚本中的符号.
        for _a in _attrs:
            if _a in globals() and not hasattr(_existing, _a):
                setattr(_existing, _a, globals()[_a])
    else:
        # (b) 未加载或被占位符 — 创建假 ModuleType 兜底 (单文件部署).
        _m = _types_j.ModuleType(_mod)
        for _a in _attrs:
            if _a in globals():
                setattr(_m, _a, globals()[_a])
        _sys_j.modules[_mod] = _m
# 重新绑定到顶层 import 路径，便于直接 from src.judges.X import Y
from src.judges.typesafe_spec import (NoulAnswer, ChoiceAnswer, ScoreAnswer, CompositeVerdict, Answer)
from src.judges.primitives import (VOICE_DRIFT, REGISTER_MATCH, ENTITY_CONSISTENCY, LITERAL_RESIDUE, SEMANTIC_LOSS, STYLE_COMPLIANCE_SCORE, ALL_PRIMITIVES, build_question_set)
from src.judges.typesafe_backend import (SystemOneBackend, JevBackend, OpenAICompatBackend, MockBackend, FallbackBackend, get_backend)
from src.judges.verdict import (synthesize_verdict, DEFAULT_THRESHOLDS)
from src.judges.judge import (build_state, evaluate_chunk, to_legacy_scores)

# Round 96: compute_cjk_ratio 在文件顶部 inline 定义, 镜像 src/utils/text_match.py, 避免 shim 后的虚拟 src.utils.text_match 包缺失该符号


# Section 12 — 统一入口点 (split_input_to_chapters)
# ────────────────────────────────────────────────────────────────────

def detect_input_format(file_path: str) -> str:
    """根据文件扩展名自动检测输入格式"""
    suffix = Path(file_path).suffix.lower()
    # R104 P5: 走 src.main namespace 取 shim 注入的 table, 而不是裸名
    # (monolith 内联不再维护 _INPUT_FORMAT_TABLE, 避免与散装组件相撞).
    _fmt_table = getattr(sys.modules.get("src.main"), "_INPUT_FORMAT_TABLE", None) or {}
    fmt = _fmt_table.get(suffix)
    if fmt is None:
        raise ValueError(
            f"无法识别输入格式: {file_path}\n"
            f"支持的扩展名: {list(_fmt_table.keys())}"
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
    # R104 P5: 走 src.splitters namespace 取 shim 注入的常量, 避免裸名 NameError
    if input_format == 'epub':
        _epub_deps_ok = getattr(sys.modules.get("src.splitters"), "_EPUB_DEPS_OK", True)
        if not _epub_deps_ok:
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
    _audit = getattr(splitter, "audit_result", None) or {}
    if _audit and _audit.get("ok") is False:
        _splitter_log_err(
            f"转换完整性审计未通过（覆盖率 {_audit.get('overall', 0.0) * 100:.3f}%），"
            f"已中止；报告: {_audit.get('report_path')}")
        sys.exit(1)
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


# Round 103 P7: 同步 src.core.scheduler 的 TaskState/TaskScheduler/VALID_TRANSITIONS
# 到本模块 globals. 解决 test_round96_p2_raw_cjk_gate 等测试期望 enum 身份相等.
# `from src.translator_agent import TranslationPipeline` 后调用 pipeline._process_*
# 使用的 TaskState 与 `from src.core.scheduler import TaskState` 是同一 enum 实例.
# 单文件部署模式 (src.* 不在 path) 走 except 分支, 保留内联定义.
# R104 P2 修复: shim 先于本块执行, 已把 src.core.scheduler 写进 sys.modules (假模块,
# __file__=None). 此处先 pop 假模块再 import, 才能拿到真实的 src.core.scheduler,
# 否则 sync 命中不了 (fake.__file__ is None), 保留 monolith 内联定义, 身份与真实 enum 不一致.
try:
    # R104 P3 修复: 父包 src.core 是 shim 占位 (__path__=[]), 必须先清掉 src.core 才能
    # import src.core.scheduler 找到真实文件. R104 P7 修正: 之前版本用宽口径
    # `for _k in list(sys.modules): if src.* and __file__ is None: pop`, 把 shim
    # 同批次创建的 src.main / src.splitters / src.utils 等假父包也清掉, 导致
    # `python -m src.translator_agent split` 报 "无法识别输入格式" (src.main
    # 没了, shim 没补回去). 改为只 pop 自己依赖的父包.
    if "src.core" in sys.modules and getattr(sys.modules.get("src.core"), "__file__", None) is None:
        sys.modules.pop("src.core", None)
    if "src.core.scheduler" in sys.modules and getattr(sys.modules.get("src.core.scheduler"), "__file__", None) is None:
        sys.modules.pop("src.core.scheduler", None)
    import src.core.scheduler as _real_sched
    if getattr(_real_sched, "__file__", None):
        TaskState = _real_sched.TaskState
        TaskScheduler = _real_sched.TaskScheduler
        VALID_TRANSITIONS = _real_sched.VALID_TRANSITIONS
except (ImportError, AttributeError):
    pass

# Round 103 P7: 同步 _cjk_ratio_cache, 让 monolith reset_cjk_ratio_cache()
# 清的是散装 src.utils.text_match._cjk_ratio_cache 同一个 dict. 否则
# test_round102_compute_cjk_ratio_cache::test_pipeline_reset_clears_module_cache
# 等测试 import _cjk_ratio_cache (来自 src.utils.text_match) 后断言清空,
# 但 monolith 清的是自己 globals 里的副本, 真实 dict 不变.
# R104 P2 修复: shim 先于本块执行, 已把 src.utils.text_match 写进 sys.modules (假模块,
# __file__=None). 此处先 pop 假模块再 import, 才能拿到真实的 src.utils.text_match,
# 让 monolith._cjk_ratio_cache 与真实 _cjk_ratio_cache 是同一 dict (is 关系).
try:
    # R104 P3 + R104 P7: 只 pop 自己依赖的父包 src.utils (而非所有 src.*),
    # 避免误删 shim 同批次创建的 src.main / src.splitters 等假父包.
    if "src.utils" in sys.modules and getattr(sys.modules.get("src.utils"), "__file__", None) is None:
        sys.modules.pop("src.utils", None)
    if "src.utils.text_match" in sys.modules and getattr(sys.modules.get("src.utils.text_match"), "__file__", None) is None:
        sys.modules.pop("src.utils.text_match", None)
    import src.utils.text_match as _real_text_match
    if getattr(_real_text_match, "__file__", None):
        _cjk_ratio_cache = _real_text_match._cjk_ratio_cache
        _CJK_RATIO_CACHE_MAX = _real_text_match._CJK_RATIO_CACHE_MAX
except (ImportError, AttributeError):
    pass



if __name__ == "__main__":
    main()