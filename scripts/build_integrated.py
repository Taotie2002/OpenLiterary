#!/usr/bin/env python3
"""
SagaTrans 整合脚本自动构建器

功能：
  从散装模块（src/）按 Section 顺序拼装，生成 src/translator_agent.py
  （即单文件评审快照）。

用法：
  python scripts/build_integrated.py            # 生成到 src/translator_agent.py
  python scripts/build_integrated.py --check    # 只检查是否一致，不写入
  python scripts/build_integrated.py --output   # 输出到 stdout 而非文件

设计目标：
  - 散装 = 事实源，所有业务逻辑先改散装
  - 整合 = 评审快照，每次优化后跑此脚本重新生成
  - 保证两者字节级一致（去除 docstring/Section header 后）

拼装顺序（与原整合脚本 Section 一一对应）：
  文件头              imports + _config 适配 + robust_json_loads
  Section 1            枚举（TaskState, DecisionLevel）→ 分散到 core/scheduler.py + core/decision_engine.py
  Section 2            LLM 适配层（utils/llm_adapter.py）
  Section 3            Scheduler（core/scheduler.py）
  Section 4            Decision Engine（core/decision_engine.py）
  Section 5            SmartChunker（utils/chunker.py）
  Section 6            Agents（4 个 agents/*.py）
  Section 7            Pipeline（pipeline.py）
  Section 8            测试工具（test_golden_set.py + test_memory_pressure.py）
  Section 9            辅助入口（main.py: init_project, debug_db）
  Section 10           统一入口（main.py: print_banner, main）
  Section 11           输入格式适配器（splitters.py）
  Section 11.5         一致性校正（consistency.py）
  Section 12           统一入口点（main.py: detect_input_format, split_input_to_chapters）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "src"
OUTPUT = SRC / "translator_agent.py"


# ========================================
# 头部：imports + _config 适配 + robust_json_loads
# ========================================
HEADER = '''#!/usr/bin/env python3
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

# MLX (可选，仅 macOS ARM64 需要)
try:
    from mlx_lm import load as _mlx_load
    from mlx_lm import generate as _mlx_generate
    from mlx_lm.sample_utils import make_sampler as _make_sampler
    import mlx.core as mx
except ImportError:
    _mlx_load = _mlx_generate = _make_sampler = mx = None

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
        chunker = {"soft_limit": 1000, "hard_limit": 2500}
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
                            "literal_residue": 7.0,
                            "average_score_min": 7.0}
        style_guide = {"avg_sentence_length": "适中、口语化",
                       "lexicon_preference": "平实、克制、英式幽默",
                       "author_priority_ratio": 0.7,
                       "genre": "儿童文学",
                       "work_type": "长篇童话"}

        def resolve_task_model(self, task_name):
            routing = self.task_routing.get(task_name, {})
            model_key = routing.get("model", "primary")
            return model_key, routing.get("params_override", {})

        def _get_model_config(self, model_key):
            if self.llm_backend == "mlx":
                return self.mlx_models.get(model_key, {})
            if self.llm_backend == "openai_api":
                nested = self.openai_api.get("models", {})
                if nested and model_key in nested:
                    return nested[model_key]
            return self.openai_models.get(model_key, {})

        def get_task_role(self, task_name):
            return self.task_routing.get(task_name, {}).get("model", "primary")

        def has_role_models(self):
            return False

        def role_models(self):
            return {}

        def get_role_config(self, role):
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
    if os.environ.get("SAGATRANS__STRICT") == "1":
        print("[ERROR] SAGATRANS__STRICT=1 模式下拒绝降级，退出。", file=sys.stderr)
        sys.exit(2)


# ────────────────────────────────────────────────────────────────────
'''


# ========================================
# Section 分隔注释
# ========================================
def section_comment(num: str, title: str) -> str:
    return f'''

# Section {num} — {title}
# ────────────────────────────────────────────────────────────────────

'''


# ========================================
# 模块提取（去除散装的 docstring + 散装模块自身的 imports）
# ========================================
def strip_module_extras(src: str) -> str:
    """Remove top decorations from modular source.

    Strips: shebang, module docstring, full imports block (incl. multiline),
            Section markers, dividers, source-origin comments.
    Keeps: all class and function definitions and their body code.
    """
    lines = src.splitlines()
    out_lines = []
    in_docstring = False
    docstring_quote = None
    in_multiline_import = False
    past_header = False
    Q = chr(34)

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#!"):
            continue

        if not past_header:
            triple_q = Q * 3
            if not in_docstring and not in_multiline_import:
                if stripped.startswith(triple_q) or stripped.startswith(chr(39)*3):
                    in_docstring = True
                    docstring_quote = stripped[:3]
                    if stripped.endswith(docstring_quote) and len(stripped) > 3:
                        in_docstring = False
                        past_header = True
                    continue
                if in_docstring:
                    if docstring_quote in stripped:
                        in_docstring = False
                        past_header = True
                    continue
                if (stripped.startswith("from ") or stripped.startswith("import ")) and stripped.endswith("("):
                    in_multiline_import = True
                    continue
                if stripped.startswith("from ") or stripped.startswith("import "):
                    continue
                if stripped == "":
                    continue
                past_header = True
                in_multiline_import = False
            elif in_docstring:
                if docstring_quote in stripped:
                    in_docstring = False
                    past_header = True
                continue
            elif in_multiline_import:
                if stripped == ")" or stripped == "),":
                    in_multiline_import = False
                continue
            else:
                past_header = True

        if line.startswith("# Section "):
            continue
        if line.startswith("# " + chr(0x2500)):
            continue
        if line.startswith("# 源"):
            continue

        out_lines.append(line)

    return chr(10).join(out_lines)


def load_module(rel_path: str) -> str:
    """加载散装模块，剥离 module docstring"""
    p = SRC / rel_path
    return strip_module_extras(p.read_text(encoding="utf-8"))


def extract_class(src: str, class_name: str) -> str:
    """从模块源码中提取指定 class 的完整定义（AST 精确定位）

    包含前导装饰器（@dataclass / @runtime_checkable 等）和缩进（如果是嵌套类）
    """
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    lines = src.splitlines()
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            # 装饰器是 class 上方的 @xxx 行（连续）
            start_lineno = n.lineno  # 1-based
            # 找最近的装饰器（可能不连续，但 ast 把它们存进 decorator_list）
            if n.decorator_list:
                first_dec_lineno = min(d.lineno for d in n.decorator_list)
                # 取装饰器行到 class 末行
                end_lineno = n.end_lineno or start_lineno
                return "\n".join(lines[first_dec_lineno - 1:end_lineno]) + "\n"
            # 无装饰器
            end_lineno = n.end_lineno or start_lineno
            return "\n".join(lines[start_lineno - 1:end_lineno]) + "\n"
    return ""


def extract_function(src: str, func_name: str) -> str:
    """从模块源码中提取指定顶层函数的完整定义（AST 精确定位）"""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(src, node) or ""
    return ""


def extract_constant(src: str, name: str) -> str:
    """从模块源码中提取指定顶层变量赋值（含类型注解 / 多行 dict / list）"""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.get_source_segment(src, node) or ""
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if getattr(tgt, "id", None) == name:
                    return ast.get_source_segment(src, node) or ""
    return ""


def build_section_1() -> str:
    """枚举：TaskState（来自 scheduler.py）+ DecisionLevel（来自 decision_engine.py）"""
    sched_src = load_module("core/scheduler.py")
    de_src = load_module("core/decision_engine.py")
    task_state = extract_class(sched_src, "TaskState")
    decision_level = extract_class(de_src, "DecisionLevel")
    return section_comment("1", "枚举与类型定义") + task_state + "\n\n\n" + decision_level


def build_section_2() -> str:
    """LLM 适配层（utils/llm_adapter.py）+ robust_json_loads + 全局缓存"""
    src = load_module("utils/llm_adapter.py")
    # robust_json_loads 必须在最前（头部有定义）
    robust = extract_function(src, "robust_json_loads")
    # 全局客户端缓存（被 get_llm_client 引用）
    cache = extract_constant(src, "_client_instances")
    # 然后是 4 个 class 和 helper 函数
    classes = [extract_class(src, c) for c in ["LLMAdapter", "MockLLMAdapter", "OpenAICompatibleAdapter", "MLXNativeAdapter"]]
    helpers = [extract_function(src, f) for f in ["_build_role_client", "get_llm_client", "get_llm_client_for_task", "get_role_model_name", "get_role_extra_body"]]
    
    body = robust + "\n\n\n" + cache + "\n\n\n" + "\n\n\n".join(classes) + "\n\n\n" + "\n\n\n".join(helpers)
    return section_comment("2", "LLM 适配层") + body


def build_section_3() -> str:
    """TaskScheduler（scheduler.py 中除 TaskState 外的部分）"""
    src = load_module("core/scheduler.py")
    scheduler = extract_class(src, "TaskScheduler")
    # 加全局 VALID_TRANSITIONS（紧跟 TaskState）
    # 这里直接从源里抓取
    lines = src.splitlines()
    out = []
    capture = False
    for line in lines:
        if line.strip().startswith("VALID_TRANSITIONS"):
            capture = True
        if capture:
            out.append(line)
            if line.strip() == "}" or (capture and line.strip().startswith("class DecisionLevel")):
                break
    valid_trans = "\n".join(out)
    return section_comment("3", "Scheduler 任务调度器") + valid_trans + "\n\n\n" + scheduler


def build_section_4() -> str:
    """DecisionEngine（decision_engine.py 除 DecisionLevel 外）"""
    src = load_module("core/decision_engine.py")
    de = extract_class(src, "DecisionEngine")
    # Round 71: 跨人物冲突 sibling 查找 + 冲突诊断
    helpers = [extract_function(src, f) for f in [
        "query_high_confidence_entities",
        "query_reviewable_entities",
        "get_sibling_canonicals_by_source",
        "detect_cross_person_conflicts",
    ]]
    seed_data = extract_function(src, "seed_entity_variants")
    # SEED_ENTITY_VARIANTS 常量
    seed_const_match = re.search(r"^SEED_ENTITY_VARIANTS = \{.*?^\}", src, flags=re.MULTILINE | re.DOTALL)
    seed_const = seed_const_match.group(0) if seed_const_match else ""
    
    body = seed_const + "\n\n\n" + de + "\n\n\n" + "\n\n\n".join(helpers) + "\n\n\n" + seed_data
    return section_comment("4", "Decision Engine 决策引擎") + body


def build_section_5() -> str:
    """SmartChunker + 引号常量（QUOTE_CHARS_*）"""
    src = load_module("utils/chunker.py")
    chunker = extract_class(src, "SmartChunker")
    # 提取模块级常量供 SmartChunker.split_markdown 引用（line 91-93）
    # extract_constant 返回值不带尾部换行，需手动 join
    quote_consts = "\n".join(
        extract_constant(src, name).rstrip("\n")
        for name in ("QUOTE_CHARS_OPEN", "QUOTE_CHARS_CLOSE")
    ) + "\n\n"
    return section_comment("5", "SmartChunker 文本切分器") + quote_consts + chunker


def build_section_5b() -> str:
    """Round 70+71: utils/text_match.py 中的 reject category helpers + case_insensitive_find.

    Round 70 修复: 整合脚本未打包 utils/text_match.py，导致 LiteraryRewriterAgent.
        _extract_reject_categories 和 TranslationPipeline._extract_reject_categories 触发 NameError.
        导出 _CATEGORY_SYNONYMS / _normalize_category / extract_reject_categories.

    Round 71 修复: 同样问题 — TranslationPipeline._case_insensitive_find 引用
        _text_match_case_insensitive_find (case_insensitive_find 的别名), 触发 NameError.
        导出 case_insensitive_find 并通过别名赋值兼容 pipeline 中的下划线前缀名.
    """
    src = load_module("utils/text_match.py")
    parts = []
    import ast as _ast
    tree = _ast.parse(src)
    # 顶层常量 + 函数定义
    for name in ("_CATEGORY_SYNONYMS", "_normalize_category", "extract_reject_categories", "case_insensitive_find"):
        snippet = extract_constant(src, name).rstrip("\n")
        if not snippet:
            for node in tree.body:
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == name:
                    snippet = (_ast.get_source_segment(src, node) or "").rstrip("\n")
                    break
        if snippet:
            parts.append(snippet)
    # Round 71: 把 case_insensitive_find 别名为 _text_match_case_insensitive_find (pipeline 引用此名)
    parts.append("# Round 71: alias 与 pipeline.py:30 import 保持一致")
    parts.append("_text_match_case_insensitive_find = case_insensitive_find")
    return section_comment("5b", "Reject Category 归一化与提取 (Round 70) + 文本匹配 helpers (Round 71)") + "\n\n\n".join(parts) + "\n\n\n"


def build_section_6() -> str:
    """4 个 Agent"""
    out = section_comment("6", "Agents")
    for cls, mod in [("ReferenceAgent", "agents/reference_agent.py"),
                       ("LiteraryRewriterAgent", "agents/rewriter_agent.py"),
                       ("CriticAgent", "agents/critic_agent.py"),
                       ("JudgeAgent", "agents/judge_agent.py")]:
        src = load_module(mod)
        out += extract_class(src, cls) + "\n\n\n"
    return out


def build_section_7() -> str:
    """TranslationPipeline"""
    src = load_module("pipeline.py")
    pipeline = extract_class(src, "TranslationPipeline")
    return section_comment("7", "Pipeline 主流程") + pipeline


def build_section_8() -> str:
    """测试工具"""
    gs_src = load_module("test_golden_set.py")
    mp_src = load_module("test_memory_pressure.py")
    
    gs_const = extract_constant(gs_src, "GOLDEN_AB_MATRIX")
    gs_class = extract_class(gs_src, "GoldenSetEvaluator")
    gs_fns = [extract_function(gs_src, f) for f in [
        "_levenshtein_distance", "run_golden_test",
        "_extract_confidence_source", "summarize_golden_results",
        "recommend_golden_combo", "run_golden_ab",
    ]]
    # run_golden_gate 实际定义在 test_memory_pressure.py（散装历史：原 §8.2 拆出去时迁过去的）
    mp_fns = [extract_function(mp_src, f) for f in ["get_system_memory", "get_process_memory", "simulate_memory_load", "run_golden_gate", "test_memory_pressure"]]
    
    prefix = (gs_const + "\n\n\n") if gs_const else ""
    body = prefix + gs_class + "\n\n\n" + "\n\n\n".join(gs_fns) + "\n\n\n" + "\n\n\n".join(mp_fns)
    return section_comment("8", "测试工具") + body


def build_section_9() -> str:
    """init_project, debug_db（main.py 中）"""
    src = load_module("main.py")
    body = extract_function(src, "init_project") + "\n\n\n" + extract_function(src, "debug_db")
    return section_comment("9", "辅助入口") + body


def build_section_10() -> str:
    """main（main.py 中；print_banner 已在 HEADER 定义）"""
    src = load_module("main.py")
    body = extract_function(src, "main")
    return section_comment("10", "统一入口") + body


def build_section_11() -> str:
    """splitters.py：3 个 Splitter class + 全部 module-level helper + 常量"""
    src = load_module("splitters.py")
    out = section_comment("11", "输入格式适配器 (EpubSplitter / TextSplitter / MdSplitter)")
    # 先提取模块级常量（_SCORE_KEY_NORMALIZATION 给 critic 评分归一化用；_GUTENBERG_* 给 EpubSplitter 用）
    for name in ("_SCORE_KEY_NORMALIZATION", "_GUTENBERG_HEADER_PATTERNS",
                 "_GUTENBERG_TOC_TITLE_PATTERN", "_GUTENBERG_TOC_ITEM_PATTERN"):
        const_text = extract_constant(src, name).rstrip("\n")
        if const_text:
            out += const_text + "\n\n\n"
    # 再 helper（避免 method 引用未定义名字）
    for fn in ['_splitter_log_info', '_splitter_log_ok', '_splitter_log_warn', '_splitter_log_err', '_normalize_chapter_title', '_normalize_scores', '_epub_clean_html_text', '_strip_epub_front_matter', '_epub_split_large_doc_by_toc', '_epub_extract_chapters', '_text_detect_chapter_title']:
        out += extract_function(src, fn) + "\n\n\n"
    for cls in ["EpubSplitter", "TextSplitter", "MdSplitter"]:
        out += extract_class(src, cls) + "\n\n\n"
    return out


def build_section_11_5() -> str:
    """consistency.py：run_consistency_check + 依赖 helper（docstring 与 imports 已剥离）"""
    src = load_module("consistency.py")
    fns = ["_load_glossary", "_load_decisions", "_scan_finals", "_find_slash_aliases",
           "_in_quote_or_note", "_detect_issues", "_generate_report", "_apply_issues",
           "run_consistency_check", "run_consistency_check_for_chapter"]
    out = section_comment("11.5", "译后命名一致性校正")
    for fn in fns:
        out += extract_function(src, fn) + "\n\n\n"
    return out


def build_section_11_6() -> str:
    """typesafe-ai judges 模块（src/judges/）

    typesafe-spec 兼容的判断原语层：
      - typesafe_spec      Noul/Choice/Score/CompositeVerdict 数据类
      - primitives         6 个原子判断（voice_drift/entity_consistency/literal_residue/...）
      - typesafe_backend   Jev/OpenAICompat/Mock 三级 fallback
      - verdict            code-side 合成器
      - judge              对外入口 evaluate_chunk + to_legacy_scores

    重构要点：原 critic/judge 双 LLM 调用 → 多 atomic primitives 并行评估 +
              code-side verdict 合成，零 LLM 成本做最终判断。
    """
    out = section_comment("11.6", "typesafe-ai judges（多原子原语 + verdict 合成）")
    # typesafe_spec 用了 dataclass + field + asdict，整合脚本头部已有 dataclass 但 field/asdict 没有
    out += "from dataclasses import dataclass, field, asdict\n\n\n"

    # 1) typesafe_spec — 所有 dataclass（NoulAnswer/ChoiceAnswer/ScoreAnswer/CompositeVerdict）
    src_spec = load_module("judges/typesafe_spec.py")
    for cls in ["NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "CompositeVerdict"]:
        out += extract_class(src_spec, cls) + "\n\n\n"

    # 1.5) 类型别名（必须在 dataclass 之后，因为 Answer = Union[NoulAnswer, ...]）
    # typesafe_backend.py 用 typing.Union，原 spec 文件里有 from typing import Union
    # 但整合后 dataclass 被单独提取，import 没了，故显式注入
    out += "from typing import Union\n"
    out += "Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]\n\n\n"

    # 2) primitives — 6 个 dict + ALL_PRIMITIVES + build_question_set()
    # Round 87 修复：primitives.py 第 35 行有 `import copy`, 用于 build_question_set 的
    # deepcopy. AST 提取只挑 target_names + build_question_set 函数体, 不会自动复制顶部
    # imports, 故 build_question_set 跑时会触发 NameError. 这里显式注入.
    out += "import copy\n\n\n"
    src_prim = load_module("judges/primitives.py")
    # dict 字面量不需要 extract，直接通过 AST 取其赋值文本
    import ast
    target_names = ("VOICE_DRIFT", "REGISTER_MATCH", "ENTITY_CONSISTENCY",
                    "LITERAL_RESIDUE", "SEMANTIC_LOSS", "STYLE_COMPLIANCE_SCORE",
                    "ALL_PRIMITIVES")
    try:
        tree = ast.parse(src_prim)
        for node in tree.body:
            # AnnAssign = 带类型注解的赋值（如 `X: Dict = {...}`）
            if isinstance(node, ast.AnnAssign):
                name = getattr(node.target, "id", None)
                if name in target_names:
                    out += ast.get_source_segment(src_prim, node) + "\n\n\n"
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    name = getattr(tgt, "id", None)
                    if name in target_names:
                        out += ast.get_source_segment(src_prim, node) + "\n\n\n"
            elif isinstance(node, ast.FunctionDef) and node.name == "build_question_set":
                out += ast.get_source_segment(src_prim, node) + "\n\n\n"
    except SyntaxError:
        pass

    # 3) typesafe_backend — 3 个 backend + get_backend()
    src_be = load_module("judges/typesafe_backend.py")
    # 注入 _DEFAULT_BACKEND 模块级状态（get_backend 用到）
    out += "from typing import Optional\n"
    out += "_DEFAULT_BACKEND: Optional[\"SystemOneBackend\"] = None\n\n\n"
    for cls in ["SystemOneBackend", "JevBackend", "OpenAICompatBackend", "MockBackend", "FallbackBackend"]:
        out += extract_class(src_be, cls) + "\n\n\n"
    # Round 87 修复：_confidence_from_probs 是 Mock/OpenAICompat backend 内部使用的 helper,
    # OpenAICompatBackend / MockBackend evaluate 方法都调用它. AST 提取只挑 class 和
    # get_backend, 不会自动复制模块级 helper, 故整合脚本运行时触发 NameError.
    out += extract_function(src_be, "_confidence_from_probs") + "\n\n\n"
    # get_backend 内部引用 _load_typesafe_config，须一并提取（否则 NameError）
    out += extract_function(src_be, "_load_typesafe_config") + "\n\n\n"
    out += extract_function(src_be, "get_backend") + "\n\n\n"

    # 4) verdict — DEFAULT_THRESHOLDS + synthesize_verdict
    src_v = load_module("judges/verdict.py")
    import ast as _ast
    try:
        tree = _ast.parse(src_v)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if getattr(tgt, "id", None) == "DEFAULT_THRESHOLDS":
                        out += _ast.get_source_segment(src_v, node) + "\n\n\n"
            elif isinstance(node, ast.FunctionDef) and node.name == "synthesize_verdict":
                out += _ast.get_source_segment(src_v, node) + "\n\n\n"
    except SyntaxError:
        pass

    # 5) judge — build_state / evaluate_chunk / to_legacy_scores
    src_j = load_module("judges/judge.py")
    for fn in ["build_state", "evaluate_chunk", "to_legacy_scores"]:
        out += extract_function(src_j, fn) + "\n\n\n"

    # Round 72 修复: synthesize_verdict 调用 aggregate_per_entity_nouls / PER_ENTITY_KEY_PREFIX,
    # 这两个定义在 primitives.py 而非 verdict.py. 必须单独打包到整合脚本顶层.
    # Round 87 修复: MockBackend (typesafe_backend.py:449) 和 critic_agent / judge 都引用
    # build_per_entity_questions, 旧版 build_integrated.py 只挑 aggregate_* + 常量, 漏掉了
    # 这个函数及其私有依赖 _sanitize_key, 整合脚本运行时会 ImportError / NameError.
    out += "# Round 72/87: synthesize_verdict / MockBackend 调用的 primitives.py 依赖\n\n"
    src_prim = load_module("judges/primitives.py")
    import ast as _ast2
    tree_p = _ast2.parse(src_prim)
    for node in tree_p.body:
        # 提取 PER_ENTITY_KEY_PREFIX 和 ALL_PRIMITIVES 常量
        if isinstance(node, _ast2.Assign):
            for tgt in node.targets:
                tname = getattr(tgt, "id", None)
                if tname in ("PER_ENTITY_KEY_PREFIX", "ALL_PRIMITIVES"):
                    out += _ast2.get_source_segment(src_prim, node) + "\n\n"
        elif isinstance(node, _ast2.FunctionDef) and node.name in (
            "aggregate_per_entity_nouls",
            "build_per_entity_questions",
            "_sanitize_key",
        ):
            out += _ast2.get_source_segment(src_prim, node) + "\n\n\n"

    # 6) src.judges.* 兼容 shim（让整合脚本内部 `from src.judges.X import Y` 也能用）
    # 真实 src.judges 模块并不存在（散装时被拍扁），故用 sys.modules 注入虚拟模块
    out += """
# ───── src.judges.* 兼容 shim（单文件部署模式） ─────
import sys as _sys_j
import types as _types_j
# 先注册父包「src」为 package（需要 __path__ 才能 import 子模块）
if "src" not in _sys_j.modules:
    _src_pkg = _types_j.ModuleType("src")
    _src_pkg.__path__ = []
    _sys_j.modules["src"] = _src_pkg
if "src.judges" not in _sys_j.modules:
    _j_pkg = _types_j.ModuleType("src.judges")
    _j_pkg.__path__ = []
    _sys_j.modules["src.judges"] = _j_pkg
# 其他 src.* 子包（如 critic 内部 from src.utils.llm_adapter import）
for _sub in ("src.utils", "src.config", "src.agents", "src.core", "src.consistency", "src.splitters", "src.pipeline", "src.main"):
    if _sub not in _sys_j.modules:
        _m = _types_j.ModuleType(_sub)
        _m.__path__ = []
        _sys_j.modules[_sub] = _m
for _mod_name, _attrs in {
    "src.judges.typesafe_spec": ["NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "CompositeVerdict", "Answer"],
    # Round 87 修复: 旧 shim 漏掉了 primitives.py 的 PER_ENTITY_KEY_PREFIX 常量 /
    # build_per_entity_questions / _sanitize_key 函数, 整合脚本中 MockBackend 等位置
    # 的 `from src.judges.primitives import PER_ENTITY_KEY_PREFIX` 触发 ImportError.
    "src.judges.primitives": ["VOICE_DRIFT", "REGISTER_MATCH", "ENTITY_CONSISTENCY", "LITERAL_RESIDUE", "SEMANTIC_LOSS", "STYLE_COMPLIANCE_SCORE", "ALL_PRIMITIVES", "build_question_set", "PER_ENTITY_KEY_PREFIX", "build_per_entity_questions", "_sanitize_key", "aggregate_per_entity_nouls"],
    "src.judges.typesafe_backend": ["SystemOneBackend", "JevBackend", "OpenAICompatBackend", "MockBackend", "FallbackBackend", "get_backend", "_DEFAULT_BACKEND"],
    "src.judges.verdict": ["synthesize_verdict", "DEFAULT_THRESHOLDS"],
    "src.judges.judge": ["build_state", "evaluate_chunk", "to_legacy_scores"],
    "src.judges": ["NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "CompositeVerdict", "Answer",
                   "VOICE_DRIFT", "REGISTER_MATCH", "ENTITY_CONSISTENCY", "LITERAL_RESIDUE", "SEMANTIC_LOSS", "STYLE_COMPLIANCE_SCORE", "ALL_PRIMITIVES", "build_question_set",
                   "PER_ENTITY_KEY_PREFIX", "build_per_entity_questions", "_sanitize_key", "aggregate_per_entity_nouls",
                   "SystemOneBackend", "JevBackend", "OpenAICompatBackend", "MockBackend", "FallbackBackend", "get_backend",
                   "synthesize_verdict", "DEFAULT_THRESHOLDS",
                   "build_state", "evaluate_chunk", "to_legacy_scores"],
}.items():
    _m = _types_j.ModuleType(_mod_name)
    for _a in _attrs:
        if _a in globals() and not _a.startswith("_sys_") and not _a.startswith("_types_"):
            setattr(_m, _a, globals()[_a])
    _sys_j.modules[_mod_name] = _m

# ───── 把散装模块的内容塞进对应的 src.* 虚拟子包 ─────
# 这样 `from src.utils.llm_adapter import ...` 等也能工作
_SRC_SUBMODULES = {
    "src.utils.llm_adapter": [
        "LLMAdapter", "MockLLMAdapter", "OpenAICompatibleAdapter", "MLXNativeAdapter",
        "get_llm_client", "get_llm_client_for_task", "get_role_model_name", "get_role_extra_body",
        "robust_json_loads", "_build_role_client", "_client_instances",
    ],
    "src.utils.chunker": ["SmartChunker"],
    "src.config": ["Config", "get_config", "config"],
    "src.core.scheduler": ["TaskState", "TaskScheduler", "VALID_TRANSITIONS"],
    "src.core.decision_engine": [
        "DecisionLevel", "DecisionEngine", "query_high_confidence_entities",
        "query_reviewable_entities", "get_sibling_canonicals_by_source",
        "detect_cross_person_conflicts", "seed_entity_variants", "SEED_ENTITY_VARIANTS",
    ],
    "src.agents.reference_agent": ["ReferenceAgent"],
    "src.agents.rewriter_agent": ["LiteraryRewriterAgent"],
    "src.agents.critic_agent": ["CriticAgent"],
    "src.agents.judge_agent": ["JudgeAgent"],
    "src.pipeline": ["TranslationPipeline"],
    "src.consistency": ["run_consistency_check", "_load_glossary", "_load_decisions",
                        "_scan_finals", "_find_slash_aliases", "_in_quote_or_note",
                        "_detect_issues", "_generate_report", "_apply_issues",
                        "run_consistency_check_for_chapter"],
    "src.splitters": ["EpubSplitter", "TextSplitter", "MdSplitter",
                      "_splitter_log_info", "_splitter_log_ok", "_splitter_log_warn", "_splitter_log_err",
                      "_normalize_chapter_title", "_normalize_scores",
                      "_epub_clean_html_text", "_strip_epub_front_matter",
                      "_epub_split_large_doc_by_toc", "_epub_extract_chapters",
                      "_text_detect_chapter_title"],
    "src.main": ["init_project", "debug_db", "print_banner", "main",
                 "detect_input_format", "split_input_to_chapters"],
    "src.test_golden_set": ["GoldenSetEvaluator", "_levenshtein_distance", "run_golden_test",
                            "GOLDEN_AB_MATRIX", "_extract_confidence_source",
                            "summarize_golden_results", "recommend_golden_combo", "run_golden_ab",
                            "run_golden_gate"],
    "src.test_memory_pressure": ["get_system_memory", "get_process_memory", "simulate_memory_load", "test_memory_pressure"],
}
for _mod, _attrs in _SRC_SUBMODULES.items():
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
"""

    return out


def build_section_12() -> str:
    """detect_input_format, split_input_to_chapters（main.py 中）"""
    src = load_module("main.py")
    body = extract_function(src, "detect_input_format") + "\n\n\n" + extract_function(src, "split_input_to_chapters")
    return section_comment("12", "统一入口点 (split_input_to_chapters)") + body


# ========================================
# 主流程
# ========================================
def build() -> str:
    parts = [HEADER]
    parts.append(build_section_1())
    parts.append(build_section_2())
    parts.append(build_section_3())
    parts.append(build_section_4())
    parts.append(build_section_5())
    parts.append(build_section_5b())
    parts.append(build_section_6())
    parts.append(build_section_7())
    parts.append(build_section_8())
    parts.append(build_section_9())
    parts.append(build_section_10())
    parts.append(build_section_11())
    parts.append(build_section_11_5())
    parts.append(build_section_11_6())
    parts.append(build_section_12())
    parts.append('\n')
    parts.append('if __name__ == "__main__":')
    parts.append('    main()')
    
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="SagaTrans 整合脚本构建器")
    parser.add_argument("--check", action="store_true", help="只检查是否一致，不写入文件")
    parser.add_argument("--output", action="store_true", help="输出到 stdout 而非文件")
    args = parser.parse_args()
    
    generated = build()
    
    if args.output:
        sys.stdout.write(generated)
        return 0
    
    if args.check:
        if OUTPUT.exists():
            existing = OUTPUT.read_text(encoding="utf-8")
            if existing == generated:
                print(f"✅ {OUTPUT.name} 与散装一致（{len(generated.splitlines())} 行）")
                return 0
            else:
                print(f"❌ {OUTPUT.name} 与散装不一致")
                print(f"   当前: {len(existing.splitlines())} 行")
                print(f"   应生成: {len(generated.splitlines())} 行")
                return 1
        else:
            print(f"❌ {OUTPUT} 不存在")
            return 1
    
    OUTPUT.write_text(generated, encoding="utf-8")
    print(f"✅ {OUTPUT.name} 已重新生成（{len(generated.splitlines())} 行 / {len(generated):,} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
