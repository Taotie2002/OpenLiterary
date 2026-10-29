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
    # 然后是 class 和 helper 函数（Round 129: 增加配额降级包装器与拆分后的构建函数）
    classes = [extract_class(src, c) for c in [
        "QuotaExhaustedError", "LLMAdapter", "MockLLMAdapter",
        "OpenAICompatibleAdapter", "FallbackRoleAdapter",
    ]]
    helpers = [extract_function(src, f) for f in [
        "_map_effort_for_fallback", "_fallback_levels_for",
        "_build_single_client", "_build_role_client",
        "get_llm_client", "get_llm_client_for_task", "get_role_model_name", "get_task_extra_body",
    ]]
    
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
    out += '''
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
'''

    return out



def extract_symbol_any(src: str, name: str) -> str:
    """提取顶层常量/函数/类（供 Section 13 镜像区使用）。"""
    seg = extract_constant(src, name)
    if seg:
        return seg
    seg = extract_function(src, name)
    if seg:
        return seg
    return extract_class(src, name)


def build_section_13() -> str:
    """镜像补全区：单体运行需要、但不属于主 Section 提取范围的散装符号（逐字镜像）。

    Round 126: 这些符号此前靠手工镜像进单体（book_lint 另有 sync_book_lint_mirror.py），
    导致 `build_integrated.py --check` 长期失真。现纳入生成器，重建即为完整单体；
    book_lint 的 sync 脚本仍可作校验（--check 幂等）。
    """
    spec = [
        ("utils/llm_adapter.py", [
            "_INLINE_THINK_RE", "_QUOTA_HINT_RE", "build_effort_params",
        ]),
        ("utils/text_match.py", [
            "entity_source_name_in_text",
            "entity_source_name_spans", "_entity_boundary_pattern",
        ]),
        ("core/scheduler.py", [
            "_source_term_pattern",
        ]),
        ("core/decision_engine.py", [
            "SEED_ENTITY_VARIANTS", "_CONF_ORDER", "load_entity_variant_seeds", "unify_entity_registry",
        ]),
        ("agents/critic_agent.py", [
            "_classify_backend_error",
        ]),
        ("judges/primitives.py", [
            "apply_entity_presence_override", "entity_present_in_text",
        ]),
        ("judges/verdict.py", [
            "DEFAULT_SOURCE_PROFILES", "_MAX_WIDENED_ZONE_HIGH", "_SOURCE_MARGIN_CONFIDENCE_KEYS", "_SOURCE_MARGIN_THRESHOLD_KEYS",
            "_apply_source_profile", "normalize_confidence_source",
        ]),
        ("consistency.py", [
            "_NOTE_MARKER_RE", "_is_common_word_variant",
        ]),
        ("pipeline.py", [
            "consolidate_dirty_chapters", "run_final_gate",
        ]),
        # Round 130: 单实例守卫（main() 引用，须随 Section 10 一并进入单体）
        ("main.py", [
            "_INSTANCE_LOCK_FD", "_MUTATING_COMMANDS", "acquire_instance_lock",
        ]),
        ("splitters.py", [
            "_AUDIT_CHAPTER_FLOOR", "_AUDIT_COVERAGE_THRESHOLD", "_AUDIT_MAX_MISSING_SPANS", "_AUDIT_TOC_SCAN_LINES",
            "_AUDIT_TOKEN_RE", "_FALLBACK_BLOCK_TAGS", "_PRIMARY_BLOCK_TAGS", "_VERSE_CLASS_RE",
            "_CHAPTER_TITLE_PATTERNS", "_apply_input_repairs", "_chapter_h1", "_input_prep_cfg", "_prep_key",
            "_VERSE_HEURISTIC_MAX_LINE", "_VERSE_HEURISTIC_MIN_BR", "_VERSE_HEURISTIC_SHORT_RATIO", "_VERSE_TAG_ATTR",
            "_audit_tokens_with_pos", "_audit_whitelist_spans", "_block_element_text", "_build_verse_manifest",
            "_check_title_present", "_el_verse_tag", "_epub_clean_html_text_ex", "_epub_extract_title",
            "_epub_oracle_text", "_extract_block_text", "_mark_verse_elements", "run_conversion_audit",
        ]),
        ("verse.py", [
            "CHUNK_TAGS_VERSION", "VERSE_MANIFEST_VERSION", "_DEFAULT_VERSE_CONFIG", "_block_lines_in_range",
            "_longest_short_line_run", "_max_line_repeat", "_normalized_index", "blocks_for_span",
            "build_chapter_tags", "build_verse_note", "fragments_for_chunk", "fragments_from_manifest",
            "load_chunk_tags", "load_verse_manifest", "locate_chunk_spans", "normalize_whitespace",
            "save_chunk_tags", "save_verse_manifest", "verse_config", "verse_lint",
        ]),
        ("book_lint.py", [
            "_BOLD_LEAD_RE", "_BRACKET_DEF_RE", "_CHAPTER_HEAD_RE", "_DANGLING_RE",
            "_FOOTNOTE_HEADER_RE", "_FRONTMATTER_LINE_RE", "_FRONTMATTER_RESIDUE_RES", "_MACHINE_FN_ID_RE",
            "_MD_DEF_RE", "_MD_REF_RE", "_META_TOKENS",
            "_NOTE_DUP_RATIO", "_NOTE_DUP_RATIO_HARD", "_NOTE_LABEL_RE", "_NOTE_TITLE_END_RE",
            "_NOTE_TITLE_RE", "_NUM_DEF_RE", "_NUM_DEF_START_RE", "_PAREN_DEF_RE",
            "_QUOTED_TERM_RE", "_SEP_LINE_RE", "_SUPER", "_SUP_DEF_RE",
            "_SUP_REF_RE", "_TR_BLOCK_RE", "_audit_text", "_book_style_config",
            "_chapter_dup_notes", "_chapter_note_dupes", "_def_key", "_find_finals",
            "_group_title_units", "_note_anchor_stop", "_note_candidates", "_note_is_dup", "_note_key",
            "_note_norm", "_note_terms", "_scan_lines", "_split_h1_sections",
            "chapter_heading", "finalize_outputs", "lint_book", "lint_texts",
            "normalize_chapter_notes", "normalize_final_style", "normalize_notes_document", "notes_normalize_outputs",
            "postprocess_chunk_text", "sup_run_to_int",
        ]),
    ]
    out = section_comment("13", "镜像补全区（book_lint / verse / 拆分器与判据辅助）")
    missing = []
    for rel, names in spec:
        src = load_module(rel)
        for name in names:
            seg = extract_symbol_any(src, name)
            if not seg:
                missing.append(f"{rel}:{name}")
                continue
            out += seg.rstrip("\n") + "\n\n\n"
    if missing:
        raise SystemExit(f"❌ Section 13 镜像缺源: {missing}")
    # typesafe 异常分类用导入块（src/agents/critic_agent.py 逐字镜像；
    # 单文件部署无 src 包时走 except 分支，靠异常名字字符串兜底分类）
    out += r'''
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
''' + "\n\n\n"
    return out


def build_section_12() -> str:
    """统一入口点（detect_input_format / split_input_to_chapters）+ 真实模块同步块。

    Round 126: 与单体逐字一致（含 R104 P5 的 shim namespace 适配与 R103 P7 的
    TaskState/TaskScheduler/_cjk_ratio_cache 真实实现同步块 —— 后者必须在 shim 之后执行，
    否则 repo 模式下 monolith.TaskScheduler 身份与 src.core.scheduler 不一致）。
    """
    return r'''
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
'''


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
    parts.append(build_section_13())
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
