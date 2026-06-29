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
  Section 12 — 统一入口点            (split_input_to_chapters, 暴露 CLI 参数)
"""

import os
import sys
import re
import json
import time
import random
import gc
import sqlite3
import threading
from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable

# 配置加载器（支持单文件模式降级）
try:
    from src.config import get_config as _get_config_external
    _config = _get_config_external()
except ImportError:
    # 单文件部署模式：src.config 不可用，使用内建默认配置
    class _BuiltinConfig:
        """单体聚合版内建默认配置（仅在 src.config 不可用时启用）"""
        llm_backend = "mock"
        task_routing = {
            "reference_extraction": {"model": "reasoning_primary", "params_override": {}},
            "literal_translation": {"model": "literal_translator", "params_override": {}},
            "literary_rewrite": {"model": "reasoning_primary", "params_override": {}},
            "critic_scoring": {"model": "reasoning_primary", "params_override": {}},
            "judge_decision": {"model": "reasoning_heavy", "params_override": {}},
        }
        mlx_models = {
            "literal_translator": {"model_id": "google/gemma-2-9b-it-mlx-4bit", "default_params": {}},
            "reasoning_primary": {"model_id": "qwen/Qwen2.5-7B-Instruct-MLX-4bit", "default_params": {}},
            "reasoning_heavy": {"model_id": "qwen/Qwen2.5-7B-Instruct-MLX-4bit", "default_params": {}},
        }
        openai_models = {
            "literal_translator": {"model_name": "gpt-4o-mini", "default_params": {}},
            "reasoning_primary": {"model_name": "gpt-4o-mini", "default_params": {}},
            "reasoning_heavy": {"model_name": "gpt-4o", "default_params": {}},
        }
        mlx_memory = {"warning_threshold": 0.8, "auto_unload_on_pressure": True}
        openai_api = {"api_base": "http://127.0.0.1:1234/v1", "api_key": "lm-studio",
                      "max_retries": 3, "retry_delay": 2.0, "request_timeout": 300,
                      "max_concurrent_requests": 4}
        pipeline = {"batch_size": 50, "max_retries": 3, "max_concurrent_chunks": 4, "poll_interval": 0.5}
        chunker = {"soft_limit": 1000, "hard_limit": 2500, "respect_scene_breaks": True}
        decision_engine = {"terminology_triggers_backtrack": True,
                           "reference_triggers_backtrack": True,
                           "style_triggers_backtrack": True,
                           "max_affected_chunks_per_decision": 50}
        critic_thresholds = {"fluency": 7.0, "readability": 7.0, "style_compliance": 7.0,
                            "voice_consistency": 7.0, "semantic_preservation": 7.0,
                            "average_score_min": 7.5}
        style_guide = {"avg_sentence_length": "较长且富有韵律",
                       "lexicon_preference": "古典、史诗感、冷硬",
                       "author_priority_ratio": 0.7}
        paths = {"db_dir": "db", "input_dir": "input", "output_dir": "output",
                 "docs_dir": "docs", "golden_test_file": "input/golden/hyperion_5k.md",
                 "decision_db": "decision_db.sqlite", "workflow_db": "workflow.db"}
        logging = {"level": "INFO", "log_llm_calls": False, "perf_log_interval": 10}

        def resolve_task_model(self, task_name: str):
            routing = self.task_routing.get(task_name, {})
            model_key = routing.get("model", "reasoning_primary")
            return model_key, routing.get("params_override", {})

        def _get_model_config(self, model_key: str):
            if self.llm_backend == "mlx":
                return self.mlx_models.get(model_key, {})
            return self.openai_models.get(model_key, {})

    _config = _BuiltinConfig()
    print("[WARN] 未找到 src.config，使用内建默认配置（单文件模式）", file=sys.stderr)

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
    TaskState.FAILED:           [TaskState.EXTRACTING_TERMS, TaskState.PERMANENTLY_FAILED],
    TaskState.PERMANENTLY_FAILED: [],  # 终态：不可转移
    TaskState.EXTRACTING_TERMS: [TaskState.TRANSLATING_RAW, TaskState.FAILED],
    TaskState.TRANSLATING_RAW:  [TaskState.REWRITING_LITERARY, TaskState.FAILED],
    TaskState.REWRITING_LITERARY: [TaskState.AUDITING, TaskState.FAILED],
    TaskState.AUDITING:         [TaskState.JUDGING, TaskState.FAILED],
    TaskState.JUDGING:          [TaskState.COMPLETED, TaskState.REWRITING_LITERARY, TaskState.FAILED],
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
    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        pass

    @abstractmethod
    def unload_model(self):
        """强制释放显存的统一接口"""
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前进程内存使用情况"""
        import psutil as _psutil
        process = _psutil.Process(os.getpid())
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

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
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
        if "【直译底稿】" in prompt:
            after_marker = prompt.split("【直译底稿】")[-1]
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
            "is_flawed": False,
            "critique": "译文流畅自然，风格契合度良好，语义保留完整。",
            "improvement_suggestions": ""
        }, ensure_ascii=False)

    def _mock_judge_decision(self) -> str:
        return json.dumps({
            "decision": "PASS",
            "final_text": "最终定稿文本",
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
    def __init__(self, api_base: str, api_key: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.api_base = api_base
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        import requests as _requests
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = _requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=300
                )
                # 429 是限流（属于 4xx 但应重试），其他 4xx 是客户端错误；5xx 由 raise_for_status 抛出后被外层重试逻辑捕获
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise RuntimeError(f"API 客户端错误 {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
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
    def __init__(self):
        try:
            from mlx_lm import load as _mlx_load
            from mlx_lm import generate as _mlx_generate
            import mlx.core as _mx
            self.mlx_load = _mlx_load
            self.mlx_generate = _mlx_generate
            self.mx = _mx
        except ImportError:
            raise RuntimeError("⚠️ 环境中未安装 mlx-lm。请在虚拟环境中执行 `pip install mlx-lm`")

        self.current_model_name = None
        self.model = None
        self.tokenizer = None

    def _load_model_if_needed(self, model_name: str):
        """按需加载，如果模型切换则强制卸载旧模型"""
        if self.current_model_name != model_name:
            if self.model is not None:
                self.unload_model()

            print(f"🚀 [MLX] 正在将模型加载至 Apple Silicon 统一内存: {model_name}...")
            self.model, self.tokenizer = self.mlx_load(model_name)
            self.current_model_name = model_name

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3, **kwargs) -> str:
        # 先检查内存，必要时在加载前卸载旧模型，避免无效加载
        if self.current_model_name and self.current_model_name != model_name:
            if self.check_memory_pressure():
                self.unload_model()
        self._load_model_if_needed(model_name)

        # 组装聊天模板
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = prompt

        # 性能计时
        start_time = time.time()
        mem_before = self.get_memory_usage()

        # MLX 生成调用
        response = self.mlx_generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
            temp=temperature
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
        """M1 环境下至关重要的显存清理操作"""
        if self.current_model_name:
            print(f"🧹 [MLX] 正在从统一内存中卸载模型: {self.current_model_name}")

        self.model = None
        self.tokenizer = None
        self.current_model_name = None

        # 1. 触发 Python 层垃圾回收
        gc.collect()

        # 2. 深入底层，强制清理 Metal 计算图和 KV Cache
        if hasattr(self.mx.metal, "clear_cache"):
            self.mx.metal.clear_cache()

        print("✅ [MLX] Metal 缓存已清空，显存已释放。")


# 单例工厂函数
_client_instance = None

def get_llm_client() -> LLMAdapter:
    """获取当前配置的 LLM 客户端实例 (单例)"""
    global _client_instance
    if _client_instance is None:
        backend = _config.llm_backend
        if backend == "mock":
            _client_instance = MockLLMAdapter()
        elif backend == "mlx":
            _client_instance = MLXNativeAdapter()
        elif backend == "openai_api":
            api_cfg = _config.openai_api
            _client_instance = OpenAICompatibleAdapter(
                api_cfg.get("api_base", "http://127.0.0.1:1234/v1"),
                api_cfg.get("api_key", "lm-studio"),
                max_retries=api_cfg.get("max_retries", 3),
                retry_delay=api_cfg.get("retry_delay", 2.0)
            )
        else:
            raise ValueError(f"不支持的 llm_backend: {backend}")
    return _client_instance


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
                last_error TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

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
                    # force 模式：覆盖 text_content 并完全重置（含 retries=0 让 PERMANENTLY_FAILED 任务可重新执行）
                    cursor.execute('''
                        INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            text_content = excluded.text_content,
                            state = ?,
                            last_error = NULL,
                            retries = 0
                    ''', (chunk_id, chapter_id, text, TaskState.PENDING.value, TaskState.PENDING.value))
                else:
                    cursor.execute('''
                        INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)
                        VALUES (?, ?, ?, ?)
                    ''', (chunk_id, chapter_id, text, TaskState.PENDING.value))
            except sqlite3.IntegrityError:
                skipped += 1
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

    def get_tasks_by_state(self, state: TaskState, batch_size: int = 10) -> List[Dict]:
        """拉取指定状态的任务批次"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM chunk_tasks
            WHERE state = ?
            ORDER BY chunk_id ASC LIMIT ?
        ''', (state.value, batch_size))
        return [dict(row) for row in cursor.fetchall()]

    def update_task_state(self, chunk_id: str, new_state: TaskState, error_msg: str = None):
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
        if error_msg:
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

    def batch_update_state(self, chunk_ids: List[str], new_state: TaskState, error_msg: str = None):
        """批量更新任务状态 - 减少数据库往返开销"""
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

        valid_placeholders = ','.join('?' for _ in valid_ids)
        if error_msg:
            cursor.execute(f'''
                UPDATE chunk_tasks
                SET state = ?, retries = retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id IN ({valid_placeholders})
            ''', (new_state.value, error_msg) + tuple(valid_ids))
        else:
            cursor.execute(f'''
                UPDATE chunk_tasks
                SET state = ?, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id IN ({valid_placeholders})
            ''', (new_state.value,) + tuple(valid_ids))
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
        """回溯引擎：将已完成的块标记为 DIRTY 重跑"""
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor.execute(f'''
            UPDATE chunk_tasks
            SET state = ?, updated_at = CURRENT_TIMESTAMP
            WHERE chunk_id IN ({placeholders}) AND state = ?
        ''', (TaskState.DIRTY.value, TaskState.COMPLETED.value) + tuple(chunk_ids))
        self.conn.commit()
        print(f"🔄 已触发 {len(chunk_ids)} 个数据块的回溯重构。")

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
        self.conn.commit()

    def add_decision(self, level: DecisionLevel, source: str, translation: str, reason: str, affected_chunks: List[str] = None):
        """插入或更新决策，并记录影响的 chunk"""
        cursor = self.conn.cursor()
        try:
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
            
            # 记录影响的 chunk（先清理旧映射，再写入新映射）
            if affected_chunks and decision_id:
                cursor.execute('DELETE FROM decision_impact WHERE decision_id = ?', (decision_id,))
                cursor.executemany(
                    'INSERT OR IGNORE INTO decision_impact (decision_id, chunk_id) VALUES (?, ?)',
                    [(decision_id, cid) for cid in affected_chunks]
                )

            self.conn.commit()
            print(f"✅ [Decision Engine] 记录 {level.name}: {source} -> {translation}")

            # 触发回溯：必须在 commit 之后调用，避免跨库事务不一致
            de_cfg = _config.decision_engine
            should_backtrack = False
            if level == DecisionLevel.TERMINOLOGY and de_cfg.get("terminology_triggers_backtrack", True):
                should_backtrack = True
            elif level == DecisionLevel.REFERENCE and de_cfg.get("reference_triggers_backtrack", True):
                should_backtrack = True
            elif level == DecisionLevel.STYLE and de_cfg.get("style_triggers_backtrack", True):
                should_backtrack = True

            if should_backtrack and affected_chunks and decision_id:
                self._trigger_backtrack(affected_chunks)
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

    def _cleanup_orphan_impacts(self, chunk_ids: List[str]):
        """清理指向已删除 chunk 的 decision_impact 孤儿记录"""
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor.execute(
            f'DELETE FROM decision_impact WHERE chunk_id IN ({placeholders})',
            tuple(chunk_ids)
        )
        self.conn.commit()

    def get_all_decisions(self):
        """为 Agent 提供 Prompt 上下文"""
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

    def set_scheduler_factory(self, factory: Callable):
        """设置调度器工厂函数（用于延迟绑定，避免循环导入）"""
        self._scheduler_factory = factory


# ────────────────────────────────────────────────────────────────────
# Section 5 — SmartChunker 文本切分器
# ────────────────────────────────────────────────────────────────────

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
        open_quotes = False  # 对话状态探针

        for p in paragraphs:
            # 1. 物理边界探测 (Scene-Aware)
            is_scene_break = bool(self.scene_break_pattern.match(p))

            # 如果遇到新场景或标题，且当前缓冲区有内容，立刻打包上一个 Chunk
            if is_scene_break and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
                open_quotes = False

            # 将当前段落加入缓冲区
            current_chunk.append(p)
            current_len += len(p)

            # 2. 对话状态探针更新
            quotes_count = p.count('"') + p.count('\u201c') + p.count('\u201d')
            if quotes_count % 2 != 0:
                open_quotes = not open_quotes

            # 3. 软硬边界触发逻辑
            if not is_scene_break:
                # 软边界：达到字数且无跨段对话悬挂
                if current_len >= self.soft_limit and not open_quotes:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    open_quotes = False
                # 硬边界：字数超限
                elif current_len >= self.hard_limit:
                    print(f"⚠️ 触发硬切分保护 (长度: {current_len})，可能存在未闭合引号。")
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    open_quotes = False

        # 收尾
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


# ────────────────────────────────────────────────────────────────────
# Section 6 — Agents
# ────────────────────────────────────────────────────────────────────

# ----- 6.1 ReferenceAgent -----

class ReferenceAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client()
        self.db = decision_engine

    def _build_prompt(self, text_chunk: str) -> str:
        return f"""你是一位精通西方文学、历史、神话和宗教的资深翻译考据专家。
你的任务是扫描给定的科幻/奇幻小说片段，识别出其中的文学典故、宗教隐喻、神话引用或历史名词，并制定翻译策略。

【严格定义】
不要提取普通的角色名字（如 Paul）或普通地名，除非它们具有明显的象征意义或典故来源。

【翻译策略池】
对于识别出的典故，你必须从以下策略中选择一种：
1. "RETAIN_AND_ANNOTATE" (保留音译/直译，并在后续生成脚注)
2. "CULTURAL_EQUIVALENT" (寻找目标语言中的等效文化意象)
3. "LITERAL_TRANSLATION" (仅作字面翻译，放弃深层隐喻)

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
  ]
}}

【待分析文本】
{text_chunk}
"""

    def _parse_and_clean_json(self, raw_response: str) -> Dict[str, Any]:
        """防御性 JSON 解析器：处理本地 LLM 可能输出的杂乱字符"""
        cleaned = re.sub(r'^```json\s*', '', raw_response, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 解析失败，模型输出格式违规: {e}")
            print(f"原始内容预览: {raw_response[:200]}...")
            return {"references": []}

    def process_chunk(self, chunk_id: str, text_chunk: str, affected_chunks: List[str] = None):
        """处理单个文本块，识别典故并自动写入决策引擎"""
        print(f"🔍 [Reference Agent] 正在考据数据块: {chunk_id}...")

        # 1. 调用 LLM (使用配置路由)
        prompt = self._build_prompt(text_chunk)
        model_key, params = _config.resolve_task_model("reference_extraction")
        model_cfg = _config._get_model_config(model_key)
        model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "qwen/Qwen2.5-7B-Instruct-MLX-4bit")

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params
        )

        # 2. 解析 JSON
        parsed_data = self._parse_and_clean_json(raw_output)
        references = parsed_data.get("references", [])

        if not references:
            print("  未发现深度典故。")
            return

        # 3. 将决策写入 Decision DB (Level 2)
        for ref in references:
            source = ref.get("source_text")
            translation = ref.get("translated_term")
            strategy = ref.get("strategy")
            reason = f"【来源】{ref.get('allusion_target')} | 【策略】{strategy} | 【依据】{ref.get('reason')}"

            if source and translation:
                self.db.add_decision(
                    level=DecisionLevel.REFERENCE,
                    source=source,
                    translation=translation,
                    reason=reason,
                    affected_chunks=affected_chunks or [chunk_id]
                )


# ----- 6.2 LiteraryRewriterAgent -----

class LiteraryRewriterAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client()
        self.db = decision_engine

    def _build_decision_context(self) -> str:
        """从 Decision DB 提取当前生效的宪法，转化为 Prompt 上下文"""
        decisions = self.db.get_all_decisions()
        if not decisions:
            return "无特殊词汇约束。"

        context = "【全局翻译决策（必须严格遵守）】\n"
        for level, source, trans in decisions:
            if level == DecisionLevel.TERMINOLOGY.value:
                context += f"- 术语: '{source}' -> 必须译为 '{trans}'\n"
            elif level == DecisionLevel.REFERENCE.value:
                context += f"- 典故: '{source}' -> 必须译为 '{trans}' (若策略为保留并加注，请生成脚注)\n"
            elif level == DecisionLevel.STYLE.value:
                context += f"- 风格: {source} -> {trans}\n"
        return context

    def _build_prompt(self, raw_translation: str, decisions_context: str, style_guide: str) -> str:
        return f"""你是一位荣获过星云奖和雨果奖的资深科幻/奇幻文学译者。
你的任务是对提供的【直译底稿】进行最高水准的文学润色。

{decisions_context}

【风格基准 (Style Guide)】
{style_guide}

【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. 当遇到需要加注的【典故】时，必须使用 Markdown 原生脚注语法。
3. 在正文中需要加注的词语后紧跟 `[^数字]`（如：拉米亚[^1]）。
4. 在你输出的全部正文**最末尾**，空两行，然后列出对应的脚注内容。脚注格式必须为：`[^数字]: 译注：[考据原因]`。

【直译底稿】
{raw_translation}

请直接输出润色后的 Markdown 文本，不要包含任何多余的开头问候或解释：
"""

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

        # 2. 诗歌/韵文特征
        poetry_markers = ['\n\n', '——', '...', 'beauty is truth', 'truth beauty']
        poetry_score = sum(1 for m in poetry_markers if m in source_text)

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

    def _build_style_guide(self, style_guide_stats: dict, source_text: str = "") -> str:
        """构建风格指南，包含动态 Author_Priority_Ratio"""
        if source_text:
            author_priority = self._infer_author_priority_ratio(source_text)
        else:
            author_priority = style_guide_stats.get('author_priority_ratio', 0.7)

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
            f"{priority_instruction}"
        )
        return style_guide

    def process_chunk(self, chunk_id: str, raw_translation: str, style_guide_stats: dict, source_text: str = ""):
        """执行润色并生成最终排版"""
        print(f"✍️ [Rewriter Agent] 正在进行文学润色: {chunk_id}...")

        style_guide = self._build_style_guide(style_guide_stats, source_text)
        decisions_context = self._build_decision_context()
        prompt = self._build_prompt(raw_translation, decisions_context, style_guide)

        # 使用配置路由：literary_rewrite
        model_key, params = _config.resolve_task_model("literary_rewrite")
        model_cfg = _config._get_model_config(model_key)
        model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "qwen/Qwen2.5-7B-Instruct-MLX-4bit")

        final_markdown = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params
        )

        return final_markdown


# ----- 6.3 CriticAgent -----

CRITIC_THRESHOLDS = {
    "fluency": 7.0,
    "style_compliance": 7.0,
    "voice_consistency": 7.0,
    "semantic_preservation": 7.0,
    "readability": 7.0,
}

class CriticAgent:
    def __init__(self):
        self.llm = get_llm_client()

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

请输出纯 JSON，不要包含 Markdown 标记。
Schema 要求：
{{
  "scores": {{
    "fluency": 8,
    "readability": 8,
    "style_compliance": 7,
    "voice_consistency": 9,
    "semantic_preservation": 8
  }},
  "is_flawed": false,
  "critique": "一段尖锐的综合评价",
  "improvement_suggestions": "针对低分项给出具体的修改建议（若无则留空）"
}}
"""

    def process_chunk(self, chunk_id: str, source_text: str, raw_trans: str, lit_trans: str, style_guide: dict) -> Dict[str, Any]:
        print(f"🧐 [Critic Agent] 正在对 {chunk_id} 进行多维度文学审计...")
        prompt = self._build_prompt(source_text, raw_trans, lit_trans, style_guide)

        # 使用配置路由：critic_scoring
        model_key, params = _config.resolve_task_model("critic_scoring")
        model_cfg = _config._get_model_config(model_key)
        model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "qwen/Qwen2.5-7B-Instruct-MLX-4bit")

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params
        )

        cleaned = re.sub(r'^```json\s*', '', raw_output, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ Critic Agent 输出异常: {e}")
            return {"is_flawed": True, "critique": "JSON解析失败", "scores": {}}

        # 自动计算 is_flawed：任意维度低于阈值即判定为有缺陷
        scores = result.get("scores", {})
        is_flawed = result.get("is_flawed", False)
        if not is_flawed and scores:
            # 从配置读取阈值
            thresholds = _config.critic_thresholds
            for dim, threshold in thresholds.items():
                if dim in scores and scores[dim] < threshold:
                    is_flawed = True
                    result["critique"] = f"{result.get('critique', '')} [自动判定: {dim}={scores[dim]} < {threshold}]"
                    break
        result["is_flawed"] = is_flawed
        return result


# ----- 6.4 JudgeAgent -----

class JudgeAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client()
        self.db = decision_engine

    def _build_prompt(self, source_text: str, lit_trans: str, critic_report: dict) -> str:
        return f"""你是一位星云奖级别的终审译者。
你需要综合【审辩者报告】，决定当前的【文学润色稿】是否可以直接定稿。

【审辩者报告】
{json.dumps(critic_report, ensure_ascii=False)}

【原文】
{source_text}
【当前润色稿】
{lit_trans}

任务与输出 JSON Schema：
{{
  "decision": "PASS" | "REJECT",
  "final_text": "如果 PASS，请输出最终平滑过的译文；如果 REJECT，此项留空",
  "reject_reason": "如果 REJECT，告诉上游的 Rewriter 必须重点修改哪里",
  "new_style_rule": {{
    "rule_description": "例如：处理独白时必须使用短促、断裂的句式。",
    "reason": "为什么这条规则对整本书很重要？"
  }}
}}"""

    def process_chunk(self, chunk_id: str, source_text: str, lit_trans: str, critic_report: dict, affected_chunks: List[str] = None) -> Dict[str, Any]:
        print(f"⚖️ [Judge Agent] 正在对 {chunk_id} 进行最终裁决...")
        prompt = self._build_prompt(source_text, lit_trans, critic_report)

        # 使用配置路由：judge_decision
        model_key, params = _config.resolve_task_model("judge_decision")
        model_cfg = _config._get_model_config(model_key)
        model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "qwen/Qwen2.5-7B-Instruct-MLX-4bit")

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params
        )

        cleaned = re.sub(r'^```json\s*', '', raw_output, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ Judge Agent 格式错误: {e}，强制裁定为 REJECT。")
            return {"decision": "REJECT", "reject_reason": "Judge 自身输出损坏，触发重试。"}

        # 自动裁决逻辑：Critic 判定有缺陷则必须 REJECT
        is_flawed = critic_report.get("is_flawed", False)
        scores = critic_report.get("scores", {})

        if is_flawed:
            result["decision"] = "REJECT"
            low_dims = [k for k, v in scores.items() if isinstance(v, (int, float)) and v < 7]
            result["reject_reason"] = f"Critic 判定不合格: {', '.join(low_dims)} 低于阈值。{critic_report.get('improvement_suggestions', '')}"

        # 额外检查：平均分过低也 REJECT (从配置读取)
        if not is_flawed and scores:
            avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / len(scores)
            avg_min = _config.critic_thresholds.get("average_score_min", 7.5)
            if avg_score < avg_min:
                result["decision"] = "REJECT"
                result["reject_reason"] = f"平均分 {avg_score:.1f} 过低，需提升整体质量。"

        # 如果 PASS 并且提炼出了新的高级规则，写入 Decision DB (Level 3)
        if result.get("decision") == "PASS" and "new_style_rule" in result and result["new_style_rule"]:
            rule = result["new_style_rule"]
            if rule.get("rule_description"):
                self.db.add_decision(
                    level=DecisionLevel.STYLE,
                    source=f"风格约束_{chunk_id}",
                    translation=rule["rule_description"],
                    reason=rule.get("reason", "Judge Agent 动态提炼"),
                    affected_chunks=affected_chunks or [chunk_id]
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

        self.llm = get_llm_client()

        # 实例化 Agents
        self.ref_agent = ReferenceAgent(self.decision_engine)
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

    def _load_intermediate(self, chunk_id: str, step: str) -> str | dict:
        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("text", data)

    def _run_raw_translator(self, source_text: str) -> str:
        prompt = f"请将以下科幻小说片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n【原文】\n{source_text}"
        # 使用配置路由：literal_translation
        model_key, params = _config.resolve_task_model("literal_translation")
        model_cfg = _config._get_model_config(model_key)
        model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "google/gemma-2-9b-it-mlx-4bit")
        return self.llm.generate(prompt, model_name=model_name, **params)

    def run(self):
        print(f"🚀 [Pipeline] 启动批处理模式处理章节: {self.chapter_id}")

        # 检查章节是否有任务，避免在空 DB 上误报"完成"
        existing_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
        if not existing_tasks:
            print(f"❌ [Pipeline] 章节 {self.chapter_id} 无任务，请先运行 init 命令。")
            return

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
                tasks = self.scheduler.get_tasks_by_state(state, batch_size=batch_size)
                if not tasks:
                    continue

                print(f"📦 [Batch] {stage_name} 阶段: 处理 {len(tasks)} 个任务")
                handler(tasks)
                any_progress = True

            if not any_progress:
                print(f"🎉 [Pipeline] 章节 {self.chapter_id} 全部处理完成！")
                break

            time.sleep(poll_interval)

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
                    intermediate.unlink()
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS)

    def _process_failed_batch(self, tasks):
        retry_chunk_ids = []
        permanent_fail_ids = []
        # 从配置读取最大重试次数
        max_retries = _config.pipeline.get("max_retries", 3)
        for task in tasks:
            chunk_id = task['chunk_id']
            retries = task.get('retries', 0)
            if retries >= max_retries:
                print(f"❌ [Pipeline] {chunk_id} 重试次数过多，转入 PERMANENTLY_FAILED 终态")
                permanent_fail_ids.append(chunk_id)
            else:
                print(f"🔁 [Pipeline] {chunk_id} 重试中 (第 {retries + 1} 次)")
                retry_chunk_ids.append(chunk_id)
        if permanent_fail_ids:
            self.scheduler.batch_update_state(
                permanent_fail_ids,
                TaskState.PERMANENTLY_FAILED,
                error_msg=f"超过重试上限 (retries>={max_retries})，需人工介入"
            )
        if retry_chunk_ids:
            self.scheduler.batch_update_state(retry_chunk_ids, TaskState.EXTRACTING_TERMS)

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
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
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
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.REWRITING_LITERARY)

    def _process_rewriting_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                raw_text = self._load_intermediate(chunk_id, "raw")
                lit_text = self.rewriter_agent.process_chunk(chunk_id, raw_text, self.style_guide, source_text)
                self._save_intermediate(chunk_id, "literary", lit_text)
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 润色异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.AUDITING)

    def _process_auditing_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                raw_text = self._load_intermediate(chunk_id, "raw")
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self.critic_agent.process_chunk(chunk_id, source_text, raw_text, lit_text, self.style_guide)
                self._save_intermediate(chunk_id, "critic_report", critic_report)
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 审计异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.JUDGING)

    def _process_judging_batch(self, tasks):
        max_retries = _config.pipeline.get("max_retries", 3)
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                retries = task.get('retries', 0)
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self._load_intermediate(chunk_id, "critic_report")
                judge_result = self.judge_agent.process_chunk(chunk_id, source_text, lit_text, critic_report, affected_chunks=[chunk_id])

                if judge_result.get("decision") == "PASS":
                    self._save_intermediate(chunk_id, "final", judge_result.get("final_text", lit_text))
                    self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED)
                    print(f"✅ [Pipeline] {chunk_id} 定稿完成。")
                elif retries >= max_retries:
                    # 重试已达上限，直接转入终态
                    self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, error_msg=judge_result.get("reject_reason"))
                    print(f"❌ [Pipeline] {chunk_id} 连续 {retries+1} 次未通过裁决，转入 PERMANENTLY_FAILED")
                else:
                    self.scheduler.update_task_state(chunk_id, TaskState.REWRITING_LITERARY, error_msg=judge_result.get("reject_reason"))
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 裁决异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))


# ────────────────────────────────────────────────────────────────────
# Section 8 — 测试工具
# ────────────────────────────────────────────────────────────────────

# ----- 8.1 Golden Set Evaluator -----

class GoldenSetEvaluator:
    """黄金测试集评估器 - 风格坍缩率量化"""

    def __init__(self, db_path: Optional[str] = None):
        self.llm = get_llm_client()
        shared_db = DecisionEngine(db_path=db_path) if db_path else DecisionEngine()
        self.db = shared_db
        self.ref_agent = ReferenceAgent(shared_db)
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
        sentences = text.split('。')
        sentences = [s for s in sentences if s.strip()]

        avg_sent_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0

        words = text.split()
        _stop_words = {'的', '了', '是', '我', '在', '有', '和', '为'}
        content_words = [w for w in words if len(w) > 1 and w not in _stop_words]
        vocab_density = len(content_words) / len(words) if words else 0

        rhetorical_markers = ['如', '似', '仿佛', '好像', '像', '犹如', '宛若']
        rhetorical_count = sum(text.count(m) for m in rhetorical_markers)
        rhetorical_density = rhetorical_count / len(sentences) if sentences else 0

        punctuation = ['，', '。', '；', '：', '——', '…', '\u201c', '\u201d', '\u2018', '\u2019']
        punct_count = sum(text.count(p) for p in punctuation)
        punct_density = punct_count / len(text) * 1000 if text else 0

        return {
            "avg_sentence_length": avg_sent_len,
            "vocabulary_density": vocab_density,
            "rhetorical_density": rhetorical_density,
            "punctuation_density": punct_density
        }


def run_golden_test():
    """运行黄金测试集（需要 input/golden/hyperion_5k.md）"""
    print("=" * 60)
    print("🧪 OpenLiterary 黄金测试集跑分")
    print("=" * 60)

    root_dir = Path(__file__).resolve().parent.parent
    test_file = root_dir / "input" / "golden" / "hyperion_5k.md"
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return

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

            raw_prompt = f"请将以下科幻小说片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n【原文】\n{chunk}"
            # 使用配置路由 literal_translation
            model_key, params = _config.resolve_task_model("literal_translation")
            model_cfg = _config._get_model_config(model_key)
            model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "google/gemma-2-9b-it-mlx-4bit")
            raw_text = evaluator.llm.generate(raw_prompt, model_name=model_name, **params)

            lit_text = evaluator.rewriter_agent.process_chunk(chunk_id, raw_text, style_guide, chunk)

            critic_report = evaluator.critic_agent.process_chunk(chunk_id, chunk, raw_text, lit_text, style_guide)

            judge_result = evaluator.judge_agent.process_chunk(chunk_id, chunk, lit_text, critic_report, affected_chunks=[chunk_id])

            final_text = judge_result.get("final_text", lit_text)
            style_eval = evaluator.evaluate_style_collapse(chunk, final_text)

            result = {
                "chunk_id": chunk_id,
                "source_length": len(chunk),
                "critic_scores": critic_report.get("scores", {}),
                "critic_flawed": critic_report.get("is_flawed", True),
                "judge_decision": judge_result.get("decision"),
                "style_collapse_rate": style_eval["style_collapse_rate"],
                "style_preservation": style_eval["average_preservation"],
                "preservation_details": style_eval["preservation_per_dimension"]
            }
            all_results.append(result)

            print(f"  Critic: flawed={critic_report.get('is_flawed')}, scores={critic_report.get('scores')}")
            print(f"  Judge: {judge_result.get('decision')}")
            print(f"  风格坍缩率: {style_eval['style_collapse_rate']:.2%}")
            print(f"  风格保持度: {style_eval['average_preservation']:.2%}")
        except Exception as e:
            print(f"⚠️ 第 {i+1} 块处理失败，跳过: {e}")
            all_results.append({"chunk_id": chunk_id, "error": str(e)})

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
    import psutil as _psutil
    mem = _psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024**3),
        "available_gb": mem.available / (1024**3),
        "used_gb": mem.used / (1024**3),
        "percent": mem.percent
    }


def get_process_memory() -> dict:
    """获取当前进程内存信息"""
    import psutil as _psutil
    process = _psutil.Process(os.getpid())
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


def test_memory_pressure():
    """运行 5 项内存压力测试"""
    print("=" * 60)
    print("🧪 OpenLiterary 内存压力测试 (16GB 限制)")
    print("=" * 60)

    sys_mem = get_system_memory()
    print(f"💻 系统内存: {sys_mem['total_gb']:.1f}GB 总计, {sys_mem['available_gb']:.1f}GB 可用")

    client = get_llm_client()
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
    print(f"\n✅ 初始化成功，数据已写入: {db_file}")


def debug_db():
    """调试工具：查看数据库中的任务状态"""
    root_dir = Path(__file__).resolve().parent.parent
    db_path = root_dir / "db" / "workflow.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, state FROM chunk_tasks")
    tasks = cursor.fetchall()

    print(f"数据库中任务总数: {len(tasks)}")
    for chunk_id, state in tasks:
        print(f"任务 {chunk_id} 当前状态: {state}")
    conn.close()


# ────────────────────────────────────────────────────────────────────
# Section 10 — 统一入口
# ────────────────────────────────────────────────────────────────────

def print_banner():
    """横幅输出到 stderr，避免污染 stdout（stdout 留给机器可解析的输出，如章节路径）"""
    print(r"""
   ____                   __    _ __                            
  / __ \____  ___  ____  / /   (_) /____  _________ ________  __
 / / / / __ \/ _ \/ __ \/ /   / / __/ _ \/ ___/ __ `/ ___/ / / /
/ /_/ / /_/ /  __/ / / / /___/ / /_/  __/ /  / /_/ / /  / /_/ / 
\____/ .___/\___/_/ /_/_____/_/\__/\___/_/   \__,_/_/   \__, /  
    /_/                                                /____/   

  OpenLiterary — AI 文学语义编译系统  (单体聚合版)
  共 {} 个模块 | 用于全量代码审计 / 单文件部署
""".format("10"), file=sys.stderr)

def main():
    print_banner()

    import argparse
    parser = argparse.ArgumentParser(description="OpenLiterary — AI 文学语义编译系统")
    parser.add_argument("command", nargs="?", default="pipeline",
                        choices=["pipeline", "init", "golden", "memory", "debug", "split"],
                        help="执行命令: pipeline (默认), init, golden, memory, debug, split")
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

    args = parser.parse_args()

    if args.command == "pipeline":
        pipeline = TranslationPipeline(chapter_id=args.chapter)
        pipeline.run()
    elif args.command == "init":
        init_project(chapter_id=args.chapter, force=args.force)
    elif args.command == "golden":
        run_golden_test()
    elif args.command == "memory":
        test_memory_pressure()
    elif args.command == "debug":
        debug_db()
    elif args.command == "split":
        if not args.input:
            print("❌ split 命令需要 --input 参数", file=sys.stderr)
            print("   用法: python -m src.translator_agent split --input <file.epub|txt|md>", file=sys.stderr)
            sys.exit(1)
        split_input_to_chapters(
            input_path=args.input,
            output_dir=args.input_dir,
            input_format=args.input_format,
            target_chars=args.chapter_size,
            min_chars=args.min_chapter_size,
        )


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


def _epub_clean_html_text(html_content: str) -> str:
    """清洗 HTML，提取纯文本（保留段落结构）"""
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    for br in soup.find_all('br'):
        br.replace_with('\n')
    for p in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        p.insert_after('\n')

    text = soup.get_text(separator='\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    return text.strip()


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
            if f'id="{anchor_id}"' in line or f'name="{anchor_id}"' in line:
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
        text = chunk_soup.get_text(separator='\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        text = text.strip()

        if not text or len(text) < 50:
            continue

        chapter_idx = chapter_idx_start + len(chapters) + 1
        ch_id = f"ch{chapter_idx:02d}"
        chapters.append((ch_id, f"# {title}\n\n{text}"))

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

        if not text or len(text) < 50:
            continue

        if len(text) <= LARGE_DOC_THRESHOLD:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
            title = title_tag.get_text(strip=True) if title_tag else f"Chapter {chapter_idx}"
            chapters.append((ch_id, f"# {title}\n\n{text}"))
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
            chapters.append((ch_id, f"# {title}\n\n{text}"))

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
        return self.chapters

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
            markdown = f"# {title}\n\n{body}"
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

    if input_format == 'epub':
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
        _splitter_log_info(f"     python -m src.translator_agent init --chapter {ch_id} --force")
        _splitter_log_info(f"     python -m src.translator_agent pipeline --chapter {ch_id}")
    if len(generated) > 5:
        _splitter_log_info(f"     ... (共 {len(generated)} 章)")

    # 章节路径输出到 stdout（供 mapfile / shell 脚本捕获）
    for f in generated:
        print(f)
    return generated


if __name__ == "__main__":
    main()
