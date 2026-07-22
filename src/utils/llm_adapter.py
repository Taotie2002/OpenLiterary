"""DEPRECATED — 模块化副本，未与生产单体脚本同步。

生产入口是 `python3 -m src.translator_agent`（单体脚本 `src/translator_agent.py`）。
此文件仅保留供历史参考。已知缺陷：
- `robust_json_loads` Strategy 2 类型不符直接 `return json.loads(...)`，未返回 empty（缺陷 B 未修复）
"""
import json
import gc
import re
import threading
import requests
import time
import random
import psutil
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from src.config import config
from json_repair import repair_json as _repair_json

def robust_json_loads(raw_response: str) -> dict:
    """Robust JSON parser for LLM outputs.

    Handles: <think> blocks, markdown fences, preamble text,
    trailing text (via raw_decode), trailing commas (via cleanup),
    and embedded JSON extraction as last resort.

    Returns parsed dict, or empty dict if all attempts fail.
    """
    if not raw_response or not raw_response.strip():
        return {}

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
        return {}

    # Strategy 1: raw_decode extracts first JSON value, ignores trailing text
    try:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(cleaned)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: clean trailing commas and retry
    try:
        fallback = re.sub(r',\s*}', '}', cleaned)
        fallback = re.sub(r',\s*]', ']', fallback)
        return json.loads(fallback)
    except json.JSONDecodeError:
        pass

    # Strategy 3: json_repair handles unescaped quotes, missing brackets, single quotes, etc.
    try:
        repaired = _repair_json(cleaned)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    # Strategy 4: extract outermost { ... } block
    try:
        brace_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group())
    except json.JSONDecodeError:
        pass

    return {}


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
        return usage["percent"] / 100 > config.get("mlx.memory.warning_threshold", 0.8)
    
    def auto_unload_if_needed(self) -> bool:
        """内存压力时自动卸载模型，返回是否触发了卸载"""
        if config.get("mlx.memory.auto_unload_on_pressure", True) and self.check_memory_pressure():
            print(f"⚠️ 内存压力过大 ({self.get_memory_usage()['percent']:.1f}%)，自动卸载模型")
            self.unload_model()
            return True
        return False

class MockLLMAdapter(LLMAdapter):
    """Mock LLM 适配器：用于开发测试，无需真实模型服务"""
    
    def __init__(self):
        self.call_count = 0
        
    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
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
        # 从 prompt 中提取原文
        if "【原文】" in prompt:
            source = prompt.split("【原文】")[-1].strip()
        else:
            source = "未知原文"
        return f"[直译] {source}"
    
    def _mock_literary_rewrite(self, prompt: str) -> str:
        markers = ["【直译底稿（语义基准，不可偏离）】", "【直译底稿】"]
        marker = next((m for m in markers if m in prompt), None)
        if marker:
            # 提取直译底稿，去除后面的指令文本
            after_marker = prompt.split(marker)[-1]
            raw = after_marker.split("请直接输出")[0].strip()
        else:
            raw = ""
        # 移除 [直译] 前缀
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
    # 类级共享限流：所有实例共用同一个 last_request_time，跨角色生效
    _last_request_time: float = 0.0
    _request_lock = threading.Lock()

    def __init__(self, api_base: str, api_key: str, max_retries: int = 3, retry_delay: float = 2.0, request_timeout: int = 300):
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

        # 频率限制：类级共享锁，跨角色实例共用同一计时器
        with self.__class__._request_lock:
            now = time.time()
            elapsed = now - self.__class__._last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            self.__class__._last_request_time = time.time()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
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
            from mlx_lm import load, generate
            import mlx.core as mx
            self.mlx_load = load
            self.mlx_generate = generate
            self.mx = mx
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

# Per-role 客户端缓存：role_key -> LLMAdapter
_client_instances: dict[str, LLMAdapter] = {}


def _build_role_client(role_cfg: Dict[str, Any]) -> LLMAdapter:
    backend = role_cfg.get("backend", "mock")
    if backend == "mock":
        return MockLLMAdapter()
    if backend == "mlx":
        return MLXNativeAdapter()
    if backend in ("openai_api", "ollama", "nim", "mistral", "custom"):
        section = config.get_section("openai_api")
        return OpenAICompatibleAdapter(
            api_base=role_cfg.get("api_base") or section.get("api_base", "http://127.0.0.1:1234/v1"),
            api_key=role_cfg.get("api_key") or section.get("api_key", "lm-studio"),
            max_retries=section.get("max_retries", 3),
            retry_delay=section.get("retry_delay", 2.0),
            request_timeout=section.get("request_timeout", 300),
        )
    raise ValueError(f"不支持的 backend: {backend}")


def get_llm_client(role: str) -> LLMAdapter:
    """按角色返回 LLM 客户端实例（首次按需构建并缓存）。"""
    if role not in _client_instances:
        role_cfg = config.get_role_config(role)
        _client_instances[role] = _build_role_client(role_cfg)
    return _client_instances[role]


def get_llm_client_for_task(task_name: str) -> LLMAdapter:
    """根据任务名（如 'reference_extraction'）解析出角色并返回客户端。"""
    return get_llm_client(config.get_task_role(task_name))


def get_role_model_name(role: str) -> str:
    """返回该角色对应的 model 标识（MLX: model_id；HTTP: model_name）。"""
    cfg = config.get_role_config(role)
    return cfg.get("model_id") or cfg.get("model_name", "")


def get_role_extra_body(role: str) -> Dict[str, Any]:
    """返回该角色的 extra_body（如思考模式开关）。"""
    return config.get_role_config(role).get("extra_body") or {}


def reset_clients() -> None:
    """清空客户端缓存（测试 / 切换配置后用）。"""
    _client_instances.clear()