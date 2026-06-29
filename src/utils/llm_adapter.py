import json
import gc
import requests
import time
import random
import psutil
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

# 全局极简配置字典（后续可迁移至 config.yaml）
SYS_CONFIG = {
    "llm_backend": "mock",  # 可选: "mock" (测试), "openai_api" (LM Studio/vLLM), "mlx" (Apple Silicon)
    "mlx_models": {
        "4b_model": "google/gemma-2-9b-it-mlx-4bit",
        "9b_model": "qwen/Qwen2.5-7B-Instruct-MLX-4bit" 
    },
    "openai_api_base": "http://127.0.0.1:1234/v1",
    "openai_api_key": "lm-studio",
    "max_retries": 3,
    "retry_delay": 2,
    "memory_warning_threshold": 0.8,
    "auto_unload_on_pressure": True,
}

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
        return usage["percent"] / 100 > SYS_CONFIG.get("memory_warning_threshold", 0.8)
    
    def auto_unload_if_needed(self) -> bool:
        """内存压力时自动卸载模型，返回是否触发了卸载"""
        if SYS_CONFIG.get("auto_unload_on_pressure", True) and self.check_memory_pressure():
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
        if "【直译底稿】" in prompt:
            # 提取直译底稿，去除后面的指令文本
            after_marker = prompt.split("【直译底稿】")[-1]
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

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
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

    def generate(self, prompt: str, model_name: str, max_tokens: int = 2048, temperature: float = 0.3) -> str:
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
    """获取当前配置的 LLM 客户端实例"""
    global _client_instance
    if _client_instance is None:
        if SYS_CONFIG["llm_backend"] == "mock":
            _client_instance = MockLLMAdapter()
        elif SYS_CONFIG["llm_backend"] == "mlx":
            _client_instance = MLXNativeAdapter()
        elif SYS_CONFIG["llm_backend"] == "openai_api":
            _client_instance = OpenAICompatibleAdapter(
                SYS_CONFIG["openai_api_base"], 
                SYS_CONFIG["openai_api_key"],
                max_retries=SYS_CONFIG.get("max_retries", 3),
                retry_delay=SYS_CONFIG.get("retry_delay", 2.0)
            )
        else:
            raise ValueError(f"不支持的 llm_backend: {SYS_CONFIG['llm_backend']}")
    return _client_instance