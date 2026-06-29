#!/usr/bin/env python3
"""
内存压力测试脚本
模拟 16GB 内存限制下的显存压力测试
"""

import sys
import time
import gc
import psutil
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from utils.llm_adapter import get_llm_client, SYS_CONFIG


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
    # 分配字节数组
    data = bytearray(int(target_gb * 1024 * 1024 * 1024))
    return data


def test_memory_pressure():
    print("=" * 60)
    print("🧪 OpenLiterary 内存压力测试 (16GB 限制)")
    print("=" * 60)
    
    # 系统信息
    sys_mem = get_system_memory()
    print(f"💻 系统内存: {sys_mem['total_gb']:.1f}GB 总计, {sys_mem['available_gb']:.1f}GB 可用")
    
    # 获取 LLM 客户端
    client = get_llm_client()
    print(f"🤖 当前后端: {SYS_CONFIG['llm_backend']}")
    
    # 测试 1: 基础内存监控
    print("\n📊 测试 1: 基础内存监控")
    proc_mem = get_process_memory()
    print(f"  进程内存: RSS={proc_mem['rss_gb']:.2f}GB, VMS={proc_mem['vms_gb']:.2f}GB")
    
    # 测试 2: 生成性能基线
    print("\n⚡ 测试 2: 生成性能基线")
    test_prompt = "请将以下文本翻译成中文：The Shrike is not a god, nor a demon, nor even a machine."
    start = time.time()
    result = client.generate(test_prompt, model_name="test-model", max_tokens=100, temperature=0.3)
    elapsed = (time.time() - start) * 1000
    print(f"  生成耗时: {elapsed:.0f}ms")
    print(f"  结果长度: {len(result)} 字符")
    
    proc_mem = get_process_memory()
    print(f"  生成后内存: RSS={proc_mem['rss_gb']:.2f}GB")
    
    # 测试 3: 内存压力模拟 (仅在有足够内存时)
    if sys_mem['available_gb'] > 4:
        print("\n🔥 测试 3: 内存压力模拟")
        # 分配 2GB 内存模拟模型加载
        load_data = simulate_memory_load(2.0)
        
        proc_mem = get_process_memory()
        print(f"  负载后进程内存: RSS={proc_mem['rss_gb']:.2f}GB ({proc_mem['percent']:.1f}%)")
        
        # 检查内存压力检测
        if hasattr(client, 'check_memory_pressure'):
            pressure = client.check_memory_pressure()
            print(f"  内存压力检测: {'触发' if pressure else '正常'}")
        
        # 释放负载
        del load_data
        gc.collect()
        time.sleep(1)
        
        proc_mem = get_process_memory()
        print(f"  释放后进程内存: RSS={proc_mem['rss_gb']:.2f}GB")
    else:
        print("\n⚠️ 系统可用内存不足，跳过内存压力模拟")
    
    # 测试 4: 连续生成压力测试
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
    
    # 测试 5: 模型卸载 (如果是 MLX)
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


if __name__ == "__main__":
    test_memory_pressure()
