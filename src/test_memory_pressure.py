"""
SagaTrans 内存压力测试工具

源：原整合脚本 §8 中的 Memory 相关部分，拆分到独立模块。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import psutil

from src.config import config as _config
from src.utils.llm_adapter import get_llm_client_for_task
from src.test_golden_set import run_golden_ab, run_golden_test


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


# ────────────────────────────────────────────────────────────────────
