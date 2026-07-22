#!/usr/bin/env python3
"""
黄金测试集跑分脚本
使用《海伯利安》前 5000 字进行测试，记录风格坍缩率
"""

import sys
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from core.scheduler import TaskScheduler, TaskState
from core.decision_engine import DecisionEngine
from utils.llm_adapter import get_llm_client_for_task
from agents.reference_agent import ReferenceAgent
from agents.rewriter_agent import LiteraryRewriterAgent
from agents.critic_agent import CriticAgent
from agents.judge_agent import JudgeAgent
from utils.chunker import SmartChunker


class GoldenSetEvaluator:
    def __init__(self, db_path: str = None):
        self.llm = get_llm_client_for_task("literal_translation")
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
        
        # 计算风格保持度 (0-1，越高越好)
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
        
        # 平均句长
        avg_sent_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 词汇密度：内容词 / 总词数
        words = text.split()
        _stop_words = {'的', '了', '是', '我', '在', '有', '和', '为'}
        content_words = [w for w in words if len(w) > 1 and w not in _stop_words]
        vocab_density = len(content_words) / len(words) if words else 0
        
        # 修辞密度：比喻、拟人、排比等
        rhetorical_markers = ['如', '似', '仿佛', '好像', '像', '犹如', '宛若']
        rhetorical_count = sum(text.count(m) for m in rhetorical_markers)
        rhetorical_density = rhetorical_count / len(sentences) if sentences else 0
        
        # 标点丰富度
        punctuation = ['，', '。', '；', '：', '——', '…', '“', '”', '‘', '’']
        punct_count = sum(text.count(p) for p in punctuation)
        punct_density = punct_count / len(text) * 1000 if text else 0
        
        return {
            "avg_sentence_length": avg_sent_len,
            "vocabulary_density": vocab_density,
            "rhetorical_density": rhetorical_density,
            "punctuation_density": punct_density
        }


def run_golden_test():
    print("=" * 60)
    print("🧪 OpenLiterary 黄金测试集跑分")
    print("=" * 60)
    
    # 读取测试文本
    test_file = ROOT_DIR / "input" / "golden" / "hyperion_5k.md"
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    with open(test_file, "r", encoding="utf-8") as f:
        source_text = f.read()
    
    print(f"📖 测试文本长度: {len(source_text)} 字符")
    
    # 切分
    chunker = SmartChunker(soft_limit=1000, hard_limit=2500)
    chunks = chunker.split_markdown(source_text)
    print(f"✂️ 切分为 {len(chunks)} 个块")
    
    evaluator = GoldenSetEvaluator(db_path=str(ROOT_DIR / "db" / "decision_db.sqlite"))
    all_results = []
    
    # 逐块处理
    for i, chunk in enumerate(chunks):
        chunk_id = f"golden_chunk{i:03d}"
        try:
            print(f"\n📦 处理第 {i+1}/{len(chunks)} 块 ({len(chunk)} 字符)...")

            # 1. Reference Agent - 典故提取
            evaluator.ref_agent.process_chunk(chunk_id, chunk, affected_chunks=[chunk_id])

            # 2. 直译（使用配置中的模型名，而非角色 key）
            from src.utils.llm_adapter import get_role_model_name
            _lit_model_name = get_role_model_name("literal_translator")
            raw_prompt = f"请将以下科幻小说片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n【原文】\n{chunk}"
            raw_text = evaluator.llm.generate(raw_prompt, model_name=_lit_model_name, max_tokens=1024, temperature=0.1)

            # 3. 文学润色
            style_guide = {
                "avg_sentence_length": "较长且富有韵律",
                "lexicon_preference": "古典、史诗感、冷硬",
                "author_priority_ratio": 0.7
            }
            lit_text = evaluator.rewriter_agent.process_chunk(chunk_id, raw_text, style_guide, chunk)

            # 4. Critic 审计
            critic_report = evaluator.critic_agent.process_chunk(chunk_id, chunk, raw_text, lit_text, style_guide)

            # 5. Judge 裁决
            judge_result = evaluator.judge_agent.process_chunk(chunk_id, chunk, lit_text, critic_report, affected_chunks=[chunk_id])

            # 6. 风格坍缩评估
            style_eval = evaluator.evaluate_style_collapse(chunk, lit_text)

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

    # 详细维度
    dims = ["avg_sentence_length", "vocabulary_density", "rhetorical_density", "punctuation_density"]
    for dim in dims:
        dim_avg = sum(r.get("preservation_details", {}).get(dim, 0) for r in valid_results) / max(len(valid_results), 1)
        print(f"  {dim}: {dim_avg:.2%}")

    # 保存结果
    output_file = ROOT_DIR / "output" / "golden_test_report.json"
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


if __name__ == "__main__":
    run_golden_test()
