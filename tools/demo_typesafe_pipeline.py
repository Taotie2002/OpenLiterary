#!/usr/bin/env python3
"""手工演示：散装 critic + judge typesafe 链路（用 ch02_chunk000 真实数据）"""
import sys, json
from pathlib import Path
sys.path.insert(0, '/home/zyu/projects/SagaTrans')

from src.agents.critic_agent import CriticAgent
from src.agents.judge_agent import JudgeAgent
from src.core.decision_engine import DecisionEngine
from src.judges.primitives import ALL_PRIMITIVES

# ch02_chunk000 真实数据
raw = json.loads(Path("output/ch02/ch02_chunk000_raw.json").read_text(encoding="utf-8"))["text"]
lit = json.loads(Path("output/ch02/ch02_chunk000_literary_v6.json").read_text(encoding="utf-8"))["text"]

source = "Mr. George Naylor, the young owner of a Hampshire farm, was a man of remarkably varied gifts."
style = {"avg_sentence_length": "适中、口语化", "lexicon_preference": "平实、克制"}
entity_reg = {"Mr. George Naylor": "乔治·内勒", "Hampshire": "汉普郡"}

de = DecisionEngine(db_path="/tmp/demo.db", scheduler_factory=lambda: None)
critic = CriticAgent()
judge = JudgeAgent(decision_engine=de)

print("=" * 60)
print("T2 typesafe-ai 重构演示（ch02_chunk000 真实数据）")
print("=" * 60)

print("\nStep 1: critic.process_chunk（typesafe-ai 路径）")
c = critic.process_chunk(
    chunk_id="ch02_chunk000",
    source_text=source,
    raw_trans=raw,
    lit_trans=lit,
    style_guide=style,
    entity_registry=entity_reg,
)
v = c["_typesafe_verdict"]
print(f"  backend       : {c['_backend_path']}")
print(f"  decision      : {v['decision']}")
print(f"  confidence    : {v['confidence']:.2f}")
print(f"  reject_reasons: {v['reject_reasons']}")
print(f"  legacy scores : {c['scores']}")

print("\nStep 2: judge.process_chunk（继承 critic verdict）")
j = judge.process_chunk(
    chunk_id="ch02_chunk000",
    source_text=source,
    lit_trans=lit,
    critic_report=c,
)
print(f"  decision : {j['decision']}")
print(f"  backend  : {j['_backend_path']}")
print(f"  reason   : {j.get('reject_reason', '')[:120]}")

print("\nStep 3: typesafe primitives 详情")
print(f"  共 {len(ALL_PRIMITIVES)} 个 atomic primitives:")
for qid, p in ALL_PRIMITIVES.items():
    print(f"    - {qid:25s} ({p['type']})")
print("\n注：当前 backend=mock_heuristic 是因为离线（API 不可达）。")
print("   网络恢复后 backend 会自动变为 openai_compat 或 jev。")
