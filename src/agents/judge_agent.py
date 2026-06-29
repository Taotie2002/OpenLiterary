import json
import re
from typing import Dict, Any, List
from utils.llm_adapter import get_llm_client
from core.decision_engine import DecisionEngine, DecisionLevel

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
        
        raw_output = self.llm.generate(
            prompt=prompt,
            model_name="qwen/Qwen2.5-7B-Instruct-MLX-4bit",
            max_tokens=1500,
            temperature=0.3
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
        
        # 额外检查：平均分过低也 REJECT
        if not is_flawed and scores:
            avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / len(scores)
            if avg_score < 7.5:
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