import json
import re
from typing import Dict, Any, List
from utils.llm_adapter import get_llm_client_for_task, get_role_model_name, get_role_extra_body, robust_json_loads
from core.decision_engine import DecisionEngine, DecisionLevel
from src.config import config

class JudgeAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client_for_task("judge_decision")
        self.role = config.get_task_role("judge_decision")
        self.db = decision_engine

    def _build_prompt(self, source_text: str, lit_trans: str, critic_report: dict) -> str:
        # P0: 将 critic_thresholds 注入为参考阈值（非硬性否决条件）
        thresholds = config.critic_thresholds
        thresholds_lines = "\n".join(
            f"  - {dim}: 参考阈值 {v}"
            for dim, v in thresholds.items() if dim != "average_score_min"
        )
        thresholds_block = f"""
【评分参考阈值（供综合判断，不是硬性否决条件）】
审辩者各维度的参考阈值为：
{thresholds_lines}

如果审辩者报告的 scores 中某维度略低于阈值，但你认为整体质量达标，可以判 PASS。
如果多个维度显著低于阈值，且润色稿确实问题明显，才判 REJECT。
"""
        return f"""你是一位星云奖级别的终审译者。
你需要综合【审辩者报告】，决定当前的【文学润色稿】是否可以直接定稿。

【审辩者报告】
{json.dumps(critic_report, ensure_ascii=False)}

【原文】
{source_text}
【当前润色稿】
{lit_trans}
{thresholds_block}
【JSON 输出规则 — 必须严格遵守】
1. 输出纯 JSON，禁止包含任何 Markdown 标记、代码块、或额外说明文字
2. 字符串值中的双引号 " 必须转义为 \"
3. 不得在数组/对象的最后一个元素后加逗号
4. ⚠️ 如果 JSON 格式错误，裁决将直接判定为 REJECT

输出 JSON Schema：
{{
  "decision": "PASS" | "REJECT",
  "reject_reason": "如果 REJECT，告诉上游的 Rewriter 必须重点修改哪里",
  "new_style_rule": {{
    "rule_description": "例如：处理独白时必须使用短促、断裂的句式。",
    "reason": "为什么这条规则对整本书很重要？"
  }}
}}

注意："final_text" 不需要你输出，系统将自动使用 Rewriter 的润色稿作为最终译文。
请严格按上述 Schema 输出唯一一个 JSON 对象。
"""

    def process_chunk(self, chunk_id: str, source_text: str, lit_trans: str, critic_report: dict, affected_chunks: List[str] = None) -> Dict[str, Any]:
        print(f"⚖️ [Judge Agent] 正在对 {chunk_id} 进行最终裁决...")
        prompt = self._build_prompt(source_text, lit_trans, critic_report)
        
        model_key, params = config.resolve_task_model("judge_decision")
        model_name = get_role_model_name(self.role) or model_key
        extra_body = get_role_extra_body(self.role)
        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
            extra_body=extra_body,
        )
        
        result = robust_json_loads(raw_output)
        if not result:
            # H7（v2）：Judge JSON 损坏 → 抛异常 → pipeline 置 FAILED → H6 恢复 JUDGING
            raise RuntimeError("Judge 输出 JSON 损坏，重新裁决。")

        scores = critic_report.get("scores", {})

        # P0: 安全网——仅在 LLM 判 PASS 但评分极低时覆盖，防止 LLM 幻觉放行劣质译文
        # 不同于旧版无条件覆盖（avg<7.5→REJECT），新版尊重 LLM 独立判断
        if scores:
            avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / len(scores)
            safety_min = config.pipeline.get("judge_safety_net_avg_min", 5.5)
            if result.get("decision") == "PASS" and avg_score < safety_min:
                result["decision"] = "REJECT"
                result["reject_reason"] = (f"安全网: Critic 平均分 {avg_score:.1f} 低于安全阈值 {safety_min}，"
                                           f"但 Judge 判定通过，疑似 LLM 幻觉。")

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