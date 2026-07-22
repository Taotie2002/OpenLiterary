import json
import re
from typing import Dict, Any
from utils.llm_adapter import get_llm_client_for_task, get_role_model_name, get_role_extra_body, robust_json_loads
from src.config import config

class CriticAgent:
    def __init__(self):
        self.llm = get_llm_client_for_task("critic_scoring")
        self.role = config.get_task_role("critic_scoring")

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

【JSON 输出规则 — 必须严格遵守】
1. 输出纯 JSON，禁止包含任何 Markdown 标记、代码块、或额外说明文字
2. 字符串值中的双引号 " 必须转义为 \"
3. 不得在数组/对象的最后一个元素后加逗号
4. 布尔值用 true/false（不加引号），数字用纯数字（不加引号）
5. ⚠️ 如果 JSON 格式错误，你的整份评语将被丢弃，本次审计作废

Schema 要求：
{{
  "scores": {{
    "fluency": 8,
    "readability": 8,
    "style_compliance": 7,
    "voice_consistency": 9,
    "semantic_preservation": 8
  }},
  "critique": "一段尖锐的综合评价",
  "improvement_suggestions": "针对低分项给出具体的修改建议（若无则留空）"
}}

请严格按上述 Schema 输出唯一一个 JSON 对象。
"""

    def process_chunk(self, chunk_id: str, source_text: str, raw_trans: str, lit_trans: str, style_guide: dict) -> Dict[str, Any]:
        print(f"🧐 [Critic Agent] 正在对 {chunk_id} 进行多维度文学审计...")
        prompt = self._build_prompt(source_text, raw_trans, lit_trans, style_guide)
        
        model_key, params = config.resolve_task_model("critic_scoring")
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
            print(f"⚠️ Critic Agent 输出异常")
            return {"critique": "JSON解析失败", "scores": {}}
        
        return result