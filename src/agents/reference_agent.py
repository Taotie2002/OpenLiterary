import json
import re
from typing import List, Dict, Any
from utils.llm_adapter import get_llm_client
from core.decision_engine import DecisionEngine, DecisionLevel
from src.config import config
from src.config import config

class ReferenceAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client()
        self.db = decision_engine
        
    def _build_prompt(self, text_chunk: str) -> str:
        """构建带有严格 JSON Schema 约束的系统提示词"""
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
        # 尝试剥离模型可能擅自添加的 ```json 标签
        cleaned = re.sub(r'^```json\s*', '', raw_response, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # 记录日志，实际生产中这里应抛出异常交由 Scheduler 触发重试
            print(f"⚠️ JSON 解析失败，模型输出格式违规: {e}")
            print(f"原始内容预览: {raw_response[:200]}...")
            return {"references": []}

    def process_chunk(self, chunk_id: str, text_chunk: str, affected_chunks: List[str] = None):
        """处理单个文本块，识别典故并自动写入决策引擎"""
        print(f"🔍 [Reference Agent] 正在考据数据块: {chunk_id}...")
        
        # 1. 调用 LLM
        prompt = self._build_prompt(text_chunk)
        
        model_key, params = config.resolve_task_model("reference_extraction")
        model_cfg = config._get_model_config(model_key)
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
                # 写入 Level 2: REFERENCE 决策，传入受影响的 chunk
                self.db.add_decision(
                    level=DecisionLevel.REFERENCE, 
                    source=source, 
                    translation=translation, 
                    reason=reason,
                    affected_chunks=affected_chunks or [chunk_id]
                )