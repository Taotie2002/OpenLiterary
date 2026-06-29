import json
import re
from typing import List, Dict
from utils.llm_adapter import get_llm_client
from core.decision_engine import DecisionEngine, DecisionLevel
from src.config import config
from src.config import config


class LiteraryRewriterAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client()
        self.db = decision_engine

    def _build_decision_context(self) -> str:
        """从 Decision DB 提取当前生效的宪法，转化为 Prompt 上下文"""
        decisions = self.db.get_all_decisions()
        if not decisions:
            return "无特殊词汇约束。"
            
        context = "【全局翻译决策（必须严格遵守）】\n"
        for level, source, trans in decisions:
            if level == DecisionLevel.TERMINOLOGY.value:
                context += f"- 术语: '{source}' -> 必须译为 '{trans}'\n"
            elif level == DecisionLevel.REFERENCE.value:
                context += f"- 典故: '{source}' -> 必须译为 '{trans}' (若策略为保留并加注，请生成脚注)\n"
            elif level == DecisionLevel.STYLE.value:
                context += f"- 风格: {source} -> {trans}\n"
        return context

    def _build_prompt(self, raw_translation: str, decisions_context: str, style_guide: str) -> str:
        """构建带有脚注排版协议的系统提示词"""
        return f"""你是一位荣获过星云奖和雨果奖的资深科幻/奇幻文学译者。
你的任务是对提供的【直译底稿】进行最高水准的文学润色。

{decisions_context}

【风格基准 (Style Guide)】
{style_guide}

【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. 当遇到需要加注的【典故】时，必须使用 Markdown 原生脚注语法。
3. 在正文中需要加注的词语后紧跟 `[^数字]`（如：拉米亚[^1]）。
4. 在你输出的全部正文**最末尾**，空两行，然后列出对应的脚注内容。脚注格式必须为：`[^数字]: 译注：[考据原因]`。

【直译底稿】
{raw_translation}

请直接输出润色后的 Markdown 文本，不要包含任何多余的开头问候或解释：
"""

    def _infer_author_priority_ratio(self, source_text: str) -> float:
        """
        动态推断 Author_Priority_Ratio (0-1)
        
        基于原文特征决定翻译时对原著风格的保留程度：
        - 1.0 = 完全保留作者风格（直译倾向）
        - 0.0 = 完全按译者风格重写（意译倾向）
        
        影响因子：
        1. 典故/引用密度：越高越倾向保留原文风格
        2. 诗歌/韵文比例：越高越倾向保留
        3. 专有名词密度：越高越倾向保留
        4. 叙事视角稳定性：不稳定时降低优先级
        5. 情感强度：高情感段落保留作者音色
        """
        # 基础比率
        base_ratio = 0.7
        
        # 1. 典故/引用检测（英文原文使用英文作者名）
        allusion_markers = ['Keats', 'Shakespeare', 'Milton', 'Dante', 'Homer', 'Shelley', 'Byron']
        allusion_count = sum(source_text.count(m) for m in allusion_markers)
        allusion_density = min(allusion_count / (len(source_text) / 1000), 1.0)  # 每千字典故数

        # 2. 诗歌/韵文特征
        poetry_markers = ['\n\n', '——', '...', 'beauty is truth', 'truth beauty']
        poetry_score = sum(1 for m in poetry_markers if m in source_text)

        # 3. 专有名词密度（大写单词、首字母大写）
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', source_text)
        proper_noun_density = min(len(proper_nouns) / (len(source_text) / 1000), 2.0)

        # 4. 叙事视角标记（英文原文使用英文代词）
        pov_markers = [' I ', ' my ', ' we ', ' he ', ' she ', ' it ']
        pov_changes = sum(source_text.count(m) for m in pov_markers)
        
        # 5. 情感词汇
        emotion_words = [' pain', ' sorrow', ' anger', ' joy', ' love', ' hate', ' fear', ' despair', ' hope', ' dream', ' soul', ' grief', ' rage']
        emotion_count = sum(source_text.count(w) for w in emotion_words)
        emotion_density = min(emotion_count / (len(source_text) / 1000), 1.0)
        
        # 动态调整
        # 典故密度高 -> 保留作者风格
        if allusion_density > 0.5:
            base_ratio += 0.15
        elif allusion_density > 0.2:
            base_ratio += 0.05
        
        # 诗歌特征 -> 保留作者韵律
        if poetry_score > 0:
            base_ratio += 0.1
        
        # 专有名词密 -> 保留术语准确性
        if proper_noun_density > 1.0:
            base_ratio += 0.05
         # 视角不稳定 -> 降低作者优先级，由译者统一声音
        if pov_changes > 20:
            base_ratio -= 0.1
        
        # 高情感 -> 保留作者音色
        if emotion_density > 0.5:
            base_ratio += 0.1
        
        # 限制在 [0.3, 0.9] 区间
        return max(0.3, min(0.9, base_ratio))

    def _build_style_guide(self, style_guide_stats: dict, source_text: str = "") -> str:
        """构建风格指南，包含动态 Author_Priority_Ratio"""
        # 动态推断优先级比率
        if source_text:
            author_priority = self._infer_author_priority_ratio(source_text)
        else:
            author_priority = style_guide_stats.get('author_priority_ratio', 0.7)
        
        # 根据优先级生成差异化指令
        if author_priority >= 0.8:
            priority_instruction = (
                "【高作者优先级模式】严格保留原著句式节奏、修辞手法、词汇选择。"
                "即使中文表达略显生硬，也要优先还原作者的原意与音色。"
            )
        elif author_priority >= 0.6:
            priority_instruction = (
                "【平衡模式】在保持作者核心风格的前提下，适度调整句式使其符合中文阅读习惯。"
                "保留关键修辞、典故处理，非核心处可本地化。"
            )
        else:
            priority_instruction = (
                "【译者主导模式】以目标语言的自然流畅为首要目标。"
                "大胆重组句式、替换词汇，只保留核心语义与关键意象。"
            )
        
        style_guide = (
            f"请模仿以下风格特征：平均句长偏向 {style_guide_stats.get('avg_sentence_length', '中等')}，"
            f"词汇倾向 {style_guide_stats.get('lexicon_preference', '文学化')}。\n"
            f"作者优先级比率: {author_priority:.2f} (0=译者主导, 1=作者主导)\n"
            f"{priority_instruction}"
        )
        return style_guide

    def process_chunk(self, chunk_id: str, raw_translation: str, style_guide_stats: dict, source_text: str = ""):
        """执行润色并生成最终排版"""
        print(f"✍️ [Rewriter Agent] 正在进行文学润色: {chunk_id}...")
        
        # 动态构建风格指南
        style_guide = self._build_style_guide(style_guide_stats, source_text)
        
        decisions_context = self._build_decision_context()
        prompt = self._build_prompt(raw_translation, decisions_context, style_guide)
        
        # 润色是决定最终质量的关键，必须使用能力最强的模型
        model_key, params = config.resolve_task_model("literary_rewrite")
        model_cfg = config._get_model_config(model_key)
        model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "qwen/Qwen2.5-7B-Instruct-MLX-4bit")
        final_markdown = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params
        )
        
        return final_markdown