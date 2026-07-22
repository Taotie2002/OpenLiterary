import json
import re
from typing import List, Dict, Optional
from utils.llm_adapter import get_llm_client_for_task, get_role_model_name, get_role_extra_body
from core.decision_engine import DecisionEngine, DecisionLevel
from src.config import config


class LiteraryRewriterAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client_for_task("literary_rewrite")
        self.role = config.get_task_role("literary_rewrite")
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
2. P4 优化：对于卡罗尔《爱丽丝梦游仙境》这类儿童文学，**禁止在润色阶段添加脚注**。
3. 如遇需要注释的典故，应在首次出现时用文内括号简注（如：渡渡鸟（一种已灭绝的鸟）），而非使用脚注。
4. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。

【人称代词规则】
1. 严格遵循原文人称代词性别：he/him/his → "他"，she/her → "她"，it/its → "它"。
2. 注意原文中人物对话和叙述视角的代词指代关系，不得混淆角色性别。

【直译底稿】
{raw_translation}

请直接输出润色后的 Markdown 文本，不要包含任何多余的开头问候或解释：
"""

    def _build_retry_prompt(
        self,
        raw_translation: str,
        prev_lit_text: str,
        decisions_context: str,
        style_guide: str,
        reject_reason: str,
        critic_feedback: Optional[str],
        retry_count: int,
        all_feedback: Optional[List[Dict]] = None,
    ) -> str:
        critic_section = ""
        if critic_feedback:
            critic_section = f"\n【Critic 具体改进建议】\n{critic_feedback}\n"

        history_section = ""
        if all_feedback and len(all_feedback) > 1:
            lines = []
            for i, fb in enumerate(all_feedback[:-1]):
                reason_text = fb.get("reason", "")[:80]
                tag = "已修复" if fb.get("resolved", False) else "需持续关注"
                if reason_text.strip():
                    lines.append(f"  第 {i+1} 轮 ({tag}): {reason_text}")
            if lines:
                history_section = (
                    "\n【过去所有被拒原因 — 需确认全部已解决】\n"
                    + "\n".join(lines) + "\n"
                )

        if retry_count <= 2:
            strategy = (
                "请在保留上一版优点的基础上**针对性修改**，只动被拒原因相关的段落，"
                "不要推倒重来。"
            )
        elif retry_count <= 4:
            strategy = (
                "你可以**重写 1-2 个段落**来系统性修复被拒原因，"
                "其他部分保持稳定。注意保持整体风格的统一。"
            )
        else:
            strategy = (
                "本轮允许**基于直译底稿重新润色全部内容**，"
                "但必须避免自第一轮以来所有被拒原因列出的问题。"
            )

        return f"""你是一位荣获过星云奖和雨果奖的资深科幻/奇幻文学译者。
这是第 {retry_count + 1} 次修订。上一版译文因以下原因被终审拒绝。

{decisions_context}

{strategy}

【上一版被拒原因（必须解决）】
{reject_reason}{critic_section}
{history_section}

【风格基准 (Style Guide)】
{style_guide}

【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. P4 优化：对于卡罗尔《爱丽丝梦游仙境》这类儿童文学，**禁止在润色阶段添加脚注**。
3. 如遇需要注释的典故，应在首次出现时用文内括号简注（如：渡渡鸟（一种已灭绝的鸟）），而非使用脚注。
4. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。

【人称代词规则】
1. 严格遵循原文人称代词性别：he/him/his → "他"，she/her → "她"，it/its → "它"。
2. 注意原文中人物对话和叙述视角的代词指代关系，不得混淆角色性别。

【直译底稿（语义基准，不可偏离）】
{raw_translation}

【上一版译文（在此基础上精修，只动被拒原因相关段落）】
{prev_lit_text}

请直接输出修订后的 Markdown 文本，重点解决被拒原因，其他部分保持稳定：
"""

    def _calculate_retry_temperature(self, base_temperature: float, retry_count: int) -> float:
        if retry_count <= 2:
            return min(base_temperature + retry_count * 0.1, 0.8)
        elif retry_count <= 4:
            return min(base_temperature + 0.3, 0.9)
        else:
            return min(base_temperature + 0.5, 1.0)

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

        # 2. 诗歌/韵文特征（P3 优化：为卡罗尔诗歌添加更多标记）
        poetry_markers = ['\n\n', '——', '...', 'beauty is truth', 'truth beauty',
                         'How doth', 'How cheerful', 'I am older', 'Who am I',
                         'Curiouser', 'said the', 'replied the']
        poetry_score = sum(1 for m in poetry_markers if m in source_text)
        # 额外检测：换行符密度（诗歌通常有更多换行）
        newline_density = source_text.count('\n') / max(len(source_text) / 100, 1)
        if newline_density > 0.5:
            poetry_score += 1

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

    def process_chunk(
        self,
        chunk_id: str,
        raw_translation: str,
        style_guide_stats: dict,
        source_text: str = "",
        prev_lit_text: Optional[str] = None,
        reject_reason: Optional[str] = None,
        critic_feedback: Optional[str] = None,
        retry_count: int = 0,
        all_feedback: Optional[List[Dict]] = None,
    ):
        print(f"✍️ [Rewriter Agent] 正在进行文学润色: {chunk_id} (retry={retry_count})...")

        style_guide = self._build_style_guide(style_guide_stats, source_text)
        decisions_context = self._build_decision_context()

        # 使用配置路由：literary_rewrite（先于 temperature 计算，确保 base 从 config 读取）
        model_key, params = config.resolve_task_model("literary_rewrite")
        model_name = get_role_model_name(self.role) or model_key
        base_temp = params.get("temperature", 0.3)

        if retry_count > 0 and prev_lit_text:
            if not reject_reason:
                reject_reason = "终审未提供具体拒因，请结合历史反馈与风格基准进行全面审视与精修。"
            prompt = self._build_retry_prompt(
                raw_translation=raw_translation,
                prev_lit_text=prev_lit_text,
                decisions_context=decisions_context,
                style_guide=style_guide,
                reject_reason=reject_reason,
                critic_feedback=critic_feedback,
                retry_count=retry_count,
                all_feedback=all_feedback,
            )
            temperature = self._calculate_retry_temperature(base_temp, retry_count)
        else:
            prompt = self._build_prompt(raw_translation, decisions_context, style_guide)
            temperature = base_temp

        params = {**params, "temperature": temperature}
        extra_body = get_role_extra_body(self.role)

        final_markdown = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
            extra_body=extra_body,
        )

        return final_markdown