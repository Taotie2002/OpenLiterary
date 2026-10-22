"""
LiteraryRewriterAgent — 文学润色 Agent

源：原整合脚本 §6 中的 LiteraryRewriterAgent 类。
职责：
  1. 依据风格指南（genre / work_type / persona / register_default）重构译文
  2. 注入 Decision DB 中的术语、典故、风格约束
  3. Few-shot 示例注入（来自 docs/few_shot_examples.json）
  4. 章节级 register 方向推断（N 方案）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import config as _config
from src.utils.text_match import extract_reject_categories
from src.utils.llm_adapter import (
    get_llm_client_for_task,
    get_llm_client,
    get_role_model_name,
    get_role_extra_body,
    robust_json_loads,
)
from src.core.decision_engine import DecisionEngine


class LiteraryRewriterAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client_for_task("literary_rewrite")
        self.role = _config.get_task_role("literary_rewrite")
        self.db = decision_engine
        # 加载 Few-shot 示例
        self._few_shot_examples = self._load_few_shot_examples()

    def _build_decision_context(self, source_text: str = "") -> str:
        """有界决策上下文（委托 DecisionEngine.format_prompt_context）。"""
        return self.db.format_prompt_context(source_text=source_text)

    # Round 38: 一次性告警 set（避免每次 retry 都打同样的 few-shot 缺失警告）
    _few_shot_warned_paths: set = set()

    def _load_few_shot_examples(self) -> str:
        """加载 Few-shot 示例，注入 Prompt 作为参考"""
        try:
            root_dir = Path(__file__).resolve().parent.parent
            # depth-independent：整合脚本在 src/ 下两层就到根，散装 src/agents/*.py 要三层
            # 探测 config.yaml 向上找
            if not (root_dir / "config.yaml").exists():
                root_dir = root_dir.parent
            few_shot_file = root_dir / "docs" / "few_shot_examples.json"
            with open(few_shot_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            examples = data.get("few_shot_examples", [])
            if not examples:
                return ""

            parts = ["【Few-shot 示例：以下为优秀润色范例，请模仿其风格、脚注格式、代词处理】\n"]
            for ex in examples[:4]:  # 最多注入 4 个示例
                parts.append(f"【示例 {ex['id']}】")
                parts.append(f"【直译底稿】\n{ex['raw_translation']}\n")
                parts.append(f"【优秀润色】\n{ex['literary_translation']}\n")
                if "edits" in ex and ex["edits"]:
                    parts.append("【修改要点】")
                    for edit in ex["edits"]:
                        parts.append(f"  - {edit['location']}: \"{edit['before']}\" -> \"{edit['after']}\" ({edit['reason']})")
                parts.append("")  # 空行分隔

            parts.append("———\n参考上述示例的脚注格式、术语一致性与人称代词处理即可；文风须以本任务的【风格指南】为准，不得套用示例的文言/史诗体。\n")
            return "\n".join(parts)
        except Exception:
            # Round 38: 一次性告警（同一个 path 只打一次），减少跑批噪音
            path_key = str(few_shot_file)
            if path_key not in self._few_shot_warned_paths:
                self._few_shot_warned_paths.add(path_key)
                print(f"  ⚠️ [Rewriter] 加载 few-shot 示例失败: {few_shot_file} 不存在或格式错误（后续同类警告将省略）", file=sys.stderr)
            return ""

    # Round 34: thin shim — delegates to src.utils.text_match.extract_reject_categories
    # Round 89: 增加 normalize 参数 (默认 True 保持向后兼容). 策略查找路径显式传 False.
    @staticmethod
    def _extract_reject_categories(reason_text: str, normalize: bool = True) -> list:
        return extract_reject_categories(reason_text, as_set=False, normalize=normalize)

    _JEV_CATEGORY_STRATEGY: dict = {
        "直译残留": "本轮必须消除译稿中的英文句式套用与逐词堆砌，以自然中文表达为目标整体重写。",
        "关键语义丢失": "保留原文所有关键意象、动作细节、隐喻、反讽，不得概括或弱化。",
        "音色漂移（severe）": "本轮保留原文的叙事口吻/角色音色，包括讽刺、克制、幽默等特征，避免被抹平为平淡中性叙述。",
        "音色漂移（subtle）": "轻微调整使叙事口吻更接近原文，避免过度风格化。",
        # Round 85: Round 82 新增 subtle+其它 escalation 类别专用策略.
        # 单独 subtle 不 reject (Round 82); 仅在叠加其它 reject 时升级.
        # Rewriter 看到此类别时, 已知 subtle 单独不够, 其它维度也已 reject,
        # 需要在解决其它核心问题的同时一并把 voice drift 拉回正轨.
        "音色漂移（subtle+其它）": (
            "本轮 Jev 判定 subtle voice drift 与其它问题叠加 (例如 实体一致性 / 直译残留 / "
            "关键语义丢失) 共同触发 reject. 请在解决其它维度的同时, 把叙事口吻向原文基线 "
            "靠拢 (保留讽刺/克制/幽默等特征), 避免被抹平为平淡中性叙述 —— 但不要过度风格化 "
            "(否则会反向触发 severe voice drift)."
        ),
        "语域偏离（under）": "提升语域，减少口语/方言化表达，匹配目标读者期望。",
        "语域偏离（over）": "降低语域，减少文言/书面化表达，避免距离感过强。",
        "整体风格偏离": "全篇风格重润色，匹配目标风格基线。",
        # Round 107 P5: 风格契合度低 — Jev Score primitive, 全局风格问题
        "风格契合度低": "全篇风格重润色：以风格基准为锚点，重新组织断句节奏、词汇密度、修辞与语域，使译文风格与目标基线对齐。局部词换无法解决全局风格偏离。",
    }

    def _build_jev_category_section(self, reject_reason: str) -> str:
        """Round 23: 在 retry prompt 顶部渲染 Jev 判定的核心问题维度 + 对应策略。

        Round 89: 策略查找路径必须用 normalize=False, 否则 `_CATEGORY_SYNONYMS`
        把 `音色漂移（severe）` 归一为 `音色漂移`, 导致 lookup miss, 落到 generic
        fallback "请针对此维度整体审视并修订。" (失去 specific 策略).
        """
        cats = self._extract_reject_categories(reject_reason, normalize=False)
        if not cats:
            return ""
        lines = []
        for c in cats:
            strategy = self._JEV_CATEGORY_STRATEGY.get(c)
            if strategy:
                lines.append(f"  - {c}：{strategy}")
            elif c == "实体一致性违反":
                lines.append(f"  - {c}：详见下方【实体一致性反馈】清单。")
            else:
                lines.append(f"  - {c}：请针对此维度整体审视并修订。")
        body = "\n".join(lines)
        return (
            "\n【⚠️ 本轮核心问题维度（Jev/primitives 判定 — 必须整体性处理）】\n"
            "本轮 critic 触发了以下 Jev primitives 维度：\n"
            f"{body}\n"
            "提示：上述每一项都应在**整体重写层面**解决（而非局部词换）。"
            "请把每一维度当作「必须 100% 解决」的硬约束。\n"
        )

    def _build_prompt(
        self,
        raw_translation: str,
        decisions_context: str,
        style_guide: str,
        narrative_memory: str = "",
        entity_registry: Optional[Dict[str, str]] = None,  # 回路修复
        verse_note: Optional[str] = None,  # Round 119 L1 韵文协议
    ) -> str:
        few_shot = self._few_shot_examples
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        mem_block = ""
        if narrative_memory:
            mem_block = f"\n【前文摘要（保持声口与指代连贯）】\n{narrative_memory}\n"
        persona = (_config.style_guide or {}).get("persona", "职业文学译者")
        footnote_policy = (_config.style_guide or {}).get("footnote_policy", "inline")

        footnote_rules = {
            "none": "禁止在润色阶段添加脚注。如遇需要注释的典故，用文内括号简注（如：渡渡鸟（一种已灭绝的鸟））。",
            "inline": "如遇需要注释的典故，应在首次出现时用文内括号简注，而非使用脚注。",
            "markdown_footnote": "如遇需要注释的典故，可使用 Markdown 脚注语法 [^1]。",
        }
        footnote_instruction = footnote_rules.get(footnote_policy, footnote_rules["inline"])

        # 回路修复：构建实体译名登记表 section（按 canonical 分组，源文别名放括号里）
        # 这样 Rewriter 看到 canonical 优先 + 所有英文别名，能推断 "George Lomax" 也属于 "乔治·洛马克斯先生"
        entity_registry_section = ""
        if entity_registry:
            # 按 canonical 分组：{canonical: [source_name1, ...]}
            grouped: Dict[str, List[str]] = {}
            for src_name, zh_canonical in entity_registry.items():
                if src_name and zh_canonical:
                    grouped.setdefault(zh_canonical, []).append(src_name)
            if grouped:
                lines = []
                for canonical, aliases in sorted(grouped.items()):
                    alias_str = " / ".join(aliases)
                    lines.append(f"  - {canonical}（源文中可能的形式：{alias_str}）")
                entity_registry_section = (
                    "\n【实体译名登记表（首次润色参考 — 严格遵循）】\n"
                    "译稿中的人名/地名/组织名必须使用下列规范译名。"
                    "源文中如出现任一英文别名（含部分匹配），都应翻译为对应中文译名。\n"
                    "【优先级规则】实体译名若与上方考据/术语决策的写法不一致，"
                    "**以本登记表规范主名为准**（统一为规范主名，不得保留考据决策中的"
                    "异体/简称/缺后缀形式）。\n"
                    + "\n".join(lines)
                    + "\n"
                )

        # Round 54: 把 config.yaml 的 forbidden_modernisms 注入 Rewriter prompt（修复 dead config）
        forbidden_modernisms = (_config.style_guide or {}).get("forbidden_modernisms", []) or []
        forbidden_section = ""
        if forbidden_modernisms:
            forbidden_str = "、".join(str(w) for w in forbidden_modernisms)
            forbidden_section = (
                "\n【禁用词列表 (CRITICAL — 译文将被自动校验)】\n"
                f"以下表达严禁出现在译文中（含其变体形式）：{forbidden_str}\n"
                "如出现即视为违背风格指南，无论文学质量如何都会被 reject。\n"
            )

        # Round 10: 实体登记表提到首位（最显眼位置）+ MUST 语言 + self-check 指令
        # 原因：Round 9 审计暴露 ch02_chunk000 漏译"绅士乔"等——LLM 即使收到 entity_registry 也常忽略
        # 修复策略：(1) 把登记 section 放在 prompt 顶部，避免被后续段落淹没
        #          (2) 用 MUST/否则被拒 等强约束语言
        #          (3) 在 output 前要求 self-check，确保所有登记实体都已呈现
        # Round 119 L1: 韵文 chunk 注入逐行对应/复沓保留协议
        verse_section = verse_note if verse_note else ""
        return f"""你是一位{persona}。
你的任务是对提供的【直译底稿】进行最高水准的文学润色。

{entity_registry_section}{forbidden_section}
{decisions_context}
{mem_block}
{few_shot}

【风格基准 (Style Guide)】
{style_guide}

【⚠️ 实体译名一致性 (CRITICAL — 译文将被自动校验)】
完成润色后必须 self-check：源文中出现的每个登记实体，译文中都必须能找到对应的规范中文译名。
如有遗漏（如 "Codders" 漏译），即使文学质量再高也会被 reject。


{verse_section}【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. {footnote_instruction}
3. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。
4. 脚注只陈述典故/文化背景/双关机制本身；不得出现任何翻译流程元语（如"保留与加注策略""保留并加注""保留音译""按约束""直译自""策略池"等字样），不得描述译者的处理方式或决策理由。
5. 原文里的方括号编辑标记（如 [later editions continued as follows]）必须保留并译出，用【】包裹（如【后来版本续之如下：】），不得删除或改写成叙述。
6. 注释一律使用 Markdown 脚注 [^n] 并带正文引标；不得新增"【章末注释】""【脚注】"等标题式注释块，不得输出无引标的裸编号注列表（如 [1] xxx 列表）。

【翻译约束 (必须同时遵守)】
1. 不得丢弃原文的比喻/意象 (除非确实不可译)
2. 不得添加原文没有的评价性词语或从句
3. 保持原文信息量，不要"锦上添花"

【语气标签校准】
- 台词前的副词 (coquettishly/angrily/softly) 作为语域校准强约束信号
- 匹配副词的情感色彩到译文语气
- 例如: coquettishly → 娇嗔/撒娇的语气, angrily → 生气的语气, softly → 轻柔的语气

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
        # 回路修复：新增 entity_feedback + entity_registry 参数（带默认值，需放在 retry_count 之后）
        entity_feedback: Optional[str] = None,
        entity_registry: Optional[Dict[str, str]] = None,
        all_feedback: Optional[List[Dict]] = None,
        specific_edits: Optional[List[Dict]] = None,
        register_target: str = "",
        low_dims: Optional[List[str]] = None,
        # Round 64: 上轮被 Jev uncertain zone 跳过的维度，避免对边界判断维度做无效重写
        skipped_dimensions: Optional[List[str]] = None,
        verse_note: Optional[str] = None,  # Round 119 L1 韵文协议
    ) -> str:
        # Round 10: 重试场景下也强化 entity section 的位置 + MUST 语言
        # 重试时 entity_feedback 已具体列出问题项，本轮必须 100% 修复
        # 实体反馈 section（最显眼位置，prompt 顶部）
        entity_feedback_section = ""
        if entity_feedback:
            # Round 29: 在 entity_feedback 顶部插入"输出前全文自检"指令
            # —— 解决"rewriter 改了一处 variant 但另一处 source_name 残留"导致下一轮又失败的循环
            self_check_preamble = (
                "\n【🔍 输出前必做：实体一致性全文自检】\n"
                "在你输出最终润色稿之前，必须按以下步骤全文自检：\n"
                "  1. 把【直译底稿】与下方【实体一致性反馈】清单中的每一项逐条对照；\n"
                "  2. 把你**准备输出**的整篇润色稿全文扫一遍，标记任何残留的英文 source_name 或中文 variant；\n"
                "  3. 把所有残留替换为清单中标注的 expected 规范主名；\n"
                "  4. 若清单中标注「missing」的实体原文中出现，**确保译稿中存在对应规范主名**。\n"
                "任何一处残留都会让本次任务再次被 reject。\n"
            )
            entity_feedback_section = (
                self_check_preamble
                + "\n【⚠️ 实体一致性反馈 (CRITICAL — 上次被拒原因，必须逐项修复)】\n"
                + f"{entity_feedback}\n"
                + "请在本次修订中**严格按上述清单**逐项修正译稿中的实体译名，"
                "尤其是清单中标注「missing」的项，必须在译稿中补回；标注「variant_instead_of_canonical」的项，必须替换为主名。\n"
            )

        # Round 23: 从 reject_reason 中解析 Jev reject categories，渲染顶部"硬约束维度"section
        # —— 让 rewriter 拿到与 entity_feedback 同级的 explicit 维度指引
        jev_categories_section = self._build_jev_category_section(reject_reason)

        # Round 64: 渲染"边界判断跳过"section —— 让 rewriter 知道哪些维度是 Jev 边界判断
        # （Noul ∈ [0.3, 0.7] 或 Choice/Score confidence < 0.5），未达 reject 门槛。
        # 本轮不需要重点修改这些维度；如确实需要修改，请基于整体润色判断而非局部修复。
        skipped_section = ""
        if skipped_dimensions:
            skipped_section = (
                "\n【ℹ️ 上轮 Jev 边界判断跳过（无需重点修改这些维度）】\n"
                "Jev 在以下维度给出边界判断（noul ∈ [0.3, 0.7] 或 confidence < 0.5），"
                "未达 reject 门槛。本轮不需要重点修改这些维度；"
                "如确实需要修改，请基于整体润色判断而非局部修复。\n"
                f"  跳过维度：{', '.join(skipped_dimensions)}\n"
            )

        # 实体登记表 section（紧跟 entity_feedback，方便 Rewriter 查阅）
        entity_registry_section = ""
        if entity_registry:
            # 按 canonical 分组（与 _build_prompt 同语义）
            grouped: Dict[str, List[str]] = {}
            for src_name, zh_canonical in entity_registry.items():
                if src_name and zh_canonical:
                    grouped.setdefault(zh_canonical, []).append(src_name)
            if grouped:
                lines = []
                for canonical, aliases in sorted(grouped.items()):
                    alias_str = " / ".join(aliases)
                    lines.append(f"  - {canonical}（源文中可能的形式：{alias_str}）")
                entity_registry_section = (
                    "\n【实体译名登记表（修订时参考 — 严格遵循）】\n"
                    "译稿中的人名/地名/组织名必须使用下列规范译名。"
                    "源文中如出现任一英文别名（含部分匹配），都应翻译为对应中文译名。\n"
                    "【优先级规则】实体译名若与上方考据/术语决策的写法不一致，"
                    "**以本登记表规范主名为准**（统一为规范主名，不得保留考据决策中的"
                    "异体/简称/缺后缀形式）。\n"
                    + "\n".join(lines)
                    + "\n"
                )

        # Round 56: 把 forbidden_modernisms 注入 retry prompt（Round 54 漏改 — retry 时约束也必须保持）
        forbidden_modernisms = (_config.style_guide or {}).get("forbidden_modernisms", []) or []
        forbidden_section = ""
        if forbidden_modernisms:
            forbidden_str = "、".join(str(w) for w in forbidden_modernisms)
            forbidden_section = (
                "\n【禁用词列表 (CRITICAL — 修订时同样适用)】\n"
                f"以下表达严禁出现在译文中（含其变体形式）：{forbidden_str}\n"
                "修订时若发现上一版有上述任何表达，必须替换为符合风格指南的替代表达。\n"
            )

        critic_section = ""
        if critic_feedback:
            # L 方案：将改进建议提升为"必执行顶层指令"，避免 rewriter 只盯 specific_edits 忽略结构性抱怨
            critic_section = (
                "\n【Critic 总体改进建议（必须执行，非可选参考）】\n"
                f"{critic_feedback}\n"
                "以上建议为终审拒绝的核心原因，必须在本次修订中**整体落实**，"
                "尤其涉及语域/文言/四字格/脚注密度的全局性问题须全篇处理。\n"
            )

        specific_edits_section = ""
        if specific_edits:
            edits_lines = []
            for i, edit in enumerate(specific_edits, 1):
                loc = edit.get("location", "?")
                orig = (edit.get("original", "") or "")[:30]
                issue = edit.get("issue", "")
                direction = (edit.get("suggested_direction", "") or "")[:50]
                edits_lines.append(f"  {i}. [{loc}] \"{orig}\" → {issue}: {direction}")
            specific_edits_section = (
                "\n【Critic 结构化修改指令（必须逐条处理，不得只改一处）】\n"
                + "\n".join(edits_lines) + "\n\n"
                # M 方案：强制逐条回应全部 specific_edits，禁止"只改一处"的打勾式空转
                "【强制要求】你必须在这一轮中**逐条处理以上全部 "
                f"{len(specific_edits)} 条修改指令**，不得只挑选其中 1 条。"
                "输出润色稿之后，另起一行用 ```json 包裹输出每条的处理结果：\n"
                '```json\n'
                '[{"index": 1, "status": "resolved|partial|skipped", "reason": "..."}, '
                '{"index": 2, "status": "resolved|partial|skipped", "reason": "..."}]\n'
                '```\n'
                f"共 {len(specific_edits)} 条，每条对应一个 index，缺一不可。\n"
            )

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

        # L 方案：当风格/音色维度持续不达标，强制"全局重润色"而非局部词换
        # Round 24 扩展：Jev reject categories（直译残留 / 关键语义丢失 / 语域偏离 / 整体风格偏离）
        #   也作为整体重润色的早触发条件——这些维度本质是"全篇级问题"，局部词换永远无法解决，
        #   而 max_quality_retries=4 意味着"重写 1-2 段落"分支跑 4 次后会直接 fallback，
        #   等不到 retry_count > 4 的整体重润色分支。
        # Round 28 扩展：entity_consistency 也触发整体重写——实体译名是全篇级约束，
        #   局部词换一个 variant 时可能让另一个 variant 残留，导致下一轮又触发新的不一致。
        style_dims = {"style_compliance", "voice_consistency"}
        low_style = [d for d in (low_dims or []) if d in style_dims]
        entity_low = "entity_consistency" in (low_dims or [])

        # Round 24: Jev primitive categories → 整体重润色策略
        # Round 95: 补齐 3 个 voice_drift 类别中可触发整体重写的两条:
        #   - 音色漂移（severe）: 严重漂移, 单点词换无法解决, 必须整体重写
        #   - 音色漂移（subtle+其它）: 与其它维度叠加, 必须一并整体修正
        # 音色漂移（subtle）单独不 reject (Round 82 设计), 不放入 GLOBAL.
        GLOBAL_REWRITE_CATEGORIES = {
            "直译残留": "本轮必须**基于直译底稿整体重写**：消除译稿中的英文句式套用与逐词堆砌，逐段重组成自然中文表达。局部词换无法解决直译残留问题。",
            "关键语义丢失": "本轮必须**整体重写**以补回原文所有关键意象、动作细节、隐喻、反讽；不得概括或弱化任何一项。局部词换无法挽回已经丢失的语义。",
            "整体风格偏离": "本轮必须**整体重润色**：以目标风格基线为锚点，重新组织全篇的断句、节奏、语域与脚注。",
            "语域偏离（under）": "本轮必须**整体重写**以提升语域：减少口语/方言化表达，逐段匹配目标读者期望的语域。",
            "语域偏离（over）": "本轮必须**整体重写**以降低语域：减少文言/书面化表达，逐段匹配目标读者期望的语域。",
            "音色漂移（severe）": (
                "本轮必须**整体重写**以拉回叙事口吻与角色音色: 保留讽刺/克制/幽默等特征,"
                "避免被抹平为平淡中性叙述. 局部词换无法解决严重音色漂移——"
                "必须从句子节奏、句式选择、叙事视角整体重做."
            ),
            "音色漂移（subtle+其它）": (
                "本轮必须**整体重写**: subtle voice drift 与其它维度 (实体一致性 / 直译残留 / "
                "关键语义丢失) 共同 reject, 需在解决其它核心问题的同时, 把叙事口吻向原文基线 "
                "靠拢 (保留讽刺/克制/幽默), 又不能过度风格化 (避免反向触发 severe)."
            ),
            # Round 107 P5: 风格契合度低 — 全局风格问题, 局部词换无法解决, 必须整体重润色
            "风格契合度低": (
                "本轮必须**整体重润色**以拉高风格契合度: 以【风格基准】为锚点, "
                "重新组织全篇的断句节奏、词汇密度、修辞手法与语域, 使译文风格与目标基线对齐. "
                "局部词换无法解决全局风格偏离——必须从句子结构、段落节奏、用词选择整体重做."
            ),
        }
        # Round 89: GLOBAL_REWRITE_CATEGORIES 的 keys 是 `语域偏离（under）` 等具体形式,
        # 策略查找路径用 normalize=False 保留具体形式.
        jev_cats = self._extract_reject_categories(reject_reason, normalize=False)
        jev_global = [c for c in jev_cats if c in GLOBAL_REWRITE_CATEGORIES]

        # Round 28: entity-specific 整体重写策略
        ENTITY_GLOBAL_STRATEGY = (
            "本轮必须**基于直译底稿整体重写**，且以【实体译名登记表】为最高优先级硬约束。"
            "清单已逐项列出每处违规（含源文形式与应替换为的规范主名）。"
            "**一次重写内**把所有违规实体译名统一为登记表规范主名，**不得**保留任何 variant / source_name 残留形式。"
            "局部词换无法解决实体一致性问题——一处修改可能让另一处 variant 残留。"
        )

        if (low_style and retry_count >= 2) or (entity_low and retry_count >= 1) or (jev_global and retry_count >= 1):
            # Round 24 + Round 28: 选择最具体的策略（优先级：Jev category > entity_consistency > style/voice）
            if jev_global:
                strategy = GLOBAL_REWRITE_CATEGORIES[jev_global[0]]
            elif entity_low:
                strategy = ENTITY_GLOBAL_STRATEGY
            else:
                strategy = (
                    "本轮回退到**基于直译底稿的整体重润色**：必须以【本章目标语域】为锚点，"
                    "重新组织全篇的断句、节奏、语域与脚注，而非仅替换个别词语。"
                    "上一版被拒的根因是全局风格/音色偏离，单点词换无法解决，必须结构性重做。"
                )
        elif retry_count <= 2:
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

        register_block = ""
        if register_target:
            register_block = (
                f"【本章目标语域（必须遵循，重写时以此为准）】\n{register_target}\n"
            )

        persona = (_config.style_guide or {}).get("persona", "职业文学译者")
        footnote_policy = (_config.style_guide or {}).get("footnote_policy", "inline")

        footnote_rules = {
            "none": "禁止在润色阶段添加脚注。如遇需要注释的典故，用文内括号简注。",
            "inline": "如遇需要注释的典故，应在首次出现时用文内括号简注，而非使用脚注。",
            "markdown_footnote": "如遇需要注释的典故，可使用 Markdown 脚注语法 [^1]。",
        }
        footnote_instruction = footnote_rules.get(footnote_policy, footnote_rules["inline"])

        register_block = ""
        if register_target:
            register_block = (
                f"【本章目标语域（必须遵循，重写时以此为准）】\n{register_target}\n"
            )

        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        # Round 119 L1: 韵文 chunk 注入逐行对应/复沓保留协议（retry 轮同样携带）
        verse_section = verse_note if verse_note else ""
        return f"""你是一位{persona}。
这是第 {retry_count + 1} 次修订。上一版译文因以下原因被终审拒绝。

{decisions_context}

{strategy}

{register_block}

【上一版被拒原因（必须解决）】
{reject_reason}{critic_section}
{jev_categories_section}
{skipped_section}
{entity_feedback_section}
{history_section}
{specific_edits_section}

【实体译名登记表（修订时参考 — 严格遵循）】
{entity_registry_section}{forbidden_section}

【风格基准 (Style Guide)】
{style_guide}

{verse_section}【排版与脚注协议 (CRITICAL)】
1. 严禁改变 Markdown 的物理段落结构。
2. {footnote_instruction}
3. 脚注仅限于参考提取阶段（Reference Agent）生成的必要考据注释，润色阶段不得新增。
4. 脚注只陈述典故/文化背景/双关机制本身；不得出现任何翻译流程元语（如"保留与加注策略""保留并加注""保留音译""按约束""直译自""策略池"等字样），不得描述译者的处理方式或决策理由。
5. 原文里的方括号编辑标记（如 [later editions continued as follows]）必须保留并译出，用【】包裹（如【后来版本续之如下：】），不得删除或改写成叙述。
6. 注释一律使用 Markdown 脚注 [^n] 并带正文引标；不得新增"【章末注释】""【脚注】"等标题式注释块，不得输出无引标的裸编号注列表（如 [1] xxx 列表）。

【人称代词规则】
1. 严格遵循原文人称代词性别：he/him/his → "他"，she/her → "她"，it/its → "它"。
2. 注意原文中人物对话和叙述视角的代词指代关系，不得混淆角色性别。

【直译底稿（语义基准，不可偏离）】
{raw_translation}

【上一版译文（在此基础上精修，只动被拒原因相关段落）】
{prev_lit_text}

请直接输出修订后的 Markdown 文本，重点解决被拒原因，其他部分保持稳定。

【CRITICAL — 必须修改】
你必须对上一版译文做出**至少一处实质性修改**（改写句子、调整语序、替换用词、补充细节等）。
如果输出与上一版完全相同（字节级一致），本次任务将直接判定失败（REWRITER_STAGNATED）。
请确保你的输出与【上一版译文】有可见差异。
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
        base_ratio = 0.7

        # 1. 典故/引用检测（英文原文使用英文作者名）
        allusion_markers = ['Keats', 'Shakespeare', 'Milton', 'Dante', 'Homer', 'Shelley', 'Byron']
        allusion_count = sum(source_text.count(m) for m in allusion_markers)
        allusion_density = min(allusion_count / (len(source_text) / 1000), 1.0)

        # 2. 诗歌/韵文特征（P1-4: 只保留作品无关的结构性标记；作品特定诗句样例已移除）
        poetry_markers = ['\n\n', '——', '...']
        poetry_score = sum(1 for m in poetry_markers if m in source_text)
        # 额外检测：换行符密度（诗歌通常有更多换行）
        newline_density = source_text.count('\n') / max(len(source_text) / 100, 1)
        if newline_density > 0.5:
            poetry_score += 1

        # 3. 专有名词密度
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
        if allusion_density > 0.5:
            base_ratio += 0.15
        elif allusion_density > 0.2:
            base_ratio += 0.05

        if poetry_score > 0:
            base_ratio += 0.1

        if proper_noun_density > 1.0:
            base_ratio += 0.05

        if pov_changes > 20:
            base_ratio -= 0.1

        if emotion_density > 0.5:
            base_ratio += 0.1

        return max(0.3, min(0.9, base_ratio))

    def _infer_register_target(self, source_text: str) -> str:
        """
        推断本章目标语域方向（N 方案：章节级 register 信号）。

        用于同时指导首轮润色与 retry：明确告诉 Rewriter 本章应"加文学度"
        还是"减文学度"，避免 ch02/13（欠润色）与 ch04 等（过润色/文言）拿
        到同质指令后随机摆动、永不收敛。

        判定逻辑：检测原文中"需要克制平实"的信号（科学论述、对话自述、
        维多利亚口语标记）与"可适度文学化"的信号（强意象/诗意描写）。
        二者平衡时给出中性指令。
        """
        register_default = (_config.style_guide or {}).get("register_default", "保持克制、平实、贴近原文的叙事语域")
        if not source_text:
            return register_default
        low = 0.0
        high = 0.0
        plain_markers = ['I suppose', 'I think', 'you see', 'of course', 'rather',
                         'however', 'perhaps', 'I mean', 'you know', 'the fact is',
                         'it seems', 'I confess', 'we were', 'I was']
        low += sum(source_text.count(m) for m in plain_markers)
        literary_markers = ['glowing', 'shimmering', 'gleamed', 'silence', 'vast',
                            'shadow', 'wonder', 'marvellous', 'strange beauty',
                            'soft', 'dim', 'ghost']
        high += sum(source_text.count(m) for m in literary_markers)
        if source_text.count('"') > 6:
            low += 1.5
        if high > low + 1.5:
            return ("本章原文意象/描写密度高，可适度文学化，但仍须以克制语域为底；"
                    "禁止文言虚词（之/乃/遂/盖）、四字格堆砌与超过 3 条脚注。")
        if low >= high + 1.0:
            return ("本章为叙述者口语化自述/科学论述，目标语域=克制平实、贴近原文的闲谈；"
                    "在直译底稿基础上做文学润色（断句、节奏、语气词），但不得退化为直译稿，"
                    "也不得文言化；脚注不超过 3 条。")
        return register_default

    def _build_style_guide(self, style_guide_stats: dict, source_text: str = "") -> str:
        """构建风格指南，包含动态 Author_Priority_Ratio 与章节级 register 目标（N 方案）"""
        if source_text:
            author_priority = self._infer_author_priority_ratio(source_text)
        else:
            author_priority = style_guide_stats.get('author_priority_ratio', 0.7)
        register_target = self._infer_register_target(source_text) if source_text else \
            (_config.style_guide or {}).get("register_default", "保持克制、平实、贴近原文的叙事语域")

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
            f"{priority_instruction}\n"
            f"【本章目标语域（必须遵循）】{register_target}"
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
        # Rewriter↔Entity 回路修复：新增 2 个参数
        entity_feedback: Optional[str] = None,  # 上轮实体的具体反馈（含「X 应统一为 Y」清单）
        entity_registry: Optional[Dict[str, str]] = None,  # 实体译名登记表 {english_source: chinese_canonical}
        retry_count: int = 0,
        all_feedback: Optional[List[Dict]] = None,
        specific_edits: Optional[List[Dict]] = None,
        narrative_memory: str = "",
        low_dims: Optional[List[str]] = None,
        # Round 64: 让 rewriter 感知被 Jev uncertain zone 跳过的维度
        skipped_dimensions: Optional[List[str]] = None,
        verse_note: Optional[str] = None,  # Round 119 L1 韵文协议
        model_role: Optional[str] = None,  # Round 116: 停滞换模型覆盖角色
    ):
        print(f"✍️ [Rewriter Agent] 正在进行文学润色: {chunk_id} (retry={retry_count})...")

        style_guide = self._build_style_guide(style_guide_stats, source_text)
        decisions_context = self._build_decision_context(source_text)
        # N 方案：章节级 register 目标，作为 retry 的方向锚点
        register_target = self._infer_register_target(source_text) if source_text else ""

        # Round 116: 停滞换模型 — 本次调用可覆盖角色（客户端/模型名/extra_body 随之切换）
        llm = self.llm
        role = self.role
        if model_role and model_role != role:
            role = model_role
            llm = get_llm_client(role)

        # 使用配置路由：literary_rewrite（先于 temperature 计算，确保 base 从 config 读取）
        model_key, params = _config.resolve_task_model("literary_rewrite")
        model_name = get_role_model_name(role) or model_key
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
                entity_feedback=entity_feedback,           # 回路修复
                entity_registry=entity_registry,           # 回路修复
                retry_count=retry_count,
                all_feedback=all_feedback,
                specific_edits=specific_edits,
                register_target=register_target,
                low_dims=low_dims,
                skipped_dimensions=skipped_dimensions,  # Round 64
                verse_note=verse_note,
            )
            temperature = self._calculate_retry_temperature(base_temp, retry_count)
        else:
            prompt = self._build_prompt(
                raw_translation, decisions_context, style_guide,
                narrative_memory=narrative_memory,
                entity_registry=entity_registry,            # 回路修复（首次润色也有登记表参考）
                verse_note=verse_note,
            )
            temperature = base_temp

        # 覆盖 temperature（用计算后的值替换 params 中的配置值）
        params = {**params, "temperature": temperature}
        extra_body = get_role_extra_body(role)

        final_markdown = llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
            extra_body=extra_body,
        )

        # 解析自评 JSON（仅 retry 轮）
        self_review = None
        clean_text = final_markdown
        if retry_count > 0 and specific_edits:
            # P3-3: 使用顶层 re（删除冗余 `import re as _re`）
            # 匹配末尾的 ```json ... ``` 块
            m = re.search(r'```json\s*(.+?)\s*```', final_markdown, re.DOTALL)
            if m:
                raw_block = m.group(1).strip()
                # P2-1: 统一走 robust_json_loads，先按 list 解析（自评是数组），失败回退 dict
                parsed = robust_json_loads(raw_block, expected_type=list)
                if not parsed:
                    parsed_dict = robust_json_loads(raw_block, expected_type=dict)
                    parsed = [parsed_dict] if parsed_dict else None
                if parsed:
                    self_review = parsed
                    clean_text = final_markdown[:m.start()].rstrip()
                else:
                    # NDJSON 兜底
                    try:
                        ndjson_items = []
                        for line in raw_block.splitlines():
                            line = line.strip()
                            if line:
                                # P2-1: NDJSON 行用 robust_json_loads 抗畸形
                                obj = robust_json_loads(line, expected_type=dict)
                                if obj:
                                    ndjson_items.append(obj)
                        if ndjson_items:
                            self_review = ndjson_items
                            clean_text = final_markdown[:m.start()].rstrip()
                    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
                        pass
            if self_review is None:
                print(f"  ⚠️ [Rewriter] {chunk_id} 自评 JSON 解析失败，降级为无自评模式")

        # 返回 (润色正文, 自评) 元组，由调用方保存到 metadata
        return clean_text, self_review


# ----- 6.3 CriticAgent -----

# 注意：阈值配置从 _config.critic_thresholds 读取，而非此处常量
