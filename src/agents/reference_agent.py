"""
ReferenceAgent — 典故考据 + 实体识别 Agent

源：原整合脚本 §6 中的 ReferenceAgent 类。
职责：
  1. 识别原文中的文学典故、宗教隐喻、神话引用、历史名词
  2. 识别本片段中的人物 / 地点实体
  3. 写入 Decision DB（REFERENCE 级决策）
  4. 增量登记实体到 entity_registry
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional

from src.config import config as _config
from src.utils.llm_adapter import (
    get_llm_client_for_task,
    get_role_model_name,
    get_role_extra_body,
    robust_json_loads,
)
from src.core.decision_engine import DecisionEngine, DecisionLevel


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的 Levenshtein 编辑距离（手写，无外部依赖）。与单体版逐字一致。"""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


class ReferenceAgent:
    def __init__(self, decision_engine: DecisionEngine, chapter_id: str = None):
        self.llm = get_llm_client_for_task("reference_extraction")
        self.role = _config.get_task_role("reference_extraction")
        self.db = decision_engine
        self.chapter_id = chapter_id
        # 获取 scheduler 用于全库扫描
        self._scheduler = decision_engine._scheduler_factory() if decision_engine._scheduler_factory else None

    def _find_chunks_containing(self, source_text: str) -> List[str]:
        """从 workflow DB 全文搜索包含 source_text 的 chunk_id。

        范围由 decision_engine.backtrack_scope 控制：
          - book（默认）：全书扫描
          - chapter：仅当前章节
        """
        if not self._scheduler:
            return []
        try:
            de_cfg = _config.decision_engine if hasattr(_config, "decision_engine") else {}
            scope = (de_cfg.get("backtrack_scope") or "book").lower()
            max_aff = int(de_cfg.get("max_affected_chunks_per_decision", 50) or 50)
            chapter_id = self.chapter_id if scope == "chapter" else None
            if scope == "chapter" and not self.chapter_id:
                return []
            return self._scheduler.find_chunks_containing(
                source_text, chapter_id=chapter_id, limit=max_aff if max_aff > 0 else None
            )
        except Exception as e:
            print(f"⚠️ [ReferenceAgent] 扫描受影响 chunk 失败: {e}")
            return []

    def _build_prompt(self, text_chunk: str) -> str:
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        work_type = (_config.style_guide or {}).get("work_type") or "长篇小说"
        hints = (_config.style_guide or {}).get("translation_hints", []) or []
        hints_block = ""
        if hints:
            hints_block = (
                "   （本书已登记考据条目，优先核对：\n"
                + "\n".join(f"      - {h}" for h in hints[:30])
                + "\n   ）"
            )
        return f"""你是一位精通西方文学、历史、神话和宗教的资深翻译考据专家。
你的任务是扫描给定的{genre}（{work_type}）片段，完成两项并行任务：
1. 识别其中的文学典故、宗教隐喻、神话引用或历史名词，并制定翻译策略。
2. 识别本片段中出现的人物/地点实体，建立实体档案。

【硬约束】
- 思考过程必须用中文。最终输出必须是严格的 JSON，禁止任何 JSON 之外的字符。
- 不要输出 "思考过程"，最终只保留 JSON。

【严格定义】
不要提取普通的角色名字（如 Paul）或普通地名，除非它们具有明显的象征意义或典故来源。

【典故识别 — 两类独立检查】
1. 专有名词/术语: 具有特定法律/社会/宗教含义的专名（直译会让中文读者误解的）
2. 文学引用/诗句: 对知名诗句/名言的引用或谐仿（判据：押韵、格律、典故化用等引文特征）
{hints_block}

【翻译策略池】
对于识别出的典故，你必须从以下策略中选择一种：
1. "RETAIN_AND_ANNOTATE" (保留音译/直译，并在后续生成脚注)
2. "CULTURAL_EQUIVALENT" (寻找目标语言中的等效文化意象)
3. "LITERAL_TRANSLATION" (仅作字面翻译，放弃深层隐喻)

【实体识别要求】
对于识别出的人物/地点实体，必须提供以下字段：
- canonical_name: 主名（中文译名，统一规范）
- source_names: 原文中出现的所有指称形式（本名、昵称、"the captain"这类称谓、代词轨迹），JSON数组
- pronoun_gender: 代词性别
- entity_type: 实体类型
- confidence: 把握程度
- translation_strategy: 翻译策略（literal=直译, transliteration=音译, free=自由意译）；首次登记后固定，后续必须沿用同一策略

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
  ],
  "entities": [
    {{
      "canonical_name": "主名（中文译名，统一规范）",
      "source_names": ["原文中的指称形式1", "指称形式2"],
      "pronoun_gender": "he/she/it/unknown",
      "entity_type": "person/place/organization/other",
      "confidence": "high/medium/low",
      "translation_strategy": "literal/transliteration/free"
    }}
  ]
}}

【待分析文本】
{text_chunk}
"""

    def _parse_and_clean_json(self, raw_response: str) -> Dict[str, Any]:
        """防御性 JSON 解析器：处理 LLM 可能输出的多种格式违规"""
        result = robust_json_loads(raw_response)
        if not result:
            print(f"⚠️ JSON 解析失败，模型输出格式违规")
            print(f"原始内容预览: {raw_response[:200]}...")
        return result if result else {"references": []}

    def _run_confirm_stage(
        self, chunk_id: str, text_chunk: str, recall_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Round 118 P1-3: 第二阶段确认（MiniMax 复核 DeepSeek 高召回初筛）.

        Round 25 曾为降本把考据收敛到单模型；质量优先恢复两阶段分工：
          - 阶段 1（DeepSeek, 低温）：高召回，宁多勿漏；
          - 阶段 2（MiniMax, thinking）：逐条核验典故来源、删误报、补漏、
            统一实体 canonical/策略，输出定稿 JSON。

        任何失败（配置缺失/网络/JSON 损坏）都返回 None，调用方回退召回结果，
        绝不因确认阶段把已有召回丢掉。
        """
        cfg = _config.pipeline.get("reference_two_stage", {}) or {}
        if not cfg.get("enabled", True):
            return None
        try:
            role = _config.get_task_role("reference_confirmation")
            confirm_llm = get_llm_client_for_task("reference_confirmation")
            model_key, params = _config.resolve_task_model("reference_confirmation")
        except Exception as e:
            print(f"⚠️ [Reference Agent] 两阶段确认不可用（保留召回结果）: {e}", file=sys.stderr)
            return None
        model_name = get_role_model_name(role) or model_key
        extra_body = get_role_extra_body(role)
        max_chars = int(cfg.get("max_recall_chars", 12000))
        recall_json = json.dumps(recall_data, ensure_ascii=False, indent=2)[:max_chars]
        prompt = f"""你是资深文学考据复核员（第二意见）。另一位考据员已对片段做「高召回」初筛，你的任务是复核并定稿。

【原文片段】
{text_chunk[:8000]}

【初筛候选（JSON）】
{recall_json}

【复核规则】
1. 逐条核验 allusion_target 是否被原文支撑；无依据、张冠李戴或明显误判 → 删除。
2. 补漏：初筛漏掉且确有出处的典故/隐喻/历史名词 → 追加（宁精勿滥）。
3. 实体：合并重复指称；canonical_name 统一为规范中文译名；source_names 必须都是原文实际出现的形式。
4. 策略只能是 RETAIN_AND_ANNOTATE / CULTURAL_EQUIVALENT / LITERAL_TRANSLATION 三选一。
5. 若初筛正确，原样保留其条目（可修正字段格式）；不要为凑数新增无依据条目。

【输出格式】
输出与初筛完全相同的纯 JSON（references / entities 两个数组），不要输出 JSON 以外的任何字符。
"""
        try:
            raw = confirm_llm.generate(
                prompt=prompt,
                model_name=model_name,
                **params,
                extra_body=extra_body,
            )
        except Exception as e:
            print(
                f"⚠️ [Reference Agent] 两阶段确认调用失败（保留召回结果）: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return None
        data = robust_json_loads(raw)
        # robust_json_loads 对垃圾输入可能返回 {}；确认结果必须至少含
        # references/entities 之一，否则视为解析失败回退召回。
        if not isinstance(data, dict) or not data:
            print("⚠️ [Reference Agent] 两阶段确认 JSON 解析失败（保留召回结果）", file=sys.stderr)
            return None
        if "references" not in data and "entities" not in data:
            print("⚠️ [Reference Agent] 两阶段确认 JSON 解析失败（保留召回结果）", file=sys.stderr)
            return None
        refs = data.get("references")
        ents = data.get("entities")
        return {
            "references": refs if isinstance(refs, list) else [],
            "entities": ents if isinstance(ents, list) else [],
        }

    def process_chunk(self, chunk_id: str, text_chunk: str, affected_chunks: List[str] = None):
        """处理单个文本块，识别典故并自动写入决策引擎，同时登记实体到注册表"""
        print(f"🔍 [Reference Agent] 正在考据数据块: {chunk_id}...")

        # 1. 调用 LLM (使用配置路由)
        prompt = self._build_prompt(text_chunk)
        model_key, params = _config.resolve_task_model("reference_extraction")
        model_name = get_role_model_name(self.role) or model_key
        extra_body = get_role_extra_body(self.role)

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
            extra_body=extra_body,
        )

        # 2. 解析 JSON
        parsed_data = self._parse_and_clean_json(raw_output)
        references = parsed_data.get("references", [])
        entities = parsed_data.get("entities", [])

        # Round 118 P1-3: 两阶段 — DeepSeek 高召回初筛 + MiniMax 逐条复核定稿。
        # 确认失败/禁用时静默回退召回结果（绝不丢数据）。
        confirmed = self._run_confirm_stage(chunk_id, text_chunk, parsed_data)
        if confirmed is not None:
            print(
                f"🔬 [Reference Agent] 两阶段确认: references "
                f"{len(references)}→{len(confirmed['references'])}, entities "
                f"{len(entities)}→{len(confirmed['entities'])}"
            )
            parsed_data = confirmed
            references = confirmed["references"]
            entities = confirmed["entities"]

        if not references and not entities:
            print("  未发现深度典故或实体。")
            return

        # 3. 将决策写入 Decision DB (Level 2: 典故)
        for ref in references:
            source = ref.get("source_text")
            translation = ref.get("translated_term")
            strategy = ref.get("strategy")
            reason = f"【来源】{ref.get('allusion_target')} | 【策略】{strategy} | 【依据】{ref.get('reason')}"

            if source and translation:
                # 扫描全章所有包含该 source_text 的 chunk
                all_affected = self._find_chunks_containing(source)
                if affected_chunks:
                    all_affected = list(set(all_affected) | set(affected_chunks))

                self.db.add_decision(
                    level=DecisionLevel.REFERENCE,
                    source=source,
                    translation=translation,
                    reason=reason,
                    affected_chunks=all_affected or [chunk_id]
                )

        # 4. 登记实体到 entity_registry (增量构建全书实体知识库)
        if entities:
            self._register_entities(chunk_id, entities)

    def _register_entities(self, chunk_id: str, entities: List[Dict]):
        """将识别到的实体增量登记到 entity_registry 表"""
        if not entities:
            return
        
        cursor = self.db.conn.cursor()
        
        for ent in entities:
            canonical = ent.get("canonical_name")
            source_names = ent.get("source_names", [])
            pronoun_gender = ent.get("pronoun_gender", "unknown")
            entity_type = ent.get("entity_type", "other")
            confidence = ent.get("confidence", "low")
            translation_strategy = ent.get("translation_strategy", "free")
            
            if not canonical or not source_names:
                continue
            
            # 确保 source_names 是列表
            if isinstance(source_names, str):
                source_names = [source_names]
            
            # 过滤代词/停用词；Round 18 审计修复：移除 `len(sn) >= 4` 长度阈值
            # 原因：
            #   1. 长度阈值导致所有 <4 字符的英文原名（如 3 字符人名 "Joe"/"Kim"/"Ana"）
            #      永远无法进入 entity_registry.source_names
            #   2. Section 11.5 的 english_residue 检测依赖 source_names 列表，
            #      短英文原名的残留检测因此"天生失效"（Round 18 问题 1）
            #   3. 短名字的"子串误匹配"风险由调用方处理：
            #      - _detect_issues 已用 `\b` 词边界匹配（L4191-4193）
            #      - format_prompt_context 已用 `(sn in src or sn.lower() in src_lower)`
            #      停用词表本身已能排除代词/冠词/介词的误判
            _STOPWORDS = {
                "a", "an", "the", "and", "or", "but", "in", "on", "at",
                "to", "for", "of", "by", "with", "from", "as", "is", "was",
                "he", "she", "it", "they", "we", "you", "i", "me", "him",
                "her", "his", "my", "your", "its", "our", "their", "them",
                "us", "this", "that", "these", "those", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "can", "could", "shall", "should", "may", "might", "not",
                # Round 121: 疑问词/连词泄漏护栏（ch06 "who"→帽匠等 6 实体污染，
                # 造成"在场却漏译"不可满足核查 → 实体硬失败循环）
                "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
            }
            source_names = [sn for sn in source_names if sn.lower() not in _STOPWORDS]
            if not source_names:
                continue
            
            source_names_json = json.dumps(source_names, ensure_ascii=False)
            
            # 查找是否已存在同一 canonical_name
            cursor.execute('SELECT id, source_names, pending_review FROM entity_registry WHERE canonical_name = ?', (canonical,))
            row = cursor.fetchone()
            
            if row:
                # 已存在：合并 source_names
                existing_id, existing_names_json, pending_review = row
                # P2-1: 统一走 robust_json_loads
                existing_names = robust_json_loads(existing_names_json, expected_type=list) if existing_names_json else []
                
                # 合并新旧 source_names（去重）
                merged_names = list(set(existing_names + source_names))
                merged_names_json = json.dumps(merged_names, ensure_ascii=False)
                
                # 合并置信度：取较高者
                confidence_order = {"high": 3, "medium": 2, "low": 1}
                new_conf_level = confidence_order.get(confidence, 1)
                cursor.execute('SELECT confidence FROM entity_registry WHERE id = ?', (existing_id,))
                current_row = cursor.fetchone()
                if current_row:
                    current_conf = confidence_order.get(current_row[0], 1)
                    final_conf = confidence if new_conf_level > current_conf else current_row[0]
                else:
                    final_conf = confidence
                
                cursor.execute('''
                    UPDATE entity_registry 
                    SET source_names = ?, confidence = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (merged_names_json, final_conf, existing_id))
                print(f"  📝 实体已存在，合并指称: {canonical} <- {source_names}")
            else:
                # 检查是否有语义相近的已有实体（简单字符串包含/编辑距离初筛）
                cursor.execute('SELECT id, canonical_name, source_names, variant_names FROM entity_registry')
                similar_found = False
                for existing_row in cursor.fetchall():
                    eid, ecanon, enames_json, evariants_json = existing_row
                    # P2-1: 统一走 robust_json_loads
                    enames = robust_json_loads(enames_json, expected_type=list) if enames_json else []
                    evariants = robust_json_loads(evariants_json, expected_type=list) if evariants_json else []
                    
                    # 双向子串包含 + 编辑距离(≤2) 初筛
                    # 注：source_names 已在 _register_entities 入口过滤停用词
                    substr_match = any(
                        sn in existing_name or existing_name in sn
                        for existing_name in enames
                        for sn in source_names
                    )
                    fuzzy_match = any(
                        _levenshtein_distance(sn, existing_name) <= 2
                        for existing_name in enames
                        for sn in source_names
                    )
                    if substr_match or fuzzy_match:
                        # Round 121: 新中文名已是该实体的登记异体 → 已知别名，
                        # 仅并入 source_names，不再标记待核查（防归一后反复重标）
                        if canonical in evariants:
                            merged_names = list(set(enames + source_names))
                            cursor.execute('''
                                UPDATE entity_registry
                                SET source_names = ?, updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            ''', (json.dumps(merged_names, ensure_ascii=False), eid))
                            similar_found = True
                            print(f"  📝 已知异体合并: {canonical} <- {source_names} (并入 {ecanon})")
                            break
                        # 发现潜在同一实体的不同译名
                        # 不自动合并，标记待核查
                        cursor.execute('''
                            UPDATE entity_registry 
                            SET pending_review = 1, review_reason = ?
                            WHERE id = ?
                        ''', (f"称谓差异大: 新实体 {canonical} 与现有 {ecanon} 可能指向同一对象", eid))
                        similar_found = True
                        print(f"  ⚠️ 发现潜在同一实体不同译名: {canonical} vs {ecanon}，标记待核查")
                        break
                
                if not similar_found:
                    # 新建条目
                    cursor.execute('''
                        INSERT INTO entity_registry (canonical_name, source_names, pronoun_gender, entity_type, first_seen_chunk, confidence, pending_review, translation_strategy)
                        VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    ''', (canonical, source_names_json, pronoun_gender, entity_type, chunk_id, confidence, translation_strategy))
                    print(f"  ✅ 新实体登记: {canonical} ({entity_type}, {confidence})")
        
        self.db.conn.commit()
