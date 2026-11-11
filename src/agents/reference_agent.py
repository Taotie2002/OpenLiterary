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
    robust_json_loads,
)
from src.core.decision_engine import (
    DecisionEngine,
    DecisionLevel,
    alias_attested_in_text,
    classify_entity_source_name,
    decision_alias_belongs_to_owner,
    decision_alias_conflicts_with_owner,
    load_decision_alias_targets,
)


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
- source_names: 本片段中能**唯一指向该实体**的专名（本名、姓氏+名字、头衔+名字），JSON数组。
  禁止把 the Duke / my Lord / Father / the Emperor / the man / 代词轨迹 放进 source_names；
  这些泛称、关系词、描述性指称一律放进 context_names。
- context_names: 本片段中指向该实体的泛称/称谓/关系词/描述性指称，JSON数组（可一对多，不作硬约束）
- pronoun_gender: 代词性别
- entity_type: 实体类型
- confidence: 把握程度
- translation_strategy: 翻译策略（literal=直译, transliteration=音译, free=自由意译）；首次登记后固定，后续必须沿用同一策略

【别名归属硬约束】
- 同一个 source_name 只能属于一个 canonical。若某指称可能指多人，或指代对象不确定，放入 context_names。
- 不要在一个角色的 source_names 里登记其他角色的名字或别称。
- 不要把 "Duke Leto" 登记给 Paul/Baron，"the Emperor" 登记给 Baron；这类跨人物错配会直接污染全书实体库。

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
      "source_names": ["可唯一识别该实体的专名1", "专名2"],
      "context_names": ["the Duke", "my Lord", "代词轨迹等泛称/关系指称"],
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
3. 实体：合并重复指称；canonical_name 统一为规范中文译名；source_names 只保留可唯一识别该实体的专名，
   泛称/关系词/描述性指称移入 context_names。source_name 跨 canonical 冲突时，宁可放入 context_names，绝不硬配。
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

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
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
            self._register_entities(chunk_id, entities, text_chunk)

    def _register_entities(self, chunk_id: str, entities: List[Dict],
                           source_text: str = None):
        """将识别到的实体增量登记到 entity_registry 表。

        Round 144 别名完整性：
          - source_names 只存可执行专名；泛称/关系/描述进入 context_names；
          - 同一专名被第二个 canonical 声明时，立即从所有 owner 的硬别名中移除，
            并写入 quarantined_names（fail-open：宁可少约束，也不制造错约束）。
        """
        if not entities:
            return

        cursor = self.db.conn.cursor()

        # 一次性载入 registry，避免每个 alias 全表扫描。
        try:
            cursor.execute(
                "SELECT id, canonical_name, source_names, context_names, quarantined_names, "
                "variant_names, confidence, pending_review, entity_type "
                "FROM entity_registry"
            )
        except Exception:
            # 兼容极端旧库：补齐列后重试。
            from src.core.decision_engine import ensure_entity_alias_columns
            ensure_entity_alias_columns(self.db.conn)
            cursor.execute(
                "SELECT id, canonical_name, source_names, context_names, quarantined_names, "
                "variant_names, confidence, pending_review, entity_type "
                "FROM entity_registry"
            )

        def _load_list(raw):
            return robust_json_loads(raw, expected_type=list) if raw else []

        def _norm(sn):
            return re.sub(r"\s+", " ", (sn or "")).strip().lower()

        def _dedupe(items):
            out, seen = [], set()
            for item in items or []:
                if not isinstance(item, str):
                    continue
                s = item.strip()
                key = _norm(s)
                if not s or key in seen:
                    continue
                seen.add(key)
                out.append(s)
            return out

        registry: Dict[str, Dict[str, Any]] = {}
        owners: Dict[str, set] = {}
        for rid, canon, src_json, ctx_json, qt_json, var_json, conf, pending, etype in cursor.fetchall():
            row = {
                "id": rid,
                "source": _dedupe(_load_list(src_json)),
                "context": _dedupe(_load_list(ctx_json)),
                "quarantine": _dedupe(_load_list(qt_json)),
                "variants": _dedupe(_load_list(var_json)),
                "confidence": conf or "low",
                "pending_review": pending or 0,
                "entity_type": etype or "",
            }
            registry[canon] = row
            for sn in row["source"]:
                owners.setdefault(_norm(sn), set()).add(canon)
        decision_targets = load_decision_alias_targets(self.db.conn)

        def _persist(canon: str) -> None:
            row = registry[canon]
            cursor.execute(
                "UPDATE entity_registry SET source_names=?, context_names=?, "
                "quarantined_names=?, variant_names=?, confidence=?, pending_review=?, "
                "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (json.dumps(_dedupe(row["source"]), ensure_ascii=False),
                 json.dumps(_dedupe(row["context"]), ensure_ascii=False),
                 json.dumps(_dedupe(row["quarantine"]), ensure_ascii=False),
                 json.dumps(_dedupe(row["variants"]), ensure_ascii=False),
                 row["confidence"], row["pending_review"], row["id"]),
            )

        # Round 18/121 保留既有结构性停用词护栏。
        _STOPWORDS = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "by", "with", "from", "as", "is", "was",
            "he", "she", "it", "they", "we", "you", "i", "me", "him",
            "her", "his", "my", "your", "its", "our", "their", "them",
            "us", "this", "that", "these", "those", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "shall", "should", "may", "might", "not",
            "who", "whom", "whose", "which", "what", "where", "when", "why", "how",
        }
        _CONF_ORDER = {"high": 3, "medium": 2, "low": 1}

        for ent in entities:
            canonical = (ent.get("canonical_name") or "").strip()
            pronoun_gender = ent.get("pronoun_gender", "unknown")
            entity_type = ent.get("entity_type", "other")
            confidence = ent.get("confidence", "low")
            translation_strategy = ent.get("translation_strategy", "free")

            if not canonical:
                continue

            raw_sources = ent.get("source_names", []) or []
            raw_contexts = ent.get("context_names", []) or []
            if isinstance(raw_sources, str):
                raw_sources = [raw_sources]
            if isinstance(raw_contexts, str):
                raw_contexts = [raw_contexts]

            aliases = _dedupe(list(raw_sources) + list(raw_contexts))
            aliases = [s for s in aliases if s.lower() not in _STOPWORDS]
            if not aliases:
                continue

            proper_candidates = [s for s in aliases if classify_entity_source_name(s) == "proper"]
            context_candidates = [s for s in aliases if classify_entity_source_name(s) != "proper"]

            existing = registry.get(canonical)
            existing_qt = {_norm(x) for x in (existing or {}).get("quarantine", [])}

            accepted: List[str] = []
            quarantined: List[str] = []
            for sn in proper_candidates:
                key = _norm(sn)
                if source_text and not alias_attested_in_text(sn, source_text):
                    quarantined.append(sn)
                    print(f"  ⚠️ 无原文出处的专名隔离: {sn!r} 不登记给 {canonical}")
                    continue
                known_canonicals = list(registry)
                if canonical not in registry:
                    known_canonicals.append(canonical)
                known_types = {c: r["entity_type"] for c, r in registry.items()}
                known_types[canonical] = entity_type
                decision_owned = decision_alias_belongs_to_owner(
                    sn, canonical, decision_targets, known_canonicals,
                    entity_types=known_types)
                decision_foreign = (not decision_owned
                                    and decision_alias_conflicts_with_owner(
                                        sn, canonical, decision_targets,
                                        known_canonicals,
                                        entity_types=known_types))
                if key in existing_qt and not decision_owned:
                    quarantined.append(sn)
                    continue
                # 任何 owner 已隔离过该专名 → 不可再被硬声明。
                if (not decision_owned and any(
                        key in {_norm(x) for x in row["quarantine"]}
                        for row in registry.values())):
                    quarantined.append(sn)
                    continue
                other_owners = set(owners.get(key, set())) - {canonical}
                if other_owners:
                    if decision_owned:
                        # decision_db 已裁决归属：只从身份无关的 owner 移除，
                        # 同一身份簇（同人重复 canonical）保留。
                        kept_owners = {canonical}
                        for owner in sorted(other_owners):
                            if decision_alias_belongs_to_owner(
                                    sn, owner, decision_targets,
                                    known_canonicals, entity_types=known_types):
                                kept_owners.add(owner)
                                continue
                            orow = registry.get(owner)
                            if not orow:
                                continue
                            orow["source"] = [
                                x for x in orow["source"] if _norm(x) != key
                            ]
                            orow["quarantine"] = _dedupe(orow["quarantine"] + [sn])
                            _persist(owner)
                            print(f"  ⚖️ decision_db 归属裁决: {sn!r} 保留给 {canonical}，"
                                  f"从 {owner} 隔离")
                        if existing is not None:
                            existing["quarantine"] = [
                                x for x in existing["quarantine"]
                                if _norm(x) != key
                            ]
                        owners[key] = kept_owners
                        accepted.append(sn)
                        continue
                    # fail-open：冲突专名从所有 owner 移除，任何 canonical 都不再执行。
                    for owner in sorted(other_owners):
                        orow = registry.get(owner)
                        if not orow:
                            continue
                        before = len(orow["source"])
                        orow["source"] = [x for x in orow["source"] if _norm(x) != key]
                        orow["quarantine"] = _dedupe(orow["quarantine"] + [sn])
                        if len(orow["source"]) != before:
                            _persist(owner)
                        print(f"  ⚠️ 专名冲突隔离: {sn!r} 同时指向 {owner} 与 {canonical}；两边均不硬执行")
                    owners[key] = set()
                    quarantined.append(sn)
                    continue
                # Round 144: 即使当前只有一个 owner，只要 decision_db 把
                # 这个专名唯一翻译到另一个身份无关的 canonical，也隔离。
                # 这能挡住“只错挂一次、没有第二 owner 触发冲突”的污染。
                if decision_foreign:
                    quarantined.append(sn)
                    print(f"  ⚠️ decision_db 归属冲突隔离: {sn!r} 不登记给 {canonical}")
                    continue
                if decision_owned and existing is not None:
                    existing["quarantine"] = [
                        x for x in existing["quarantine"] if _norm(x) != key
                    ]
                accepted.append(sn)
                owners.setdefault(key, set()).add(canonical)

            if existing is not None:
                existing["source"] = _dedupe(existing["source"] + accepted)
                existing["context"] = _dedupe(existing["context"] + context_candidates)
                existing["quarantine"] = _dedupe(existing["quarantine"] + quarantined)
                cur_conf = _CONF_ORDER.get(existing["confidence"], 1)
                new_conf = _CONF_ORDER.get(confidence, 1)
                if new_conf > cur_conf:
                    existing["confidence"] = confidence
                _persist(canonical)
                print(f"  📝 实体已存在，合并专名: {canonical} <- {accepted or '[]'}")
                continue

            # 新 canonical：保留既有“近似名 → pending_review”行为。
            similar_found = False
            if proper_candidates:
                for ecanon, erow in list(registry.items()):
                    enames = erow["source"]
                    substr_match = any(
                        sn in existing_name or existing_name in sn
                        for existing_name in enames for sn in proper_candidates
                    )
                    fuzzy_match = any(
                        _levenshtein_distance(sn, existing_name) <= 2
                        for existing_name in enames for sn in proper_candidates
                    )
                    if not (substr_match or fuzzy_match):
                        continue
                    if canonical in erow["variants"]:
                        erow["source"] = _dedupe(erow["source"] + accepted)
                        erow["context"] = _dedupe(erow["context"] + context_candidates)
                        erow["quarantine"] = _dedupe(erow["quarantine"] + quarantined)
                        for sn in accepted:
                            owners.setdefault(_norm(sn), set()).discard(canonical)
                            owners.setdefault(_norm(sn), set()).add(ecanon)
                        _persist(ecanon)
                        similar_found = True
                        print(f"  📝 已知异体合并: {canonical} <- {accepted} (并入 {ecanon})")
                        break
                    erow["pending_review"] = 1
                    cursor.execute(
                        "UPDATE entity_registry SET pending_review=1, review_reason=?, "
                        "updated_at=CURRENT_TIMESTAMP WHERE id=?",
                        (f"称谓差异大: 新实体 {canonical} 与现有 {ecanon} 可能指向同一对象",
                         erow["id"]),
                    )
                    similar_found = True
                    print(f"  ⚠️ 发现潜在同一实体不同译名: {canonical} vs {ecanon}，标记待核查")
                    break

            if not similar_found:
                pending = 0 if accepted else 1
                cursor.execute(
                    "INSERT INTO entity_registry "
                    "(canonical_name, source_names, context_names, quarantined_names, "
                    "pronoun_gender, entity_type, first_seen_chunk, confidence, "
                    "pending_review, translation_strategy) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (canonical,
                     json.dumps(_dedupe(accepted), ensure_ascii=False),
                     json.dumps(_dedupe(context_candidates), ensure_ascii=False),
                     json.dumps(_dedupe(quarantined), ensure_ascii=False),
                     pronoun_gender, entity_type, chunk_id, confidence, pending,
                     translation_strategy),
                )
                registry[canonical] = {
                    "id": cursor.lastrowid,
                    "source": _dedupe(accepted),
                    "context": _dedupe(context_candidates),
                    "quarantine": _dedupe(quarantined),
                    "variants": [],
                    "confidence": confidence,
                    "pending_review": pending,
                    "entity_type": entity_type,
                }
                print(f"  ✅ 新实体登记: {canonical} ({entity_type}, {confidence}, "
                      f"专名={len(accepted)}, 上下文={len(context_candidates)})")

        self.db.conn.commit()
