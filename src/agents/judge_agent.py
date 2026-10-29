"""
JudgeAgent — 最终裁决 Agent

源：原整合脚本 §6 中的 JudgeAgent 类。
职责：
  1. 综合 Critic 6 维评分与 Rewriter 自评，做出 PASS / REJECT 决策
  2. 注入 critic_thresholds 作为参考（非硬性否决）
  3. 实体一致性检查（强制退回润色触发条件）
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from src.config import config as _config
from src.utils.llm_adapter import (
    get_llm_client_for_task,
    get_role_model_name,
    robust_json_loads,
)
from src.core.decision_engine import DecisionEngine
from src.judges.typesafe_spec import CompositeVerdict


class JudgeAgent:
    def __init__(self, decision_engine: DecisionEngine):
        self.llm = get_llm_client_for_task("judge_decision")
        self.role = _config.get_task_role("judge_decision")
        self.db = decision_engine

    @staticmethod
    def _sanitize_critic_report_for_llm(critic_report: dict) -> dict:
        """Round 106 P2: 移除 mock_heuristic 的假分数/verdict，避免带偏 LLM 裁决。

        mock 后端（Jev 不可用）生成的 scores/_typesafe_verdict/critique 是确定性
        假数据，喂给 LLM judge 会使其把假分数当真实质量信号（ch05 实测 7/7 chunk
        被 REJECT）。移除这些字段，保留 entity_consistency_notes（本地核查，可靠）
        和 specific_edits，让 LLM 纯凭译文质量独立判断。
        """
        sanitized = dict(critic_report)
        for key in ("scores", "_typesafe_verdict", "critique", "improvement_suggestions"):
            sanitized.pop(key, None)
        sanitized["_note"] = (
            "（critic 后端为 mock_heuristic，Jev 不可用，无可靠评分；"
            "请纯凭【原文】与【当前润色稿】的翻译质量独立判断，不要假设存在低分维度）"
        )
        return sanitized

    def _second_opinion_reject_edits(
        self,
        chunk_id: str,
        source_text: str,
        lit_trans: str,
        verdict: "CompositeVerdict",
        reject_reason: str,
    ) -> tuple:
        """Round 118 P1-2: REJECT 第二意见 → 可执行编辑清单.

        Jev 的 reject_reasons 常只有概率数字（如 "noul=0.82"），rewriter 无法执行，
        导致 REWRITER_STAGNATED（ch32 实测）。本方法在 REJECT 后触发一次独立 LLM
        复核（judge_decision 角色），把拒因翻译成逐条可执行编辑；不改变 Jev 的
        decision（绝不弱化仲裁），只补 rewriter 所需的具体指令。

        Returns:
            (edits, info): edits 兼容 pipeline 的 specific_edits 结构；
            info 为审计元数据（confirmed_reject=false 表示第二意见认为拒因
            无可执行修改点，仅供参考，不用于翻案）。
        """
        cfg = _config.pipeline.get("judge_second_opinion", {}) or {}
        if not cfg.get("enabled", True):
            return [], {"enabled": False}
        max_edits = int(cfg.get("max_edits", 20))
        reasons = (reject_reason or "; ".join(verdict.reject_reasons))[:1500]
        summary_json = json.dumps(verdict.primitive_summary, ensure_ascii=False)[:2500]
        prompt = f"""你是资深文学翻译终审编辑。另一位校准裁判已给出 REJECT，但拒因多为概率信号，润色师无法执行。
你的任务：仅产出**可执行的修改指令**（不要重新评分、不要讨论是否改判）。

【原文】
{source_text[:2500]}

【当前润色稿】
{lit_trans[:2500]}

【裁判拒因】
{reasons}

【原语信号（JSON）】
{summary_json}

【输出 JSON — 严格遵守】
{{"edits": [{{"original": "译文中的问题片段（照抄，便于定位）", "problem": "问题说明（一句话）", "suggested_direction": "具体修改方向或替换建议"}}], "confirmed": true}}

【规则】
1. 只列真实存在、可定位的修改点；每条 original 必须能在润色稿中找到。
2. 若拒因确无可执行修改点（疑似概率噪声），edits 输出空数组，confirmed 置 false。
3. 最多 {max_edits} 条；不要输出 JSON 以外的任何文字。"""
        model_key, params = _config.resolve_task_model("judge_decision")
        model_name = get_role_model_name(self.role) or model_key
        params = dict(params)
        params["temperature"] = float(cfg.get("temperature", 0.2))
        try:
            raw = self.llm.generate(
                prompt=prompt, model_name=model_name, **params,
            )
            data = robust_json_loads(raw)
            if not isinstance(data, dict):
                raise RuntimeError("第二意见输出非 JSON 对象")
            edits: List[Dict[str, Any]] = []
            for i, e in enumerate(data.get("edits") or [], 1):
                if not isinstance(e, dict):
                    continue
                direction = str(e.get("suggested_direction", "")).strip()
                problem = str(e.get("problem", "")).strip()
                if not direction and not problem:
                    continue
                edits.append({
                    "index": i,
                    "original": str(e.get("original", ""))[:400],
                    "problem": problem[:300],
                    "suggested_direction": direction[:300],
                })
            confirmed = bool(data.get("confirmed", True))
            info = {
                "triggered": True,
                "edits": len(edits),
                "confirmed_reject": confirmed,
                "model_role": "judge_decision",
            }
            print(f"🧭 [Judge Agent] {chunk_id} REJECT 第二意见: edits={len(edits)} confirmed={confirmed}")
            return edits[:max_edits], info
        except Exception as e:
            print(
                f"⚠️ [Judge Agent] {chunk_id} REJECT 第二意见失败"
                f"（保留原裁决，不阻断流程）: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return [], {"triggered": True, "error": f"{type(e).__name__}: {e}"}

    def _build_prompt(self, source_text: str, lit_trans: str, critic_report: dict, rewriter_self_review: Optional[list] = None, verse_note: Optional[str] = None) -> str:
        # P0: 将 critic_thresholds 注入为参考阈值（非硬性否决条件）
        thresholds = _config.critic_thresholds_for_book()
        thresholds_lines = "\n".join(
            f"  - {dim}: 参考阈值 {v}"
            for dim, v in thresholds.items() if dim != "average_score_min"
        )
        # Round 106 P2: 无 scores 时（mock 后端已清洗），不引用阈值，纯凭译文质量判断
        if critic_report.get("scores"):
            thresholds_block = f"""
【评分参考阈值（供综合判断，不是硬性否决条件）】
审辩者各维度的参考阈值为：
{thresholds_lines}

如果审辩者报告的 scores 中某维度略低于阈值，但你认为整体质量达标，可以判 PASS。
如果多个维度显著低于阈值，且润色稿确实问题明显，才判 REJECT。
"""
        else:
            thresholds_block = """
【判断依据】
审辩者未提供可靠评分（后端不可用）。请纯凭【原文】与【当前润色稿】的翻译质量
（流畅度、可读性、风格、语义保真、实体译名一致性）独立判断：质量达标判 PASS；
确有明显问题（直译腔、生硬拗口、漏译、实体译名错误）才判 REJECT，并在
reject_reason 中说明具体位置与改法。
"""

        self_review_block = ""
        specific_edits = critic_report.get("specific_edits", [])
        if specific_edits:
            self_review_block = "\n【Rewriter 自评报告】\n"
            if rewriter_self_review:
                for item in rewriter_self_review:
                    idx = item.get("index", "?")
                    status = item.get("status", "?")
                    reason = (item.get("reason", "") or "")[:80]
                    self_review_block += f"  Edit #{idx}: status={status}, reason={reason}\n"
            else:
                self_review_block += "  （Rewriter 未提供自评，以下由你独立判断）\n"

        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        persona = (_config.style_guide or {}).get("persona", "职业文学译者")
        # Round 119 L1: 韵文对齐检查注入（仅 verse chunk 携带）
        verse_block = ""
        if verse_note:
            verse_block = (
                "\n【韵文对齐检查（本片段含韵文，必须逐行核对）】\n"
                f"{verse_note}"
                "若润色稿把韵文合并成散文行、诗行数明显不足（低于协议下限）"
                "或复沓（refrain）丢失，应判 REJECT，并在 reject_reason 中"
                "要求逐行对齐重排。\n")
        return f"""你是一位{persona}，担任本次{genre}终审。
你需要综合【审辩者报告】，决定当前的【文学润色稿】是否可以直接定稿。

【审辩者报告】
{json.dumps(critic_report, ensure_ascii=False)}
{self_review_block}
【原文】
{source_text}
【当前润色稿】
{lit_trans}
{verse_block}
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
  }},
  "unresolved_edits": [
    {{
      "index": 1,
      "location": "...",
      "issue": "...",
      "reason": "为什么这条没解决"
    }}
  ]
}}

约束：
1. unresolved_edits 是可选字段，仅在有 specific_edits 且 Rewriter 未完全解决时才输出
2. 如果全部解决或没有 specific_edits，unresolved_edits 应为空数组 []
3. unresolved_edits 不改变 decision 主逻辑——即使整体 PASS 也可以有 unresolved_edits 作为参考

注意："final_text" 不需要你输出，系统将自动使用 Rewriter 的润色稿作为最终译文。
请严格按上述 Schema 输出唯一一个 JSON 对象。
"""

    def process_chunk(self, chunk_id: str, source_text: str, lit_trans: str, critic_report: dict, affected_chunks: List[str] = None, rewriter_self_review: Optional[list] = None, verse_note: Optional[str] = None) -> Dict[str, Any]:
        """最终裁决

        T2 重构：
          1. 优先 typesafe-ai verdict 合成（code-side，无 LLM 成本）
          2. 若 critic_report 携带 _typesafe_verdict，则直接复用其 verdict
          3. 否则 fallback 旧 LLM 路径
        """
        # ───── T2 typesafe-ai 路径：从 critic 传过来的 verdict 直接使用 ─────
        verdict: Optional[CompositeVerdict] = critic_report.get("_typesafe_verdict")
        if isinstance(verdict, dict) and "decision" in verdict:
            verdict = CompositeVerdict(**{k: v for k, v in verdict.items() if k in (
                "decision", "confidence", "reject_reasons", "primitive_summary",
                "confidence_source", "confidence_calibrated",
            )})

        # Round 106: mock_heuristic 的 verdict 不可信。Jev API 不可用（如 451 区域封锁）
        # 时 critic 回退到 mock 启发式，其 _typesafe_verdict 是确定性假分数（ch05 实测
        # entity_consistency=0 / literal_residue 极低 → 整章 7/7 chunk PERMANENTLY_FAILED）。
        # 默认不信任（trust_mock_heuristic=false）：强制走 LLM 独立裁决，让真正的质量门生效。
        if (isinstance(verdict, CompositeVerdict)
                and not _config.pipeline.get("trust_mock_heuristic", False)
                and critic_report.get("_backend_path") == "mock_heuristic"):
            print(f"⚖️ [Judge Agent] {chunk_id} critic 后端为 mock_heuristic（Jev 不可用），"
                  f"跳过 typesafe verdict 复用，改用 LLM 独立裁决")
            verdict = None

        if isinstance(verdict, CompositeVerdict):
            backend_path = critic_report.get("_backend_path", "unknown")
            reject_reason = "; ".join(verdict.reject_reasons) or "全部维度通过"
            result = {
                "decision": verdict.decision,
                "reject_reason": reject_reason,
                "unresolved_edits": [],
                "_typesafe_verdict": verdict.to_dict(),
                "_backend_path": backend_path,
            }
            # 安全网：typesafe 路径也保留（即便 verdict=PASS，confidence 极低时仍拒）
            safety_min = float(_config.pipeline.get("judge_safety_net_avg_min", 5.5))
            scores = critic_report.get("scores", {})
            if scores and result["decision"] == "PASS":
                avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / max(1, len(scores))
                if avg_score < safety_min and verdict.confidence < 0.5:
                    result["decision"] = "REJECT"
                    result["reject_reason"] = (
                        f"安全网(typesafe): 平均分 {avg_score:.1f} < {safety_min} 且"
                        f" verdict 置信度 {verdict.confidence:.2f} 偏低，疑似启发式噪声"
                    )
            # Round 118 P1-2: REJECT 第二意见 — 把概率拒因翻译成可执行编辑。
            # 不改变 decision（绝不弱化 Jev），仅补 rewriter 所需的具体指令。
            if result["decision"] == "REJECT":
                second_edits, second_info = self._second_opinion_reject_edits(
                    chunk_id, source_text, lit_trans, verdict, result["reject_reason"],
                )
                if second_edits:
                    result["unresolved_edits"] = second_edits
                if second_info:
                    result["_second_opinion"] = second_info
            # style rule 提取暂不做（typesafe 不生成新规则，避免误判；保留旧路径能力）
            return result

        # ───── 旧 LLM 路径（fallback） ─────
        # Round 106 P2: mock_heuristic 的分数/verdict 不可信，喂给 LLM 会带偏裁决
        # （ch05 实测 LLM judge 反复引用 mock 假分数 readability=2.5/fluency=2.5 作为拒因，
        #  导致 7/7 chunk 全被 REJECT）。清洗 critic_report：移除 mock 的假分数/verdict，
        #  让 LLM 纯凭译文质量独立判断。
        if (not _config.pipeline.get("trust_mock_heuristic", False)
                and critic_report.get("_backend_path") == "mock_heuristic"):
            critic_report = self._sanitize_critic_report_for_llm(critic_report)
        print(f"⚖️ [Judge Agent] 正在对 {chunk_id} 进行最终裁决...")
        prompt = self._build_prompt(source_text, lit_trans, critic_report, rewriter_self_review=rewriter_self_review, verse_note=verse_note)

        model_key, params = _config.resolve_task_model("judge_decision")
        model_name = get_role_model_name(self.role) or model_key

        raw_output = self.llm.generate(
            prompt=prompt,
            model_name=model_name,
            **params,
        )

        result = robust_json_loads(raw_output)
        if not result:
            raise RuntimeError("Judge 输出 JSON 损坏，重新裁决。")

        scores = critic_report.get("scores", {})
        # Round 106: mock_heuristic 的分数不可信（Jev 不可用时的兜底假分数），
        # 安全网不用 mock 分数否决 LLM 的 PASS，否则 LLM 判过也会被 mock 拉回 REJECT。
        _mock_untrusted_safety = (
            not _config.pipeline.get("trust_mock_heuristic", False)
            and critic_report.get("_backend_path") == "mock_heuristic"
        )

        if scores and not _mock_untrusted_safety:
            avg_score = sum(v for v in scores.values() if isinstance(v, (int, float))) / max(1, len(scores))
            safety_min = _config.pipeline.get("judge_safety_net_avg_min", 5.5)
            if result.get("decision") == "PASS" and avg_score < safety_min:
                result["decision"] = "REJECT"
                result["reject_reason"] = (f"安全网: Critic 平均分 {avg_score:.1f} 低于安全阈值 {safety_min}，"
                                           f"但 Judge 判定通过，疑似 LLM 幻觉。")

        # style rule 提取（仅旧 LLM 路径支持；typesafe 不生成新规则，避免误判）
        if result.get("decision") == "PASS" and "new_style_rule" in result and result["new_style_rule"]:
            rule = result["new_style_rule"]
            desc = (rule.get("rule_description") or "").strip()
            if desc:
                import hashlib
                fp = hashlib.sha1(desc.encode("utf-8")).hexdigest()[:12]
                self.db.add_decision(
                    level=DecisionLevel.STYLE,
                    source=f"style_{fp}",
                    translation=desc,
                    reason=rule.get("reason", "Judge Agent 动态提炼"),
                    affected_chunks=None,
                )

        result["_backend_path"] = "legacy"
        return result


# ────────────────────────────────────────────────────────────────────
