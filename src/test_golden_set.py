"""
SagaTrans 黄金测试集（Golden Set）跑分与质量门禁

源：原整合脚本 §8 中的 GoldenSet 相关部分，拆分到独立模块。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import config as _config
from src.core.decision_engine import DecisionEngine
from src.agents.reference_agent import ReferenceAgent
from src.agents.rewriter_agent import LiteraryRewriterAgent
from src.agents.critic_agent import CriticAgent
from src.agents.judge_agent import JudgeAgent
from src.utils.chunker import SmartChunker
from src.utils.llm_adapter import (
    get_llm_client,
    get_role_model_name,
)


# Round 118 P2-7 / Round 128: 黄金集 A/B — 直译阶段「模型 × effort」四组合.
# 重写阶段跟随 task_routing（本轮不切换），衡量的是“直译×重写”全流程结果。
GOLDEN_AB_MATRIX: List[Dict[str, Any]] = [
    {"name": "literal_effort_none", "role": "literal_translator", "effort": None},
    {"name": "literal_effort_xhigh", "role": "literal_translator", "effort": "xhigh"},
    {"name": "primary_effort_none", "role": "primary", "effort": None},
    {"name": "primary_effort_max", "role": "primary", "effort": "max"},
]


class GoldenSetEvaluator:
    """黄金测试集评估器 - 风格坍缩率量化"""

    def __init__(self, db_path: Optional[str] = None, literal_role: Optional[str] = None):
        # Round 118 P2-7: literal_role 可被 A/B 组合覆盖（DeepSeek vs MiniMax）
        self.literal_role = literal_role or _config.get_task_role("literal_translation")
        self.llm = get_llm_client(self.literal_role)
        shared_db = DecisionEngine(db_path=db_path) if db_path else DecisionEngine()
        self.db = shared_db
        self.ref_agent = ReferenceAgent(shared_db, chapter_id="golden_test")
        self.rewriter_agent = LiteraryRewriterAgent(shared_db)
        self.critic_agent = CriticAgent()
        self.judge_agent = JudgeAgent(shared_db)

    def evaluate_style_collapse(self, source_text: str, translated_text: str) -> dict:
        """评估风格坍缩率：对比原文与译文的风格特征"""
        source_stats = self._analyze_style(source_text)
        trans_stats = self._analyze_style(translated_text)

        preservation = {}
        for key in source_stats:
            if source_stats[key] > 0:
                ratio = min(trans_stats[key] / source_stats[key], 1.0)
                preservation[key] = ratio

        avg_preservation = sum(preservation.values()) / len(preservation) if preservation else 0
        collapse_rate = 1.0 - avg_preservation

        return {
            "source_style": source_stats,
            "translated_style": trans_stats,
            "preservation_per_dimension": preservation,
            "average_preservation": avg_preservation,
            "style_collapse_rate": collapse_rate
        }

    def _analyze_style(self, text: str) -> dict:
        """提取文本风格统计特征"""
        # 语言检测：是否包含中文字符
        has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))
        sent_split = '。' if has_cjk else r'[.!?]+'
        sentences = [s.strip() for s in re.split(sent_split, text) if s.strip()]

        avg_sent_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0

        words = text.split()
        _stop_words = {'的', '了', '是', '我', '在', '有', '和', '为', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        content_words = [w for w in words if len(w) > 1 and w.lower() not in _stop_words]
        vocab_density = len(content_words) / len(words) if words else 0

        rhetorical_markers = ['如', '似', '仿佛', '好像', '像', '犹如', '宛若', 'like', 'as if', 'as though', 'seem', 'appear', 'resemble']
        rhetorical_count = sum(text.lower().count(m) for m in rhetorical_markers)
        rhetorical_density = rhetorical_count / len(sentences) if sentences else 0

        punctuation = ['，', '。', '；', '：', '——', '…', '\u201c', '\u201d', '\u2018', '\u2019', ',', ';', ':', '—', '...', '"', "'", '?', '!']
        punct_count = sum(text.count(p) for p in punctuation)
        punct_density = punct_count / len(text) * 1000 if text else 0

        return {
            "avg_sentence_length": avg_sent_len,
            "vocabulary_density": vocab_density,
            "rhetorical_density": rhetorical_density,
            "punctuation_density": punct_density
        }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的 Levenshtein 编辑距离（手写，无外部依赖）"""
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


def run_golden_test(
    ab_override: Optional[Dict[str, Any]] = None,
    report_path: Optional[Any] = None,
):
    """运行黄金测试集（需要 input/golden/hyperion_5k.md）.

    Round 118 P2-7 / Round 128: ab_override={"name","role","effort"} 时按 A/B 组合
    覆盖直译阶段的模型与 effort 档位；report_path 可指定本轮报告输出路径
    （默认 output/golden_test_report.json）。
    """
    ab = dict(ab_override or {})
    print("=" * 60)
    print("🧪 SagaTrans 黄金测试集跑分")
    print("=" * 60)

    root_dir = Path(__file__).resolve().parent.parent
    test_file = root_dir / "input" / "golden" / "hyperion_5k.md"
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return []

    with open(test_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    print(f"📖 测试文本长度: {len(source_text)} 字符")

    # 使用配置的 chunker 参数
    chunker_cfg = _config.chunker
    chunker = SmartChunker(
        soft_limit=chunker_cfg.get("soft_limit", 1000),
        hard_limit=chunker_cfg.get("hard_limit", 2500)
    )
    chunks = chunker.split_markdown(source_text)
    print(f"✂️ 切分为 {len(chunks)} 个块")

    literal_role = ab.get("role") or _config.get_task_role("literal_translation")
    evaluator = GoldenSetEvaluator(
        db_path=str(root_dir / "db" / "decision_db.sqlite"),
        literal_role=literal_role,
    )
    if ab:
        print(
            f"🧪 A/B 组合: {ab.get('name')} "
            f"(role={literal_role}, effort={ab.get('effort') or 'none'})"
        )
    all_results = []

    # 从配置获取风格指南
    style_guide = _config.style_guide

    for i, chunk in enumerate(chunks):
        chunk_id = f"golden_chunk{i:03d}"
        try:
            print(f"\n📦 处理第 {i+1}/{len(chunks)} 块 ({len(chunk)} 字符)...")

            evaluator.ref_agent.process_chunk(chunk_id, chunk, affected_chunks=[chunk_id])

            genre = (_config.style_guide or {}).get("genre") or "文学小说"
            gloss = evaluator.db.format_prompt_context(source_text=chunk)
            raw_prompt = (
                f"请将以下{genre}片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n"
                f"{gloss}\n\n【原文】\n{chunk}"
            )
            # 使用配置路由 literal_translation（A/B 时按组合覆盖 role + effort）
            model_key, params = _config.resolve_task_model("literal_translation")
            params = dict(params)
            if ab:
                ab_effort = ab.get("effort")
                if ab_effort:
                    params["effort"] = str(ab_effort).strip().lower()
                else:
                    params.pop("effort", None)
            model_name = get_role_model_name(literal_role) or model_key
            raw_text = evaluator.llm.generate(raw_prompt, model_name=model_name, **params)

            raw_result = evaluator.rewriter_agent.process_chunk(chunk_id, raw_text, style_guide, chunk)
            lit_text, _ = raw_result if isinstance(raw_result, tuple) else (raw_result, None)

            critic_report = evaluator.critic_agent.process_chunk(chunk_id, chunk, raw_text, lit_text, style_guide)

            judge_result = evaluator.judge_agent.process_chunk(chunk_id, chunk, lit_text, critic_report, affected_chunks=[chunk_id])

            final_text = lit_text
            style_eval = evaluator.evaluate_style_collapse(chunk, final_text)

            result = {
                "chunk_id": chunk_id,
                "source_length": len(chunk),
                "critic_scores": critic_report.get("scores", {}),
                "judge_decision": judge_result.get("decision"),
                "judge_backend": judge_result.get("_backend_path"),
                "confidence_source": _extract_confidence_source(judge_result),
                "literal_role": literal_role,
                "literal_effort": params.get("effort"),
                "ab": ab.get("name") if ab else None,
                "style_collapse_rate": style_eval["style_collapse_rate"],
                "style_preservation": style_eval["average_preservation"],
                "preservation_details": style_eval["preservation_per_dimension"]
            }
            all_results.append(result)

            print(f"  Critic: scores={critic_report.get('scores')}")
            print(f"  Judge: {judge_result.get('decision')}")
            print(f"  风格坍缩率: {style_eval['style_collapse_rate']:.2%}")
            print(f"  风格保持度: {style_eval['average_preservation']:.2%}")
        except Exception as e:
            print(f"⚠️ 第 {i+1} 块处理失败，跳过: {e}")
            all_results.append({"chunk_id": chunk_id, "error": str(e)})
        done = i + 1
        print(f"  📊 [{done}/{len(chunks)}] 完成 ({done/max(len(chunks),1)*100:.0f}%)")

    # 汇总报告
    print("\n" + "=" * 60)
    print("📊 黄金测试集汇总报告")
    print("=" * 60)

    total_chunks = len(all_results)
    error_count = sum(1 for r in all_results if "error" in r)
    # 仅对成功处理的块计算聚合指标，跳过错误条目避免 KeyError
    valid_results = [r for r in all_results if "error" not in r]
    pass_count = sum(1 for r in valid_results if r["judge_decision"] == "PASS")
    avg_collapse = sum(r["style_collapse_rate"] for r in valid_results) / max(len(valid_results), 1)
    avg_preservation = sum(r["style_preservation"] for r in valid_results) / max(len(valid_results), 1)

    print(f"总块数: {total_chunks}")
    print(f"错误数: {error_count}")
    print(f"通过数: {pass_count} ({pass_count/max(total_chunks,1)*100:.1f}%)")
    print(f"平均风格坍缩率: {avg_collapse:.2%}")
    print(f"平均风格保持度: {avg_preservation:.2%}")

    dims = ["avg_sentence_length", "vocabulary_density", "rhetorical_density", "punctuation_density"]
    for dim in dims:
        dim_avg = sum(r.get("preservation_details", {}).get(dim, 0) for r in valid_results) / max(len(valid_results), 1)
        print(f"  {dim}: {dim_avg:.2%}")

    output_file = Path(report_path) if report_path else (root_dir / "output" / "golden_test_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_chunks": total_chunks,
                "error_count": error_count,
                "pass_rate": pass_count / max(total_chunks, 1),
                "avg_style_collapse_rate": avg_collapse,
                "avg_style_preservation": avg_preservation,
                "ab_stats": summarize_golden_results(all_results),
            },
            "details": all_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细报告已保存: {output_file}")
    return all_results


def _extract_confidence_source(judge_result: Dict[str, Any]) -> str:
    """Round 118 P1-1/P2-7: 从 judge 结果提取概率来源, 供 A/B 分组统计.

    - typesafe verdict 自带 confidence_source（jev / openai_compat / mock_heuristic / _uncalibrated）
    - 旧 LLM 路径无 verdict → "legacy" 后端映射为 "_uncalibrated"（非 Jev 校准）
    """
    verdict = (judge_result or {}).get("_typesafe_verdict") or {}
    src = verdict.get("confidence_source")
    if src:
        return str(src)
    backend = (judge_result or {}).get("_backend_path") or "unknown"
    if backend == "legacy":
        return "_uncalibrated"
    return str(backend)


def summarize_golden_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Round 118 P2-7: 聚合单轮 golden 结果, 按 verdict 来源分组统计.

    calibrated_ratio = confidence_source=="jev" 的有效样本占比。降级环境
    （无 TYPESAFE_API_KEY / 后端 fallback）的高通过率不得作为 A/B 结论或门禁依据。
    """
    valid = [r for r in results if "error" not in r]
    total = len(results)
    valid_count = len(valid)
    pass_count = sum(1 for r in valid if r.get("judge_decision") == "PASS")

    def _avg(values: List[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    chunk_avgs: List[float] = []
    for r in valid:
        scores = r.get("critic_scores") or {}
        nums = [v for v in scores.values() if isinstance(v, (int, float))]
        if nums:
            chunk_avgs.append(sum(nums) / len(nums))

    backend_dist: Dict[str, int] = {}
    confidence_dist: Dict[str, int] = {}
    for r in valid:
        backend = str(r.get("judge_backend") or "unknown")
        backend_dist[backend] = backend_dist.get(backend, 0) + 1
        conf = str(r.get("confidence_source") or "unknown")
        confidence_dist[conf] = confidence_dist.get(conf, 0) + 1

    return {
        "total": total,
        "valid_count": valid_count,
        "error_count": total - valid_count,
        "pass_count": pass_count,
        "pass_rate": (pass_count / valid_count) if valid_count else 0.0,
        "avg_critic_score": _avg(chunk_avgs),
        "avg_style_preservation": _avg([float(r.get("style_preservation", 0.0)) for r in valid]),
        "avg_style_collapse_rate": _avg([float(r.get("style_collapse_rate", 0.0)) for r in valid]),
        "judge_backend_dist": backend_dist,
        "confidence_source_dist": confidence_dist,
        "calibrated_ratio": (confidence_dist.get("jev", 0) / valid_count) if valid_count else 0.0,
    }


def recommend_golden_combo(
    combos: Dict[str, Dict[str, Any]],
    min_calibrated: float = 0.5,
) -> Dict[str, Any]:
    """Round 118 P2-7: 在 Jev 校准占比达标的组合中选推荐（质量优先排序）.

    排序键: (pass_rate, avg_critic_score, avg_style_preservation) 依次取高。
    无达标组合时明确返回 name=None — 拒绝用降级数据得出结论。
    """
    calibrated = [
        name for name, c in combos.items()
        if (c.get("summary") or {}).get("valid_count", 0) > 0
        and (c.get("summary") or {}).get("calibrated_ratio", 0.0) >= min_calibrated
    ]
    if not calibrated:
        no_samples = all(
            (c.get("summary") or {}).get("valid_count", 0) == 0 for c in combos.values()
        )
        if no_samples:
            return {
                "name": None,
                "reason": "no_valid_samples",
                "note": "A/B 无有效样本（黄金集缺失或全部处理失败），无法判定",
            }
        return {
            "name": None,
            "reason": "no_calibrated_combo",
            "note": "所有组合的 verdict 均非 Jev 校准（环境降级），A/B 结论与门禁不可用",
        }

    def _key(name: str) -> Tuple[float, float, float]:
        s = combos[name]["summary"]
        return (
            float(s.get("pass_rate", 0.0)),
            float(s.get("avg_critic_score", 0.0)),
            float(s.get("avg_style_preservation", 0.0)),
        )

    best = max(calibrated, key=_key)
    return {
        "name": best,
        "reason": "best_calibrated_pass_rate",
        "calibrated_candidates": calibrated,
    }


def run_golden_ab(
    matrix: Optional[List[Dict[str, Any]]] = None,
    report_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """Round 118 P2-7 / Round 128: 黄金集 A/B — 直译模型 × effort 档位对比.

    每个组合跑完整 直译→重写→critic→judge 流程（重写阶段按 task_routing 定，
    A/B 只切直译阶段），统计 pass_rate / critic 均分 / 风格保持度，并按
    judge `_backend_path` 与 `confidence_source` 分组；仅 Jev 校准占比 ≥
    config.pipeline.golden_ab.min_calibrated_ratio 的组合参与推荐。
    """
    matrix = [dict(m) for m in (matrix or GOLDEN_AB_MATRIX)]
    root_dir = Path(__file__).resolve().parent.parent
    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    combos: Dict[str, Dict[str, Any]] = {}
    for idx, combo in enumerate(matrix):
        name = str(combo.get("name") or f"combo_{idx}")
        print("\n" + "=" * 60)
        print(f"🧪 Golden A/B 组合 [{name}]: role={combo.get('role')}, effort={combo.get('effort') or 'none'}")
        print("=" * 60)
        results = run_golden_test(
            ab_override=combo,
            report_path=output_dir / f"golden_ab_{name}.json",
        )
        combos[name] = {
            "config": combo,
            "summary": summarize_golden_results(results),
            "results": results,
        }

    min_calibrated = float((_config.pipeline.get("golden_ab", {}) or {}).get("min_calibrated_ratio", 0.5))
    recommendation = recommend_golden_combo(combos, min_calibrated=min_calibrated)
    report = {
        "matrix": matrix,
        "min_calibrated_ratio": min_calibrated,
        "combos": combos,
        "recommendation": recommendation,
    }

    out = Path(report_path) if report_path else (output_dir / "golden_ab_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("📊 Golden A/B 汇总（仅 Jev 校准组合可作结论）")
    print("=" * 60)
    for name, c in combos.items():
        s = c["summary"]
        print(
            f"  {name:<18} pass={s['pass_count']}/{s['valid_count']} ({s['pass_rate']:.1%}) "
            f"critic={s['avg_critic_score']:.2f} 风格保持={s['avg_style_preservation']:.1%} "
            f"校准占比={s['calibrated_ratio']:.0%} backend={s['judge_backend_dist']}"
        )
    if recommendation.get("name"):
        print(f"✅ 推荐组合: {recommendation['name']} ({recommendation.get('reason')})")
    else:
        print(f"⚠️ 无推荐: {recommendation.get('note')}")
    print(f"📄 A/B 报告已保存: {out}")
    return report

