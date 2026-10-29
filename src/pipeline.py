"""SagaTrans 翻译流水线（散装源）。

本模块定义 ``TranslationPipeline``，由 ``src/main.py`` 与 ``src/translator_agent.py``
（整合脚本）共同调用。**事实源**：散装 ``src/`` 目录；**整合副本**：
``src/translator_agent.py`` 由 ``scripts/build_integrated.py`` 自动生成。

Round 71-80 修改均在本模块体现，整合脚本与之保持一致。
"""
import json
import re
import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import config as _config

from src.core.scheduler import TaskScheduler, TaskState
from src.core.decision_engine import (
    DecisionEngine,
    get_sibling_canonicals_by_source,
    query_reviewable_entities,
)
from src.utils.llm_adapter import (
    get_llm_client_for_task,
    get_llm_client,
    get_role_model_name,
    robust_json_loads,
    _map_effort_for_fallback,
)
from src.utils.text_match import entity_source_name_in_text, entity_source_name_spans
from src.agents.reference_agent import ReferenceAgent
from src.agents.rewriter_agent import LiteraryRewriterAgent
from src.agents.critic_agent import CriticAgent
from src.agents.judge_agent import JudgeAgent
# Round 58 修复: pipeline.py 需要从 src.splitters re-export _normalize_scores,
# 测试 (tests/utils/test_normalize_scores.py::TestPipelineImport) 直接
# `from src.pipeline import _normalize_scores`. 原写法不会把名字挂到
# src.pipeline 命名空间, 触发 ImportError. 改为 as-import 形式保证
# src.pipeline._normalize_scores is src.splitters._normalize_scores.
from src.splitters import _normalize_scores as _normalize_scores
from src.utils.text_match import (
    case_insensitive_find,
    compute_cjk_ratio,
    extract_reject_categories,
    reset_cjk_ratio_cache,
)


class TranslationPipeline:
    def __init__(self, chapter_id: str):
        self.chapter_id = chapter_id
        # 使用绝对路径定位数据库（相对于此文件的上级目录）
        self._root_dir = Path(__file__).resolve().parent.parent
        self.scheduler = TaskScheduler(db_path=str(self._root_dir / "db" / "workflow.db"))
        self.decision_engine = DecisionEngine(
            db_path=str(self._root_dir / "db" / "decision_db.sqlite"),
            scheduler_factory=lambda: self.scheduler
        )

        # Round 117 P0: 动态风格剖面加载（StyleProfiler 产物）→ 运行时 style_guide 覆盖层。
        # 剖面由 `python3 -m src.translator_agent style-profile` 生成并落盘 input/book_profile.json;
        # 缺失/损坏时安静跳过（保持 config.yaml 基线, 不注入任何作品特定先验）。
        self.book_profile = None
        try:
            from src.style_profiler import load_profile, profile_to_style_guide
            _prof_path = self._root_dir / "input" / "book_profile.json"
            _prof = load_profile(_prof_path)
            if _prof:
                self.book_profile = _prof
                _config.set_runtime_style_guide(profile_to_style_guide(_prof, chapter_id=chapter_id))
                print(f"🎨 [StyleProfiler] 已加载作品风格剖面 ← {_prof_path.name} "
                      f"(source={_prof.get('source')}, confidence={_prof.get('confidence')})")
        except Exception as _prof_err:
            print(f"⚠️ [StyleProfiler] 剖面加载失败，忽略: {_prof_err}", file=sys.stderr)

        self.llm = get_llm_client_for_task("literal_translation")
        self.literal_role = _config.get_task_role("literal_translation")

        # 实例化 Agents
        self.ref_agent = ReferenceAgent(self.decision_engine, chapter_id)
        self.rewriter_agent = LiteraryRewriterAgent(self.decision_engine)
        self.critic_agent = CriticAgent()
        self.judge_agent = JudgeAgent(self.decision_engine)

        # 绝对路径 output
        self.output_dir = self._root_dir / "output" / chapter_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Round 13: LLM 调用计数器（按 task_type 分组）
        # Round 65 修订：每次 process_chunk 调用 + quality_retries 次重试 = 实际 LLM 调用次数
        # （之前 +=1 漏算 retry，导致 cost_report 严重低估 literary_rewriting / critic_scoring / judge_decision 成本）
        self._llm_call_counter = {
            "reference_extraction": 0,
            "literal_translation": 0,
            "literary_rewriting": 0,
            "critic_scoring": 0,
            "judge_decision": 0,
        }

        # 从配置加载风格指南
        # Round 79: 之前只 snapshot 3 个键 (avg_sentence_length / lexicon_preference /
        # author_priority_ratio), 缺 genre/persona/work_type/register_default/
        # footnote_policy/forbidden_modernisms 6+ 个键. partial dict 传给 critic_agent
        # 被注入 critic prompt, 导致 critic 看不到 persona/禁用词/时代语域约束, 影响评分准确性.
        # 现在直接用完整 _config.style_guide dict, 让下游 Agent 看到所有约束.
        # (rewriter_agent 内部本来就用 _config.style_guide.get(...) 读, 不受影响.)
        self.style_guide = dict(_config.style_guide) if _config.style_guide else {}

        # Round 71: 跨人物冲突 sibling 缓存. 每次审计调用 _check_entity_consistency 时,
        # 第一次会从 entity_registry 计算 sibling 集合, 后续复用. 在 _process_auditing_batch
        # 入口重置, 避免跨章节脏数据.
        self._sibling_canonicals_cache: Optional[Dict[str, Set[str]]] = None

        # Round 119 L1: 韵文通道运行时缓存（tags/manifest 惰性加载；无韵文时零开销）
        self._verse_tags_cache: Optional[dict] = None
        self._verse_manifest_cache: Optional[dict] = None
        self._verse_note_cache: Dict[str, str] = {}

        # Round 92: 最佳文学稿 fallback 状态. 跨 retry 追踪最高 critic 平均分对应的 lit_text,
        # fallback 时 (stagnation / oscillation / max_retries / safety_net) 用此版本,
        # 而不是当前被 critic 拒掉的版本. 用户看到的是"最高质量的版本"而不是"最差一次的版本".
        # key=chunk_id, value=(avg_score, lit_text)
        self._best_lit: Dict[str, Tuple[float, str]] = {}
        # Round 118 P2-8: 候选稿池（top-N by critic avg），供 Jev 配对仲裁做最终选择。
        # key=chunk_id, value=List[(avg_score, lit_text)] 按 avg 降序、文本去重
        self._lit_candidates: Dict[str, List[Tuple[float, str]]] = {}
        self._pairwise_info: Dict[str, Dict[str, Any]] = {}

        # Round 93: 每 chunk 各 retry 的 critic avg_score 历史. 用于检测"连续下降"模式
        # (e.g., retry 0=6.5, retry 1=6.0, retry 2=5.5), 提前 fallback 节省 LLM 调用.
        # key=chunk_id, value=List[float] 按 retry 顺序追加
        self._score_history: Dict[str, List[float]] = {}
        # Round 98 P1: 实体违规计数历史. 类似 _score_history, 但跟踪每次 audit 后
        # serious 实体违规数. 用于检测"实体违规连续 N 次未修复"模式, 提前 fallback
        # 节省 rewriter+critic LLM 调用. ch32 实测每次 retry 仍 serious=1-2,
        # rewriter 反复失败, 早停可在 quality_retries=2 时止损.
        self._entity_violation_history: Dict[str, List[int]] = {}
        # Round 99 P1: per-chunk entity_registry 缓存. 同一 chunk 的所有 retry 用同一份 source_text,
        # registry 在 chapter 内不变 (除非 ext 模式重载). 缓存避免每次 retry 重跑 query_reviewable_entities
        # SQL + JSON parse. chapter 切换时清空 (在 run() 重置, 见 _reset_per_run_cache).
        self._entity_registry_cache: Dict[str, Dict[str, str]] = {}
        self._entity_variants_cache: Dict[str, Dict[str, List[str]]] = {}
        # Round 99 P2: intermediate 文件读缓存. key=(chunk_id, step), value=loaded data.
        # 同 chunk × step 多次加载走内存, 写时自动 invalidate (见 _save_intermediate).
        self._intermediate_cache: Dict[Tuple[str, str], object] = {}
        # Round 100 P1: narrative_memory 读缓存. _load_prior_memory 每次 rewriter retry
        # 都重读 _memory.jsonl + JSON parse + filter. 同 chunk 跨 retry 内容不变, 缓存命中.
        # _append_chunk_memory 时 invalidate 对应 chunk 的 cache entry (内容变了).
        self._narrative_memory_cache: Dict[str, str] = {}
        # Round 100 P2: _compute_avg_score 缓存 (key=id(critic_report))
        self._avg_score_cache: Dict[int, float] = {}

    # Round 92: 最佳文学稿追踪 + fallback 选择
    # Round 100 P2: 实例方法 (非 staticmethod), 因为需要访问 self._avg_score_cache
    # Round 103 P1: 改为 classmethod, 兼容测试
    # `TranslationPipeline._compute_avg_score(cr)` (无 self) 调用.
    # 实例方法 `self._compute_avg_score(cr)` 也工作 (Python 自动传 cls=self).
    #
    # Round 103 P3: 删除 Round 100 P2 的 id() 缓存. 原设计假设 id() 在
    # critic_report 生命周期内稳定, 但 CPython 在 dict 被 GC 后可复用 id() —
    # 实测导致测试 test_zero_filtered_at_update_time 出现 [7.0, 7.0] 而非
    # [7.0, 6.0] 的 bug. avg 计算本身是 O(6) sum/len, 微秒级, 不值得为它
    # 引入 id() 缓存的 correctness risk.
    #
    # 如未来确有性能需求, 用 weakref.WeakKeyDictionary 替代 (但 dict 不
    # 支持 weakref, 须包装或换用 @dataclass(frozen=True) CriticReport 类).

    @classmethod
    def _compute_avg_score(cls, critic_report: dict) -> float:
        """从 critic_report 提取所有数值维度 score 的算术平均. 无 scores 返回 0.0.

        Round 92: 引入. Round 100 P2: 加 id() 缓存 (Round 103 P3 移除).
        """
        if not isinstance(critic_report, dict):
            return 0.0
        scores = critic_report.get("scores", {}) or {}
        numeric = [v for v in scores.values() if isinstance(v, (int, float))]
        return (sum(numeric) / len(numeric)) if numeric else 0.0

    def _update_best_lit(self, chunk_id: str, lit_text: str, critic_report: dict) -> None:
        """把当前 retry 的 (avg_score, lit_text) 与已存最佳比较, 若更高则更新.

        设计原则:
          - 即使 critic reject (judge=REJECT) 也保存当前版本 — 因为它的 avg_score 可能比之前高
          - 即使 critic pass (judge=PASS) 也保存 — 防御性, 万一后续 retry 不小心覆盖更高分版本
          - 当 chunk 完成 (judge PASS, fallback 触发, 或 chapter end) 时, _best_lit 保留
            直至下一轮 processing 清理
          - Round 93: 同时维护 _score_history[chunk_id], 追加当前 avg_score 用于"连续下降"检测
        """
        avg = self._compute_avg_score(critic_report)
        existing = self._best_lit.get(chunk_id)
        if existing is None or avg > existing[0]:
            self._best_lit[chunk_id] = (avg, lit_text)
        # Round 118 P2-8: 维护候选池（去重，按 avg 降序截断）
        if lit_text:
            self._ensure_candidate_pools()
            cands = self._lit_candidates.setdefault(chunk_id, [])
            if all(t != lit_text for _s, t in cands):
                cands.append((avg, lit_text))
                cands.sort(key=lambda st: st[0], reverse=True)
                max_cands = int(
                    (_config.pipeline.get("pairwise_arbitration", {}) or {})
                    .get("max_candidates", 4)
                )
                del cands[max(1, max_cands):]
        # Round 93: 同步记录 score 历史 (无论是否 best). 仅记录数值 > 0 的样本,
        # critic 评分空时 (avg=0.0) 不污染历史.
        if avg > 0.0:
            self._score_history.setdefault(chunk_id, []).append(avg)

    def _is_declining_streak(self, chunk_id: str, current_score: float, consecutive: int = 3) -> bool:
        """检查 chunk 是否处于"连续下降"模式.

        Returns True 当 _score_history[chunk_id] 末尾 `consecutive` 个分数严格递减.

        设计:
          - 默认 consecutive=3: 连续 3 次 retry 分数递减视为不可能改善, 提前 fallback
          - 边界: history 长度 < consecutive → 返回 False (样本不足, 不下结论)
          - 例外: 任一分数 == 0.0 (无评分) 不计入 (避免 critic 返回空误触发)
          - 严格递减: s[i-1] > s[i] (非 ">=", 允许偶然相等)

        用法: judge 阶段 REJECT 后, 调用此方法判断是否直接 fallback (跳过 rewriter).
        """
        hist = self._score_history.get(chunk_id, [])
        # 末尾 consecutive 个 + 当前 score = consecutive 个样本
        samples = hist[-consecutive + 1:] + [current_score] if consecutive > 1 else [current_score]
        if len(samples) < consecutive:
            return False
        # 仅保留 > 0 的样本 (过滤 critic 空评分)
        samples = [s for s in samples if s > 0.0]
        if len(samples) < consecutive:
            return False
        # 严格递减检查
        return all(samples[i] > samples[i + 1] for i in range(len(samples) - 1))

    # Round 98 P2: 非进展 streak — 比 _is_declining_streak 更激进
    # _is_declining_streak 要求严格递减 (s[i-1] > s[i]), 遇到「持平」场景不触发
    # (rewriter 输出虽然不恶化但也不进步, 浪费 LLM). 本检查: 末尾 N 个样本任意两次间
    # 都满足 next <= prev (即单调不增), 即触发. 实际语义: 连续 N 次 retry 没有净改善.
    def _is_no_progress_streak(self, chunk_id: str, current_score: float, window: int = 3) -> bool:
        """检查 chunk 是否处于「连续不增」模式 (含持平).

        Returns True 当 _score_history[chunk_id] 末尾 window 个样本与当前 score 形成
        window 个连续非增对比 (即: s[0] >= s[1] >= ... >= s[window-1] >= current).

        设计:
          - 默认 window=3: 与 _is_declining_streak 默认值一致 (可独立调)
          - 边界: history 长度 < window → 返回 False (样本不足)
          - 例外: 任一分数 == 0.0 不计入 (过滤 critic 空评分)
          - 单调不增: s[i] <= s[i+1] (允许相等, 与 _is_declining_streak 的严格递减互补)
        """
        if window <= 1:
            return False
        hist = self._score_history.get(chunk_id, [])
        # 末尾 window-1 个历史 + 当前 score = window 个样本
        samples = hist[-(window - 1):] + [current_score] if window > 1 else [current_score]
        if len(samples) < window:
            return False
        # 仅保留 > 0 的样本 (过滤 critic 空评分)
        samples = [s for s in samples if s > 0.0]
        if len(samples) < window:
            return False
        # 单调不增检查 (next <= prev)
        return all(samples[i] >= samples[i + 1] for i in range(len(samples) - 1))

    # Round 99 P1: chapter 切换时清空 per-chunk 缓存, 避免跨 chapter 脏数据.
    def _reset_per_run_cache(self) -> None:
        """chapter 切换时调用 — 清空 entity_registry 缓存和 entity_violation history.

        设计: 与 sibling_canonicals_cache 同样在 _process_auditing_batch 起手清空,
        但 entity_violation_history 与 score_history 是累积统计, 不应在 batch 间清空
        (它们按 chunk_id 索引, 不同 chapter 用不同 chunk_id, 互不干扰).
        这里只清空 _entity_registry_cache 和 _entity_variants_cache, 因为它们
        按 chunk_id 索引但数据可能因 chapter 切换而不准确.
        """
        # 容错: 若字段未初始化 (例如测试桩或外部调用), 跳过清空
        if hasattr(self, '_entity_registry_cache') and self._entity_registry_cache is not None:
            self._entity_registry_cache.clear()
        if hasattr(self, '_entity_variants_cache') and self._entity_variants_cache is not None:
            self._entity_variants_cache.clear()
        if hasattr(self, '_intermediate_cache') and self._intermediate_cache is not None:
            self._intermediate_cache.clear()
        if hasattr(self, '_narrative_memory_cache') and self._narrative_memory_cache is not None:
            self._narrative_memory_cache.clear()
        # Round 103 P3: 删除 _avg_score_cache 实例字段 (Round 100 P2 引入)
        # 在 _reset_per_run_cache 中清理. 现 _compute_avg_score 不再缓存, 无需清理.
        # 保留 hasattr 检查向后兼容 (若未来重新引入缓存).
        if hasattr(self, '_avg_score_cache') and self._avg_score_cache is not None:
            self._avg_score_cache.clear()
        # Round 102 P1: cjk_ratio 缓存是模块级 (非实例字段), 跨 chapter 累积无意义, 同步清空.
        try:
            reset_cjk_ratio_cache()
        except (NameError, AttributeError):
            pass  # 单文件 shim 模式下模块级 cache 名称不同, 容错
        # 注: _intermediate_cache 字段在 __init__ 初始化 (line 3386),
        #     这里只清空, 不重复声明.

    # ── Round 116: 停滞换模型（生成端多样性）────────────────────────────────
    _RW_SWITCH_STEP = "rewrite_switch"

    def _read_rewriter_switch_state(
        self, chunk_id: str, base_role: str, alt_role: str, max_switches: int
    ) -> dict:
        """读取换模型 sidecar，决定本轮润色应使用的角色。

        sidecar 格式: {"switches": [{"from","to","retry","reason","timestamp"}, ...]}
        语义：当前角色 = 最后一次 switch 的 to（无记录则 base_role）；
              next_role = base/alt 交替；can_switch 受 max_switches 上限约束。
        """
        switches: list = []
        try:
            data = self._load_intermediate(chunk_id, self._RW_SWITCH_STEP)
            if isinstance(data, dict):
                switches = list(data.get("switches", []) or [])
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
            switches = []
        current_role = base_role
        for sw in switches:
            to_role = (sw or {}).get("to")
            if to_role:
                current_role = to_role
        next_role = alt_role if current_role != alt_role else base_role
        try:
            max_sw = int(max_switches)
        except (TypeError, ValueError):
            max_sw = 1
        return {
            "current_role": current_role,
            "switches": len(switches),
            "can_switch": len(switches) < max(0, max_sw),
            "next_role": next_role,
        }

    def _record_rewriter_model_switch(
        self, chunk_id: str, from_role: str, to_role: str, retry: int, reason: str
    ) -> None:
        """追加一条换模型记录（sidecar），供下一轮 _read_rewriter_switch_state 读取。"""
        data: dict = {}
        try:
            loaded = self._load_intermediate(chunk_id, self._RW_SWITCH_STEP)
            if isinstance(loaded, dict):
                data = loaded
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
            data = {}
        switches = list(data.get("switches", []) or [])
        switches.append({
            "from": from_role,
            "to": to_role,
            "retry": retry,
            "reason": (reason or "")[:200],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._save_intermediate(chunk_id, self._RW_SWITCH_STEP, {"switches": switches})

    def _get_best_lit(self, chunk_id: str, fallback_text: str) -> str:
        """返回该 chunk 跨 retry 的最佳文学稿; 若无最佳 (例如从未跑 critic), 返回 fallback_text."""
        entry = self._best_lit.get(chunk_id)
        if entry is None or not entry[1]:
            return fallback_text
        return entry[1]

    def _ensure_candidate_pools(self) -> None:
        """Round 118 P2-8: 惰性初始化候选池 (兼容半初始化实例, 例如测试用 __new__ 构造)."""
        if getattr(self, "_lit_candidates", None) is None:
            self._lit_candidates = {}
        if getattr(self, "_pairwise_info", None) is None:
            self._pairwise_info = {}

    def _pairwise_pick_best(
        self,
        chunk_id: str,
        fallback_text: str,
        source_text: Optional[str] = None,
        raw_text: Optional[str] = None,
    ) -> str:
        """Round 118 P2-8: 早停/兜底定稿时, 用 Jev 配对仲裁在 top-2 候选间选更优稿.

        只在定稿路径调用; 任何失败/平票/不可用都回退 _get_best_lit（保持原行为）。
        仲裁只做"选稿", 不改变裁决阈值语义; 结果写入 self._pairwise_info 供审计。
        """
        best = self._get_best_lit(chunk_id, fallback_text)
        cfg = _config.pipeline.get("pairwise_arbitration", {}) or {}
        if not cfg.get("enabled", True) or not source_text:
            return best
        self._ensure_candidate_pools()
        cands = self._lit_candidates.get(chunk_id, [])
        if len(cands) < 2:
            return best
        (score_a, text_a), (score_b, text_b) = cands[0], cands[1]
        if not text_a or not text_b or text_a == text_b:
            return best
        try:
            if raw_text is None:
                try:
                    raw_text = self._load_intermediate(chunk_id, "raw")
                except Exception:
                    raw_text = ""
            from src.judges.judge import build_state
            from src.judges.primitives import build_question_set
            from src.judges.typesafe_backend import get_backend
            from src.judges.pairwise import arbitrate_answers

            style = dict(getattr(self, "style_guide", None) or {})
            entity_registry = self._build_entity_registry_for_rewriter(source_text)
            questions = build_question_set()
            backend = get_backend()
            eff_timeout = getattr(backend, "timeout", 30.0)
            state_a = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_a, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            state_b = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_b, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            answers_a = backend.evaluate(state_a, questions, timeout=eff_timeout)
            answers_b = backend.evaluate(state_b, questions, timeout=eff_timeout)
            result = arbitrate_answers(
                answers_a, answers_b,
                min_confidence=float(cfg.get("min_confidence", 0.6)),
            )
            winner = result.get("winner")
            backend_path = getattr(
                backend, "last_backend",
                getattr(backend, "backend_kind", type(backend).__name__),
            )
            info = {
                "winner": winner,
                "backend": backend_path,
                "votes": result.get("votes", {}),
                "candidate_scores": [round(score_a, 2), round(score_b, 2)],
            }
            self._pairwise_info[chunk_id] = info
            print(
                f"⚖️ [Pipeline] {chunk_id} pairwise 仲裁: winner={winner} "
                f"backend={backend_path} votes={result.get('votes')}"
            )
            if winner == "B":
                return text_b
            if winner == "A":
                return text_a
            return best
        except Exception as e:
            print(
                f"⚠️ [Pipeline] {chunk_id} pairwise 仲裁不可用（回退 best score）: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return best

    def _save_intermediate(self, chunk_id: str, step: str, data: str | dict):
        """保存 chunk intermediate 文件. Round 99 P2: 写入时 invalidate 缓存."""
        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            if isinstance(data, str):
                json.dump({"text": data}, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
        # Invalidate 缓存 — 下次加载走文件, 反映最新内容
        if hasattr(self, "_intermediate_cache") and self._intermediate_cache is not None:
            self._intermediate_cache.pop((chunk_id, step), None)

    def _load_intermediate(self, chunk_id: str, step: str):
        """加载 chunk intermediate 文件 (raw / literary / critic_report 等).

        Round 99 P2: 内存读缓存. 同 chunk × step 在 chapter 内多次加载走内存,
        省文件 I/O + JSON parse. 仅缓存 step='file' (即 chunk_id + '_' + step.json
        文件路径), 缓存粒度 = 单 chunk × 单 step.

        写入 (via _save_intermediate) 会 invalidate 对应 cache entry.
        """
        # 缓存 key
        cache_key = (chunk_id, step)
        if not hasattr(self, "_intermediate_cache") or self._intermediate_cache is None:
            self._intermediate_cache = {}
        if cache_key in self._intermediate_cache:
            return self._intermediate_cache[cache_key]

        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "text" in data:
            result = data["text"]
        else:
            result = data
        self._intermediate_cache[cache_key] = result
        return result

    # Round 96: final.json 写入前的 CJK sanity 闸门.
    #
    # 设计动机: ch02 / ch32 实测中, rewriter LLM 偶发"输出英文+中文注释"而不是
    # 中文译文 (典型 prompt misunderstanding 或模型选择错误). 旧代码直接保存
    # LLM 输出, 下游消费者无从察觉. Round 96 在写入前检查 CJK 比例:
    #   1) primary text 通过 (cjk >= threshold) → 直接保存, 仅追加 cjk_ratio 字段
    #   2) primary 未通过 → 依次尝试 literary/raw intermediate, 选 CJK 最高的合规版本
    #   3) 全部未通过 → 仍保存 primary (保留重试痕迹), 但标记 sanity_failed=true
    #                   + sanity_inspected 列出所有检查过的 step 的 CJK 比例
    #
    # 阈值通过 config.yaml pipeline.final_cjk_threshold 配置 (默认 0.5).
    # 关闭方法: 设 final_cjk_threshold=0 或 0.0, 则闸门失效走原 _save_intermediate.
    def _save_final_with_sanity(
        self,
        chunk_id: str,
        data,
        source_step: str = "literary",
        inspect_steps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """final.json 写入前 CJK 闸门 (Round 96).

        Args:
            chunk_id: 章节 chunk id.
            data: 写入数据, 可以是 str (被 _save_intermediate 包裹) 或
                  dict 形如 {"text": ..., "metadata": ...}.
            source_step: data 来自哪个 step (literary / raw / best_lit) — 仅日志使用.
            inspect_steps: fallback 候选 step 列表 (默认 ['literary', 'raw']).

        Returns:
            实际写入 final.json 的 dict (含 text + metadata).
        """
        if inspect_steps is None:
            inspect_steps = ["literary", "raw"]

        # Round 96: 阈值通过 config 取; 0 表示禁用闸门 (向后兼容)
        threshold_cfg = _config.pipeline.get("final_cjk_threshold", 0.5)
        try:
            cjk_threshold = float(threshold_cfg)
        except (TypeError, ValueError):
            cjk_threshold = 0.5
        
        # Round 107 P6: 短文本/前言 CJK 阈值放宽.
        # ch01 倒查: 前言/目录 (65 字符, 含英文公司名/版权符号/年份) CJK 比例天然低 (0.308),
        # 0.50 阈值误判. 对短文本 (<100 字符) 或前言特征文本, 放宽到 0.30.
        if isinstance(data, dict):
            _txt = (data or {}).get("text", "") or ""
        else:
            _txt = data if isinstance(data, str) else ""
        _is_short_or_frontmatter = (
            len(_txt) < 100
            or any(kw in _txt for kw in ("目录", "版权", "献给", "版权所有", "题献", "Dedication", "Contents"))
        )
        # final sanity 的 cjk_threshold 是 cjk_ratio 的门槛: cjk_ratio >= threshold → pass.
        # 原 0.50: cjk_ratio < 0.50 → fail. 放宽到 0.30: cjk_ratio < 0.30 → fail.
        if _is_short_or_frontmatter and cjk_threshold > 0.30:
            cjk_threshold = 0.30

        # 关闭闸门 (threshold <= 0) — 走原 _save_intermediate 路径
        if cjk_threshold <= 0:
            self._save_intermediate(chunk_id, "final", data)
            if isinstance(data, str):
                return {"text": data, "metadata": {}}
            return dict(data)

        # 提取 primary text + metadata
        if isinstance(data, str):
            primary_text = data
            primary_metadata: Dict[str, Any] = {}
        else:
            primary_text = (data or {}).get("text", "") or ""
            primary_metadata = dict((data or {}).get("metadata", {}) or {})

        primary_cjk = compute_cjk_ratio(primary_text)

        # primary 已合规 → 直接保存, 仅追加 cjk_ratio 元数据
        if primary_cjk >= cjk_threshold:
            merged = {
                "text": primary_text,
                "metadata": {**primary_metadata, "cjk_ratio": round(primary_cjk, 3)},
            }
            self._save_intermediate(chunk_id, "final", merged)
            return merged

        # primary 未通过 → 尝试 fallback 候选 (按 cjk 降序, 取合规 + 比 primary 高)
        candidates: List[Tuple[str, str, float]] = []
        for step in inspect_steps:
            if step == source_step:
                continue  # 跳过与 primary 同源的 step (避免重复检查)
            try:
                loaded = self._load_intermediate(chunk_id, step)
                cand_text = loaded if isinstance(loaded, str) else (loaded or {}).get("text", "") or ""
                if cand_text and cand_text.strip():
                    cjk = compute_cjk_ratio(cand_text)
                    candidates.append((step, cand_text, cjk))
            except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                continue

        candidates.sort(key=lambda x: -x[2])

        chosen_step = None
        chosen_text: Optional[str] = None
        chosen_cjk = 0.0
        for step, cand_text, cjk in candidates:
            if cjk >= cjk_threshold and cjk > primary_cjk:
                chosen_step, chosen_text, chosen_cjk = step, cand_text, cjk
                break

        if chosen_step is not None and chosen_text is not None:
            merged = {
                "text": chosen_text,
                "metadata": {
                    **primary_metadata,
                    "cjk_ratio": round(chosen_cjk, 3),
                    "sanity_fallback": True,
                    "sanity_source_step": chosen_step,
                    "sanity_original_step": source_step,
                    "sanity_original_cjk": round(primary_cjk, 3),
                    "sanity_threshold": cjk_threshold,
                },
            }
            self._save_intermediate(chunk_id, "final", merged)
            print(
                f"⚠️ [Pipeline] {chunk_id} CJK sanity fallback: "
                f"{source_step} cjk={primary_cjk:.2f} → {chosen_step} cjk={chosen_cjk:.2f}"
            )
            return merged

        # 全部未通过 → 仍保存 primary (保留重试痕迹), 标记 sanity_failed
        sanity_inspected = [{"step": source_step, "cjk": round(primary_cjk, 3)}]
        sanity_inspected.extend(
            {"step": s, "cjk": round(c, 3)} for s, _, c in candidates
        )
        merged = {
            "text": primary_text,
            "metadata": {
                **primary_metadata,
                "cjk_ratio": round(primary_cjk, 3),
                "sanity_failed": True,
                "sanity_threshold": cjk_threshold,
                "sanity_inspected": sanity_inspected,
            },
        }
        self._save_intermediate(chunk_id, "final", merged)
        print(
            f"❌ [Pipeline] {chunk_id} CJK sanity 失败: primary={primary_cjk:.2f}, "
            f"all fallback < {cjk_threshold:.2f} ({sanity_inspected})"
        )
        return merged

    # Round 119 L1: 韵文通道 helpers（tags/manifest 惰性加载 + 文本包含兜底）。
    # 说明：verse 相关导入全部放在函数内，导入失败时静默降级为空（非韵文场景无影响）。
    def _verse_tags(self) -> dict:
        if getattr(self, '_verse_tags_cache', None) is None:
            try:
                from src.verse import load_chunk_tags
                self._verse_tags_cache = load_chunk_tags(self._root_dir / "db")
            except Exception:
                self._verse_tags_cache = {}
        return self._verse_tags_cache

    def _verse_manifest(self) -> dict:
        if getattr(self, '_verse_manifest_cache', None) is None:
            try:
                from src.verse import load_verse_manifest
                self._verse_manifest_cache = load_verse_manifest(self._root_dir / "input")
            except Exception:
                self._verse_manifest_cache = {}
        return self._verse_manifest_cache

    def _verse_fragments_for(self, chunk_id: str, source_text: str) -> list:
        """chunk 的韵文片段列表（无韵文/未启用时返回 []）。"""
        if not chunk_id:
            return []
        try:
            from src.verse import fragments_for_chunk, verse_config
            if not verse_config().get("enabled", True):
                return []
            return fragments_for_chunk(self.chapter_id, chunk_id, source_text,
                                       self._verse_tags(), self._verse_manifest())
        except Exception:
            return []

    def _verse_note_for(self, chunk_id: str, source_text: str) -> str:
        """韵文协议提示块（按 chunk 缓存；无韵文返回 ""）。"""
        if not chunk_id:
            return ""
        cache = getattr(self, '_verse_note_cache', None)
        if cache is None:
            cache = {}
            self._verse_note_cache = cache
        if chunk_id not in cache:
            try:
                from src.verse import build_verse_note
                cache[chunk_id] = build_verse_note(
                    self._verse_fragments_for(chunk_id, source_text))
            except Exception:
                cache[chunk_id] = ""
        return cache[chunk_id]

    def _verse_lint_gate(self, chunk_id: str, source_text: str, final_text) -> Optional[dict]:
        """PASS 定稿前的韵文对齐检查。None=无韵文/未启用；否则 verse_lint 结果 + 元数据。"""
        try:
            from src.verse import verse_lint, verse_config
        except Exception:
            return None
        cfg = verse_config()
        if not cfg.get("enabled", True):
            return None
        frags = self._verse_fragments_for(chunk_id, source_text)
        if not frags:
            return None
        res = verse_lint(frags, final_text,
                         min_line_ratio=float(cfg.get("min_line_ratio", 0.6)),
                         max_line_chars=int(cfg.get("max_line_chars", 56)),
                         refrain_ratio=float(cfg.get("refrain_ratio", 0.5)))
        res["retried"] = self._verse_retry_used(chunk_id)
        res["retry_on_fail"] = bool(cfg.get("retry_on_fail", True))
        res["chunk_id"] = chunk_id
        return res

    def _mark_verse_retry(self, chunk_id: str, verse_gate: dict) -> None:
        """记录「verse 定向重写已发生」（一次性限额；init --force 时重置）。"""
        try:
            self._save_intermediate(chunk_id, "verse_retry", {
                "reason": verse_gate.get("reason", ""),
                "expected_lines": verse_gate.get("expected_lines"),
                "found_lines": verse_gate.get("found_lines"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        except Exception as exc:
            print(f"⚠️ [Pipeline] {chunk_id} verse_retry 标记写入失败: {exc}", file=sys.stderr)

    def _verse_retry_used(self, chunk_id: str) -> bool:
        return (self.output_dir / f"{chunk_id}_verse_retry.json").exists()

    def _seed_pun_decisions(self, source_text: str) -> int:
        """L2: pun_watchlist 命中 chunk 时强制落 REFERENCE 决策（幂等；无 impact 不回溯）。

        source_key 直接用 anchor 原词，使 format_prompt_context 的命中过滤能在
        含该词的后续 chunk prompt 中带出策略；affected_chunks=None 保证不触发回溯。
        """
        guide = getattr(self, "style_guide", None)
        if not isinstance(guide, dict) or not source_text:
            return 0
        watch = guide.get("pun_watchlist") or []
        if not watch:
            return 0
        engine = getattr(self, "decision_engine", None)
        if engine is None or not hasattr(engine, "add_decision"):
            return 0
        seeded = getattr(self, "_pun_seeded_keys", None)
        if seeded is None:
            seeded = set()
            self._pun_seeded_keys = seeded
        src_lower = source_text.lower()
        n = 0
        for item in watch:
            if not isinstance(item, dict):
                continue
            candidates = [str(a).strip() for a in
                          (list(item.get("src_anchors") or []) + list(item.get("anchors") or []))
                          if str(a).strip()]
            hits = [a for a in candidates if len(a) >= 2 and a.lower() in src_lower]
            if not hits:
                continue
            strategy = str(item.get("strategy") or "").strip() or "双关需在译文保留笑点，必要时章末加注"
            tag = str(item.get("source") or "").strip()
            for anchor in hits[:2]:
                dedupe = "pun:" + anchor.lower()
                if dedupe in seeded:
                    continue
                try:
                    from src.core.decision_engine import DecisionLevel as _DL
                    engine.add_decision(
                        _DL.REFERENCE,
                        anchor,
                        f"[双关·保留笑点] {strategy}",
                        f"pun_watchlist: {tag}",
                        affected_chunks=None,
                    )
                    n += 1
                except Exception as exc:
                    print(f"⚠️ [Pipeline] pun 决策写入失败 {anchor}: {exc}", file=sys.stderr)
                seeded.add(dedupe)
        return n

    def _dual_draft_cfg(self) -> dict:
        """L3: 双稿开关（默认关闭；小场景 A/B 后定夺）。"""
        cfg = _config.pipeline.get("dual_draft", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        try:
            max_chunks = int(cfg.get("max_chunks", 25) or 25)
        except (TypeError, ValueError):
            max_chunks = 25
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "alternate_role": cfg.get("alternate_role") or "primary",
            "alternate_effort": cfg.get("alternate_effort"),
            "base_role": cfg.get("base_role") or "",
            "max_chunks": max_chunks,
        }

    def _dual_draft_kind(self, chunk_id: str, source_text: str) -> str:
        """'' | 'verse' | 'pun' —— 高价值片段路由判定（L3）。"""
        try:
            if self._verse_fragments_for(chunk_id, source_text):
                return "verse"
        except Exception:
            pass
        guide = getattr(self, "style_guide", None)
        if isinstance(guide, dict) and source_text:
            src_lower = source_text.lower()
            for item in (guide.get("pun_watchlist") or []):
                if not isinstance(item, dict):
                    continue
                anchors = list(item.get("src_anchors") or []) + list(item.get("anchors") or [])
                for anchor in anchors:
                    a = str(anchor).strip()
                    if len(a) >= 2 and a.lower() in src_lower:
                        return "pun"
        return ""

    def _dual_critic_avg(self, chunk_id: str, source_text: str, raw_text: str, text: str):
        """L3: critic 单稿均分（失败返回 None）。"""
        if not text:
            return None
        try:
            report = self.critic_agent.process_chunk(
                chunk_id, source_text, raw_text or "", text, self.style_guide,
                entity_registry=self._build_entity_registry_for_rewriter(source_text),
                entity_variants=self._build_entity_variants_for_critic(source_text),
            )
            avg = self._compute_avg_score(report)
            return float(avg) if avg else None
        except Exception:
            return None

    def _dual_pairwise_winner(self, chunk_id: str, source_text: str, raw_text: str,
                              text_a: str, text_b: str):
        """L3: 两份候选稿 Jev 配对仲裁 → (winner 'a'|'b'|None, info dict)。"""
        cfg = _config.pipeline.get("pairwise_arbitration", {}) or {}
        info = {"winner": None, "backend": None, "votes": {}}
        try:
            from src.judges.judge import build_state
            from src.judges.primitives import build_question_set
            from src.judges.typesafe_backend import get_backend
            from src.judges.pairwise import arbitrate_answers

            style = dict(getattr(self, "style_guide", None) or {})
            entity_registry = self._build_entity_registry_for_rewriter(source_text)
            questions = build_question_set()
            backend = get_backend()
            eff_timeout = getattr(backend, "timeout", 30.0)
            state_a = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_a, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            state_b = build_state(
                source_text=source_text, raw_translation=raw_text or "",
                literary_translation=text_b, style_guide_stats=style,
                entity_registry=entity_registry,
            )
            answers_a = backend.evaluate(state_a, questions, timeout=eff_timeout)
            answers_b = backend.evaluate(state_b, questions, timeout=eff_timeout)
            result = arbitrate_answers(
                answers_a, answers_b,
                min_confidence=float(cfg.get("min_confidence", 0.6)),
            )
            winner_raw = result.get("winner")
            backend_path = getattr(
                backend, "last_backend",
                getattr(backend, "backend_kind", type(backend).__name__),
            )
            info = {"winner": winner_raw, "backend": backend_path,
                    "votes": result.get("votes", {})}
            if winner_raw == "B":
                return "b", info
            if winner_raw == "A":
                return "a", info
            return None, info
        except Exception as e:
            print(f"⚠️ [Pipeline] {chunk_id} 双稿 pairwise 不可用: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return None, info

    def _dual_draft_run(self, chunk_id: str, source_text: str, raw_text: str,
                        prev_lit_text, reject_reason, critic_feedback, entity_feedback,
                        all_feedback, specific_edits, narrative_memory, low_dims,
                        skipped_dimensions, draft_a: str) -> str:
        """L3: 高价值片段双稿仲裁 — 不满足条件时原样返回 draft_a。"""
        cfg = self._dual_draft_cfg()
        if not cfg.get("enabled") or not draft_a or not source_text:
            return draft_a
        count = int(getattr(self, "_dual_draft_count", 0) or 0)
        if count >= int(cfg.get("max_chunks") or 25):
            return draft_a
        try:
            existing = self._load_intermediate(chunk_id, "dual_draft_meta")
            if isinstance(existing, dict) and existing.get("chosen"):
                return draft_a
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
            pass
        kind = self._dual_draft_kind(chunk_id, source_text)
        if not kind:
            return draft_a
        alt_role = cfg.get("alternate_role") or "primary"
        base_role = cfg.get("base_role") or _config.get_task_role("literary_rewrite")
        try:
            draft_b_raw = self.rewriter_agent.process_chunk(
                chunk_id, raw_text, self.style_guide, source_text,
                prev_lit_text=prev_lit_text,
                reject_reason=reject_reason,
                critic_feedback=critic_feedback,
                entity_feedback=entity_feedback,
                entity_registry=self._build_entity_registry_for_rewriter(source_text),
                retry_count=0,
                all_feedback=all_feedback,
                specific_edits=specific_edits,
                narrative_memory=narrative_memory,
                low_dims=low_dims,
                skipped_dimensions=skipped_dimensions,
                verse_note=self._verse_note_for(chunk_id, source_text),
                model_role=alt_role,
                model_effort=cfg.get("alternate_effort"),
            )
        except TypeError:
            return draft_a
        except Exception as e:
            print(f"⚠️ [Pipeline] {chunk_id} 双稿第二稿生成失败, 保留首稿: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return draft_a
        draft_b = draft_b_raw[0] if isinstance(draft_b_raw, tuple) else draft_b_raw
        if not draft_b or draft_b == draft_a:
            return draft_a
        score_a = self._dual_critic_avg(chunk_id, source_text, raw_text, draft_a)
        score_b = self._dual_critic_avg(chunk_id, source_text, raw_text, draft_b)
        winner, pair_info = self._dual_pairwise_winner(
            chunk_id, source_text, raw_text, draft_a, draft_b)
        if winner == "b":
            chosen, chosen_text, reason = "b", draft_b, "pairwise"
        elif winner == "a":
            chosen, chosen_text, reason = "a", draft_a, "pairwise"
        elif score_b is not None and (score_a is None or score_b > score_a):
            chosen, chosen_text, reason = "b", draft_b, "critic"
        else:
            chosen, chosen_text = "a", draft_a
            reason = "critic" if score_a is not None else "fallback"
        self._save_intermediate(chunk_id, "dual_draft_meta", {
            "chosen": chosen,
            "reason": reason,
            "kind": kind,
            "roles": {"a": base_role, "b": alt_role},
            "critic_avg": {"a": score_a, "b": score_b},
            "pairwise": pair_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        self._dual_draft_count = count + 1
        print(f"⚖️ [Pipeline] {chunk_id} 双稿仲裁({kind}): 选 {chosen}"
              f"（{reason}; critic a={score_a} b={score_b}; pairwise={pair_info.get('winner')}）")
        return chosen_text

    def _build_literal_prompt(self, source_text: str, chunk_id: str = "") -> str:
        """组装直译 prompt（含有界术语表 + 本书用词提示 + 韵文协议）；供生产路径与单测共用。"""
        genre = (_config.style_guide or {}).get("genre") or "文学小说"
        glossary = self.decision_engine.format_prompt_context(source_text=source_text)
        verse_note = self._verse_note_for(chunk_id, source_text) if chunk_id else ""
        return (
            f"请将以下{genre}片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n"
            f"{glossary}\n\n"
            f"{self._build_translation_hints_block()}"
            f"{verse_note}"
            f"【原文】\n{source_text}"
        )

    @staticmethod
    def _build_translation_hints_block() -> str:
        """本书用词提示块（P1-4：作品特定提示只允许来自风格剖面 translation_hints）。

        旧版曾在此硬编码「1920年代英式俚语 wild-cat / 丁尼生名句」等作品特定先验，
        会污染其他作品的直译；现在无 profile 时只给作品无关的通用指令。
        """
        hints = (_config.style_guide or {}).get("translation_hints", []) or []
        if hints:
            lines = "\n".join(f"- {h}" for h in hints[:30])
            return (
                "⚠️ 本书已确认的年代/地域用词与引用（先识别含义再翻译，不要按字面意思直译）：\n"
                f"{lines}\n\n"
            )
        return (
            "⚠️ 注意识别时代/地域俚语、行业行话与文学引用：先识别含义再翻译，不要按字面意思直译；"
            "无法确认的引用保留可考据的译法，并在术语表中登记。\n\n"
        )

    def _run_raw_translator(self, source_text: str, chunk_id: str = "") -> str:
        prompt = self._build_literal_prompt(source_text, chunk_id)
        # 使用配置路由：literal_translation
        model_key, params = _config.resolve_task_model("literal_translation")
        role = self.literal_role
        llm = self.llm
        params = dict(params)
        # Round 118 P2-5 / Round 128: 难度路由 — 高难度片段切到更强模型（默认 primary/MiniMax）
        # 并按段加档 effort（hard_effort）；普通片段仍走任务默认（不下发档位）。
        dcfg = _config.pipeline.get("difficulty_routing", {}) or {}
        if dcfg.get("enabled", True):
            difficulty = self._chunk_difficulty(source_text)
            if difficulty >= float(dcfg.get("hard_threshold", 2.0)):
                role = dcfg.get("hard_role", "primary")
                llm = self._get_role_llm(role)
                hard_effort = str(dcfg.get("hard_effort") or "").strip().lower()
                if hard_effort:
                    params["effort"] = hard_effort
                print(
                    f"🧩 [Pipeline] 难度路由: difficulty={difficulty:.2f} ≥ "
                    f"{float(dcfg.get('hard_threshold', 2.0)):.1f} → role={role} "
                    f"(effort={hard_effort or 'default'})"
                )
        # Round 133: no_content 重试适配 override（降档/换角救稿）优先于难度路由结果
        if chunk_id:
            ov = self._get_retry_override(chunk_id)
            if ov and ov.get("sig") == "no_content" and ov.get("stage") == "TRANSLATING_RAW":
                ov_role = str(ov.get("role") or "")
                if ov_role and ov_role != role:
                    role = ov_role
                    llm = self._get_role_llm(role)
                ov_eff = ov.get("effort")
                if ov_eff:
                    params["effort"] = str(ov_eff).strip().lower()
                print(
                    f"🔻 [Pipeline] {chunk_id} 重试适配生效: role={role} "
                    f"effort={params.get('effort') or 'default'}"
                )
        model_name = get_role_model_name(role) or model_key
        self._note_last_call(chunk_id, "TRANSLATING_RAW", role, params.get("effort"), model_name)
        return llm.generate(prompt, model_name=model_name, **params)

    def _get_role_llm(self, role: str):
        """按角色缓存 LLM 客户端（难度路由切模型时复用，避免重复构造）。"""
        cache = getattr(self, "_role_llm_cache", None)
        if cache is None:
            cache = {}
            self._role_llm_cache = cache
        if role not in cache:
            from src.utils.llm_adapter import get_llm_client
            cache[role] = get_llm_client(role)
        return cache[role]

    def _count_reference_hits(self, source_text: str) -> int:
        """统计本片段命中的已登记典故数（DecisionLevel.REFERENCE = 2）。"""
        if not source_text:
            return 0
        src_lower = source_text.lower()
        try:
            cur = self.decision_engine.conn.cursor()
            cur.execute("SELECT source_key FROM decision_db WHERE level = 2")
            hits = 0
            for (source_key,) in cur.fetchall():
                if not source_key:
                    continue
                if source_key in source_text:
                    hits += 1
                elif len(source_key) >= 3 and source_key.lower() in src_lower:
                    hits += 1
            return hits
        except Exception:
            return 0

    def _chunk_difficulty(self, source_text: str) -> float:
        """Round 118 P2-5: 片段难度启发式打分（用于直译阶段模型路由）。

        信号：
          - 已登记典故密度（考据越密越难；每命中 1 条 +1，封顶 3）
          - 长句占比（≥35 词句子占比 ≥0.3 → +1.5，≥0.15 → +0.75）
          - 引文特征（含成对引号且引文长度 ≥6 词 → +1）
        """
        if not source_text:
            return 0.0
        score = float(min(3, self._count_reference_hits(source_text)))
        sentences = [s for s in re.split(r"[.!?]+\s*", source_text) if s.strip()]
        if sentences:
            long_ratio = sum(
                1 for s in sentences if len(s.split()) >= 35
            ) / len(sentences)
            if long_ratio >= 0.3:
                score += 1.5
            elif long_ratio >= 0.15:
                score += 0.75
        for m in re.finditer(r"[\"“]([^\"”]{20,})[\"”]", source_text):
            if len(m.group(1).split()) >= 6:
                score += 1.0
                break
        return score

    @staticmethod
    def _compute_text_similarity(a: str, b: str) -> float:
        """Round 6: 文本相似度（字符级 SequenceMatcher 比值）

        用于 oscillation 检测：连续 retry 译文相似度 ≥ 阈值 → LLM 原地打转 → 早停。

        返回 0.0~1.0；空字符串特殊处理（视为 0.0 避免除零）。
        """
        if not a or not b:
            return 0.0
        if len(a) > 4000 or len(b) > 4000:
            # 长文本截断对比（避免 difflib 平方复杂度爆栈）
            a, b = a[:4000], b[:4000]
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()


    # Round 34: thin shim — delegates to src.utils.text_match.extract_reject_categories
    # （行为完全等价 Round 16 实现；保留类方法签名避免破坏 pipeline 内部调用点）
    @staticmethod
    def _extract_reject_categories(reason_text: str) -> set:
        return extract_reject_categories(reason_text, as_set=True)
    # Round 34: thin shim — delegates to src.utils.text_match.case_insensitive_find
    @staticmethod
    def _case_insensitive_find(name: str, haystack: "str | None") -> "str | None":
        return case_insensitive_find(name, haystack)
    def _memory_path(self) -> Path:
        return self.output_dir / "_memory.jsonl"

    def _append_chunk_memory(self, chunk_id: str, lit_text: str, source_text: str = "") -> None:
        """定稿后写入轻量章内记忆，供后续 chunk 润色注入。

        Round 100 P1: 写入后 invalidate _narrative_memory_cache (新 chunk 加入让
        后续 chunk 的 prior_memory 视图变化).
        """
        if hasattr(self, '_narrative_memory_cache') and self._narrative_memory_cache is not None:
            # 全清: 新写入影响所有"当前 chunk 之后"的视图, 但最简方案是全清
            self._narrative_memory_cache.clear()
        text = lit_text if isinstance(lit_text, str) else str(lit_text)
        # 规则摘要：取前 2–3 句或截断 240 字
        parts = re.split(r"[。！？\.\!\?]\s*", text.strip())
        sentences = [p.strip() for p in parts if p and p.strip()]
        summary = "。".join(sentences[:3])
        if summary and not summary.endswith("。"):
            summary += "。"
        if len(summary) > 240:
            summary = summary[:240] + "…"
        rec = {
            "chunk_id": chunk_id,
            "summary": summary or text[:200],
            "source_preview": (source_text or "")[:80],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            with open(self._memory_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"  ⚠️ 写入 chapter memory 失败: {e}")

    def _load_prior_memory(self, current_chunk_id: str, max_entries: int = 3) -> str:
        """读取本章已完成 chunk 的摘要（不含当前 chunk）。

        Round 100 P1: 内存读缓存. 同 chunk 跨 retry 内容不变, 缓存命中跳过文件 I/O
        和 JSON parse. _append_chunk_memory 写入时会 invalidate 对应 cache entry.
        """
        if not hasattr(self, '_narrative_memory_cache') or self._narrative_memory_cache is None:
            self._narrative_memory_cache = {}
        cache_key = current_chunk_id
        if cache_key in self._narrative_memory_cache:
            return self._narrative_memory_cache[cache_key]

        path = self._memory_path()
        if not path.exists():
            return ""
        lines = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # P2-1: memory jsonl 用 robust_json_loads 抗损坏行
                    rec = robust_json_loads(line, expected_type=dict)
                    if not rec:
                        continue
                    if rec.get("chunk_id") == current_chunk_id:
                        continue
                    lines.append(rec)
        except OSError:
            return ""
        recent = lines[-max_entries:]
        if not recent:
            return ""
        result = "\n".join(
            f"- [{r.get('chunk_id', '?')}] {r.get('summary', '')}" for r in recent
        )
        self._narrative_memory_cache[cache_key] = result
        return result

    def run(self) -> bool:
        print(f"🚀 [Pipeline] 启动批处理模式处理章节: {self.chapter_id}")

        # Round 99 P1: chapter 切换时清空 per-chunk 缓存, 防止跨 chapter 脏数据
        self._reset_per_run_cache()

        # 检查章节是否有任务，避免在空 DB 上误报"完成"
        existing_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
        if not existing_tasks:
            # 区分两种空：真未 init vs init 被跳过但 DB 无行/无产出
            print(f"❌ [Pipeline] 章节 {self.chapter_id} 无可执行任务（可能 init 被跳过或 DB 无记录），"
                  f"请用 --force 重新初始化。")
            return False

        # Round 104: 恢复"COMPLETED 但 final.json 缺失"的 chunk (回溯删 final 后重跑失败的残留)
        try:
            self._recover_completed_without_final()
        except Exception as e:
            print(f"⚠️ [Pipeline] COMPLETED 恢复检查异常（不影响主流程）: {e}")

        # 从配置读取 pipeline 参数
        pipe_cfg = _config.pipeline
        batch_size = pipe_cfg.get("batch_size", 50)
        poll_interval = pipe_cfg.get("poll_interval", 0.5)

        # 阶段顺序定义：按流水线顺序处理，每阶段批量处理
        pipeline_stages = [
            (TaskState.DIRTY, self._process_dirty_batch, "回溯重跑"),
            (TaskState.FAILED, self._process_failed_batch, "失败重试"),
            (TaskState.PENDING, self._process_pending_batch, "初始化"),
            (TaskState.EXTRACTING_TERMS, self._process_extracting_terms_batch, "术语提取"),
            (TaskState.TRANSLATING_RAW, self._process_translating_raw_batch, "直译"),
            (TaskState.REWRITING_LITERARY, self._process_rewriting_batch, "文学润色"),
            (TaskState.AUDITING, self._process_auditing_batch, "审计评分"),
            (TaskState.JUDGING, self._process_judging_batch, "最终裁决"),
        ]

        # Round 130: deferred 回溯的章级合并预算计数（本次运行内累计）
        self._backtrack_sweeps_done = 0

        while True:
            any_progress = False

            for state, handler, stage_name in pipeline_stages:
                tasks = self.scheduler.get_tasks_by_state(state, batch_size=batch_size, chapter_id=self.chapter_id)
                if not tasks:
                    continue

                print(f"📦 [Batch] {stage_name} 阶段: 处理 {len(tasks)} 个任务")
                handler(tasks)
                any_progress = True

            if not any_progress:
                # Round 130: deferred 回溯在队列清空点（章末）合并应用一次；
                # 治理预算（章级轮数 + 单块轮数）保证不收敛循环无法再出现。
                # immediate 模式下此调用为 no-op（决策变更已即时标 DIRTY）。
                try:
                    if self._consolidate_deferred_backtracks():
                        any_progress = True
                except Exception as e:
                    print(f"⚠️ [Pipeline] 章末合并回溯异常（不影响主流程）: {e}")

            if not any_progress:
                # 仅表示队列已清空，不代表产出完整；最终是否成功由 _verify_final_outputs 决定
                print(f"⏹ [Pipeline] 章节 {self.chapter_id} 任务队列已清空，进入产出校验…")
                break

            time.sleep(poll_interval)

        # Round 7: 译后一致性校正（章节级 scope，auto-apply）
        # 修复 alias_mixed / english_residue / cjk_variant 三类命名漂移
        # QA_REPORT 显示 162 处不一致散落在历史章节里，启用后可批量修复
        post_consistency_cfg = _config.pipeline.get("post_consistency", {}) or {}
        if post_consistency_cfg.get("enabled", True):
            try:
                self._run_post_consistency(
                    chapter_scoped=bool(post_consistency_cfg.get("chapter_scoped", True))
                )
            except Exception as e:
                print(f"⚠️ [Pipeline] 译后一致性校正异常（不影响主流程）: {e}")

        # Round 15: entity_registry 冲突检测（生成 entity_registry_conflicts.md）
        reg_audit_cfg = _config.pipeline.get("entity_registry_audit", {}) or {}
        if reg_audit_cfg.get("enabled", True):
            try:
                conflicts = self._entity_registry_conflict_audit()
                if conflicts:
                    cross = sum(1 for c in conflicts if c["conflict_type"] == "cross_entity")
                    variant = sum(1 for c in conflicts if c["conflict_type"] == "variant_for_same_person")
                    lines = []
                    lines.append(f"# entity_registry 冲突检测 - {self.chapter_id}\n\n")
                    lines.append(f"- 总冲突数: {len(conflicts)} (跨人物: {cross}, 同人变体: {variant})\n\n")
                    lines.append("## 跨人物冲突（明确错误，需人工修复）\n\n")
                    for c in [x for x in conflicts if x["conflict_type"] == "cross_entity"]:
                        lines.append("- **\"" + c["source_name"] + "\"** ← " + str(c["canonicals"]) + "\n")
                    lines.append("\n## 同人变体（可接受但建议整合）\n\n")
                    for c in [x for x in conflicts if x["conflict_type"] == "variant_for_same_person"]:
                        lines.append("- **\"" + c["source_name"] + "\"** ← " + str(c["canonicals"]) + "\n")
                    lines.append("\n注：本报告为诊断用，**不会自动修改 entity_registry**。\n")
                    lines.append("修复方式：人工编辑 decision_db.sqlite 的 entity_registry 表 source_names 列。\n")
                    conflict_path = self.output_dir / "entity_registry_conflicts.md"
                    conflict_path.write_text("".join(lines), encoding="utf-8")
                    print(f"⚠️ [Entity Registry] 发现 {cross} 个跨人物冲突 + {variant} 个同人变体（详见 entity_registry_conflicts.md）")
                else:
                    print(f"✅ [Entity Registry] 无冲突")
            except Exception as e:
                print(f"⚠️ [Pipeline] entity_registry 冲突检测异常（不影响主流程）: {e}")

        # Round 9: 实体完整性审计（纯诊断，生成 entity_audit.md）
        entity_audit_cfg = _config.pipeline.get("entity_audit", {}) or {}
        if entity_audit_cfg.get("enabled", True):
            try:
                self._post_pipeline_entity_audit()
            except Exception as e:
                print(f"⚠️ [Pipeline] 实体审计异常（不影响主流程）: {e}")

        # Round 13: LLM 调用成本报告
        cost_cfg = _config.pipeline.get("cost_report", {}) or {}
        if cost_cfg.get("enabled", True):
            try:
                self._print_llm_cost_report()
            except Exception as e:
                print(f"⚠️ [Pipeline] 成本报告异常（不影响主流程）: {e}")

        # Round 14: intermediate 文件清理
        try:
            self._cleanup_intermediates()
        except Exception as e:
            print(f"⚠️ [Pipeline] 中间文件清理异常（不影响主流程）: {e}")

        # Round 107 P7: 恢复回溯重跑失败的 final_backup.json (在验证前)
        try:
            self._restore_dirty_backups()
        except Exception as e:
            print(f"⚠️ [Pipeline] 恢复 dirty backup 异常（不影响主流程）: {e}")

        # 完成后交叉验证：final.json 落盘数应与任务数一致，否则视为假阳性完成
        return self._verify_final_outputs(len(existing_tasks))

    def _run_post_consistency(self, chapter_scoped: bool = True) -> None:
        """Round 7: 章节完成后自动跑 consistency（章节级 scope）

        Round 105: 受 pipeline.consistency_auto_fix 开关控制。
          - false（默认）: 仅生成 consistency_diff.md 报告，不落地修改 final.json
            （ch32 实测自动修正会把普通词当人名异体替换，改坏正文）。
          - true: 直接 apply 修正。
          CLI 子命令 `consistency --apply` 不受此开关影响（显式人工操作）。

        Args:
            chapter_scoped: True=仅修正当前章节的 final.json（生产环境推荐）
                           False=扫描整个 output/（CI/全量审计用，会修改所有章节）
        """
        from src.consistency import run_consistency_check_for_chapter, run_consistency_check
        auto_fix = bool(_config.pipeline.get('consistency_auto_fix', False))
        dry_run = not auto_fix
        if chapter_scoped:
            n = run_consistency_check_for_chapter(self.chapter_id, dry_run=dry_run)
        else:
            # 全局模式：复用 CLI 子命令行为（覆盖整个 output/）
            n = run_consistency_check(dry_run=dry_run)
        if n and n > 0:
            if dry_run:
                print(f"  ⏸️ [Consistency] 发现 {n} 处命名不一致（consistency_auto_fix=false，仅报告未修改；详见 consistency_diff.md）")
            else:
                print(f"  ✅ [Consistency] 本章修正 {n} 处命名不一致")

    def _print_llm_cost_report(self) -> None:
        """Round 13: 输出 LLM 调用成本报告（按 task_type 分组）"""
        counter = self._llm_call_counter
        total = sum(counter.values())

        display_names = {
            "reference_extraction": "典故/实体提取",
            "literal_translation": "字面直译",
            "literary_rewriting": "文学润色",
            "critic_scoring": "Critic 评分",
            "judge_decision": "最终裁决",
        }
        est_tokens_per_call = {
            "reference_extraction": 4000,
            "literal_translation": 2500,
            "literary_rewriting": 5000,
            "critic_scoring": 4500,
            "judge_decision": 3500,
        }

        lines = []
        lines.append("# LLM 成本报告 - " + self.chapter_id + "\n\n")
        lines.append("- 生成时间: " + time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        lines.append("- 总 LLM 调用次数: **" + str(total) + "**\n\n")
        lines.append("| Task 类型 | 调用次数 | 估算 tokens | 占比 |\n")
        lines.append("|---|---:|---:|---:|\n")
        sorted_types = sorted(counter.items(), key=lambda kv: -kv[1])
        for task_type, count in sorted_types:
            if count == 0:
                continue
            pct = (count / total * 100) if total > 0 else 0
            est_tokens = count * est_tokens_per_call.get(task_type, 3000)
            lines.append("| " + display_names.get(task_type, task_type) + " | " + str(count) + " | ~" + format(est_tokens, ",") + " | " + format(pct, ".1f") + "% |\n")
        total_tokens = sum(c * est_tokens_per_call.get(t, 3000) for t, c in counter.items())
        lines.append("| **合计** | **" + str(total) + "** | **~" + format(total_tokens, ",") + "** | 100% |\n\n")
        lines.append("- Round 6 oscillation + Round 12 stagnation + Round 36 字节停滞 + Round 73 pre-rewriter 短路 早停链 已显著减少 literary_rewriting / critic_scoring / judge_decision 调用次数\n")

        # Round 27: 真实 token 累计（按 role；OpenAI-compatible API 直接解析 usage；Mock 估算 len//4）。
        #              注意：同 role 多 task 共享 adapter 时的累计会合并显示
        try:
            from src.utils.llm_adapter import get_llm_client
            seen_roles = set()
            role_rows = []
            for task_type in counter.keys():
                role = _config.get_task_role(task_type)
                if role in seen_roles:
                    continue
                seen_roles.add(role)
                client = get_llm_client(role)
                u = client.get_cumulative_usage()
                role_rows.append((role, u))
            if role_rows:
                total_real = sum(u["total_tokens"] for _, u in role_rows)
                lines.append("\n## 真实 token 累计（Round 27，按 role 汇总）\n\n")
                lines.append("| Role (后端) | 调用次数 | Prompt tokens | Completion tokens | 合计 |\n")
                lines.append("|---|---:|---:|---:|---:|\n")
                for role, u in sorted(role_rows, key=lambda kv: -kv[1]["total_tokens"]):
                    if u["calls"] == 0:
                        continue
                    lines.append(
                        "| " + role + " | " + str(u["calls"]) + " | "
                        + format(u["prompt_tokens"], ",") + " | "
                        + format(u["completion_tokens"], ",") + " | "
                        + format(u["total_tokens"], ",") + " |\n"
                    )
                lines.append("| **合计** | | | | **" + format(total_real, ",") + "** |\n\n")
                # 对账：估算 vs 真实（Round 27 + Round 31 加偏差告警）
                if total_tokens > 0 and total_real > 0:
                    ratio = total_real / total_tokens
                    delta_pct = abs(ratio - 1.0) * 100
                    lines.append(
                        "- 对账：估算 " + format(total_tokens, ",")
                        + " → 真实 " + format(total_real, ",")
                        + f"（比值 {ratio:.2f}×，偏差 {delta_pct:.1f}%）\n"
                    )
                    # Round 31: 偏差 > 30% 提示用户更新 mental model
                    if delta_pct > 30:
                        lines.append(
                            "- ⚠️ **估算偏差 > 30%**：实际 token 与 Round 13 估算差距显著，"
                            "建议更新 est_tokens_per_call 表以提升后续章节预测精度。\n"
                        )

            # Round 35: 自动建议 est_tokens_per_call 更新值（基于本章真实 per-call 平均）
            try:
                                # Round 84: 补全 Round 81 fix —— 之前 round 81 只修了 lines.append("\n") 和内层 f-string, 
                # 外层 markdown 结构 (section header / 段落 / 表格 header) 仍含字面 \n,
                # 写入 markdown 后渲染为字面 \n 而非真换行. 现在统一改为 \n.
                lines.append("\n### est_tokens_per_call 建议值（Round 35）\n\n")
                lines.append("基于本章真实 token / 调用次数计算的每调用平均：\n\n")
                lines.append("| Role | 当前每调用真实平均 | 当前估算 | 建议替换值 |\n")
                lines.append("|---|---:|---:|---:|\n")
                seen_roles_suggest = set()
                for task_type in counter.keys():
                    role = _config.get_task_role(task_type)
                    if role in seen_roles_suggest:
                        continue
                    seen_roles_suggest.add(role)
                    client = get_llm_client(role)
                    u = client.get_cumulative_usage()
                    if u["calls"] == 0:
                        continue
                    avg = u["total_tokens"] / u["calls"]
                    current_est = est_tokens_per_call.get(task_type, 3000)
                    delta_pct_est = abs(avg - current_est) / current_est * 100
                    suggested = round(avg / 500) * 500
                    marker = "**" if delta_pct_est > 30 else ""
                    lines.append(
                        f"| {role} | {avg:,.0f} | {current_est:,} | "
                        f"{marker}{suggested:,}{marker}（偏差 {delta_pct_est:.0f}%）|" + "\n"
                    )
                # Round 81: 真换行符 (修复前是字面反斜杠 + n, 不会被 markdown 渲染)
                lines.append("\n")  # Round 81: 真换行符 (修复前字面 \n, 不会被 markdown 渲染)
            except Exception as _sug_err:
                print(f"  ⚠️ [_print_llm_cost_report] est_tokens_per_call 建议生成失败: {_sug_err}", file=sys.stderr)

        except Exception as _real_err:

            print(f"  ⚠️ [_print_llm_cost_report] 真实 token 统计失败: {_real_err}", file=sys.stderr)

        # Round 18: 统计 Jev 后端实际利用率（扫描 critic_report.json 的 _backend_path）
        backend_counter = {"jev": 0, "openai_compat": 0, "mock_heuristic": 0, "legacy": 0, "unknown": 0}
        try:
            for cr_path in self.output_dir.glob("*_critic_report.json"):
                try:
                    with open(cr_path, "r", encoding="utf-8") as f:
                        cr = json.load(f)
                    bp = cr.get("_backend_path", "unknown")
                    backend_counter[bp] = backend_counter.get(bp, 0) + 1
                except (OSError, json.JSONDecodeError):
                    pass
            total_critic = sum(backend_counter.values())
            if total_critic > 0:
                lines.append("\n## Jev 后端利用率（Round 18）\n\n")
                lines.append("| Backend | 调用次数 | 占比 | 说明 |\n")
                lines.append("|---|---:|---:|---|\n")
                bp_desc = {
                    "jev": "✅ Jev 原生（typesafe_sdk 直连）",
                    "openai_compat": "⚠️ OpenAI 兼容 fallback（MiniMax/DeepSeek）",
                    "mock_heuristic": "ℹ️ Mock 启发式（兜底）",
                    "legacy": "ℹ️ 旧 Critic LLM 路径",
                    "unknown": "❓ 未识别",
                }
                for bp in ("jev", "openai_compat", "mock_heuristic", "legacy", "unknown"):
                    n = backend_counter.get(bp, 0)
                    if n == 0:
                        continue
                    pct = n / total_critic * 100
                    lines.append("| " + bp + " | " + str(n) + " | " + f"{pct:.1f}%" + " | " + bp_desc.get(bp, "") + " |\n")
                lines.append("\n")
                jev_pct = backend_counter.get("jev", 0) / total_critic * 100
                if jev_pct >= 95:
                    lines.append("- ✅ Jev 实际使用率 " + f"{jev_pct:.1f}%" + "（接近 100% 说明 fallback 链路健康）\n")
                elif jev_pct >= 50:
                    lines.append("- 🟡 Jev 实际使用率 " + f"{jev_pct:.1f}%" + "（部分 chunk 走了 fallback，检查 _backend_diagnostics）\n")
                else:
                    lines.append("- 🔴 Jev 实际使用率 " + f"{jev_pct:.1f}%" + "（多数 chunk 走了 fallback，调查原因）\n")

                # Round 62: 统计 skipped_dimensions — 让用户看到哪些维度被 Jev 不确定跳过
                # 帮助诊断：是 Jev 真给 clean verdict，还是边界值被 uncertain zone 过滤？
                # Round 75 修复: 原代码引用 chapter_dirs 但从未定义, 整个 try 块静默失败
                # (except Exception 吞掉). 改为 [self.output_dir] (本章节 critic_report 都在该目录下).
                try:
                    from collections import Counter as _SkCounter
                    skipped_counter: _SkCounter = _SkCounter()
                    total_with_skipped = 0
                    # Round 75 修复: 原代码引用 chapter_dirs 但从未定义, 整个 try 块静默失败
                    # (except Exception 吞掉). 改为 [self.output_dir] (本章节 critic_report 都在该目录下).
                    chapter_dirs = [self.output_dir]
                    for chunk_dir in chapter_dirs:
                        for cr_path in chunk_dir.glob("*_critic_report.json"):
                            try:
                                cr = json.loads(cr_path.read_text(encoding="utf-8"))
                            except Exception:
                                continue
                            tv = cr.get("_typesafe_verdict", {}) or {}
                            skipped = tv.get("skipped_dimensions", []) or []
                            if skipped:
                                total_with_skipped += 1
                                for dim in skipped:
                                    skipped_counter[dim] += 1
                    if skipped_counter:
                        lines.append("\n## Jev 边界判断 / 不确定跳过（Round 62）\n\n")
                        lines.append("| 维度 | 跳过次数 | 说明 |\n")
                        lines.append("|---|---:|---|\n")
                        dim_desc = {
                            "voice_drift": "音色漂移（Choice conf < 0.5）",
                            "register_match": "语域契合（Choice conf < 0.5）",
                            "entity_consistency": "实体一致性（Noul ∈ [0.3, 0.7]）",
                            "literal_residue": "直译残留（Noul ∈ [0.3, 0.7]）",
                            "semantic_loss": "关键语义丢失（Noul ∈ [0.3, 0.7]）",
                            "style_compliance_score": "风格契合度（Score conf < 0.6）",
                        }
                        for dim, n in skipped_counter.most_common():
                            lines.append(f"| {dim} | {n} | {dim_desc.get(dim, '—')} |\n")
                        lines.append("\n")
                        lines.append(f"- {total_with_skipped}/{total_critic} chunk 至少有一个维度被跳过\n")
                        lines.append("- 跳过 ≠ 通过：被跳过的维度 Jev 不确定，仍可能有问题但未达 reject 门槛\n")
                except Exception as _sk_err:
                    print(f"  ⚠️ [_print_llm_cost_report] skipped_dimensions 统计失败: {_sk_err}", file=sys.stderr)
        except Exception as _bp_err:
            print(f"  ⚠️ [_print_llm_cost_report] Jev 利用率统计失败: {_bp_err}", file=sys.stderr)

        # Round 32: 章节翻译质量摘要（扫描 final.json 分类 fallback）
        try:
            quality_counter = {
                "PASS": 0,
                "SAFETY_NET_PASSED": 0,
                "PERMANENTLY_FAILED_AFTER_MAX_RETRIES": 0,
                "EARLY_STOP": 0,
                "REWRITER_STAGNATED": 0,  # Round 36 字节停滞
                "PRE_REWRITER_STAGNATION": 0,  # Round 73 pre-rewriter 短路
            }
            total_quality_retries = 0
            max_retries_seen = 0
            for fin_path in self.output_dir.glob("*_final.json"):
                try:
                    with open(fin_path, "r", encoding="utf-8") as f:
                        fin = json.load(f)
                    if not isinstance(fin, dict):
                        continue
                    md = fin.get("metadata", {}) or {}
                    if md.get("fallback"):
                        reason = md.get("reason", "UNKNOWN")
                        quality_counter[reason] = quality_counter.get(reason, 0) + 1
                        qr = int(md.get("quality_retries", 0) or 0)
                        total_quality_retries += qr
                        max_retries_seen = max(max_retries_seen, qr)
                    else:
                        quality_counter["PASS"] += 1
                except (OSError, json.JSONDecodeError):
                    pass
            total_chunks = sum(quality_counter.values())
            if total_chunks > 0:
                pass_pct = quality_counter["PASS"] / total_chunks * 100
                lines.append("\n## 翻译质量摘要（Round 32）\n\n")
                lines.append("| 状态 | chunk 数 | 占比 |\n")
                lines.append("|---|---:|---:|\n")
                status_emoji = {
                    "PASS": "✅",
                    "SAFETY_NET_PASSED": "🟡",
                    "PERMANENTLY_FAILED_AFTER_MAX_RETRIES": "🔴",
                    "EARLY_STOP": "⏭️",
                    "REWRITER_STAGNATED": "⏹️",
                    "PRE_REWRITER_STAGNATION": "⏭️",  # Round 73: pre-rewriter 短路
                }
                for status, n in quality_counter.items():
                    if n == 0:
                        continue
                    pct = n / total_chunks * 100
                    emoji = status_emoji.get(status, "")
                    lines.append(f"| {emoji} {status} | {n} | {pct:.1f}% |\n")
                lines.append(f"| **合计** | **{total_chunks}** | 100% |\n\n")
                lines.append(f"- 一次通过率：**{pass_pct:.1f}%** ({quality_counter['PASS']}/{total_chunks})\n")
                if total_quality_retries > 0:
                    avg_qr = total_quality_retries / max(1, total_chunks - quality_counter["PASS"])
                    lines.append(f"- 失败 chunk 平均重试轮次：{avg_qr:.2f}（最高 {max_retries_seen}）\n")
                # 健康度判断
                if pass_pct >= 90:
                    lines.append("- ✅ 章节翻译质量良好（一次通过率 ≥ 90%）\n")
                elif pass_pct >= 70:
                    lines.append("- 🟡 章节翻译质量中等（一次通过率 70-90%，失败 chunk 已 fallback 定稿）\n")
                else:
                    lines.append(f"- 🔴 章节翻译质量差（一次通过率 {pass_pct:.1f}%），建议人工复核 fallback chunk\n")
        except Exception as _qs_err:
            print(f"  ⚠️ [_print_llm_cost_report] 质量摘要统计失败: {_qs_err}", file=sys.stderr)

        # Round 33: backend diagnostics 摘要（让用户诊断 typesafe 调用失败模式）
        try:
            diag_counter = {"auth": 0, "rate_limit": 0, "timeout": 0, "network": 0,
                            "bad_request": 0, "schema_mismatch": 0, "api_error": 0,
                            "unknown": 0}
            total_diag = 0
            for cr_path in self.output_dir.glob("*_critic_report.json"):
                try:
                    with open(cr_path, "r", encoding="utf-8") as f:
                        cr = json.load(f)
                    diags = cr.get("_backend_diagnostics", {}) or {}
                    if isinstance(diags, dict):
                        for backend_key, diag in diags.items():
                            if isinstance(diag, dict):
                                cat = diag.get("category", "unknown")
                                diag_counter[cat] = diag_counter.get(cat, 0) + 1
                                total_diag += 1
                except (OSError, json.JSONDecodeError):
                    pass
            if total_diag > 0:

                lines.append("\n## Backend Diagnostics 摘要（Round 33）\n\n")
                lines.append("本章累计 backend 失败诊断数：" + str(total_diag) + "\n\n")
                lines.append("| Category | 次数 | 含义 / 建议动作 |\n")
                lines.append("|---|---:|---|\n")
                diag_desc = {
                    "auth": "🔑 认证失败 — 检查 TYPESAFE_API_KEY",
                    "rate_limit": "🚦 速率限制 — 等待或升级配额",
                    "timeout": "⏱️ 超时 — 增加 timeout 或重试",
                    "network": "🌐 网络故障 — 检查 DNS / 防火墙 / 证书",
                    "bad_request": "📝 请求格式错误 — 检查 question schema",
                    "schema_mismatch": "🔧 响应解析失败 — 更新 verdict parser",
                    "api_error": "⚠️ 服务端错误 — 检查 typesafe status page",
                    "unknown": "❓ 未分类",
                }
                for cat, n in sorted(diag_counter.items(), key=lambda kv: -kv[1]):
                    if n == 0:
                        continue
                    desc = diag_desc.get(cat, "❓ 未分类")
                    lines.append(f"| {cat} | {n} | {desc} |\n")
                # 健康度判断
                if diag_counter.get("auth", 0) > 0:
                    lines.append("\n- 🔴 **认证失败次数 > 0**：检查 .env 中的 TYPESAFE_API_KEY 是否有效\n")
                elif diag_counter.get("network", 0) > total_diag * 0.5:
                    lines.append("\n- 🔴 **网络失败占比 > 50%**：检查 DNS / 防火墙 / 证书（参考 Round 18 Jev 利用率）\n")
                elif diag_counter.get("rate_limit", 0) > 0:
                    lines.append("\n- 🟡 出现速率限制：考虑降速或升级 typesafe 配额\n")
        except Exception as _bd_err:
            print(f"  ⚠️ [_print_llm_cost_report] backend diagnostics 统计失败: {_bd_err}", file=sys.stderr)
        # Round 74 修复: 原代码从 summary.get("retry_efficiency") 读, 但 summary 此时尚未填入 retry_efficiency,
        # 导致重试效率表永远不渲染. 改为在表渲染前先计算 re_eff (本地权威), 同时供 cost_summary.json 复用.
        try:
            re_eff_total_retries = 0
            re_eff_wasted = 0
            re_eff_early_stop = 0
            for fin_path in self.output_dir.glob("*_final.json"):
                try:
                    with open(fin_path, "r", encoding="utf-8") as f:
                        fin = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(fin, dict):
                    continue
                md = fin.get("metadata", {}) or {}
                qr = int(md.get("quality_retries", 0) or 0)
                re_eff_total_retries += qr
                if md.get("fallback") and md.get("reason") == "PERMANENTLY_FAILED_AFTER_MAX_RETRIES":
                    re_eff_wasted += qr
                # Round 74: PRE_REWRITER_STAGNATION 也算早停命中
                if md.get("reason") in ("EARLY_STOP", "REWRITER_STAGNATED", "PRE_REWRITER_STAGNATION"):
                    re_eff_early_stop += 1
            re_eff_useful = max(0, re_eff_total_retries - re_eff_wasted)
            re_eff_efficiency_pct = round(re_eff_useful / re_eff_total_retries * 100, 1) if re_eff_total_retries > 0 else None
            if re_eff_total_retries > 0 or re_eff_early_stop > 0:
                eff_str = f"{re_eff_efficiency_pct:.1f}%" if re_eff_efficiency_pct is not None else "-"
                lines.append("\n## 重试效率（Round 44）\n\n")
                lines.append("| 指标 | 值 | 说明 |\n")
                lines.append("|---|---:|---|\n")
                lines.append(f"| 总重试轮次 | {re_eff_total_retries} | 所有 fallback chunk 累计 quality_retries |\n")
                lines.append(f"| 有收益重试 | {re_eff_useful} | 最终通过/安全网放行的 chunk 累计 retry |\n")
                lines.append(f"| 浪费重试 | {re_eff_wasted} | 走到 PERMANENTLY_FAILED_AFTER_MAX_RETRIES 的 retry |\n")
                lines.append(f"| 重试效率 | {eff_str} | useful_retries / total_retries |\n")
                lines.append(f"| 早停命中 | {re_eff_early_stop} | EARLY_STOP / REWRITER_STAGNATED / PRE_REWRITER_STAGNATION 触发的 chunk 数 |\n\n")
        except Exception:
            pass

        # Round 37: per-chunk 摘要表（结合 final.json metadata + critic_report scores + backend path）
        try:
            chunk_rows = []
            for fin_path in sorted(self.output_dir.glob("*_final.json")):
                cid = re.sub(r"_final\.json$", "", fin_path.name)
                try:
                    with open(fin_path, "r", encoding="utf-8") as f:
                        fin = json.load(f)
                    md = fin.get("metadata", {}) or {}
                    if md.get("fallback"):
                        status = md.get("reason", "FALLBACK")
                        emoji = {"SAFETY_NET_PASSED": "🟡", "PERMANENTLY_FAILED_AFTER_MAX_RETRIES": "🔴", "EARLY_STOP": "⏭️", "REWRITER_STAGNATED": "⏹️", "PRE_REWRITER_STAGNATION": "⏭️"}.get(status, "🔴")  # Round 74
                    else:
                        status = "PASS"
                        emoji = "✅"
                    qr = int(md.get("quality_retries", 0) or 0)
                    avg = md.get("avg_score")
                except (OSError, json.JSONDecodeError):
                    status, emoji, qr, avg = "?", "❓", 0, None

                # 配对 critic_report 拿 backend_path
                backend_path = "?"
                cr_path = self.output_dir / f"{cid}_critic_report.json"
                if cr_path.exists():
                    try:
                        with open(cr_path, "r", encoding="utf-8") as f:
                            cr = json.load(f)
                        backend_path = cr.get("_backend_path", "?")
                    except (OSError, json.JSONDecodeError):
                        pass

                chunk_rows.append((cid, emoji, status, qr, avg, backend_path))
            if chunk_rows:
                lines.append("\n## Per-Chunk 摘要（Round 37）\n\n")
                lines.append("| Chunk | 状态 | reason | 重试 | 平均分 | Backend |\n")
                lines.append("|---|---|---|---:|---:|---|\n")
                for cid, emoji, status, qr, avg, bp in chunk_rows:
                    avg_str = f"{avg:.2f}" if isinstance(avg, (int, float)) else "-"
                    lines.append(f"| {cid} | {emoji} | {status} | {qr} | {avg_str} | {bp} |\n")
                # 异常标记
                failed = [r for r in chunk_rows if r[2] not in ("PASS", "SAFETY_NET_PASSED")]
                if failed:
                    lines.append(f"\n- 🔴 {len(failed)} 个 chunk 未通过（PASS/SAFETY 之外）：{[r[0] for r in failed]}\n")
        except Exception as _pc_err:
            print(f"  ⚠️ [_print_llm_cost_report] per-chunk 摘要统计失败: {_pc_err}", file=sys.stderr)

        # Round 39: 估算花费（基于 Round 27 真实 token × 公开单价，仅参考）
        # 单价可在 cost_report 里直接编辑覆盖；缺省按国内常见公开定价
        try:
            PRICING = {
                # role → (input ¥/M tokens, output ¥/M tokens)
                "primary": (0.8, 2.0),                # MiniMax-M3
                "secondary": (0.4, 1.2),              # Round 125: Qwen-flash（估算）
                "literal_translator": (0.1, 0.2),     # deepseek-flash
                "mock": (0.0, 0.0),                   # mock 后端无花费
                "jev": (0.5, 1.5),                    # Jev (typesafe_sdk) — 估算
            }
            lines.append("\n## 估算花费（Round 39，仅供参考）\n\n")
            lines.append("基于 Round 27 真实 per-role token 累计 × 公开单价（单位 ¥，按 1M tokens 折算）：\n\n")
            lines.append("| Role | Prompt tokens | Completion tokens | 估算花费 |\n")
            lines.append("|---|---:|---:|---:|\n")
            total_cost = 0.0
            seen_roles_cost = set()
            for task_type in counter.keys():
                role = _config.get_task_role(task_type)
                if role in seen_roles_cost:
                    continue
                seen_roles_cost.add(role)
                client = get_llm_client(role)
                u = client.get_cumulative_usage()
                if u["calls"] == 0:
                    continue
                p_in, p_out = PRICING.get(role, (0.0, 0.0))
                cost = (u["prompt_tokens"] * p_in + u["completion_tokens"] * p_out) / 1_000_000
                total_cost += cost
                lines.append(
                    f"| {role} | {u['prompt_tokens']:,} | {u['completion_tokens']:,} | ¥{cost:.4f} |\n"
                )
            lines.append(f"| **合计** | | | **¥{total_cost:.4f}** |\n\n")
            lines.append("- 默认单价（¥/M tokens, input/output）：\n")
            for role, (p_in, p_out) in PRICING.items():
                lines.append(f"  - {role}: {p_in} / {p_out}\n")
            lines.append("- ⚠️ 仅为估算，实际花费以 provider 账单为准\n")
        except Exception as _cost_err:
            print(f"  ⚠️ [_print_llm_cost_report] 估算花费失败: {_cost_err}", file=sys.stderr)

        # Round 40: entity registry 章节快照（统计总实体 / 总 variant + 本章活跃实体）
        try:
            if self.decision_engine:
                all_entities = self.decision_engine.conn.execute(
                    "SELECT canonical_name, source_names, variant_names FROM entity_registry"
                ).fetchall()
                total_entities = len(all_entities)
                total_variants = 0
                # 收集本章 final.json 的所有 text 字段（拼成大字符串判断哪些实体出现过）
                chapter_text = ""
                for fin_path in self.output_dir.glob("*_final.json"):
                    try:
                        with open(fin_path, "r", encoding="utf-8") as f:
                            fin = json.load(f)
                        txt = fin.get("text", "") if isinstance(fin, dict) else ""
                        if isinstance(txt, str):
                            chapter_text += txt + "\n"
                    except (OSError, json.JSONDecodeError):
                        pass
                chapter_active = []
                for row in all_entities:
                    canonical = row[0]
                    src_names_json = row[1]
                    var_names_json = row[2] if len(row) > 2 else None
                    var_names = json.loads(var_names_json) if var_names_json else []
                    src_names = json.loads(src_names_json) if src_names_json else []
                    total_variants += len(var_names)
                    # Round 40 fix: 该实体是否在本章出现过（按 canonical / source_name / variant 任一匹配）
                    active = (canonical and canonical in chapter_text) \
                        or any(sn in chapter_text for sn in src_names if sn) \
                        or any(vn in chapter_text for vn in var_names if vn)
                    if active:
                        chapter_active.append((canonical, len(var_names)))
                lines.append("\n## Entity Registry 摘要（Round 40）\n\n")
                lines.append(f"- 全库实体总数：**{total_entities}**\n")
                lines.append(f"- 全库 variant 名总数：**{total_variants}**\n")
                lines.append(f"- 本章活跃实体（出现于 final.json）：**{len(chapter_active)}**\n\n")
                if chapter_active:
                    lines.append("| Canonical | Variant 数 |\n")
                    lines.append("|---|---:|\n")
                    for canonical, vc in sorted(chapter_active, key=lambda kv: -kv[1])[:20]:
                        lines.append(f"| {canonical} | {vc} |\n")
                    if len(chapter_active) > 20:
                        lines.append(f"| ... | (共 {len(chapter_active)} 个) |\n")
        except Exception as _er_err:
            print(f"  ⚠️ [_print_llm_cost_report] entity registry 摘要失败: {_er_err}", file=sys.stderr)

        report_text = "".join(lines)
        report_path = self.output_dir / "cost_report.md"
        report_path.write_text(report_text, encoding="utf-8")

        # Round 41: machine-readable JSON summary alongside cost_report.md
        try:
            summary = {
                "chapter_id": self.chapter_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_llm_calls": int(total),
                "total_tokens_estimated": int(total_tokens),
                "task_breakdown": {t: int(c) for t, c in counter.items() if c > 0},
            }
            # Round 27: real tokens per role
            try:
                role_tokens = {}
                for task_type in counter.keys():
                    role = _config.get_task_role(task_type)
                    if role in role_tokens:
                        continue
                    u = get_llm_client(role).get_cumulative_usage()
                    if u["calls"] > 0:
                        role_tokens[role] = {
                            "calls": u["calls"],
                            "prompt_tokens": u["prompt_tokens"],
                            "completion_tokens": u["completion_tokens"],
                            "total_tokens": u["total_tokens"],
                        }
                summary["real_tokens_by_role"] = role_tokens
            except Exception:
                summary["real_tokens_by_role"] = None
            # Round 39: estimated cost
            try:
                PRICING = {"primary": (0.8, 2.0), "secondary": (0.4, 1.2), "literal_translator": (0.1, 0.2), "mock": (0.0, 0.0), "jev": (0.5, 1.5)}
                total_cost = 0.0
                cost_by_role = {}
                for role, u in (summary.get("real_tokens_by_role") or {}).items():
                    p_in, p_out = PRICING.get(role, (0.0, 0.0))
                    cost = (u["prompt_tokens"] * p_in + u["completion_tokens"] * p_out) / 1_000_000
                    cost_by_role[role] = round(cost, 6)
                    total_cost += cost
                summary["estimated_cost_yuan_by_role"] = cost_by_role
                summary["estimated_cost_yuan_total"] = round(total_cost, 6)
            except Exception:
                summary["estimated_cost_yuan_total"] = None
            # Round 18: Jev utilization
            try:
                backend_counter = {"jev": 0, "openai_compat": 0, "mock_heuristic": 0, "legacy": 0, "unknown": 0}
                for cr_path in self.output_dir.glob("*_critic_report.json"):
                    try:
                        with open(cr_path, "r", encoding="utf-8") as f:
                            cr = json.load(f)
                        bp = cr.get("_backend_path", "unknown")
                        backend_counter[bp] = backend_counter.get(bp, 0) + 1
                    except (OSError, json.JSONDecodeError):
                        pass
                total_critic = sum(backend_counter.values())
                summary["backend_distribution"] = backend_counter
                summary["jev_utilization_pct"] = round(backend_counter["jev"] / total_critic * 100, 1) if total_critic > 0 else None
            except Exception:
                summary["jev_utilization_pct"] = None
            # Round 32: pass rate
            try:
                pass_count = sum(1 for fin_path in self.output_dir.glob("*_final.json")
                                 if not json.loads(fin_path.read_text(encoding="utf-8")).get("metadata", {}).get("fallback"))
                total_finals = sum(1 for _ in self.output_dir.glob("*_final.json"))
                summary["pass_pct"] = round(pass_count / total_finals * 100, 1) if total_finals > 0 else None
                summary["total_chunks"] = total_finals
                summary["passed_chunks"] = pass_count
            except Exception:
                summary["pass_pct"] = None
            # Round 44+74: 重试效率汇总 — 复用本函数顶部已计算的 re_eff_* 变量 (Round 74 修复 pre-render bug)
            try:
                summary["retry_efficiency"] = {
                    "total_quality_retries": re_eff_total_retries,
                    "useful_retries": re_eff_useful,
                    "wasted_retries": re_eff_wasted,
                    "efficiency_pct": re_eff_efficiency_pct,
                    "early_stop_count": re_eff_early_stop,
                }
            except Exception:
                summary["retry_efficiency"] = None
            # Round 45: 持久化 Round 35 建议值到 cost_summary.json（未来可读取）
            try:
                est_suggestions = {}
                seen_roles_r45 = set()
                for task_type in counter.keys():
                    role = _config.get_task_role(task_type)
                    if role in seen_roles_r45:
                        continue
                    seen_roles_r45.add(role)
                    client = get_llm_client(role)
                    u = client.get_cumulative_usage()
                    if u["calls"] == 0:
                        continue
                    avg = u["total_tokens"] / u["calls"]
                    current_est = est_tokens_per_call.get(task_type, 3000)
                    delta_pct = abs(avg - current_est) / current_est * 100 if current_est > 0 else 0
                    est_suggestions[role] = {
                        "task_type": task_type,
                        "real_avg_tokens": round(avg, 1),
                        "current_estimated": current_est,
                        "suggested_estimated": round(avg / 500) * 500,
                        "delta_pct": round(delta_pct, 1),
                        "needs_update": delta_pct > 30,
                    }
                summary["est_tokens_per_call_suggestion"] = est_suggestions
            except Exception:
                summary["est_tokens_per_call_suggestion"] = None
            json_path = self.output_dir / "cost_summary.json"
            json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            # Round 42: 一致性自校验（cost_summary.json vs cost_report.md）
            drift_warnings = []
            try:
                md_text = report_text
                # total_llm_calls: 章节末尾 "合计                N 次"
                import re as _re
                md_total_match = _re.search(r"合计\s+(\d+)\s+次", md_text)
                md_total = int(md_total_match.group(1)) if md_total_match else None
                if md_total is not None and md_total != summary.get("total_llm_calls"):
                    drift_warnings.append(
                        f"total_llm_calls 漂移：json={summary.get('total_llm_calls')} vs md={md_total}"
                    )
                # pass_pct: "一次通过率：**25.0%** (1/4)"
                md_pass_match = _re.search(r"一次通过率：\*\*([\d.]+)%\*\*", md_text)
                md_pass = float(md_pass_match.group(1)) if md_pass_match else None
                if md_pass is not None and summary.get("pass_pct") is not None:
                    if abs(md_pass - summary["pass_pct"]) > 0.1:
                        drift_warnings.append(
                            f"pass_pct 漂移：json={summary['pass_pct']:.1f}% vs md={md_pass:.1f}%"
                        )
                # jev_utilization_pct: "Jev 实际使用率 50.0%"
                md_jev_match = _re.search(r"Jev 实际使用率 ([\d.]+)%", md_text)
                md_jev = float(md_jev_match.group(1)) if md_jev_match else None
                if md_jev is not None and summary.get("jev_utilization_pct") is not None:
                    if abs(md_jev - summary["jev_utilization_pct"]) > 0.1:
                        drift_warnings.append(
                            f"jev_utilization_pct 漂移：json={summary['jev_utilization_pct']:.1f}% vs md={md_jev:.1f}%"
                        )
                # Round 44: retry_efficiency — wasted_retries + useful_retries 数字一致性
                re_eff = summary.get("retry_efficiency") or {}
                md_wasted_match = _re.search(r"浪费重试 \| (\d+)", md_text)
                md_wasted = int(md_wasted_match.group(1)) if md_wasted_match else None
                if md_wasted is not None and re_eff.get("wasted_retries") is not None:
                    if md_wasted != re_eff["wasted_retries"]:
                        drift_warnings.append(
                            f"retry_efficiency.wasted_retries 漂移：json={re_eff['wasted_retries']} vs md={md_wasted}"
                        )
                if drift_warnings:
                    print(
                        f"⚠️ [Cost Report Drift] {len(drift_warnings)} 项 metric 在 cost_report.md 与 cost_summary.json 之间不一致：",
                        file=sys.stderr,
                    )
                    for w in drift_warnings:
                        print(f"  - {w}", file=sys.stderr)
            except Exception as _drift_err:
                pass  # drift check failure is non-fatal
        except Exception as _js_err:
            print(f"  ⚠️ [_print_llm_cost_report] cost_summary.json dump failed: {_js_err}", file=sys.stderr)

        print()
        print("[LLM Cost] " + self.chapter_id + " 调用统计:")
        for task_type, count in sorted_types:
            if count > 0:
                print("    " + display_names.get(task_type, task_type).ljust(20) + " " + str(count).rjust(3) + " 次")
        print("    " + "合计".ljust(20) + " " + str(total).rjust(3) + " 次")
        print("    详细: " + str(report_path))

    def _cleanup_intermediates(self) -> None:
        """Round 14: 章节完成后清理 intermediate 文件

        保留：
          - chunk_xxx_final.json（最终译文）
          - chunk_xxx_literary.json（canonical 润色稿）
          - chunk_xxx_critic_report.json（最后一次 critic 报告）
          - chunk_xxx_raw.json（直译底稿）
          - chunk_xxx_literary_vN.json（N = 该 chunk 最终 quality_retries；保留最后一版）
          - chunk_xxx_rewrite_meta_vN.json（同上，最后一次）

        删除（对于 COMPLETED chunks）：
          - chunk_xxx_literary_v0..vN-1.json（已被 vN 覆盖）
          - chunk_xxx_rewrite_meta_v0..vN-1.json（同上）

        PERMANENTLY_FAILED chunks 默认保留全部中间文件（人工 inspect 需要）。
        可通过 config.pipeline.intermediate_cleanup.preserve_for_failed=false 关闭。
        """
        cfg = _config.pipeline.get("intermediate_cleanup", {}) or {}
        if not cfg.get("enabled", True):
            return
        preserve_for_failed = bool(cfg.get("preserve_for_failed", True))

        cleaned_files = 0
        kept_files = 0
        for chunk_file in sorted(self.output_dir.glob("*_final.json")):
            chunk_id = chunk_file.stem.replace("_final", "")
            # 检查 chunk 状态
            task_state = None
            try:
                tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
                for t in tasks:
                    if t.get("chunk_id") == chunk_id:
                        task_state = t.get("state")
                        break
            except Exception:
                pass

            # PERMANENTLY_FAILED 默认保留
            if preserve_for_failed and task_state == "PERMANENTLY_FAILED":
                kept_files += 1
                continue

            # 找最后一次 retry 的 quality_retries（从 DB 取）
            final_qr = 0
            try:
                tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
                for t in tasks:
                    if t.get("chunk_id") == chunk_id:
                        final_qr = int(t.get("quality_retries", 0) or 0)
                        break
            except Exception:
                pass

            # 删除 v0..vN-1（保留 vN 即最后一次）
            for r in range(final_qr):
                for pattern in [f"{chunk_id}_literary_v{r}.json", f"{chunk_id}_rewrite_meta_v{r}.json"]:
                    f = self.output_dir / pattern
                    if f.exists():
                        try:
                            f.unlink()
                            cleaned_files += 1
                        except OSError:
                            pass

        # 报告清理结果
        total_chunks = sum(1 for _ in self.output_dir.glob("*_final.json"))
        preserved_failed = sum(
            1 for t in self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
            if t.get("state") == "PERMANENTLY_FAILED"
        ) if preserve_for_failed else 0
        if cleaned_files > 0:
            print(
                f"🧹 [Cleanup] 清理 {cleaned_files} 个中间文件"
                f"（{total_chunks - preserved_failed} COMPLETED chunk 去重 + "
                f"{preserved_failed} PERMANENTLY_FAILED 保留 inspect）"
            )
        elif total_chunks > 0:
            print(
                f"🧹 [Cleanup] 无中间文件需清理（{total_chunks} chunks："
                f"{total_chunks - preserved_failed} COMPLETED 无 retry 历史 + "
                f"{preserved_failed} PERMANENTLY_FAILED 保留）"
            )

    def _entity_registry_conflict_audit(self) -> List[Dict]:
        """Round 15: 检测 entity_registry 中 source_name 冲突

        冲突定义：同一英文 source_name 被登记到 ≥2 个 canonical 时
        影响：consistency.py 无法确定替换目标，导致 Round 7/9 行为不可预期
        解决：本方法只生成诊断报告（entity_registry_conflicts.md），运维手动修复

        Returns:
            [{source_name: str, canonicals: [str, ...], conflict_type: str}, ...]
        """
        if not self.decision_engine:
            return []
        # Round 15: 查全部实体（含 pending_review=1）—— 冲突可能就藏在待核查里
        try:
            cursor = self.decision_engine.conn.execute(
                "SELECT canonical_name, source_names FROM entity_registry"
            )
            entities = [(row[0], row[1], "[]") for row in cursor.fetchall()]
        except Exception:
            return []

        # 反向索引：source_name → [canonical, ...]
        from collections import defaultdict
        src_to_canonicals = defaultdict(list)
        for row in entities:
            canonical = row[0]
            sn_json = row[1]
            sn_list = robust_json_loads(sn_json, expected_type=list) if sn_json else []
            for sn in sn_list:
                if sn:
                    src_to_canonicals[sn].append(canonical)

        # 找冲突：source_name → ≥2 canonicals
        conflicts = []
        for src_name, canonicals in src_to_canonicals.items():
            if len(canonicals) < 2:
                continue
            # 区分冲突类型
            # 同名变体（如 "Mr. Lomax" 同时给 洛马克斯先生 + 乔治·洛马克斯先生）：可能是 acceptable（同一人不同译名）
            # 跨人物冲突（如 "Joe" 同时给 安东尼·凯德 + 绅士乔）：明确错误
            conflict_type = "cross_entity"  # 默认
            canon_set = set(canonicals)
            # 简单启发式：如果所有 canonical 都包含相同的人名姓氏，可能是变体
            # 否则视为跨人物冲突
            unique_first_chars = set()
            for c in canonicals:
                # 取中文译名前 2 字作为"姓氏"
                if len(c) >= 2:
                    unique_first_chars.add(c[:2])
            if len(unique_first_chars) == 1:
                conflict_type = "variant_for_same_person"
            else:
                conflict_type = "cross_entity"

            conflicts.append({
                "source_name": src_name,
                "canonicals": canonicals,
                "conflict_type": conflict_type,
            })

        return conflicts

    def _post_pipeline_entity_audit(self) -> dict:
        """Round 9: 实体完整性审计（纯诊断，不改译文）

        对每个 chunk：
          1. 从 entity_registry 找"本章 source_text 出现过"的实体（任一 source_name in source_text）
          2. 检查 final_text 中是否含 canonical 或任一已登记 variant_names
          3. 若均缺失 → 计入 missing（提示 Rewriter 漏译）

        输出：output/{chapter_id}/entity_audit.md（人类可读）+ entity_audit.json（结构化）

        Returns:
            {chunk_id: [missing_canonical_names, ...], ...}
        """
        if not self.decision_engine:
            return {}
        try:
            from src.core.decision_engine import query_reviewable_entities
            entities = query_reviewable_entities(self.decision_engine.conn)
        except Exception:
            return {}

        # 把 chapter 内所有 chunk 的 source_text 预加载
        all_tasks = {t["chunk_id"]: t.get("text_content", "") for t in
                     self.scheduler.get_all_tasks_by_chapter(self.chapter_id)}

        report_lines = []
        report_data = {}
        for chunk_file in sorted(self.output_dir.glob("*_final.json")):
            chunk_id = chunk_file.stem.replace("_final", "")
            try:
                data = json.loads(chunk_file.read_text(encoding="utf-8"))
                text = data.get("text", "") if isinstance(data, dict) else str(data)
            except (json.JSONDecodeError, OSError):
                continue
            source_text = all_tasks.get(chunk_id, "")
            if not source_text:
                continue

            chunk_missing = []
            chunk_present = []
            for row in entities:
                canonical = row[0]
                if not canonical:
                    continue
                source_names_json = row[1]
                variant_names_json = row[2] if len(row) > 2 else None
                source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
                variant_names = robust_json_loads(variant_names_json, expected_type=list) if variant_names_json else []

                # Round 112: 身份映射 canonical（未翻译英文主名）不进入实体审计
                if not self._is_enforceable_canonical(canonical, source_names):
                    continue

                # 哪些 source_name 在本章出现过？
                # Round 106: 过滤无效 source_name（动词短语/关系代词/泛称，如 "I could"/
                # "his wife"/"the fellow"），避免 reference agent 误登记的脏指称导致
                # 审计误报"漏译"（ch05 实测 女主人公 因 "I could" 被误判 missing）。
                # Round 133b: 别名吞并抑制（与 _check_entity_consistency 同语义）—
                # 剔除被其它实体更长别名吞并的伪命中（chunk071 "Mother"⊂"Reverend Mother"）。
                valid_sns = [
                    sn for sn in source_names
                    if sn and self._is_valid_entity_source_name(sn)
                ]
                effective_sns, _swallowed = self._partition_swallowed_source_names(
                    source_text, canonical, valid_sns,
                )
                matched_sources = [
                    sn for sn in effective_sns
                    if entity_source_name_in_text(sn, source_text)
                ]
                if not matched_sources:
                    continue

                # 检查 final_text 是否含 canonical 或任一 variant
                # Round 105: 用 _canonical_present_in_text 容差判定（与一致性检查同语义），
                # 消除括注名（镇（town）/科德斯（Codders））与前缀写法（乔治·洛马克斯 vs 乔治·洛马克斯先生）的假阳性
                present_forms = [canonical] + [v for v in variant_names if v]
                found = any(self._canonical_present_in_text(form, text) for form in present_forms)
                if found:
                    chunk_present.append({
                        "canonical": canonical,
                        "matched_sources": matched_sources,
                    })
                else:
                    chunk_missing.append({
                        "canonical": canonical,
                        "matched_sources": matched_sources,
                    })

            report_data[chunk_id] = {
                "missing": chunk_missing,
                "present": chunk_present,
            }

        # 写 markdown 报告
        cfg = _config.pipeline.get("entity_audit", {}) or {}
        min_missing = int(cfg.get("min_chunk_missing_threshold", 1))
        total_missing = sum(len(d["missing"]) for d in report_data.values())

        report_lines.append(f"# 实体完整性审计 — {self.chapter_id}\n")
        report_lines.append(f"- 生成时间: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        report_lines.append(f"- 扫描 chunk 数: {len(report_data)}\n")
        report_lines.append(f"- 漏译实体总数: {total_missing}\n")
        report_lines.append(f"- 仅供诊断；不会修改 final.json\n\n")

        chunks_with_missing = [(cid, d) for cid, d in report_data.items()
                                if len(d["missing"]) >= min_missing]
        if not chunks_with_missing:
            report_lines.append("✅ 本章所有已登记实体均已在 final.json 中正确呈现。\n")
        else:
            for cid, d in chunks_with_missing:
                report_lines.append(f"## {cid} 漏译 {len(d['missing'])} 个实体\n")
                for ent in d["missing"]:
                    src_form = " / ".join(ent["matched_sources"][:3])
                    report_lines.append(f"- **{ent['canonical']}**（源文: {src_form}）\n")
                report_lines.append("\n")

        report_path = self.output_dir / "entity_audit.md"
        report_path.write_text("".join(report_lines), encoding="utf-8")

        # 结构化 JSON（供后续程序消费）
        json_path = self.output_dir / "entity_audit.json"
        json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2),
                             encoding="utf-8")

        if total_missing > 0:
            print(f"⚠️ [Entity Audit] 本章共漏译 {total_missing} 个实体（详见 entity_audit.md）")
        else:
            print(f"✅ [Entity Audit] 本章所有已登记实体均已翻译")
        return report_data

    def _recover_completed_without_final(self) -> List[str]:
        """Round 104: 恢复"COMPLETED 但 final.json 缺失"的 chunk.

        根因: 回溯 (COMPLETED -> DIRTY) 会删除 final.json (_process_dirty_batch),
        若重跑因 bug 未重新落盘 final.json, chunk 会卡在 COMPLETED 且无产出.
        此时 --force (smart-force) 会保留 COMPLETED chunk, 用户无法恢复.
        本方法在 run() 启动时检测这类 chunk 并重置为 DIRTY, 让其重新走完整流水线.

        Returns:
            被恢复的 chunk_id 列表.
        """
        recovered: List[str] = []
        try:
            tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
        except Exception:
            return recovered
        for t in tasks:
            if t.get("state") != TaskState.COMPLETED.value:
                continue
            chunk_id = t.get("chunk_id")
            if not chunk_id:
                continue
            final_path = self.output_dir / f"{chunk_id}_final.json"
            if final_path.exists():
                continue
            # COMPLETED 但无 final.json → 重置为 DIRTY 重跑 (合法转移, 不增计数)
            try:
                # Round 130（审计③）: 感知返回值 —— 拒绝时不得假报已恢复
                if self.scheduler.update_task_state(chunk_id, TaskState.DIRTY):
                    recovered.append(chunk_id)
                    print(f"♻️ [Pipeline] {chunk_id} 状态为 COMPLETED 但 final.json 缺失，重置为 DIRTY 重跑")
                else:
                    print(f"⚠️ [Pipeline] {chunk_id} 重置 DIRTY 被拒绝（非法转移），跳过（状态未变）")
            except Exception as e:
                print(f"⚠️ [Pipeline] {chunk_id} 恢复失败: {e}")
        return recovered

    def _restore_dirty_backups(self):
        """Round 107 P7: 恢复回溯重跑失败的 final_backup.json.

        回溯重跑 (COMPLETED→DIRTY) 时 final.json 被重命名为 final_backup.json.
        若重跑成功, 新 final.json 会写入; Round 130 版本择优在此裁决:
        新版 critic 均分低于旧版时保留旧版（备份版覆盖回 final.json）,
        新版至少不差时删除无用 backup.
        若重跑失败 (PERMANENTLY_FAILED), final.json 不存在, 恢复 backup:
        - final_backup.json → final.json
        - 状态 PERMANENTLY_FAILED → COMPLETED (保留之前的合格译文)
        """
        restored = []
        for backup in self.output_dir.glob("*_final_backup.json"):
            chunk_id = backup.stem.replace("_final_backup", "")
            final_path = self.output_dir / f"{chunk_id}_final.json"
            if final_path.exists():
                # 重跑成功, 新 final.json 已写入.
                # Round 130: 版本择优 — 新版 critic 均分低于旧版(超出 tolerance)时
                # 用备份版覆盖 final.json（保留更优旧版），返回 True。
                if self._apply_version_merit(chunk_id, backup, final_path):
                    continue
                # 新版至少不差 → 清理无用 backup（含 critic 备份）
                try:
                    backup.unlink()
                except OSError:
                    pass
                self._cleanup_critic_backup(chunk_id)
                continue
            # 重跑失败, 恢复 backup
            try:
                backup.rename(final_path)
                # 状态恢复: PERMANENTLY_FAILED → COMPLETED (Round 124: 显式放行的终态恢复)
                # Round 130（审计③）: 感知返回值，拒绝时不再假报 COMPLETED
                try:
                    _state_ok = self.scheduler.update_task_state(
                        chunk_id, TaskState.COMPLETED, allow_terminal_restore=True)
                except Exception as _st_exc:
                    _state_ok = False
                    print(f"  ⚠️ {chunk_id} 状态恢复异常: {_st_exc}")
                restored.append(chunk_id)
                if _state_ok:
                    print(f"♻️ [Pipeline] {chunk_id} 回溯重跑失败, 已恢复之前的 final.json (COMPLETED)")
                else:
                    print(f"⚠️ [Pipeline] {chunk_id} 已恢复 final.json，但状态恢复被拒绝（状态未变，请人工核查 chunk 状态）")
            except OSError as e:
                print(f"  ⚠️ 恢复 {chunk_id} backup 失败: {e}")
        return restored

    def _cleanup_critic_backup(self, chunk_id: str) -> None:
        """Round 130: 清理已裁决的 *_critic_report_backup.json（防止残留堆积）。"""
        cb = self.output_dir / f"{chunk_id}_critic_report_backup.json"
        if cb.exists():
            try:
                cb.unlink()
            except OSError as e:
                print(f"  ⚠️ 清理 {cb.name} 失败: {e}")

    def _apply_version_merit(self, chunk_id: str, backup: Path, final_path: Path) -> bool:
        """Round 130 版本择优: 对比备份版与新版 critic 均分, 决定是否保留旧版.

        新版均分 < 旧版均分 - tolerance → 用备份版覆盖 final.json（旧版更优），
        并把备份 critic 报告归位为 critic_report.json；返回 True。
        其余情况返回 False（调用方按"无用 backup"清理）。
        旧版/新版评分缺失（无 critic 文件，如 legacy chunk）时不做裁决。
        """
        vm_cfg = _config.pipeline.get("version_merit", {}) or {}
        if not vm_cfg.get("enabled", True):
            return False
        critic_new = self.output_dir / f"{chunk_id}_critic_report.json"
        critic_old = self.output_dir / f"{chunk_id}_critic_report_backup.json"
        old_score = self._critic_avg_from_file(critic_old)
        new_score = self._critic_avg_from_file(critic_new)
        if old_score <= 0 or new_score <= 0:
            return False
        try:
            tolerance = float(vm_cfg.get("tolerance", 0.0) or 0.0)
        except (TypeError, ValueError):
            tolerance = 0.0
        if new_score >= old_score - tolerance:
            return False
        try:
            final_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            backup.unlink()
            if critic_old.exists():
                critic_old.replace(critic_new)  # 旧版评分归位, 保持 final 与评分一致
        except OSError as e:
            print(f"  ⚠️ [Version Merit] {chunk_id} 恢复旧版本失败: {e}")
            return False
        print(f"🛡️ [Version Merit] {chunk_id} 新版均分 {new_score:.2f} < 旧版 {old_score:.2f}，保留旧版")
        return True

    def _critic_avg_from_file(self, path: Path) -> float:
        """读取 critic_report json 的数值维度均分; 缺失/损坏返回 0.0。"""
        try:
            return self._compute_avg_score(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def _verify_final_outputs(self, expected_count: int, strict: bool = True) -> bool:
        """校验本章 final.json 实际落盘是否匹配任务。

        防"任务队列空即报成功"的假阳性完成。除比数外，还比对 chunk_id
        集合：残留旧 final（上一轮多余 chunk）或缺失都判残缺，避免 stale 假通过。

        Round 76: 分离 completeness 与 quality 校验。
          - completeness 失败（缺 final.json / 残留旧文件）→ return False（真残缺）
          - 仅 quality 降级（state=PERMANENTLY_FAILED 但 final.json 已落盘的 fallback 章节）
            → 视为"产出完整但质量降级"，return True，仅 warning 告知。
        旧版本将 fallback 章节也判 False，导致 CLI 退出码 = 1 与磁盘上
        已有 final.json 的事实不一致；调用方可读 cost_report 看到 fallback 分类。

        Round 119 T4: DIRTY 单独处理 —— DIRTY 且带 final 是"回溯标记未整合"的
        陈旧产物，return False（先 consolidate 收敛再合并；DIRTY 不入成书）。

        返回 True 表示产出完整（可能含 fallback 降级），False 表示真残缺。
        """
        if expected_count <= 0:
            return True
        actual_files = list(self.output_dir.glob("*_final.json"))
        actual_ids = {f.stem.rsplit("_final", 1)[0] for f in actual_files}
        expected_ids = {t["chunk_id"] for t in self.scheduler.get_all_tasks_by_chapter(self.chapter_id)}

        missing = expected_ids - actual_ids
        stale = actual_ids - expected_ids
        if missing or stale:
            print(f"❌ [Pipeline] 章节 {self.chapter_id} 产出残缺/残留："
                  f"期望 {len(expected_ids)} 个，缺失 {len(missing)}，残留旧文件 {len(stale)}。"
                  f"请用 --force 重新初始化后重跑。")
            if missing:
                print(f"   缺失 chunk: {sorted(missing)}")
            if stale:
                print(f"   残留 chunk: {sorted(stale)}")
            return False
        if strict:
            all_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
            dirty_chunks = [t["chunk_id"] for t in all_tasks
                            if t["state"] == TaskState.DIRTY.value]
            if dirty_chunks:
                print(f"❌ [Pipeline] 章节 {self.chapter_id} 有 {len(dirty_chunks)} 个 chunk 处于 DIRTY（陈旧待整合）："
                      f"{dirty_chunks}")
                print(f"   请运行 consolidate 收敛 DIRTY 后再合并（T4: DIRTY 不入成书）。")
                return False
            bad = [t["chunk_id"] for t in all_tasks
                    if t["state"] != TaskState.COMPLETED.value]
            if bad:
                fallback_chunks = []
                missing_chunks = []
                for cid in bad:
                    if (self.output_dir / f"{cid}_final.json").exists():
                        fallback_chunks.append(cid)
                    else:
                        missing_chunks.append(cid)
                if fallback_chunks:
                    print(f"⚠️ [Pipeline] 章节 {self.chapter_id} 有 {len(fallback_chunks)} 个 chunk 质量未达标(fallback)，"
                          f"译文已写入但状态为 PERMANENTLY_FAILED：{fallback_chunks}")
                if missing_chunks:
                    print(f"❌ [Pipeline] 章节 {self.chapter_id} 有 {len(missing_chunks)} 个 chunk 完全无产出：{missing_chunks}")
                    return False
                # Round 76: 仅 fallback（无 missing）→ 产出完整但质量降级，warning 已告知。
                return True
        return True

    def _process_pending_batch(self, tasks):
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS)

    def _process_dirty_batch(self, tasks):
        for task in tasks:
            chunk_id = task['chunk_id']
            print(f"🔄 [Pipeline] {chunk_id} 标记为 DIRTY，重置到术语提取")
            # Round 108: 清空上一轮的分数历史/最佳版本 — 回溯重跑是新 episode,
            # 旧历史会污染 streak 早停 (ch08 chunk006: 旧历史 8.157→7.943 与重跑
            # 7.94 拼成「单调不增」假阳性, 重跑仅 1 次 retry 就被提前 fallback)。
            self._score_history.pop(chunk_id, None)
            self._best_lit.pop(chunk_id, None)
            self._ensure_candidate_pools()
            self._lit_candidates.pop(chunk_id, None)
            self._pairwise_info.pop(chunk_id, None)
            # Round 107 P7: final.json 不删除, 重命名为 final_backup.json.
            # ch07 倒查: 回溯重跑 (COMPLETED→DIRTY) 删除 final.json 后, 若重跑失败
            # (NO_PROGRESS_STREAK 等), chunk 从 COMPLETED 降级为 PERMANENTLY_FAILED 且无产出.
            # 保留备份: 重跑成功会覆盖, 重跑失败则 _restore_dirty_backups 恢复.
            final_path = self.output_dir / f"{chunk_id}_final.json"
            backup_path = self.output_dir / f"{chunk_id}_final_backup.json"
            if final_path.exists():
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                    final_path.rename(backup_path)
                    print(f"  💾 已备份 final.json → final_backup.json (重跑失败可恢复)")
                except OSError as e:
                    print(f"  ⚠️ 备份 final.json 失败: {e}")
            # Round 130: critic_report 保留为 *_critic_report_backup.json，
            # 供章末"版本择优"（_apply_version_merit）对比新旧评分。
            critic_path = self.output_dir / f"{chunk_id}_critic_report.json"
            critic_backup = self.output_dir / f"{chunk_id}_critic_report_backup.json"
            if critic_path.exists():
                try:
                    if critic_backup.exists():
                        critic_backup.unlink()
                    critic_path.rename(critic_backup)
                except OSError as e:
                    print(f"  ⚠️ 备份 critic_report 失败: {e}")
            for step in ["raw", "literary"]:
                intermediate = self.output_dir / f"{chunk_id}_{step}.json"
                if intermediate.exists():
                    try:
                        intermediate.unlink()
                    except OSError as e:
                        print(f"  ⚠️ 清理失败 {intermediate}: {e}")
            # 清理版本历史文件
            for pattern in [f"{chunk_id}_literary_v*.json", f"{chunk_id}_rewrite_meta_v*.json"]:
                for f in self.output_dir.glob(pattern):
                    try:
                        f.unlink()
                    except OSError as e:
                        print(f"  ⚠️ 清理失败 {f}: {e}")
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS, reset_counters=True)

    def _consolidate_deferred_backtracks(self) -> bool:
        """Round 130: 章末合并式回溯（decision_engine.backtrack_mode=deferred）。

        在批处理循环"队列清空"点调用：把 decision_engine 登记的本章待回溯 chunk
        合并应用一次（标记 DIRTY → 主循环继续重跑这些块）。

        治理预算（防自激循环）：
          - 章级：每章每次运行最多应用 max_backtrack_sweeps_per_chapter 轮（默认 1）；
            预算用尽后仅冻结（队列登记保留至下一次运行，打印告警）。
          - 单块：累计回溯轮数上限由 scheduler.trigger_backtrack 执行（超限冻结，
            并写出 backtrack_frozen.md 明细）。

        Returns:
            True 表示有 chunk 被标记 DIRTY（调用方应继续主循环）。
        """
        de_cfg = _config.decision_engine or {}
        mode = str(de_cfg.get("backtrack_mode", "deferred") or "deferred").lower()
        if mode == "immediate":
            return False
        if not getattr(self, "decision_engine", None):
            return False
        try:
            max_sweeps = int(de_cfg.get("max_backtrack_sweeps_per_chapter", 1) or 0)
        except (TypeError, ValueError):
            max_sweeps = 1
        sweeps_done = int(getattr(self, "_backtrack_sweeps_done", 0) or 0)
        task_ids = [t["chunk_id"] for t in self.scheduler.get_all_tasks_by_chapter(self.chapter_id)]
        if sweeps_done >= max_sweeps:
            pending = self.decision_engine.count_pending_backtracks(task_ids)
            if pending:
                print(f"🧊 [Backtrack] 本章合并预算已用尽（{max_sweeps} 轮），"
                      f"冻结 {pending} 项待回溯登记（保留至下次运行）")
            return False
        entries = self.decision_engine.take_pending_backtracks(task_ids)
        if not entries:
            return False
        stats = self.scheduler.trigger_backtrack([e["chunk_id"] for e in entries])
        self._backtrack_sweeps_done = sweeps_done + 1
        dirty = stats.get("dirty", []) if isinstance(stats, dict) else []
        over_cap = stats.get("over_cap", []) if isinstance(stats, dict) else []
        print(f"🧩 [Backtrack] 章末合并回溯：登记 {len(entries)} 个 → 标记 DIRTY {len(dirty)} 个"
              + (f"，治理冻结 {len(over_cap)} 个（超单块预算）" if over_cap else ""))
        if over_cap:
            self._write_backtrack_frozen_report(over_cap, entries)
        return bool(dirty)

    def _write_backtrack_frozen_report(self, frozen_ids: List[str], entries: List[Dict]) -> None:
        """把治理冻结的 chunk 写入 output/<chapter>/backtrack_frozen.md（诊断用）。"""
        try:
            meta = {e["chunk_id"]: e for e in entries if isinstance(e, dict)}
            lines = ["# 回溯治理冻结记录 (Round 130)\n\n",
                     "- 冻结原因：单块累计回溯轮数达到 max_backtrack_rounds_per_chunk\n\n"]
            for cid in frozen_ids:
                e = meta.get(cid, {})
                lines.append(f"- `{cid}` ← source_key={e.get('source_key', '?')} "
                             f"(queue_rounds={e.get('queue_rounds', '?')})\n")
            path = self.output_dir / "backtrack_frozen.md"
            path.write_text("".join(lines), encoding="utf-8")
            print(f"  📄 冻结明细: {path}")
        except OSError as e:
            print(f"  ⚠️ 写 backtrack_frozen.md 失败: {e}")

    # ────────────────────────────────────────────────────────────────
    # Round 133: 重试适配 — 「仅推理无正文」(no_content) 签名驱动的
    # 自动降档（max→xhigh→high→…）与换角色救稿（重试预算耗尽后最后 1 次）。
    # 背景：ch01 通宵跑实测 7 块 PERMANENTLY_FAILED 全部为 effort=max 思考链
    # （5-7 万字符）吃光 max_tokens、正文为空；同参数重试必然复现。
    # 只对命中签名的块生效（override 按块持久化），正常块零影响。
    # ────────────────────────────────────────────────────────────────
    _NO_CONTENT_MARKERS = ("无正文", "未闭合 think 块", "reasoning_only")

    @classmethod
    def _is_no_content_signature(cls, err_text) -> bool:
        """判定异常文本是否为「仅推理无正文」签名（thinking 耗尽 max_tokens）。"""
        text = str(err_text or "")
        return any(m in text for m in cls._NO_CONTENT_MARKERS)

    @staticmethod
    def _effort_levels_table() -> Dict[str, tuple]:
        """provider → 合法 effort 档位表（优先 src.config 单一事实源；单体兜底内置副本）。"""
        try:
            from src.config import EFFORT_LEVELS_BY_PROVIDER as _tbl
            return dict(_tbl)
        except Exception:
            return {
                "minimax": ("low", "medium", "high", "xhigh", "max"),
                "deepseek": ("none", "low", "high", "xhigh", "max"),
                "qwen": ("none", "low", "medium", "xhigh"),
            }

    @classmethod
    def _provider_of_role(cls, role: str) -> Optional[str]:
        """按角色配置的 model_name/api_base 推断 effort provider；未知返回 None。"""
        try:
            role_models = getattr(_config, "role_models", None)
            if callable(role_models):
                role_models = role_models()
            if isinstance(role_models, dict) and role_models and role not in role_models:
                return None  # 未定义角色：不用 legacy 兜底模型猜 provider
        except Exception:
            pass
        try:
            role_cfg = _config.get_role_config(role) or {}
        except Exception:
            return None
        target = f"{role_cfg.get('model_name', '')} {role_cfg.get('api_base', '')}".lower()
        for provider in cls._effort_levels_table():
            if provider in target:
                return provider
        return None

    def _retry_adapt_cfg(self) -> Dict[str, Any]:
        cfg = dict(_config.pipeline.get("retry_adaptation", {}) or {})
        cfg.setdefault("enabled", True)
        cfg.setdefault("downgrade_on_signature", ["no_content"])
        cfg.setdefault("fallback_role_switch", True)
        cfg.setdefault("minimax_protect", True)
        cfg.setdefault("quality_escalate_back", True)
        return cfg

    def _load_model_capabilities(self) -> Dict[str, Any]:
        """读取 L1 档位探测结果（db/model_capabilities.json）；缺失/过期 → 空 dict。"""
        try:
            probe_cfg = dict(_config.pipeline.get("model_probe", {}) or {})
        except Exception:
            probe_cfg = {}
        rel = str(probe_cfg.get("path") or "db/model_capabilities.json")
        path = Path(rel)
        if not path.is_absolute():
            root = getattr(self, "_root_dir", None) or Path.cwd()
            path = Path(root) / rel
        try:
            if not path.exists():
                return {}
            ttl_days = float(probe_cfg.get("ttl_days", 7) or 7)
            if ttl_days > 0 and (time.time() - path.stat().st_mtime) > ttl_days * 86400:
                return {}
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _effort_ladder_for_role(self, role: str) -> tuple:
        """角色可用档位阶梯：provider 全档位 → 探测文件 accepted 过滤（有则用）。"""
        provider = self._provider_of_role(role)
        if not provider:
            return ()
        ladder = tuple(self._effort_levels_table().get(provider) or ())
        caps = self._load_model_capabilities()
        rec = dict((caps.get("models") or {}).get(role) or {})
        levels_rec = dict(rec.get("levels") or {})
        if rec.get("provider") == provider and levels_rec:
            accepted = [lv for lv in ladder if (levels_rec.get(lv) or {}).get("accepted")]
            if accepted:
                return tuple(accepted)
        return ladder

    def _downgrade_one_step(self, role: str, effort) -> Optional[str]:
        """按 provider 档位表降一档；降无可降/档位未知 → None。"""
        ladder = self._effort_ladder_for_role(role)
        eff = str(effort or "").strip().lower()
        if not ladder or not eff or eff not in ladder:
            return None
        i = ladder.index(eff)
        return ladder[i - 1] if i > 0 else None

    def _escalate_one_step(self, role: str, effort) -> Optional[str]:
        ladder = self._effort_ladder_for_role(role)
        eff = str(effort or "").strip().lower()
        if not ladder or not eff or eff not in ladder:
            return None
        i = ladder.index(eff)
        return ladder[i + 1] if i + 1 < len(ladder) else None

    def _get_retry_override(self, chunk_id: str) -> Dict[str, Any]:
        try:
            return dict(self.scheduler.get_retry_override(chunk_id) or {})
        except Exception:
            return {}

    def _set_retry_override(self, chunk_id: str, payload: Dict[str, Any]) -> bool:
        try:
            return bool(self.scheduler.set_retry_override(chunk_id, payload))
        except Exception:
            return False

    def _task_default_effort(self, task_name: str) -> Optional[str]:
        try:
            _, params = _config.resolve_task_model(task_name)
            eff = params.get("effort")
            return str(eff).strip().lower() if eff else None
        except Exception:
            return None

    def _note_last_call(self, chunk_id: str, stage: str, role: str, effort, model_name: str = "") -> None:
        """记录块最近一次 LLM 调用的 (stage, role, effort)，供 no_content 签名记录使用。"""
        if not chunk_id:
            return
        cache = getattr(self, "_last_call_info", None)
        if cache is None:
            cache = {}
            self._last_call_info = cache
        cache[chunk_id] = {
            "stage": stage,
            "role": role or "",
            "effort": (str(effort).strip().lower() if effort else None),
            "model_name": model_name or "",
        }

    def _record_no_content_signal(self, chunk_id: str, stage: str) -> None:
        """命中「仅推理无正文」签名 → 持久化降档 override（下次重试自动降一档）。"""
        cfg = self._retry_adapt_cfg()
        if not cfg.get("enabled", True):
            return
        sigs = {str(s).strip().lower() for s in (cfg.get("downgrade_on_signature") or ["no_content"])}
        if "no_content" not in sigs:
            return
        info = (getattr(self, "_last_call_info", {}) or {}).get(chunk_id) or {}
        used_role = info.get("role") or ""
        used_eff = info.get("effort")
        prev = self._get_retry_override(chunk_id)
        next_eff = self._downgrade_one_step(used_role, used_eff) if used_eff else None
        if next_eff is None and prev.get("effort"):
            next_eff = prev.get("effort")  # 已降无可降：保留现有 override
        delta = int(prev.get("down_delta", 0) or 0) + (
            1 if (next_eff and next_eff != prev.get("effort")) else 0)
        payload = {
            "sig": "no_content",
            "stage": stage,
            "role": prev.get("role") or used_role,
            "effort": next_eff,
            "used_role": used_role,
            "used_effort": used_eff,
            "down_delta": delta,
            "rescue_used": bool(prev.get("rescue_used")),
            "escalated": bool(prev.get("escalated")),
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if self._set_retry_override(chunk_id, payload):
            if next_eff:
                print(f"🔻 [Pipeline] {chunk_id} no_content 签名 → 下次重试降档 "
                      f"{used_eff or 'default'}→{next_eff} (role={payload['role'] or '?'})")
            else:
                print(f"🔻 [Pipeline] {chunk_id} no_content 签名（{used_eff or 'default'} 已无更低档）")

    def _rescue_role_for(self, stage: str, failing_role: str) -> Optional[str]:
        """换角色救稿目标选择（fallback_role_switch）。

        RAW 救稿默认 secondary/literal_translator（minimax_protect 时不选 primary）；
        REWRITE 救稿排除失败角色，protect 时优先 secondary。
        """
        cfg = self._retry_adapt_cfg()
        protect = bool(cfg.get("minimax_protect", True))
        if stage == "TRANSLATING_RAW":
            cands = ["secondary", "literal_translator"]
            if not protect:
                cands.append("primary")
        else:
            if protect and failing_role != "secondary":
                cands = ["secondary", "literal_translator", "primary"]
            else:
                cands = ["primary", "secondary", "literal_translator"]
        cands = [r for r in cands if r and r != failing_role]
        return cands[0] if cands else None

    def _rescue_effort_for(self, target_role: str, prev_effort) -> Optional[str]:
        """救稿档位：跨家映射到目标 role 合法档位后，再降一档（避免同档复现 no_content）。"""
        ladder = self._effort_ladder_for_role(target_role)
        eff = str(prev_effort or "").strip().lower()
        if not ladder or not eff:
            return None
        mapped = _map_effort_for_fallback(eff, ladder)
        if mapped is None:
            return None
        i = ladder.index(mapped)
        return ladder[i - 1] if i > 0 else ladder[i]

    def _maybe_rescue_no_content(self, task) -> Optional[str]:
        """重试预算耗尽的救稿决策。

        返回 recovery stage value（已写入换角色 override，调用方把任务放回该阶段再试 1 次）；
        返回 None 表示放弃救稿（照旧转 PERMANENTLY_FAILED）。
        """
        cfg = self._retry_adapt_cfg()
        if not cfg.get("enabled", True) or not cfg.get("fallback_role_switch", True):
            return None
        last_error = task.get("last_error", "") or ""
        if not self._is_no_content_signature(last_error):
            return None
        chunk_id = task.get("chunk_id", "")
        ov = self._get_retry_override(chunk_id)
        if ov.get("rescue_used"):
            return None
        stage = ov.get("stage") or self._get_recovery_stage(last_error).value
        failing_role = ov.get("used_role") or ov.get("role") or ""
        rescue_role = self._rescue_role_for(stage, failing_role)
        if not rescue_role:
            return None
        prev_eff = ov.get("used_effort") or ov.get("effort")
        rescue_eff = self._rescue_effort_for(rescue_role, prev_eff)
        payload = dict(ov)
        payload.update({
            "sig": "no_content",
            "stage": stage,
            "role": rescue_role,
            "effort": rescue_eff,
            "used_role": failing_role,
            "used_effort": prev_eff,
            "rescue_used": True,
            "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if not self._set_retry_override(chunk_id, payload):
            # fail-safe：override 落盘失败则放弃救稿（否则 rescue_used 无法持久化 → 无限重入）
            print(f"⚠️ [Pipeline] {chunk_id} 救稿 override 落盘失败，放弃换角色（转 PF）")
            return None
        print(f"🆘 [Pipeline] {chunk_id} no_content 重试耗尽 → 换角色救稿: "
              f"{failing_role or '?'}@{prev_eff or 'default'} → {rescue_role}@{rescue_eff or 'default'}")
        return stage

    def _get_recovery_stage(self, last_error: str) -> TaskState:
        if not last_error:
            return TaskState.EXTRACTING_TERMS
        # 兼容两种格式：
        # 1) [STAGE_NAME] 错误描述
        # 2) JSON {"judge_reason": ...}（quality_retry 路径）
        stage_match = re.search(r'^\[(\w+)\]', last_error)
        if stage_match:
            stage = stage_match.group(1)
        elif '"judge_reason"' in last_error:
            stage = "REWRITING_LITERARY"
        else:
            return TaskState.EXTRACTING_TERMS
        STAGE_MAP = {
            "EXTRACTING_TERMS": TaskState.EXTRACTING_TERMS,
            "TRANSLATING_RAW": TaskState.TRANSLATING_RAW,
            "REWRITING_LITERARY": TaskState.REWRITING_LITERARY,
            "AUDITING": TaskState.AUDITING,
            "JUDGING": TaskState.JUDGING,
        }
        return STAGE_MAP.get(stage, TaskState.EXTRACTING_TERMS)

    def _try_fallback_final(self, chunk_id: str) -> bool:
        """API 级重试耗尽时，从已有中间产物抢救 fallback final.json

        优先用文学润色稿，回退到直译稿。若两者都不存在则不保存。
        Returns True 如果成功保存了 fallback。
        """
        for step, label in [("literary", "文学润色稿"), ("raw", "直译稿")]:
            try:
                data = self._load_intermediate(chunk_id, step)
                text = data if isinstance(data, str) else data.get("text", str(data))
                if text and text.strip():
                    self._save_final_with_sanity(chunk_id, {
                        "text": text,
                        "metadata": {
                            "fallback": True,
                            "reason": f"API_FAILURE_FALLBACK_FROM_{step.upper()}",
                            "source_step": step,
                        }
                    }, source_step="literary")
                    print(f"  ↪ 已从{label}抢救 fallback final: {chunk_id}")
                    return True
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
        return False

    def _process_failed_batch(self, tasks):
        retry_by_stage = {}
        permanent_fail_ids = []
        max_retries = _config.pipeline.get("max_retries", 3)
        for task in tasks:
            chunk_id = task['chunk_id']
            retries = task.get('retries', 0)
            if retries >= max_retries:
                # Round 133: no_content 签名 + 预算耗尽 → 换角色救稿一次（默认开启）
                rescue_stage = self._maybe_rescue_no_content(task)
                if rescue_stage:
                    stage_state = self._get_recovery_stage(f"[{rescue_stage}]")
                    retry_by_stage.setdefault(stage_state, []).append(chunk_id)
                    continue
                print(f"❌ [Pipeline] {chunk_id} 重试次数过多，转入 PERMANENTLY_FAILED 终态")
                self._try_fallback_final(chunk_id)
                permanent_fail_ids.append(chunk_id)
            else:
                recovery_stage = self._get_recovery_stage(task.get('last_error', ''))
                retry_by_stage.setdefault(recovery_stage, []).append(chunk_id)
                print(f"🔁 [Pipeline] {chunk_id} 重试中 (第 {retries + 1} 次) → {recovery_stage.value}")
        if permanent_fail_ids:
            self.scheduler.batch_update_state(
                permanent_fail_ids,
                TaskState.PERMANENTLY_FAILED,
                error_msgs={cid: f"超过重试上限 (retries>={max_retries})，需人工介入" for cid in permanent_fail_ids}
            )
        for stage, ids in retry_by_stage.items():
            if ids:
                print(f"🔁 [Pipeline] {len(ids)} 个任务恢复到 {stage.value}")
                self.scheduler.batch_update_state(ids, stage)

    def _process_extracting_terms_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                self.ref_agent.process_chunk(chunk_id, source_text, affected_chunks=[chunk_id])
                self._llm_call_counter["reference_extraction"] += 1  # Round 13
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 术语提取异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[EXTRACTING_TERMS] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.TRANSLATING_RAW)

    def _process_translating_raw_batch(self, tasks):
        success_ids = []
        # Round 96 P2: raw 翻译必须含中文. ch02 案例中 LLM 返回了英文源文 (或模型混淆),
        # 该英文稿被喂给 rewriter 造成连锁失败. 此处加 CJK 闸门, 不通过则 FAILED 触发 TRANSLATING_RAW 重试.
        try:
            raw_cjk_threshold = float(_config.pipeline.get("min_raw_cjk_ratio", 0.3))
        except (TypeError, ValueError):
            raw_cjk_threshold = 0.3
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                self._seed_pun_decisions(source_text)  # L2: pun_watchlist 命中即强制落 decision_db
                raw_text = self._run_raw_translator(source_text, chunk_id)
                self._llm_call_counter["literal_translation"] += 1  # Round 13
                # Round 96 P2: 检查 raw_text 是否真的是中文 (LLM 偶发返回英文源文)
                raw_cjk = compute_cjk_ratio(raw_text) if raw_text else 0.0
                if raw_cjk_threshold > 0 and raw_cjk < raw_cjk_threshold:
                    print(
                        f"⚠️ [Pipeline] {chunk_id} raw 翻译 CJK={raw_cjk:.3f} < "
                        f"{raw_cjk_threshold:.3f} (LLM 返回英文?), FAILED 重试"
                    )
                    # Round 96 P2: 删除 raw 中间产物, 避免污染下游 fallback
                    try:
                        (self.output_dir / f"{chunk_id}_raw.json").unlink(missing_ok=True)
                    except OSError:
                        pass
                    self.scheduler.update_task_state(
                        chunk_id,
                        TaskState.FAILED,
                        error_msg=(
                            f"[TRANSLATING_RAW] raw_not_chinese "
                            f"(cjk_ratio={raw_cjk:.3f} < {raw_cjk_threshold:.3f}, "
                            f"len={len(raw_text or '')})"
                        ),
                    )
                    continue  # 不进 success_ids
                self._save_intermediate(chunk_id, "raw", raw_text)
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 直译异常: {e}")
                if self._is_no_content_signature(e):
                    self._record_no_content_signal(task['chunk_id'], "TRANSLATING_RAW")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[TRANSLATING_RAW] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.REWRITING_LITERARY)

    def _process_rewriting_batch(self, tasks):
        success_ids = []
        entity_feedback: str = ''
        entity_registry_for_rewriter: Dict[str, str] = {}
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                quality_retries = task.get('quality_retries', 0)
                low_dims = []
                dim_scores = {}

                reject_reason_raw = task.get('last_error', '')
                reject_reason = reject_reason_raw
                critic_feedback = None
                specific_edits = []
                # Round 64: 默认空，由 feedback['_typesafe_verdict'] 填充
                skipped_dimensions: List[str] = []
                if reject_reason_raw:
                    # P2-1: 用 robust_json_loads 抗 critic 报告畸形；返回 {} 走原 raw 字符串兜底
                    feedback = robust_json_loads(reject_reason_raw, expected_type=dict)
                    if feedback:
                        reject_reason = feedback.get('judge_reason', '')
                        critic_feedback = feedback.get('critic_suggestions', '')
                        # 回路修复：提取 entity_consistency_notes（之前一直被丢弃，导致 Rewriter 看不到实体反馈）
                        entity_feedback = feedback.get('entity_consistency_notes', '') or ''
                        specific_edits = feedback.get("specific_edits", [])
                        low_dims = feedback.get('low_dims', [])
                        dim_scores = feedback.get('scores', {})
                        # Round 64: 从 _typesafe_verdict 提取 skipped_dimensions（避免对边界判断维度做无效重写）
                        typesafe_verdict = feedback.get('_typesafe_verdict', {}) or {}
                        skipped_dimensions = list(typesafe_verdict.get('skipped_dimensions', []) or [])
                        if low_dims:
                            low_dim_detail = ", ".join(
                                f"{d}({dim_scores.get(d, '?')}分)" for d in low_dims
                            )
                            critic_feedback = f"低分维度：{low_dim_detail}\n" + (critic_feedback or '')

                all_feedback = []
                if reject_reason_raw and quality_retries > 0:
                    for i in range(quality_retries):
                        try:
                            meta = self._load_intermediate(chunk_id, f"rewrite_meta_v{i}")
                            all_feedback.append({
                                "reason": (meta.get("reject_reason_summary") or "")[:80],
                                "resolved": False,
                            })
                        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                            pass

                prev_lit_text = None
                try:
                    prev_lit_text = self._load_intermediate(chunk_id, "literary")
                except (FileNotFoundError, KeyError):
                    pass

                try:
                    raw_text = self._load_intermediate(chunk_id, "raw")
                except (FileNotFoundError, KeyError):
                    print(f"⚠️ [Pipeline] {chunk_id} 缺失 raw 底稿，转入 FAILED 等待恢复")
                    self.scheduler.update_task_state(chunk_id, TaskState.FAILED, error_msg="[TRANSLATING_RAW] raw file missing, needs re-translation")
                    continue

                narrative_memory = self._load_prior_memory(chunk_id)
                # 回路修复：构建 entity_registry 注入 Rewriter（让 Rewriter 看到完整登记表以自我核对）
                entity_registry_for_rewriter = self._build_entity_registry_for_rewriter(source_text)

                # Round 73: pre-rewriter 停滞早停 — 在调用 rewriter LLM 前检查上一次 reject categories
                # 与上上一次是否完全相同 (Round 69 synonym mapping 后). 若相同, rewriter 不会再产生
                # 新效果, 直接 fallback 省一次昂贵的 LLM 调用 + 配套 critic + judge 三连击.
                # 区别于 post-rewriter stagnation (Round 12): 后者跑完一轮 LLM 才判断, 这里在 LLM 调用前拦截.
                stag_cfg_p73 = (_config.pipeline.get("oscillation_detection", {}) or {}).get("stagnation_detection", {}) or {}
                if bool(stag_cfg_p73.get("enabled", True)) and quality_retries >= 1:
                    stag_consecutive_p73 = int(stag_cfg_p73.get("consecutive_required", 1))
                    if quality_retries >= 1 + stag_consecutive_p73:
                        try:
                            prev_meta_obj = self._load_intermediate(
                                chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                            )
                            prev_reject_summary = (prev_meta_obj.get("reject_reason_summary", "") or "") if prev_meta_obj else ""
                            if prev_reject_summary:
                                prev_prev_meta_obj = self._load_intermediate(
                                    chunk_id, f"rewrite_meta_v{quality_retries - 1 - stag_consecutive_p73}"
                                )
                                prev_prev_summary = (prev_prev_meta_obj.get("reject_reason_summary", "") or "") if prev_prev_meta_obj else ""
                                if prev_prev_summary:
                                    prev_cats = self._extract_reject_categories(prev_prev_summary)
                                    curr_cats = self._extract_reject_categories(prev_reject_summary)
                                    if prev_cats and prev_cats == curr_cats:
                                        # Round 92: 用跨 retry 最佳文学稿 (而非当前被拒的 prev_lit_text)
                                        candidate = prev_lit_text or self._load_intermediate(chunk_id, "literary") or ""
                                        final_text = self._get_best_lit(chunk_id, candidate)
                                        self._save_final_with_sanity(chunk_id, {
                                            "text": final_text,
                                            "metadata": {
                                                "fallback": True,
                                                "reason": "PRE_REWRITER_STAGNATION",
                                                "judge_reason": prev_reject_summary[:200],
                                                "quality_retries": quality_retries,
                                                "stagnant_categories": sorted(prev_cats),
                                            }
                                        }, source_step="literary")
                                        self.scheduler.update_task_state(
                                            chunk_id, TaskState.PERMANENTLY_FAILED,
                                            error_msg=f"[PRE_REWRITER_STAGNATION] 跳过 rewriter LLM: cats {sorted(prev_cats)} 连续 {stag_consecutive_p73 + 1} 次不变"
                                        )
                                        print(f"⏹️ [Pipeline] {chunk_id} pre-rewriter 停滞 (retry={quality_retries}, cats={sorted(prev_cats)}), 跳过 rewriter LLM")
                                        continue
                        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                            # meta 缺失 (首次 retry), 正常走 rewriter
                            pass

                # Round 65: 重试计入实际 LLM 调用次数（之前 +=1 漏算 retry，导致 cost_report 严重低估 literary_rewriting 成本）
                self._llm_call_counter["literary_rewriting"] += 1 + quality_retries
                # Round 116: 停滞换模型（生成端多样性）— 决定本 chunk 本轮润色所用角色
                sw_cfg = _config.pipeline.get("stagnation_model_switch", {}) or {}
                sw_enabled = bool(sw_cfg.get("enabled", False))
                sw_base_role = sw_cfg.get("base_role") or _config.get_task_role("literary_rewrite")
                sw_alt_role = sw_cfg.get("alternate_role") or ""
                sw_state = None
                model_role = sw_base_role
                if sw_enabled and sw_alt_role:
                    sw_state = self._read_rewriter_switch_state(
                        chunk_id, sw_base_role, sw_alt_role, sw_cfg.get("max_switches", 1)
                    )
                    model_role = sw_state.get("current_role") or sw_base_role
                # Round 129: 换到 alternate（更强模型）时加档 effort
                model_effort = (sw_cfg.get("alternate_effort") if model_role != sw_base_role else None)
                # Round 133: no_content 重试适配 override（降档/换角救稿）优先于停滞换档；
                # 降档后若因质量重进润色（quality_retries≥1）→ 升档补偿一次（防死循环）。
                ov = self._get_retry_override(chunk_id)
                if ov and ov.get("sig") == "no_content" and ov.get("stage") == "REWRITING_LITERARY":
                    if ov.get("role"):
                        model_role = ov["role"]
                    if ov.get("effort"):
                        model_effort = str(ov["effort"]).strip().lower()
                    raccfg = self._retry_adapt_cfg()
                    if (raccfg.get("quality_escalate_back", True)
                            and not ov.get("escalated")
                            and model_effort
                            and quality_retries >= 1
                            and int(ov.get("down_delta", 0) or 0) > 0):
                        up_eff = self._escalate_one_step(model_role, model_effort)
                        if up_eff and up_eff != model_effort:
                            print(
                                f"🔼 [Pipeline] {chunk_id} 降档后质量重进润色 → 升档补偿一次 "
                                f"{model_effort}→{up_eff} (role={model_role})"
                            )
                            model_effort = up_eff
                            ov["escalated"] = True
                            ov["effort"] = up_eff
                            self._set_retry_override(chunk_id, ov)
                    if model_effort:
                        print(
                            f"🔻 [Pipeline] {chunk_id} 重试适配生效: role={model_role} "
                            f"effort={model_effort}"
                        )
                self._note_last_call(
                    chunk_id, "REWRITING_LITERARY", model_role,
                    model_effort or self._task_default_effort("literary_rewrite"),
                    get_role_model_name(model_role),
                )
                raw_result = self.rewriter_agent.process_chunk(
                    chunk_id, raw_text, self.style_guide, source_text,
                    prev_lit_text=prev_lit_text,
                    reject_reason=reject_reason,
                    critic_feedback=critic_feedback,
                    entity_feedback=entity_feedback,        # 回路修复
                    entity_registry=entity_registry_for_rewriter,  # 回路修复
                    retry_count=quality_retries,
                    all_feedback=all_feedback,
                    specific_edits=specific_edits,
                    narrative_memory=narrative_memory,
                    low_dims=low_dims,
                    skipped_dimensions=skipped_dimensions,  # Round 64
                    verse_note=self._verse_note_for(chunk_id, source_text),  # Round 119 L1
                    model_role=model_role,  # Round 116: 停滞换模型
                    model_effort=model_effort,  # Round 129: 换模型加档
                )
                # process_chunk 返回 (lit_text, self_review) 元组
                if isinstance(raw_result, tuple):
                    lit_text, self_review = raw_result
                else:
                    lit_text, self_review = raw_result, None  # 向后兼容旧返回值
                self._save_intermediate(chunk_id, "literary", lit_text)

                # L3: 高价值片段（verse/pun-dense）双稿仲裁 — 仅首轮，每 chunk 一次
                if quality_retries == 0 and lit_text:
                    try:
                        lit_text = self._dual_draft_run(
                            chunk_id, source_text, raw_text, prev_lit_text,
                            reject_reason, critic_feedback, entity_feedback,
                            all_feedback, specific_edits, narrative_memory,
                            low_dims, skipped_dimensions, lit_text,
                        )
                        self._save_intermediate(chunk_id, "literary", lit_text)
                    except Exception as _dd_exc:
                        print(f"⚠️ [Pipeline] {chunk_id} 双稿仲裁异常，保留首稿: {_dd_exc}", file=sys.stderr)

                # Round 36: rewriter 字节级停滞检测 — 当 retry 输出与 prev_lit_text 完全相同时，
                # 说明 LLM 没消化任何反馈，继续 retry 只是浪费 token。直接走 fallback。
                # （区别于 Round 6 oscillation：基于相似度阈值；此处为精确字节比较，
                #  两者并行运行，任一触发即 fallback。）
                if prev_lit_text and lit_text == prev_lit_text and quality_retries >= 1:
                    print(f"⏹️ [Pipeline] {chunk_id} rewriter 输出与 prev_lit_text 完全相同 (retry={quality_retries})，触发 REWRITER_STAGNATED 早停")
                    stag_meta = {'retry_count': quality_retries, 'quality_retries': quality_retries, 'reject_reason_summary': (reject_reason or '')[:200], 'critic_feedback_summary': (critic_feedback or '')[:200], 'all_feedback_summaries': all_feedback, 'self_review': self_review, 'stagnation': 'REWRITER_STAGNATED', 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}
                    # Round 116: 字节停滞优先换模型再试一轮；达上限后走原 fallback
                    if sw_enabled and sw_alt_role and sw_state and sw_state.get("can_switch"):
                        switch_to = sw_state.get("next_role") or sw_alt_role
                        self._record_rewriter_model_switch(
                            chunk_id, model_role, switch_to, quality_retries, "REWRITER_STAGNATED"
                        )
                        stag_meta["model_role"] = model_role
                        stag_meta["model_switch_to"] = switch_to
                        self._save_intermediate(chunk_id, f'rewrite_meta_v{quality_retries}', stag_meta)
                        print(f"🔁 [Pipeline] {chunk_id} 字节停滞 retry={quality_retries}：换模型 {model_role} → {switch_to} 再试一轮")
                        continue
                    self._save_intermediate(chunk_id, f'rewrite_meta_v{quality_retries}', stag_meta)
                    self._save_final_with_sanity(chunk_id, {
                        "text": lit_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "REWRITER_STAGNATED",
                            "judge_reason": f"rewriter 在 retry={quality_retries} 时输出与 prev_lit_text 完全相同，未消化任何反馈",
                            "quality_retries": quality_retries,
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=f"rewriter 字节停滞 retry={quality_retries}"
                    )
                    continue

                meta = {
                    "retry_count": quality_retries,
                    "quality_retries": quality_retries,
                    "reject_reason_summary": (reject_reason or "")[:200],
                    "critic_feedback_summary": (critic_feedback or "")[:200],
                    "all_feedback_summaries": all_feedback,
                    "self_review": self_review,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._save_intermediate(chunk_id, f"literary_v{quality_retries}", lit_text)

                # Round 6: oscillation 早停检测（similarity-based）
                # Round 12: stagnation 早停检测（reject_reason-based）——并行运行，任一触发即 fallback
                osc_cfg = _config.pipeline.get("oscillation_detection", {}) or {}
                osc_enabled = bool(osc_cfg.get("enabled", True))
                osc_sim_threshold = float(osc_cfg.get("similarity_threshold", 0.95))
                osc_consecutive = int(osc_cfg.get("consecutive_required", 2))
                stag_cfg = osc_cfg.get("stagnation_detection", {}) or {}
                stag_enabled = bool(stag_cfg.get("enabled", True))
                stag_consecutive = int(stag_cfg.get("consecutive_required", 2))
                stag_exact_only = bool(stag_cfg.get("exact_match_only", True))

                is_oscillation = False
                osc_sim = 0.0
                if osc_enabled and prev_lit_text and quality_retries >= 1:
                    osc_sim = self._compute_text_similarity(prev_lit_text, lit_text)
                    if osc_sim >= osc_sim_threshold:
                        is_oscillation = True

                # Round 12 stagnation 检测（reject_reason-based）
                # Round 16: 默认用 primitive-category 集合对比（Jev-aware）
                #   - 旧 exact_match_only=true 永远不触发（Jev noul 每次浮动）
                #   - match_mode=category 比较 primitive 集合（"实体一致性违反" vs "noul=0.33"）
                #   - match_mode=exact 保留原字符串严格匹配
                #   - match_mode=substring 保留旧"包含关系"宽松匹配
                current_reject_summary = (reject_reason or "")[:200]
                is_stagnation = False
                match_mode = stag_cfg.get("match_mode")  # Round 16 新增
                if match_mode is None:
                    # 向后兼容：未指定 match_mode 时按 exact_match_only 推断
                    match_mode = "exact" if stag_exact_only else "substring"
                if stag_enabled and quality_retries >= 1 and current_reject_summary:
                    try:
                        prev_meta_obj = self._load_intermediate(
                            chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                        )
                        prev_reject_summary = prev_meta_obj.get("reject_reason_summary", "") or ""
                        if match_mode == "category":
                            # Round 16: 比较 primitive category 集合（忽略 noul 浮动）
                            prev_cats = self._extract_reject_categories(prev_reject_summary)
                            curr_cats = self._extract_reject_categories(current_reject_summary)
                            is_stagnation = bool(prev_cats) and prev_cats == curr_cats
                        elif match_mode == "exact":
                            is_stagnation = (prev_reject_summary == current_reject_summary)
                        else:  # "substring"
                            is_stagnation = (
                                prev_reject_summary in current_reject_summary
                                or current_reject_summary in prev_reject_summary
                            ) if prev_reject_summary else False
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                        pass

                # 把 oscillation / stagnation 信息写进 meta，便于下一 retry 读取
                meta["oscillation"] = is_oscillation
                meta["oscillation_similarity"] = round(osc_sim, 4) if osc_sim else 0.0
                meta["oscillation_threshold"] = osc_sim_threshold
                meta["stagnation"] = is_stagnation
                meta["reject_reason_summary"] = current_reject_summary
                self._save_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}", meta)

                # 触发条件：连续 oscillation / stagnation 任一满足 → 强制 fallback
                # 语义：consecutive_required=N 表示当前 + 前 N 次都需一致（总共 N+1 个一致信号）
                # Round 12 默认 N=1：当前 + 前 1 次 stagnation = 2 个一致信号即触发（更激进，省 LLM）
                should_force_fallback = False
                force_reason = ""
                if is_oscillation and quality_retries >= osc_consecutive:
                    try:
                        prev_meta_obj = self._load_intermediate(
                            chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                        )
                        # 累计 oscillation 计数：当前 + 前 N-1 次 prev_meta 是否都 oscillation
                        osc_count = 1  # 当前
                        cur_meta = prev_meta_obj
                        for _ in range(osc_consecutive - 1):
                            if not cur_meta or not bool(cur_meta.get("oscillation", False)):
                                osc_count = 0
                                break
                            try:
                                nxt_meta = self._load_intermediate(
                                    chunk_id, f"rewrite_meta_v{quality_retries - 2 - _}"
                                )
                                cur_meta = nxt_meta
                                osc_count += 1
                            except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                                osc_count = 0
                                break
                        if osc_count >= osc_consecutive:
                            should_force_fallback = True
                            force_reason = (
                                f"连续{osc_consecutive + 1}次 retry 相似度"
                                f"≥{osc_sim_threshold:.0%}（sim={osc_sim:.4f}）"
                            )
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                        pass
                if not should_force_fallback and is_stagnation and quality_retries >= stag_consecutive:
                    try:
                        prev_meta_obj = self._load_intermediate(
                            chunk_id, f"rewrite_meta_v{quality_retries - 1}"
                        )
                        stag_count = 1  # 当前
                        cur_meta = prev_meta_obj
                        for _ in range(stag_consecutive - 1):
                            if not cur_meta or not bool(cur_meta.get("stagnation", False)):
                                stag_count = 0
                                break
                            try:
                                nxt_meta = self._load_intermediate(
                                    chunk_id, f"rewrite_meta_v{quality_retries - 2 - _}"
                                )
                                cur_meta = nxt_meta
                                stag_count += 1
                            except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                                stag_count = 0
                                break
                        if stag_count >= stag_consecutive:
                            should_force_fallback = True
                            force_reason = (
                                f"连续{stag_consecutive + 1}次 retry reject_reason 未变（Rewriter 没修）"
                            )
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
                        pass

                if should_force_fallback:
                    print(
                        f"⏹️ [Pipeline] {chunk_id} retry={quality_retries} 强制 fallback：{force_reason}"
                    )
                    # Round 116: 强制 fallback 前优先换模型再试一轮
                    if sw_enabled and sw_alt_role and sw_state and sw_state.get("can_switch"):
                        switch_to = sw_state.get("next_role") or sw_alt_role
                        self._record_rewriter_model_switch(
                            chunk_id, model_role, switch_to, quality_retries,
                            f"STAGNATION:{force_reason}"
                        )
                        switch_meta = {
                            "retry_count": quality_retries,
                            "quality_retries": quality_retries,
                            "reject_reason_summary": current_reject_summary,
                            "stagnation": "STAGNATION_MODEL_SWITCH",
                            "model_role": model_role,
                            "model_switch_to": switch_to,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                        self._save_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}", switch_meta)
                        print(f"🔁 [Pipeline] {chunk_id} 强制 fallback 前换模型 {model_role} → {switch_to} 再试：{force_reason}")
                        continue
                    # Round 92: fallback 写入跨 retry 最佳版本, 而非当前被拒 lit_text
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text)
                    self._save_final_with_sanity(chunk_id, best_text)
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=f"[STAGNATION] {force_reason}，提前 fallback",
                    )
                    continue  # 跳过 success_ids，不再进 AUDITING

                success_ids.append(chunk_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ [Pipeline] {task['chunk_id']} 润色异常: {e}")
                if self._is_no_content_signature(e):
                    self._record_no_content_signal(task['chunk_id'], "REWRITING_LITERARY")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[REWRITING_LITERARY] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.AUDITING)

    @staticmethod
    def _strip_parenthetical(name: str) -> str:
        """Round 105: 剥离规范名中的括注（全角/半角），返回纯名。

        例: 镇（town）→镇 / 科德斯（Codders）→科德斯 / 卡特拉姆勋爵（第九代…）→卡特拉姆勋爵
        """
        if not name:
            return name
        import re as _re_p
        return _re_p.sub(r"（[^）]*）|\([^)]*\)", "", name).strip()

    def _canonical_present_in_text(self, canonical: str, lit_text: str) -> bool:
        """Round 105: 判定规范名（或其显著部分）是否出现在译文中。

        用于 missing 核查，消除两类系统性假阳性（ch32 倒查根因）：
          1. 规范名带括注（镇（town）/科德斯（Codders）/卡特拉姆勋爵（第九代…））
             — 译文永远不会含括注，整串匹配必然 missing。
          2. 前缀/后缀写法（译文用 乔治·洛马克斯，规范名 乔治·洛马克斯先生）
             — 同一人不同形式，不应判硬 missing（会触发 rewriter 互斥指令死循环）。

        判定为"已出现"（返回 True）的条件（任一）：
          - 完整规范名在 lit_text；
          - 去括注后的 base（len≥2）在 lit_text；
          - base 的显著前缀或后缀（长度 ≥ max(3, 2/3·len)）在 lit_text；
          - base 长度 < 2（如 镇（town）→镇）：单字名无法可靠核验，视为已出现
            （抑制 missing，避免对单字名死循环）。
        """
        if not lit_text or not canonical:
            return False
        if canonical in lit_text:
            return True
        base = self._strip_parenthetical(canonical)
        if not base:
            return False
        if len(base) < 2:
            # 单字 base（镇/城/国…）无法可靠核验，抑制 missing 假阳性
            return True
        if base != canonical and base in lit_text:
            return True
        # Round 115: 斜杠备选 canonical（如 "国民党/民族主义者"）— 译文通常只用其中
        # 一个备选（ch06 chunk012 用「民族主义者」），整串/前后缀容差都会差在斜杠上
        # → 误判 missing → 实体硬失败死循环。任一备选字面命中即算在场。
        if "/" in base or "／" in base:
            for alt in re.split(r"[/／]", base):
                alt = alt.strip()
                if len(alt) >= 2 and alt in lit_text:
                    return True
        ref = base
        if len(ref) < 4:
            # 短名（2-3 字）不做前缀/后缀容差，避免误判（如 黛西 的前缀 黛 太短）
            return ref in lit_text
        threshold = max(3, (len(ref) * 2) // 3)
        if threshold >= len(ref):
            return ref in lit_text
        prefix = ref[:threshold]
        suffix = ref[-threshold:]
        return prefix in lit_text or suffix in lit_text

    # ch06 倒查：reference agent 把非实体短语登记为 source_name（sir/his eyelids/
    # the support/a sharp command），导致实体一致性核查误报 missing → 硬失败死循环。
    _NON_ENTITY_SOURCE_NAMES = {
        "sir", "madam", "mister", "miss", "mr", "mrs", "ms",
        "the man", "the woman", "the fellow", "the other",
        "a man", "a woman", "a gentleman", "a lady",
        # Round 106: ch05 倒查补充的泛称/描述（reference agent 误登记）
        "the lady", "the gentleman", "the young man", "the old man",
        "an englishwoman", "the englishwoman", "the local castle",
        "he (implied)", "she (implied)", "it (implied)",
        # Round 107: ch03 倒查补充 — 代词/反身代词/描述性短语
        "himself", "herself", "itself", "themselves",
        "he/his/him", "she/her/her", "it/its",
        "my boy", "my man", "my friend", "my old friend",
        "an old friend of yours", "a friend of yours", "an old friend",
        "this young fellow", "that young fellow", "this old fellow",
        "this man", "that man", "this woman", "that woman",
    }
    _NON_ENTITY_POSSESSIVE_NOUNS = {
        "eyelids", "eyelid", "impressions", "impression",
        "face", "eyes", "hands", "voice", "head",
        # Round 106: 关系名词（his/her + 关系 = 指代另一人，非本实体专名）
        "wife", "husband", "brother", "sister", "son", "daughter",
        "mother", "father", "parent", "child", "children",
        "friend", "friends", "colleague", "colleagues",
        # Round 107: ch03 倒查补充
        "boy", "man", "fellow", "gentleman", "lady",
    }
    # Round 106: 动词短语（代词主语 + 情态/助动词/常见动词）— 明显非实体指称
    # （ch05 实测 "I could"/"I should"/"I had" 被登记为女主人公 source_name → 误报 missing）
    _NON_ENTITY_VERB_RE = re.compile(
        r"^(i|he|she|it|we|they)\s+"
        r"(could|should|would|might|may|must|can|will|shall|had|has|have|did|do|does|"
        r"is|am|are|was|were|said|told|asked|looked|turned|stood|sat|walked|ran|"
        r"came|went|felt|knew|thought|saw|heard|made|got|took|gave|put|let|set|"
        r"began|started|tried|kept|left|meant|happened|seemed|appeared|proved)\b"
    )
    # Round 107: 代词/反身代词（含斜杠分隔的多代词，如 "he/his/him"）
    _NON_ENTITY_PRONOUN_RE = re.compile(
        r"^(i|you|he|she|it|we|they|me|him|her|us|them|my|your|his|her|its|our|their|"
        r"mine|yours|hers|ours|theirs|myself|yourself|himself|herself|itself|ourselves|"
        r"themselves)(/(\w+))*$"
    )
    # Round 107: 描述性短语（冠词 + 形容词 + 名词 + of + 代词，如 "an old friend of yours"）
    _NON_ENTITY_DESCR_RE = re.compile(
        r"^(a|an|the)\s+\w+\s+\w+\s+of\s+(my|your|his|her|its|our|their|yours|his|hers)$"
    )
    # Round 107: this/that + 名词短语（2-3 词, 如 "this country", "this young fellow"）
    # Round 107 P8: 扩展到 2 词 (this/that + 单个普通名词), ch07 倒查 "this country" 误提取.
    _NON_ENTITY_THIS_THAT_RE = re.compile(
        r"^(this|that)\s+\w+(\s+\w+)?$"
    )
    # Round 107 P3: like + 名词短语（如 "Like all women"）— 比较/类比短语, 非实体指称
    _NON_ENTITY_LIKE_RE = re.compile(
        r"^like\s+\w+(\s+\w+){0,3}$"
    )
    # Round 111: 限定词 + 全小写普通名词短语（如 "the stairs"/"the chauffeur"/"my maid"/
    # "the unemployed young man"）— 叙事中的泛指描述，非专名。
    # ch09 chunk004 根因: reference agent 把 "the stairs"→楼梯 登记为实体后，实体核查按
    # "规范主名必须字面出现" 判 missing → 硬失败退回重写；而中文习惯动词化表达（上楼/下楼），
    # rewriter 无法两全 → 输出字节停滞 → REWRITER_STAGNATED → PERMANENTLY_FAILED。
    # 判定基于原文大小写（大小写敏感）：真专名首字母大写（the Comrades/the King/
    # the Tower），不受影响；仅当限定词后的所有词全小写时视为普通名词短语。
    _NON_ENTITY_ARTICLE_NOUN_RE = re.compile(
        r"^(?:The|the|A|a|An|an|My|my|Your|your|His|his|Her|her|Its|its|Our|our|Their|their)\s+"
        r"[a-z][a-z'\-]*(\s+[a-z][a-z'\-]*){0,3}$"
    )
    # Round 113: 主语代词开头的叙事从句（如 "I wanted to know"/"I haven't written"/
    # "he (指代 Professor Wynward)"）— 是叙事文本，不是实体指称。
    # ch16 chunk006 根因: reference agent 把叙事从句登记为 "女主人公（未具名叙述者）"
    # 的 source_name；Round 106 的动词白名单是有限枚举（"wanted/haven't written"
    # 不在表内）被再次击穿 → 核查要求描述性主名字面出现 → 不可满足 → 实体硬失败 →
    # rewriter 停滞 → PERMANENTLY_FAILED。主语代词 + 任何后续内容一律视为从句。
    _NON_ENTITY_CLAUSE_RE = re.compile(
        r"^(i|you|he|she|it|we|they)(['’][a-z]+)?\s+\S"
    )
    # Round 113: 描述性占位 canonical（无名角色的描述串，如
    # "女主人公（未具名叙述者）"/"美国绅士（未具名）"）— 译文永远不会字面出现整串，
    # 核查必然判 missing → 硬失败死循环；这类实体没有可核查的专名，不进入核查/审计/
    # registry（与 Round 111/112 同语义：防不可满足的硬约束）。
    _NON_ENFORCEABLE_CANONICAL_RE = re.compile(
        r"未具名|无具名|无名|匿名|未给出姓名|隐含"
    )

    @classmethod
    def _is_valid_entity_source_name(cls, sn: str) -> bool:
        """判定 source_name 是否为有效实体指称（而非泛称/身体部位/普通短语）。"""
        if not sn:
            return False
        low = sn.strip().lower()
        if not low:
            return False
        if low in cls._NON_ENTITY_SOURCE_NAMES:
            return False
        m = re.match(r"^(his|her|its|their)\s+(\w+)$", low)
        if m and m.group(2) in cls._NON_ENTITY_POSSESSIVE_NOUNS:
            return False
        # Round 106: 动词短语（代词主语 + 动词）
        if cls._NON_ENTITY_VERB_RE.match(low):
            return False
        # Round 113: 主语代词 + 后续内容的从句（泛化 Round 106 的有限动词白名单，
        # 覆盖 "I wanted to know"/"I haven't written" 等叙事片段）
        if cls._NON_ENTITY_CLAUSE_RE.match(low):
            return False
        # Round 107: 代词/反身代词（含斜杠分隔的多代词）
        if cls._NON_ENTITY_PRONOUN_RE.match(low):
            return False
        # Round 107: 描述性短语（冠词 + 形容词 + 名词 + of + 代词）
        if cls._NON_ENTITY_DESCR_RE.match(low):
            return False
        # Round 107: this/that + 形容词 + 名词
        if cls._NON_ENTITY_THIS_THAT_RE.match(low):
            return False
        # Round 107 P3: like + 名词短语 (类比/比较, 非实体指称)
        if cls._NON_ENTITY_LIKE_RE.match(low):
            return False
        # Round 111: 限定词 + 全小写普通名词短语（用原文大小写判定, 不能走 low）
        if cls._NON_ENTITY_ARTICLE_NOUN_RE.match(sn.strip()):
            return False
        return True

    @classmethod
    def _is_enforceable_canonical(cls, canonical: str, source_names: list) -> bool:
        """Round 112: 判定 canonical 是否可作实体核查基准（防"未翻译 canonical"死循环）。

        ch10 chunk007 根因: 登记表 canonical=" Cloak Room"（ASCII + 前导空格，与
        source_name "Cloak Room" 规范化后恒等）→ 核查要求中文译文出现英文串，
        而正常译法是「行李寄存票/衣帽间」→ 不可满足 → 实体硬失败退回重写 →
        rewriter 字节停滞 → REWRITER_STAGNATED → PERMANENTLY_FAILED。

        判定: canonical 无 CJK 且与某个 source_name 规范化（折叠空白+小写）后相等
        → 身份映射（reference agent 未给出中文译名），不含翻译信息，不进入核查/
        registry（Morris Cowley 等"保留英文原名"场景由译文自然处理，无需强约束）。

        Round 113: 描述性占位 canonical（未具名/无名/匿名 等标记）同样剔除 —
        ch16 chunk006 "女主人公（未具名叙述者）" 根因（字面要求不可满足）。
        """
        if not canonical:
            return False
        cs = re.sub(r"\s+", " ", canonical).strip()
        if not cs:
            return False
        if cls._NON_ENFORCEABLE_CANONICAL_RE.search(cs):
            return False
        if re.search(r"[\u4e00-\u9fff\u3400-\u4dbf]", cs):
            return True
        norm = lambda s: re.sub(r"\s+", " ", (s or "")).strip().lower()
        if any(norm(sn) == norm(cs) for sn in (source_names or []) if sn):
            return False
        return True

    def _find_near_variant(self, canonical: str, lit_text: str) -> Optional[str]:
        """Round 108: 检测 lit_text 中与规范名一字之差的近变体（译名漂移）。

        ch07 chunk016 根因：规范名「艾琳·布伦特夫人」，译文写成「伊琳·布伦特夫人」
        （一字之差）。本地核查的后缀容差（「特夫人」）抑制了 missing，已登记变体表
        又为空 → 本地核查零报告，只有 Jev 能发现，而拒因不可执行 → rewriter 停滞。
        这里主动检测一字之差近变体，报 variant_instead_of_canonical（可执行）。

        只处理含分隔符（·）的规范名（如 艾琳·布伦特夫人）：按 CJK 段扫描一字之差
        串，重构后整串须出现在 lit_text（强精度守卫）。
        纯 CJK 规范名不做检测（ch07 chunk007 实测：「敲诈者」与动词短语「敲诈谁/
        敲诈勒」一字之差碰撞，CJK 无空格无法可靠分词，误报会诱导 rewriter 改坏正文）。

        Returns:
            找到近变体串则返回，否则 None。
        """
        if not lit_text or not canonical:
            return None
        base = self._strip_parenthetical(canonical)
        if not base or len(base) < 3:
            return None
        try:
            from src.utils.entity_stopwords import is_plausible_name_variant
        except Exception:
            is_plausible_name_variant = None

        def _one_char_diff(a: str, b: str) -> bool:
            if len(a) != len(b):
                return False
            return sum(1 for x, y in zip(a, b) if x != y) == 1

        def _plausible(ref: str, cand: str) -> bool:
            if is_plausible_name_variant is None:
                return True
            try:
                ok, _reason = is_plausible_name_variant(ref, cand)
                return bool(ok)
            except Exception:
                return True

        # 只处理含分隔符的规范名（如 艾琳·布伦特夫人）— 分隔符提供天然 token 边界，
        # 重构守卫（整串须出现在 lit_text）保证精度。纯 CJK 规范名不做检测（见 docstring）。
        if re.fullmatch(r'[\u4e00-\u9fff]+', base):
            return None
        segments = [s for s in re.split(r'[^\u4e00-\u9fff]+', base) if len(s) >= 2]
        if not segments:
            return None
        for seg in segments:
            length = len(seg)
            cands = set(re.findall(r'(?=([\u4e00-\u9fff]{%d}))' % length, lit_text))
            cands = {c for c in cands if c != seg and _one_char_diff(seg, c) and _plausible(seg, c)}
            if not cands:
                continue
            # 尝试 base 中 seg 的每个出现位置
            start = 0
            while True:
                idx = base.find(seg, start)
                if idx < 0:
                    break
                for cand in cands:
                    near = base[:idx] + cand + base[idx + length:]
                    if near in lit_text:
                        return near
                start = idx + 1
        return None

    def _get_sibling_canonicals(self) -> Dict[str, Set[str]]:
        """Round 71/133b: 全 registry 的 source_name → {canonical} 映射（含 pending 行）。

        lazy-init + 按审计批次缓存（_process_auditing_batch 入口重置 _sibling_canonicals_cache）。
        兼容测试桩：测试用 __new__ 构造的实例也可用（hasattr 兜底）。
        """
        if not hasattr(self, "_sibling_canonicals_cache") or \
                self._sibling_canonicals_cache is None:
            try:
                self._sibling_canonicals_cache = get_sibling_canonicals_by_source(
                    self.decision_engine.conn
                )
            except Exception:
                self._sibling_canonicals_cache = {}
        return self._sibling_canonicals_cache or {}

    def _partition_swallowed_source_names(
        self, source_text: str, canonical: str, valid_sns: List[str],
    ) -> Tuple[List[str], Set[str]]:
        """Round 133b: 别名吞并抑制 — 剔除"被其它实体更长别名吞并"的伪命中。

        chunk071 根因: entity_registry 里 "Mother" 被同时登记给 杰西卡/杰西卡夫人，
        而 chunk071 原文中 "Mother" 唯一出现于 "Reverend Mother"（盖乌斯·海伦·莫希阿姆
        的别名）内部 → Round 114 词边界防护放行（"Mother" 是合法词）→ 杰西卡夫人被
        判"原文在场但译文缺失" → rewriter 收到不可执行指令（"Reverend Mother" 那处
        必须出现「杰西卡夫人」）→ 改坏正文 → NO_PROGRESS_STREAK → PERMANENTLY_FAILED。

        判定: 别名 a 的所有命中 span 都被某"更长注册别名 b"(len(b) > len(a), a 是 b 的
        子串, b 归属某 canonical ≠ 当前 canonical) 的命中 span 包含 → a 为伪命中, 剔除;
        只要一处未被吞并即保留（真在场）。

        Returns:
            (effective, swallowed): effective=真在场别名（按原顺序）;
            swallowed=出现在原文但被吞并的别名集合。
        """
        if not valid_sns:
            return [], set()
        siblings = self._get_sibling_canonicals()
        own_spans_cache: Dict[str, List[Tuple[int, int]]] = {}
        coverer_spans_cache: Dict[str, List[Tuple[int, int]]] = {}

        effective: List[str] = []
        swallowed: Set[str] = set()
        for sn in valid_sns:
            if not sn:
                continue
            if sn not in own_spans_cache:
                own_spans_cache[sn] = entity_source_name_spans(sn, source_text)
            spans = own_spans_cache[sn]
            if not spans:
                continue
            # 候选覆盖别名: 更长 + 包含 sn + 归属于≥1 个其它 canonical
            coverers = [
                other for other in siblings.keys()
                if len(other) > len(sn) and sn in other
                and any(oc != canonical for oc in (siblings.get(other) or set()))
                and self._is_valid_entity_source_name(other)
            ]
            fully_swallowed = False
            if coverers:
                cov_spans: List[Tuple[int, int]] = []
                for other in coverers:
                    if other not in coverer_spans_cache:
                        coverer_spans_cache[other] = entity_source_name_spans(other, source_text)
                    cov_spans.extend(coverer_spans_cache[other])
                fully_swallowed = all(
                    any(cs <= s and e <= ce for (cs, ce) in cov_spans)
                    for (s, e) in spans
                )
            if fully_swallowed:
                swallowed.add(sn)
            else:
                effective.append(sn)
        return effective, swallowed

    def _check_entity_consistency(self, chunk_id: str, lit_text: str, source_text: str) -> List[Dict]:
        """实体一致性核查：检查译文中实体译名是否与 entity_registry 一致

        Returns:
            List[Dict]: 不一致项列表，每项包含 entity, expected, found, chunk_id
        """
        if not self.decision_engine:
            return []

        # P3-1: 改用公共 helper（语义：所有非待核查实体，不过滤 confidence）
        entities = query_reviewable_entities(self.decision_engine.conn)

        if not entities:
            return []

        # Round 71: 跨人物冲突抑制 — 同一英文 source_name 被登记到多个 canonical 时,
        # 若 lit_text 中出现任何一个 sibling canonical (共享 source_name 的其它译名),
        # 则视为 rewriter 用了别名译名, 不报 "missing". variant_instead_of_canonical
        # 不受影响 (variant 是真问题, sibling 命中不影响 variant 路径).
        # 缓存: 单次审计内多次调用 _check_entity_consistency 时复用同一份.
        sibling_canonicals = self._get_sibling_canonicals()

        inconsistencies = []
        for row in entities:
            canonical = row[0]
            source_names_json = row[1]
            source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
            variant_names = robust_json_loads(row[2], expected_type=list) if len(row) > 2 and row[2] else []

            if len(canonical) < 2:
                continue
            # Round 112: 身份映射（canonical ≡ source_name 且无 CJK）不进入核查 —
            # ch10 chunk007 " Cloak Room" 根因（不可满足的英文主名要求）
            if not self._is_enforceable_canonical(canonical, source_names):
                continue

            # B2 修复：只检查本章原文出现的实体（按 source_text 过滤），
            # 避免全库实体每次 AUDITING 都被遍历、variant 误发现累积、
            # quality_retries 无限增长。修前 ch32 quality_retries=112+ 死循环。
            # ch06: 过滤非实体 source_name（sir/his eyelids 等），避免误判"实体出场"
            valid_source_names = [sn for sn in source_names if self._is_valid_entity_source_name(sn)]
            # Round 133b: 别名吞并抑制 — 剔除被其它实体更长别名吞并的伪命中
            # （chunk071 "Mother"⊂"Reverend Mother" 根因, 详见 _partition_swallowed_source_names）
            effective_source_names, _swallowed = self._partition_swallowed_source_names(
                source_text, canonical, valid_source_names,
            )
            appears_in_source = any(
                (sn and entity_source_name_in_text(sn, source_text))
                for sn in effective_source_names
            )
            if not appears_in_source:
                continue

            # Round 21: 收集所有出现在 lit_text 中的 variant（不只第一个）
            matched_variants = []  # [(variant_found_in_lit, source_name_matched)]
            for vn in variant_names:
                found_form = self._case_insensitive_find(vn, lit_text)
                if found_form:
                    # 反查该 variant 在原文中以哪种 source_name 形式出现
                    matched_src = None
                    for sn in source_names:
                        if self._case_insensitive_find(sn, source_text):
                            matched_src = sn
                            break
                    matched_variants.append((found_form, matched_src))

            # Round 21: 收集所有出现在 lit_text 中的 source_name
            matched_source_names = []
            for sn in source_names:
                found_form = self._case_insensitive_find(sn, lit_text)
                if found_form:
                    matched_source_names.append(found_form)

            # 去重：source_name 命中后也算 variant_instead_of_canonical（英文残留）
            all_violations = list(matched_variants)
            for sn in matched_source_names:
                if not any(sn == src for _, src in matched_variants):
                    all_violations.append((sn, sn))

            near_variant = None
            if all_violations:
                # Round 21: 为每一个违规单独报告一条 inconsistency（rewriter 一次性全部修）
                for found_val, src_name in all_violations:
                    inconsistencies.append({
                        "entity": canonical,
                        "expected": canonical,
                        "found": found_val,
                        "source_name": src_name,
                        "type": "variant_instead_of_canonical",
                        "chunk_id": chunk_id,
                    })
                # 只要有违规就 continue，不再走 missing 分支（variant 已表明该实体被引用）
                # 但仍要 try 发现新变体（_maybe_discover_entity_variant 可能在 variant_names 已包含的 case 误触发）
            else:
                # Round 108: 一字之差近变体检测（译名漂移，如 伊琳·布伦特夫人 vs 艾琳·布伦特夫人）。
                # 比 missing 更可执行：给 rewriter 具体的 found→expected 替换清单。
                near_variant = self._find_near_variant(canonical, lit_text)
                if near_variant:
                    matched_source = None
                    for sn in effective_source_names:
                        if entity_source_name_in_text(sn, source_text):
                            matched_source = sn
                            break
                    inconsistencies.append({
                        "entity": canonical,
                        "expected": canonical,
                        "found": near_variant,
                        "source_name": matched_source,
                        "type": "variant_instead_of_canonical",
                        "chunk_id": chunk_id,
                    })
            if not all_violations and not near_variant and not self._canonical_present_in_text(canonical, lit_text):
                # 只有原文中确实出现了该实体时才报 missing
                # 避免"全书角色本章未出场"的假阳性
                matched_source = None
                for sn in effective_source_names:
                    if entity_source_name_in_text(sn, source_text):
                        matched_source = sn
                        break
                if matched_source:
                    # Round 71: 跨人物冲突抑制 — 若 lit_text 中已经出现共享 source_name 的
                    # 其它 canonical, 说明 rewriter 用了别名译名, 不算 missing.
                    # 例: canonical="乔治·洛马克斯先生", source_name="Mr. Lomax" 在源文,
                    #      lit_text 含 "洛马克斯先生" (同 source_name 的 sibling canonical).
                    siblings = sibling_canonicals.get(matched_source) or set()
                    if siblings and any(sib != canonical and sib in lit_text for sib in siblings):
                        # sibling 命中, 抑制 missing 假阳性
                        pass
                    else:
                        inconsistencies.append({
                            "entity": canonical,
                            "expected": canonical,
                            "found": None,
                            "source_name": matched_source,
                            "type": "missing",
                            "chunk_id": chunk_id,
                        })

            self._maybe_discover_entity_variant(canonical, lit_text, variant_names)

        return inconsistencies

    def _build_entity_registry_for_rewriter(self, source_text: str) -> Dict[str, str]:
        """回路修复 helper：构建 Rewriter 可见的实体译名登记表

        Round 99 P1: per-chunk 缓存. 同一 chunk 的所有 retry 用同一份 source_text,
        registry 在 chapter 内不变. 用 chunk_id 作为缓存 key (隐式从 source_text 提取).

        只注入本章原文出现的实体（与 _check_entity_consistency 同语义），
        避免无关实体污染 prompt 同时节约 token。

        关键：把每个 entity 的所有 source_name 都加进 dict（不只是第一个），
        这样 Rewriter 能在 prompt 里看到所有英文别名 → 中文译名的映射，
        即使 source_text 使用的是 source_names[2] 也能正确翻译。
        修前只取第一个 source_name 时，遇到 "Mr. George Lomax" 在源文里写
        "George Lomax" 时 Rewriter 找不到匹配的指引。

        Returns:
            {english_source_name: chinese_canonical} dict（每个 entity 可能多 key）
        """
        if not self.decision_engine:
            return {}

        # Round 99 P1: per-chunk 缓存. 用 source_text hash 作为 key (chunk_id 不可得,
        # 因为此 helper 是无状态调用). hash 同一 source_text 一致, 缓存命中走短路.
        cache_key = hash(source_text)
        if cache_key in self._entity_registry_cache:
            return self._entity_registry_cache[cache_key]

        try:
            from src.core.decision_engine import query_reviewable_entities
            rows = query_reviewable_entities(self.decision_engine.conn)
        except Exception:
            return {}
        out: Dict[str, str] = {}
        for row in rows:
            canonical = row[0]
            source_names_json = row[1]
            source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
            # Round 111: 过滤无效 source_name（限定词+全小写普通名词短语等，如 "the stairs"/
            # "the chauffeur"/"my maid"）— 与 _check_entity_consistency 同语义，避免
            # rewriter/critic 收到"把描述性短语统一为规范主名"的不可执行指令（ch09 chunk004）。
            # Round 112: 身份映射 canonical（如 " Cloak Room"→"Cloak Room"）同步剔除。
            if not self._is_enforceable_canonical(canonical, source_names):
                continue
            valid_sns = [sn for sn in source_names if self._is_valid_entity_source_name(sn)]
            # Round 133b: 别名吞并抑制 — swallowed 别名不注入/不当在场证据
            # （chunk071 "Mother"→杰西卡夫人 污染 prompt 根因）
            effective_sns, swallowed_sns = self._partition_swallowed_source_names(
                source_text, canonical, valid_sns,
            )
            # 只取本章出现的实体
            if not any((sn and entity_source_name_in_text(sn, source_text)) for sn in effective_sns):
                continue
            # 把每个匹配的 source_name 都映射到 canonical（修前 bug: 只取第一个）
            for sn in valid_sns:
                if sn in swallowed_sns:
                    continue
                if sn and sn not in out:
                    out[sn] = canonical
        # 缓存结果 (同一 source_text 在 chapter 内稳定)
        self._entity_registry_cache[cache_key] = out
        return out

    def _build_entity_variants_for_critic(self, source_text: str) -> Dict[str, List[str]]:
        """Round 61 helper：为 critic 构造 per-entity 变体表（供 typesafe Jev atomic Noul 使用）

        Round 99 P1: per-chunk 缓存 (与 _build_entity_registry_for_rewriter 同样策略).

        与 _build_entity_registry_for_rewriter 同语义：只注入本章原文出现的实体。
        Returns:
            {english_source_name: [variant_zh_1, variant_zh_2, ...]}
        """
        if not self.decision_engine:
            return {}

        cache_key = hash(source_text)
        if cache_key in self._entity_variants_cache:
            return self._entity_variants_cache[cache_key]

        try:
            from src.core.decision_engine import query_reviewable_entities
            rows = query_reviewable_entities(self.decision_engine.conn)
        except Exception:
            return {}
        out: Dict[str, List[str]] = {}
        for row in rows:
            canonical = row[0]
            source_names_json = row[1]
            source_names = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
            variant_names_json = row[2] if len(row) > 2 else None
            variant_names = robust_json_loads(variant_names_json, expected_type=list) if variant_names_json else []
            # Round 111: 与 _build_entity_registry_for_rewriter 同语义过滤无效 source_name
            # Round 112: 身份映射 canonical 同步剔除（无翻译信息）
            if not self._is_enforceable_canonical(canonical, source_names):
                continue
            valid_sns = [sn for sn in source_names if self._is_valid_entity_source_name(sn)]
            # Round 133b: 别名吞并抑制（与 _build_entity_registry_for_rewriter 同语义）
            effective_sns, swallowed_sns = self._partition_swallowed_source_names(
                source_text, canonical, valid_sns,
            )
            if not any((sn and entity_source_name_in_text(sn, source_text)) for sn in effective_sns):
                continue
            for sn in valid_sns:
                if sn in swallowed_sns:
                    continue
                if sn and sn not in out:
                    out[sn] = list(variant_names) if variant_names else []
        # Round 99 P1: 缓存 variants 结果
        self._entity_variants_cache[cache_key] = out
        return out

    def _maybe_discover_entity_variant(self, canonical: str, lit_text: str, existing_variants: list):
        """检测润色稿中可能的新变体（与 canonical 一字之差）

        B3 Round 2 修复：旧实现"只要发现一字之差就入注册表" → 与 canonical 冲突 →
        触发 entity_hard_fail 死循环（每次都查到新"变体"）。

        修复策略：
          - 加 min_occurrence 门槛（默认 2 次）：变体必须在同一 chunk 内出现 ≥N 次才入注册表
          - 加 auto_register 配置开关（Round 105 起默认 false，仅 soft-hint，需显式开启）
          - Round 105: 硬长度护栏（canonical < variant_min_auto_len 默认 3 字绝不自动入表）
          - Round 105: 普通词护栏（src.utils.entity_stopwords）：普通词/高频差异字
            一律不作为人名异体，杜绝 东西/卡特/警察/伦敦金 等污染
          - 低于门槛的候选写入 soft_candidates 字段（不持久化，仅返回供调用方参考）

        Args:
            canonical: 已登记的主名（中文）
            lit_text: 润色稿
            existing_variants: 已登记的 variant 列表
        """
        try:
            if not canonical or not re.fullmatch(r'[一-鿿]{2,}', canonical):
                return [], []
            length = len(canonical)
            # 用 finditer 取所有匹配位置，便于统计出现次数
            all_matches = re.findall(r'(?=([一-鿿]{%d}))' % length, lit_text)
            from collections import Counter
            cand_counts = Counter(all_matches)

            # B3 配置：min_occurrence 门槛 + auto_register 开关
            min_occurrence = int(_config.pipeline.get("variant_min_occurrence", 2))
            # Round 105: 默认改为 False。一字之差自动入表误报率极高（ch32 实测把
            # 东西/卡特/警察/伦敦金 等普通词登记为人名异体，随后被 consistency 3c
            # 盲目替换改坏正文）。默认只记 soft-hint 供人工复核，需显式开启才入表。
            auto_register = bool(_config.pipeline.get("variant_auto_register", False))
            # Round 105: 硬长度护栏。短规范名（<3 字）的一字之差几乎必为普通词
            # （黛西←东西 / 卡洛←卡特），即使 auto_register 开启也绝不自动入表。
            min_auto_len = int(_config.pipeline.get("variant_min_auto_len", 3))

            # Round 105: 普通词护栏（共享 src.utils.entity_stopwords）。
            try:
                from src.utils.entity_stopwords import is_plausible_name_variant
            except Exception:  # 单文件部署兜底
                is_plausible_name_variant = None

            to_register = []
            soft_candidates = []
            for cand, count in cand_counts.items():
                if cand == canonical or cand in existing_variants:
                    continue
                if self.decision_engine.is_canonical_name(cand):
                    continue
                diff = sum(1 for a, b in zip(canonical, cand) if a != b)
                if diff != 1:
                    continue
                # Round 105: 普通词/高频差异字一律不作为人名异体（连 soft-hint 也不记，
                # 避免噪声）。真异体（差异字冷僻）才进入后续门槛判断。
                if is_plausible_name_variant is not None:
                    plausible, _reason = is_plausible_name_variant(canonical, cand)
                    if not plausible:
                        continue
                # 只考虑一字之差的近音/近形变体
                can_auto = auto_register and length >= min_auto_len
                if count >= min_occurrence and can_auto:
                    self.decision_engine.add_entity_variant(canonical, cand)
                    to_register.append((cand, count))
                else:
                    # 低于门槛 / 未开启自动入表：仅记为 soft candidate，不写入 registry
                    soft_candidates.append((cand, count))

            return to_register, soft_candidates
        except Exception:
            return [], []

    @staticmethod
    def _audit_early_stop_count(critic_report: Dict[str, Any], threshold: float) -> int:
        """Round 115: AUDIT_EARLY_STOP 只统计"结论性低分"维度。

        ch06 chunk012 根因: 3 个维度（literal_residue 3.1 / semantic_preservation 3.9 /
        entity_consistency 2.4）实际都被 Jev verdict 标为 uncertain 并列入
        skipped_dimensions（R8 起 synthesize_verdict 据此不判 REJECT），但
        AUDIT_EARLY_STOP（R99 P3）仍按原始 legacy 分数把它们计入"低于阈值" →
        首轮直接 fallback，跳过 judge —— 前提"judge 必然 REJECT"对 uncertain
        维度不成立，导致合格译文被误杀。改为只统计 verdict 未跳过、分数仍低于
        阈值的维度。
        """
        scores = (critic_report or {}).get("scores") or {}
        skipped = set(
            ((critic_report or {}).get("_typesafe_verdict") or {}).get("skipped_dimensions") or []
        )
        # legacy scores 键与 primitive 维度名不同名（semantic_preservation↔semantic_loss 等）
        _alias = {
            "voice_drift": "voice_consistency",
            "style_compliance_score": "style_compliance",
            "semantic_loss": "semantic_preservation",
        }
        skipped_legacy = {_alias.get(d, d) for d in skipped}
        return sum(
            1 for k, v in scores.items()
            if k not in skipped_legacy and isinstance(v, (int, float)) and v < threshold
        )

    def _process_auditing_batch(self, tasks):
        success_ids = []
        # Round 71: 每次审计批次开始重置跨人物冲突 sibling 缓存, 避免跨章节脏数据
        self._sibling_canonicals_cache = None
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                # Round 68: 提前到循环顶部提取 quality_retries（修复 Round 65 引入的 NameError）
                # 旧位置在 entity_hard_fail 之后定义，导致 line 1893 自加时变量未绑定
                quality_retries = task.get('quality_retries', 0) or 0
                source_text = task['text_content']
                raw_text = self._load_intermediate(chunk_id, "raw")
                lit_text = self._load_intermediate(chunk_id, "literary")
                # 回路修复：把 entity_registry 也注入 Critic/Jev
                # （修前 pipeline 只在 Rewriter 调用时构造 registry，Critic 完全拿不到，
                #  Jev 看到的 entity_registry 是空 → entity_consistency 维度永远 Noul~0.5 随机噪声）
                # 用与 Rewriter 完全相同的过滤逻辑，保证 Jev 与本地 _check_entity_consistency 看到同一份登记表。
                entity_registry_for_critic = self._build_entity_registry_for_rewriter(source_text)
                # Round 61: 把 variants 也传给 critic（per-entity atomic Noul 需要）
                entity_variants_for_critic = self._build_entity_variants_for_critic(source_text)
                # Round 65: 每次 rewriter retry 都会触发新一轮 critic_scoring，计入实际调用次数
                self._llm_call_counter["critic_scoring"] += 1 + quality_retries
                critic_report = self.critic_agent.process_chunk(
                    chunk_id, source_text, raw_text, lit_text, self.style_guide,
                    entity_registry=entity_registry_for_critic,
                    entity_variants=entity_variants_for_critic,
                )

                # Q3 修复：Critic LLM 偶发拼错评分维度名（如 voice_consistence），
                # 归一化后再入库，避免脏维度名在 Judge 阈值查表时被静默忽略。
                if isinstance(critic_report, dict) and "scores" in critic_report:
                    critic_report["scores"] = _normalize_scores(critic_report.get("scores"))

                # 实体一致性核查：检查译文中实体译名是否与 entity_registry 一致
                entity_notes = self._check_entity_consistency(chunk_id, lit_text, source_text)
                if entity_notes:
                    critic_report["entity_consistency_notes"] = entity_notes
                    print(f"  ⚠️ 实体一致性核查: {len(entity_notes)} 条不一致项")

                self._save_intermediate(chunk_id, "critic_report", critic_report)
                # Round 92: 追踪当前 retry 的 lit_text 与 critic score, fallback 时用最佳版
                self._update_best_lit(chunk_id, lit_text, critic_report)
                # Round 99 P3: audit-phase early stop. 旧实现仅在 _process_judging_batch 检查
                # early_stop (judge REJECT 后), 但若首轮 critic 评分已极差, 跑 judge LLM 是浪费
                # (judge 必然 REJECT, rewriter 也不可能在 1 retry 内修复极差译文).
                # 把 early_stop 提前到 audit 阶段: quality_retries == 0 时,
                # 若 very_low_count >= max_low_dims (即多数维度评分 < threshold),
                # 直接 fallback, 跳过 judge + 后续 rewriter cycle.
                early_stop_cfg_p99 = _config.pipeline.get("early_stop", {}) or {}
                early_stop_enabled_p99 = bool(early_stop_cfg_p99.get("enabled", True))
                # Round 106: mock_heuristic 的分数不可信（Jev 不可用时的兜底假分数），
                # 不触发 AUDIT_EARLY_STOP，让 LLM judge 独立裁决。
                _mock_untrusted_p99 = (
                    not _config.pipeline.get("trust_mock_heuristic", False)
                    and critic_report.get("_backend_path") == "mock_heuristic"
                )
                if (early_stop_enabled_p99
                        and quality_retries == 0
                        and not _mock_untrusted_p99
                        and critic_report.get("scores")):
                    es_threshold_p99 = float(early_stop_cfg_p99.get("low_score_threshold", 3.0))
                    es_max_low_p99 = int(early_stop_cfg_p99.get("max_low_dims", 3))
                    # Round 115: 只统计结论性低分 — verdict 明确跳过（uncertain/low_conf）
                    # 的维度不计入，否则与 R8 verdict 合成语义冲突（ch06 chunk012 根因）。
                    very_low_count_p99 = self._audit_early_stop_count(critic_report, es_threshold_p99)
                    if very_low_count_p99 >= es_max_low_p99:
                        stop_reason_p99 = (
                            f"AUDIT_EARLY_STOP: {very_low_count_p99}项低于{es_threshold_p99}分"
                            f"（{critic_report['scores']}），跳过 judge + rewriter 直接 fallback"
                        )
                        print(f"⏭️ [Pipeline] {chunk_id} {stop_reason_p99}")
                        best_text_p99 = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text_p99,
                            "metadata": {
                                "fallback": True,
                                "reason": "AUDIT_EARLY_STOP",
                                "early_stop_detail": stop_reason_p99,
                                "scores": critic_report["scores"],
                                "quality_retries": quality_retries,
                            }
                        }, source_step="literary")
                        self.scheduler.update_task_state(
                            chunk_id, TaskState.PERMANENTLY_FAILED,
                            error_msg=stop_reason_p99
                        )
                        continue
                if not critic_report.get("scores"):
                    print(f"⚠️ [Pipeline] {chunk_id} Critic 评分为空（JSON 解析失败），转入 FAILED 等待重试")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.FAILED,
                        error_msg="[AUDITING] Critic 返回空 scores"
                    )
                    continue

                # 实体硬失败标志 + 降级阈值 + serious 列表 (Round 101 P1 提前到这里, 高分直通检查需要用)
                entity_hard_fail = bool(_config.pipeline.get("entity_hard_fail", True))
                degradation_threshold = int(_config.pipeline.get("entity_hard_fail_degradation_threshold", 3))
                serious = [
                    n for n in (entity_notes or [])
                    if n.get("type") in ("variant_instead_of_canonical", "missing")
                ]
                # Round 101 P1: audit 高分直通 — 所有维度评分 ≥ high_threshold + 无实体违规 + CJK 高,
                # 直接 COMPLETED, 跳过 judge LLM 调用. 与 AUDIT_EARLY_STOP (R99 P3) 互补:
                #   R99 P3: 评分低 → PERMANENTLY_FAILED (跳 judge + rewriter)
                #   R101 P1: 评分高 → COMPLETED (跳 judge)
                # 触发条件: quality_retries == 0 (首轮, 避免误判 retry 中可能恶化的 chunk)
                #           AND all_scores >= high_threshold
                #           AND serious == 0 (无严重实体违规)
                #           AND lit_cjk >= final_cjk_threshold (译文是中文)
                high_pass_cfg = _config.pipeline.get("audit_high_pass", {}) or {}
                high_pass_enabled = bool(high_pass_cfg.get("enabled", True))
                # Round 68: quality_retries 已提升到循环顶部, 此处不再重复定义.
                # Round 106: mock_heuristic 的分数不可信，不触发高分直通，让 LLM judge 裁决。
                _mock_untrusted_hp = (
                    not _config.pipeline.get("trust_mock_heuristic", False)
                    and critic_report.get("_backend_path") == "mock_heuristic"
                )
                if (high_pass_enabled and quality_retries == 0
                        and not _mock_untrusted_hp
                        and not (entity_hard_fail and serious)):
                    high_threshold = float(high_pass_cfg.get("high_score_threshold", 8.0))
                    scores_dict = critic_report.get("scores", {}) or {}
                    all_high = (
                        scores_dict
                        and all(
                            isinstance(v, (int, float)) and v >= high_threshold
                            for v in scores_dict.values()
                        )
                    )
                    lit_cjk_r101 = compute_cjk_ratio(lit_text)
                    final_threshold = float(_config.pipeline.get("final_cjk_threshold", 0.5))
                    cjk_ok = (final_threshold <= 0) or (lit_cjk_r101 >= final_threshold)
                    if all_high and cjk_ok:
                        # 直接 COMPLETED, 跳过 judge LLM
                        print(
                            f"✅ [Pipeline] {chunk_id} audit 高分直通 (all_scores≥{high_threshold}, "
                            f"cjk={lit_cjk_r101:.2f}), 跳过 judge 直接 COMPLETED"
                        )
                        # 保存 final (用 literary 版, 通过 sanity gate)
                        final_payload = {
                            "text": lit_text,
                            "metadata": {
                                "fallback": False,
                                "reason": "AUDIT_HIGH_PASS",
                                "scores": scores_dict,
                                "lit_cjk_ratio": round(lit_cjk_r101, 3),
                                "quality_retries": quality_retries,
                                "audit_short_circuit": True,
                            }
                        }
                        self._save_final_with_sanity(chunk_id, final_payload, source_step="literary")
                        self.scheduler.update_task_state(
                            chunk_id, TaskState.COMPLETED,
                            error_msg=f"[AUDIT_HIGH_PASS] all_scores≥{high_threshold}, cjk={lit_cjk_r101:.2f}, judge skipped"
                        )
                        continue

                # 实体硬失败：严重不一致时强制一轮质量退回（notes-only 不够）
                # P0 修复：增加降级机制，避免死锁
                # 注: entity_hard_fail / degradation_threshold / serious 已在 Round 101 P1 高分直通检查处定义

                # Round 98 P1: 实体违规连续 N 次早停
                # ch32 实测: 每次 retry 仍有 1-2 个 serious 违规 (variant_instead_of_canonical),
                # rewriter 反复修复但失败. 在 entity_stagnation_threshold 次 retry 后 fallback,
                # 节省 rewriter+critic+judge LLM 调用.
                # 触发条件: 最近 N 次 audit 每次都有 ≥1 个 serious 违规 (不管走 entity_hard_fail
                #           还是 degradation pass 路径), 且 quality_retries >= entity_stagnation_threshold.
                # 行为: fallback 到 _get_best_lit (可能仍有违规, 但已是 rewriter 能拿到的最高分);
                #       标 PERMANENTLY_FAILED + error_msg 含 [ENTITY_VIOLATION_STREAK].
                entity_stagnation_threshold = int(_config.pipeline.get("entity_stagnation_threshold", 2))
                entity_hist = self._entity_violation_history.setdefault(chunk_id, [])
                entity_hist.append(len(serious))
                if len(entity_hist) > 5:
                    entity_hist.pop(0)  # 仅保留最近 5 次, 避免内存增长
                recent_serious = entity_hist[-entity_stagnation_threshold:]
                if (entity_stagnation_threshold > 0
                        and len(recent_serious) == entity_stagnation_threshold
                        and all(c > 0 for c in recent_serious)
                        and entity_hard_fail
                        and quality_retries >= entity_stagnation_threshold):
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text)
                    self._save_final_with_sanity(chunk_id, {
                        "text": best_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "ENTITY_VIOLATION_STREAK",
                            "serious_history": recent_serious,
                            "entity_stagnation_threshold": entity_stagnation_threshold,
                            "judge_reason": f"实体违规连续 {entity_stagnation_threshold} 次未修复 (counts={recent_serious})",
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=(
                            f"[ENTITY_VIOLATION_STREAK] 实体违规连续 {entity_stagnation_threshold} 次未修复 "
                            f"(counts={recent_serious}), 提前 fallback"
                        )
                    )
                    print(
                        f"⏹️ [Pipeline] {chunk_id} 实体违规连续 {entity_stagnation_threshold} 次未修复 "
                        f"(counts={recent_serious}), 提前 fallback"
                    )
                    continue

                # 降级逻辑：质量重试 > 0 时，仅在严重不一致项 > 阈值时才强制退回
                # 避免因少量实体问题导致死锁
                if entity_hard_fail and serious and quality_retries == 0:
                    # 回路修复：把 entity_block 也写入 feedback（之前只放在 specific_edits，缺失形式化清单）
                    entity_lines = []
                    for s_e in serious[:30]:
                        ent = s_e.get("entity", "")
                        exp = s_e.get("expected", "")
                        found = s_e.get("found", "")
                        ntype = s_e.get("type", "")
                        # Round 20: 把 source_name 一并展示（rewriter 可定位源文位置）
                        src_name = s_e.get("source_name", "")
                        src_hint = f"，源文形式：「{src_name}」" if src_name else ""
                        if exp and found and found != exp:
                            entity_lines.append(f"- 「{found}」应统一为「{exp}」（{ntype}{src_hint}）")
                        elif exp:
                            entity_lines.append(f"- 实体「{ent}」须使用主名「{exp}」（{ntype}{src_hint}）")
                    entity_block = ("实体一致性要求（必须遵循）：\n" + "\n".join(entity_lines)) if entity_lines else ""

                    feedback = {
                        "judge_reason": f"实体一致性硬失败: {len(serious)} 项译名不一致，须按 entity_registry 修正",
                        "critic_suggestions": json.dumps(serious, ensure_ascii=False)[:500],
                        "entity_consistency_notes": entity_block,  # 回路修复：补这个字段
                        "low_dims": ["entity_consistency"],
                        "scores": critic_report.get("scores", {}),
                        "specific_edits": [
                            {
                                "location": s.get("entity", "?"),
                                "original": s.get("found") or "",
                                "issue": s.get("type"),
                                "suggested_direction": f"统一为 {s.get('expected')}" + (f"（源文:{s.get('source_name', '')}）" if s.get("source_name") else ""),
                            }
                            for s in serious[:10]
                        ],
                        # Round 64: 把 _typesafe_verdict 透传给 rewriter，让它知道哪些维度被 Jev uncertain 跳过
                        "_typesafe_verdict": critic_report.get("_typesafe_verdict", {}) or {},
                    }
                    self.scheduler.update_task_state(
                        chunk_id,
                        TaskState.REWRITING_LITERARY,
                        error_msg=json.dumps(feedback, ensure_ascii=False),
                        quality_retry=True,
                    )
                    print(f"🔁 [Pipeline] {chunk_id} 实体硬失败，退回文学润色")
                    continue
                elif entity_hard_fail and serious and quality_retries > 0 and len(serious) > degradation_threshold:
                    # 降级逻辑：质量重试 > 0 且严重不一致项 > 阈值时，仍强制退回
                    # B1 修复：先检查是否达到 max_quality_retries_entity 上限，
                    #         达上限则降级放行，避免死循环（实测 ch32 quality_retries=112+）
                    max_qr_entity = int(_config.pipeline.get("max_quality_retries_entity", 4))
                    if quality_retries >= max_qr_entity:
                        print(f"🟡 [Pipeline] {chunk_id} 实体一致性达上限放行 (qr={quality_retries} >= max_qr_entity={max_qr_entity}, serious={len(serious)})")
                        success_ids.append(chunk_id)
                        continue
                    # 回路修复：同样构造 entity_block
                    entity_lines = []
                    for s_e in serious[:30]:
                        ent = s_e.get("entity", "")
                        exp = s_e.get("expected", "")
                        found = s_e.get("found", "")
                        ntype = s_e.get("type", "")
                        if exp and found and found != exp:
                            entity_lines.append(f"- 「{found}」应统一为「{exp}」（{ntype}）")
                        elif exp:
                            entity_lines.append(f"- 实体「{ent}」须使用主名「{exp}」（{ntype}）")
                    entity_block = ("实体一致性要求（必须遵循）：\n" + "\n".join(entity_lines)) if entity_lines else ""

                    feedback = {
                        "judge_reason": f"实体一致性硬失败(降级): {len(serious)} 项译名不一致，须按 entity_registry 修正",
                        "critic_suggestions": json.dumps(serious, ensure_ascii=False)[:500],
                        "entity_consistency_notes": entity_block,  # 回路修复
                        "low_dims": ["entity_consistency"],
                        "scores": critic_report.get("scores", {}),
                        "specific_edits": [
                            {
                                "location": s.get("entity", "?"),
                                "original": s.get("found") or "",
                                "issue": s.get("type"),
                                "suggested_direction": f"统一为 {s.get('expected')}",
                            }
                            for s in serious[:10]
                        ],
                        # Round 64: 把 _typesafe_verdict 透传给 rewriter
                        "_typesafe_verdict": critic_report.get("_typesafe_verdict", {}) or {},
                    }
                    self.scheduler.update_task_state(
                        chunk_id,
                        TaskState.REWRITING_LITERARY,
                        error_msg=json.dumps(feedback, ensure_ascii=False),
                        quality_retry=True,
                    )
                    print(f"🔁 [Pipeline] {chunk_id} 实体硬失败(降级)，退回文学润色 (quality_retries={quality_retries})")
                    continue
                elif entity_hard_fail and serious and quality_retries > 0 and len(serious) <= degradation_threshold:
                    # 降级逻辑：质量重试 > 0 且严重不一致项 ≤ 阈值时，允许继续进入裁决阶段
                    print(f"🟡 [Pipeline] {chunk_id} 实体一致性降级放行 (quality_retries={quality_retries}, serious={len(serious)})")
                    # 不 continue，继续进入 success_ids

                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 审计异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[AUDITING] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.JUDGING)

    def _fallback_to_rewriter(self, chunk_id, lit_text, critic_report, judge_result):
        scores = critic_report.get("scores", {})
        thresholds = _config.critic_thresholds_for_book()
        low_dims = [k for k, v in scores.items()
                    if isinstance(v, (int, float)) and v < thresholds.get(k, 7.0)]
        specific_edits = critic_report.get("specific_edits", [])[:20]
        # Round 118 P1-2: typesafe 路径下 critic_report.specific_edits 恒为空，
        # 可执行编辑由 judge 的 REJECT 第二意见产出（unresolved_edits）并入。
        if not specific_edits and judge_result.get("unresolved_edits"):
            specific_edits = judge_result.get("unresolved_edits", [])[:20]
        if specific_edits:
            for e in specific_edits:
                e["suggested_direction"] = (e.get("suggested_direction", "") or "")[:200]
        # 实体一致性：将 registry 主名反馈给 Rewriter，避免重试仍用错译名（修复 ch02 实体死锁）
        entity_notes = critic_report.get("entity_consistency_notes", []) or []
        entity_block = ""
        if entity_notes:
            lines = []
            for n in entity_notes[:30]:
                ent = n.get("entity", "")
                exp = n.get("expected", "")
                found = n.get("found", "")
                ntype = n.get("type", "")
                if exp and found and found != exp:
                    lines.append(f"- 「{found}」应统一为「{exp}」（{ntype}）")
                elif exp:
                    lines.append(f"- 实体「{ent}」须使用主名「{exp}」（{ntype}）")
            if lines:
                entity_block = "实体一致性要求（必须遵循）：\n" + "\n".join(lines)
        enriched_feedback = json.dumps({
            "judge_reason": judge_result.get("reject_reason", ""),
            "critic_suggestions": critic_report.get("improvement_suggestions", ""),
            "entity_consistency_notes": entity_block,
            "low_dims": low_dims,
            "scores": scores,
            "specific_edits": specific_edits,
            # Round 64: 把 _typesafe_verdict 透传给 rewriter（让 rewriter 感知 uncertain 跳过维度）
            "_typesafe_verdict": critic_report.get("_typesafe_verdict", {}) or {},
            # Round 118 P1-2: 第二意见审计元数据（confirmed_reject 仅参考，不翻案）
            "_second_opinion": judge_result.get("_second_opinion", {}) or {},
        }, ensure_ascii=False)
        self.scheduler.update_task_state(
            chunk_id, TaskState.REWRITING_LITERARY,
            error_msg=enriched_feedback, quality_retry=True
        )
        print(f"🔁 [Pipeline] {chunk_id} 裁决未通过，退回文学润色")

    def _process_judging_batch(self, tasks):
        max_quality_retries = _config.pipeline.get("max_quality_retries", 6)
        early_stop_cfg = _config.pipeline.get("early_stop", {})
        early_stop_enabled = early_stop_cfg.get("enabled", True)
        early_stop_max_low = early_stop_cfg.get("max_low_dims", 3)
        early_stop_threshold = early_stop_cfg.get("low_score_threshold", 3.0)
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                quality_retries = task.get('quality_retries', 0)
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self._load_intermediate(chunk_id, "critic_report")

                # Round 123: mock_heuristic 是 Jev/critic LLM 不可用时的兜底假分数,
                # 不得触发 judging 阶段 EARLY_STOP — 与 audit 阶段 R106 护栏
                # (_mock_untrusted_p99) 同语义。根因: ch09 chunk013/044 critic
                # 连续 429 回退 mock 后, 假分数 readability=2.5/fluency=2.5/
                # literal_residue=0.0 (3 维<4.0) 被 EARLY_STOP 秒杀成
                # PERMANENTLY_FAILED, 而同一份 mock 分数在 audit 阶段是放行的。
                # 未显式 trust mock 时跳过 EARLY_STOP, 交由 rewriter 正常轮次。
                mock_untrusted_judging = (
                    isinstance(critic_report, dict)
                    and not _config.pipeline.get("trust_mock_heuristic", False)
                    and critic_report.get("_backend_path") == "mock_heuristic"
                )

                # 加载 Rewriter 自评（如果有）
                self_review = None
                if quality_retries > 0:
                    try:
                        meta = self._load_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}")
                        if isinstance(meta, dict):
                            self_review = meta.get("self_review")
                    except Exception:
                        pass

                # Round 65: 每次 rewriter retry 都会触发新一轮 judge_decision，计入实际调用次数
                self._llm_call_counter["judge_decision"] += 1 + quality_retries
                judge_result = self.judge_agent.process_chunk(
                    chunk_id, source_text, lit_text, critic_report,
                    affected_chunks=[chunk_id],
                    rewriter_self_review=self_review,
                    verse_note=self._verse_note_for(chunk_id, source_text),  # Round 119 L1
                )

                # Round 93: judge REJECT 后检查是否"连续下降" — 若连续 3 次 retry 分数递减,
                # 跳过 rewriter fallback, 直接用 best_lit 写入 final. 节省 ~2 次重试 LLM 调用
                # (rewriter + critic + judge 共 3 LLM 调用每次).
                current_avg = self._compute_avg_score(critic_report)
                # Round 108: streak 早停只适用于 REJECT 路径 — judge 已 PASS 时直接定稿。
                # 修前无条件执行: 旧分数历史（如回溯重跑前）会把通过轮误杀
                # (ch08 chunk006: 全部维度通过, 却被 NO_PROGRESS_STREAK 打成 PERMANENTLY_FAILED)。
                judge_is_pass = judge_result.get("decision") == "PASS"
                decline_streak = int(_config.pipeline.get("score_decline_streak", 4))
                if not judge_is_pass and self._is_declining_streak(chunk_id, current_avg, consecutive=decline_streak):
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    self._save_final_with_sanity(chunk_id, {
                        "text": best_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "SCORE_DECLINE_STREAK",
                            "judge_reason": judge_result.get("reject_reason", ""),
                            "quality_retries": quality_retries,
                            "score_history": list(self._score_history.get(chunk_id, [])[-5:]),
                            "current_avg": round(current_avg, 2),
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=f"[SCORE_DECLINE] 连续 {decline_streak} 次 retry 分数递减, 提前 fallback (history={self._score_history.get(chunk_id, [])[-5:]})"
                    )
                    print(f"⏹️ [Pipeline] {chunk_id} 连续 {decline_streak} 次 retry 分数递减, 提前 fallback (current={current_avg:.2f}, history={self._score_history.get(chunk_id, [])[-5:]})")
                    continue
                # Round 98 P2: 非进展 streak — 末尾 N 个 score 与当前 score 单调不增
                # (含持平) 时提前 fallback. 比 _is_declining_streak (严格递减) 更激进,
                # 抓住「rewriter 不恶化但也不进步」场景, 节省 rewriter+critic LLM 调用.
                # 与 SCORE_DECLINE 互补: SCORE_DECLINE 要求 s[i] > s[i+1], NO_PROGRESS 允许 s[i] == s[i+1].
                no_progress_window = int(_config.pipeline.get("no_progress_window", 3))
                if (not judge_is_pass and no_progress_window > 1
                        and self._is_no_progress_streak(chunk_id, current_avg, window=no_progress_window)):
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    self._save_final_with_sanity(chunk_id, {
                        "text": best_text,
                        "metadata": {
                            "fallback": True,
                            "reason": "NO_PROGRESS_STREAK",
                            "judge_reason": judge_result.get("reject_reason", ""),
                            "quality_retries": quality_retries,
                            "score_history": list(self._score_history.get(chunk_id, [])[-5:]),
                            "current_avg": round(current_avg, 2),
                            "no_progress_window": no_progress_window,
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=(
                            f"[NO_PROGRESS_STREAK] 连续 {no_progress_window} 次 retry 单调不增 (含持平), "
                            f"提前 fallback (history={self._score_history.get(chunk_id, [])[-5:]})"
                        )
                    )
                    print(
                        f"⏹️ [Pipeline] {chunk_id} 连续 {no_progress_window} 次 retry 单调不增, "
                        f"提前 fallback (current={current_avg:.2f}, history={self._score_history.get(chunk_id, [])[-5:]})"
                    )
                    continue
                # Round 97: CJK 重试耗尽早停 — rewriter 反复输出英文, 不浪费 LLM 调用
                # 触发条件: REJECT 仅因 CJK 比例过低 (Round 96 P3 闸门), 且已 retry >= cjk_max_retries 次.
                # 行为: fallback 到 raw 翻译稿 (raw 经 P2 闸门保证是中文). 比 max_quality_retries (4) 早停,
                #       节省 2-4 次 rewriter+critic LLM 调用.
                reject_reason_raw = judge_result.get("reject_reason", "") or ""
                cjk_max_retries = int(_config.pipeline.get("cjk_max_retries", 2))
                cjk_only_reject = (
                    "CJK 比例过低" in reject_reason_raw
                    and cjk_max_retries > 0
                    and quality_retries >= cjk_max_retries
                )
                if cjk_only_reject:
                    # 加载 raw 翻译稿作为 fallback (raw 经 P2 闸门保证是中文, 即使文学性差)
                    raw_fallback = ""
                    try:
                        raw_data = self._load_intermediate(chunk_id, "raw")
                        raw_fallback = raw_data if isinstance(raw_data, str) else (
                            raw_data.get("text", "") if isinstance(raw_data, dict) else ""
                        )
                    except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError):
                        pass
                    if not raw_fallback or not raw_fallback.strip():
                        # raw 也不可用 → 用 lit (最佳一版, 即便英文)
                        raw_fallback = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    raw_cjk = compute_cjk_ratio(raw_fallback)
                    self._save_final_with_sanity(chunk_id, {
                        "text": raw_fallback,
                        "metadata": {
                            "fallback": True,
                            "reason": "CJK_RETRY_EXHAUSTED",
                            "judge_reason": reject_reason_raw[:200],
                            "quality_retries": quality_retries,
                            "cjk_max_retries": cjk_max_retries,
                            "raw_cjk_ratio": round(raw_cjk, 3),
                            "lit_cjk_ratio": round(compute_cjk_ratio(lit_text), 3),
                        }
                    }, source_step="literary")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.PERMANENTLY_FAILED,
                        error_msg=(
                            f"[CJK_RETRY_EXHAUSTED] rewriter 重试 {quality_retries} 次仍输出英文 "
                            f"(lit cjk={compute_cjk_ratio(lit_text):.2f}), 提前 fallback 到 raw "
                            f"(raw cjk={raw_cjk:.2f}), 节省 {max_quality_retries - quality_retries} 次 LLM 调用"
                        )
                    )
                    print(
                        f"⏹️ [Pipeline] {chunk_id} CJK 重试耗尽 (lit cjk={compute_cjk_ratio(lit_text):.2f}, "
                        f"raw cjk={raw_cjk:.2f}, retries={quality_retries}/{max_quality_retries}), "
                        f"fallback 到 raw, 节省 {max_quality_retries - quality_retries} 次 LLM 调用"
                    )
                    continue
                if judge_result.get("decision") == "PASS":
                    # Round 119 L1: 韵文对齐闸门 — PASS 前先 lint；未达标触发一次定向重写。
                    verse_gate = self._verse_lint_gate(chunk_id, source_text, lit_text)
                    if (verse_gate is not None and not verse_gate["ok"]
                            and not verse_gate.get("retried")
                            and verse_gate.get("retry_on_fail", True)):
                        self._mark_verse_retry(chunk_id, verse_gate)
                        judge_result["reject_reason"] = (
                            "【韵文对齐未达标】" + (verse_gate.get("reason") or "")
                            + "；请严格按韵文协议重排：逐行对应、保留复沓（refrain）、"
                            + "不得把韵文并入散文段落。")
                        self._fallback_to_rewriter(chunk_id, lit_text, critic_report, judge_result)
                        continue
                    # Round 92: judge PASS 时也用最佳版本. 即使当前 retry 通过 judge,
                    # 之前的 retry 可能有更高 critic score (未达 judge pass 阈值但更高分).
                    best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                    if verse_gate is not None and not verse_gate["ok"]:
                        # L1: 一次定向重写后仍不达标 — 放行但落档（gate ⑥ 另行核验）
                        base_text = (best_text.get("text", "") if isinstance(best_text, dict)
                                     else str(best_text or ""))
                        best_text = {"text": base_text, "metadata": {"verse_lint": {
                            "ok": False,
                            "reason": verse_gate.get("reason", ""),
                            "expected_lines": verse_gate.get("expected_lines"),
                            "found_lines": verse_gate.get("found_lines"),
                        }}}
                        print(f"⚠️ [Pipeline] {chunk_id} verse_lint 未达标（已用过一次定向重写）: {verse_gate.get('reason', '')}")
                    self._save_final_with_sanity(chunk_id, best_text)
                    self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED)
                    self._append_chunk_memory(chunk_id, best_text if isinstance(best_text, str) else best_text.get("text", str(best_text)) if isinstance(best_text, dict) else str(best_text), source_text)
                    print(f"✅ [Pipeline] {chunk_id} 定稿完成。")
                elif quality_retries >= max_quality_retries:
                    # O 方案：安全网放行——若各维度已接近阈值（仅差 1-2 分），
                    # 不再整章 PERMANENTLY_FAILED 丢失译文，改为带告警定稿。
                    scores = critic_report.get("scores", {}) if isinstance(critic_report, dict) else {}
                    thresholds = _config.critic_thresholds_for_book()
                    numeric = [v for v in scores.values() if isinstance(v, (int, float))]
                    avg_score = (sum(numeric) / len(numeric)) if numeric else 0.0
                    safety_net_min = float(_config.pipeline.get("judge_safety_net_avg_min", 5.5))
                    # 是否有维度低于阈值超过 1 分（即严重偏离）
                    severe_count = sum(
                        1 for k, v in scores.items()
                        if isinstance(v, (int, float))
                        and v < thresholds.get(k, 7.0) - 1.0
                    )
                    if avg_score >= safety_net_min and severe_count == 0:
                        # Round 92: 安全网放行时也用最佳版本, 而非当前被拒的 lit_text
                        best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "SAFETY_NET_PASSED",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                                "avg_score": round(avg_score, 2),
                            }
                        }, source_step="literary")
                        self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED, error_msg=f"[SAFETY_NET] 平均分{avg_score:.2f}≥{safety_net_min}，接近阈值放行")
                        print(f"🟡 [Pipeline] {chunk_id} 安全网放行（平均分{avg_score:.2f}，未达PERMANENTLY_FAILED）")
                    else:
                        # Round 92: max retries fallback 也用最佳版本
                        best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "PERMANENTLY_FAILED_AFTER_MAX_RETRIES",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        }, source_step="literary")
                        self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, error_msg=judge_result.get("reject_reason") or "质量重试耗尽")
                        print(f"⚠️ [Pipeline] {chunk_id} 质量重试耗尽，已 fallback 写入 final（最佳一版润色稿）")
                elif early_stop_enabled and quality_retries == 0:
                    # early_stop 仅在首轮裁决生效：避免 rework 一两次就放弃可恢复的 chunk
                    # 第二轮起若仍不达标，正常走 fallback 完成 max_quality_retries 轮 rework
                    # 详见 config.yaml pipeline.early_stop.apply_only_first_round
                    scores = critic_report.get("scores", {})
                    very_low_count = sum(
                        1 for v in scores.values()
                        if isinstance(v, (int, float)) and v < early_stop_threshold
                    )
                    if very_low_count >= early_stop_max_low and mock_untrusted_judging:
                        # Round 123: mock 假分数不可信 — 不早停, 落回 rewriter 正常轮次
                        # (与 audit 阶段 R106 护栏同语义, 见上方 mock_untrusted_judging)
                        print(f"🛡️ [Pipeline] {chunk_id} EARLY_STOP 拦截: critic 后端 mock_heuristic（假分数不可信），转 rewriter 正常重试")
                    elif very_low_count >= early_stop_max_low:
                        stop_reason = (
                            f"EARLY_STOP: {very_low_count}项低于{early_stop_threshold}分"
                            f"（{scores}），放弃重试"
                        )
                        print(f"⏭️ [Pipeline] {chunk_id} {stop_reason}")
                        # Round 92: early stop 也用最佳版本 (虽然首轮平均分低, 但其它维度可能还行)
                        best_text = self._pairwise_pick_best(chunk_id, lit_text, source_text, raw_text=None)
                        self._save_final_with_sanity(chunk_id, {
                            "text": best_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "EARLY_STOP",
                                "early_stop_detail": stop_reason,
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        }, source_step="literary")
                        self.scheduler.update_task_state(
                            chunk_id, TaskState.PERMANENTLY_FAILED,
                            error_msg=stop_reason
                        )
                        continue
                    self._fallback_to_rewriter(chunk_id, lit_text, critic_report, judge_result)
                else:
                    self._fallback_to_rewriter(chunk_id, lit_text, critic_report, judge_result)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 裁决异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[JUDGING] {e}")


def consolidate_dirty_chapters(
    project_root=None,
    max_rounds: int = 3,
    pipeline_factory=None,
    db_path=None,
    output_dir=None,
) -> dict:
    """T2: 全书决策收敛闭环 — 重跑所有含 DIRTY chunk 的章节直到收敛。

    D2/D3 根因: 章节串行执行时，已结束章节上的回溯标记（DIRTY）没有
    消费点，全书跑完后 merge 会把"回溯前"陈旧 final 收录进成书。
    本函数在全书写完之后统一收口: 扫描 workflow.db 的 DIRTY chunk →
    按章序重跑该章管线（DIRTY 阶段在阶段表首位）→ 重扫；循环到
    DIRTY=0 或 max_rounds。每个重跑 chunk 的旧/新 final 文本差异写入
    output/consolidation_diff.md（旧稿在重跑前采样，不依赖 backup 存活）。

    Returns:
        {
          "converged": bool,     # 最终 DIRTY=0
          "rounds": int,         # 实际执行轮数（无 DIRTY 时为 0）
          "rerun": [...],        # 重跑过的 chunk（按发生顺序）
          "remaining": {...},    # 未收敛残留 {chapter: [chunk_id]}
          "diff_path": str|None,
        }
    """
    import difflib
    from datetime import datetime

    rounds_cap = max(1, int(max_rounds))
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    db_file = Path(db_path) if db_path else root / "db" / "workflow.db"
    out_root = Path(output_dir) if output_dir else root / "output"

    result = {"converged": True, "rounds": 0, "rerun": [], "remaining": {}, "diff_path": None}
    if not db_file.exists():
        print(f"⚠️ [Consolidate] workflow.db 不存在（{db_file}），跳过收敛检查")
        return result

    if pipeline_factory is None:
        pipeline_factory = lambda ch: TranslationPipeline(chapter_id=ch)

    scheduler = TaskScheduler(db_path=str(db_file))

    def _dirty_rows():
        cursor = scheduler.conn.execute(
            "SELECT chapter_id, chunk_id FROM chunk_tasks WHERE state = ? "
            "ORDER BY chapter_id ASC, chunk_id ASC",
            (TaskState.DIRTY.value,),
        )
        return [tuple(row) for row in cursor.fetchall()]

    def _read_final_text(chapter_id, chunk_id):
        path = out_root / chapter_id / f"{chunk_id}_final.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data.get("text", "") if isinstance(data, dict) else str(data)

    diff_records = []
    while result["rounds"] < rounds_cap:
        rows = _dirty_rows()
        if not rows:
            break
        result["rounds"] += 1
        round_no = result["rounds"]
        by_chapter = {}
        for chapter_id, chunk_id in rows:
            by_chapter.setdefault(chapter_id, []).append(chunk_id)
        print(f"🔁 [Consolidate] 第 {round_no}/{rounds_cap} 轮: "
              f"{len(rows)} 个 DIRTY chunk, {len(by_chapter)} 章待收敛")
        for chapter_id in sorted(by_chapter):
            chunk_ids = by_chapter[chapter_id]
            pre_texts = {cid: _read_final_text(chapter_id, cid) for cid in chunk_ids}
            try:
                ok = bool(pipeline_factory(chapter_id).run())
            except Exception as exc:
                ok = False
                print(f"  ⚠️ [Consolidate] 章节 {chapter_id} 重跑异常: {exc}")
            if not ok:
                print(f"  ⚠️ [Consolidate] 章节 {chapter_id} 重跑未完全通过（fallback 已兜底的可能）")
            for cid in chunk_ids:
                diff_records.append({
                    "round": round_no,
                    "chapter": chapter_id,
                    "chunk": cid,
                    "old": pre_texts.get(cid),
                    "new": _read_final_text(chapter_id, cid),
                })
                result["rerun"].append(cid)

    remaining_rows = _dirty_rows()
    result["converged"] = len(remaining_rows) == 0
    for chapter_id, chunk_id in remaining_rows:
        result["remaining"].setdefault(chapter_id, []).append(chunk_id)

    lines = [
        "# 全书决策收敛报告 (consolidation_diff)",
        "",
        f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"- 轮次: {result['rounds']} / max_rounds={rounds_cap}",
        f"- 重跑 chunk: {len(result['rerun'])} 个",
        f"- 最终状态: {'✅ 收敛 (DIRTY=0)' if result['converged'] else '❌ 未收敛'}",
        "",
    ]
    if result["remaining"]:
        lines.append("## 未收敛残留")
        lines.append("")
        for chapter_id in sorted(result["remaining"]):
            lines.append(f"- {chapter_id}: {', '.join(result['remaining'][chapter_id])}")
        lines.append("")
    if not diff_records:
        lines.append("（本次扫描 DIRTY=0，无重跑）")
        lines.append("")
    else:
        lines.append("## 重跑明细")
        lines.append("")
        for rec in diff_records:
            lines.append(f"### [{rec['chapter']}/{rec['chunk']}] (轮 {rec['round']})")
            old_text, new_text = rec["old"], rec["new"]
            if old_text is None and new_text is None:
                lines.append("- 旧/新 final 均缺失（重跑失败且无 backup 可恢复）")
            elif old_text is None:
                lines.append(f"- 无旧稿对照；新稿 {len(new_text)} 字")
            elif new_text is None:
                lines.append(f"- 重跑无新稿（已回滚旧稿）；旧稿 {len(old_text)} 字")
            elif old_text == new_text:
                lines.append(f"- 文本无变化（仅决策核销）；{len(new_text)} 字")
            else:
                lines.append(f"- 文本变化: {len(old_text)} 字 → {len(new_text)} 字 "
                             f"(Δ{len(new_text) - len(old_text):+d})")
                old_lines = old_text.splitlines()
                new_lines = new_text.splitlines()
                preview = []
                for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes():
                    if tag == "equal":
                        continue
                    for ln in old_lines[i1:i2][:3]:
                        preview.append("- " + ln[:80])
                    for ln in new_lines[j1:j2][:3]:
                        preview.append("+ " + ln[:80])
                    if len(preview) >= 8:
                        break
                lines.append("- 差异预览:")
                lines.append("")
                lines.append("```diff")
                lines.extend(preview[:8])
                lines.append("```")
            lines.append("")
    lines.append("---")
    lines.append("*由 consolidate 子命令自动生成*")
    lines.append("")

    diff_path = out_root / "consolidation_diff.md"
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        diff_path.write_text("\n".join(lines), encoding="utf-8")
        result["diff_path"] = str(diff_path)
    except OSError as exc:
        print(f"⚠️ [Consolidate] 报告写入失败: {exc}")

    if result["converged"]:
        print(f"✅ [Consolidate] 收敛完成: DIRTY=0 "
              f"(重跑 {len(result['rerun'])} chunk, {result['rounds']} 轮)")
    else:
        total_left = sum(len(v) for v in result["remaining"].values())
        print(f"❌ [Consolidate] 未收敛: 仍余 {total_left} 个 DIRTY chunk"
              f"（max_rounds={rounds_cap} 已用尽）")
        for chapter_id in sorted(result["remaining"]):
            print(f"   {chapter_id}: {', '.join(result['remaining'][chapter_id])}")
    return result


def run_final_gate(
    project_root=None,
    db_path=None,
    output_dir=None,
    qa_report_path=None,
    conversion_audit_path=None,
) -> dict:
    """T4: 交付门禁 — 完成信号不变量化（fail-closed）。

    ① 状态机：全部 chunk COMPLETED 且 DIRTY=0
    ② 产物：每章 final.json 数 == DB 任务数；无 *_final_backup.json 残留
    ③ QA_REPORT：无 fallback 章节条目（## 段）
    ④ T5 lint（finals + merged）全绿
    ⑤ T1 转换审计报告结论为 "✅ 通过"
    ⑥ L1 韵文对齐（verse_lint，门禁内联自动运行）
    ⑦ L2 双关核销（pun_audit，门禁内联自动运行；Round 130 起无需手动补跑）
    任一为红 → ok=False（translate_book.sh 以非零退出中止 / ⛔）。
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    db_file = Path(db_path) if db_path else root / "db" / "workflow.db"
    out_root = Path(output_dir) if output_dir else root / "output"
    qa_file = Path(qa_report_path) if qa_report_path else out_root / "QA_REPORT.md"
    audit_file = (Path(conversion_audit_path) if conversion_audit_path
                  else root / "input" / "_conversion_audit.md")

    failures: list = []
    warnings: list = []
    stats: dict = {"tasks": 0, "states": {}, "chapters": 0, "qa_fallbacks": 0}
    by_state: dict = {}
    expected: dict = {}

    # ① 状态机 + ② 产物
    rows = []
    scheduler = None
    if not db_file.exists():
        failures.append(f"① workflow.db 不存在: {db_file}")
    else:
        scheduler = TaskScheduler(db_path=str(db_file))
        rows = scheduler.conn.execute(
            "SELECT chapter_id, chunk_id, state FROM chunk_tasks").fetchall()
    stats["tasks"] = len(rows)
    for ch, _cid, st in rows:
        by_state[st] = by_state.get(st, 0) + 1
        expected[ch] = expected.get(ch, 0) + 1
    stats["states"] = by_state
    stats["chapters"] = len(expected)

    dirty_n = by_state.get(TaskState.DIRTY.value, 0)
    if dirty_n:
        failures.append(f"① DIRTY={dirty_n}（存在未整合脏块，先跑 consolidate）")
    non_completed = [(cid, st) for _ch, cid, st in rows if st != TaskState.COMPLETED.value]
    if non_completed:
        sample = ", ".join(f"{cid}:{st}" for cid, st in non_completed[:5])
        failures.append(f"① 非 COMPLETED 任务 {len(non_completed)} 个（如 {sample}）")

    for ch, exp in sorted(expected.items()):
        finals = list((out_root / ch).glob("*_final.json"))
        if len(finals) != exp:
            failures.append(f"② {ch}: final.json {len(finals)} 个 != DB 任务 {exp} 个")
    backups = [str(p.relative_to(root)) for p in out_root.glob("ch*/*_final_backup.json")]
    if backups:
        failures.append(f"② 残留 final_backup {len(backups)} 个: {backups[:5]}")

    # ③ QA_REPORT
    if not qa_file.exists():
        failures.append(f"③ QA_REPORT 不存在: {qa_file}")
    else:
        falls = [ln[3:].strip() for ln in qa_file.read_text(encoding="utf-8").split("\n")
                 if ln.startswith("## ")]
        stats["qa_fallbacks"] = len(falls)
        if falls:
            failures.append(f"③ QA_REPORT 记录 {len(falls)} 个 fallback 章节: {falls[:5]}")

    # ④ T5 lint
    try:
        from src.book_lint import lint_book
        lint_res = lint_book(project_root=str(root))
        stats["lint_failures"] = len(lint_res.get("failures", []))
        if not lint_res.get("ok"):
            failures.append(
                f"④ lint 未通过（{stats['lint_failures']} 项，详见 output/book_lint.md）")
    except Exception as exc:
        stats["lint_failures"] = -1
        failures.append(f"④ lint 执行异常: {exc}")

    # ⑤ T1 转换审计
    if not audit_file.exists():
        failures.append(f"⑤ 转换审计报告不存在: {audit_file}")
    else:
        audit_text = audit_file.read_text(encoding="utf-8")
        cov = re.search(r'全书覆盖率:\s*([\d.]+)%', audit_text)
        if cov:
            stats["audit_coverage"] = float(cov.group(1))
        if "✅ 通过" not in audit_text:
            failures.append(f"⑤ 转换审计未通过（详见 {audit_file}）")

    # ⑥ L1 韵文对齐（verse_lint；manifest 无韵文块或功能关闭时跳过）
    try:
        from src.verse import (load_verse_manifest, load_chunk_tags,
                               fragments_for_chunk, verse_lint, verse_config)
        _vcfg = verse_config()
        _vmani = load_verse_manifest(root / "input")
        _vtags = load_chunk_tags(root / "db")
        _vblocks_n = sum(len(v or []) for v in (_vmani.get("chapters") or {}).values())
        verse_stats = {"verse_blocks": _vblocks_n, "checked": 0, "failed": 0}
        stats["verse"] = verse_stats
        if _vcfg.get("enabled", True) and _vblocks_n:
            _vtext_rows = []
            if scheduler is not None:
                _vtext_rows = scheduler.conn.execute(
                    "SELECT chunk_id, chapter_id, text_content FROM chunk_tasks").fetchall()
            _vfail = []
            _vreport = ["# 韵文对齐报告 (verse_lint)", "",
                        f"- 韵文块总数: {_vblocks_n}", ""]
            for _vcid, _vch, _vtext in _vtext_rows:
                _vfinal = out_root / _vch / f"{_vcid}_final.json"
                if not _vfinal.exists():
                    continue
                _vfrags = fragments_for_chunk(_vch, _vcid, _vtext or "", _vtags, _vmani)
                if not _vfrags:
                    continue
                try:
                    _vdata = json.loads(_vfinal.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                _vftext = _vdata.get("text", "") if isinstance(_vdata, dict) else str(_vdata)
                _vres = verse_lint(_vfrags, _vftext,
                                   min_line_ratio=float(_vcfg.get("min_line_ratio", 0.6)),
                                   max_line_chars=int(_vcfg.get("max_line_chars", 56)),
                                   refrain_ratio=float(_vcfg.get("refrain_ratio", 0.5)))
                verse_stats["checked"] += 1
                if _vres["ok"]:
                    _vreport.append(
                        f"- ✅ {_vcid}: lines {_vres['found_lines']}/{_vres['expected_lines']}, "
                        f"refrain {_vres['refrain_target']}/{_vres['refrain_source']}")
                else:
                    verse_stats["failed"] += 1
                    _vreport.append(f"- ❌ {_vcid}: {_vres['reason']}")
                    _vfail.append(f"{_vcid}({_vres['reason']})")
            if verse_stats["checked"] == 0 and _vblocks_n:
                failures.append(
                    "⑥ 韵文清单非空但未匹配到任何 chunk（manifest/tags 可能陈旧，需重新 split+init）")
            if _vfail:
                failures.append(f"⑥ verse_lint 未达标 {len(_vfail)} 个 chunk: {_vfail[:5]}")
            (out_root / "verse_lint.md").write_text(
                "\n".join(_vreport) + "\n", encoding="utf-8")
    except Exception as _v_exc:
        stats["verse"] = {"error": str(_v_exc)}
        failures.append(f"⑥ verse_lint 执行异常: {_v_exc}")
    # ⑦ L2 双关核销（pun_audit；watchlist 为空时跳过）
    try:
        from src.style_profiler import load_profile as _load_profile
        _prof = _load_profile(root / "input" / "book_profile.json")
        _watch = (_prof or {}).get("pun_watchlist") or []
        if _watch:
            _pun_json = out_root / "pun_audit.json"
            # Round 130（外部审计②）: 门禁内联自动运行 pun_audit（纯静态、无 LLM 调用），
            # 消除"必须手动先跑 scripts/pun_audit.py，否则 gate ⑦ 必红"的顺序耦合。
            _pun_script = root / "scripts" / "pun_audit.py"
            if _pun_script.exists():
                import subprocess as _subprocess
                try:
                    _pun_proc = _subprocess.run(
                        [sys.executable, str(_pun_script),
                         "--project-root", str(root), "--no-fail"],
                        cwd=str(root), capture_output=True, text=True, timeout=120)
                    if _pun_proc.returncode != 0:
                        warnings.append(
                            f"⑦ pun_audit 自动执行异常(rc={_pun_proc.returncode}): "
                            f"{(_pun_proc.stderr or _pun_proc.stdout or '').strip()[:200]}")
                except Exception as _pun_run_exc:
                    failures.append(
                        f"⑦ pun_audit 自动执行失败: {type(_pun_run_exc).__name__}: {_pun_run_exc}")
            if not _pun_json.exists():
                failures.append("⑦ pun_audit 未运行（缺 output/pun_audit.json；自动补跑不可用，请手动跑 scripts/pun_audit.py）")
            else:
                _pun_report = json.loads(_pun_json.read_text(encoding="utf-8"))
                _pun_sum = _pun_report.get("summary") or {}
                stats["pun"] = _pun_sum
                _pun_miss = int(_pun_sum.get("uncovered_nonwaived", 0) or 0)
                if _pun_miss:
                    failures.append(f"⑦ pun_watchlist 未核销 {_pun_miss} 项（详见 output/pun_audit.md）")
    except Exception as _p_exc:
        failures.append(f"⑦ pun 审计执行异常: {_p_exc}")

    if stats["tasks"] == 0 and db_file.exists():
        failures.append("① workflow.db 无任务行（未 init？）")

    result = {"ok": not failures, "failures": failures, "warnings": warnings, "stats": stats}
    print("🎯 [Gate] 交付门禁检查：")
    print(f"  ① 状态机: {stats['tasks']} 个任务, states={by_state}")
    print(f"  ② 产物: {stats['chapters']} 章, final_backup={len(backups)}")
    print(f"  ③ QA_REPORT: fallback {stats['qa_fallbacks']} 章")
    print(f"  ④ lint: failures={stats.get('lint_failures')}")
    if stats.get("audit_coverage") is not None:
        print(f"  ⑤ 转换审计: 覆盖率 {stats['audit_coverage']}%")
    _vs = stats.get("verse") or {}
    if _vs.get("error"):
        print(f"  ⑥ 韵文: 异常 {_vs['error']}")
    elif _vs:
        print(f"  ⑥ 韵文: blocks={_vs.get('verse_blocks', 0)}, "
              f"checked={_vs.get('checked', 0)}, failed={_vs.get('failed', 0)}")
    _ps = stats.get("pun") or {}
    if _ps:
        print(f"  ⑦ 双关核销: covered={_ps.get('covered', 0)}/{_ps.get('total', 0)}, "
              f"waived={_ps.get('waived', 0)}, uncovered={_ps.get('uncovered_nonwaived', 0)}")
    for f in failures:
        print(f"  ❌ {f}")
    if result["ok"]:
        print("  ✅ gate 全绿（可交付）")
    return result

