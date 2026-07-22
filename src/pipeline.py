"""DEPRECATED — 模块化副本，未与生产单体脚本同步。

生产入口是 `python3 -m src.translator_agent`（单体脚本 `src/translator_agent.py`）。
此文件仅保留供历史参考，不应被 import 或直接运行。已知缺陷：
- `robust_json_loads` 类型不符不返回 empty（缺陷 B 未修复）
- `run()` 无 bool 返回值 + 无 `_verify_final_outputs`（缺陷 A 未修复）
- banner 仍含「🎉 全部处理完成」
"""
import sys
import os
import re
import time
import json
from pathlib import Path

# 1. 设置根目录锚点
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

# 2. 导入核心组件
from core.scheduler import TaskScheduler, TaskState
from core.decision_engine import DecisionEngine
from utils.llm_adapter import get_llm_client_for_task, get_role_model_name, get_role_extra_body
from agents.reference_agent import ReferenceAgent
from agents.rewriter_agent import LiteraryRewriterAgent
from agents.critic_agent import CriticAgent
from agents.judge_agent import JudgeAgent
from src.config import config

class TranslationPipeline:
    def __init__(self, chapter_id: str):
        self.chapter_id = chapter_id
        # 使用绝对路径定位数据库
        self.scheduler = TaskScheduler(db_path=str(ROOT_DIR / "db" / "workflow.db"))
        self.decision_engine = DecisionEngine(
            db_path=str(ROOT_DIR / "db" / "decision_db.sqlite"),
            scheduler_factory=lambda: self.scheduler
        )

        self.llm = get_llm_client_for_task("literal_translation")
        self.literal_role = config.get_task_role("literal_translation")

        # 实例化 Agents
        self.ref_agent = ReferenceAgent(self.decision_engine)
        self.rewriter_agent = LiteraryRewriterAgent(self.decision_engine)
        self.critic_agent = CriticAgent()
        self.judge_agent = JudgeAgent(self.decision_engine)
        
        # 绝对路径 output
        self.output_dir = ROOT_DIR / "output" / chapter_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sg = config.style_guide
        self.style_guide = {
            "avg_sentence_length": sg.get("avg_sentence_length", "适中、口语化，贴近原文句式长度"),
            "lexicon_preference": sg.get("lexicon_preference", "平实、克制、英式幽默，避免过度意译、四字格堆砌与装饰性发明"),
            "author_priority_ratio": sg.get("author_priority_ratio", 0.7),
            "genre": sg.get("genre", "儿童文学/维多利亚童话"),
            "work_type": sg.get("work_type", "长篇童话"),
        }

    def _save_intermediate(self, chunk_id: str, step: str, data: str | dict):
        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            if isinstance(data, str):
                json.dump({"text": data}, f, ensure_ascii=False, indent=2)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_intermediate(self, chunk_id: str, step: str) -> str | dict:
        file_path = self.output_dir / f"{chunk_id}_{step}.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("text", data)

    def _run_raw_translator(self, source_text: str) -> str:
        prompt = f"请将以下科幻小说片段进行直译。要求：字面忠实，不丢失任何细节，不进行文学润色。\n\n【原文】\n{source_text}"
        model_key, params = config.resolve_task_model("literal_translation")
        model_name = get_role_model_name(self.literal_role) or model_key
        extra_body = get_role_extra_body(self.literal_role)
        return self.llm.generate(prompt, model_name=model_name, extra_body=extra_body, **params)

    def run(self):
        print(f"🚀 [Pipeline] 启动批处理模式处理章节: {self.chapter_id}")

        # 检查章节是否有任务，避免在空 DB 上误报"完成"
        existing_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
        if not existing_tasks:
            print(f"❌ [Pipeline] 章节 {self.chapter_id} 无任务，请先运行 init 命令。")
            return
        
        pipe_cfg = config.pipeline
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
        
        while True:
            any_progress = False
            
            for state, handler, stage_name in pipeline_stages:
                # 批量获取该状态的任务
                tasks = self.scheduler.get_tasks_by_state(state, batch_size=batch_size, chapter_id=self.chapter_id)
                if not tasks:
                    continue
                
                print(f"📦 [Batch] {stage_name} 阶段: 处理 {len(tasks)} 个任务")
                handler(tasks)
                any_progress = True
            
            if not any_progress:
                print(f"🎉 [Pipeline] 章节 {self.chapter_id} 全部处理完成！")
                break
            
            time.sleep(poll_interval)

    def _process_pending_batch(self, tasks):
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS)

    def _process_dirty_batch(self, tasks):
        for task in tasks:
            chunk_id = task['chunk_id']
            print(f"🔄 [Pipeline] {chunk_id} 标记为 DIRTY，重置到术语提取")
            for step in ["raw", "literary", "critic_report", "final"]:
                intermediate = self.output_dir / f"{chunk_id}_{step}.json"
                if intermediate.exists():
                    intermediate.unlink()
            # 清理版本历史文件
            for pattern in [f"{chunk_id}_literary_v*.json", f"{chunk_id}_rewrite_meta_v*.json"]:
                for f in self.output_dir.glob(pattern):
                    f.unlink()
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS, reset_counters=True)

    def _get_recovery_stage(self, last_error: str) -> TaskState:
        if not last_error:
            return TaskState.EXTRACTING_TERMS
        stage_match = re.search(r'^\[(\w+)\]', last_error)
        if not stage_match:
            return TaskState.EXTRACTING_TERMS
        stage = stage_match.group(1)
        STAGE_MAP = {
            "EXTRACTING_TERMS": TaskState.EXTRACTING_TERMS,
            "TRANSLATING_RAW": TaskState.TRANSLATING_RAW,
            "REWRITING_LITERARY": TaskState.REWRITING_LITERARY,
            "AUDITING": TaskState.AUDITING,
            "JUDGING": TaskState.JUDGING,
        }
        return STAGE_MAP.get(stage, TaskState.EXTRACTING_TERMS)

    def _try_fallback_final(self, chunk_id: str) -> bool:
        """API 级重试耗尽时，从已有中间产物抢救 fallback final.json"""
        for step, label in [("literary", "文学润色稿"), ("raw", "直译稿")]:
            try:
                data = self._load_intermediate(chunk_id, step)
                text = data if isinstance(data, str) else data.get("text", str(data))
                if text and text.strip():
                    self._save_intermediate(chunk_id, "final", {
                        "text": text,
                        "metadata": {
                            "fallback": True,
                            "reason": f"API_FAILURE_FALLBACK_FROM_{step.upper()}",
                            "source_step": step,
                        }
                    })
                    print(f"  ↪ 已从{label}抢救 fallback final: {chunk_id}")
                    return True
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                continue
        return False

    def _process_failed_batch(self, tasks):
        max_retries = config.pipeline.get("max_retries", 3)
        retry_by_stage = {}
        permanent_fail_ids = []
        for task in tasks:
            chunk_id = task['chunk_id']
            retries = task.get('retries', 0)
            if retries >= max_retries:
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
                error_msg=f"超过重试上限 (retries>={max_retries})，需人工介入"
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
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 术语提取异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[EXTRACTING_TERMS] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.TRANSLATING_RAW)

    def _process_translating_raw_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                raw_text = self._run_raw_translator(source_text)
                self._save_intermediate(chunk_id, "raw", raw_text)
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 直译异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[TRANSLATING_RAW] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.REWRITING_LITERARY)

    def _process_rewriting_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                quality_retries = task.get('quality_retries', 0)

                reject_reason_raw = task.get('last_error', '')
                reject_reason = reject_reason_raw
                critic_feedback = None
                if reject_reason_raw:
                    try:
                        feedback = json.loads(reject_reason_raw)
                        reject_reason = feedback.get('judge_reason', '')
                        critic_feedback = feedback.get('critic_suggestions', '')
                        low_dims = feedback.get('low_dims', [])
                        dim_scores = feedback.get('scores', {})
                        if low_dims:
                            low_dim_detail = ", ".join(
                                f"{d}({dim_scores.get(d, '?')}分)" for d in low_dims
                            )
                            critic_feedback = f"低分维度：{low_dim_detail}\n" + (critic_feedback or '')
                    except (json.JSONDecodeError, TypeError):
                        pass

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

                lit_text = self.rewriter_agent.process_chunk(
                    chunk_id, raw_text, self.style_guide, source_text,
                    prev_lit_text=prev_lit_text,
                    reject_reason=reject_reason,
                    critic_feedback=critic_feedback,
                    retry_count=quality_retries,
                    all_feedback=all_feedback,
                )
                self._save_intermediate(chunk_id, "literary", lit_text)

                meta = {
                    "retry_count": quality_retries,
                    "quality_retries": quality_retries,
                    "reject_reason_summary": (reject_reason or "")[:200],
                    "critic_feedback_summary": critic_feedback[:200] if critic_feedback else None,
                    "all_feedback_summaries": all_feedback,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._save_intermediate(chunk_id, f"literary_v{quality_retries}", lit_text)
                self._save_intermediate(chunk_id, f"rewrite_meta_v{quality_retries}", meta)

                success_ids.append(chunk_id)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"❌ [Pipeline] {task['chunk_id']} 润色异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[REWRITING_LITERARY] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.AUDITING)

    def _process_auditing_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                raw_text = self._load_intermediate(chunk_id, "raw")
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self.critic_agent.process_chunk(chunk_id, source_text, raw_text, lit_text, self.style_guide)
                self._save_intermediate(chunk_id, "critic_report", critic_report)
                if not critic_report.get("scores"):
                    print(f"⚠️ [Pipeline] {chunk_id} Critic 评分为空（JSON 解析失败），转入 FAILED 等待重试")
                    self.scheduler.update_task_state(
                        chunk_id, TaskState.FAILED,
                        error_msg="[AUDITING] Critic 返回空 scores"
                    )
                    continue
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 审计异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=f"[AUDITING] {e}")
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.JUDGING)

    def _fallback_to_rewriter(self, chunk_id, lit_text, critic_report, judge_result):
        scores = critic_report.get("scores", {})
        thresholds = config.critic_thresholds
        low_dims = [k for k, v in scores.items()
                    if isinstance(v, (int, float)) and v < thresholds.get(k, 7.0)]
        specific_edits = critic_report.get("specific_edits", [])[:20]
        if specific_edits:
            for e in specific_edits:
                e["suggested_direction"] = (e.get("suggested_direction", "") or "")[:200]
        # 实体一致性：将 registry 主名反馈给 Rewriter，避免重试仍用错译名
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
        }, ensure_ascii=False)
        self.scheduler.update_task_state(
            chunk_id, TaskState.REWRITING_LITERARY,
            error_msg=enriched_feedback, quality_retry=True
        )
        print(f"🔁 [Pipeline] {chunk_id} 裁决未通过，退回文学润色")

    def _process_judging_batch(self, tasks):
        max_quality_retries = config.pipeline.get("max_quality_retries", 6)
        early_stop_cfg = config.pipeline.get("early_stop", {})
        early_stop_enabled = early_stop_cfg.get("enabled", True)
        early_stop_max_low = early_stop_cfg.get("max_low_dims", 3)
        early_stop_threshold = early_stop_cfg.get("low_score_threshold", 4.0)
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                quality_retries = task.get('quality_retries', 0)
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self._load_intermediate(chunk_id, "critic_report")
                judge_result = self.judge_agent.process_chunk(chunk_id, source_text, lit_text, critic_report, affected_chunks=[chunk_id])

                if judge_result.get("decision") == "PASS":
                    self._save_intermediate(chunk_id, "final", lit_text)
                    self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED)
                    print(f"✅ [Pipeline] {chunk_id} 定稿完成。")
                elif quality_retries >= max_quality_retries:
                    # P1 优化：安全网放行——若各维度已接近阈值（仅差 1-2 分），
                    # 不再整章 PERMANENTLY_FAILED 丢失译文，改为带告警定稿。
                    scores = critic_report.get("scores", {}) if isinstance(critic_report, dict) else {}
                    thresholds = config.critic_thresholds or {}
                    numeric = [v for v in scores.values() if isinstance(v, (int, float))]
                    avg_score = (sum(numeric) / len(numeric)) if numeric else 0.0
                    safety_net_min = float(config.pipeline.get("judge_safety_net_avg_min", 5.5))
                    severe_count = sum(
                        1 for k, v in scores.items()
                        if isinstance(v, (int, float))
                        and v < thresholds.get(k, 7.0) - 1.0
                    )
                    if avg_score >= safety_net_min and severe_count == 0:
                        self._save_intermediate(chunk_id, "final", {
                            "text": lit_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "SAFETY_NET_PASSED",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                                "avg_score": round(avg_score, 2),
                            }
                        })
                        self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED, error_msg=f"[SAFETY_NET] 平均分{avg_score:.2f}≥{safety_net_min}，接近阈值放行")
                        print(f"🟡 [Pipeline] {chunk_id} 安全网放行（平均分{avg_score:.2f}，未达PERMANENTLY_FAILED）")
                    else:
                        self._save_intermediate(chunk_id, "final", {
                            "text": lit_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "PERMANENTLY_FAILED_AFTER_MAX_RETRIES",
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        })
                        self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, error_msg=judge_result.get("reject_reason") or "质量重试耗尽")
                        print(f"⚠️ [Pipeline] {chunk_id} 质量重试耗尽，已 fallback 写入 final（最后一版润色稿）")
                elif early_stop_enabled and quality_retries == 0:
                    scores = critic_report.get("scores", {})
                    very_low_count = sum(
                        1 for v in scores.values()
                        if isinstance(v, (int, float)) and v < early_stop_threshold
                    )
                    if very_low_count >= early_stop_max_low:
                        stop_reason = (
                            f"EARLY_STOP: {very_low_count}项低于{early_stop_threshold}分"
                            f"（{scores}），放弃重试"
                        )
                        print(f"⏭️ [Pipeline] {chunk_id} {stop_reason}")
                        self._save_intermediate(chunk_id, "final", {
                            "text": lit_text,
                            "metadata": {
                                "fallback": True,
                                "reason": "EARLY_STOP",
                                "early_stop_detail": stop_reason,
                                "judge_reason": judge_result.get("reject_reason", ""),
                                "quality_retries": quality_retries,
                            }
                        })
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

if __name__ == "__main__":
    pipeline = TranslationPipeline(chapter_id="ch01")
    pipeline.run()