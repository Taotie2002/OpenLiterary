import sys
import os
import time
import json
from pathlib import Path

# 1. 设置根目录锚点
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

# 2. 导入核心组件
from core.scheduler import TaskScheduler, TaskState
from core.decision_engine import DecisionEngine
from utils.llm_adapter import get_llm_client
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
        
        self.llm = get_llm_client()
        
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
            "avg_sentence_length": sg.get("avg_sentence_length", "较长且富有韵律"),
            "lexicon_preference": sg.get("lexicon_preference", "古典、史诗感、冷硬"),
            "author_priority_ratio": sg.get("author_priority_ratio", 0.7)
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
        model_cfg = config._get_model_config(model_key)
        model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "google/gemma-2-9b-it-mlx-4bit")
        return self.llm.generate(prompt, model_name=model_name, **params)

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
                tasks = self.scheduler.get_tasks_by_state(state, batch_size=batch_size)
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
        chunk_ids = [t['chunk_id'] for t in tasks]
        self.scheduler.batch_update_state(chunk_ids, TaskState.EXTRACTING_TERMS)

    def _process_failed_batch(self, tasks):
        max_retries = config.pipeline.get("max_retries", 3)
        retry_chunk_ids = []
        permanent_fail_ids = []
        for task in tasks:
            chunk_id = task['chunk_id']
            retries = task.get('retries', 0)
            if retries >= max_retries:
                print(f"❌ [Pipeline] {chunk_id} 重试次数过多，转入 PERMANENTLY_FAILED 终态")
                permanent_fail_ids.append(chunk_id)
            else:
                print(f"🔁 [Pipeline] {chunk_id} 重试中 (第 {retries + 1} 次)")
                retry_chunk_ids.append(chunk_id)
        if permanent_fail_ids:
            self.scheduler.batch_update_state(
                permanent_fail_ids,
                TaskState.PERMANENTLY_FAILED,
                error_msg=f"超过重试上限 (retries>={max_retries})，需人工介入"
            )
        if retry_chunk_ids:
            self.scheduler.batch_update_state(retry_chunk_ids, TaskState.EXTRACTING_TERMS)

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
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
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
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.REWRITING_LITERARY)

    def _process_rewriting_batch(self, tasks):
        success_ids = []
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                raw_text = self._load_intermediate(chunk_id, "raw")
                # 传递 source_text 用于动态 Author_Priority_Ratio 推断
                lit_text = self.rewriter_agent.process_chunk(chunk_id, raw_text, self.style_guide, source_text)
                self._save_intermediate(chunk_id, "literary", lit_text)
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 润色异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
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
                success_ids.append(chunk_id)
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 审计异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))
        if success_ids:
            self.scheduler.batch_update_state(success_ids, TaskState.JUDGING)

    def _process_judging_batch(self, tasks):
        max_retries = config.pipeline.get("max_retries", 3)
        for task in tasks:
            try:
                chunk_id = task['chunk_id']
                source_text = task['text_content']
                retries = task.get('retries', 0)
                lit_text = self._load_intermediate(chunk_id, "literary")
                critic_report = self._load_intermediate(chunk_id, "critic_report")
                judge_result = self.judge_agent.process_chunk(chunk_id, source_text, lit_text, critic_report, affected_chunks=[chunk_id])

                if judge_result.get("decision") == "PASS":
                    self._save_intermediate(chunk_id, "final", judge_result.get("final_text", lit_text))
                    self.scheduler.update_task_state(chunk_id, TaskState.COMPLETED)
                    print(f"✅ [Pipeline] {chunk_id} 定稿完成。")
                elif retries >= max_retries:
                    # 重试已达上限，直接转入终态
                    self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, error_msg=judge_result.get("reject_reason"))
                    print(f"❌ [Pipeline] {chunk_id} 连续 {retries+1} 次未通过裁决，转入 PERMANENTLY_FAILED")
                else:
                    self.scheduler.update_task_state(chunk_id, TaskState.REWRITING_LITERARY, error_msg=judge_result.get("reject_reason"))
            except Exception as e:
                print(f"❌ [Pipeline] {task['chunk_id']} 裁决异常: {e}")
                self.scheduler.update_task_state(task['chunk_id'], TaskState.FAILED, error_msg=str(e))

if __name__ == "__main__":
    pipeline = TranslationPipeline(chapter_id="ch01")
    pipeline.run()