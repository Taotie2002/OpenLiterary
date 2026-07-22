# src/core/scheduler.py
import sqlite3
import json
import time
import threading
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional

class TaskState(Enum):
    # 基础流程状态
    PENDING = "PENDING"
    EXTRACTING_TERMS = "EXTRACTING_TERMS"
    TRANSLATING_RAW = "TRANSLATING_RAW"
    REWRITING_LITERARY = "REWRITING_LITERARY"
    AUDITING = "AUDITING"
    JUDGING = "JUDGING"

    # 终态与异常态
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"           # 失败，等待重试 (retries < 3)
    PERMANENTLY_FAILED = "PERMANENTLY_FAILED"  # 终态：重试耗尽，需人工介入

    # 回溯专属状态
    DIRTY = "DIRTY"             # 因 Decision DB 变更被标记为已污染，需局部重跑

VALID_TRANSITIONS = {
    TaskState.PENDING:          [TaskState.EXTRACTING_TERMS],
    TaskState.DIRTY:            [TaskState.EXTRACTING_TERMS],
    # H6: FAILED 状态可恢复到任意中间阶段（由 _get_recovery_stage 解析 last_error 决定）
    TaskState.FAILED:           [TaskState.EXTRACTING_TERMS, TaskState.TRANSLATING_RAW,
                                 TaskState.REWRITING_LITERARY, TaskState.AUDITING,
                                 TaskState.JUDGING, TaskState.PERMANENTLY_FAILED],
    TaskState.PERMANENTLY_FAILED: [],  # 终态：不可转移
    TaskState.EXTRACTING_TERMS: [TaskState.TRANSLATING_RAW, TaskState.FAILED],
    TaskState.TRANSLATING_RAW:  [TaskState.REWRITING_LITERARY, TaskState.FAILED],
    TaskState.REWRITING_LITERARY: [TaskState.AUDITING, TaskState.FAILED],
    TaskState.AUDITING:         [TaskState.JUDGING, TaskState.FAILED],
    TaskState.JUDGING:          [TaskState.COMPLETED, TaskState.REWRITING_LITERARY, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    TaskState.COMPLETED:        [TaskState.DIRTY],
}

class TaskScheduler:
    def __init__(self, db_path="db/workflow.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._db_path_str = str(self.db_path)
        self._init_db()

    @property
    def conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path_str, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    def _init_db(self):
        """初始化任务流转表"""
        cursor = self.conn.cursor()
        
        # 任务序列表（quality_retries 用于质量重试，与 retries（API 错误重试）分离）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_tasks (
                chunk_id TEXT PRIMARY KEY,
                chapter_id TEXT,
                text_content TEXT,
                state TEXT NOT NULL,
                retries INTEGER DEFAULT 0,
                quality_retries INTEGER DEFAULT 0,
                last_error TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 存量数据库兼容迁移：新增列（已存在时忽略错误）
        try:
            cursor.execute('ALTER TABLE chunk_tasks ADD COLUMN quality_retries INTEGER DEFAULT 0')
            self.conn.commit()
        except sqlite3.OperationalError:
            pass  # 列已存在，跳过
        
        # 批处理支持索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_state ON chunk_tasks(state)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chapter ON chunk_tasks(chapter_id)')
        self.conn.commit()
    
    def init_chapter_tasks(self, chapter_id: str, chunks: list[str], force: bool = False):
        """批量注入章节任务

        Args:
            chapter_id: 章节 ID
            chunks: 切分后的文本块列表
            force: 是否覆盖已存在的 chunk（同时重置 text_content 和 retries）
        """
        cursor = self.conn.cursor()
        # 记录执行前的行数，用于准确统计 overwritten（SQLite ON CONFLICT DO UPDATE 的 rowcount 总是 1）
        if force:
            cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            count_before = cursor.fetchone()[0]
        skipped = 0
        for i, text in enumerate(chunks):
            chunk_id = f"{chapter_id}_chunk{i:03d}"
            try:
                if force:
                    # force 模式：覆盖 text_content 并完全重置（含 retries=0, quality_retries=0 让 PERMANENTLY_FAILED 任务可重新执行）
                    cursor.execute('''
                        INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(chunk_id) DO UPDATE SET
                            text_content = excluded.text_content,
                            state = ?,
                            last_error = NULL,
                            retries = 0,
                            quality_retries = 0
                    ''', (chunk_id, chapter_id, text, TaskState.PENDING.value, TaskState.PENDING.value))
                else:
                    cursor.execute('''
                        INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)
                        VALUES (?, ?, ?, ?)
                    ''', (chunk_id, chapter_id, text, TaskState.PENDING.value))
            except sqlite3.IntegrityError:
                skipped += 1
        self.conn.commit()
        # 用前后 COUNT 差值计算真正被覆盖的 chunk 数
        overwritten = 0
        if force:
            cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            count_after = cursor.fetchone()[0]
            inserted = max(count_after - count_before, 0)
            overwritten = len(chunks) - inserted
        if skipped > 0:
            print(f"⚠️ 跳过 {skipped} 个已存在的 chunk（若原文已变更请使用 --force 重新初始化）")
        if force and overwritten > 0:
            print(f"🔄 --force 模式：覆盖更新 {overwritten} 个已有 chunk")
        print(f"✅ 章节 {chapter_id} 初始化完成，共 {len(chunks)} 个执行块。")

    def get_tasks_by_state(self, state: TaskState, batch_size: int = 10, chapter_id: str | None = None) -> List[Dict]:
        cursor = self.conn.cursor()
        if chapter_id:
            cursor.execute('''
                SELECT * FROM chunk_tasks 
                WHERE state = ? AND chapter_id = ?
                ORDER BY chunk_id ASC LIMIT ?
            ''', (state.value, chapter_id, batch_size))
        else:
            cursor.execute('''
                SELECT * FROM chunk_tasks 
                WHERE state = ? 
                ORDER BY chunk_id ASC LIMIT ?
            ''', (state.value, batch_size))
        return [dict(row) for row in cursor.fetchall()]

    def update_task_state(self, chunk_id: str, new_state: TaskState, error_msg: str = None, quality_retry: bool = False):
        """更新任务状态（含转移合法性校验）

        Args:
            chunk_id: 任务 ID
            new_state: 目标状态
            error_msg: 错误信息（有值时自动递增 retries）
            quality_retry: 是否为质量重试（True=增 quality_retries，False=增 retries）
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT state FROM chunk_tasks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        if row:
            from_state = TaskState(row[0])
            allowed = VALID_TRANSITIONS.get(from_state, [])
            if new_state not in allowed:
                print(f"⚠️ [Scheduler] 非法转移 {chunk_id}: {row[0]} -> {new_state.value}")
                return
        if error_msg and quality_retry:
            cursor.execute('''
                UPDATE chunk_tasks 
                SET state = ?, quality_retries = quality_retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            ''', (new_state.value, error_msg, chunk_id))
        elif error_msg:
            cursor.execute('''
                UPDATE chunk_tasks 
                SET state = ?, retries = retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            ''', (new_state.value, error_msg, chunk_id))
        else:
            cursor.execute('''
                UPDATE chunk_tasks 
                SET state = ?, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
            ''', (new_state.value, chunk_id))
        self.conn.commit()

    def batch_update_state(self, chunk_ids: List[str], new_state: TaskState, error_msg: str = None, quality_retry: bool = False, reset_counters: bool = False):
        """批量更新任务状态 - 减少数据库往返开销

        Args:
            chunk_ids: 任务 ID 列表
            new_state: 目标状态
            error_msg: 错误信息（有值时自动递增 retries 或 quality_retries）
            quality_retry: 是否为质量重试（True=增 quality_retries，False=增 retries）
            reset_counters: 是否重置 retries 和 quality_retries 为 0（用于 DIRTY 回溯后重新计数）
        """
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)

        # 获取当前状态并校验转移合法性
        cursor.execute(f'''
            SELECT chunk_id, state FROM chunk_tasks WHERE chunk_id IN ({placeholders})
        ''', tuple(chunk_ids))
        current_states = {row[0]: row[1] for row in cursor.fetchall()}

        valid_ids = []
        for cid in chunk_ids:
            cur = current_states.get(cid)
            if cur is None:
                print(f"⚠️ [Scheduler] chunk_id {cid} 不存在，跳过")
                continue
            from_state = TaskState(cur)
            allowed = VALID_TRANSITIONS.get(from_state, [])
            if new_state not in allowed:
                print(f"⚠️ [Scheduler] 非法转移 {cid}: {cur} -> {new_state.value} (允许: {[s.value for s in allowed]})")
                continue
            valid_ids.append(cid)

        if not valid_ids:
            return

        valid_placeholders = ','.join('?' for _ in valid_ids)
        if error_msg and quality_retry:
            cursor.execute(f'''
                UPDATE chunk_tasks 
                SET state = ?, quality_retries = quality_retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id IN ({valid_placeholders})
            ''', (new_state.value, error_msg) + tuple(valid_ids))
        elif error_msg:
            cursor.execute(f'''
                UPDATE chunk_tasks 
                SET state = ?, retries = retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id IN ({valid_placeholders})
            ''', (new_state.value, error_msg) + tuple(valid_ids))
        elif reset_counters:
            cursor.execute(f'''
                UPDATE chunk_tasks 
                SET state = ?, retries = 0, quality_retries = 0, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id IN ({valid_placeholders})
            ''', (new_state.value,) + tuple(valid_ids))
        else:
            cursor.execute(f'''
                UPDATE chunk_tasks 
                SET state = ?, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE chunk_id IN ({valid_placeholders})
            ''', (new_state.value,) + tuple(valid_ids))
        self.conn.commit()

    def get_all_tasks_by_chapter(self, chapter_id: str) -> List[Dict]:
        """获取章节所有任务"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM chunk_tasks WHERE chapter_id = ? ORDER BY chunk_id', (chapter_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_task(self, chunk_id: str) -> Optional[Dict]:
        """获取单个任务"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM chunk_tasks WHERE chunk_id = ?', (chunk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def trigger_backtrack(self, chunk_ids: List[str]):
        """回溯引擎：将所有涉及的数据块标记为 DIRTY 重跑。
        跳过 PENDING 和 EXTRACTING_TERMS——尚未产出翻译内容，无需回溯，
        后续阶段的 decision_context 自然读到新决策。"""
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor.execute(f'''
            UPDATE chunk_tasks 
            SET state = ?, updated_at = CURRENT_TIMESTAMP
            WHERE chunk_id IN ({placeholders})
            AND state NOT IN (?, ?)
        ''', (TaskState.DIRTY.value, TaskState.PENDING.value, TaskState.EXTRACTING_TERMS.value) + tuple(chunk_ids))
        self.conn.commit()
        updated = cursor.rowcount
        print(f"🔄 已触发 {updated} 个数据块的回溯重构（标记 DIRTY）。")

    def get_completed_chunks(self, chapter_id: str) -> List[str]:
        """获取已完成的 chunk_id 列表"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT chunk_id FROM chunk_tasks 
            WHERE chapter_id = ? AND state = ?
            ORDER BY chunk_id
        ''', (chapter_id, TaskState.COMPLETED.value))
        return [row['chunk_id'] for row in cursor.fetchall()]

    def delete_tasks(self, chunk_ids: List[str]):
        """永久删除指定任务（用于清理永久失败的任务）"""
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor.execute(f'DELETE FROM chunk_tasks WHERE chunk_id IN ({placeholders})', tuple(chunk_ids))
        self.conn.commit()