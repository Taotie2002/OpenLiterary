"""
SagaTrans 任务状态机（TaskState + TaskScheduler）

源：原整合脚本 §1（枚举）+ §3（调度器），拆分到独立模块。
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


# Section 1 — 枚举与类型定义
# ────────────────────────────────────────────────────────────────────

class TaskState(Enum):
    """任务生命周期状态"""
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
    TaskState.FAILED:           [TaskState.EXTRACTING_TERMS, TaskState.TRANSLATING_RAW,
                                 TaskState.REWRITING_LITERARY, TaskState.AUDITING,
                                 TaskState.JUDGING, TaskState.PERMANENTLY_FAILED],
    TaskState.PERMANENTLY_FAILED: [],  # 终态：不可转移
    TaskState.EXTRACTING_TERMS: [TaskState.TRANSLATING_RAW, TaskState.FAILED],
    TaskState.TRANSLATING_RAW:  [TaskState.REWRITING_LITERARY, TaskState.FAILED],
    TaskState.REWRITING_LITERARY: [TaskState.AUDITING, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    # AUDITING 可退回润色：实体硬失败等质量问题无需再走 Judge
    # Round 107: 高分直通 (R101 P1) 从 AUDITING 直接 COMPLETED, 跳过 judge.
    # 之前缺 COMPLETED 导致转移被拒, chunk 卡 AUDITING 死循环, 最终被 NO_PROGRESS_STREAK 误杀.
    TaskState.AUDITING:         [TaskState.JUDGING, TaskState.COMPLETED, TaskState.REWRITING_LITERARY, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    TaskState.JUDGING:          [TaskState.COMPLETED, TaskState.REWRITING_LITERARY, TaskState.FAILED, TaskState.PERMANENTLY_FAILED],
    TaskState.COMPLETED:        [TaskState.DIRTY],
}



# T3：回溯触发精度 — 词边界精筛正则。
def _source_term_pattern(source_text: str):
    """构造「词边界」精筛正则（回溯触发精度）。

    - 仅字母参与边界（(?<![A-Za-z])/(?![A-Za-z])），避免 "well" 命中
      "dwelling"/"farewell" 之类子串；数字/CJK 不设边界（保护 "1984"、
      中文短语）。
    - 连字符与空白视作可互换分隔（变形容差：tea-party ≈ tea party）。
    - 大小写不敏感（源文大小写与决策条目常不一致）。
    - 大小写敏感：决策条目按原文摘录，大小写即语义（"Time" 不应命中
      任意小写 time；小写子串型伪匹配已由字母边界拦截）。唯一放宽：
      小写首字母词容忍句首大写（"please your Majesty" ≈ "Please ..."），
      仅放宽首字符、词内其余字符仍严格。
    返回 None 表示归一后为空。
    """
    norm = re.sub(r'\s+', ' ', (source_text or '').strip())
    raw_parts = [p for p in re.split(r'[\s\-]+', norm) if p]
    if not raw_parts:
        return None
    parts = [re.escape(p) for p in raw_parts]
    head = raw_parts[0][:1]
    if head.isalpha() and head.islower():
        parts[0] = (f'[{re.escape(head)}{re.escape(head.upper())}]'
                    f'{re.escape(raw_parts[0][1:])}')
    body = r'[\s\-]+'.join(parts)
    return re.compile(rf'(?<![A-Za-z]){body}(?![A-Za-z])')




# Section 3 — Scheduler 任务调度器
# ────────────────────────────────────────────────────────────────────

class TaskScheduler:
    def __init__(self, db_path="db/workflow.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
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

        # 任务序列表
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
        
        # 存量数据库兼容迁移
        try:
            cursor.execute('ALTER TABLE chunk_tasks ADD COLUMN quality_retries INTEGER DEFAULT 0')
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        
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
            # Round 11: smart-force 模式——保留已 COMPLETED 的 chunk，避免重译浪费 LLM 调用
            cursor.execute(
                'SELECT chunk_id, state FROM chunk_tasks WHERE chapter_id = ?',
                (chapter_id,)
            )
            existing_states = {row[0]: row[1] for row in cursor.fetchall()}
        skipped = 0
        preserved_completed = 0
        for i, text in enumerate(chunks):
            chunk_id = f"{chapter_id}_chunk{i:03d}"
            try:
                if force:
                    # Round 11: smart-force——若 chunk 已是 COMPLETED 状态，跳过重置（保留 final.json）
                    if existing_states.get(chunk_id) == TaskState.COMPLETED.value:
                        preserved_completed += 1
                        continue
                    # force 模式（但仅对非 COMPLETED）：覆盖 text_content 并完全重置
                    # 让 PERMANENTLY_FAILED/FAILED/PENDING 任务可重新执行
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
        # --force 时删除不在新 chunk 列表中的旧 DB 行（避免 stale 行骗过产出校验）
        if force:
            new_ids = {f"{chapter_id}_chunk{i:03d}" for i in range(len(chunks))}
            cursor.execute('SELECT chunk_id FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            old_ids = {row[0] for row in cursor.fetchall()}
            orphan_ids = old_ids - new_ids
            if orphan_ids:
                placeholders = ','.join(['?'] * len(orphan_ids))
                cursor.execute(f'DELETE FROM chunk_tasks WHERE chunk_id IN ({placeholders})', tuple(orphan_ids))
                print(f"  🧹 清理 {len(orphan_ids)} 个旧 chunk DB 行: {orphan_ids}")
        self.conn.commit()
        # Round 11: smart-force 报告保留的 COMPLETED chunk 数
        if force and preserved_completed:
            print(f"  🔒 smart-force：保留 {preserved_completed} 个 COMPLETED chunk（不重译）")

        # 用前后 COUNT 差值计算真正被覆盖的 chunk 数
        # Round 11: 减去 preserved_completed（那些没动 DB 的），否则会显示"覆盖更新"全部 chunk
        overwritten = 0
        if force:
            cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
            count_after = cursor.fetchone()[0]
            inserted = max(count_after - count_before, 0)
            overwritten = len(chunks) - inserted - preserved_completed
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
        """更新任务状态（含转移合法性校验）"""
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

    def batch_update_state(self, chunk_ids: List[str], new_state: TaskState, error_msgs: Dict[str, str] = None, quality_retry: bool = False, reset_counters: bool = False):
        """批量更新任务状态 - 减少数据库往返开销

        Args:
            chunk_ids: 任务 ID 列表
            new_state: 目标状态
            error_msgs: 可选，按 chunk_id 分错误信息 {chunk_id: error_msg}
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

        # 逐条执行以保留各自的 error_msg
        for cid in valid_ids:
            msg = error_msgs.get(cid) if error_msgs else None
            if msg and quality_retry:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, quality_retries = quality_retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, msg, cid))
            elif msg:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, retries = retries + 1, last_error = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, msg, cid))
            elif reset_counters:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, retries = 0, quality_retries = 0, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, cid))
            else:
                cursor.execute('''
                    UPDATE chunk_tasks
                    SET state = ?, last_error = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE chunk_id = ?
                ''', (new_state.value, cid))
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
        """回溯引擎：仅将已定稿(COMPLETED)的数据块标记为 DIRTY 重跑。

        与 VALID_TRANSITIONS 对齐：只允许 COMPLETED -> DIRTY。
        禁止对 JUDGING/AUDITING 等中间态打 DIRTY（否则会阻断 COMPLETED 并可能死循环）。
        尚未产出译文的 PENDING / EXTRACTING_TERMS 等也不回溯——后续阶段会自然读到新决策。
        """
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        # 仅 COMPLETED 可进入 DIRTY，与 VALID_TRANSITIONS[COMPLETED] 一致
        cursor.execute(f'''
            UPDATE chunk_tasks
            SET state = ?, updated_at = CURRENT_TIMESTAMP
            WHERE chunk_id IN ({placeholders})
            AND state = ?
        ''', (TaskState.DIRTY.value,) + tuple(chunk_ids) + (TaskState.COMPLETED.value,))
        self.conn.commit()
        updated = cursor.rowcount
        skipped = len(chunk_ids) - updated
        msg = f"🔄 已触发 {updated} 个数据块的回溯重构（标记 DIRTY）"
        if skipped > 0:
            msg += f"（跳过 {skipped} 个非 COMPLETED 状态）"
        print(msg + "。")

    def find_chunks_containing(
        self,
        source_text: str,
        chapter_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """全文搜索包含 source_text 的 chunk_id（T3：两级筛选）。

        一级：SQL LIKE 粗筛（原样 + 空白归一两种形态，后者用 SQLite
        REPLACE 把块内 \\n/\\r/\\t 折成空格，覆盖跨行折行短语）；二级：
        Python 词边界精筛（见 _source_term_pattern）。limit 在精筛后截断。

        Args:
            source_text: 要匹配的原文片段
            chapter_id: 若给定则仅扫该章；None 表示全书
            limit: 最多返回条数（None=不截断）
        """
        if not source_text:
            return []
        pattern = _source_term_pattern(source_text)
        if pattern is None:
            return []
        norm_term = re.sub(r'\s+', ' ', source_text).strip()

        def _like_esc(s: str) -> str:
            return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        raw_pat = f"%{_like_esc(source_text)}%"
        norm_pat = f"%{_like_esc(norm_term)}%"
        norm_sql = ("REPLACE(REPLACE(REPLACE(text_content, char(10), ' '), "
                    "char(13), ' '), char(9), ' ')")
        cond = f"text_content LIKE ? ESCAPE '\\' OR {norm_sql} LIKE ? ESCAPE '\\'"
        cursor = self.conn.cursor()
        if chapter_id:
            sql = (
                "SELECT chunk_id, text_content FROM chunk_tasks "
                f"WHERE chapter_id = ? AND ({cond}) "
                "ORDER BY chunk_id ASC"
            )
            params: list = [chapter_id, raw_pat, norm_pat]
        else:
            sql = (
                "SELECT chunk_id, text_content FROM chunk_tasks "
                f"WHERE {cond} "
                "ORDER BY chunk_id ASC"
            )
            params = [raw_pat, norm_pat]
        cursor.execute(sql, tuple(params))
        matched = []
        for chunk_id, text_content in cursor.fetchall():
            norm = re.sub(r'\s+', ' ', text_content or '')
            if pattern.search(norm):
                matched.append(chunk_id)
        if limit is not None and limit > 0:
            matched = matched[:int(limit)]
        return matched

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


# ────────────────────────────────────────────────────────────────────
