import sqlite3
import threading
from enum import IntEnum
from typing import List, Optional, Callable
from pathlib import Path

class DecisionLevel(IntEnum):
    TERMINOLOGY = 1  # 术语级，全局必须一致
    REFERENCE = 2    # 典故级，影响重写策略
    STYLE = 3        # 风格约束级，限制重写器词汇与句式

class DecisionEngine:
    def __init__(self, db_path="db/decision_db.sqlite", scheduler_factory: Optional[Callable] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._db_path_str = str(self.db_path)
        self._scheduler_factory = scheduler_factory
        self._init_tables()

    @property
    def conn(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path_str, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
        return self._local.conn

    def _init_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_db (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level INTEGER NOT NULL,
                source_key TEXT NOT NULL,
                translation TEXT NOT NULL,
                reason TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_source ON decision_db(source_key)')
        
        # 记录决策影响的 chunk 映射表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_impact (
                decision_id INTEGER,
                chunk_id TEXT,
                FOREIGN KEY(decision_id) REFERENCES decision_db(id),
                PRIMARY KEY (decision_id, chunk_id)
            )
        ''')
        self.conn.commit()

    def add_decision(self, level: DecisionLevel, source: str, translation: str, reason: str, affected_chunks: List[str] = None):
        """插入或更新决策，并记录影响的 chunk"""
        cursor = self.conn.cursor()
        try:
            # 原子化 UPSERT：INSERT ... ON CONFLICT DO UPDATE
            cursor.execute('''
                INSERT INTO decision_db (level, source_key, translation, reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    level = excluded.level,
                    translation = excluded.translation,
                    reason = excluded.reason,
                    updated_at = CURRENT_TIMESTAMP
            ''', (level.value, source, translation, reason))
            decision_id = cursor.lastrowid
            
            # 获取真实的决策 ID (UPSERT 后 lastrowid 在冲突时可能为 0)
            if not decision_id:
                cursor.execute('SELECT id FROM decision_db WHERE source_key = ?', (source,))
                row = cursor.fetchone()
                decision_id = row[0] if row else None
            
            # 记录影响的 chunk（先清理旧映射，再写入新映射）
            if affected_chunks and decision_id:
                cursor.execute('DELETE FROM decision_impact WHERE decision_id = ?', (decision_id,))
                cursor.executemany(
                    'INSERT OR IGNORE INTO decision_impact (decision_id, chunk_id) VALUES (?, ?)',
                    [(decision_id, cid) for cid in affected_chunks]
                )

            self.conn.commit()
            print(f"✅ [Decision Engine] 记录 {level.name}: {source} -> {translation}")

            # 触发回溯：必须在 commit 之后调用，避免跨库事务不一致
            if affected_chunks and decision_id:
                self._trigger_backtrack(affected_chunks)
        except Exception as e:
            print(f"❌ 决策写入失败: {e}")
            raise

    def _trigger_backtrack(self, chunk_ids: List[str]):
        """触发回溯：通过工厂函数获取 scheduler 并标记 DIRTY"""
        if not self._scheduler_factory:
            return
        try:
            scheduler = self._scheduler_factory()
            scheduler.trigger_backtrack(chunk_ids)
        except Exception as e:
            # 回溯失败是跨库一致性事件，不应静默吞掉
            raise RuntimeError(f"回溯触发失败，Decision DB 与 Workflow DB 可能不一致: {e}") from e

    def _cleanup_orphan_impacts(self, chunk_ids: List[str]):
        """清理指向已删除 chunk 的 decision_impact 孤儿记录"""
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        placeholders = ','.join('?' for _ in chunk_ids)
        cursor.execute(
            f'DELETE FROM decision_impact WHERE chunk_id IN ({placeholders})',
            tuple(chunk_ids)
        )
        self.conn.commit()

    def get_all_decisions(self):
        """为 Agent 提供 Prompt 上下文"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT level, source_key, translation FROM decision_db ORDER BY level ASC')
        return cursor.fetchall()

    def get_decisions_for_chunk(self, chunk_id: str):
        """获取影响特定 chunk 的所有决策"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT d.level, d.source_key, d.translation, d.reason
            FROM decision_db d
            JOIN decision_impact i ON d.id = i.decision_id
            WHERE i.chunk_id = ?
            ORDER BY d.level ASC
        ''', (chunk_id,))
        return cursor.fetchall()

    def set_scheduler_factory(self, factory: Callable):
        """设置调度器工厂函数（用于延迟绑定，避免循环导入）"""
        self._scheduler_factory = factory