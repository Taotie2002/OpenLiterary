"""
SagaTrans 决策引擎（DecisionLevel + DecisionEngine）

源：原整合脚本 §1（DecisionLevel 枚举）+ §4（DecisionEngine 类），拆分到独立模块。
"""
from __future__ import annotations

import sqlite3
import re
import threading
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any

from src.config import config as _config
from src.utils.llm_adapter import robust_json_loads


class DecisionLevel(IntEnum):
    """决策等级"""
    TERMINOLOGY = 1  # 术语级，全局必须一致
    REFERENCE = 2    # 典故级，影响重写策略
    STYLE = 3        # 风格约束级，限制重写器词汇与句式


# ────────────────────────────────────────────────────────────────────


# Section 4 — Decision Engine 决策引擎
# ────────────────────────────────────────────────────────────────────

class DecisionEngine:
    def __init__(self, db_path="db/decision_db.sqlite", scheduler_factory: Optional[Callable] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
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

        # 实体注册表：增量构建的全书实体知识库
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,          -- 主名（中文译名）
                source_names TEXT NOT NULL,            -- JSON数组：原文出现的所有指称形式
                pronoun_gender TEXT,                   -- he/she/it/unknown
                entity_type TEXT,                      -- person/place/organization/other
                first_seen_chunk TEXT,                 -- 首次出现的 chunk_id
                confidence TEXT,                       -- high/medium/low
                pending_review INTEGER DEFAULT 0,      -- 0/1：是否有待人工/后续核查的歧义
                review_reason TEXT,                    -- 待核查原因说明
                translation_strategy TEXT DEFAULT 'free', -- literal/transliteration/free：翻译策略，首次登记后固定
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_canonical ON entity_registry(canonical_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_first_seen ON entity_registry(first_seen_chunk)')

        # 中文异体写法列（如 爱丽丝/艾丽丝）；旧库幂等迁移
        try:
            self.conn.execute("SELECT variant_names FROM entity_registry LIMIT 1")
        except Exception:
            self.conn.execute("ALTER TABLE entity_registry ADD COLUMN variant_names TEXT")

        try:
            self.conn.execute("SELECT translation_strategy FROM entity_registry LIMIT 1")
        except Exception:
            self.conn.execute("ALTER TABLE entity_registry ADD COLUMN translation_strategy TEXT DEFAULT 'free'")

        # 决策锁定列（Round 121）：防跨章译文翻转 → 回溯乒乓。
        # locked=1 的决策拒绝覆盖性更新且不触发 backtrack；旧库幂等迁移
        try:
            self.conn.execute("SELECT locked FROM decision_db LIMIT 1")
        except Exception:
            self.conn.execute("ALTER TABLE decision_db ADD COLUMN locked INTEGER DEFAULT 0")

        # Round 130: deferred 回溯队列 — 决策变更先登记，章末合并应用一次
        # （防"决策翻转→回溯已完成块→重跑再翻转"自激长跑；同一 chunk 幂等合并）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_backtrack (
                chunk_id TEXT PRIMARY KEY,
                decision_id INTEGER,
                source_key TEXT,
                queue_rounds INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()

        # 种子：将已知中文异体登记进 entity_registry（幂等，仅合并不覆盖）
        try:
            seed_entity_variants(self.conn)
        except Exception as e:
            print(f"  ⚠️ 实体异体种子失败: {e}", file=sys.stderr)

    def is_canonical_name(self, name: str) -> bool:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT 1 FROM entity_registry WHERE canonical_name = ? LIMIT 1", (name,))
            return cur.fetchone() is not None
        except Exception:
            return False

    def add_entity_variant(self, canonical: str, variant: str):
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT id, variant_names FROM entity_registry WHERE canonical_name = ?", (canonical,))
            row = cur.fetchone()
            if not row:
                return
            vid, vn_json = row
            variants = robust_json_loads(vn_json, expected_type=list) if vn_json else []
            if variant in variants:
                return
            variants.append(variant)
            cur.execute(
                "UPDATE entity_registry SET variant_names = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(variants, ensure_ascii=False), vid),
            )
            self.conn.commit()
        except Exception as e:
            print(f"  ⚠️ 登记实体异体失败 ({canonical}<-{variant}): {e}", file=sys.stderr)

    def add_decision(self, level: DecisionLevel, source: str, translation: str, reason: str, affected_chunks: List[str] = None):
        """插入或更新决策，并记录影响的 chunk。

        幂等：同一 source_key 且 translation 未变时不触发 backtrack。
        STYLE 默认不回溯（见 style_triggers_backtrack）；即使开启也不得依赖「当前 JUDGING chunk」。

        Round 130: backtrack_mode=deferred（默认）时，变更不再即时标记 DIRTY，
        而是登记进 pending_backtrack 队列，由 pipeline 在章末（队列清空点）合并应用一次。
        """
        cursor = self.conn.cursor()
        try:
            # 幂等检测：translation 未变则跳过 backtrack（仍可合并 impact）
            cursor.execute(
                'SELECT id, translation, locked FROM decision_db WHERE source_key = ?',
                (source,),
            )
            existing = cursor.fetchone()
            translation_unchanged = bool(
                existing and existing[1] == translation
            )

            # Round 121 锁定决策：拒绝覆盖性变更，保住跨章一致译名
            # （翻转会被 _find_chunks_containing 放大成全书回溯乒乓）
            if existing and int(existing[2] or 0) == 1 and not translation_unchanged:
                print(
                    f"🔒 [Decision Lock] {source}: 保留锁定译文「{existing[1]}」，"
                    f"忽略新译文「{translation}」（不触发回溯）"
                )
                return

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
            if existing:
                decision_id = existing[0]
            
            # 记录影响的 chunk：追加合并（不删除历史 impact）
            if affected_chunks and decision_id:
                cursor.executemany(
                    'INSERT OR IGNORE INTO decision_impact (decision_id, chunk_id) VALUES (?, ?)',
                    [(decision_id, cid) for cid in affected_chunks]
                )

            self.conn.commit()
            print(f"✅ [Decision Engine] 记录 {level.name}: {source} -> {translation}")

            if translation_unchanged:
                print(f"  ℹ️ 决策未变更 (source_key={source})，跳过回溯")
                return

            # 触发回溯：必须在 commit 之后调用，避免跨库事务不一致
            de_cfg = _config.decision_engine
            should_backtrack = False
            if level == DecisionLevel.TERMINOLOGY and de_cfg.get("terminology_triggers_backtrack", True):
                should_backtrack = True
            elif level == DecisionLevel.REFERENCE and de_cfg.get("reference_triggers_backtrack", True):
                should_backtrack = True
            elif level == DecisionLevel.STYLE and de_cfg.get("style_triggers_backtrack", False):
                # 默认 False：风格规则只累积进后续 prompt，不重跑已完成 chunk
                should_backtrack = True

            if should_backtrack and affected_chunks and decision_id:
                # 截断受影响 chunk 数量，避免单次决策雪崩式重译
                max_affected = int(de_cfg.get("max_affected_chunks_per_decision", 50) or 50)
                chunks_to_dirty = list(dict.fromkeys(affected_chunks))  # 保序去重
                if max_affected > 0 and len(chunks_to_dirty) > max_affected:
                    print(
                        f"  ⚠️ 受影响 chunk {len(chunks_to_dirty)} 个，"
                        f"截断为 max_affected_chunks_per_decision={max_affected}"
                    )
                    chunks_to_dirty = sorted(chunks_to_dirty)[:max_affected]
                backtrack_mode = str(de_cfg.get("backtrack_mode", "deferred") or "deferred").lower()
                if backtrack_mode == "immediate":
                    self._trigger_backtrack(chunks_to_dirty)
                else:
                    self._queue_backtrack(chunks_to_dirty, decision_id, source)
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

    def _queue_backtrack(self, chunk_ids: List[str], decision_id: Optional[int], source_key: str):
        """Round 130 deferred 模式：把受影响 chunk 登记进待回溯队列（幂等合并）。

        队列由 pipeline 在章末合并点经 take_pending_backtracks 取出后统一应用；
        同一 chunk 多次命中只保留一行并累计 queue_rounds，避免重复重跑。
        """
        if not chunk_ids:
            return
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT INTO pending_backtrack (chunk_id, decision_id, source_key, queue_rounds, updated_at)
            VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
            ON CONFLICT(chunk_id) DO UPDATE SET
                decision_id = excluded.decision_id,
                source_key = excluded.source_key,
                queue_rounds = COALESCE(queue_rounds, 0) + 1,
                updated_at = CURRENT_TIMESTAMP
        ''', [(cid, decision_id, source_key) for cid in chunk_ids])
        self.conn.commit()
        print(f"⏳ [Backtrack Queue] 登记 {len(chunk_ids)} 个待回溯 chunk（章末合并应用）: {source_key[:40]}")

    def take_pending_backtracks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """取出并删除指定 chunk 的待回溯登记（章末合并应用点调用）。

        Returns: [{"chunk_id", "decision_id", "source_key", "queue_rounds"}, ...]
        """
        if not chunk_ids:
            return []
        cur = self.conn.cursor()
        ph = ','.join('?' for _ in chunk_ids)
        rows = cur.execute(
            f'SELECT chunk_id, decision_id, source_key, COALESCE(queue_rounds, 0) AS queue_rounds '
            f'FROM pending_backtrack WHERE chunk_id IN ({ph})',
            tuple(chunk_ids)).fetchall()
        if rows:
            cur.execute(f'DELETE FROM pending_backtrack WHERE chunk_id IN ({ph})', tuple(chunk_ids))
            self.conn.commit()
        return [dict(r) for r in rows]

    def count_pending_backtracks(self, chunk_ids: Optional[List[str]] = None) -> int:
        """统计待回溯登记数（可限定 chunk 列表；None=全库）。"""
        cur = self.conn.cursor()
        if chunk_ids:
            ph = ','.join('?' for _ in chunk_ids)
            return int(cur.execute(
                f'SELECT COUNT(*) FROM pending_backtrack WHERE chunk_id IN ({ph})',
                tuple(chunk_ids)).fetchone()[0])
        return int(cur.execute('SELECT COUNT(*) FROM pending_backtrack').fetchone()[0])

    def get_all_decisions(self):
        """为 Agent 提供 Prompt 上下文（全量；优先用 format_prompt_context）"""
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

    def format_prompt_context(
        self,
        source_text: str = "",
        max_terminology: Optional[int] = None,
        max_reference: Optional[int] = None,
        max_style: Optional[int] = None,
        max_entities: Optional[int] = None,
    ) -> str:
        """有界决策/实体上下文：优先收录与 source_text 命中的条目，硬上限由配置控制。

        不再把 decision_db 全量无过滤灌进 prompt。
        """
        de_cfg = _config.decision_engine if hasattr(_config, "decision_engine") else {}
        max_terminology = max_terminology if max_terminology is not None else int(de_cfg.get("context_max_terminology", 40) or 40)
        max_reference = max_reference if max_reference is not None else int(de_cfg.get("context_max_reference", 20) or 20)
        max_style = max_style if max_style is not None else int(de_cfg.get("context_max_style", 10) or 10)
        max_entities = max_entities if max_entities is not None else int(de_cfg.get("context_max_entities", 30) or 30)

        src = source_text or ""
        src_lower = src.lower()

        cursor = self.conn.cursor()
        cursor.execute("SELECT level, source_key, translation FROM decision_db ORDER BY level ASC, id DESC")
        rows = cursor.fetchall()

        terms: List[str] = []
        refs: List[str] = []
        styles: List[str] = []
        # 命中优先：先扫 hit，再扫非 hit 填满配额
        term_hits, term_rest = [], []
        ref_hits, ref_rest = [], []
        style_all = []

        for level, source, trans in rows:
            line_t = f"- 术语: '{source}' -> 必须译为 '{trans}'"
            line_r = f"- 典故: '{source}' -> 必须译为 '{trans}' (若策略为保留并加注，请生成脚注)"
            line_s = f"- 风格: {trans}"
            hit = bool(source) and (source in src or (len(source) >= 3 and source.lower() in src_lower))
            if level == DecisionLevel.TERMINOLOGY.value:
                (term_hits if hit else term_rest).append(line_t)
            elif level == DecisionLevel.REFERENCE.value:
                (ref_hits if hit else ref_rest).append(line_r)
            elif level == DecisionLevel.STYLE.value:
                style_all.append(line_s)

        terms = (term_hits + term_rest)[:max_terminology]
        refs = (ref_hits + ref_rest)[:max_reference]
        styles = style_all[:max_style]

        entity_terms: List[str] = []
        try:
            rows = query_high_confidence_entities(self.conn)
            ent_hits, ent_rest = [], []
            for row in rows:
                canonical, source_names_json, variant_names_json, strategy = row[0], row[1], row[2], (row[3] if len(row) > 3 else 'free')
                if not canonical:
                    continue
                clean = re.sub(r"[^\u4e00-\u9fff\w\s]", "", canonical)[:30]
                if not clean or len(clean) < 2:
                    continue
                sn_list = robust_json_loads(source_names_json, expected_type=list) if source_names_json else []
                hit = any(
                    (sn and (sn in src or sn.lower() in src_lower))
                    for sn in sn_list
                ) or (clean in src)
                strategy_label = {"literal": "直译", "transliteration": "音译", "free": "自由意译"}.get(strategy, "自由意译")
                line = f"- 实体: '{clean}' (源语指称: {source_names_json}) -> 必须译为 '{clean}' [策略: {strategy_label}]"
                _vn = robust_json_loads(variant_names_json, expected_type=list) if variant_names_json else []
                if _vn:
                    line += f"；禁止写法（异体）: {', '.join(_vn)}"
                (ent_hits if hit else ent_rest).append(line)
            entity_terms = (ent_hits + ent_rest)[:max_entities]
        except Exception:
            pass

        if not terms and not refs and not styles and not entity_terms:
            return "无特殊词汇约束。"

        parts = ["【全局翻译决策（必须严格遵守；已按相关度截断）】"]
        if terms:
            parts.append("\n".join(terms))
        if refs:
            parts.append("\n".join(refs))
        if styles:
            parts.append("\n".join(styles))
        if entity_terms:
            parts.append("\n【实体译名约束（高置信度，必须严格遵守）】")
            parts.append("\n".join(entity_terms))
        truncated = (
            len(term_hits) + len(term_rest) > max_terminology
            or len(ref_hits) + len(ref_rest) > max_reference
            or len(style_all) > max_style
        )
        if truncated:
            parts.append(
                f"\n（上下文已截断：terminology≤{max_terminology}, "
                f"reference≤{max_reference}, style≤{max_style}, entities≤{max_entities}）"
            )
        return "\n".join(parts)

    def set_scheduler_factory(self, factory: Callable):
        """设置调度器工厂函数（用于延迟绑定，避免循环导入）"""
        self._scheduler_factory = factory


# P3-1: entity_registry 公共查询 helpers（双变体——confidence 过滤语义不等价）
def query_high_confidence_entities(conn) -> List[Tuple[str, str, str, str]]:
    """返回高置信度且无待核查的实体 (canonical_name, source_names_json, variant_names_json, translation_strategy)。

    用途：prompt 上下文注入、译后一致性报告。
    等价 SQL：`WHERE confidence = 'high' AND pending_review = 0`
    """
    cursor = conn.cursor()
    cursor.execute(
        'SELECT canonical_name, source_names, variant_names, translation_strategy FROM entity_registry '
        'WHERE confidence = "high" AND pending_review = 0'
    )
    return cursor.fetchall()


def query_reviewable_entities(conn) -> List[Tuple[str, str]]:
    """返回所有非待核查实体 (canonical_name, source_names_json, variant_names_json)。

    用途：译文核查 (Pipeline._check_entity_consistency)。
    **不过滤 confidence** —— low/medium 也要查，因为不一致检测需要更宽覆盖。
    等价 SQL：`WHERE pending_review = 0`
    """
    cursor = conn.cursor()
    cursor.execute(
        'SELECT canonical_name, source_names, variant_names FROM entity_registry '
        'WHERE pending_review = 0'
    )
    return cursor.fetchall()


def get_sibling_canonicals_by_source(conn) -> Dict[str, Set[str]]:
    """Round 71: 对 entity_registry 中每个 source_name, 返回它登记到的所有 canonical 集合.

    包含 pending_review = 0 与 = 1 的全部行 (与运行时 audit 仅看 0 不同).
    原因: ReferenceAgent 标记 pending_review=1 的行同样是有效 source_name 注册,
          如果不纳入, 跨任务冲突会被静默忽略 → rewriter 收到自相矛盾的反馈
          → 死循环 (ch32 实测 6 轮 quality_retries 全失败).

    返回: {source_name: {canonical, ...}}. 即使无冲突的 source_name 也保留
           (调用方可直接 set lookup, 不必先判 None).
    """
    cursor = conn.cursor()
    cursor.execute(
        'SELECT canonical_name, source_names FROM entity_registry'
    )
    rows = cursor.fetchall()
    out: Dict[str, Set[str]] = {}
    for canonical, source_names_json in rows:
        if not source_names_json:
            continue
        try:
            import json as _json
            srcs = _json.loads(source_names_json)
        except (ValueError, TypeError):
            continue
        if not isinstance(srcs, list):
            continue
        for sn in srcs:
            if isinstance(sn, str) and sn:
                out.setdefault(sn, set()).add(canonical)
    return out


def detect_cross_person_conflicts(conn) -> Dict[str, List[str]]:
    """Round 71: 检测 entity_registry 中 source_name 跨人物冲突 (含 pending_review).

    冲突定义: 同一英文 source_name 被登记到 ≥2 个 canonical 时 (不论 pending_review 状态).
    返回: {source_name: [canonical, ...]} 仅包含冲突条目.

    Note: 与 src/pipeline.py:_entity_registry_conflict_audit 同语义, 但本函数
          包含 pending_review = 1 的行, 用于运行时 sibling 查找.
    """
    siblings = get_sibling_canonicals_by_source(conn)
    return {sn: sorted(cs) for sn, cs in siblings.items() if len(cs) >= 2}


_CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def unify_entity_registry(conn, dry_run: bool = True) -> Dict[str, Dict]:
    """Round 105 P2-2: 归一 entity_registry 中「同人多规范名」冲突。

    背景（ch32 倒查根因）：同一人物被登记到多个 canonical（George Lomax →
    乔治·洛马克斯阁下 / 洛马克斯先生 / 乔治·洛马克斯先生），运行时 missing 核查
    只认 pending_review=0 的那一个，rewriter 用其它形式即被判 missing → 硬失败死循环。

    策略：
      1. 按共享 source_name 聚类（共享 ≥1 个 source_name 视为同一人）。
      2. 每个 ≥2 成员的簇选一个 winner：
         优先级 confidence(high>medium>low) → 更早 created_at → 更长规范名 → 字典序。
      3. 其余成员降级 pending_review=1（退出运行时核查，仍保留供 sibling 查找）。

    Args:
        conn: sqlite 连接（entity_registry 表）
        dry_run: True 仅返回计划不写库；False 实际 UPDATE。

    Returns:
        {cluster_repr: {"winner": canonical, "demoted": [canonical, ...]}}
        仅包含被处理的（≥2 成员）簇。
    """
    import json as _json

    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, canonical_name, source_names, confidence, created_at, pending_review "
        "FROM entity_registry"
    )
    rows = cursor.fetchall()

    # 聚类：canonical -> set(专名 source_names)
    # Round 105: 只用「含大写字母」的 source_name 聚类（专名/人名），排除
    # "a gentleman"/"the young man"/"the other"/"town" 等通用短语——否则并查集
    # 会经通用指称把不同人物传递性合并（ch32 实测 绅士乔 经 安东尼·凯德 误并入
    # 男爵洛洛普雷特济尔 簇）。
    canon_srcs: Dict[str, set] = {}
    meta: Dict[str, dict] = {}
    for rid, canon, srcs_json, conf, created, pending in rows:
        try:
            srcs = _json.loads(srcs_json) if srcs_json else []
        except (ValueError, TypeError):
            srcs = []
        specific = {
            s for s in srcs
            if isinstance(s, str) and s and any(ch.isupper() for ch in s)
        }
        canon_srcs.setdefault(canon, set()).update(specific)
        meta[canon] = {
            "id": rid, "confidence": conf or "low",
            "created": created or "", "pending": pending or 0,
        }

    # 并查集聚类（共享 source_name 连通）
    parent = {c: c for c in canon_srcs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    canons = list(canon_srcs)
    for i in range(len(canons)):
        for j in range(i + 1, len(canons)):
            if canon_srcs[canons[i]] & canon_srcs[canons[j]]:
                union(canons[i], canons[j])

    clusters: Dict[str, list] = {}
    for c in canons:
        clusters.setdefault(find(c), []).append(c)

    plan: Dict[str, Dict] = {}
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        # 选 winner：confidence → 去括注 base 长度（更长=更具体）→ 更早 created_at → 字典序
        import re as _re_u
        def _base_len(c):
            return len(_re_u.sub(r"（[^）]*）|\([^)]*\)", "", c).strip())
        def _key(c):
            m = meta[c]
            return (
                _CONF_ORDER.get(m["confidence"], 3),
                -_base_len(c),
                m["created"],
                c,
            )
        members_sorted = sorted(members, key=_key)
        winner = members_sorted[0]
        demoted = [c for c in members_sorted[1:] if meta[c]["pending"] == 0]
        if not demoted:
            continue
        plan[" / ".join(sorted(members))] = {"winner": winner, "demoted": demoted}
        if not dry_run:
            for c in demoted:
                cursor.execute(
                    "UPDATE entity_registry SET pending_review = 1, "
                    "review_reason = 'Round105 unify: 同人多规范名降级 (winner=' || ? || ')' "
                    "WHERE id = ? AND pending_review = 0",
                    (winner, meta[c]["id"]),
                )
    if not dry_run:
        conn.commit()
    return plan


# ────────────────────────────────────────────────────────────────────


# P1-4: 作品特定实体异体种子已从代码外移 — 默认空。
# 读取顺序：config.yaml `entity_variants` 段优先（可为空/可覆盖）；为空时回退
# per-work 作品资源文件 input/entity_variants.yaml（可选）。
SEED_ENTITY_VARIANTS: Dict[str, List[str]] = {}


def load_entity_variant_seeds() -> Dict[str, List[str]]:
    """P1-4 读取器：config 段优先，为空回退 per-work 资源文件（缺失 = 无种子）。"""
    try:
        from src.config import get_config as _get_cfg
        cfg_variants = getattr(_get_cfg(), "entity_variants", None)
        if isinstance(cfg_variants, dict) and cfg_variants:
            return cfg_variants
    except Exception:
        pass
    try:
        from pathlib import Path
        import yaml as _yaml
        seed_path = Path("input/entity_variants.yaml")
        if seed_path.exists():
            data = _yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return {str(k): list(v or []) for k, v in data.items()}
    except Exception:
        pass
    return {}


def seed_entity_variants(conn, variants: Optional[Dict[str, List[str]]] = None):
    """将 per-work 实体异体种子合并进 entity_registry.variant_names（幂等，不覆盖既有异体）。

    P1-4: 种子不再硬编码在通用代码里 — 默认读取配置 entity_variants 段，
    为空时回退 per-work 资源 input/entity_variants.yaml（均可选；缺失 = 不注入）。
    """
    if variants is None:
        variants = load_entity_variant_seeds()
    cur = conn.cursor()
    for canonical, variant_list in (variants or {}).items():
        variant_list = list(variant_list or [])
        if not canonical or not variant_list:
            continue
        cur.execute("SELECT id, variant_names FROM entity_registry WHERE canonical_name = ?", (canonical,))
        row = cur.fetchone()
        if not row:
            continue
        vid, vn_json = row
        existing = robust_json_loads(vn_json, expected_type=list) if vn_json else []
        merged = list(dict.fromkeys(existing + variant_list))
        if merged == existing:
            continue
        cur.execute(
            "UPDATE entity_registry SET variant_names = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(merged, ensure_ascii=False), vid),
        )
    conn.commit()
