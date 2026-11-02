"""
SagaTrans 决策引擎（DecisionLevel + DecisionEngine）

源：原整合脚本 §1（DecisionLevel 枚举）+ §4（DecisionEngine 类），拆分到独立模块。
"""
from __future__ import annotations

import sqlite3
import re
import json
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
                    # Round 134: variant_names 语义 = 同人白名单（允许写法）。
                    # 语境称谓（公爵/雷托公爵）与简称保留多样性，不得强改全名。
                    line += f"；同人异写（允许，按语域择自然者）: {', '.join(_vn)}"
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


def build_name_unification_maps(conn, text: str = "", preferred_forms: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Round 134: 从 entity_registry 生成「全文名称统一」映射。

    只在 pending_review=0 的 winner 行上工作（其 variant_names 是 unify 写入的白名单）：
      - policy=allow 的别名（公爵 / 雷托公爵 / 杰西卡）→ 只进 guards，绝不替换；
      - policy=replace 的别名（菲德-劳萨 / 保罗·厄崔迪 / 彼得·德弗里斯）→ 统一到 target；
      - token 级漂移映射：来自对齐结果（劳萨→罗萨 / 厄崔迪→阿特雷德斯），
        另对规范名 token 扫描正文，找出 ≥3 次的一字漂移形（费德→菲德 / 皮特→彼得）。
      - Round 137 非人名扩展：锚点/桥接不再限 person（组织/术语/地名同样统一）；
        新增「长公共段漂移扫描」捕获长度伸缩的二次音译（戈姆·贾巴尔 vs 戈姆刺 /
        阿特雷迪斯 vs 阿特雷德斯），候选一律经 LLM 批量裁决后才应用。
      - preferred_forms（可选，config 注入）：任何形式 → 首选形式，优先应用，
        用于「公认译名」这类外挂知识，而不必改代码（通用机制，无作品默认值）。

    Returns:
        {"forms": [[variant, target, winner], ...],
         "tokens": {from: to, ...},
         "authorities": [winner canonical, ...],
         "guards": [合法形式, ...]}
    """
    try:
        from src.utils.entity_stopwords import is_plausible_name_variant
    except Exception:  # 单文件部署兜底
        is_plausible_name_variant = None

    rows = _r134_registry_rows(conn)
    if not rows:
        return {"forms": [], "tokens": {}, "token_candidates": {},
                "authorities": [], "guards": []}
    text = text or ""

    forms: list = []
    tokens: Dict[str, str] = {}
    candidates: Dict[str, dict] = {}   # 扫描发现的漂移候选（需 LLM 裁决后才应用）
    authorities: list = []
    guards: set = set()
    all_canonicals: set = set()
    all_known: set = set()             # Round 136: canonical + 全部 variant（含 pending 行）
    person_compound: list = []  # Round 136: 人物+复合名（「一字之差」扫描锚点源）
    name_compound: list = []   # Round 137: 所有实体类型的「规范名 + 英文指称」（长公共段扫描锚点源）
    semantic_pending: list = []  # Round 137: 同源语义别名（零字面交集/证据不足，报告用）
    for _rid, canonical, _sn_json, vns_json, _etype, _conf, _created, _pending in rows:
        if canonical:
            _c0 = _r134_strip_parens(canonical)
            all_canonicals.add(_c0)
            all_known.add(_c0)
        for _v0 in (robust_json_loads(vns_json, expected_type=list) if vns_json else []):
            if isinstance(_v0, str) and _v0:
                all_known.add(_r134_strip_parens(_v0))

    # Round 136: 决策译名查找表（source_key 归一 → 当前译文），供「决策桥接」用。
    # 归一后精确匹配，锁定「同一英文指称」这一同一人证据。
    decision_lookup: Dict[str, str] = {}
    try:
        for _sk, _tr in conn.execute(
                "SELECT source_key, translation FROM decision_db "
                "WHERE source_key IS NOT NULL AND source_key != '' "
                "AND translation IS NOT NULL AND translation != ''"):
            _nk = source_name_normalized(_sk)
            _tr_s = (_tr or "").strip()
            if _nk and _tr_s:
                decision_lookup.setdefault(_nk, _tr_s)
                # Round 137: 冠词剥离别名——登记表来源 "the Residency" 与决策键
                # "Residency" 指同一指称，不剥冠词会静默错过桥接/语义别名检查。
                _nk2 = re.sub(r"^(?:the|a|an) ", "", _nk)
                if _nk2 and _nk2 != _nk:
                    decision_lookup.setdefault(_nk2, _tr_s)
    except Exception:
        decision_lookup = {}

    for _rid, canonical, sn_json, vns_json, etype, _conf, _created, pending in rows:
        if canonical:
            guards.add(_r134_strip_parens(canonical))
        if pending:
            continue  # 只在 winner（已归一）行上工作
        w = _r134_strip_parens(canonical or "")
        if not w:
            continue
        authorities.append(w)
        vns = robust_json_loads(vns_json, expected_type=list) if vns_json else []
        srcs = [x for x in (robust_json_loads(sn_json, expected_type=list) or [])
                if isinstance(x, str) and x][:4]
        # Round 136: 「一字之差」扫描锚点——仅人物 + 复合名（普通词邻域密集，防误伤）。
        if (etype or "").strip() in ("person", "人物", "") and (
                len(canonical_tokens(w)) >= 2 or vns):
            person_compound.append((w, srcs))
        # Round 137: 「长公共段」扫描锚点——扩展到全部实体类型（组织/术语/地名如
        # 宇航公会 / 戈姆刺 / 总督府，译名书写不一致同样须全文统一）。风险由三层
        # 兜底：锚点 token 过滤（非纯头衔/非语境指称词/不含功能字）+ 窗口边界规则
        # + 候选一律 LLM 批量裁决（宁缺勿滥）。
        if len(w) >= 2:
            name_compound.append((w, srcs))
        for v in vns:
            if not isinstance(v, str) or not v:
                continue
            cls = classify_entity_variant(v, w)
            guards.add(_r134_strip_parens(v))
            if cls["policy"] == "replace":
                forms.append([v, cls["target"], w])
                for a, b in cls.get("token_map") or []:
                    if a and b and a != b and len(a) >= 2:
                        tokens[a] = b

        # Round 136: 决策桥接的「分量级一致性」——同一人物的人名分量（first name /
        # last name）用字必须一致，但粒度各自保留（原文全名↔译文全名、原文简称↔
        # 译文简称，互不展开/收缩）。证据链：winner.source_names 的某英文指称 s 的
        # 当前决策译文 t（decision_db: s→t）；t 未登记为任何实体形式时，若 t 与
        # winner 分量对齐命中 replace（classify ④ 长度唯一匹配，或 ② 逐位配对），
        # 则把正文别名 t 统一为对齐结果。例（Dune ch01 实测）：Piter→皮特 且
        # winner=彼得·德·弗里斯（source 含 Piter）→ 正文「皮特」→「彼得」。
        if decision_lookup and sn_json:
            # Round 137: 桥接不再限 person 实体；但非人名必须额外过「逐分量相似度
            # ≥0.6」门槛（见下方 cls_b 之后）——整体对齐下限 0.5 会把同源不同指称
            # 的译法（Guildsman→公会成员 vs winner 宇航公会）误判为书写漂移。
            _etype_s = (etype or "").strip()
            _is_person_like = _etype_s in ("person", "people", "人物", "")
            for _raw in (robust_json_loads(sn_json, expected_type=list) or []):
                if not isinstance(_raw, str) or not _raw:
                    continue
                # 只信任「专名级」来源：关系指称（your father）与纯泛称/无专名
                # 证据的来源（lad / the child）不进桥接——其译文是语境指称，
                # 不属"人名分量"，统一会破坏称谓/语域多样性。
                if source_name_is_relational(_raw):
                    continue
                if not source_name_proper_tokens(_raw):
                    continue
                t = decision_lookup.get(source_name_normalized(_raw))
                if not t:
                    _ns = source_name_normalized(_raw)
                    t = decision_lookup.get(re.sub(r"^(?:the|a|an) ", "", _ns))
                if not t or t == w or len(t) < 2:
                    continue
                if t in all_known or any(t in k for k in all_known):
                    continue
                if t in guards or any(t in g for g in guards):
                    continue
                if text and t not in text:
                    continue
                if any(f0[0] == t for f0 in forms):
                    continue
                cls_b = classify_entity_variant(t, w)
                if cls_b["policy"] != "replace":
                    # Round 137: 同源不同形（Residency→官邸 vs winner 总督府）属
                    # 「语义别名」类——零字面交集不自动替换，记入报告（留
                    # preferred_forms / 人工处理，宁缺勿滥）。
                    if not _is_person_like and len(semantic_pending) < 20:
                        _note = f"{t}→{w}｜来源 {_raw}｜语义别名（未自动替换）"
                        if _note not in semantic_pending:
                            semantic_pending.append(_note)
                    continue
                if not _is_person_like and not _align_variant_tokens(t, w, min_token_sim=0.6):
                    # 非人名桥接的逐分量门槛（见上方注释）：证据不足 → 保守跳过。
                    if len(semantic_pending) < 20:
                        _note = f"{t}→{w}｜来源 {_raw}｜相似度不足（未自动替换）"
                        if _note not in semantic_pending:
                            semantic_pending.append(_note)
                    continue
                tgt = cls_b["target"]
                if not tgt or tgt == t:
                    continue
                forms.append([t, tgt, w])
                for a, b in cls_b.get("token_map") or []:
                    if a and b and a != b and len(a) >= 2:
                        tokens[a] = b

    # token 级漂移发现：对规范名 token 扫正文，找 ≥3 次的一字漂移形
    if text:
        # 漂移发现只覆盖「人物 + 复合名」的 token —— 单 token 普通名词/地名（大厅/城堡）
        # 一律不进扫描，避免 Round 105 那类「改坏正文」事故重演。
        authority_tokens: Dict[str, tuple] = {}   # token → (所属主名, 英文指称)
        for a, a_srcs in person_compound:
            for t in canonical_tokens(a):
                if len(t) >= 2 and not canonical_is_title_only(t):
                    authority_tokens.setdefault(t, (a, a_srcs))
        for _v, tgt, _w in forms:
            for t in canonical_tokens(tgt):
                if len(t) >= 2 and not canonical_is_title_only(t):
                    authority_tokens.setdefault(t, (_w, []))
        # 双向要求：权威 token 本身必须在正文出现 ≥1 次（否则不统一一个正文没用的形）
        authority_tokens = {t: v for t, v in authority_tokens.items() if t in text}
        cand_by_len: Dict[int, set] = {}
        cand_cnt_by_len: Dict[int, dict] = {}   # Round 137: 预计数，避免逐候选 O(n) count
        for L in {len(t) for t in authority_tokens}:
            _ngrams = re.findall(r"(?=([\u4e00-\u9fff]{%d}))" % L, text)
            cand_by_len[L] = set(_ngrams)
            _cnt: dict = {}
            for _g in _ngrams:
                _cnt[_g] = _cnt.get(_g, 0) + 1
            cand_cnt_by_len[L] = _cnt
        # Round 137: guards/all_canonicals 子串类排除「按长度预筛一次」，替代逐锚点
        # × 全窗口重算（ch01 实测 6000 万次 any → 130 万次，扫描耗时降 3 倍+）。
        cand_list_by_len: Dict[int, list] = {}
        for L, _cands in cand_by_len.items():
            cand_list_by_len[L] = sorted(
                c for c in _cands
                if c not in guards
                and not any(c in g for g in guards)
                and not any(c in c2 for c2 in all_canonicals)
            )
        for tok in sorted(authority_tokens, key=len, reverse=True):
            ent_name, ent_srcs = authority_tokens[tok]
            for cand in cand_list_by_len.get(len(tok), ()):
                if cand == tok or cand in tokens or cand in candidates or cand in guards:
                    continue
                if is_plausible_name_variant is not None:
                    ok, _reason = is_plausible_name_variant(tok, cand)
                    if not ok:
                        continue
                cnt = cand_cnt_by_len.get(len(tok), {}).get(cand, 0)
                if cnt < 3:
                    continue
                ctxs = []
                for m in re.finditer(re.escape(cand), text):
                    ctxs.append(text[max(0, m.start() - 12):m.end() + 12].replace("\n", " "))
                    if len(ctxs) >= 2:
                        break
                candidates[cand] = {
                    "to": tok, "count": cnt, "contexts": ctxs,
                    "entity": ent_name, "sources": list(ent_srcs),
                }

    # Round 137: 长公共段漂移扫描（非人名一致性扩展，与上面「一字之差」互补）。
    # 对每个权威 token（不限 person），在「名分隔符已折叠」的扫描视图里找与它共享
    # ≥2 字连续片段、但整词书写不同的专名形——覆盖长度伸缩的二次音译：
    #   戈姆·贾巴尔（gom jabbar 的「名+尾」写法）vs 戈姆刺；阿特雷迪斯 vs 阿特雷德斯。
    # 窗口三重护栏：① 首尾对齐（首/末字等于锚点首/末字，或原文邻字为标点/功能字）；
    # ② 左延伸一律以锚点首字为界（防「用戈姆刺」这类句子片段）；③ 窗口边缘与内部
    # 出现功能字（的/了/会…）直接丢弃；④ 相似度 [0.4, 0.95] + 出现次数门限。
    # 命中一律进 candidates 交 LLM 批量裁决（绝不由扫描直接替换，宁缺勿滥）。
    if text:
        scan_text, s_pos = _name_scan_view(text)
        seen_best: Dict[str, tuple] = {}
        anchors_long: Dict[str, tuple] = {}
        for a, a_srcs in name_compound:
            if len(a) < 3:
                continue  # 2 字名漂移由「一字之差」扫描覆盖；长段扫描对其仅产噪声
            for tok in canonical_tokens(a):
                if len(tok) >= 3 and not canonical_is_title_only(tok) \
                        and not _is_context_reference_word(tok) \
                        and not any(c in _ZH_ANCHOR_STOP for c in tok):
                    anchors_long.setdefault(tok, (a, a_srcs))
        for _v, tgt, _w in forms:
            for tok in canonical_tokens(tgt):
                if len(tok) >= 3 and not canonical_is_title_only(tok) \
                        and not _is_context_reference_word(tok) \
                        and not any(c in _ZH_ANCHOR_STOP for c in tok):
                    anchors_long.setdefault(tok, (_w, []))
        anchors_long = {t: v for t, v in anchors_long.items() if t in scan_text}
        # R1（Round 137）: 窗口内若包含任一已登记名形（锚点 token / 白名单 guards），
        # 多出的部分就是谓语/修饰（哈克南人 / 男爵问道 / 保罗低头看）→ 句子片段，
        # 非书写漂移，直接剔除（防「名字+动词」垃圾淹没裁决队列）。
        _r1 = set(anchors_long) | set(authority_tokens)
        for _g in guards:
            if _g and len(_g) >= 3 and all(_is_cjk_char(c) for c in _g):
                _r1.add(_g)
        if anchors_long:
            anchor_bigrams = set()
            for t in anchors_long:
                for k in range(len(t) - 1):
                    anchor_bigrams.add(t[k:k + 2])
            bigram_pos: Dict[str, list] = {}
            for k in range(len(scan_text) - 1):
                bg = scan_text[k:k + 2]
                if bg in anchor_bigrams:
                    bigram_pos.setdefault(bg, []).append(k)
            dead = set(guards) | set(tokens) | set(candidates)
            for a in sorted(anchors_long, key=len, reverse=True):
                owner, owner_srcs = anchors_long[a]
                lens = [L for L in (len(a) - 1, len(a), len(a) + 1, len(a) + 2)
                        if 3 <= L <= 6]
                if not lens:
                    continue
                for k in range(len(a) - 1):
                    for p in bigram_pos.get(a[k:k + 2], ()):
                        for L in lens:
                            lo = max(0, p - (L - 2))
                            hi = min(p, len(scan_text) - L)
                            for s0 in range(lo, hi + 1):
                                win = scan_text[s0:s0 + L]
                                if win == a or win in dead or win in a:
                                    continue
                                if any(T != win and T in win for T in _r1):
                                    continue
                                if not all(_is_cjk_char(c) for c in win):
                                    continue
                                if win[0] in _ZH_SCAN_EDGE_STOP or win[-1] in _ZH_SCAN_EDGE_STOP:
                                    continue
                                if any(c in _ZH_SCAN_EDGE_STOP for c in win):
                                    continue
                                if not (win[0] == a[0] or s0 == 0
                                        or not _is_cjk_char(scan_text[s0 - 1])):
                                    continue
                                right_ok = (win[-1] == a[-1] or s0 + L >= len(scan_text)
                                            or not _is_cjk_char(scan_text[s0 + L])
                                            or scan_text[s0 + L] in _ZH_SCAN_EDGE_STOP)
                                if not right_ok:
                                    continue
                                run = _longest_common_run(win, a)
                                if len(run) < 2:
                                    continue
                                if len(run) >= len(a) and len(win) > len(a):
                                    # 整锚点连续命中 + 额外字 = 「名字后接谓语/修饰」
                                    # 的句子片段（莫希阿姆说 / 哈克南军队），非书写漂移。
                                    continue
                                if len(run) < len(a) and len(run) < len(win):
                                    _pw, _pa = win.find(run), a.find(run)
                                    if (_pw == 0 and _pa == len(a) - len(run)) or \
                                            (_pw == len(win) - len(run) and _pa == 0):
                                        # 公共段「窗口头/锚点尾」（或镜像）= 头衔+谓语式
                                        # 片段（男爵问道 vs 哈克南男爵），非书写漂移。
                                        continue
                                sim = name_char_similarity(win, a)
                                if sim < 0.4 or sim > 0.95:
                                    continue
                                prev = seen_best.get(win)
                                if prev is None or sim > prev[0]:
                                    seen_best[win] = (sim, a, owner, owner_srcs, s0)
            for win, (_sim, a, owner, owner_srcs, s0) in sorted(seen_best.items()):
                if win in candidates or win in tokens or win in guards:
                    continue
                run = _longest_common_run(win, a)
                cnt = scan_text.count(win)
                if not (cnt >= 2 or (cnt >= 1 and len(run) >= 3)):
                    continue
                raw_form = text[s_pos[s0]:s_pos[s0 + len(win) - 1] + 1]
                if raw_form in candidates or raw_form in tokens or raw_form in guards:
                    continue
                if any(g and raw_form in g for g in guards):
                    continue
                if any(raw_form in c for c in all_canonicals):
                    continue
                if any(k and k in raw_form for k in all_known):
                    continue
                if _is_context_reference_word(raw_form) or raw_form not in text:
                    continue
                ctxs = []
                for m in re.finditer(re.escape(raw_form), text):
                    ctxs.append(text[max(0, m.start() - 12):m.end() + 12].replace("\n", " "))
                    if len(ctxs) >= 2:
                        break
                candidates[raw_form] = {
                    "to": a, "count": cnt, "contexts": ctxs,
                    "entity": owner, "sources": list(owner_srcs),
                    "form": raw_form,   # Round 137: 携带原文形 → 裁决通过后走 forms 通道
                }

    # preferred_forms：外挂首选形式（配置注入），优先于一切替换
    if preferred_forms:
        for src, pref in preferred_forms.items():
            if not src or not pref or src == pref:
                continue
            pref_tokens = canonical_tokens(pref)
            src_tokens = canonical_tokens(src)
            if len(pref_tokens) == len(src_tokens) and len(pref_tokens) >= 1:
                for a, b in zip(src_tokens, pref_tokens):
                    if a != b and len(a) >= 2:
                        tokens[a] = b
            forms.append([src, pref, pref])

    # token 冲突消解：被 guards / 其它 canonical 占据的来源形一律不做来源
    tokens = {a: b for a, b in tokens.items()
              if a != b and a not in guards and not any(a in c for c in all_canonicals)}
    return {"forms": forms, "tokens": tokens, "token_candidates": candidates,
            "authorities": authorities, "guards": sorted(guards),
            "semantic_pending": semantic_pending}


def apply_name_unification(text: str, maps: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Round 134: 在正文上执行名称统一（分隔符复原 → 异体整形 → token 漂移）。

    返回 (new_text, stats)。stats 含各规则替换计数 + 示例，供报告/审计。
    """
    stats: Dict[str, Any] = {"sep_restore": 0, "forms": 0, "tokens": 0, "examples": []}
    if not text or not maps:
        return text, stats

    def _record(rule, before, after, n=1):
        if n <= 0:
            return
        stats[rule] += n
        if len(stats["examples"]) < 12:
            stats["examples"].append(f"{before}→{after}" + (f"×{n}" if n > 1 else ""))

    # ⓪ 分隔符复原：规范名/目标形的「紧凑形」（去 ·/-）→ 标准形
    targets = list(dict.fromkeys(
        [a for a in maps.get("authorities", []) if a] +
        [t for _v, t, _w in maps.get("forms", []) if t]
    ))
    for target in sorted(targets, key=len, reverse=True):
        toks = canonical_tokens(target)
        if len(toks) < 2:
            continue
        compact = "".join(toks)
        if compact == target or compact not in text:
            continue
        n = text.count(compact)
        if n:
            text = text.replace(compact, target)
            _record("sep_restore", compact, target, n)

    # ① 注册异体整形（分隔符不敏感：彼得德弗里斯 / 保罗·厄崔迪 都命中）
    sep_pat = r"[\s" + re.escape(_NAME_SEP_CHARS) + r"]*"
    for variant, target, _w in sorted(maps.get("forms", []), key=lambda x: -len(x[0])):
        toks = canonical_tokens(variant)
        if not toks:
            continue
        pat = re.compile(sep_pat.join(re.escape(t) for t in toks))
        if not pat.search(text):
            continue
        counter = {"n": 0}

        def _rep(m, _t=target):
            got = m.group(0)
            if got == _t:
                return got
            counter["n"] += 1
            return _t

        text = pat.sub(_rep, text)
        _record("forms", variant, target, counter["n"])

    # ② token 级漂移（长的先，避免短形吃长形）
    for src in sorted(maps.get("tokens", {}), key=len, reverse=True):
        dst = maps["tokens"][src]
        if not src or src == dst or src not in text:
            continue
        n = text.count(src)
        text = text.replace(src, dst)
        _record("tokens", src, dst, n)
    return text, stats


def detect_cross_person_conflicts(conn) -> Dict[str, List[str]]:
    """Round 71: 检测 entity_registry 中 source_name 跨人物冲突 (含 pending_review).

    冲突定义: 同一英文 source_name 被登记到 ≥2 个 canonical 时 (不论 pending_review 状态).
    返回: {source_name: [canonical, ...]} 仅包含冲突条目.

    Note: 与 src/pipeline.py:_entity_registry_conflict_audit 同语义, 但本函数
          包含 pending_review = 1 的行, 用于运行时 sibling 查找.
    """
    siblings = get_sibling_canonicals_by_source(conn)
    return {sn: sorted(cs) for sn, cs in siblings.items() if len(cs) >= 2}


# ════════════════════════════════════════════════════════════════════
# Round 134: 通用实体命名工具（同人归一 / 称谓白名单 / 音译漂移统一）
#
# 设计红线（用户拍板）：**不写死任何作品专名**。全部判定建立在
# 「头衔/关系/泛指词表 + 名称相似度 + 登记证据」三类通用信号上：
#   ① 英文 source_name 的「专名证据」= 去掉冠词/所有格/头衔/关系/修饰词后
#      剩余的大写词（Duke Leto → {leto}；the old Duke → ∅）——这是聚类证据。
#   ② 中文 canonical 的「纯头衔/称谓」判定（公爵 / 老公爵 / 圣母 …）——
#      这类名字允许并存（文学多样性），绝不参与名称相似度合并，也绝不被替换。
#   ③ 名称 token 对齐（·/- 分段的）区分「简称（allow）」与「音译漂移（replace）」。
#
# 用途：
#   - unify_entity_registry()          同人多规范名合并：winner + 其余入白名单
#   - classify_entity_variant()        单对 (variant, winner) 的 allow/replace 判定
#   - build_name_unification_maps()    生成全文名称映射（供 consistency 译后统一）
#   - apply_name_unification()         在正文上执行映射（带护栏与计数）
# ════════════════════════════════════════════════════════════════════

# 英文头衔/角色词（出现于 source_name 时不构成专名证据）
_EN_ROLE_WORDS = frozenset({
    "duke", "duchess", "lord", "lady", "baron", "baroness", "sir", "dame",
    "king", "queen", "prince", "princess", "emperor", "empress", "archduke",
    "count", "countess", "earl", "viscount", "marquis", "marquess",
    "mother", "father", "sister", "brother", "priest", "priestess",
    "reverend", "reverence", "bishop", "archbishop", "pope", "cardinal",
    "pastor", "deacon", "nun", "monk", "abbot", "abbess", "holiness",
    "grace", "excellency", "highness", "majesty", "worship",
    "doctor", "dr", "mr", "mrs", "ms", "miss", "master", "mistress",
    "madame", "madam", "monsieur", "mademoiselle", "signor", "signora",
    "senor", "senora", "don", "dona", "captain", "commander", "general",
    "colonel", "major", "lieutenant", "sergeant", "admiral", "ensign",
    "corporal", "marshal", "warrant", "chief", "elder", "khan", "tsar",
    "czar", "sultan", "sheik", "emir", "pasha", "nabob", "mayor",
    "governor", "president", "minister", "chancellor", "ambassador",
    "consul", "senator", "steward", "warden", "champion", "knight",
})

# 英文关系词（"X's father" / "my mother" 之类不作为专名证据）
_EN_RELATION_WORDS = frozenset({
    "father", "mother", "son", "daughter", "brother", "sister", "wife",
    "husband", "widow", "widower", "grandfather", "grandmother", "grandson",
    "granddaughter", "uncle", "aunt", "nephew", "niece", "cousin", "heir",
    "heiress", "child", "children", "parent", "parents", "spouse",
    "godfather", "godmother", "stepfather", "stepmother", "partner",
})

# 英文泛指修饰词（old/young/dead…）——计入「头衔键」，用于区分 老公爵 vs 公爵
_EN_GENERIC_MODIFIERS = frozenset({
    "old", "young", "late", "dead", "living", "new", "former", "current",
    "present", "great", "little", "elder", "older", "younger", "dear",
    "poor", "good", "worthy", "honourable", "honorable", "esteemed",
    "beloved", "first", "second", "third", "fourth", "last", "next",
    "previous", "aforementioned", "departed", "recent", "future",
})

_EN_DETERMINERS = frozenset({
    "the", "a", "an", "my", "his", "her", "their", "our", "your", "its",
    "this", "that", "these", "those", "of", "and", "or", "who", "whom",
    "whose", "which", "is", "are", "was", "were", "be", "been", "being",
    "there", "here",
})

# 中文头衔/角色词（canonical 侧的「称谓类」判定）
_ZH_ROLE_WORDS = frozenset({
    "公爵", "女公爵", "大公", "伯爵", "女伯爵", "子爵", "男爵", "女男爵",
    "侯爵", "勋爵", "爵士", "阁下", "大人", "老爷", "夫人", "太太", "小姐",
    "女士", "先生", "陛下", "殿下", "国王", "女王", "皇帝", "皇后", "公主",
    "王子", "亲王", "王后", "储君", "太后", "神父", "牧师", "主教", "大主教",
    "教皇", "枢机", "修女", "修士", "嬷嬷", "教母", "教父", "圣母", "先知",
    "祭司", "大祭司", "萨满", "长老", "族长", "酋长", "可汗", "汗", "苏丹",
    "埃米尔", "帕夏", "沙阿", "将军", "元帅", "上校", "中校", "少校", "上尉",
    "中尉", "少尉", "队长", "舰长", "船长", "总统", "主席", "总理", "首相",
    "大臣", "部长", "总督", "省长", "市长", "博士", "教授", "医生", "护士",
    "警督", "警监", "警司", "警官", "警长", "侦探", "校长", "院长", "主任",
    "管家", "领班", "老板", "骑士", "侍女", "侍从", "仆人", "女仆", "男仆",
    "祖父", "外祖父", "祖母", "外祖母", "父亲", "母亲", "哥哥", "弟弟",
    "姐姐", "妹妹", "叔叔", "舅舅", "阿姨", "姨妈", "姑姑", "伯父", "伯母",
    "儿子", "女儿", "丈夫", "妻子", "遗孀", "继承人", "圣者", "圣人",
})

# 中文泛指修饰/前缀（老/大/小…）
_ZH_GENERIC_MODIFIERS = (
    "现任", "前任", "已故", "伟大的", "伟大",
    "老", "大", "小", "新", "前", "少", "幼",
)

_NAME_SEP_CHARS = "·∙・‧-–—ー/／"
_NAME_SEP_RE = re.compile(r"[" + re.escape(_NAME_SEP_CHARS) + r"]+")


# Round 137: 长公共段漂移扫描的「功能字」集合——候选窗口边缘/内部出现这些字，
# 一律视为句子片段而非专名（的了着过们之其…会能要中上下里外…）。
_ZH_SCAN_EDGE_STOP = frozenset(
    "的了着过们之其此该些个种呢吗吧啊呀嘛哦是在和与或及而但且不没又都也"
    "还被把让给使向对从由以因为于同跟连我你他她它这那哪谁什怎会能要"
    "中上下里外内间处时后前边侧更最很太再只才就"
)

# Round 137: 锚点过滤用的「纯功能字」小集（的组织性更强，公会/中央 类真实专名不受影响）。
_ZH_ANCHOR_STOP = frozenset("的了着过之其此该些个们")


def _is_cjk_char(ch):
    """Round 137: 单字符是否基本区汉字（漂移窗口构建用）。"""
    return bool(ch) and "\u4e00" <= ch <= "\u9fff"


def _name_scan_view(text):
    """Round 137: 构建「名分隔符已折叠」的扫描视图（戈姆·贾巴 → 戈姆贾巴）。

    仅删除被两个汉字夹住的分隔符段（·/-/空格等），标点/英文边界原样保留——
    名字内部的分隔符折叠后，跨分隔符的写法（戈姆·贾巴尔）才能与规范形
    （戈姆刺）做公共子串对齐。返回 (scan_text, pos_map)：
    pos_map[i] 为 scan_text[i] 在原文中的位置（用于回映原文形）。
    """
    chars = []
    pos = []
    n = len(text or "")
    i = 0
    while i < n:
        ch = text[i]
        if (ch in _NAME_SEP_CHARS or ch == " ") and chars:
            j = i
            while j < n and (text[j] in _NAME_SEP_CHARS or text[j] == " "):
                j += 1
            nxt = text[j] if j < n else ""
            if _is_cjk_char(chars[-1]) and _is_cjk_char(nxt):
                i = j
                continue
        chars.append(ch)
        pos.append(i)
        i += 1
    return "".join(chars), pos


def _r134_strip_parens(name):
    """剥离 canonical 中的括注（与 pipeline._strip_parenthetical 同语义）。"""
    if not name:
        return name or ""
    return re.sub(r"（[^）]*）|\([^)]*\)", "", name).strip()


def _en_tokens(source_name):
    return re.findall(r"[A-Za-z][A-Za-z'’\-\.]*", source_name or "")


def _en_normalize_token(tok):
    if not tok:
        return ""
    t = tok.strip().strip(".").lower()
    if t.endswith("'s") or t.endswith("’s"):
        t = t[:-2]
    return t


def source_name_is_relational(source_name):
    """Round 134: source_name 是否为「X's father / my mother」式关系指称。

    关系指称指向的是「另一个人的亲属身份」，不是该专名本人的名字——
    必须排除出聚类证据（否则 老公爵 会经 "Leto's father" 误并入 雷托 簇）。
    """
    toks = [t for t in _en_tokens((source_name or "").lower()) if t]
    for i, raw in enumerate(toks):
        tl = _en_normalize_token(raw)
        if tl not in _EN_RELATION_WORDS:
            continue
        prev = toks[i - 1] if i > 0 else ""
        prev_norm = prev.strip().strip(".").lower()
        if prev_norm.endswith("'s") or prev_norm.endswith("’s") or prev_norm in _EN_DETERMINERS:
            return True
    return False


def source_name_proper_tokens(source_name):
    """Round 134: source_name 的专名证据（小写 token 集合）。

    Duke Leto Atreides → {leto, atreides}；the old Duke → ∅；Leto's father → ∅。
    """
    if not source_name or source_name_is_relational(source_name):
        return frozenset()
    out = set()
    for raw in _en_tokens(source_name):
        if not raw or not raw[0].isupper():
            continue  # 专名证据只认大写词
        tl = _en_normalize_token(raw)
        if not tl or tl in _EN_DETERMINERS or tl in _EN_ROLE_WORDS \
                or tl in _EN_GENERIC_MODIFIERS or tl in _EN_RELATION_WORDS:
            continue
        out.add(tl)
    return frozenset(out)


def source_name_normalized(source_name):
    """Round 134: source_name 归一（小写、去所有格、标点折叠为空格）——用于精确共享证据。"""
    if not source_name:
        return ""
    s = source_name.lower()
    s = re.sub(r"[’']s\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def source_name_generic_keys(source_name):
    """Round 134: 泛指指称 → 头衔键（frozenset）。

    "the Duke" → {frozenset({"duke"})}；"the old Duke" → {{"duke","old"}}；
    "my Lord" → {{"lord"}}；"the man" → ∅（非头衔）；关系指称 → ∅。
    头衔键用于「纯头衔行 → 专名簇」的桥接证据，且**带修饰词的键不参与桥接**
    （保证 老公爵(the old Duke) 不会桥进 公爵(the Duke) 所在簇）。
    """
    if not source_name or source_name_is_relational(source_name):
        return set()
    toks = [_en_normalize_token(t) for t in _en_tokens(source_name)]
    toks = [t for t in toks if t and t not in _EN_DETERMINERS and t not in _EN_RELATION_WORDS]
    if not toks:
        return set()
    if any(t not in _EN_ROLE_WORDS and t not in _EN_GENERIC_MODIFIERS for t in toks):
        return set()  # 含专名/普通名词 → 不是纯头衔形式
    if not any(t in _EN_ROLE_WORDS for t in toks):
        return set()  # 无头衔词 → 不能作为桥接键
    return {frozenset(toks)}


def generic_key_is_plain(key):
    """Round 134: 头衔键是否为「无修饰词」的朴素形式（可用于桥接）。"""
    return bool(key) and not any(t in _EN_GENERIC_MODIFIERS for t in key)


def canonical_is_title_only(name):
    """Round 134: canonical 是否为纯头衔/称谓（公爵 / 老公爵 / 圣母 / 外祖父）。

    这类名字本身不携带专名信息，允许在正文里与具体主名并存（文学多样性），
    且绝不参与「名称相似度」合并 / 替换。散文式描述名（女主人公（未具名叙述者））
    也会命中此判定，同样受保护。
    """
    n = _r134_strip_parens(name or "")
    if not n:
        return False
    if any(ch.isascii() and ch.isalpha() for ch in n):
        return False  # 含拉丁字母 → 按专名处理
    if re.search(r"未具名|无名|匿名|隐含", n):
        return True  # 无名角色描述串：不可核查，同样不做合并/替换
    core = n
    for m in _ZH_GENERIC_MODIFIERS:
        if core.startswith(m) and len(core) > len(m):
            core = core[len(m):]
            break
    if core in _ZH_ROLE_WORDS:
        return True
    rest = core
    for w in sorted(_ZH_ROLE_WORDS, key=len, reverse=True):
        if rest.startswith(w):
            rest = rest[len(w):]
    return rest == ""


# Round 136: 中文敬语/关系指称（分量统一 ④ 护栏用）——这类是语境称谓，不参与
# 人名分量替换（用户规则：只约束「人名分量」用字；称谓/敬语多样性保留）。
_ZH_REFERENCE_WORDS = frozenset({
    "令尊", "令堂", "令郎", "令爱", "令千金", "尊夫人", "贤弟", "贤妹",
    "家父", "家母", "家兄", "舍弟", "舍妹", "贵干", "足下", "在下",
})


def _is_context_reference_word(word) -> bool:
    """Round 136: 判定 word 是否「语境指称/普通词」（分量统一 ④ 的危险输入）。

    孩子/少主/小子/老巫婆/令尊 这类描述性、敬语性指称绝不参与人名分量替换；
    判据（任一命中即保护）：
      - 已确认普通词黑名单（entity_stopwords.COMMON_WORDS）；
      - 中文敬语/关系指称表（_ZH_REFERENCE_WORDS）；
      - 带泛指修饰前缀（老/小/少/幼/新/前/大…，与 canonical_is_title_only 同表）。
    """
    if not word:
        return True
    if word in _ZH_REFERENCE_WORDS:
        return True
    try:
        from src.utils.entity_stopwords import is_likely_common_word
        if is_likely_common_word(word):
            return True
    except Exception:
        pass
    for m in _ZH_GENERIC_MODIFIERS:
        if word.startswith(m) and len(word) > len(m):
            return True
    return False


def canonical_tokens(name):
    """Round 134: canonical 按 ·/-/空格 等分隔符切 token（无分隔符的名字整体成 1 token）。"""
    n = _r134_strip_parens(name or "")
    return [p for p in _NAME_SEP_RE.split(n) if p]


def canonical_tokens_with_spans(name):
    """Round 134: 返回 [(token, start, end)]，用于按原分隔符形态重建目标串。"""
    n = _r134_strip_parens(name or "")
    out = []
    pos = 0
    for part in re.split(r"([" + re.escape(_NAME_SEP_CHARS) + r"\s]+)", n):
        if not part:
            continue
        if re.fullmatch(r"[" + re.escape(_NAME_SEP_CHARS) + r"\s]+", part):
            pos += len(part)
            continue
        out.append((part, pos, pos + len(part)))
        pos += len(part)
    return out


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(cur[-1], prev[j + 1]))
        prev = cur
    return prev[-1]


def name_char_similarity(a, b):
    """Dice 系数（基于 LCS）——用于 token 对齐与漂移判定。"""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return 2.0 * _lcs_len(a, b) / (len(a) + len(b))


def _align_variant_tokens(variant, winner, min_token_sim=0.5):
    """Round 134: 把 variant 的每个 token 依次对齐到 winner 的 token 段（可跨段拼接）。

    返回 list[(variant_token, w_start, w_end, sim, combined)]；无法全对齐返回 None。
    """
    vt = canonical_tokens(variant)
    wt = canonical_tokens(winner)
    if not vt or not wt:
        return None
    ai = 0
    alignments = []
    for tok in vt:
        best = None
        for s in range(ai, len(wt)):
            for e in range(s, len(wt)):
                comb = "".join(wt[s:e + 1])
                sim = 1.0 if tok == comb else name_char_similarity(tok, comb)
                if best is None or sim > best[3]:
                    best = (tok, s, e, sim, comb)
        if best is None or best[3] < min_token_sim:
            return None
        alignments.append(best)
        ai = best[2] + 1
    return alignments


def _longest_common_run(a, b, min_len=2):
    """Round 134: 返回 a、b 的最长公共子串（长度 ≥ min_len，否则空串）。"""
    if not a or not b:
        return ""
    best = ""
    la, lb = len(a), len(b)
    dp = [0] * (lb + 1)
    for i in range(1, la + 1):
        prev = 0
        for j in range(1, lb + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
                if dp[j] > len(best):
                    best = a[i - dp[j]:i]
            else:
                dp[j] = 0
            prev = cur
    return best if len(best) >= min_len else ""


def _split_by_common_run(a, b):
    """Round 134: 按最长公共子串把 a、b 拆成 (a_stem, w_stem)，用于逐位漂移映射。

    仅在公共子串位于两侧同侧、且两侧 stem 均 ≥2 字时返回；否则 None。
    """
    run = _longest_common_run(a, b)
    if not run:
        return None
    ia = a.find(run)
    ib = b.find(run)
    if ia < 0 or ib < 0:
        return None
    a_stem = a[:ia] + a[ia + len(run):]
    w_stem = b[:ib] + b[ib + len(run):]
    if len(a_stem) >= 2 and len(w_stem) >= 2:
        return a_stem, w_stem
    return None


def classify_entity_variant(variant, winner):
    """Round 134: 对 (variant, winner) 判定「允许并存」还是「须统一替换」。

    返回 {"policy": "equal"|"allow"|"replace", "target": str,
          "token_map": [(from, to), ...], "reason": str}

    规则（全部通用，不含作品专名）：
      - equal：同形 → 无事可做。
      - allow：① 纯头衔/称谓（公爵/老公爵）；② 含中文头衔词的复合写法（雷托公爵）；
               ③ variant 是 winner 的 token 简称（杰西卡 ⊂ 杰西卡夫人）→
               保留正文多样性，仅在登记表里作为「允许写法」。
      - replace：音译漂移（菲德-劳萨 vs 菲德-罗萨·哈克南 / 彼得·德弗里斯 /
               皮特 vs 彼得 / 厄崔迪 vs 阿特雷德斯）→ 全文统一到 target。
                Round 136：同名分量（first/last name）用字必须一致，但粒度各自
                保留——简称统一到对应分量形（皮特→彼得），不展开成全名。
    """
    v = _r134_strip_parens(variant or "")
    w = _r134_strip_parens(winner or "")
    if not v or not w or v == w:
        return {"policy": "equal", "target": w or v, "token_map": [], "reason": "同形"}
    if canonical_is_title_only(v):
        return {"policy": "allow", "target": w, "token_map": [], "reason": "纯头衔/称谓"}
    vt = canonical_tokens(v)
    wt = canonical_tokens(w)
    # 含中文头衔词的复合写法（如 雷托公爵）= 语境称谓 → 允许并存（正文多样性保障）
    if any(tok in _ZH_ROLE_WORDS or tok in _ZH_GENERIC_MODIFIERS for tok in vt) \
            or any(rw in v for rw in _ZH_ROLE_WORDS):
        return {"policy": "allow", "target": w, "token_map": [], "reason": "含头衔词的语境称谓"}
    # ⓪ 子串简称（杰西卡 ⊂ 杰西卡夫人 / 菲德 ⊂ 菲德-罗萨·哈克南）→ 允许并存
    if len(v) >= 2 and v in w and len(v) < len(w):
        return {"policy": "allow", "target": w, "token_map": [], "reason": "子串简称"}
    # ① 简称：variant 的 token 序列是 winner 的子序列（全等命中）
    if len(vt) <= len(wt):
        idx = 0
        ok = True
        for tok in vt:
            while idx < len(wt) and wt[idx] != tok:
                idx += 1
            if idx >= len(wt):
                ok = False
                break
            idx += 1
        if ok:
            return {"policy": "allow", "target": w, "token_map": [], "reason": "token 简称"}
    spans = canonical_tokens_with_spans(w)
    alignments = _align_variant_tokens(v, w)
    if alignments:
        s_min = alignments[0][1]
        e_max = alignments[-1][2]
        target = w[spans[s_min][1]:spans[e_max][2]] if spans else w
        token_map = [(tok, comb) for tok, _s, _e, sim, comb in alignments
                     if tok != comb and sim < 1.0]
        if target != v:
            return {"policy": "replace", "target": target, "token_map": token_map,
                    "reason": "音译漂移（token 对齐）"}
        return {"policy": "allow", "target": w, "token_map": [], "reason": "对齐后同形"}
    # ② 对齐失败但 token 数相等且存在全等 token → 逐位配对（厄崔迪→阿特雷德斯）
    if len(vt) == len(wt) and len(vt) >= 2:
        exact_pairs = sum(1 for a, b in zip(vt, wt) if a == b)
        if exact_pairs >= 1:
            token_map = [(a, b) for a, b in zip(vt, wt) if a != b and len(a) >= 2]
            return {"policy": "replace", "target": w, "token_map": token_map,
                    "reason": "音译漂移（逐位配对）"}
    # ③ 单 token 长名（无分隔符，如 厄崔迪家族 vs 阿特雷德斯家族）：按最长公共字串
    #    拆 stem 逐位配对（仅在两侧 stem 均 ≥2 字时启用，保守优先）
    if len(v) >= 4 and len(w) >= 4 and max(len(v), len(w)) <= 2 * min(len(v), len(w)) + 2:
        stems = _split_by_common_run(v, w)
        if stems:
            a_stem, w_stem = stems
            return {"policy": "replace", "target": w, "token_map": [(a_stem, w_stem)],
                    "reason": "音译漂移（公共字串拆分）"}
    # ④ Round 136: 单 token 别名 vs 复合名 —— 零字面交集的分量漂移
    #    （皮特 vs 彼得·德·弗里斯）。同一人物的「同名分量」（first name / last
    #    name）用字必须一致，但粒度各自保留：简称统一到对应分量形（皮特→彼得），
    #    绝不展开成全名。仅当分量长度唯一匹配时执行（多义/歧义一律保守放行）；
    #    到达此处时所有组合相似度均 <min_token_sim（0.5），近形已在对齐路径处理。
    if len(vt) == 1 and len(wt) >= 2 and 2 <= len(v) <= 4 \
            and not _is_context_reference_word(v):
        cands = [t for t in wt
                 if len(t) == len(v) and t != v and not canonical_is_title_only(t)
                 and not _is_context_reference_word(t)]
        if len(cands) == 1:
            return {"policy": "replace", "target": cands[0],
                    "token_map": [(v, cands[0])],
                    "reason": "单 token 分量漂移（长度唯一匹配）"}
    return {"policy": "allow", "target": w, "token_map": [], "reason": "证据不足，保守放行"}


_CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def _r134_registry_rows(conn) -> List[tuple]:
    """Round 134: 容错读取 entity_registry（旧库/测试 fixture 可能缺 entity_type 等列）。

    统一归一为 (id, canonical_name, source_names, variant_names, entity_type,
    confidence, created_at, pending_review) 8 元组；缺失列取缺省值。
    """
    try:
        cur = conn.execute("SELECT * FROM entity_registry")
    except Exception:
        return []
    names = [d[0] for d in (cur.description or [])]
    rows = []
    for r in cur.fetchall():
        d = dict(zip(names, r))
        rows.append((
            d.get("id"),
            d.get("canonical_name"),
            d.get("source_names"),
            d.get("variant_names"),
            d.get("entity_type", "") or "",
            d.get("confidence"),
            d.get("created_at") or "",
            d.get("pending_review") or 0,
        ))
    return rows


def unify_entity_registry(conn, dry_run: bool = True) -> Dict[str, Dict]:
    """Round 134: 同人多规范名归一（通用版，取代 Round 105 原型）。

    背景（ch01 倒查根因）：同一人物被登记到多个 canonical（Duke Leto →
    雷托公爵 / 公爵 / 雷托·阿特雷德斯），运行时核查只认 pending_review=0 的一个，
    rewriter 用其它形式即被判 missing / variant_instead_of_canonical → 硬失败
    死循环（ch01 实测 6 个 chunk 因此 PERMANENTLY_FAILED）。

    聚类证据（全部通用，不写死任何作品专名）：
      E1 共享专名 source_name（归一后精确相等）。冠词/所有格/头衔/关系/修饰词不算
         专名证据 —— "Leto's father"、"the old Duke" 因此不能把 老公爵 拉进雷托簇。
      E2 名称漂移等价：·/- 分段 token 对齐（较短一方全部 token 可对齐进较长一方的
         token 段，每 token 相似度 ≥0.5，且至少一个全等）。纯头衔名不参与。
      E3 桥接：无专名证据的纯头衔行（公爵）的头衔键与**恰好一个**专名簇的朴素
         头衔键重合 → 并入该簇；带修饰词的键（the old Duke）不参与 → 老公爵独立。
    实体类型冲突（person vs place 等）不合并。

    winner 规则：confidence → 含「·」→ 去括注长度 → 更早 created_at → 字典序。

    合并动作：
      - winner 收编全部成员的 source_names（合并去重）；
      - 被降级 canonical 写入 winner.variant_names（白名单：正文允许并存，交
        classify_entity_variant 再分「称谓类 allow」/「音译漂移 replace」）；
      - 被降级行 pending_review=1；winner pending_review=0（解除歧义冻结）；
      - Class B（音译漂移）成员在 replace_pairs 里列出 (variant → target)。

    Args:
        conn: sqlite 连接（entity_registry 表）
        dry_run: True 仅返回计划不写库；False 实际 UPDATE（幂等，可反复调用）。

    Returns:
        {cluster_repr: {"winner", "demoted", "whitelist", "replace_pairs",
                        "evidence", "changed"}}；仅含 ≥2 成员的簇。
    """
    cursor = conn.cursor()
    rows = _r134_registry_rows(conn)
    if not rows:
        return {}

    meta: Dict[str, dict] = {}
    order: list = []
    for rid, canon, srcs_json, vns_json, etype, conf, created, pending in rows:
        if not canon:
            continue
        try:
            srcs = json.loads(srcs_json) if srcs_json else []
        except (ValueError, TypeError):
            srcs = []
        proper_src, proper_tokens, generic_keys, plain_keys = set(), set(), set(), set()
        for sn in srcs:
            if not isinstance(sn, str) or not sn:
                continue
            pt = source_name_proper_tokens(sn)
            if pt:
                proper_src.add(source_name_normalized(sn))
                proper_tokens |= set(pt)
                continue
            for k in source_name_generic_keys(sn):
                generic_keys.add(k)
                if generic_key_is_plain(k):
                    plain_keys.add(k)
        meta[canon] = {
            "id": rid, "confidence": conf or "low", "created": created or "",
            "pending": pending or 0, "etype": (etype or "").strip(),
            "proper_src": proper_src, "proper_tokens": proper_tokens,
            "generic_keys": generic_keys, "plain_keys": plain_keys,
            "title_only": canonical_is_title_only(canon),
            "tokens": canonical_tokens(canon),
            "srcs": [s for s in srcs if isinstance(s, str) and s],
            "variants": robust_json_loads(vns_json, expected_type=list) if vns_json else [],
        }
        order.append(canon)

    parent = {c: c for c in order}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def _type_conflict(a, b):
        ta, tb = meta[a]["etype"], meta[b]["etype"]
        return bool(ta and tb and ta != tb)

    def _drift_equivalent(a, b):
        ma, mb = meta[a], meta[b]
        if ma["title_only"] or mb["title_only"]:
            return False
        short, long_ = (a, b) if len(ma["tokens"]) <= len(mb["tokens"]) else (b, a)
        st, lt = meta[short]["tokens"], meta[long_]["tokens"]
        if not st or not lt or len(st) > len(lt):
            return False
        align = _align_variant_tokens(short, long_)
        if not align:
            return False
        if not any(tok == comb for tok, _s, _e, _sim, comb in align):
            return False
        return True

    # E1/E2：专名簇
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            if find(a) == find(b) or _type_conflict(a, b):
                continue
            if (meta[a]["proper_src"] & meta[b]["proper_src"]) or _drift_equivalent(a, b):
                union(a, b)

    # E3：纯头衔行桥接（带修饰词的头衔键不参与 → 老公爵独立）
    for c in order:
        if not meta[c]["title_only"] or not meta[c]["plain_keys"]:
            continue
        cands = set()
        for other in order:
            if other == c or find(other) == find(c):
                continue
            if meta[other]["title_only"]:
                continue
            if meta[other]["plain_keys"] & meta[c]["plain_keys"]:
                cands.add(find(other))
        if len(cands) == 1:
            union(c, cands.pop())

    clusters: Dict[str, list] = {}
    for c in order:
        clusters.setdefault(find(c), []).append(c)

    # updated_at 列容错：旧测试 fixture / 老库可能没有该列
    try:
        has_updated = any(r[1] == "updated_at"
                          for r in conn.execute("PRAGMA table_info(entity_registry)"))
    except Exception:
        has_updated = False
    ts_sql = ", updated_at = CURRENT_TIMESTAMP" if has_updated else ""

    plan: Dict[str, Dict] = {}
    for root, members in clusters.items():
        if len(members) < 2:
            continue

        def _base_len(c):
            return len(_r134_strip_parens(c))

        def _key(c):
            m = meta[c]
            return (
                _CONF_ORDER.get(m["confidence"], 3),
                0 if "·" in c else 1,
                -_base_len(c),
                m["created"],
                c,
            )

        members_sorted = sorted(members, key=_key)
        winner = members_sorted[0]
        demoted = [c for c in members_sorted[1:]]
        if not demoted:
            continue

        evidence = []
        for d in demoted:
            if meta[d]["proper_src"] & meta[winner]["proper_src"]:
                evidence.append("shared_source:" + sorted(meta[d]["proper_src"] & meta[winner]["proper_src"])[0])
            elif meta[d]["title_only"]:
                evidence.append("title_bridge:" + ",".join(sorted(next(iter(meta[d]["plain_keys"]), ""))))
            else:
                evidence.append("name_drift")

        whitelist = list(demoted)
        replace_pairs = []
        for d in demoted:
            cls = classify_entity_variant(d, winner)
            if cls["policy"] == "replace" and cls["target"] != d:
                replace_pairs.append([d, cls["target"]])

        merged_srcs = list(dict.fromkeys(meta[winner]["srcs"] + [s for d in demoted for s in meta[d]["srcs"]]))
        merged_variants = list(dict.fromkeys([v for v in meta[winner]["variants"] if v] + whitelist))

        entry = {
            "winner": winner,
            "demoted": demoted,
            "whitelist": whitelist,
            "replace_pairs": replace_pairs,
            "evidence": evidence,
            "merged_source_names": merged_srcs,
            "merged_variants": merged_variants,
            "changed": 0,
        }

        if not dry_run:
            ch = 0
            if meta[winner]["pending"] != 0 or merged_srcs != meta[winner]["srcs"] \
                    or merged_variants != meta[winner]["variants"]:
                cursor.execute(
                    "UPDATE entity_registry SET pending_review = 0, review_reason = ?, "
                    "source_names = ?, variant_names = ?" + ts_sql + " WHERE id = ?",
                    (f"Round134 unify: winner，收编 {len(demoted)} 个同人别名",
                     json.dumps(merged_srcs, ensure_ascii=False),
                     json.dumps(merged_variants, ensure_ascii=False),
                     meta[winner]["id"]),
                )
                ch += 1
            for d in demoted:
                reason = f"Round134 unify: 同人别名并入 {winner}"
                if meta[d]["pending"] == 0:
                    cursor.execute(
                        "UPDATE entity_registry SET pending_review = 1, review_reason = ?"
                        + ts_sql + " WHERE id = ? AND pending_review = 0",
                        (reason, meta[d]["id"]),
                    )
                    ch += 1
                else:
                    cursor.execute(
                        "UPDATE entity_registry SET review_reason = ?" + ts_sql + " "
                        "WHERE id = ? AND (review_reason IS NULL OR review_reason != ?)",
                        (reason, meta[d]["id"], reason),
                    )
                    ch += cursor.rowcount
            entry["changed"] = ch

        plan[" / ".join(sorted(members))] = entry

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
