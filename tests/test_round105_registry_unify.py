"""Round 105 P2-2: 同人多规范名归一 + rewriter prompt 冲突优先级。

ch32 倒查根因：同一人物登记到多个 canonical（George Lomax → 3 个），运行时
missing 核查只认 pending_review=0 的一个，rewriter 用其它形式被判 missing →
硬失败死循环；且考据决策与登记表互斥，rewriter 收到矛盾指令。
"""
from __future__ import annotations

import json
import sqlite3

import pytest


@pytest.fixture
def registry():
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            confidence TEXT DEFAULT 'high',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pending_review INTEGER DEFAULT 0,
            review_reason TEXT
        );
    """)
    # George Lomax 同人多规范名（共享专名 Mr. Lomax / George）
    conn.execute("INSERT INTO entity_registry (canonical_name, source_names, confidence, created_at, pending_review) VALUES (?,?,?,?,?)",
                 ("洛马克斯先生", json.dumps(["Mr. Lomax", "George", "Lomax"]), "high", "2026-01-01", 0))
    conn.execute("INSERT INTO entity_registry (canonical_name, source_names, confidence, created_at, pending_review) VALUES (?,?,?,?,?)",
                 ("乔治·洛马克斯先生", json.dumps(["Mr. Lomax", "Mr. George Lomax", "George", "sir"]), "high", "2026-01-02", 0))
    # 不同人物（仅共享通用指称 the young man，不应合并）
    conn.execute("INSERT INTO entity_registry (canonical_name, source_names, confidence, created_at, pending_review) VALUES (?,?,?,?,?)",
                 ("安东尼·凯德", json.dumps(["the young man", "Anthony Cade"]), "high", "2026-01-01", 0))
    conn.execute("INSERT INTO entity_registry (canonical_name, source_names, confidence, created_at, pending_review) VALUES (?,?,?,?,?)",
                 ("巴黎的流浪艺人", json.dumps(["the young man"]), "high", "2026-01-01", 0))
    conn.commit()
    yield conn
    conn.close()


class TestUnifyEntityRegistry:
    def test_clusters_same_person_by_proper_name(self, registry):
        from src.core.decision_engine import unify_entity_registry
        plan = unify_entity_registry(registry, dry_run=True)
        # George Lomax 簇应被识别
        assert any("洛马克斯先生" in k and "乔治·洛马克斯先生" in k for k in plan), plan
        # 通用指称 the young man 不应把两个不同人物合并
        assert not any("安东尼·凯德" in k and "巴黎的流浪艺人" in k for k in plan), plan

    def test_winner_is_longer_base(self, registry):
        from src.core.decision_engine import unify_entity_registry
        plan = unify_entity_registry(registry, dry_run=True)
        for k, v in plan.items():
            if "洛马克斯" in k:
                assert v["winner"] == "乔治·洛马克斯先生", v
                assert v["demoted"] == ["洛马克斯先生"], v

    def test_dry_run_does_not_write(self, registry):
        from src.core.decision_engine import unify_entity_registry
        unify_entity_registry(registry, dry_run=True)
        cur = registry.execute(
            "SELECT pending_review FROM entity_registry WHERE canonical_name='洛马克斯先生'"
        )
        assert cur.fetchone()[0] == 0, "dry_run 不应写库"

    def test_apply_demotes_losers(self, registry):
        from src.core.decision_engine import unify_entity_registry
        unify_entity_registry(registry, dry_run=False)
        cur = registry.execute(
            "SELECT pending_review FROM entity_registry WHERE canonical_name='洛马克斯先生'"
        )
        assert cur.fetchone()[0] == 1, "apply 应降级 loser"
        cur = registry.execute(
            "SELECT pending_review FROM entity_registry WHERE canonical_name='乔治·洛马克斯先生'"
        )
        assert cur.fetchone()[0] == 0, "winner 应保持 pending_review=0"


class TestPromptPrioritization:
    @staticmethod
    def _rewriter():
        from src.translator_agent import LiteraryRewriterAgent as LRA
        p = LRA.__new__(LRA)
        p._few_shot_examples = ""
        return p

    def test_retry_prompt_has_priority_rule(self):
        """retry prompt 的实体登记表 section 应含冲突优先级声明。"""
        p = self._rewriter()
        prompt = p._build_retry_prompt(
            raw_translation="直译稿",
            prev_lit_text="上一版",
            decisions_context="考据: George Lomax -> 乔治·洛马克斯",
            style_guide="风格",
            reject_reason="实体一致性硬失败",
            critic_feedback=None,
            retry_count=1,
            entity_feedback="实体「乔治·洛马克斯先生」须使用主名",
            entity_registry={"George": "乔治·洛马克斯先生"},
        )
        assert "优先级规则" in prompt, "retry prompt 应含实体名冲突优先级声明"
        assert "以本登记表规范主名为准" in prompt

    def test_initial_prompt_has_priority_rule(self):
        p = self._rewriter()
        prompt = p._build_prompt(
            raw_translation="直译稿",
            decisions_context="考据: George Lomax -> 乔治·洛马克斯",
            style_guide="风格",
            entity_registry={"George": "乔治·洛马克斯先生"},
        )
        assert "优先级规则" in prompt, "首次润色 prompt 应含实体名冲突优先级声明"
