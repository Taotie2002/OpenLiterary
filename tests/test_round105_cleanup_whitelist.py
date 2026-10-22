"""Round 105 P3: entity_registry 变体清理脚本的白名单策略回归测试。

ch06 倒查：旧「黑名单 + 高频差异字」护栏双向都漏——既放过 帝国/法庭/法律/英镑/
女佣/托国/非私 等垃圾（差异字不高频），又可能误杀真音译变体。清理脚本改为
白名单：只保留 CONFIRMED_VARIANTS + 种子变体，其余「一字之差」候选一律移除，
裸警衔词（警监/警司/警督）因盲替换风险也移除。本测试锁定该行为防回归。
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _load_clean_module():
    path = _ROOT / "scripts" / "clean_entity_registry.py"
    spec = importlib.util.spec_from_file_location("clean_entity_registry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    yield conn
    conn.close()


def _insert(conn, canonical, variants):
    conn.execute(
        "INSERT INTO entity_registry (canonical_name, source_names, variant_names) VALUES (?, ?, ?)",
        (canonical, "[]", json.dumps(variants, ensure_ascii=False)),
    )


def _variants_of(conn, canonical):
    row = conn.execute(
        "SELECT variant_names FROM entity_registry WHERE canonical_name = ?", (canonical,)
    ).fetchone()
    return json.loads(row[0]) if row and row[0] else []


class TestWhitelistCleanup:
    def test_garbage_one_char_diff_removed(self, registry):
        mod = _load_clean_module()
        _insert(registry, "法国", ["外国", "帝国", "法庭", "法律", "托国", "美国", "出国"])
        _insert(registry, "楼梯", ["楼上", "楼去", "楼下"])
        _insert(registry, "女仆", ["女士", "女佣", "男仆", "女儿"])
        changes = mod.clean_common_word_variants(registry, dry_run=False)
        removed = {c: r for c, r in changes}
        assert set(removed["法国"]) == {"外国", "帝国", "法庭", "法律", "托国", "美国", "出国"}
        assert set(removed["楼梯"]) == {"楼上", "楼去", "楼下"}
        assert set(removed["女仆"]) == {"女士", "女佣", "男仆", "女儿"}
        assert _variants_of(registry, "法国") == []

    def test_confirmed_variants_kept(self, registry):
        mod = _load_clean_module()
        _insert(registry, "奥布洛维奇家族", ["奥博洛维奇家族"])
        _insert(registry, "保皇党", ["保皇派"])
        _insert(registry, "巴特尔警督", ["巴特尔警监"])
        assert mod.clean_common_word_variants(registry, dry_run=False) == []
        assert _variants_of(registry, "奥布洛维奇家族") == ["奥博洛维奇家族"]
        assert _variants_of(registry, "保皇党") == ["保皇派"]
        assert _variants_of(registry, "巴特尔警督") == ["巴特尔警监"]

    def test_seed_variants_kept(self, registry, monkeypatch):
        """P1-4: 种子变体以 config.entity_variants 为准（不再硬编码在通用代码里）。"""
        mod = _load_clean_module()
        from src import config as cfg_mod
        cfg = cfg_mod.get_config()
        monkeypatch.setitem(cfg._data, "entity_variants", {"爱丽丝": ["艾丽丝"]})
        _insert(registry, "爱丽丝", ["艾丽丝"])
        assert mod.clean_common_word_variants(registry, dry_run=False) == []
        assert _variants_of(registry, "爱丽丝") == ["艾丽丝"]

    def test_bare_rank_words_removed(self, registry):
        mod = _load_clean_module()
        _insert(registry, "巴特尔警督", ["警监", "警司", "警督", "巴特尔警司"])
        changes = mod.clean_common_word_variants(registry, dry_run=False)
        removed = {c: r for c, r in changes}
        assert set(removed["巴特尔警督"]) == {"警监", "警司", "警督", "巴特尔警司"}

    def test_idempotent(self, registry):
        mod = _load_clean_module()
        _insert(registry, "法国", ["外国", "帝国", "托国"])
        _insert(registry, "保皇党", ["保皇派"])
        mod.clean_common_word_variants(registry, dry_run=False)
        assert mod.clean_common_word_variants(registry, dry_run=False) == []

    def test_dry_run_does_not_write(self, registry):
        mod = _load_clean_module()
        _insert(registry, "法国", ["外国", "帝国"])
        mod.clean_common_word_variants(registry, dry_run=True)
        assert _variants_of(registry, "法国") == ["外国", "帝国"]
