"""Round 134: 同人归一（winner + 白名单）与音译漂移全文统一 —— 通用机制测试。

覆盖：
  - E1/E2/E3 聚类：共享专名 / token 对齐 / 纯头衔桥接；带修饰词的旧头衔
    （the old Marquis 类）不得并入（老公爵场景的通用化）；
  - winner 规则：confidence → 含「·」→ base 长度 → created_at → 字典序；
  - 白名单回写：demoted 入 winner.variant_names，demoted pending_review=1；
  - classify_entity_variant：称谓类 allow / 音译漂移 replace（含 token_map）；
  - maps 应用：replace 统一 + allow 保留 + 分隔符复原；
  - 扫描候选（一字漂移）必须经 LLM 裁决才应用；无 LLM → 保守不替换；
  - _apply_name_unification_pass：dry-run 零写入（DB/final.json），apply 幂等。

全部 fixture 使用虚构人名，不写死任何作品专名。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest


def _hash(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def _mk_db(tmp_path: Path, rows):
    """rows: [(canonical, [sources], etype), ...] -> sqlite 文件（真实 schema 子集）"""
    db = tmp_path / "decision_db.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE entity_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,
            source_names TEXT NOT NULL,
            variant_names TEXT,
            entity_type TEXT,
            confidence TEXT DEFAULT 'high',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pending_review INTEGER DEFAULT 0,
            review_reason TEXT,
            updated_at TIMESTAMP
        );
    """)
    for canonical, sources, etype in rows:
        conn.execute(
            "INSERT INTO entity_registry (canonical_name, source_names, variant_names, "
            "entity_type, confidence, created_at, pending_review) VALUES (?,?,?,?,?,?,?)",
            (canonical, json.dumps(sources, ensure_ascii=False), None, etype,
             "high", "2026-01-01", 0),
        )
    conn.commit()
    return db, conn


# 聚类 fixture：侯爵（纯头衔）/ 佐恩侯爵（复合称谓）/ 佐恩·巴列（专名·形）/
# 佐恩·巴勒（一字漂移）/ 老侯爵（带修饰词的旧头衔 —— 不同人，不得并入）
CLUSTER_ROWS = [
    ("侯爵", ["the Marquis"], "person"),
    ("佐恩侯爵", ["the Marquis", "Marquis Zorn"], "person"),
    ("佐恩·巴列", ["Marquis Zorn", "Zorn Barley"], "person"),
    ("佐恩·巴勒", ["Marquis Zorn", "Zorn Barley"], "person"),
    ("老侯爵", ["the old Marquis"], "person"),
]


@pytest.fixture
def cluster_db(tmp_path):
    db, conn = _mk_db(tmp_path, CLUSTER_ROWS)
    yield db, conn
    conn.close()


def _fin(text, tmp_path: Path, name="smoke000_final.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"text": text, "metadata": {"chunk_id": "smoke000"}},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    return [{"chapter": "smoke", "file": name, "text": text, "path": str(p)}]


class TestUnifyGenericClusters:
    def test_cluster_merge_and_winner_rules(self, cluster_db):
        from src.core.decision_engine import unify_entity_registry
        _db, conn = cluster_db
        plan = unify_entity_registry(conn, dry_run=True)
        hit = [v for v in plan.values() if "佐恩·巴列" in (v["winner"], *v["demoted"])]
        assert len(hit) == 1, plan
        e = hit[0]
        assert e["winner"] == "佐恩·巴列", e            # 含「·」优先 + 字典序小者
        assert set(e["demoted"]) == {"佐恩侯爵", "佐恩·巴勒", "侯爵"}, e
        assert [p[0] for p in e["replace_pairs"]] == ["佐恩·巴勒"], e  # 音译漂移类

    def test_old_title_with_modifier_stays_independent(self, cluster_db):
        from src.core.decision_engine import unify_entity_registry
        _db, conn = cluster_db
        plan = unify_entity_registry(conn, dry_run=True)
        for members in plan.values():
            assert "老侯爵" not in (members["winner"], *members["demoted"]), plan

    def test_apply_writes_whitelist_and_demotes(self, cluster_db):
        from src.core.decision_engine import unify_entity_registry
        _db, conn = cluster_db
        unify_entity_registry(conn, dry_run=False)
        cur = conn.execute(
            "SELECT pending_review, variant_names, source_names FROM entity_registry "
            "WHERE canonical_name='佐恩·巴列'")
        pending, vns, srcs = cur.fetchone()
        assert pending == 0
        assert set(json.loads(vns)) == {"佐恩侯爵", "佐恩·巴勒", "侯爵"}
        assert "the Marquis" in json.loads(srcs)          # winner 收编成员 source_names
        for loser in ("侯爵", "佐恩侯爵", "佐恩·巴勒"):
            cur = conn.execute(
                "SELECT pending_review FROM entity_registry WHERE canonical_name=?", (loser,))
            assert cur.fetchone()[0] == 1, loser


class TestClassifyGeneric:
    def test_title_and_composite_forms_allow(self):
        from src.core.decision_engine import classify_entity_variant as cls
        assert cls("侯爵", "佐恩·巴列")["policy"] == "allow"
        assert cls("佐恩侯爵", "佐恩·巴列")["policy"] == "allow"
        assert cls("佐恩", "佐恩·巴列")["policy"] == "allow"      # 子串简称

    def test_align_drift_replace(self):
        from src.core.decision_engine import classify_entity_variant as cls
        r = cls("佐恩·巴勒", "佐恩·巴列")
        assert r["policy"] == "replace" and r["target"] == "佐恩·巴列", r
        assert ("巴勒", "巴列") in r["token_map"], r

    def test_common_run_split_replace(self):
        from src.core.decision_engine import classify_entity_variant as cls
        r = cls("巴莱顿家族", "巴勒顿家族")
        assert r["policy"] == "replace" and r["target"] == "巴勒顿家族", r


class TestMapsApplyGeneric:
    def test_apply_replaces_drift_keeps_whitelist(self, cluster_db, tmp_path):
        from src.core.decision_engine import (
            unify_entity_registry, build_name_unification_maps, apply_name_unification,
        )
        _db, conn = cluster_db
        unify_entity_registry(conn, dry_run=False)
        text = "佐恩·巴勒出场。佐恩侯爵与侯爵同来。佐恩巴列离开。老侯爵已逝。"
        maps = build_name_unification_maps(conn, text)
        new_text, st = apply_name_unification(text, maps)
        assert "佐恩·巴列出场" in new_text and "佐恩·巴勒" not in new_text
        assert "佐恩侯爵与侯爵同来" in new_text            # allow 类保留多样性
        assert "佐恩·巴列离开" in new_text                 # 分隔符复原
        assert "老侯爵已逝" in new_text
        assert st["sep_restore"] >= 1 and st["forms"] >= 1, st


class TestScanCandidates:
    @staticmethod
    def _cold_pair():
        from src.utils.entity_stopwords import COMMON_CHARS, COMMON_WORDS
        for c in "栎烽洵珂玦珀砺翀":
            cand = "洛" + c
            if c not in COMMON_CHARS and cand not in COMMON_WORDS:
                return cand
        pytest.skip("no cold char available")

    def test_one_char_drift_is_candidate_not_auto_token(self, tmp_path):
        from src.core.decision_engine import build_name_unification_maps
        cand = self._cold_pair()
        _db, conn = _mk_db(tmp_path, [("洛琛·薇尔", ["Luochen Weier"], "person")])
        text = f"洛琛·薇尔说道。「{cand}来了。」{cand}点头。{cand}离开。"
        maps = build_name_unification_maps(conn, text)
        assert cand in maps["token_candidates"], maps["token_candidates"]
        assert maps["token_candidates"][cand]["to"] == "洛琛"
        assert cand not in maps["tokens"], "未经 LLM 裁决不得直接进入 tokens"
        conn.close()

    def test_vetting_skips_without_llm(self, tmp_path):
        from src.consistency import _vet_name_drift_candidates
        cand = self._cold_pair()
        cache = tmp_path / "_cache.json"
        approved = _vet_name_drift_candidates(
            {cand: {"to": "洛琛", "count": 3, "contexts": [], "entity": "洛琛·薇尔",
                    "sources": ["Luochen Weier"]}},
            {"llm_vetting": False}, cache,
        )
        assert approved == {}, "LLM 关闭时必须保守不替换"

    def test_vetted_candidate_applies(self, tmp_path):
        from src.core.decision_engine import build_name_unification_maps, apply_name_unification
        cand = self._cold_pair()
        _db, conn = _mk_db(tmp_path, [("洛琛·薇尔", ["Luochen Weier"], "person")])
        text = f"洛琛·薇尔说道。「{cand}来了。」{cand}点头。{cand}离开。"
        maps = build_name_unification_maps(conn, text)
        maps["tokens"].update({cand: "洛琛"})      # 模拟 LLM 裁决通过
        new_text, st = apply_name_unification(text, maps)
        assert cand not in new_text and new_text.count("洛琛") >= 4
        assert st["tokens"] == 3, st
        conn.close()


class TestDryRunAndApplyPass:
    def test_dry_run_zero_write(self, cluster_db, tmp_path):
        from src.consistency import _apply_name_unification_pass
        db, conn = cluster_db
        conn.close()  # pass 自行打开；先确保 fixture 连接不锁库
        db_hash = _hash(db)
        finals = _fin("佐恩·巴勒出场。佐恩巴列离开。", tmp_path)
        fin_hash = _hash(finals[0]["path"])
        out = _apply_name_unification_pass(
            finals, str(db), report_path=tmp_path / "rep.md",
            cache_path=tmp_path / "cache.json", scope_label="dry", dry_run=True,
        )
        assert _hash(db) == db_hash, "dry-run 写了真实 DB"
        assert _hash(finals[0]["path"]) == fin_hash, "dry-run 改了 final.json"
        assert out["files_pending"] >= 1 and out["files"] == 0
        rep = (tmp_path / "rep.md").read_text(encoding="utf-8")
        assert "DRY RUN" in rep and "待写入文件" in rep

    def test_apply_changes_and_is_idempotent(self, cluster_db, tmp_path, monkeypatch):
        import src.consistency as C
        db, conn = cluster_db
        conn.close()
        monkeypatch.setattr(C, "_load_name_unification_cfg",
                            lambda: {"enabled": True, "llm_vetting": False,
                                     "preferred_forms": {}})
        finals = _fin("佐恩·巴勒出场。佐恩巴列离开。", tmp_path)
        out1 = C._apply_name_unification_pass(
            finals, str(db), report_path=tmp_path / "rep.md",
            cache_path=tmp_path / "cache.json", scope_label="apply", dry_run=False,
        )
        text1 = json.loads(Path(finals[0]["path"]).read_text(encoding="utf-8"))["text"]
        assert "佐恩·巴列出场" in text1 and "佐恩巴列" not in text1
        assert out1["files"] == 1
        out2 = C._apply_name_unification_pass(
            finals, str(db), report_path=tmp_path / "rep.md",
            cache_path=tmp_path / "cache.json", scope_label="apply2", dry_run=False,
        )
        total2 = out2["sep_restore"] + out2["forms"] + out2["tokens"]
        assert total2 == 0 and out2["files"] == 0, "二次应用必须幂等"
