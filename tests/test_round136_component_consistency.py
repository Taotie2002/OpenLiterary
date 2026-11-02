"""Round 136: 同一人物·人名分量一致性（用字统一 + 粒度保留）。

用户规则（Dune ch01 实测触发）：已经识别为一个人的，其人名的同一部分
（first name / last name）用字必须一致；但粒度各自保留——原文全名→译文
全名，原文简称→译文简称，互不展开/收缩。

实现两条腿：
  1. classify_entity_variant ④「单 token 分量漂移（长度唯一匹配）」：
     零字面交集的别名按分量长度唯一对齐（皮特 vs 彼得·德·弗里斯 → 彼得）；
     附语境指称护栏（孩子/少主/老巫婆/令尊/封皮 类描述词绝不替换）。
  2. build_name_unification_maps「决策桥接」：winner.source_names 中某英文
     指称 s 的当前决策译文 t（decision_db: s→t）未登记为任何实体形式时，
     若 t 与 winner 分量对齐命中 replace，则把正文别名 t 统一为对齐结果。
     桥接只信「专名级」来源（person 实体 + 非关系指称 + 英文专名证据），
     例：Piter→皮特 且 winner=彼得·德·弗里斯（source 含 Piter）→
     正文「皮特」→「彼得」（简称仍是简称）。
"""
from __future__ import annotations

import importlib
import json
import sqlite3

import pytest


MIRROR = [
    ("monolith", "src.translator_agent"),
    ("modular", "src.core.decision_engine"),
]


@pytest.fixture(params=MIRROR, ids=[m[0] for m in MIRROR])
def de_mod(request):
    return importlib.import_module(request.param[1])


def _mk_db(tmp_path, entities, decisions):
    """entities: [(canonical, sources, variants, etype, pending)]
    decisions: [(source_key, translation)]"""
    db = tmp_path / "dec.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE entity_registry (id INTEGER PRIMARY KEY, canonical_name TEXT,"
        " source_names TEXT, variant_names TEXT, entity_type TEXT,"
        " confidence TEXT, created_at TEXT, pending_review INTEGER)")
    for i, (canon, srcs, vns, etype, pending) in enumerate(entities, 1):
        conn.execute(
            "INSERT INTO entity_registry VALUES (?,?,?,?,?,?,?,?)",
            (i, canon, json.dumps(srcs, ensure_ascii=False),
             json.dumps(vns, ensure_ascii=False) if vns else None,
             etype, "high", "", pending))
    conn.execute(
        "CREATE TABLE decision_db (id INTEGER PRIMARY KEY, level INTEGER,"
        " source_key TEXT, translation TEXT)")
    for i, (sk, tr) in enumerate(decisions, 1):
        conn.execute("INSERT INTO decision_db VALUES (?,?,?,?)", (i, 2, sk, tr))
    conn.commit()
    return conn


class TestClassifyComponentDrift:
    def test_zero_overlap_short_vs_full(self, de_mod):
        cls = de_mod.classify_entity_variant("皮特", "彼得·德·弗里斯")
        assert cls["policy"] == "replace"
        assert cls["target"] == "彼得", "简称统一到对应分量，不得展开为全名"
        assert ("皮特", "彼得") in cls["token_map"]

    def test_reverse_direction_follows_winner(self, de_mod):
        cls = de_mod.classify_entity_variant("彼得", "皮特·德·弗里斯")
        assert cls["policy"] == "replace"
        assert cls["target"] == "皮特"

    def test_multi_token_zero_overlap_keeps_full_extent(self, de_mod):
        cls = de_mod.classify_entity_variant("皮特·德·弗里斯", "彼得·德·弗里斯")
        assert cls["policy"] == "replace"
        assert cls["target"] == "彼得·德·弗里斯", "全名保持全名"

    @pytest.mark.parametrize("v", ["孩子", "少主", "小子", "老巫婆", "令尊"])
    def test_context_reference_words_protected(self, de_mod, v):
        cls = de_mod.classify_entity_variant(v, "保罗·阿特雷德斯")
        assert cls["policy"] == "allow", f"{v} 是语境指称，不得参与分量替换"

    def test_single_token_winner_shape_protected(self, de_mod):
        # 真实数据形状：物体/单 token 名（弹簧锁扣封面）不触发分量路径
        cls = de_mod.classify_entity_variant("封皮", "弹簧锁扣封面")
        assert cls["policy"] == "allow"

    def test_titles_and_substring_shorts_still_allowed(self, de_mod):
        for v, w in [("公爵", "雷托·阿特雷德斯"), ("圣母", "盖乌斯·海伦·莫希阿姆"),
                     ("保罗", "保罗·阿特雷德斯"), ("杰西卡", "杰西卡夫人")]:
            assert de_mod.classify_entity_variant(v, w)["policy"] == "allow"

    def test_length_ambiguity_conservative(self, de_mod):
        # 零字面交集 + 两个同长候选 → 歧义 → 保守放行
        cls = de_mod.classify_entity_variant("卡萝", "海伦·丽兹·约翰")
        assert cls["policy"] == "allow"


class TestDecisionBridge:
    def _setup(self, de_mod, tmp_path, text):
        conn = _mk_db(
            tmp_path,
            entities=[("彼得·德·弗里斯", ["Piter", "the man"],
                       None, "person", 0)],
            decisions=[("Piter", "皮特")])
        maps = de_mod.build_name_unification_maps(conn, text)
        return maps

    def test_bridge_replaces_only_present_forms(self, de_mod, tmp_path):
        text = "皮特说。“皮特！”彼得·德·弗里斯笑了笑。"
        maps = self._setup(de_mod, tmp_path, text)
        assert ["皮特", "彼得", "彼得·德·弗里斯"] in maps["forms"]
        new_text, st = de_mod.apply_name_unification(text, maps)
        assert "皮特" not in new_text
        assert "彼得·德·弗里斯" in new_text, "全名不得被收缩"
        assert "彼得说" in new_text or "彼得" in new_text, "简称替换为简称"
        assert st["forms"] >= 2

    def test_bridge_requires_text_presence(self, de_mod, tmp_path):
        maps = self._setup(de_mod, tmp_path, "完全无关的正文。")
        assert not any(f[0] == "皮特" for f in maps["forms"])

    def test_bridge_non_person_low_sim_blocked(self, de_mod, tmp_path):
        # Round 137: 桥接已扩展至非人名实体，但「逐分量相似度 ≥0.6」门槛必须拦住
        # 同源不同指称——Guildsman=成员 vs 宇航公会=组织（整体相似度 0.5）。
        conn = _mk_db(
            tmp_path,
            entities=[("宇航公会", ["Guildsman"], None, "organization", 0)],
            decisions=[("Guildsman", "公会成员")])
        maps = de_mod.build_name_unification_maps(conn, "公会成员来了。")
        assert not any(f[0] == "公会成员" for f in maps["forms"])
        assert "公会成员" not in maps["tokens"]

    def test_bridge_skips_relational_source(self, de_mod, tmp_path):
        conn = _mk_db(
            tmp_path,
            entities=[("雷托·阿特雷德斯", ["your father"], None, "person", 0)],
            decisions=[("your father", "令尊")])
        maps = de_mod.build_name_unification_maps(conn, "令尊说道。")
        assert not any(f[0] == "令尊" for f in maps["forms"])

    def test_bridge_skips_no_proper_evidence_source(self, de_mod, tmp_path):
        # 'lad' 无专名证据（纯泛称）→ 不桥接（其译文 孩子 是语境指称）
        conn = _mk_db(
            tmp_path,
            entities=[("保罗·阿特雷德斯", ["lad"], None, "person", 0)],
            decisions=[("lad", "孩子")])
        maps = de_mod.build_name_unification_maps(conn, "那 孩子 跑开了。")
        assert not any(f[0] == "孩子" for f in maps["forms"])

    def test_bridge_skips_registered_forms(self, de_mod, tmp_path):
        # 决策译文已是其它实体 canonical → 不得窃取
        conn = _mk_db(
            tmp_path,
            entities=[("彼得·德·弗里斯", ["Piter"], None, "person", 0),
                      ("皮特", ["Piter the man"], None, "person", 0)],
            decisions=[("Piter", "皮特")])
        maps = de_mod.build_name_unification_maps(conn, "皮特来了。")
        assert not any(f[0] == "皮特" for f in maps["forms"])
