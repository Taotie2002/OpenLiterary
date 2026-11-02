"""Round 137: 非人名（术语/组织/专名）一致性扩展。

用户规则（Dune ch01 实测触发）：已经识别为同一实体的——不论人物还是组织/术语/
地名——其书写漂移（二次音译/错字）应全文统一；但语义别名（零字面交集：官邸 vs
总督府）与句子片段（男爵问道 / 哈克南人 / 宇航公会的飞船）绝不自动替换。

实现三条腿：
  1. 决策桥接扩展：etype 门控放开到全部实体类型，非人名加「逐分量相似度 ≥0.6」
     门槛——拦 Guildsman→公会成员 vs 宇航公会 这类同源不同指称；放
     阿特雷迪斯家族 → 阿特雷德斯家族 这类真漂移。
  2. 长公共段漂移扫描：对全部实体类型的规范名 token，在「名分隔符已折叠」视图里
     找共享 ≥2 字连续片段、整词书写不同的专名形（戈姆·贾巴尔 vs 戈姆刺 /
     阿特雷迪斯 vs 阿特雷德斯），带窗口边界/功能字/整锚点延伸等护栏；候选一律
     经 LLM 批量裁决后才应用（宁缺勿滥）。
  3. 分隔符候选走 forms 通道：裁决通过的含分隔符形（戈姆·贾巴尔）按分隔符不敏感
     正则替换，兼容同名的无分隔写法。
"""
from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import pytest


MIRROR = [
    ("monolith", "src.translator_agent"),
    ("modular", "src.core.decision_engine"),
]
CONSISTENCY_MIRROR = [
    ("monolith", "src.translator_agent"),
    ("modular", "src.consistency"),
]


@pytest.fixture(params=MIRROR, ids=[m[0] for m in MIRROR])
def de_mod(request):
    return importlib.import_module(request.param[1])


@pytest.fixture(params=CONSISTENCY_MIRROR, ids=[m[0] for m in CONSISTENCY_MIRROR])
def cons_mod(request):
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


def _fin(text, tmp_path, name="ch99_chunk007_final.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"text": text, "metadata": {}}, ensure_ascii=False),
                    encoding="utf-8")
    return [{"chapter": "ch99", "file": name, "text": text, "path": str(path)}]


class TestBridgeNonPersonExtended:
    def test_org_bridge_good_pair_alignment_applies(self, de_mod, tmp_path):
        conn = _mk_db(
            tmp_path,
            entities=[("阿特雷德斯家族", ["Atreides family"], None, "organization", 0)],
            decisions=[("Atreides family", "阿特雷迪斯家族")])
        text = "阿特雷迪斯家族声名远扬。阿特雷德斯家族在此集结。"
        maps = de_mod.build_name_unification_maps(conn, text)
        assert any(f[0] == "阿特雷迪斯家族" and f[1] == "阿特雷德斯家族"
                   for f in maps["forms"]), maps["forms"]
        new_text, st = de_mod.apply_name_unification(text, maps)
        assert "阿特雷迪斯家族" not in new_text
        assert new_text.count("阿特雷德斯家族") == 2
        assert st["forms"] >= 1

    def test_org_bridge_blocks_member_vs_org(self, de_mod, tmp_path):
        # Guildsman=成员 vs 宇航公会=组织：同源但指称不同（逐分量 sim 0.5 <0.6）。
        conn = _mk_db(
            tmp_path,
            entities=[("宇航公会", ["Guildsman"], None, "organization", 0)],
            decisions=[("Guildsman", "公会成员")])
        maps = de_mod.build_name_unification_maps(conn, "公会成员来了。宇航公会派人来。")
        assert not any(f[0] == "公会成员" for f in maps["forms"])
        assert "公会成员" not in maps["tokens"]
        assert any("公会成员" in x and "宇航公会" in x
                   for x in maps.get("semantic_pending") or [])

    def test_zero_overlap_semantic_alias_noted_not_replaced(self, de_mod, tmp_path):
        # Residency→官邸（决策）vs winner 总督府：零字面交集 → 不自动替换，仅记录。
        conn = _mk_db(
            tmp_path,
            entities=[("总督府", ["the Residency"], None, "place", 0)],
            decisions=[("Residency", "官邸")])
        maps = de_mod.build_name_unification_maps(conn, "官邸的大门紧闭。总督府灯火通明。")
        assert not any(f[0] == "官邸" for f in maps["forms"])
        assert "官邸" not in maps["tokens"]
        assert any("官邸" in x and "总督府" in x
                   for x in maps.get("semantic_pending") or [])


class TestLongRunScan:
    def test_gom_jabbar_candidate(self, de_mod, tmp_path):
        conn = _mk_db(
            tmp_path,
            entities=[("戈姆刺", ["gom jabbar"], None, "other", 0)],
            decisions=[("gom jabbar", "戈姆刺")])
        text = ("她取出戈姆·贾巴尔碰触了他，戈姆·贾巴尔就会要命。"
                "这就是戈姆·贾巴尔。戈姆刺是检定工具。")
        maps = de_mod.build_name_unification_maps(conn, text)
        cands = maps["token_candidates"]
        assert sorted(cands) == ["戈姆·贾巴尔"], cands
        info = cands["戈姆·贾巴尔"]
        assert info["to"] == "戈姆刺"
        assert info["count"] == 3
        assert info["form"] == "戈姆·贾巴尔"

    def test_atreides_single_occurrence_candidate(self, de_mod, tmp_path):
        conn = _mk_db(
            tmp_path,
            entities=[("保罗·阿特雷德斯", ["Paul"], None, "person", 0)],
            decisions=[])
        text = "他看见了红色的阿特雷迪斯鹰徽。保罗·阿特雷德斯握紧了拳头。"
        maps = de_mod.build_name_unification_maps(conn, text)
        cands = maps["token_candidates"]
        assert "阿特雷迪斯" in cands, cands
        assert cands["阿特雷迪斯"]["to"] == "阿特雷德斯"
        assert cands["阿特雷迪斯"]["count"] == 1

    def test_phrase_extensions_not_candidates(self, de_mod, tmp_path):
        # 「锚点 + 功能字/谓语延伸」= 句子片段：宇航公会的飞船 / 莫希阿姆说道 /
        # 哈克南男爵点了（R0 整锚点延伸）与 男爵问道（R2 头衔+谓语）一律剔除。
        conn = _mk_db(
            tmp_path,
            entities=[("宇航公会", ["Guild"], None, "organization", 0),
                      ("盖乌斯·海伦·莫希阿姆", ["Gaius"], None, "person", 0),
                      ("弗拉基米尔·哈克南男爵", ["Baron"], None, "person", 0)],
            decisions=[])
        text = ("宇航公会的飞船起航了。宇航公会的用法很奇怪。"
                "莫希阿姆说道：“我拒绝。”莫希阿姆说道：“我拒绝。”"
                "哈克南男爵点了点头。男爵问道：“谁？”男爵问道：“谁？”")
        maps = de_mod.build_name_unification_maps(conn, text)
        assert maps["token_candidates"] == {}, maps["token_candidates"]

    def test_known_form_extension_not_candidate(self, de_mod, tmp_path):
        # R1: 窗口内包含已登记名形 + 延伸（哈克南人 vs 哈克南）→ 句子片段剔除。
        conn = _mk_db(
            tmp_path,
            entities=[("菲德-罗萨·哈克南", ["Feyd"], None, "person", 0),
                      ("弗拉基米尔·哈克南男爵", ["Baron"], None, "person", 0)],
            decisions=[])
        maps = de_mod.build_name_unification_maps(
            conn, "哈克南人越过沙丘。哈克南人苦恼。弗拉基米尔·哈克南男爵沉默。")
        assert maps["token_candidates"] == {}, maps["token_candidates"]

    def test_count_and_run_threshold(self, de_mod, tmp_path):
        # 单次出现且公共段仅 2 字（run≥3 才放行单次）→ 不产生候选。
        conn = _mk_db(
            tmp_path,
            entities=[("戈姆刺", ["gom jabbar"], None, "other", 0)],
            decisions=[])
        maps = de_mod.build_name_unification_maps(
            conn, "戈姆刺是工具。这就是戈姆·贾巴尔。")
        assert maps["token_candidates"] == {}


class TestSepFormRouting:
    def test_approved_sep_candidate_applies_and_is_idempotent(
            self, cons_mod, tmp_path, monkeypatch):
        conn = _mk_db(
            tmp_path,
            entities=[("戈姆刺", ["gom jabbar"], None, "other", 0)],
            decisions=[("gom jabbar", "戈姆刺")])
        conn.close()
        monkeypatch.setattr(cons_mod, "_load_name_unification_cfg",
                            lambda: {"enabled": True, "llm_vetting": True,
                                     "preferred_forms": {}})
        cache = tmp_path / "cache.json"
        cache.write_text(json.dumps({"戈姆·贾巴尔->戈姆刺": True}, ensure_ascii=False),
                         encoding="utf-8")
        text = ("她取出戈姆·贾巴尔碰触了他，戈姆·贾巴尔就会要命。"
                "这就是戈姆·贾巴尔。另外戈姆贾巴尔也一样。戈姆刺是检定工具。")
        finals = _fin(text, tmp_path)
        out = cons_mod._apply_name_unification_pass(
            finals, str(tmp_path / "dec.sqlite"),
            report_path=tmp_path / "rep.md", cache_path=cache,
            scope_label="r137", dry_run=False)
        new_text = json.loads(Path(finals[0]["path"]).read_text(encoding="utf-8"))["text"]
        assert "戈姆·贾巴尔" not in new_text
        assert "戈姆贾巴尔" not in new_text, "分隔符不敏感替换须覆盖无分隔写法"
        assert new_text.count("戈姆刺") == 5  # 原有 1 处 + 替换 4 处
        assert any("戈姆·贾巴尔" in x for x in out["approved"])
        rep = (tmp_path / "rep.md").read_text(encoding="utf-8")
        assert "戈姆·贾巴尔" in rep
        out2 = cons_mod._apply_name_unification_pass(
            finals, str(tmp_path / "dec.sqlite"),
            report_path=tmp_path / "rep2.md", cache_path=cache,
            scope_label="r137b", dry_run=False)
        assert out2["sep_restore"] + out2["forms"] + out2["tokens"] == 0, "二次应用必须幂等"
