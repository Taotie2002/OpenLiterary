"""Round 120 (L2): pun_audit — 核销/豁免/退出码/边界匹配。"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location("pun_audit", _ROOT / "scripts" / "pun_audit.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)


def _mk_decision_db(path: Path, rows):
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE decision_db (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "level INTEGER NOT NULL, source_key TEXT NOT NULL UNIQUE, translation TEXT NOT NULL, "
        "reason TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    for k, t in rows:
        con.execute(
            "INSERT INTO decision_db (level, source_key, translation, reason) VALUES (2,?,?,?)",
            (k, t, "test"))
    con.commit()
    con.close()


def test_anchor_hit_word_boundary():
    assert pa._anchor_hit("tale", "A long and sad tale!")
    assert not pa._anchor_hit("tale", "The tallest man")
    assert pa._anchor_hit("尾巴", "它的尾巴很长")


def test_audit_covered_uncovered_and_waiver():
    watchlist = [
        {"source": "tale/tail", "anchors": ["tale", "tail"], "strategy": "落点或加注"},
        {"source": "draw", "anchors": ["draw"], "strategy": "多义"},
    ]
    texts = [("merged", "他讲了一个长长的故事[^1]。\n\n[^1]: tale/tail 双关：原文谐音‘尾巴’。")]
    decisions = ["tale/tail [双关·保留笑点] 落点或加注 pun_watchlist"]
    result = pa.audit(watchlist, texts, decisions, set())
    s = result["summary"]
    assert s["total"] == 2
    assert s["covered"] == 1
    assert s["uncovered_nonwaived"] == 1
    item0 = result["items"][0]
    assert item0["decision_ok"] and item0["landed"]
    waived = pa.audit(watchlist, texts, decisions, {"draw"})
    assert waived["summary"]["uncovered_nonwaived"] == 0
    assert waived["summary"]["waived"] == 1


def test_decision_ok_via_src_anchors():
    watchlist = [{"source": "whiting (fish vs. shoe-blacking)", "src_anchors": ["whiting"],
                  "anchors": ["牙鳕", "白垩粉"], "strategy": "双关"}]
    texts = [("merged", "“牙鳕”这个词……")]
    decisions = ["whiting [双关·保留笑点] 建议查阅牙鳕"]
    r = pa.audit(watchlist, texts, decisions, set())
    item = r["items"][0]
    assert item["decision_ok"] and item["landed"] and item["covered"]


def test_dup_notes_across_chapter_finals_flagged():
    watchlist = [{"source": "tale/tail", "anchors": ["tale", "tail"], "strategy": "落点"}]
    note = "\n\n[^1]: tale/tail 双关：原文谐音‘尾巴’。"
    texts = [
        ("merged", "他讲了一个长长的故事[^1]。" + note),
        ("ch03/ch03_chunk000", "他讲了一个长长的故事[^1]。" + note),
        ("ch04/ch04_chunk000", "……另一个故事[^1]。" + note),
    ]
    decisions = ["tale/tail [双关·保留笑点] 落点 pun_watchlist"]
    r = pa.audit(watchlist, texts, decisions, set())
    item = r["items"][0]
    assert item["dup_notes"], "两章 final 同注应判重复"
    assert not item["covered"]
    # merged 与单章 final 同注（merge 拼接传播）不算重复
    r2 = pa.audit(watchlist, texts[:2], decisions, set())
    assert not r2["items"][0]["dup_notes"]
    assert r2["items"][0]["covered"]


def test_landed_via_decision_render_fragments():
    watchlist = [{
        "source": "I see what I eat / I eat what I see",
        "src_anchors": ["I see what I eat", "I eat what I see"],
        "anchors": ["所见即所食", "所食即所见"],
        "kind": "polysemy", "strategy": "正文保留逻辑诡辩 + 必要时尾注说明",
    }]
    texts = [("ch08/ch08_chunk001",
              "“你照样也可以说‘我见我吃的’就是‘我吃我见的’！”\n\n[^2]: 此处戏仿语言哲学诡辩。")]
    decisions = ["i see what i eat / i eat what i see 我见我吃 【来源】倒置游戏"]
    renders = [("i see what i eat / i eat what i see", "我吃我所见 / 我见我吃")]
    r = pa.audit(watchlist, texts, decisions, set(), decision_renders=renders)
    item = r["items"][0]
    assert item["landed"] and item["covered"], item
    assert "我见我吃" in item["render_anchors"]
    # 无决定译文时维持原严格判定（不迁就）
    r2 = pa.audit(watchlist, texts, decisions, set())
    assert not r2["items"][0]["landed"]


def test_main_exit_codes_and_json(tmp_path):
    root = tmp_path
    (root / "input").mkdir()
    (root / "output").mkdir()
    (root / "db").mkdir()
    (root / "input" / "book_profile.json").write_text(json.dumps({
        "genre": "童话",
        "pun_watchlist": [{"source": "tale/tail", "anchors": ["tale", "tail"],
                           "strategy": "双关落点"}],
    }, ensure_ascii=False), encoding="utf-8")
    (root / "output" / "translated_full.md").write_text(
        "他讲了一个长长的故事[^1]。\n\n[^1]: tale/tail 双关说明。\n", encoding="utf-8")
    _mk_decision_db(root / "db" / "decision_db.sqlite", [("tale", "[双关] 双关落点")])

    rc = pa.main(["--project-root", str(root)])
    assert rc == 0
    data = json.loads((root / "output" / "pun_audit.json").read_text(encoding="utf-8"))
    assert data["summary"]["covered"] == 1

    # 去掉决策 → 未核销 → 退出码 1（--no-fail 则 0）
    (root / "db" / "decision_db.sqlite").unlink()
    rc = pa.main(["--project-root", str(root)])
    assert rc == 1
    rc = pa.main(["--project-root", str(root), "--no-fail"])
    assert rc == 0
