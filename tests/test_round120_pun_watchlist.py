"""Round 120 (L2): style_profiler pun_watchlist — 归一/兜底/透传。"""
from __future__ import annotations

from src import style_profiler as sp


def test_normalize_structured_and_derived_anchors():
    prof = sp.normalize_profile({
        "genre": "童话",
        "pun_watchlist": [
            {"source": "tale/tail", "kind": "homophone", "strategy": "尾巴/故事双关"},
            {"term": "draw", "note": "抽签/画画多义", "keywords": "draw, 抽签"},
            "game",
        ],
    })
    wl = prof["pun_watchlist"]
    assert len(wl) == 3
    assert wl[0]["source"] == "tale/tail"
    assert wl[0]["anchors"] == ["tale", "tail"]
    assert wl[1]["source"] == "draw"
    assert "draw" in wl[1]["anchors"] and "抽签" in wl[1]["anchors"]
    assert wl[2]["source"] == "game" and wl[2]["anchors"] == ["game"]


def test_hint_fallback_synthesis():
    prof = sp.normalize_profile({
        "genre": "童话",
        "translation_hints": [
            "tale/tail = 双关：尾巴与故事的谐音，需要落点或加注",
            "thimble = 缝纫用顶针",
        ],
    })
    wl = prof["pun_watchlist"]
    assert len(wl) == 1
    assert wl[0]["source"] == "tale/tail"
    assert wl[0]["kind"] == "hint"
    assert "tail" in wl[0]["src_anchors"]
    assert "尾巴" in wl[0]["anchors"]


def test_normalize_derives_src_anchors():
    prof = sp.normalize_profile({"pun_watchlist": [
        {"source": "whiting (fish vs. shoe-blacking)", "anchors": ["牙鳕", "白垩粉"]},
        {"source": "Tale/tail", "src_anchors": ["tale", "tail"], "anchors": ["尾巴", "故事"]},
    ]})
    wl = prof["pun_watchlist"]
    assert wl[0]["src_anchors"] == ["whiting"]
    assert wl[1]["src_anchors"] == ["tale", "tail"]


def test_hint_dedup_against_structured():
    prof = sp.normalize_profile({
        "translation_hints": ['"whiting" = 一词双关：①牙鳕 ②白垩粉，须保留双关'],
        "pun_watchlist": [{"source": "whiting (fish vs. shoe-blacking)",
                           "anchors": ["牙鳕", "白垩粉"]}],
    })
    assert len(prof["pun_watchlist"]) == 1
    assert prof["pun_watchlist"][0]["kind"] != "hint"


def test_hint_english_anchors_replaced_by_cn_strategy():
    prof = sp.normalize_profile({
        "pun_watchlist": [{
            "source": '"crumbs would all wash off in the sea"', "kind": "hint",
            "anchors": ['"crumbs would all wash off in the sea"'],
            "strategy": "与上文 whiting（白垩粉）双关呼应（白粉会被海水冲掉）",
        }],
    })
    wl = prof["pun_watchlist"]
    assert len(wl) == 1
    assert wl[0]["anchors"] == ["白垩粉"]


def test_split_anchors_dedup():
    assert sp._split_anchors("draw = 拉/画，抽签") == ["draw", "拉", "画", "抽签"]
    assert sp._split_anchors("") == []


def test_guide_passthrough_and_heuristic_default():
    prof = sp.normalize_profile({"pun_watchlist": [{"source": "pun"}]})
    guide = sp.profile_to_style_guide(prof)
    assert guide["pun_watchlist"][0]["source"] == "pun"
    hp = sp.heuristic_profile()
    assert hp["pun_watchlist"] == []


def test_summary_line_counts_pun():
    prof = sp.normalize_profile({"pun_watchlist": ["a/b"]})
    assert "pun=1" in sp.summary_line(prof)
