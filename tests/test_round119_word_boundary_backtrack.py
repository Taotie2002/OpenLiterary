"""Round 119 (T3): 回溯触发精度 — find_chunks_containing 词边界精筛.

背景（D4）: SQL LIKE 粗筛按任意子串命中 → 短词/连字符词触发大量假阳性
回溯（"well" 命中 "dwelling"/"farewell" 等），97 个 DIRTY 中相当部分是
子串假阳性。T3 两级化：LIKE 粗筛 → Python 词边界精筛。

口径（与方案 T3 一致）:
  - 词首尾仅以字母为边界: "well" 不再命中 "dwelling"；数字/CJK 词
    不设边界（保护 "1984"、"乌龟"）。
  - 连字符词变形容差: "tea-party" 同时命中 "tea party"（反之亦然）。
  - 大小写不敏感（源文大小写与决策条目常不一致）。
  - limit 在精筛之后截断（避免粗筛假阳性吃掉名额）。
"""
from __future__ import annotations

import pytest


def _mk_scheduler(tmp_path):
    from src.core.scheduler import TaskScheduler
    s = TaskScheduler(str(tmp_path / "wf.db"))
    rows = [
        ("ch01_chunk000", "ch01", "She fell down the well and wondered.", "COMPLETED"),
        ("ch01_chunk001", "ch01", "The dwelling was old; he said farewell.", "COMPLETED"),
        ("ch01_chunk002", "ch01", "We had a tea party at the tea-party room.", "COMPLETED"),
        ("ch02_chunk000", "ch02", "A swelling crowd arrived.", "COMPLETED"),
        ("ch02_chunk001", "ch02", "乌龟慢慢爬。And 1984 came.", "COMPLETED"),
        ("ch02_chunk002", "ch02", "abc1984def", "COMPLETED"),
        ("ch02_chunk003", "ch02", "The Well was deep.", "COMPLETED"),
        ("ch03_chunk000", "ch03", "that would be four thousand\nmiles down, I think", "COMPLETED"),
    ]
    for cid, ch, txt, st in rows:
        s.conn.execute(
            "INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state) "
            "VALUES (?,?,?,?)", (cid, ch, txt, st))
    s.conn.commit()
    return s


def test_word_boundary_rejects_substring_hits(tmp_path):
    s = _mk_scheduler(tmp_path)
    got = s.find_chunks_containing("well")
    # dwelling / farewell / swelling 均不命中；小写首词容忍句首大写（Well）
    assert got == ["ch01_chunk000", "ch02_chunk003"]


def test_hyphen_variant_tolerance(tmp_path):
    s = _mk_scheduler(tmp_path)
    assert s.find_chunks_containing("tea-party") == ["ch01_chunk002"]
    assert s.find_chunks_containing("tea party") == ["ch01_chunk002"]


def test_case_sensitive_like_source_quotes(tmp_path):
    s = _mk_scheduler(tmp_path)
    # 大写首词严格大小写（"Time" 不应命中任意小写 time）
    assert s.find_chunks_containing("Well") == ["ch02_chunk003"]
    assert s.find_chunks_containing("The Well") == ["ch02_chunk003"]
    # 多词短语仅首词容忍句首大写（句首大写只作用于首词）
    assert s.find_chunks_containing("the well") == ["ch01_chunk000"]


def test_wrapped_phrase_found_via_normalized_coarse(tmp_path):
    s = _mk_scheduler(tmp_path)
    # 原文跨行折行（"four thousand\nmiles down"）：原样 LIKE 命中不了，
    # 空白归一粗筛 + 词边界精筛应命中
    assert s.find_chunks_containing("four thousand miles down") == ["ch03_chunk000"]


def test_digit_term_letters_boundary(tmp_path):
    s = _mk_scheduler(tmp_path)
    # "abc1984def" 中的 1984 不算命中（字母邻接）；独立 "1984" 命中
    assert s.find_chunks_containing("1984") == ["ch02_chunk001"]


def test_cjk_term_matches(tmp_path):
    s = _mk_scheduler(tmp_path)
    assert s.find_chunks_containing("乌龟") == ["ch02_chunk001"]


def test_limit_applied_after_precision_filter(tmp_path):
    s = _mk_scheduler(tmp_path)
    got = s.find_chunks_containing("well", limit=1)
    assert got == ["ch01_chunk000"]


def test_chapter_scope_still_applies(tmp_path):
    s = _mk_scheduler(tmp_path)
    assert s.find_chunks_containing("Well", chapter_id="ch02") == ["ch02_chunk003"]
    assert s.find_chunks_containing("well", chapter_id="ch01") == ["ch01_chunk000"]


def test_empty_source_returns_empty(tmp_path):
    s = _mk_scheduler(tmp_path)
    assert s.find_chunks_containing("") == []
    assert s.find_chunks_containing("   ") == []
