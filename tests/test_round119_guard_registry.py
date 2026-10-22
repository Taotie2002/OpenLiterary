"""Round 119 (T8): 缺陷→守卫映射常驻化（方案附录 A）。

运营规则：新缺陷"先写复现测试 → 修复 → 入守卫"，不得口头修复。
本测守卫对每一类已知缺陷断言其可执行守卫仍存在（被删除/改名即红）：

  D1  转换丢失        → splitters.run_conversion_audit + gate ⑤ 审计检查
  D2/D3 DIRTY 无消费点 → pipeline.consolidate_dirty_chapters + gate ①
                        + _verify_final_outputs "陈旧待整合"
  D4  回溯词边界       → scheduler._source_term_pattern + 词边界测试文件
  D5 样式/元语/制式    → book_lint.normalize_final_style + gate ④ + prompt 禁元语
  D6/D7 页眉/标题      → book_lint._FRONTMATTER_RESIDUE_RES + chapter_heading
  D8  注释豁免         → consistency._in_quote_or_note + 豁免测试文件
  D9  悬空承诺         → book_lint._DANGLING_RE 用于 lint_book
  D10 backup 残留      → gate ② "残留 final_backup" 检查
  D11 诗体结构丢失     → verse manifest + verse_lint + gate ⑥（L1）
  D12 双关未核销/载体丢失 → pun_watchlist + scripts/pun_audit.py + gate ⑦（L2）
  D13 文学度量漂移     → scripts/literary_metrics.py + docs/baselines 基线（L6/P8）
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _gate_body() -> str:
    src = _read('src/pipeline.py')
    start = src.index('def run_final_gate(')
    return src[start:]


def test_guard_d1_conversion_audit_wired():
    assert 'def run_conversion_audit' in _read('src/splitters.py')
    assert '转换审计' in _gate_body()


def test_guard_d2d3_dirty_wired():
    assert 'def consolidate_dirty_chapters' in _read('src/pipeline.py')
    gate = _gate_body()
    assert 'DIRTY=' in gate and 'consolidate' in gate
    assert '陈旧待整合' in _read('src/pipeline.py')


def test_guard_d4_word_boundary_wired():
    assert 'def _source_term_pattern' in _read('src/core/scheduler.py')
    assert (ROOT / 'tests' / 'test_round119_word_boundary_backtrack.py').exists()


def test_guard_d5_style_wired():
    bl = _read('src/book_lint.py')
    assert '_META_TOKENS' in bl and 'def normalize_final_style' in bl
    assert 'lint 未通过' in _gate_body()
    for path in ('src/translator_agent.py', 'src/agents/rewriter_agent.py'):
        assert '不得出现任何翻译流程元语' in _read(path)


def test_guard_d6d7_frontmatter_heading_wired():
    bl = _read('src/book_lint.py')
    assert '_FRONTMATTER_RESIDUE_RES' in bl
    assert 'def chapter_heading' in bl and 'def postprocess_chunk_text' in bl


def test_guard_d8_note_exempt_wired():
    assert 'def _in_quote_or_note' in _read('src/consistency.py')
    assert (ROOT / 'tests' / 'test_round119_consistency_note_exempt.py').exists()


def test_guard_d9_dangling_wired():
    bl = _read('src/book_lint.py')
    assert '_DANGLING_RE' in bl
    start = bl.index('def lint_book(')
    assert '_DANGLING_RE' in bl[start:start + 4000], 'D9 检测须用于 lint_book 正文'


def test_guard_d10_backup_wired():
    assert '残留 final_backup' in _gate_body()


def test_round119_guard_test_files_present():
    expected = {
        'test_round119_converter_pre.py', 'test_round119_word_boundary_backtrack.py',
        'test_round119_consolidation.py', 'test_round119_consistency_note_exempt.py',
        'test_round119_mirror_parity.py', 'test_round119_quote_normalize.py',
        'test_round119_footnote_lint.py', 'test_round119_heading_frontmatter.py',
        'test_round119_final_gate.py',
    }
    missing = [f for f in expected if not (ROOT / 'tests' / f).exists()]
    assert not missing, f"守卫测试缺失: {missing}"


def test_guard_d11_verse_channel_wired():
    sp = _read('src/splitters.py')
    assert '_build_verse_manifest' in sp and 'verse_manifest' in sp
    assert 'def verse_lint' in _read('src/verse.py')
    gate = _gate_body()
    assert '⑥' in gate and 'verse_lint' in gate
    assert (ROOT / 'tests' / 'test_round119_verse.py').exists()


def test_guard_d12_pun_closure_wired():
    assert 'pun_watchlist' in _read('src/style_profiler.py')
    assert (ROOT / 'scripts' / 'pun_audit.py').exists()
    gate = _gate_body()
    assert '⑦' in gate and 'pun_audit' in gate
    for f in ('test_round120_pun_watchlist.py', 'test_round120_pun_audit.py',
              'test_round120_pun_gate.py'):
        assert (ROOT / 'tests' / f).exists(), f'守卫测试缺失: {f}'


def test_guard_d13_literary_metrics_baseline():
    assert (ROOT / 'scripts' / 'literary_metrics.py').exists()
    baseline = ROOT / 'docs' / 'baselines' / 'literary_metrics_2026-10-14.json'
    assert baseline.exists(), '文学度量基线缺失'
    data = json.loads(baseline.read_text(encoding='utf-8'))
    for key in ('ai_density_per_1k', 'translationese_total', 'ngram4'):
        assert key in data, f'基线缺字段: {key}'
