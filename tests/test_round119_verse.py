"""Round 119 (L1): 韵文专用通道 — verse_manifest / chunk 标签 / prompt 注入 / verse_lint.

背景（Alice 全量普查）:
  - 7 首诗/歌中 6 首散文化（<p class="poem"> 内 <br/> 被并成散文行）；
    Fury 诗整首丢失（T1 已修）；"闪烁，闪烁" 式译文生硬。
  - 方案 L1: 转换层 verse_manifest.json → init 打 chunk 标签
    （db/chunk_tags.json 旁路）→ prompt 注入（literal/rewrite/judge）→
    verse_lint（PASS 前失败 → 一次定向重写；gate ⑥ 全书核验）。

本测试覆盖（散装实现；单体镜像由 test_round119_mirror_parity.py 守护）:
  1. 转换层: p.poem/<br/> 行结构保留、heuristic 分类、manifest 行号/偏移定位。
  2. verse.py: 交集打标、包含兜底、协议块文本、verse_lint 通过/失败/复沓。
  3. 管线 glue: _verse_note_for / _verse_lint_gate / verse_retry 一次性限额。
  4. Agents: rewriter / judge prompt 注入 verse_note 的有无对照。
  5. 真实 EPUB 全量: manifest 覆盖已知韵文（Fury 诗/蝙蝠歌/美汤歌）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_DIR = Path('/tmp/alice_epub/OEBPS')
BOOK_EPUB = Path('/home/zyu/books/Alices_Adventures_in_Wonderland.epub')
_need_fixtures = pytest.mark.skipif(
    not FIXTURE_DIR.exists(), reason='真实 EPUB 样本缺失: /tmp/alice_epub')


def _fixture(name: str) -> str:
    return (FIXTURE_DIR / f'7126951391209738045_{name}.htm.xhtml').read_text(encoding='utf-8')


# ── 1. 转换层 ──────────────────────────────────────────────────────

def test_heuristic_short_br_lines_become_verse():
    from src.splitters import _epub_clean_html_text_ex
    html = ('<html><body><p>prose one.</p>'
            '<p>line one<br/>line two<br/>line three</p>'
            '<p>This is a much longer prose paragraph with <br/> one single break and '
            'continuing text long enough that it should not look like verse at all.</p>'
            '</body></html>')
    vout = []
    text = _epub_clean_html_text_ex(html, vout)
    assert 'line one\nline two\nline three' in text
    assert [b['tag'] for b in vout] == ['heuristic']
    assert vout[0]['text'].split('\n') == ['line one', 'line two', 'line three']
    # 长行 br 不触发启发式：保持旧行为（并成一行）
    assert 'one single break and continuing text' in text


def test_single_line_poem_not_recorded():
    from src.splitters import _epub_clean_html_text_ex
    html = '<html><body><p class="poem">A single line poem.</p></body></html>'
    vout = []
    text = _epub_clean_html_text_ex(html, vout)
    assert 'A single line poem.' in text
    assert vout == []  # 单行块无行结构可 lint，不记录


def test_pre_recorded_as_pre_tag():
    from src.splitters import _epub_clean_html_text_ex
    html = '<html><body><pre>line A\nline B\nline C</pre></body></html>'
    vout = []
    _epub_clean_html_text_ex(html, vout)
    assert [b['tag'] for b in vout] == ['pre']


@_need_fixtures
def test_real_poem_br_lines_preserved():
    from src.splitters import _epub_clean_html_text_ex
    vout = []
    _epub_clean_html_text_ex(_fixture('11-h-7'), vout)
    joined = '\n'.join(b['text'] for b in vout)
    assert 'Twinkle, twinkle, little bat!' in joined
    assert 'little bat!\n' in joined, 'br 分行未保留（被并成散文行）'
    soup_out = []
    _epub_clean_html_text_ex(_fixture('11-h-10'), soup_out)
    assert any('Beau—ootiful Soo—oop!' in b['text'] for b in soup_out), '美汤歌未识别'
    assert any(b['text'].count('\n') >= 5 for b in soup_out)


def test_build_verse_manifest_line_numbers():
    from src.splitters import _build_verse_manifest
    md = '# T\n\npara one\n\nv1\nv2\nv3\n\npara two\n'
    chapters = [('ch01', md)]
    blocks = {'ch01': [{'tag': 'pre', 'text': 'v1\nv2\nv3'}]}
    m = _build_verse_manifest(chapters, blocks)
    e = m['chapters']['ch01'][0]
    assert (e['start_line'], e['end_line'], e['line_count']) == (5, 7, 3)
    assert md[e['start_char']:e['end_char']] == 'v1\nv2\nv3'
    assert m['version'] == 1


# ── 2. verse.py ────────────────────────────────────────────────────

def _manifest_entry(md, poem_text, tag='class'):
    from src.splitters import _build_verse_manifest
    m = _build_verse_manifest([('ch01', md)], {'ch01': [{'tag': tag, 'text': poem_text}]})
    return m['chapters']['ch01']


def test_build_chapter_tags_intersection():
    from src.utils.chunker import SmartChunker
    from src.verse import build_chapter_tags
    poem = 'v one line\nv two line\nv three line'
    md = '# T\n\n' + 'A' * 200 + '\n\n' + poem + '\n\n' + 'B' * 200
    entries = _manifest_entry(md, poem)
    chunks = SmartChunker(soft_limit=100, hard_limit=5000).split_markdown(md)
    tags = build_chapter_tags('ch01', md, chunks, entries)
    assert list(tags) == ['ch01_chunk001'], tags
    blocks = tags['ch01_chunk001']['blocks']
    assert len(blocks) == 1 and blocks[0]['lines'] == poem.split('\n')


def test_fragments_from_manifest_containment_fallback():
    from src.verse import fragments_from_manifest
    blocks = [{'tag': 'class', 'start_char': 0, 'end_char': 10,
               'text': 'line one long\nline two long'}]
    chunk = '  前文 line one   long\nline two long   后文 '
    frags = fragments_from_manifest(chunk, blocks)
    assert frags and frags[0]['lines'] == ['line one long', 'line two long']


def test_fragments_from_manifest_rejects_single_common_line():
    """Round 120 P1: 单行命中（Fury 诗 "would be"）不得混入散文 chunk。"""
    from src.verse import fragments_from_manifest
    fury = ("“Fury said to a\nmouse, That he\nmet in the\nhouse,\n"
            "‘Let us\nboth go to\nlaw: I will\nprosecute\nyou.—Come,\n"
            "I’ll take no\ndenial; We\nmust have a\ntrial: For\nreally this\n"
            "morning I’ve\nnothing to\ndo.’\nSaid the\nmouse to the\ncur, ‘Such\n"
            "a trial, dear\nSir, With no\njury or judge,\nwould be\n"
            "wasting our\nbreath.’")
    blocks = [{'tag': 'pre', 'start_char': 0, 'end_char': len(fury), 'text': fury}]
    prose = "…the best thing to get us dry would be a caucus-race…"
    assert fragments_from_manifest(prose, blocks) == []
    prose2 = "she added in a whisper, half afraid that it would be offended again"
    assert fragments_from_manifest(prose2, blocks) == []


def test_fragments_from_manifest_partial_consecutive_run():
    """部分命中需按块序连续 ≥2 行；单行(即使整句包含)不再放行。"""
    from src.verse import fragments_from_manifest
    blocks = [{'tag': 'class', 'text': 'line one long\nline two long\nline three long'}]
    frags = fragments_from_manifest("xx line two long line three long yy", blocks)
    assert frags and frags[0]['lines'] == ['line two long', 'line three long']
    assert fragments_from_manifest("xx line two long yy", blocks) == []


def test_fragments_for_chunk_prefers_tags():
    from src.verse import fragments_for_chunk
    tags = {'chapters': {'ch01': {'ch01_chunk000': {'verse': True,
            'blocks': [{'tag': 'class', 'lines': ['tag line']}]}}}}
    manifest = {'chapters': {'ch01': [{'tag': 'class', 'text': 'manifest line',
                                       'start_char': 0, 'end_char': 10}]}}
    frags = fragments_for_chunk('ch01', 'ch01_chunk000', 'zzz', tags, manifest)
    assert frags[0]['lines'] == ['tag line']
    frags2 = fragments_for_chunk('ch01', 'ch01_chunk999', 'manifest line', tags, manifest)
    assert frags2 and frags2[0]['lines'] == ['manifest line']


def test_build_verse_note_contains_requirements():
    from src.verse import build_verse_note
    lines = ['Beau—ootiful Soo—oop!', 'Soo—oop of the e—e—evening',
             'Beau—ootiful Soo—oop!', 'Soo—oop of the e—e—evening',
             'Beau—ootiful Soo—oop!']
    note = build_verse_note([{'tag': 'class', 'lines': lines}])
    assert '源共 5 行' in note
    assert '逐行对应' in note and '复沓' in note and '换行' in note
    assert build_verse_note([]) == ''


def test_verse_lint_pass_and_fail():
    from src.verse import verse_lint
    src = ['Beau—ootiful Soo—oop!', 'Soo—oop of the e—e—evening',
           'Beau—ootiful Soo—oop!', 'Soo—oop of the e—e—evening',
           'Beau—ootiful Soo—oop!', 'Beautiful Soup!']
    frags = [{'tag': 'class', 'lines': src}]
    good = '\n'.join(['美丽的汤！', '黄昏的汤，呃——', '美丽的汤！',
                      '黄昏的汤，呃——', '美丽的汤！', '美丽的汤！'])
    res = verse_lint(frags, good)
    assert res['ok'] is True and res['found_lines'] == 6

    bad = '这是一整段把韵文并进去的散文行，' * 12
    res_bad = verse_lint(frags, bad)
    assert res_bad['ok'] is False
    assert 'verse_lines' in res_bad['reason']

    # 行数够但复沓丢失 → refrain 失败
    no_refrain = '\n'.join(['甲一', '乙二', '丙三', '丁四', '戊五', '己六'])
    res_rf = verse_lint(frags, no_refrain)
    assert res_rf['ok'] is False and 'refrain' in res_rf['reason']


def test_verse_lint_accepts_dict_text():
    from src.verse import verse_lint
    frags = [{'tag': 'class', 'lines': ['a line one', 'a line two']}]
    res = verse_lint(frags, {'text': '译一\n译二'})
    assert res['ok'] is True


def test_chunk_tags_file_roundtrip(tmp_path):
    from src.verse import (load_chunk_tags, save_chunk_tags, load_verse_manifest,
                           save_verse_manifest)
    assert load_chunk_tags(tmp_path) == {}
    assert load_verse_manifest(tmp_path) == {}
    save_chunk_tags(tmp_path, {'version': 1, 'chapters': {'ch01': {}}})
    save_verse_manifest(tmp_path, {'version': 1, 'chapters': {}})
    assert load_chunk_tags(tmp_path)['chapters'] == {'ch01': {}}
    assert load_verse_manifest(tmp_path)['version'] == 1


# ── 3. 管线 glue ───────────────────────────────────────────────────

def _make_pipeline(tmp_path, with_verse=True):
    from src.pipeline import TranslationPipeline
    from src.verse import save_chunk_tags, save_verse_manifest
    (tmp_path / 'input').mkdir(exist_ok=True)
    (tmp_path / 'db').mkdir(exist_ok=True)
    if with_verse:
        lines = ['v one', 'v two', 'v three', 'v four', 'v five', 'v six']
        save_verse_manifest(tmp_path / 'input', {
            'version': 1,
            'chapters': {'ch01': [{'tag': 'class', 'text': '\n'.join(lines),
                                   'start_char': 0, 'end_char': 1, 'line_count': 6}]},
        })
        save_chunk_tags(tmp_path / 'db', {
            'version': 1,
            'chapters': {'ch01': {'ch01_chunk000': {
                'verse': True, 'blocks': [{'tag': 'class', 'lines': lines}]}}},
        })
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.chapter_id = 'ch01'
    p._root_dir = tmp_path
    p.output_dir = tmp_path / 'output' / 'ch01'
    p.output_dir.mkdir(parents=True, exist_ok=True)
    p._verse_tags_cache = None
    p._verse_manifest_cache = None
    p._verse_note_cache = {}
    return p


def test_pipeline_verse_glue(tmp_path):
    p = _make_pipeline(tmp_path)
    note = p._verse_note_for('ch01_chunk000', 'whatever')
    assert '逐行对应' in note and '源共 6 行' in note
    assert p._verse_note_for('ch01_chunk999', 'x') == ''

    gate = p._verse_lint_gate('ch01_chunk000', 'src', '一长句把韵文并掉的散文' * 20)
    assert gate is not None and gate['ok'] is False and gate['retried'] is False
    p._mark_verse_retry('ch01_chunk000', gate)
    assert p._verse_retry_used('ch01_chunk000') is True
    gate2 = p._verse_lint_gate('ch01_chunk000', 'src', '一长句把韵文并掉的散文' * 20)
    assert gate2['retried'] is True

    good = '\n'.join(['译一', '译二', '译三', '译四', '译五', '译六'])
    gate3 = p._verse_lint_gate('ch01_chunk000', 'src', good)
    assert gate3['ok'] is True


def test_pipeline_verse_disabled(tmp_path, monkeypatch):
    p = _make_pipeline(tmp_path)
    import src.verse as verse_mod

    def _cfg():
        cfg = dict(verse_mod._DEFAULT_VERSE_CONFIG)
        cfg['enabled'] = False
        return cfg

    monkeypatch.setattr(verse_mod, 'verse_config', _cfg)
    assert p._verse_note_for('ch01_chunk000', 'x') == ''
    assert p._verse_lint_gate('ch01_chunk000', 'src', 'y') is None


# ── 4. Agents ──────────────────────────────────────────────────────

def test_rewriter_prompt_verse_injection():
    from src.agents.rewriter_agent import LiteraryRewriterAgent
    r = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
    r._few_shot_examples = ''
    note = '【韵文/歌词处理协议（本片段含韵文，源共 3 行）】\n要求 X\n\n'
    p1 = r._build_prompt('raw', 'dec', 'sg', verse_note=note)
    assert '韵文/歌词处理协议' in p1
    assert p1.index('韵文/歌词处理协议') < p1.index('【排版与脚注协议')
    assert '韵文' not in r._build_prompt('raw', 'dec', 'sg')
    p2 = r._build_retry_prompt('raw', 'prev', 'dec', 'sg', 'reason', None, 1,
                               verse_note=note)
    assert '韵文/歌词处理协议' in p2


def test_judge_prompt_verse_injection():
    from src.agents.judge_agent import JudgeAgent
    j = JudgeAgent.__new__(JudgeAgent)
    report = {'scores': {'fluency': 8}}
    note = '【韵文/歌词处理协议（源共 5 行）】\n\n'
    p1 = j._build_prompt('src', 'lit', report, verse_note=note)
    assert '韵文对齐检查' in p1 and '【韵文/歌词处理协议' in p1
    assert '韵文对齐检查' not in j._build_prompt('src', 'lit', report)


# ── 5. 真实 EPUB 全量 ──────────────────────────────────────────────

@pytest.mark.skipif(not BOOK_EPUB.exists(), reason='Alice EPUB 源文件缺失')
def test_full_epub_manifest_covers_known_verse(tmp_path):
    from src.splitters import EpubSplitter
    sp = EpubSplitter(str(BOOK_EPUB), str(tmp_path))
    sp.split()
    sp.write_files()
    mani = json.loads((tmp_path / 'verse_manifest.json').read_text(encoding='utf-8'))
    assert mani['version'] == 1
    all_blocks = [(ch, b) for ch, bs in mani['chapters'].items() for b in bs]
    assert len(all_blocks) >= 7, f'韵文块过少: {len(all_blocks)}'
    assert any(b['tag'] == 'pre' and b['line_count'] >= 20 for _ch, b in all_blocks), \
        'Fury 长尾诗未进入 manifest'
    assert any('Twinkle' in b['text'] for _ch, b in all_blocks), '蝙蝠歌缺失'
    assert any('Beau' in b['text'] for _ch, b in all_blocks), '美汤歌缺失'
    for ch, b in all_blocks:
        md = (tmp_path / f'{ch}.md').read_text(encoding='utf-8')
        assert md[b['start_char']:b['end_char']] == b['text'], f'{ch} 偏移错位'
        assert sum(1 for ln in b['text'].split('\n') if ln.strip()) == b['line_count']


def test_pipeline_verse_helpers_tolerate_bypassed_init():
    """老测试常用 __new__ 绕开 __init__；helpers 不得依赖 __init__ 设置的缓存."""
    from src.pipeline import TranslationPipeline
    p = TranslationPipeline.__new__(TranslationPipeline)
    assert p._verse_tags() == {}
    assert p._verse_manifest() == {}
    assert p._verse_note_for("ch_test_chunk000", "正文") == ""
