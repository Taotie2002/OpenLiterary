"""Round 119 (T1): 转换层信息守恒 — <pre> 保全 / 标题拼接 / 转换完整性审计.

背景（D1/D7）:
  - Alice EPUB 全量普查: <pre> 恰 2 处（Mouse's Tale 韵文、地址条），旧
    block 提取（仅 p/标题）把两处静默丢弃。
  - 章节标题经 <br/> 或多片段拼接丢空格: "CHAPTER I.<br/>Down…" →
    "CHAPTER I.Down…"; 页眉 h2 多片段 → "…eBook ofAlice's…"。
  - 旧 _validate_split 用同一套清洗器当基线（同源盲区），无法发现上述丢失。

测试目标（对应方案 T1 验收）:
  1. _epub_clean_html_text 保留 <pre> 行结构（Fury 诗 / 地址条）。
  2. 标题提取修复: br 与多片段均以空格连接; _normalize_chapter_title
     修复 "CHAPTER I.Down the Rabbit-Hole"（并保护 U.S.A. 型缩写）。
  3. 目录 table 兜底提取后被 _strip_epub_front_matter 正确裁剪。
  4. run_conversion_audit: 独立 oracle token 覆盖率; 缺 pre → 红 +
     missing span 含 Fury; 全量 → 绿; 古腾堡许可证/目录区域豁免。
  5. 全量 EPUB 集成: Fury/地址条进正文、标题无粘连、审计 ≥99.5%。
"""
from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_DIR = Path('/tmp/alice_epub/OEBPS')
BOOK_EPUB = Path('/home/zyu/books/Alices_Adventures_in_Wonderland.epub')

_need_fixtures = pytest.mark.skipif(
    not FIXTURE_DIR.exists(), reason='真实 EPUB 样本缺失: /tmp/alice_epub')


def _fixture(name: str) -> str:
    return (FIXTURE_DIR / f'7126951391209738045_{name}.htm.xhtml').read_text(encoding='utf-8')


# ── 1. <pre> 保全 ────────────────────────────────────────────────────

@_need_fixtures
def test_pre_poem_preserved_with_line_structure():
    from src.splitters import _epub_clean_html_text
    out = _epub_clean_html_text(_fixture('11-h-3'))  # CHAPTER III (Mouse's Tale)
    assert 'Fury said to a' in out
    assert 'prosecute' in out
    # 行结构保留：相邻诗行不被压成一行
    assert 'Fury said to a\nmouse, That he' in out
    # 内联 <i> 不切断行（law: <i>I</i> will …）
    assert 'law: I will' in out


@_need_fixtures
def test_pre_address_strip_preserved():
    from src.splitters import _epub_clean_html_text
    out = _epub_clean_html_text(_fixture('11-h-2'))  # CHAPTER II (地址条)
    assert 'Hearthrug' in out
    assert 'Hearthrug,\nnear the Fender,' in out


# ── 2. 标题拼接 ──────────────────────────────────────────────────────

@_need_fixtures
def test_title_extraction_joins_fragments():
    from bs4 import BeautifulSoup
    from src.splitters import _epub_extract_title
    s3 = BeautifulSoup(_fixture('11-h-3'), 'html.parser')
    assert _epub_extract_title(s3, 'fallback') == 'CHAPTER III. A Caucus-Race and a Long Tale'
    s0 = BeautifulSoup(_fixture('11-h-0'), 'html.parser')
    assert _epub_extract_title(s0, 'fallback') == \
        "The Project Gutenberg eBook of Alice's Adventures in Wonderland"


def test_normalize_chapter_title_widened():
    from src.splitters import _normalize_chapter_title
    assert _normalize_chapter_title('CHAPTER I.Down the Rabbit-Hole') == \
        'CHAPTER I. Down the Rabbit-Hole'
    assert _normalize_chapter_title('CHAPTER X.The Lobster Quadrille') == \
        'CHAPTER X. The Lobster Quadrille'
    # 旧行为保持
    assert _normalize_chapter_title('II.The Machine') == 'II. The Machine'
    assert _normalize_chapter_title('1.Introduction') == '1. Introduction'
    # 缩写保护: 节间命名链不误插空格
    assert _normalize_chapter_title('U.S.A. Trilogy') == 'U.S.A. Trilogy'


# ── 3. 目录 table 裁剪 ───────────────────────────────────────────────

@_need_fixtures
def test_titlepage_toc_table_stripped():
    from src.splitters import _epub_clean_html_text, _strip_epub_front_matter
    text = _strip_epub_front_matter(_epub_clean_html_text(_fixture('11-h-0')))
    assert 'Alice’s Adventures in Wonderland' in text
    assert 'by Lewis Carroll' in text
    assert 'This eBook is for the use' not in text  # 许可证 header 已整体丢弃
    assert 'Down the Rabbit-Hole' not in text       # 目录项被裁剪
    assert 'Contents' not in text


# ── 4. 转换完整性审计 ────────────────────────────────────────────────

_HTML_PRE = ('<html><body><p>Hello world.</p>'
             '<pre>Fury said to a\nmouse, That he</pre>'
             '<p>The end.</p></body></html>')
_MD_FULL = 'Hello world.\n\nFury said to a\nmouse, That he\n\nThe end.'
_MD_LOST = 'Hello world.\n\nThe end.'

_HTML_TP = ('<html><head><title>Alice | Project Gutenberg</title></head><body>'
            '<p>Alice’s Adventures in Wonderland</p>'
            '<h2>Contents</h2>'
            '<table><tr><td>CHAPTER I.</td><td>Down the Rabbit-Hole</td></tr>'
            '<tr><td>CHAPTER II.</td><td>The Pool of Tears</td></tr></table>'
            '<p>page one text.</p></body></html>')


def test_audit_missing_pre_goes_red(tmp_path):
    from src.splitters import run_conversion_audit
    res = run_conversion_audit(
        {'ch01': [_HTML_PRE]}, [('ch01', _MD_LOST)], [],
        tmp_path / '_conversion_audit.md')
    assert res['ok'] is False
    assert res['overall'] < 0.995
    assert any('Fury' in s['excerpt'] for s in res['missing_spans'])
    report = (tmp_path / '_conversion_audit.md').read_text(encoding='utf-8')
    assert 'Fury' in report


def test_audit_full_text_goes_green(tmp_path):
    from src.splitters import run_conversion_audit
    res = run_conversion_audit(
        {'ch01': [_HTML_PRE]}, [('ch01', _MD_FULL)], [],
        tmp_path / '_conversion_audit.md')
    assert res['ok'] is True
    assert res['overall'] == 1.0
    assert res['missing_spans'] == []


def test_audit_whitelists_license_and_toc(tmp_path):
    from src.splitters import run_conversion_audit
    md = 'Alice’s Adventures in Wonderland\n\npage one text.'
    res = run_conversion_audit(
        {'ch01': [_HTML_TP]}, [('ch01', md)], [],
        tmp_path / '_conversion_audit.md')
    assert res['ok'] is True, res['missing_spans']
    assert res['overall'] == 1.0


# ── 5. 全量 EPUB 集成 ────────────────────────────────────────────────

def test_split_cli_aborts_on_audit_failure(tmp_path, monkeypatch):
    """审计未通过 → split 入口非零退出（split 子命令硬闸）。"""
    import src.main as m

    class _FakeSplitter:
        def __init__(self, *args, **kwargs):
            self.chapters = [('ch01', '# t\n\nbody')]
            self.audit_result = {'ok': False, 'overall': 0.5, 'report_path': 'r'}

        def split(self):
            return self.chapters

        def write_files(self):
            return ['unused']

    monkeypatch.setattr(m, 'EpubSplitter', _FakeSplitter)
    with pytest.raises(SystemExit) as exc:
        m.split_input_to_chapters('/tmp/x.epub', str(tmp_path))
    assert exc.value.code == 1


@pytest.mark.skipif(not BOOK_EPUB.exists(), reason='Alice EPUB 源文件缺失')
def test_full_epub_split_t1_acceptance(tmp_path):
    from src.splitters import EpubSplitter
    splitter = EpubSplitter(str(BOOK_EPUB), str(tmp_path))
    chapters = dict(splitter.split())

    caucus = [md for md in chapters.values() if 'A Caucus-Race and a Long Tale' in md]
    assert caucus, 'CHAPTER III 章节缺失'
    assert 'Fury said to a' in caucus[0], 'Fury 诗未进入正文'
    assert '# CHAPTER III. A Caucus-Race and a Long Tale' in caucus[0]

    pool = [md for md in chapters.values() if 'The Pool of Tears' in md]
    assert pool, 'CHAPTER II 章节缺失'
    assert 'Hearthrug' in pool[0], '地址条未进入正文'
    assert '# CHAPTER II. The Pool of Tears' in pool[0]

    # 标题粘连回归（旧缺陷: "ofAlice's" / "I.Down"）
    for md in chapters.values():
        first_line = md.split('\n', 1)[0]
        assert not first_line.startswith('# The Project Gutenberg eBook ofAlice')

    assert splitter.audit_result.get('ok') is True, splitter.audit_result
    assert splitter.audit_result.get('overall', 0.0) >= 0.995
