"""Round 119 (T7): 双落点镜像防漂移.

散装 src/ 与单体 src/translator_agent.py 存在同源复制（8524↔128、8612↔231
一线，T3 又新增 scheduler 一处）。本轮 T1/T3/T6 的函数级改动必须双落点
一致，本测试断言：

  1. 源码段一致（AST 抽取逐字比对）:
     - src.core.scheduler._source_term_pattern / TaskScheduler.find_chunks_containing
     - src.consistency._in_quote_or_note（含 _NOTE_MARKER_RE 常量行）
  2. 行为一致（真实 EPUB fixture）:
     - _epub_clean_html_text / _epub_extract_title / _strip_epub_front_matter
       两处实现逐字节相等。
  3. repo 模式下 TaskScheduler 为共享实例（monolith 底部真实模块替换生效，
     即生产 CLI 用的就是散装实现，而非内联旧副本）。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = Path('/tmp/alice_epub/OEBPS')
_need_fixtures = pytest.mark.skipif(
    not FIXTURE_DIR.exists(), reason='真实 EPUB 样本缺失: /tmp/alice_epub')

_FIXTURE_NAMES = ('11-h-0', '11-h-2', '11-h-3', '11-h-13')


def _extract(path: str, name: str, cls: str | None = None) -> str:
    src = (ROOT / path).read_text(encoding='utf-8')
    tree = ast.parse(src)
    if cls is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(src, node)
    else:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == name:
                        return ast.get_source_segment(src, sub)
    raise AssertionError(f'未找到 {cls}.{name} in {path}')


def _fixture(name: str) -> str:
    return (FIXTURE_DIR / f'7126951391209738045_{name}.htm.xhtml').read_text(encoding='utf-8')


# ── 1. 源码段一致 ────────────────────────────────────────────────────

def test_source_term_pattern_source_parity():
    a = _extract('src/core/scheduler.py', '_source_term_pattern')
    b = _extract('src/translator_agent.py', '_source_term_pattern')
    assert a == b, '词边界正则两处实现漂移'


def test_find_chunks_containing_source_parity():
    a = _extract('src/core/scheduler.py', 'find_chunks_containing', cls='TaskScheduler')
    b = _extract('src/translator_agent.py', 'find_chunks_containing', cls='TaskScheduler')
    assert a == b, 'find_chunks_containing 两处实现漂移'


def test_in_quote_or_note_source_parity():
    a = _extract('src/consistency.py', '_in_quote_or_note')
    b = _extract('src/translator_agent.py', '_in_quote_or_note')
    assert a == b, '_in_quote_or_note 两处实现漂移'


def test_consolidate_source_parity():
    a = _extract('src/pipeline.py', 'consolidate_dirty_chapters')
    b = _extract('src/translator_agent.py', 'consolidate_dirty_chapters')
    assert a == b, 'consolidate_dirty_chapters 两处实现漂移'


def test_verify_final_outputs_source_parity():
    a = _extract('src/pipeline.py', '_verify_final_outputs', cls='TranslationPipeline')
    b = _extract('src/translator_agent.py', '_verify_final_outputs', cls='TranslationPipeline')
    assert a == b, '_verify_final_outputs 两处实现漂移'


def test_run_final_gate_source_parity():
    a = _extract('src/pipeline.py', 'run_final_gate')
    b = _extract('src/translator_agent.py', 'run_final_gate')
    assert a == b, 'run_final_gate 两处实现漂移'


def test_note_marker_re_constant_parity():
    def const_line(path: str):
        for line in (ROOT / path).read_text(encoding='utf-8').splitlines():
            if line.startswith('_NOTE_MARKER_RE'):
                return line
        return None
    assert const_line('src/consistency.py') == const_line('src/translator_agent.py')


# ── 2. 转换层行为一致（真实 fixture） ────────────────────────────────

@_need_fixtures
def test_epub_clean_html_text_behavior_parity():
    from src import splitters
    import src.translator_agent as monolith
    for name in _FIXTURE_NAMES:
        html = _fixture(name)
        assert (splitters._epub_clean_html_text(html)
                == monolith._epub_clean_html_text(html)), name


@_need_fixtures
def test_epub_extract_title_behavior_parity():
    from bs4 import BeautifulSoup
    from src import splitters
    import src.translator_agent as monolith
    for name in _FIXTURE_NAMES:
        html = _fixture(name)
        t1 = splitters._epub_extract_title(BeautifulSoup(html, 'html.parser'), 'fallback')
        t2 = monolith._epub_extract_title(BeautifulSoup(html, 'html.parser'), 'fallback')
        assert t1 == t2, name


@_need_fixtures
def test_strip_front_matter_behavior_parity():
    from src import splitters
    import src.translator_agent as monolith
    for name in _FIXTURE_NAMES:
        text = splitters._epub_clean_html_text(_fixture(name))
        assert (splitters._strip_epub_front_matter(text)
                == monolith._strip_epub_front_matter(text)), name


def test_strip_front_matter_gutenberg_synthetic_parity():
    from src import splitters
    import src.translator_agent as monolith
    text = (
        'The Project Gutenberg eBook of Alice\n\n'
        '*** START OF THE PROJECT GUTENBERG EBOOK ALICE ***\n\n'
        'Contents\n\n'
        'CHAPTER I. Down the Rabbit-Hole\n'
        'CHAPTER II. The Pool of Tears\n\n\n'
        'CHAPTER I. Down the Rabbit-Hole\n\n'
        'Alice was beginning to get very tired.\n'
    )
    assert (splitters._strip_epub_front_matter(text)
            == monolith._strip_epub_front_matter(text))


# ── 3. repo 模式共享实例 ─────────────────────────────────────────────

def test_task_scheduler_shared_instance_in_repo_mode():
    import src.translator_agent as monolith
    from src.core.scheduler import TaskScheduler
    assert monolith.TaskScheduler is TaskScheduler, (
        'monolith 未替换为真实 TaskScheduler — 生产 CLI 将用到内联副本')


# ── 4. book_lint 单源（T5） ──────────────────────────────────────────

_BOOK_LINT_FUNCS = [
    'sup_run_to_int', '_scan_lines', 'normalize_final_style', '_find_finals',
    'finalize_outputs', '_def_key', '_audit_text', 'lint_texts',
    '_chapter_dup_notes', '_book_style_config', 'lint_book',
    'postprocess_chunk_text', 'chapter_heading',
    '_note_norm', '_note_key', '_note_is_dup', '_note_terms', '_note_candidates',
    '_group_title_units', '_split_h1_sections', 'normalize_chapter_notes',
    'normalize_notes_document', 'notes_normalize_outputs', '_chapter_note_dupes',
]
_BOOK_LINT_CONSTS = [
    '_SUPER', '_FOOTNOTE_HEADER_RE', '_SUP_DEF_RE', '_MD_DEF_RE',
    '_NUM_DEF_RE', '_NUM_DEF_START_RE', '_SUP_REF_RE', '_MD_REF_RE',
    '_META_TOKENS', '_FRONTMATTER_LINE_RE', '_CHAPTER_HEAD_RE',
    '_FRONTMATTER_RESIDUE_RES', '_DANGLING_RE',
    '_NOTE_TITLE_RE', '_NOTE_TITLE_END_RE', '_TR_BLOCK_RE', '_BRACKET_DEF_RE',
    '_PAREN_DEF_RE', '_BOLD_LEAD_RE', '_SEP_LINE_RE', '_MACHINE_FN_ID_RE',
    '_NOTE_LABEL_RE', '_QUOTED_TERM_RE', '_NOTE_DUP_RATIO', '_NOTE_DUP_RATIO_HARD',
    '_NOTE_ANCHOR_STOP',
]


def _extract_assign(path: str, name: str) -> str:
    src = (ROOT / path).read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.get_source_segment(src, node)
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == name):
            return ast.get_source_segment(src, node)
    raise AssertionError(f'未找到赋值 {name} in {path}')


@pytest.mark.parametrize('name', _BOOK_LINT_FUNCS)
def test_book_lint_function_source_parity(name):
    a = _extract('src/book_lint.py', name)
    b = _extract('src/translator_agent.py', name)
    assert a == b, f'{name} 两处实现漂移'


@pytest.mark.parametrize('name', _BOOK_LINT_CONSTS)
def test_book_lint_constant_source_parity(name):
    a = _extract_assign('src/book_lint.py', name)
    b = _extract_assign('src/translator_agent.py', name)
    assert a == b, f'{name} 常量漂移'


def _fresh_book_lint():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'book_lint_fresh', ROOT / 'src' / 'book_lint.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_book_lint_behavior_parity():
    import src.translator_agent as monolith
    bl = _fresh_book_lint()
    messy = ('**脚注（保留与加注策略）**\n他喊「快跑」，说"go"。¹\n\n'
             '¹ 原注：某典故。\n\n² 悬空注释。\n')
    assert bl.normalize_final_style(messy) == monolith.normalize_final_style(messy)
    merge_text = '# The Project Gutenberg eBook of X\n\n第3章 山间小屋\n\n正文行。\n'
    assert (bl.postprocess_chunk_text(merge_text, 'ch01', 0)
            == monolith.postprocess_chunk_text(merge_text, 'ch01', 0))


def test_book_lint_chapter_heading_behavior_parity(tmp_path):
    import src.translator_agent as monolith
    bl = _fresh_book_lint()
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'ch01.md').write_text(
        '# CHAPTER I. Down the Rabbit-Hole\n\n正文。\n', encoding='utf-8')
    d = tmp_path / 'output' / 'ch01'
    d.mkdir(parents=True)
    (d / 'ch01_chunk000_final.json').write_text(
        json.dumps({"text": "第一章 掉进兔子洞\n\n正文。\n"}, ensure_ascii=False),
        encoding='utf-8')
    for style in ('bilingual', 'zh', 'keep'):
        assert (bl.chapter_heading(tmp_path, 'ch01', style)
                == monolith.chapter_heading(tmp_path, 'ch01', style))


# ── 5. Round 119 L1 韵文通道（散装↔单体 + verse.py 单源） ─────────────

_L1_SPLITTER_FUNCS = [
    '_mark_verse_elements', '_el_verse_tag', '_block_element_text',
    '_extract_block_text', '_epub_clean_html_text_ex', '_epub_clean_html_text',
    '_build_verse_manifest', '_epub_split_large_doc_by_toc', '_epub_extract_chapters',
]


@pytest.mark.parametrize('name', _L1_SPLITTER_FUNCS)
def test_l1_splitters_source_parity(name):
    a = _extract('src/splitters.py', name)
    b = _extract('src/translator_agent.py', name)
    assert a == b, f'{name} 两处实现漂移'


_L1_SPLITTER_METHODS = [
    ('EpubSplitter', '__init__'),
    ('EpubSplitter', 'split'),
    ('EpubSplitter', 'write_files'),
]


@pytest.mark.parametrize('cls,name', _L1_SPLITTER_METHODS)
def test_l1_splitters_method_parity(cls, name):
    a = _extract('src/splitters.py', name, cls=cls)
    b = _extract('src/translator_agent.py', name, cls=cls)
    assert a == b, f'{cls}.{name} 两处实现漂移'


_L1_PIPELINE_METHODS = [
    '_verse_tags', '_verse_manifest', '_verse_fragments_for', '_verse_note_for',
    '_verse_lint_gate', '_mark_verse_retry', '_verse_retry_used',
    '_verse_lint_gate', '_mark_verse_retry', '_verse_retry_used',
    '_seed_pun_decisions',
    '_dual_draft_cfg', '_dual_draft_kind', '_dual_critic_avg',
    '_dual_pairwise_winner', '_dual_draft_run',
    '__init__',
    '_read_rewriter_switch_state', '_record_rewriter_model_switch',
    '_build_literal_prompt', '_run_raw_translator',
    '_process_translating_raw_batch', '_process_rewriting_batch', '_process_judging_batch',
]


@pytest.mark.parametrize('name', _L1_PIPELINE_METHODS)
def test_l1_pipeline_method_parity(name):
    a = _extract('src/pipeline.py', name, cls='TranslationPipeline')
    b = _extract('src/translator_agent.py', name, cls='TranslationPipeline')
    assert a == b, f'TranslationPipeline.{name} 两处实现漂移'


_L1_AGENT_METHODS = [
    ('src/agents/rewriter_agent.py', 'LiteraryRewriterAgent', '_build_prompt'),
    ('src/agents/rewriter_agent.py', 'LiteraryRewriterAgent', '_build_retry_prompt'),
    ('src/agents/rewriter_agent.py', 'LiteraryRewriterAgent', 'process_chunk'),
    ('src/agents/judge_agent.py', 'JudgeAgent', '_build_prompt'),
    ('src/agents/judge_agent.py', 'JudgeAgent', 'process_chunk'),
]


@pytest.mark.parametrize('path,cls,name', _L1_AGENT_METHODS)
def test_l1_agent_method_parity(path, cls, name):
    a = _extract(path, name, cls=cls)
    b = _extract('src/translator_agent.py', name, cls=cls)
    assert a == b, f'{cls}.{name} 两处实现漂移'


def test_l1_init_project_parity():
    a = _extract('src/main.py', 'init_project')
    b = _extract('src/translator_agent.py', 'init_project')
    assert a == b, 'init_project 两处实现漂移'


_L1_VERSE_FUNCS = [
    'verse_config', 'normalize_whitespace', 'load_verse_manifest', 'save_verse_manifest',
    'load_chunk_tags', 'save_chunk_tags', '_normalized_index', 'locate_chunk_spans',
    '_block_lines_in_range', 'blocks_for_span', 'fragments_from_manifest',
    'fragments_for_chunk', 'build_chapter_tags', '_max_line_repeat', 'build_verse_note',
    '_longest_short_line_run', 'verse_lint',
]


@pytest.mark.parametrize('name', _L1_VERSE_FUNCS)
def test_l1_verse_source_parity(name):
    a = _extract('src/verse.py', name)
    b = _extract('src/translator_agent.py', name)
    assert a == b, f'verse.{name} 两处实现漂移'


@pytest.mark.parametrize('name',
                         ['VERSE_MANIFEST_VERSION', 'CHUNK_TAGS_VERSION',
                          '_DEFAULT_VERSE_CONFIG'])
def test_l1_verse_constant_parity(name):
    a = _extract_assign('src/verse.py', name)
    b = _extract_assign('src/translator_agent.py', name)
    assert a == b, f'verse 常量 {name} 漂移'


@_need_fixtures
def test_l1_clean_html_ex_verse_parity():
    from src import splitters
    import src.translator_agent as monolith
    for name in _FIXTURE_NAMES:
        html = _fixture(name)
        v1, v2 = [], []
        t1 = splitters._epub_clean_html_text_ex(html, verse_out=v1)
        t2 = monolith._epub_clean_html_text_ex(html, verse_out=v2)
        assert t1 == t2 and v1 == v2, name
