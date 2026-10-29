#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""book_lint 镜像同步：把 src/book_lint.py 的守卫函数/常量逐字同步到
src/translator_agent.py 的镜像区（Round 119 T5 纪律；tests/test_round119_mirror_parity.py 校验）。

用法: python3 scripts/sync_book_lint_mirror.py [--check]
  --check  仅校验两处是否一致（不一致 → 退出码 1；不做写入）
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'book_lint.py'
DST = ROOT / 'src' / 'translator_agent.py'

# 与 tests/test_round119_mirror_parity.py 的 _BOOK_LINT_FUNCS/_BOOK_LINT_CONSTS 保持同步
FUNCS = [
    'sup_run_to_int', '_scan_lines', 'normalize_final_style', '_find_finals',
    'finalize_outputs', '_def_key', '_audit_text', 'lint_texts',
    '_chapter_dup_notes', '_book_style_config', 'lint_book',
    'postprocess_chunk_text', 'chapter_heading',
    '_note_norm', '_note_key', '_note_anchor_stop', '_note_is_dup', '_note_terms', '_note_candidates',
    '_group_title_units', '_split_h1_sections', 'normalize_chapter_notes',
    'normalize_notes_document', 'notes_normalize_outputs', '_chapter_note_dupes',
]
CONSTS = [
    '_SUPER', '_FOOTNOTE_HEADER_RE', '_SUP_DEF_RE', '_MD_DEF_RE',
    '_NUM_DEF_RE', '_NUM_DEF_START_RE', '_SUP_REF_RE', '_MD_REF_RE',
    '_META_TOKENS', '_FRONTMATTER_LINE_RE', '_CHAPTER_HEAD_RE',
    '_FRONTMATTER_RESIDUE_RES', '_DANGLING_RE',
    '_NOTE_TITLE_RE', '_NOTE_TITLE_END_RE', '_TR_BLOCK_RE', '_BRACKET_DEF_RE',
    '_PAREN_DEF_RE', '_BOLD_LEAD_RE', '_SEP_LINE_RE', '_MACHINE_FN_ID_RE',
    '_NOTE_LABEL_RE', '_QUOTED_TERM_RE', '_NOTE_DUP_RATIO', '_NOTE_DUP_RATIO_HARD',
]


def _segments(path: Path, names: list) -> dict:
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    out[t.id] = ast.get_source_segment(src, node)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args(argv)

    want = _segments(SRC, FUNCS + CONSTS)
    have = _segments(DST, FUNCS + CONSTS)
    missing = sorted(set(want) - set(have) - {'lint_book'})
    drift = sorted(n for n in set(want) & set(have) if want[n] != have[n])
    if args.check:
        if missing or drift:
            print(f'❌ 镜像不一致: missing={missing} drift={drift}')
            return 1
        print('✅ 镜像一致')
        return 0

    dst_text = DST.read_text(encoding='utf-8')
    dst_lines = dst_text.split('\n')

    def _line_range(src_text: str, name: str):
        tree = ast.parse(src_text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node.lineno - 1, (node.end_lineno or node.lineno)
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        return node.lineno - 1, (node.end_lineno or node.lineno)
        return None

    # 1) 逐个替换既有条目（自底向上）
    repl = []
    for name, want_seg in want.items():
        rng = _line_range(dst_text, name)
        if rng and want_seg != have.get(name):
            repl.append((rng[0], rng[1], want_seg.split('\n')))
    for start, end, seg_lines in sorted(repl, reverse=True):
        dst_lines[start:end] = seg_lines

    # 2) 新条目：整段插入到 chapter_heading 之后
    if missing:
        dst_text2 = '\n'.join(dst_lines)
        ch = _line_range(dst_text2, 'chapter_heading')
        assert ch, 'translator_agent 未找到 chapter_heading 镜像锚点'
        block = []
        for name in FUNCS + CONSTS:
            if name in missing:
                block += want[name].split('\n') + ['']
        block += ['# ── Round 121 L5 镜像区结束 ──', '']
        dst_lines[ch[1]:ch[1]] = block
    DST.write_text('\n'.join(dst_lines), encoding='utf-8')
    print(f'✅ 已同步（替换 {len(repl)} 项，新增 {len(missing)} 项）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
