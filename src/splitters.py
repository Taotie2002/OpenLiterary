"""
SagaTrans 输入格式适配器（EpubSplitter / TextSplitter / MdSplitter）

源：原整合脚本 §11，拆分到独立模块。

职责：将原始输入（EPUB / TXT / MD）切分为多个章节 Markdown 文件，
      输出到 input/chXX.md，供 SmartChunker 二次切分。

注意：本模块为输入适配层，与翻译管线业务逻辑完全解耦。
"""
from __future__ import annotations

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple


# Section 11 — 输入格式适配器 (EpubSplitter / TextSplitter / MdSplitter)
# ─────────────────────────────────────────────────────────────────────────────
# 职责：将原始输入（EPUB / TXT / MD）切分为多个章节 Markdown 文件，
#       输出到 input/chXX.md，供 SmartChunker 二次切分。
# 注意：本节为输入适配层，与翻译管线业务逻辑完全解耦。

def _splitter_log_info(msg: str):
    print(f"[INFO] {msg}", file=sys.stderr)


def _splitter_log_ok(msg: str):
    print(f"[OK] {msg}", file=sys.stderr)


def _splitter_log_warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)


def _splitter_log_err(msg: str):
    print(f"[ERR] {msg}", file=sys.stderr)


# ────────────────────────────────────────────────────────────────────
# 11.1 EpubSplitter — EPUB → 章节 Markdown
# ─────────────────────────────────────────────────────────────────────────────

# EPUB 解析依赖（懒加载，缺失时给出友好提示）
try:
    import ebooklib
    from ebooklib import epub as _epub_lib
    from bs4 import BeautifulSoup
    _EPUB_DEPS_OK = True
except ImportError as _e_dep:
    _EPUB_DEPS_OK = False
    _EPUB_DEPS_ERROR = str(_e_dep)


def _normalize_chapter_title(title: str) -> str:
    """规范章节标题：在「序号.」与后续正文之间补空格。

    源 EPUB 常见 `II.The Machine` / `CHAPTER I.Down the Rabbit-Hole` 这类
    序号与英文紧接、无空格的形式，拼成 Markdown 标题后观感差。统一规范为
    `II. The Machine` / `CHAPTER I. Down the Rabbit-Hole`。同时处理
    「第N章xxx」保持原样。
    """
    if not title:
        return title
    # T1：在任意「字母数字 + 句点 + 紧接大写字母」处插入空格（不再限于
    # 行首，修复 "CHAPTER I.Down…" 型）。两组负向守卫避免误伤缩写链：
    #   - 句点前已是「字母数字+句点」（如 U.S.A.、1.2.）不插入
    #   - 大写字母后紧跟另一个句点（如 U.S.A.、Ph.D.）不插入
    fixed = re.sub(r'(?<![A-Za-z0-9]\.)([A-Za-z0-9])\.(?=[A-Z](?!\.))',
                   r'\1. ', title.strip())
    return fixed


# Critic LLM 偶发把维度名拼错（如 voice_consistency → voice_consistence），
# 导致下游 Judge 阈值查表时该脏维度被静默忽略。此处用白名单归一化，
# 把已知拼写变体映射回规范维度名，未命中白名单的键原样保留。
_SCORE_KEY_NORMALIZATION = {
    "voice_consistence": "voice_consistency",
    "voice_consistancy": "voice_consistency",
    "voice consistence": "voice_consistency",
    "style_complience": "style_compliance",
    "style_compliance_": "style_compliance",
    "semantic_preservation": "semantic_preservation",
    "semantic_preserv": "semantic_preservation",
    "literal_residue": "literal_residue",
    "literal_residues": "literal_residue",
    "literal_residu": "literal_residue",
    "fluency": "fluency",
    "readability": "readability",
}


def _normalize_scores(scores: Optional[dict]) -> dict:
    """归一化 Critic 评分维度名，剔除脏维度名对 Judge 阈值逻辑的污染。

    只处理 dict 且值为数值（int/float）的项；非数值或空值原样跳过，
    避免把 critique 文本当分数累加。
    """
    if not isinstance(scores, dict):
        return scores if isinstance(scores, dict) else {}
    normalized: dict = {}
    for k, v in scores.items():
        key = k
        if isinstance(k, str):
            key = _SCORE_KEY_NORMALIZATION.get(k.strip().lower(), k.strip())
        if isinstance(v, (int, float)):
            normalized[key] = v
    return normalized


# T1：块级提取口径。主路径 = p/标题/pre；blockquote/li/dl(dd/dt)/table(tr)
# 仅在「无主路径后代」时兜底提取（此前这些容器内容被静默丢弃）。
_PRIMARY_BLOCK_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre']
_FALLBACK_BLOCK_TAGS = ['blockquote', 'li', 'dd', 'dt', 'tr', 'table']

# T-L1（Round 119）：韵文元素识别。三类来源：
#   1. class 命中 poem/stanza/verse/song/lyric/epigraph（显式标注）；
#   2. <pre>（天然保行，如老鼠长尾诗 / 地址条）；
#   3. 短行启发式：元素含 ≥2 个 <br/> 且多数行较短（未标注的韵文）。
# 命中元素在 br→'\n' 替换前打上 data 属性；提取时保留行结构并记录
# （供 input/verse_manifest.json 与 verse_lint 使用）。
_VERSE_TAG_ATTR = 'data-sagatrans-verse'
_VERSE_CLASS_RE = re.compile(r'poem|stanza|verse|song|lyric|epigraph', re.IGNORECASE)
_VERSE_HEURISTIC_MIN_BR = 2
_VERSE_HEURISTIC_MAX_LINE = 40
_VERSE_HEURISTIC_SHORT_RATIO = 0.6


def _mark_verse_elements(soup) -> None:
    """T-L1：在 br→'\\n' 替换前标记韵文元素（class 命中或短行 br 启发式）。"""
    for el in soup.find_all(['p', 'pre', 'blockquote', 'div', 'section', 'dd']):
        cls = ' '.join(el.get('class') or [])
        if cls and _VERSE_CLASS_RE.search(cls):
            el[_VERSE_TAG_ATTR] = 'class'
            continue
        if el.name == 'pre':
            continue  # pre 天然保行；tag 在提取时按行数判定
        br_count = len(el.find_all('br'))
        if br_count < _VERSE_HEURISTIC_MIN_BR:
            continue
        lines = [ln.strip() for ln in el.get_text('\n').split('\n') if ln.strip()]
        if len(lines) < 3:
            continue
        short = sum(1 for ln in lines if len(ln) <= _VERSE_HEURISTIC_MAX_LINE)
        if short / len(lines) >= _VERSE_HEURISTIC_SHORT_RATIO:
            el[_VERSE_TAG_ATTR] = 'heuristic'


def _el_verse_tag(el) -> Optional[str]:
    """元素的韵文来源标签：class/启发式 标记 或 pre（多行时）。"""
    tag = el.get(_VERSE_TAG_ATTR)
    if tag:
        return str(tag)
    if el.name == 'pre':
        return 'pre'
    return None


def _block_element_text(el) -> str:
    """提取单个块级元素文本（pre/韵文保留行结构，其余以空格连接片段）。"""
    if el.name == 'pre' or el.get(_VERSE_TAG_ATTR):
        # 韵文/地址条：逐行 strip 保留行结构；内联标签（<i> 等）不切断行，
        # 故不带分隔符取文本（片段自带空格）。连续空行压成一个（保留诗节）。
        lines: list = []
        for raw in el.get_text('').split('\n'):
            ln = raw.strip()
            if not ln:
                if lines and lines[-1] != '':
                    lines.append('')
                continue
            lines.append(ln)
        while lines and lines[-1] == '':
            lines.pop()
        return '\n'.join(lines)
    return el.get_text(' ', strip=True)


def _extract_block_text(root, verse_out: Optional[list] = None) -> str:
    """按块级元素提取纯文本，以 '\\n\\n' 拼接还原段落边界。

    不含 div（div 常为容器，其 get_text 会把整章压成一段，反而丢失段落
    结构）。无任何主路径块级标签时退化为整体文本（历史行为）。
    T-L1：verse_out 非 None 时收集多行韵文块（{"tag", "text"}）。
    """
    primary = root.find_all(_PRIMARY_BLOCK_TAGS)
    if not primary:
        return root.get_text('\n')
    parts = []
    for el in root.find_all(_PRIMARY_BLOCK_TAGS + _FALLBACK_BLOCK_TAGS):
        if el.name in _FALLBACK_BLOCK_TAGS:
            # 兜底仅取「主路径不可见」的元素：含主块级后代（已被主路径提取）
            # 或含更内层兜底元素（避免重复提取）时跳过
            if el.find(_PRIMARY_BLOCK_TAGS) or el.find(_FALLBACK_BLOCK_TAGS):
                continue
        t = _block_element_text(el)
        if t:
            parts.append(t)
            if verse_out is not None and '\n' in t:
                tag = _el_verse_tag(el)
                if tag:
                    verse_out.append({'tag': tag, 'text': t})
    return '\n\n'.join(parts)


def _epub_clean_html_text_ex(html_content: str, verse_out: Optional[list] = None) -> str:
    """清洗 HTML，提取纯文本（保留段落结构）。

    P 方案修复：原先对每个块级标签 insert_after('\\n') 再用
    get_text(separator='\\n')，因源 XML 段内自带换行，最终整章塌成 1 段。
    改为逐个提取块级元素文本、以 '\\n\\n' 拼接，确定性还原段落边界，
    且对段内换行免疫。
    T1：块级口径扩展为 p/标题/pre（保留韵文行结构）+ 容器类兜底提取。
    T-L1：先标记韵文元素；verse_out 非 None 时收集多行韵文块。
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    _mark_verse_elements(soup)

    for br in soup.find_all('br'):
        br.replace_with('\n')

    text = _extract_block_text(soup, verse_out)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 注意：不能用 \s（含 \n）配合 MULTILINE 去修整行首尾，否则会把段落间的
    # \n\n 也吃掉、重新塌成 1 段。仅修剪行首尾的空格/制表符。
    text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', text)
    return text.strip()


def _epub_clean_html_text(html_content: str) -> str:
    """_epub_clean_html_text_ex 的纯文本包装（不收集韵文块，兼容旧调用）。"""
    return _epub_clean_html_text_ex(html_content)


def _build_verse_manifest(chapters: list, chapter_verse: dict) -> dict:
    """T-L1：由各章韵文块记录构建 input/verse_manifest.json 结构。

    逐块在章节 md 中按序定位（字符偏移 → 行号）；定位失败的块跳过并告警。
    """
    chapters_out: dict = {}
    for ch_id, md in chapters:
        blocks = chapter_verse.get(ch_id) or []
        entries: list = []
        cursor = 0
        for b in blocks:
            t = re.sub(r'[ \t]+', ' ', b.get('text') or '')
            if not t.strip():
                continue
            pos = md.find(t, cursor)
            if pos < 0:
                pos = md.find(t)
            if pos < 0:
                _splitter_log_warn(
                    f"{ch_id} 韵文块未在 md 中定位（跳过）: {t[:30]!r}")
                continue
            cursor = pos + len(t)
            start_line = md.count('\n', 0, pos) + 1
            end_line = md.count('\n', 0, pos + len(t)) + 1
            line_count = sum(1 for ln in t.split('\n') if ln.strip())
            entries.append({
                'tag': b.get('tag', 'unknown'),
                'start_line': start_line,
                'end_line': end_line,
                'start_char': pos,
                'end_char': pos + len(t),
                'line_count': line_count,
                'text': t,
            })
        chapters_out[ch_id] = entries
    return {'version': 1, 'chapters': chapters_out}


# Q2 修复：古腾堡计划 EPUB 头部含许可证公告 + 目录（"Contents"/"目录"），
# 这些非正文文本若随章节被翻译，会污染最终译文。此处按常见特征裁剪。
_GUTENBERG_HEADER_PATTERNS = [
    re.compile(r'Project Gutenberg', re.IGNORECASE),
    re.compile(r'www\.gutenberg\.org', re.IGNORECASE),
    re.compile(r'This eBook is for the use of anyone'),
    re.compile(r'古腾堡计划', re.IGNORECASE),
]
# 目录标题行：单独出现即为目录标志（需配合紧随其后的编号条目块才裁剪）
_GUTENBERG_TOC_TITLE_PATTERN = re.compile(
    r'^\s*(contents|table of contents|目录)\s*$', re.IGNORECASE | re.MULTILINE)
# 编号目录项：罗马数字（I./II.）、阿拉伯数字（1.）或 Chapter 序号
# （Chapter 1 / CHAPTER X.）开头的行
_GUTENBERG_TOC_ITEM_PATTERN = re.compile(
    r'^\s*(?:[IVXLC]+\.?|\d+\.?|chapter\s+[IVXLC\d]+\.?)(?:\s|$)',
    re.IGNORECASE | re.MULTILINE)


def _strip_epub_front_matter(text: str) -> str:
    """裁剪古腾堡计划 EPUB 头部的许可证公告与目录块，仅保留正文。

    安全策略（不误删正文）：
    - 仅跳过含「古腾堡版权/许可证」特征的块；
    - 仅当某个块是「目录标题（Contents/目录）」且紧随其后含 ≥2 个编号
      目录项的连续块时，才把该目录区域一并跳过；仅在文本前部 30 块内
      探测（防误删正文中的普通编号列表）。
    单处出现的 "I. The Time Machine" 等正文段落不会被误判为目录。
    """
    if not text:
        return text
    blocks = re.split(r'\n\s*\n', text)

    start_idx = 0
    # 1) 跳过连续的首部版权/许可证块
    for i, block in enumerate(blocks):
        if any(p.search(block) for p in _GUTENBERG_HEADER_PATTERNS):
            start_idx = i + 1
        else:
            break
    if start_idx >= len(blocks):
        return ""

    # 2) 目录块：Contents 标题块 + 紧随其后的编号目录项块（≥2 项才裁剪）。
    #    T1：li/table 兜底提取后目录项会进入文本（此前被静默丢弃），沿用
    #    「≥2 项」安全策略；允许 Contents 不在首块（标题页场景）。
    scan_limit = min(len(blocks), start_idx + 30)
    for j in range(start_idx, scan_limit):
        if not _GUTENBERG_TOC_TITLE_PATTERN.search(blocks[j]):
            continue
        k = j + 1
        item_count = 0
        while k < len(blocks):
            matches = len(_GUTENBERG_TOC_ITEM_PATTERN.findall(blocks[k]))
            if matches == 0:
                break
            item_count += matches
            k += 1
        if item_count >= 2:
            blocks = blocks[:j] + blocks[k:]
            break

    return '\n\n'.join(blocks).strip()


def _epub_extract_title(soup, fallback_title: str) -> str:
    """提取章节标题：优先正文标题标签（h1-h3），退回 <title> 或兜底名。

    T1：<br/> 分隔的标题先在标签内替换为空格；get_text 用空格连接片段，
    避免「CHAPTER I.<br/>Down…」粘连成 "I.Down" 或「…eBook of
    <span>Alice's…」粘连成 "ofAlice's"。
    """
    title_tag = soup.find(['h1', 'h2', 'h3']) or soup.find('title')
    if not title_tag:
        return fallback_title
    for br in title_tag.find_all('br'):
        br.replace_with(' ')
    return title_tag.get_text(' ', strip=True)


def _check_title_present(ch_id: str, title: str, text: str) -> None:
    """T1：H1 与正文标题一致性（信息性告警）。

    标题未在正文开头复现时记录 INFO（通常意味着标题来自页眉/元数据，
    属「页眉残留」类问题线索），不阻断切分。
    """
    normalized = _normalize_chapter_title(title)
    if normalized and normalized[:60] not in text[:400]:
        _splitter_log_info(
            f"{ch_id} 标题未在正文再现（可能来自页眉/元数据）: {normalized}")


# ────────────────────────────────────────────────────────────────────
# T1：转换完整性审计（独立 oracle，token 覆盖率）
#
# 设计：oracle 只做最小清洗（去掉脚本/导航/页眉等非正文容器）后原样取
# 文本，不经过 _extract_block_text 的块选择逻辑，避免「同源盲区」（旧的
# 字符数基线用同一套清洗器做对照，静默丢失无法被发现）。
# 豁免（whitelist）：古腾堡许可证特征串、前部「Contents + ≥2 目录项」
# 区域 —— 这些是 Q2 有意裁剪的 front matter，不算丢失。
# ────────────────────────────────────────────────────────────────────

_AUDIT_TOKEN_RE = re.compile(r'[A-Za-z0-9]+|[\u4e00-\u9fff]')
_AUDIT_COVERAGE_THRESHOLD = 0.995
_AUDIT_CHAPTER_FLOOR = 0.90    # 单章最低覆盖率（防整体达标掩盖单章崩坏）
_AUDIT_MAX_MISSING_SPANS = 20
_AUDIT_TOC_SCAN_LINES = 120    # 目录区域仅在前部扫描（防误豁免正文）


def _epub_oracle_text(html_content: str) -> str:
    """独立 oracle 文本：仅去除非正文容器/元数据，其余原样取文本。"""
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'title']):
        tag.decompose()
    return soup.get_text('\n')


def _audit_whitelist_spans(text: str) -> list:
    """返回 oracle 文本中「策略豁免」的字符区间（许可证串 + 目录区域）。"""
    spans = []
    for pat in _GUTENBERG_HEADER_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))

    lines = text.split('\n')
    starts = []
    pos = 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1

    for i in range(min(len(lines), _AUDIT_TOC_SCAN_LINES)):
        if not _GUTENBERG_TOC_TITLE_PATTERN.search(lines[i]):
            continue
        j = i + 1
        items = 0
        end = j
        while j < len(lines):
            ln = lines[j]
            if ln.strip() == '':
                # 空行透明：其后仍紧跟目录项行才继续
                k = j + 1
                while k < len(lines) and lines[k].strip() == '':
                    k += 1
                if k < len(lines) and len(lines[k]) < 120 \
                        and _GUTENBERG_TOC_ITEM_PATTERN.search(lines[k]):
                    j = k
                    continue
                break
            if len(ln) >= 120:
                break
            if _GUTENBERG_TOC_ITEM_PATTERN.search(ln):
                items += 1
                j += 1
                end = j
                continue
            # 续行（如表格 <td> 分列后的标题行）：其后仍紧跟目录项行才纳入
            k = j + 1
            while k < len(lines) and lines[k].strip() == '':
                k += 1
            if k < len(lines) and len(lines[k]) < 120 \
                    and _GUTENBERG_TOC_ITEM_PATTERN.search(lines[k]):
                j = k
                end = j
                continue
            break
        if items >= 2:
            # 末尾续行（表格末行标题后无下一项行）：前一行是目录项则一并纳入
            if end < len(lines):
                tail = lines[end]
                if tail.strip() and len(tail) < 120 \
                        and not _GUTENBERG_TOC_ITEM_PATTERN.search(tail) \
                        and end - 1 >= i and _GUTENBERG_TOC_ITEM_PATTERN.search(lines[end - 1]):
                    end += 1
            spans.append((starts[i], starts[end - 1] + len(lines[end - 1])))
            break
    return spans


def _audit_tokens_with_pos(text: str) -> list:
    return [(m.group(0), m.start()) for m in _AUDIT_TOKEN_RE.finditer(text)]


def run_conversion_audit(
    chapter_sources: dict,
    chapters: list,
    skipped_docs: list,
    output_path,
    threshold: float = _AUDIT_COVERAGE_THRESHOLD,
) -> dict:
    """转换完整性审计：oracle token 多重集 vs md 产出 token 多重集。

    返回 {"ok", "overall", "chapter_results", "missing_spans", "report_path"}。
    低于阈值（或单章低于下限）→ ok=False；报告文件始终写出（可审计）。
    """
    from collections import Counter
    from datetime import datetime

    chapter_results: dict = {}
    missing_spans: list = []
    total_oracle = 0
    total_missing = 0
    total_whitelisted = 0

    for ch_id, md in chapters:
        srcs = chapter_sources.get(ch_id, [])
        oracle_counter: Counter = Counter()
        token_stream: list = []
        whitelisted = 0
        for html in srcs:
            otext = _epub_oracle_text(html)
            wl_spans = _audit_whitelist_spans(otext)
            for tok, pos in _audit_tokens_with_pos(otext):
                if any(s <= pos < e for s, e in wl_spans):
                    whitelisted += 1
                    continue
                oracle_counter[tok] += 1
                token_stream.append((tok, pos, otext))

        md_counter = Counter(_AUDIT_TOKEN_RE.findall(md))
        missing_counter = oracle_counter - md_counter
        missing_n = sum(missing_counter.values())
        total_n = sum(oracle_counter.values())
        coverage = (1.0 - missing_n / total_n) if total_n else 1.0

        # 缺失 span：按 oracle 顺序定位未覆盖 token，合并邻近出现（≤40 字符间隔）
        if missing_n:
            remain = missing_counter.copy()
            runs = []
            cur = None
            prev_text = None
            prev_end = None
            for tok, pos, otext in token_stream:
                if remain.get(tok, 0) <= 0:
                    continue
                remain[tok] -= 1
                end = pos + len(tok)
                if cur is not None and otext is prev_text and pos - prev_end <= 40:
                    cur['end'] = end
                else:
                    if cur is not None:
                        runs.append(cur)
                    cur = {'text': otext, 'start': pos, 'end': end}
                prev_text = otext
                prev_end = end
            if cur is not None:
                runs.append(cur)
            for r in runs:
                s = max(0, r['start'] - 40)
                e = min(len(r['text']), r['end'] + 40)
                excerpt = re.sub(r'\s+', ' ', r['text'][s:e]).strip()
                missing_spans.append({'chapter': ch_id, 'excerpt': excerpt})

        chapter_results[ch_id] = {
            'sources': len(srcs),
            'oracle_tokens': total_n,
            'missing': missing_n,
            'coverage': coverage,
        }
        total_oracle += total_n
        total_missing += missing_n
        total_whitelisted += whitelisted

    overall = (1.0 - total_missing / total_oracle) if total_oracle else 1.0
    min_cov = min((r['coverage'] for r in chapter_results.values()), default=1.0)
    ok = overall >= threshold and min_cov >= _AUDIT_CHAPTER_FLOOR

    report = [
        '# 转换完整性审计报告 (T1)',
        '',
        f'- 生成时间: {datetime.now().isoformat(timespec="seconds")}',
        f'- 阈值: 全书覆盖率 ≥ {threshold * 100:.1f}%；单章下限 {_AUDIT_CHAPTER_FLOOR * 100:.0f}%',
        f'- 全书覆盖率: {overall * 100:.3f}%',
        f'- 策略豁免 token 数（许可证/目录区域）: {total_whitelisted}',
        f'- 结论: {"✅ 通过" if ok else "❌ 未通过"}',
        '',
        '## 逐章覆盖率',
        '',
        '| 章 | 源文档数 | oracle tokens | 缺失 | 覆盖率 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for ch_id, r in chapter_results.items():
        report.append(
            f"| {ch_id} | {r['sources']} | {r['oracle_tokens']} | "
            f"{r['missing']} | {r['coverage'] * 100:.3f}% |")

    report.append('')
    report.append(f'## 未覆盖片段 Top-{_AUDIT_MAX_MISSING_SPANS}')
    report.append('')
    if missing_spans:
        for s in missing_spans[:_AUDIT_MAX_MISSING_SPANS]:
            report.append(f"- [{s['chapter']}] …{s['excerpt']}…")
    else:
        report.append('- （无）')

    report.append('')
    report.append('## 策略豁免说明')
    report.append('')
    report.append('- 古腾堡许可证特征串区域（Project Gutenberg / www.gutenberg.org 等）')
    report.append('- 前部「Contents + ≥2 个编号目录项」区域（Q2 有意裁剪的目录）')
    if skipped_docs:
        report.append('')
        report.append('## 按策略跳过的短文档（<50 字符，不计入审计）')
        report.append('')
        for d in skipped_docs:
            report.append(f"- {d.get('id', '?')} ({d.get('chars', 0)} chars)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(report) + '\n', encoding='utf-8')

    return {
        'ok': ok,
        'overall': overall,
        'chapter_results': chapter_results,
        'missing_spans': missing_spans[:_AUDIT_MAX_MISSING_SPANS],
        'report_path': str(output_path),
    }


def _epub_split_large_doc_by_toc(html_content: str, toc_entries: list, chapter_idx_start: int,
                                 sources_out: Optional[dict] = None,
                                 verse_out: Optional[dict] = None,
                                 lead_out: Optional[list] = None) -> list:
    """
    将单个超大 HTML 文档按 TOC 锚点切分。
    返回 [(chapter_id, markdown), ...]
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    toc_anchors = []
    for entry in toc_entries:
        href = getattr(entry, 'href', '') or ''
        if isinstance(href, str) and '#' in href:
            anchor_id = href.split('#', 1)[1]
            title = getattr(entry, 'title', '') or ''
            toc_anchors.append((title, anchor_id))

    if not toc_anchors:
        return []

    lines = html_content.split('\n')
    anchor_lines = {}
    for i, line in enumerate(lines):
        for title, anchor_id in toc_anchors:
            if f'id="{anchor_id}"' in line or f"id='{anchor_id}'" in line or f'name="{anchor_id}"' in line or f"name='{anchor_id}'" in line:
                if anchor_id not in anchor_lines:
                    anchor_lines[anchor_id] = (i, title)

    sorted_anchors = sorted(anchor_lines.items(), key=lambda x: x[1][0])

    # 接缝回填（Kindle split 格式）：文档开头到第一个锚点之间的正文是物理
    # 切分的上一章续文（如 ~200KB 分片的 index_split_NNN.html），不处理会被
    # 整段丢弃。把这段交给调用方回填到上一章末尾（实测《Why Nations Fail》
    # 此类分片共丢 ~17 万字符）。lead_out 非 None 时收集。
    if lead_out is not None and sorted_anchors:
        lead_html = '\n'.join(lines[:sorted_anchors[0][1][0]])
        if '<head' in lead_html and '</head>' in lead_html:
            lead_html = lead_html.split('</head>', 1)[1]
        lead_soup = BeautifulSoup(lead_html, 'html.parser')
        _mark_verse_elements(lead_soup)
        for br in lead_soup.find_all('br'):
            br.replace_with('\n')
        lead_blocks: list = []
        lead_text = _extract_block_text(lead_soup, lead_blocks)
        lead_text = re.sub(r'[ \t]+', ' ', lead_text)
        lead_text = re.sub(r'\n{3,}', '\n\n', lead_text)
        lead_text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', lead_text)
        lead_text = _strip_epub_front_matter(lead_text.strip())
        if lead_text:
            lead_out.append({'text': lead_text, 'html': lead_html, 'verse': lead_blocks})

    chapters = []
    for i, (anchor_id, (line_no, toc_title)) in enumerate(sorted_anchors):
        start_line = line_no
        end_line = sorted_anchors[i + 1][1][0] if i + 1 < len(sorted_anchors) else len(lines)
        chunk_html = '\n'.join(lines[start_line:end_line])

        title = toc_title or f"Section {i+1}"

        chunk_soup = BeautifulSoup(chunk_html, 'html.parser')
        _mark_verse_elements(chunk_soup)
        for br in chunk_soup.find_all('br'):
            br.replace_with('\n')
        # T1：与 _epub_clean_html_text 同口径（p/标题/pre + 容器兜底）
        verse_blocks: list = []
        text = _extract_block_text(chunk_soup, verse_blocks)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', text)
        text = text.strip()
        # Q2：剔除古腾堡头部许可证/目录，避免非正文进入译文
        text = _strip_epub_front_matter(text)

        if not text or len(text) < 50:
            continue

        chapter_idx = chapter_idx_start + len(chapters) + 1
        ch_id = f"ch{chapter_idx:02d}"
        if sources_out is not None:
            sources_out[ch_id] = [chunk_html]
        if verse_out is not None:
            verse_out[ch_id] = verse_blocks
        chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))

    return chapters


# ── input_prep：切分后自动排除非正文 + 文本伪影修复（config.yaml input_prep 段） ──

def _input_prep_cfg() -> dict:
    """读取 config.yaml 的 input_prep 配置（不可用时返回 {} = 功能关闭）。"""
    try:
        from src.config import get_config as _gc
        return dict(_gc().input_prep or {})
    except Exception:
        return {}


def _prep_key(s: str) -> str:
    """标题归一（忽略大小写/多余空白）供 skip_titles 匹配。"""
    return " ".join((s or "").strip().casefold().split())


def _chapter_h1(md: str) -> str:
    """取章节 Markdown 的 H1 标题（前几行内首个 '# ' 行）。"""
    for line in (md or "").splitlines()[:5]:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _apply_input_repairs(text: str, repairs) -> tuple:
    """按 input_prep.repairs 表顺序替换文本伪影，返回 (新文本, 替换次数)。"""
    n = 0
    for item in repairs or []:
        try:
            old, new = str(item[0]), str(item[1])
        except (TypeError, IndexError, KeyError):
            continue
        if old and old in text:
            n += text.count(old)
            text = text.replace(old, new)
    return text, n


def _epub_extract_chapters(book, sources_out: Optional[dict] = None,
                           skipped_out: Optional[list] = None,
                           verse_out: Optional[dict] = None) -> list:
    """从 EPUB 提取章节，返回 [(chapter_id, markdown), ...]

    sources_out / skipped_out：可选输出（T1 审计用）——每章对应的源
    XHTML 片段 与 被跳过的短文档清单。
    verse_out：可选输出（T-L1）——每章的韵文块记录（供 verse_manifest）。
    """
    chapters = []
    chapter_idx = 0

    toc_entries = []
    try:
        for item in book.toc:
            toc_entries.append(item)
            if isinstance(item, tuple) and len(item) == 2:
                section, sub_items = item
                toc_entries.append(section)
                if isinstance(sub_items, list):
                    toc_entries.extend(sub_items)
    except Exception:
        pass

    LARGE_DOC_THRESHOLD = 50000

    for item_id in book.spine:
        if isinstance(item_id, tuple):
            item_id = item_id[0]
        item = book.get_item_with_id(item_id)
        if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        html_content = item.get_content().decode('utf-8', errors='ignore')
        verse_blocks: list = []
        text = _epub_clean_html_text_ex(html_content, verse_blocks)
        # Q2：剔除古腾堡头部许可证/目录，避免非正文进入译文
        text = _strip_epub_front_matter(text)

        if not text or len(text) < 50:
            if skipped_out is not None:
                skipped_out.append({"id": item.get_name(), "chars": len(text or "")})
            continue

        if len(text) <= LARGE_DOC_THRESHOLD:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title = _epub_extract_title(soup, f"Chapter {chapter_idx}")
            _check_title_present(ch_id, title, text)
            if sources_out is not None:
                sources_out[ch_id] = [html_content]
            if verse_out is not None:
                verse_out[ch_id] = verse_blocks
            chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))
            continue

        lead_box: list = []
        sub_chapters = _epub_split_large_doc_by_toc(html_content, toc_entries, chapter_idx,
                                                    sources_out=sources_out,
                                                    verse_out=verse_out,
                                                    lead_out=lead_box)
        if sub_chapters:
            if lead_box and chapters:
                lead = lead_box[0]
                if len(lead.get('text') or '') >= 40:
                    prev_id, prev_md = chapters[-1]
                    chapters[-1] = (prev_id, prev_md.rstrip() + '\n\n' + lead['text'])
                    if sources_out is not None:
                        sources_out.setdefault(prev_id, []).append(lead['html'])
                    if verse_out is not None and lead.get('verse'):
                        verse_out.setdefault(prev_id, []).extend(lead['verse'])
                    _splitter_log_info(
                        f"接缝回填：{len(lead['text'])} 字符续文 → {prev_id}")
            chapters.extend(sub_chapters)
            chapter_idx += len(sub_chapters)
        else:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title = _epub_extract_title(soup, f"Chapter {chapter_idx}")
            _check_title_present(ch_id, title, text)
            if sources_out is not None:
                sources_out[ch_id] = [html_content]
            if verse_out is not None:
                verse_out[ch_id] = verse_blocks
            chapters.append((ch_id, f"# {_normalize_chapter_title(title)}\n\n{text}"))

    return chapters


class EpubSplitter:
    """EPUB → 章节 Markdown 切分器

    用法:
        splitter = EpubSplitter("book.epub", "input/")
        chapters = splitter.split()  # 返回 [(ch_id, md_text), ...]
        splitter.write_files()     # 写入磁盘
    """

    def __init__(self, epub_path: str, output_dir: str):
        if not _EPUB_DEPS_OK:
            raise RuntimeError(
                f"❌ 缺少依赖: {_EPUB_DEPS_ERROR}\n"
                "请先安装: pip install ebooklib beautifulsoup4 lxml"
            )
        self.epub_path = Path(epub_path)
        self.output_dir = Path(output_dir)
        self.book = None
        self.chapters: list = []
        self._chapter_sources: dict = {}
        self._skipped_docs: list = []
        self._chapter_verse: dict = {}
        self.verse_manifest: dict = {}
        self.audit_result: dict = {}
        self._excluded_ids: set = set()

    def load(self):
        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB 文件不存在: {self.epub_path}")
        _splitter_log_info(f"读取 EPUB: {self.epub_path}")
        self.book = _epub_lib.read_epub(str(self.epub_path))

        title = self.book.get_metadata('DC', 'title')
        author = self.book.get_metadata('DC', 'creator')
        if title:
            _splitter_log_info(f"书名: {title[0][0]}")
        if author:
            _splitter_log_info(f"作者: {author[0][0]}")
        return self

    def split(self) -> list:
        """切分章节，返回 [(ch_id, markdown), ...]"""
        if self.book is None:
            self.load()
        self.chapters = _epub_extract_chapters(
            self.book, sources_out=self._chapter_sources, skipped_out=self._skipped_docs,
            verse_out=self._chapter_verse)
        self._apply_input_prep()
        if not self.chapters:
            _splitter_log_warn("未提取到任何章节内容")
        else:
            _splitter_log_ok(f"共提取 {len(self.chapters)} 章")
            self._validate_split()
            _kept = [(cid, md) for cid, md in self.chapters
                     if cid not in self._excluded_ids]
            self.verse_manifest = _build_verse_manifest(_kept, self._chapter_verse)
            n_verse = sum(len(v or []) for v in (self.verse_manifest.get('chapters') or {}).values())
            if n_verse:
                _splitter_log_info(
                    f"韵文通道：检测到 {n_verse} 个韵文块（verse_manifest 随 write_files 落盘）")
        return self.chapters

    def _apply_input_prep(self):
        """input_prep：文本伪影修复 + 非正文标记（写盘时入 skip_dir，不参与翻译）。

        被排除章节仍保留在 self.chapters（完整性审计口径不变）。
        """
        cfg = _input_prep_cfg()
        if not cfg or cfg.get("enabled", True) is False:
            return
        repairs = cfg.get("repairs") or []
        if repairs:
            total = 0
            fixed = []
            for ch_id, md in self.chapters:
                md, n = _apply_input_repairs(md, repairs)
                total += n
                fixed.append((ch_id, md))
            self.chapters = fixed
            if total:
                _splitter_log_info(f"input_prep：文本伪影修复 {total} 处")
        skip_titles = {_prep_key(t) for t in (cfg.get("skip_titles") or []) if t}
        self._excluded_ids = set()
        if not skip_titles:
            return
        labels = []
        for ch_id, md in self.chapters:
            title = _chapter_h1(md)
            if title and _prep_key(title) in skip_titles:
                self._excluded_ids.add(ch_id)
                labels.append(f"{ch_id}（{title}）")
        if self._excluded_ids:
            _splitter_log_info(
                f"input_prep：排除 {len(self._excluded_ids)} 个非正文文档"
                f"（写 {cfg.get('skip_dir') or '_skip'}/，不翻译）: " + "、".join(labels))

    def _validate_split(self):
        """P4：切分完整性 + 段落结构回归校验。

        - T1：完整性改用独立 oracle token 覆盖率审计（run_conversion_audit），
          替换原「同源清洗器字符数」基线（该基线与被测逻辑同源，存在盲区）。
        - 段落结构：每章按空行拆出的段落数应>1（除非原文确实极短），
          防止再次出现"整章塌成 1 段"的回归。
        """
        try:
            self.audit_result = run_conversion_audit(
                self._chapter_sources, self.chapters, self._skipped_docs,
                self.output_dir / "_conversion_audit.md",
            )
            if self.audit_result["ok"]:
                _splitter_log_ok(
                    "转换完整性审计通过：覆盖率 "
                    f"{self.audit_result['overall'] * 100:.2f}% "
                    f"(报告: {self.audit_result['report_path']})")
            else:
                _splitter_log_err(
                    "转换完整性审计未通过：覆盖率 "
                    f"{self.audit_result['overall'] * 100:.3f}% "
                    f"(阈值 {_AUDIT_COVERAGE_THRESHOLD * 100:.1f}%)，"
                    f"详见 {self.audit_result['report_path']}")
        except Exception as e:
            # 审计异常按「未通过」处理（fail-closed，防盲区放行）
            self.audit_result = {"ok": False, "overall": 0.0,
                                 "report_path": "", "error": str(e)}
            _splitter_log_err(f"转换完整性审计异常（按未通过处理）: {e}")

        try:
            single_para = 0
            for ch_id, md in self.chapters:
                paras = [p for p in re.split(r'\n{2,}', md) if p.strip()]
                if len(paras) <= 1 and len(md) > 500:
                    single_para += 1
            if single_para > 0:
                _splitter_log_warn(
                    f"段落结构预警：{single_para} 章疑似仍塌缩为单段落（>500字符却无空行分段），请检查切分逻辑"
                )
            else:
                _splitter_log_ok("段落结构校验通过：各章均含多段落边界")
        except Exception as e:
            _splitter_log_warn(f"校验跳过（非致命）：{e}")

    def write_files(self) -> list:
        """将切分结果写入 output_dir/chXX.md，返回「参与翻译」的文件路径列表

        input_prep 标出的非正文文档改写到 output_dir/<skip_dir>/（不计入返回列表，
        CLI/mapfile 与翻译流程天然跳过）；同时清理上次切分遗留的陈旧 chXX.md。
        """
        if not self.chapters:
            self.split()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _cfg = _input_prep_cfg()
        _skip_dir_name = str((_cfg or {}).get("skip_dir") or "_skip")

        generated = []
        pairs = []  # [(ch_id, path)] 参与翻译的章节（供 summary 使用）
        for ch_id, markdown in self.chapters:
            if ch_id in self._excluded_ids:
                skip_dir = self.output_dir / _skip_dir_name
                skip_dir.mkdir(parents=True, exist_ok=True)
                (skip_dir / f"{ch_id}.md").write_text(markdown, encoding='utf-8')
                _splitter_log_info(
                    f"非正文排除: {_skip_dir_name}/{ch_id}.md（{len(markdown)} chars，不参与翻译）")
                continue
            out_file = self.output_dir / f"{ch_id}.md"
            out_file.write_text(markdown, encoding='utf-8')
            generated.append(str(out_file))
            pairs.append((ch_id, str(out_file)))
            _splitter_log_ok(f"{ch_id}.md ({len(markdown)} chars)")

        # 清理陈旧章节文件（同目录重切时的残留，如曾被排除但仍在根目录的 chXX）
        if _cfg and _cfg.get("enabled", True) is not False:
            root_kept = {ch_id for ch_id, _ in pairs}
            for stale in self.output_dir.glob("ch*.md"):
                if re.fullmatch(r'ch\d+', stale.stem) and stale.stem not in root_kept:
                    try:
                        stale.unlink()
                        _splitter_log_info(f"清理陈旧章节: {stale.name}")
                    except OSError:
                        pass
            skip_dir = self.output_dir / _skip_dir_name
            if skip_dir.is_dir():
                for stale in skip_dir.glob("ch*.md"):
                    if re.fullmatch(r'ch\d+', stale.stem) and stale.stem not in self._excluded_ids:
                        try:
                            stale.unlink()
                            _splitter_log_info(f"清理陈旧章节: {_skip_dir_name}/{stale.name}")
                        except OSError:
                            pass

        summary = self.output_dir / "_chapters_summary.txt"
        summary.write_text(
            "\n".join(
                f"{ch_id}\t{Path(f).stat().st_size} bytes"
                for ch_id, f in pairs
            ),
            encoding='utf-8'
        )
        _splitter_log_info(f"汇总: {summary}")

        if self.verse_manifest:
            manifest_file = self.output_dir / "verse_manifest.json"
            manifest_file.write_text(
                json.dumps(self.verse_manifest, ensure_ascii=False, indent=2),
                encoding='utf-8')
            n_blocks = sum(
                len(v or []) for v in (self.verse_manifest.get('chapters') or {}).values())
            _splitter_log_info(f"韵文清单: {manifest_file}（{n_blocks} 个韵文块）")
        return generated


# ────────────────────────────────────────────────────────────────────
# 11.2 TextSplitter — 纯文本 → 章节 Markdown
# ─────────────────────────────────────────────────────────────────────────────

# 章节标题正则（按优先级匹配）
# 支持中英文常见模式
_CHAPTER_TITLE_PATTERNS = [
    re.compile(r'^\s*(?:Chapter|CHAPTER|第[一二三四五六七八九十百千零\d]+章|序章|楔子|引子|前言|后记|尾声)\s*[\s\d.、:：\-—]*[^\n]*$', re.IGNORECASE),
    re.compile(r'^\s*\[(?:CHAPTER|Chapter|第[一二三四五六七八九十百千零\d]+章)[^\]]*\]\s*$', re.IGNORECASE),
    re.compile(r'^\s*#{1,2}\s*[^\n]+$'),  # Markdown H1/H2
]


def _text_detect_chapter_title(line: str) -> bool:
    """判断一行是否为章节标题"""
    if len(line) > 80:
        return False
    for pat in _CHAPTER_TITLE_PATTERNS:
        if pat.match(line):
            return True
    return False


class TextSplitter:
    """纯文本 → 章节 Markdown 切分器

    策略（按优先级）:
      1. 检测显式章节标记（"Chapter N" / "第N章" / Markdown H1/H2）
      2. 若未找到任何标记，按 target_chars 均分
      3. 段落内连续空行作为段落边界

    参数:
      text_path: 输入 TXT 文件路径
      output_dir: 输出目录
      target_chars: 均分模式下的目标章节字数（默认 5000）
      min_chars: 短于该字数的章节会被合并到下一章（默认 1000）
    """

    def __init__(self, text_path: str, output_dir: str,
                 target_chars: int = 5000, min_chars: int = 1000):
        self.text_path = Path(text_path)
        self.output_dir = Path(output_dir)
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.chapters: list = []

    def load_text(self) -> str:
        if not self.text_path.exists():
            raise FileNotFoundError(f"文本文件不存在: {self.text_path}")
        _splitter_log_info(f"读取文本: {self.text_path}")
        text = self.text_path.read_text(encoding='utf-8', errors='ignore')
        _splitter_log_info(f"总字符数: {len(text)}")
        return text

    def _find_title_lines(self, text: str) -> list:
        """扫描全文，定位所有章节标题行号"""
        title_lines = []
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if _text_detect_chapter_title(line):
                title_lines.append((i, line.strip()))
        return title_lines, lines

    def split(self) -> list:
        """切分章节，返回 [(ch_id, markdown), ...]"""
        text = self.load_text()

        title_lines, lines = self._find_title_lines(text)

        if title_lines:
            return self._split_by_titles(text, title_lines, lines)
        else:
            _splitter_log_warn("未发现显式章节标记，按字数均分")
            return self._split_by_size(text)

    def _split_by_titles(self, text: str, title_lines: list, lines: list) -> list:
        """按显式章节标记切分"""
        chapters = []
        for i, (line_no, title) in enumerate(title_lines):
            start_line = line_no
            end_line = title_lines[i + 1][0] if i + 1 < len(title_lines) else len(lines)
            body = '\n'.join(lines[start_line:end_line]).strip()

            if not body or len(body) < 50:
                continue

            chapter_idx = len(chapters) + 1
            ch_id = f"ch{chapter_idx:02d}"
            markdown = f"# {_normalize_chapter_title(title)}\n\n{body}"
            chapters.append((ch_id, markdown))

        # 若显式切分后章节太少（或每章过大），再尝试二次均分
        if len(chapters) < 2 and len(text) > self.target_chars * 2:
            _splitter_log_warn("显式标记过少，启用均分兜底")
            return self._split_by_size(text)

        return chapters

    def _split_by_size(self, text: str) -> list:
        """按目标字数均分"""
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if not paragraphs:
            return []

        chapters = []
        current_paragraphs = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)

            # 若当前章节非空、且加入新段会超 target_chars 且当前已达 min_chars，则打包
            if current_paragraphs and current_len + p_len > self.target_chars and current_len >= self.min_chars:
                chapter_text = '\n\n'.join(current_paragraphs)
                chapter_idx = len(chapters) + 1
                ch_id = f"ch{chapter_idx:02d}"
                chapters.append((ch_id, f"# Part {chapter_idx}\n\n{chapter_text}"))
                current_paragraphs = []
                current_len = 0

            current_paragraphs.append(p)
            current_len += p_len

        # 收尾
        if current_paragraphs:
            chapter_text = '\n\n'.join(current_paragraphs)
            chapter_idx = len(chapters) + 1
            ch_id = f"ch{chapter_idx:02d}"
            chapters.append((ch_id, f"# Part {chapter_idx}\n\n{chapter_text}"))

        return chapters

    def write_files(self) -> list:
        if not self.chapters:
            self.chapters = self.split()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for ch_id, markdown in self.chapters:
            out_file = self.output_dir / f"{ch_id}.md"
            out_file.write_text(markdown, encoding='utf-8')
            generated.append(str(out_file))
            _splitter_log_ok(f"{ch_id}.md ({len(markdown)} chars)")

        summary = self.output_dir / "_chapters_summary.txt"
        summary.write_text(
            "\n".join(
                f"{ch_id}\t{Path(f).stat().st_size} bytes"
                for ch_id, f in zip([c[0] for c in self.chapters], generated)
            ),
            encoding='utf-8'
        )
        _splitter_log_info(f"汇总: {summary}")
        return generated


# ────────────────────────────────────────────────────────────────────
# 11.3 MdSplitter — 整本 Markdown → 多章节（均分 or 按 H1）
# ─────────────────────────────────────────────────────────────────────────────

class MdSplitter:
    """整本 Markdown → 多章节切分器

    策略:
      1. 按 H1 标题切分
      2. 若无 H1，则按 H2 切分
      3. 若无任何标题，按 target_chars 均分

    用于：用户已有整本 Markdown（如从 EPUB/TXT 转出来但没分章节）
    """

    def __init__(self, md_path: str, output_dir: str,
                 target_chars: int = 5000, min_chars: int = 1000):
        self.md_path = Path(md_path)
        self.output_dir = Path(output_dir)
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.chapters: list = []

    def load(self) -> str:
        if not self.md_path.exists():
            raise FileNotFoundError(f"MD 文件不存在: {self.md_path}")
        _splitter_log_info(f"读取 MD: {self.md_path}")
        text = self.md_path.read_text(encoding='utf-8', errors='ignore')
        _splitter_log_info(f"总字符数: {len(text)}")
        return text

    def split(self) -> list:
        text = self.load()

        # 按 H1 切分
        parts = re.split(r'(?m)^# .+$', text)
        headings = re.findall(r'(?m)^# (.+)$', text)

        if len(parts) > 1 and headings:
            chapters = []
            for i, (heading, body) in enumerate(zip(headings, parts[1:])):
                body = body.strip()
                if not body or len(body) < 50:
                    continue
                chapter_idx = len(chapters) + 1
                ch_id = f"ch{chapter_idx:02d}"
                chapters.append((ch_id, f"# {heading.strip()}\n\n{body}"))
            if chapters:
                return chapters

        # 兜底：复用 TextSplitter 的均分逻辑
        _splitter_log_warn("未发现 H1 标题，启用均分兜底")
        fake_path = self.md_path.with_suffix('.txt')
        fake_text = TextSplitter(
            str(fake_path), str(self.output_dir),
            target_chars=self.target_chars, min_chars=self.min_chars
        )
        # 直接借用均分方法
        return fake_text._split_by_size(text)

    def write_files(self) -> list:
        if not self.chapters:
            self.chapters = self.split()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        generated = []
        for ch_id, markdown in self.chapters:
            out_file = self.output_dir / f"{ch_id}.md"
            out_file.write_text(markdown, encoding='utf-8')
            generated.append(str(out_file))
            _splitter_log_ok(f"{ch_id}.md ({len(markdown)} chars)")

        summary = self.output_dir / "_chapters_summary.txt"
        summary.write_text(
            "\n".join(
                f"{ch_id}\t{Path(f).stat().st_size} bytes"
                for ch_id, f in zip([c[0] for c in self.chapters], generated)
            ),
            encoding='utf-8'
        )
        _splitter_log_info(f"汇总: {summary}")
        return generated


# ────────────────────────────────────────────────────────────────────
