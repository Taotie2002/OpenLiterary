"""成书样式单源化 + lint（Round 119 T5）。

事实源：本文件；整合副本：src/translator_agent.py（逐字镜像，见
tests/test_round119_mirror_parity.py）。

职责:
  - normalize_final_style  引号统一（“”）/ 脚注制式统一（[^n]）/
    注释区块标题行与策略元语清除 / 悬空定义转章末译者注（报告留痕）；
  - finalize_outputs       对 output/ch*/*_final.json 批量执行（dry-run/apply）；
  - lint_texts / lint_book 成书级 lint（finals + merged 双跑）；
  - postprocess_chunk_text merge 前处理（页眉/版本行/内嵌章节标题剥离）；
  - chapter_heading        merge 层章节标题（bilingual|zh|keep 三制式）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SUPER = "¹²³⁴⁵⁶⁷⁸⁹"

# 注释区块标题行（"**脚注：**" / "**【脚注】**" / "**脚注（…策略）**" /
# "**脚注**（典故保留并加注）：" 等）。仅匹配"整行即标题"的形态。
_FOOTNOTE_HEADER_RE = re.compile(
    r'^\s*\*{0,2}\s*【?脚注】?\s*\*{0,2}'
    r'(?:\s*[（(][^）)\n]{0,48}[）)])?'
    r'\s*\*{0,2}\s*[：:]?\s*\*{0,2}\s*$'
)
_SUP_DEF_RE = re.compile(r'^\s*([¹²³⁴⁵⁶⁷⁸⁹]+)\s+(\S.*)$')
_MD_DEF_RE = re.compile(r'^\s*\[\^([^\]]+)\]:\s?(.*)$')
_NUM_DEF_RE = re.compile(r'^\s*(\d+)\.\s+(\S.*)$')
_NUM_DEF_START_RE = re.compile(r'^[“”‘’「『（【*\u4e00-\u9fffA-Za-z]')
_SUP_REF_RE = re.compile(r'[¹²³⁴⁵⁶⁷⁸⁹]+')
_MD_REF_RE = re.compile(r'\[\^([^\]]+)\](?!:)')

# 策略元语禁用表（T5/D5）
_META_TOKENS = ["保留与加注策略", "保留并加注", "策略池", "RETAIN_AND_ANNOTATE",
                "加注策略", "保留音译", "按约束", "直译自"]

# merge 前处理：页眉/版本行（对齐真实字符串，含中英两制式）
_FRONTMATTER_LINE_RE = re.compile(
    r'^\s*#?\s*(?:The Project Gutenberg eBook\b.*|THE MILLENNIUM FULCRUM EDITION\b.*'
    r'|古腾堡计划.*|千[禧年]*年支点版.*|目录|Contents)\s*$')
_CHAPTER_HEAD_RE = re.compile(r'^\s*#?\s*第[一二三四五六七八九十百千零〇\d]+\s*[章.。．]')

# merged 页眉残留（lint）
_FRONTMATTER_RESIDUE_RES = [
    ("Gutenberg 英文残留", re.compile(r'The Project Gutenberg eBook')),
    ("Gutenberg 中文残留", re.compile(r'古腾堡')),
    ("版本行残留", re.compile(r'MILLENNIUM FULCRUM|千[禧年]*年支点版')),
    ("标题粘连残留", re.compile(r'ofAlice')),
    ("章节标题缺空格", re.compile(r'CHAPTER\s+[IVXLCDM]+\.(?=[A-Z])')),
]
# 悬空承诺（D9 类，先报不修）
_DANGLING_RE = re.compile(r'[：:]\s*——\s*$')


def sup_run_to_int(run: str) -> int:
    """上标数字串 → 整数（"¹²" → 12）。"""
    return int(''.join(str(_SUPER.index(c) + 1) for c in run))


def _scan_lines(text: str):
    """一次扫描：行分类 + 注释定义 + 引用集合。

    Returns:
        (items, defs, refs)
        items: [('header', None) | ('def', def_dict) | ('body', line)]
        defs:  [{'style': 'sup'|'md'|'num', 'id': str, 'content': str}]
        refs:  set[str]（正文中的 [^id] 与上标引用，统一为字符串 id）
    """
    items = []
    defs = []
    header_seen = False
    for line in text.split('\n'):
        if _FOOTNOTE_HEADER_RE.match(line):
            header_seen = True
            items.append(('header', None))
            continue
        m = _SUP_DEF_RE.match(line)
        if m:
            d = {'style': 'sup', 'id': str(sup_run_to_int(m.group(1))), 'content': m.group(2)}
            defs.append(d)
            items.append(('def', d))
            continue
        m = _MD_DEF_RE.match(line)
        if m:
            d = {'style': 'md', 'id': m.group(1), 'content': m.group(2)}
            defs.append(d)
            items.append(('def', d))
            continue
        if header_seen:
            m = _NUM_DEF_RE.match(line)
            if m and _NUM_DEF_START_RE.match(m.group(2)):
                d = {'style': 'num', 'id': m.group(1), 'content': m.group(2)}
                defs.append(d)
                items.append(('def', d))
                continue
        items.append(('body', line))
    body_text = '\n'.join(p for k, p in items if k == 'body')
    refs = set()
    for mm in _SUP_REF_RE.finditer(body_text):
        refs.add(str(sup_run_to_int(mm.group(0))))
    for mm in _MD_REF_RE.finditer(body_text):
        refs.add(mm.group(1))
    return items, defs, refs


def normalize_final_style(text: str):
    """T5 规范化 pass（幂等）。Returns: (new_text, changes dict)。"""
    changes = {"headers_removed": 0, "sup_defs_converted": 0, "md_defs_kept": 0,
               "orphans_relocated": 0, "cjk_quotes": 0, "straight_quotes": 0}
    items, defs, refs = _scan_lines(text)
    out = []
    relocated = []
    for kind, payload in items:
        if kind == 'header':
            changes["headers_removed"] += 1
            continue
        if kind == 'def':
            d = payload
            if d['style'] == 'num':
                relocated.append(d['content'])
                continue
            if d['id'] in refs:
                out.append("[^" + str(d['id']) + "]: " + d['content'])
                if d['style'] == 'sup':
                    changes["sup_defs_converted"] += 1
                else:
                    changes["md_defs_kept"] += 1
            else:
                relocated.append(d['content'])
            continue
        line = payload
        line = _SUP_REF_RE.sub(lambda m: "[^" + str(sup_run_to_int(m.group(0))) + "]", line)
        out.append(line)
    if relocated:
        changes["orphans_relocated"] = len(relocated)
        while out and out[-1].strip() == '':
            out.pop()
        if out and out[-1].strip() == '---':
            out.pop()
        out.append('')
        for content in relocated:
            out.append('> **译者注：** ' + content)
    t = '\n'.join(out)
    if '「' in t or '」' in t:
        changes["cjk_quotes"] += t.count('「') + t.count('」')
        t = t.replace('「', '“').replace('」', '”')
    if '『' in t or '』' in t:
        changes["cjk_quotes"] += t.count('『') + t.count('』')
        t = t.replace('『', '‘').replace('』', '’')
    if '"' in t:
        changes["straight_quotes"] = t.count('"')
        parts = t.split('"')
        rebuilt = parts[0]
        for k, p in enumerate(parts[1:]):
            rebuilt += ('“' if k % 2 == 0 else '”') + p
        t = rebuilt
    return t, changes


def _find_finals(out_root: Path):
    return sorted(out_root.glob("ch*/*_final.json"))


def finalize_outputs(project_root=None, output_dir=None, apply: bool = False) -> dict:
    """对全部 final.json 执行 normalize（默认 dry-run），写 output/finalize_diff.md。"""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    finals = _find_finals(out_root)
    stats = {"files": len(finals), "changed": 0, "headers_removed": 0,
             "sup_defs_converted": 0, "md_defs_kept": 0, "orphans_relocated": 0,
             "cjk_quotes": 0, "straight_quotes": 0}
    details = []
    for fp in finals:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            print(f"  ⚠️ [Finalize] 跳过损坏 JSON: {fp}")
            continue
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        new_text, ch = normalize_final_style(text)
        if new_text == text:
            continue
        stats["changed"] += 1
        for k in ("headers_removed", "sup_defs_converted", "md_defs_kept",
                  "orphans_relocated", "cjk_quotes", "straight_quotes"):
            stats[k] += ch[k]
        details.append((str(fp.relative_to(root)), ch))
        if apply and isinstance(data, dict):
            data["text"] = new_text
            meta = data.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            meta["style_normalized"] = True
            data["metadata"] = meta
            fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 成书样式规范化报告 (finalize_diff)", ""]
    lines.append(f"- 模式: {'APPLY (已写入)' if apply else 'DRY RUN (仅报告)'}")
    lines.append(f"- 扫描 final.json: {stats['files']} 个；需要变更: {stats['changed']} 个")
    lines.append(f"- 注释标题行清除: {stats['headers_removed']}；上标定义转 [^n]: {stats['sup_defs_converted']}")
    lines.append(f"- 悬空定义转译者注: {stats['orphans_relocated']}；「」引号: {stats['cjk_quotes']}；直引号: {stats['straight_quotes']}")
    lines.append("")
    if details:
        lines.append("## 明细")
        lines.append("")
        for path, ch in details:
            lines.append(f"- `{path}`: " + ", ".join(f"{k}={v}" for k, v in ch.items() if v))
    else:
        lines.append("（无需变更，样式已统一）")
    lines.append("")
    lines.append("---")
    lines.append("*由 finalize 子命令自动生成*")
    lines.append("")
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "finalize_diff.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"stats": stats, "report": str(report_path), "applied": bool(apply)}


def _def_key(content: str) -> str:
    m = re.match(r'^\s*\*\*(.+?)\*\*', content)
    if m:
        return m.group(1).strip()
    m = re.match(r'^\s*[“「](.+?)[”」]', content)
    if m:
        return m.group(1).strip()
    return content.strip()[:12]


def _audit_text(text: str) -> dict:
    """单文本 lint 审计（返回 problems 与统计）。"""
    problems = []
    straight = text.count('"')
    if straight:
        problems.append(f"直引号 {straight} 处")
    cjk_q = sum(text.count(c) for c in '「」『』')
    if cjk_q:
        problems.append(f"「」/『』引号 {cjk_q} 处")
    items, defs, refs = _scan_lines(text)
    sup_defs = [d for d in defs if d['style'] == 'sup']
    if sup_defs:
        problems.append(f"上标脚注定义 {len(sup_defs)} 行（未转 [^n]）")
    def_ids = {d['id'] for d in defs}
    orphans = [d for d in defs if d['id'] not in refs]
    if orphans:
        problems.append(f"悬空脚注定义 {len(orphans)} 条: {[d['id'] for d in orphans]}")
    dangling_refs = sorted(refs - def_ids)
    meta = sum(text.count(t) for t in _META_TOKENS)
    if meta:
        problems.append(f"策略元语 {meta} 处")
    keys = {}
    dups = []
    for d in defs:
        k = _def_key(d['content'])
        if k in keys and keys[k] != d['content']:
            dups.append(k)
        keys[k] = d['content']
    if dups:
        problems.append(f"同键重复注释: {dups}")
    return {"problems": problems, "defs": defs, "orphan_count": len(orphans),
            "dangling_refs": dangling_refs, "meta_count": meta}


def lint_texts(items) -> dict:
    """items: [(label, text), ...] → {'ok': bool, 'failures': [...], 'warnings': [...]}"""
    failures = []
    warnings = []
    for label, text in items:
        a = _audit_text(text)
        for p in a["problems"]:
            failures.append(f"{label}: {p}")
        if a["dangling_refs"]:
            warnings.append(f"{label}: 无定义引用 {a['dangling_refs']}")
    return {"ok": not failures, "failures": failures, "warnings": warnings}


def _chapter_dup_notes(entries) -> list:
    """entries: [(label, text)]（同章）。返回重复注 failures/warnings。"""
    failures, warnings = [], []
    seen = {}
    for label, text in entries:
        _items, defs, _refs = _scan_lines(text)
        for d in defs:
            k = _def_key(d['content'])
            if not k:
                continue
            if k in seen:
                prev_label, prev_content = seen[k]
                if prev_content == d['content']:
                    failures.append(f"重复注（同键同文）: {k} @ {prev_label} / {label}")
                else:
                    import difflib as _dl
                    r = _dl.SequenceMatcher(None, prev_content, d['content']).ratio()
                    if r >= 0.8:
                        failures.append(f"重复注（同键近似 r={r:.2f}）: {k} @ {prev_label} / {label}")
                    else:
                        warnings.append(f"同键不同注: {k} @ {prev_label} / {label}")
            else:
                seen[k] = (label, d['content'])
    return failures, warnings


def _book_style_config() -> dict:
    try:
        from src.config import config as _cfg
        data = _cfg.get("book", {}) if hasattr(_cfg, "get") else {}
        if data:
            return data
    except Exception:
        pass
    try:
        from src.config import get_config as _get_config
        return _get_config().get("book", {}) or {}
    except Exception:
        return {}


def lint_book(project_root=None, output_dir=None, merged_path=None) -> dict:
    """成书级 lint：finals（逐 chunk）+ merged（整书）双跑；写 output/book_lint.md。"""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    merged = Path(merged_path) if merged_path else out_root / "translated_full.md"
    failures, warnings = [], []
    stats = {"finals": 0}

    by_chapter = {}
    for fp in _find_finals(out_root):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            failures.append(f"{fp.name}: JSON 解析失败")
            continue
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        label = f"{fp.parent.name}/{fp.name}"
        stats["finals"] += 1
        by_chapter.setdefault(fp.parent.name, []).append((label, text))
        a = _audit_text(text)
        for p in a["problems"]:
            failures.append(f"{label}: {p}")
        if a["dangling_refs"]:
            warnings.append(f"{label}: 无定义引用 {a['dangling_refs']}")
    for ch, entries in sorted(by_chapter.items()):
        df, dw = _chapter_dup_notes(entries)
        failures.extend(f"{ch}: {x}" for x in df)
        warnings.extend(f"{ch}: {x}" for x in dw)

    if merged.exists():
        mtext = merged.read_text(encoding="utf-8")
        for name, pat in _FRONTMATTER_RESIDUE_RES:
            n = len(pat.findall(mtext))
            if n:
                failures.append(f"merged: {name} {n} 处")
        style = str(_book_style_config().get("chapter_titles", "bilingual"))
        headings = [ln.strip() for ln in mtext.split('\n') if ln.startswith('# ')]
        stats["headings"] = len(headings)
        if style in ("bilingual", "zh"):
            for h in headings[1:]:
                if not re.match(r'^#\s*第[一二三四五六七八九十百千零〇\d]+\s*章', h):
                    failures.append(f"merged: 章节标题不符合制式({style}): {h[:40]}")
        for h in headings:
            for name, pat in _FRONTMATTER_RESIDUE_RES:
                if pat.search(h):
                    failures.append(f"merged: 标题含残留（{name}）: {h[:40]}")
        dangling = [ln.strip()[:60] for ln in mtext.split('\n') if _DANGLING_RE.search(ln)]
        for d in dangling:
            warnings.append(f"merged: 疑似悬空承诺: {d}")
        a = _audit_text(mtext)
        for p in a["problems"]:
            failures.append(f"merged: {p}")
        machine_ids = _MACHINE_FN_ID_RE.findall(mtext)
        if machine_ids:
            failures.append(
                f"merged: 机器脚注 id 残留 {len(machine_ids)} 处（应先跑 notes-normalize）")
        for label, body in _split_h1_sections(mtext):
            dups = _chapter_note_dupes(body)
            for d in dups[:5]:
                failures.append(f"merged/{label}: {d}")
            if len(dups) > 5:
                failures.append(f"merged/{label}: 另有 {len(dups) - 5} 组重复注未列出")
    else:
        warnings.append("merged 未生成（translated_full.md 缺失），跳过整书级检查")

    result = {"ok": not failures, "failures": failures, "warnings": warnings, "stats": stats}
    lines = ["# 成书 lint 报告 (book_lint)", ""]
    lines.append(f"- 结论: {'✅ 全绿' if result['ok'] else '❌ 未通过'}")
    lines.append(f"- 扫描 final.json: {stats['finals']} 个")
    lines.append("")
    if failures:
        lines.append(f"## 失败项 ({len(failures)})")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")
    if warnings:
        lines.append(f"## 警告项 ({len(warnings)})")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("---")
    lines.append("*由 lint 子命令自动生成*")
    lines.append("")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "book_lint.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def postprocess_chunk_text(text: str, chapter_id: str, chunk_idx: int) -> str:
    """merge 前处理：剥离页眉/版本行残留 + chunk 内嵌章节标题行。"""
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if _FRONTMATTER_LINE_RE.match(s):
            continue
        if _CHAPTER_HEAD_RE.match(s):
            continue
        out.append(line)
    return '\n'.join(out)


def chapter_heading(project_root, chapter_id: str, style: str = None) -> str:
    """merge 层章节标题。默认 config.book.chapter_titles（bilingual）。

    bilingual: "第一章 掉进兔子洞（Down the Rabbit-Hole）"
    zh:        "第一章 掉进兔子洞"
    keep:      input/<ch>.md 原始 H1（清理 Gutenberg 前缀）
    """
    root = Path(project_root)
    if style is None:
        style = str(_book_style_config().get("chapter_titles", "bilingual"))
    orig = ""
    md = root / "input" / f"{chapter_id}.md"
    if md.exists():
        for line in md.read_text(encoding="utf-8").split('\n')[:10]:
            s = line.strip()
            if s.startswith('# '):
                orig = s[2:].strip()
                break
    orig_clean = re.sub(r'^The Project Gutenberg eBook of\s*', '', orig)
    zh = ""
    out_dir = root / "output" / chapter_id
    firsts = sorted(out_dir.glob("*_final.json"))
    if firsts:
        try:
            data = json.loads(firsts[0].read_text(encoding="utf-8"))
            text = data.get("text", "") if isinstance(data, dict) else str(data)
        except Exception:
            text = ""
        for line in text.split('\n')[:20]:
            m = re.match(r'^#?\s*(第[一二三四五六七八九十百千零〇\d]+\s*[章.。．]\s*\S.*)$', line.strip())
            if m and len(m.group(1)) <= 60:
                zh = m.group(1).strip().rstrip('。．.')
                zh = re.sub(r'\[\^[^\]]+\]', '', zh)
                break
    if style == "keep":
        return orig_clean or chapter_id
    if style == "zh":
        return zh or orig_clean or chapter_id
    orig_short = re.sub(r'^CHAPTER\s+[IVXLCDM]+\.\s*', '', orig_clean)
    if zh and orig_short and orig_short not in zh:
        return f"{zh}（{orig_short}）"
    return zh or orig_short or chapter_id


# ── Round 121 L5: 章级注释统一（去重 / 孤儿改锚 / 顺编号）──────────────
# 交付文本注释载体：md 脚注 [^id] / 编号注 [n]·（n）（含【译注】块内）/
# 悬块 > **译者注：** / 段落式 **词条**：释义 / 标题块 【章末注释】。
# 目标：无重复注、无孤儿注、脚注编号全书唯一且顺序稳定。
_NOTE_TITLE_RE = re.compile(r'^\s*\*{0,2}\s*【([^】]{1,12})】\s*\*{0,2}\s*$')
_NOTE_TITLE_END_RE = re.compile(
    r'^\s*\*{0,2}\s*【(?:章末注释|本章注释|章末注|尾注|文末注)】\s*\*{0,2}\s*$')
_TR_BLOCK_RE = re.compile(r'^\s*>\s*\*\*译者注[:：]\*\*\s*(.*)$')
_BRACKET_DEF_RE = re.compile(r'^\s*[\[［](\d{1,3})[\]］]\s*(\S.*)$')
_PAREN_DEF_RE = re.compile(r'^\s*[（(](\d{1,3})[）)]\s*(\S.*)$')
_BOLD_LEAD_RE = re.compile(r'^\s*\*\*([^*]{1,40})\*\*\s*[：:]\s*\S')
_SEP_LINE_RE = re.compile(r'^\s*-{3,}\s*$')
_MACHINE_FN_ID_RE = re.compile(r'\[\^[^\]]*_c\d+_[^\]]*\]')
_NOTE_LABEL_RE = re.compile(r'^\s*(?:\*+\s*)?(?:译者注|译注|脚注|注释)\s*[:：]?\s*')
_QUOTED_TERM_RE = re.compile(r'[“「『]([^”」』]{2,36})[”」』]')
# 去重比率：通用统计兜底，校准自一本真实书的注释样本
# （重复组 r≈0.31~0.39、对照组 r≤0.21）；换书若观察到误合并/漏合并，
# 用新样本重校准这两个比率即可——非任何作品专属规则。
# 拉丁锚点停用改为章内频率动态派生（Round 130 审计④：不再硬编码任何作品
# 专属词——历史初始样本中每注都含作者/书名词，曾致高频词误合并；
# 现由 _note_anchor_stop() 按章自动识别）。
_NOTE_DUP_RATIO = 0.28
_NOTE_DUP_RATIO_HARD = 0.75


def _note_norm(content: str) -> str:
    """注释内容归一化（去重比较用）：剥 markdown/标签/空白。"""
    t = re.sub(r'[*`>]', '', content)
    t = re.sub(r'^\s*\[\^[^\]]+\]:\s*', '', t)
    t = _NOTE_LABEL_RE.sub('', t)
    return re.sub(r'\s+', '', t)


def _note_key(content: str) -> str:
    """注释主键：粗体词条 / 引号词条 / 归一化前 12 字。"""
    m = re.match(r'^\s*\*\*(.+?)\*\*', content)
    if m:
        return m.group(1).strip()
    m = re.match(r'^\s*[“「『](.+?)[”」』]', content)
    if m:
        return m.group(1).strip()
    return _note_norm(content)[:12]


def _note_anchor_stop(contents) -> set:
    """章内频率派生的停用锚点（纯统计，零作品先验，Round 130 审计④）。

    在 ≥60%（且 ≥2 条）注释中都出现的拉丁词（≥4 字母）不构成"两注同源"
    的证据——某词若高频遍布全章注释（典型：作者名/书名/时代词），它对区分
    不同注释没有信息量。初始样本中 Carroll 类词即由此机制自然覆盖，不再
    硬编码任何作品专属停用表。
    """
    import math
    from collections import Counter
    texts = [str(c) for c in contents if c]
    n = len(texts)
    if n < 2:
        return set()
    cnt: Counter = Counter()
    for c in texts:
        cnt.update({w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", c)})
    threshold = max(2, math.ceil(n * 0.6))
    return {w for w, k in cnt.items() if k >= threshold}


def _note_is_dup(a: str, b: str, stop=frozenset()) -> bool:
    """内容级重复判定（保守，校准值见上方注释）。

    stop: 章内频率派生的停用锚点（_note_anchor_stop()）；生产调用方须显式传入，
    缺省空集 = 不做停用（仅供单测/直调）。
    """
    import difflib
    na, nb = _note_norm(a), _note_norm(b)
    if not na or not nb or min(len(na), len(nb)) < 30:
        return False
    r = difflib.SequenceMatcher(None, na, nb).ratio()
    if r >= _NOTE_DUP_RATIO_HARD:
        return True
    if r < _NOTE_DUP_RATIO:
        return False
    la = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", a)} - stop
    lb = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", b)} - stop
    if la & lb:
        return True
    ca, cb = set(), set()
    for run in re.findall(r'[\u4e00-\u9fff]{3,}', na):
        ca.update(run[i:i + 4] for i in range(len(run) - 3))
    for run in re.findall(r'[\u4e00-\u9fff]{3,}', nb):
        cb.update(run[i:i + 4] for i in range(len(run) - 3))
    return len(ca & cb) >= 4


def _note_terms(content: str) -> list:
    """孤儿注改锚候选词（长词优先，只从词条头部提取）。"""
    cands = []
    m = re.match(r'^\s*\*\*([^*]{1,40})\*\*', content)
    if m:
        cands.append(m.group(1))
    for m in _QUOTED_TERM_RE.finditer(content):
        cands.append(m.group(1))
    m = re.match(r'^\s*([^：:]{2,24})[：:]', content)
    if m:
        cands.append(m.group(1))
    seen, out = set(), []
    for t in cands:
        t = re.sub(r'^\s*\[[^\]]*\]\s*', '', t).strip(' *“”"「」『』\u3000')
        pieces = [t, t.split('（')[0], t.split('(')[0]]
        pieces += re.findall(r'[\u4e00-\u9fff]{2,}', t)
        pieces += re.findall(r"[A-Za-z][A-Za-z'-]{3,}", t)
        for p in pieces:
            p = p.strip(' *“”"「」『』：「:」（()）')
            if len(p) >= 2 and p not in seen:
                seen.add(p)
                out.append(p)
    out.sort(key=len, reverse=True)
    return out


def _note_candidates(text: str):
    """注释单元扫描。→ (blocks, units)；blocks 含字符偏移，units 含 kind/id/content。"""
    raw_blocks = [{'start': m.start(), 'end': m.end(), 'seg': m.group(0)}
                  for m in re.finditer(r'(?m)(?:^[ \t]*\S.*(?:\n|$))+', text)]
    blocks = []
    for _b in raw_blocks:
        _ls = _b['seg'].split('\n')
        _body_ls = [l for l in _ls if l.strip()]
        if (len(_body_ls) > 1
                and all(_TR_BLOCK_RE.match(l) for l in _body_ls)):
            _pos = _b['start']
            for _l in _ls:
                if _l.strip():
                    blocks.append({'start': _pos, 'end': _pos + len(_l), 'seg': _l})
                _pos += len(_l) + 1
        else:
            blocks.append(_b)
    units = []
    prev = None
    for bi, b in enumerate(blocks):
        seg = b['seg']
        first = seg.split('\n', 1)[0]
        m_md = _MD_DEF_RE.match(first)
        m_tr = _TR_BLOCK_RE.match(first)
        m_title = _NOTE_TITLE_RE.match(first)
        m_br = _BRACKET_DEF_RE.match(first)
        m_pa = _PAREN_DEF_RE.match(first)
        if m_md:
            u = {'kind': 'md', 'id': m_md.group(1), 'content': m_md.group(2), 'block': bi}
        elif m_tr:
            rest = [re.sub(r'^\s*>\s?', '', l) for l in seg.split('\n')[1:]]
            u = {'kind': 'tr', 'id': None,
                 'content': ' '.join([m_tr.group(1)] + rest).strip(), 'block': bi}
        elif m_title and len(seg.rstrip('\n').split('\n')) == 1:
            u = {'kind': 'title', 'id': None, 'content': m_title.group(1), 'block': bi,
                 'title_end': bool(_NOTE_TITLE_END_RE.match(first))}
        elif (m_br or m_pa) and prev in ('sep', 'num', 'title'):
            g = m_br or m_pa
            u = {'kind': 'num', 'id': g.group(1), 'content': g.group(2), 'block': bi}
        elif _SEP_LINE_RE.match(first):
            u = {'kind': 'sep', 'id': None, 'content': '', 'block': bi}
        elif (_BOLD_LEAD_RE.match(first) and len(seg.strip()) <= 800):
            u = {'kind': 'para', 'id': None, 'content': seg.strip(), 'block': bi}
        else:
            u = {'kind': 'body', 'id': None, 'content': seg, 'block': bi}
        units.append(u)
        prev = u['kind']
    return blocks, units


def _group_title_units(units) -> None:
    """标题块打组：title 及紧随其后的连续 num 单元。"""
    i, n = 0, len(units)
    while i < n:
        if units[i]['kind'] == 'title':
            j = i + 1
            while j < n and units[j]['kind'] == 'num':
                j += 1
            if j - i > 1:
                for k in range(i, j):
                    units[k]['group'] = i
            i = j
        else:
            i += 1


def _split_h1_sections(text: str):
    """按 H1 切章 → [(章标题, 正文)]（前置段标题为 '(前置)'）。"""
    parts = re.split(r'(?m)^(# .*)$', text)
    out = [('(前置)', parts[0])]
    for k in range(1, len(parts), 2):
        out.append((parts[k].strip(), parts[k + 1] if k + 1 < len(parts) else ''))
    return out


def normalize_chapter_notes(text: str, start_num: int = 1):
    """章级注释统一：去重 → 孤儿改锚/章末归位 → 顺序重编号（幂等）。

    Returns: (new_text, changes: dict, next_num: int)
    """
    blocks, units = _note_candidates(text)
    _group_title_units(units)
    changes = {"merged_dups": 0, "orphans_attached": 0, "md_relocated": 0,
               "num_relocated": 0, "title_blocks_moved": 0, "renumbered": 0,
               "title_shells_dropped": 0,
               "machine_ids": len(_MACHINE_FN_ID_RE.findall(text))}

    # 1) 去重（标题块/分隔行/正文豁免）
    cand = [u for u in units if u['kind'] in ('md', 'num', 'tr', 'para')
            and 'group' not in u]
    parent = list(range(len(cand)))

    def _root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    _stop = _note_anchor_stop([c['content'] for c in cand])
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if _note_is_dup(cand[i]['content'], cand[j]['content'], stop=_stop):
                ri, rj = _root(i), _root(j)
                if ri != rj:
                    parent[rj] = ri
    clusters = {}
    for i in range(len(cand)):
        clusters.setdefault(_root(i), []).append(i)
    dropped = set()
    remap = {}

    def _rank(u):
        score = 2 if u['kind'] == 'md' else (1 if u['kind'] == 'num' else 0)
        return (score, len(_note_norm(u['content'])))

    for members in clusters.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda i: _rank(cand[i]), reverse=True)
        surv = cand[members[0]]
        for mi in members[1:]:
            u = cand[mi]
            dropped.add(u['block'])
            changes["merged_dups"] += 1
            if (u['kind'] == 'md' and surv['kind'] == 'md'
                    and u['id'] != surv['id']):
                remap[u['id']] = surv['id']

    # 2) 保留区（去重删除块置空以稳定偏移）内统计引用
    kept = text
    for bi in dropped:
        b = blocks[bi]
        kept = kept[:b['start']] + (' ' * (b['end'] - b['start'])) + kept[b['end']:]
    body_spans = [(u['block'], blocks[u['block']]) for u in units
                  if u['kind'] == 'body' and u['block'] not in dropped]

    def _count_refs(uid, md=True):
        pat = (re.compile(r'\[\^' + re.escape(uid) + r'\](?!:)') if md
               else re.compile(r'[\[［]' + re.escape(uid) + r'[\]］](?!\s*[:：])'))
        cnt, pos = 0, None
        for bi, b in body_spans:
            seg = kept[b['start']:b['end']]
            for m in pat.finditer(seg):
                cnt += 1
                if pos is None:
                    pos = b['start'] + m.start()
        return cnt, pos

    ref_info = {}
    for u in cand:
        if u['kind'] in ('md', 'num') and u['block'] not in dropped:
            ref_info[u['block']] = _count_refs(u['id'], md=(u['kind'] == 'md'))
    for old, new in remap.items():
        for u in cand:
            if u['kind'] == 'md' and u['id'] == new and u['block'] in ref_info:
                c, p = _count_refs(old, md=True)
                cc, pp = ref_info[u['block']]
                ref_info[u['block']] = (cc + c, pp if pp is not None else p)
                break

    # 3) 孤儿处理：改锚（正文词条）或章末归位
    seg_override = {}
    md_fail, num_fail = [], []

    def _block_text(bi):
        return seg_override.get(bi, blocks[bi]['seg'])

    # 编号注列表内交叉引用：同一连续列表内 {编号 → 临时 uid}
    run_map = {}
    runs, cur_run = [], []
    for u in units:
        if u['kind'] == 'num' and 'group' not in u:
            cur_run.append(u)
        else:
            if len(cur_run) > 1:
                runs.append(cur_run)
            cur_run = []
    if len(cur_run) > 1:
        runs.append(cur_run)
    for run in runs:
        m = {}
        for u in run:
            m[str(int(u['id']))] = ('n{0}'.format(u['block'])
                                    if u['kind'] == 'num' else u['id'])
        run_map[id(run)] = m
    uid_run = {}
    for run in runs:
        for u in run:
            uid_run[u['block']] = run_map[id(run)]

    def _rewrite_cross(content, block):
        m = uid_run.get(block)
        if not m:
            return content
        return re.sub(r'\[(\d+)\](?![\]:])',
                      lambda mm: '[^' + m.get(mm.group(1), mm.group(1)) + ']'
                      if mm.group(1) in m else mm.group(0), content)

    for u in cand:
        if u['kind'] not in ('md', 'num', 'tr') or u['block'] in dropped:
            continue
        cnt, _p = ref_info.get(u['block'], (0, None))
        if u['kind'] == 'tr':
            cnt = 0
        if cnt:
            continue
        hit = None
        for term in _note_terms(u['content']):
            for bi, b in body_spans:
                seg = kept[b['start']:b['end']]
                if term in seg:
                    hit = bi
                    break
            if hit is not None:
                break
        uid = u['id'] if u['kind'] == 'md' else 'n{0}'.format(u['block'])
        if hit is not None:
            cur = _block_text(hit)
            cur = cur.rstrip('\n')
            nl = '\n' if blocks[hit]['seg'].endswith('\n') else ''
            seg_override[hit] = cur + '[^' + uid + ']' + nl
            if u['kind'] in ('num', 'tr'):
                seg_override[u['block']] = '[^{0}]: {1}\n'.format(
                    uid, _rewrite_cross(u['content'], u['block']))
            changes["orphans_attached"] += 1
        elif u['kind'] == 'tr':
            continue
        else:
            dropped.add(u['block'])
            if u['kind'] == 'md':
                md_fail.append(u['content'])
                changes["md_relocated"] += 1
            else:
                num_fail.append('[{0}] {1}'.format(
                    u['id'], _rewrite_cross(u['content'], u['block'])))
                changes["num_relocated"] += 1

    # 3b) 章末注释块条目：尽力改锚（编辑器要求"有注无引"→角标；失败者留在块内）
    for tu in units:
        if not (tu['kind'] == 'title' and tu.get('title_end') and 'group' in tu):
            continue
        members = [u for u in units
                   if u.get('group') == tu['group'] and u['kind'] == 'num']
        attached = 0
        for u in members:
            terms = _note_terms(u['content'])
            extra = []
            for _run in re.findall(r'[\u4e00-\u9fff]{4,}', u['content']):
                if _run not in extra:
                    extra.append(_run)
                for _i in range(len(_run) - 3):
                    _w = _run[_i:_i + 4]
                    if _w not in extra:
                        extra.append(_w)
            extra = [r for r in extra if r not in terms]
            terms = terms + extra
            terms.sort(key=len, reverse=True)
            hit = None
            for term in terms:
                if len(term) < 4 and term not in _note_terms(u['content']):
                    continue
                for bi, b in body_spans:
                    seg = kept[b['start']:b['end']]
                    if term in seg:
                        hit = bi
                        break
                if hit is not None:
                    break
            if hit is None:
                continue
            uid = 'n{0}'.format(u['block'])
            cur = _block_text(hit).rstrip('\n')
            nl = '\n' if blocks[hit]['seg'].endswith('\n') else ''
            seg_override[hit] = cur + '[^' + uid + ']' + nl
            seg_override[u['block']] = '[^{0}]: {1}\n'.format(uid, u['content'])
            attached += 1
            changes["orphans_attached"] += 1
        if members and attached == len(members):
            dropped.add(tu['block'])

    # 3c) 空壳注释标题（无任何条目的【注释】块）：删除
    # 仅限"【注释】"制式（结构化编号注的标准标题）；【译注】等可承载
    # 纯段落说明文字，标题存在即合理，不做清理.
    _shell_re = re.compile(r'^\s*\*{0,2}\s*【注释】\s*\*{0,2}\s*$')
    for u in units:
        if u['kind'] != 'title' or 'group' in u or u['block'] in dropped:
            continue
        if _shell_re.match(blocks[u['block']]['seg'].split('\n', 1)[0]):
            dropped.add(u['block'])
            changes["title_shells_dropped"] += 1

    # 4) 章末标题块归位（【章末注释】等；已在章末者不动）
    end_blocks = []
    for u in units:
        if u['kind'] != 'title' or not u.get('title_end') or 'group' not in u:
            continue
        if u['block'] in dropped:
            continue
        grp = [x for x in units if x.get('group') == u['group']]
        last = max(x['block'] for x in grp)
        if not any(x['kind'] == 'body' and x['block'] > last for x in units):
            continue
        movers = [x for x in grp
                  if x['block'] not in dropped and x['block'] not in seg_override]
        movers.sort(key=lambda x: x['block'])
        for x in movers:
            dropped.add(x['block'])
            end_blocks.append(blocks[x['block']]['seg'].rstrip('\n'))
        if movers:
            changes["title_blocks_moved"] += 1

    # 5) 组装
    out_parts, cur = [], 0
    for bi, b in enumerate(blocks):
        out_parts.append(text[cur:b['start']])
        if bi in dropped:
            seg = ''
        elif bi in seg_override:
            seg = seg_override[bi]
        else:
            seg = b['seg']
        out_parts.append(seg)
        cur = b['end']
    out_parts.append(text[cur:])
    assembled = ''.join(out_parts)
    tail = []
    if end_blocks:
        tail.append('---\n\n' + '\n\n'.join(end_blocks))
    if num_fail:
        tail.append('---\n\n**【译注】**\n\n' + '\n\n'.join(num_fail))
    if md_fail:
        tail.append('---\n\n' + '\n\n'.join('> **译者注：** ' + c for c in md_fail))
    if tail:
        assembled = assembled.rstrip('\n') + '\n\n' + '\n\n'.join(tail) + '\n'

    # 6) 顺序重编号（全书连续；被去重旧 id 的引用先并入幸存者）
    if remap:
        assembled = re.sub(
            r'\[\^([^\]]+)\](?!:)',
            lambda m: '[^' + remap.get(m.group(1), m.group(1)) + ']', assembled)
    order = []
    for m in re.finditer(r'\[\^([^\]]+)\](?!:)', assembled):
        if m.group(1) not in order:
            order.append(m.group(1))
    for m in re.finditer(r'\[\^([^\]]+)\]:', assembled):
        if m.group(1) not in order:
            order.append(m.group(1))
    mapping = {old: str(start_num + i) for i, old in enumerate(order)}

    def _sub_ref(m):
        return '[^' + mapping.get(m.group(1), m.group(1)) + ']'

    assembled = re.sub(r'\[\^([^\]]+)\](?!:)', _sub_ref, assembled)
    assembled = re.sub(r'\[\^([^\]]+)\]:',
                       lambda m: '[^' + mapping.get(m.group(1), m.group(1)) + ']:',
                       assembled)
    changes["renumbered"] = len(order)

    # 6.5) 脚注定义章末集中（raw md 可读性；编辑器多次建议"集中至章末"）
    def_lines = {}
    kept_lines = []
    for line in assembled.split('\n'):
        m = re.match(r'^\[\^(\d+)\]:\s(.*)$', line)
        if m:
            def_lines[int(m.group(1))] = line
        else:
            kept_lines.append(line)
    if def_lines:
        while kept_lines and kept_lines[-1].strip() == '':
            kept_lines.pop()
        if kept_lines and kept_lines[-1].strip() == '---':
            kept_lines.pop()
        while kept_lines and kept_lines[-1].strip() == '':
            kept_lines.pop()
        assembled = ('\n'.join(kept_lines).rstrip('\n') + '\n\n---\n\n'
                     + '\n'.join(def_lines[k] for k in sorted(def_lines)) + '\n')
    return assembled, changes, start_num + len(order)


def normalize_notes_document(text: str):
    """全书：按 H1 切章逐章统一（编号全书连续）。→ (new_text, report)"""
    parts = re.split(r'(?m)^(# .*)$', text)
    agg = {"merged_dups": 0, "orphans_attached": 0, "md_relocated": 0,
           "num_relocated": 0, "title_blocks_moved": 0, "renumbered": 0,
           "title_shells_dropped": 0,
           "machine_ids": 0}
    details = []
    counter = 1
    pre_new, ch0, counter = normalize_chapter_notes(parts[0], counter)
    for key in agg:
        agg[key] += ch0.get(key, 0)
    if any(ch0.values()):
        details.append(('(前置)', ch0))
    out_txt = [pre_new]
    for k in range(1, len(parts), 2):
        head = parts[k]
        body = parts[k + 1] if k + 1 < len(parts) else ''
        body_new, ch, counter = normalize_chapter_notes(body, counter)
        for key in agg:
            agg[key] += ch.get(key, 0)
        if any(ch.values()):
            details.append((head.strip(), ch))
        out_txt.append(head)
        out_txt.append(body_new)
    report = {"stats": agg, "chapters": details, "next_num": counter}
    return ''.join(out_txt), report


def notes_normalize_outputs(project_root=None, output_dir=None, apply: bool = False) -> dict:
    """对 translated_full.md 执行章级注释统一；默认 dry-run，写 output/note_policy_diff.md。"""
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    merged = out_root / "translated_full.md"
    stats = {"files": 0, "changed": 0, "merged_dups": 0, "orphans_attached": 0,
             "md_relocated": 0, "num_relocated": 0, "title_blocks_moved": 0,
             "renumbered": 0, "title_shells_dropped": 0, "machine_ids": 0}
    details = []
    if merged.exists():
        stats["files"] = 1
        text = merged.read_text(encoding="utf-8")
        new_text, rep = normalize_notes_document(text)
        if new_text != text:
            stats["changed"] = 1
            for key in ("merged_dups", "orphans_attached", "md_relocated",
                        "num_relocated", "title_blocks_moved", "renumbered",
                        "title_shells_dropped", "machine_ids"):
                stats[key] = rep["stats"].get(key, 0)
            details = rep["chapters"]
            if apply:
                merged.write_text(new_text, encoding="utf-8")
    lines = ["# 章级注释统一报告 (note_policy_diff)", "",
             "- 模式: {0}".format('APPLY (已写入)' if apply else 'DRY RUN (仅报告)'),
             "- 处理 file: {0} 个；需要变更: {1} 个".format(stats['files'], stats['changed']),
             "- 去重合并: {0}；孤儿改锚: {1}；md 转译者注: {2}；编号注转译注块: {3}".format(
                 stats['merged_dups'], stats['orphans_attached'],
                 stats['md_relocated'], stats['num_relocated']),
             "- 标题块搬移: {0}；空壳标题清理: {1}；顺编号: {2}；机器 id 清除: {3}".format(
                 stats['title_blocks_moved'], stats['title_shells_dropped'],
                 stats['renumbered'], stats['machine_ids']),
             ""]
    if details:
        lines.append("## 明细")
        lines.append("")
        for label, ch in details:
            lines.append("- `{0}`: {1}".format(
                label, ", ".join("{0}={1}".format(k, v) for k, v in ch.items() if v)))
    else:
        lines.append("（无需变更）")
    lines += ["", "---", "*由 notes-normalize 子命令自动生成*", ""]
    out_root.mkdir(parents=True, exist_ok=True)
    report_path = out_root / "note_policy_diff.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"stats": stats, "report": str(report_path), "applied": bool(apply)}


def _chapter_note_dupes(text: str) -> list:
    """章切片的注释重复检测（供 lint；与 normalize 同判据）。"""
    _blocks, units = _note_candidates(text)
    _group_title_units(units)
    cand = [u for u in units if u['kind'] in ('md', 'num', 'tr', 'para')
            and 'group' not in u]
    out = []
    _stop = _note_anchor_stop([c['content'] for c in cand])
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            if _note_is_dup(cand[i]['content'], cand[j]['content'], stop=_stop):
                out.append("重复注: {0} ~ {1}".format(
                    _note_key(cand[i]['content'])[:18],
                    _note_key(cand[j]['content'])[:18]))
    return out
