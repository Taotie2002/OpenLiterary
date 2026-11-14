"""Round 155: deterministic proofreading for the merged book.

This module is deliberately **not** a style optimizer.  It only reports
objective mechanical defects: encoding damage, control/invisible characters,
artifact headings left by chunk assembly, unbalanced brackets, duplicate
punctuation, raw HTML, broken chapter shells, and a few conservative layout
warnings.  ``--apply`` is limited to removing artifact headings and invisible
characters; it never rewrites wording or punctuation.  A second safe class is
pipeline/editorial scaffolding that can never be part of the literary text
(``译名冲突说明``, ``术语处理说明`` blocks and ``### 译注`` assembly headings).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.judgment_layer import atomic_write_json, atomic_write_text
from src.book_lint import resolve_book_output_path

SCHEMA_VERSION = 1
REPORT_NAME = "proofread_report.json"

_ARTIFACT_HEADING_RE = re.compile(
    r'^\s*#{2,6}\s*(?:译文|注释|正文|原文|脚注|译注)\s*$')
_PIPELINE_META_HEADER_RE = re.compile(
    r'^\s*\*{0,2}术语处理说明\*{0,2}\s*$')
_PIPELINE_META_LINE_RE = re.compile(
    r'^\s*(?:>\s*)?\*{0,2}(?:译名冲突说明|原指令中|实体译名登记表)')
_REPLACEMENT_RE = re.compile('\ufffd')
_INVISIBLE_RE = re.compile('[\u200b\u200c\u200d\ufeff]')
_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
_HTML_TAG_RE = re.compile(r'</?[A-Za-z][^>\n]{0,120}>')
_DUP_PUNCT_RE = re.compile(r'([，。；：、])\1+')
_REPEAT_EMPHASIS_RE = re.compile(r'([？！])\1+')
_SPACE_BEFORE_PUNCT_RE = re.compile(r'[ \t]+[，。；：、？！]')
_HALF_PUNCT_RE = re.compile(r'(?<=[\u4e00-\u9fff])[;:!?,](?=[\u4e00-\u9fff\s]|$)')
_FOOTNOTE_REF_RE = re.compile(r'\[\^[^\]]+\]')
_TERMINAL_RE = re.compile(r'[。！？…”’」』）】》.!?;:—]$')

_BRACKET_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ('（', '）', '圆括号'),
    ('【', '】', '方头括号'),
    ('《', '》', '书名号'),
)
_QUOTE_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ('“', '”', '双引号'),
    ('‘', '’', '单引号'),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line_of(text: str, pos: int) -> int:
    return text.count('\n', 0, pos) + 1


def _chapter_map(text: str) -> Tuple[List[str], List[Tuple[int, str, int, int]]]:
    """Return per-line chapter labels and H1 section spans."""
    lines = text.split('\n')
    labels: List[str] = []
    current = '(前置)'
    sections: List[Tuple[int, str, int, int]] = []
    for idx, line in enumerate(lines):
        if line.startswith('# '):
            current = line.strip()
            sections.append((idx, current, idx + 1, len(lines)))
        labels.append(current)
    # Fix section ends now that all headings are known.
    fixed: List[Tuple[int, str, int, int]] = []
    for n, (start, heading, body_start, _end) in enumerate(sections):
        end = sections[n + 1][0] if n + 1 < len(sections) else len(lines)
        fixed.append((start, heading, body_start, end))
    return labels, fixed


def _excerpt(line: str, limit: int = 120) -> str:
    return re.sub(r'\s+', ' ', line.strip())[:limit]


def _add_issue(issues: List[dict], *, rule: str, severity: str,
               message: str, chapter: str, line: int, excerpt: str,
               fixable: bool = False) -> None:
    issues.append({
        "rule": rule,
        "severity": severity,
        "chapter": chapter,
        "line": int(line),
        "message": message,
        "excerpt": excerpt,
        "fixable": bool(fixable),
    })


def _scan_text(text: str, input_lengths: Optional[Dict[str, int]] = None) -> List[dict]:
    """Return deterministic proofreading findings for *text*."""
    issues: List[dict] = []
    lines = text.split('\n')
    labels, sections = _chapter_map(text)

    def chapter_at(pos: int) -> Tuple[str, int]:
        line = _line_of(text, pos)
        return labels[line - 1] if 0 < line <= len(labels) else '(前置)', line

    # Invisible/control/encoding damage.
    for rule, regex, severity, message, fixable in (
        ("replacement_char", _REPLACEMENT_RE, "blocking",
         "Unicode 替换字符 U+FFFD（编码损坏）", False),
        ("invisible_char", _INVISIBLE_RE, "blocking",
         "零宽/BOM 不可见字符", True),
        ("control_char", _CONTROL_RE, "blocking",
         "C0/C1 控制字符", True),
    ):
        for match in regex.finditer(text):
            chapter, line = chapter_at(match.start())
            _add_issue(
                issues, rule=rule, severity=severity, message=message,
                chapter=chapter, line=line,
                excerpt=_excerpt(lines[line - 1]), fixable=fixable)

    # Chunk-assembly artifact headings.
    for idx, line in enumerate(lines, 1):
        if _ARTIFACT_HEADING_RE.match(line):
            _add_issue(
                issues, rule="artifact_heading", severity="blocking",
                message="chunk 拼接残留标题（不是正文）", chapter=labels[idx - 1],
                line=idx, excerpt=_excerpt(line), fixable=True)
        if _PIPELINE_META_HEADER_RE.match(line):
            _add_issue(
                issues, rule="pipeline_meta_block", severity="blocking",
                message="流水线术语处理说明残留（不是正文）", chapter=labels[idx - 1],
                line=idx, excerpt=_excerpt(line), fixable=True)
        if _PIPELINE_META_LINE_RE.match(line):
            _add_issue(
                issues, rule="pipeline_meta_line", severity="blocking",
                message="流水线/实体登记说明残留（不是正文）", chapter=labels[idx - 1],
                line=idx, excerpt=_excerpt(line), fixable=True)

    # Raw HTML and repeated punctuation.
    for match in _HTML_TAG_RE.finditer(text):
        chapter, line = chapter_at(match.start())
        _add_issue(
            issues, rule="html_tag", severity="blocking",
            message="正文残留 HTML 标签", chapter=chapter, line=line,
            excerpt=_excerpt(lines[line - 1]))
    for match in _DUP_PUNCT_RE.finditer(text):
        chapter, line = chapter_at(match.start())
        _add_issue(
            issues, rule="duplicate_punctuation", severity="blocking",
            message="连续重复标点", chapter=chapter, line=line,
            excerpt=_excerpt(lines[line - 1]))
    for match in _REPEAT_EMPHASIS_RE.finditer(text):
        chapter, line = chapter_at(match.start())
        _add_issue(
            issues, rule="repeated_emphasis", severity="warning",
            message="连续重复问号/叹号（可能是刻意强调，需确认）",
            chapter=chapter, line=line,
            excerpt=_excerpt(lines[line - 1]))

    # Layout warnings: half-width punctuation and spaces before CJK punctuation.
    for match in _HALF_PUNCT_RE.finditer(text):
        chapter, line = chapter_at(match.start())
        _add_issue(
            issues, rule="halfwidth_punctuation", severity="warning",
            message="中文正文中混入半角标点", chapter=chapter, line=line,
            excerpt=_excerpt(lines[line - 1]))
    for match in _SPACE_BEFORE_PUNCT_RE.finditer(text):
        chapter, line = chapter_at(match.start())
        _add_issue(
            issues, rule="space_before_punctuation", severity="warning",
            message="中文标点前有多余空格", chapter=chapter, line=line,
            excerpt=_excerpt(lines[line - 1]))

    # Paired symbols.  Bracket imbalance is objective; quote imbalance is a
    # warning because multi-paragraph speeches are legitimate in some editions.
    for _start, heading, body_start, body_end in sections:
        body = '\n'.join(lines[body_start:body_end])
        for open_c, close_c, label in _BRACKET_PAIRS:
            if body.count(open_c) != body.count(close_c):
                _add_issue(
                    issues, rule="unbalanced_bracket", severity="blocking",
                    message=f"{label}不成对（{open_c}{body.count(open_c)}/{close_c}{body.count(close_c)}）",
                    chapter=heading, line=body_start + 1,
                    excerpt=_excerpt(heading))
        for open_c, close_c, label in _QUOTE_PAIRS:
            if body.count(open_c) != body.count(close_c):
                _add_issue(
                    issues, rule="unbalanced_quote", severity="warning",
                    message=f"{label}数量不平衡（{open_c}{body.count(open_c)}/{close_c}{body.count(close_c)}），需人工确认是否跨段引语",
                    chapter=heading, line=body_start + 1,
                    excerpt=_excerpt(heading))

    # Chapter shell: duplicates, empty sections, blank line after H1.
    seen_headings: Dict[str, int] = {}
    for _start, heading, body_start, body_end in sections:
        seen_headings[heading] = seen_headings.get(heading, 0) + 1
        if body_end <= body_start or not '\n'.join(lines[body_start:body_end]).strip():
            _add_issue(
                issues, rule="empty_chapter", severity="blocking",
                message="章节正文为空", chapter=heading, line=_start + 1,
                excerpt=_excerpt(heading))
        if body_start < len(lines) and lines[body_start].strip():
            _add_issue(
                issues, rule="heading_spacing", severity="warning",
                message="H1 标题后缺少空行", chapter=heading, line=_start + 1,
                excerpt=_excerpt(heading))
    for heading, count in seen_headings.items():
        if count > 1:
            _add_issue(
                issues, rule="duplicate_heading", severity="blocking",
                message=f"H1 标题重复 {count} 次", chapter=heading, line=1,
                excerpt=_excerpt(heading))
    if not sections and text.strip():
        _add_issue(
            issues, rule="missing_heading", severity="blocking",
            message="成书没有任何 H1 章节标题", chapter="(全书)", line=1,
            excerpt=_excerpt(text.split('\n', 1)[0]))

    # Chapter length sanity (only when input chapter files are available).
    if input_lengths:
        ordered_inputs = [input_lengths[k] for k in sorted(input_lengths)]
        if len(ordered_inputs) == len(sections):
            for (_start, heading, body_start, body_end), src_len in zip(sections, ordered_inputs):
                body_len = len('\n'.join(lines[body_start:body_end]))
                if src_len <= 0:
                    continue
                ratio = body_len / src_len
                if ratio < 0.20 or ratio > 1.20:
                    _add_issue(
                        issues, rule="chapter_length", severity="warning",
                        message=f"章节长度比例异常（译文/原文={ratio:.2f}）",
                        chapter=heading, line=_start + 1, excerpt=_excerpt(heading))

    # A single-line paragraph ending in a CJK character without terminal
    # punctuation is a conservative missing-punctuation signal.
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s or len(s) < 20:
            continue
        if idx > 0 and lines[idx - 1].strip():
            continue
        if idx + 1 < len(lines) and lines[idx + 1].strip():
            continue
        if s.startswith(('#', '>', '-', '*', '|', '[^', '——', '= =')):
            continue
        if _FOOTNOTE_REF_RE.sub('', s).rstrip().endswith(('。', '！', '？', '…', '”', '’', '」', '』', '）', '】', '》', '.', '!', '?', ':', '—')):
            continue
        if _TERMINAL_RE.search(s):
            continue
        if re.search(r'[\u4e00-\u9fff]$', s):
            _add_issue(
                issues, rule="paragraph_end", severity="warning",
                message="单行段落末尾缺少终止标点", chapter=labels[idx],
                line=idx + 1, excerpt=_excerpt(s))

    # Report trailing whitespace once per chapter, not once per line.
    trailing: Dict[str, int] = {}
    for idx, line in enumerate(lines):
        if line == line.rstrip():
            continue
        tail = line[len(line.rstrip()):]
        # Exactly two trailing spaces are the standard Markdown hard-break
        # marker (used by verse in this corpus), not a formatting defect.
        if line.strip() and tail == '  ':
            continue
        trailing[labels[idx]] = trailing.get(labels[idx], 0) + 1
    for chapter, count in trailing.items():
        _add_issue(
            issues, rule="trailing_whitespace", severity="warning",
            message=f"异常行尾空白 {count} 行（已排除两个空格的 Markdown 硬换行）",
            chapter=chapter, line=1, excerpt="")

    return issues


def _apply_safe(text: str) -> Tuple[str, List[dict]]:
    """Remove only deterministic, non-semantic artifacts."""
    fixed: List[dict] = []
    lines = text.split('\n')
    kept: List[str] = []
    skip_pipeline_block = False
    for idx, line in enumerate(lines, 1):
        if _PIPELINE_META_HEADER_RE.match(line):
            fixed.append({"rule": "pipeline_meta_block", "line": idx,
                          "action": "removed", "excerpt": _excerpt(line)})
            skip_pipeline_block = True
            continue
        if skip_pipeline_block:
            stripped = line.strip()
            if not stripped or stripped.startswith("- "):
                if stripped:
                    fixed.append({"rule": "pipeline_meta_block", "line": idx,
                                  "action": "removed", "excerpt": _excerpt(line)})
                continue
            skip_pipeline_block = False
        if _PIPELINE_META_LINE_RE.match(line):
            fixed.append({"rule": "pipeline_meta_line", "line": idx,
                          "action": "removed", "excerpt": _excerpt(line)})
            continue
        if _ARTIFACT_HEADING_RE.match(line):
            fixed.append({"rule": "artifact_heading", "line": idx,
                          "action": "removed", "excerpt": _excerpt(line)})
            continue
        kept.append(line)
    new_text = '\n'.join(kept)
    for rule, regex, action in (
        ("invisible_char", _INVISIBLE_RE, "removed"),
        ("control_char", _CONTROL_RE, "removed"),
    ):
        new_text, count = regex.subn('', new_text)
        if count:
            fixed.append({"rule": rule, "line": 0, "action": action,
                          "count": count})
    return new_text, fixed


def proofread_book(
    project_root: Optional[Any] = None,
    output_dir: Optional[Any] = None,
    merged_path: Optional[Any] = None,
    apply: bool = False,
    json_out: Optional[Any] = None,
) -> Dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    out_root = Path(output_dir) if output_dir else root / "output"
    merged = (Path(merged_path) if merged_path
              else resolve_book_output_path(root, out_root))
    if not merged.exists():
        return {"ok": False, "error": f"merged_not_found: {merged}",
                "issues": [], "summary": {"blocking": 0, "warning": 0, "fixed": 0}}

    try:
        original = merged.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {"ok": False, "error": f"merged_read_error: {exc}",
                "issues": [], "summary": {"blocking": 0, "warning": 0, "fixed": 0}}
    text = original
    fixed: List[dict] = []
    if apply:
        text, fixed = _apply_safe(text)
        if text != original:
            atomic_write_text(merged, text)

    input_lengths: Dict[str, int] = {}
    input_dir = root / "input"
    if input_dir.exists():
        for path in sorted(input_dir.glob("ch*.md")):
            try:
                input_lengths[path.stem] = len(path.read_text(encoding="utf-8"))
            except OSError:
                pass

    issues = _scan_text(text, input_lengths=input_lengths)
    blocking = sum(1 for issue in issues if issue["severity"] == "blocking")
    warnings = sum(1 for issue in issues if issue["severity"] == "warning")
    by_rule: Dict[str, int] = {}
    for issue in issues:
        by_rule[issue["rule"]] = by_rule.get(issue["rule"], 0) + 1

    merged_sha = _sha256_bytes(text.encode("utf-8"))
    report_path = (Path(json_out) if json_out
                   else out_root / REPORT_NAME)
    report = {
        "schema_version": SCHEMA_VERSION,
        "ok": blocking == 0,
        "generated_at": datetime.now().astimezone().isoformat(),
        "merged_path": str(merged),
        "original_sha256": _sha256_bytes(original.encode("utf-8")),
        "merged_sha256": merged_sha,
        "changed": text != original,
        "apply": bool(apply),
        "summary": {
            "blocking": blocking,
            "warning": warnings,
            "fixed": len(fixed),
            "by_rule": by_rule,
        },
        "issues": issues,
        "fixed": fixed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)

    md_lines = ["# 成书机械排错报告 (proofread)", ""]
    md_lines.append(f"- 模式: {'APPLY (安全修复已写入)' if apply else 'DRY RUN (仅报告)'}")
    md_lines.append(f"- 成书: `{merged}`")
    md_lines.append(f"- SHA-256: `{merged_sha}`")
    md_lines.append(f"- 阻断项: {blocking}；警告: {warnings}；安全修复: {len(fixed)}")
    md_lines.append("")
    if blocking or warnings:
        md_lines.append("## 发现")
        md_lines.append("")
        for issue in issues[:200]:
            md_lines.append(
                f"- [{issue['severity']}] `{issue['rule']}` {issue['chapter']}:"
                f"{issue['line']} — {issue['message']}：{issue['excerpt']}")
        if len(issues) > 200:
            md_lines.append(f"- ……另有 {len(issues) - 200} 条，见 JSON 报告")
    else:
        md_lines.append("（未发现机械性错误）")
    if fixed:
        md_lines += ["", "## 安全修复", ""]
        for item in fixed[:100]:
            md_lines.append(
                f"- `{item.get('rule')}` line={item.get('line')} "
                f"{item.get('action')} {item.get('excerpt', '')}".rstrip())
    md_lines += ["", "---", "*由 proofread 子命令自动生成*", ""]
    atomic_write_text(report_path.with_suffix(".md"), "\n".join(md_lines))

    print(f"🧹 proofread({'apply' if apply else 'dry-run'}): "
          f"blocking={blocking}, warning={warnings}, fixed={len(fixed)}")
    print(f"   报告: {report_path}")
    return {
        "ok": blocking == 0,
        "changed": text != original,
        "merged_path": str(merged),
        "merged_sha256": merged_sha,
        "report_path": str(report_path),
        "issues": issues,
        "fixed": fixed,
        "summary": report["summary"],
    }
