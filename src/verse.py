"""
SagaTrans 韵文通道（Round 119 L1）— 单源实现

链路：转换层 `input/verse_manifest.json` → init 侧 chunk 标签 `db/chunk_tags.json`
      → 运行时提示词注入（literal / rewrite / judge）→ verse_lint
      （PASS 定稿前失败 → 一次定向重写，走既有 quality retry 通道；书级核验在 gate）。

设计原则：
- 纯函数 + 文件读写分离；不依赖 LLM，可离线单测。
- manifest 以「字符偏移 + 行号」记录韵文块；chunk 标签按字符交集计算；
  运行时另有「归一化文本包含」兜底（容忍手工切分与空白差异）。
- lint 阈值全部来自 config.verse（默认保守：宁少报，不误伤散文）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSE_MANIFEST_VERSION = 1
CHUNK_TAGS_VERSION = 1

_DEFAULT_VERSE_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_line_ratio": 0.6,
    "max_line_chars": 56,
    "refrain_ratio": 0.5,
    "retry_on_fail": True,
}


def verse_config() -> Dict[str, Any]:
    """读取 config.yaml `verse:` 节（缺失/异常时返回内建默认，向后兼容）。"""
    cfg = dict(_DEFAULT_VERSE_CONFIG)
    try:
        from src.config import config as _config
        section = _config.get_section("verse") or {}
        for k, v in section.items():
            cfg[k] = v
    except Exception:
        pass
    return cfg


def normalize_whitespace(text: str) -> str:
    """归一化空白：全部空白折叠为单个空格并去首尾（用于包含/重复判定）。"""
    return re.sub(r"\s+", " ", text or "").strip()


# ── manifest 读写（input/verse_manifest.json） ─────────────────────────

def load_verse_manifest(input_dir) -> dict:
    path = Path(input_dir) / "verse_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}


def save_verse_manifest(input_dir, manifest: dict) -> None:
    path = Path(input_dir) / "verse_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


# ── chunk 标签读写（db/chunk_tags.json） ───────────────────────────────

def load_chunk_tags(db_dir) -> dict:
    path = Path(db_dir) / "chunk_tags.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}


def save_chunk_tags(db_dir, tags: dict) -> None:
    path = Path(db_dir) / "chunk_tags.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tags, ensure_ascii=False, indent=2), encoding="utf-8")


# ── chunk ↔ 韵文交集（init 时计算） ────────────────────────────────────

def _normalized_index(text: str) -> Tuple[str, List[int]]:
    """返回 (归一化文本, 每个归一化字符对应的原文本偏移)。"""
    norm_chars: List[str] = []
    orig_idx: List[int] = []
    last_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if not last_space:
                norm_chars.append(" ")
                orig_idx.append(i)
                last_space = True
        else:
            norm_chars.append(ch)
            orig_idx.append(i)
            last_space = False
    return "".join(norm_chars), orig_idx


def locate_chunk_spans(markdown_text: str, chunks: List[str]) -> List[Optional[Tuple[int, int]]]:
    """按序定位每个 chunk 在 markdown_text 中的 (start_char, end_char)；失败 None。

    归一化空白后查找，再映射回原文本偏移，容忍段落间空白差异。
    """
    norm_text, orig_idx = _normalized_index(markdown_text)
    spans: List[Optional[Tuple[int, int]]] = []
    cursor = 0
    for chunk in chunks:
        nchunk = normalize_whitespace(chunk)
        if not nchunk:
            spans.append(None)
            continue
        pos = norm_text.find(nchunk, cursor)
        if pos < 0:
            pos = norm_text.find(nchunk)
        if pos < 0:
            spans.append(None)
            continue
        end_norm = pos + len(nchunk)
        start_char = orig_idx[pos]
        end_char = orig_idx[end_norm - 1] + 1
        spans.append((start_char, end_char))
        cursor = end_norm
    return spans


def _block_lines_in_range(block: dict, start: int, end: int) -> List[str]:
    """取韵文块中字符区间 [start, end) 覆盖的非空行。"""
    text = block.get("text") or ""
    base = int(block.get("start_char", 0) or 0)
    lines: List[str] = []
    pos = base
    for ln in text.split("\n"):
        ln_s, ln_e = pos, pos + len(ln)
        if ln.strip() and ln_s < end and ln_e > start:
            lines.append(ln)
        pos = ln_e + 1
    return lines


def blocks_for_span(verse_blocks: List[dict], start: int, end: int) -> List[dict]:
    """与 chunk 字符区间相交的韵文片段：[{"tag", "lines"}]。"""
    frags: List[dict] = []
    for b in verse_blocks or []:
        bs = int(b.get("start_char", -1) or -1)
        be = int(b.get("end_char", -1) or -1)
        if bs < 0 or be <= bs or bs >= end or be <= start:
            continue
        lines = _block_lines_in_range(b, max(bs, start), min(be, end))
        if lines:
            frags.append({"tag": b.get("tag", "unknown"), "lines": lines})
    return frags


def fragments_from_manifest(chunk_text: str, verse_blocks: List[dict]) -> List[dict]:
    """运行时兜底：按归一化「文本包含」匹配（整块连通命中 → 全行；否则按块序连续 ≥2 行命中）。

    Round 120 P1: 单行命中会误报（如 Fury 诗 "would be" 混入散文 chunk），
    故仅接受整块连通命中或连续 ≥2 行的命中段。
    """
    norm_chunk = normalize_whitespace(chunk_text)
    if not norm_chunk:
        return []
    frags: List[dict] = []
    for b in verse_blocks or []:
        lines = [ln for ln in (b.get("text") or "").split("\n") if ln.strip()]
        if not lines:
            continue
        norm_block = normalize_whitespace(b.get("text") or "")
        if norm_block and norm_block in norm_chunk:
            frags.append({"tag": b.get("tag", "unknown"), "lines": lines})
            continue
        hit = [i for i, ln in enumerate(lines)
               if len(ln.strip()) >= 8 and normalize_whitespace(ln) in norm_chunk]
        if not hit:
            continue
        best = cur = [hit[0]]
        for i in hit[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                if len(cur) > len(best):
                    best = cur
                cur = [i]
        if len(cur) > len(best):
            best = cur
        if len(best) >= 2:
            frags.append({"tag": b.get("tag", "unknown"),
                          "lines": [lines[i] for i in best]})
    return frags


def fragments_for_chunk(chapter_id: str, chunk_id: str, chunk_text: str,
                        tags: dict, manifest: dict) -> List[dict]:
    """chunk 的韵文片段：tags 优先（精确交集），miss 时 manifest 包含兜底。"""
    entry = ((tags.get("chapters") or {}).get(chapter_id) or {}).get(chunk_id)
    if isinstance(entry, dict) and entry.get("blocks"):
        return entry["blocks"]
    blocks = ((manifest.get("chapters") or {}).get(chapter_id)) or []
    if not blocks:
        return []
    return fragments_from_manifest(chunk_text, blocks)


def build_chapter_tags(chapter_id: str, markdown_text: str, chunks: List[str],
                       verse_blocks: List[dict]) -> Dict[str, dict]:
    """计算本章 verse chunk 标签：{chunk_id: {"verse": true, "blocks": [{"tag","lines"}]}}。"""
    result: Dict[str, dict] = {}
    if not verse_blocks or not chunks:
        return result
    spans = locate_chunk_spans(markdown_text, chunks)
    for idx, text in enumerate(chunks):
        frags: List[dict] = []
        span = spans[idx] if idx < len(spans) else None
        if span is not None:
            frags = blocks_for_span(verse_blocks, span[0], span[1])
        if not frags:
            frags = fragments_from_manifest(text, verse_blocks)
        if frags:
            result[f"{chapter_id}_chunk{idx:03d}"] = {"verse": True, "blocks": frags}
    return result


# ── 提示词注入 ────────────────────────────────────────────────────────

def _max_line_repeat(lines: List[str], min_len: int = 4) -> int:
    """归一化后行重复的最大次数（复沓/refrain 探测）。"""
    counts: Dict[str, int] = {}
    for ln in lines:
        key = normalize_whitespace(ln).strip("“”\"'‘’「」")
        if len(key) < min_len:
            continue
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) if counts else 0


def build_verse_note(fragments: List[dict]) -> str:
    """生成注入 literal/rewrite/judge prompt 的韵文处理协议块；无韵文返回 ""。"""
    lines: List[str] = []
    for f in fragments or []:
        lines.extend(f.get("lines") or [])
    lines = [ln.strip() for ln in lines if ln and ln.strip()]
    if not lines:
        return ""
    n = len(lines)
    target = max(2, int(round(n * 0.8)))
    refrain = _max_line_repeat(lines)
    parts = [
        f"【韵文/歌词处理协议（本片段含韵文，源共 {n} 行）】",
        f"1. 逐行对应：每个源行对应一个译行，行与行之间保留换行；译文行数尽量贴近源文"
        f"（源 {n} 行 → 目标 ≥ {target} 行）；严禁把韵文合并进散文段落。",
    ]
    if refrain >= 2:
        parts.append(
            f"2. 复沓（refrain）保留：原文有重复出现的歌词句（最多重复 {refrain} 次），"
            f"译文应按相同的重复节奏重现。")
    parts.append(
        "3. 节奏与尾韵：优先保留童谣/歌谣的节奏感；尾韵尽力而为，不得为押韵牺牲语义。")
    parts.append(
        "4. 韵文正文内不得插入注释或译者注；注释集中放在章末或脚注中。")
    parts.append(
        "5. 本片段为韵文体裁，分行结构是硬性结构要求（不是排版建议），"
        "不得因顺句而合并诗行。")
    return "\n".join(parts) + "\n\n"


# ── verse_lint ────────────────────────────────────────────────────────

def _longest_short_line_run(lines: List[str], max_chars: int) -> int:
    """最长「连续短行」段落长度（空行透明，长行重置；容忍诗节空行）。"""
    best = 0
    cur = 0
    for ln in lines:
        if not ln:
            continue
        if len(ln) <= max_chars:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def verse_lint(fragments: List[dict], translated_text,
               min_line_ratio: float = 0.6,
               max_line_chars: int = 56,
               refrain_ratio: float = 0.5) -> dict:
    """源行数 vs 译行数偏差 + 复沓存留检查。

    返回 {"ok", "expected_lines", "found_lines", "refrain_source",
          "refrain_target", "failures", "reason"}。
    """
    src_lines: List[str] = []
    for f in fragments or []:
        src_lines.extend(ln.strip() for ln in (f.get("lines") or []) if ln and ln.strip())
    expected = len(src_lines)
    res: Dict[str, Any] = {
        "ok": True, "expected_lines": expected, "found_lines": 0,
        "refrain_source": 0, "refrain_target": 0, "failures": [], "reason": "",
    }
    if expected == 0:
        return res
    text = translated_text
    if isinstance(text, dict):
        text = text.get("text", "")
    tlines = [ln.strip() for ln in str(text or "").split("\n")]
    found = _longest_short_line_run(tlines, max_line_chars)
    res["found_lines"] = found
    if expected <= 2:
        need = expected
    else:
        need = max(2, int(round(expected * min_line_ratio)))
    if found < need:
        res["failures"].append(f"verse_lines:{found}/{expected}(need>={need})")
    r_src = _max_line_repeat(src_lines)
    res["refrain_source"] = r_src
    if r_src >= 3:
        r_tgt = _max_line_repeat([ln for ln in tlines if ln])
        res["refrain_target"] = r_tgt
        need_r = max(2, int(round(r_src * refrain_ratio)))
        if r_tgt < need_r:
            res["failures"].append(f"refrain:{r_tgt}/{r_src}(need>={need_r})")
    res["ok"] = not res["failures"]
    res["reason"] = "；".join(res["failures"])
    return res
