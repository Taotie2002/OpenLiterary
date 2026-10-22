# -*- coding: utf-8 -*-
"""build_epub.py — 由 output/translated_full.md 生成交付用 EPUB3。

用法:
  python3 scripts/build_epub.py [--md output/translated_full.md] [--out output/book.epub]
                                [--title 书名] [--author 作者] [--lang zh-CN]

规则:
  - `# X` -> 章节标题；首个 `# ` 段落含「作者：」时视为书名页
  - `---` 行 -> 忽略（章间分隔）
  - 段内单换行 -> <br/>（韵文/诗行）
  - [^n] 引用 -> 链接至 notes.xhtml#fn-n；文末 `[^n]: ...` 定义 -> 注释章节（可缺省）
  - 支持 **粗** / *斜* 内联（注释区）
书名页元数据：命令行 --title/--author 优先，其次从文档头部（书名/作者/出版信息行）抽取。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import sys

from ebooklib import epub

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = text.replace("\n", "<br/>")
    return text


def paras_to_html(block: str, ref_link: bool = True) -> str:
    out = []
    for raw in re.split(r"\n\s*\n", block.strip()):
        p = raw.strip()
        if not p or p == "---":
            continue
        p = inline(p)
        if ref_link:
            p = re.sub(r"\[\^(\d+)\]",
                       lambda m: (f'<a class="fnref" href="notes.xhtml#fn-{m.group(1)}" '
                                  f'id="ref-{m.group(1)}"><sup>[{m.group(1)}]</sup></a>'), p)
        out.append(f"<p>{p}</p>")
    return "\n".join(out)


CSS = """body { font-family: serif; line-height: 1.7; margin: 5%; }
h1 { text-align: center; margin: 1em 0 1.2em; font-size: 1.35em; }
h1.chap { border-bottom: 1px solid #ccc; padding-bottom: .4em; }
p { margin: 0.35em 0; text-indent: 2em; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 10%; }
a.fnref { text-decoration: none; }
sup { font-size: 0.75em; }
.titlepage { text-align: center; margin-top: 15%; }
.titlepage h1 { font-size: 1.9em; }
.titlepage p { text-indent: 0; }
#notes p { text-indent: 0; margin: 0.6em 0; }
#notes .fn { font-size: 0.95em; }
"""


def build(md_path: str, out_path: str, title: str | None = None,
          author: str | None = None, lang: str = "zh-CN") -> dict:
    src = open(md_path, encoding="utf-8").read()

    # 1) 切出注释定义（文末 [^n]: ...）
    defs: dict[str, str] = {}
    body_lines = []
    for line in src.split("\n"):
        m = re.match(r"^\[\^(\d+)\]:\s?(.*)$", line)
        if m:
            defs[m.group(1)] = m.group(2)
        else:
            body_lines.append(line)
    body = "\n".join(body_lines)

    # 2) 切章节；首个含「作者：」的 # 段落视为书名页
    chunks = re.split(r"(?m)^# (.+)$", body)
    front = chunks[0].strip()
    pairs = [(chunks[i].strip(), chunks[i + 1]) for i in range(1, len(chunks), 2)]
    if pairs and "作者：" in pairs[0][1]:
        front = ("# " + pairs[0][0] + "\n" + pairs[0][1]).strip()
        pairs = pairs[1:]
    chapters = pairs
    if not chapters:
        raise SystemExit("[build_epub] 未找到任何 `# 标题` 章节")

    # 3) 书名页元数据抽取
    title_en = None
    title_zh = None
    author_doc = None
    extras: list[str] = []
    for line in front.split("\n"):
        t = line.strip()
        if not t:
            continue
        if t.startswith("# "):
            title_en = t[2:].strip()
        elif t.startswith("作者："):
            author_doc = t[3:].strip()
        elif ("古登堡" in t or "Gutenberg" in t or t.startswith("千禧")):
            extras.append(t)          # 出版信息行，原样放书名页
        elif title_zh is None and CJK_RE.search(t) and len(t) <= 40:
            title_zh = t
        else:
            extras.append(t)

    book_title = title or title_zh or title_en or "无名之书"
    book_author = author or author_doc or ""

    book = epub.EpubBook()
    book.set_identifier("urn:sha1:" + hashlib.sha1(src.encode("utf-8")).hexdigest())
    book.set_title(book_title)
    if book_author:
        book.add_author(book_author)
    book.set_language(lang)
    for e in extras[:3]:
        book.add_metadata("DC", "source", e)

    style = epub.EpubItem(uid="style", file_name="style/main.css",
                          media_type="text/css", content=CSS)
    book.add_item(style)

    tp = epub.EpubHtml(title="书名页", file_name="titlepage.xhtml", lang=lang)
    parts = ['<div class="titlepage">', f"<h1>{html.escape(book_title)}</h1>"]
    if title_en and title_en != book_title:
        parts.append(f'<p class="subtitle">{html.escape(title_en)}</p>')
    if book_author:
        parts.append(f"<p>作者：{html.escape(book_author)}</p>")
    for e in extras:
        parts.append(f"<p>{html.escape(e)}</p>")
    parts.append("</div>")
    tp.content = "\n".join(parts)
    tp.add_item(style)
    book.add_item(tp)

    spine = ["nav", tp]
    toc = [tp]
    for idx, (ch_title, block) in enumerate(chapters, 1):
        c = epub.EpubHtml(title=ch_title, file_name=f"ch{idx:02d}.xhtml", lang=lang)
        c.content = f'<h1 class="chap">{html.escape(ch_title)}</h1>\n' + paras_to_html(block)
        c.add_item(style)
        book.add_item(c)
        spine.append(c)
        toc.append(c)

    if defs:
        notes_items = []
        for n in sorted((k for k in defs if k.isdigit()), key=int):
            notes_items.append(f'<p class="fn" id="fn-{n}">[{n}] {inline(defs[n])}</p>')
        nt = epub.EpubHtml(title="注释", file_name="notes.xhtml", lang=lang)
        nt.content = ('<h1 class="chap">注释</h1>\n<div id="notes">\n'
                      + "\n".join(notes_items) + "\n</div>")
        nt.add_item(style)
        book.add_item(nt)
        spine.append(nt)
        toc.append(nt)

    book.toc = toc
    book.spine = spine
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(out_path, book)

    refs = set(re.findall(r"\[\^(\d+)\]", body))
    missing = sorted(refs - set(defs))
    return {"chapters": len(chapters), "notes": len(defs), "refs": len(refs),
            "missing_defs": missing, "out": out_path, "title": book_title}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SagaTrans EPUB3 出书")
    ap.add_argument("--md", default="output/translated_full.md")
    ap.add_argument("--out", default="output/book.epub")
    ap.add_argument("--title", default=None, help="覆盖书名（默认取文档头部）")
    ap.add_argument("--author", default=None, help="覆盖作者（默认取文档头部）")
    ap.add_argument("--lang", default="zh-CN")
    a = ap.parse_args()
    r = build(a.md, a.out, a.title, a.author, a.lang)
    print(f"[build_epub] 《{r['title']}》 chapters={r['chapters']} notes={r['notes']} "
          f"refs={r['refs']} missing={r['missing_defs']} -> {r['out']}")
    sys.exit(0 if not r["missing_defs"] else 1)
