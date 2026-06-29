#!/usr/bin/env python3
"""
EPUB → 章节 Markdown 预处理工具
用法:
    python tools/epub_to_md.py <input.epub> <output_dir> [--lang zh|en]
"""
import sys
import re
import argparse
from pathlib import Path

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"❌ 缺少依赖: {e}", file=sys.stderr)
    print("请先安装: pip install ebooklib beautifulsoup4 lxml", file=sys.stderr)
    sys.exit(1)


def log_info(msg: str):
    print(f"[INFO] {msg}", file=sys.stderr)


def log_ok(msg: str):
    print(f"[OK] {msg}", file=sys.stderr)


def log_warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)


def log_err(msg: str):
    print(f"[ERR] {msg}", file=sys.stderr)


def clean_html_text(html_content: str) -> str:
    """清洗 HTML，提取纯文本"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # 移除不需要的标签
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    # 处理常见标签：保留段落结构
    for br in soup.find_all('br'):
        br.replace_with('\n')
    for p in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        p.insert_after('\n')

    text = soup.get_text(separator='\n')

    # 清洗：合并多余空行，去除首尾空白
    text = re.sub(r'[ \t]+', ' ', text)          # 多空格合并
    text = re.sub(r'\n{3,}', '\n\n', text)       # 3+ 换行 → 2
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # 行首尾空白
    text = text.strip()

    return text


def split_large_doc_by_toc_or_headings(html_content: str, toc_entries: list, chapter_idx_start: int) -> list[tuple[str, str]]:
    """
    将单个超大 HTML 文档按 TOC 锚点或 H1/H2 标题切分。
    返回 [(chapter_id, markdown), ...]
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    sections: list[tuple[str, str]] = []

    # 1) 尝试从 TOC 解析锚点
    toc_anchors = []
    for entry in toc_entries:
        href = getattr(entry, 'href', '') or ''
        if isinstance(href, str) and '#' in href:
            anchor_id = href.split('#', 1)[1]
            title = getattr(entry, 'title', '') or ''
            toc_anchors.append((title, anchor_id))

    if toc_anchors:
        for title, anchor_id in toc_anchors:
            el = soup.find(id=anchor_id)
            if el:
                sections.append((title or "Section", el))

    # 2) 没有 TOC 锚点时，按 H1 切分
    if not sections:
        headings = soup.find_all('h1')
        if len(headings) >= 2:
            for h in headings:
                sections.append((h.get_text(strip=True) or "Untitled", h))
        else:
            headings = soup.find_all('h2')
            for h in headings:
                sections.append((h.get_text(strip=True) or "Untitled", h))

    if not sections:
        return []

    # 3) 按顺序切分：每章从当前标题到下个标题前
    # 用 DOM 层级关系定位下一个标题的祖先同辈
    chapters = []
    all_top_parents = []  # 每个章节的"顶级父节点"
    for i, (title, el) in enumerate(sections):
        # 找到该元素所在的"段落块"父节点
        # 向上找到没有同级 title 的最近祖先（通常为 <body> 或 <div>）
        top_parent = el.parent
        all_top_parents.append((title, el, top_parent))

    # 简单实现：找到每个章节元素在文档中的位置（通过 enumerate parents）
    # 然后切分文本时按行号分
    # 用 page_source 按行处理：找到每个 anchor 对应的行
    lines = html_content.split('\n')
    anchor_lines = {}  # anchor_id -> line_number
    for i, line in enumerate(lines):
        for title, anchor_id in toc_anchors:
            if f'id="{anchor_id}"' in line or f'name="{anchor_id}"' in line:
                if anchor_id not in anchor_lines:
                    anchor_lines[anchor_id] = (i, title)

    sorted_anchors = sorted(anchor_lines.items(), key=lambda x: x[1][0])
    for i, (anchor_id, (line_no, toc_title)) in enumerate(sorted_anchors):
        start_line = line_no
        end_line = sorted_anchors[i + 1][1][0] if i + 1 < len(sorted_anchors) else len(lines)
        chunk_lines = lines[start_line:end_line]
        chunk_html = '\n'.join(chunk_lines)

        title = toc_title or f"Section {i+1}"

        # 清理 HTML → 纯文本
        chunk_soup = BeautifulSoup(chunk_html, 'html.parser')
        text = chunk_soup.get_text(separator='\n')
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        text = text.strip()

        if not text or len(text) < 50:
            continue

        chapter_idx = chapter_idx_start + len(chapters) + 1
        ch_id = f"ch{chapter_idx:02d}"
        markdown = f"# {title}\n\n{text}"
        chapters.append((ch_id, markdown))

    return chapters


def extract_chapters(book: epub.EpubBook) -> list[tuple[str, str]]:
    """
    从 EPUB 提取章节，返回 [(chapter_id, markdown_text), ...]
    策略：
      1. 按 spine 顺序遍历 document 项
      2. 单个文档过大（>50K 字符）时，按 TOC 锚点或 H1/H2 标题二次切分
      3. 跳过太短的片段（<50 字符）
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
        text = clean_html_text(html_content)

        if not text or len(text) < 50:
            continue

        if len(text) <= LARGE_DOC_THRESHOLD:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
            title = title_tag.get_text(strip=True) if title_tag else f"Chapter {chapter_idx}"
            chapters.append((ch_id, f"# {title}\n\n{text}"))
            continue

        sub_chapters = split_large_doc_by_toc_or_headings(
            html_content, toc_entries, chapter_idx
        )
        if sub_chapters:
            chapters.extend(sub_chapters)
            chapter_idx += len(sub_chapters)
        else:
            chapter_idx += 1
            ch_id = f"ch{chapter_idx:02d}"
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find(['h1', 'h2', 'h3', 'title'])
            title = title_tag.get_text(strip=True) if title_tag else f"Chapter {chapter_idx}"
            chapters.append((ch_id, f"# {title}\n\n{text}"))

    return chapters


def epub_to_chapter_mds(epub_path: str, output_dir: str, lang: str = 'auto') -> list[str]:
    """
    主转换函数
    返回生成的章节文件路径列表
    """
    epub_path = Path(epub_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not epub_path.exists():
        log_err(f"文件不存在: {epub_path}")
        return []

    log_info(f"读取 EPUB: {epub_path}")
    book = epub.read_epub(str(epub_path))

    # 元数据
    title = book.get_metadata('DC', 'title')
    author = book.get_metadata('DC', 'creator')
    if title:
        log_info(f"书名: {title[0][0]}")
    if author:
        log_info(f"作者: {author[0][0]}")

    chapters = extract_chapters(book)
    if not chapters:
        log_warn("未提取到任何章节内容")
        return []

    log_ok(f"共提取 {len(chapters)} 章")
    generated = []

    for ch_id, markdown in chapters:
        out_file = out_dir / f"{ch_id}.md"
        out_file.write_text(markdown, encoding='utf-8')
        generated.append(str(out_file))
        log_ok(f"{ch_id}.md ({len(markdown)} chars)")

    # 生成汇总信息文件
    summary = out_dir / "_chapters_summary.txt"
    summary.write_text(
        "\n".join(f"{ch_id}\t{Path(f).stat().st_size} bytes" for ch_id, f in zip([c[0] for c in chapters], generated)),
        encoding='utf-8'
    )
    log_info(f"汇总: {summary}")

    # 仅输出文件路径到 stdout（供 mapfile 读取）
    for f in generated:
        print(f)

    return generated


def main():
    parser = argparse.ArgumentParser(description="EPUB 转章节 Markdown")
    parser.add_argument("epub", help="输入 EPUB 文件路径")
    parser.add_argument("output", help="输出目录（将生成 ch01.md, ch02.md...）")
    parser.add_argument("--lang", choices=['zh', 'en', 'auto'], default='auto',
                        help="语言提示（暂未使用，预留）")
    args = parser.parse_args()

    generated = epub_to_chapter_mds(args.epub, args.output, args.lang)

    if generated:
        log_info(f"完成！共生成 {len(generated)} 个章节文件")
        log_info("下一步：")
        log_info(f"  1. 检查 {args.output}/ 下的章节文件")
        log_info(f"  2. 修改 config.yaml 设置 llm_backend 等")
        log_info(f"  3. 运行翻译管线：")
        for f in generated:
            ch_id = Path(f).stem
            log_info(f"     python -m src.translator_agent init --chapter {ch_id} --force")
            log_info(f"     python -m src.translator_agent pipeline --chapter {ch_id}")
        sys.exit(0)
    else:
        log_err("转换失败，无输出")
        sys.exit(1)


if __name__ == "__main__":
    main()