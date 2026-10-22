"""
SagaTrans 文本切分器（SmartChunker）

源：原整合脚本 §5，拆分到独立模块。

设计：
  - 软切分阈值（soft_limit）：达到此值且对话闭合时切分
  - 硬切分阈值（hard_limit）：超过此值强制切分（极端防错）
  - 场景断点优先（Scene-Aware）：标题 / *** / --- 等边界
  - 引号状态机：ASCII / 弯引号双重奇偶跟踪，避免切断对话
"""
from __future__ import annotations

import re
from typing import List

from src.config import config as _config


# Section 5 — SmartChunker 文本切分器
# ────────────────────────────────────────────────────────────────────

# P3-2: CJK 引号字符集常量（SmartChunker / 各 Splitter 共用）
QUOTE_CHARS_OPEN = "\u201c\u2018\u300c\u300e"   # “ ‘ 「 『
QUOTE_CHARS_CLOSE = "\u201d\u2019\u300d\u300f"  # ” ’ 」 』


class SmartChunker:
    def __init__(self, soft_limit: int = None, hard_limit: int = None):
        """
        :param soft_limit: 软切分阈值（字符数）。达到此值且对话闭合时切分。
        :param hard_limit: 强制切分阈值。极端防错机制，超过此值即使对话未闭合也强制切分。
        """
        chunker_cfg = _config.chunker
        self.soft_limit = soft_limit if soft_limit is not None else chunker_cfg.get("soft_limit", 1000)
        self.hard_limit = hard_limit if hard_limit is not None else chunker_cfg.get("hard_limit", 2500)

        # 匹配 Markdown 一级边界：标题 (#) 或场景转场 (***, ---)
        self.scene_break_pattern = re.compile(r'^(#+|\*\*\*|---)\s*')

    def split_markdown(self, markdown_text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', markdown_text) if p.strip()]

        chunks = []
        current_chunk = []
        current_len = 0
        # 引号状态：>0 表示有未闭合引号（弯引号计数 + ASCII toggle 贡献）
        open_quotes = 0
        ascii_double_open = False  # ASCII " 奇偶切换
        ascii_single_open = False  # ASCII ' 奇偶切换（缩写 don't 可能误计，硬切分兜底）

        def _reset_quote_state():
            nonlocal open_quotes, ascii_double_open, ascii_single_open
            open_quotes = 0
            ascii_double_open = False
            ascii_single_open = False

        for p in paragraphs:
            # 1. 物理边界探测 (Scene-Aware)
            is_scene_break = bool(self.scene_break_pattern.match(p))

            # 如果遇到新场景或标题，且当前缓冲区有内容，立刻打包上一个 Chunk
            if is_scene_break and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
                _reset_quote_state()

            # 将当前段落加入缓冲区
            current_chunk.append(p)
            current_len += len(p)

            # 2. 对话状态探针更新
            # - ASCII " / ' ：toggle（英文直引号无开闭字形）
            # - 弯引号 / 日式引号：开 +1 / 闭 -1
            for ch in p:
                if ch == '"':
                    if ascii_double_open:
                        open_quotes = max(0, open_quotes - 1)
                        ascii_double_open = False
                    else:
                        open_quotes += 1
                        ascii_double_open = True
                elif ch == "'":
                    if ascii_single_open:
                        open_quotes = max(0, open_quotes - 1)
                        ascii_single_open = False
                    else:
                        open_quotes += 1
                        ascii_single_open = True
                elif ch in QUOTE_CHARS_OPEN:
                    open_quotes += 1
                elif ch in QUOTE_CHARS_CLOSE:
                    open_quotes = max(0, open_quotes - 1)

            # 3. 软硬边界触发逻辑
            if not is_scene_break:
                # 软边界：达到字数且无未闭合引号
                if current_len >= self.soft_limit and open_quotes == 0:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    _reset_quote_state()
                # 硬边界：字数超限
                elif current_len >= self.hard_limit:
                    print(f"⚠️ 触发硬切分保护 (长度: {current_len})，可能存在未闭合引号。")
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    _reset_quote_state()

        # 收尾
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks


# ────────────────────────────────────────────────────────────────────