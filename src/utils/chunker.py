import re
from pathlib import Path

class SmartChunker:
    def __init__(self, soft_limit: int = 1000, hard_limit: int = 2500):
        """
        :param soft_limit: 软切分阈值（字符数）。达到此值且对话闭合时切分。
        :param hard_limit: 强制切分阈值。极端防错机制，超过此值即使对话未闭合也强制切分。
        """
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        
        # 匹配 Markdown 一级边界：标题 (#) 或场景转场 (***, ---)
        self.scene_break_pattern = re.compile(r'^(#+|\*\*\*|---)\s*')

    def split_markdown(self, markdown_text: str) -> list[str]:
        # 按自然段进行粗拆分，过滤掉纯空白段落
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', markdown_text) if p.strip()]
        
        chunks = []
        current_chunk = []
        current_len = 0
        open_quotes = False  # 对话状态探针
        
        for p in paragraphs:
            # 1. 物理边界探测 (Scene-Aware)
            is_scene_break = bool(self.scene_break_pattern.match(p))
            
            # 如果遇到新场景或标题，且当前缓冲区有内容，立刻打包上一个 Chunk
            if is_scene_break and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
                open_quotes = False
            
            # 将当前段落加入缓冲区
            current_chunk.append(p)
            current_len += len(p)
            
            # 2. 对话状态探针更新
            # 统计段落中的所有引号形式
            quotes_count = p.count('"') + p.count('“') + p.count('”')
            if quotes_count % 2 != 0:
                open_quotes = not open_quotes  # 状态翻转
                
            # 3. 软硬边界触发逻辑
            if not is_scene_break:
                # 软边界：达到字数且无跨段对话悬挂
                if current_len >= self.soft_limit and not open_quotes:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    open_quotes = False
                # 硬边界：字数超限（防止原著存在极长且不规范的内心独白排版）
                elif current_len >= self.hard_limit:
                    print(f"⚠️ 触发硬切分保护 (长度: {current_len})，可能存在未闭合引号。")
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                    open_quotes = False
                    
        # 收尾：将循环结束后遗留的最后一点文本打包
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks