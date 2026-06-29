import sys
import os
from pathlib import Path

# 获取项目根目录 (即 translator-agent/)
# 假设 src/init_project.py 在 src/ 目录下，所以 parent 是项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 将项目根目录加入 sys.path，确保能导入 src/ 下的模块
sys.path.append(str(ROOT_DIR / "src"))

from core.scheduler import TaskScheduler
from utils.chunker import SmartChunker

def initialize_translation_project(chapter_id: str, force: bool = False):
    # 1. 强制打印所有路径信息，确保我们在同一频道
    db_dir = ROOT_DIR / "db"
    db_file = db_dir / "workflow.db"
    input_file = ROOT_DIR / "input" / f"{chapter_id}.md"

    print(f"DEBUG: 项目根目录: {ROOT_DIR}")
    print(f"DEBUG: 检查输入文件: {input_file} -> {'存在' if input_file.exists() else '不存在'}")
    print(f"DEBUG: 目标数据库路径: {db_file}")

    if not input_file.exists():
        print(f"❌ 找不到文件: {input_file}")
        return

    # 2. 确保目录存在
    db_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()
        
    # 3. 注入任务调度器
    scheduler = TaskScheduler(db_path=str(db_file))
    
    # 4. 关键：确保 chunks 不为空
    chunks = SmartChunker(soft_limit=1000).split_markdown(markdown_text)
    if not chunks:
        print("❌ 警告：没有切分出任何 chunk！")
        return

    scheduler.init_chapter_tasks(chapter_id=chapter_id, chunks=chunks, force=force)
    print(f"\n✅ 初始化成功，数据已写入: {db_file}")

if __name__ == "__main__":
    # 此处可修改你要初始化的章节 ID
    initialize_translation_project(chapter_id="ch01")