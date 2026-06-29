import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
db_path = ROOT_DIR / "db" / "workflow.db"

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()
cursor.execute("SELECT chunk_id, state FROM chunk_tasks")
tasks = cursor.fetchall()

print(f"数据库中任务总数: {len(tasks)}")
for chunk_id, state in tasks:
    print(f"任务 {chunk_id} 当前状态: {state}")