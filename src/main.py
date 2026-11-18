"""
SagaTrans CLI 统一入口

源：原整合脚本 §9（辅助入口）+ §10（统一入口）+ §12（统一入口点）。

子命令：
  init          初始化翻译任务（切分章节 → 写入 workflow.db）
  pipeline      跑单章完整流水线
  split         切分输入文件（EPUB / TXT / MD → input/ch*.md）
  consistency   译后命名一致性校正（干跑 / 应用）
  golden        黄金测试集跑分
  golden-gate   质量门禁（参考人工基线）
  memory        内存压力测试
  debug         数据库调试工具
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

from src.consistency import run_consistency_check

# Round 78: 补全散装入口缺失的 import. src/main.py 作为 CLI 统一入口, 引用了
#   多个散装模块的符号 (TaskScheduler, SmartChunker, TranslationPipeline, 等),
#   但顶部 import 仅 run_consistency_check — 导致任何 `python3 -m src.main <cmd>`
#   都会 NameError (如用户报告的 `name 'QUOTE_CHARS_OPEN' is not defined`).
#
#   整合脚本 src/translator_agent.py 由 build_integrated.py 生成, 把这些符号
#   全部内联平铺, 所以整合模式 (默认推荐) 不受影响. 散装模式 (开发者/CI 用)
#   需要这些 import 真正起作用.
#
#   设计原则: 散装 = 事实源. 散装入口必须能跑; 整合脚本用于外部审计/单文件部署.
from src.config import config as _config
from src.inflight_recovery import audit_and_write
from src.task_logging import TaskLoggingError, make_logger, set_active_logger
from src.utils.chunker import SmartChunker
from src.core.scheduler import TaskScheduler
from src.pipeline import TranslationPipeline
from src.rescue import (
    collect_final_status,
    ledger_requeue_candidates,
    write_qa_report,
    write_rescue_report,
)
from src.test_golden_set import run_golden_ab, run_golden_test, run_injection_ab
from src.test_memory_pressure import test_memory_pressure, run_golden_gate
from src.splitters import (
    EpubSplitter, TextSplitter, MdSplitter,
    _EPUB_DEPS_OK, _splitter_log_ok, _splitter_log_info, _splitter_log_err,
)
# Round 78: _EPUB_DEPS_ERROR 是 conditional name (仅在 EPUB 依赖缺失时定义),
# 用 getattr fallback 避免 ImportError 阻断散装入口.
_EPUB_DEPS_ERROR = getattr(__import__("src.splitters", fromlist=["_EPUB_DEPS_ERROR"]),
                            "_EPUB_DEPS_ERROR", "")


# Section 9 — 辅助入口
# ────────────────────────────────────────────────────────────────────

def init_project(chapter_id: str = "ch01", force: bool = False) -> bool:
    """初始化翻译项目：切分文本并写入任务数据库

    Args:
        chapter_id: 章节 ID
        force: 是否覆盖已存在的 chunk（同时重置 text_content 和 retries）

    Returns:
        bool: True 表示初始化成功；False 表示失败（输入文件缺失或切分为空），
              调用方应据此决定是否向 shell 报告非零退出码。
    """
    root_dir = Path(__file__).resolve().parent.parent
    db_dir = root_dir / "db"
    db_file = db_dir / "workflow.db"
    input_file = root_dir / "input" / f"{chapter_id}.md"

    if not input_file.exists():
        print(f"❌ 找不到文件: {input_file}", file=sys.stderr)
        return False

    db_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    scheduler = TaskScheduler(db_path=str(db_file))

    # 使用配置的 chunker 参数
    chunker_cfg = _config.chunker
    chunks = SmartChunker(
        soft_limit=chunker_cfg.get("soft_limit", 1000),
        hard_limit=chunker_cfg.get("hard_limit", 2500)
    ).split_markdown(markdown_text)
    if not chunks:
        print("❌ 警告：没有切分出任何 chunk！", file=sys.stderr)
        return False

    scheduler.init_chapter_tasks(chapter_id=chapter_id, chunks=chunks, force=force)

    # Round 119 L1：韵文 chunk 标签（旁路 db/chunk_tags.json，不改 workflow.db schema）。
    # manifest 存在则按字符交集打标（含空表清理旧标签）；缺失时同样置空，防跨书残留。
    try:
        from src.verse import (load_verse_manifest, load_chunk_tags,
                               save_chunk_tags, build_chapter_tags)
        _manifest = load_verse_manifest(root_dir / "input")
        _blocks = ((_manifest.get("chapters") or {}).get(chapter_id)) or []
        _tags = load_chunk_tags(db_dir)
        _tags.setdefault("version", 1)
        _tags.setdefault("chapters", {})[chapter_id] = build_chapter_tags(
            chapter_id, markdown_text, chunks, _blocks)
        save_chunk_tags(db_dir, _tags)
        _n_verse = len(_tags["chapters"][chapter_id])
        if _n_verse:
            print(f"  🏮 韵文标签: {_n_verse} 个 verse chunk（db/chunk_tags.json）")
    except Exception as _verse_err:
        print(f"  ⚠️ 韵文标签计算失败（非致命）: {_verse_err}")

    # --force：清理上一轮残留的 orphan final.json（避免 stale 文件骗过产出校验）
    if force:
        out_dir = root_dir / "output" / chapter_id
        if out_dir.is_dir():
            valid_ids = {f"{chapter_id}_chunk{i:03d}" for i in range(len(chunks))}
            for f in out_dir.glob("*_final.json"):
                fid = f.stem.rsplit("_final", 1)[0]
                if fid not in valid_ids:
                    f.unlink()
                    print(f"  🧹 清理残留 final: {f.name}")
            # L1：force 视为全新尝试，重置 verse 定向重写的一次性限额标记
            for f in out_dir.glob("*_verse_retry.json"):
                f.unlink(missing_ok=True)

    print(f"\n✅ 初始化成功，数据已写入: {db_file}")
    return True


def debug_db():
    """调试工具：查看数据库中的任务状态"""
    root_dir = Path(__file__).resolve().parent.parent
    db_path = root_dir / "db" / "workflow.db"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, state, retries, quality_retries, last_error FROM chunk_tasks")
    tasks = cursor.fetchall()

    print(f"数据库中任务总数: {len(tasks)}")
    for chunk_id, state, retries, qr, last_error in tasks:
        print(f"  {chunk_id} | {state} | retries={retries} | quality_retries={qr}")
        if last_error:
            print(f"    last_error: {last_error[:120]}")
    conn.close()


# ────────────────────────────────────────────────────────────────────


# Section 10 — 统一入口
# ────────────────────────────────────────────────────────────────────

def print_banner():
    """横幅输出到 stderr，避免污染 stdout（stdout 留给机器可解析的输出，如章节路径）

    散装模式：动态扫描 src/ 目录统计散装模块数。
    """
    _src_dir = Path(__file__).resolve().parent
    _module_files = sorted(_src_dir.rglob("*.py"))
    _module_count = len([f for f in _module_files if f.name != "__init__.py"])
    print(rf"""
   _____                  ______
  / ___/____ _____ _____ /_  __/________ _____  _____
  \__ \/ __ `/ __ `/ __ `// / / ___/ __ `/ __ \/ ___/
 ___/ / /_/ / /_/ / /_/ // / / /  / /_/ / / / (__  )
/____/\__,_/\__, /\__,_//_/ /_/   \__,_/_/ /_/____/
           /____/

  SagaTrans — AI 文学语义编译系统  (散装模块化版)
  共 {_module_count} 个模块 | 散装 = 事实源，整合脚本定期同步
""", file=sys.stderr)

# ────────────────────────────────────────────────────────────────────
# Round 130: 单实例守卫（外部审计①）——文档承诺落地为代码约束。
# 与 scripts/translate_book.sh 共用同一把 flock（默认 db/.sagatrans.lock，
# flock(1) 与 fcntl.flock 底层同源，可互相阻塞）：
#   - 脚本持锁后 export SAGATRANS_LOCK_HELD=1，其子进程（本 CLI）跳过重复加锁；
#   - 应急绕过：SAGATRANS_ALLOW_CONCURRENT=1（会在 stderr 打警告，风险自担）。
# 历史事故：双进程让 MiniMax 配额瞬间打爆、chunk 状态互相覆盖、整章重译两遍。
# ────────────────────────────────────────────────────────────────────
_INSTANCE_LOCK_FD: Optional[int] = None
_INSTANCE_GUARD_FD: Optional[int] = None

_MUTATING_COMMANDS = {"pipeline", "init", "split", "style-profile", "story-graph",
                      "consolidate", "finalize", "notes-normalize", "requeue",
                      "rescue", "qa-report", "recalibrate-critic", "proofread",
                      "book-repair", "book-link-audit", "name-authority-align",
                      "decision-authority-align",
                      "decision-target-authority-repair", "term-form-contract",
                      "query-gold-set", "locked-target-authority-repair",
                      "confirmed-term-drift-repair"}


def _default_lock_path() -> Path:
    """Project-root lock path, independent of the caller's cwd."""
    return Path(__file__).resolve().parent.parent / "db" / ".sagatrans.lock"


def _guard_lock_path(primary: Path) -> Path:
    """Return the secondary runtime lock paired with ``primary``.

    The primary path lives inside the repository and can be unlinked by a
    cleanup/test command.  An old writer then keeps the deleted inode while a
    new writer opens the replacement path, so both can pass ``flock``.  This
    second lock lives outside the repository and is keyed by the absolute
    primary path; it closes that split-brain window without breaking
    interoperability with writers that still hold the primary lock.
    """
    override = os.environ.get("SAGATRANS_GUARD_LOCK_PATH")
    if override:
        return Path(override).expanduser()
    runtime_dir = Path(os.environ.get("SAGATRANS_LOCK_RUNTIME_DIR") or "/tmp")
    canonical = os.path.abspath(os.path.expanduser(str(primary)))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    return runtime_dir / f"sagatrans-{uid}-{digest}.lock"


def _acquire_flock(path: Path, label: str) -> Optional[int]:
    """Acquire one non-blocking flock; ``None`` means explicit bypass."""
    import fcntl

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        if os.environ.get("SAGATRANS_ALLOW_CONCURRENT") == "1":
            print(f"⚠️ [Lock] 无法创建锁文件（{label}） {path}: {e}——"
                  f"SAGATRANS_ALLOW_CONCURRENT=1 放行", file=sys.stderr)
            return None
        print(f"⛔ [Lock] 无法创建锁文件（{label}） {path}: {e}", file=sys.stderr)
        print("   单实例保护不可用，拒绝执行；修复目录权限或设置 SAGATRANS_LOCK_PATH。",
              file=sys.stderr)
        return -1
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            holder = path.read_text(encoding="utf-8").strip().replace("\n", " ") or "(无持有者信息)"
        except OSError:
            holder = "(无持有者信息)"
        try:
            os.close(fd)
        except OSError:
            pass
        if os.environ.get("SAGATRANS_ALLOW_CONCURRENT") == "1":
            print(f"⚠️ [Lock] 已有 SagaTrans 实例持有锁文件（{label}）（{holder}）——"
                  f"SAGATRANS_ALLOW_CONCURRENT=1 放行", file=sys.stderr)
            return None
        print(f"⛔ [Lock] 已有 SagaTrans 实例在运行：{holder}", file=sys.stderr)
        print(f"   锁文件（{label}）: {path}；确认无残留进程后可设 "
              f"SAGATRANS_ALLOW_CONCURRENT=1 强制放行（风险自担）", file=sys.stderr)
        return -1
    return fd


def _command_needs_lock(command: str, args: object) -> bool:
    """Return True for commands that can write DB/profile/graph artifacts."""
    if command in _MUTATING_COMMANDS:
        return True
    if command == "consistency" and getattr(args, "apply", False):
        return True
    if command == "setup-payoff-audit" and getattr(args, "apply", False):
        return True
    if command == "book-repair" and getattr(args, "apply", False):
        return True
    if command in {"pending-reconcile", "name-only-fastpath",
                   "completed-name-repair"} and getattr(
            args, "apply", False):
        return True
    return False

# Round 141: Ctrl+C 防误触。第一次只警告并继续，第二次才真正中止。
_SIGINT_WARNED = False


def _install_sigint_guard():
    """安装两段式 SIGINT 处理器；返回 handler 供测试直接调用。"""
    global _SIGINT_WARNED
    _SIGINT_WARNED = False
    import signal

    def _handler(_signum, _frame):
        global _SIGINT_WARNED
        if not _SIGINT_WARNED:
            _SIGINT_WARNED = True
            print("\n⚠️ 检测到 Ctrl+C：当前任务尚未完成。", file=sys.stderr, flush=True)
            print("   若只是误触，无需操作，程序会继续当前步骤。", file=sys.stderr, flush=True)
            print("   若确认中止，请再按一次 Ctrl+C；已完成 chunk 与 DB 状态会保留，可断点续跑。",
                  file=sys.stderr, flush=True)
            return
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        return None
    return _handler


def acquire_instance_lock(lock_path: Optional[Path] = None) -> bool:
    """非阻塞获取单实例锁。

    Returns:
        True  — 已持有（或由祖先进程持有 / 放行开关生效），可继续执行；
        False — 已有另一实例在运行，调用方应中止。
    """
    global _INSTANCE_LOCK_FD, _INSTANCE_GUARD_FD
    if os.environ.get("SAGATRANS_LOCK_HELD") == "1":
        return True
    import time as _time
    override = os.environ.get("SAGATRANS_LOCK_PATH")
    path = Path(lock_path or override) if (lock_path or override) else _default_lock_path()
    fd = _acquire_flock(path, "主锁")
    if fd == -1:
        return False
    if fd is None:  # explicit concurrent bypass
        return True

    guard_path = _guard_lock_path(path)
    guard_fd = _acquire_flock(guard_path, "运行期副锁")
    if guard_fd == -1:
        try:
            os.close(fd)
        except OSError:
            pass
        return False
    if guard_fd is None:  # explicit concurrent bypass
        try:
            os.close(fd)
        except OSError:
            pass
        return True
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (f"pid={os.getpid()} started={_time.strftime('%F %T')} "
                      f"cmd={' '.join(sys.argv[1:]) or 'pipeline'}\n").encode("utf-8"))
    except OSError:
        pass
    _INSTANCE_LOCK_FD = fd  # 进程存活期间保持打开；fd 随进程退出关闭 → 自动释放锁
    _INSTANCE_GUARD_FD = guard_fd
    return True


def main():
    _install_sigint_guard()
    print_banner()

    import argparse
    parser = argparse.ArgumentParser(description="SagaTrans — AI 文学语义编译系统")
    parser.add_argument("command", nargs="?", default="pipeline",
                        choices=["pipeline", "init", "golden", "golden-gate", "memory", "debug", "split", "consistency", "consolidate", "finalize", "lint", "gate", "style-profile", "story-graph", "story-graph-query", "setup-payoff-audit", "drift-audit", "notes-normalize", "proofread", "book-repair", "book-link-audit", "pending-reconcile", "name-only-fastpath", "completed-name-repair", "confirmed-term-drift-repair", "name-authority-audit", "name-authority-align", "decision-authority-align", "decision-target-authority-repair", "locked-target-authority-repair", "term-form-audit", "locked-target-authority-audit", "term-form-contract", "query-gold-set", "requeue", "rescue", "qa-report", "recalibrate-critic"],
                        help="执行命令: pipeline (默认), init, golden, golden-gate, memory, debug, split, consistency, consolidate, finalize, lint, gate, style-profile, story-graph, story-graph-query, setup-payoff-audit, drift-audit, notes-normalize, proofread, book-repair, book-link-audit, pending-reconcile, name-only-fastpath, completed-name-repair, confirmed-term-drift-repair, name-authority-audit, name-authority-align, decision-authority-align, decision-target-authority-repair, locked-target-authority-repair, term-form-audit, locked-target-authority-audit, term-form-contract, query-gold-set, requeue, rescue, qa-report, recalibrate-critic")
    parser.add_argument("--chapter", default="ch01", help="章节 ID (默认: ch01)")
    parser.add_argument("--chapters", default=None,
                        help="style-profile 命令：抽样章节，逗号分隔（默认均匀抽 5 章）")
    parser.add_argument("--deep", action="store_true",
                        help="style-profile 命令：全书分段深读 + 跨段合并 + L1/L2 验证（Round 131）")
    parser.add_argument("--deep-max-chars", type=int, default=60000,
                        help="style-profile --deep：单段原文字符上限（默认 60000）")
    parser.add_argument("--deep-overlap-chars", type=int, default=2000,
                        help="style-profile --deep：相邻分段重叠字符数（默认 2000）")
    parser.add_argument("--derive-digests", action="store_true",
                        help="style-profile：不调用 LLM，从 style_profile_audit.json 确定性回填章源哈希与 chapter_digests")
    parser.add_argument("--graph-path", default=None,
                        help="story-graph / story-graph-query 命令：图路径（默认读取 config.story_graph.path）")
    parser.add_argument("--allow-stale", action="store_true",
                        help="story-graph-query / setup-payoff-audit：允许读取 stale 图作研究用途（不能用于交付或 --apply）")
    parser.add_argument("--require-answerable", action="store_true",
                        help="story-graph 命令：实体覆盖率不可答时返回 2")
    parser.add_argument("--query", "--question", "--claim", dest="query", default=None,
                        help="story-graph-query 命令：问题或叙事主张")
    parser.add_argument("--note-id", default=None,
                        help="story-graph-query 命令：已验证注记 ID；其 evidence 作为场景种子")
    parser.add_argument("--top-k", type=int, default=10,
                        help="story-graph-query 命令：返回场景数（默认 10）")
    parser.add_argument("--max-hops", type=int, choices=[1, 2], default=2,
                        help="story-graph-query 命令：最大 incidence 跳数（默认 2）")
    parser.add_argument("--llm-judge", action="store_true",
                        help="story-graph-query 命令：对候选做一次本地 LLM 必要证据复核（失败 fail-open）")
    parser.add_argument("--no-dpp", action="store_true",
                        help="story-graph-query 命令：关闭 DPP 多样性重排")
    parser.add_argument("--no-note-similarity", action="store_true",
                        help="story-graph-query 命令：关闭问题→已验证注记的相似度种子（严格别名模式）")
    parser.add_argument("--eval-cases", default=None,
                        help="story-graph-query 命令：运行 gold-span 评测 JSON")
    parser.add_argument("--manifest-path", default=None,
                        help="book-link-audit：build manifest 路径（默认 output/build_manifest.json）")
    parser.add_argument("--merged-path", default=None,
                        help="book-link-audit：merged 成书路径（--write-manifest 时必需）")
    parser.add_argument("--write-manifest", action="store_true",
                        help="book-link-audit：merge 后写 output/build_manifest.json")
    parser.add_argument("--update-merged", action="store_true",
                        help="book-link-audit：仅更新 manifest 的 merged 哈希（notes-normalize/proofread 后）")
    parser.add_argument("--contracts-path", default=None,
                        help="term-form-contract：合同制品路径（默认 input/term_form_contracts.json）")
    parser.add_argument("--stage", default="postprocess",
                        help="book-link-audit --update-merged：合并后处理阶段标签")
    parser.add_argument("--require-sealed", action="store_true",
                        help="book-link-audit：要求任务日志已封存且哈希一致")
    parser.add_argument("--require-current", action="store_true",
                        help="book-link-audit：profile/story graph stale 也判失败")
    parser.add_argument("--profile-path", default=None,
                        help="setup-payoff-audit 命令：book_profile.json 路径（默认 input/book_profile.json）")
    parser.add_argument("--top-n", type=int, default=5,
                        help="setup-payoff-audit 命令：每个 setup 保留的候选数（默认 5）")
    parser.add_argument("--audit-batch-size", type=int, default=8,
                        help="setup-payoff-audit 命令：LLM 选候选的批大小（默认 8，上限 10）")
    parser.add_argument("--audit-in", default=None,
                        help="setup-payoff-audit 命令：复用已有候选审计 JSON（不重复调用模型）")
    parser.add_argument("--llm", action="store_true",
                        help="setup-payoff-audit 命令：调用本地模型从候选 ID 中选择 payoff（失败 fail-open）")
    parser.add_argument("--l2", action="store_true",
                        help="setup-payoff-audit 命令：对 L1 通过的配对做低成本 L2 复核")
    parser.add_argument("--fix-from-review", action="store_true",
                        help="pipeline 命令：读取 review_dirty.json，二次校验锚句后定向重跑结构 P0 chunk")
    parser.add_argument("--review-dirty-path", default="output/review_dirty.json",
                        help="pipeline --fix-from-review：review_dirty JSON 路径（默认 output/review_dirty.json）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新初始化（init）或强制重审已校准报告（recalibrate-critic）")
    # 新增：split 命令专用参数
    parser.add_argument("--input", help="split 命令：输入文件路径（EPUB/TXT/MD）")
    parser.add_argument("--input-dir", default="input", help="split 命令：输出目录（默认 input/）")
    parser.add_argument("--input-format", choices=["auto", "epub", "txt", "md"], default="auto",
                        help="split 命令：输入格式（默认 auto=按扩展名自动判断）")
    parser.add_argument("--chapter-size", type=int, default=5000,
                        help="split 命令：TXT/MD 均分模式下的目标章节字数（默认 5000）")
    parser.add_argument("--min-chapter-size", type=int, default=1000,
                        help="split 命令：短于此字数的章节会被合并到下一章（默认 1000）")
    parser.add_argument("--apply", action="store_true",
                        help="consistency / notes-normalize / proofread / book-repair：执行自动修正；book-repair 只做有 source/raw/literary 证据的确定性修复")
    parser.add_argument("--safe-only", action="store_true",
                        help="pending-reconcile --apply：只消费 satisfied / 已 DIRTY 的 covered，outstanding 一律保留且不触发 DIRTY")
    parser.add_argument("--locked-target-only", action="store_true",
                        help="pending-reconcile --apply：只把 locked_decision_target_missing 行转 DIRTY，其余行全部保留")
    parser.add_argument("--verify", action="store_true",
                        help="book-repair：只复核当前 final 与修复 manifest，不写正文")
    parser.add_argument("--window", type=int, default=None,
                        help="drift-audit 命令：滑动窗口大小（默认 config.pipeline.drift_audit.window=5）")
    parser.add_argument("--jev", action="store_true",
                        help="drift-audit 命令：附加 Jev 窗口配对检查（需要 Jev 可用）")
    parser.add_argument("--json-out", default=None,
                        help="drift-audit / golden --judgment-ab / proofread / book-repair / query-gold-set：报告 JSON 输出路径")
    parser.add_argument("--ab", action="store_true",
                        help="golden/golden-gate 命令：运行 直译模型×thinking 4 组合 A/B 对比")
    parser.add_argument("--judgment-ab", action="store_true",
                        help="golden 命令：仅比较判断层注记注入 on/off（同章同摘录，不乘模型×effort 矩阵）")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="consolidate 命令：最大收敛轮数（默认 3）")
    parser.add_argument("--chunks", default=None,
                        help="requeue 命令：逗号分隔的 chunk_id；与 --from-ledger 二选一或并用")
    parser.add_argument("--from-ledger", action="store_true",
                        help="requeue 命令：从 output/rescue/failure_ledger.jsonl 选取仍隔离的 PF 块")
    parser.add_argument("--all", action="store_true",
                        help="rescue/completed-name-repair 命令：扫描全书；省略时只处理 --chapter 指定章节")
    parser.add_argument("--reset-rescue-budget", action="store_true",
                        help="rescue 命令：仅复位目标章节未完成块的救援/重试预算（算法升级后受控复活）")
    parser.add_argument("--workers", type=int, default=1,
                        help="recalibrate-critic 命令：并行重审线程数（默认 1）")
    parser.add_argument("--limit", type=int, default=None,
                        help="recalibrate-critic 命令：最多重审多少个目标 chunk（探针用）")
    parser.add_argument("--low-score-threshold", type=float, default=7.0,
                        help="recalibrate-critic 命令：低分报告阈值（默认 7.0）")
    parser.add_argument("--retries", type=int, default=1,
                        help="recalibrate-critic 命令：非校准/瞬时失败重试次数（默认 1）")
    parser.add_argument("--state-only", action="store_true",
                        help="name-only-fastpath：允许无文本变化但 pending 已满足的 DIRTY 块，仅以真实重审闭环状态")

    args = parser.parse_args()

    # P0-4: 章节 ID 合法性校验（路径遍历防御）
    # Round 52 修复：re.fullmatch 替代 re.match（避免 $ 在 final \n 前的边界 bug — "ch01\n" 原可通过）
    if args.chapter and not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', args.chapter):
        print(f"❌ 非法 chapter_id: '{args.chapter}' (仅允许字母/数字/下划线/连字符，1-32 字符)", file=sys.stderr)
        sys.exit(1)

    # Round 158 Phase 0: every CLI run gets a persisted, timestamped task log.
    # Write commands fail closed if the log cannot be created; read-only
    # diagnostics are allowed to continue with a stderr warning.
    _task_root = Path(__file__).resolve().parent.parent
    _task_read_only = not _command_needs_lock(args.command, args)
    try:
        _task_logger = make_logger(
            _task_root, args.command, read_only=_task_read_only,
        )
        _task_logger.start(chapter=args.chapter)
        set_active_logger(_task_logger)
    except TaskLoggingError as _task_log_exc:
        print(f"❌ [TaskLog] 无法创建本次运行日志，拒绝执行: {_task_log_exc}",
              file=sys.stderr)
        sys.exit(1)

    # Round 130: 单实例守卫（审计①）——写共享状态的命令必须独占锁；
    # 只读命令（gate/lint/drift-audit 等）不拦截，便于运行中诊断。
    if _command_needs_lock(args.command, args):
        if not acquire_instance_lock():
            try:
                _task_logger.finish(1, status="failed", message="lock_busy")
            except TaskLoggingError:
                pass
            sys.exit(1)

    _task_logger.event("command_started", stage="RUN", chapter=args.chapter)

    # Round 158 Phase 0: after the flock is held, publish a read-only recovery
    # report for chunks left in an active stage.  This never resets state and
    # never deletes final/critic backups; it only makes the interruption
    # auditable before the stage table resumes it.
    if args.command in {"pipeline", "rescue"}:
        _recovery_chapter = None if (args.command == "rescue" and args.all) else args.chapter
        try:
            _recovery = audit_and_write(
                _task_root, chapter_id=_recovery_chapter,
                run_id=_task_logger.run_id,
            )
            _rs = _recovery.get("summary") or {}
            if int(_rs.get("nonterminal") or 0):
                print(
                    f"♻️ [Recovery] 检测到 {_rs.get('nonterminal')} 个上次中断的非终态块；"
                    f"报告: {_recovery.get('report_path')}（状态与 backup 均未改动）",
                    file=sys.stderr,
                )
            _task_logger.event(
                "recovery_audit", stage="RECOVERY", status="ok",
                message="read-only in-flight recovery report",
                report_path=_recovery.get("report_path"),
                nonterminal=_rs.get("nonterminal", 0),
                by_state=_rs.get("by_state", {}),
            )
        except Exception as _recovery_exc:
            print(f"⚠️ [Recovery] 中断恢复审计失败（不阻塞主流程）: {_recovery_exc}",
                  file=sys.stderr)
            try:
                _task_logger.event(
                    "recovery_audit", stage="RECOVERY", status="failed",
                    error_class=type(_recovery_exc).__name__,
                    message=str(_recovery_exc),
                )
            except TaskLoggingError:
                pass

    # P4-9: 顶层异常包装（除已规范的 consistency/golden-gate 外，统一捕获并退出）
    try:
        if args.command == "pipeline":
            pipeline = TranslationPipeline(chapter_id=args.chapter)
            if args.fix_from_review:
                fix_res = pipeline.apply_review_fixes(Path(args.review_dirty_path))
                if not fix_res.get("ok"):
                    print(f"❌ [ReviewFix] 无法应用终审修复: {fix_res.get('error')}",
                          file=sys.stderr)
                    sys.exit(1)
                skipped = fix_res.get("skipped") or []
                if skipped:
                    reasons: dict = {}
                    for item in skipped:
                        reason = str(item.get("reason") or "unknown")
                        reasons[reason] = reasons.get(reason, 0) + 1
                    detail = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
                    print(f"ℹ️ [ReviewFix] {len(skipped)} 条未应用（{detail}）")
            if not pipeline.run():
                sys.exit(1)
        elif args.command == "init":
            if not init_project(chapter_id=args.chapter, force=args.force):
                sys.exit(1)
        elif args.command == "style-profile":
            from src.style_profiler import (
                derive_profile_digests,
                generate_book_profile,
                load_profile,
                save_profile,
                summary_line,
            )
            _chapters = [c.strip() for c in args.chapters.split(",") if c.strip()] if args.chapters else None
            if args.derive_digests:
                import shutil

                _profile_path = (Path(args.profile_path) if args.profile_path
                                 else Path.cwd() / "input" / "book_profile.json")
                _audit_path = Path.cwd() / "output" / "style_profile_audit.json"
                _prof = load_profile(_profile_path)
                if not _prof:
                    print(f"❌ style-profile --derive-digests: 无法读取 {_profile_path}",
                          file=sys.stderr)
                    sys.exit(1)
                try:
                    _audit = json.loads(_audit_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"❌ style-profile --derive-digests: 无法读取 {_audit_path}: {exc}",
                          file=sys.stderr)
                    sys.exit(1)
                _derived = derive_profile_digests(Path.cwd(), _prof, _audit)
                if not _derived.get("ok"):
                    print(f"❌ style-profile --derive-digests: {_derived.get('error')}",
                          file=sys.stderr)
                    sys.exit(1)
                _backup = _profile_path.with_name(
                    _profile_path.stem + ".pre_digest_backfill.json")
                if not _backup.exists():
                    shutil.copy2(_profile_path, _backup)
                save_profile(_derived["profile"], _profile_path)
                _summary = _derived.get("summary") or {}
                print(f"✅ chapter_digests 已写入 {_profile_path}: "
                      f"complete={_summary.get('complete')} "
                      f"chapters={_summary.get('chapters_complete')}/"
                      f"{_summary.get('chapters_total')} "
                      f"missing={','.join(_summary.get('missing') or []) or '-'}")
                print(f"   backup: {_backup}")
                sys.exit(0)
            res = generate_book_profile(
                Path.cwd(), chapters=_chapters, force=args.force,
                deep=args.deep,
                deep_max_chars=args.deep_max_chars,
                deep_overlap_chars=args.deep_overlap_chars,
            )
            if not res.get("ok"):
                print(f"❌ style-profile 失败: {res.get('error')}", file=sys.stderr)
                sys.exit(1)
            if res.get("skipped"):
                print(f"ℹ️ 已存在风格剖面，未重新生成: {res['path']}（--force 可强制重生成）")
            else:
                _prof = res.get("profile") or {}
                if res.get("saved", True):
                    print(f"✅ 风格剖面已写入 {res['path']}（samples: {', '.join(res.get('samples') or [])}）")
                else:
                    print(f"⚠️ 风格剖面未写入（LLM 不可用，未注入作品特定先验）: {_prof.get('note', '')}")
                print(f"   {summary_line(_prof)}")
                if res.get("deep"):
                    _cov = res.get("coverage") or {}
                    print(f"   deep: merge_mode={res.get('merge_mode') or '-'} "
                          f"chapters={_cov.get('chapters_with_active_notes', 0)}/{_cov.get('chapters_total', 0)} "
                          f"segments={_cov.get('segments_ok', 0)}/{_cov.get('segments_total', 0)} "
                          f"notes={res.get('active_notes', 0)} active/"
                          f"{_cov.get('active_after_limits', res.get('notes_after_limits', 0))} after-limits "
                          f"setup={_cov.get('setup_payoff_active', 0)}")
                    if res.get("graph_path"):
                        _gc = res.get("graph_coverage") or {}
                        print(f"   story_graph: {res.get('graph_path')} "
                              f"entity_coverage={_gc.get('entity_coverage', '-')} "
                              f"answerable={_gc.get('answerable', '-')}")
        elif args.command == "story-graph":
            from src.story_graph import build_and_save_story_graph
            from src.style_profiler import load_profile
            _root = Path.cwd()
            _prof = load_profile(_root / "input" / "book_profile.json")
            _graph = build_and_save_story_graph(
                _root, profile=_prof,
                path=Path(args.graph_path) if args.graph_path else None,
            )
            _cov = _graph.get("coverage") or {}
            _out = Path(args.graph_path) if args.graph_path else \
                _root / str((_config.get("story_graph", {}) or {}).get("path") or "input/story_graph.json")
            print(f"🧭 story-graph: {_out}")
            print(f"   entities={_cov.get('entities_with_appearance', 0)}/{_cov.get('entities_total', 0)} "
                  f"entity_coverage={_cov.get('entity_coverage', 0):.1%} "
                  f"chapters={_cov.get('chapters_with_entities', 0)}/{_cov.get('chapters_total', 0)} "
                  f"answerable={_cov.get('answerable')}")
            print(f"   scenes={_cov.get('scenes_with_entities', 0)}/{_cov.get('scenes_total', 0)} "
                  f"scene_coverage={_cov.get('scene_coverage', 0):.1%} "
                  f"mentions_in={(_graph.get('diagnostics') or {}).get('mentions_in_edges', 0)}")
            if _cov.get("uncovered_entities"):
                print(f"   uncovered_entities={len(_cov.get('uncovered_entities') or [])}")
            if _cov.get("uncovered_chapters"):
                print(f"   uncovered_chapters={','.join(_cov.get('uncovered_chapters') or [])}")
            if args.require_answerable and not _cov.get("answerable"):
                sys.exit(2)
        elif args.command == "book-link-audit":
            from src.book_linker import (
                DEFAULT_AUDIT_JSON,
                DEFAULT_AUDIT_MD,
                DEFAULT_MANIFEST_PATH,
                audit_book_links,
                build_manifest,
                update_manifest_merged,
                write_book_link_audit,
                write_manifest,
            )
            _root = Path.cwd()
            _manifest_path = (Path(args.manifest_path) if args.manifest_path
                              else _root / DEFAULT_MANIFEST_PATH)
            _merged_path = Path(args.merged_path) if args.merged_path else None
            if args.write_manifest:
                if not _merged_path:
                    print("❌ book-link-audit --write-manifest 需要 --merged-path",
                          file=sys.stderr)
                    sys.exit(2)
                _manifest = build_manifest(
                    _root, _merged_path,
                    input_dir=Path(args.input_dir) if args.input_dir else None,
                    run_id=os.environ.get("SAGATRANS_RUN_ID"))
                write_manifest(_manifest, _manifest_path)
                _totals = _manifest.get("totals") or {}
                print(f"🧾 build manifest: {_manifest_path} "
                      f"chapters={_totals.get('chapters')} "
                      f"chunks={_totals.get('db_chunks')} "
                      f"finals={_totals.get('finals_present')} "
                      f"missing={_totals.get('finals_missing')} "
                      f"orphan={_totals.get('orphan_finals')}")
                print(f"   merged: {(_manifest.get('merged') or {}).get('path')} "
                      f"{(_manifest.get('merged') or {}).get('sha256', '')[:12]}")
                sys.exit(0 if (_manifest.get("merged") or {}).get("exists")
                         and not int(_totals.get("finals_missing") or 0) else 1)
            if args.update_merged:
                if not _merged_path:
                    print("❌ book-link-audit --update-merged 需要 --merged-path",
                          file=sys.stderr)
                    sys.exit(2)
                _upd = update_manifest_merged(
                    _root, _manifest_path, _merged_path, stage=args.stage)
                if not _upd.get("ok"):
                    print(f"❌ book-link-audit --update-merged: {_upd.get('error')} "
                          f"{(json.dumps(_upd.get('details') or [], ensure_ascii=False)[:300])}",
                          file=sys.stderr)
                    sys.exit(1)
                print(f"🧾 merged 哈希已更新 ({args.stage}): "
                      f"{_upd.get('sha256', '')[:12]} chars={_upd.get('chars')}")
                sys.exit(0)
            _audit = audit_book_links(
                _root, manifest_path=_manifest_path, merged_path=_merged_path,
                input_dir=Path(args.input_dir) if args.input_dir else None,
                require_current=bool(args.require_current),
                require_sealed=bool(args.require_sealed))
            _json_path = Path(args.json_out) if args.json_out else _root / DEFAULT_AUDIT_JSON
            _md_path = _root / DEFAULT_AUDIT_MD
            write_book_link_audit(_audit, _json_path, _md_path)
            print(f"🔗 book-link-audit: {_audit.get('status')} "
                  f"passed={len(_audit.get('passed') or [])} "
                  f"warn={len(_audit.get('warnings') or [])} "
                  f"failed={len(_audit.get('failed') or [])}")
            for _name in (_audit.get("failed") or [])[:12]:
                print(f"   ❌ {_name}")
            for _name in (_audit.get("warnings") or [])[:8]:
                print(f"   ⚠️ {_name}")
            print(f"   report: {_json_path} / {_md_path}")
            sys.exit(0 if _audit.get("ok") else 1)
        elif args.command == "story-graph-query":
            from src.artifact_freshness import eval_freshness, load_story_graph_checked
            from src.style_profiler import load_profile
            from src.story_graph_query import (
                StoryGraphQueryIndex,
                default_story_graph_query_generate,
                evaluate_story_graph_query,
                format_query_result,
            )
            from src.judgment_layer import atomic_write_json
            _root = Path.cwd()
            _rel = str((_config.get("story_graph", {}) or {}).get("path")
                       or "input/story_graph.json")
            _path = Path(args.graph_path) if args.graph_path else _root / _rel
            _in_dir = Path(args.input_dir) if args.input_dir else _root / "input"
            _profile_path = (Path(args.profile_path) if args.profile_path
                             else _root / "input" / "book_profile.json")
            _prof = load_profile(_profile_path) if _profile_path.exists() else None
            _checked = load_story_graph_checked(
                _root, _path, _prof,
                require_current=not args.allow_stale,
                input_dir=_in_dir,
            )
            _graph = _checked.get("graph")
            if not _graph:
                print(f"❌ story-graph-query: 无法读取图 artifact: {_path}",
                      file=sys.stderr)
                sys.exit(1)
            if not _checked.get("ok"):
                print(f"❌ story-graph-query: 图已陈旧或缺少血缘 ({_checked.get('status')}): "
                      f"{', '.join(_checked.get('reasons') or [])}",
                      file=sys.stderr)
                print("   重建: python3 -m src.translator_agent story-graph；"
                      "仅作研究可显式加 --allow-stale", file=sys.stderr)
                sys.exit(1)
            if _checked.get("status") != "fresh":
                print(f"⚠️ story-graph-query: 使用非新鲜图（{_checked.get('status')}）: "
                      f"{', '.join(_checked.get('reasons') or [])}",
                      file=sys.stderr)
            _index = StoryGraphQueryIndex(_graph, input_dir=_in_dir)
            _gen = default_story_graph_query_generate if args.llm_judge else None
            if args.eval_cases:
                _cases_path = Path(args.eval_cases)
                try:
                    _payload = json.loads(_cases_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"❌ story-graph-query eval: 读取失败 {_cases_path}: {exc}",
                          file=sys.stderr)
                    sys.exit(1)
                _eval_check = eval_freshness(
                    _payload, _root, _prof, graph=_graph, input_dir=_in_dir)
                if not _eval_check.get("ok"):
                    _det = _eval_check.get("details") or {}
                    print(f"❌ story-graph-query eval: 评测制品无效/过期，拒绝计算指标 "
                          f"(reasons={','.join(_eval_check.get('reasons') or [])}; "
                          f"invalid_cases={_det.get('invalid_cases', 0)}; "
                          f"stale_notes={_det.get('stale_notes', 0)})",
                          file=sys.stderr)
                    sys.exit(1)
                _cases = _payload.get("cases") if isinstance(_payload, dict) else _payload
                _eval = evaluate_story_graph_query(
                    _index, _cases or [], top_k=args.top_k,
                    max_hops=args.max_hops, use_dpp=not args.no_dpp,
                    use_note_similarity=not args.no_note_similarity,
                    llm_generate=_gen,
                )
                _m = _eval.get("metrics") or {}
                print(f"🧪 story-graph-query eval: cases={_m.get('cases', 0)} "
                      f"R@5={_m.get('R@5', 0):.1%} R@10={_m.get('R@10', 0):.1%} "
                      f"AE@5={_m.get('AE@5', 0):.1%} AE@10={_m.get('AE@10', 0):.1%} "
                      f"anchor_precision@1={_m.get('anchor_precision@1', 0):.1%}")
                print(f"   evidence: gold_cases={_m.get('gold_cases', 0)} "
                      f"any@10={_m.get('any_evidence_R@10', 0):.1%} "
                      f"all@10={_m.get('all_evidence_R@10', 0):.1%} "
                      f"precision@2={_m.get('evidence_precision@2', 0):.1%} "
                      f"no_seed={_m.get('no_seed_rate', 0):.1%} "
                      f"negative_empty={_m.get('negative_empty_rate', 0):.1%}")
                if _m.get("note_evidence_cases"):
                    print(f"   note_evidence: n={_m.get('note_evidence_cases')} "
                          f"R@5={_m.get('note_evidence_R@5', 0):.1%} "
                          f"R@10={_m.get('note_evidence_R@10', 0):.1%} "
                          f"anchor_precision@1={_m.get('note_evidence_anchor_precision@1', 0):.1%}")
                if _m.get("alias_only_cases"):
                    print(f"   alias_only: n={_m.get('alias_only_cases')} "
                          f"R@5={_m.get('alias_only_R@5', 0):.1%} "
                          f"R@10={_m.get('alias_only_R@10', 0):.1%} "
                          f"anchor_precision@1={_m.get('alias_only_anchor_precision@1', 0):.1%}")
                if _m.get("note_similarity_cases"):
                    print(f"   note_similarity: n={_m.get('note_similarity_cases')} "
                          f"R@5={_m.get('note_similarity_R@5', 0):.1%} "
                          f"R@10={_m.get('note_similarity_R@10', 0):.1%} "
                          f"anchor_precision@1={_m.get('note_similarity_anchor_precision@1', 0):.1%}")
                if args.json_out:
                    atomic_write_json(Path(args.json_out), _eval)
                    print(f"   json_out: {args.json_out}")
                sys.exit(0)
            if not args.query and not args.note_id:
                print("❌ story-graph-query: 需要 --query/--question/--claim 或 --note-id",
                      file=sys.stderr)
                sys.exit(2)
            _result = _index.query(
                query=args.query or "", note_id=args.note_id,
                top_k=args.top_k, max_hops=args.max_hops,
                use_dpp=not args.no_dpp,
                use_note_similarity=not args.no_note_similarity,
                llm_generate=_gen,
            )
            print(format_query_result(_result, max_items=max(1, min(10, args.top_k))))
            if args.json_out:
                atomic_write_json(Path(args.json_out), _result)
                print(f"   json_out: {args.json_out}")
        elif args.command == "setup-payoff-audit":
            import shutil

            from src.judgment_layer import atomic_write_json
            from src.setup_payoff_audit import (
                DEFAULT_AUDIT_PATH,
                build_setup_payoff_audit,
                default_setup_payoff_audit_generate,
                format_setup_payoff_audit_summary,
                promote_setup_payoff_pairs,
                select_setup_payoff_candidates,
                verify_setup_payoff_selections,
            )
            from src.artifact_freshness import load_story_graph_checked
            from src.story_graph import build_story_graph, save_story_graph
            from src.story_graph_query import StoryGraphQueryIndex
            from src.style_profiler import load_profile, save_profile

            _root = Path.cwd()
            _profile_path = (Path(args.profile_path) if args.profile_path
                             else _root / "input" / "book_profile.json")
            _profile = load_profile(_profile_path)
            if not _profile:
                print(f"❌ setup-payoff-audit: 无法读取剖面 {_profile_path}",
                      file=sys.stderr)
                sys.exit(1)
            _rel = str((_config.get("story_graph", {}) or {}).get("path")
                       or "input/story_graph.json")
            _graph_path = Path(args.graph_path) if args.graph_path else _root / _rel
            _input_dir = Path(args.input_dir) if args.input_dir else _root / "input"
            _checked = load_story_graph_checked(
                _root, _graph_path, _profile,
                require_current=(not args.allow_stale) or bool(args.apply),
                input_dir=_input_dir,
            )
            _graph = _checked.get("graph")
            if not _graph:
                print(f"❌ setup-payoff-audit: 无法读取图 artifact {_graph_path}",
                      file=sys.stderr)
                sys.exit(1)
            if not _checked.get("ok"):
                print(f"❌ setup-payoff-audit: 图已陈旧或缺少血缘 ({_checked.get('status')}): "
                      f"{', '.join(_checked.get('reasons') or [])}",
                      file=sys.stderr)
                print("   先运行 story-graph；research-only 可使用 --allow-stale，"
                      "--apply 永远要求 fresh", file=sys.stderr)
                sys.exit(1)
            if _checked.get("status") != "fresh":
                print(f"⚠️ setup-payoff-audit: 使用非新鲜图（{_checked.get('status')}）: "
                      f"{', '.join(_checked.get('reasons') or [])}",
                      file=sys.stderr)
            _index = StoryGraphQueryIndex(_graph, input_dir=_input_dir)
            if args.audit_in:
                try:
                    _audit = json.loads(Path(args.audit_in).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    print(f"❌ setup-payoff-audit: 无法读取 --audit-in {args.audit_in}: {exc}",
                          file=sys.stderr)
                    sys.exit(1)
                if not isinstance(_audit, dict) or not isinstance(_audit.get("cases"), list):
                    print("❌ setup-payoff-audit: --audit-in 不是有效审计制品",
                          file=sys.stderr)
                    sys.exit(1)
                print(f"♻️ setup-payoff-audit: 复用审计制品 {args.audit_in}")
            else:
                _audit = build_setup_payoff_audit(
                    _profile, _index, top_n=args.top_n, max_hops=args.max_hops)
                if args.llm:
                    _audit = select_setup_payoff_candidates(
                        _audit, _index.chapter_texts,
                        default_setup_payoff_audit_generate,
                        batch_size=args.audit_batch_size)
                if args.l2:
                    if not args.llm:
                        print("⚠️ setup-payoff-audit: --l2 需要 --llm 选出的候选；本次 L2 将无事可做")
                    _audit = verify_setup_payoff_selections(
                        _audit, _index.chapter_texts, llm_generate=None,
                        batch_size=10)
            print(format_setup_payoff_audit_summary(_audit))
            if args.apply:
                if not args.audit_in and not (args.llm and args.l2):
                    print("❌ setup-payoff-audit --apply 需要同时指定 --llm 与 --l2，"
                          "或使用已完成 L2 的 --audit-in",
                          file=sys.stderr)
                    sys.exit(2)
                if args.audit_in and int((_audit.get("summary") or {}).get(
                        "promotion_eligible") or 0) <= 0:
                    print("❌ setup-payoff-audit: --audit-in 没有 promotion_eligible 配对；"
                          "请先跑 --llm --l2",
                          file=sys.stderr)
                    sys.exit(2)
                _promoted = promote_setup_payoff_pairs(_profile, _audit)
                # Compare against a fresh graph built from the current DB/profile,
                # not a possibly stale on-disk artifact.
                _baseline_graph = build_story_graph(
                    _root, profile=_profile, input_dir=_input_dir)
                _new_graph = build_story_graph(
                    _root, profile=_promoted["profile"], input_dir=_input_dir)
                _old_cov = _baseline_graph.get("coverage") or {}
                _new_cov = _new_graph.get("coverage") or {}
                _old_cert = float(_old_cov.get("certified_entity_coverage_of_trusted") or 0.0)
                _new_cert = float(_new_cov.get("certified_entity_coverage_of_trusted") or 0.0)

                def _unresolved(payload):
                    for gap in (payload.get("knowledge_gaps") or []):
                        if str(gap.get("code") or "") == "unresolved_aliases":
                            return int(gap.get("count") or 0)
                    return 0

                _old_alias = _unresolved(_baseline_graph)
                _new_alias = _unresolved(_new_graph)
                if _new_cert + 1e-9 < _old_cert or _new_alias > _old_alias:
                    print("❌ setup-payoff-audit apply: 图谱不变量失败；未写 profile/graph "
                          f"(cert {_old_cert:.4f}->{_new_cert:.4f}, "
                          f"unresolved_alias {_old_alias}->{_new_alias})",
                          file=sys.stderr)
                    sys.exit(1)
                _backup = _profile_path.with_name(
                    _profile_path.stem + ".pre_setup_payoff_audit.json")
                if not _backup.exists():
                    shutil.copy2(_profile_path, _backup)
                save_profile(_promoted["profile"], _profile_path)
                save_story_graph(_new_graph, path=_graph_path, project_root=_root)
                _audit["apply"] = {
                    "activated": _promoted["activated"],
                    "already_resolved": _promoted["already_resolved"],
                    "terminal": _promoted["terminal"],
                    "skipped_cap": _promoted["skipped_cap"],
                    "existing_active": _promoted["existing_active"],
                    "available_slots": _promoted["available_slots"],
                    "certified_entity_coverage_of_trusted": {
                        "before": _old_cert, "after": _new_cert},
                    "unresolved_aliases": {"before": _old_alias, "after": _new_alias},
                    "profile_backup": str(_backup),
                }
                print(f"✅ setup-payoff-audit apply: activated={len(_promoted['activated'])} "
                      f"terminal={len(_promoted['terminal'])} "
                      f"cert={_old_cert:.4f}->{_new_cert:.4f} "
                      f"unresolved_alias={_old_alias}->{_new_alias}")
            _out_path = (Path(args.json_out) if args.json_out
                         else _root / DEFAULT_AUDIT_PATH)
            atomic_write_json(_out_path, _audit)
            print(f"   json_out: {_out_path}")
        elif args.command == "drift-audit":
            from src.drift_audit import (
                audit_chapter,
                collect_chunk_texts,
                jev_window_check,
                summarize_report,
            )
            _report = audit_chapter(Path.cwd(), chapter=args.chapter, window=args.window)
            if args.jev:
                _chunks = collect_chunk_texts(Path.cwd() / "output" / args.chapter)
                _report["jev"] = jev_window_check(
                    _chunks, window=args.window or _report["window"]
                )
                _jv = _report["jev"]
                if _jv.get("available"):
                    print(f"🔬 Jev 窗口检查 (backend={_jv.get('backend')}): "
                          f"drifted_pairs={_jv.get('drifted_pairs')}")
                else:
                    print(f"⚠️ Jev 窗口检查不可用: {_jv.get('error')}", file=sys.stderr)
            print(summarize_report(_report))
            if args.json_out:
                Path(args.json_out).write_text(
                    json.dumps(_report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[out] {args.json_out}")
            sys.exit(0 if not _report.get("drift_flags") else 2)
        elif args.command == "golden":
            if args.judgment_ab:
                from src.style_profiler import load_profile
                _root = Path.cwd()
                _prof = load_profile(_root / "input" / "book_profile.json")
                if _prof is None:
                    print("⚠️ [JudgmentA/B] 未找到可读的 input/book_profile.json；"
                          "将只生成 keep_disabled 报告，不调用模型", file=sys.stderr)
                _ab_res = run_injection_ab(
                    args.chapter, _prof,
                    report_path=Path(args.json_out) if args.json_out else None,
                )
                if _ab_res.get("error"):
                    print(f"❌ [JudgmentA/B] {_ab_res.get('error')}", file=sys.stderr)
                    sys.exit(1)
                # Exit 2 means "no injected notes / no model call", matching
                # story-graph --require-answerable's explicit not-ready signal.
                sys.exit(0 if _ab_res.get("ok") else 2)
            elif args.ab:
                run_golden_ab()
            else:
                run_golden_test()
        elif args.command == "golden-gate":
            sys.exit(0 if run_golden_gate(use_ab=args.ab) else 1)
        elif args.command == "memory":
            test_memory_pressure()
        elif args.command == "debug":
            debug_db()
        elif args.command == "consistency":
            n_issues = run_consistency_check(dry_run=not args.apply)
            sys.exit(0 if n_issues == 0 else 2)
        elif args.command == "pending-reconcile":
            from src.pending_backtrack_reconcile import (
                apply_pending_backtrack_reconcile,
                audit_pending_backtracks,
            )
            _pr_root = Path(__file__).resolve().parent.parent
            if args.apply:
                _pr_res = apply_pending_backtrack_reconcile(
                    _pr_root, apply=True, safe_only=args.safe_only,
                    locked_target_only=args.locked_target_only)
            else:
                _pr_res = audit_pending_backtracks(_pr_root)
            _pr_counts = _pr_res.get("counts") or {}
            _pr_mode = "apply" if args.apply else "read-only"
            print("🧾 pending-reconcile(%s): total=%s, %s" % (
                _pr_mode,
                _pr_res.get("total", 0),
                ", ".join(f"{k}={v}" for k, v in _pr_counts.items())))
            if args.apply:
                print(f"   consumed={_pr_res.get('consumed', 0)} "
                      f"triggered_dirty={_pr_res.get('triggered_dirty', 0)}")
            if not _pr_res.get("ok"):
                print("   ⛔ 仍有 outstanding/blocked/orphan；详见 "
                      f"{_pr_res.get('md_path')}", file=sys.stderr)
            sys.exit(0 if _pr_res.get("ok") else 2)
        elif args.command == "term-form-audit":
            from src.term_form_audit import audit_term_forms
            _tfa_root = Path(__file__).resolve().parent.parent
            _tfa_res = audit_term_forms(
                _tfa_root,
                json_out=(Path(args.json_out) if args.json_out else None),
            )
            _tfa_counts = _tfa_res.get("counts") or {}
            print("🔎 term-form-audit(read-only): " + (
                ", ".join(f"{key}={value}"
                          for key, value in _tfa_counts.items()) or "clean"))
            for _tfa_group in (_tfa_res.get("groups") or [])[:12]:
                print(f"   group: #{_tfa_group.get('decision_id')} "
                      f"{_tfa_group.get('source_key')} → "
                      f"{_tfa_group.get('target')} / "
                      f"{_tfa_group.get('classification')} / "
                      f"pending={len(_tfa_group.get('pending_chunks') or [])}")
            print(f"   报告: {_tfa_res.get('md_path')}")
            for _tfa_err in (_tfa_res.get("errors") or [])[:8]:
                print(f"   ❌ {_tfa_err}", file=sys.stderr)
            sys.exit(0 if _tfa_res.get("ok") else 2)
        elif args.command == "locked-target-authority-audit":
            from src.locked_target_authority_audit import (
                audit_locked_target_authorities,
            )
            _lta_root = Path(__file__).resolve().parent.parent
            _lta_res = audit_locked_target_authorities(
                _lta_root,
                json_out=(Path(args.json_out) if args.json_out else None),
            )
            _lta_counts = _lta_res.get("counts") or {}
            print("🔒 locked-target-authority-audit(read-only): " + (
                ", ".join(f"{key}={value}"
                          for key, value in _lta_counts.items()) or "clean"))
            for _lta_group in (_lta_res.get("groups") or [])[:12]:
                print(f"   group: #{_lta_group.get('decision_id')} "
                      f"{_lta_group.get('source_key')} → "
                      f"{_lta_group.get('target')} / "
                      f"{_lta_group.get('classification')} / "
                      f"hits={_lta_group.get('target_hits')}/"
                      f"{_lta_group.get('authority_chunks')}")
            print(f"   报告: {_lta_res.get('md_path')}")
            for _lta_err in (_lta_res.get("errors") or [])[:8]:
                print(f"   ❌ {_lta_err}", file=sys.stderr)
            sys.exit(0 if _lta_res.get("ok") else 2)
        elif args.command == "locked-target-authority-repair":
            from src.decision_authority_align import (
                apply_locked_target_authority_repairs,
                audit_locked_target_authority_repairs,
            )
            _ltr_root = Path(__file__).resolve().parent.parent
            if args.apply:
                _ltr_res = apply_locked_target_authority_repairs(
                    _ltr_root, apply=True, run_id=_task_logger.run_id)
            else:
                _ltr_res = audit_locked_target_authority_repairs(_ltr_root)
            _ltr_counts = (_ltr_res.get("counts")
                           or _ltr_res.get("plan_counts") or {})
            _ltr_mode = "apply" if args.apply else "read-only"
            print(f"🔧 locked-target-authority-repair({_ltr_mode}): " + (
                ", ".join(f"{key}={value}"
                          for key, value in _ltr_counts.items()) or "clean"))
            for _ltr_cand in (_ltr_res.get("candidates") or [])[:12]:
                print(f"   candidate: Decision #{_ltr_cand.get('decision_id')} "
                      f"{_ltr_cand.get('source_key')} "
                      f"{_ltr_cand.get('old_translation')} → "
                      f"{_ltr_cand.get('new_translation')} "
                      f"pending={len(_ltr_cand.get('pending_chunks') or [])}")
            for _ltr_blocked in (_ltr_res.get("blocked_candidates") or [])[:8]:
                print(f"   blocked: Decision #{_ltr_blocked.get('decision_id')} "
                      f"{_ltr_blocked.get('reason')}")
            print(f"   报告: {_ltr_res.get('md_path')}")
            for _ltr_err in (_ltr_res.get("errors") or [])[:8]:
                print(f"   ❌ {_ltr_err}", file=sys.stderr)
            sys.exit(0 if _ltr_res.get("ok") else 2)
        elif args.command == "term-form-contract":
            from src.term_form_contracts import (
                apply_term_form_contracts,
                build_term_form_contracts,
            )
            _tfc_root = Path(__file__).resolve().parent.parent
            _tfc_path = (Path(args.contracts_path) if args.contracts_path
                         else None)
            if args.apply:
                _tfc_res = apply_term_form_contracts(
                    _tfc_root, contracts_path=_tfc_path,
                    run_id=_task_logger.run_id,
                )
            else:
                _tfc_res = build_term_form_contracts(
                    _tfc_root, contracts_path=_tfc_path,
                    report_json=(Path(args.json_out) if args.json_out else None),
                )
            _tfc_mode = "apply" if args.apply else "read-only"
            print("🧩 term-form-contract(%s): " % _tfc_mode + (
                ", ".join(f"{key}={value}"
                          for key, value in (_tfc_res.get("counts") or {}).items())
                or "no contracts"))
            for _tfc_contract in (_tfc_res.get("contracts") or [])[:12]:
                print(f"   contract: #{_tfc_contract.get('decision_id')} "
                      f"{_tfc_contract.get('source_key')} / "
                      f"{_tfc_contract.get('status')} / "
                      f"forms=" + ",".join(
                          form.get("form", "")
                          for form in _tfc_contract.get("forms") or []))
            if _tfc_res.get("artifact_path"):
                print(f"   artifact: {_tfc_res.get('artifact_path')}")
            print(f"   报告: {_tfc_res.get('md_path')}")
            for _tfc_err in (_tfc_res.get("errors") or [])[:8]:
                print(f"   ❌ {_tfc_err}", file=sys.stderr)
            sys.exit(0 if _tfc_res.get("ok") else 2)
        elif args.command == "name-authority-audit":
            from src.name_authority_audit import audit_name_authorities
            _na_root = Path(__file__).resolve().parent.parent
            _na_res = audit_name_authorities(_na_root)
            _na_counts = _na_res.get("counts") or {}
            print("🧭 name-authority-audit(read-only): " + (
                ", ".join(f"{k}={v}" for k, v in _na_counts.items()) or "clean"))
            print(f"   multi_form_groups={_na_res.get('multi_form_count', 0)} "
                  f"needs_form_contract={_na_res.get('multi_form_needing_contract', 0)}")
            print(f"   报告: {_na_res.get('md_path')}")
            if not _na_res.get("ok"):
                print("   ⛔ 权威分歧尚未对齐；name-only 写闭环不得启用。",
                      file=sys.stderr)
            for _na_err in (_na_res.get("errors") or [])[:8]:
                print(f"   ❌ {_na_err}", file=sys.stderr)
            sys.exit(0 if _na_res.get("ok") else 2)
        elif args.command == "name-authority-align":
            from src.name_authority_align import align_name_authorities
            _nal_root = Path(__file__).resolve().parent.parent
            _nal_res = align_name_authorities(
                _nal_root,
                apply=args.apply,
                json_out=(Path(args.json_out) if args.json_out else None),
                run_id=_task_logger.run_id,
            )
            _nal_mode = "apply" if args.apply else "read-only"
            _nal_counts = (_nal_res.get("counts")
                           or _nal_res.get("plan_counts") or {})
            print(f"🧭 name-authority-align({_nal_mode}): " + (
                ", ".join(f"{k}={v}" for k, v in _nal_counts.items()) or "clean"))
            _nal_after = _nal_res.get("audit_after") or {}
            if _nal_after:
                print(f"   audit_after: ok={_nal_after.get('ok')} "
                      f"blocking={_nal_after.get('blocking_count', 0)} "
                      f"status={_nal_after.get('status')}")
            print(f"   write_attempted={_nal_res.get('write_attempted')} "
                  f"applied={_nal_res.get('applied_count', 0)}")
            for _nal_act in (_nal_res.get("actions") or [])[:12]:
                print(f"   action: id={_nal_act.get('registry_id')} "
                      f"{_nal_act.get('action')} {_nal_act.get('status', 'planned')}")
            for _nal_manual in (_nal_res.get("manual_review") or [])[:8]:
                print("   manual: " + json.dumps(_nal_manual, ensure_ascii=False))
            print(f"   报告: {_nal_res.get('md_path')}")
            for _nal_err in (_nal_res.get("errors") or [])[:8]:
                print(f"   ❌ {_nal_err}", file=sys.stderr)
            sys.exit(0 if _nal_res.get("ok") else 2)
        elif args.command == "decision-authority-align":
            from src.decision_authority_align import (
                apply_decision_authority_alignment,
                audit_decision_authorities,
            )
            _dac_root = Path(__file__).resolve().parent.parent
            if args.apply:
                _dac_res = apply_decision_authority_alignment(
                    _dac_root, apply=True, run_id=_task_logger.run_id)
            else:
                _dac_res = audit_decision_authorities(_dac_root)
            _dac_counts = _dac_res.get("counts") or _dac_res.get("plan_counts") or {}
            _dac_mode = "apply" if args.apply else "read-only"
            print(f"⚖️ decision-authority-align({_dac_mode}): " + (
                ", ".join(f"{key}={value}"
                          for key, value in _dac_counts.items()) or "clean"))
            for _dac_cand in (_dac_res.get("candidates")
                              or _dac_res.get("planned") or [])[:12]:
                print(f"   candidate: Decision #{_dac_cand.get('decision_id')} "
                      f"{_dac_cand.get('source_key')} "
                      f"{_dac_cand.get('old_translation')} → "
                      f"{_dac_cand.get('new_translation')} "
                      f"repair_chunks={len(_dac_cand.get('repair_chunks') or [])}")
            for _dac_act in (_dac_res.get("actions") or [])[:12]:
                print(f"   action: Decision #{_dac_act.get('decision_id')} "
                      f"{_dac_act.get('status')} "
                      f"queued={len(_dac_act.get('queued_chunks') or [])}")
            for _dac_blocked in (_dac_res.get("blocked_candidates") or [])[:8]:
                print("   manual: " + json.dumps(
                    {"decision_id": _dac_blocked.get("decision_id"),
                     "source_key": _dac_blocked.get("source_key"),
                     "reason": _dac_blocked.get("reason")},
                    ensure_ascii=False))
            print(f"   报告: {_dac_res.get('md_path')}")
            for _dac_err in (_dac_res.get("errors") or [])[:8]:
                print(f"   ❌ {_dac_err}", file=sys.stderr)
            sys.exit(0 if _dac_res.get("ok") else 2)
        elif args.command == "decision-target-authority-repair":
            from src.decision_target_authority_repair import (
                apply_decision_target_repairs,
                audit_decision_target_repairs,
            )
            _dtar_root = Path(__file__).resolve().parent.parent
            if args.apply:
                _dtar_res = apply_decision_target_repairs(
                    _dtar_root, apply=True, run_id=_task_logger.run_id)
            else:
                _dtar_res = audit_decision_target_repairs(_dtar_root)
            _dtar_counts = (_dtar_res.get("plan_counts")
                            or _dtar_res.get("counts") or {})
            _dtar_mode = "apply" if args.apply else "read-only"
            print(f"🎯 decision-target-authority-repair({_dtar_mode}): " + (
                ", ".join(f"{key}={value}"
                          for key, value in _dtar_counts.items()) or "clean"))
            for _dtar_act in (_dtar_res.get("actions")
                              or _dtar_res.get("candidates") or [])[:12]:
                _dtar_queued = len(
                    _dtar_act.get("queued_chunks")
                    or _dtar_act.get("stale_chunks") or [])
                print(f"   action: Decision #{_dtar_act.get('decision_id')} "
                      f"{_dtar_act.get('source_key')} "
                      f"{_dtar_act.get('old_translation')} → "
                      f"{_dtar_act.get('new_translation')} "
                      f"pending={len(_dtar_act.get('pending_chunks') or [])} "
                      f"queued={_dtar_queued}")
            for _dtar_blocked in (_dtar_res.get("blocked_groups") or [])[:12]:
                print(f"   blocked: {_dtar_blocked.get('source_key')} / "
                      f"{_dtar_blocked.get('reason')}")
            print(f"   报告: {_dtar_res.get('md_path')}")
            for _dtar_err in (_dtar_res.get("errors") or [])[:8]:
                print(f"   ❌ {_dtar_err}", file=sys.stderr)
            sys.exit(0 if _dtar_res.get("ok") else 2)
        elif args.command == "query-gold-set":
            from src.artifact_freshness import load_story_graph_checked
            from src.query_gold_set import build_query_gold_set
            from src.style_profiler import load_profile

            _qg_root = Path(__file__).resolve().parent.parent
            _qg_in = (Path(args.input_dir) if args.input_dir
                      else _qg_root / "input")
            if not _qg_in.is_absolute():
                _qg_in = _qg_root / _qg_in
            _qg_profile_path = (Path(args.profile_path) if args.profile_path
                                else _qg_in / "book_profile.json")
            if not _qg_profile_path.is_absolute():
                _qg_profile_path = _qg_root / _qg_profile_path
            _qg_profile = load_profile(_qg_profile_path)
            if not _qg_profile:
                print(f"❌ query-gold-set: 无法读取剖面 {_qg_profile_path}",
                      file=sys.stderr)
                sys.exit(1)
            _qg_rel = str((_config.get("story_graph", {}) or {}).get("path")
                          or "input/story_graph.json")
            _qg_graph_path = (Path(args.graph_path) if args.graph_path
                              else _qg_root / _qg_rel)
            if not _qg_graph_path.is_absolute():
                _qg_graph_path = _qg_root / _qg_graph_path
            _qg_checked = load_story_graph_checked(
                _qg_root, _qg_graph_path, _qg_profile,
                require_current=True, input_dir=_qg_in,
            )
            if not _qg_checked.get("ok") or not _qg_checked.get("graph"):
                print("❌ query-gold-set: 图不是 fresh artifact "
                      f"({_qg_checked.get('status')}): "
                      f"{', '.join(_qg_checked.get('reasons') or [])}",
                      file=sys.stderr)
                sys.exit(1)
            _qg_res = build_query_gold_set(
                _qg_root,
                profile=_qg_profile,
                graph=_qg_checked.get("graph"),
                profile_path=_qg_profile_path,
                graph_path=_qg_graph_path,
                input_dir=_qg_in,
                output_path=(Path(args.json_out) if args.json_out else None),
            )
            print("🧪 query-gold-set: " + (
                ", ".join(f"{key}={value}"
                          for key, value in (_qg_res.get("counts") or {}).items())
                or "no cases"))
            print(f"   status={_qg_res.get('status')} "
                  f"eval_written={_qg_res.get('eval_written')} "
                  f"case_count={_qg_res.get('case_count', 0)}")
            if _qg_res.get("gaps"):
                print("   gaps: " + ", ".join(
                    f"{key}={value}" for key, value in _qg_res["gaps"].items()),
                    file=sys.stderr)
            for _qg_err in (_qg_res.get("errors") or [])[:8]:
                print(f"   ❌ {_qg_err}", file=sys.stderr)
            print(f"   报告: {_qg_res.get('report_path')}")
            sys.exit(0 if _qg_res.get("ok") else 2)
        elif args.command == "name-only-fastpath":
            from src.name_only_fastpath import (
                apply_name_only_fastpath,
                audit_name_only_fastpath,
            )
            _nf_root = Path(__file__).resolve().parent.parent
            if args.apply:
                _nf_res = apply_name_only_fastpath(
                    _nf_root, chapter_id=args.chapter, apply=True,
                    run_id=_task_logger.run_id, limit=args.limit,
                    allow_state_only=args.state_only,
                )
            else:
                _nf_res = audit_name_only_fastpath(
                    _nf_root, chapter_id=args.chapter,
                    allow_state_only=args.state_only)
            _nf_counts = _nf_res.get("counts") or {}
            _nf_mode = "apply" if args.apply else "read-only"
            print(f"🧪 name-only-fastpath({_nf_mode}): " + ", ".join(
                f"{k}={v}" for k, v in _nf_counts.items()))
            print(f"   报告: {_nf_res.get('md_path')}")
            if _nf_res.get("authority_blocked"):
                print("   ⛔ 名称权威审计未清零；写闭环已拒绝执行。",
                      file=sys.stderr)
            for _nf_rec in (_nf_res.get("candidates") or [])[:12]:
                print(f"   candidate: {_nf_rec.get('chunk_id')} "
                      f"mode={_nf_rec.get('change_mode', 'name_only')} "
                      f"before={str(_nf_rec.get('final_sha256_before'))[:12]} "
                      f"after={str(_nf_rec.get('final_sha256_after'))[:12]}")
            for _nf_act in (_nf_res.get("actions") or [])[:12]:
                print(f"   action: {_nf_act.get('chunk_id')} "
                      f"{_nf_act.get('status')} / {_nf_act.get('reason')}")
            if _nf_res.get("failed"):
                for _nf_err in _nf_res["failed"][:8]:
                    print(f"   ❌ {_nf_err}", file=sys.stderr)
            sys.exit(0 if _nf_res.get("ok") else 1)
        elif args.command == "completed-name-repair":
            from src.name_only_fastpath import (
                apply_completed_name_repair,
                audit_completed_name_repair,
            )
            _cn_root = Path(__file__).resolve().parent.parent
            _cn_chapter = None if getattr(args, "all", False) else args.chapter
            if args.apply:
                _cn_res = apply_completed_name_repair(
                    _cn_root, chapter_id=_cn_chapter, apply=True,
                    run_id=_task_logger.run_id, limit=args.limit,
                )
            else:
                _cn_res = audit_completed_name_repair(
                    _cn_root, chapter_id=_cn_chapter)
            _cn_counts = _cn_res.get("counts") or {}
            _cn_mode = "apply" if args.apply else "read-only"
            print(f"🧹 completed-name-repair({_cn_mode}): " + ", ".join(
                f"{k}={v}" for k, v in _cn_counts.items()))
            print(f"   报告: {_cn_res.get('md_path')}")
            if _cn_res.get("authority_blocked"):
                print("   ⛔ 名称权威审计未清零；COMPLETED 修复已拒绝执行。",
                      file=sys.stderr)
            for _cn_rec in (_cn_res.get("candidates") or [])[:12]:
                print(f"   candidate: {_cn_rec.get('chunk_id')} "
                      f"{_cn_rec.get('source_key')} → "
                      f"{_cn_rec.get('canonical_name')} "
                      f"({len(_cn_rec.get('changed_spans') or [])} 处)")
            for _cn_act in (_cn_res.get("actions") or [])[:12]:
                print(f"   action: {_cn_act.get('chunk_id')} "
                      f"{_cn_act.get('status')} / {_cn_act.get('reason')}")
            if _cn_res.get("after_counts"):
                print("   pending_after: " + ", ".join(
                    f"{k}={v}" for k, v in _cn_res["after_counts"].items()))
            sys.exit(0 if _cn_res.get("ok") else 1)
        elif args.command == "confirmed-term-drift-repair":
            from src.name_only_fastpath import (
                apply_confirmed_term_drift_repair,
                audit_confirmed_term_drift_repair,
            )
            _ct_root = Path(__file__).resolve().parent.parent
            _ct_chapter = None if getattr(args, "all", False) else args.chapter
            if args.apply:
                _ct_res = apply_confirmed_term_drift_repair(
                    _ct_root, chapter_id=_ct_chapter, apply=True,
                    run_id=_task_logger.run_id, limit=args.limit,
                )
            else:
                _ct_res = audit_confirmed_term_drift_repair(
                    _ct_root, chapter_id=_ct_chapter)
            _ct_counts = _ct_res.get("counts") or {}
            _ct_mode = "apply" if args.apply else "read-only"
            print(f"🧭 confirmed-term-drift-repair({_ct_mode}): " + ", ".join(
                f"{k}={v}" for k, v in _ct_counts.items()))
            print(f"   报告: {_ct_res.get('md_path')}")
            for _ct_rec in (_ct_res.get("candidates") or [])[:12]:
                print(f"   candidate: Decision #{_ct_rec.get('decision_id')} "
                      f"{_ct_rec.get('source_key')} / "
                      f"{_ct_rec.get('wrong_form')} → "
                      f"{_ct_rec.get('canonical_name')} / chunks="
                      f"{len(_ct_rec.get('chunks') or [])}")
            for _ct_act in (_ct_res.get("actions") or [])[:12]:
                print(f"   action: Decision #{_ct_act.get('decision_id')} "
                      f"{_ct_act.get('status')} / {_ct_act.get('reason')} / "
                      f"chunks={len(_ct_act.get('chunks') or [])}")
            for _ct_blocked in (_ct_res.get("blocked") or [])[:8]:
                print(f"   blocked: Decision #{_ct_blocked.get('decision_id')} "
                      f"{_ct_blocked.get('source_key')} / "
                      f"{_ct_blocked.get('reason')}", file=sys.stderr)
            for _ct_err in (_ct_res.get("errors") or [])[:8]:
                print(f"   ❌ {_ct_err}", file=sys.stderr)
            sys.exit(0 if _ct_res.get("ok") else 1)
        elif args.command == "consolidate":
            from src.pipeline import consolidate_dirty_chapters
            _cons_res = consolidate_dirty_chapters(max_rounds=args.max_rounds)
            sys.exit(0 if _cons_res.get("converged") else 1)
        elif args.command == "requeue":
            # Round 135: PF 终态块复活（≥1 次评审通道）——把 PERMANENTLY_FAILED
            # 恢复为 DIRTY 重跑；--force 绕 pf_requeue_max 预算。复活块走 dirty 通道：
            # final.json 自动转 backup，重跑失败由 _restore_dirty_backups 恢复旧版。
            _rq_root = Path(__file__).resolve().parent.parent
            _rq_sched = TaskScheduler(db_path=str(_rq_root / "db" / "workflow.db"))
            _rq_chunks = []
            if args.chunks:
                _rq_chunks.extend(c.strip() for c in args.chunks.split(",") if c.strip())
            if args.from_ledger:
                _ledger_ids = ledger_requeue_candidates(_rq_root, args.chapter)
                for _cid in _ledger_ids:
                    if _cid not in _rq_chunks:
                        _rq_chunks.append(_cid)
                print(f"📒 requeue ledger: {args.chapter} 发现 {len(_ledger_ids)} 个隔离块")
            _rq_res = _rq_sched.requeue_permanently_failed(
                args.chapter, chunk_ids=(_rq_chunks or None), force=args.force)
            _rq_n = len(_rq_res["requeued"])
            print(f"♻️ requeue: {args.chapter} 共 {_rq_n} 个 PF 块恢复 DIRTY")
            if _rq_res["requeued"]:
                print(f"    {', '.join(_rq_res['requeued'])}")
                print(f"    → 运行: python3 -m src.translator_agent pipeline --chapter {args.chapter}")
            if _rq_res["over_cap"]:
                print(f"    ⚠️ 复活预算用尽(pf_requeue_max): {', '.join(_rq_res['over_cap'])}（--force 可强制）")
            sys.exit(0 if _rq_n or not _rq_res["over_cap"] else 1)
        elif args.command == "rescue":
            _rs_root = Path(__file__).resolve().parent.parent
            _rs_db = _rs_root / "db" / "workflow.db"
            _rs_sched = TaskScheduler(db_path=str(_rs_db))
            if args.reset_rescue_budget:
                if not args.apply:
                    print("❌ --reset-rescue-budget 需要与 --apply 同用", file=sys.stderr)
                    sys.exit(2)
                _reset_chapters = []
                if args.all:
                    _reset_chapters = [row[0] for row in _rs_sched.conn.execute(
                        "SELECT DISTINCT chapter_id FROM chunk_tasks ORDER BY chapter_id").fetchall()]
                else:
                    _reset_chapters = [args.chapter]
                for _rch in _reset_chapters:
                    _reset_ids = _rs_sched.reset_rescue_budget(_rch)
                    print(f"♻️ [Rescue] {_rch}: 已复位 {len(_reset_ids)} 个块的救援/重试预算")
            _rs_rows = _rs_sched.conn.execute(
                "SELECT DISTINCT chapter_id, state, chunk_id FROM chunk_tasks "
                "WHERE state IN ('PERMANENTLY_FAILED','FAILED','DIRTY') "
                "ORDER BY chapter_id, chunk_id").fetchall()
            _rs_by_ch = {}
            for _ch, _st, _cid in _rs_rows:
                _rs_by_ch.setdefault(_ch, []).append((_cid, _st))
            _rs_final = collect_final_status(_rs_root, _rs_root / "output")
            for _item in _rs_final.get("fallbacks", []):
                _ch = str(_item.get("chapter") or "")
                if _ch:
                    _rs_by_ch.setdefault(_ch, [])
                    _cid = str(_item.get("chunk_id") or "")
                    _rs_by_ch[_ch].append((_cid, "FALLBACK_FINAL"))
            if not args.all:
                _rs_by_ch = {args.chapter: _rs_by_ch.get(args.chapter, [])}
            _rs_by_ch = {k: v for k, v in _rs_by_ch.items() if v}
            print(f"🧭 rescue: {len(_rs_by_ch)} 章存在可救援失败块")
            for _ch in sorted(_rs_by_ch):
                print(f"   {_ch}: {len(_rs_by_ch[_ch])} 个待判定块")
            if not args.apply:
                print("   只读模式；加 --apply 执行定向救援")
                sys.exit(0)
            _rs_failed = []
            for _ch in sorted(_rs_by_ch):
                print(f"♻️ [Rescue] 处理章节 {_ch}")
                try:
                    _ok = TranslationPipeline(chapter_id=_ch).run()
                except Exception as _rs_exc:
                    _ok = False
                    print(f"❌ [Rescue] {_ch} 异常: {_rs_exc}", file=sys.stderr)
                if not _ok:
                    _rs_failed.append(_ch)
            try:
                write_rescue_report(_rs_root)
            except Exception:
                pass
            # run() may return True when a chapter has a complete fallback set;
            # that is not a successful rescue.  Re-scan the state machine and
            # current final artifacts so this command has a truthful exit code.
            _rs_remaining = []
            for _ch in sorted(_rs_by_ch):
                try:
                    for _task in _rs_sched.get_all_tasks_by_chapter(_ch):
                        if _task.get("state") != "COMPLETED":
                            _rs_remaining.append(
                                (str(_task.get("chunk_id")), str(_task.get("state"))))
                except Exception as _scan_exc:
                    print(f"⚠️ [Rescue] {_ch} 残留扫描异常: {_scan_exc}", file=sys.stderr)
                    _rs_remaining.append((f"{_ch}:scan", "SCAN_ERROR"))
            _rs_final_left = collect_final_status(_rs_root, _rs_root / "output").get("fallbacks", [])
            if _rs_remaining or _rs_final_left:
                print(f"❌ rescue 后仍有 {len(_rs_remaining)} 个非完成块、"
                      f"{len(_rs_final_left)} 个 fallback final；需进入 gate 判定", file=sys.stderr)
                if _rs_remaining:
                    print("   " + ", ".join(f"{cid}:{st}" for cid, st in _rs_remaining[:12]),
                          file=sys.stderr)
                if _rs_final_left:
                    print("   " + ", ".join(str(x.get("chunk_id")) for x in _rs_final_left[:12]),
                          file=sys.stderr)
            sys.exit(0 if not _rs_failed and not _rs_remaining and not _rs_final_left else 1)
        elif args.command == "qa-report":
            _qa_root = Path(__file__).resolve().parent.parent
            _qa_res = write_qa_report(_qa_root, _qa_root / "output")
            print(f"📝 QA_REPORT: fallback={len(_qa_res.get('fallbacks', []))}, "
                  f"finals={_qa_res.get('final_count', 0)} → {_qa_res.get('path')}")
            sys.exit(0 if _qa_res.get("ok") else 1)
        elif args.command == "recalibrate-critic":
            from src.recalibrate import run_recalibrate_critic
            _rc_chunks = [c.strip() for c in (args.chunks or "").split(",") if c.strip()]
            _rc_res = run_recalibrate_critic(
                chapter_id=args.chapter,
                all_chapters=bool(args.all),
                chunk_ids=_rc_chunks or None,
                limit=args.limit,
                workers=args.workers,
                apply=bool(args.apply),
                low_score_threshold=args.low_score_threshold,
                retries=args.retries,
                force=bool(args.force),
            )
            _rc_mode = "apply" if _rc_res.get("applied") else "dry-run"
            print(f"🧪 recalibrate-critic({_rc_mode}): selected={_rc_res.get('selected')} "
                  f"recalibrated={_rc_res.get('recalibrated')} failed={_rc_res.get('failed')} "
                  f"low_score={_rc_res.get('low_score')}")
            if _rc_res.get("report_path"):
                print(f"   报告: {_rc_res['report_path']}")
            if _rc_res.get("errors"):
                for _rc_err in _rc_res["errors"][:12]:
                    print(f"   ❌ {_rc_err}", file=sys.stderr)
            sys.exit(0 if _rc_res.get("ok") else 1)
        elif args.command == "gate":
            from src.pipeline import run_final_gate
            _gate_res = run_final_gate()
            if not _gate_res.get("ok"):
                try:
                    _task_logger.event(
                        "gate_failed", stage="GATE", status="failed",
                        message="final gate rejected the delivery",
                        failures=(_gate_res.get("failures") or [])[:20],
                    )
                except TaskLoggingError:
                    pass
            sys.exit(0 if _gate_res.get("ok") else 1)
        elif args.command == "finalize":
            from src.book_lint import finalize_outputs
            finalize_outputs(apply=args.apply)
            sys.exit(0)
        elif args.command == "notes-normalize":
            from src.book_lint import notes_normalize_outputs
            _nn_res = notes_normalize_outputs(apply=args.apply)
            _nn_stats = _nn_res.get("stats") or {}
            print(f"notes-normalize: files={_nn_stats.get('files', 0)} "
                  f"changed={_nn_stats.get('changed', 0)} → {_nn_res.get('report', '')}")
            sys.exit(0)
        elif args.command == "proofread":
            from src.proofread import proofread_book
            _pr_res = proofread_book(
                project_root=Path(__file__).resolve().parent.parent,
                apply=bool(args.apply),
                json_out=args.json_out,
            )
            if _pr_res.get("error"):
                print(f"❌ proofread: {_pr_res['error']}", file=sys.stderr)
            sys.exit(0 if _pr_res.get("ok") else 1)
        elif args.command == "book-repair":
            from src.book_repair import apply_book_repairs, verify_book_repairs
            _br_root = Path(__file__).resolve().parent.parent
            if args.verify:
                _br_res = verify_book_repairs(
                    project_root=_br_root,
                    output_dir=_br_root / "output",
                )
                print(
                    f"🩺 book-repair(verify): scanned={_br_res.get('scanned', 0)} "
                    f"pending={len(_br_res.get('pending_chunks') or [])} "
                    f"blocking={len(_br_res.get('blocking_issues') or [])} "
                    f"manifest_current={_br_res.get('manifest_current')}")
            else:
                _br_res = apply_book_repairs(
                    project_root=_br_root,
                    output_dir=_br_root / "output",
                    apply=bool(args.apply),
                    json_out=args.json_out,
                )
            for _br_issue in (_br_res.get("blocking_issues") or [])[:12]:
                print(
                    f"   ❌ {_br_issue.get('chunk_id')}: "
                    f"{_br_issue.get('kind')} — {_br_issue.get('detail')}",
                    file=sys.stderr)
            for _br_stale in (_br_res.get("stale_manifest") or [])[:8]:
                print(
                    f"   ❌ 修复凭证陈旧: {_br_stale.get('chunk_id')}",
                    file=sys.stderr)
            for _br_missing in (_br_res.get("missing_sources") or [])[:8]:
                print(f"   ❌ 缺少 DB source 证据: {_br_missing}", file=sys.stderr)
            for _br_bad_final in (_br_res.get("malformed_finals") or [])[:8]:
                print(f"   ❌ final 无法解析: {_br_bad_final}", file=sys.stderr)
            sys.exit(0 if _br_res.get("ok") else 1)
        elif args.command == "lint":
            from src.book_lint import lint_book
            sys.exit(0 if lint_book().get("ok") else 1)
        elif args.command == "split":
            if not args.input:
                # P0-3: split 提示命令按 __package__ 自适应
                if __package__:
                    cmd_hint = f"python -m {__package__} split --input <file.epub|txt|md>"
                else:
                    cmd_hint = "python translator_agent.py split --input <file.epub|txt|md>"
                print(f"❌ split 命令需要 --input 参数", file=sys.stderr)
                print(f"   用法: {cmd_hint}", file=sys.stderr)
                sys.exit(1)
            split_input_to_chapters(
                input_path=args.input,
                output_dir=args.input_dir,
                input_format=args.input_format,
                target_chars=args.chapter_size,
                min_chars=args.min_chapter_size,
            )
    except KeyboardInterrupt:
        print("⛔ 已确认中止（Ctrl+C x2）。进度已保存；重新运行命令会从数据库断点续跑。",
              file=sys.stderr, flush=True)
        try:
            _task_logger.event("run_interrupted", stage="RUN", status="interrupted")
            _task_logger.finish(130, status="interrupted")
        except TaskLoggingError:
            pass
        sys.exit(130)
    except SystemExit as _task_exit:
        _task_code = getattr(_task_exit, "code", 1)
        if _task_code is None:
            _task_code = 0
        if not isinstance(_task_code, int):
            _task_code = 1
        try:
            if _task_code != 0:
                _task_logger.event(
                    "command_failed", stage="RUN", status="failed",
                    exit_code=_task_code,
                    message="command exited non-zero",
                )
            _task_logger.finish(_task_code)
        except TaskLoggingError:
            pass
        raise
    except Exception as e:
        # P4-9: 顶层异常统一退出（保留 KeyboardInterrupt/SystemExit 上抛）
        print(f"❌ 命令 {args.command} 异常终止: {e}", file=sys.stderr)
        import traceback as _tb_p49
        _tb_p49.print_exc()
        try:
            _task_logger.event(
                "command_failed", stage="RUN", status="failed",
                error_class=type(e).__name__, message=str(e),
            )
            _task_logger.finish(1, status="failed")
        except TaskLoggingError:
            pass
        sys.exit(1)
    else:
        try:
            _task_logger.finish(0)
        except TaskLoggingError:
            pass


# ────────────────────────────────────────────────────────────────────


# Section 12 — 统一入口点 (split_input_to_chapters)
# ─────────────────────────────────────────────────────────────────────────────

# 输入格式检测表
_INPUT_FORMAT_TABLE = {
    '.epub': 'epub',
    '.txt': 'txt',
    '.md': 'md',
    '.markdown': 'md',
}


def detect_input_format(file_path: str) -> str:
    """根据文件扩展名自动检测输入格式"""
    suffix = Path(file_path).suffix.lower()
    fmt = _INPUT_FORMAT_TABLE.get(suffix)
    if fmt is None:
        raise ValueError(
            f"无法识别输入格式: {file_path}\n"
            f"支持的扩展名: {list(_INPUT_FORMAT_TABLE.keys())}"
        )
    return fmt


def split_input_to_chapters(
    input_path: str,
    output_dir: str,
    input_format: str = 'auto',
    target_chars: int = 5000,
    min_chars: int = 1000,
) -> list:
    """
    统一入口：根据输入格式自动选择 Splitter 并切分章节。

    参数:
      input_path: 输入文件路径（EPUB / TXT / MD）
      output_dir: 输出目录（将生成 chXX.md）
      input_format: 'auto' | 'epub' | 'txt' | 'md'
      target_chars: TXT/MD 均分模式下的目标章节字数
      min_chars: 短于此字数的章节会被合并

    返回:
      生成的章节文件路径列表（仅 stdout，方便 mapfile 捕获）

    仅输出文件路径到 stdout，日志走 stderr。
    """
    if input_format == 'auto':
        input_format = detect_input_format(input_path)

    # P4-10: 依赖预检 — EPUB 格式在构造 splitter 前检查依赖，避免 RuntimeError 滞后
    if input_format == 'epub':
        if not _EPUB_DEPS_OK:
            raise RuntimeError(
                f"❌ 缺少 EPUB 解析依赖: {_EPUB_DEPS_ERROR}\n"
                "   请先安装: pip install ebooklib beautifulsoup4 lxml"
            )
        splitter = EpubSplitter(input_path, output_dir)
    elif input_format == 'txt':
        splitter = TextSplitter(input_path, output_dir, target_chars, min_chars)
    elif input_format == 'md':
        splitter = MdSplitter(input_path, output_dir, target_chars, min_chars)
    else:
        raise ValueError(f"不支持的格式: {input_format}")

    splitter.chapters = splitter.split()
    _audit = getattr(splitter, "audit_result", None) or {}
    if _audit and _audit.get("ok") is False:
        _splitter_log_err(
            f"转换完整性审计未通过（覆盖率 {_audit.get('overall', 0.0) * 100:.3f}%），"
            f"已中止；报告: {_audit.get('report_path')}")
        sys.exit(1)
    generated = splitter.write_files()

    # 汇总信息（stderr）
    _splitter_log_ok(f"完成！共生成 {len(generated)} 个章节文件")
    _splitter_log_info("下一步：")
    for f in generated[:5]:
        ch_id = Path(f).stem
        # P0-3: 根据 __package__ 自适应提示命令（单文件部署 vs python -m）
        if __package__:
            _cmd_prefix = f"python -m {__package__}"
        else:
            _cmd_prefix = "python translator_agent.py"
        _splitter_log_info(f"     {_cmd_prefix} init --chapter {ch_id} --force")
        _splitter_log_info(f"     {_cmd_prefix} pipeline --chapter {ch_id}")
    if len(generated) > 5:
        _splitter_log_info(f"     ... (共 {len(generated)} 章)")

    # 章节路径输出到 stdout（供 mapfile / shell 脚本捕获）
    for f in generated:
        print(f)
    return generated


if __name__ == "__main__":
    main()
