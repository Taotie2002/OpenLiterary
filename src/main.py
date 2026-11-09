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
from src.utils.chunker import SmartChunker
from src.core.scheduler import TaskScheduler
from src.pipeline import TranslationPipeline
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

_MUTATING_COMMANDS = {"pipeline", "init", "split", "style-profile", "story-graph",
                      "consolidate", "finalize", "notes-normalize", "requeue"}


def _command_needs_lock(command: str, args: object) -> bool:
    """Return True for commands that can write DB/profile/graph artifacts."""
    if command in _MUTATING_COMMANDS:
        return True
    if command == "consistency" and getattr(args, "apply", False):
        return True
    if command == "setup-payoff-audit" and getattr(args, "apply", False):
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
    global _INSTANCE_LOCK_FD
    if os.environ.get("SAGATRANS_LOCK_HELD") == "1":
        return True
    import fcntl
    import time as _time
    path = Path(lock_path) if lock_path else Path.cwd() / "db" / ".sagatrans.lock"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        print(f"⚠️ [Lock] 无法创建锁文件 {path}: {e}（继续执行）", file=sys.stderr)
        return True
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
            print(f"⚠️ [Lock] 已有 SagaTrans 实例在运行（{holder}）——"
                  f"SAGATRANS_ALLOW_CONCURRENT=1 放行", file=sys.stderr)
            return True
        print(f"⛔ [Lock] 已有 SagaTrans 实例在运行：{holder}", file=sys.stderr)
        print(f"   锁文件: {path}；确认无残留进程后可设 SAGATRANS_ALLOW_CONCURRENT=1 强制放行（风险自担）",
              file=sys.stderr)
        return False
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (f"pid={os.getpid()} started={_time.strftime('%F %T')} "
                      f"cmd={' '.join(sys.argv[1:]) or 'pipeline'}\n").encode("utf-8"))
    except OSError:
        pass
    _INSTANCE_LOCK_FD = fd  # 进程存活期间保持打开；fd 随进程退出关闭 → 自动释放锁
    return True


def main():
    _install_sigint_guard()
    print_banner()

    import argparse
    parser = argparse.ArgumentParser(description="SagaTrans — AI 文学语义编译系统")
    parser.add_argument("command", nargs="?", default="pipeline",
                        choices=["pipeline", "init", "golden", "golden-gate", "memory", "debug", "split", "consistency", "consolidate", "finalize", "lint", "gate", "style-profile", "story-graph", "story-graph-query", "setup-payoff-audit", "drift-audit", "notes-normalize", "requeue"],
                        help="执行命令: pipeline (默认), init, golden, golden-gate, memory, debug, split, consistency, consolidate, finalize, lint, gate, style-profile, story-graph, story-graph-query, setup-payoff-audit, drift-audit, notes-normalize, requeue")
    parser.add_argument("--chapter", default="ch01", help="章节 ID (默认: ch01)")
    parser.add_argument("--chapters", default=None,
                        help="style-profile 命令：抽样章节，逗号分隔（默认均匀抽 5 章）")
    parser.add_argument("--deep", action="store_true",
                        help="style-profile 命令：全书分段深读 + 跨段合并 + L1/L2 验证（Round 131）")
    parser.add_argument("--deep-max-chars", type=int, default=60000,
                        help="style-profile --deep：单段原文字符上限（默认 60000）")
    parser.add_argument("--deep-overlap-chars", type=int, default=2000,
                        help="style-profile --deep：相邻分段重叠字符数（默认 2000）")
    parser.add_argument("--graph-path", default=None,
                        help="story-graph / story-graph-query 命令：图路径（默认读取 config.story_graph.path）")
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
    parser.add_argument("--force", action="store_true", help="强制重新初始化（覆盖已存在的 chunk，重置状态和重试计数）")
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
                        help="consistency / notes-normalize 命令：执行自动修正（notes-normalize 默认仅生成差异报告）")
    parser.add_argument("--window", type=int, default=None,
                        help="drift-audit 命令：滑动窗口大小（默认 config.pipeline.drift_audit.window=5）")
    parser.add_argument("--jev", action="store_true",
                        help="drift-audit 命令：附加 Jev 窗口配对检查（需要 Jev 可用）")
    parser.add_argument("--json-out", default=None,
                        help="drift-audit / golden --judgment-ab 命令：报告 JSON 输出路径")
    parser.add_argument("--ab", action="store_true",
                        help="golden/golden-gate 命令：运行 直译模型×thinking 4 组合 A/B 对比")
    parser.add_argument("--judgment-ab", action="store_true",
                        help="golden 命令：仅比较判断层注记注入 on/off（同章同摘录，不乘模型×effort 矩阵）")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="consolidate 命令：最大收敛轮数（默认 3）")

    args = parser.parse_args()

    # P0-4: 章节 ID 合法性校验（路径遍历防御）
    # Round 52 修复：re.fullmatch 替代 re.match（避免 $ 在 final \n 前的边界 bug — "ch01\n" 原可通过）
    if args.chapter and not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', args.chapter):
        print(f"❌ 非法 chapter_id: '{args.chapter}' (仅允许字母/数字/下划线/连字符，1-32 字符)", file=sys.stderr)
        sys.exit(1)

    # Round 130: 单实例守卫（审计①）——写共享状态的命令必须独占锁；
    # 只读命令（gate/lint/drift-audit 等）不拦截，便于运行中诊断。
    if _command_needs_lock(args.command, args):
        if not acquire_instance_lock():
            sys.exit(1)

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
            from src.style_profiler import generate_book_profile, summary_line
            _chapters = [c.strip() for c in args.chapters.split(",") if c.strip()] if args.chapters else None
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
        elif args.command == "story-graph-query":
            from src.story_graph import load_story_graph
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
            _graph = load_story_graph(_path)
            if not _graph:
                print(f"❌ story-graph-query: 无法读取图 artifact: {_path}",
                      file=sys.stderr)
                sys.exit(1)
            _in_dir = Path(args.input_dir) if args.input_dir else _root / "input"
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
            from src.story_graph import (
                build_story_graph,
                load_story_graph,
                save_story_graph,
            )
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
            _graph = load_story_graph(_graph_path)
            if not _graph:
                print(f"❌ setup-payoff-audit: 无法读取图 artifact {_graph_path}",
                      file=sys.stderr)
                sys.exit(1)
            _input_dir = Path(args.input_dir) if args.input_dir else _root / "input"
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
            _rq_res = _rq_sched.requeue_permanently_failed(args.chapter, force=args.force)
            _rq_n = len(_rq_res["requeued"])
            print(f"♻️ requeue: {args.chapter} 共 {_rq_n} 个 PF 块恢复 DIRTY")
            if _rq_res["requeued"]:
                print(f"    {', '.join(_rq_res['requeued'])}")
                print(f"    → 运行: python3 -m src.translator_agent pipeline --chapter {args.chapter}")
            if _rq_res["over_cap"]:
                print(f"    ⚠️ 复活预算用尽(pf_requeue_max): {', '.join(_rq_res['over_cap'])}（--force 可强制）")
            sys.exit(0 if _rq_n or not _rq_res["over_cap"] else 1)
        elif args.command == "gate":
            from src.pipeline import run_final_gate
            _gate_res = run_final_gate()
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
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        # P4-9: 顶层异常统一退出（保留 KeyboardInterrupt/SystemExit 上抛）
        print(f"❌ 命令 {args.command} 异常终止: {e}", file=sys.stderr)
        import traceback as _tb_p49
        _tb_p49.print_exc()
        sys.exit(1)


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
