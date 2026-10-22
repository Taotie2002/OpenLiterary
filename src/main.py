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
from src.test_golden_set import run_golden_ab, run_golden_test
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

def main():
    print_banner()

    import argparse
    parser = argparse.ArgumentParser(description="SagaTrans — AI 文学语义编译系统")
    parser.add_argument("command", nargs="?", default="pipeline",
                        choices=["pipeline", "init", "golden", "golden-gate", "memory", "debug", "split", "consistency", "consolidate", "finalize", "lint", "gate", "style-profile", "drift-audit"],
                        help="执行命令: pipeline (默认), init, golden, golden-gate, memory, debug, split, consistency, consolidate, finalize, lint, gate, style-profile, drift-audit")
    parser.add_argument("--chapter", default="ch01", help="章节 ID (默认: ch01)")
    parser.add_argument("--chapters", default=None,
                        help="style-profile 命令：抽样章节，逗号分隔（默认均匀抽 5 章）")
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
                        help="consistency 命令：执行自动修正（默认仅生成差异报告）")
    parser.add_argument("--window", type=int, default=None,
                        help="drift-audit 命令：滑动窗口大小（默认 config.pipeline.drift_audit.window=5）")
    parser.add_argument("--jev", action="store_true",
                        help="drift-audit 命令：附加 Jev 窗口配对检查（需要 Jev 可用）")
    parser.add_argument("--json-out", default=None,
                        help="drift-audit 命令：报告 JSON 输出路径")
    parser.add_argument("--ab", action="store_true",
                        help="golden/golden-gate 命令：运行 直译模型×thinking 4 组合 A/B 对比")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="consolidate 命令：最大收敛轮数（默认 3）")

    args = parser.parse_args()

    # P0-4: 章节 ID 合法性校验（路径遍历防御）
    # Round 52 修复：re.fullmatch 替代 re.match（避免 $ 在 final \n 前的边界 bug — "ch01\n" 原可通过）
    if args.chapter and not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', args.chapter):
        print(f"❌ 非法 chapter_id: '{args.chapter}' (仅允许字母/数字/下划线/连字符，1-32 字符)", file=sys.stderr)
        sys.exit(1)

    # P4-9: 顶层异常包装（除已规范的 consistency/golden-gate 外，统一捕获并退出）
    try:
        if args.command == "pipeline":
            pipeline = TranslationPipeline(chapter_id=args.chapter)
            if not pipeline.run():
                sys.exit(1)
        elif args.command == "init":
            if not init_project(chapter_id=args.chapter, force=args.force):
                sys.exit(1)
        elif args.command == "style-profile":
            from src.style_profiler import generate_book_profile, summary_line
            _chapters = [c.strip() for c in args.chapters.split(",") if c.strip()] if args.chapters else None
            res = generate_book_profile(Path.cwd(), chapters=_chapters, force=args.force)
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
            if args.ab:
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
        elif args.command == "gate":
            from src.pipeline import run_final_gate
            _gate_res = run_final_gate()
            sys.exit(0 if _gate_res.get("ok") else 1)
        elif args.command == "finalize":
            from src.book_lint import finalize_outputs
            finalize_outputs(apply=args.apply)
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
    except (KeyboardInterrupt, SystemExit):
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
