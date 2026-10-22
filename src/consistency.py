"""
SagaTrans 译后命名一致性校正（consistency 子命令）

源：原整合脚本 §11.5，拆分到独立模块。

职责：
  1. 扫描 output/ch*/chunk*_final.json 中的所有命名
  2. 与 Decision DB 的 entity_registry 做交叉比对
  3. 输出一致性差异报告（output/consistency_diff.md）
  4. 支持 --apply 模式：自动修正不一致项
"""
from __future__ import annotations

import json
import os
import sys
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from src.utils.llm_adapter import robust_json_loads
from typing import Any, Dict, List, Optional, Tuple


# Section 11.5 — 译后命名一致性校正
# ─────────────────────────────────────────────────────────────────────────────

def _load_glossary(db_path: str) -> dict:
    db = sqlite3.connect(db_path)
    entities = {}
    try:
        try:
            db.execute("SELECT variant_names FROM entity_registry LIMIT 1")
        except Exception:
            db.execute("ALTER TABLE entity_registry ADD COLUMN variant_names TEXT")
            db.commit()
        try:
            seed_entity_variants(db)
        except Exception:
            pass
        cursor = db.execute(
            "SELECT canonical_name, source_names, entity_type, confidence, variant_names "
            "FROM entity_registry WHERE confidence = 'high' AND pending_review = 0"
        )
        for row in cursor:
            entities[row[0]] = {
                "source_names": robust_json_loads(row[1], expected_type=list) if row[1] else [],
                "type": row[2],
                "confidence": row[3],
                "variant_names": robust_json_loads(row[4], expected_type=list) if row[4] else [],
            }
    except Exception as e:
        print(f"  ⚠️ 读取 entity_registry 失败: {e}", file=sys.stderr)
    db.close()
    return entities


def _load_decisions(db_path: str) -> dict:
    """从 decision_db 加载术语/典故决策"""
    db = sqlite3.connect(db_path)
    decisions = {}
    try:
        cursor = db.execute(
            "SELECT source_key, translation, level FROM decision_db WHERE level IN (1, 2)"
        )
        for row in cursor:
            decisions[row[0]] = {"translation": row[1], "level": row[2]}
    except Exception as e:
        print(f"  ⚠️ 读取 decision_db 失败: {e}", file=sys.stderr)
    db.close()
    return decisions


def _scan_finals(output_dir: str, flat: bool = False) -> list:
    """扫描 final.json

    Args:
        output_dir: 根目录（默认嵌套模式：output/<chapter>/<final.json>）
        flat: True 时把 output_dir 当作单章目录（output_dir 下直接是 final.json）

    Returns:
        [{chapter, file, text, path}, ...]
    """
    finals = []
    root = Path(output_dir)
    if not root.exists():
        return finals

    if flat:
        # Round 7: 章节级 scope（pipeline 末尾自动校正用）—— 直接把 output_dir 视为单章目录
        chapter_label = root.name
        for f in sorted(root.iterdir()):
            if not f.is_file() or not f.name.endswith("_final.json"):
                continue
            data = robust_json_loads(f.read_text(encoding="utf-8"), expected_type=dict)
            if not data:
                print(f"  ⚠️ 跳过 {f}: JSON 解析失败", file=sys.stderr)
                continue
            text = data.get("text", "") if isinstance(data, dict) else str(data)
            finals.append({
                "chapter": chapter_label,
                "file": f.name,
                "text": text,
                "path": str(f),
            })
        return finals

    # 原嵌套模式：output/<chapter>/*_final.json
    for ch_dir in sorted(root.iterdir()):
        if not ch_dir.is_dir():
            continue
        for f in sorted(ch_dir.iterdir()):
            if not f.name.endswith("_final.json"):
                continue
            # P2-1: 统一走 robust_json_loads 抗损坏 final.json
            data = robust_json_loads(f.read_text(encoding="utf-8"), expected_type=dict)
            if not data:
                print(f"  ⚠️ 跳过 {f}: JSON 解析失败", file=sys.stderr)
                continue
            text = data.get("text", "") if isinstance(data, dict) else str(data)
            finals.append({
                "chapter": ch_dir.name,
                "file": f.name,
                "text": text,
                "path": str(f),
            })
    return finals


def _find_slash_aliases(canonical: str) -> list:
    """'省长/地方市长' → ['省长', '地方市长']"""
    parts = [p.strip() for p in canonical.split("/")]
    return parts if len(parts) > 1 else []


# 中文异体漂移种子（可扩展）：canonical -> 易混用异体写法
# 通过 seed_entity_variants() 注入 entity_registry.variant_names。


# Round 119 (T6): 尾注区标记（段落首行上标 ¹²³… 或 "注：" 前缀），整段注释豁免
_NOTE_MARKER_RE = re.compile(r"^\s*(?:[¹²³⁴⁵⁶⁷⁸⁹]+|注\s*[:：])")


def _in_quote_or_note(text: str, pos: int) -> bool:
    """判断 pos 是否位于引号内或译注/脚注区块内（英文残留误报豁免）。"""
    q = 0
    for ch in text[:pos]:
        if ch in '“”"\'‘’':
            q += 1
    if q % 2 == 1:
        return True
    pre = text[max(0, pos - 200):pos]
    if "译注" in pre or "脚注" in pre or "此处保留" in pre:
        return True
    # 脚注定义行（以 [^ 起头）内的英文源词附注明豁免
    line_start = text.rfind("\n", 0, pos)
    line_end = text.find("\n", pos)
    line = text[line_start + 1: line_end if line_end != -1 else None]
    if line.lstrip().startswith("[^"):
        return True
    # Round 119 (T6): 尾注区豁免 — 段落首行为上标注（¹²³…）/"注："前缀时整段豁免
    # （长注释 200 字符 pre 窗口够不到段首"脚注"字样，D8 ch08 误报即此根因）
    para_start = line_start + 1
    while para_start > 0:
        prev_nl = para_start - 1
        prev_start = text.rfind("\n", 0, prev_nl) + 1
        if not text[prev_start:prev_nl].strip():
            break
        para_start = prev_start
    first_nl = text.find("\n", para_start)
    first_line = text[para_start: first_nl if first_nl != -1 else len(text)]
    if _NOTE_MARKER_RE.match(first_line):
        return True
    # Round 119 (T6): "（原文 …）" 括注内的英文源词附注豁免
    lp = text.rfind("（", 0, pos)
    if lp != -1:
        rp = text.find("）", lp + 1)
        if lp < pos < rp and text[lp + 1:rp].lstrip().startswith("原文"):
            return True
    return False


def _is_common_word_variant(variant: str, canonical: str = None) -> bool:
    """Round 105: 判定 variant 是否为普通词/假异体（不应作为人名异体替换）。

    复用 src.utils.entity_stopwords 黑名单 + 一字之差高频差异字护栏。
    导入失败（单文件部署）时退化为仅拦截已知 ch32 污染项，保证安全。

    ch06 倒查：variant_names 被「一字之差」误发现污染（法国→外国/美国/帝国/法庭，
    绅士乔→绅士一/绅士可，楼梯→楼上/楼下/楼去）。这些假异体的共同特征是
    变体侧差异字为高频字（一/外/去/上/下…），而真名异体的差异字偏冷僻。
    故在黑名单之外，追加「一字之差 + 变体侧差异字 ∈ 高频字集」护栏。
    """
    try:
        from src.utils.entity_stopwords import is_likely_common_word
        return is_likely_common_word(variant)
    except Exception:
        # 兜底黑名单（与 entity_stopwords.COMMON_WORDS 的 ch32 污染项一致）
        _FALLBACK = {
            "东西", "卡特", "警察", "警督", "警已", "警方", "警惕", "警司", "警官",
            "材敦", "普敦", "伦次",
            "伦敦金", "伦敦来", "伦敦最", "伦敦高", "伦敦时", "伦敦警", "伦敦可",
        }
        return variant in _FALLBACK


def _detect_issues(finals: list, entities: dict, decisions: dict) -> list:
    """
    检测三类问题：
    3a. 别名混用 —— 同一实体在同章节内混用多个译名
    3b. 英源残留 —— 英文名意外出现在中文译文中（脚注原文附注除外）
    """
    issues = []

    for fin in finals:
        text = fin["text"]

        # 3a. 实体名一致性（仅检测含 / 的多译名实体）
        for canonical, info in entities.items():
            aliases = _find_slash_aliases(canonical)
            if not aliases:
                continue  # 无 slash 别名，跳过
            all_names = [canonical] + aliases
            primary = canonical.split("/")[0].strip()  # 取 / 前作为主名
            counts = {n: text.count(n) for n in all_names}

            if counts.get(primary, 0) > 0:
                for n in aliases:
                    if counts.get(n, 0) > 0:
                        has_cjk_alias = bool(re.search(r'[\u4e00-\u9fff]', n))
                        pat_alias = re.escape(n) if has_cjk_alias else r'\b' + re.escape(n) + r'\b'
                        for m in re.finditer(pat_alias, text):
                            ctx = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
                            issues.append({
                                "chapter": fin["chapter"],
                                "file": fin["file"],
                                "path": fin["path"],
                                "position": m.start(),
                                "variant": n,
                                "canonical": primary,
                                "context": ctx,
                                "issue_type": "alias_mixed",
                            })

        # 3b. 英源词残留检测
        for canonical, info in entities.items():
            primary = canonical.split("/")[0].strip()
            for sn in info.get("source_names", []):
                # 先用 in 快速预检（无边界，仅做性能优化），再用 regex 精确匹配
                if sn.lower() in text.lower():
                    # CJK 源名用子串匹配（中文无词边界），非 CJK 用 \b 边界
                    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', sn))
                    pat = re.escape(sn) if has_cjk else r'(?<!\w)' + re.escape(sn) + r'(?!\w)'
                    for m in re.finditer(pat, text, re.IGNORECASE):
                        before = text[max(0, m.start() - 3): m.start()]
                        if "（" in before or "(" in before:
                            continue  # 脚注中的原文附注不算残留
                        if _in_quote_or_note(text, m.start()):
                            continue  # 引号内引文 / 译注中的英文源词不算残留
                        ctx = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
                        issues.append({
                            "chapter": fin["chapter"],
                            "file": fin["file"],
                            "path": fin["path"],
                            "position": m.start(),
                            "variant": sn,
                            "canonical": primary,
                            "context": ctx,
                            "issue_type": "english_residue",
                        })

        # 3c. 中文异体漂移检测（数据源：entity_registry.variant_names）
        # Round 105: 普通词护栏。variant_names 历史上被「一字之差」误发现污染
        # （东西/卡特/警察/伦敦金 等普通词被登记为人名异体）。盲目替换会改坏正文
        # （某样东西→某样黛西 / 卡特姆勋爵→卡洛姆勋爵）。命中普通词黑名单的
        # variant 一律跳过（不报告、不替换）。
        for canonical, info in entities.items():
            variants = info.get("variant_names", [])
            if not variants:
                continue
            for v in variants:
                if v not in text:
                    continue
                if _is_common_word_variant(v):
                    continue
                for m in re.finditer(re.escape(v), text):
                    if _in_quote_or_note(text, m.start()):
                        continue
                    ctx = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
                    issues.append({
                        "chapter": fin["chapter"],
                        "file": fin["file"],
                        "path": fin["path"],
                        "position": m.start(),
                        "variant": v,
                        "canonical": canonical,
                        "context": ctx,
                        "issue_type": "cjk_variant",
                    })

    return issues


def _generate_report(finals: list, issues: list, dry_run: bool) -> str:
    """生成差异报告 Markdown"""
    lines = []
    now = datetime.now().isoformat(timespec="seconds")

    lines.append("# 译后术语一致性校正报告")
    lines.append("")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 扫描章节: {len(finals)} 个 chunk")
    lines.append(f"- 发现不一致: {len(issues)} 处")
    lines.append(f"- 模式: {'DRY RUN (仅报告)' if dry_run else '已自动修正'}")
    lines.append("")

    if issues:
        by_type: dict = {}
        for iss in issues:
            by_type.setdefault(iss["issue_type"], []).append(iss)

        type_labels = {
            "alias_mixed": "同实体混用多个别名",
            "english_residue": "英文源词残留于中文译文中",
            "cjk_variant": "中文译名异体漂移",
        }

        for t, items in by_type.items():
            label = type_labels.get(t, t)
            lines.append(f"## {label} ({len(items)} 处)")
            lines.append("")
            for item in items:
                lines.append(f"### [{item['chapter']}/{item['file']}] @ {item['position']}")
                lines.append(f"- **变体**: `{item['variant']}` → **规范**: `{item['canonical']}`")
                lines.append(f"- 上下文: ...{item['context']}...")
                lines.append("")
    else:
        lines.append("未发现任何名称不一致。")
        lines.append("")

    lines.append("---")
    lines.append("*由 consistency 子命令自动生成*")
    lines.append("")

    return "\n".join(lines)


def _apply_issues(finals: list, issues: list) -> int:
    """应用修正到 final.json，返回实际修改的文件数"""
    by_file: dict = {}
    for iss in issues:
        if iss["issue_type"] in ("alias_mixed", "english_residue", "cjk_variant"):
            by_file.setdefault(iss["path"], []).append(iss)

    patched_files = 0
    for filepath, reps in by_file.items():
        p = Path(filepath)
        # P2-1: 统一走 robust_json_loads 抗损坏 final.json
        data = robust_json_loads(p.read_text(encoding="utf-8"), expected_type=dict)
        if not data:
            continue
        text = data.get("text", "") if isinstance(data, dict) else str(data)

        reps_sorted = sorted(reps, key=lambda x: x["position"], reverse=True)
        changes = 0
        for r in reps_sorted:
            if r["variant"] != r["canonical"]:
                old = text[r["position"]:r["position"] + len(r["variant"])]
                if old == r["variant"]:
                    text = text[:r["position"]] + r["canonical"] + text[r["position"] + len(r["variant"]):]
                    changes += 1

        if changes > 0:
            if isinstance(data, dict):
                data["text"] = text
                meta = data.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                meta["consistency_patched"] = True
                meta["consistency_patches"] = meta.get("consistency_patches", 0) + changes
                meta["consistency_patched_at"] = datetime.now().isoformat(timespec="seconds")
                data["metadata"] = meta
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            patched_files += 1
            print(f"  ✅ {p.name}: {changes} 处修正")

    return patched_files


def run_consistency_check(dry_run: bool = True, check_only: bool = False,
                          db_path: str = "", output_dir: str = "") -> int:
    """
    译后一致性校正入口。
    返回发现的 issues 数量。
    """
    _root = Path(__file__).resolve().parent.parent
    if not db_path:
        db_path = str(_root / "db" / "decision_db.sqlite")
    if not output_dir:
        output_dir = str(_root / "output")

    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}", file=sys.stderr)
        print("   请先运行 init 初始化项目。", file=sys.stderr)
        return -1

    print("📖 加载术语对照表...")
    entities = _load_glossary(db_path)
    decisions = _load_decisions(db_path)
    print(f"    entity_registry: {len(entities)} 个 high-confidence 实体")
    print(f"    decision_db:     {len(decisions)} 条决策")

    print("\n🔍 扫描 final.json...")
    finals = _scan_finals(output_dir)
    if not finals:
        print("    ⚠️ 没有找到 final.json（output/ 目录为空？）")
        return 0
    print(f"    找到 {len(finals)} 个文件")

    print("\n🔎 检测名称不一致...")
    issues = _detect_issues(finals, entities, decisions)
    print(f"    发现 {len(issues)} 处")

    if check_only:
        return len(issues)

    print("\n📝 生成报告...")
    report = _generate_report(finals, issues, dry_run=dry_run)
    report_path = Path(output_dir) / "consistency_diff.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"    → {report_path}")

    if not dry_run and issues:
        print("\n✏️ 执行修正...")
        patched = _apply_issues(finals, issues)
        print(f"    修改了 {patched} 个文件")
    elif dry_run and issues:
        print("\n⏸️  DRY RUN 模式：未修改任何文件（--apply 以执行修正）")

    print("\n🏁 完成")
    return len(issues)


def run_consistency_check_for_chapter(
    chapter_id: str,
    dry_run: bool = False,
    db_path: str = "",
    output_root: str = "",
) -> int:
    """Round 7: 章节级 scope 的 consistency 校正（供 pipeline.run() 末尾调用）

    与 run_consistency_check 的区别：
      - scope 仅限 output/{chapter_id}/*_final.json（无嵌套扫描）
      - 报告写到 {chapter_dir}/consistency_diff.md（而非 output/consistency_diff.md）
      - 默认 dry_run=False（auto-trigger 时应直接 apply，CLI 子命令仍走 dry_run）

    Args:
        chapter_id: 章节 ID（如 'ch32'）
        dry_run: True 仅生成报告，False 直接修改 final.json
        db_path: decision_db.sqlite 路径（默认 _root/db/decision_db.sqlite）
        output_root: output 根目录（默认 _root/output）

    Returns:
        发现的 issues 数量（-1 表示失败）
    """
    _root = Path(__file__).resolve().parent.parent
    if not db_path:
        db_path = str(_root / "db" / "decision_db.sqlite")
    if not output_root:
        output_root = str(_root / "output")

    chapter_dir = Path(output_root) / chapter_id
    if not Path(db_path).exists():
        print(f"❌ 数据库不存在: {db_path}", file=sys.stderr)
        return -1
    if not chapter_dir.exists():
        print(f"⚠️ 章节目录不存在: {chapter_dir}（跳过 consistency 校正）")
        return 0

    print(f"🔧 [Consistency] 章节级 scope: {chapter_id}")
    entities = _load_glossary(db_path)
    decisions = _load_decisions(db_path)
    finals = _scan_finals(str(chapter_dir), flat=True)
    if not finals:
        print(f"    ⚠️ {chapter_dir} 无 final.json（跳过）")
        return 0
    print(f"    扫描 {len(finals)} 个 final.json")

    issues = _detect_issues(finals, entities, decisions)
    print(f"    发现 {len(issues)} 处不一致")

    # 报告写到 chapter 目录内（避免污染全局报告）
    report = _generate_report(finals, issues, dry_run=dry_run)
    report_path = chapter_dir / "consistency_diff.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"    → 报告: {report_path}")

    if not dry_run and issues:
        patched = _apply_issues(finals, issues)
        print(f"    ✏️ 已修正 {patched} 个文件")
    elif dry_run and issues:
        print(f"    ⏸️ DRY RUN：未修改文件")

    return len(issues)


# ────────────────────────────────────────────────────────────────────
