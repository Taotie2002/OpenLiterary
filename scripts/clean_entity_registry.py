#!/usr/bin/env python3
"""Round 105 P3: entity_registry 数据清理（可重跑）。

ch32 倒查后，注册表被「一字之差」误发现污染：
  - variant_names 混入普通词（东西/卡特/警察/伦敦金/材敦…）→ consistency 3c 改坏正文。
  - 同人多规范名冲突（George Lomax → 3 个 canonical）→ missing 硬失败死循环。

本脚本做两件事（默认 dry-run，加 --apply 才写库）：
  1. 清理 variant_names 中的普通词（复用 src.utils.entity_stopwords 黑名单）。
  2. 归一同人多规范名（unify_entity_registry：选 winner，其余降级 pending_review=1）。

用法：
  python3 scripts/clean_entity_registry.py            # dry-run，仅打印计划
  python3 scripts/clean_entity_registry.py --apply    # 实际写库
  python3 scripts/clean_entity_registry.py --no-unify # 只清普通词变体，不做归一
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.utils.entity_stopwords import (  # noqa: E402
    differing_char,
    is_likely_common_word,
)
from src.core.decision_engine import unify_entity_registry  # noqa: E402


# 确认的合法变体白名单（canonical, variant）。清理时只保留这些 + 种子变体，
# 其余「一字之差」候选一律视为污染移除。
# 依据：ch06/ch32 倒查，逐条人工确认。音译真异体 / 同人不同译名才入白名单。
CONFIRMED_VARIANTS = {
    ("奥布洛维奇家族", "奥博洛维奇家族"),  # 音译真异体（博/布）
    ("保皇党", "保皇派"),                  # 同一派系同义词
    ("巴特尔警督", "巴特尔警监"),           # 同一人物警衔不一致
}

# 裸警衔词：泛指警衔，盲替换会改坏正文（其他警官也可能用这些词），绝不作变体。
BARE_RANK_WORDS = frozenset({"警监", "警司", "警督"})


def _keep_set() -> set:
    """白名单 = 确认变体 + per-work 种子变体。

    P1-4: 种子已外移 — SEED_ENTITY_VARIANTS（模块常量）现为空；作品特定异体
    以 config.yaml `entity_variants` 段为准，为空时回退作品资源文件
    input/entity_variants.yaml（均可选；缺失 = 无种子）。
    """
    from src.core.decision_engine import SEED_ENTITY_VARIANTS, load_entity_variant_seeds
    configured = load_entity_variant_seeds()
    seeds = {**SEED_ENTITY_VARIANTS, **configured}
    keep = set(CONFIRMED_VARIANTS)
    for canonical, variants in seeds.items():
        for v in (variants or []):
            keep.add((canonical, v))
    return keep


def clean_common_word_variants(conn, dry_run: bool = True) -> list:
    """移除 variant_names 中的污染变体。返回 [(canonical, removed_variants), ...]

    Round 105 P3 升级（白名单策略）：
      旧版用「黑名单 + 高频差异字」护栏，双向都漏——既放过 帝国/法庭/英镑/女佣/
      托国/非私 等垃圾（差异字不高频），又可能误杀真音译变体。
      现改为白名单：只保留 CONFIRMED_VARIANTS + 种子变体，其余「一字之差」候选
      一律视为污染移除；裸警衔词（警监/警司/警督）因盲替换风险也移除。
      非一字之差的变体（多字差异的合法异体）保留。
    """
    keep = _keep_set()
    cursor = conn.cursor()
    cursor.execute("SELECT id, canonical_name, variant_names FROM entity_registry")
    rows = cursor.fetchall()
    changes = []
    for rid, canonical, vns_json in rows:
        if not vns_json:
            continue
        try:
            vns = json.loads(vns_json)
        except (ValueError, TypeError):
            continue
        if not isinstance(vns, list):
            continue
        kept, removed = [], []
        for v in vns:
            if not isinstance(v, str) or not v:
                continue
            if (canonical, v) in keep:
                kept.append(v)
                continue
            if is_likely_common_word(v):
                removed.append(v)
                continue
            if v in BARE_RANK_WORDS:
                removed.append(v)
                continue
            if differing_char(canonical, v) is not None:
                removed.append(v)
                continue
            kept.append(v)
        if removed:
            changes.append((canonical, removed))
            if not dry_run:
                cursor.execute(
                    "UPDATE entity_registry SET variant_names = ? WHERE id = ?",
                    (json.dumps(kept, ensure_ascii=False), rid),
                )
    if not dry_run:
        conn.commit()
    return changes


def main() -> int:
    ap = argparse.ArgumentParser(description="entity_registry 数据清理")
    ap.add_argument("--apply", action="store_true", help="实际写库（默认 dry-run）")
    ap.add_argument("--no-unify", action="store_true", help="跳过同人多规范名归一")
    ap.add_argument("--db", default=str(_ROOT / "db" / "decision_db.sqlite"))
    args = ap.parse_args()

    dry_run = not args.apply
    mode = "DRY-RUN（加 --apply 写库）" if dry_run else "APPLY（写库）"
    print(f"🧹 entity_registry 清理 — {mode}")
    print(f"   db: {args.db}")

    conn = sqlite3.connect(args.db)
    try:
        # 1. 清理普通词变体
        print("\n① 清理 variant_names 中的普通词…")
        changes = clean_common_word_variants(conn, dry_run=dry_run)
        if not changes:
            print("   无普通词变体需清理")
        for canonical, removed in changes:
            print(f"   - {canonical}: 移除 {removed}")
        print(f"   共 {len(changes)} 个实体受影响")

        # 2. 同人多规范名归一
        if not args.no_unify:
            print("\n② 归一同人多规范名…")
            plan = unify_entity_registry(conn, dry_run=dry_run)
            if not plan:
                print("   无冲突需归一")
            for cluster, info in plan.items():
                print(f"   - 簇 [{cluster}]")
                print(f"       winner: {info['winner']}")
                print(f"       降级: {info['demoted']}")
            print(f"   共 {len(plan)} 个簇")
    finally:
        conn.close()

    if dry_run:
        print("\n⏸️  DRY-RUN 完成，未写库。确认无误后加 --apply 执行。")
    else:
        print("\n✅ 清理完成，已写库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
