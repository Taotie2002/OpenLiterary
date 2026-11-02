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

        # 3c. 中文异体漂移检测 —— Round 134 起从本函数移除：
        #     variant_names 现在表达「同人白名单」（含语境称谓 公爵/雷托公爵），
        #     一律替换会破坏文学多样性；改由 _apply_name_unification_pass 按
        #     allow/replace 分类处理（音译漂移统一 + 分隔符复原，含 LLM 裁决）。
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


# ════════════════════════════════════════════════════════════════════
# Round 134: 译后「名称统一」pass（同人归一 → 别名白名单 → 音译漂移全文统一）
#
# 与旧 3c（variant_names 盲替换）的区别：
#   - 旧 3c 把所有 variant_names 一律替换为 canonical —— 会把「公爵/雷托公爵」
#     这类语境称谓也替换掉，破坏文学多样性，且 variant_names 曾被普通词污染。
#   - 新 pass 先做同人归一（unify_entity_registry），再用 classify_entity_variant
#     分成两类：allow（称谓类，仅登记、正文不动）与 replace（音译漂移，全文统一）。
#   - 扫描发现的一字漂移（费德→菲德 / 皮特→彼得）必须经 LLM 一次性批量裁决才应用；
#     LLM 不可用/解析失败 → 保守不改（宁缺勿滥）。
# ════════════════════════════════════════════════════════════════════

def _load_name_unification_cfg() -> dict:
    try:
        from src.config import config as _cfg
        return dict((getattr(_cfg, "pipeline", {}) or {}).get("name_unification", {}) or {})
    except Exception:
        return {}


def _vet_name_drift_candidates(candidates: dict, cfg: dict, cache_path: Path) -> dict:
    """Round 134: 对「扫描发现」的漂移候选做 LLM 一次性批量裁决。

    Args:
        candidates: {cand: {"to", "count", "contexts", "entity", "sources"}}
        cfg: name_unification 配置段
        cache_path: 裁决缓存 JSON（按 "cand->to" 键持久化，避免重复调用）

    Returns:
        {cand: to} —— 仅包含被裁决为「同一专名的不同写法」的候选。
        任何失败（网络/解析/配额）→ 返回已裁决的部分或 {}（保守：不替换）。
    """
    if not candidates:
        return {}
    cache: dict = {}
    try:
        if cache_path.exists():
            cache = robust_json_loads(cache_path.read_text(encoding="utf-8"), expected_type=dict) or {}
    except Exception:
        cache = {}
    approved: dict = {}
    to_ask = []
    for cand, info in candidates.items():
        key = f"{cand}->{info.get('to')}"
        if key in cache:
            if cache[key]:
                approved[cand] = info.get("to")
            continue
        to_ask.append((cand, info))
    if not to_ask:
        return approved
    if not bool(cfg.get("llm_vetting", True)):
        return approved
    # Round 137: 按出现次数降序送审——多出现候选优先，防止高价值目标被截断挤出。
    to_ask.sort(key=lambda ci: -int((ci[1] or {}).get("count") or 0))

    lines = []
    for cand, info in to_ask[:80]:
        ctx = " / ".join(info.get("contexts") or [])[:160]
        srcs = "、".join(info.get("sources") or [])[:80]
        lines.append(
            f'- 候选「{cand}」→ 规范「{info.get("to")}」 | 实体: {info.get("entity", "")} '
            f'| 英文指称: {srcs} | 正文出现 {info.get("count", 0)} 次 | 上下文: {ctx}'
        )
    prompt = (
        "你在校验小说译文中的「专名书写漂移」（译名不统一的另一种写法）。下列每行给出"
        "一个候选写法与规范写法、所属实体、英文指称与候选写法在正文中的上下文。\n"
        "请判断：候选写法是否为**同一个专名的另一种书写/音译/错字形式**——专名包括"
        "人物、地名、组织、器物、概念术语等（如人物「玛丽安/玛莉安」、地名"
        "「卡斯提尔/卡斯蒂亚」、组织「银月商会/银月商团」这类整词改写）。\n"
        "判 false 的情形：① 普通中文词或句子片段（尤其含「的/了/会/是」等功能字、"
        "或名字后接谓语动词）；② 指称不同对象（组织 vs 其成员/分支、家族 vs 个人、"
        "器物 vs 使用者）；③ 两个不同的专名。\n"
        "只输出 JSON（不要解释、不要代码围栏）：\n"
        '{"results":[{"from":"<候选>","to":"<规范>","is_name_drift":true}]}\n\n'
        + "\n".join(lines)
    )
    try:
        from src.utils.llm_adapter import (
            get_llm_client, get_role_model_name, get_task_extra_body,
        )
        from src.config import config as _cfg
        role = _cfg.get_task_role("name_vetting")
        client = get_llm_client(role)
        model_name = get_role_model_name(role)
        extra = get_task_extra_body("name_vetting")
        raw = client.generate(
            prompt=prompt, model_name=model_name, max_tokens=4000,
            temperature=0.0, **extra,
        )
    except Exception as e:
        print(f"  ⚠️ [NameUnify] 漂移候选裁决失败（保守跳过）: {e}", file=sys.stderr)
        return approved

    try:
        data = robust_json_loads(raw, expected_type=dict)
    except Exception:
        data = None
    if not isinstance(data, dict):
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        data = robust_json_loads(m.group(0), expected_type=dict) if m else None
    if not isinstance(data, dict):
        print("  ⚠️ [NameUnify] 裁决结果解析失败（保守跳过）", file=sys.stderr)
        return approved
    asked = {c: info.get("to") for c, info in to_ask}
    for item in (data.get("results") or []):
        if not isinstance(item, dict):
            continue
        c, t = item.get("from"), item.get("to")
        if c not in asked:
            continue
        ok = bool(item.get("is_name_drift"))
        cache[f"{c}->{asked[c]}"] = ok
        if ok:
            approved[c] = asked[c]
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return approved


def _apply_name_unification_pass(
    finals: list,
    db_path: str,
    report_path: Path,
    cache_path: Path,
    scope_label: str = "",
    dry_run: bool = False,
) -> dict:
    """Round 134: 章节/全书级名称统一（登记表归一 + 别名整形 + 漂移 token 统一）。

    Args:
        finals: 待处理 final.json 列表（[{chapter,file,text,path}]）
        dry_run: True 时在数据库**临时拷贝**上试算并生成报告，真实 DB 与
                 final.json 均不写入；LLM 候选裁决跳过（候选写入报告，apply 时裁决）。

    Returns: {"files", "files_pending", "sep_restore", "forms", "tokens",
              "approved", "rejected", "pending", "clusters", "dry_run"}
    """
    cfg = _load_name_unification_cfg()
    if not bool(cfg.get("enabled", True)):
        return {"files": 0, "sep_restore": 0, "forms": 0, "tokens": 0,
                "approved": [], "rejected": [], "pending": [], "clusters": 0,
                "dry_run": bool(dry_run), "skipped": True}
    out = {"files": 0, "files_pending": 0, "sep_restore": 0, "forms": 0, "tokens": 0,
           "approved": [], "rejected": [], "pending": [], "clusters": 0,
           "dry_run": bool(dry_run)}
    tmp_path = None
    conn = None
    try:
        if dry_run:
            # 临时拷贝：归一/映射在“与真库一致的视图”上试算，真库零写入
            import tempfile
            _fd, _tmp_name = tempfile.mkstemp(prefix="sagatrans_nu_", suffix=".sqlite")
            os.close(_fd)
            tmp_path = Path(_tmp_name)
            conn = sqlite3.connect(str(tmp_path))
            try:
                _src_conn = sqlite3.connect(db_path)
                _src_conn.backup(conn)
                _src_conn.close()
            except Exception as e:
                print(f"  ⚠️ [NameUnify] dry-run 数据库拷贝失败: {e}", file=sys.stderr)
        else:
            conn = sqlite3.connect(db_path)
    except Exception as e:
        print(f"  ⚠️ [NameUnify] 数据库打开失败: {e}", file=sys.stderr)
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return out
    try:
        from src.core.decision_engine import (
            unify_entity_registry, build_name_unification_maps, apply_name_unification,
            _NAME_SEP_CHARS,
        )
        plan = unify_entity_registry(conn, dry_run=False)
        out["clusters"] = len(plan)
        corpus = "\n".join(f.get("text", "") for f in finals)
        maps = build_name_unification_maps(
            conn, corpus, preferred_forms=cfg.get("preferred_forms") or {},
        )
        cands = maps.get("token_candidates") or {}
        if cands and dry_run:
            out["pending"] = sorted(f"{c}→{i.get('to')}" for c, i in cands.items())
        elif cands:
            approved = _vet_name_drift_candidates(cands, cfg, cache_path)
            for c, t in approved.items():
                # Round 137: 长公共段扫描的候选携带原文形（可能含 ·/空格等分隔符，
                # 戈姆·贾巴尔）→ 走 forms 通道，apply 的 forms 阶段按分隔符不敏感
                # 正则替换（兼容同一名字的无分隔写法）；「一字之差」旧候选（纯 CJK
                # 同长形）不带 form，维持 tokens 字面替换，行为与 Round 136 一致。
                _info = cands.get(c) or {}
                if _info.get("form"):
                    maps.setdefault("forms", []).append(
                        [_info["form"], t, _info.get("entity") or t])
                else:
                    maps["tokens"][c] = t
            out["approved"] = sorted(f"{c}→{t}" for c, t in approved.items())
            out["rejected"] = sorted(f"{c}→{i.get('to')}" for c, i in cands.items()
                                     if c not in approved)
        for fin in finals:
            new_text, st = apply_name_unification(fin.get("text", ""), maps)
            if new_text and new_text != fin.get("text", ""):
                if dry_run:
                    out["files_pending"] += 1
                else:
                    try:
                        data = robust_json_loads(
                            Path(fin["path"]).read_text(encoding="utf-8"), expected_type=dict,
                        )
                        if data:
                            data["text"] = new_text
                            meta = data.get("metadata")
                            if not isinstance(meta, dict):
                                meta = {}
                            n_patch = st["sep_restore"] + st["forms"] + st["tokens"]
                            meta["name_unification_patched"] = True
                            meta["name_unification_patches"] = int(meta.get("name_unification_patches", 0)) + n_patch
                            meta["name_unification_at"] = datetime.now().isoformat(timespec="seconds")
                            data["metadata"] = meta
                            Path(fin["path"]).write_text(
                                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
                            )
                            fin["text"] = new_text
                            out["files"] += 1
                    except Exception as e:
                        print(f"  ⚠️ [NameUnify] 写入失败 {fin.get('file')}: {e}", file=sys.stderr)
            out["sep_restore"] += st["sep_restore"]
            out["forms"] += st["forms"]
            out["tokens"] += st["tokens"]
        # 报告
        try:
            lines = [f"# Round 134 名称统一报告 — {scope_label}", ""]
            mode = ("DRY RUN（仅报告；真实 DB 与 final.json 均未写入）" if dry_run
                    else "已应用（登记表 + 正文）")
            lines.append(f"- 模式: **{mode}**")
            total = out["sep_restore"] + out["forms"] + out["tokens"]
            lines.append(f"- 同人归一簇数: **{out['clusters']}**（登记表 winner/白名单）")
            lines.append(f"- 正文替换合计: **{total}** 处 "
                         f"(分隔符复原 {out['sep_restore']} / 别名整形 {out['forms']} / token 漂移 {out['tokens']})")
            if dry_run:
                lines.append(f"- 待写入文件: {out['files_pending']} / {len(finals)}")
            else:
                lines.append(f"- 受影响文件: {out['files']} / {len(finals)}")
            lines.append("")
            lines.append("## 统一映射（音译漂移 → 规范形）")
            for v, t, w in maps.get("forms", [])[:40]:
                lines.append(f"- `{v}` → `{t}`（{w}）")
            token_map = maps.get("tokens") or {}
            if token_map:
                lines.append("")
                lines.append("## token 级统一（含 LLM 裁决通过的候选）")
                for a in sorted(token_map, key=lambda x: -len(x))[:60]:
                    lines.append(f"- `{a}` → `{token_map[a]}`")
            sem = maps.get("semantic_pending") or []
            if sem:
                lines.append("")
                lines.append(f"## 同源语义别名（{len(sem)}；零字面交集/证据不足，未自动替换）")
                lines += [f"- `{x}`" for x in sem[:20]]
            if out["approved"]:
                lines.append("")
                lines.append(f"## LLM 裁决通过（{len(out['approved'])}）")
                lines += [f"- `{x}`" for x in out["approved"][:40]]
            if out["rejected"]:
                lines.append("")
                lines.append(f"## LLM 裁决否决（{len(out['rejected'])}，正文保持不变）")
                lines += [f"- `{x}`" for x in out["rejected"][:40]]
            if out["pending"]:
                lines.append("")
                lines.append(f"## LLM 裁决候选（{len(out['pending'])}，dry-run 未裁决；apply 时自动裁决）")
                lines += [f"- `{x}`" for x in out["pending"][:40]]
            allowed_only = [g for g in maps.get("guards", [])
                            if g and g not in set(maps.get("authorities", []))]
            lines.append("")
            lines.append("## 白名单（允许并存的语境称谓/简称，正文不动）")
            lines.append(f"- {len(allowed_only)} 项（variant_names policy=allow；详见 entity_registry）")
            if allowed_only:
                lines.append(f"- 示例: {'、'.join(sorted(set(allowed_only))[:12])}")
            lines.append("")
            report_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"  ⚠️ [NameUnify] 报告写入失败: {e}", file=sys.stderr)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return out


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

    # Round 134: 名称统一 pass（全书 scope；dry_run 仅出报告）
    try:
        nu = _apply_name_unification_pass(
            finals, db_path,
            report_path=Path(output_dir) / "name_unification_report.md",
            cache_path=Path(output_dir) / "_name_drift_vetting.json",
            scope_label="(global)",
            dry_run=dry_run,
        )
        if not nu.get("skipped"):
            n_repl = nu["sep_restore"] + nu["forms"] + nu["tokens"]
            print(f"\n🔗 [NameUnify] 同人归一 {nu['clusters']} 簇，正文替换 {n_repl} 处"
                  f"（文件 {nu['files']}）→ name_unification_report.md")
    except Exception as e:
        print(f"⚠️ [NameUnify] 名称统一异常（不影响主流程）: {e}")

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

    # Round 134: 名称统一 pass —— 应用范围为 output/** 全文（音译漂移须跨章统一），
    # 由 name_unification.enabled 独立门控（不受 legacy consistency_auto_fix 影响，
    # 因其替换全部经过 winner 白名单 + LLM 裁决护栏）。
    try:
        finals_all = _scan_finals(output_root) or finals
        nu = _apply_name_unification_pass(
            finals_all, db_path,
            report_path=Path(output_root) / "name_unification_report.md",
            cache_path=Path(output_root) / "_name_drift_vetting.json",
            scope_label=f"{chapter_id} → 全文 output/**",
            dry_run=False,
        )
        if not nu.get("skipped"):
            n_repl = nu["sep_restore"] + nu["forms"] + nu["tokens"]
            print(f"    🔗 [NameUnify] 同人归一 {nu['clusters']} 簇，正文替换 {n_repl} 处"
                  f"（文件 {nu['files']}）→ name_unification_report.md")
    except Exception as e:
        print(f"    ⚠️ [NameUnify] 名称统一异常（不影响主流程）: {e}")

    return len(issues)


# ────────────────────────────────────────────────────────────────────
