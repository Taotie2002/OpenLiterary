#!/usr/bin/env python3
"""Round 152: deterministic repair of remaining name/title contamination.

Every edit is anchored to an exact Chinese surface.  The rules were derived by
comparing final.json with the English chunk source and, for the glossary/name
appendix, with the source definitions themselves.  Unknown text is never
rewritten.  Dry-run is the default; ``--apply`` writes per-chunk backups.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


_ROOT = Path(__file__).resolve().parent.parent

# Exact, chunk-scoped replacements.  Each tuple is
# (old, new, rule, expected_count).  A missing expected occurrence is reported
# and never guessed around.
_CHUNK_RULES: Dict[str, List[Tuple[str, str, str, int]]] = {
    "ch01_chunk019": [
        ("你听说过盖乌斯之药[^4]吗？", "你听说过真言者之药[^4]吗？",
         "truthsayer_drug_body", 1),
        ("一位盖乌斯[^6]", "一位真言者[^6]", "truthsayer_role_body", 1),
        ("任何盖乌斯都看不见", "任何真言者都看不见",
         "truthsayer_role_body", 1),
        ("[^4]: 盖乌斯之药：", "[^4]: 真言者之药：",
         "truthsayer_drug_footnote", 1),
        ("[^5]: 真言恍惚：盖乌斯服用", "[^5]: 真言恍惚：真言者服用",
         "truthsayer_role_footnote", 1),
        ("[^6]: 盖乌斯：", "[^6]: 真言者：", "truthsayer_role_footnote", 1),
    ],
    "ch01_chunk071": [
        ("“海伦警告过您吗？”", "“圣母警告过您吗？”",
         "reverend_mother_not_helen", 1),
    ],
    "ch03_chunk068": [
        ("他看见母亲——盖乌斯·海伦·莫希阿姆的弟子——也读懂了",
         "他看见母亲也读懂了", "mother_is_jessica_not_mohiam", 1),
    ],
    "ch03_chunk079": [
        ("盖乌斯站在房间中央", "杰西卡站在房间中央",
         "jessica_scene_not_mohiam", 1),
        ("盖乌斯找到一件长袍", "杰西卡找到一件长袍",
         "jessica_scene_not_mohiam", 1),
        ("盖乌斯用腰带束好长袍", "杰西卡用腰带束好长袍",
         "jessica_scene_not_mohiam", 1),
    ],
    "ch03_chunk100": [
        ("保罗把左手放在护盾", "雷托把左手放在护盾",
         "leto_action_not_paul", 1),
    ],
    "ch03_chunk112": [
        ("他害怕杰西卡的审问，盖乌斯想。",
         "他害怕真言者的审问，杰西卡想。",
         "truthsayer_and_jessica_thought", 1),
    ],
    "ch03_chunk114": [
        ("“真言者？干完今晚这桩活，我可不愿去想还要面对那个杰西卡，”疤脸说。",
         "“我嘛，干完今晚这桩活，我可不愿去想还要面对那个真言者，”疤脸说。",
         "truthsayer_dialogue_source_alignment", 1),
        ("真言者听着他们的话，观察着，记录着。",
         "杰西卡听着他们的话，观察着，记录着。",
         "truthsayer_listener_is_jessica", 1),
    ],
    "ch03_chunk152": [
        ("他挨着母亲盖乌斯·海伦·莫希阿姆坐着",
         "他挨着母亲坐着", "mother_is_jessica_not_mohiam", 1),
    ],
    "ch03_chunk169": [
        ("它就像盖乌斯之药", "它就像真言者之药", "truthsayer_drug_body", 1),
    ],
    "ch04_chunk134": [
        ("他们或许会抛弃那女人，他的母亲——盖乌斯·海伦·莫希阿姆在姐妹会里教出来的那种女人——但会救下那个男孩。",
         "他们或许会抛弃那女人，也就是他的母亲，但会救下那个男孩。",
         "mother_is_jessica_not_mohiam", 1),
    ],
    "ch05_chunk057": [
        ("他朝杰西卡看了一眼。杰西卡夫人点了点头。",
         "他朝杰西卡看了一眼。她点了点头。",
         "single_jessica_reference", 1),
    ],
    "ch05_chunk067": [
        ("他认出了母亲——盖乌斯·海伦·莫希阿姆。",
         "他认出了母亲。", "mother_is_jessica_not_mohiam", 1),
    ],
    "ch05_chunk082": [
        ("海伦伯爵和芬林夫人", "芬林伯爵和芬林夫人",
         "fenring_not_helen", 1),
    ],
    "ch05_chunk120": [
        ("我想是在等海伦[^5]", "我想是在等圣母[^5]",
         "reverend_mother_not_helen", 1),
        ('[^5]: “海伦”：贝尼·杰瑟里特姐妹会中居于领导地位、常承担女先知职能的女性尊称。',
         '[^5]: “圣母”：贝尼·杰瑟里特姐妹会中居于领导地位、常承担女先知职能的女性尊称。',
         "reverend_mother_not_helen", 1),
    ],
    "ch05_chunk148": [
        ("这是幻觉，盖乌斯对自己说。", "这是幻觉，杰西卡对自己说。",
         "jessica_thought_not_mohiam", 1),
    ],
    "ch06_chunk000": [
        ("据我母亲——盖乌斯·海伦·莫希阿姆——说",
         "据我母亲说", "mother_is_jessica_not_mohiam", 1),
    ],
    "ch06_chunk015": [
        ("弗拉基米尔·哈克南男爵男爵", "弗拉基米尔·哈克南男爵",
         "duplicate_honorific", 1),
    ],
    "ch06_chunk021": [
        ("海伦伯爵", "芬林伯爵", "fenring_not_helen", 3),
    ],
    "ch06_chunk024": [
        ("男爵承认过，盖乌斯·海伦·莫希阿姆——那个贝尼·杰瑟里特姐妹会——曾经是他的武器。而那只能意味着一切的终结——甚至包括盖乌斯自己的儿子。",
         "男爵承认过，那个贝尼·杰瑟里特女巫就是他的武器。而那只能意味着一切的终结——甚至包括她自己的儿子。",
         "mother_is_jessica_not_mohiam", 1),
    ],
    "ch06_chunk026": [
        ("他的母亲杰西卡和契妮", "他的母亲和契妮",
         "mother_is_jessica_not_mohiam", 1),
    ],
    "ch06_chunk037": [
        ("我的母亲盖乌斯·海伦·莫希阿姆", "我的母亲",
         "mother_is_jessica_not_mohiam", 1),
    ],
    "ch06_chunk087": [
        ("雷托公爵公爵", "雷托公爵", "duplicate_honorific", 1),
    ],
    "ch06_chunk114": [
        ("他的母亲——盖乌斯·海伦·莫希阿姆——曾把它拿给他看",
         "他的母亲曾把它拿给他看", "mother_is_jessica_not_mohiam", 1),
    ],
    "ch07_chunk009": [
        ("他的母亲杰西卡", "他的母亲", "mother_is_jessica_not_mohiam", 1),
    ],
    "ch07_chunk012": [
        ("海伦盖乌斯·海伦·莫希阿姆独自坐在入口正对面",
         "圣母独自坐在入口正对面", "reverend_mother_not_helen", 1),
        ("**海伦**：贝尼·杰瑟里特姐妹会中高阶女性成员或女先知的敬称。",
         "**圣母**：贝尼·杰瑟里特姐妹会高阶女性成员的敬称；此处指杰西卡，她已接受生命之水成为圣母。",
         "reverend_mother_not_helen", 1),
    ],
    "ch07_chunk024": [
        ("却能像海伦一样看透生命之水", "却能像圣母一样看透生命之水",
         "reverend_mother_not_helen", 1),
        ('“海伦”是贝尼·杰瑟里特姐妹会高阶成员的尊称',
         '“圣母”是贝尼·杰瑟里特姐妹会高阶成员的尊称',
         "reverend_mother_not_helen", 1),
    ],
    "ch07_chunk046": [
        ("圣母盖乌斯·海伦·莫希阿姆以皇帝盖乌斯的身份出现",
         "圣母盖乌斯·海伦·莫希阿姆以皇帝的真言者身份出现",
         "emperor_truthsayer_not_gaius", 1),
    ],
    "ch07_chunk056": [
        ("贝尼·杰瑟里特姐妹会的现任圣母、盖乌斯，杰西卡夫人之母",
         "贝尼·杰瑟里特姐妹会的现任圣母，杰西卡夫人之母",
         "malformed_mohiam_footnote", 1),
    ],
    "ch07_chunk063": [
        ("“海伦[^4]，”", "“圣母[^4]，”", "reverend_mother_not_helen", 1),
        ("，”海伦说。", "，”圣母说。", "reverend_mother_not_helen", 1),
        ("[^4]: 海伦：对贝尼·杰瑟里特姐妹会高阶成员的敬称",
         "[^4]: 圣母：对贝尼·杰瑟里特姐妹会高阶成员的敬称",
         "reverend_mother_not_helen", 1),
    ],
    "ch07_chunk084": [
        ("贝尼·杰瑟里特姐妹会的盖乌斯也在发抖",
         "贝尼·杰瑟里特姐妹会的真言者也在发抖",
         "truthsayer_not_gaius", 1),
        ('“盖乌斯”指其成员中的女先知。', '“真言者”指其成员中的女先知。',
         "truthsayer_not_gaius", 1),
    ],
    "ch07_chunk089": [
        ("“母亲”：保罗对盖乌斯（盖乌斯·海伦·莫希阿姆）的称呼。盖乌斯是贝尼·杰瑟里特姐妹会成员",
         "“母亲”：保罗对杰西卡夫人的称呼。杰西卡是贝尼·杰瑟里特姐妹会成员",
         "mother_is_jessica_not_mohiam", 1),
    ],
    "ch07_chunk122": [
        ("或像盖乌斯·海伦·莫希阿姆那样的海伦",
         "或像盖乌斯·海伦·莫希阿姆那样的圣母",
         "reverend_mother_not_helen", 1),
    ],
    "ch07_chunk137": [
        ("潮湿之母（海伦盖乌斯·海伦·莫希阿姆[^5]）",
         "潮湿之母（圣母们[^5]）", "reverend_mother_not_helen", 1),
        ("[^5]: 盖乌斯·海伦·莫希阿姆：贝尼·杰瑟里特姐妹会的海伦。",
         "[^5]: 圣母（Reverend Mothers）：贝尼·杰瑟里特姐妹会的高阶女性成员。",
         "reverend_mother_not_helen", 1),
    ],
    "ch07_chunk146": [
        ("哈西米尔·海伦伯爵", "哈西米尔·芬林伯爵", "fenring_not_helen", 1),
        ("海伦应对导致", "芬林应对导致", "fenring_not_helen", 1),
        ("海伦是沙达姆四世", "芬林是沙达姆四世", "fenring_not_helen", 1),
        ("海伦伯爵担任过的帝国职任", "芬林伯爵担任过的帝国职任",
         "fenring_not_helen", 1),
    ],
    "ch07_chunk157": [
        ("贝尼·杰瑟里特姐妹会的一名海伦", "贝尼·杰瑟里特姐妹会的一名圣母",
         "reverend_mother_not_helen", 1),
    ],
    "ch07_chunk158": [
        ("海伦：最初是贝尼·杰瑟里特姐妹会", "圣母：最初是贝尼·杰瑟里特姐妹会",
         "reverend_mother_not_helen", 1),
        ("她使体内的一种“使人开悟的毒”发生转化",
         "一名圣母使体内的一种“使人开悟的毒”发生转化",
         "reverend_mother_not_helen", 1),
    ],
}

_ATREIDES_RE = re.compile(r"阿特雷迪斯|阿特雷德斯|阿特雷德|亚崔迪|亚崔德")


def _replace_exact(text: str, old: str, new: str, rule: str,
                   expected: int) -> Tuple[str, Dict[str, Any], bool]:
    count = text.count(old)
    if count != expected:
        return text, {
            "rule": rule,
            "from": old,
            "to": new,
            "expected": expected,
            "found": count,
            "count": 0,
        }, False
    return text.replace(old, new), {
        "rule": rule,
        "from": old,
        "to": new,
        "expected": expected,
        "found": count,
        "count": count,
    }, True


def repair_text(chunk_id: str, text: str) -> Tuple[str, List[Dict[str, Any]],
                                                     List[Dict[str, Any]]]:
    """Apply exact Round 152 rules.  Return text, changes, skipped expectations."""
    changes: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    out = text
    for old, new, rule, expected in _CHUNK_RULES.get(chunk_id, []):
        out, change, ok = _replace_exact(out, old, new, rule, expected)
        (changes if ok else skipped).append(change)

    unified = _ATREIDES_RE.sub("厄崔迪", out)
    count = len(_ATREIDES_RE.findall(out))
    if count:
        changes.append({
            "rule": "atreides_canonical_transliteration",
            "from": "阿特雷迪斯/阿特雷德斯/阿特雷德/亚崔迪/亚崔德",
            "to": "厄崔迪",
            "expected": count,
            "found": count,
            "count": count,
        })
    return unified, changes, skipped


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# Round 152 姓名/称谓残留修复",
        "",
        f"- 模式: {payload['mode']}",
        f"- 扫描: {payload['scanned']}",
        f"- 命中 chunk: {payload['matched_chunks']}",
        f"- 替换次数: {payload['replacements']}",
        f"- 已写入 chunk: {payload['applied_chunks']}",
        "",
    ]
    for row in payload["rows"]:
        lines.append(f"## `{row['chunk_id']}`")
        for change in row["changes"]:
            lines.append(f"- `{change['rule']}`: {change['count']} 次")
        for skip in row["skipped"]:
            lines.append(
                f"- 跳过 `{skip['rule']}`: expected={skip['expected']} "
                f"found={skip['found']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round 152 姓名/称谓残留确定性修复（默认 dry-run）")
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument("--chunks", default=None, help="逗号分隔 chunk_id")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = root / "output"
    json_out = Path(args.json_out) if args.json_out else (
        output_dir / "round152_name_residue_repair.json")
    report = Path(args.report) if args.report else (
        output_dir / "round152_name_residue_repair.md")
    now = _dt.datetime.now().astimezone()
    backup_root = output_dir / "repair_backups" / (
        "round152_name_" + now.strftime("%Y%m%d_%H%M%S"))

    if args.chunks:
        ids = [x.strip() for x in args.chunks.split(",") if x.strip()]
        paths = [output_dir / cid.split("_", 1)[0] / f"{cid}_final.json"
                 for cid in ids]
    else:
        paths = sorted(output_dir.glob("ch*/*_final.json"))

    rows: List[Dict[str, Any]] = []
    applied_chunks = 0
    replacements = 0
    skipped_total = 0
    for final_path in paths:
        cid = final_path.stem.rsplit("_final", 1)[0]
        data = _read_json(final_path)
        if not data:
            continue
        text = str(data.get("text") or "")
        if not text:
            continue
        repaired, changes, skipped = repair_text(cid, text)
        if not changes and not skipped:
            continue
        replacements += sum(int(x["count"]) for x in changes)
        skipped_total += len(skipped)
        row = {
            "chunk_id": cid,
            "changes": changes,
            "skipped": skipped,
            "before_sha256": _sha256(text),
            "after_sha256": _sha256(repaired),
            "applied": False,
        }
        rows.append(row)
        if args.apply and not skipped and repaired != text:
            chapter = cid.split("_", 1)[0]
            backup_path = backup_root / chapter / final_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_path, backup_path)
            data["text"] = repaired
            meta = data.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            meta["round152_name_residue_repair"] = {
                "schema_version": 1,
                "applied_at": now.isoformat(),
                "rules": changes,
                "backup": str(backup_path.relative_to(root)),
            }
            data["metadata"] = meta
            _atomic_write_json(final_path, data)
            row["applied"] = True
            applied_chunks += 1

    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "mode": "apply" if args.apply else "dry-run",
        "scanned": len(paths),
        "matched_chunks": len(rows),
        "replacements": replacements,
        "applied_chunks": applied_chunks,
        "skipped_rules": skipped_total,
        "rows": rows,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    _write_markdown(report, payload)
    print(json.dumps({
        "mode": payload["mode"],
        "scanned": payload["scanned"],
        "matched_chunks": payload["matched_chunks"],
        "replacements": payload["replacements"],
        "applied_chunks": payload["applied_chunks"],
        "skipped_rules": payload["skipped_rules"],
        "json": str(json_out),
        "report": str(report),
    }, ensure_ascii=False))
    return 0 if skipped_total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
