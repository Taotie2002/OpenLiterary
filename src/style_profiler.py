"""
StyleProfiler — 动态风格剖面生成器 (Round 117 P0)

背景（用户决策 + docs/audit_project_quality_2026-07-24.md P1-4）：
  config.yaml 曾把作品风格写死（如「1920s 英式轻喜剧」），prompt 里还残留
  俚语表/丁尼生引用/星云奖人设等作品特定先验。翻译其他作品时这些先验会污染
  直译与润色。正确做法：翻译前由本模块抽样原文，让 LLM 生成结构化「作品风格
  剖面」(book profile)，落盘 input/book_profile.json；TranslationPipeline 启动
  时加载为运行时 style_guide 覆盖层（src.config.set_runtime_style_guide）。

两种模式（用户指定思路）：
  1. known_work — LLM 认出这是知名已出版作品 → 按该作品公认风格设置，source=known_work；
  2. sampled    — 不知名作品 → 基于抽样章节原文归纳，source=sampled。
  LLM 判定需要给出 evidence_quotes（抽样原文中的英文短句）作为支撑。

剖面 JSON schema（schema_version=2；v1 字段全部保留）：
  genre / era / region / persona / register_default / footnote_policy /
  forbidden_modernisms / translation_hints / pun_watchlist / chapter_register_map /
  evidence_quotes / critic_thresholds（可选——按书覆盖 critic 评分阈值，0-10 数值） /
  work_guess（title/title_zh/author/year） / confidence / source / samples_used /
  narrative_arc / setup_payoff / voice_notes / generation

v2 注记统一 envelope（见 src/judgment_layer.py）：
  id / type(arc|setup_payoff|voice) / chapters / claim / evidence /
  confidence / status / source_model / created_at

LLM 失败时回退 heuristic_profile()：中性默认 + confidence=0 + source=fallback，
保证管线仍可运行且不注入任何作品特定先验。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.judgment_layer import (
    JUDGMENT_SCHEMA_VERSION,
    atomic_write_json,
    build_audit_payload,
    chapter_ids_between,
    dedupe_notes,
    enforce_note_limits,
    ensure_unique_note_ids,
    normalize_judgment_fields,
    normalize_notes,
    restrict_notes_to_candidate_quotes,
    replace_notes_in_profile,
    validate_note_evidence,
    verify_notes_l2,
    write_judgment_audit,
)

DEFAULT_PROFILE_PATH = "input/book_profile.json"
PROFILE_VERSION = JUDGMENT_SCHEMA_VERSION

NEUTRAL_REGISTER = "保持克制、平实、贴近原文的叙事语域"
NEUTRAL_PERSONA = "忠实原著语域的职业文学译者"
VALID_FOOTNOTE_POLICIES = ("none", "inline", "markdown_footnote")
VALID_SOURCES = ("known_work", "sampled", "mixed", "fallback")


def _split_anchors(source: str) -> List[str]:
    """从 pun source 串（"tale/tail"、"draw = 拉/画"）派生审计锚点（保序去重）。"""
    out: List[str] = []
    normalized = str(source or "")
    for sep in ("=", "，", ",", "、", ";", "；", "\n"):
        normalized = normalized.replace(sep, "/")
    for part in normalized.split("/"):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out


def _derive_src_anchors(source: str) -> List[str]:
    """从 source 串派生源侧（英文）定位锚点：取括号插注前的英文片段，保序去重。

    例: "whiting (fish vs. shoe-blacking)" → ["whiting"]；
        "William the Conqueror ... Edgar Atheling (…) " → ["William the Conqueror", "Edgar Atheling"]。
    """
    out: List[str] = []
    seen: set = set()
    for frag in _split_anchors(source):
        s = re.split(r"[（(]", str(frag))[0]
        for part in re.split(r"[.…]+|--|—", s):
            p = part.strip().strip('"\'‘’“”').strip(" ,;:。；：!?！？)）")
            if len(p) < 2 or not re.search(r"[A-Za-z]{2,}", p):
                continue
            pk = p.lower()
            if pk in seen:
                continue
            seen.add(pk)
            out.append(p)
    return out[:8]


_CN_HINT_STOPWORDS = (
    "双关", "谐音", "戏拟", "落点", "加注", "注释", "保留", "需要",
    "建议", "呼应", "上文", "此处", "处理", "直译", "音译", "可以", "或者",
)


def _hint_cn_anchors(text: str, cap: int = 4) -> List[str]:
    """从 hint 说明（中文）抽译文侧核销锚点：括号/引号内中文术语优先。"""
    out: List[str] = []
    for grp in re.findall(r'[（(「『“"]([^（()」』”"]{1,24})[）)」』”"]', str(text)):
        g = grp.strip()
        if 2 <= len(g) <= 6 and re.fullmatch(r"[\u4e00-\u9fff]+", g) and g not in out:
            out.append(g)
    if not out:
        for seg in re.split(r"[，。、；：,.!?;:…\s]+", str(text)):
            if re.search(r"[A-Za-z]", seg):
                continue
            for run in re.findall(r"[\u4e00-\u9fff]{2,}", seg):
                for part in re.split(r"[与和或及等的]", run):
                    p = part.strip()
                    if len(p) < 2 or p in _CN_HINT_STOPWORDS or p in out:
                        continue
                    if any(stop in p for stop in ("双关", "谐音", "戏拟")):
                        continue
                    out.append(p)
                    if len(out) >= cap:
                        break
                if len(out) >= cap:
                    break
            if len(out) >= cap:
                break
    return out[:cap]


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ────────────────────────────────────────────────────────────────────
# 抽样
# ────────────────────────────────────────────────────────────────────

def spread_sample_chapters(files: Sequence[Path], limit: int = 5) -> List[Path]:
    """从排序后的章节文件中均匀取 limit 个（覆盖首/中/尾）。"""
    files = sorted(files)
    if len(files) <= limit:
        return list(files)
    n = len(files)
    idxs = {round(i * (n - 1) / (limit - 1)) for i in range(limit)}
    return [files[i] for i in sorted(idxs)]


def load_chapter_samples(
    input_dir: Path,
    chapters: Optional[Sequence[str]] = None,
    limit: int = 5,
    per_chapter_chars: int = 6000,
) -> List[Dict[str, str]]:
    """读取抽样章节。chapters 指定时按指定顺序取；否则均匀抽样。

    返回 [{"chapter_id": "ch01", "text": "..."}]（每章截取前 per_chapter_chars）。
    """
    input_dir = Path(input_dir)
    if chapters:
        files = [input_dir / f"{c}.md" for c in chapters]
        files = [f for f in files if f.exists()]
    else:
        files = spread_sample_chapters(sorted(input_dir.glob("ch*.md")), limit)
    out: List[Dict[str, str]] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        out.append({"chapter_id": f.stem, "text": text[:per_chapter_chars]})
    return out


# ────────────────────────────────────────────────────────────────────
# LLM prompt / 解析
# ────────────────────────────────────────────────────────────────────

_PROFILE_SCHEMA_BLOCK = """{
  "work_guess": {"recognized": true|false, "title": "书名或空", "title_zh": "中文通行译名或空", "author": "作者或空", "year": "首版年或空"},
  "source": "known_work | sampled",
  "confidence": 0.0-1.0,
  "genre": "体裁（从原文特征得出，如：侦探小说/儿童文学/科幻小说/荒诞童话）",
  "era": "时代（如：1920s / 维多利亚时期；不确定给空）",
  "region": "地域文化背景（如：英国；不确定给空）",
  "persona": "翻译该作品应采用的译者人设（一句话）",
  "register_default": "全书默认语域目标（一句话，指导润色时的文学度/口语度）",
  "footnote_policy": "none | inline | markdown_footnote（依据典故密度与体裁推荐）",
  "forbidden_modernisms": ["与作品时代/语域冲突、应禁用的现代用词", "..."] ,
  "translation_hints": ["确实出现在抽样文本中的年代/地域用词、行话、引用，格式：英文原词 = 含义与处理建议", "..."],
  "pun_watchlist": [
    {"source": "英文原词/词组（如 tale/tail、draw）", "kind": "homophone|polysemy|allusion|name-pun", "src_anchors": ["源侧定位锚点：英文原词/短句片段（供在原文中定位，≥1 个）"], "anchors": ["审计锚点：译文正文或注释中应出现的中/英关键词（≥1 个）"], "strategy": "建议处理策略（双关落点/正文兼顾/尾注说明）", "hint": "位置提示（如 ch07 茶会）"},
    "..."
  ],
  "chapter_register_map": {"ch01": "该章语域目标（仅针对抽样章）", "..."},
  "evidence_quotes": ["支撑你风格判断的抽样原文英文短句（每章至少 1 条）", "..."],
  "narrative_arc": [],
  "setup_payoff": [],
  "voice_notes": [],
  "generation": {"model": "生成模型或空", "params": {}, "time": "ISO 时间"}
}"""


def build_profile_prompt(
    samples: Sequence[Dict[str, str]],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    meta_block = ""
    if extra_meta:
        meta_block = f"\n【可用元数据】\n{json.dumps(extra_meta, ensure_ascii=False)}\n"
    samples_block = "\n\n".join(
        f"===== {s['chapter_id']}（抽样节选）=====\n{s['text']}" for s in samples
    )
    return f"""你是一位文学翻译项目的风格分析专家。项目即将翻译一部英文文学作品，
你需要先阅读抽样章节，产出一份【作品风格剖面】JSON，供直译/润色/评论/终审四个环节共享。

【第一步：判定作品是否知名】
- 若你根据抽样文本与元数据认出这是知名已出版作品：genre/era/region/persona/register_default
  以该作品在文学史上的公认风格为准（可参考抽样文本核对），source 填 "known_work"，
  work_guess 填写书名/中文通行译名/作者/首版年，confidence 给高；不确认中文译名时 title_zh 留空。
- 若认不出：完全依据抽样文本特征归纳（语言年代感、句式、对话风格、体裁特征），
  source 填 "sampled"，work_guess.recognized=false，confidence 按证据充分度给。
- 严禁臆造：不确定的作品名/年份给空字符串；不确定的字段给保守值。

【第二步：归纳风格要素】
- translation_hints 只登记确实出现在抽样文本中的词句（英文原词 = 含义/处理建议），
  不要为任何未在抽样中出现的词编造提示。
- pun_watchlist 登记抽样文本（含 translation_hints）中的双关/谐音/一词多义笑点；
  每条必须给 src_anchors（英文原词片段，供在原文中定位）与 anchors（审计据此在译文正文/注释中找落点），
  确实没有双关时给空数组。
- forbidden_modernisms 列出与作品时代/语域明显冲突的现代中文用词（网络用语等）。
- footnote_policy：典故/引用密度高 → inline 或 markdown_footnote；几乎无典故 → none。
- chapter_register_map 只覆盖抽样章（章 ID 见下）。
- evidence_quotes 必须给出抽样原文中的英文短句作为判断依据（每抽样章至少 1 条）。

【输出 JSON Schema — 严格遵守】
{_PROFILE_SCHEMA_BLOCK}
{meta_block}
【抽样原文】
{samples_block}

请输出纯 JSON（无 markdown 代码块，无额外说明）："""


def normalize_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """校验/归一 LLM 返回的剖面 dict（缺失字段补默认值）。"""
    if not isinstance(data, dict):
        raise ValueError("profile must be a dict")

    def _str(key: str, default: str = "") -> str:
        v = data.get(key)
        return str(v).strip() if isinstance(v, (str, int, float)) and str(v).strip() else default

    def _str_list(key: str, cap: int = 50) -> List[str]:
        v = data.get(key) or []
        if isinstance(v, dict):
            v = [f"{k} = {val}" for k, val in v.items()]
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for item in v:
            if isinstance(item, dict):
                term = item.get("term") or item.get("phrase") or ""
                note = item.get("meaning") or item.get("note") or item.get("explanation") or ""
                item = f"{term} = {note}" if term else json.dumps(item, ensure_ascii=False)
            s = str(item).strip()
            if s:
                out.append(s)
        return out[:cap]

    def _pun_list(key: str, cap: int = 40) -> List[Dict[str, Any]]:
        v = data.get(key) or []
        if isinstance(v, dict):
            v = [v]
        if not isinstance(v, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append({"source": s, "kind": "",
                                "src_anchors": _derive_src_anchors(s),
                                "anchors": _split_anchors(s),
                                "strategy": "", "hint": ""})
                continue
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("term") or item.get("phrase") or "").strip()
            if not source:
                continue
            anchors_raw = item.get("anchors") or item.get("keywords") or []
            if isinstance(anchors_raw, str):
                anchors = _split_anchors(anchors_raw)
            elif isinstance(anchors_raw, list):
                anchors = [str(a).strip() for a in anchors_raw if str(a).strip()]
            else:
                anchors = []
            if not anchors:
                anchors = _split_anchors(source)
            src_raw = item.get("src_anchors") or item.get("source_anchors") or []
            if isinstance(src_raw, str):
                src_anchors = [x.strip() for x in _split_anchors(src_raw) if x.strip()]
            elif isinstance(src_raw, list):
                src_anchors = [str(a).strip() for a in src_raw if str(a).strip()]
            else:
                src_anchors = []
            if not src_anchors:
                src_anchors = _derive_src_anchors(source)
            out.append({
                "source": source,
                "kind": str(item.get("kind") or item.get("type") or "").strip(),
                "src_anchors": src_anchors[:8],
                "anchors": anchors[:8],
                "strategy": str(item.get("strategy") or item.get("suggestion") or item.get("note") or "").strip(),
                "hint": str(item.get("hint") or item.get("location") or item.get("position") or "").strip(),
            })
        return out[:cap]

    fp = _str("footnote_policy", "inline").lower()
    if fp not in VALID_FOOTNOTE_POLICIES:
        fp = "inline"
    src = _str("source", "sampled").lower()
    if src not in VALID_SOURCES:
        src = "sampled"
    try:
        conf = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    cmap_raw = data.get("chapter_register_map") or {}
    cmap: Dict[str, str] = {}
    if isinstance(cmap_raw, dict):
        for k, v in cmap_raw.items():
            k, v = str(k).strip(), str(v).strip()
            if k and v:
                cmap[k] = v

    wg_raw = data.get("work_guess") or {}
    work_guess = {
        "recognized": bool(wg_raw.get("recognized", False)) if isinstance(wg_raw, dict) else False,
        "title": str(wg_raw.get("title") or "").strip() if isinstance(wg_raw, dict) else "",
        "title_zh": str(wg_raw.get("title_zh") or "").strip() if isinstance(wg_raw, dict) else "",
        "author": str(wg_raw.get("author") or "").strip() if isinstance(wg_raw, dict) else "",
        "year": str(wg_raw.get("year") or "").strip() if isinstance(wg_raw, dict) else "",
    }

    def _thresholds(key: str) -> Dict[str, float]:
        """可选评分阈值覆盖块（Round 130 审计④：值随剖面落盘、
        load_profile 的重归一不再吞掉它；仅接受 0-10 数值，其他丢弃）。"""
        blk = data.get(key)
        out: Dict[str, float] = {}
        if isinstance(blk, dict):
            for k, v in blk.items():
                if (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and 0.0 <= float(v) <= 10.0):
                    out[str(k)] = float(v)
        return out

    generated_at = str(data.get("generated_at") or "").strip() or _now()
    generation_raw = data.get("generation") if isinstance(data.get("generation"), dict) else {}
    generation = {
        "model": str(generation_raw.get("model") or data.get("source_model") or "").strip(),
        "params": generation_raw.get("params") if isinstance(generation_raw.get("params"), dict) else {},
        "time": str(generation_raw.get("time") or generated_at).strip(),
    }
    for key, value in generation_raw.items():
        if key not in generation:
            generation[key] = value
    source_model = generation.get("model") or ""

    prof = {
        "schema_version": PROFILE_VERSION,
        "source": src,
        "confidence": conf,
        "genre": _str("genre", "文学小说"),
        "era": _str("era"),
        "region": _str("region"),
        "persona": _str("persona", NEUTRAL_PERSONA),
        "register_default": _str("register_default", NEUTRAL_REGISTER),
        "footnote_policy": fp,
        "forbidden_modernisms": _str_list("forbidden_modernisms"),
        "translation_hints": _str_list("translation_hints"),
        "pun_watchlist": _pun_list("pun_watchlist"),
        "chapter_register_map": cmap,
        "evidence_quotes": _str_list("evidence_quotes", cap=12),
        "critic_thresholds": _thresholds("critic_thresholds"),
        "work_guess": work_guess,
        "samples_used": [str(s) for s in (data.get("samples_used") or [])] if isinstance(data.get("samples_used"), list) else [],
        "generated_at": generated_at,
        "generation": generation,
    }
    prof.update(normalize_judgment_fields(data, source_model=source_model,
                                          created_at=generated_at))
    # load_profile() runs this normalizer before the pipeline sees the
    # artifact.  Preserve the deep-read coverage/diagnostics block so a v2
    # profile does not lose its provenance and coverage story on every load.
    if isinstance(data.get("deep_read"), dict):
        prof["deep_read"] = dict(data["deep_read"])
    # L2: translation_hints 中提及双关/谐音/戏拟的条目并入 pun_watchlist（LLM 漏报兜底；
    #     与既有条目同词（结构化已登记）时不重复合成）
    _GENERIC_EN = {
        "would", "could", "should", "their", "there", "these", "those",
        "which", "while", "about", "after", "before", "where", "little",
        "other", "every", "first", "still", "being", "under", "never",
        "always", "thing", "things", "with", "from",
    }

    def _key_tokens(*texts: str) -> set:
        toks: set = set()
        for t in texts:
            toks.update(w.lower() for w in re.findall(r"[A-Za-z]{4,}", str(t))
                        if w.lower() not in _GENERIC_EN)
        return toks

    seen_pun = {str(it.get("source") or "").strip().strip('"\'').lower()
                for it in prof["pun_watchlist"]}
    seen_tokens: set = set()
    for it in prof["pun_watchlist"]:
        seen_tokens |= _key_tokens(it.get("source") or "",
                                   " ".join(it.get("src_anchors") or []),
                                   " ".join(it.get("anchors") or []))
    for hint in list(prof["translation_hints"]):
        s = str(hint)
        if not ("双关" in s or "谐音" in s or "戏拟" in s or "pun" in s.lower()):
            continue
        head, _, tail = s.partition("=")
        key = head.strip().strip('"\'').lower()
        if not key or key in seen_pun:
            continue
        if _key_tokens(head) & seen_tokens:
            continue
        prof["pun_watchlist"].append({
            "source": head.strip() or s[:32],
            "kind": "hint",
            "src_anchors": _derive_src_anchors(head.strip()) or [head.strip()[:40]],
            "anchors": _hint_cn_anchors(tail or s),
            "strategy": tail.strip() or s,
            "hint": "translation_hints",
        })
        seen_pun.add(key)
        seen_tokens |= _key_tokens(head)

    # L2: 清理历史 profile 中与结构化条目同词的 hint 合成项；hint 锚点若全非中文，
    #     从 strategy 说明重新派生中文核销锚点（否则无法在译文中核销）。
    structured_tokens: set = set()
    for it in prof["pun_watchlist"]:
        if str(it.get("kind") or "") != "hint":
            structured_tokens |= _key_tokens(it.get("source") or "",
                                             " ".join(it.get("src_anchors") or []),
                                             " ".join(it.get("anchors") or []))
    cleaned: List[Dict[str, Any]] = []
    for it in prof["pun_watchlist"]:
        if str(it.get("kind") or "") == "hint":
            if _key_tokens(it.get("source") or "") & structured_tokens:
                continue
            anchors = it.get("anchors") or []
            if not any(re.search(r"[\u4e00-\u9fff]", str(a)) for a in anchors):
                cn = _hint_cn_anchors(str(it.get("strategy") or ""))
                if cn:
                    it = dict(it, anchors=cn)
        cleaned.append(it)
    prof["pun_watchlist"] = cleaned
    return prof


def heuristic_profile(
    samples: Optional[Sequence[Dict[str, str]]] = None,
    note: str = "",
) -> Dict[str, Any]:
    """LLM 不可用时的中性回退剖面（不注入任何作品特定先验）。"""
    return {
        "schema_version": PROFILE_VERSION,
        "source": "fallback",
        "confidence": 0.0,
        "genre": "文学小说",
        "era": "",
        "region": "",
        "persona": NEUTRAL_PERSONA,
        "register_default": NEUTRAL_REGISTER,
        "footnote_policy": "inline",
        "forbidden_modernisms": [],
        "translation_hints": [],
        "pun_watchlist": [],
        "chapter_register_map": {},
        "evidence_quotes": [],
        "critic_thresholds": {},
        "work_guess": {"recognized": False, "title": "", "title_zh": "",
                       "author": "", "year": ""},
        "samples_used": [s["chapter_id"] for s in (samples or [])],
        "note": note,
        "generated_at": _now(),
        "narrative_arc": [],
        "setup_payoff": [],
        "voice_notes": [],
        "generation": {"model": "", "params": {}, "time": _now()},
    }


def parse_profile_response(
    raw: str,
    samples: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """解析 LLM 输出 → 规范化剖面；失败回退 heuristic_profile。"""
    from src.utils.llm_adapter import robust_json_loads

    try:
        data = robust_json_loads(raw)
    except Exception:
        data = {}
    if isinstance(data, dict) and isinstance(data.get("profile"), dict):
        data = data["profile"]
    if not isinstance(data, dict) or not data:
        return heuristic_profile(samples, note="parse_failed")
    try:
        prof = normalize_profile(data)
    except Exception as exc:  # noqa: BLE001 - 任何归一失败都回退中性
        return heuristic_profile(samples, note=f"normalize_failed: {type(exc).__name__}: {exc}")
    if samples and not prof.get("samples_used"):
        prof["samples_used"] = [s["chapter_id"] for s in samples]
    return prof


# ────────────────────────────────────────────────────────────────────
# LLM 调用（task_routing: style_profiling）
# ────────────────────────────────────────────────────────────────────

def default_style_profiler_generate(prompt: str, task_name: str = "style_profiling") -> str:
    from src.config import get_config
    from src.utils.llm_adapter import (
        get_llm_client_for_task,
        get_role_model_name,
    )

    cfg = get_config()
    if task_name not in (cfg.task_routing or {}):
        task_name = "style_profiling"
    model_key, params = cfg.resolve_task_model(task_name)
    role = cfg.get_task_role(task_name)
    llm = get_llm_client_for_task(task_name)
    model_name = get_role_model_name(role) or model_key
    return llm.generate(prompt, model_name=model_name, **params)


def build_profile(
    samples: Sequence[Dict[str, str]],
    llm_generate: Optional[Callable[[str], str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """抽样章节 → LLM → 剖面 dict。LLM 异常/解析失败时回退 heuristic。"""
    if not samples:
        return heuristic_profile(None, note="no_samples")
    gen = llm_generate or default_style_profiler_generate
    prompt = build_profile_prompt(samples, extra_meta)
    try:
        raw = gen(prompt)
    except Exception as exc:  # noqa: BLE001
        return heuristic_profile(samples, note=f"llm_error: {type(exc).__name__}: {exc}")
    return parse_profile_response(raw, samples)


# ────────────────────────────────────────────────────────────────────
# Round 131 deep mode: all chapters -> segment extraction -> merge -> L1/L2
# ────────────────────────────────────────────────────────────────────

def load_chapter_documents(
    input_dir: Path,
    chapters: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Load complete chapter sources (not samples)."""
    in_dir = Path(input_dir)
    if chapters:
        files = [in_dir / f"{c}.md" for c in chapters]
        files = [p for p in files if p.exists()]
    else:
        files = sorted(in_dir.glob("ch*.md"))
    out: List[Dict[str, Any]] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        out.append({"chapter_id": p.stem, "text": text, "path": str(p)})
    return out


def segment_chapter_text(
    text: str,
    max_chars: int = 60000,
    overlap_chars: int = 2000,
) -> List[Dict[str, Any]]:
    """Paragraph-aware windows.  Every source character is reviewed at least once."""
    src = str(text or "")
    limit = max(1000, int(max_chars or 60000))
    overlap = max(0, min(int(overlap_chars or 0), limit // 2))
    if len(src) <= limit:
        return [{"text": src, "char_start": 0, "char_end": len(src),
                 "part_index": 1, "part_total": 1}]
    raw: List[Dict[str, Any]] = []
    start = 0
    while start < len(src):
        end = min(len(src), start + limit)
        if end < len(src):
            floor = start + int(limit * 0.65)
            best = -1
            for marker in ("\n\n", "\n", ". ", "。", "! ", "? "):
                idx = src.rfind(marker, floor, end)
                if idx > best:
                    best = idx + (len(marker) if marker in ("\n\n", ". ", "! ", "? ") else 1)
            if best > start + 200:
                end = best
        raw.append({"text": src[start:end], "char_start": start,
                    "char_end": end, "part_index": len(raw) + 1,
                    "part_total": 0})
        if end >= len(src):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = start + 1
        start = next_start
    if len(raw) > 1 and len(raw[-1]["text"]) < max(500, overlap // 2):
        prev = raw[-2]
        prev["text"] = src[prev["char_start"]:]
        prev["char_end"] = len(src)
        raw.pop()
    for i, seg in enumerate(raw, 1):
        seg["part_index"] = i
        seg["part_total"] = len(raw)
    return raw


def build_deep_extract_prompt(chapter_id: str, segment: Dict[str, Any],
                              chapter_chars: int) -> str:
    return f"""你是文学翻译项目的全书深读分析员。现在只阅读第 {chapter_id} 章的一个分段，
目标是抽取能指导后续章节翻译的“判断层注记”。不要翻译正文，不要改写原文。

【分段位置】
- 章节总字符: {chapter_chars}
- 分段 {segment['part_index']}/{segment['part_total']}
- 本章字符区间: {segment['char_start']}–{segment['char_end']}

抽取规则：
1. notes 只能包含三类：voice（叙述/角色音色）、arc（人物/符号/主题的阶段性变化）、
   setup_payoff（本段出现的伏笔或回收线索）。
2. 每条 note 的 claim 用中文写 1–2 句，只描述叙事/语气/弧线；禁止规定术语或译名
   （不得出现“应译作”“统一译为”“标准译名”等）。
3. evidence 至少 1 条，quote 必须从下面【分段原文】中逐字复制一段连续英文，
   不得翻译、改写、补标点或拼接跨段文字。chapter 固定为 {chapter_id}。
4. setup_payoff 在单段内无法确定回收时，只登记 setup 线索为 setup_payoff，confidence 给低值；
   后续合并阶段再配对。不要为了配对而虚构尾章内容。
5. 每段上限：voice ≤4、arc ≤3、setup_payoff ≤4。没有可靠内容就输出空数组。
6. register_summary 用一句话描述本段语域；synopsis 用 1–2 句概括本段叙事功能。

只输出 JSON，不要 markdown：
{{
  "part_id": "{chapter_id}:{segment['part_index']}",
  "register_summary": "≤60字",
  "synopsis": "≤120字",
  "notes": [
    {{"id": "voice-{chapter_id}-p{segment['part_index']}-001", "type": "voice|arc|setup_payoff",
      "chapters": ["{chapter_id}"], "claim": "中文判断（≤160字）",
      "evidence": [{{"chapter": "{chapter_id}", "quote": "原文逐字引文（≤200字符）"}}],
      "confidence": 0.0-1.0}}
  ]
}}

【分段原文】
{segment['text']}
"""


def parse_deep_extract_response(raw: str) -> Dict[str, Any]:
    from src.utils.llm_adapter import robust_json_loads

    try:
        data = robust_json_loads(raw)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {"notes": [], "register_summary": "", "synopsis": "",
                "parse_error": "not_a_json_object"}
    return {
        "part_id": str(data.get("part_id") or ""),
        "register_summary": str(data.get("register_summary") or "").strip()[:180],
        "synopsis": str(data.get("synopsis") or "").strip()[:400],
        "notes": data.get("notes") if isinstance(data.get("notes"), list) else [],
        "parse_error": "",
    }


def build_deep_merge_prompt(
    chapter_ids: Sequence[str],
    segment_payloads: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    base_profile: Optional[Dict[str, Any]],
    partial: bool = False,
) -> str:
    compact_profile = {}
    if isinstance(base_profile, dict):
        compact_profile = {k: base_profile.get(k) for k in (
            "genre", "era", "region", "persona", "register_default",
            "footnote_policy", "translation_hints", "forbidden_modernisms",
            "work_guess", "source", "confidence",
        )}
    summaries = [{
        "part_id": p.get("part_id"),
        "register_summary": p.get("register_summary"),
        "synopsis": p.get("synopsis"),
        "notes": len(p.get("notes") or []),
    } for p in segment_payloads]
    # Keep the merge prompt bounded without systematically starving the tail
    # chapters: setup/payoff evidence often only becomes pairable there.
    cands_all = sorted(
        candidates,
        key=lambda n: (-float(n.get("confidence") or 0.0), str(n.get("id") or "")),
    )
    cap = 400
    if len(cands_all) <= cap:
        cands = cands_all
    else:
        chapter_order = sorted({str(c) for n in cands_all
                               for c in (n.get("chapters") or []) if c})
        tail = set(chapter_order[-max(2, len(chapter_order) // 4):])
        tail_cands = [n for n in cands_all
                      if any(str(c) in tail for c in (n.get("chapters") or []))]
        tail_quota = min(len(tail_cands), max(80, cap // 3))
        chosen = tail_cands[:tail_quota]
        chosen_ids = {str(n.get("id") or "") for n in chosen}
        for n in cands_all:
            if len(chosen) >= cap:
                break
            if str(n.get("id") or "") not in chosen_ids:
                chosen.append(n)
        cands = chosen
    compact_cands = [{
        "id": n.get("id"),
        "type": n.get("type"),
        "chapters": n.get("chapters"),
        "claim": n.get("claim"),
        "confidence": n.get("confidence"),
        "evidence": [{"chapter": e.get("chapter"), "quote": e.get("quote")}
                     for e in (n.get("evidence") or [])],
    } for n in cands]
    scope_label = "本批章节" if partial else "全部章节"
    setup_rule = (
        "3. setup_payoff 只有在候选证据同时包含较早章与较晚章时才配成一条；chapters 按 [最早章, 最晚章]，\n"
        "   evidence 至少包含两章各一条。若本批只看到较早的 setup、尚不能确定回收，仍保留为 setup_payoff，\n"
        "   chapters 只写已见章、confidence ≤0.55，交由下一层合并配对；不得虚构后文。"
        if partial else
        "3. setup_payoff 只有在候选证据同时包含较早章与较晚章时才配成一条；chapters 按 [最早章, 最晚章]，\n"
        "   evidence 至少包含两章各一条。无法确定回收时不要输出 setup_payoff（可降为 arc 或丢弃）。"
    )
    return f"""你是全书深读的合并编辑。输入是逐段抽取结果（含摘要与候选注记），
不是原文。请合并重复项、消除互相矛盾的判断，并把可确定的伏笔与后文回收配成一对。

硬性规则：
1. 不得新增任何 evidence quote；只能逐字复用候选中已有的 quote。没有候选证据就丢弃该判断。
2. {scope_label} id：{", ".join(chapter_ids)}。chapter_register_map 必须覆盖{scope_label}每一个章节。
{setup_rule}
4. arc 上限 20 条、setup_payoff 上限 30 条、voice 上限 50 条（每章最多 8 条）。
5. claim 只描述语气/人物弧线/伏笔回收/叙事视角；禁止规定术语、译名、统一译法。
6. status 一律输出 active；后续机械与交叉模型校验会决定是否降级。

只输出 JSON：
{{
  "chapter_register_map": {{"ch01": "该章语域目标", "...": "..."}},
  "narrative_arc": [统一 envelope...],
  "setup_payoff": [统一 envelope...],
  "voice_notes": [统一 envelope...]
}}

【基础剖面（v1 字段只作上下文，不在此重复输出）】
{json.dumps(compact_profile, ensure_ascii=False)}

【分段摘要】
{json.dumps(summaries, ensure_ascii=False)}

【候选注记（quote 逐字复用）】
{json.dumps(compact_cands, ensure_ascii=False)}
"""


def parse_deep_merge_response(raw: str) -> Dict[str, Any]:
    from src.utils.llm_adapter import robust_json_loads

    try:
        data = robust_json_loads(raw)
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get("profile"), dict):
        data = data["profile"]
    if not isinstance(data, dict):
        return {"parse_error": "not_a_json_object"}
    return data


def build_setup_pairing_prompt(
    chapter_ids: Sequence[str],
    setup_candidates: Sequence[Dict[str, Any]],
) -> str:
    """Ask only for candidate-id pairs; code assembles evidence deterministically."""
    compact = [{
        "id": n.get("id"),
        "type": n.get("type"),
        "chapters": n.get("chapters") or [],
        "claim": n.get("claim"),
        "confidence": n.get("confidence"),
        "evidence": [{"chapter": e.get("chapter"), "quote": e.get("quote")}
                     for e in (n.get("evidence") or [])[:2]],
    } for n in setup_candidates]
    return f"""你是全书伏笔配对编辑。输入是逐段抽取后保留的单章 setup_payoff 候选。
请只判断哪些候选共同构成同一叙事线的“较早埋线 → 较晚回收”。

硬性规则：
1. 只能从候选 id 中选择 setup_id 与 payoff_id，不得发明 id、quote 或新 claim。
2. setup_id 的章节必须早于 payoff_id；证据不足或不在同一叙事线时省略该对。
3. 同一候选最多进入一对；宁可少配，不得为了覆盖率硬配。
4. 输出只含 pair 的 id 与简短 reason，不要复制 evidence。

章节顺序：{", ".join(chapter_ids)}
只输出 JSON：
{{"pairs": [{{"setup_id": "候选 id", "payoff_id": "候选 id", "reason": "≤60字"}}]}}

【setup_payoff 候选】
{json.dumps(compact, ensure_ascii=False)}
"""


def parse_setup_pairing_response(raw: str) -> Dict[str, Any]:
    from src.utils.llm_adapter import robust_json_loads

    try:
        data = robust_json_loads(raw)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {"parse_error": "not_a_json_object", "pairs": []}
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        return {"parse_error": "pairs_not_a_list", "pairs": []}
    return {"parse_error": "", "pairs": pairs}


def apply_setup_pairs(
    pairs: Sequence[Dict[str, Any]],
    setup_candidates: Sequence[Dict[str, Any]],
    chapter_ids: Sequence[str],
    source_model: str = "",
    created_at: str = "",
) -> List[Dict[str, Any]]:
    """Build validated-envelope candidates from model-selected candidate ids."""
    by_id = {str(n.get("id") or ""): n for n in setup_candidates}
    order = {cid: idx for idx, cid in enumerate(chapter_ids)}
    seen_pairs: set = set()
    used: set = set()
    out: List[Dict[str, Any]] = []
    for item in pairs or []:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("setup_id") or "").strip()
        pid = str(item.get("payoff_id") or "").strip()
        if not sid or not pid or sid == pid or (sid, pid) in seen_pairs:
            continue
        early = by_id.get(sid)
        late = by_id.get(pid)
        if not early or not late or sid in used or pid in used:
            continue
        early_ev = (early.get("evidence") or [{}])[0]
        late_ev = (late.get("evidence") or [{}])[0]
        early_ch = str(early_ev.get("chapter") or "")
        late_ch = str(late_ev.get("chapter") or "")
        if not early_ch or not late_ch or early_ch == late_ch:
            continue
        if order.get(early_ch, len(order)) >= order.get(late_ch, len(order)):
            continue
        claim = (f"{str(early.get('claim') or '').strip()}；后文回收："
                 f"{str(late.get('claim') or '').strip()}").strip("；")[:600]
        if not claim:
            continue
        try:
            conf_early = float(early.get("confidence") or 0.0)
            conf_late = float(late.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf_early = conf_late = 0.0
        pair_id = "setup-pair-" + hashlib.sha1(
            (sid + "\x00" + pid).encode("utf-8")).hexdigest()[:16]
        out.append({
            "id": pair_id,
            "type": "setup_payoff",
            "chapters": [early_ch, late_ch],
            "claim": claim,
            "evidence": [dict(early_ev), dict(late_ev)],
            "confidence": round((min(1.0, max(0.0, conf_early))
                                 + min(1.0, max(0.0, conf_late))) / 2.0, 3),
            "status": "active",
            "pairing_status": "paired",
            "source_model": source_model,
            "created_at": created_at,
            "meta": {"setup_id": sid, "payoff_id": pid,
                     "pair_reason": str(item.get("reason") or "")[:160]},
        })
        seen_pairs.add((sid, pid))
        used.add(sid)
        used.add(pid)
    return out


def pair_setup_candidates(
    chapter_ids: Sequence[str],
    setup_candidates: Sequence[Dict[str, Any]],
    gen: Callable[[str], str],
    merge_max_chars: int = 45000,
    source_model: str = "",
    created_at: str = "",
) -> Dict[str, Any]:
    """Candidate-id-only global setup pairing (Auto Ontology-style commit)."""
    prompt = build_setup_pairing_prompt(chapter_ids, setup_candidates)
    out: Dict[str, Any] = {"pairs": [], "notes": [], "error": "",
                           "prompt_chars": len(prompt), "called": False}
    if merge_max_chars > 0 and len(prompt) > merge_max_chars:
        out["error"] = f"pair_prompt_too_large:{len(prompt)}>{merge_max_chars}"
        return out
    out["called"] = True
    try:
        parsed = parse_setup_pairing_response(gen(prompt) or "")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"llm_error:{type(exc).__name__}: {exc}"
        return out
    if parsed.get("parse_error"):
        out["error"] = str(parsed["parse_error"])
        return out
    notes = apply_setup_pairs(
        parsed.get("pairs") or [], setup_candidates, chapter_ids,
        source_model=source_model, created_at=created_at)
    out["pairs"] = parsed.get("pairs") or []
    out["notes"] = notes
    return out


def _candidate_chapters(note: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for raw in note.get("chapters") or []:
        cid = str(raw or "").strip()
        if cid and cid not in out:
            out.append(cid)
    for ev in note.get("evidence") or []:
        cid = str(ev.get("chapter") or "").strip()
        if cid and cid not in out:
            out.append(cid)
    return out


def _invoke_deep_merge(
    chapter_ids: Sequence[str],
    segment_payloads: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    base_profile: Optional[Dict[str, Any]],
    gen: Callable[[str], str],
    merge_max_chars: int,
    partial: bool,
    source_model: str,
    created_at: str,
) -> Dict[str, Any]:
    """Run one bounded merge call and mechanically enforce candidate quotes."""
    prompt = build_deep_merge_prompt(
        chapter_ids, segment_payloads, candidates, base_profile, partial=partial)
    out: Dict[str, Any] = {
        "called": False, "ok": False, "parsed": False, "notes": [],
        "chapter_register_map": {}, "error": "", "prompt_chars": len(prompt),
    }
    if merge_max_chars > 0 and len(prompt) > merge_max_chars:
        out["error"] = f"merge_prompt_too_large:{len(prompt)}>{merge_max_chars}"
        return out
    out["called"] = True
    try:
        merged = parse_deep_merge_response(gen(prompt) or "")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"llm_error:{type(exc).__name__}: {exc}"
        return out
    parse_error = str(merged.get("parse_error") or "")
    if parse_error or not merged:
        out["error"] = parse_error or "empty_merge_response"
        return out
    out["parsed"] = True
    cmap = merged.get("chapter_register_map")
    if isinstance(cmap, dict):
        wanted = set(chapter_ids)
        for cid in chapter_ids:
            val = str(cmap.get(cid) or "").strip()
            if cid in wanted and val:
                out["chapter_register_map"][cid] = val
    notes: List[Dict[str, Any]] = []
    for key, hint in (("voice_notes", "voice"), ("setup_payoff", "setup_payoff"),
                      ("narrative_arc", "arc"), ("arc_notes", "arc")):
        notes.extend(normalize_notes(
            merged.get(key) or [], type_hint=hint,
            source_model=source_model, created_at=created_at,
        ))
    # Enforce the candidate boundary at every merge level, not only at the
    # final call.  Otherwise an invented quote from a partial merge would
    # become an allowed "candidate" for the next level.
    notes = restrict_notes_to_candidate_quotes(notes, candidates)
    notes = ensure_unique_note_ids(dedupe_notes(notes))
    out["notes"] = notes
    if not notes:
        out["error"] = "merge_returned_no_usable_notes"
        return out
    out["ok"] = True
    return out


def _split_deep_merge_batches(
    chapter_ids: Sequence[str],
    segment_payloads: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    base_profile: Optional[Dict[str, Any]],
    merge_max_chars: int,
    merge_batch_max_candidates: int = 80,
) -> List[List[Dict[str, Any]]]:
    """Greedily partition candidates by chapter order under a prompt budget."""
    order = {cid: idx for idx, cid in enumerate(chapter_ids)}

    def sort_key(note: Dict[str, Any]):
        chapters = _candidate_chapters(note)
        first = min((order.get(c, len(order)) for c in chapters), default=len(order))
        try:
            conf = float(note.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        return (first, -conf, str(note.get("id") or ""))

    ordered = sorted(candidates, key=sort_key)
    payload_by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    for payload in segment_payloads:
        cid = str(payload.get("chapter_id") or "")
        if cid:
            payload_by_chapter.setdefault(cid, []).append(payload)

    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for note in ordered:
        if current and merge_batch_max_candidates > 0 \
                and len(current) >= merge_batch_max_candidates:
            batches.append(current)
            current = []
        trial = current + [note]
        trial_chapters = [cid for cid in chapter_ids
                          if any(cid in _candidate_chapters(n) for n in trial)]
        trial_payloads = [p for cid in trial_chapters
                          for p in payload_by_chapter.get(cid, [])]
        prompt = build_deep_merge_prompt(
            trial_chapters, trial_payloads, trial, base_profile, partial=True)
        if current and merge_max_chars > 0 and len(prompt) > merge_max_chars:
            batches.append(current)
            current = [note]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def merge_deep_candidates(
    chapter_ids: Sequence[str],
    segment_payloads: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    base_profile: Optional[Dict[str, Any]],
    gen: Callable[[str], str],
    merge_max_chars: int = 45000,
    merge_batch_max_candidates: int = 80,
    source_model: str = "",
    created_at: str = "",
) -> Dict[str, Any]:
    """Merge candidates once, or hierarchically when one prompt is too large.

    The single-call path is retained for small books.  Large books split by
    chapter order into partial merges which may keep unresolved setup notes;
    a final merge pairs them across the whole book.  Every layer is fail-open
    and deterministic in chapter/id order.
    """
    chapter_ids = list(chapter_ids)
    candidates = [dict(n) for n in candidates]
    prompts: List[int] = []
    errors: List[str] = []
    merge_calls = 0
    if not candidates:
        return {"notes": [], "chapter_register_map": {}, "mode": "no_candidate_notes",
                "merge_called": False, "merge_calls": 0, "prompt_chars": [],
                "batch_count": 0, "errors": []}

    full_prompt = build_deep_merge_prompt(
        chapter_ids, segment_payloads, candidates, base_profile, partial=False)
    prompts.append(len(full_prompt))
    if merge_max_chars <= 0 or len(full_prompt) <= merge_max_chars:
        res = _invoke_deep_merge(
            chapter_ids, segment_payloads, candidates, base_profile, gen,
            merge_max_chars, False, source_model, created_at)
        merge_calls += int(bool(res.get("called")))
        if res.get("ok"):
            return {"notes": res["notes"], "chapter_register_map": res["chapter_register_map"],
                    "mode": "merged", "merge_called": merge_calls > 0,
                    "merge_calls": merge_calls, "prompt_chars": prompts,
                    "batch_count": 1, "errors": errors}
        mode = "merge_empty_fallback" if res.get("parsed") else "candidate_fallback"
        if res.get("error"):
            errors.append(str(res["error"]))
        return {"notes": candidates, "chapter_register_map": {},
                "mode": mode, "merge_called": merge_calls > 0,
                "merge_calls": merge_calls, "prompt_chars": prompts,
                "batch_count": 1, "errors": errors}

    batches = _split_deep_merge_batches(
        chapter_ids, segment_payloads, candidates, base_profile, merge_max_chars,
        merge_batch_max_candidates=merge_batch_max_candidates)
    partial_active: List[Dict[str, Any]] = []
    audit_notes: List[Dict[str, Any]] = []
    merged_cmap: Dict[str, str] = {}
    fallback_batches = 0
    payload_by_chapter: Dict[str, List[Dict[str, Any]]] = {}
    for payload in segment_payloads:
        cid = str(payload.get("chapter_id") or "")
        if cid:
            payload_by_chapter.setdefault(cid, []).append(payload)
    for batch in batches:
        batch_chapters = [cid for cid in chapter_ids
                          if any(cid in _candidate_chapters(n) for n in batch)]
        batch_payloads = [p for cid in batch_chapters
                          for p in payload_by_chapter.get(cid, [])]
        res = _invoke_deep_merge(
            batch_chapters, batch_payloads, batch, base_profile, gen,
            merge_max_chars, True, source_model, created_at)
        merge_calls += int(bool(res.get("called")))
        prompts.append(int(res.get("prompt_chars") or 0))
        if res.get("ok"):
            merged_cmap.update(res.get("chapter_register_map") or {})
            for note in res.get("notes") or []:
                if note.get("status") == "active":
                    partial_active.append(note)
                else:
                    audit_notes.append(note)
        else:
            fallback_batches += 1
            partial_active.extend(dict(n) for n in batch)
            if res.get("error"):
                errors.append(f"batch:{res.get('error')}")

    combined = ensure_unique_note_ids(dedupe_notes(partial_active))
    final_prompt = build_deep_merge_prompt(
        chapter_ids, segment_payloads, combined, base_profile, partial=False)
    prompts.append(len(final_prompt))
    if merge_max_chars <= 0 or len(final_prompt) <= merge_max_chars:
        res = _invoke_deep_merge(
            chapter_ids, segment_payloads, combined, base_profile, gen,
            merge_max_chars, False, source_model, created_at)
        merge_calls += int(bool(res.get("called")))
        if res.get("ok"):
            merged_cmap.update(res.get("chapter_register_map") or {})
            active = [n for n in res.get("notes") or [] if n.get("status") == "active"]
            audit_notes.extend(n for n in res.get("notes") or []
                               if n.get("status") != "active")
            mode = ("hierarchical_merged" if not fallback_batches
                    else "hierarchical_partial_merged")
            return {"notes": active + audit_notes, "chapter_register_map": merged_cmap,
                    "mode": mode, "merge_called": True,
                    "merge_calls": merge_calls, "prompt_chars": prompts,
                    "batch_count": len(batches), "errors": errors}
        if res.get("error"):
            errors.append(f"final:{res.get('error')}")
        return {"notes": combined + audit_notes, "chapter_register_map": merged_cmap,
                "mode": "hierarchical_candidate_fallback", "merge_called": True,
                "merge_calls": merge_calls, "prompt_chars": prompts,
                "batch_count": len(batches), "errors": errors}

    # Very long serials can still exceed the window after partial merges.  Pair
    # only the compact setup candidates globally, and keep the already-merged
    # voice/arc notes rather than dropping the whole artifact.  The model picks
    # candidate ids only; code assembles the two-chapter envelope and evidence.
    setup_notes = [n for n in combined if n.get("type") == "setup_payoff"]
    other_notes = [n for n in combined if n.get("type") != "setup_payoff"]
    if setup_notes:
        pair_res = pair_setup_candidates(
            chapter_ids, setup_notes, gen, merge_max_chars=merge_max_chars,
            source_model=source_model, created_at=created_at)
        prompts.append(int(pair_res.get("prompt_chars") or 0))
        merge_calls += int(bool(pair_res.get("called")))
        if pair_res.get("notes"):
            paired = list(pair_res["notes"])
            used_ids = set()
            for n in paired:
                meta = n.get("meta") or {}
                used_ids.add(str(meta.get("setup_id") or ""))
                used_ids.add(str(meta.get("payoff_id") or ""))
            unpaired = []
            for n in setup_notes:
                if str(n.get("id") or "") in used_ids:
                    continue
                item = dict(n)
                item["status"] = "dormant"
                item["pairing_status"] = "unpaired"
                item.setdefault("meta", {})["pairing_status"] = "unpaired"
                unpaired.append(item)
            return {"notes": other_notes + paired + unpaired + audit_notes,
                    "chapter_register_map": merged_cmap,
                    "mode": "hierarchical_setup_pairing", "merge_called": True,
                    "merge_calls": merge_calls, "prompt_chars": prompts,
                    "batch_count": len(batches), "errors": errors}
        if pair_res.get("error"):
            errors.append(f"setup_pairing:{pair_res.get('error')}")
    dormant_setup = []
    for n in setup_notes:
        item = dict(n)
        item["status"] = "dormant"
        item["pairing_status"] = "unpaired"
        item.setdefault("meta", {})["pairing_status"] = "unpaired"
        dormant_setup.append(item)
    return {"notes": other_notes + dormant_setup + audit_notes,
            "chapter_register_map": merged_cmap,
            "mode": "hierarchical_combined_fallback", "merge_called": True,
            "merge_calls": merge_calls, "prompt_chars": prompts,
            "batch_count": len(batches), "errors": errors}


def _task_model_label(task_name: str) -> str:
    try:
        from src.config import get_config
        from src.utils.llm_adapter import get_role_model_name
        cfg = get_config()
        role = cfg.get_task_role(task_name)
        model_key, _ = cfg.resolve_task_model(task_name)
        return get_role_model_name(role) or model_key
    except Exception:
        return ""


def build_deep_profile(
    project_root: Path,
    base_profile: Dict[str, Any],
    chapters: Optional[Sequence[str]] = None,
    input_dir: Optional[Path] = None,
    llm_generate: Optional[Callable[[str], str]] = None,
    merge_llm_generate: Optional[Callable[[str], str]] = None,
    l2_generate: Optional[Callable[[str], str]] = None,
    max_chars: int = 60000,
    overlap_chars: int = 2000,
    min_similarity: float = 0.9,
    merge_max_chars: int = 45000,
    merge_batch_max_candidates: int = 80,
) -> Dict[str, Any]:
    """Deep read all chapters, merge, then L1+L2.  Never edits chapter files."""
    root = Path(project_root)
    indir = Path(input_dir) if input_dir else root / "input"
    docs = load_chapter_documents(indir, chapters=chapters)
    if not docs:
        return {"ok": False, "error": f"未在 {indir} 找到章节文件"}
    chapter_texts = {d["chapter_id"]: d["text"] for d in docs}
    chapter_ids = [d["chapter_id"] for d in docs]
    extract_gen = llm_generate or (
        lambda p: default_style_profiler_generate(p, "style_profiling_deep"))
    # Extraction and merge have different failure modes: extraction needs long
    # literary reasoning, while merge is a bounded normalization task.  Using
    # the xhigh extraction role for merge repeatedly hit the gateway's 300s
    # read timeout on real Dune candidates, so production uses a separate,
    # cheaper task.  Injected test generators still drive both stages.
    merge_gen = merge_llm_generate or llm_generate or (
        lambda p: default_style_profiler_generate(p, "style_profiling_merge"))

    segments: List[Tuple[str, Dict[str, Any]]] = []
    for doc in docs:
        for seg in segment_chapter_text(doc["text"], max_chars=max_chars,
                                        overlap_chars=overlap_chars):
            segments.append((doc["chapter_id"], seg))

    payloads: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    created_at = _now()
    for cid, seg in segments:
        prompt = build_deep_extract_prompt(cid, seg, len(chapter_texts[cid]))
        try:
            raw = extract_gen(prompt)
            parsed = parse_deep_extract_response(raw)
        except Exception as exc:  # noqa: BLE001
            parsed = {"notes": [], "register_summary": "", "synopsis": "",
                      "parse_error": f"llm_error:{type(exc).__name__}: {exc}"}
        parsed["chapter_id"] = cid
        if not parsed.get("part_id"):
            parsed["part_id"] = f"{cid}:{seg['part_index']}"
        parsed["char_start"] = seg["char_start"]
        parsed["char_end"] = seg["char_end"]
        if parsed.get("parse_error"):
            errors.append({"part_id": parsed["part_id"], "error": parsed["parse_error"]})
        payloads.append(parsed)
        notes = normalize_notes(parsed.get("notes") or [], source_model=_task_model_label("style_profiling_deep"),
                                created_at=created_at)
        candidates.extend(notes)

    # Per-segment register summaries are a deterministic fallback for chapters
    # the merge model omits.  The deep artifact should cover every chapter even
    # when one merge call returns a partial map.
    register_by_chapter: Dict[str, List[str]] = {}
    for p in payloads:
        cid = str(p.get("chapter_id") or "")
        rs = str(p.get("register_summary") or "").strip()
        if cid and rs:
            bucket = register_by_chapter.setdefault(cid, [])
            if rs not in bucket:
                bucket.append(rs)

    merge_res = merge_deep_candidates(
        chapter_ids, payloads, candidates, base_profile, merge_gen,
        merge_max_chars=merge_max_chars,
        merge_batch_max_candidates=merge_batch_max_candidates,
        source_model=_task_model_label("style_profiling_merge"),
        created_at=created_at,
    )
    merge_mode = str(merge_res.get("mode") or "candidate_fallback")
    merge_called = bool(merge_res.get("merge_called"))
    merge_calls = int(merge_res.get("merge_calls") or 0)
    merge_batch_count = int(merge_res.get("batch_count") or 0)
    merge_prompt_chars = list(merge_res.get("prompt_chars") or [])
    merge_errors = list(merge_res.get("errors") or [])
    merge_error = "; ".join(str(e) for e in merge_errors)
    merged_cmap: Dict[str, str] = dict(merge_res.get("chapter_register_map") or {})
    selected_notes: List[Dict[str, Any]] = [dict(n) for n in merge_res.get("notes") or []]

    profile = dict(base_profile or heuristic_profile(note="deep_base_missing"))
    final_cmap = dict(profile.get("chapter_register_map") or {})
    final_cmap.update(merged_cmap)
    for cid in chapter_ids:
        if not str(final_cmap.get(cid) or "").strip() and register_by_chapter.get(cid):
            final_cmap[cid] = "；".join(register_by_chapter[cid])[:240]
    if final_cmap:
        profile["chapter_register_map"] = final_cmap

    # A merge model is allowed to recombine extraction quotes, never invent a
    # quote that merely happens to exist elsewhere in the book.  This mirrors
    # Auto Ontology's candidate-only extraction contract.
    selected_notes = restrict_notes_to_candidate_quotes(selected_notes, candidates)
    selected_notes = ensure_unique_note_ids(dedupe_notes(selected_notes))
    limits = enforce_note_limits(selected_notes)
    selected_notes = limits["notes"]

    l1 = validate_note_evidence(selected_notes, chapter_texts, min_similarity=min_similarity)
    l2 = verify_notes_l2(l1["notes"], chapter_texts, llm_generate=l2_generate)
    profile = replace_notes_in_profile(profile, l2["notes"])
    profile["schema_version"] = PROFILE_VERSION
    _deep_model = _task_model_label("style_profiling_deep")
    _merge_model = _task_model_label("style_profiling_merge")
    profile["generation"] = {
        "model": _deep_model,
        "merge_model": _merge_model,
        "params": {"mode": "deep", "max_chars": max_chars,
                   "overlap_chars": overlap_chars,
                   "segments": len(segments), "l1_min_similarity": min_similarity,
                   "merge_max_chars": merge_max_chars,
                   "merge_batch_max_candidates": merge_batch_max_candidates,
                   "merge_calls": merge_calls,
                   "merge_strategy": "hierarchical" if merge_batch_count > 1 else "single"},
        "time": _now(),
        "prompt_version": "r131.deep.v3",
    }
    profile["generated_at"] = _now()
    active_notes_out = [n for n in l2["notes"] if n.get("status") == "active"]
    covered_chapters = set()
    for n in active_notes_out:
        for cid in chapter_ids_between(n.get("chapters") or []):
            if cid in chapter_texts:
                covered_chapters.add(cid)
    segments_total = len(segments)
    segments_ok = sum(1 for p in payloads if not p.get("parse_error"))
    register_covered = sum(1 for cid in chapter_ids
                           if str(final_cmap.get(cid) or "").strip())
    coverage = {
        "chapters_total": len(chapter_ids),
        "chapters_with_active_notes": len(covered_chapters),
        "chapter_coverage": round(len(covered_chapters) / len(chapter_ids), 4)
        if chapter_ids else 0.0,
        "chapter_register_total": len(chapter_ids),
        "chapter_register_covered": register_covered,
        "chapter_register_coverage": round(register_covered / len(chapter_ids), 4)
        if chapter_ids else 0.0,
        "segments_total": segments_total,
        "segments_ok": segments_ok,
        "segment_coverage": round(segments_ok / segments_total, 4)
        if segments_total else 0.0,
        "extraction_candidates": len(candidates),
        "notes_after_limits": len(selected_notes),
        "active_after_limits": sum(1 for n in selected_notes if n.get("status") == "active"),
        "notes_after_l1": len(l1["notes"]),
        "active_notes": len(active_notes_out),
        "setup_payoff_active": sum(1 for n in active_notes_out
                                   if n.get("type") == "setup_payoff"),
    }
    deep_complete = bool(
        segments_total
        and segments_ok == segments_total
        and bool(active_notes_out)
    )
    deep_status = (
        "complete" if deep_complete
        else ("empty" if not active_notes_out else "partial")
    )
    profile["deep_read"] = {
        "chapters": chapter_ids,
        "segments": len(segments),
        "complete": deep_complete,
        "status": deep_status,
        "split_strategy": "chapter_segments_with_overlap_then_cross_segment_merge",
        "merge_mode": merge_mode,
        "merge_called": merge_called,
        "merge_calls": merge_calls,
        "merge_batch_count": merge_batch_count,
        "merge_prompt_chars": merge_prompt_chars,
        "merge_quality": ("ok" if merge_mode in (
            "merged", "hierarchical_merged", "hierarchical_partial_merged",
            "hierarchical_setup_pairing") else "degraded"),
        "coverage": coverage,
        "extract_errors": errors,
        "merge_error": merge_error,
    }
    audit = build_audit_payload(
        profile, l1_audit=l1["audit"], l2_audit=l2["audit"],
        generation=profile.get("generation"),
        extra={"segment_payloads": [{
            "chapter_id": p.get("chapter_id"),
            "part_id": p.get("part_id"), "char_start": p.get("char_start"),
            "char_end": p.get("char_end"), "synopsis": p.get("synopsis"),
            "register_summary": p.get("register_summary"),
            "notes_emitted": len(p.get("notes") or []),
            "parse_error": p.get("parse_error", ""),
        } for p in payloads],
        "merge_error": merge_error, "merge_mode": merge_mode,
        "merge_called": merge_called, "merge_calls": merge_calls,
        "merge_batch_count": merge_batch_count,
        "merge_prompt_chars": merge_prompt_chars,
        "merge_errors": merge_errors, "coverage": coverage},
    )
    return {"ok": True, "complete": deep_complete, "profile": profile, "audit": audit,
            "chapter_texts": chapter_texts, "segments": len(segments),
            "candidate_notes": len(candidates), "notes_after_limits": len(selected_notes),
            "active_notes": len(active_notes_out), "merge_mode": merge_mode,
            "coverage": coverage}


# ────────────────────────────────────────────────────────────────────
# 剖面 → 运行时 style_guide 覆盖层
# ────────────────────────────────────────────────────────────────────

def profile_to_style_guide(
    profile: Optional[Dict[str, Any]],
    chapter_id: Optional[str] = None,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把剖面映射为 config.style_guide 覆盖 dict（与 config.yaml 基线合并）。

    chapter_id 命中 chapter_register_map 时，register_default 用该章目标；
    其余通用键（avg_sentence_length 等）沿用 base（config.yaml 基线）。
    """
    if base is None:
        try:
            from src.config import get_config
            base = dict(getattr(get_config(), "base_style_guide", {}) or {})
        except Exception:
            base = {}
    guide = dict(base)
    if not profile:
        return guide

    guide["genre"] = profile.get("genre") or guide.get("genre") or "文学小说"
    guide["persona"] = profile.get("persona") or NEUTRAL_PERSONA
    fp = profile.get("footnote_policy")
    guide["footnote_policy"] = fp if fp in VALID_FOOTNOTE_POLICIES else "inline"

    reg = profile.get("register_default") or NEUTRAL_REGISTER
    cmap = profile.get("chapter_register_map") or {}
    if chapter_id and isinstance(cmap, dict) and cmap.get(chapter_id):
        reg = str(cmap[chapter_id])
    guide["register_default"] = reg

    guide["forbidden_modernisms"] = list(profile.get("forbidden_modernisms") or [])
    guide["translation_hints"] = list(profile.get("translation_hints") or [])
    guide["pun_watchlist"] = list(profile.get("pun_watchlist") or [])

    era_region = "，".join(x for x in (profile.get("era"), profile.get("region")) if x)
    if era_region:
        guide["work_era_region"] = era_region
    wg = profile.get("work_guess") or {}
    if wg.get("title"):
        guide["work_title"] = wg["title"]
    guide["_profile_source"] = profile.get("source")
    guide["_profile_confidence"] = profile.get("confidence")
    return guide


# ────────────────────────────────────────────────────────────────────
# 落盘 / 加载 / 顶层入口
# ────────────────────────────────────────────────────────────────────

def save_profile(profile: Dict[str, Any], path: Path) -> None:
    """Atomically publish a profile artifact.

    A deep profile is consumed by a long-running translation pipeline and by
    story-graph rebuilds.  A torn JSON file from a crashed write would make the
    judgment layer disappear silently on the next load, so publish by rename.
    """
    atomic_write_json(Path(path), profile)


def load_profile(path: Path) -> Optional[Dict[str, Any]]:
    """加载剖面文件；不存在/损坏返回 None（调用方安静跳过）。"""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data:
        return None
    try:
        return normalize_profile(data)
    except Exception:
        return None


def profile_is_deep(profile: Optional[Dict[str, Any]]) -> bool:
    """Return True only for a completed deep-read artifact.

    ``schema_version`` alone is not enough: shallow profiles generated after
    Round 131 are schema v2 but carry no ``deep_read`` block.  The one-click
    launcher must upgrade those instead of treating them as finished.
    """
    if not isinstance(profile, dict):
        return False
    deep_read = profile.get("deep_read")
    if isinstance(deep_read, dict) and deep_read:
        # New artifacts carry an explicit completion bit.  A failed/empty deep
        # run remains useful as a fail-open partial artifact, but it must not
        # suppress the one-click retry on the next invocation.
        if "complete" in deep_read:
            return bool(deep_read.get("complete"))
        # Backward compatibility for early Round-131 artifacts that predate
        # the completion bit.
        return True
    generation = profile.get("generation")
    params = generation.get("params") if isinstance(generation, dict) else None
    return isinstance(params, dict) and params.get("mode") == "deep"


def summary_line(profile: Dict[str, Any]) -> str:
    wg = profile.get("work_guess") or {}
    title = wg.get("title") or "(未识别)"
    notes = []
    for key in ("narrative_arc", "setup_payoff", "voice_notes"):
        val = profile.get(key) or []
        notes.append(sum(1 for n in val if isinstance(n, dict)
                         and n.get("status") == "active"))
    return (
        f"source={profile.get('source')} conf={profile.get('confidence')} "
        f"genre={profile.get('genre')} era={profile.get('era') or '-'} work={title} "
        f"hints={len(profile.get('translation_hints') or [])} "
        f"pun={len(profile.get('pun_watchlist') or [])} "
        f"notes_active={sum(notes)}"
    )


def _try_build_story_graph(root: Path, profile: Optional[Dict[str, Any]]) -> tuple:
    """Build the deterministic graph artifact without ever blocking profiling."""
    try:
        from src.config import get_config
        cfg = get_config().get("story_graph", {}) or {}
        if isinstance(cfg, dict) and cfg.get("enabled", True) is False:
            return "", {"enabled": False}
        from src.story_graph import DEFAULT_GRAPH_PATH, build_and_save_story_graph
        graph = build_and_save_story_graph(root, profile=profile)
        rel = str((cfg or {}).get("path") or DEFAULT_GRAPH_PATH) if isinstance(cfg, dict) else DEFAULT_GRAPH_PATH
        return str(Path(root) / rel), graph.get("coverage") or {}
    except Exception as exc:  # noqa: BLE001 - graph is advisory
        print(f"⚠️ [StoryGraph] 构建失败（不阻塞剖面生成）: {type(exc).__name__}: {exc}")
        return "", {}


def generate_book_profile(
    project_root: Path = Path("."),
    chapters: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
    limit: int = 5,
    llm_generate: Optional[Callable[[str], str]] = None,
    input_dir: Optional[Path] = None,
    deep: bool = False,
    l2_generate: Optional[Callable[[str], str]] = None,
    deep_max_chars: int = 60000,
    deep_overlap_chars: int = 2000,
) -> Dict[str, Any]:
    """顶层入口：抽样（或全书深读）→ LLM → 落盘 input/book_profile.json。

    返回 {"ok": bool, ...}。已存在且非 force 时 skipped=True（不重新生成）。
    """
    root = Path(project_root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    out_path = Path(path) if path else root / DEFAULT_PROFILE_PATH

    existing = load_profile(out_path) if out_path.exists() else None
    # A shallow schema-v2 profile is not a completed deep artifact.  Only a
    # profile carrying deep_read provenance may satisfy ``--deep``; otherwise
    # the one-click launcher would silently keep a v1-quality profile forever.
    if existing is not None and not force and (not deep or profile_is_deep(existing)):
        prof = existing
        graph_path, graph_stats = _try_build_story_graph(root, prof)
        return {"ok": True, "skipped": True, "reason": "profile_exists", "profile": prof,
                "path": str(out_path), "graph_path": graph_path,
                "graph_coverage": graph_stats}

    samples = load_chapter_samples(in_dir, chapters=chapters, limit=limit)
    if not samples:
        return {"ok": False, "error": f"未在 {in_dir} 找到章节样本（ch*.md）", "path": str(out_path)}

    if dry_run:
        prof = heuristic_profile(samples, note="dry_run")
        return {"ok": True, "dry_run": True, "profile": prof, "path": str(out_path), "samples": [s["chapter_id"] for s in samples]}

    if deep:
        base_profile = existing
        if not base_profile or base_profile.get("source") == "fallback":
            base_profile = build_profile(samples, llm_generate=llm_generate)
        if base_profile.get("source") == "fallback":
            return {
                "ok": True,
                "profile": base_profile,
                "path": str(out_path),
                "saved": False,
                "deep": True,
                "error": "base_profile_fallback",
                "note": base_profile.get("note", ""),
            }
        deep_res = build_deep_profile(
            root, base_profile, chapters=chapters, input_dir=in_dir,
            llm_generate=llm_generate, l2_generate=l2_generate,
            max_chars=deep_max_chars, overlap_chars=deep_overlap_chars,
        )
        if not deep_res.get("ok"):
            return {"ok": False, "error": deep_res.get("error"), "path": str(out_path)}
        prof = deep_res["profile"]
        saved = True
        save_profile(prof, out_path)
        audit_paths = write_judgment_audit(root, deep_res["audit"])
        graph_path, graph_stats = _try_build_story_graph(root, prof)
        return {
            "ok": True,
            "profile": prof,
            "path": str(out_path),
            "saved": saved,
            "samples": [s["chapter_id"] for s in samples],
            "source": prof.get("source"),
            "confidence": prof.get("confidence"),
            "deep": True,
            "segments": deep_res.get("segments", 0),
            "complete": bool(deep_res.get("complete", False)),
            "candidate_notes": deep_res.get("candidate_notes", 0),
            "notes_after_limits": deep_res.get("notes_after_limits", 0),
            "active_notes": deep_res.get("active_notes", 0),
            "merge_mode": deep_res.get("merge_mode", ""),
            "coverage": deep_res.get("coverage") or {},
            "audit_paths": audit_paths,
            "graph_path": graph_path,
            "graph_coverage": graph_stats,
        }

    prof = build_profile(samples, llm_generate=llm_generate)
    # LLM 不可用时只回退中性 profile —— 不落盘，避免用空剖面遮蔽后续真实生成
    saved = prof.get("source") != "fallback"
    if saved:
        save_profile(prof, out_path)
    graph_path, graph_stats = _try_build_story_graph(root, prof) if (saved or out_path.exists()) else ("", {})
    return {
        "ok": True,
        "profile": prof,
        "path": str(out_path),
        "saved": saved,
        "samples": [s["chapter_id"] for s in samples],
        "source": prof.get("source"),
        "confidence": prof.get("confidence"),
        "deep": False,
        "graph_path": graph_path,
        "graph_coverage": graph_stats,
    }
