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

剖面 JSON schema（schema_version=1）：
  genre / era / region / persona / register_default / footnote_policy /
  forbidden_modernisms / translation_hints / pun_watchlist / chapter_register_map /
  evidence_quotes / critic_thresholds（可选——按书覆盖 critic 评分阈值，0-10 数值） /
  work_guess / confidence / source / samples_used

LLM 失败时回退 heuristic_profile()：中性默认 + confidence=0 + source=fallback，
保证管线仍可运行且不注入任何作品特定先验。
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_PROFILE_PATH = "input/book_profile.json"
PROFILE_VERSION = 1

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
  "work_guess": {"recognized": true|false, "title": "书名或空", "author": "作者或空", "year": "首版年或空"},
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
  "evidence_quotes": ["支撑你风格判断的抽样原文英文短句（每章至少 1 条）", "..."]
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
  work_guess 填写书名/作者/首版年，confidence 给高。
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
        "generated_at": _now(),
    }
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
        "work_guess": {"recognized": False, "title": "", "author": "", "year": ""},
        "samples_used": [s["chapter_id"] for s in (samples or [])],
        "note": note,
        "generated_at": _now(),
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

def default_style_profiler_generate(prompt: str) -> str:
    from src.config import get_config
    from src.utils.llm_adapter import (
        get_llm_client_for_task,
        get_role_model_name,
    )

    cfg = get_config()
    model_key, params = cfg.resolve_task_model("style_profiling")
    role = cfg.get_task_role("style_profiling")
    llm = get_llm_client_for_task("style_profiling")
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


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


def summary_line(profile: Dict[str, Any]) -> str:
    wg = profile.get("work_guess") or {}
    title = wg.get("title") or "(未识别)"
    return (
        f"source={profile.get('source')} conf={profile.get('confidence')} "
        f"genre={profile.get('genre')} era={profile.get('era') or '-'} work={title} "
        f"hints={len(profile.get('translation_hints') or [])} "
        f"pun={len(profile.get('pun_watchlist') or [])}"
    )


def generate_book_profile(
    project_root: Path = Path("."),
    chapters: Optional[Sequence[str]] = None,
    path: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
    limit: int = 5,
    llm_generate: Optional[Callable[[str], str]] = None,
    input_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """顶层入口：抽样 → LLM → 落盘 input/book_profile.json。

    返回 {"ok": bool, ...}。已存在且非 force 时 skipped=True（不重新生成）。
    """
    root = Path(project_root)
    in_dir = Path(input_dir) if input_dir else root / "input"
    out_path = Path(path) if path else root / DEFAULT_PROFILE_PATH

    if out_path.exists() and not force:
        prof = load_profile(out_path)
        return {"ok": True, "skipped": True, "reason": "profile_exists", "profile": prof, "path": str(out_path)}

    samples = load_chapter_samples(in_dir, chapters=chapters, limit=limit)
    if not samples:
        return {"ok": False, "error": f"未在 {in_dir} 找到章节样本（ch*.md）", "path": str(out_path)}

    if dry_run:
        prof = heuristic_profile(samples, note="dry_run")
        return {"ok": True, "dry_run": True, "profile": prof, "path": str(out_path), "samples": [s["chapter_id"] for s in samples]}

    prof = build_profile(samples, llm_generate=llm_generate)
    # LLM 不可用时只回退中性 profile —— 不落盘，避免用空剖面遮蔽后续真实生成
    saved = prof.get("source") != "fallback"
    if saved:
        save_profile(prof, out_path)
    return {
        "ok": True,
        "profile": prof,
        "path": str(out_path),
        "saved": saved,
        "samples": [s["chapter_id"] for s in samples],
        "source": prof.get("source"),
        "confidence": prof.get("confidence"),
    }
