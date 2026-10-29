"""Round 34: 共享文本匹配 helpers."""
from __future__ import annotations
import re
from typing import List, Optional, Set, Tuple


# Round 69: 不同代码路径产出的 reject category 名称同义映射
# 例: Jev verdict 用 "实体一致性违反"，pipeline entity_hard_fail 用 "实体一致性硬失败" / "实体一致性硬失败(降级)"
# stagnation 检测需把这些视为同一类别，否则 ch32 实测 450+ retries 死循环（同一根因反复触发但不同 label）
_CATEGORY_SYNONYMS: dict = {
    "实体一致性硬失败": "实体一致性违反",
    "实体一致性硬失败(降级)": "实体一致性违反",
    "实体一致性违反(per-entity 聚合)": "实体一致性违反",
    "音色漂移（severe）": "音色漂移",
}

# Round 102 P1: compute_cjk_ratio 按 id(text) 缓存.
# 设计动机:
#   lit_text / raw_fallback / cand_text 等文本对象在同一次 audit+judge 周期内被反复传入
#   compute_cjk_ratio (e.g. R97 CJK_RETRY_EXHAUSTED 路径里 lit_text 被算 3 次). 同 str
#   对象 id() 稳定, 用 id() 作 key 命中率高; re.sub + ord 循环每次 ~O(N), ch32 单 chunk
#   千字级文本一次 ~0.1ms, 一章 30+ 次冗余计算累积明显.
#   key 用 id() 是 fast-path: str 本身可 hash, 但用 id() 可以命中"同一对象"语义,
#   避免不同对象但内容相同的字符串占用 cache slot (literal 调用常见).
# 内存控制:
#   - str 不支持 weakref, 所以不能用 WeakValueDictionary. 用 holder 对象持有 weakref.ref
#     监视 str 存活; cache entry 检测到 text 被 GC (ref() 返回 None) 即清理.
#   - chapter 切换 / 重置由调用方显式调用 reset_cjk_ratio_cache().
#   - 上限保护: _CJK_RATIO_CACHE_MAX 防极端 literal 调用场景下无界增长.
# Round 102 P1: id(text) 缓存.
# 设计: cache 直接持有 str 对象 (强引用), 通过 dict key=id(text) 命中.
# 同一 str 对象在 dict 内只占一份 (key + value 共享 str), 内存开销 ~= 1 个 dict entry.
# 内存管理:
#   - 长期运行中 str 被 GC 时, dict entry 会留下 stale value (str 对象已被回收).
#     id() 在 CPython 中可能被复用, 造成假命中 (不同 str 同一 id). 这是已知 trade-off:
#     CJK ratio 只是 sanity gate 的输入, 误判只会导致一次错误的 gate 决策 (不是翻译错误).
#     用户可显式调用 reset_cjk_ratio_cache() 在 chapter 切换时清空.
#   - FIFO 上限 _CJK_RATIO_CACHE_MAX=512 防止极端 literal 调用下无界增长.
_cjk_ratio_cache: "dict[int, float]" = {}
_CJK_RATIO_CACHE_MAX = 512  # 经验值: 一章几十个 chunk, 每 chunk ~10 个不同 str, 留 ~10x 余量


def _cjk_ratio_cache_get(text: str):
    """查 cache. 返回 cached float 或 None. 不做 id 碰撞检测 — 见上方注释."""
    return _cjk_ratio_cache.get(id(text))


def _cjk_ratio_cache_put(text: str, ratio: float) -> None:
    """写入 cache. 触发上限时按 FIFO 淘汰最老 entry (避免无界增长)."""
    if len(_cjk_ratio_cache) >= _CJK_RATIO_CACHE_MAX:
        # 简单 FIFO 淘汰首个 entry; 上限仅在极端 literal 调用场景触发
        try:
            oldest_key = next(iter(_cjk_ratio_cache))
            _cjk_ratio_cache.pop(oldest_key, None)
        except StopIteration:
            pass
    _cjk_ratio_cache[id(text)] = ratio


def reset_cjk_ratio_cache() -> None:
    """Chapter 切换时清空 cjk_ratio 缓存. 由 TranslationPipeline._reset_per_run_cache 调用."""
    _cjk_ratio_cache.clear()


def _normalize_category(category: str) -> str:
    """Round 69: 把同义 category 名称归一化到一个规范形式."""
    return _CATEGORY_SYNONYMS.get(category, category)


def extract_reject_categories(
    reason_text: str,
    as_set: bool = False,
    normalize: bool = True,
):
    """从 reject_reason 文本里抽出 category 列表 (按出现顺序去重).

    Args:
        reason_text: critic / judge 输出的 reject 原因字符串, 多个 category 用 `;` 分隔
        as_set:     True 返回 set (无序), False 返回 list (有序)
        normalize:  是否应用 _CATEGORY_SYNONYMS 归一化.
                    - True (默认): stagnation 检测用, 把 `音色漂移（severe）` 和
                      `音色漂移（subtle）` 视为同一类 (都属于 voice drift)
                    - False: 策略查找用, 保留具体形式以便 lookup
                      `_JEV_CATEGORY_STRATEGY` / `GLOBAL_REWRITE_CATEGORIES`
                      (这些 dicts 的 keys 用具体形式)
    Round 89: 增加 normalize 参数, 让 stagnation 检测和策略查找可独立控制.
    """
    if not reason_text:
        return set() if as_set else []
    seen: Set[str] = set()
    ordered: List[str] = []
    for seg in reason_text.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        idx = seg.find(":")
        if idx > 0:
            raw = seg[:idx].strip()
            c = _normalize_category(raw) if normalize else raw
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
    return seen if as_set else ordered


def case_insensitive_find(name: str, haystack: Optional[str]) -> Optional[str]:
    if not name or not haystack:
        return None
    if not name.isascii():
        return name if name in haystack else None
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    m = pat.search(haystack)
    return m.group(0) if m else (name if name in haystack else None)


def entity_source_name_in_text(source_name: Optional[str], source_text: Optional[str]) -> bool:
    """Round 114: source_name 是否以"词/短语边界"形式出现在原文中（而非裸子串）。

    ch30 chunk000 根因: 实体 "伯特"←"Bert" 因原文 "Bertillon measurements" 含子串
    "Bert" 被误判"实体在场" → 实体核查要求译文出现「伯特」→ 不可满足 → 实体硬失败
    → rewriter 停滞 → PERMANENTLY_FAILED。同类: "Europe"⊂"European"、
    "Herzoslovakia"⊂"Herzoslovakian"、"drawing-room"⊂"drawing-rooms"。

    用 Unicode 字母 lookaround（而非 \\b）：兼容以标点结尾的短语（"Mr."/"H. I."）；
    所有格/连字符形式（"Bert's"/"Bert—"）仍视为在场。大小写敏感（英文原文与登记表
    形式一致）。
    """
    if not source_name or not source_text:
        return False
    return re.search(_entity_boundary_pattern(source_name), source_text) is not None


def _entity_boundary_pattern(source_name: str) -> str:
    """Round 133b: entity source_name 的"词/短语边界"正则（与 Round 114 语义一致）。"""
    return r"(?<![^\W\d_])" + re.escape(source_name) + r"(?![^\W\d_])"


def entity_source_name_spans(
    source_name: Optional[str], source_text: Optional[str],
) -> List[Tuple[int, int]]:
    """Round 133b: entity_source_name_in_text 的 span 版本 — 返回全部命中 [start, end) 区间。

    供"更长别名吞并抑制"使用（chunk071 根因："Mother"(杰西卡夫人 的别名) 在原文中
    唯一出现于 "Reverend Mother"(盖乌斯·海伦·莫希阿姆 的别名) 内部 → 词边界防护
    放行 → 判杰西卡夫人在场 → 译文被判 missing → rewriter 收到不可执行指令 → 死循环）。
    语义与 entity_source_name_in_text 完全一致（大小写敏感、Unicode 字母边界）。
    """
    if not source_name or not source_text:
        return []
    return [
        (m.start(), m.end())
        for m in re.finditer(_entity_boundary_pattern(source_name), source_text)
    ]


# Round 96: CJK 字符比例计算 — final sanity 闸门的基础原语.
#
# 设计: 把字符分为 CJK / 其他, 去掉空白后求 CJK 占比.
# CJK 范围: Unified Ideographs (4E00-9FFF) + Extension A (3400-4DBF).
#   不含 Hiragana/Katakana/Hangul — 这是英语→中文翻译, 目标语言是中文.
#   假阳控制: 中文标点「」、。，！？ 算"非 CJK" — 它们属于正常中文内容的辅助符号,
#   翻译中确实有英文人名/地名/术语, 比例 0.5~0.7 是合理下限, 0.5 以下基本可视为
#   "英文主导" (如 ch02 那种 LLM 输出英文 + 中文注释的情况).
#
# 旧 BUG: ch02 final.json 是英文主导 + 中文注释, metadata 没标记 fallback,
# 下游消费者无从察觉. Round 96 用此函数 + _save_final_with_sanity 兜底.
def compute_cjk_ratio(text: str) -> float:
    """计算 CJK 字符占非空白字符的比例.

    Returns:
        float in [0, 1]. 空字符串返回 0.0.

    Round 102 P1: 按 id(text) 缓存. 同 str 对象重复调用直接命中, 跳过 re.sub + ord 循环.
    缓存由 reset_cjk_ratio_cache() 显式控制 (chapter 切换时调用).
    """
    if not text:
        return 0.0
    cached = _cjk_ratio_cache_get(text)
    if cached is not None:
        return cached
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        _cjk_ratio_cache_put(text, 0.0)
        return 0.0
    cjk_count = 0
    for ch in stripped:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF):
            cjk_count += 1
    ratio = cjk_count / len(stripped)
    _cjk_ratio_cache_put(text, ratio)
    return ratio
