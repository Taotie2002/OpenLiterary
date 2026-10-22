"""Round 105: 实体变体发现的普通词/停用词护栏。

背景（ch32 倒查根因）：
  `_maybe_discover_entity_variant` 用「一字之差 + 出现≥N次」自动把候选词登记为
  人名异体。对短规范名（黛西/卡洛/警监/伦敦塔）该启发式完全失效——任何与规范名
  共享一个字的普通词（东西/卡特/警察/伦敦金…）都会被登记，进而：
    1) 被 consistency 3c 自动修正盲目替换 → 改坏正文（某样东西→某样黛西）；
    2) 喂给 Jev per-entity Noul → 评分噪声；
    3) 触发 _check_entity_consistency 的 variant_instead_of_canonical 假阳性。

本模块提供共享护栏，供「变体发现」与「consistency 3c」复用：
  - COMMON_WORDS: 已确认的普通词/垃圾变体黑名单（含 ch32 实测污染项）。主过滤器。
  - COMMON_CHARS: 高频汉字集合。一字之差候选的「差异字」若为高频字 → 普通词。
  - is_likely_common_word / is_plausible_name_variant: 判定入口。

设计原则：
  - 音译人名常用高频字（马/克/斯/尔/维/雷…），故**不用**「整词高频字占比」判普通词，
    否则会误杀 洛马克斯/雷维尔 等真名。
  - 主过滤靠 COMMON_WORDS 黑名单（真实词典词）；一字之差场景再叠加「差异字高频」检查。
  - 宁可漏报真异体（转 soft-hint 人工复核），不可误报普通词（会改坏正文）。
"""
from __future__ import annotations

from typing import Optional, Tuple

# 高频汉字（覆盖现代汉语最高频 ~300 字）。
# 用途：一字之差候选的「差异字」落在该集合 → 判定为普通词而非人名异体。
# 人名用字通常偏冷僻（黛/洛/斯/邦/德尔等），普通词用字偏高频（东/西/警/金/来/最…）。
COMMON_CHARS: frozenset = frozenset(
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
    "尔意条状点再做外孩声西果将头把分还认问由什每没口体全机重明力理信马步听真话别打走叫见远近深浅宽窄厚薄粗细长短高低大小多少新旧好坏美丑真假对错是非黑白红黄蓝绿金银铜铁石木水火土风雨雪云雷电光明暗冷热温凉干湿软硬轻重快慢早晚今明昨昼夜晨昏朝夕春夏秋冬东南西北中央"
    "警察官方司督已去过来回到达至向往朝向对面前后左右上下里外内中间旁边侧邻远近深浅"
    "东西南北上下左右前后里外中间旁边侧近远深浅宽窄厚薄粗细长短高低大小多少新旧好坏美丑真假对错是非"
    "谁啥哪咋吗呢吧啊呀嘛啦哇哟呐哦噢嗯嘿喂哎唉喔"
    "金来最时可可高可时警金来最时"
)

# 已确认的普通词 / 垃圾变体黑名单（ch32 实测污染 + 常见误报）。
# 命中即判定为普通词，绝不作为人名异体登记或替换。
COMMON_WORDS: frozenset = frozenset({
    # ch32 实测污染项
    "东西", "卡特", "警察", "警督", "警已", "警方", "警惕", "警司", "警官",
    "材敦", "普敦", "伦次",
    "伦敦金", "伦敦来", "伦敦最", "伦敦高", "伦敦时", "伦敦警", "伦敦可",
    # 高频普通词（2 字）
    "我们", "你们", "他们", "她们", "它们", "自己", "什么", "怎么", "这样",
    "那样", "这个", "那个", "这些", "那些", "没有", "不是", "可以", "已经",
    "因为", "所以", "但是", "如果", "虽然", "然而", "于是", "然后", "接着",
    "突然", "忽然", "渐渐", "慢慢", "轻轻", "重重", "深深", "浅浅",
    "时候", "时间", "地方", "事情", "问题", "办法", "意思", "感觉",
    "声音", "光线", "空气", "天气", "日子", "年月", "世界", "国家", "人民",
    "生活", "工作", "学习", "学校", "家庭", "朋友", "老师", "学生", "孩子",
    "大人", "老人", "青年", "少年", "男人", "女人", "丈夫", "妻子", "父亲",
    "母亲", "哥哥", "姐姐", "弟弟", "妹妹", "叔叔", "阿姨", "舅舅", "姨妈",
    "警卫", "警局", "警长", "警方", "警官", "警司", "警督", "警惕",
})


def is_likely_common_word(word: str) -> bool:
    """判定一个中文词是否「已确认的普通词」（而非人名/地名异体）。

    仅依赖 COMMON_WORDS 黑名单（真实词典词）。**不用**整词高频字占比——
    音译人名常用高频字（马/克/斯/尔…），占比法会误杀 洛马克斯/雷维尔 等真名。
    """
    if not word:
        return True
    return word in COMMON_WORDS


def differing_char(canonical: str, candidate: str) -> Optional[str]:
    """返回 canonical 与 candidate 唯一不同的字符；非「恰好一字之差」返回 None。"""
    if not canonical or not candidate or len(canonical) != len(candidate):
        return None
    diff_pos = -1
    diff_count = 0
    for i, (a, b) in enumerate(zip(canonical, candidate)):
        if a != b:
            diff_count += 1
            diff_pos = i
    if diff_count != 1:
        return None
    return candidate[diff_pos]


def is_plausible_name_variant(canonical: str, candidate: str) -> Tuple[bool, str]:
    """判定 candidate 是否可能是 canonical 的人名/地名异体（一字之差场景）。

    返回 (是否可信, 原因)。可信条件（全部满足）：
      1. 恰好一字之差。
      2. candidate 不在 COMMON_WORDS 黑名单。
      3. 差异字不是高频字（高频差异字 → 普通词）。
    """
    if not canonical or not candidate:
        return False, "empty"
    if candidate == canonical:
        return False, "same_as_canonical"
    dchar = differing_char(canonical, candidate)
    if dchar is None:
        return False, "not_one_char_diff"
    if is_likely_common_word(candidate):
        return False, f"candidate_is_common_word({candidate})"
    if dchar in COMMON_CHARS:
        return False, f"differing_char_is_common({dchar})"
    return True, "ok"
