"""Round 121 L5: 章级注释统一（去重 / 孤儿改锚 / 顺编号）+ lint 守卫。"""
from __future__ import annotations

import re

from src.book_lint import (
    _chapter_note_dupes,
    _note_anchor_stop,
    _note_is_dup,
    _note_keys_compatible,
    _note_merge_key,
    lint_book,
    normalize_chapter_notes,
    normalize_notes_document,
)

HATTER = ("帽匠（Hatter）：维多利亚时代制帽匠用硝酸汞处理毛毡，长期接触导致慢性汞中毒，"
          "症状包括神经紊乱、口齿不清，故帽匠总是语无伦次、行为怪异。")
HARE = ("三月兔（March Hare）：英语民间传说中，三月正值野兔发情期，常被描述为癫狂如中邪。"
        "Carroll 据此典故塑造此角色。")
DORMOUSE = ("睡鼠（Dormouse）：榛睡鼠，一种欧洲特有的小型啮齿动物，以冬眠嗜睡著称。"
            "Carroll 借此习性塑造了永远昏睡的角色。")
WHITING_A = ("**牙鳕**（whiting）：英文一词双关——既指一种白色小鱼（牙鳕／银鳕），又指擦白皮鞋用的白垩粉。"
             "Carroll 在本章 Gryphon 与 Mock Turtle 的后续对话中借 “It does the boots and shoes” "
             "将两层意思打连环，构成全章核心笑点。")
WHITING_B = ("whiting：一词双关，既指“牙鳕”（一种海鱼），又指“擦白鞋用的白垩粉”；"
             "与 blacking 共同构成“擦鞋用品”的文字游戏。")
WHITING_C = ("**whiting（牙鳕）**：一词双关——既指一种银白色的食用海鱼（牙鳕），又指擦白鞋用的白垩粉"
             "（whiting chalk）。本章后半段中鹰头狮将此双关打连环，是全章关键笑点所在。")


def _refs(text):
    return re.findall(r'\[\^([^\]]+)\](?!:)', text)


def test_frequency_derived_stopword_no_false_merge():
    """Round 130 审计④: 停用锚点改为章内频率派生（零作品先验）。

    Carroll 在两注中共现（2/2 ≥ 60%）→ 不构成同源证据 → 不得误合并。
    """
    stop = _note_anchor_stop([HARE, DORMOUSE])
    assert "carroll" in stop
    assert not _note_is_dup(HARE, DORMOUSE, stop=stop)


def test_whiting_cluster_merge_and_remap():
    body = (
        "“我真喜欢那首关于牙鳕[^ch11_c3_1]的怪歌！”爱丽丝说。\n\n"
        "> **译者注：** " + WHITING_C + "\n\n"
        "“海里的靴子和鞋子，是用一只牙鳕擦的。[^ch11_c5_2]现在你知道了。”\n\n"
        "[^ch11_c3_1]: " + WHITING_A + "\n\n"
        "[^ch11_c5_2]: " + WHITING_B + "\n"
    )
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["merged_dups"] == 2
    assert new.count("[^1]:") == 1
    assert len(_refs(new)) == 2
    assert "^ch11" not in new


def test_orphan_attach_by_term():
    body = (
        "“小鳄鱼，何以\n把闪亮的尾巴改进！”\n\n"
        "“那真是险些脱身！”爱丽丝说，然后游了很远很远的一段距离才停下来。\n\n"
        "> **译者注：** “小鳄鱼”一诗系卡罗尔戏仿维多利亚时代儿童童诗《蜘蛛与苍蝇》之作，"
        "是全书著名的戏仿段落说明。\n"
    )
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["orphans_attached"] == 1
    assert "改进！”[^1]" in new
    assert "[^1]: “小鳄鱼”一诗系" in new


def test_orphan_relocate_without_term():
    body = ("正文一段，说些别的事情，篇幅足够以避免误判为注释段落本身。\n\n"
            "[^zz]: 一条找不到锚点的考据注释，讲的是某个只在此处出现的罕见典故名词。\n")
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["md_relocated"] == 1
    assert "> **译者注：** 一条找不到锚点的考据注释" in new
    assert "[^zz]" not in new


def test_orphan_relocation_reaches_duplicate_fixed_point():
    para = ("**克里姆克尔纤维/克里姆克尔绳**：用厄卡兹胡富藤的藤条编成的“爪纤维”。"
            "用克里姆克尔纤维打成的结，在结线被拉紧时，会像爪子一样越收越紧，直到预先设定的极限。"
            "（更详细的研究，见霍尔扬斯·冯布鲁克的《厄卡兹的绞杀藤》。）[^ch07_c155_5]")
    definition = ("[^ch07_c155_5]: 胡富藤是克里姆克尔纤维的原料；厄卡兹是这种藤蔓的原产地。"
                  "《厄卡兹的绞杀藤》是霍尔扬斯·冯布鲁克所著的研究。")
    body = ("正文讲述沙漠中的行旅与部落生活，篇幅足够长以避免被误判成注释段落本身。\n\n"
            + para + "\n\n---\n\n" + definition + "\n")
    assert _chapter_note_dupes(body) == []

    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["dangling_refs_removed"] == 1
    assert ch["md_relocated"] == 1
    # 两条注虽然共享材料/产地信息，但显式词条分别是“克里姆克尔纤维”
    # 与“胡富藤”，不能在缺少双语别名证据时合并；保留两条完整注释，
    # 同时确保重复检测与归一化使用同一兼容性口径。
    assert ch["merged_dups"] == 0
    assert _chapter_note_dupes(new) == []
    assert normalize_chapter_notes(new, 1)[0] == new


def test_distinct_headwords_are_not_merged_by_shared_context():
    body = (
        "正文提到保罗[^a]与圣母[^b]，叙事内容足够长以便通过注释扫描。\n\n"
        "[^a]: “保罗-穆阿迪布”是保罗·厄崔迪的姓名与宗教称号的合称；"
        "“圣母”是贝尼·杰瑟里特姐妹会高阶成员的尊称。\n\n"
        "[^b]: **圣母**：贝尼·杰瑟里特姐妹会高阶女性成员的敬称；此处指杰西卡。\n"
    )
    new, changes, _ = normalize_chapter_notes(body, 1)

    assert changes["merged_dups"] == 0
    assert "保罗-穆阿迪布" in new
    assert "**圣母**" in new


def test_chinese_concept_bridge_cannot_merge_distinct_notes():
    sisters = ("“贝尼·杰瑟里特姐妹会”是由女性组成的神秘修会，成员彼此以姐妹相称；"
               "“尊母”是其中的高阶成员称号。")
    reverend_mother = ("**尊母**：贝尼·杰瑟里特姐妹会中担任领袖与导师的女性，"
                       "其训练涵盖预知术。")
    assert not _note_keys_compatible(
        sisters, reverend_mother,
        _note_merge_key(sisters), _note_merge_key(reverend_mother))
    assert _note_keys_compatible(
        WHITING_A, WHITING_B,
        _note_merge_key(WHITING_A), _note_merge_key(WHITING_B))


def test_cluster_bridge_cannot_collapse_a_third_headword():
    first = (
        "**甲一（alpha）**：甲一（alpha）是用于测试的一组术语；本条注释说明二者的关系、"
        "用法和区别，并补充足够长的解释，以便与另外两条注释形成内容相似但身份不同的"
        "测试样例；这一条刻意写得更长，以保证聚类时保留不含后续桥接词条的版本。")
    bridge = (
        "**alpha（甲一）**：alpha（甲一）是用于测试的一组术语；本条还解释丙三（alpha）"
        "这组术语的关系、用法和区别，并补充足够长的解释，以便形成相似样例。")
    third = (
        "**丙三（alpha）**：丙三（alpha）是用于测试的一组术语；本条注释说明二者的关系、"
        "用法和区别，并补充足够长的解释，以便与另外两条注释形成内容相似但身份不同的测试样例。")
    body = (
        "正文提到甲一[^a]、alpha[^b] 与丙三[^c]，这段叙事足够长以通过注释扫描。\n\n"
        f"[^a]: {first}\n\n[^b]: {bridge}\n\n[^c]: {third}\n"
    )

    new, changes, _ = normalize_chapter_notes(body, 1)

    assert changes["merged_dups"] == 1
    assert new.count("[^1]") == 3
    assert new.count("[^2]") == 2
    assert "**丙三（alpha）**" in new
    assert normalize_chapter_notes(new, 1)[0] == new


def test_title_block_moved_to_chapter_end():
    body = (
        "第一段正文，足够长以构成叙事。\n\n"
        "**【章末注释】**\n\n"
        "（1）“Curiouser and curiouser!”：卡罗尔自造双重比较级以示荒诞之用的著名例句说明。\n\n"
        "（2）“Esq.”：Esquire 之缩写，维多利亚时代置于姓名之后的尊称，可视作“先生”的注释说明。\n\n"
        "第二段正文，接续叙事，应当保持在注释块搬移之后的位置不变。\n"
    )
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["title_blocks_moved"] == 1
    assert new.index("**【章末注释】**") > new.index("第二段正文")


def test_note_title_shell_dropped():
    body = ("正文第一段，足够长以便通过扫描。\n\n---\n\n**【注释】**\n\n\n\n"
            "正文第二段继续前行。\n\n---\n\n**【译注】**\n\n"
            "此段乃对原文的说明文字，标题存在即合理，不应被清理。\n")
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["title_shells_dropped"] == 1
    assert "**【注释】**" not in new
    assert "**【译注】**" in new
    assert new == normalize_chapter_notes(new, 1)[0]


def test_machine_ids_renumber_and_idempotent():
    body = (
        "爱丽丝[^ch01_c0_1]看着帽匠[^ch01_c1_2]，又看看三月兔[^ch01_c0_1]。\n\n"
        "[^ch01_c0_1]: 第一条注释内容足够长放在这里用于测试编号迁移处理。\n\n"
        "[^ch01_c1_2]: 第二条注释内容同样需要足够长度以避免短注豁免规则。\n"
    )
    new, ch, nxt = normalize_chapter_notes(body, 1)
    assert ch["machine_ids"] == 5
    assert new.count("[^1]:") == 1 and new.count("[^2]:") == 1
    assert len(_refs(new)) == 3
    again, _ch2, _n2 = normalize_chapter_notes(new, 1)
    assert again == new
    assert nxt == 3


def test_cross_ref_rewritten_inside_list():
    body = (
        "帽匠的神情十分古怪，发疯的传闻众人皆知，这段正文足够长以通过检查。\n\n"
        "---\n\n[1] " + HATTER + "\n\n[2] “发疯”（went mad）：见 [1]，指帽匠因汞中毒而"
        "精神失常，原文暗示其疯病并非突发的说明文字。\n"
    )
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["orphans_attached"] == 2
    assert "见 [^" in new
    assert "见 [1]" not in new


def test_multiline_tr_split_independent():
    body = (
        "“再唱副歌！”鹰头狮喊道，声音洪亮，这段正文足够长以便识别词条锚点。\n\n"
        "> **译者注：** **再唱副歌！** 原文 “Chorus again!” —— 副歌本是合唱歌曲中反复出现的段落说明。\n"
        "> **译者注：** **甲鱼汤** 原文 “Turtle Soup” —— 维多利亚时代伦敦高档餐馆的招牌名贵汤品说明。\n"
    )
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["orphans_attached"] == 1
    def_lines = [l for l in new.split("\n") if l.startswith("[^")]
    assert len(def_lines) == 1 and "再唱副歌" in def_lines[0]
    assert "甲鱼汤" not in def_lines[0]
    assert new.count("> **译者注：**") == 1


def test_document_global_numbering():
    doc = (
        "# 第一章 A\n\n正文甲[^c0_1]足够长的叙事内容。\n\n"
        "[^c0_1]: 第一章的注释内容，长度足够用于编号连续性测试说明。\n\n"
        "# 第二章 B\n\n正文乙[^c1_1]足够长的叙事内容。\n\n"
        "[^c1_1]: 第二章的注释内容，长度足够用于编号连续性测试说明。\n"
    )
    new, rep = normalize_notes_document(doc)
    assert "[^1]:" in new and "[^2]:" in new
    assert rep["stats"]["renumbered"] == 2


def test_lint_guards_machine_id_and_dup(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "translated_full.md").write_text(
        "# 第一章 测试（Test）\n\n正文一段[^ch01_c0_1]足够长的叙事内容。\n\n"
        "[^ch01_c0_1]: " + WHITING_B + "\n\n"
        "> **译者注：** " + WHITING_B + "\n",
        encoding="utf-8",
    )
    res = lint_book(project_root=tmp_path)
    assert not res["ok"]
    joined = "\n".join(res["failures"])
    assert "机器脚注 id 残留" in joined
    assert "重复注" in joined


def test_chapter_note_dupes_detects_whiting_pair():
    txt = ("[^a]: " + WHITING_B + "\n\n> **译者注：** " + WHITING_A + "\n")
    dups = _chapter_note_dupes(txt)
    assert dups, "whiting 重复组未被检出"


def test_defs_centralized_to_chapter_end():
    body = ("正文甲[^c1]与正文乙[^c2]，都足够长的叙事内容用于测试。\n\n"
            "[^c1]: 第一条注释内容足够长用于集中测试的说明文字甲。\n\n"
            "[^c2]: 第二条注释内容足够长用于集中测试的说明文字乙。\n")
    new, _ch, _ = normalize_chapter_notes(body, 1)
    lines = new.rstrip("\n").split("\n")
    assert lines[-1].startswith("[^2]:") and lines[-2].startswith("[^1]:")
    assert new.index("[^1]:") > new.index("正文乙")


def test_end_note_items_best_effort_attach():
    body = ("爱丽丝的右脚　先生，\n炉边小毯，\n近火栏处。\n\n"
            "**【章末注释】**\n\n"
            "（1）“Hearthrug, near the Fender”：维多利亚时期英国家庭壁炉前的固定陈设"
            "（hearthrug 即炉边小毯，用来护足防溅的说明）。\n\n"
            "末段正文继续前行，足够长以便通过扫描。\n")
    new, ch, _ = normalize_chapter_notes(body, 1)
    assert ch["orphans_attached"] == 1
    assert "近火栏处。[^1]" in new
    assert "**【章末注释】**" not in new
    assert "[^1]: " in new
    assert new == normalize_chapter_notes(new, 1)[0]
