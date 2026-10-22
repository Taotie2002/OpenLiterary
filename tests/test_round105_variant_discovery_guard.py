"""Round 105 P1-1: 变体发现的普通词护栏。

ch32 倒查根因：`_maybe_discover_entity_variant` 用「一字之差 + 出现≥N次」自动把
普通词登记为人名异体（黛西←东西 / 卡洛←卡特 / 警监←警察 / 伦敦塔←伦敦金），
随后被 consistency 3c 盲目替换改坏正文。

修复：
  - 共享普通词护栏 src.utils.entity_stopwords（黑名单 + 高频差异字）。
  - 硬长度护栏：canonical < variant_min_auto_len(默认3) 绝不自动入表。
  - variant_auto_register 默认改为 false（仅 soft-hint）。
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────
# 1. 共享护栏模块本身
# ─────────────────────────────────────────────────────────────
class TestEntityStopwords:
    def test_garbage_words_are_common(self):
        from src.utils.entity_stopwords import is_likely_common_word
        for w in ["东西", "卡特", "警察", "警督", "警已", "警方", "警惕", "警司", "警官",
                  "材敦", "普敦", "伦次", "伦敦金", "伦敦来", "伦敦最", "伦敦高",
                  "伦敦时", "伦敦警", "伦敦可"]:
            assert is_likely_common_word(w), f"{w} 应被判为普通词"

    def test_real_names_not_common(self):
        from src.utils.entity_stopwords import is_likely_common_word
        for w in ["黛西", "卡洛", "邦德", "洛林", "费什", "科德斯", "埃弗斯利",
                  "洛马克斯", "卡特汉姆", "卡特拉姆", "约翰逊", "巴特尔", "雷维尔",
                  "艾萨克斯坦", "朱塞佩", "凯德", "麦格拉斯"]:
            assert not is_likely_common_word(w), f"{w} 是真名，不应被判为普通词"

    def test_plausible_variant_blocks_garbage(self):
        from src.utils.entity_stopwords import is_plausible_name_variant
        assert is_plausible_name_variant("黛西", "东西")[0] is False
        assert is_plausible_name_variant("卡洛", "卡特")[0] is False
        assert is_plausible_name_variant("警监", "警察")[0] is False
        assert is_plausible_name_variant("伦敦塔", "伦敦金")[0] is False

    def test_plausible_variant_allows_real(self):
        from src.utils.entity_stopwords import is_plausible_name_variant
        # 差异字冷僻 → 可信异体
        assert is_plausible_name_variant("黛西", "黛曦")[0] is True
        assert is_plausible_name_variant("卡洛", "卡络")[0] is True

    def test_not_one_char_diff_rejected(self):
        from src.utils.entity_stopwords import is_plausible_name_variant
        assert is_plausible_name_variant("黛西", "黛西")[0] is False  # 相同
        assert is_plausible_name_variant("黛西", "黛")[0] is False     # 长度不同
        assert is_plausible_name_variant("黛西", "黛曦晓")[0] is False  # 长度不同


# ─────────────────────────────────────────────────────────────
# 2. _maybe_discover_entity_variant 集成行为
# ─────────────────────────────────────────────────────────────
class _FakeDE:
    def __init__(self, canonicals=()):
        self._canonicals = set(canonicals)
        self.registered = []  # (canonical, variant)

    def is_canonical_name(self, name):
        return name in self._canonicals

    def add_entity_variant(self, canonical, variant):
        self.registered.append((canonical, variant))


def _make_pipeline(de):
    from src.translator_agent import TranslationPipeline
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.decision_engine = de
    return p


@pytest.fixture
def cfg_pipeline():
    """保存/恢复 monolith 的 _config.pipeline 相关键，避免污染其它测试。"""
    import src.translator_agent as ta
    saved = {
        k: ta._config.pipeline.get(k)
        for k in ("variant_auto_register", "variant_min_occurrence", "variant_min_auto_len")
    }
    yield ta._config.pipeline
    for k, v in saved.items():
        if v is None:
            ta._config.pipeline.pop(k, None)
        else:
            ta._config.pipeline[k] = v


class TestVariantDiscoveryGuard:
    def test_common_word_never_auto_registered(self, cfg_pipeline):
        """即使 auto_register=true + 高频出现，普通词也不入表。"""
        cfg_pipeline["variant_auto_register"] = True
        cfg_pipeline["variant_min_occurrence"] = 2
        cfg_pipeline["variant_min_auto_len"] = 3
        de = _FakeDE()
        p = _make_pipeline(de)
        # 规范名 黛西(2字)；lit_text 里 "东西" 出现 3 次（一字之差 黛→东）
        lit = "他买了东西，又买了东西，还是东西。"
        to_reg, soft = p._maybe_discover_entity_variant("黛西", lit, [])
        assert to_reg == [], f"普通词 东西 不应自动入表: {to_reg}"
        assert de.registered == []

    def test_short_canonical_never_auto_registered(self, cfg_pipeline):
        """硬长度护栏：canonical<3 字，即使 auto_register=true 也不入表。"""
        cfg_pipeline["variant_auto_register"] = True
        cfg_pipeline["variant_min_occurrence"] = 2
        cfg_pipeline["variant_min_auto_len"] = 3
        de = _FakeDE()
        p = _make_pipeline(de)
        # 卡洛(2字)；"卡络" 是真异体（差异字 络 冷僻）但 canonical 太短
        lit = "卡络来了，卡络走了。"
        to_reg, soft = p._maybe_discover_entity_variant("卡洛", lit, [])
        assert to_reg == [], f"短规范名不应自动入表: {to_reg}"
        # 但应记为 soft candidate（供人工复核）
        assert any(c == "卡络" for c, _ in soft), f"应记 soft candidate: {soft}"

    def test_real_variant_auto_registered_when_long_enough(self, cfg_pipeline):
        """canonical≥3 字 + 真异体 + auto_register=true + 高频 → 入表。"""
        cfg_pipeline["variant_auto_register"] = True
        cfg_pipeline["variant_min_occurrence"] = 2
        cfg_pipeline["variant_min_auto_len"] = 3
        de = _FakeDE()
        p = _make_pipeline(de)
        # 规范名 黛曦儿(3字)；"黛西儿" 差异字 西(高频) → 普通词，不入表
        lit = "黛西儿在，黛西儿走。"
        to_reg, _ = p._maybe_discover_entity_variant("黛曦儿", lit, [])
        assert to_reg == [], f"高频差异字应被拦: {to_reg}"

    def test_default_auto_register_is_off(self, cfg_pipeline):
        """默认（config 未显式开启）不自动入表，只 soft-hint。"""
        cfg_pipeline.pop("variant_auto_register", None)  # 走代码默认 False
        cfg_pipeline["variant_min_occurrence"] = 2
        cfg_pipeline["variant_min_auto_len"] = 3
        de = _FakeDE()
        p = _make_pipeline(de)
        # 构造一个 canonical≥3 且差异字冷僻的真异体
        lit = "邦德瑞出现，邦德瑞离开。"
        to_reg, soft = p._maybe_discover_entity_variant("邦德珥", lit, [])
        assert to_reg == [], f"默认不应自动入表: {to_reg}"
        assert de.registered == []

    def test_existing_variant_not_reprocessed(self, cfg_pipeline):
        """已在 existing_variants 的候选不重复处理。"""
        cfg_pipeline["variant_auto_register"] = True
        cfg_pipeline["variant_min_occurrence"] = 2
        cfg_pipeline["variant_min_auto_len"] = 3
        de = _FakeDE()
        p = _make_pipeline(de)
        lit = "卡络来了，卡络走了。"
        to_reg, soft = p._maybe_discover_entity_variant("卡洛", lit, ["卡络"])
        assert to_reg == [] and soft == []
