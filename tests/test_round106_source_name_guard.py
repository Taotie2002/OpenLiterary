"""Round 106: source_name 有效性护栏扩展回归测试。

ch05 倒查：reference agent 把非实体短语登记为 source_name（"I could"/"I should"/
"I had" 登记为女主人公、"his wife" 登记为前侯爵夫人、"the fellow" 登记为凯德等），
导致实体一致性核查误报 missing → 硬失败（chunk003），实体审计误报"漏译"。

修复：扩展 _is_valid_entity_source_name 护栏，覆盖三类无效指称：
  - 动词短语（代词主语 + 情态/助动词/常见动词）：I could / he said
  - 关系代词（his/her + 关系名词）：his wife / her husband
  - 泛称描述：the fellow / the young man / an Englishwoman
本测试锁定护栏判定，防止回归。
"""
from __future__ import annotations

import pytest

from src.pipeline import TranslationPipeline as P


class TestSourceNameGuard:
    @pytest.mark.parametrize("sn", [
        "I could", "I should", "I had", "he said", "she looked", "they were",
        "sir", "madam", "mr", "mrs",
        "his wife", "his brother", "his daughter", "her husband", "his son",
        "the fellow", "the young man", "the old man", "a gentleman",
        "the lady", "an Englishwoman", "the man", "a woman",
        "he (implied)", "she (implied)",
        "his eyelids", "his impressions",
        # Round 111: 限定词 + 全小写普通名词短语统一判无效（ch09 chunk004 "the stairs"
        # 硬失败根因）；"the financier" 从 Round 106 的 valid 列表移来此处。
        "the financier", "the stairs", "the chauffeur", "my maid",
        "the unemployed young man", "The manager",
    ])
    def test_invalid_source_names_rejected(self, sn):
        assert P._is_valid_entity_source_name(sn) is False, f"{sn!r} 应被判为无效 source_name"

    @pytest.mark.parametrize("sn", [
        "Mr. Isaacstein", "George", "the Tower", "the King", "the City",
        "the Balkans", "the Abbey", "the Hon. George Lomax",
        "the Sûreté", "the Obolovitch family", "His Majesty's Government",
        "the Bois", "the Matoppos", "the Wallace Collection",
        "George Lomax", "Mr. Lomax", "Virginia Revel",
        # Round 111: 大写专名（限定词 + 大写词）不受新规则影响
        "the Comrades", "the Countess",
    ])
    def test_valid_source_names_accepted(self, sn):
        assert P._is_valid_entity_source_name(sn) is True, f"{sn!r} 是合法指称，不应被拒"

    def test_empty_rejected(self):
        assert P._is_valid_entity_source_name("") is False
        assert P._is_valid_entity_source_name("   ") is False
