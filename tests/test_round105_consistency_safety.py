"""Round 105 P1-2: 一致性自动修正安全护栏。

ch32 倒查根因：consistency 3c 用 entity_registry.variant_names 盲目子串替换，
而 variant_names 被「一字之差」误发现污染（东西/卡特/警察…），导致：
  某样东西→某样黛西 / 卡特姆勋爵→卡洛姆勋爵 / 那名警察→那名警监。

修复：
  - 3c 命中普通词黑名单的 variant 一律跳过（不报告、不替换）。
  - pipeline.consistency_auto_fix 默认 false：章节末尾仅报告、不落地修改。
"""
from __future__ import annotations

import pytest


def _fin(text, path="ch32/ch32_chunk000_final.json"):
    return {"chapter": "ch32", "file": "ch32_chunk000_final.json",
            "path": path, "text": text}


def _entities(variants_by_canonical):
    """variants_by_canonical: {canonical: [variant, ...]}"""
    out = {}
    for canon, vns in variants_by_canonical.items():
        out[canon] = {"source_names": [], "type": "person",
                      "confidence": "high", "variant_names": vns}
    return out


class TestCjkVariantCommonWordFilter:
    def test_common_word_variant_not_flagged(self):
        """普通词 variant（东西/卡特/警察）不应被 3c 报告。"""
        from src.consistency import _detect_issues
        text = "他把某样东西托在手掌上。卡特姆勋爵带着那名警察。"
        finals = [_fin(text)]
        entities = _entities({
            "黛西": ["东西"],
            "卡洛": ["卡特"],
            "警监": ["警察"],
        })
        issues = _detect_issues(finals, entities, {})
        cjk = [i for i in issues if i["issue_type"] == "cjk_variant"]
        assert cjk == [], f"普通词不应被报告为 cjk_variant: {cjk}"

    def test_real_variant_still_flagged(self):
        """真异体（非普通词）仍应被 3c 报告。"""
        from src.consistency import _detect_issues
        # 规范名 黛曦，异体 黛西（非普通词，差异字冷僻）
        text = "黛西站在门口。"
        finals = [_fin(text)]
        entities = _entities({"黛曦": ["黛西"]})
        issues = _detect_issues(finals, entities, {})
        cjk = [i for i in issues if i["issue_type"] == "cjk_variant"]
        assert any(i["variant"] == "黛西" and i["canonical"] == "黛曦" for i in cjk), \
            f"真异体应被报告: {cjk}"

    def test_is_common_word_variant_helper(self):
        from src.consistency import _is_common_word_variant
        assert _is_common_word_variant("东西") is True
        assert _is_common_word_variant("卡特") is True
        assert _is_common_word_variant("警察") is True
        assert _is_common_word_variant("黛西") is False
        assert _is_common_word_variant("洛马克斯") is False


class TestPostConsistencyAutoFixSwitch:
    """_run_post_consistency 受 consistency_auto_fix 开关控制。"""

    def _make_pipeline(self, chapter_id="ch32"):
        from src.translator_agent import TranslationPipeline
        p = TranslationPipeline.__new__(TranslationPipeline)
        p.chapter_id = chapter_id
        return p

    @staticmethod
    def _patch_consistency(monkeypatch, fake):
        """直接 patch sys.modules['src.consistency'] 上的函数。

        monolith 的 shim 会把 src.consistency 注册为假模块（__file__=None），
        字符串式 monkeypatch 解析会踩到假包路径问题，故直接对模块对象 setattr。
        _run_post_consistency 内部 `from src.consistency import ...` 在调用时
        从 sys.modules 取属性，patch 模块对象即生效。
        """
        import sys
        cons_mod = sys.modules["src.consistency"]
        monkeypatch.setattr(cons_mod, "run_consistency_check_for_chapter", fake)

    def test_default_is_report_only(self, monkeypatch):
        """默认（consistency_auto_fix 未设/false）→ dry_run=True，不修改文件。"""
        import src.translator_agent as ta
        p = self._make_pipeline()
        calls = {}

        def fake_for_chapter(chapter_id, dry_run=False, **kw):
            calls["dry_run"] = dry_run
            calls["chapter"] = chapter_id
            return 0

        self._patch_consistency(monkeypatch, fake_for_chapter)
        # 确保配置为默认（false）
        saved = ta._config.pipeline.get("consistency_auto_fix")
        ta._config.pipeline.pop("consistency_auto_fix", None)
        try:
            p._run_post_consistency(chapter_scoped=True)
        finally:
            if saved is None:
                ta._config.pipeline.pop("consistency_auto_fix", None)
            else:
                ta._config.pipeline["consistency_auto_fix"] = saved
        assert calls.get("dry_run") is True, f"默认应 dry_run=True: {calls}"

    def test_auto_fix_true_applies(self, monkeypatch):
        """consistency_auto_fix=true → dry_run=False，落地修改。"""
        import src.translator_agent as ta
        p = self._make_pipeline()
        calls = {}

        def fake_for_chapter(chapter_id, dry_run=False, **kw):
            calls["dry_run"] = dry_run
            return 0

        self._patch_consistency(monkeypatch, fake_for_chapter)
        saved = ta._config.pipeline.get("consistency_auto_fix")
        ta._config.pipeline["consistency_auto_fix"] = True
        try:
            p._run_post_consistency(chapter_scoped=True)
        finally:
            if saved is None:
                ta._config.pipeline.pop("consistency_auto_fix", None)
            else:
                ta._config.pipeline["consistency_auto_fix"] = saved
        assert calls.get("dry_run") is False, f"auto_fix=true 应 dry_run=False: {calls}"
