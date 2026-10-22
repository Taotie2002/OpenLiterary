"""Round 96: final.json CJK sanity 闸门 — final 输出保底是中文译文.

设计动机:
  ch02 / ch32 实测中, rewriter LLM 偶发"输出英文+中文注释"而不是中文译文
  (典型 prompt misunderstanding 或模型选择错误). 旧代码直接保存 LLM 输出,
  下游消费者无从察觉. Round 96 在写入前检查 CJK 比例, 不达标时:
    1. 优先 fallback 到 literary/raw intermediate
    2. 全部失败 → 标记 sanity_failed=true + sanity_inspected 列出所有 step

测试覆盖:
  - compute_cjk_ratio 单元测试
  - _save_final_with_sanity 三种场景:
    * primary 通过 → 直接保存 + 仅追加 cjk_ratio
    * primary 未通过, 有 fallback 合规版本 → 选最优 fallback
    * primary + 全部 fallback 都未通过 → sanity_failed=true
  - threshold=0 关闭闸门 (向后兼容)
  - data 为字符串的场景
  - 整合脚本 src.translator_agent.py 同步
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# compute_cjk_ratio 单元测试
# ---------------------------------------------------------------------------

class TestComputeCjkRatio:
    def test_pure_chinese(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("你好世界这是一段中文文本") == pytest.approx(1.0, abs=1e-6)

    def test_pure_english(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("Hello world this is English") == pytest.approx(0.0, abs=1e-6)

    def test_empty_string(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("") == 0.0

    def test_whitespace_only(self):
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("   \n\t  ") == 0.0

    def test_mixed_2_of_5(self):
        # 5 个非空白字符, 2 个 CJK → 0.4
        from src.utils.text_match import compute_cjk_ratio
        assert compute_cjk_ratio("你好abc") == pytest.approx(0.4, abs=1e-6)

    def test_whitespace_ignored(self):
        # 空白字符不计入分母
        from src.utils.text_match import compute_cjk_ratio
        # "你好 abc" 去空白 = "你好abc" 5 字符, 2 CJK = 0.4
        assert compute_cjk_ratio("你好 abc") == pytest.approx(0.4, abs=1e-6)

    def test_chinese_with_punctuation(self):
        # 中文 + 中文标点
        from src.utils.text_match import compute_cjk_ratio
        # 字符: 你/好/世/界/，/。/中/文 → 8 字符, 4 CJK = 0.5
        assert compute_cjk_ratio("你好世界，中文。") == pytest.approx(0.75, abs=1e-6)

    def test_ext_a_chars(self):
        # CJK Extension A 范围 (3400-4DBF)
        from src.utils.text_match import compute_cjk_ratio
        # "㐀" (U+3400) 是 CJK Ext A 第一个字
        assert compute_cjk_ratio("㐀") == pytest.approx(1.0, abs=1e-6)

    def test_english_with_chinese_annotation(self):
        # ch02 实际场景: 英文 + 简短中文注释
        from src.utils.text_match import compute_cjk_ratio
        text = "Castle tour. [译注: 城堡之旅]"
        ratio = compute_cjk_ratio(text)
        # 字符: Castle/tour./[译注:/城堡之旅] - 去空白, 中文部分约 6/20 ≈ 0.30
        assert 0.2 < ratio < 0.4


# ---------------------------------------------------------------------------
# _save_final_with_sanity 集成测试 — 模拟 TranslationPipeline 行为
# ---------------------------------------------------------------------------

class TestSaveFinalWithSanity:
    """_save_final_with_sanity 行为验证 — 通过隔离测试避免污染全局状态."""

    def _make_pipeline_stub(self, tmp_path, cjk_threshold=0.5):
        """构造一个 TranslationPipeline 桩, 配置 output_dir 到 tmp_path."""
        # 用 MagicMock 模拟避免真实 init
        pipeline = MagicMock()
        pipeline.output_dir = tmp_path
        pipeline._save_intermediate = self._make_save_intermediate(tmp_path)
        pipeline._load_intermediate = self._make_load_intermediate(tmp_path)
        # 提取真实方法绑定 (绕过 MagicMock 自动拦截)
        from src.pipeline import TranslationPipeline as RealTP
        # 直接绑定真实方法到 stub 实例
        bound_method = RealTP._save_final_with_sanity.__get__(pipeline, MagicMock)
        return pipeline, bound_method

    def _make_save_intermediate(self, base_dir):
        def _save_intermediate(chunk_id, step, data):
            file_path = base_dir / f"{chunk_id}_{step}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                if isinstance(data, str):
                    json.dump({"text": data}, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        return _save_intermediate

    def _make_load_intermediate(self, base_dir):
        def _load_intermediate(chunk_id, step):
            file_path = base_dir / f"{chunk_id}_{step}.json"
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            return data
        return _load_intermediate

    def test_primary_passes_threshold(self, tmp_path, monkeypatch):
        """primary CJK >= threshold → 直接保存, 仅追加 cjk_ratio."""
        import src.config
        _cfg = src.config.config

        # 设置 threshold = 0.5
        monkeypatch.setitem(_cfg.pipeline, "final_cjk_threshold", 0.5)

        pipeline, bound = self._make_pipeline_stub(tmp_path)
        primary_text = "你好世界这是一段中文文本" * 5  # 纯中文

        result = bound("ch01_chunk000", primary_text, source_step="literary")

        # primary 应被保存
        saved = json.loads((tmp_path / "ch01_chunk000_final.json").read_text())
        assert saved["text"] == primary_text
        # cjk_ratio 应被追加到 metadata
        assert "metadata" in saved
        assert saved["metadata"]["cjk_ratio"] == pytest.approx(1.0, abs=1e-3)
        # 不应有 sanity_fallback 或 sanity_failed
        assert "sanity_fallback" not in saved["metadata"]
        assert "sanity_failed" not in saved["metadata"]

    def test_primary_dict_passes(self, tmp_path, monkeypatch):
        """primary 是 dict 形式, CJK >= threshold → 直接保存."""
        import src.config
        _cfg = src.config.config

        monkeypatch.setitem(_cfg.pipeline, "final_cjk_threshold", 0.5)

        pipeline, bound = self._make_pipeline_stub(tmp_path)
        primary = {
            "text": "你好世界这是一段中文文本" * 5,
            "metadata": {"existing_field": "preserve_me"},
        }

        result = bound("ch01_chunk000", primary, source_step="literary")

        saved = json.loads((tmp_path / "ch01_chunk000_final.json").read_text())
        assert saved["text"] == primary["text"]
        # 原有 metadata 保留 + cjk_ratio 追加
        assert saved["metadata"]["existing_field"] == "preserve_me"
        assert "cjk_ratio" in saved["metadata"]

    def test_primary_fails_fallback_to_literary(self, tmp_path, monkeypatch):
        """primary 英文主导 → fallback 到 literary (中文)."""
        import src.config
        _cfg = src.config.config

        monkeypatch.setitem(_cfg.pipeline, "final_cjk_threshold", 0.5)

        # 预先放一个 literary (合规中文)
        lit_text = "文学润色稿：乔治·洛马克斯先生回到了伦敦。" * 3
        (tmp_path / "ch01_chunk000_literary.json").write_text(
            json.dumps({"text": lit_text}, ensure_ascii=False)
        )

        pipeline, bound = self._make_pipeline_stub(tmp_path)
        # primary 是英文主导 (模拟 ch02 bug)
        primary = "Castle tour. [译注: 城堡之旅]" * 5

        result = bound("ch01_chunk000", primary, source_step="best_lit")  # 非 inspect_steps 成员, 不会跳过 literary

        saved = json.loads((tmp_path / "ch01_chunk000_final.json").read_text())
        # 应 fallback 到 literary
        assert saved["text"] == lit_text
        assert saved["metadata"]["sanity_fallback"] is True
        assert saved["metadata"]["sanity_source_step"] == "literary"
        assert saved["metadata"]["sanity_original_cjk"] < 0.5
        assert saved["metadata"]["sanity_threshold"] == 0.5

    def test_primary_fails_fallback_to_raw(self, tmp_path, monkeypatch):
        """primary 失败, literary 也失败 → fallback 到 raw."""
        import src.config
        _cfg = src.config.config

        monkeypatch.setitem(_cfg.pipeline, "final_cjk_threshold", 0.5)

        # 文学稿也是英文 (ch02 真实场景)
        (tmp_path / "ch01_chunk000_literary.json").write_text(
            json.dumps({"text": "Castle tour. English text."}, ensure_ascii=False)
        )
        # raw 是合规中文
        raw_text = "直译稿：城堡之旅"
        (tmp_path / "ch01_chunk000_raw.json").write_text(
            json.dumps({"text": raw_text}, ensure_ascii=False)
        )

        pipeline, bound = self._make_pipeline_stub(tmp_path)
        primary = "Some English output."

        result = bound("ch01_chunk000", primary, source_step="best_lit")

        saved = json.loads((tmp_path / "ch01_chunk000_final.json").read_text())
        # 应 fallback 到 raw (literary 也被检查但不合规)
        assert saved["text"] == raw_text
        assert saved["metadata"]["sanity_fallback"] is True
        assert saved["metadata"]["sanity_source_step"] == "raw"

    def test_all_fail_marks_sanity_failed(self, tmp_path, monkeypatch):
        """primary + literary + raw 都未通过 → 仍保存 primary + sanity_failed=true."""
        import src.config
        _cfg = src.config.config

        monkeypatch.setitem(_cfg.pipeline, "final_cjk_threshold", 0.5)

        # literary 和 raw 都是英文
        (tmp_path / "ch01_chunk000_literary.json").write_text(
            json.dumps({"text": "English literary text only."}, ensure_ascii=False)
        )
        (tmp_path / "ch01_chunk000_raw.json").write_text(
            json.dumps({"text": "English raw translation."}, ensure_ascii=False)
        )

        pipeline, bound = self._make_pipeline_stub(tmp_path)
        primary = "Primary English only."

        result = bound("ch01_chunk000", primary, source_step="best_lit")

        saved = json.loads((tmp_path / "ch01_chunk000_final.json").read_text())
        # primary 被保留 (不丢数据)
        assert saved["text"] == primary
        # sanity_failed 标记 + inspected 列表
        assert saved["metadata"]["sanity_failed"] is True
        # Round 107 P6: 短文本 (<100 字符) CJK 阈值放宽到 0.30
        assert saved["metadata"]["sanity_threshold"] == 0.3
        inspected = saved["metadata"]["sanity_inspected"]
        steps = [x["step"] for x in inspected]
        assert "literary" in steps
        assert "raw" in steps
        # 所有 inspected 的 cjk 都 < 0.5
        for item in inspected:
            assert item["cjk"] < 0.5

    def test_threshold_zero_disables_gate(self, tmp_path, monkeypatch):
        """threshold=0 关闭闸门 → 走原 _save_intermediate."""
        import src.config
        _cfg = src.config.config

        monkeypatch.setitem(_cfg.pipeline, "final_cjk_threshold", 0)

        pipeline, bound = self._make_pipeline_stub(tmp_path)
        primary = "Pure English text without any Chinese."

        result = bound("ch01_chunk000", primary, source_step="best_lit")

        saved = json.loads((tmp_path / "ch01_chunk000_final.json").read_text())
        # 关闭闸门时, 应直接保存 primary (即使 CJK=0)
        assert saved["text"] == primary
        # 不应有 sanity_failed 标记
        assert "sanity_failed" not in saved
        # cjk_ratio 仍可能存在 (metadata 不变)
        # 但不应有 sanity_* 字段


# ---------------------------------------------------------------------------
# 整合脚本一致性测试 (Round 96: 散装 + 整合必须同步)
# ---------------------------------------------------------------------------

class TestIntegratedScriptSync:
    """验证 src/translator_agent.py 与 src/pipeline.py 同步."""

    def test_helper_exists_in_both(self):
        """_save_final_with_sanity 在两个文件中都存在."""
        import src.pipeline as p
        import src.translator_agent as t
        assert hasattr(p.TranslationPipeline, "_save_final_with_sanity")
        assert hasattr(t.TranslationPipeline, "_save_final_with_sanity")

    def test_helper_signature_match(self):
        """两个文件的 _save_final_with_sanity 签名一致."""
        import inspect
        import src.pipeline as p
        import src.translator_agent as t

        sig_p = inspect.signature(p.TranslationPipeline._save_final_with_sanity)
        sig_t = inspect.signature(t.TranslationPipeline._save_final_with_sanity)
        # 参数列表应一致 (允许 source_step 默认值差异, 但参数名应一致)
        params_p = list(sig_p.parameters.keys())
        params_t = list(sig_t.parameters.keys())
        assert params_p == params_t, f"signature mismatch: {params_p} vs {params_t}"

    def test_cjk_ratio_available_in_both(self):
        """compute_cjk_ratio / cjk_ratio 在两个文件都可用."""
        # R-终验收修复: 用模块对象直取, 避免 r103/r105 组合测试后 sys.modules
        # 父包属性失配（真实 src.utils.text_match 已缓存, 但 src.utils 上未挂载
        # text_match 属性）造成的顺序依赖假失败; 断言意图（模块与整合脚本均
        # 可调用 compute_cjk_ratio）保持不变.
        import importlib
        tm = importlib.import_module("src.utils.text_match")
        assert hasattr(tm, "compute_cjk_ratio")
        assert callable(tm.compute_cjk_ratio)

        # 整合脚本有自己的内联版本 (避免虚拟 src.utils.text_match 包)
        import src.translator_agent as t
        assert hasattr(t, "compute_cjk_ratio")
        assert callable(t.compute_cjk_ratio)

    def test_final_call_sites_migrated_in_integrated(self):
        """整合脚本中 final.json 写入都走闸门 (除 _save_final_with_sanity 内部递归外)."""
        import src.translator_agent as t

        # 读源码, 数 _save_intermediate(chunk_id, "final", ...) 直接调用
        import inspect
        source = inspect.getsource(t.TranslationPipeline)
        # 数 _save_final_with_sanity 调用 (含闸门方法自己的 4 个内部调用)
        sanity_calls = source.count("self._save_final_with_sanity(")
        # 数 _save_intermediate(chunk_id, "final", ...) 直接调用 (应该只在 _save_final_with_sanity 内部 4 处)
        direct_calls = source.count('self._save_intermediate(chunk_id, "final", ')
        # direct_calls 应 == 4 (闸门方法内部)
        assert direct_calls == 4, f"expected 4 internal direct calls, got {direct_calls}"
        # sanity_calls 应 >= 4 (闸门内 0 + 外部迁移 9 处) — 实际是 1 def + 9 calls + 内部 0 = 10
        # 但 def 也算一个, 所以是 10
        assert sanity_calls >= 9, f"expected >=9 _save_final_with_sanity references, got {sanity_calls}"


# ---------------------------------------------------------------------------
# Ch02 / Ch32 真实场景回归测试
# ---------------------------------------------------------------------------

class TestCh02RegressionScenario:
    """复现 ch02 真实 bug: LLM 输出英文+中文注释, final.json 没标记."""

    def test_ch02_scenario_marks_sanity_failed(self, tmp_path, monkeypatch):
        """ch02 bug 场景: primary + literary + raw 都英文 → sanity_failed=true."""
        import src.config
        _cfg = src.config.config

        monkeypatch.setitem(_cfg.pipeline, "final_cjk_threshold", 0.5)

        # ch02 实际场景: literary.json 包含英文 + 中文注释
        ch02_literary = '''# 1Anthony Cade Signs on

1 Anthony Cade Signs on

"Gentleman Joe!"

"Why, if it isn't old Jimmy McGrath."

Castle's Select Tour, represented by seven depressed-looking
females and three perspiring males...

【上一版译文（在此基础上精修，只动被拒原因相关段落）】
译文经过文学润色，呈现史诗感与古典韵律。

[^1]: 译注：济慈（1795-1821），英国浪漫主义诗人。'''

        (tmp_path / "ch02_chunk000_literary.json").write_text(
            json.dumps({"text": ch02_literary}, ensure_ascii=False)
        )
        (tmp_path / "ch02_chunk000_raw.json").write_text(
            json.dumps({"text": ch02_literary}, ensure_ascii=False)  # raw 也是英文
        )

        # 直接模拟 _save_final_with_sanity 调用 (不通过 MagicMock)
        import re
        from src.pipeline import TranslationPipeline
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.output_dir = tmp_path
        # 替换为本地版本 (隔离测试)
        pipeline._save_intermediate = lambda cid, step, data: (
            (tmp_path / f"{cid}_{step}.json").write_text(
                json.dumps({"text": data} if isinstance(data, str) else data,
                          ensure_ascii=False, indent=2)
            )
        )
        pipeline._load_intermediate = lambda cid, step: json.loads(
            (tmp_path / f"{cid}_{step}.json").read_text()
        ).get("text", "")

        result = pipeline._save_final_with_sanity(
            "ch02_chunk000", ch02_literary, source_step="literary"
        )

        # ch02 场景下 literary 是英文主导, raw 也是, 应被标 sanity_failed
        saved = json.loads((tmp_path / "ch02_chunk000_final.json").read_text())
        # ch02 文学稿的实际 CJK 比例: 含中文注释但主体英文 → < 0.5
        cjk = sum(1 for ch in ch02_literary if 0x4E00 <= ord(ch) <= 0x9FFF) / \
              len(re.sub(r"\s+", "", ch02_literary))
        if cjk < 0.5:
            assert saved["metadata"].get("sanity_failed") is True
            assert saved["metadata"]["sanity_inspected"][0]["step"] == "literary"
