"""Round 58: 单元测试 - src.splitters._normalize_scores + pipeline.py 导入.

保护目标（Q3 修复）：
  - Critic LLM 拼错的维度名（voice_consistence 等）必须归一化
  - 未识别键原样保留
  - 非 dict 输入安全降级
  - pipeline.py 的 _normalize_scores 引用必须能解析（无 NameError）
"""
from __future__ import annotations

import pytest

from src.splitters import _normalize_scores


# ---------------------------------------------------------------------------
# 已知脏键归一化
# ---------------------------------------------------------------------------

class TestKnownTypoNormalization:
    def test_voice_consistence_normalized(self):
        result = _normalize_scores({"voice_consistence": 8.0})
        assert "voice_consistency" in result

    def test_voice_consistancy_normalized(self):
        result = _normalize_scores({"voice_consistancy": 7.5})
        assert "voice_consistency" in result

    def test_style_complience_normalized(self):
        result = _normalize_scores({"style_complience": 6.0})
        assert "style_compliance" in result

    def test_semantic_preserv_normalized(self):
        result = _normalize_scores({"semantic_preserv": 7.0})
        assert "semantic_preservation" in result

    def test_literal_residues_normalized(self):
        result = _normalize_scores({"literal_residues": 8.0})
        assert "literal_residue" in result

    def test_case_insensitive_lookup(self):
        # 大小写归一化（lower 化后查白名单）
        result = _normalize_scores({"VOICE_CONSISTENCE": 8.0})
        assert "voice_consistency" in result

    def test_whitespace_stripped(self):
        # 前后空白容错
        result = _normalize_scores({" voice_consistence ": 8.0})
        assert "voice_consistency" in result


# ---------------------------------------------------------------------------
# 已规范键保持
# ---------------------------------------------------------------------------

class TestCanonicalKeysPreserved:
    def test_canonical_keys_pass_through(self):
        scores = {
            "voice_consistency": 8.0,
            "style_compliance": 7.0,
            "semantic_preservation": 9.0,
            "literal_residue": 7.5,
            "fluency": 8.5,
            "readability": 8.0,
        }
        result = _normalize_scores(scores)
        for k in scores:
            assert k in result

    def test_unknown_key_preserved_as_is(self):
        # 未识别键原样保留
        result = _normalize_scores({"custom_dimension": 7.0})
        assert "custom_dimension" in result
        assert result["custom_dimension"] == 7.0


# ---------------------------------------------------------------------------
# 非数值过滤
# ---------------------------------------------------------------------------

class TestNonNumericFiltering:
    def test_string_value_dropped(self):
        # Critic 有时会把 critique 文本塞到 scores 里 — 非数值应跳过
        result = _normalize_scores({"voice_consistency": "很高"})
        assert "voice_consistency" not in result

    def test_none_value_dropped(self):
        result = _normalize_scores({"voice_consistency": None})
        assert "voice_consistency" not in result

    def test_list_value_dropped(self):
        result = _normalize_scores({"voice_consistency": [8.0, 9.0]})
        assert "voice_consistency" not in result

    def test_mixed_numeric_and_non_numeric(self):
        result = _normalize_scores({
            "voice_consistency": 8.0,
            "style_compliance": "invalid",
            "literal_residue": 7.5,
        })
        assert result == {
            "voice_consistency": 8.0,
            "literal_residue": 7.5,
        }


# ---------------------------------------------------------------------------
# 输入安全
# ---------------------------------------------------------------------------

class TestInputSafety:
    def test_none_input_returns_empty_dict(self):
        assert _normalize_scores(None) == {}

    def test_empty_dict_input_returns_empty_dict(self):
        assert _normalize_scores({}) == {}

    def test_non_dict_input_returns_empty_dict(self):
        # list 输入安全降级
        assert _normalize_scores([1, 2, 3]) == {}
        # 字符串输入安全降级
        assert _normalize_scores("not a dict") == {}


# ---------------------------------------------------------------------------
# pipeline.py 引用能解析
# ---------------------------------------------------------------------------

class TestPipelineImport:
    def test_pipeline_can_resolve_normalize_scores(self):
        # Round 58 修复：pipeline.py 之前漏 import _normalize_scores
        # 此测试保证 _normalize_scores 能从 src.pipeline 命名空间访问到
        from src.pipeline import _normalize_scores as _ns_func
        assert _ns_func is _normalize_scores

    def test_pipeline_module_does_not_have_name_error(self):
        # 模拟 _process_auditing_batch 中实际调用场景
        # 通过 import + 闭包方式验证不触发 NameError
        from src.pipeline import _normalize_scores
        critic_report = {"scores": {"voice_consistence": 8.0, "fluency": 9.0}}
        if isinstance(critic_report, dict) and "scores" in critic_report:
            critic_report["scores"] = _normalize_scores(critic_report.get("scores"))
        # 应成功执行，无 NameError
        assert "voice_consistency" in critic_report["scores"]
        assert "fluency" in critic_report["scores"]
