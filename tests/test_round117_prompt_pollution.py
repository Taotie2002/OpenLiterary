"""Round 117 P0/P1-4/P1-5: 跨作品 prompt 污染负样本 + Mock 路由契约。

覆盖：
  - 用中性 style_guide 构建直译/考据/润色/评论/终审 5 类 prompt，断言不含作品特定字符串
    （wild-cat / 星云奖 / 爱丽丝 / 丁尼生 / Curiouser / morganatic 等）
  - translation_hints 只有在风格剖面注入时才出现（合法动态注入）
  - Mock 路由契约: 5 类真实 prompt 都命中各自的 Mock 分支（P1-5 回归）
  - 源码中不再残留旧硬编码 prompt 片段（精确串断言）
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

from src import config as cfg_mod
from src.utils.llm_adapter import robust_json_loads
from src.translator_agent import (
    JudgeAgent,
    LiteraryRewriterAgent,
    MockLLMAdapter,
    ReferenceAgent,
    TranslationPipeline,
)

BANNED = [
    "wild-cat", "wild cat", "Kind hearts", "burning deck", "morganatic",
    "星云奖", "雨果奖", "卡罗尔", "爱丽丝", "丁尼生", "海曼斯",
    "Curiouser", "beauty is truth", "1920年代英式俚语",
]


class _StubDecisionEngine:
    def format_prompt_context(self, source_text: str = "") -> str:
        return "【有界术语表】(stub)"


def _neutral_guide() -> dict:
    return {
        "avg_sentence_length": "适中、口语化，贴近原文句式长度",
        "lexicon_preference": "平实、克制",
        "author_priority_ratio": 0.7,
        "genre": "文学小说",
        "work_type": "长篇小说",
        "persona": "忠实原著语域的职业文学译者",
        "footnote_policy": "inline",
        "register_default": "保持克制、平实、贴近原文的叙事语域",
        "forbidden_modernisms": [],
        "translation_hints": [],
    }


@pytest.fixture
def neutral_guide():
    cfg_mod.config.set_runtime_style_guide(_neutral_guide())
    yield _neutral_guide()
    cfg_mod.config.set_runtime_style_guide(None)


def _build_all_prompts() -> dict:
    stub = types.SimpleNamespace(
        decision_engine=_StubDecisionEngine(),
        _build_translation_hints_block=TranslationPipeline._build_translation_hints_block,
    )
    literal = TranslationPipeline._build_literal_prompt(stub, "He walked along the trail at dusk.")

    ref = ReferenceAgent.__new__(ReferenceAgent)
    reference = ref._build_prompt("Alice was beginning to get very tired.")

    rew = LiteraryRewriterAgent.__new__(LiteraryRewriterAgent)
    rew._few_shot_examples = ""
    rewrite = rew._build_prompt("[直译] text", "", "【风格基准】stub")

    critic = __import__("src.translator_agent", fromlist=["CriticAgent"]).CriticAgent.__new__(
        __import__("src.translator_agent", fromlist=["CriticAgent"]).CriticAgent)
    critic_prompt = critic._build_prompt("source", "raw", "lit", _neutral_guide())

    judge = JudgeAgent.__new__(JudgeAgent)
    judge_prompt = judge._build_prompt("source", "lit", {"scores": {}})

    return {
        "literal": literal,
        "reference": reference,
        "rewrite": rewrite,
        "critic": critic_prompt,
        "judge": judge_prompt,
    }


def _legacy_prompt_fragments() -> list:
    """旧版 prompt 中逐字存在的作品特定片段（精确匹配，避免误伤注释）。"""
    return [
        "- 1920年代英式俚语: 'wild-cat' = 投机的/未经证实的",
        "- 采矿行话: 'wild-cat trails' = 不靠谱的淘金线索",
        '"Kind hearts are more than coronets" → 丁尼生名句，需要注释',
        '"The boy stood on the burning deck" → 海曼斯诗，需要注释',
        "你是一位星云奖级别的终审",
    ]


def test_neutral_prompts_have_no_work_specific_pollution(neutral_guide):
    prompts = _build_all_prompts()
    for name, prompt in prompts.items():
        for banned in BANNED:
            assert banned not in prompt, f"{name} prompt 含作品特定字符串: {banned!r}"


def test_translation_hints_injected_only_from_profile(neutral_guide):
    guide = _neutral_guide()
    guide["translation_hints"] = ["foo-hint = 本书抽样中新出现的方言词（示例）"]
    cfg_mod.config.set_runtime_style_guide(guide)
    stub = types.SimpleNamespace(
        decision_engine=_StubDecisionEngine(),
        _build_translation_hints_block=TranslationPipeline._build_translation_hints_block,
    )
    literal = TranslationPipeline._build_literal_prompt(stub, "Some text.")
    assert "foo-hint = 本书抽样中新出现的方言词（示例）" in literal

    ref = ReferenceAgent.__new__(ReferenceAgent)
    reference = ref._build_prompt("Some text.")
    assert "foo-hint = 本书抽样中新出现的方言词（示例）" in reference


def test_mock_router_contract_on_real_prompts(neutral_guide):
    prompts = _build_all_prompts()
    mock = MockLLMAdapter()

    lit_out = mock.generate(prompts["literal"], model_name="mock")
    assert lit_out.startswith("[直译] ")

    ref_out = mock.generate(prompts["reference"], model_name="mock")
    ref_data = robust_json_loads(ref_out, expected_type=dict)
    assert isinstance(ref_data, dict) and "references" in ref_data

    rew_out = mock.generate(prompts["rewrite"], model_name="mock")
    assert "[Mock Response #" not in rew_out

    crit_out = mock.generate(prompts["critic"], model_name="mock")
    crit_data = robust_json_loads(crit_out, expected_type=dict)
    assert isinstance(crit_data, dict) and "scores" in crit_data

    judge_out = mock.generate(prompts["judge"], model_name="mock")
    judge_data = robust_json_loads(judge_out, expected_type=dict)
    assert isinstance(judge_data, dict) and judge_data.get("decision") in {"PASS", "REJECT"}


def test_source_files_do_not_contain_legacy_hardcoded_fragments():
    scanned = [
        ROOT / "src" / "translator_agent.py",
        ROOT / "src" / "pipeline.py",
        ROOT / "src" / "agents" / "reference_agent.py",
        ROOT / "src" / "utils" / "llm_adapter.py",
        ROOT / "src" / "agents" / "judge_agent.py",
        ROOT / "src" / "agents" / "rewriter_agent.py",
    ]
    for path in scanned:
        text = path.read_text(encoding="utf-8")
        for frag in _legacy_prompt_fragments():
            assert frag not in text, f"{path.name} 仍残留旧硬编码片段: {frag!r}"


def test_base_config_style_guide_is_neutral():
    from src.config import Config
    fresh = Config()
    base = fresh.base_style_guide
    assert base.get("genre") == "文学小说"
    assert "1920s" not in str(base.get("register_default", ""))
    assert "英式轻喜剧" not in str(base.get("persona", ""))
    assert base.get("translation_hints") == []
    # entity_variants 默认空（种子已外移）
    assert fresh.entity_variants == {}
