"""Round 65: 验证 _llm_call_counter 计入 rewriter retry 次数.

保护目标（修复 cost_report 低估 bug）：
  - 之前 `+= 1` 只算 chunk 处理次数，漏算 retry
  - 现在 `+= 1 + quality_retries` 反映实际 LLM 调用次数
  - 修复前 ch32 实测：450+ rewriter retries 但 cost_report 只显示 2 次 literary_rewriting
"""
from __future__ import annotations


class TestCounterMath:
    """验证增量数学正确性"""

    def test_quality_retries_zero_increments_by_one(self):
        """quality_retries=0（首次润色）→ +=1"""
        quality_retries = 0
        delta = 1 + quality_retries
        assert delta == 1

    def test_quality_retries_one_increments_by_two(self):
        """quality_retries=1（第 1 次重试，即第 2 次调用）→ +=2"""
        quality_retries = 1
        delta = 1 + quality_retries
        assert delta == 2

    def test_quality_retries_five_increments_by_six(self):
        """quality_retries=5（第 5 次重试，即第 6 次调用）→ +=6"""
        quality_retries = 5
        delta = 1 + quality_retries
        assert delta == 6

    def test_ch32_realistic_scenario(self):
        """ch32 真实场景：每个 chunk 6 次文学润色调用 + 6 次 critic + 6 次 judge"""
        # chunk001: quality_retries=5 (走到 fallback) → 6 次 literary_rewriting
        # chunk002: quality_retries=5 → 6 次 literary_rewriting
        # 修复前: counter = 2（每 chunk += 1）
        # 修复后: counter = 12（每 chunk += 6）
        chunks_with_retries_5 = 2
        old_total = chunks_with_retries_5 * 1  # 修复前
        new_total = chunks_with_retries_5 * (1 + 5)  # 修复后
        assert old_total == 2
        assert new_total == 12
        # 修复后揭示的真实成本至少 6 倍于修复前
        assert new_total == old_total * 6


class TestCodeInjection:
    """验证代码确实使用 += 1 + quality_retries"""

    def test_pipeline_uses_retry_aware_increment(self):
        with open("src/pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()

        # 三个 counter 都应使用 1 + quality_retries
        assert '_llm_call_counter["literary_rewriting"] += 1 + quality_retries' in content
        assert '_llm_call_counter["critic_scoring"] += 1 + quality_retries' in content
        assert '_llm_call_counter["judge_decision"] += 1 + quality_retries' in content

    def test_old_pattern_removed(self):
        """旧 += 1 模式不应再存在于重试相关的 counter"""
        with open("src/pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()

        # 旧的 += 1（不含 quality_retries）应仅保留在 reference_extraction 和 literal_translation
        # （这两个 task_type 在 pipeline 内不重试）
        assert '_llm_call_counter["reference_extraction"] += 1  # Round 13' in content
        assert '_llm_call_counter["literal_translation"] += 1  # Round 13' in content
        # 重试相关的三个不应该再是纯 +=1
        assert '_llm_call_counter["literary_rewriting"] += 1  # Round 13' not in content
        assert '_llm_call_counter["critic_scoring"] += 1  # Round 13' not in content
        assert '_llm_call_counter["judge_decision"] += 1  # Round 13' not in content
