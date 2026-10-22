"""Round 68: 回归测试 — _process_auditing_batch 必须先定义 quality_retries 再使用。

保护目标（修复 Round 65 引入的 NameError）：
  - Round 65 修改: critic_scoring += 1 + quality_retries
  - 但 quality_retries 在 line 1924 才定义（晚于 line 1893 的引用）
  - 旧测试未覆盖 _process_auditing_batch 的执行路径（仅验证模块加载）
  - Round 68: 把 quality_retries 提取提前到循环顶部，并添加测试锁定此修复
"""
from __future__ import annotations


class TestCodeStructure:
    """代码结构层面验证 quality_retries 定义在引用之前"""

    def test_quality_retries_defined_in_auditing_batch(self):
        with open("src/pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()

        # 在 _process_auditing_batch 函数体内
        func_start = content.find("def _process_auditing_batch")
        next_func = content.find("\n    def ", func_start + 1)
        func_body = content[func_start:next_func] if next_func != -1 else content[func_start:]

        # 找所有 quality_retries 出现位置
        lines = func_body.split("\n")
        first_definition_line = None
        for i, line in enumerate(lines):
            if "quality_retries = task.get" in line:
                first_definition_line = i
                break
        assert first_definition_line is not None, "应至少有一处 quality_retries = task.get(...)"

        # 检查在 first_definition_line 之前没有 quality_retries 引用
        for i, line in enumerate(lines[:first_definition_line]):
            if "quality_retries" in line and "quality_retries = task.get" not in line:
                # 注释除外
                if not line.strip().startswith("#"):
                    raise AssertionError(
                        f"质量重试引用早于定义: 第 {i} 行: {line.strip()}\n"
                        f"定义在第 {first_definition_line} 行: {lines[first_definition_line].strip()}"
                    )

    def test_critic_scoring_counter_uses_quality_retries(self):
        """Round 65: critic_scoring counter 计入 quality_retries"""
        with open("src/pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()

        func_start = content.find("def _process_auditing_batch")
        next_func = content.find("\n    def ", func_start + 1)
        func_body = content[func_start:next_func] if next_func != -1 else content[func_start:]

        # critic_scoring counter 必须使用 quality_retries
        assert '_llm_call_counter["critic_scoring"] += 1 + quality_retries' in func_body


class TestRound65RegressionCoverage:
    """Round 65 引入的三个 += 1 + quality_retries 必须都能正常执行"""

    def test_three_counter_increments_all_use_quality_retries(self):
        """三个重试相关的 counter 都使用 quality_retries"""
        with open("src/pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()

        # 三处必须都有 quality_retries
        assert '_llm_call_counter["literary_rewriting"] += 1 + quality_retries' in content
        assert '_llm_call_counter["critic_scoring"] += 1 + quality_retries' in content
        assert '_llm_call_counter["judge_decision"] += 1 + quality_retries' in content

    def test_three_definitions_all_present(self):
        """三处都先定义 quality_retries 再用"""
        with open("src/pipeline.py", "r", encoding="utf-8") as f:
            content = f.read()

        # _process_rewriting_batch: literary_rewriting
        rew_start = content.find("def _process_rewriting_batch")
        rew_end = content.find("\n    def ", rew_start + 1)
        rew_body = content[rew_start:rew_end]
        rew_lines = rew_body.split("\n")
        rew_def = next(i for i, l in enumerate(rew_lines) if "quality_retries = task.get" in l)
        rew_use = next(i for i, l in enumerate(rew_lines) if "literary_rewriting\"] += 1 + quality_retries" in l)
        assert rew_def < rew_use, f"rewriting: 定义 {rew_def} > 使用 {rew_use}"

        # _process_auditing_batch: critic_scoring
        aud_start = content.find("def _process_auditing_batch")
        aud_end = content.find("\n    def ", aud_start + 1)
        aud_body = content[aud_start:aud_end]
        aud_lines = aud_body.split("\n")
        aud_def = next(i for i, l in enumerate(aud_lines) if "quality_retries = task.get" in l)
        aud_use = next(i for i, l in enumerate(aud_lines) if "critic_scoring\"] += 1 + quality_retries" in l)
        assert aud_def < aud_use, f"auditing: 定义 {aud_def} > 使用 {aud_use}"

        # _process_judging_batch: judge_decision
        jdg_start = content.find("def _process_judging_batch")
        jdg_end = content.find("\n    def ", jdg_start + 1)
        jdg_body = content[jdg_start:jdg_end]
        jdg_lines = jdg_body.split("\n")
        jdg_def = next(i for i, l in enumerate(jdg_lines) if "quality_retries = task.get" in l)
        jdg_use = next(i for i, l in enumerate(jdg_lines) if "judge_decision\"] += 1 + quality_retries" in l)
        assert jdg_def < jdg_use, f"judging: 定义 {jdg_def} > 使用 {jdg_use}"
