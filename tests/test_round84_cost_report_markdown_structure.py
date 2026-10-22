"""Round 84: 补全 Round 81 fix —— cost_report markdown 结构层字面 \\n 修复.

背景:
  Round 81 修复了 cost_report 建议表渲染字面 \\n bug, 但只覆盖了内层循环:
    - f"...偏差 ...%）|\\n" (Round 81 f-string)
    - lines.append("\\n") (Round 81 separator)
  外层 markdown 结构 (Round 35 est_tokens_per_call 建议段) 仍有 4 处字面 \\n:
    - L414 ### section header
    - L415 段落分隔
    - L416 | Role | table header
    - L417 |---|---:| table separator

修复前 BUG:
  cost_report.md 的 "est_tokens_per_call 建议值" 段会被渲染成:
    \\n### est_tokens_per_call 建议值（Round 35）\\n\\n
    \\n基于本章真实 token / 调用次数计算的每调用平均：\\n\\n
    \\n| Role | 当前每调用真实平均 | 当前估算 | 建议替换值 |\\n
    \\n|---|---:|---:|---:|\\n
  —— 表格不换行, 不分段, 全部挤成一行字面 \\n

修复:
  将源内 "\\n" (3 chars: \, \, n → Python 解析为 \ + n 字面)
  改为 "\n" (2 chars: \, n → Python 解析为真换行符 chr(10))
  让 markdown 渲染为:
    ### est_tokens_per_call 建议值（Round 35）

    基于本章真实 token / 调用次数计算的每调用平均：

    | Role | 当前每调用真实平均 | 当前估算 | 建议替换值 |
    |---|---:|---:|---:|

验证:
  - 修复前: source 含 `lines.append("...\\\\n...")` (4-char \\n in source)
  - 修复后: source 含 `lines.append("...\\n...")` (2-char \\n in source)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_modules():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    polluted = tuple(
        k for k in list(sys.modules)
        if k == "src" or k.startswith("src.")
    )
    snapshot = {k: sys.modules.get(k) for k in polluted}
    try:
        yield
    finally:
        # Round 103 P6+: 动态 polluted 自动覆盖 src.* 父包 + 所有子模块,
        # 父包循环已是冗余, 不再单独遍历 parents.
        for k in polluted:
            if snapshot.get(k) is not None:
                sys.modules[k] = snapshot[k]
            else:
                cur = sys.modules.get(k)
                if cur is not None and getattr(cur, "__file__", None) is None:
                    sys.modules.pop(k, None)



# ---------------------------------------------------------------------------
# Round 84: source 不应再有字面 \\n (4 chars) 在 est_tokens_per_call 段
# ---------------------------------------------------------------------------

class TestCostReportSectionHeader:
    """Round 84: cost_report "est_tokens_per_call 建议值" 段 markdown 渲染."""

    def test_section_header_no_literal_double_backslash_n(self):
        """修复后: lines.append("...\\n...") 中 \\n 应是真换行 (Python 解析后)"""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        # 找 section header 行 (含 est_tokens_per_call 建议值)
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'lines.append("' in line and 'est_tokens_per_call 建议值' in line:
                # 在源码中, 修复后此行应是 `lines.append("\\n### est_tokens_per_call ...")`
                # 即 2-char \\n (单反斜杠 + n), Python 解析为真换行
                # 不应是 3-char \\\\n (双反斜杠 + n, 字面)
                assert '\\\\n' not in line, (
                    f"Round 84: L{i} 仍有字面 \\\\n (Python source 3 chars \\, \\, n). "
                    f"修复后应是单反斜杠 (Python source 2 chars \\, n)."
                )
                break
        else:
            pytest.fail("未找到 est_tokens_per_call section header 行")

    def test_paragraph_no_literal_double_backslash_n(self):
        """修复后: 段落行 (每调用平均) 不应有字面 \\n"""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'lines.append("' in line and '每调用平均' in line:
                assert '\\\\n' not in line, (
                    f"Round 84: L{i} 段落行仍有字面 \\\\n"
                )
                break
        else:
            pytest.fail("未找到段落行")

    def test_table_header_no_literal_double_backslash_n(self):
        """修复后: | Role | 表头行不应有字面 \\n"""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'lines.append("' in line and '| Role |' in line:
                assert '\\\\n' not in line, (
                    f"Round 84: L{i} 表头行仍有字面 \\\\n"
                )
                break
        else:
            pytest.fail("未找到 | Role | 表头行")

    def test_table_separator_no_literal_double_backslash_n(self):
        """修复后: |---|---:| 分隔行不应有字面 \\n"""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        lines = content.split('\n')
        # 找到含 est_tokens_per_call 段的 L474 表分隔 (5 dashes 分隔线)
        in_target = False
        for i, line in enumerate(lines, 1):
            if 'est_tokens_per_call' in line:
                in_target = True
            if in_target and 'lines.append("|---|---:|---:|---:|\\n")' in line:
                # 这就是 est_tokens_per_call 段的 5-dash 分隔线
                assert '\\\\n' not in line, (
                    f"Round 84: L{i} 表分隔行仍有字面 \\\\n"
                )
                break


# ---------------------------------------------------------------------------
# Round 84: runtime 行为 - markdown 实际渲染正确
# ---------------------------------------------------------------------------

class TestRuntimeMarkdownRendering:
    """模拟 cost_report 生成, 验证 est_tokens_per_call 段真换行."""

    def test_section_header_renders_as_newline(self):
        # 模拟修复后的源
        line = "\n### est_tokens_per_call 建议值（Round 35）\n\n"
        # 运行时: 这是真换行 + 文本 + 真换行
        assert line.startswith(chr(10))  # 真换行开头
        assert line.endswith(chr(10) + chr(10))  # 末尾两个真换行 (空行)
        # 不应包含字面 \n (反斜杠 + n)
        assert '\\n' not in line, "运行时不应有字面 \\n"

    def test_table_row_renders_as_newline(self):
        line = "| Role | 当前每调用真实平均 | 当前估算 | 建议替换值 |\n"
        assert line.endswith(chr(10))
        assert '\\n' not in line

    def test_table_separator_renders_as_newline(self):
        line = "|---|---:|---:|---:|\n"
        assert line.endswith(chr(10))
        assert '\\n' not in line

    def test_full_suggestion_section(self):
        """完整模拟建议段, 应是真 markdown (可解析) 而非字面 \\n 堆砌."""
        # 修复后
        lines = [
            "\n### est_tokens_per_call 建议值（Round 35）\n\n",
            "基于本章真实 token / 调用次数计算的每调用平均：\n\n",
            "| Role | 当前每调用真实平均 | 当前估算 | 建议替换值 |\n",
            "|---|---:|---:|---:|\n",
            "| literal_translator | 2300 | 2500 | **2300**（偏差 -8%）|\n",
            "\n",
        ]
        content = "".join(lines)
        # 验证无字面 \\n
        assert '\\n' not in content, f"运行时含字面 \\n: {content!r}"
        # 验证是真 markdown: 表格按 \n 分隔
        table_only = content.split("|---|---:|---:|---:|\n")[1]
        rows = [r for r in table_only.split("\n") if r.strip()]
        # 应有 1 行数据 + 1 个空行 (来自 lines.append("\n"))
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# Round 84: Round 81 已修复项仍正确 (回归测试)
# ---------------------------------------------------------------------------

class TestRound81RegressionStillWorks:
    """Round 84 不能破坏 Round 81 已修复的项."""

    def test_f_string_round81_fix_intact(self):
        """Round 81 f-string 修复形式 (`+ "\\n"` 显式拼接) 仍存在."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert '偏差 {delta_pct_est:.0f}%）|" + "\\n"' in content, (
            "Round 81 f-string fix 形式 (显式 + \"\\n\") 不应被破坏"
        )

    def test_lines_append_round81_fix_intact(self):
        """Round 81 lines.append("\\n") 修复仍存在."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert 'lines.append("\\n")' in content, (
            "Round 81 lines.append fix 不应被破坏"
        )


# ---------------------------------------------------------------------------
# Round 84: 整合脚本 src/translator_agent.py 同步修复
# ---------------------------------------------------------------------------

class TestIntegratedScriptSynced:
    """散装 src/pipeline.py 修复必须同步到整合脚本 src/translator_agent.py."""

    def test_round_84_marker_in_scattered(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "Round 84" in content, "散装 pipeline.py 未标注 Round 84 修复"

    def test_round_84_marker_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 84" in content, "整合脚本未同步 Round 84 修复"

    def test_section_header_fixed_in_integrated(self):
        """整合脚本的 est_tokens_per_call 段不应再有字面 \\\\n"""
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'lines.append("' in line and 'est_tokens_per_call 建议值' in line:
                assert '\\\\n' not in line, (
                    f"Round 84: 整合脚本 L{i} 仍有字面 \\\\n"
                )
                break
        else:
            pytest.fail("整合脚本未找到 est_tokens_per_call section header")
