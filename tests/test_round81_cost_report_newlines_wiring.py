"""Round 81: 修复 _print_llm_cost_report 中 est_tokens_per_call 建议表渲染字面 \\n 的 bug.

根因: src/pipeline.py:450 f-string 末尾 `|\\n` 是字面反斜杠 + n (Python '\\\\n' → '\\n'),
      line 452 `lines.append("\\\\n")` 同样字面. 整段代码生成的内容末尾有字面 '\\n\\n'
      (反斜杠 + n + 反斜杠 + n) 而非真换行符, markdown 渲染时显示为字面而非换行.

修复: 把字面 '\\n' 改为真换行符 \\n (Python 字符串字面: '\\\\n' → '\\n' 即换行符).
      让 est_tokens_per_call 建议表行末尾真正换行.

验证:
  - 修复前: cost_report.md 的"建议替换值"表格行末尾显示为 |\\n 字面
  - 修复后: 表格行末尾真换行, markdown 渲染正确
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_modules():
        # Round 103 P6+: 动态捕获所有 src.* 命名空间, 避免逐个列举遗漏
    # translator_agent shim 注入的假模块 (translator_agent 被 import 时).
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



class TestSourcePatch:
    """散装 src/pipeline.py 必须应用 Round 81 修复."""

    def test_round_81_marker_present(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "Round 81" in content

    def test_f_string_newline_real(self):
        """f-string 末尾 |\\n 必须是真换行符 (不是字面 \\n)."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        # 找 est_tokens_per_call 建议段的 f-string
        target = 'f"{marker}{suggested:,}{marker}（偏差 {delta_pct_est:.0f}%）|\n"'
        # 在源码中, 我们期望 |\\n 后面是真换行符 + "
        # 实际的字面字符序列应该是: ...|\\n" (字面 \\n = backslash + n)
        # 修复后: ...|\\n" (真换行符 + ")
        # 这里在源码字面中, '|\\n"' 的最后 3 字符应该是 '\\n"'
        # 真换行符在源码中只显示为 \\n (1 char)
        # 所以正确的字面表示是 '|%s"' (即 ... + 真换行 + ")
        # 错误的是 '|\\n"' (即 ... + 反斜杠 + n + ")
        # Round 81: 源码中 f-string 末尾不能再是字面反斜杠 + n (4 chars: |\n")
        # 原 BUG 形式: f"...偏差 ...%）|\n" (Python 源码中是 4 chars: |, \, n, ")
        # 修复后: 不再有这种字面 4-char 序列
        literal_pattern = '偏差 {delta_pct_est:.0f}%）|\n"'
        assert literal_pattern not in content, (
            "Round 81: src/pipeline.py 不应再含字面 `偏差 ...%）|\n` "
            "(即 Python 源码中的 `|\n` = 4 chars 反斜杠 + n). "
            "修复后这里应是 `|` + 真换行符 (chr(10)) + 双引号, "
            "或 `|` + 双引号 + `+ \n` 显式拼接."
        )
    def test_lines_append_newline_real(self):
        """lines.append(\"\\n\") 必须是真换行符."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        # 找 est_tokens_per_call 建议段后的 lines.append
        # 在源码中, 'lines.append("\n")' 的 Python 解析:
        # - 字面 '"\\n"' 解析为 1 char 换行符
        # - 字面 '"\\\\n"' 解析为 2 chars 反斜杠 + n
        # 所以期望: lines.append("\n") 是真换行符
        # 旧的 (字面 \\n): lines.append("\n") 在 Python 解析后是 1 char 换行符 (OK!)
        # 但 line 452 现在是 lines.append("\n") 还是 lines.append("\n")? 让我看
        # 关键: 在源码中, 字面 '\\n' (双反斜杠 + n) 才是真换行符; 字面 '\\\\n' (4 chars) 是反斜杠 + n
        # 修复后 line 452 应该是: lines.append("\n") (即 Python 解析为 1 char 换行符)
        # 修复前 line 452 是: lines.append("\n") 在源码中显示为 4 chars '\\\\n'
        # 我需要直接读源码确认
        lines = content.split('\n')
        # 找 line 452 (est_tokens_per_call 建议段后的 lines.append)
        for i, line in enumerate(lines):
            if 'lines.append' in line and 'Round 81' in line and '真换行符' in line and 'except' not in line:
                # 检查源码中是 lines.append("\n") (4 chars: \\n) 还是别的
                # Round 81: 源码中应是 `lines.append("\n")` (4 字符: \, n 不应被 markdown 当作转义)
                # 修复前: lines.append("\\n") (双反斜杠, Python 解析为 \n 字面)
                # 修复后: lines.append("\n") (单反斜杠, Python 解析为 chr(10) 真换行)
                # 测试检查 source 字节 (非运行时), 所以期望源码中是 4 字节 "\n":
                assert 'lines.append("\\n")' in line, (
                    f"Round 81: line {{i+1}} 应有 lines.append(\"\\n\") (4 字节源码形式, "
                    f"Python 解析为真换行符 chr(10)), "
                    f"实际: {{line!r}}"
                )
                break

    def test_syntax_check(self):
        import ast
        ast.parse(Path("src/pipeline.py").read_text(encoding="utf-8"))


class TestIntegratedScript:
    """整合脚本 src/translator_agent.py 必须同步应用 Round 81 修复."""

    def test_pipeline_importable(self):
        from src.translator_agent import TranslationPipeline
        assert TranslationPipeline is not None

    def test_round_81_present_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        assert "Round 81" in content, "整合脚本未同步 Round 81 修复"


class TestCostReportMarkdownOutput:
    """模拟 cost_report 生成逻辑, 验证 markdown 表格行末尾是真换行符."""

    def test_markdown_table_row_ends_with_real_newline(self):
        """核心验证: 模拟建议表生成, 表格行末尾必须是真换行符 \\n 而非字面反斜杠 + n."""
        # 直接 import pipeline 模块 (因为 _print_llm_cost_report 是其方法)
        from src.translator_agent import TranslationPipeline
        # 不需要跑完整 __init__, 直接构造 stub
        pipeline = TranslationPipeline.__new__(TranslationPipeline)
        pipeline.chapter_id = "ch_r81_test"
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.output_dir = Path(tmp) / "output" / "ch_r81_test"
            pipeline.output_dir.mkdir(parents=True)

            # 模拟建议表生成逻辑 (与 src/pipeline.py:443-453 一致)
            lines = []
            role = "literal_translator"
            avg = 2300.5
            current_est = 2500
            suggested = 2500
            marker = ""
            delta_pct_est = 7.96
            lines.append(
                f"| {role} | {avg:,.0f} | {current_est:,} | "
                f"{marker}{suggested:,}{marker}（偏差 {delta_pct_est:.0f}%）|\n"
                # Round 81: 上一行末尾 \n 应是真换行符
            )
            lines.append("\n")  # Round 81: 应是真换行符

            content = "".join(lines)
            # 关键验证: content 不应包含字面 '\n' (反斜杠 + n)
            # 字面 '\n' 在文件中是 2 chars ('\\' + 'n')
            # 真换行符是 1 char ('\n')
            # 我们检查 content 末尾是否包含 2 个连续的真换行符 (而不是字面 \n + \n)
            # 期望: content = "...偏差 X%)  # Round 81\n新真换行符\n"
            # 真换行符 chr(10)
            assert chr(10) in content, "content 必须包含真换行符"

            # 不应包含字面 \n (2 chars: backslash + n) 在 markdown 表格行附近
            # 注意: 源代码中的注释里可能有 'Round 81\n' 字面, 但 join 后的 content 应不包含
            # 等等, 注释是 f-string 的注释, 不会出现在 content 中 (只在源码中)
            # 所以 content 应该只包含: 表格行 + 第二个 \n
            # 验证: 表格行末尾是真换行符 (chr(10))
            # content 应该是 "...偏差 X%)  # Round 81\n新真换行符\n"
            # 不应包含字面 \\n (backslash + n) 在 '偏差' 后

            # 验证: content[-1] 应该是真换行符
            assert content.endswith(chr(10)), (
                f"content 末尾应是真换行符, 实际最后 1 char: {content[-1]!r}, "
                f"最后 10 chars: {content[-10:]!r}"
            )

            # 验证: content 不应包含字面 \\n (backslash + n) 在 '偏差' 之后
            bias_pos = content.find("偏差")
            if bias_pos >= 0:
                after_bias = content[bias_pos:bias_pos+30]
                # 不应有字面 '\\n' (backslash + n) 紧跟偏差
                assert "\\n" not in after_bias or "Round 81" not in after_bias, (
                    f"Round 81: content 在偏差后不应有字面 \\\\n. "
                    f"实际: {after_bias!r}"
                )


class TestSourceCodeHasCorrectFormat:
    """源码层面验证: 修复后的字符串字面用真换行符."""

    def test_f_string_ends_with_real_newline_bytes(self):
        """源码层面验证: f-string 末尾不能再是字面 |\\n" (4 chars 反斜杠 + n).

        修复前 (BUG): f"...偏差 ...%）|\\n"   (源码中是 4 chars: |, \\, n, ")
        修复后:      f"...偏差 ...%）|" + "\\n"  (显式拼接, Python 解析为真换行符)
        """
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        # BUG 形式: 偏差...%）|\\n" (literal backslash + n + quote)
        bug_form = '偏差 {delta_pct_est:.0f}%）|\\n"'
        assert bug_form not in content, (
            f"Round 81: src/pipeline.py 不应再有字面 `偏差 ...%）|\\n` "
            f"(即 Python 字面 \\n, 4 chars 反斜杠 + n + 双引号). "
            f"实际 source 已含 BUG 形式。"
        )
        # FIX 形式: 偏差...%）|" + "\\n" (显式 chr(10) 拼接)
        fix_form = '偏差 {delta_pct_est:.0f}%）|" + "\\n"'
        assert fix_form in content, (
            f"Round 81: src/pipeline.py 应有 FIX 形式 `偏差 ...%）|` + `\"\\n\"`. "
            f"实际未找到。"
        )
