"""Round 76: 修复 _verify_final_outputs 将 fallback 章节误判为"产出残缺"的 bug.

根因: _verify_final_outputs 在 strict=True 分支下, 当任何 chunk state != COMPLETED 时
      (典型场景: PERMANENTLY_FAILED 但 final.json 已落盘的 fallback 章节),
      无差别 return False → pipeline.run() 返回 False → CLI 退出码 = 1.
      但 fallback 章节的 final.json 实际已写入磁盘, 用户可读; 把它判为"残缺"
      与磁盘事实不一致, 是 false positive.

修复: 分离 completeness 与 quality 校验.
  - completeness 失败 (缺 final.json / 残留旧文件) → return False (真残缺)
  - 仅 quality 降级 (state=PERMANENTLY_FAILED 但 final.json 已落盘) → return True
    (产出完整但质量降级, warning 已告知用户)

向后兼容: 调用方 (CLI / CI) 仍可通过 cost_report.md 看到 fallback 分类,
         不再依赖错误退出码表达"有 fallback"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_translator_agent_module():
    """隔离父包, 避免测试间相互污染 (历史教训)."""
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



def _make_pipeline(output_dir, tasks):
    """构造 TranslationPipeline stub (跳过 __init__, 直接注入必要属性)."""
    from src.translator_agent import TranslationPipeline

    class _StubScheduler:
        def __init__(self, tasks):
            self._tasks = tasks

        def get_all_tasks_by_chapter(self, chapter_id):
            return self._tasks

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.chapter_id = "ch_test"
    pipeline.output_dir = output_dir
    pipeline.scheduler = _StubScheduler(tasks=tasks)
    return pipeline


class TestSourcePatch:
    """散装 src/pipeline.py 必须应用 Round 76 修复."""

    def test_round_76_marker_present(self):
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        assert "Round 76" in content, "Round 76 修复标记必须存在"

    def test_quality_separation_logic_present(self):
        """_verify_final_outputs 必须区分 completeness 失败和 quality 失败."""
        content = Path("src/pipeline.py").read_text(encoding="utf-8")
        func_start = content.find("def _verify_final_outputs")
        assert func_start > 0, "_verify_final_outputs 函数必须存在"
        # Round 76 关键修复: missing_chunks 非空时才 return False
        # 而不是无差别 return False
        func_body = content[func_start:func_start + 3500]
        assert "missing_chunks" in func_body
        # Round 76: fallback_chunks 不再触发 return False, 仅 warning
        # 关键代码模式: "if missing_chunks:" 应在 fallback warning 之后, 且 return False 受其控制
        assert "if missing_chunks:" in func_body and "return False" in func_body
        # 必须有 "return True" 在 fallback 之后 (fallback 单独不再算残缺)
        assert "Round 76" in func_body

    def test_syntax_check(self):
        """防止 syntax error (历史教训: Round 75 第一次实现时破坏 syntax)."""
        import ast
        ast.parse(Path("src/pipeline.py").read_text(encoding="utf-8"))


class TestIntegratedScript:
    """整合脚本 src/translator_agent.py 必须同步应用 Round 76 修复."""

    def test_translation_pipeline_importable(self):
        from src.translator_agent import TranslationPipeline
        assert TranslationPipeline is not None

    def test_round_76_present_in_integrated(self):
        content = Path("src/translator_agent.py").read_text(encoding="utf-8")
        # 整合脚本是自动生成的, 必须与散装同步含 Round 76 标记
        # (散装改了, build_integrated 必须同步; 测试时已运行 build_integrated)
        assert "Round 76" in content, "整合脚本未同步 Round 76 修复, 请跑 scripts/build_integrated.py"


def _write_final(output_dir, chunk_id, with_metadata=True):
    """写一个 final.json, 默认带 metadata.fallback=True."""
    path = output_dir / f"{chunk_id}_final.json"
    if with_metadata:
        data = {
            "text": f"译文 for {chunk_id}",
            "metadata": {"fallback": True, "reason": "PERMANENTLY_FAILED_AFTER_MAX_RETRIES"},
        }
    else:
        data = {"text": f"译文 for {chunk_id}"}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class TestVerifyFinalOutputsBehavior:
    """_verify_final_outputs 行为测试 - 覆盖 5 种场景."""

    def test_all_completed_returns_true(self, tmp_path, capsys):
        """场景 1: 全部 COMPLETED → return True (baseline 行为)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "COMPLETED"},
            {"chunk_id": "ch_test_chunk001", "state": "COMPLETED"},
        ]
        for t in tasks:
            _write_final(output_dir, t["chunk_id"], with_metadata=False)

        pipeline = _make_pipeline(output_dir, tasks)
        result = pipeline._verify_final_outputs(expected_count=2)
        assert result is True, "全 COMPLETED 必须返回 True"

        # 不应打印任何 fallback warning
        captured = capsys.readouterr()
        assert "fallback" not in captured.out.lower(), (
            f"全 COMPLETED 不应触发 fallback warning, 但 stdout: {captured.out!r}"
        )

    def test_only_fallback_returns_true_round76(self, tmp_path, capsys):
        """场景 2 (Round 76 核心): 仅 fallback 章节 (PERMANENTLY_FAILED + final.json 存在)
        → 必须返回 True (而非旧版本的 False)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "COMPLETED"},
            {"chunk_id": "ch_test_chunk001", "state": "PERMANENTLY_FAILED"},
            {"chunk_id": "ch_test_chunk002", "state": "PERMANENTLY_FAILED"},
        ]
        # 全部 final.json 都已落盘 (包括 fallback 的两个)
        for t in tasks:
            _write_final(output_dir, t["chunk_id"])

        pipeline = _make_pipeline(output_dir, tasks)
        result = pipeline._verify_final_outputs(expected_count=3)
        assert result is True, (
            "Round 76: fallback 章节 (final.json 已落盘) 必须判为产出完整, "
            "不应判残缺 (旧行为返回 False, 导致 CLI 退出码误报 1)"
        )

        # 必须打印 fallback warning 告知用户
        captured = capsys.readouterr()
        assert "fallback" in captured.out.lower(), (
            f"fallback 章节必须打印 warning 告知用户, 但 stdout: {captured.out!r}"
        )
        assert "ch_test_chunk001" in captured.out
        assert "ch_test_chunk002" in captured.out

    def test_truly_missing_returns_false(self, tmp_path, capsys):
        """场景 3: 真残缺 (有 chunk state != COMPLETED 且 final.json 不存在) → return False."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "COMPLETED"},
            {"chunk_id": "ch_test_chunk001", "state": "COMPLETED"},
            # chunk002: 状态非 COMPLETED 且 final.json 不存在 (真残缺)
            {"chunk_id": "ch_test_chunk002", "state": "PERMANENTLY_FAILED"},
        ]
        # 仅写 chunk000 和 chunk001 的 final.json
        _write_final(output_dir, "ch_test_chunk000", with_metadata=False)
        _write_final(output_dir, "ch_test_chunk001", with_metadata=False)
        # chunk002 final.json 故意不写

        pipeline = _make_pipeline(output_dir, tasks)
        result = pipeline._verify_final_outputs(expected_count=3)
        assert result is False, (
            "真残缺 (final.json 缺失) 必须返回 False, 不能因 Round 76 改动而放宽"
        )

        captured = capsys.readouterr()
        assert "ch_test_chunk002" in captured.out, "必须点名缺失 chunk"

    def test_mixed_fallback_and_missing(self, tmp_path, capsys):
        """场景 4: fallback + 真残缺混合 → 必须返回 False (真残缺主导)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "PERMANENTLY_FAILED"},  # fallback (有 final.json)
            {"chunk_id": "ch_test_chunk001", "state": "COMPLETED"},
            {"chunk_id": "ch_test_chunk002", "state": "PERMANENTLY_FAILED"},  # 真残缺 (无 final.json)
        ]
        _write_final(output_dir, "ch_test_chunk000")  # fallback OK
        _write_final(output_dir, "ch_test_chunk001", with_metadata=False)
        # chunk002 故意不写 → 真残缺

        pipeline = _make_pipeline(output_dir, tasks)
        result = pipeline._verify_final_outputs(expected_count=3)
        assert result is False, (
            "混合场景: 有真残缺 chunk 时必须返回 False, "
            "Round 76 不能让 fallback 掩盖真残缺"
        )

    def test_stale_final_returns_false(self, tmp_path, capsys):
        """场景 5: 残留旧 final.json (上一轮多余 chunk) → return False (真残缺信号)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "COMPLETED"},
            {"chunk_id": "ch_test_chunk001", "state": "COMPLETED"},
        ]
        # 当前任务的 final.json
        _write_final(output_dir, "ch_test_chunk000", with_metadata=False)
        _write_final(output_dir, "ch_test_chunk001", with_metadata=False)
        # 残留旧文件 (上一轮的多余 chunk)
        _write_final(output_dir, "ch_test_chunk_old", with_metadata=False)

        pipeline = _make_pipeline(output_dir, tasks)
        result = pipeline._verify_final_outputs(expected_count=2)
        assert result is False, "残留旧 final.json 必须判残缺 (stale 检测)"


class TestRegressionBackwardCompat:
    """回归测试: Round 76 改动不应破坏已有功能."""

    def test_expected_count_zero_returns_true(self, tmp_path):
        """expected_count=0 的边界情况 (历史 baseline)."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = []
        pipeline = _make_pipeline(output_dir, tasks)
        result = pipeline._verify_final_outputs(expected_count=0)
        assert result is True, "expected_count=0 必须返回 True (baseline)"

    def test_strict_false_skips_state_check(self, tmp_path, capsys):
        """strict=False 时, 即使有非 COMPLETED 状态, 只要 final.json 齐全就 return True."""
        output_dir = tmp_path / "output" / "ch_test"
        output_dir.mkdir(parents=True)
        tasks = [
            {"chunk_id": "ch_test_chunk000", "state": "PERMANENTLY_FAILED"},
        ]
        _write_final(output_dir, "ch_test_chunk000")

        pipeline = _make_pipeline(output_dir, tasks)
        result = pipeline._verify_final_outputs(expected_count=1, strict=False)
        assert result is True, "strict=False 应跳过 state 校验"
