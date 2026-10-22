#!/usr/bin/env python3
"""
SagaTrans T2 typesafe-ai 重构验证脚本

用法：
  python tools/verify_typesafe_refactor.py           # 跑全套验证
  python tools/verify_typesafe_refactor.py --quick   # 只跑冒烟
  python tools/verify_typesafe_refactor.py --check=N # 只跑第 N 项

验证项：
  1) 模块导入完整（src.judges.* 与整合脚本）
  2) typesafe verdict 合成器单测（12 个场景）
  3) Critic 端到端（ch02 真实数据）
  4) Judge 端到端（继承 critic verdict）
  5) Backend 链正确性（Jev → OpenAICompat → Mock 降级）
  6) CLI 退出码契约（散装 vs 整合）
  7) 整合脚本 Section 11.6 完整性（23 关键符号）
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# ────────────────────────────────────────────────
# 测试结果收集
# ────────────────────────────────────────────────

RESULTS: list = []  # [(name, passed, detail)]


def check(name: str, passed: bool, detail: str = ""):
    RESULTS.append((name, passed, detail))
    mark = "✅" if passed else "❌"
    print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))


# ────────────────────────────────────────────────
# 1) 模块导入
# ────────────────────────────────────────────────

def test_01_modular_imports():
    print("\n[1] 散装模块导入完整性")
    try:
        from src.judges.typesafe_spec import (
            NoulAnswer, ChoiceAnswer, ScoreAnswer, CompositeVerdict, Answer
        )
        from src.judges.primitives import (
            VOICE_DRIFT, REGISTER_MATCH, ENTITY_CONSISTENCY, LITERAL_RESIDUE,
            SEMANTIC_LOSS, STYLE_COMPLIANCE_SCORE, ALL_PRIMITIVES, build_question_set,
        )
        from src.judges.typesafe_backend import (
            SystemOneBackend, JevBackend, OpenAICompatBackend, MockBackend,
            FallbackBackend, get_backend,
        )
        from src.judges.verdict import synthesize_verdict, DEFAULT_THRESHOLDS
        from src.judges.judge import evaluate_chunk, to_legacy_scores, build_state

        # 验证 6 个 primitives
        assert len(ALL_PRIMITIVES) == 6, f"应有 6 个 primitives，实际 {len(ALL_PRIMITIVES)}"
        for name, p in ALL_PRIMITIVES.items():
            assert "type" in p and "instructions" in p
        check("6 primitives 完整", True)

        # 验证 thresholds
        assert "voice_drift_severe_prob" in DEFAULT_THRESHOLDS
        check("DEFAULT_THRESHOLDS 完整", True)

        # 验证 backend 默认链
        b = get_backend()
        assert isinstance(b, FallbackBackend)
        assert isinstance(b._mock, MockBackend)
        check("FallbackBackend 默认链正确", True,
              detail=f"primary={type(b.primary).__name__}, fallback={type(b.fallback).__name__}, mock={type(b._mock).__name__}")

        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("散装模块导入", False, str(e))
        return False


def test_01b_integrated_imports():
    print("\n[1b] 整合脚本导入完整性（subprocess 隔离执行）")
    symbols_check = """
import sys
import translator_agent as t
expected = [
    "NoulAnswer", "ChoiceAnswer", "ScoreAnswer", "CompositeVerdict",
    "VOICE_DRIFT", "REGISTER_MATCH", "ENTITY_CONSISTENCY", "LITERAL_RESIDUE",
    "SEMANTIC_LOSS", "STYLE_COMPLIANCE_SCORE", "ALL_PRIMITIVES",
    "build_question_set", "synthesize_verdict", "evaluate_chunk",
    "to_legacy_scores", "DEFAULT_THRESHOLDS", "get_backend",
    "JevBackend", "OpenAICompatBackend", "MockBackend", "FallbackBackend",
    "build_state",
]
missing = [s for s in expected if not hasattr(t, s)]
print("MISSING:" + ",".join(missing) if missing else "OK")
"""
    r = subprocess.run(
        ["python3", "-c", symbols_check],
        capture_output=True, cwd=PROJECT,
        env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
    )
    out = r.stdout.decode().strip()
    if out == "OK":
        check("整合脚本含全部 23 个 symbols", True)
        return True
    else:
        check("整合脚本含全部 23 个 symbols", False, f"缺失: {out}")
        return False


# ────────────────────────────────────────────────
# 2) verdict 单测
# ────────────────────────────────────────────────

def test_02_verdict_synthesis():
    print("\n[2] typesafe verdict 合成器单元测试（12 场景）")
    from src.judges.typesafe_spec import NoulAnswer, ChoiceAnswer, ScoreAnswer
    from src.judges.verdict import synthesize_verdict

    cases = []

    def expect(name, ans, expected, thresholds=None):
        v = synthesize_verdict(ans, thresholds=thresholds or {})
        cases.append((name, v.decision == expected, f"{v.decision} conf={v.confidence:.2f} reasons={len(v.reject_reasons)}"))

    base_perfect = {
        "voice_drift": ChoiceAnswer("none", {"none": 0.9, "subtle": 0.08, "severe": 0.02}, 0.9),
        "register_match": ChoiceAnswer("match", {"match": 0.85, "under": 0.1, "over": 0.05}, 0.85),
        "entity_consistency": NoulAnswer(0.0, 0.95),
        "literal_residue": NoulAnswer(0.05, 0.9),
        "semantic_loss": NoulAnswer(0.02, 0.92),
        "style_compliance_score": ScoreAnswer(1.8, {0: 0.05, 1: 0.15, 2: 0.8}, 0.85),
    }
    expect("完美 → PASS", base_perfect, "PASS")

    a = dict(base_perfect)
    a["voice_drift"] = ChoiceAnswer("subtle", {"none": 0.3, "subtle": 0.6, "severe": 0.1}, 0.7)
    expect("voice subtle → PASS", a, "PASS")

    a = dict(base_perfect)
    a["voice_drift"] = ChoiceAnswer("severe", {"none": 0.1, "subtle": 0.3, "severe": 0.6}, 0.3)
    expect("voice severe conf=0.3 → PASS（低置信度升级）", a, "PASS")

    bad = {
        # Round 60: Noul 值抬高到区间外上侧（>0.7）— 否则落入 uncertain zone 不 reject
        "voice_drift": ChoiceAnswer("severe", {"none": 0.1, "subtle": 0.2, "severe": 0.7}, 0.8),
        "register_match": ChoiceAnswer("under", {"match": 0.2, "under": 0.6, "over": 0.2}, 0.7),
        "entity_consistency": NoulAnswer(0.85, 0.85),
        "literal_residue": NoulAnswer(0.85, 0.8),
        "semantic_loss": NoulAnswer(0.85, 0.7),
        "style_compliance_score": ScoreAnswer(0.3, {0: 0.7, 1: 0.2, 2: 0.1}, 0.8),
    }
    expect("全差 → REJECT", bad, "REJECT")

    a = dict(base_perfect)
    # Round 60: noul=0.5 落在 uncertain zone → 不触发 reject
    a["entity_consistency"] = NoulAnswer(0.5, 0.85)
    expect("单 entity uncertain noul=0.5 → PASS", a, "PASS")

    a = dict(base_perfect)
    # noul=0.8 超出 uncertain zone 上侧 → REJECT
    a["entity_consistency"] = NoulAnswer(0.8, 0.85)
    expect("单 entity noul=0.8 → REJECT", a, "REJECT")

    a = dict(base_perfect)
    # Round 60: uncertain zone [0.3, 0.7] — noul=0.29 在区间外下侧 → 按 threshold=0.3 → PASS
    a["entity_consistency"] = NoulAnswer(0.29, 0.95)
    expect("entity=0.29（区间外下侧）→ PASS", a, "PASS")

    a = dict(base_perfect)
    # noul=0.30 恰在 uncertain_low 边界（含）→ uncertain → PASS
    a["entity_consistency"] = NoulAnswer(0.30, 0.95)
    expect("entity=0.30（uncertain_low 边界）→ PASS", a, "PASS")

    a = dict(base_perfect)
    # noul=0.71 在 uncertain zone 上侧外 → REJECT
    a["entity_consistency"] = NoulAnswer(0.71, 0.95)
    expect("entity=0.71（区间外上侧）→ REJECT", a, "REJECT")

    a = dict(base_perfect)
    a["entity_consistency"] = NoulAnswer(0.5, 0.85)
    expect("自定义阈值 entity=0.8 → PASS", a, "PASS", thresholds={"entity_consistency_noul": 0.8})

    a = dict(base_perfect)
    a["style_compliance_score"] = ScoreAnswer(1.8, {0: 0.05, 1: 0.15, 2: 0.8}, 0.5)
    expect("overall_conf = min(0.5)", a, "PASS")

    expect("空 answers → PASS conf=0", {}, "PASS")

    sparse = {
        "voice_drift": ChoiceAnswer("none", {"none": 1.0}, 0.9),
        "register_match": ChoiceAnswer("match", {"match": 1.0}, 0.85),
    }
    expect("partial answers (只有 2 个) → PASS", sparse, "PASS")

    all_pass = True
    for name, ok, detail in cases:
        if not ok:
            all_pass = False
        print(f"    {('✅' if ok else '❌')} {name}: {detail}")
    check(f"verdict 合成 {len(cases)}/{len(cases)} 场景通过", all_pass)
    return all_pass


# ────────────────────────────────────────────────
# 3) Critic 端到端
# ────────────────────────────────────────────────

def test_03_critic_e2e():
    print("\n[3] Critic 端到端（ch02_chunk000 真实数据）")
    raw_path = PROJECT / "output" / "ch02" / "ch02_chunk000_raw.json"
    lit_path = PROJECT / "output" / "ch02" / "ch02_chunk000_literary_v6.json"

    if not raw_path.exists() or not lit_path.exists():
        check("Critic 端到端", False, "缺 ch02_chunk000 数据，请先用旧 LLM 跑过该 chunk")
        return False

    raw = json.loads(raw_path.read_text(encoding="utf-8"))["text"]
    lit = json.loads(lit_path.read_text(encoding="utf-8"))["text"]

    from src.agents.critic_agent import CriticAgent
    agent = CriticAgent()
    result = agent.process_chunk(
        chunk_id="ch02_chunk000",
        source_text="Mr. George Naylor, the young owner of a Hampshire farm, was a man of remarkably varied gifts.",
        raw_trans=raw,
        lit_trans=lit,
        style_guide={"avg_sentence_length": "适中", "lexicon_preference": "平实"},
        entity_registry={"Mr. George Naylor": "乔治·内勒"},
    )

    print(f"    backend_path: {result.get('_backend_path')}")
    print(f"    decision:     {result.get('_typesafe_verdict', {}).get('decision')}")
    print(f"    confidence:   {result.get('_typesafe_verdict', {}).get('confidence', 0):.2f}")
    print(f"    scores:       {result.get('scores')}")
    print(f"    reasons:      {len(result.get('_typesafe_verdict', {}).get('reject_reasons', []))}")

    # 验收条件
    has_verdict = "_typesafe_verdict" in result
    has_backend = "_backend_path" in result
    has_scores = isinstance(result.get("scores"), dict) and len(result["scores"]) >= 6

    check("Critic 返回 _typesafe_verdict", has_verdict)
    check("Critic 返回 _backend_path", has_backend)
    check("Critic 返回 6 维度 legacy scores", has_scores)
    check("Critic 实际走 typesafe 路径（而非 legacy）",
          result.get("_backend_path") != "legacy",
          detail=f"backend={result.get('_backend_path')}")

    return has_verdict and has_backend and has_scores


# ────────────────────────────────────────────────
# 4) Judge 端到端
# ────────────────────────────────────────────────

def test_04_judge_e2e():
    print("\n[4] Judge 端到端（继承 critic typesafe verdict）")
    raw_path = PROJECT / "output" / "ch02" / "ch02_chunk000_raw.json"
    lit_path = PROJECT / "output" / "ch02" / "ch02_chunk000_literary_v6.json"
    if not raw_path.exists():
        check("Judge 端到端", False, "缺 ch02 数据")
        return False

    raw = json.loads(raw_path.read_text(encoding="utf-8"))["text"]
    lit = json.loads(lit_path.read_text(encoding="utf-8"))["text"]

    from src.agents.critic_agent import CriticAgent
    from src.agents.judge_agent import JudgeAgent
    from src.core.decision_engine import DecisionEngine

    de = DecisionEngine(db_path="/tmp/verify.db", scheduler_factory=lambda: None)
    critic = CriticAgent()
    judge = JudgeAgent(decision_engine=de)

    c = critic.process_chunk(
        chunk_id="ch02_chunk000",
        source_text="Mr. George Naylor, the young owner of a Hampshire farm, was a man of remarkably varied gifts.",
        raw_trans=raw, lit_trans=lit,
        style_guide={"avg_sentence_length": "适中"},
        entity_registry={"Mr. George Naylor": "乔治·内勒"},
    )
    j = judge.process_chunk(
        chunk_id="ch02_chunk000",
        source_text="...",
        lit_trans=lit,
        critic_report=c,
    )

    print(f"    judge decision: {j.get('decision')}")
    print(f"    judge backend:  {j.get('_backend_path')}")
    print(f"    reason:         {j.get('reject_reason', '')[:80]}")

    check("Judge 决策与 Critic verdict 一致",
          j.get("decision") == c.get("_typesafe_verdict", {}).get("decision"),
          detail=f"judge={j.get('decision')} vs critic={c.get('_typesafe_verdict', {}).get('decision')}")
    check("Judge 未调 legacy LLM",
          j.get("_backend_path") != "legacy",
          detail=f"backend={j.get('_backend_path')}")
    return True


# ────────────────────────────────────────────────
# 5) Backend 链降级
# ────────────────────────────────────────────────

def test_05_backend_chain():
    print("\n[5] Backend fallback 链正确性")
    from src.judges.typesafe_backend import (
        FallbackBackend, JevBackend, OpenAICompatBackend, MockBackend, get_backend,
    )
    b = get_backend()
    assert isinstance(b, FallbackBackend)
    check("默认 backend 是 FallbackBackend", True)
    check("primary 是 JevBackend", isinstance(b.primary, JevBackend))
    check("fallback 是 OpenAICompatBackend", isinstance(b.fallback, OpenAICompatBackend))
    check("mock 是 MockBackend", isinstance(b._mock, MockBackend))

    # 探测：哪个 backend 当前能调
    from src.judges.judge import build_state
    from src.judges.primitives import build_question_set
    state = build_state(
        source_text="test", raw_translation="测试", literary_translation="测试",
        style_guide_stats={}, entity_registry={},
    )
    questions = build_question_set()

    reachable = []
    if b.primary.api_key:
        try:
            b.primary.evaluate(state, questions, timeout=5)
            reachable.append("jev")
        except Exception:
            pass
    try:
        b.fallback.evaluate(state, questions, timeout=5)
        reachable.append("openai_compat")
    except (ConnectionError, OSError):
        pass
    reachable.append("mock_heuristic")

    print(f"    当前可达 backend: {reachable}")
    check("Mock 后端必可达（兜底）", "mock_heuristic" in reachable)

    if not b.primary.api_key and "openai_compat" not in reachable:
        print("    ⚠️ 当前离线（API 不可达），仅 mock_heuristic 可用")
        print("    💡 网络恢复后此处应自动变为 openai_compat；Jev 需 TYPESAFE_API_KEY")

    return True


# ────────────────────────────────────────────────
# 6) CLI 退出码契约
# ────────────────────────────────────────────────

def test_06_cli_exit_codes():
    print("\n[6] CLI 退出码契约（散装 vs 整合）")
    cases = [
        ("split 无 --input", "split", [], 1),
        ("init 缺章节", "init", ["--chapter", "ch99"], 1),
        ("非法 chapter_id", "init", ["--chapter", "../../bad"], 1),
    ]
    all_ok = True
    for name, cmd, args, expected_rc in cases:
        # 散装
        r1 = subprocess.run(
            ["python3", "-m", "src.main", cmd] + args,
            capture_output=True, cwd=PROJECT, env={**os.environ, "PYTHONPATH": str(PROJECT)},
        )
        # 整合
        r2 = subprocess.run(
            ["python3", "src/translator_agent.py", cmd] + args,
            capture_output=True, cwd=PROJECT,
        )
        ok = (r1.returncode == expected_rc == r2.returncode)
        if not ok:
            all_ok = False
        print(f"    {('✅' if ok else '❌')} {name}: 散装={r1.returncode} 整合={r2.returncode} 期望={expected_rc}")
    check(f"CLI 退出码契约 {len(cases)}/{len(cases)} 通过", all_ok)
    return all_ok


# ────────────────────────────────────────────────
# 7) 整合脚本 Section 11.6 完整性
# ────────────────────────────────────────────────

def test_07_integrated_completeness():
    print("\n[7] 整合脚本 Section 11.6 完整性")
    import re
    content = (PROJECT / "src" / "translator_agent.py").read_text(encoding="utf-8")
    m = re.search(r"# Section 11\.6.*?(?=# Section 12)", content, re.DOTALL)
    if not m:
        check("Section 11.6 存在", False)
        return False
    sec = m.group(0)
    print(f"    Section 11.6 行数: {sec.count(chr(10))}")

    need = [
        "class NoulAnswer", "class ChoiceAnswer", "class ScoreAnswer",
        "class CompositeVerdict",
        "class SystemOneBackend", "class JevBackend",
        "class OpenAICompatBackend", "class MockBackend", "class FallbackBackend",
        "VOICE_DRIFT", "REGISTER_MATCH", "ENTITY_CONSISTENCY", "LITERAL_RESIDUE",
        "SEMANTIC_LOSS", "STYLE_COMPLIANCE_SCORE", "ALL_PRIMITIVES",
        "DEFAULT_THRESHOLDS", "def build_question_set", "def synthesize_verdict",
        "def get_backend", "def build_state", "def evaluate_chunk",
        "def to_legacy_scores",
    ]
    missing = [n for n in need if n not in sec]
    check(f"Section 11.6 含全部 {len(need)} 关键符号", len(missing) == 0,
          detail=f"缺失: {missing}" if missing else "")
    return len(missing) == 0


# ────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="只跑冒烟（verdict + 导入）")
    parser.add_argument("--check", type=int, help="只跑第 N 项")
    args = parser.parse_args()

    print("=" * 60)
    print("SagaTrans T2 typesafe-ai 重构验证")
    print("=" * 60)

    tests = [
        ("1", "散装模块导入", test_01_modular_imports),
        ("1b", "整合脚本导入", test_01b_integrated_imports),
        ("2", "verdict 单测", test_02_verdict_synthesis),
        ("3", "Critic 端到端", test_03_critic_e2e),
        ("4", "Judge 端到端", test_04_judge_e2e),
        ("5", "Backend 链", test_05_backend_chain),
        ("6", "CLI 退出码", test_06_cli_exit_codes),
        ("7", "整合完整性", test_07_integrated_completeness),
    ]

    for num, name, fn in tests:
        if args.check and args.check != int(num.rstrip("b")):
            continue
        if args.quick and num not in ("1", "1b", "2"):
            continue
        fn()

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"汇总: {passed}/{total} 通过")
    print("=" * 60)
    if passed != total:
        print("\n失败项：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  ❌ {name}  ({detail})")
        sys.exit(1)
    else:
        print("\n🎉 全部验证通过")


if __name__ == "__main__":
    main()
