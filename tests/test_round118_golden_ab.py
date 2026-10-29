"""Round 118 P2-7 / Round 128: 黄金集 A/B — 直译模型×effort 档位组合对比与来源分组统计.

覆盖:
  - GOLDEN_AB_MATRIX 四组合结构 (literal_translator/primary × effort none/档位)
  - run_golden_test 的 ab_override 覆盖 role + effort 并落盘新字段
  - summarize_golden_results 按 judge_backend / confidence_source 分组
  - recommend_golden_combo 仅在 Jev 校准占比达标时给推荐 (降级环境拒绝结论)
  - run_golden_ab 聚合与报告落盘
  - golden-gate --ab: 无校准组合时拒绝放行; 有推荐时用其结果过门禁
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import src.test_golden_set as g


class _FakeLLM:
    def __init__(self, calls):
        self.calls = calls

    def generate(self, prompt, model_name=None, extra_body=None, **params):
        self.calls.append({
            "model_name": model_name,
            "extra_body": extra_body,
            "params": params,
        })
        return "直译稿"


class _FakeAgent:
    def __init__(self, out):
        self.out = out

    def process_chunk(self, *args, **kwargs):
        return self.out


class _FakeEvaluator:
    instance = None
    last_init = None

    def __init__(self, db_path=None, literal_role=None):
        type(self).last_init = {"db_path": db_path, "literal_role": literal_role}
        type(self).instance = self
        self.literal_role = literal_role
        self.calls = []
        self.llm = _FakeLLM(self.calls)
        self.ref_agent = _FakeAgent("refs")
        self.db = types.SimpleNamespace(format_prompt_context=lambda source_text="": "gloss")
        self.rewriter_agent = _FakeAgent("润色稿")
        self.critic_agent = _FakeAgent({"scores": {"fluency": 8.0}, "_backend_path": "jev"})
        self.judge_agent = _FakeAgent({
            "decision": "PASS",
            "_backend_path": "jev",
            "_typesafe_verdict": {"confidence_source": "jev"},
        })

    def evaluate_style_collapse(self, source_text, translated_text):
        return {
            "style_collapse_rate": 0.1,
            "average_preservation": 0.9,
            "preservation_per_dimension": {"avg_sentence_length": 0.9},
        }


class _FakeChunker:
    def __init__(self, soft_limit=1000, hard_limit=2500):
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit

    def split_markdown(self, text):
        return ["chunk one", "chunk two"]


class _FakeConfig:
    chunker = {"soft_limit": 1000, "hard_limit": 2500}
    style_guide = {"genre": "测试"}
    pipeline = {"golden_ab": {"min_calibrated_ratio": 0.5}}

    def get_task_role(self, task_name):
        return "literal_translator"

    def resolve_task_model(self, task_name):
        return "literal_translator", {
            "temperature": 0.1,
            "extra_body": {"vendor": task_name},
        }


def _prepare_golden_env(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir(exist_ok=True)
    golden_dir = tmp_path / "input" / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    (golden_dir / "hyperion_5k.md").write_text("source text", encoding="utf-8")
    monkeypatch.setattr(g, "__file__", str(src_dir / "test_golden_set.py"))
    monkeypatch.setattr(g, "_config", _FakeConfig())
    monkeypatch.setattr(g, "GoldenSetEvaluator", _FakeEvaluator)
    monkeypatch.setattr(g, "SmartChunker", _FakeChunker)
    monkeypatch.setattr(g, "get_role_model_name", lambda role: f"model:{role}")


def _chunk_result(decision="PASS", backend="jev", conf="jev", score=8.0):
    return {
        "chunk_id": f"c_{backend}_{conf}_{decision}_{score}",
        "critic_scores": {"fluency": score},
        "judge_decision": decision,
        "judge_backend": backend,
        "confidence_source": conf,
        "style_collapse_rate": 0.1,
        "style_preservation": 0.9,
    }


# ---------------------------------------------------------------------------
# (1) 矩阵结构
# ---------------------------------------------------------------------------

class TestGoldenAbMatrix:
    def test_four_combos_two_models_two_effort_states(self):
        combos = g.GOLDEN_AB_MATRIX
        assert len(combos) == 4
        axis = {(c["role"], c.get("effort")) for c in combos}
        assert axis == {
            ("literal_translator", None),
            ("literal_translator", "xhigh"),
            ("primary", None),
            ("primary", "max"),
        }
        names = [c["name"] for c in combos]
        assert len(set(names)) == 4


# ---------------------------------------------------------------------------
# (2) run_golden_test 的 A/B 覆盖与落盘字段
# ---------------------------------------------------------------------------

class TestGoldenTestAbOverride:
    def test_override_plumbs_role_and_effort(self, tmp_path, monkeypatch):
        _prepare_golden_env(tmp_path, monkeypatch)
        report = tmp_path / "report.json"
        results = g.run_golden_test(
            ab_override={"name": "primary_effort_max", "role": "primary", "effort": "max"},
            report_path=report,
        )
        assert _FakeEvaluator.last_init["literal_role"] == "primary"
        call = _FakeEvaluator.instance.calls[0]
        assert call["params"]["effort"] == "max"
        assert call["model_name"] == "model:primary"
        # Round 127: extra_body 为任务级配置，A/B 覆盖角色不再改变它
        assert call["extra_body"] == {"vendor": "literal_translation"}

        assert results[0]["ab"] == "primary_effort_max"
        assert results[0]["literal_role"] == "primary"
        assert results[0]["literal_effort"] == "max"
        assert results[0]["judge_backend"] == "jev"
        assert results[0]["confidence_source"] == "jev"

        data = json.loads(report.read_text(encoding="utf-8"))
        stats = data["summary"]["ab_stats"]
        assert stats["pass_rate"] == 1.0
        assert stats["calibrated_ratio"] == 1.0
        assert data["details"][0]["ab"] == "primary_effort_max"

    def test_default_run_keeps_config_literal_settings(self, tmp_path, monkeypatch):
        _prepare_golden_env(tmp_path, monkeypatch)
        results = g.run_golden_test(report_path=tmp_path / "r.json")
        assert _FakeEvaluator.last_init["literal_role"] == "literal_translator"
        call = _FakeEvaluator.instance.calls[0]
        assert "effort" not in call["params"], "literal_translation 任务默认不下发 effort"
        assert call["model_name"] == "model:literal_translator"
        assert results[0]["ab"] is None
        assert results[0]["literal_effort"] is None


# ---------------------------------------------------------------------------
# (3) 分组统计与推荐
# ---------------------------------------------------------------------------

class TestSummarizeAndRecommend:
    def test_summarize_groups_by_backend_and_source(self):
        results = [
            _chunk_result("PASS", "jev", "jev", 8.0),
            _chunk_result("REJECT", "openai_compat", "openai_compat", 6.0),
            {"chunk_id": "broken", "error": "boom"},
        ]
        s = g.summarize_golden_results(results)
        assert s["total"] == 3
        assert s["valid_count"] == 2
        assert s["error_count"] == 1
        assert s["pass_rate"] == 0.5
        assert s["avg_critic_score"] == 7.0
        assert s["judge_backend_dist"] == {"jev": 1, "openai_compat": 1}
        assert s["confidence_source_dist"] == {"jev": 1, "openai_compat": 1}
        assert s["calibrated_ratio"] == 0.5

    def test_summarize_empty_results(self):
        s = g.summarize_golden_results([])
        assert s["valid_count"] == 0
        assert s["pass_rate"] == 0.0
        assert s["calibrated_ratio"] == 0.0

    def test_legacy_judge_maps_to_uncalibrated(self):
        result = g._extract_confidence_source({"decision": "PASS", "_backend_path": "legacy"})
        assert result == "_uncalibrated"

    def _combo(self, pass_rate, critic, calibrated):
        return {"summary": {
            "valid_count": 2,
            "calibrated_ratio": calibrated,
            "pass_rate": pass_rate,
            "avg_critic_score": critic,
            "avg_style_preservation": 0.9,
        }}

    def test_recommend_ignores_degraded_high_pass_rate(self):
        combos = {
            "degraded": self._combo(1.0, 9.9, 0.0),
            "calibrated_hi": self._combo(1.0, 6.0, 1.0),
            "calibrated_lo": self._combo(0.5, 9.0, 1.0),
        }
        rec = g.recommend_golden_combo(combos, min_calibrated=0.5)
        assert rec["name"] == "calibrated_hi"
        assert set(rec["calibrated_candidates"]) == {"calibrated_hi", "calibrated_lo"}

    def test_recommend_none_when_all_degraded(self):
        combos = {"a": self._combo(1.0, 9.0, 0.0), "b": self._combo(0.9, 8.0, 0.25)}
        rec = g.recommend_golden_combo(combos, min_calibrated=0.5)
        assert rec["name"] is None
        assert rec["reason"] == "no_calibrated_combo"

    def test_recommend_respects_threshold(self):
        combos = {"a": self._combo(1.0, 9.0, 0.4)}
        assert g.recommend_golden_combo(combos, min_calibrated=0.5)["name"] is None
        assert g.recommend_golden_combo(combos, min_calibrated=0.4)["name"] == "a"


# ---------------------------------------------------------------------------
# (4) run_golden_ab 聚合与报告
# ---------------------------------------------------------------------------

class TestRunGoldenAb:
    def test_aggregates_combos_and_recommends(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        monkeypatch.setattr(g, "__file__", str(src_dir / "test_golden_set.py"))
        monkeypatch.setattr(g, "_config", _FakeConfig())
        calls = []

        def fake_run_golden_test(ab_override=None, report_path=None):
            calls.append(ab_override["name"])
            n_pass = {"a": 2, "b": 1}[ab_override["name"]]
            return [
                _chunk_result("PASS" if (i + 1) <= n_pass else "REJECT", "jev", "jev")
                for i in range(2)
            ]

        monkeypatch.setattr(g, "run_golden_test", fake_run_golden_test)
        matrix = [
            {"name": "a", "role": "literal_translator", "effort": None},
            {"name": "b", "role": "primary", "effort": "max"},
        ]
        report = g.run_golden_ab(matrix=matrix, report_path=tmp_path / "ab.json")
        assert calls == ["a", "b"]
        assert report["combos"]["a"]["summary"]["pass_rate"] == 1.0
        assert report["combos"]["b"]["summary"]["pass_rate"] == 0.5
        assert report["recommendation"]["name"] == "a"
        assert (tmp_path / "ab.json").exists()

    def test_refuses_recommendation_in_degraded_env(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        monkeypatch.setattr(g, "__file__", str(src_dir / "test_golden_set.py"))
        monkeypatch.setattr(g, "_config", _FakeConfig())

        def fake_run_golden_test(ab_override=None, report_path=None):
            return [_chunk_result("PASS", "openai_compat", "openai_compat")]

        monkeypatch.setattr(g, "run_golden_test", fake_run_golden_test)
        matrix = [{"name": "m", "role": "primary", "effort": "max"}]
        report = g.run_golden_ab(matrix=matrix, report_path=tmp_path / "ab.json")
        assert report["recommendation"]["name"] is None
        assert report["recommendation"]["reason"] == "no_calibrated_combo"


# ---------------------------------------------------------------------------
# (5) golden-gate --ab 分支
# ---------------------------------------------------------------------------

class TestGoldenGateAb:
    def test_gate_rejects_uncalibrated_ab(self, monkeypatch):
        import src.test_memory_pressure as mp
        monkeypatch.setattr(mp, "run_golden_ab", lambda: {
            "recommendation": {"name": None, "reason": "no_calibrated_combo"},
        })
        assert mp.run_golden_gate(use_ab=True) is False

    def test_gate_uses_recommended_combo_results(self, monkeypatch):
        import src.test_memory_pressure as mp
        results = [{"chunk_id": "c1", "judge_decision": "PASS", "critic_scores": {"a": 8.0}}]
        monkeypatch.setattr(mp, "run_golden_ab", lambda: {
            "recommendation": {"name": "x"},
            "combos": {"x": {"results": results}},
        })

        def _must_not_run(*args, **kwargs):
            raise AssertionError("use_ab=True 时不应再跑单轮 run_golden_test")

        monkeypatch.setattr(mp, "run_golden_test", _must_not_run)
        assert mp.run_golden_gate(use_ab=True) is True


# ---------------------------------------------------------------------------
# (6) CLI 接线
# ---------------------------------------------------------------------------

class TestCliWiring:
    def test_ab_flag_wired_in_both_entrypoints(self):
        for path in ("src/main.py", "src/translator_agent.py"):
            src = Path(path).read_text(encoding="utf-8")
            assert 'parser.add_argument("--ab"' in src, path
            assert "run_golden_ab(" in src, path
            assert "use_ab=args.ab" in src, path
