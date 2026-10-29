"""Round 133: no_content 重试适配 — 自动降档 / 换角色救稿 / override 持久化。

背景：ch01 通宵跑实测 7 块 PERMANENTLY_FAILED 全部是 effort=max 思考链
（5-7 万字符）吃光 max_tokens、正文为空（reasoning-only dump）；同参数重试
必然复现。修复：
  1. 命中「仅推理无正文」签名 → 下次重试自动降一档（max→xhigh→high→…）；
  2. 重试预算耗尽仍未成功 → 换角色救稿 1 次（RAW→secondary；minimax_protect）；
  3. 降档后因质量重进润色 → 升档补偿 1 次（防死循环）；
  4. override 按块持久化（workflow.db.retry_effort_overrides 列）。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config import config as _cfg
from src.core.scheduler import TaskState


def _make_pipeline_stub(tmp_path):
    """构造 TranslationPipeline stub（不触发 __init__，隔离根目录）。"""
    from src.core.scheduler import TaskScheduler
    from src.pipeline import TranslationPipeline

    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = tmp_path
    pipeline.chapter_id = "ch_test"
    pipeline._root_dir = tmp_path  # 隔离 model_capabilities.json 探测
    pipeline.scheduler = TaskScheduler(db_path=str(tmp_path / "workflow.db"))
    return pipeline


@pytest.fixture
def pipeline(tmp_path):
    p = _make_pipeline_stub(tmp_path)
    yield p
    try:
        p.scheduler.conn.close()
    except Exception:
        pass


@pytest.fixture
def retry_adapt_cfg():
    """临时覆盖 pipeline.retry_adaptation 配置并恢复。"""
    pipe = _cfg._data.setdefault("pipeline", {})
    backup = dict(pipe.get("retry_adaptation", {}) or {})
    pipe["retry_adaptation"] = dict(backup)

    def _apply(**kw):
        pipe["retry_adaptation"].update(kw)

    yield _apply
    pipe["retry_adaptation"] = backup


def _insert_failed(pipeline, chunk_id, retries, last_error, state="FAILED"):
    cur = pipeline.scheduler.conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO chunk_tasks (chunk_id, chapter_id, text_content, state, retries, last_error) "
        "VALUES (?, 'ch_test', '源文本', ?, ?, ?)", (chunk_id, state, retries, last_error))
    pipeline.scheduler.conn.commit()


def _state_of(pipeline, chunk_id):
    row = pipeline.scheduler.conn.execute(
        "SELECT state FROM chunk_tasks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return row[0]


# ────────────────────────────────────────────────────────────────────
# 签名判定
# ────────────────────────────────────────────────────────────────────
class TestNoContentSignature:
    def test_positive_reasoning_only(self, pipeline):
        msg = ("模型仅输出推理、无正文（reasoning=64435 chars）——"
               "疑似 thinking 耗尽 max_tokens，判定调用失败")
        assert pipeline._is_no_content_signature(msg) is True

    def test_positive_unclosed_think(self, pipeline):
        assert pipeline._is_no_content_signature("模型输出为未闭合 think 块（疑推理截断），判定调用失败") is True

    def test_negative(self, pipeline):
        assert pipeline._is_no_content_signature(
            "HTTPSConnectionPool(host='token.ai-hf.cn', port=443): Read timed out. (read timeout=300)") is False
        assert pipeline._is_no_content_signature("") is False
        assert pipeline._is_no_content_signature(None) is False


# ────────────────────────────────────────────────────────────────────
# 档位阶梯（降档 / 升档）
# ────────────────────────────────────────────────────────────────────
class TestEffortLadder:
    def test_minimax_chain(self, pipeline):
        assert pipeline._downgrade_one_step("primary", "max") == "xhigh"
        assert pipeline._downgrade_one_step("primary", "xhigh") == "high"
        assert pipeline._downgrade_one_step("primary", "low") is None

    def test_deepseek_chain(self, pipeline):
        assert pipeline._downgrade_one_step("secondary", "max") == "xhigh"
        assert pipeline._downgrade_one_step("secondary", "none") is None

    def test_unknown_effort_or_provider(self, pipeline):
        assert pipeline._downgrade_one_step("secondary", "bogus") is None
        assert pipeline._downgrade_one_step("secondary", None) is None
        assert pipeline._effort_ladder_for_role("nonexistent_role") == ()

    def test_escalate(self, pipeline):
        assert pipeline._escalate_one_step("primary", "xhigh") == "max"
        assert pipeline._escalate_one_step("primary", "max") is None
        assert pipeline._escalate_one_step("secondary", "none") == "low"

    def test_probe_file_filters_ladder(self, pipeline, monkeypatch):
        """探测文件把 primary 的 max 标为不可用 → 阶梯过滤掉 max。"""
        caps = {"models": {"primary": {"provider": "minimax", "levels": {
            "low": {"accepted": True}, "medium": {"accepted": True},
            "high": {"accepted": True}, "xhigh": {"accepted": True},
            "max": {"accepted": False},
        }}}}
        monkeypatch.setattr(pipeline, "_load_model_capabilities", lambda: caps)
        assert pipeline._effort_ladder_for_role("primary") == ("low", "medium", "high", "xhigh")
        assert pipeline._downgrade_one_step("primary", "xhigh") == "high"


# ────────────────────────────────────────────────────────────────────
# override 持久化（workflow.db）
# ────────────────────────────────────────────────────────────────────
class TestOverridePersistence:
    def test_roundtrip(self, pipeline):
        _insert_failed(pipeline, "ch_test_chunk000", 0, None)
        payload = {"sig": "no_content", "stage": "TRANSLATING_RAW", "role": "primary", "effort": "xhigh"}
        assert pipeline.scheduler.set_retry_override("ch_test_chunk000", payload) is True
        got = pipeline.scheduler.get_retry_override("ch_test_chunk000")
        assert got["effort"] == "xhigh" and got["sig"] == "no_content"
        assert pipeline.scheduler.clear_retry_override("ch_test_chunk000") is True
        assert pipeline.scheduler.get_retry_override("ch_test_chunk000") is None

    def test_missing_chunk_or_record(self, pipeline):
        assert pipeline.scheduler.get_retry_override("no_such_chunk") is None
        assert pipeline.scheduler.set_retry_override("no_such_chunk", {"sig": "x"}) is False

    def test_force_init_clears_override(self, pipeline):
        _insert_failed(pipeline, "ch_test_chunk000", 0, None, state="PENDING")
        pipeline.scheduler.set_retry_override("ch_test_chunk000", {"sig": "no_content"})
        pipeline.scheduler.init_chapter_tasks("ch_test", ["新文本"], force=True)
        assert pipeline.scheduler.get_retry_override("ch_test_chunk000") is None


# ────────────────────────────────────────────────────────────────────
# 签名记录 → 降档 override
# ────────────────────────────────────────────────────────────────────
class TestRecordSignal:
    def test_record_writes_downgrade(self, pipeline):
        _insert_failed(pipeline, "ch_test_chunk000", 0, None)
        pipeline._note_last_call("ch_test_chunk000", "TRANSLATING_RAW", "primary", "max", "MiniMax-M3.1-Flash")
        pipeline._record_no_content_signal("ch_test_chunk000", "TRANSLATING_RAW")
        ov = pipeline.scheduler.get_retry_override("ch_test_chunk000")
        assert ov["effort"] == "xhigh" and ov["role"] == "primary"
        assert ov["used_effort"] == "max" and ov["down_delta"] == 1
        assert ov["rescue_used"] is False and ov["stage"] == "TRANSLATING_RAW"

    def test_chained_record_descends(self, pipeline):
        _insert_failed(pipeline, "ch_test_chunk000", 0, None)
        pipeline._note_last_call("ch_test_chunk000", "REWRITING_LITERARY", "secondary", "max")
        pipeline._record_no_content_signal("ch_test_chunk000", "REWRITING_LITERARY")
        pipeline._note_last_call("ch_test_chunk000", "REWRITING_LITERARY", "secondary", "xhigh")
        pipeline._record_no_content_signal("ch_test_chunk000", "REWRITING_LITERARY")
        ov = pipeline.scheduler.get_retry_override("ch_test_chunk000")
        assert ov["effort"] == "high" and ov["down_delta"] == 2

    def test_disabled_config_no_override(self, pipeline, retry_adapt_cfg):
        retry_adapt_cfg(enabled=False)
        _insert_failed(pipeline, "ch_test_chunk000", 0, None)
        pipeline._note_last_call("ch_test_chunk000", "TRANSLATING_RAW", "primary", "max")
        pipeline._record_no_content_signal("ch_test_chunk000", "TRANSLATING_RAW")
        assert pipeline.scheduler.get_retry_override("ch_test_chunk000") is None


# ────────────────────────────────────────────────────────────────────
# 换角色救稿（重试预算耗尽）
# ────────────────────────────────────────────────────────────────────
class TestRescue:
    def test_raw_role_matrix(self, pipeline):
        assert pipeline._rescue_role_for("TRANSLATING_RAW", "primary") == "secondary"
        assert pipeline._rescue_role_for("TRANSLATING_RAW", "secondary") == "literal_translator"
        assert pipeline._rescue_role_for("TRANSLATING_RAW", "") == "secondary"

    def test_rewrite_role_matrix(self, pipeline):
        assert pipeline._rescue_role_for("REWRITING_LITERARY", "secondary") == "primary"
        assert pipeline._rescue_role_for("REWRITING_LITERARY", "primary") == "secondary"
        assert pipeline._rescue_role_for("REWRITING_LITERARY", "") == "secondary"

    def test_minimax_protect_off(self, pipeline, retry_adapt_cfg):
        retry_adapt_cfg(minimax_protect=False)
        assert pipeline._rescue_role_for("TRANSLATING_RAW", "primary") == "secondary"
        assert pipeline._rescue_role_for("TRANSLATING_RAW", "secondary") == "literal_translator"

    def test_rescue_effort_mapping(self, pipeline):
        # xhigh → 目标家映射后同档 xhigh → 再降一档 = high
        assert pipeline._rescue_effort_for("secondary", "xhigh") == "high"
        assert pipeline._rescue_effort_for("primary", "high") == "medium"
        assert pipeline._rescue_effort_for("secondary", None) is None

    def test_maybe_rescue_end_to_end(self, pipeline):
        cid = "ch_test_chunk000"
        err = ("[TRANSLATING_RAW] 模型仅输出推理、无正文（reasoning=64435 chars）——"
               "疑似 thinking 耗尽 max_tokens，判定调用失败")
        _insert_failed(pipeline, cid, 3, err)
        pipeline._note_last_call(cid, "TRANSLATING_RAW", "primary", "high", "MiniMax-M3.1-Flash")
        pipeline._record_no_content_signal(cid, "TRANSLATING_RAW")
        tasks = pipeline.scheduler.get_tasks_by_state(TaskState.FAILED, 10)
        assert len(tasks) == 1
        stage = pipeline._maybe_rescue_no_content(tasks[0])
        assert stage == "TRANSLATING_RAW"
        ov = pipeline.scheduler.get_retry_override(cid)
        assert ov["rescue_used"] is True and ov["role"] == "secondary"
        assert ov["effort"] == pipeline._rescue_effort_for("secondary", "high")
        # 救稿已用过 → 再判即放弃
        tasks2 = pipeline.scheduler.get_tasks_by_state(TaskState.FAILED, 10)
        assert pipeline._maybe_rescue_no_content(tasks2[0]) is None

    def test_no_signature_no_rescue(self, pipeline):
        cid = "ch_test_chunk001"
        _insert_failed(pipeline, cid, 3, "[TRANSLATING_RAW] Read timed out. (read timeout=300)")
        tasks = pipeline.scheduler.get_tasks_by_state(TaskState.FAILED, 10)
        assert pipeline._maybe_rescue_no_content(tasks[0]) is None


# ────────────────────────────────────────────────────────────────────
# _process_failed_batch 集成：救稿而非 PF
# ────────────────────────────────────────────────────────────────────
class TestFailedBatchIntegration:
    def test_rescue_instead_of_pf(self, pipeline):
        cid = "ch_test_chunk000"
        err = "[TRANSLATING_RAW] 模型仅输出推理、无正文（reasoning=64435 chars）——判定调用失败"
        _insert_failed(pipeline, cid, 3, err)
        pipeline._try_fallback_final = MagicMock(return_value=False)
        tasks = pipeline.scheduler.get_tasks_by_state(TaskState.FAILED, 10)
        pipeline._process_failed_batch(tasks)
        assert _state_of(pipeline, cid) == "TRANSLATING_RAW"
        assert pipeline.scheduler.get_retry_override(cid)["rescue_used"] is True

    def test_pf_when_no_signature(self, pipeline):
        cid = "ch_test_chunk002"
        _insert_failed(pipeline, cid, 3, "[TRANSLATING_RAW] Read timed out. (read timeout=300)")
        pipeline._try_fallback_final = MagicMock(return_value=False)
        tasks = pipeline.scheduler.get_tasks_by_state(TaskState.FAILED, 10)
        pipeline._process_failed_batch(tasks)
        assert _state_of(pipeline, cid) == "PERMANENTLY_FAILED"
        assert pipeline.scheduler.get_retry_override(cid) is None


# ────────────────────────────────────────────────────────────────────
# 单体镜像一致性
# ────────────────────────────────────────────────────────────────────
class TestMonolithMirror:
    def test_monolith_has_round133_methods(self):
        from src.translator_agent import TranslationPipeline as Mono
        for name in (
            "_is_no_content_signature", "_record_no_content_signal", "_note_last_call",
            "_maybe_rescue_no_content", "_rescue_role_for", "_rescue_effort_for",
            "_downgrade_one_step", "_escalate_one_step", "_effort_ladder_for_role",
            "_retry_adapt_cfg", "_load_model_capabilities",
        ):
            assert hasattr(Mono, name), f"monolith 缺少 {name}（需重建 build_integrated）"

    def test_monolith_scheduler_has_override_methods(self):
        from src.core.scheduler import TaskScheduler as ModSched
        from src.translator_agent import TaskScheduler as MonoSched
        for cls in (ModSched, MonoSched):
            for name in ("get_retry_override", "set_retry_override", "clear_retry_override"):
                assert hasattr(cls, name), f"{cls} 缺少 {name}"
