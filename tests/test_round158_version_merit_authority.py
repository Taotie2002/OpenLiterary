"""Round 158: locked Decision authority outranks critic-mean version merit."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.config import config as _cfg
from src.core.decision_engine import DecisionEngine, DecisionLevel
from src.core.scheduler import TaskScheduler
from src.pipeline import TranslationPipeline


def _pipeline(tmp_path, source_text: str, engine: DecisionEngine):
    pipeline = TranslationPipeline.__new__(TranslationPipeline)
    pipeline.output_dir = tmp_path
    pipeline.chapter_id = "ch_test"
    pipeline.decision_engine = engine
    pipeline.scheduler = MagicMock(spec=TaskScheduler)
    pipeline.scheduler.get_task.return_value = {
        "chunk_id": "ch_test_chunk000",
        "text_content": source_text,
    }
    pipeline._score_history = {}
    pipeline._best_lit = {}
    return pipeline


def _write_scores(tmp_path, chunk_id: str, value: float, suffix: str = "") -> None:
    (tmp_path / f"{chunk_id}_critic_report{suffix}.json").write_text(
        json.dumps({"scores": {"fluency": value}}), encoding="utf-8")


def _locked(engine: DecisionEngine, source: str, target: str) -> None:
    engine.add_decision(
        DecisionLevel.REFERENCE, source, target, "test authority",
        affected_chunks=None,
    )
    engine.conn.execute(
        "UPDATE decision_db SET locked=1 WHERE source_key=?", (source,))
    engine.conn.commit()


def test_missing_locked_target_prevents_score_based_rollback(tmp_path):
    old_merit = dict(_cfg.pipeline.get("version_merit", {}) or {})
    try:
        _cfg.pipeline["version_merit"] = {"enabled": True, "tolerance": 0.0}
        engine = DecisionEngine(db_path=str(tmp_path / "decision.sqlite"))
        _locked(engine, "kindjal", "坎贾尔匕首")
        pipeline = _pipeline(tmp_path, "Halleck lifted a naked kindjal.", engine)
        chunk_id = "ch_test_chunk000"
        final = tmp_path / f"{chunk_id}_final.json"
        backup = tmp_path / f"{chunk_id}_final_backup.json"
        final.write_text(json.dumps({"text": "他拿起坎贾尔匕首。"}, ensure_ascii=False), encoding="utf-8")
        backup.write_text(json.dumps({"text": "他拿起金贾尔匕首。"}, ensure_ascii=False), encoding="utf-8")
        _write_scores(tmp_path, chunk_id, 8.0)
        _write_scores(tmp_path, chunk_id, 9.5, "_backup")

        restored = pipeline._apply_version_merit(chunk_id, backup, final)

        assert restored is False
        assert json.loads(final.read_text(encoding="utf-8"))["text"] == "他拿起坎贾尔匕首。"
    finally:
        _cfg.pipeline["version_merit"] = old_merit


def test_false_positive_locked_title_prevents_score_based_rollback(tmp_path):
    old_merit = dict(_cfg.pipeline.get("version_merit", {}) or {})
    try:
        _cfg.pipeline["version_merit"] = {"enabled": True, "tolerance": 0.0}
        engine = DecisionEngine(db_path=str(tmp_path / "decision.sqlite"))
        _locked(engine, "the One", "那一位")
        pipeline = _pipeline(
            tmp_path,
            "Your mother wanted me to be the one to tell you.",
            engine,
        )
        chunk_id = "ch_test_chunk000"
        final = tmp_path / f"{chunk_id}_final.json"
        backup = tmp_path / f"{chunk_id}_final_backup.json"
        final.write_text(json.dumps({"text": "你母亲想让我来告诉你。"}, ensure_ascii=False), encoding="utf-8")
        backup.write_text(
            json.dumps({"text": "你母亲想让我成为那一位来告诉你。"}, ensure_ascii=False),
            encoding="utf-8",
        )
        _write_scores(tmp_path, chunk_id, 8.0)
        _write_scores(tmp_path, chunk_id, 9.5, "_backup")

        restored = pipeline._apply_version_merit(chunk_id, backup, final)

        assert restored is False
        assert json.loads(final.read_text(encoding="utf-8"))["text"] == "你母亲想让我来告诉你。"
    finally:
        _cfg.pipeline["version_merit"] = old_merit

