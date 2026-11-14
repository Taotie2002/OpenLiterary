"""Round 153: select only calibrated low-average chunks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "requeue_low_critic.py")
_SPEC = importlib.util.spec_from_file_location("requeue_low_critic", _SCRIPT)
requeue = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(requeue)


def _report(root: Path, cid: str, avg_scores):
    chapter = cid.split("_", 1)[0]
    out = root / "output" / chapter
    out.mkdir(parents=True, exist_ok=True)
    payload = {"scores": {"x": avg_scores}, "_backend_path": "jev"}
    (out / f"{cid}_critic_report.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_selects_only_below_threshold_sorted_and_chapter_filtered(tmp_path):
    _report(tmp_path, "ch01_chunk001", 6.5)
    _report(tmp_path, "ch01_chunk002", 7.5)
    _report(tmp_path, "ch02_chunk001", 6.2)
    rows = requeue.select_candidates(tmp_path, 7.0, ["ch01"])
    assert [r["chunk_id"] for r in rows] == ["ch01_chunk001"]
    rows_all = requeue.select_candidates(tmp_path, 7.0)
    assert [r["chunk_id"] for r in rows_all] == ["ch02_chunk001", "ch01_chunk001"]
