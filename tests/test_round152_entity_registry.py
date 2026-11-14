"""Round 152: narrow repairs for known registry contamination."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "repair_entity_registry_round152.py")
_SPEC = importlib.util.spec_from_file_location("repair_entity_registry_round152", _SCRIPT)
repair = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(repair)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE entity_registry ("
        "id INTEGER PRIMARY KEY, canonical_name TEXT, source_names TEXT, "
        "variant_names TEXT, context_names TEXT, quarantined_names TEXT, "
        "pending_review INTEGER DEFAULT 0, review_reason TEXT)"
    )
    rows = [
        (1, "保罗·厄崔迪",
         ["Paul Atreides", "Duke Leto Atreides", "Duke Leto"],
         [], [], []),
        (2, "盖乌斯·海伦·莫希阿姆", ["Gaius Helen Mohiam"],
         ["圣母", "真言者"], ["Reverend Mother"], []),
        (3, "厄崔迪家族", ["House Atreides"], [], [], []),
        (4, "阿特雷迪斯家族", ["House Atreides"], [], [], []),
        (5, "阿特雷德家族", ["House Atreides"], [], [], []),
        (6, "亚崔迪家族", [], [], [], []),
    ]
    conn.executemany(
        "INSERT INTO entity_registry VALUES (?,?,?,?,?,?,0,NULL)",
        [(rid, name, json.dumps(src, ensure_ascii=False),
          json.dumps(var, ensure_ascii=False), json.dumps(ctx, ensure_ascii=False),
          json.dumps(q, ensure_ascii=False))
         for rid, name, src, var, ctx, q in rows],
    )
    conn.commit()
    return conn


def test_repairs_paul_mohiam_and_atreides_family():
    conn = _conn()
    result = repair.repair_registry(conn, dry_run=False)
    assert result["changed_rows"] == 5
    paul = conn.execute(
        "SELECT source_names, quarantined_names FROM entity_registry WHERE id=1"
    ).fetchone()
    assert json.loads(paul[0]) == ["Paul Atreides"]
    assert set(json.loads(paul[1])) == {"Duke Leto Atreides", "Duke Leto"}
    mohiam = conn.execute(
        "SELECT variant_names, context_names FROM entity_registry WHERE id=2"
    ).fetchone()
    assert json.loads(mohiam[0]) == []
    assert set(json.loads(mohiam[1])) >= {"Reverend Mother", "圣母", "真言者"}
    states = dict(conn.execute(
        "SELECT canonical_name, pending_review FROM entity_registry WHERE canonical_name LIKE '%家族'"
    ).fetchall())
    assert states == {
        "厄崔迪家族": 0, "阿特雷迪斯家族": 1,
        "阿特雷德家族": 1, "亚崔迪家族": 1,
    }


def test_dry_run_does_not_write():
    conn = _conn()
    result = repair.repair_registry(conn, dry_run=True)
    assert result["changed_rows"] == 5
    raw = conn.execute("SELECT source_names FROM entity_registry WHERE id=1").fetchone()[0]
    assert "Duke Leto" in json.loads(raw)
