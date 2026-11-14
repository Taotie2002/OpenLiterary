"""Round 150: conservative, non-LLM repair of cross-person name contamination."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repair_name_contamination.py"
_SPEC = importlib.util.spec_from_file_location("repair_name_contamination", _SCRIPT)
repair = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(repair)


def _write(path: Path, text: str, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "text": text,
        "metadata": metadata or {},
    }, ensure_ascii=False), encoding="utf-8")


def test_positional_name_repair_is_exact_and_keeps_full_name():
    final = "盖乌斯看着海伦。盖乌斯·海伦·莫希阿姆说：该死！"
    literary = "杰西卡看着尊母。盖乌斯·海伦·莫希阿姆说：该死！"
    repaired, changes = repair._repair_text(final, literary)
    assert repaired == literary
    assert [(c["from"], c["to"]) for c in changes] == [
        ("盖乌斯", "杰西卡"), ("海伦", "尊母"),
    ]


def test_repair_refuses_name_count_mismatch():
    assert repair._repair_text("盖乌斯和盖乌斯夫人。", "杰西卡。") is None


def test_repair_refuses_nonsuspect_sequence_mismatch():
    assert repair._repair_text("盖乌斯看着老妇人。", "杰西卡看着尊母。") is None


def test_dry_run_does_not_modify_final_and_apply_creates_backup(tmp_path, monkeypatch):
    root = tmp_path
    final_path = root / "output" / "ch01" / "ch01_chunk000_final.json"
    _write(final_path, "盖乌斯看着海伦。", {"name_unification_patched": True})
    _write(root / "output" / "ch01" / "ch01_chunk000_literary.json",
           "杰西卡看着尊母。")
    args = [
        "repair_name_contamination.py",
        "--root", str(root),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert repair.main() == 0
    assert json.loads(final_path.read_text(encoding="utf-8"))["text"] == "盖乌斯看着海伦。"

    monkeypatch.setattr(sys, "argv", args + ["--apply"])
    assert repair.main() == 0
    data = json.loads(final_path.read_text(encoding="utf-8"))
    assert data["text"] == "杰西卡看着尊母。"
    assert data["metadata"]["name_contamination_repair"]["replacements"]
    backups = list((root / "output" / "repair_backups").glob("name_contamination_*/*/*.json"))
    assert len(backups) == 1
