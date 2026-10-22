"""Round 120 (L6): literary_metrics — 计算/剥离/基线对比。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location("literary_metrics", _ROOT / "scripts" / "literary_metrics.py")
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)


def test_strip_markup_removes_notes_and_headings():
    text = "# 似乎\n\n正文。\n\n[^1]: 似乎似乎\n"
    m = lm.compute_metrics(text)
    assert m["ai_flavor_counts"]["似乎"] == 0


def test_compute_metrics_counts_and_repeat():
    text = "他似乎微微地笑了笑他似乎微微地笑了笑。\n\n进行了一次尝试。\n"
    m = lm.compute_metrics(text)
    assert m["ai_flavor_counts"]["似乎"] == 2
    assert m["ai_flavor_counts"]["微微"] == 2
    assert m["ai_flavor_total"] == 4
    assert m["translationese_counts"]["进行一次"] == 1
    assert m["chars_cjk"] > 0
    assert m["ngram4"]["repeat_rate"] > 0


def test_quotes_and_baseline_compare():
    text = '他说："你好。"‘在的轻声叹气。’\n'
    m = lm.compute_metrics(text)
    assert m["quotes"]["ascii_double"] == 2
    baseline = {"ai_density_per_1k": m["ai_density_per_1k"] + 1.0,
                "translationese_total": 100, "quotes": {"ascii_double": 0},
                "notes": {"orphans": 0}, "ngram4": {"repeat_rate": 0.5}}
    rows = lm.compare_to_baseline(m, baseline)
    by = {r["metric"]: r for r in rows}
    assert by["ai_density_per_1k"]["worse"] is False
    assert by["translationese_total"]["worse"] is False
    assert by["quotes.ascii_double"]["worse"] is True
    assert by["ngram4.repeat_rate"]["worse"] is False
