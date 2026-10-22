"""Round 118 P2-6: 章节级漂移审计（Jev 滑动窗口 vs profile 语域映射）.

确定性统计层（离线可用）:
  - 收集 output/<chapter>/*_final.json 定稿文本, 按 chunk 序号排序;
  - 每 chunk 计算风格指标（平均句长 / CJK 比例 / 对话句占比 / 每千字文言词密度）;
  - 滑动窗口均值 vs book_profile 映射基线（avg_sentence_length / register_default）;
  - 首窗→末窗趋势 + 基线偏差超阈值 → drift_flags（附可执行说明）。

Jev 层（可选, CLI `--jev`）:
  - 相邻窗口各取代表片段, 作为 voice_drift / register_match 的配对输入,
    用同一 backend（Jev → fallback）评估窗口间口吻/语域是否漂移;
  - Jev 不可用时返回 available=False, 不影响统计层结论（设计原则: 审计不依赖网络）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.text_match import compute_cjk_ratio

_SENT_RE = re.compile(r"[。！？!?]+")
_LITERARY_RE = re.compile(r"[之乎矣焉其亦乃哉]")
_DIALOGUE_RE = re.compile(r"[「『“\"]")
_CHUNK_NUM_RE = re.compile(r"chunk(\d+)")


def collect_chunk_texts(chapter_dir: Path) -> List[Tuple[str, str]]:
    """收集章节下所有 *_final.json 的定稿文本, 按 chunk 序号升序返回。"""
    found: List[Tuple[int, str, str]] = []
    if not chapter_dir.exists():
        return []
    for path in chapter_dir.glob("*_final.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = data.get("text") if isinstance(data, dict) else None
        if not text or not isinstance(text, str):
            continue
        m = _CHUNK_NUM_RE.search(path.stem)
        num = int(m.group(1)) if m else 0
        chunk_id = path.stem[: m.start() + len(m.group(0))] if m else path.stem
        found.append((num, chunk_id, text))
    found.sort(key=lambda item: item[0])
    return [(cid, text) for _n, cid, text in found]


def chunk_style_metrics(text: str) -> Dict[str, float]:
    """单 chunk 风格指标（确定性, 无 LLM）。"""
    if not text:
        return {
            "avg_sentence_len": 0.0,
            "cjk_ratio": 0.0,
            "dialogue_ratio": 0.0,
            "literary_density": 0.0,
        }
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    n_chars = len(text)
    return {
        "avg_sentence_len": n_chars / max(1, len(sentences)),
        "cjk_ratio": round(compute_cjk_ratio(text), 4),
        "dialogue_ratio": round(len(_DIALOGUE_RE.findall(text)) / max(1, len(sentences)), 4),
        "literary_density": round(len(_LITERARY_RE.findall(text)) / max(1, n_chars) * 1000.0, 4),
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_profile_guide(root: Path, chapter: str) -> Dict[str, Any]:
    try:
        from src.style_profiler import load_profile, profile_to_style_guide
        prof = load_profile(root / "input" / "book_profile.json")
        if not prof:
            return {}
        return dict(profile_to_style_guide(prof, chapter_id=chapter))
    except Exception:
        return {}


def audit_chapter(
    root: Path,
    chapter: str = "ch01",
    window: Optional[int] = None,
    profile_guide: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """统计层审计: 指标 + 窗口趋势 + 基线偏差, 返回结构化报告。"""
    cfg: Dict[str, Any] = {}
    try:
        from src.config import config as _config
        cfg = _config.pipeline.get("drift_audit", {}) or {}
    except Exception:
        cfg = {}
    win = int(window if window is not None else cfg.get("window", 5))
    win = max(1, win)
    trend_threshold = float(cfg.get("trend_threshold", 0.25))
    sentence_tol = float(cfg.get("sentence_len_tolerance", 0.30))

    chunks = collect_chunk_texts(root / "output" / chapter)
    per_chunk = [
        {"chunk_id": cid, "metrics": chunk_style_metrics(text)}
        for cid, text in chunks
    ]
    windows: List[Dict[str, Any]] = []
    for start in range(0, len(per_chunk), win):
        block = per_chunk[start:start + win]
        if not block:
            continue
        metrics = block[0]["metrics"].keys()
        windows.append({
            "range": [block[0]["chunk_id"], block[-1]["chunk_id"]],
            "size": len(block),
            "mean": {k: round(_mean([b["metrics"][k] for b in block]), 4) for k in metrics},
        })

    flags: List[Dict[str, Any]] = []
    if len(windows) >= 2:
        first, last = windows[0]["mean"], windows[-1]["mean"]
        for key in ("avg_sentence_len", "dialogue_ratio", "literary_density", "cjk_ratio"):
            base = abs(first.get(key, 0.0)) or 1e-9
            delta = (last.get(key, 0.0) - first.get(key, 0.0)) / base
            if abs(delta) >= trend_threshold:
                flags.append({
                    "type": "window_trend",
                    "metric": key,
                    "delta_ratio": round(delta, 3),
                    "detail": f"{key} 首窗 {first.get(key, 0.0):.2f} → 末窗 {last.get(key, 0.0):.2f}",
                })

    guide = profile_guide if profile_guide is not None else _load_profile_guide(root, chapter)
    target_len = guide.get("avg_sentence_length")
    actual_mean_len = _mean([c["metrics"]["avg_sentence_len"] for c in per_chunk])
    if target_len and actual_mean_len and per_chunk:
        try:
            target = float(target_len)
            if target > 0:
                dev = (actual_mean_len - target) / target
                if abs(dev) >= sentence_tol:
                    flags.append({
                        "type": "profile_deviation",
                        "metric": "avg_sentence_len",
                        "delta_ratio": round(dev, 3),
                        "detail": (
                            f"全书均值 {actual_mean_len:.1f} 与 profile 基线 {target:.1f} "
                            f"偏差 {dev*100:.0f}%"
                        ),
                    })
        except (TypeError, ValueError):
            pass

    return {
        "chapter": chapter,
        "chunk_count": len(per_chunk),
        "window": win,
        "per_chunk": per_chunk,
        "windows": windows,
        "drift_flags": flags,
        "profile": {
            "source": guide.get("_profile_source"),
            "register_default": guide.get("register_default"),
            "avg_sentence_length_target": guide.get("avg_sentence_length"),
        },
    }


def jev_window_check(
    chunks: List[Tuple[str, str]],
    window: int = 5,
    excerpt_chars: int = 600,
) -> Dict[str, Any]:
    """Jev 层: 相邻窗口配对评估 voice_drift / register_match（可选, 需要 backend）。"""
    if len(chunks) < 2:
        return {"available": False, "error": "chunks < 2"}
    try:
        from src.judges.judge import build_state
        from src.judges.primitives import build_question_set
        from src.judges.typesafe_backend import get_backend
        from src.core.decision_engine import DecisionEngine  # noqa: F401  仅确认依赖可导入
    except Exception as e:
        return {"available": False, "error": f"import failed: {type(e).__name__}: {e}"}
    try:
        backend = get_backend()
        primary_key = getattr(getattr(backend, "primary", None), "api_key", None)
        if primary_key is None:
            return {"available": False, "error": "backend 无 Jev 主链 (可能是单后端 mock)"}
        if not primary_key:
            return {"available": False, "error": "no TYPESAFE_API_KEY"}
    except Exception as e:
        return {"available": False, "error": f"backend init failed: {type(e).__name__}: {e}"}

    texts = [t for _cid, t in chunks]
    blocks = [texts[i:i + window] for i in range(0, len(texts), window)]
    questions = build_question_set()
    pairs: List[Dict[str, Any]] = []
    for i in range(len(blocks) - 1):
        left = "".join(blocks[i])[:excerpt_chars]
        right = "".join(blocks[i + 1])[:excerpt_chars]
        if not left or not right:
            continue
        # 复用 voice_drift/register_match 语义: 把左窗当作"基线稿"、右窗当作"待查稿"
        state = build_state(
            source_text=left, raw_translation=left,
            literary_translation=right, style_guide_stats={},
        )
        try:
            answers = backend.evaluate(state, questions)
        except Exception as e:
            pairs.append({"pair": i, "error": f"{type(e).__name__}: {e}"})
            continue
        voice = answers.get("voice_drift")
        register = answers.get("register_match")
        entry: Dict[str, Any] = {"pair": i}
        drifted = False
        if voice is not None:
            probs = getattr(voice, "probabilities", {}) or {}
            entry["voice_severe_prob"] = probs.get("severe", 0.0)
            drifted = drifted or probs.get("severe", 0.0) >= 0.5
        if register is not None:
            probs = getattr(register, "probabilities", {}) or {}
            mismatch = probs.get("under", 0.0) + probs.get("over", 0.0)
            entry["register_mismatch"] = round(mismatch, 3)
            drifted = drifted or mismatch >= 0.4
        entry["drifted"] = drifted
        pairs.append(entry)
    return {
        "available": True,
        "backend": getattr(backend, "last_backend", getattr(backend, "backend_kind", "unknown")),
        "pairs": pairs,
        "drifted_pairs": [p.get("pair") for p in pairs if p.get("drifted")],
    }


def summarize_report(report: Dict[str, Any]) -> str:
    """渲染审计报告为可读文本（CLI 输出）。"""
    lines = [
        f"章节漂移审计: {report.get('chapter')} | chunks={report.get('chunk_count')} "
        f"| window={report.get('window')}",
    ]
    prof = report.get("profile") or {}
    if prof.get("avg_sentence_length_target") or prof.get("register_default"):
        lines.append(
            f"profile 基线: 目标句长={prof.get('avg_sentence_length_target')}, "
            f"语域={prof.get('register_default') or '—'} (source={prof.get('source') or '—'})"
        )
    for w in report.get("windows") or []:
        mean = w["mean"]
        lines.append(
            f"  窗口 {w['range'][0]}..{w['range'][1]} (n={w['size']}): "
            f"句长≈{mean['avg_sentence_len']:.1f}, 文言密度={mean['literary_density']:.1f}/千字, "
            f"对话句比={mean['dialogue_ratio']:.2f}, CJK={mean['cjk_ratio']:.2f}"
        )
    flags = report.get("drift_flags") or []
    if flags:
        lines.append(f"⚠️ 漂移警告 {len(flags)} 条:")
        for f in flags:
            lines.append(f"  - [{f.get('type')}] {f.get('metric')}: {f.get('detail')}")
    else:
        lines.append("✅ 未检测到超阈值漂移")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """独立入口: python -m src.drift_audit --chapter ch01 [--window 5] [--jev] [--json out]"""
    import argparse
    parser = argparse.ArgumentParser(description="SagaTrans 章节漂移审计")
    parser.add_argument("--chapter", default="ch01")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--jev", action="store_true", help="附加 Jev 窗口配对检查（需要 Jev 可用）")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args(argv)
    root = Path.cwd()
    report = audit_chapter(root, chapter=args.chapter, window=args.window)
    if args.jev:
        chunks = collect_chunk_texts(root / "output" / args.chapter)
        report["jev"] = jev_window_check(chunks, window=args.window or report["window"])
        jv = report["jev"]
        if jv.get("available"):
            print(f"🔬 Jev 窗口检查 (backend={jv.get('backend')}): "
                  f"drifted_pairs={jv.get('drifted_pairs')}")
        else:
            print(f"⚠️ Jev 窗口检查不可用: {jv.get('error')}", file=sys.stderr)
    print(summarize_report(report))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[out] {args.json_out}")
    return 0 if not report.get("drift_flags") else 2


if __name__ == "__main__":
    raise SystemExit(main())
