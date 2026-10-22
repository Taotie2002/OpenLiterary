#!/usr/bin/env python3
"""Round 116: Jev 配对仲裁 CLI（bench 工具，不接线主流水线）。

对同一 source/raw 的两个候选润色稿（A/B）跑同一组 typesafe primitives，
输出每维胜负、分歧清单与总体建议（JSON 可选落盘）。

用法：
  python3 scripts/jev_pairwise.py --source src.txt --raw raw.txt --a cand_deepseek.txt --b cand_minimax.txt \
      [--entity-registry registry.json] [--min-confidence 0.5] [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Jev pairwise arbitration bench")
    ap.add_argument("--source", required=True, help="原文文件")
    ap.add_argument("--raw", required=True, help="直译底稿文件")
    ap.add_argument("--a", required=True, help="候选 A 文件（如 DeepSeek 润色稿）")
    ap.add_argument("--b", required=True, help="候选 B 文件（如 MiniMax 润色稿）")
    ap.add_argument("--entity-registry", default=None, help="JSON 文件: {source_name: canonical}")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="结果 JSON 落盘路径（可选）")
    ap.add_argument("--require-jev", action="store_true",
                    help="Jev 不可达时直接失败（禁止静默降级到 openai fallback）")
    args = ap.parse_args(argv)

    # 注意导入顺序：必须先导入真实 src.judges.* 子模块，再导入 translator_agent。
    # 原因：translator_agent.py Round 103 shim 会把"尚未导入"的 src.judges 替换为
    # 假 ModuleType（__path__=[]），此后 `from src.judges.pairwise import ...` 会报
    # "No module named 'src.judges.pairwise'; 'src.judges' is not a package"。
    from src.judges.pairwise import arbitrate_answers, summarize
    import src.config as _cfg_mod
    from src.translator_agent import FallbackBackend, build_state, build_question_set, get_backend

    def _evaluate(state):
        """精简 backend 探测：Jev → OpenAICompat，返回 (answers, backend_path, jev_error)。"""
        backend = get_backend()
        if isinstance(backend, FallbackBackend):
            jev_error = None
            if backend.primary.api_key:
                try:
                    return backend.primary.evaluate(state, questions, timeout=backend.timeout), "jev", None
                except Exception as e:
                    jev_error = f"{type(e).__name__}: {e}"
                    print(f"⚠️ [jev_pairwise] Jev 不可达: {jev_error}", file=sys.stderr)
            else:
                jev_error = "no TYPESAFE_API_KEY"
                print("⚠️ [jev_pairwise] 未配置 TYPESAFE_API_KEY, Jev 不可用", file=sys.stderr)
            if args.require_jev:
                raise SystemExit(f"❌ --require-jev: Jev 不可用（{jev_error}）")
            if backend.allow_openai_fallback:
                print("⚠️ [jev_pairwise] 降级到 openai_compat（结果不再是 Jev 仲裁）", file=sys.stderr)
                return backend.fallback.evaluate(state, questions, timeout=backend.timeout), "openai_compat", jev_error
            raise RuntimeError("Jev 不可达且未开启 openai fallback")
        return backend.evaluate(state, questions), type(backend).__name__, None

    style_guide = dict(_cfg_mod.config.style_guide or {})
    entity_registry = {}
    if args.entity_registry:
        entity_registry = json.loads(_read(args.entity_registry))

    source = _read(args.source)
    raw = _read(args.raw)
    cand_a = _read(args.a)
    cand_b = _read(args.b)

    questions = build_question_set()
    state_a = build_state(
        source_text=source, raw_translation=raw, literary_translation=cand_a,
        style_guide_stats=style_guide, entity_registry=entity_registry or None,
    )
    state_b = build_state(
        source_text=source, raw_translation=raw, literary_translation=cand_b,
        style_guide_stats=style_guide, entity_registry=entity_registry or None,
    )

    answers_a, backend_path_a, jev_error_a = _evaluate(state_a)
    answers_b, backend_path_b, jev_error_b = _evaluate(state_b)

    # 答案覆盖度警告：缺失 primitive 时静默跳过是最常见的误判来源
    prims = {"voice_drift", "register_match", "entity_consistency",
             "literal_residue", "semantic_loss", "style_compliance_score"}
    for label, ans in (("A", answers_a), ("B", answers_b)):
        miss = sorted(p for p in prims if p not in (ans or {}))
        if miss:
            print(f"⚠️ [jev_pairwise] 候选 {label} 缺失 primitive: {miss}", file=sys.stderr)

    result = arbitrate_answers(answers_a, answers_b, min_confidence=args.min_confidence)
    result["_backend_path"] = backend_path_a if backend_path_a == backend_path_b else f"{backend_path_a}|{backend_path_b}"
    jev_error = jev_error_a or jev_error_b
    if jev_error:
        result["_jev_error"] = jev_error
        print(f"⚠️ [jev_pairwise] 本次结果并非 Jev 仲裁（backend={result['_backend_path']}）: {jev_error}",
              file=sys.stderr)

    print(summarize(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
