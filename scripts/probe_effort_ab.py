#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 138: effort 档位 A/B 探测（默认 max vs xhigh）—— 润色 / 裁决 两条轨道。

目的：用**生产同构 prompt**回答"高 effort（max）相对 xhigh 的质量收益
是否值得它的时间成本"。对已完成章节的困难块做双档生成，再由独立裁判盲评。

用法（在仓库根目录运行）：
  # ① 离线自检：只选样本 + 构建两轨 prompt，不调任何 LLM
  python3 scripts/probe_effort_ab.py rewrite --dry-run

  # ② 润色轨道（DeepSeek secondary；不占 MiniMax 配额，安全）
  ALL_PROXY=socks5h://127.0.0.1:10808 python3 scripts/probe_effort_ab.py rewrite --n 12

  # ③ 裁决轨道（MiniMax primary；默认关闭，须显式 --minimax，注意配额）
  ALL_PROXY=socks5h://127.0.0.1:10808 python3 scripts/probe_effort_ab.py judge --minimax --n 6

不干扰主流程的保证：
  - db/workflow.db 只读；db/decision_db.sqlite 用临时拷贝（绝不写生产库）
  - 不写 output/ / db/ / input/；结果全部落 /tmp/sagatrans_effort_ab/<时间戳>/
  - 顺序单发调用 + 可调间隔（--sleep），避免与主流程抢网关
  - judge 轨道直接消耗 MiniMax 配额（主流程同样依赖），故默认禁用、默认 n 小

统计口径提醒：N 小（10 量级）只给方向性证据；"max 零胜"即可认定为无增益，
"max 胜率≥60%"才值得继续保留 max。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.config import config as _config  # noqa: E402
from src.utils.llm_adapter import get_role_model_name, robust_json_loads  # noqa: E402

WORKFLOW_DB = ROOT / "db" / "workflow.db"
DECISION_DB = ROOT / "db" / "decision_db.sqlite"
OUT_ROOT = Path(os.environ.get("SAGATRANS_PROBE_OUT", "/tmp/sagatrans_effort_ab"))
TMP_DB_DIR = Path(tempfile.mkdtemp(prefix="sagatrans_effort_ab_db_"))

_DE_COPY = None  # DecisionEngine on temp copy（全局复用）


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def get_de():
    """DecisionEngine(临时拷贝)；绝不触碰生产 decision_db。"""
    global _DE_COPY
    if _DE_COPY is None:
        from src.core.decision_engine import DecisionEngine
        copy_path = TMP_DB_DIR / "decision_db.sqlite"
        shutil.copy(str(DECISION_DB), str(copy_path))
        _DE_COPY = DecisionEngine(db_path=str(copy_path))
    return _DE_COPY


def set_chapter_style(chapter_id: str) -> dict:
    """复刻 pipeline 启动逻辑：book_profile → 运行时 style_guide（按章）。"""
    try:
        from src.style_profiler import load_profile, profile_to_style_guide
        prof = load_profile(ROOT / "input" / "book_profile.json")
        if prof:
            _config.set_runtime_style_guide(profile_to_style_guide(prof, chapter_id=chapter_id))
    except Exception as exc:
        log(f"⚠️ style profile 加载失败（用 config 基线）: {exc}")
    return dict(_config.style_guide or {})


def build_entity_registry(source_text: str) -> dict:
    """简化版 _build_entity_registry_for_rewriter：只按"出现在本块原文"过滤。

    注：未复刻 Round 111/112/133b 的全部过滤器（对双档 A/B 公平性无影响，
    两份 prompt 注入同一登记表；绝对保真度略低于生产）。
    """
    from src.core.decision_engine import query_reviewable_entities
    out: dict = {}
    try:
        rows = query_reviewable_entities(get_de().conn)
    except Exception:
        return out
    low = source_text.lower()
    for row in rows:
        canonical = row[0]
        names_json = row[1]
        names = robust_json_loads(names_json, expected_type=list) if names_json else []
        for sn in names:
            if not sn or not isinstance(sn, str) or sn in out:
                continue
            if sn in source_text or (len(sn) >= 3 and sn.lower() in low):
                out[sn] = canonical
    return out


# ─────────────────────────── 样本选择 ───────────────────────────

def _ref_keys() -> list:
    keys = []
    try:
        for (sk,) in get_de().conn.execute("SELECT source_key FROM decision_db WHERE level = 2"):
            if sk:
                keys.append(str(sk))
    except Exception:
        pass
    return keys


def _count_ref_hits(source_text: str, ref_keys: list) -> int:
    src_lower = source_text.lower()
    hits = 0
    for k in ref_keys:
        if k in source_text:
            hits += 1
        elif len(k) >= 3 and k.lower() in src_lower:
            hits += 1
    return hits


def chunk_difficulty(source_text: str, ref_keys: list) -> float:
    """复刻 Pipeline._chunk_difficulty 公式（Round 118 P2-5）。"""
    if not source_text:
        return 0.0
    score = float(min(3, _count_ref_hits(source_text, ref_keys)))
    sentences = [s for s in re.split(r"[.!?]+\s*", source_text) if s.strip()]
    if sentences:
        long_ratio = sum(1 for s in sentences if len(s.split()) >= 35) / len(sentences)
        if long_ratio >= 0.3:
            score += 1.5
        elif long_ratio >= 0.15:
            score += 0.75
    for m in re.finditer(r"[\"\u201c]([^\"\u201d]{20,})[\"\u201d]", source_text):
        if len(m.group(1).split()) >= 6:
            score += 1.0
            break
    return score


def select_samples(track: str, n: int, chapters: list, only_chunks: list) -> list:
    """从 COMPLETED 块中按难度排序选样本；track=rewrite 需 raw，judge 需 critic。"""
    ref_keys = _ref_keys()
    conn = sqlite3.connect(f"file:{WORKFLOW_DB}?mode=ro", uri=True)
    ph = ",".join("?" * len(chapters))
    rows = conn.execute(
        f"SELECT chunk_id, chapter_id, text_content, state, retries, quality_retries "
        f"FROM chunk_tasks WHERE chapter_id IN ({ph}) AND state='COMPLETED'",
        chapters,
    ).fetchall()
    conn.close()
    pool = []
    for cid, ch, src, state, retries, qretries in rows:
        if only_chunks and cid not in only_chunks:
            continue
        if not src:
            continue
        outdir = ROOT / "output" / ch
        if not (outdir / f"{cid}_final.json").exists():
            continue
        if track == "rewrite" and not (outdir / f"{cid}_raw.json").exists():
            continue
        if track == "judge" and not (outdir / f"{cid}_critic_report.json").exists():
            continue
        diff = chunk_difficulty(src, ref_keys)
        rank = diff * 10 + (retries + qretries) * 2
        pool.append({
            "chunk": cid, "chapter": ch, "source": src,
            "difficulty": round(diff, 2), "retries": retries, "quality_retries": qretries,
            "rank": rank,
        })
    pool.sort(key=lambda x: (-x["rank"], -x["difficulty"], x["chunk"]))
    # Round 138: 章节轮转取样，避免样本全部落在同一章
    from collections import defaultdict, deque
    buckets = defaultdict(deque)
    for s_ in pool:
        buckets[s_["chapter"]].append(s_)
    chosen = []
    ch_order = sorted(buckets)
    while len(chosen) < n and any(buckets.values()):
        for ch in ch_order:
            if buckets[ch] and len(chosen) < n:
                chosen.append(buckets[ch].popleft())
    return chosen


def load_prod_json(chapter: str, chunk: str, kind: str):
    path = ROOT / "output" / chapter / f"{chunk}_{kind}.json"
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


# ─────────────────────────── 盲评裁判 ───────────────────────────

def jev_pairwise(source: str, raw_text: str, style: dict, reg: dict,
                  text_a: str, text_b: str) -> dict:
    """生产 dual-draft 同款 Jev 配对仲裁（调用方负责把 label 映射回 effort）。"""
    from src.judges.judge import build_state
    from src.judges.primitives import build_question_set
    from src.judges.typesafe_backend import get_backend
    from src.judges.pairwise import arbitrate_answers

    cfg = _config.pipeline.get("pairwise_arbitration", {}) or {}
    backend = get_backend()
    questions = build_question_set()
    timeout = getattr(backend, "timeout", 30.0)
    state_a = build_state(source_text=source, raw_translation=raw_text or "",
                          literary_translation=text_a, style_guide_stats=style,
                          entity_registry=reg)
    state_b = build_state(source_text=source, raw_translation=raw_text or "",
                          literary_translation=text_b, style_guide_stats=style,
                          entity_registry=reg)
    ans_a = backend.evaluate(state_a, questions, timeout=timeout)
    ans_b = backend.evaluate(state_b, questions, timeout=timeout)
    res = arbitrate_answers(ans_a, ans_b, min_confidence=float(cfg.get("min_confidence", 0.6)))
    res["backend"] = getattr(backend, "last_backend",
                             getattr(backend, "backend_kind", type(backend).__name__))
    return res


def critic_mean_score(source: str, raw_text: str, style: dict, draft: str):
    """兜底裁判：生产 critic prompt + critic_scoring 任务参数（DS, effort=none）。"""
    from src.agents.critic_agent import CriticAgent
    critic = CriticAgent()
    prompt = critic._build_prompt(source, raw_text, draft, style)
    model_key, params = _config.resolve_task_model("critic_scoring")
    model_name = get_role_model_name(critic.role) or model_key
    out = critic.llm.generate(prompt=prompt, model_name=model_name, **params)
    data = robust_json_loads(out) or {}
    scores = data.get("scores") or {}
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return (sum(vals) / len(vals)) if vals else None, scores


# ─────────────────────────── 轨道执行 ───────────────────────────

def preflight_efforts(llm, model_name: str, efforts: list) -> dict:
    """Round 138: 档位预检 — 确认网关照单执行，而非 400-剥离成无档位调用。

    检测方式：捕获调用期间 stderr 是否出现「剥离后重试」警告（adapter 在服务端
    拒绝 reasoning_effort 时会打印并去掉该参数重发；此时 A/B 两档实际等价，探测无意义）。
    """
    import contextlib
    import io

    results = {}
    prompt = "请逐步推理后作答：17 × 23 = ?"
    for eff in ["none"] + list(efforts):
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stderr(buf):
                out = llm.generate(prompt=prompt, model_name=model_name,
                                   temperature=0.1, max_tokens=3000, effort=eff)
            dt = time.perf_counter() - t0
            results[eff] = {"ok": True, "stripped": ("剥离后重试" in buf.getvalue()),
                            "reasoning_chars": len(getattr(llm, "last_reasoning", "") or ""),
                            "latency": round(dt, 1), "out_len": len(out or "")}
        except Exception as exc:
            results[eff] = {"ok": False, "stripped": ("剥离后重试" in buf.getvalue()),
                            "err": f"{type(exc).__name__}: {exc}"}
        log(f"  预检 effort={eff}: " + json.dumps(results[eff], ensure_ascii=False))
        time.sleep(1.0)
    return results


def _preflight_bad(pf: dict, efforts: list) -> list:
    return [e for e in efforts
            if not (pf.get(e, {}).get("ok") and not pf[e].get("stripped"))]


def run_rewrite_track(args, samples: list, outdir: Path) -> list:
    from src.agents.rewriter_agent import LiteraryRewriterAgent
    agent = LiteraryRewriterAgent(get_de())
    model_name = get_role_model_name(agent.role)
    if not args.ignore_preflight:
        pf = preflight_efforts(agent.llm, model_name, args.efforts)
        (outdir / "preflight.json").write_text(
            json.dumps(pf, ensure_ascii=False, indent=1), encoding="utf-8")
        bad = _preflight_bad(pf, args.efforts)
        if bad:
            log(f"❌ 预检失败：档位 {bad} 未被网关接受（400-剥离或调用失败）→ A/B 无区分度。"
                f"详见 {outdir}/preflight.json；改用网关合法档位后重跑，或 --ignore-preflight 强行继续")
            raise SystemExit(3)
        log(f"✅ 预检通过：{args.efforts} 均被网关接受")
    results = []
    log(f"润色轨道：{len(samples)} 块 × {args.efforts}（模型={model_name}）")
    for i, s in enumerate(samples, 1):
        cid, ch = s["chunk"], s["chapter"]
        style = set_chapter_style(ch)
        reg = build_entity_registry(s["source"])
        raw_text = load_prod_json(ch, cid, "raw")["text"]
        order = list(args.efforts)
        random.Random(int(hashlib.md5(cid.encode()).hexdigest(), 16)).shuffle(order)
        drafts = {}
        for eff in order:
            t0 = time.perf_counter()
            try:
                res = agent.process_chunk(
                    cid, raw_text, style, s["source"],
                    entity_registry=reg, narrative_memory="",
                    retry_count=0, model_effort=eff,
                )
                lit = res[0] if isinstance(res, tuple) else res
                dt = time.perf_counter() - t0
                drafts[eff] = {"ok": bool(lit), "text": lit, "latency": round(dt, 1), "err": ""}
            except Exception as exc:
                dt = time.perf_counter() - t0
                drafts[eff] = {"ok": False, "text": None, "latency": round(dt, 1),
                               "err": f"{type(exc).__name__}: {exc}"}
                log(f"  ❌ {cid} effort={eff} 调用失败: {drafts[eff]['err'][:150]}")
            time.sleep(args.sleep)
        entry = {"track": "rewrite", **{k: s[k] for k in
                 ("chunk", "chapter", "difficulty", "retries", "quality_retries")},
                 "model": get_role_model_name(agent.role), "drafts": drafts,
                 "referee": None}
        ok_effs = [e for e in args.efforts if drafts.get(e, {}).get("ok")]
        if len(ok_effs) == 2:
            e1, e2 = args.efforts[0], args.efforts[1]
            flip = random.Random(int(hashlib.md5((cid + "#flip").encode()).hexdigest(), 16)).random() < 0.5
            text_a, text_b = (drafts[e1]["text"], drafts[e2]["text"]) if flip else (drafts[e2]["text"], drafts[e1]["text"])
            eff_of_label = {"A": e1, "B": e2} if flip else {"A": e2, "B": e1}
            ref = {"labels": {"A": eff_of_label["A"], "B": eff_of_label["B"]}}
            try:
                res = jev_pairwise(s["source"], raw_text, style, reg, text_a, text_b)
                winner_label = res.get("winner")
                ref.update({"kind": "jev_pairwise", "winner_label": winner_label,
                            "winner_effort": eff_of_label.get(winner_label),
                            "votes": res.get("votes"), "backend": res.get("backend")})
            except Exception as exc:
                log(f"  ⚠️ {cid} Jev 裁判不可用（{type(exc).__name__}: {exc}）→ critic 兜底")
                try:
                    ma, sa = critic_mean_score(s["source"], raw_text, style, text_a)
                    mb, sb = critic_mean_score(s["source"], raw_text, style, text_b)
                    winner_label = None
                    if ma is not None and mb is not None:
                        winner_label = "A" if ma > mb else ("B" if mb > ma else None)
                    ref.update({"kind": "critic_fallback", "winner_label": winner_label,
                                "winner_effort": eff_of_label.get(winner_label),
                                "scores_A": sa, "scores_B": sb, "mean_A": ma, "mean_B": mb})
                except Exception as exc2:
                    ref.update({"kind": "unjudged", "err": f"jev: {exc}; critic: {exc2}"})
            entry["referee"] = ref
        else:
            entry["referee"] = {"kind": "arm_failed"}
        results.append(entry)
        save_entry(outdir, entry)
        ref = entry.get("referee") or {}
        lat = " / ".join(f"{e}={drafts.get(e, {}).get('latency', '?')}s" for e in args.efforts)
        log(f"[{i}/{len(samples)}] {cid} diff={s['difficulty']} | {lat} | "
            f"裁判={ref.get('kind')} winner={ref.get('winner_effort', ref.get('winner_label'))}")
    return results


def run_judge_track(args, samples: list, outdir: Path) -> list:
    from src.agents.judge_agent import JudgeAgent
    judge = JudgeAgent(get_de())
    results = []
    log(f"裁决轨道：{len(samples)} 块 × {args.efforts}（模型={get_role_model_name(judge.role)}，"
        f"⚠️ 消耗 MiniMax 配额）")
    for i, s in enumerate(samples, 1):
        cid, ch = s["chunk"], s["chapter"]
        set_chapter_style(ch)
        lit = load_prod_json(ch, cid, "final")["text"]
        critic_report = load_prod_json(ch, cid, "critic_report")
        prompt = judge._build_prompt(s["source"], lit, critic_report,
                                     rewriter_self_review=None, verse_note=None)
        model_key, params = _config.resolve_task_model("judge_decision")
        model_name = get_role_model_name(judge.role) or model_key
        order = list(args.efforts)
        random.Random(int(hashlib.md5(cid.encode()).hexdigest(), 16)).shuffle(order)
        arms = {}
        for eff in order:
            t0 = time.perf_counter()
            try:
                raw_out = judge.llm.generate(prompt=prompt, model_name=model_name,
                                             **{**params, "effort": eff})
                dt = time.perf_counter() - t0
                parsed = robust_json_loads(raw_out) or {}
                decision = parsed.get("decision")
                reason = parsed.get("reject_reason", "")
                # 生产同款安全网（skill）— 两臂同规则
                scores = critic_report.get("scores") or {}
                mock_untrusted = (
                    not _config.pipeline.get("trust_mock_heuristic", False)
                    and critic_report.get("_backend_path") == "mock_heuristic")
                if scores and not mock_untrusted and decision == "PASS":
                    vals = [v for v in scores.values() if isinstance(v, (int, float))]
                    avg = sum(vals) / max(1, len(vals))
                    if avg < float(_config.pipeline.get("judge_safety_net_avg_min", 5.5)):
                        decision = "REJECT(安全网)"
                arms[eff] = {"ok": True, "latency": round(dt, 1),
                             "decision": decision, "reject_reason": reason[:200],
                             "raw_len": len(raw_out or "")}
            except Exception as exc:
                dt = time.perf_counter() - t0
                arms[eff] = {"ok": False, "latency": round(dt, 1),
                             "err": f"{type(exc).__name__}: {exc}"}
                log(f"  ❌ {cid} effort={eff} 裁决失败: {arms[eff]['err'][:150]}")
            time.sleep(args.sleep)
        e1, e2 = args.efforts[0], args.efforts[1]
        agree = None
        if arms.get(e1, {}).get("ok") and arms.get(e2, {}).get("ok"):
            agree = arms[e1]["decision"] == arms[e2]["decision"]
        entry = {"track": "judge", **{k: s[k] for k in
                 ("chunk", "chapter", "difficulty", "retries", "quality_retries")},
                 "model": model_name, "arms": arms, "agree": agree}
        results.append(entry)
        save_entry(outdir, entry)
        lat = " / ".join(f"{e}={arms.get(e, {}).get('latency', '?')}s" for e in args.efforts)
        dec = " vs ".join(str(arms.get(e, {}).get("decision", "FAIL")) for e in args.efforts)
        log(f"[{i}/{len(samples)}] {cid} diff={s['difficulty']} | {lat} | 决策: {dec} | "
            f"{'一致' if agree else ('不一致' if agree is False else '不可比')}")
    return results


def save_entry(outdir: Path, entry: dict) -> None:
    track_dir = outdir / entry["track"]
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / f"{entry['chunk']}.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(outdir / "results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_summary(outdir: Path, rewrite_results: list, judge_results: list, efforts: list) -> Path:
    e1, e2 = efforts[0], efforts[1]
    lines = ["# effort A/B 探测摘要（Round 138）", "",
             f"- 生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             f"- 档位: {e1} vs {e2}", ""]
    if rewrite_results:
        wins = {e1: 0, e2: 0, "tie": 0, "unjudged": 0}
        lat = {e1: [], e2: []}
        fail = {e1: 0, e2: 0}
        for r in rewrite_results:
            for e in efforts:
                d = r["drafts"].get(e) or {}
                lat[e].append(d.get("latency"))
                if not d.get("ok"):
                    fail[e] += 1
            ref = r.get("referee") or {}
            if ref.get("kind") in ("jev_pairwise", "critic_fallback") and ref.get("winner_effort"):
                wins[ref["winner_effort"]] += 1
            elif ref.get("winner_label") is None and ref.get("kind") in ("jev_pairwise", "critic_fallback"):
                wins["tie"] += 1
            else:
                wins["unjudged"] += 1
        n_judged = wins[e1] + wins[e2] + wins["tie"]
        lines += ["## 润色轨道", "",
                  f"- 样本: {len(rewrite_results)} 块（困难块优先）",
                  f"- 裁判胜负: {e1}={wins[e1]} / {e2}={wins[e2]} / tie={wins['tie']} "
                  f"(未判={wins['unjudged']})",
                  f"- 调用失败率: {e1}={fail[e1]} / {e2}={fail[e2]}",
                  f"- 延迟中位数: {e1}={median(lat[e1])}s / {e2}={median(lat[e2])}s", ""]
        if n_judged >= max(4, len(rewrite_results) // 2):
            if wins[e1] == 0 and fail[e1] >= fail[e2]:
                lines.append(f"**建议**: 证据不支持 {e1}（0 胜+失败率不低）→ 可下调为 {e2}，"
                             f"预计省时约 {median(lat[e1])}→{median(lat[e2])}s/块")
            elif wins[e1] > 2 * max(1, wins[e2]):
                lines.append(f"**建议**: {e1} 优势明显 → 维持 {e1}")
            else:
                lines.append(f"**建议**: 差异在噪声内（{wins[e1]} vs {wins[e2]}）→ "
                             f"结合失败率/延迟决策（倾向 {e2 if fail[e1] >= fail[e2] else e1}）")
        else:
            lines.append("**建议**: 裁判覆盖不足，仅作延迟/失败率参考")
        lines.append("")
    if judge_results:
        agree = sum(1 for r in judge_results if r.get("agree") is True)
        disagree = sum(1 for r in judge_results if r.get("agree") is False)
        lat = {e1: [], e2: []}
        for r in judge_results:
            for e in efforts:
                lat[e].append((r["arms"].get(e) or {}).get("latency"))
        lines += ["## 裁决轨道", "",
                  f"- 样本: {len(judge_results)} 块",
                  f"- 决策一致: {agree} / 不一致: {disagree}（不一致=两档给出不同 PASS/REJECT）",
                  f"- 延迟中位数: {e1}={median(lat[e1])}s / {e2}={median(lat[e2])}s", ""]
        if disagree == 0 and agree > 0:
            lines.append(f"**建议**: 两档裁决完全一致 → {e2} 可替代 {e1}（省时按延迟差）")
        elif disagree > 0:
            lines.append("**建议**: 存在决策翻转，需人工审阅翻转样本后再定")
        lines.append("")
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return outdir / "summary.md"


def dry_run(track: str, samples: list, efforts: list) -> None:
    """离线自检：构建两轨 prompt（不调 LLM）。"""
    log(f"DRY-RUN {track}: {len(samples)} 样本, 档位 {efforts}")
    for s in samples:
        cid, ch = s["chunk"], s["chapter"]
        style = set_chapter_style(ch)
        if track == "rewrite":
            from src.agents.rewriter_agent import LiteraryRewriterAgent
            agent = LiteraryRewriterAgent(get_de())
            reg = build_entity_registry(s["source"])
            raw_text = load_prod_json(ch, cid, "raw")["text"]
            style_txt = agent._build_style_guide(style, s["source"])
            dec_ctx = agent._build_decision_context(s["source"])
            prompt = agent._build_prompt(raw_text, dec_ctx, style_txt,
                                         narrative_memory="", entity_registry=reg, verse_note=None)
            log(f"  {cid} diff={s['difficulty']} prompt={len(prompt)}ch "
                f"style={len(style_txt)}ch registry={len(reg)}条")
        else:
            from src.agents.judge_agent import JudgeAgent
            judge = JudgeAgent(get_de())
            lit = load_prod_json(ch, cid, "final")["text"]
            critic_report = load_prod_json(ch, cid, "critic_report")
            prompt = judge._build_prompt(s["source"], lit, critic_report)
            key, params = _config.resolve_task_model("judge_decision")
            log(f"  {cid} diff={s['difficulty']} prompt={len(prompt)}ch "
                f"model={get_role_model_name(judge.role) or key} params={params}")
    log("DRY-RUN 完成：prompt 构建全部成功（未发任何网络请求）")


def main() -> int:
    ap = argparse.ArgumentParser(description="effort 档位 A/B 探测（Round 138）")
    ap.add_argument("track", choices=["rewrite", "judge"])
    ap.add_argument("--n", type=int, default=None, help="样本数（rewrite 默认 12；judge 默认 6）")
    ap.add_argument("--efforts", default="max,xhigh", help="两档，逗号分隔（默认 max,xhigh）")
    ap.add_argument("--chapters", default="ch01,ch02", help="样本来源章节（默认 ch01,ch02）")
    ap.add_argument("--chunks", default="", help="只测指定块（逗号分隔，覆盖难度选择）")
    ap.add_argument("--sleep", type=float, default=3.0, help="每次 LLM 调用间隔秒（默认 3）")
    ap.add_argument("--dry-run", action="store_true", help="只构建 prompt 自检，不发请求")
    ap.add_argument("--preflight-only", action="store_true",
                    help="只跑档位预检（~1 分钟）后退出：验证 max/xhigh 是否被网关真实接受")
    ap.add_argument("--ignore-preflight", action="store_true", help="跳过档位预检（不推荐）")
    ap.add_argument("--minimax", action="store_true",
                    help="judge 轨道必填：确认消耗 MiniMax 配额")
    ap.add_argument("--yes", action="store_true", help="跳过交互确认")
    ap.add_argument("--out", default="", help="结果目录（默认 /tmp/sagatrans_effort_ab/<ts>）")
    args = ap.parse_args()

    efforts = [x.strip().lower() for x in args.efforts.split(",") if x.strip()]
    if len(efforts) != 2:
        log("❌ --efforts 需恰好两档，如 max,xhigh")
        return 2
    if any(len(e) < 2 for e in efforts):
        log("❌ --efforts 解析异常（疑似单字符/空档位），请检查参数")
        return 2
    args.efforts = efforts  # Round 138 fix: 轨道函数统一读解析后的列表（此前按字符串逐字符迭代）
    n = args.n if args.n is not None else (12 if args.track == "rewrite" else 6)
    chapters = [x.strip() for x in args.chapters.split(",") if x.strip()]
    only = [x.strip() for x in args.chunks.split(",") if x.strip()]

    if args.track == "judge" and not args.dry_run and not args.minimax:
        log("❌ judge 轨道直接消耗 MiniMax 配额（主流程依赖）。如确认要测，加 --minimax。")
        return 2

    log(f"选择样本: track={args.track} n={n} chapters={chapters}")
    samples = select_samples(args.track, n, chapters, only)
    for s in samples:
        log(f"  {s['chunk']} diff={s['difficulty']} retries={s['retries']}/qr={s['quality_retries']}")
    if not samples:
        log("❌ 未找到符合条件的样本")
        return 2

    if args.dry_run:
        dry_run(args.track, samples, efforts)
        return 0

    if args.preflight_only:
        if args.track != "rewrite":
            log("❌ --preflight-only 仅支持 rewrite 轨道（judge 预检会消耗 MiniMax 配额）")
            return 2
        from src.agents.rewriter_agent import LiteraryRewriterAgent
        outdir = Path(args.out) if args.out else OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M")
        outdir.mkdir(parents=True, exist_ok=True)
        agent = LiteraryRewriterAgent(get_de())
        pf = preflight_efforts(agent.llm, get_role_model_name(agent.role), efforts)
        (outdir / "preflight.json").write_text(
            json.dumps(pf, ensure_ascii=False, indent=1), encoding="utf-8")
        bad = _preflight_bad(pf, efforts)
        log(("✅ 预检通过：档位均被网关接受" if not bad else
             f"❌ 预检失败：{bad} 未通过（400-剥离或网络失败）") + f" → {outdir}/preflight.json")
        return 0 if not bad else 3

    if args.track == "judge" and not args.yes:
        log("⚠️ judge 轨道会占用 MiniMax 配额，可能影响主流程（触发配额降级）。")
        ans = input("确认继续？[y/N] ").strip().lower()
        if ans != "y":
            log("已取消")
            return 0

    outdir = Path(args.out) if args.out else OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M")
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"结果目录: {outdir}")
    t0 = time.time()
    rw_results = run_rewrite_track(args, samples, outdir) if args.track == "rewrite" else []
    jd_results = run_judge_track(args, samples, outdir) if args.track == "judge" else []
    summary = write_summary(outdir, rw_results, jd_results, efforts)
    log(f"✅ 完成，用时 {time.time() - t0:.0f}s。摘要: {summary}")
    print(summary.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("已中断")
        sys.exit(130)
