#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SagaTrans 网关超时诊断 probe — 只读探测，结果写 /tmp/probe_gw_diag*"""
import datetime, json, os, socket, sqlite3, sys, time
import requests
from dotenv import load_dotenv

load_dotenv("/home/zyu/projects/SagaTrans/.env")
sys.path.insert(0, "/home/zyu/projects/SagaTrans")
os.chdir("/home/zyu/projects/SagaTrans")

BASE = "https://token.ai-hf.cn/v1"
QWEN_KEY = os.environ.get("QWEN_API_KEY", "")
DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
QWEN_MODEL = "Qwen3.8-Flash-Next-SS"
DS_MODEL = "Deepseek-V4.1-Flash-1017-SS"
PROXY = "socks5h://127.0.0.1:10808"

RAW_DIR = "/tmp/probe_gw_diag_raw"; os.makedirs(RAW_DIR, exist_ok=True)
LOG_PATH = "/tmp/probe_gw_diag_results.jsonl"
T0 = time.time()

def say(x):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {x}", flush=True)

con = sqlite3.connect("db/workflow.db")
txt = con.execute("select text_content from chunk_tasks where chunk_id='ch01_chunk000'").fetchone()[0]
try:
    from src.agents.reference_agent import ReferenceAgent
    prompt_extract = ReferenceAgent._build_prompt(None, txt)
    say(f"real extraction prompt built: {len(prompt_extract)} chars")
except Exception as e:
    prompt_extract = "你是文学考据专家，输出纯JSON。\n【待分析文本】\n" + txt
    say(f"prompt fallback ({type(e).__name__}: {e}); len={len(prompt_extract)}")

recall_json = json.dumps({"references": [{"source_text": "the balances are correct", "allusion_target": "测试条目", "strategy": "LITERAL_TRANSLATION", "translated_term": "天平是正的", "reason": "测试"}] * 3,
                          "entities": [{"canonical_name": "沙达姆四世", "source_names": ["Shaddam IV"], "pronoun_gender": "he", "entity_type": "person", "confidence": "high", "translation_strategy": "transliteration"}] * 3}, ensure_ascii=False, indent=2)
prompt_confirm = f"""你是资深文学考据复核员（第二意见）。另一位考据员已对片段做「高召回」初筛，你的任务是复核并定稿。

【原文片段】
{txt[:8000]}

【初筛候选（JSON）】
{recall_json[:12000]}

【复核规则】
1. 逐条核验 allusion_target 是否被原文支撑；无依据、张冠李戴或明显误判 → 删除。
2. 补漏：初筛漏掉且确有出处的典故/隐喻/历史名词 → 追加（宁精勿滥）。
3. 实体：合并重复指称；canonical_name 统一为规范中文译名；source_names 必须都是原文实际出现的形式。
4. 策略只能是 RETAIN_AND_ANNOTATE / CULTURAL_EQUIVALENT / LITERAL_TRANSLATION 三选一。
5. 若初筛正确，原样保留其条目（可修正字段格式）；不要为凑数新增无依据条目。

【输出格式】
输出与初筛完全相同的纯 JSON（references / entities 两个数组），不要输出 JSON 以外的任何字符。
"""

def _session(use_proxy):
    s = requests.Session(); s.trust_env = False
    if use_proxy:
        s.proxies = {"http": PROXY, "https": PROXY}
    return s

def call(name, key, model, prompt, effort=None, max_tokens=20000, temperature=0.1, use_proxy=False, read_timeout=360):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    if effort is not None:
        payload["reasoning_effort"] = effort
    s = _session(use_proxy)
    t0 = time.time()
    rec = {"name": name, "model": model, "effort": effort, "proxy": use_proxy,
           "prompt_chars": len(prompt), "max_tokens": max_tokens}
    try:
        r = s.post(f"{BASE}/chat/completions",
                   headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                   json=payload, timeout=(15, read_timeout), verify=True)
        el = time.time() - t0
        rec.update(status=r.status_code, elapsed=round(el, 1))
        try:
            j = r.json()
            ch = (j.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            usage = j.get("usage") or {}
            rec.update(finish=ch.get("finish_reason"), content_chars=len(content),
                       reasoning_chars=len(reasoning),
                       comp_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"),
                       preview=content[:80].replace("\n", " "))
            with open(f"{RAW_DIR}/{name}.json", "w") as f:
                json.dump(j, f, ensure_ascii=False, indent=1)
        except Exception as e2:
            rec.update(err=f"json parse: {e2}", body_preview=r.text[:160])
        say(f"{name} | {rec.get('status')} | {rec.get('elapsed')}s | finish={rec.get('finish')} | content={rec.get('content_chars')} reasoning={rec.get('reasoning_chars')} | comp_tok={rec.get('comp_tokens')} | {(rec.get('preview') or '')[:48]}")
    except Exception as e:
        el = time.time() - t0
        rec.update(status=f"EXC:{type(e).__name__}", elapsed=round(el, 1), err=str(e)[:200])
        say(f"{name} | {rec['status']} | {el:.1f}s | {rec['err'][:110]}")
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def stream_probe(name, key, model, prompt, effort=None, max_tokens=20000, ttf_budget=75):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.1, "stream": True}
    if effort is not None:
        payload["reasoning_effort"] = effort
    s = _session(False)
    t0 = time.time(); first = None; n = 0; sample = []
    rec = {"name": name, "model": model, "effort": effort, "mode": "stream", "prompt_chars": len(prompt)}
    try:
        r = s.post(f"{BASE}/chat/completions",
                   headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                   json=payload, timeout=(15, ttf_budget), stream=True, verify=True)
        rec["status"] = r.status_code
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                n += 1
                if first is None:
                    first = time.time() - t0
                    say(f"{name} first SSE event at {first:.1f}s")
                if n <= 3:
                    sample.append(line[:90])
                if n >= 60 or "[DONE]" in line:
                    break
            if time.time() - t0 - (first or 0) > ttf_budget:
                break
        r.close()
        rec.update(first_event_s=round(first, 1) if first else None, n_events=n,
                   elapsed=round(time.time() - t0, 1), preview=" | ".join(sample)[:180])
        say(f"{name} | {rec.get('status')} | first={rec.get('first_event_s')}s | events={n} | elapsed={rec.get('elapsed')}s")
    except Exception as e:
        rec.update(status=f"EXC:{type(e).__name__}", elapsed=round(time.time() - t0, 1),
                   first_event_s=round(first, 1) if first else None, n_events=n, err=str(e)[:200])
        say(f"{name} | {rec['status']} | first={rec.get('first_event_s')}s | events={n} | {rec['err'][:100]}")
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def ok(rec):
    return rec.get("status") == 200 and (rec.get("content_chars") or 0) > 0

def budget():
    return time.time() - T0 < 1800

say("== probe start ==")
say("env proxy vars: " + json.dumps({k: v for k, v in os.environ.items() if "proxy" in k.lower()}))
t_dns = time.time(); socket.getaddrinfo("token.ai-hf.cn", 443)
say(f"dns ok {time.time() - t_dns:.2f}s")

r0 = call("P0_DS_tiny", DS_KEY, DS_MODEL, "只回复两个字：可用", max_tokens=16, read_timeout=60)
r0b = call("P0b_DS_tiny_max", DS_KEY, DS_MODEL, "只回复两个字：可用", effort="max", max_tokens=20000, read_timeout=180)
r0c = call("P0c_DS_tiny_none", DS_KEY, DS_MODEL, "只回复两个字：可用", effort="none", max_tokens=20000, read_timeout=120)
r1 = call("P1_QW_tiny", QWEN_KEY, QWEN_MODEL, "只回复两个字：可用", max_tokens=16, read_timeout=60)
r1b = call("P1b_QW_tiny_low20k", QWEN_KEY, QWEN_MODEL, "只回复两个字：可用", effort="low", max_tokens=20000, read_timeout=120)
if not ok(r1b) and budget():
    call("P1c_QW_tiny_none20k", QWEN_KEY, QWEN_MODEL, "只回复两个字：可用", effort="none", max_tokens=20000, read_timeout=120)

r2 = call("P2_QW_extract_real_low", QWEN_KEY, QWEN_MODEL, prompt_extract, effort="low", max_tokens=20000, read_timeout=360) if budget() else {}
if r2 and not ok(r2) and budget():
    stream_probe("P6_QW_extract_stream_low", QWEN_KEY, QWEN_MODEL, prompt_extract, effort="low")

r4 = call("P4_DS_confirm_max60k", DS_KEY, DS_MODEL, prompt_confirm, effort="max", max_tokens=60000, read_timeout=360) if budget() else {}
if r4 and r4.get("status") == 200 and (r4.get("content_chars") or 0) == 0 and budget():
    call("P4b_DS_confirm_high60k", DS_KEY, DS_MODEL, prompt_confirm, effort="high", max_tokens=60000, read_timeout=360)
    if budget():
        call("P4c_DS_confirm_none20k", DS_KEY, DS_MODEL, prompt_confirm, effort="none", max_tokens=20000, read_timeout=120)

if r2 and not ok(r2) and budget():
    call("P2b_QW_extract_real_none", QWEN_KEY, QWEN_MODEL, prompt_extract, effort="none", max_tokens=20000, read_timeout=360)
    if budget():
        call("P3_QW_extract_real_low_proxy", QWEN_KEY, QWEN_MODEL, prompt_extract, effort="low", max_tokens=20000, read_timeout=360, use_proxy=True)

say(f"== probe done, wall={time.time()-T0:.0f}s ==")
