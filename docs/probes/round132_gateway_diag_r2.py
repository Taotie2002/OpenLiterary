#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SagaTrans 网关诊断 Round2 — 确认阶段档位矩阵 + Qwen-none 复测。结果写 /tmp/probe_gw_r2*"""
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
HTTP_PROXY = "http://127.0.0.1:10808"

RAW_DIR = "/tmp/probe_gw_r2_raw"; os.makedirs(RAW_DIR, exist_ok=True)
LOG_PATH = "/tmp/probe_gw_r2_results.jsonl"
T0 = time.time()

def say(x):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {x}", flush=True)

def budget(need):
    return (1600 - (time.time() - T0)) > need

con = sqlite3.connect("db/workflow.db")
txt = con.execute("select text_content from chunk_tasks where chunk_id='ch01_chunk000'").fetchone()[0]
from src.agents.reference_agent import ReferenceAgent
prompt_extract = ReferenceAgent._build_prompt(None, txt)
say(f"extraction prompt: {len(prompt_extract)} chars")

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
prompt_rewrite = "把下面英文段落译成中文（文学翻译，保持叙事风格与节奏）：\n\n" + txt[:2000]

def _session(proxy_url=None):
    s = requests.Session(); s.trust_env = False
    if proxy_url:
        s.proxies = {"http": proxy_url, "https": proxy_url}
    return s

def call(name, key, model, prompt, effort=None, max_tokens=20000, temperature=0.1, proxy_url=None, read_timeout=360):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature}
    if effort is not None:
        payload["reasoning_effort"] = effort
    s = _session(proxy_url)
    t0 = time.time()
    rec = {"name": name, "model": model, "effort": effort, "proxy": proxy_url or "direct",
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
        say(f"{name} | {rec.get('status')} | {rec.get('elapsed')}s | finish={rec.get('finish')} | content={rec.get('content_chars')} reasoning={rec.get('reasoning_chars')} | comp_tok={rec.get('comp_tokens')} | {(rec.get('preview') or '')[:44]}")
    except Exception as e:
        el = time.time() - t0
        rec.update(status=f"EXC:{type(e).__name__}", elapsed=round(el, 1), err=str(e)[:200])
        say(f"{name} | {rec['status']} | {el:.1f}s | {rec['err'][:110]}")
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def stream_until_done(name, key, model, prompt, effort=None, max_tokens=60000, max_wait=330):
    """流式：一路读到 [DONE] 或 max_wait，观测长生成是否被 300s 腰斩。"""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.1, "stream": True}
    if effort is not None:
        payload["reasoning_effort"] = effort
    s = _session(None)
    t0 = time.time(); first = None; n = 0; c_chars = 0; r_chars = 0; done = False
    rec = {"name": name, "model": model, "effort": effort, "mode": "stream", "prompt_chars": len(prompt), "max_tokens": max_tokens}
    try:
        r = s.post(f"{BASE}/chat/completions",
                   headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                   json=payload, timeout=(15, 90), stream=True, verify=True)
        rec["status"] = r.status_code
        for line in r.iter_lines(decode_unicode=True):
            now = time.time() - t0
            if line and line.startswith("data:"):
                n += 1
                if first is None:
                    first = now
                body = line[5:].strip()
                if body == "[DONE]":
                    done = True
                    break
                try:
                    d = json.loads(body)
                    delta = (d.get("choices") or [{}])[0].get("delta") or {}
                    c_chars += len(delta.get("content") or "")
                    r_chars += len(delta.get("reasoning_content") or delta.get("reasoning") or "")
                except Exception:
                    pass
            if now > max_wait:
                break
        r.close()
        rec.update(first_event_s=round(first, 1) if first else None, n_events=n, done=done,
                   elapsed=round(time.time() - t0, 1), content_chars=c_chars, reasoning_chars=r_chars)
        say(f"{name} | {rec.get('status')} | first={rec.get('first_event_s')}s | events={n} | done={done} | elapsed={rec.get('elapsed')}s | content={c_chars} reasoning={r_chars}")
    except Exception as e:
        rec.update(status=f"EXC:{type(e).__name__}", elapsed=round(time.time() - t0, 1),
                   first_event_s=round(first, 1) if first else None, n_events=n, done=done,
                   content_chars=c_chars, reasoning_chars=r_chars, err=str(e)[:200])
        say(f"{name} | {rec['status']} | first={rec.get('first_event_s')}s | events={n} | elapsed={rec.get('elapsed')}s | content={c_chars} reasoning={r_chars} | {rec['err'][:90]}")
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec

def ok(r):
    return r.get("status") == 200 and (r.get("content_chars") or 0) > 0

say("== R2 probe start ==")
t_dns = time.time(); socket.getaddrinfo("token.ai-hf.cn", 443); say(f"dns ok {time.time()-t_dns:.2f}s")

r1 = call("R1_DS_confirm_none20k", DS_KEY, DS_MODEL, prompt_confirm, effort="none", max_tokens=20000, read_timeout=120)
r5 = call("R5_QW_extract_none_rep2", QWEN_KEY, QWEN_MODEL, prompt_extract, effort="none", max_tokens=20000, read_timeout=120)
r4 = call("R4_QW_extract_none_httpproxy", QWEN_KEY, QWEN_MODEL, prompt_extract, effort="none", max_tokens=20000, read_timeout=120, proxy_url=HTTP_PROXY)
r2 = call("R2_DS_confirm_low60k", DS_KEY, DS_MODEL, prompt_confirm, effort="low", max_tokens=60000, read_timeout=360)

if budget(750):
    stream_until_done("R7_DS_confirm_max_stream", DS_KEY, DS_MODEL, prompt_confirm, effort="max")
if (not ok(r2)) and budget(500):
    call("R3_DS_confirm_high60k", DS_KEY, DS_MODEL, prompt_confirm, effort="high", max_tokens=60000, read_timeout=360)
if ok(r2) and (r2.get("elapsed") or 999) < 150 and budget(500):
    call("R3x_DS_confirm_xhigh60k", DS_KEY, DS_MODEL, prompt_confirm, effort="xhigh", max_tokens=60000, read_timeout=360)
if budget(420):
    call("R6_DS_rewrite_max20k", DS_KEY, DS_MODEL, prompt_rewrite, effort="max", max_tokens=20000, read_timeout=360)

say(f"== R2 probe done, wall={time.time()-T0:.0f}s ==")
