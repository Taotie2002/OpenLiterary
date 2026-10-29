#!/usr/bin/env python3
"""Round 133 L1: 模型 effort 档位探测 → db/model_capabilities.json

用途：
  - 在流程外（人工/巡检时）摸清每个在用模型的所有 effort 档位是否可用、
    延迟、是否产推理，记录在案；
  - pipeline 启动时只读该文件（缺失/过期静默跳过），用于：
      ① 重试降档阶梯按实测 accepted 档位过滤（翻译流程 no_content 自动降档）；
      ② 档位合法性交叉校验。
  - 绝不在翻译流程中自动探测（避免烧配额、拖慢排队）。

用法：
  python3 tools/probe_llm_efforts.py                   # 全部角色（models 段全部）
  python3 tools/probe_llm_efforts.py --role primary --role secondary
  python3 tools/probe_llm_efforts.py --timeout 300 --output db/model_capabilities.json

注意：
  - 每档一次微型请求（max_tokens≈16），总消耗极小；
  - MiniMax 无 none 档（服务端强制 adaptive），档位表见 src/config.py；
  - 网络代理沿用进程环境变量（ALL_PROXY/HTTPS_PROXY）。
"""
from __future__ import annotations

import argparse
import json
import re as _re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TINY_PROMPT = "只回复两个字：可用"


def _now_local() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _probe_level(api_base: str, api_key: str, model_name: str, level: str,
                 timeout: float, tiny_max_tokens: int) -> Dict[str, Any]:
    """单档微型请求探测。返回 {accepted, latency_s, has_reasoning, content_len, finish_reason, error}。"""
    import requests
    from src.utils.llm_adapter import build_effort_params

    url = api_base.rstrip("/") + "/chat/completions"
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": TINY_PROMPT}],
        "max_tokens": tiny_max_tokens,
        "temperature": 0.0,
    }
    body.update(build_effort_params(api_base, model_name, level))
    started = time.time()
    rec: Dict[str, Any] = {
        "accepted": False, "latency_s": None, "has_reasoning": False,
        "content_len": 0, "finish_reason": None, "error": None,
    }
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body, timeout=timeout,
        )
        rec["latency_s"] = round(time.time() - started, 2)
        if resp.status_code != 200:
            rec["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}".replace("\n", " ")
            return rec
        data = resp.json()
        choice = ((data.get("choices") or [{}])[0]) or {}
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not reasoning and "<think" in content:
            m = _re.search(r"<think(.*?)</think>", content, _re.DOTALL)
            if m:
                reasoning = m.group(1)
                content = content[m.end():]
        rec["accepted"] = True
        rec["has_reasoning"] = bool(str(reasoning).strip())
        rec["content_len"] = len(content.strip())
        rec["finish_reason"] = choice.get("finish_reason")
    except Exception as e:  # noqa: BLE001 — 探测工具需记录任意失败
        rec["latency_s"] = round(time.time() - started, 2)
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="模型 effort 档位探测（L1）")
    ap.add_argument("--role", action="append", default=None,
                    help="只探测指定角色（可重复；缺省=config.yaml models 全部）")
    ap.add_argument("--output", default=None, help="输出路径（缺省=config pipeline.model_probe.path）")
    ap.add_argument("--timeout", type=float, default=180.0, help="单请求超时秒数（缺省 180）")
    ap.add_argument("--max-tokens", type=int, default=None, help="微型请求 max_tokens 覆盖")
    args = ap.parse_args()

    from src.config import EFFORT_LEVELS_BY_PROVIDER, effort_provider_of, get_config

    cfg = get_config()
    probe_cfg = dict((cfg.pipeline or {}).get("model_probe", {}) or {})
    tiny_max_tokens = int(args.max_tokens or probe_cfg.get("tiny_max_tokens", 16) or 16)
    ttl_days = float(probe_cfg.get("ttl_days", 7) or 7)
    out_path = Path(args.output or (probe_cfg.get("path") or "db/model_capabilities.json"))
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    all_roles = list((cfg.role_models or {}).keys()) if cfg.has_role_models() else []
    roles = args.role or all_roles
    if not roles:
        print("[ERR] 未找到任何角色（config.yaml models 段为空？）", file=sys.stderr)
        return 2

    result: Dict[str, Any] = {
        "generated_at": _now_local(),
        "tool": "tools/probe_llm_efforts.py",
        "ttl_days": ttl_days,
        "models": {},
    }
    for role in roles:
        role_cfg = cfg.get_role_config(role) or {}
        model_name = role_cfg.get("model_name", "")
        api_base = role_cfg.get("api_base", "")
        api_key = role_cfg.get("api_key", "")
        provider = effort_provider_of(model_name, api_base)
        if not provider:
            print(f"[SKIP] {role}: 无法识别 provider (model={model_name})")
            continue
        levels = EFFORT_LEVELS_BY_PROVIDER.get(provider, ())
        rec: Dict[str, Any] = {
            "provider": provider, "model_name": model_name,
            "api_base": api_base, "levels": {},
        }
        print(f"== {role} ({provider}) {model_name} ==")
        for lv in levels:
            r = _probe_level(api_base, api_key, model_name, lv, args.timeout, tiny_max_tokens)
            rec["levels"][lv] = r
            status = "OK  " if r["accepted"] else "FAIL"
            note = r["error"] or (
                f"finish={r['finish_reason']} content={r['content_len']} "
                f"reasoning={'Y' if r['has_reasoning'] else 'N'}"
            )
            print(f"  [{status}] effort={lv:6s} {r['latency_s']}s  {note}")
        result["models"][role] = rec

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已写入 {out_path}（generated_at={result['generated_at']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
