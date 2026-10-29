#!/usr/bin/env python3
"""probe_role.py — 单个角色 LLM 连通性探针（新增/切换模型后先跑它）。

用法:
    python3 scripts/probe_role.py secondary
    python3 scripts/probe_role.py secondary --thinking on
    python3 scripts/probe_role.py secondary --prompt "把这句话译成中文: Hello, world."

说明:
  - 角色名来自 config.yaml `models:` 段的 key（literal_translator / primary / secondary）。
  - 输出路由到的 model_name、api_base、延迟、回复片段与累计 usage。
  - 失败时打印异常分类，便于区分 401(鉴权) / 404(model id) / 429(限流) / 网络错误。
  - 需要代理时自行加 ALL_PROXY（SDK 走系统环境变量）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config as _config  # noqa: E402
from src.utils.llm_adapter import (  # noqa: E402
    get_llm_client,
    get_task_extra_body,
    get_role_model_name,
)

DEFAULT_PROMPT = (
    "用一句话把下面这句话翻译成中文，只输出译文："
    "The cold wind blew across the empty square."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="SagaTrans 角色探针")
    parser.add_argument("role", help="config.yaml models: 段中的角色 key")
    parser.add_argument("--effort", default=None,
                        help="覆盖任务级 effort 档位（如 low/high/max；默认不发该参数）")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--task", default=None,
                        help="可选：透传该任务 task_routing.extra_body 的供应商参数")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--list-models", action="store_true",
                        help="只列 api_base/models 上的可用 model id（自建网关核对模型名用）")
    args = parser.parse_args()

    role_cfg = _config.get_role_config(args.role)
    if not role_cfg:
        print(f"❌ 角色 {args.role} 未在 config.yaml models: 段中定义")
        return 2

    model_name = get_role_model_name(args.role) or args.role
    client = get_llm_client(args.role)
    base = getattr(client, "api_base", "?")
    api_key = str(role_cfg.get("api_key", ""))
    masked = (api_key[:6] + "…") if api_key and not api_key.startswith("${") else api_key
    print(f"role={args.role}  model={model_name}")
    print(f"api_base={base}  api_key={masked or '<空>'}")
    if api_key.startswith("${"):
        print(f"⚠️ api_key 未解析（环境变量缺失），请先在 .env / shell 设置 {api_key[2:-1]}")

    if args.list_models:
        import requests
        url = str(base).rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            resp = requests.get(url, headers=headers, timeout=20)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 无法访问 {url}: {type(exc).__name__}: {exc}")
            return 1
        if resp.status_code != 200:
            print(f"❌ {url} → HTTP {resp.status_code}: {resp.text[:200]}")
            return 1
        try:
            ids = [m.get("id", "?") for m in resp.json().get("data", [])]
        except Exception:  # noqa: BLE001
            print(f"⚠️ 非标准 /models 响应: {resp.text[:200]}")
            return 1
        print(f"✅ {url} 可用模型 {len(ids)} 个：")
        for mid in sorted(ids):
            tag = ""
            low = mid.lower()
            if "qwen" in low:
                tag = " ← qwen"
            elif "deepseek" in low:
                tag = " ← deepseek"
            print(f"   {mid}{tag}")
        return 0

    params = {"max_tokens": args.max_tokens, "temperature": 0.1}
    if args.effort:
        params["effort"] = args.effort

    t0 = time.time()
    try:
        out = client.generate(
            prompt=args.prompt,
            model_name=model_name,
            **params,
            extra_body=get_task_extra_body(args.task),
        )
    except Exception as exc:  # noqa: BLE001 — 探针需要展示任意异常分类
        dt = time.time() - t0
        print(f"❌ 调用失败（{dt:.1f}s）: {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    dt = time.time() - t0
    print(f"✅ {dt:.1f}s  回复: {str(out).strip()[:200]!r}")
    try:
        print("usage:", client.get_cumulative_usage())
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
