#!/usr/bin/env python3
"""Recompute the final run-log seal after shell-level tee writes finish."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.task_logging import now_iso, sha256_file, validate_jsonl  # noqa: E402


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a SagaTrans run log")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else Path(
        os.environ.get("SAGATRANS_TASK_LOG_DIR") or (PROJECT / "output" / "logs" / "runs"))
    jsonl = log_dir / f"{args.run_id}.jsonl"
    text = log_dir / f"{args.run_id}.log"
    seal = log_dir / f"{args.run_id}.sha256"
    if not jsonl.exists() and not text.exists():
        print(f"ℹ️ [TaskLog] 未找到运行日志，跳过封存: {args.run_id}")
        return 0
    validation = validate_jsonl(jsonl, expected_run_id=args.run_id)
    payload = {
        "run_id": args.run_id,
        "sealed_at": now_iso(),
        "sealed_by": "shell_finalizer",
        "status": "validated" if validation.get("ok") else "partial",
        "jsonl": str(jsonl),
        "log": str(text),
        "jsonl_sha256": sha256_file(jsonl) if jsonl.exists() else "",
        "log_sha256": sha256_file(text) if text.exists() else "",
        "seq_last": int(validation.get("seq_last") or 0),
        "validation": validation,
    }
    _atomic_write(seal, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"🔐 [TaskLog] 已封存 {args.run_id} "
          f"(jsonl={payload['jsonl_sha256'][:12]}, log={payload['log_sha256'][:12]})")
    try:
        from src.book_linker import update_manifest_run_evidence
        manifest_path = PROJECT / "output" / "build_manifest.json"
        if update_manifest_run_evidence(PROJECT, manifest_path, args.run_id, payload):
            print("🧾 [BookLink] build manifest 已补写日志封存哈希")
    except Exception as exc:  # noqa: BLE001 - sealing is still valid without a manifest
        print(f"⚠️ [BookLink] manifest 封存哈希回写失败: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
    return 0 if validation.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
