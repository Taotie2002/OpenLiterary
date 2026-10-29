#!/usr/bin/env python3
"""SagaTrans 流式日志过滤器。

用途：给管道中的每一行输出加时间戳，并剥离 ANSI 颜色码，
同时保证「无换行符的片段」（如交互式提示）立即透传。

用法：
    ./scripts/translate_book.sh book.epub --reset 2>&1 \
        | python3 tools/stream_log_filter.py | tee logs/YYYYMMDD_book.run.log

注意：不要用 `perl -pe` 做同类过滤 —— perl 按“行”读取，遇到不含
换行符的提示（如 `确认清空？[y/N] `）会一直缓存到用户按键后才吐出，
表现为“提示不显示，按了键才出现”。本脚本按字节块流式透传，无此问题。
"""
from __future__ import annotations

import os
import re
import sys
import time

_CSI = re.compile(rb"\x1b\[[0-9;?]*[ -/]*[@-~]")           # 完整 CSI 序列（颜色等）
_OSC = re.compile(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")   # 完整 OSC 序列（标题等）
_PARTIAL = re.compile(rb"\x1b(?:\[[0-9;?]*[ -/]*|\][^\x07\x1b]*)?\Z")


def _ts() -> bytes:
    return time.strftime("[%Y-%m-%d %H:%M:%S] ").encode("utf-8")


def main() -> int:
    src = sys.stdin.buffer
    dst = sys.stdout.buffer
    at_line_start = True
    hold = b""
    try:
        while True:
            chunk = src.read1(65536)
            if not chunk:
                break
            if hold:
                chunk = hold + chunk
                hold = b""
            # 尾部的“不完整转义序列”留到下一块，避免跨块截断
            m = _PARTIAL.search(chunk)
            if m:
                hold = chunk[m.start():]
                chunk = chunk[:m.start()]
            chunk = _OSC.sub(b"", chunk)
            chunk = _CSI.sub(b"", chunk)
            if not chunk:
                continue
            out = bytearray()
            parts = chunk.split(b"\n")
            for seg in parts[:-1]:
                if at_line_start:
                    out += _ts()
                out += seg + b"\n"
                at_line_start = True
            tail = parts[-1]
            if tail:
                if at_line_start:
                    out += _ts()
                out += tail
                at_line_start = False
            dst.write(bytes(out))
            dst.flush()
    except BrokenPipeError:
        # 下游（如 tee 之后用户中断）关闭时静默退出
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
