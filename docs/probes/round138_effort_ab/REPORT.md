# Round 138：润色 effort A/B 探测（max vs xhigh）——结论与归档

- 跑批：2026-11-01（原始结果 /tmp/sagatrans_effort_ab/20261101_0623/）；归档：2026-11-02
- 脚本：`scripts/probe_effort_ab.py`（rewrite 轨道；judge 轨道未跑）
- 数据：本目录 results.jsonl / summary.md / preflight.json / rewrite/*.json

## 问题
高 effort（max）相对 xhigh，对"困难块润色"的最终质量增益是否值得其时间/失败代价？

## 方法
- 样本：12 个 diff=4.0 困难块（ch01/ch02，生产 retry/低质量分优先选取）
- 双档生成：DeepSeek-V4.1-Flash-1017-SS，生产同构 rewriter prompt（同温度）
- 裁判：Jev 双盲 pairwise（随机 A/B 标签；低置信降级 critic 兜底）
- 档位预检：stripped=false，max/xhigh 均被网关真实接受（排除"参数被剥离"伪 A/B）
- 不干扰保证：workflow.db 只读、decision_db 临时拷贝、结果只写 /tmp

## 结果

| 指标 | max | xhigh |
|---|---|---|
| 判官块级胜场 | 0 | 2（ch01_chunk021、ch02_chunk004） |
| 决定性维度票（63 票内） | 1 | 4（tie 18 / skipped 40） |
| 调用失败 | 3/12（全为 no_content：推理 51k–54k 字符烧光预算） | 1/12（同签名，推理 29k） |
| 延迟中位数 | 130.3s | 86.6s（+50%） |

- max 崩而 xhigh 一次成功的块：ch02_chunk080、ch01_chunk062。
- 任何块均无 max 胜出；Jev 低置信跳过 40/63 票（样本小），但"max 零胜"按脚本口径即可认定无增益，失败率/延迟为硬数据。

## 结论与已应用变更（2026-11-02）
- `task_routing.literary_rewrite.effort`: max → **xhigh**
- `pipeline.difficulty_routing.hard_effort`: max → **xhigh**（另据 Round 132：MiniMax 高难 max ~10% 推理失控）
- xhigh 偶发 no_content 由 Round 133 重试降档链（xhigh→high→…）兜底，机制保留。
- 未测：judge 轨道（MiniMax max vs xhigh）——等主流程空窗期、配额宽裕时再跑。

## 复现
```bash
ALL_PROXY=socks5h://127.0.0.1:10808 python3 scripts/probe_effort_ab.py rewrite --n 12 --sleep 5
ALL_PROXY=socks5h://127.0.0.1:10808 python3 scripts/probe_effort_ab.py judge --minimax --n 6   # 直接消耗 MiniMax 配额，谨慎
```
