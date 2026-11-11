# Round 145 A/B: MoArk d3-flash vs d3-turbo

日期：2026-11-10  
样本：`ch07_chunk019/032/077/157/159`，实体表使用 Round 144 清洗后的 `/tmp` 副本。  
后端：真实 MoArk `/v1/systemone`，`backend_kind="jev"`（保留校准阈值档），审计记录真实 `provider=moark`。

| chunk | 实体数 | 问题数 | d3-flash | d3-turbo | 均分 Δ | 备注 |
|---|---:|---:|---|---|---:|---|
| 019 | 1 | 7 | REJECT | REJECT | -0.171 | 均为 Jessica 缺失 |
| 032 | 0 | 6 | PASS | PASS | -1.300 | 无实体问题 |
| 077 | 1 | 7 | PASS | PASS | -1.143 | 均触发 Irulan 假阳性护栏 |
| 157 | 18 | 24 | PASS | PASS | -0.328 | 实体覆盖充分 |
| 159 | 31 | 37 | PASS | REJECT | -0.714 | d3-turbo 多 1 个 reject |

结论：

- d3-flash 判定更稳定，保持全局默认。
- d3-turbo 在 5/5 块均分更低，并在 chunk159 发生 PASS→REJECT 漂移；不设为默认，只用于确认困难 chunk 的 A/B。
- 37 问实测约 23 秒；64 问单批可能超过旧 30 秒，`typesafe.timeout` 已调为 60 秒。
- `output_tokens` 当前均为 0（MoArk 未上报），成本监控不依赖该字段。

Key 备注：本机 `TYPESAFE_API_KEY` 仍是过期 Jev key，`MOARK_API_KEY` 才是可用 d3 key。配置已改为 `${MOARK_API_KEY}`；后端 key 优先级为显式参数 > config/MOARK env > legacy TYPESAFE env，避免旧 key 反向覆盖。
