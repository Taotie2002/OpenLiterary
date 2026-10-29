# SagaTrans

```
   _____                  ______
  / ___/____ _____ _____ /_  __/________ _____  _____
  \__ \/ __ `/ __ `/ __ `// / / ___/ __ `/ __ \/ ___/
 ___/ / /_/ / /_/ / /_/ // / / /  / /_/ / / / (__  )
/____/\__,_/\__, /\__,_//_/ /_/   \__,_/_/ /_/____/
           /____/
```

**AI 文学语义编译系统** —— 基于多 Agent 协同与增量回溯机制的长篇文学翻译系统，专为解决长篇文学翻译中的术语不一致、风格坍缩与典故缺失问题而设计。

> 翻译不是搬运，而是编译：**语义提取 → 中间表示 → 目标语言重构**。

## 核心设计哲学

- **文学编译而非翻译**：整本书被解构为编译流水线，每个阶段都有可检查的中间产物（chunk 初译 / 润色稿 / 维度评分 / 裁决记录），任何质量问题都能定位到具体阶段与片段。
- **决策驱动**：`Decision DB`（SQLite）维护全局翻译决策（术语、典故、风格、注释策略）；决策变更会触发受影响 chunk 的增量回溯重译，保证全书一致。
- **多模型分工路由**：`literal_translator`(Qwen) 承担直译与考据召回；`secondary`(DeepSeek) 承担文学润色（Round 129 起，实测 max 稳定）、审计评分与考据确认；`primary`(MiniMax-M3.1) 保留最终裁决、风格剖面、成书审读与困难片段直译——三家族分工降低同源盲区。思考档位按任务配置（`task_routing.<task>.effort`，合法档位逐家校验、2026-10 实测；如 judge=max、rewrite=max、critic=none、直译=none；`none`=显式关闭思考），高难度片段经难度路由切至主力模型并加档，停滞救稿/双稿第二稿同样加档。**配额韧性（Round 129）**：MiniMax 配额/限流持续耗尽时 primary 角色自动降级到 secondary，流水线不中断。
- **动态风格剖面**：风格提示词由 `StyleProfiler` 从原文抽样动态生成（按作品隔离落盘 `input/book_profile.json`），不硬编码任何作品先验。
- **Jev 仲裁核心**：TypeSafe System One 概率校准裁决；不可用时按 `confidence_source` 分档降级并标记未校准（绝不静默当作 Jev 结论使用）。

## 流水线

```
split → init → pipeline → style-profile → consolidate → finalize
      → notes-normalize → lint → consistency → (golden-gate) → gate → build_epub
```

- **pipeline**：chunk 级「直译 → 文学润色 → 批评评分 → 裁决 → 回溯」闭环，状态机 + SQLite 持久化，可断点续跑。
- **回溯治理（Round 130）**：决策变更默认**章末合并式回溯**（`decision_engine.backtrack_mode: deferred`）——先登记 `pending_backtrack` 队列，章节队列清空时合并应用一次；单块累计回溯轮数（`max_backtrack_rounds_per_chunk`）与章级合并轮数（`max_backtrack_sweeps_per_chapter`）超预算即冻结（明细见 `output/chXX/backtrack_frozen.md`）；重跑新版 critic 均分低于备份版时自动保留旧版（`pipeline.version_merit`）。设 `backtrack_mode: immediate` 可回退旧行为。
- **consolidate / finalize**：跨 chunk 合并、章内衔接与终稿整合，产出 `output/ch*/ch*_chunk*_final.json` → `output/translated_full.md`。
- **lint / consistency / gate**：脚注、引号、韵文、双关、术语一致性与交付门禁。

## 质量体系

- **交付门禁 gate（七项）**：状态机完成度 / 产物完整性 / QA_REPORT / lint / EPUB→MD 转换审计（覆盖率） / 韵文校验 / 双关核销。
- **成书级审读** `scripts/editor_review.py`：P0/P1/P2 分级（悬空引语、翻译腔、语域漂移、注密、标点规范等），支持 `--chapters` 定点复审与 `--fail-on-p0` 门禁。
- **文学指标** `scripts/literary_metrics.py`：AI 味密度 / 翻译腔计数 / 重复率。
- **双关核销** `scripts/pun_audit.py`：全书双关点位登记与覆盖门禁（gate ⑦ 已内联自动运行，无需手动补跑）。
- **Golden Set**：基准样本跑分（`python3 -m src.main golden`）。

## 快速开始

### 1. 安装

```bash
pip install -r requirements-core.txt   # 核心最小依赖
pip install -r requirements.txt        # + EPUB 解析/打包等
```

Python >= 3.10。

### 2. 配置 key

```bash
cp .env.example .env   # 填入真实 key（.env 不入库）
```

| 变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 文学润色 / 审计评分 / 考据确认（`secondary`）：自建 DeepSeek 网关（默认 `api_base: token.ai-hf.cn`，模型 Deepseek-V4.1-Flash-1017-SS） |
| `QWEN_API_KEY` | 直译 / 考据召回（`literal_translator`）：自建 Qwen 网关（默认同上 base；模型名在 `config.yaml`，可用探针 `--list-models` 核对） |
| `MINIMAX_API_KEY` | 主力模型：最终裁决 / 风格剖面 / 成书审读 / 困难片段直译（api.minimax.cn）；配额耗尽时自动降级到 secondary |
| `TYPESAFE_API_KEY` | TypeSafe Jev 仲裁（可选；缺失时自动降级并标记未校准） |

### 3. 一键翻译

```bash
./scripts/translate_book.sh book.epub            # 全自动：EPUB→MD→风格剖面→全流程→门禁
./scripts/translate_book.sh book.epub --reset    # 重置后重跑
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--skip-epub` | 跳过 EPUB→MD 转换（`input/ch*.md` 已就绪时） |
| `--merge-only` | 只重跑合并/终稿/门禁，不调用 LLM |
| `--consistency` | 追加一致性校正 |
| `--golden-gate` | 追加 Golden Set 门禁 |
| `--dry-run` | 只走流程骨架（mock 后端） |

> 切分自带清理（`config.yaml` → `input_prep`）：前页/目录/索引等非正文写入
> `input/_skip/`（不参与翻译），文本伪影按修复表自动替换。`--reset` 重建
> `input/` 后的效果与手工清理一致，无需再加 `--skip-epub`。

### 3.1 维护：重跑单个章节

```bash
# 官方推荐（smart-force）：保留 COMPLETED，把 PERMANENTLY_FAILED / PENDING 重置重试
python3 -m src.translator_agent init --chapter ch09 --force
python3 -m src.translator_agent pipeline --chapter ch09   # 管线末尾自带章节级 consistency / 实体审计 / cost / 清理

# 完全干净重跑：连 COMPLETED 一起重译，并清掉旧产物
python3 - <<'EOF'
import sqlite3
c = sqlite3.connect('db/workflow.db')
print('deleted', c.execute("DELETE FROM chunk_tasks WHERE chapter_id='ch09'").rowcount)
c.commit()
EOF
rm -rf output/ch09
python3 -m src.translator_agent init --chapter ch09
python3 -m src.translator_agent pipeline --chapter ch09
```

> 单实例守卫已内置（Round 130）：`scripts/translate_book.sh` 与
> `python3 -m src.translator_agent <写库命令>` 共用一把 flock（`db/.sagatrans.lock`），
> 第二个实例会被直接拒绝并打印持有者 pid/启动时间；应急放行用
> `SAGATRANS_ALLOW_CONCURRENT=1`（风险自担）。历史教训：双进程会让 MiniMax 配额
> 瞬间打爆（日志特征：大量 `429 api.minimax.cn`、`非法转移`、chunk 状态互相覆盖，
> 整章被重译两遍）；若在旧版本见到该特征请先停掉所有进程，再按本节 smart-force 重跑。
> 若要用 Jev，确认正确的 `TYPESAFE_API_KEY` 且代理可用（`ALL_PROXY=socks5h://...`）。
> Round 129 起单进程内 MiniMax 配额/限流持续耗尽会自动降级到 DeepSeek（stderr
> 打 `[Quota Fallback]`），不会再长时间卡在 429 退避；配额 5 小时窗重置后重启恢复主力。

### 4. 出书（EPUB3）

```bash
python3 scripts/build_epub.py --md output/translated_full.md --out output/book.epub
```

产物含：书名页、按 `# 标题` 切分的章节、目录（nav + NCX）、`[^n]` 尾注超链接、内置样式；书名/作者默认从文档头部抽取，可用 `--title/--author` 覆盖。

## 目录结构

```
.
├── config.yaml            # 全局配置：per-role 后端/模型、任务路由、质量阈值
├── db/                    # 运行态：工作流 DB + Decision DB + chunk 标签（不入库）
├── docs/                  # 本地工作文档（审读指南/审计/基线；默认不入库）
├── input/                 # 原文 MD（EPUB 自动转换）与书级风格剖面
├── output/                # 译文与 QA 报告（不入库）
├── scripts/
│   ├── translate_book.sh  # 一键全流程
│   ├── build_epub.py      # EPUB3 出书
│   ├── editor_review.py   # 成书级编辑审读
│   ├── literary_metrics.py# 文学指标
│   ├── pun_audit.py       # 双关核销
│   └── build_integrated.py# 单体聚合构建器
├── src/
│   ├── translator_agent.py# 生产入口（单体聚合版）
│   ├── main.py            # CLI 统一入口
│   ├── pipeline.py        # 模块化流水线
│   ├── core/              # 调度器 / 决策引擎
│   ├── agents/            # 直译/润色/批评/裁决 4 个 Agent
│   ├── judges/            # Jev / pairwise 裁决实现
│   ├── utils/             # 切分 / 文本匹配 / LLM 适配
│   ├── style_profiler.py  # 动态风格剖面
│   ├── verse.py           # 韵文通道
│   ├── consistency.py     # 一致性校正
│   └── book_lint.py       # 成书 lint
├── tests/                 # round 粒度回归测试
└── tools/                 # 辅助脚本（epub_to_md 等）
```

## 环境变量覆盖（不改 config.yaml 切换模型）

```bash
export SAGATRANS__LLM_BACKEND=openai_api
export SAGATRANS__MODELS__PRIMARY__MODEL_NAME=MiniMax-M3.1-Flash
export SAGATRANS__MODELS__PRIMARY__API_BASE=https://api.minimax.cn/v1
export SAGATRANS__MODELS__PRIMARY__API_KEY=$MINIMAX_API_KEY
./scripts/translate_book.sh book.epub
```

## 模型配置（config.yaml `models:` 段，唯一事实源）

每个角色一块，集中写连接 + 模型名（Effort 档位在 task_routing，见下）：

```yaml
models:
  literal_translator:
    api_base: https://token.ai-hf.cn/v1
    api_key: ${QWEN_API_KEY}
    model_name: Qwen3.8-Flash-Next-SS
  primary:   { ... }    # MiniMax-M3.1-Flash
  secondary: { ... }    # Deepseek-V4.1-Flash-1017-SS
```

`primary` 可配 `fallback_role: secondary`（Round 129 配额韧性）：MiniMax 配额/限流
持续耗尽（429 连续 / 配额类 402-403）时，primary 角色调用永久降级到 secondary，
流水线不中断；降级为进程内单向（重启恢复主力）。

思考档位是任务级配置（`task_routing.<task>.effort`；缺省=不下发参数）。
合法档位逐家校验（2026-10 实测）：MiniMax `low/medium/high/xhigh/max`（服务端强制
adaptive，无 none 档）；DeepSeek `none/low/high/xhigh/max`；Qwen `none/low/medium/xhigh`
（`none` = 显式关闭思考；YAML 裸 `off` 会被归一为 `none`）。
供应商特有透传参数可用任务级 `extra_body`（最后合并、同名覆盖）：

```yaml
task_routing:
  judge_decision:
    model: primary
    effort: max
```

本地推理（MLX）支持已移除；全部模型均为 OpenAI 兼容 HTTP 接口。

## 测试

```bash
python3 -m pytest tests/ -q     # 回归测试
python3 -m src.main golden      # Golden Set 跑分
python3 -m src.main gate        # 交付门禁
```

## 说明

- 请仅对**公有领域或已获授权**的作品使用本系统，翻译产物版权责任由使用者承担。
- `.env`、`db/`、`output/`、`input/ch*.md` 等运行态文件默认不入库（见 `.gitignore`）；`docs/` 下的本地工作文档同样不上传。
- 网络受限环境可在 shell 层配置代理（如 `ALL_PROXY=socks5://127.0.0.1:10808`）。
