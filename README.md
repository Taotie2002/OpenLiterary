# SagaTrans

> **OpenLiterary · AI 文学语义编译系统**

[GitHub Repository](https://github.com/Taotie2002/OpenLiterary) · [Quick Start](#快速开始) · [Architecture](#架构)

SagaTrans 把长篇文学翻译实现为一条可检查、可回滚的编译流水线：

**切分 → 直译 → 文学润色 → 多维批评 → 裁决 → 决策回溯 → 合并 → 终审 → 交付门禁**

它面向的不是短文本翻译，而是整本书：需要统一术语、保持叙述声音、追踪伏笔回收，并在中断、限流和模型失败后继续运行。

## 为什么是“编译”

普通翻译流程把原文直接交给一个模型，得到译文；出现问题时只能重译或人工通读。SagaTrans 把这件事拆成带中间产物的编译过程：

- **每个 chunk 都有可检查状态**：初译、润色稿、critic 报告、judge 裁决和回溯记录分别落盘。
- **全局决策不依赖 prompt 记忆**：术语、译名、典故和风格决策写入 SQLite Decision DB，变化时定向重跑受影响片段。
- **判断层与执行层分离**：全书深读只产出 advisory 注记制品；执行层继续按 chunk 工作，不被一次超长调用绑死。
- **质量门禁是确定性的**：术语一致性、脚注、韵文、双关、文本完整性和交付检查不交给单一 LLM 自行判断。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 多 Agent 翻译闭环 | 直译、文学润色、五维 critic、最终裁决与回溯重译组成 chunk 级状态机 |
| Decision DB | 全局维护术语、译名、典故和风格决策；冲突显式记录，变更触发定向回溯 |
| 动态风格剖面 | 从当前作品原文生成，不硬编码任何书先验；支持 `style-profile --deep` 全书分段深读 |
| 全书判断层 | v2 剖面保存 voice / arc / setup-payoff 注记，经 L1 原文匹配与 L2 交叉复核后作为 advisory 制品 |
| 故事知识图谱 | `story-graph` 从实体注册表与章节原文确定性构建出现/共现边，叠加已验证的叙事边，并输出显式 `knowledge_gaps` |
| 成书级终审 | `editor_review.py` 支持逐章与 `--global` 弧线审读；结构 P0 可经机械锚句校验后写入 `review_dirty` |
| 可回滚治理 | 章末合并式回溯、chunk/章级预算、PF 复活、旧稿 merit 保护、哈希绑定的 review-fix sidecar |
| 交付工具链 | 成书清洗、一致性校正、双关核销、文学指标、交付 gate 与 EPUB3 导出 |

## 架构

```text
 EPUB / TXT / MD
        │
        ▼
      split ───────────────► input/chXX.md
        │                         │
        ▼                         ▼
  workflow DB  ◄──────────  chunk task state machine
        │                         │
        │        ┌────────────────┴────────────────┐
        │        ▼                                 ▼
        │   literal translation ──► literary rewrite
        │        │                                 │
        │        └──────────────► critic ◄─────────┘
        │                           │
        │                           ▼
        └──────────────────────► judge / Jev
                                    │
                         decision change? ──► backtrack queue
                                    │
                                    ▼
                           final chunk artifacts
                                    │
             consolidate ─► finalize ─► notes-normalize
                                    │
             lint / consistency / editor review / gate
                                    │
                                    ▼
                         output/<中文书名>.md ─► EPUB3
```

## 全书判断层

Round 131 的判断层是一条独立的离线制品链，不替代 Decision DB，也不直接修改译文：

1. `style-profile --deep` 按章分段读取全书，重叠窗口跨段合并，避免把整本书塞进一次调用。
2. 注记统一为 `voice | arc | setup_payoff` envelope，包含章节范围、claim、原文证据、置信度、状态和来源模型。
3. **L1** 对切分后的 `input/chXX.md` 做 exact → normalized → fuzzy 匹配；无证据即丢弃。
4. **L2** 由低成本模型小批复核；术语/译名规范性断言直接拒绝，硬约束只属于 Decision DB。
5. 通过验证的 active 注记用于全书故事图与 `editor_review --global`；运行时是否注入由 A/B 单独控制。

> 当前 `judgment_injection.enabled: false`。真实 ch03 Jev 校准 A/B 中两臂均 5/5 PASS，
> critic 均分 Δ=-0.07，按 L3 协议保持关闭。深读剖面仍供故事图和全书终审使用；只有新的
> Jev 校准 A/B 没有回归时才应重新开启。

失败策略是不对称的：**注记验证失败 fail-open**（丢弃或 dormant，不阻塞翻译）；**交付门禁 fail-closed**（硬闸不通过则不交付）。

## 模型路由

所有模型均为 OpenAI-compatible HTTP 后端；模型定义在 `config.yaml` 的 `models` 段，任务到模型的映射在 `task_routing`。

| 角色 | 默认后端 | 主要职责 |
| --- | --- | --- |
| `literal_translator` | Qwen | 字面直译、典故/实体召回、低成本结构化分类 |
| `secondary` | DeepSeek | 文学润色、critic、考据确认、深读抽取/合并 |
| `primary` | MiniMax | 最终裁决、风格剖面、成书审读、困难片段直译 |
| Jev | TypeSafe（可选） | 概率校准裁决；不可用时降级并显式标记 `confidence_source` |

设计要点：

- 三个模型家族分工，降低同源评分盲区；思考档位按任务配置，例如 `judge=max`、`rewrite=xhigh`、`critic=none`。
- 困难 chunk 可路由到 `primary` 并提高思考档；停滞救稿和双稿第二稿也会加档。
- MiniMax 配额持续耗尽时，`primary` 可在进程内单向降级到 `secondary`，流水线不中断。
- 所有 API key 使用环境变量占位；`.env`、DB、译文和原文运行产物均不入库。

## 快速开始

### 环境

- Python **3.10+**
- 可访问的 OpenAI-compatible 模型端点
- 可选：TypeSafe API key（用于 Jev 校准裁决）

```bash
git clone https://github.com/Taotie2002/OpenLiterary.git
cd OpenLiterary

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt   # 核心管线
pip install -r requirements.txt        # 额外包含 EPUB 工具
```

### 配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入 QWEN_API_KEY / DEEPSEEK_API_KEY / MINIMAX_API_KEY
# TYPESAFE_API_KEY 可选
```

模型地址、模型名、超时和质量阈值统一在 `config.yaml` 调整。临时切换模型无需改文件：

```bash
export SAGATRANS__MODELS__PRIMARY__MODEL_NAME=your-model
export SAGATRANS__MODELS__PRIMARY__API_BASE=https://your-endpoint/v1
export SAGATRANS__MODELS__PRIMARY__API_KEY=$YOUR_KEY
```

### 一键翻译

```bash
./scripts/translate_book.sh book.epub
./scripts/translate_book.sh book.epub --reset      # 清空状态后重跑
./scripts/translate_book.sh book.epub --skip-epub  # 复用已切分的 input/chXX.md
```

脚本依次完成 EPUB 转换、切分、初始化、逐章翻译、深读剖面、合并、终稿清洗、lint/gate 和故事图构建；任一步失败都会保留可恢复状态。

### 手动流程

```bash
python3 -m src.translator_agent split --input book.epub
python3 -m src.translator_agent init --chapter ch01
python3 -m src.translator_agent pipeline --chapter ch01
python3 -m src.translator_agent style-profile --deep
python3 -m src.translator_agent consolidate --max-rounds 3
python3 -m src.translator_agent finalize --apply
python3 -m src.translator_agent notes-normalize --apply
python3 -m src.translator_agent lint
python3 -m src.translator_agent gate
python3 scripts/build_epub.py --out output/book.epub
```

## 常用命令

| 命令 | 用途 |
| --- | --- |
| `pipeline --chapter chXX` | 跑一章的翻译/润色/critic/judge/回溯闭环 |
| `golden` / `golden-gate` | Golden Set 跑分与质量门禁 |
| `golden --judgment-ab --chapter chXX` | 只比较判断层注记 off/on，不乘入模型×effort 矩阵 |
| `style-profile --deep` | 全书分段深读、L1/L2 验证并生成 v2 剖面 |
| `story-graph --require-answerable` | 重建故事图；实体索引不可答时退出码 2 |
| `editor_review.py --global --review-dirty` | 成书级全书审读并生成机械校验过的结构修复清单 |
| `pipeline --fix-from-review` | 读取 `review_dirty.json`，定向重跑结构 P0 chunk |
| `consistency` / `lint` / `gate` | 一致性、成书格式与交付门禁 |
| `drift-audit --jev` | 翻译漂移与 Jev 窗口一致性诊断 |
| `memory` | 长文内存压力自检 |

## 质量体系

- **Chunk 级**：五维 critic + judge 裁决；低分可触发救稿、双稿或回溯。
- **章级**：实体名归一、音译漂移统一、术语/典故回溯和章末一致性报告。
- **成书级**：`editor_review.py` 检查悬空承诺、重复注释、翻译腔、声音断裂、注释过密和弧线违背。
- **结构修复**：只有机械锚定且唯一落块的 P0 才进入 `review_dirty`；sidecar 绑定原 final 哈希，旧反馈不会污染后续重跑。
- **交付级**：状态机完成度、产物完整性、QA、book lint、EPUB→MD 审计、韵文、双关和最终 gate。

长篇运行还有三条硬规则：

1. `scripts/translate_book.sh` 与写库 CLI 共用 `db/.sagatrans.lock`，第二个实例会被拒绝。
2. Ctrl+C 第一次只警告，第二次才中止；已完成 chunk 与 DB 状态保留。
3. 修改散装 `src/` 后必须重建单体并校验：

```bash
python3 scripts/build_integrated.py
python3 scripts/build_integrated.py --check
python3 scripts/sync_book_lint_mirror.py --check
```

## 产物

| 路径 | 内容 |
| --- | --- |
| `output/chXX/chXX_chunk*_raw.json` | 直译中间稿 |
| `output/chXX/chXX_chunk*_literary.json` | 润色稿 |
| `output/chXX/chXX_chunk*_critic_report.json` | 五维评分与问题说明 |
| `output/chXX/chXX_chunk*_final.json` | 最终 chunk 稿 |
| `output/<中文书名>.md` | 合并、清洗后的成书文本 |
| `output/style_profile_audit.*` | 深读注记验证、丢弃与状态明细 |
| `input/book_profile.json` | 书级风格/叙事剖面（运行产物，不入库） |
| `input/story_graph.json` | 全书实体/故事图（运行产物，不入库） |
| `output/editor_review*.json` | 成书审读结果与锚点校验信息 |
| `output/review_dirty.json` | 可定向修复的结构 P0 清单 |
| `output/book.epub` | EPUB3 成品 |

## 测试

```bash
python3 -m pytest -q tests/
python3 -m pytest -q tests/test_round131_*.py
python3 -m pytest -q tests/test_round119_mirror_parity.py
```

当前工作树已在 Python 3.10 与 3.13 下通过全量回归，并在单体镜像、book_lint 镜像和真实模型样本上完成验收。涉及付费模型的 A/B 与终审必须使用校准后的 judge 结果；mock 或降级结果不会被当作启用依据。

## 目录

```text
.
├── config.yaml              # 模型、任务路由、流水线与质量门槛
├── scripts/
│   ├── translate_book.sh    # 一键全流程
│   ├── build_epub.py        # EPUB3 导出
│   ├── editor_review.py     # 成书级终审 / review_dirty
│   ├── build_integrated.py  # 散装 src → 单体入口
│   └── sync_book_lint_mirror.py
├── src/
│   ├── translator_agent.py  # 生产单体入口（由 src/ 生成）
│   ├── main.py              # 模块化 CLI
│   ├── pipeline.py          # chunk 状态机与质量闭环
│   ├── judgment_layer.py    # 注记 schema、L1/L2、切片注入
│   ├── story_graph.py       # 实体/叙事知识图谱
│   ├── style_profiler.py    # 抽样与全书深读剖面
│   ├── core/                # scheduler / decision engine
│   ├── agents/              # 直译 / 润色 / critic / rewriter
│   ├── judges/              # Jev / pairwise 裁决
│   └── utils/               # LLM、切分、文本匹配
├── tests/                   # round 粒度回归测试
└── tools/                   # EPUB 转换等辅助工具
```

## 运行说明

- 本系统只应处理**公有领域或已获授权**的作品；使用者承担译文版权责任。
- 外部代理、模型端点和配额策略因部署环境而异；网络受限时在 shell 层配置 `ALL_PROXY`。
- `input/ch*.md`、`input/book_profile.json`、`input/story_graph.json`、`db/`、`output/` 和 `docs/*.md` 中的本地工作文档默认不入库。
- `src/` 是事实源，`src/translator_agent.py` 是生成产物；不要直接编辑单体文件。

## 设计原则

**可观测优先，确定性优先，失败可恢复。**

LLM 负责提出语言与叙事判断；代码负责约束候选、验证证据、维护状态、执行回溯并决定是否交付。模型可以失败，流水线不应因此丢失已完成的工作。
