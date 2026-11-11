# SagaTrans

> **OpenLiterary · AI 文学语义编译系统**

[GitHub Repository](https://github.com/Taotie2002/OpenLiterary)

SagaTrans 把一个长篇小说翻译任务实现为一条**有状态、可审计、可回滚的语义编译流水线**。它不把整本书塞进一次超长对话，而是把翻译拆成章节、chunk、决策、判断注记和交付制品，让每个中间结果都能被检查、重放和定向修复。

```text
切分 → 风格剖面 → 直译 → 文学润色 → 多维审计 → 校准裁决
     → 决策回溯 → 全书收敛 → 合并 → 终审 → 交付门禁 → EPUB3
```

---

## 1. 这项工程在研究什么

长篇文学翻译与短文本翻译的困难不同。一本书会持续几百万字符，超出任何单一上下文窗口；人物称谓、叙述声音、典故、伏笔和章节语气又必须在全书范围内保持一致。简单地把书切碎后逐段调用模型，会产生三个系统性问题：

1. **全局约束失忆**：后面的 chunk 看不到前面的术语和风格决定，译名与语气逐步漂移。
2. **质量信号不可靠**：生成模型既写译文又给自己打分，容易形成同源盲区。
3. **失败不可逆**：一次超时、限额或错误裁决可能导致重译整章，且无法判断旧稿是否更好。

SagaTrans 的研究立场是：**把翻译当作编译，把一致性当作事实数据库问题，把质量当作证据裁决问题，把交付当作状态机不变量问题。** 模型负责提出语言候选，代码负责维护约束、验证证据、调度重跑并决定能否交付。

### 核心命题

| 命题 | 工程实现 |
| --- | --- |
| 长篇翻译需要跨上下文记忆 | SQLite Decision DB 保存术语、译名、典故与风格决策，变化时定向回溯 |
| 生成与判断应尽量解耦 | 直译/润色模型与 System One 校准判断后端分离；未校准路径显式标记 |
| 判断应当是制品，而不是隐式 prompt 状态 | 全书注记写入 `book_profile.json`，经 L1/L2 验证后供检索与终审使用 |
| 全局理解不应阻塞局部执行 | 判断层是 advisory 且 fail-open；交付门禁是硬约束且 fail-closed |
| 不确定性必须可恢复 | 每个 chunk 有版本、备份、merit 比较、DIRTY 队列和 PF 复活预算 |
| 交付不能依赖“队列看起来跑完了” | 八项确定性 gate 通过才算完成，mock/未校准裁决占比也会被检查 |

---

## 2. 设计主张

### 2.1 文学编译，而不是“大 prompt 翻译”

一个 chunk 会经历直译、润色、审计、裁决和可能的回流。每一步都有独立产物：`raw.json`、`literary.json`、`critic_report.json`、`final.json`、回溯/双稿 sidecar。主流程只消费通过状态机检查的制品，不依赖某个模型的“记忆”。

### 2.2 硬约束与软判断分层

- **硬约束**：术语、译名、实体 canonical 形式、脚注/韵文/格式规则、交付状态。它们属于 Decision DB、book lint 和 gate。
- **软判断**：叙述声音、人物语气、章节弧线、伏笔是否回收、平移风格是否偏移。它们可以作为注记进入故事图和审读上下文，但不能覆盖 Decision DB。

这条边界很重要：如果让“全书注记”直接修改译名，系统会得到两个互相矛盾的事实源。SagaTrans 明确规定，注记只能是 advisory；L2 复核也会拒绝包含术语/译名规范性断言的高层注记。

### 2.3 生成模型与判断模型分工

生成端追求文风、句法和创造性；判断端追求可重复、可校准的概率信号。当前默认分工如下：

| 角色 | 默认后端 | 主要职责 |
| --- | --- | --- |
| `literal_translator` | Qwen | 直译、典故/实体召回、轻量分类、判断层 L2 复核 |
| `secondary` | DeepSeek | 文学润色、考据确认、旧 critic 路径、深读抽取/合并 |
| `primary` | MiniMax | 最终 LLM 裁决、风格剖面、成书审读、困难片段直译、救稿/双稿 |
| System One | MoArk `d3-flash` | 校准概率裁决；`d3-turbo` 仅用于同批质量 A/B |

“思考档位”是任务级参数，例如 `judge_decision=max`、`literary_rewrite=xhigh`、`critic_scoring=none`。具体模型、地址、effort 和 `max_tokens` 都在 `config.yaml` 的 `models` 与 `task_routing` 中调整。

### 2.4 失败策略是不对称的

- **判断层 fail-open**：深读注记验证失败、故事图构建失败或查询失败，不阻塞正文翻译；无效注记被丢弃或降为 dormant。
- **交付层 fail-closed**：状态未完成、产物缺失、lint/韵文/双关/校准比例不达标，任何一项为红都不交付。

模型可以失败；流水线不能因此丢失已经完成的工作，也不能把未校准结果伪装成完成品。

---

## 3. 系统模型

### 3.1 四层事实

理解本项目时，建议按“事实层”而不是按“模型调用”来阅读：

| 层 | 位置 | 作用 | 是否可手改 |
| --- | --- | --- | --- |
| 输入层 | `input/chXX.md`、`input/verse_manifest.json`、`input/_conversion_audit.md` | 切分后的原文与转换审计；所有证据匹配都以章源文件为准 | 可检查，不建议手改 MD |
| 决策层 | `db/decision_db.sqlite`、`db/workflow.db` | 全局术语/译名/典故/风格决策，以及 chunk 状态机 | 通过 CLI/脚本修改，禁止并发写 |
| 生成层 | `output/chXX/*_raw/literary/critic_report/final.json` | 每个 chunk 的候选稿、审计、裁决和最终稿 | 可检查；不要绕过状态机直接替换 final |
| 判断/派生层 | `input/book_profile.json`、`input/story_graph.json`、`output/review_dirty.json` | 全书注记、知识图谱、成书审读问题 | advisory；可重建，不反向成为术语事实源 |

`src/` 是代码事实源；`src/translator_agent.py` 是自动生成的单文件镜像。运行产物的默认路径由项目根派生，不在 `config.yaml` 中配置。

### 3.2 Chunk 状态机

```text
PENDING
  └─→ EXTRACTING_TERMS
        └─→ TRANSLATING_RAW
              └─→ REWRITING_LITERARY
                    └─→ AUDITING
                          ├─→ JUDGING ─→ COMPLETED
                          ├─→ REWRITING_LITERARY   (质量问题回流)
                          ├─→ FAILED                (API/执行失败，可重试)
                          └─→ PERMANENTLY_FAILED    (预算耗尽，保留 fallback 稿)

COMPLETED ─→ DIRTY (决策变更/结构 P0 修复触发定向重跑)
DIRTY     ─→ EXTRACTING_TERMS
FAILED    ─→ 任一可恢复阶段
```

- `COMPLETED` 表示 chunk 已通过该轮质量闭环，不等于全书可交付；还需要全书收敛和 gate。
- `DIRTY` 表示旧稿受新决策或终审结构问题影响，必须重跑后才能再次交付。
- `PERMANENTLY_FAILED`（PF）是“本次重试预算内的终态”，不是永久黑洞：可以通过 `pf_requeue_max`、`requeue --force` 或后续回溯治理重新进入 DIRTY。
- 回溯重跑的新稿若 critic 均分不如备份稿，`version_merit` 会保留旧版，防止“越修越差”。

---

## 4. 总体架构

```text
EPUB / TXT / MD
      │
      ▼
split + input_prep ───────────────► input/chXX.md
      │                              input/verse_manifest.json
      │                              input/_conversion_audit.md
      ▼
style-profile (抽样或 --deep) ─────► input/book_profile.json
      │                              （叙述声音/弧线/伏笔注记）
      ▼
init ──────────────────────────────► workflow.db(task states)
      │
      ▼
per-chapter pipeline
  ┌──────────────────────────────────────────────────────────┐
  │ reference extraction → canonical decisions               │
  │ literal translation → literary rewrite                  │
  │ critic/System One audit → judge → rewrite if rejected    │
  │ decision change → DIRTY → chapter-local retry            │
  └──────────────────────────────────────────────────────────┘
      │                              decision_db.sqlite
      ▼
consolidate 全书 DIRTY 收敛 ────────► output/consolidation_diff.md
      │
      ▼
story-graph 重建 ───────────────────► input/story_graph.json
      │
      ▼
finalize → merge → notes-normalize → lint
      │                              output/<中文书名>.md
      ▼
可选：editor_review / story-graph-query / setup-payoff-audit
      │
      ▼
gate（8 项 fail-closed） ──► output/book.epub（可选）
```

---

## 5. 工作流详解

以下顺序与 `scripts/translate_book.sh` 的实际执行逻辑一致。手工调试时也应遵守“先收敛，再合并，再终审，最后 gate”的顺序。

### 5.1 输入转换与切分

1. `split` 接受 `.epub`、`.txt` 或 `.md`，输出 `input/chXX.md`。
2. `input_prep` 把封面、版权页、目录、索引等非正文文档移入 `input/_skip/`，并执行配置中的字间断裂修复。
3. EPUB 转换同时产出 `input/verse_manifest.json` 和 `input/_conversion_audit.md`；后者是 gate ⑤ 的输入。
4. `SmartChunker` 先按标题、`***`/`---`、空行等场景断点切分；`soft_limit` 后优先在断点切，超过 `hard_limit` 才强制切。

### 5.2 风格剖面与全书判断层

翻译前先确保 `input/book_profile.json` 存在：

- `style-profile`：抽样若干章生成 v1 剖面；
- `style-profile --deep`：分段读取全书，跨段合并注记，经 L1 原文匹配和 L2 低成本复核后生成 v2 剖面；
- 判断层注记的类型是 `voice | arc | setup_payoff`，统一 envelope 包含章节范围、claim、evidence、confidence、status；
- 注记是 advisory。L1 只对切分后的章源文件做 exact → 归一化 → 滑窗匹配；L2 拒绝术语/译名规范性断言；
- 当前 `judgment_injection.enabled: false`：真实同章 A/B 没有证明 critic 收益；深读制品仍供故事图和 `editor_review --global` 使用。

剖面生成失败是 fail-open：系统回退到 `config.yaml` 中性风格基线，继续翻译。

### 5.3 初始化与 chunk 任务

`init --chapter chXX` 做三件事：切 chunk、写入 `db/workflow.db`、写入韵文标签/旁路元数据。`--force` 是 smart-force：通常只重置未完成块，除非明确需要重建整个章节。

每个 chunk 的输入、状态、retry 计数和错误信息都在 workflow DB 中；进程中断后再次运行会从 DB 状态继续，而不是重新翻译所有内容。

### 5.4 单章质量闭环

1. **实体/典故提取**：两阶段考据先召回，再确认，写入 Decision DB。
2. **直译**：普通片段走 `literal_translator`；难句/典故密集片段由 `difficulty_routing` 切到 `primary` 并提高 effort。
3. **文学润色**：把源文、直译稿、Decision DB 上下文、风格剖面和结构 P0 反馈交给 `secondary`。
4. **审计**：默认走 System One 校准 primitives；不可用时才进入显式标记的旧 LLM critic 路径。
5. **裁决**：代码合成 verdict，必要时用 `judge_decision` 补充可执行修改意见；REJECT 回流 rewriter。
6. **恢复策略**：CJK 比例闸门、实体硬失败、停滞换模型、振荡/不进步早停、双稿仲裁、pairwise 仲裁和 `no_content` 自动降档共同避免死循环。
7. **章末收口**：实体审计、注册表冲突诊断、一致性报告、成本报告和中间文件清理。

一次章节运行失败并不等于全部工作丢失：已完成 chunk 留在 DB/final 中，PF 块保留 fallback 稿和中间文件；后续可以用回溯、`consolidate`、`requeue` 或 `pipeline --fix-from-review` 继续修复。

### 5.5 全书收敛与回溯治理

跨 chunk 的术语/典故变化会触发 DIRTY。生产默认使用 `backtrack_mode: deferred`：变更先进入队列，`consolidate --max-rounds N` 在章节清空点合并应用，避免“改决策 → 重跑 → 再改决策”的自激循环。

治理预算包括：

- `max_affected_chunks_per_decision`：单次决策最多影响多少块；
- `max_backtrack_sweeps_per_chapter`：每章最多消费几轮回溯；
- `max_backtrack_rounds_per_chunk`：单块累计回溯次数；
- `pf_requeue_max`：PF 块的自动复活次数；
- `version_merit`：重跑稿与备份稿的择优保护。

未收敛的 DIRTY 会阻止合并和交付；只有显式 `--allow-incomplete` 才允许放行，且该选项不属于生产流程。

### 5.6 合并、终稿规范化与 lint

1. `finalize --apply`：统一引号、脚注制式、清除策略元语和悬空定义；先写 `output/finalize_diff.md` 再落盘。
2. `merge`：把各章 final 合成为 `output/<中文书名>.md`，同时生成 `output/QA_REPORT.md`；章节标题制式由 `book.chapter_titles` 控制。
3. `notes-normalize --apply`：章级注释去重、孤儿改锚、顺编号，输出 `output/note_policy_diff.md`。
4. `lint`：同时检查 chunk 级 final 和合并后的整书，输出 `output/book_lint.md`。
5. 可选 `consistency --apply`：命名一致性校正；默认只报告，避免普通词被误替换。

### 5.7 成书终审与定向修复

`scripts/editor_review.py` 是 merge 之后的成书级审读，不属于一键脚本的默认步骤：

```bash
python3 scripts/editor_review.py --json output/editor_review.json --md output/editor_review.md
python3 scripts/editor_review.py --global --review-dirty \
  --json output/editor_review.json --md output/editor_review.md
```

终审只报告问题，不直接改全文。只有锚句能在 final 文本中机械校验并唯一落到一个 chunk 的 P0 结构问题，才会写入 `output/review_dirty.json`；随后用：

```bash
python3 -m src.main pipeline --fix-from-review
```

定向重跑对应的 chunk。`review_dirty` 的 sidecar 绑定原 final 哈希，旧审读结果不会污染后续新稿。

### 5.8 交付门禁与清理

`gate` 是 fail-closed 的，默认检查八项：

1. workflow DB：全部 chunk `COMPLETED` 且 `DIRTY=0`；
2. 产物数量：每章 `final.json` 数量等于 DB 任务数，且无备份残留；
3. `QA_REPORT.md` 不包含 fallback 章节；
4. `book_lint` 全绿；
5. EPUB 转换审计通过；
6. 韵文对齐（`verse_lint`）通过；
7. 双关 watchlist 核销（`pun_audit`）通过；
8. 校准裁决占比达到 `pipeline.final_gate.min_calibrated_ratio`（默认 0.95，按全书和单章检查；`jev` 是现役 MoArk d3 的兼容标签）。

`--clean` 必须在 gate 之后执行。gate ⑧ 需要读取每个 chunk 的 `critic_report.json` 中的 `_backend_path` 来核算校准来源；提前清理会导致审计缺证。一键脚本已经固定了这个顺序。

### 5.9 可选 EPUB3 导出

一键脚本产出的是 Markdown 成书文件，不自动导出 EPUB。gate 通过后可运行：

```bash
python3 scripts/build_epub.py --md output/沙丘.md --out output/book.epub
```

导出器支持中文章节标题、`[^n]` 脚注链接、粗体/斜体和基础书名页元数据。EPUB 是交付形态转换，不是质量门禁的一部分；应先通过 gate 再导出。

---

## 6. 文件结构

```text
SagaTrans/
├── config.yaml                 # 运行配置：模型、任务路由、质量门槛、风格基线
├── .env.example                # 密钥模板；复制为 .env 后填写
├── requirements-core.txt       # 生产最小依赖（含 System One SDK/httpx2）
├── requirements.txt            # 核心依赖 + EPUB 处理依赖
├── README.md
├── db/
│   ├── workflow.db             # chunk 任务状态机、重试/治理计数
│   ├── decision_db.sqlite      # 术语/典故/风格/实体注册（硬约束事实源）
│   ├── model_capabilities.json # effort 探测结果（可选）
│   └── .sagatrans.lock         # 单实例写锁
├── input/
│   ├── chXX.md                 # 切分后的章源；证据匹配的基准
│   ├── _skip/                  # 非正文文档
│   ├── verse_manifest.json     # 韵文块清单
│   ├── _conversion_audit.md    # 转换覆盖率审计
│   ├── book_profile.json       # 当前作品的风格/叙事剖面（运行产物）
│   └── story_graph.json        # 全书知识图谱（运行产物）
├── output/
│   ├── chXX/                   # 每章 raw/literary/critic_report/final 制品
│   ├── <中文书名>.md           # 合并后的成书 Markdown
│   ├── QA_REPORT.md
│   ├── book_lint.md
│   ├── consolidate_diff.md
│   ├── finalize_diff.md
│   ├── note_policy_diff.md
│   ├── review_dirty.json
│   └── book.epub               # 可选 EPUB3 成品
├── scripts/
│   ├── translate_book.sh       # 一键主流程（含锁、preflight、gate 顺序）
│   ├── build_epub.py           # Markdown → EPUB3
│   ├── editor_review.py        # 成书级审读 / review_dirty
│   ├── build_integrated.py     # 散装 src → 单文件镜像
│   ├── sync_book_lint_mirror.py# book_lint 镜像一致性
│   ├── pun_audit.py            # 双关 watchlist 核销
│   ├── literary_metrics.py     # AI 味/翻译腔/重复/诗体统计
│   ├── clean_entity_registry.py# 实体注册表清理与并名（默认 dry-run）
│   ├── probe_role.py           # 单角色连通性探针
│   └── probe_effort_ab.py      # effort 档位 A/B 探针
├── src/
│   ├── translator_agent.py     # 自动生成的生产单体入口（禁止手改）
│   ├── main.py                 # 模块化 CLI
│   ├── config.py               # YAML/.env/环境变量覆盖加载
│   ├── pipeline.py             # chunk 状态机、质量闭环、最终 gate
│   ├── judgment_layer.py       # 判断层 schema、L1/L2、切片注入
│   ├── story_graph.py          # 确定性实体/场景图 + 验证叙事边
│   ├── story_graph_query.py    # 只读图谱检索与评测
│   ├── setup_payoff_audit.py   # 伏笔候选审计与受控提升
│   ├── style_profiler.py       # 抽样/深读风格剖面
│   ├── book_lint.py            # final/merged lint、finalize、notes-normalize
│   ├── consistency.py          # 命名一致性校正
│   ├── verse.py                # 韵文识别与 verse_lint
│   ├── core/                   # scheduler、decision_engine
│   ├── agents/                 # reference、rewriter、critic、judge
│   ├── judges/                 # System One、pairwise、verdict 合成
│   └── utils/                  # LLM adapter、chunker、文本匹配、停用词
├── tests/                      # round 粒度回归与契约测试
├── tools/                      # 辅助探针/日志过滤/演示
├── docs/                       # 研究计划/审计说明（本地默认不入库）与 probes A/B 报告
└── .github/workflows/          # 3.10/3.13 双解释器 CI
```

### 生成文件纪律

- `src/` 是事实源；修改 Python 业务代码后运行 `scripts/build_integrated.py` 重新生成 `src/translator_agent.py`。
- `book_lint` 的部分函数/常量会在 `translator_agent.py` 内形成镜像区；修改 `src/book_lint.py` 后运行 `scripts/sync_book_lint_mirror.py`。
- 不要直接编辑 `src/translator_agent.py` 的生成区，否则下一次构建会覆盖。
- `input/book_profile.json`、`input/story_graph.json`、`db/`、`output/`、`.env` 是本机运行产物，不应提交。

---

## 7. 配置系统

### 7.1 密钥与覆盖顺序

日常只需要维护项目根 `.env`：

```bash
cp .env.example .env
# 编辑 .env，填写：
# QWEN_API_KEY / DEEPSEEK_API_KEY / MINIMAX_API_KEY / MOARK_API_KEY
```

`config.yaml` 使用 `${ENV_NAME}` 占位符。加载顺序是：

1. `config.yaml` 中的持久配置；
2. 项目根 `.env` 展开 `${...}`（shell 同名变量只在 `.env` 缺项时兜底）；
3. `SAGATRANS__<PATH>` 环境变量只作为 CI/临时覆盖，例如：

```bash
SAGATRANS__MODELS__PRIMARY__MODEL_NAME=your-model \
SAGATRANS__MODELS__PRIMARY__API_BASE=https://api.example.com/v1 \
python3 -m src.main pipeline --chapter ch01
```

不要把 key 写进 `config.yaml`、README 或测试 fixture。旧 `TYPESAFE_API_KEY` 不再是现役判断后端的必需项；现役后端使用 `MOARK_API_KEY`。

### 7.2 主要配置段

| 段 | 说明 |
| --- | --- |
| `models` | 三个角色的 `api_base`、`api_key`、`model_name`；唯一模型定义处 |
| `llm_backend` | `openai_api` 为生产，`mock` 仅用于 dry-run/回归 |
| `openai_api` | HTTP 重试、退避和超时 |
| `task_routing` | 任务 → 角色、effort、temperature、max_tokens、extra_body |
| `pipeline` | chunk 调度、质量重试、早停、双稿、回溯治理、gate ⑧ 阈值 |
| `judgment_injection` | 全书注记是否注入生成 prompt；当前默认关闭 |
| `story_graph` | 图谱路径、共现窗口、覆盖率与场景长度 |
| `decision_engine` | 决策回溯触发条件、scope、队列与预算 |
| `critic_thresholds` | 通用 critic 参考门槛，可被作品 profile 覆盖 |
| `typesafe_thresholds` | 校准 verdict 的概率/置信/Noul 不确定区间 |
| `style_guide` | 无 profile 时的中性风格基线 |
| `typesafe` | MoArk d3-flash 判断后端的连接、批大小和 timeout |
| `book` / `verse` | 成书标题/章节制式与韵文对齐参数 |

`paths`、`logging`、模型 `alias`、历史 `golden_test` 路由块以及若干没有运行读取点的旧开关已从配置删除；路径由项目根派生，运行日志沿用 Python 默认配置。这样配置文件只表达当前真正生效的契约。

---

## 8. 脚本使用方法

### 8.1 环境安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-core.txt
python3 -m pip install -r requirements.txt       # 含 EPUB 转换依赖
python3 -m pip install pytest                     # 开发/回归需要
```

一键脚本会按以下顺序选择解释器：`SAGATRANS_PYTHON` → 项目 `.venv/bin/python` → `PATH` 上的 `python3`。如果依赖装在 conda 或其他环境，请显式指定，避免“装在一个 Python、运行时用另一个 Python”：

```bash
SAGATRANS_PYTHON=/path/to/python ./scripts/translate_book.sh book.epub
```

preflight 会检查 Python ≥ 3.10、`typesafe_sdk`/`httpx2` 和可解析的 `MOARK_API_KEY`；在非 dry-run、非 `--merge-only` 模式下缺失会直接中止，而不是静默进入未校准路径。

### 8.2 一键主流程

```bash
# 最常用：完整翻译 + 收敛 + 合并 + notes-normalize + lint + gate
./scripts/translate_book.sh book.epub

# 复用已经切分好的 input/chXX.md
./scripts/translate_book.sh book.epub --skip-epub

# 清空 DB/output 后重来（破坏性操作）
./scripts/translate_book.sh book.epub --reset

# 只跑流程，不调用真实 LLM；split 仍会执行
./scripts/translate_book.sh book.epub --dry-run

# 已有 final.json，只做收口：consolidate → 图谱 → finalize → merge → lint → gate
./scripts/translate_book.sh --merge-only

# 清理中间产物（必须在 gate 之后；脚本内部 --clean 已按此顺序执行）
./scripts/translate_book.sh book.epub --clean

# 只清理旧中间产物，不翻译
./scripts/translate_book.sh --clean-only
```

常用选项：

| 选项 | 作用 |
| --- | --- |
| `--force` | 强制重新初始化章节（smart-force 主要重置未完成块） |
| `--dry-run` | 使用 mock 后端；不消耗模型额度，gate 跳过 |
| `--skip-epub` | 不转换输入，直接使用现有 `input/ch*.md` |
| `--reset` | 删除历史 DB/output，重新开始 |
| `--clean` | gate 后清理 raw/literary/critic 等中间文件；保留 final 与 DB |
| `--clean-only` | 独立清理中间文件，不运行翻译 |
| `--merge-only` | 不翻译，直接收口已有 finals |
| `--no-normalize` | 跳过 finalize 样式规范化 |
| `--skip-deep-profile` | 跳过全书深读，沿用现有/抽样剖面 |
| `--consistency` | 合并后额外运行命名一致性报告 |
| `--golden-gate` | 运行 Golden Gate 质量门禁（失败仅 WARN） |
| `--allow-incomplete` | 调试用，放行 DIRTY/lint/gate 硬闸；不要用于生产 |
| `--auto-merge-on-failure` + `--yes` | API 耗尽时自动合并已完成章节，适合人工值守的恢复场景 |

一键脚本会在 `db/.sagatrans.lock` 上持有单实例锁，Python 写库命令和 shell 脚本共用同一把锁。Ctrl+C 第一次只警告，第二次才中止；已经完成的 chunk 和 DB 状态会保留，可断点续跑。

### 8.3 手工 CLI

常用命令均从项目根运行。`src.main` 和 `src.translator_agent` 都能作为入口，生产单体通常用后者。

```bash
# 输入切分与初始化
python3 -m src.main split --input book.epub --input-dir input
python3 -m src.main init --chapter ch01

# 单章闭环
python3 -m src.main pipeline --chapter ch01

# 全书判断层与图谱
python3 -m src.main style-profile --deep
python3 -m src.main story-graph --require-answerable
python3 -m src.main story-graph-query --query "杰西卡如何成为圣母？" --top-k 10

# 决策收敛与成书收口
python3 -m src.main consolidate --max-rounds 3
python3 -m src.main finalize --apply
./scripts/translate_book.sh --merge-only      # 合并 + notes-normalize + lint + gate

# 诊断与交付
python3 -m src.main lint
python3 -m src.main gate
python3 scripts/editor_review.py --global --review-dirty
python3 scripts/pun_audit.py
python3 scripts/literary_metrics.py --print
python3 scripts/build_epub.py --out output/book.epub
```

### 8.4 命令用途速查

| 命令 | 用途 |
| --- | --- |
| `pipeline --chapter chXX` | 跑一章的翻译/润色/审计/裁决/回溯闭环 |
| `pipeline --fix-from-review` | 读取 `review_dirty.json`，二次校验锚句后定向重跑 P0 chunk |
| `consolidate --max-rounds N` | 消费 pending backtrack，收敛 DIRTY；未收敛返回非零 |
| `finalize --apply` | 统一成书引号/脚注/策略元语 |
| `notes-normalize --apply` | 注释去重、孤儿改锚、顺编号 |
| `lint` / `gate` | 成书格式检查；八项交付不变量检查 |
| `style-profile [--deep]` | 抽样或全书深读生成/升级 `book_profile.json` |
| `story-graph` | 确定性重建实体/场景/叙事图；`--require-answerable` 不可答时退出 2 |
| `story-graph-query` | 只读场景检索、注记匹配、DPP 重排、评测与 trace |
| `setup-payoff-audit` | 只读候选审计；加 `--llm --l2 --apply` 才受控提升已验证配对 |
| `editor_review.py --global` | 全书弧线审读；`--review-dirty` 生成可机械修复的 P0 清单 |
| `editor_review.py --global --retrieval` | 可选注入有界跨章证据；默认关闭，A/B 未显示稳定收益 |
| `requeue --chapter chXX [--force]` | 将 PF 块放回 DIRTY；消耗 `pf_requeue_max` 预算 |
| `drift-audit` | 章级语域/句长漂移诊断；`--jev` 是历史参数名，现走 System One 窗口检查 |
| `golden` / `golden-gate [--ab]` | Golden Set 跑分、模型×effort A/B、人工基线动态门禁 |
| `golden --judgment-ab --chapter chXX` | 只比较判断层注记 off/on，不乘入模型矩阵 |

### 8.5 测试、镜像与回归

```bash
# 快速回归
python3 -m pytest -q tests/

# 重点契约
python3 -m pytest -q tests/test_round119_mirror_parity.py
python3 -m pytest -q tests/test_round131_*.py tests/test_round145_*.py

# 修改散装代码后
python3 scripts/build_integrated.py
python3 scripts/build_integrated.py --check
python3 scripts/sync_book_lint_mirror.py --check

# 配置语法检查
python3 -c "import yaml; yaml.safe_load(open('config.yaml', encoding='utf-8'))"
```

CI 使用 Python 3.10 和 3.13 两个解释器跑同一套测试；`TestPython310SyntaxCompat` 只检查语法，因此新增代码还应避免依赖仅某个解释器环境里偶然存在的包。真实付费模型 A/B 必须保留报告并区分 mock、legacy、校准 System One 的结果。

---

## 9. 判断层与全书知识图谱

这是项目相对传统 chunk 流水线最有研究价值的部分，也是当前默认最保守的部分。

### 9.1 判断层注记

- 深层阅读产出统一 envelope：`id, type, chapters, claim, evidence, confidence, status, source_model, created_at`；
- evidence 必须能落到 `input/chXX.md`，而不是落到模型看到的拼接文本；
- L1 先 exact，再归一化 exact，最后滑窗相似度；不命中即丢弃；
- L2 用低成本模型小批复核；术语/译名断言直接 rejected；
- 通过验证的 active 注记可用于故事图与全书终审；是否进入生成 prompt 由 `judgment_injection` 单独控制，当前关闭。

### 9.2 知识图谱的确定性底座

`story-graph` 不把 LLM 当作事实源：

- 实体、canonical、变体、章节出现和共现边来自 Decision DB 与章源；
- 叙事边只来自已验证、active 的判断层注记；
- 场景节点按段落/标题/分隔线装箱，实体到场景的 `mentions_in` 与章级 `appears_in` 分离；
- 每条边带 provenance，无法解析的别名/缺口写入显式 `knowledge_gaps`，不做静默补全；
- 图谱可无 LLM 重建，并记录指纹以便检索结果可复现。

### 9.3 检索与伏笔回收

`story-graph-query` 是只读研究工具：可从别名、已验证注记或问题相似度出发，最多两跳取场景，可选 LLM 必要证据复核。它不写图、不改认证、不注入翻译回路。

`setup-payoff-audit` 只让模型从已给出的候选 ID 中选择回收场景，evidence 由代码从 setup/payoff 两个章节组装，随后过 L1/L2；`--apply` 之前还会比较图谱基线的 certified coverage 和 unresolved aliases。模型不能凭空返回 quote 或新 ID。

这两条链路的价值是提供可审计的全书叙事检索和知识缺口报告，而不是让 LLM 变成“全书事实数据库”。术语与译名仍由 Decision DB 决定。

---

## 10. 质量语义与恢复边界

### 10.1 fallback 与 PERMANENTLY_FAILED

质量重试耗尽时，pipeline 会把当时的最佳版本写入 final，并标记 `fallback=True`。这保证“已有产出不丢”，但该 chunk 会进入 `PERMANENTLY_FAILED`，gate ③ 通常不会让整书通过。之后可以通过：

- 后续决策回溯或 `consolidate` 触发受控重跑；
- `requeue --chapter chXX` / `requeue --force` 把 PF 恢复为 DIRTY；
- 终审 `review_dirty` 只对结构 P0 做定向修复；
- `version_merit` 在重跑失败时恢复旧备份。

因此，一次章节运行中的部分 fallback 不是“最终判决”；但也不能把 fallback 当完成品交付。

### 10.2 零人工审核的前提

本项目追求尽量少的人工干预，但这要求 gate 真的能发现降级：

- 不能以“进程退出 0”替代“交付成功”；
- 不能让 mock/openai_compat/legacy 冒充校准 System One；
- 不能让 `--allow-incomplete` 成为常态；
- 不能让 `--clean` 在 gate 前删除审计证据。

### 10.3 什么时候需要人工介入

以下情况建议人工检查而不是继续自动重跑：转换审计覆盖率异常、实体 registry 硬冲突持续不收敛、gate ⑧ 校准占比低于阈值、连续多章 fallback、终审锚句无法唯一映射、故事图 alias 缺口持续扩大。相关报告都在 `output/` 与 `input/`，且带时间/路径信息。

---

## 11. 给人类研究者与自动化协作者

### 人类研究者

- 若维护者保留了本地研究笔记，可查阅 `docs/PLAN_round131_global_judgment_layer.md`、`docs/ROUND142_SAG2_RESEARCH.md` 和 `docs/ROUND131_AUTO_ONTOLOGY_NOTES.md`；这些 `docs/*.md` 默认不入库；
- 从已跟踪的 `docs/probes/` 看真实付费模型的 A/B 报告；
- 把“生成质量”“判断可信度”“一致性治理”“交付不变量”分开评价，避免用单一 BLEU/风格分掩盖系统性问题。

### 自动化协作者

- 修改业务代码只改 `src/` 散装模块；不要手改 `src/translator_agent.py` 的生成区；
- 改 `src/book_lint.py` 后必须同步镜像并跑 parity；
- 任何写 `db/`、`output/`、`input/book_profile.json` 的生产命令都要先确认单实例锁；
- 不要为通过测试而降低断言；配置清理、schema 兼容和 gate 行为都应有回归测试；
- 没有用户明确授权，不要提交、push、清空生产 DB 或启动全量付费翻译。

---

## 12. 边界与声明

- 本项目只应处理**公有领域或已获授权**的作品；译文版权与使用责任由部署者承担。
- 模型端点、代理和配额策略因环境而异；网络受限时可在 shell 层配置 `ALL_PROXY`。
- `.env`、`input/ch*.md`、`input/book_profile.json`、`input/story_graph.json`、`db/`、`output/` 和本地研究文档默认不入库。
- README 描述的是当前代码契约；历史 Round 计划、审计记录和 A/B 报告用于解释演进，不应覆盖当前配置与 gate 语义。

---

**一句话总结：模型提出译文，代码保存事实；判断层提供上下文，交付 gate 决定结果。**
