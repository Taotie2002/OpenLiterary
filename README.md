# SagaTrans

> **OpenLiterary · AI 文学语义编译系统**

[GitHub Repository](https://github.com/Taotie2002/OpenLiterary) · [CI](https://github.com/Taotie2002/OpenLiterary/actions/workflows/tests.yml)

> 代码仓库名为 **OpenLiterary**，系统/产品名为 **SagaTrans**；下文统一使用后者。

SagaTrans 把长篇小说翻译实现为一条**有状态、可审计、可回滚的语义编译流水线**。它不把整本书塞进一次超长对话，而是把翻译拆成章节、chunk、决策、判断注记和交付制品，让每一个中间结果都能被检查、重放和定向修复。

```text
切分 → 风格剖面 → 直译 → 文学润色 → 多维审计 → 校准裁决
     → 决策回溯 → 全书收敛 → 终检修复 → 合并
     → 注释/机械收口 → QA 重建 → lint → 十三项交付门禁 → 可选 EPUB3
```

本项目面向的是“**零人工值守下的高质量候选稿**”，不是“无人值守的文学定稿”。代码负责守住事实、证据、状态和交付不变量；文学判断仍需要读者。

---

## 0. 快速开始

```bash
git clone https://github.com/Taotie2002/OpenLiterary.git
cd OpenLiterary

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-core.txt
python3 -m pip install -r requirements.txt
python3 -m pip install pytest

cp .env.example .env
# 编辑 .env：填写 QWEN_API_KEY / DEEPSEEK_API_KEY / MINIMAX_API_KEY / MOARK_API_KEY

./scripts/translate_book.sh book.epub
```

已有 `input/chXX.md` 时：

```bash
./scripts/translate_book.sh book.epub --skip-epub
```

已有 `final.json`、只想重做全书收口时：

```bash
./scripts/translate_book.sh --merge-only
```

如果依赖安装在 conda 或其他解释器中，必须显式指定同一个 Python，不要让“安装解释器”和“运行解释器”分裂：

```bash
SAGATRANS_PYTHON=/path/to/python ./scripts/translate_book.sh book.epub
```

---

## 1. 这项工程在研究什么

长篇小说翻译不同于短文本翻译。它同时面对四个断点：

1. **上下文断点**：全书远超单次上下文窗口，后面的 chunk 看不到前面的术语、人物关系和语气决定。
2. **一致性断点**：译名、称谓、典故、脚注、叙述声音和章节语气必须在全书范围内保持稳定。
3. **判断断点**：生成模型既写译文又给自己打分，容易产生同源盲区；未校准的分数不能冒充概率裁决。
4. **交付断点**：进程退出 0、队列看起来跑完，都不等于文本真的完整、可重建、可交付。

SagaTrans 的基本回答是：

> **把翻译当作编译，把一致性当作事实数据库问题，把质量当作证据裁决问题，把交付当作状态机不变量问题。**

模型负责提出语言候选；代码负责维护约束、验证证据、调度重跑，并决定什么时候允许交付。

### 核心命题

| 命题 | 工程实现 |
| --- | --- |
| 长篇翻译需要跨上下文记忆 | SQLite Decision DB 保存术语、译名、典故和风格决策；变更时定向回溯 |
| 生成与判断应尽量解耦 | 直译/润色模型与 System One 校准判断后端分离；未校准路径显式标记 |
| 判断应当是制品，而不是隐式 prompt 状态 | 全书注记写入 `book_profile.json`，经 L1/L2 验证后供图谱与检索使用 |
| 全局理解不应阻塞局部执行 | 判断层、图谱、查询 fail-open；无效注记丢弃或降级，不阻塞正文 |
| 不确定性必须可恢复 | 每个 chunk 有版本、备份、merit 比较、DIRTY 队列、PF 复活预算和救援账本 |
| 交付不能依赖“看起来完成” | 十三项确定性 gate 检查状态、产物、校准、fallback、机械排错、链接证据与回溯队列 |

---

## 2. 设计立场

### 2.1 文学编译，而不是“大 prompt 翻译”

一个 chunk 会经历直译、润色、审计、裁决和可能的回流。每一步都有独立产物：

```text
source → raw.json → literary.json → critic_report.json → final.json
```

主流程只消费通过状态机检查的制品，不依赖某个模型的“记忆”。回溯、救援、终检修复和合并都围绕这些制品工作，而不是围绕一段无法追踪的对话历史工作。

### 2.2 硬约束与软判断分层

- **硬约束**：术语、译名、实体 canonical 形式、脚注/韵文/格式规则、chunk 状态和交付证据。它们属于 Decision DB、book lint 和 gate。
- **软判断**：叙述声音、人物语气、章节弧线、伏笔是否回收、平移风格是否偏移。它们可以作为注记进入故事图和审读上下文，但不能覆盖 Decision DB。

这条边界是系统的关键。如果让“全书注记”直接修改译名，就会得到两个互相矛盾的事实源。SagaTrans 明确规定：注记只能是 advisory；L2 复核也会拒绝包含术语/译名规范性断言的高层注记。

### 2.3 生成模型与判断模型分工

生成端追求文风、句法和创造性；判断端追求可重复、可校准的概率信号。当前 `config.yaml` 的默认分工如下（模型名可替换，角色契约不变）：

| 角色 | 默认后端 | 主要职责 |
| --- | --- | --- |
| `literal_translator` | Qwen | 直译、典故/实体召回、轻量分类、判断层 L2 复核 |
| `secondary` | DeepSeek | 文学润色、考据确认、旧 critic 路径、深读抽取/合并 |
| `primary` | MiniMax | 最终 LLM 裁决、风格剖面、成书审读、困难片段直译、救稿/双稿 |
| System One | MoArk `d3-flash` | 校准概率裁决；`d3-turbo` 仅作历史对照，当前没有自动 A/B |

报告和 gate 中的 `jev` 是现役 MoArk d3 路径保留的**兼容标签**，不是另一个独立模型。`confidence_source=jev` / `_backend_path=jev` 表示这次裁决来自校准 System One；`legacy`、`openai_compat`、`mock` 都不得冒充校准结果。

“思考档位”是任务级参数，例如 `judge_decision=max`、`literary_rewrite=xhigh`、`critic_scoring=none`。具体模型、地址、effort 和 `max_tokens` 都在 `config.yaml` 的 `models` 与 `task_routing` 中调整。

### 2.4 失败策略是不对称的

- **判断层 fail-open**：深读注记验证失败、故事图构建失败或查询失败，不阻塞正文翻译；无效注记被丢弃或降为 dormant。
- **交付层 fail-closed**：状态未完成、产物缺失、lint/韵文/双关/校准比例不达标、manifest 或运行证据不成立，任何一项为红都不交付。

模型可以失败；流水线不能因此丢失已经完成的工作，也不能把未校准结果伪装成完成品。

---

## 3. 系统模型

### 3.1 五类持久事实

理解本项目时，建议按“事实层”而不是按“模型调用”来阅读：

| 层 | 位置 | 作用 | 是否可手改 |
| --- | --- | --- | --- |
| 输入层 | `input/chXX.md`、`input/verse_manifest.json`、`input/_conversion_audit.md` | 切分后的原文与转换审计；所有证据匹配以章源文件为准 | 可检查，不建议手改 MD |
| 决策层 | `db/decision_db.sqlite`、`db/workflow.db` | 全局术语/译名/典故/风格决策，以及 chunk 状态机 | 通过 CLI/脚本修改，禁止绕过状态机 |
| 生成层 | `output/chXX/*_raw/literary/critic_report/final.json` | 每个 chunk 的候选稿、审计、裁决和最终稿 | 可检查；不要绕过状态机直接替换 final |
| 判断/派生层 | `input/book_profile.json`、`input/story_graph.json`、`output/review_dirty.json` | 全书注记、知识图谱、成书审读问题 | advisory；可重建，不反向成为术语事实源 |
| 交付证据层 | `output/build_manifest.json`、`output/logs/runs/<run_id>.*` | source → chunk → final → merged 血缘，以及运行事件与封存哈希 | 只读审计；不能手改后继续声称可交付 |

`src/` 是代码事实源；`src/translator_agent.py` 是自动生成的单文件镜像。路径默认以项目根为基准；`story_graph.path` 和 `pipeline.model_probe.path` 可在 `config.yaml` 覆盖，其余默认值见各模块常量。

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
                          └─→ PERMANENTLY_FAILED    (救援预算耗尽，保留 fallback 稿)

COMPLETED ─→ DIRTY (决策变更/结构 P0 修复触发定向重跑)
DIRTY     ─→ EXTRACTING_TERMS
FAILED    ─→ 任一可恢复阶段
PF        ─→ DIRTY (自动救援：定向续跑 → 整块重跑 → 再隔离)
```

- `COMPLETED` 表示 chunk 已通过该轮质量闭环，不等于全书可交付；还需要全书收敛和 gate。
- `DIRTY` 表示旧稿受新决策或终审结构问题影响，必须重跑后才能再次交付。
- `PERMANENTLY_FAILED`（PF）是“救援预算耗尽后的隔离态”，不是永久黑洞：下一章运行会自动尝试定向续跑；只有 `pipeline.rescue.max_targeted_attempts` 与 `max_full_requeues` 都耗尽后才保持隔离。也可用 `requeue --chapter` / `--chunks` / `--from-ledger` 显式复活。
- 回溯重跑的新稿若 critic 均分不如备份稿，`version_merit` 会保留旧版，防止“越修越差”。

### 3.3 单实例与写事务

所有可能写 DB、profile、graph 或 final 的命令都必须拿同一把单实例锁：

- 主锁：`db/.sagatrans.lock`；
- 运行期副锁：位于仓库外，按主锁绝对路径哈希生成，防止主锁文件被删除/替换后产生两个 writer。

手工执行写命令前，先确认两把锁都空闲。以下检查会同时持有主锁与默认 `/tmp` 运行期副锁（副锁名为 `sagatrans-<uid>-<主锁绝对路径 SHA-256 前 16 位>.lock`）：

```bash
PROJECT_ROOT="$(pwd -P)"
PRIMARY="$PROJECT_ROOT/db/.sagatrans.lock"
GUARD="/tmp/sagatrans-$(id -u)-$(printf '%s' "$PRIMARY" | sha256sum | cut -c1-16).lock"
(
  exec 9>"$PRIMARY"
  exec 8>"$GUARD"
  flock -n 9 || { echo "primary lock busy"; exit 1; }
  flock -n 8 || { echo "runtime guard busy"; exit 1; }
  echo "locks free"
)
```

只运行 `flock -n db/.sagatrans.lock true` 只能检查主锁，不能证明运行期副锁空闲；实际写命令会同时获取两把锁。若设置了 `SAGATRANS_LOCK_PATH`、`SAGATRANS_GUARD_LOCK_PATH` 或 `SAGATRANS_LOCK_RUNTIME_DIR`，请按覆盖后的路径检查。

`SAGATRANS_ALLOW_CONCURRENT=1` 只用于明确知道风险的故障排查，不作为生产开关。

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
  │ calibrated audit → judge → rewrite if rejected           │
  │ decision change → DIRTY → chapter-local retry            │
  └──────────────────────────────────────────────────────────┘
      │                              decision_db.sqlite
      ▼
per-chapter rescue（PF/fallback 定向续跑） ─► output/rescue/failure_ledger.jsonl
      │
      ▼
consolidate 全书 DIRTY 收敛 ────────► output/consolidation_diff.md
      │
      ▼
story-graph 重建 ───────────────────► input/story_graph.json
      │
      ▼
finalize → book-repair → recalibrate-critic
      │
      ▼
merge ──────────────────────────────► output/<中文书名>.md
      │                              output/build_manifest.json
      ▼
notes-normalize → proofread → qa-report → lint
      │
      ▼
gate（①–⑬ fail-closed） ──► 可选 editor_review / EPUB3
```

`editor_review`、`story-graph-query` 和 `setup-payoff-audit` 是研究/审读旁路，不在一键主流程内；它们不能替代 gate，也不能绕过 Decision DB 修改硬约束。

---

## 5. 工作流详解

以下顺序与 `scripts/translate_book.sh` 的实际执行逻辑一致。手工调试时也应遵守“先收敛，再做终检修复与合并，再完成注释/机械收口和 QA 重建，最后 lint 与 gate”的顺序。

### 5.1 输入转换与切分

1. `split` 接受 `.epub`、`.txt` 或 `.md`，输出 `input/chXX.md`。
2. `input_prep` 把封面、版权页、目录、索引等非正文文档移入 `input/_skip/`，并执行配置中的字间断裂修复。
3. EPUB 转换同时产出 `input/verse_manifest.json` 和 `input/_conversion_audit.md`；后者是 gate ⑤ 的输入。
4. `SmartChunker` 先按标题、`***` / `---`、空行等场景断点切分；`soft_limit` 后优先在断点切，超过 `hard_limit` 才强制切。

### 5.2 风格剖面与全书判断层

翻译前先确保 `input/book_profile.json` 存在：

- `style-profile`：抽样若干章生成 v1 剖面；
- `style-profile --deep`：分段读取全书，跨段合并注记，经 L1 原文匹配和 L2 低成本复核后生成 v2 剖面；
- 注记类型是 `voice | arc | setup_payoff`，统一 envelope 包含章节范围、claim、evidence、confidence、status；
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

一次章节运行失败并不等于全部工作丢失：已完成 chunk 留在 DB/final 中，失败块保留 raw/literary/fallback 稿和中间文件；后续可以用自动救援、回溯、`consolidate`、`requeue` 或 `pipeline --fix-from-review` 继续修复。

#### 5.4.1 失败块自动救援

失败块不再直接成为不可处理的终点，而是进入受预算约束的救援阶梯：

1. `classify_failure` 把错误归为 `api_or_no_content`、`content_sensitive`、`quality`、`cjk`、`entity`、`structure` 或 `unknown`；
2. 优先按已有 raw/literary 续跑（`recommended_resume_stage`），避免每次失败都从术语提取重做；
3. 定向救援最多 `pipeline.rescue.max_targeted_attempts` 次；用完后整块重跑最多 `max_full_requeues` 次；
4. 本章救援增量耗时受 `chapter_budget_seconds` 限制；预算耗尽才写 `PERMANENTLY_FAILED`；
5. 每次排队、完成、隔离都追加到 `output/rescue/failure_ledger.jsonl`，并生成 `output/rescue/rescue_report.md` 的根因 cohort。

救援只改状态机和候选稿，不会把失败稿升级为“通过”。如果最终仍有 `fallback=True` / `quality_waived=True` 的 final，gate ⑨ 会硬失败；不能靠 DB 的 `COMPLETED` 或进程退出码洗白。

HTTP 422 且上游返回 `input new_sensitive` 时属于永久客户端拒绝，不进入普通 API 重试：pipeline 在第一次拒绝后立即持久化换角色 override，按 `retry_adaptation.sensitive_fallback_roles` 先跨 provider、再尝试已成功产出 raw 的 Qwen；只有所有备用角色都拒绝才隔离。该 override 按 chunk 落库，进程重启后仍然生效。

### 5.5 全书收敛与回溯治理

跨 chunk 的术语/典故变化会触发 DIRTY。生产默认使用 `backtrack_mode: deferred`：变更先进入队列，`consolidate --max-rounds N` 在章节清空点合并应用，避免“改决策 → 重跑 → 再改决策”的自激循环。

治理预算包括：

- `max_affected_chunks_per_decision`：单次决策最多影响多少块；
- `max_backtrack_sweeps_per_chapter`：每章最多消费几轮回溯；
- `max_backtrack_rounds_per_chunk`：单块累计回溯次数；
- `pf_requeue_max`：PF 块的自动复活次数；
- `version_merit`：重跑稿与备份稿的择优保护。

未收敛的 DIRTY 会阻止合并和交付；只有显式 `--allow-incomplete` 才允许放行，且该选项不属于生产流程。

### 5.6 专名/术语治理闭环

Round 158 的一个关键修正，是把“只改专名的修复”与“需要重译的 DIRTY”分开处理。

- `name-authority-audit`：只读检查 Decision DB、entity registry 和锁定目标是否一致；产生 quarantine / align / manual-review 分类。
- `name-authority-align --apply`：只执行权威审计允许的动作；不猜译名。
- `pending-reconcile`：把遗留 pending backtrack 分为 satisfied / covered / outstanding / orphan；默认只读，`--apply --safe-only` 才消费已满足行。
- `name-only-fastpath --chapter chXX --apply`：只对“纯专名/固定译名问题且其他评分过关”的 DIRTY 块复用确定性替换，重审证据后把状态从 DIRTY 闭回 COMPLETED；真正的语义低分块仍走正常重译。
- `completed-name-repair --all --apply`：只处理 COMPLETED 块中“单一活跃权威专名 + 恰好一字 CJK 漂移”的可证明问题，修复后仍保持 COMPLETED，但必须经过 calibrated critic 重审并消费 pending；不带 `--apply` 时只做只读分诊。
- `confirmed-term-drift-repair`：只处理 `locked-target-authority-audit` 已确认的具体术语漂移，要求错误形在每个 source-positive 块中与源文出现次数一致。

这些命令共同解决一个常见假象：**文本已经被确定性替换成功，但 workflow DB 仍把 chunk 当 DIRTY，后续又整块重跑，等于替换白做。** 正确的闭环必须同时更新 final、critic 证据、pending 状态和 Decision 锁；任何一步缺失都不能宣称完成。

### 5.7 故事图与叙事检索

`story-graph` 不把 LLM 当作事实源：

- 实体、canonical、变体、章节出现和共现边来自 Decision DB 与章源；
- 叙事边只来自已验证、active 的判断层注记；
- 场景节点按段落/标题/分隔线装箱，实体到场景的 `mentions_in` 与章级 `appears_in` 分离；
- 每条边带 provenance，无法解析的别名/缺口写入显式 `knowledge_gaps`，不做静默补全；
- 图谱可无 LLM 重建，并记录指纹以便检索结果可复现。

`story-graph-query` 是只读研究工具：可从别名、已验证注记或问题相似度出发，最多两跳取场景，可选 LLM 必要证据复核。它不写图、不改认证、不注入翻译回路。`setup-payoff-audit` 只让模型从已给出的候选 ID 中选择回收场景，evidence 由代码从 setup/payoff 两个章节组装，随后过 L1/L2；模型不能凭空返回 quote 或新 ID。

“先检索证据、再由模型判断”的思路与 SAG2 类检索系统有相通之处，但本项目的图谱事实仍由确定性代码维护，模型只提供候选判断，不成为 ontology 或译名事实源。

### 5.8 成书规范化与终检修复

1. `finalize --apply`：统一引号、脚注制式、清除策略元语和悬空定义；先写 `output/finalize_diff.md` 再落盘。
2. `book-repair --apply`：全书终检修复，只处理 source/raw/literary 能逐项证明的姓名错位、机械可证的同位语污染、已证实错误的注释定义、正文已引用但缺少定义冒号的 Markdown 脚注、夹在可靠前后锚点之间且终稿完全没有对应的正文段漏译，以及 profile/实体注册表已确认的规范表面名漂移。不调用 LLM，不改写无法证明的句子。
3. `book-repair --verify`：只复核当前 final 与修复 manifest，不写正文。
4. `recalibrate-critic --all --apply`：对 deterministic repair 之后已经 stale 的 critic 报告做真实重审；只跳过 `exact` / `repaired` 证据，不能批量伪造 provenance。

`book-repair` 属于正式工作流，而不是对合并稿做事后补丁：它修的是带 DB source 证据的 final 文件，每次写入都留备份、前后哈希和 manifest，随后再由 finals 重新合并。这样“事后修全书”仍是可重建、可复验的主线产物；若把它留在手工脚本或直接改合并稿，就会产生一个无法从 finals 重建的第二事实源。

### 5.9 合并、manifest 与运行日志封存

merge 是唯一把 per-chunk finals 组装成成书的阶段。它同时完成两件事：

1. 按章节顺序合并 final，生成 `output/<中文书名>.md`；
2. 写 `output/build_manifest.json`，记录 source → chunk → final → merged 的全部哈希、章节顺序、DB 任务数、profile/graph 新鲜度和本次 run 的日志证据。

随后 `notes-normalize` 与 `proofread` 只改合并成书，不改 final；每次变化都用 `book-link-audit --update-merged` 更新 manifest 中的 merged 哈希和阶段历史。这样 gate ⑫ 可以证明成书确实由当前 finals 产生，而不是由某个旧合并稿冒充。

运行日志位于 `output/logs/runs/<run_id>.jsonl` 与 `.log`：

- 每行都有毫秒级 `+08:00` ISO-8601 时间戳；
- `seq` 在同一 run 内严格递增，不跳号；
- JSONL 记录 `run_start`、`command_started`、artifact、stage、`gate_passed/gate_failed` 和 `run_end`；
- `.sha256` 同时记录 `jsonl_bytes`、`jsonl_sha256`、`log_sha256` 和 `seq_last`；
- 封存之后允许后续命令**纯追加**审计事件；审计只接受“已封存字节前缀完全不变”的追加。截断、改写前缀或换一份日志都会 fail-closed。这就是 `run.seal` 的契约。

### 5.10 注释统一、机械排错与 QA 重建

1. `notes-normalize --apply`：章级注释去重、孤儿改锚、顺编号、合并同章重复定义；不改正文语义。
2. `proofread --apply`：编码/不可见字符、chunk 拼接残留标题、HTML、重复标点、括号与章节壳等机械排错。`--apply` 只删除可证明非正文的残留标题和不可见控制字符。
3. `qa-report`：扫描当前 final 集合，重建带 `final_count` 与每块 SHA-256 的机器可核验 `output/QA_REPORT.md`。
4. `lint`：同时检查 chunk 级 final 和合并后的整书，输出 `output/book_lint.md`。
5. 可选 `consistency --apply`：命名一致性校正；默认只报告，避免普通词被误替换。

### 5.11 交付门禁：十三项 fail-closed 不变量

`gate` 默认检查：

1. **状态机**：workflow DB 全部 chunk `COMPLETED`，无 DIRTY/PF 残留；
2. **产物数量**：每章 `final.json` 数量等于 DB 任务数，且没有不应存在的 backup；
3. **QA_REPORT**：不记录 fallback，且其 `final_count`、chunk 集合和每块哈希与当前 final 完全一致；
4. **book lint**：无 failure；非阻断 warning 保留在 `output/book_lint.md`，不能把“通过”表述成“零缺陷”；
5. **转换审计**：EPUB/TXT/MD 转换覆盖率满足要求；
6. **韵文对齐**：`verse_lint` 无失败；
7. **双关 watchlist**：核销覆盖完整，无未解释 uncovered；
8. **校准裁决**：达到 `pipeline.final_gate.min_calibrated_ratio`，并按全书和单章检查；同时校验 critic 报告与当前 final 的 `exact/repaired/stale/unbound` 关系；
9. **fallback final**：任意 final 的 `metadata.fallback`、`quality_waived` 或 `quality_degraded` 为真，直接判红；final 无法解析也判红；
10. **机械排错**：proofread 的阻断项为 0；gate 独立复核写入 `output/proofread_gate_report.md`；
11. **全书修复**：book-repair 无待修/未决高危项，当前 final 与修复 manifest 一致；
12. **全书链接**：`output/build_manifest.json` 存在且 source/chunk/final/merged 哈希与顺序一致；manifest 的 run 证据存在，JSONL 事件序列有效；若已封存，封存前缀必须有效。`gate.book_link_required: true` 时，manifest 完全缺失也判红；严格交付审计再加 `--require-sealed`；
13. **回溯队列**：`pending_backtrack` 不得存在未解释的 outstanding/blocked/orphan；satisfied/covered 必须由受控 apply 路径消费。

`--clean` 必须在 gate 之后执行，并且只在 gate 全绿时执行。gate ⑧ 需要读取每个 chunk 的 `critic_report.json` 中的 `_backend_path` 和 `_audit_target` 来核算校准来源及其对应的 final；提前清理会导致审计缺证。一键脚本已固定顺序并加了 `gate_ok` 守卫；gate 失败时会保留全部中间产物。

严格交付审计可以单独运行：

```bash
python3 -m src.main book-link-audit \
  --require-sealed \
  --merged-path "output/<书名>.md" \
  --json-out output/book_link_audit.json
```

`gate` 全绿表示**没有硬性不变量失败**，不表示 warning 数为 0，也不表示文学质量已经定稿。

### 5.12 成书终审与可选 EPUB3

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

一键脚本产出的是 Markdown 成书文件，不自动导出 EPUB。gate 通过后可运行：

```bash
python3 scripts/build_epub.py --md "output/<书名>.md" --out output/book.epub
```

EPUB 是交付形态转换，不是质量门禁的一部分；应先通过 gate 再导出。

---

## 6. 交付证据链

### 6.1 运行日志

每次运行都有唯一的 `SAGATRANS_RUN_ID`，并写入：

```text
output/logs/runs/<run_id>.jsonl
output/logs/runs/<run_id>.log
output/logs/runs/<run_id>.sha256
output/logs/runs/<run_id>.seq
```

`.jsonl` 是机器读的事实流；`.log` 是带时间戳的人类可读流；`.sha256` 是封存包络。`task_log_finalize.py` 会在 shell 收尾时重算封存，使哈希覆盖 tee 最终写入的文本日志。

### 6.2 Build manifest

`output/build_manifest.json` 至少记录：

- `chapter_order` 与每章 source 哈希；
- 每个 chunk 的 DB 状态、source 证据、final 路径和 final 哈希；
- merged 成书的路径、哈希、字符数和阶段历史；
- profile/graph 的新鲜度和指纹；
- `run_evidence`：run_id、JSONL/log/seal 路径、字节数、哈希、封存时间和最后 seq。

`book-link-audit` 会从当前磁盘重新推导这些事实，而不是相信 manifest 的自述。manifest 存在时，任何哈希/顺序不一致都会判红。

### 6.3 证据命令

```bash
python3 -m src.main book-link-audit --require-sealed
python3 -m src.main gate
python3 -m pytest -q tests/test_round158_book_linker.py
```

---

## 7. 项目结构

```text
SagaTrans/
├── config.yaml                 # 运行配置：模型、任务路由、质量门槛、风格基线
├── .env.example                # 密钥模板；复制为 .env 后填写
├── requirements-core.txt       # 生产最小依赖（含 typesafe-sdk / httpx2）
├── requirements.txt            # 核心依赖 + EPUB 处理依赖
├── README.md
├── db/
│   ├── workflow.db             # chunk 任务状态机、重试/治理计数
│   ├── decision_db.sqlite      # 术语/典故/风格/实体注册（硬约束事实源）
│   ├── model_capabilities.json # effort 探测结果（可选）
│   └── .sagatrans.lock         # 单实例主锁
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
│   ├── build_manifest.json     # source → chunk → final → merged 交付血缘
│   ├── book_link_audit.json    # 全书链接审计机器报告（手工 --json-out 时生成，可选）
│   ├── book_link_audit.md
│   ├── logs/runs/              # run_id.jsonl / .log / .sha256 / .seq
│   ├── QA_REPORT.md
│   ├── rescue/
│   │   ├── failure_ledger.jsonl  # append-only 失败/救援证据
│   │   └── rescue_report.md      # 根因 cohort 摘要
│   ├── book_lint.md
│   ├── book_repair_report.json  # 全书终检修复机器报告与哈希 manifest
│   ├── book_repair_report.md    # 修复规则、证据、备份与未决项
│   ├── repair_backups/          # book-repair 写入前的逐文件字节备份
│   ├── proofread_report.json    # 机械排错机器报告
│   ├── proofread_report.md      # 机械排错人类可读报告
│   ├── proofread_gate_report.json # gate ⑩ 独立复核
│   ├── proofread_gate_report.md
│   ├── consolidation_diff.md
│   ├── finalize_diff.md
│   ├── note_policy_diff.md
│   ├── recalibrate_critic_report.json
│   ├── pending_backtrack_reconcile.json/.md
│   ├── review_dirty.json
│   └── book.epub               # 可选 EPUB3 成品
├── scripts/
│   ├── translate_book.sh       # 一键主流程（锁、preflight、封存、gate 顺序）
│   ├── task_log_tee.py         # shell 输出 → 带时间戳的 run log
│   ├── task_log_finalize.py    # 收尾重算封存并回写 manifest
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
│   ├── rescue.py               # 失败分类、救援账本、QA 机器证据
│   ├── judgment_layer.py       # 判断层 schema、L1/L2、切片注入
│   ├── story_graph.py          # 确定性实体/场景图 + 验证叙事边
│   ├── story_graph_query.py    # 只读图谱检索与评测
│   ├── setup_payoff_audit.py   # 伏笔候选审计与受控提升
│   ├── style_profiler.py       # 抽样/深读风格剖面
│   ├── recalibrate.py          # 校准裁决重审与证据补齐
│   ├── proofread.py            # 成书机械排错（纯确定性，无 LLM）
│   ├── book_repair.py          # 全书证据门控修复与哈希 manifest
│   ├── book_linker.py          # build manifest、链接审计、日志封存校验
│   ├── book_lint.py            # final/merged lint、finalize、notes-normalize
│   ├── task_logging.py         # run_id、时间戳、JSONL、seq、封存与校验
│   ├── name_authority_audit.py # 专名权威审计
│   ├── name_authority_align.py # 受控隔离/对齐
│   ├── name_only_fastpath.py   # 纯专名 DIRTY 快速闭环
│   ├── pending_backtrack_reconcile.py # 遗留回溯队列分诊
│   ├── consistency.py          # 命名一致性校正
│   ├── splitters.py            # EPUB/TXT/MD 输入切分
│   ├── drift_audit.py          # 章级语域/句长漂移诊断
│   ├── verse.py                # 韵文识别与 verse_lint
│   ├── core/                   # scheduler、decision_engine
│   ├── agents/                 # reference、rewriter、critic、judge
│   ├── judges/                 # System One、pairwise、verdict 合成
│   └── utils/                  # LLM adapter、chunker、文本匹配、停用词
├── tests/                      # round 粒度回归与契约测试
├── tools/                      # 辅助探针/日志过滤/演示
├── docs/                       # 本地研究计划/审计说明；已跟踪 probes 例外
└── .github/workflows/          # 3.10/3.13 双解释器 CI
```

### 生成文件纪律

- `src/` 是事实源；修改 Python 业务代码后运行 `scripts/build_integrated.py` 重新生成 `src/translator_agent.py`。
- `book_lint` 的部分函数/常量会在 `translator_agent.py` 内形成镜像区；修改 `src/book_lint.py` 后运行 `scripts/sync_book_lint_mirror.py`。
- 不要直接编辑 `src/translator_agent.py` 的生成区，否则下一次构建会覆盖。
- `input/book_profile.json`、`input/story_graph.json`、`db/`、`output/`、`.env` 是本机运行产物，不应提交。
- `docs/*.md` 默认是本地研究记录；仓库只跟踪少数 `docs/probes/` A/B 报告和基线夹具。

---

## 8. 配置与密钥

### 8.1 `.env` 是正常使用的唯一密钥源

日常只需要维护项目根 `.env`：

```bash
cp .env.example .env
# QWEN_API_KEY / DEEPSEEK_API_KEY / MINIMAX_API_KEY / MOARK_API_KEY
```

`config.yaml` 只保存 `${ENV_NAME}` 占位符。`src/config.py` 启动时读取项目根 `.env` 到内存并展开占位符，不会把密钥导出到 `os.environ`。加载顺序是：

1. `config.yaml` 持久配置；
2. 项目根 `.env` 展开 `${...}`；shell 同名变量只在 `.env` 缺项时兜底；
3. `.env` 中的 `SAGATRANS__<PATH>` 覆盖，随后真实环境变量覆盖；

`SAGATRANS__...` 仅用于 CI、排障和一次性实验，不是日常配置方式。示例：

```bash
SAGATRANS__MODELS__PRIMARY__MODEL_NAME=your-model \
SAGATRANS__MODELS__PRIMARY__API_BASE=https://api.example.com/v1 \
python3 -m src.main pipeline --chapter ch01
```

不要把 key 写进 `config.yaml`、README 或测试 fixture。旧 `TYPESAFE_API_KEY` 只保留兼容；现役判断后端使用 `MOARK_API_KEY`。

### 8.2 主要配置段

| 段 | 说明 |
| --- | --- |
| `models` | 三个角色的 `api_base`、`api_key`、`model_name`；唯一模型定义处 |
| `llm_backend` | `openai_api` 为生产，`mock` 仅用于 dry-run/回归 |
| `openai_api` | HTTP 重试、退避和超时 |
| `task_routing` | 任务 → 角色、effort、temperature、max_tokens、extra_body |
| `pipeline` | chunk 调度、质量重试、早停、双稿、回溯治理、救援、name-only fast path、gate ⑧ 阈值 |
| `judgment_injection` | 全书注记是否注入生成 prompt；当前默认关闭 |
| `story_graph` | 图谱路径、共现窗口、覆盖率与场景长度 |
| `decision_engine` | 决策回溯触发条件、scope、队列与预算 |
| `critic_thresholds` | 通用 critic 参考门槛，可被作品 profile 覆盖 |
| `typesafe_thresholds` | 校准 verdict 的概率/置信/Noul 不确定区间 |
| `style_guide` | 无 profile 时的中性风格基线 |
| `typesafe` | MoArk `d3-flash` 判断后端的连接、批大小和 timeout |
| `gate` | `book_link_required` 等交付证据开关 |
| `book` / `verse` | 成书标题/章节制式与韵文对齐参数 |

### 8.3 解释器与依赖

- 支持 Python 3.10+；CI 固定测试 3.10 与 3.13。
- `requirements-core.txt` 已显式声明 `typesafe-sdk==0.7.1` 与 `httpx2>=2.0.0,<3`。生产判断后端不再依赖 conda 环境里“偶然存在”的包。
- 一键脚本按 `SAGATRANS_PYTHON` → 项目 `.venv/bin/python` → `PATH` 的 `python3` 顺序选择解释器。
- preflight 会检查 Python 版本、`yaml/requests/psutil/json_repair`；在非 dry-run、非 `--merge-only` 模式下还会检查 `typesafe_sdk` / `httpx2` 和可解析的 `MOARK_API_KEY`。
- `python` 命令不保证存在；请使用 `python3`，或显式设置 `SAGATRANS_PYTHON`。

---

## 9. 脚本使用方法

### 9.1 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-core.txt
python3 -m pip install -r requirements.txt
python3 -m pip install pytest
```

手工直接调用核心 CLI 时，正文流水线不需要 EPUB 工具；但默认一键脚本在未加 `--skip-epub` 时仍会探测并自动安装 `ebooklib` / `beautifulsoup4` / `lxml`。只有 `--skip-epub` 才会跳过 EPUB 转换，也跳过这段依赖检查。

### 9.2 一键主流程

```bash
# 完整翻译 + 收敛 + 终检修复 + 合并 + notes-normalize + proofread + qa-report + lint + gate
./scripts/translate_book.sh book.epub

# 复用已经切分好的 input/chXX.md
./scripts/translate_book.sh book.epub --skip-epub

# 破坏性重置：删除 DB、output/、input/ 和 db/chunk_tags.json 后重来
./scripts/translate_book.sh book.epub --reset

# 只跑流程，不调用真实 LLM；split 仍会执行，gate 跳过
./scripts/translate_book.sh book.epub --dry-run

# 已有 final.json，只做收口
./scripts/translate_book.sh --merge-only

# 清理中间产物（仅在 gate 全绿后执行；gate 失败时保留证据）
./scripts/translate_book.sh book.epub --clean

# 只清理旧中间产物，不翻译
./scripts/translate_book.sh --clean-only
```

`--reset` 会删除 `db/workflow.db`（含 WAL/SHM）、`db/decision_db.sqlite`（含 WAL/SHM）、`db/chunk_tags.json`、整个 `output/`，以及整个 `input/`。`input/` 里若有手工整理或尚未备份的章源，会一并丢失。脚本只在 stdin 是交互终端（`[[ -t 0 ]]`）时显示 `[y/N]`；CI、管道、`nohup` 等非交互调用不会提示，直接执行删除。请先备份，并避免把 `--reset` 与 `--skip-epub` 组合使用。

### 9.3 常用选项

| 选项 | 作用 |
| --- | --- |
| `--force` | 强制重新初始化章节（smart-force 主要重置未完成块） |
| `--dry-run` | 使用 mock 后端；不消耗模型额度，gate 跳过 |
| `--skip-epub` | 不转换输入，直接使用现有 `input/ch*.md` |
| `--reset` | 破坏性重置：删除 DB（含 WAL/SHM）、`db/chunk_tags.json`、`output/` 和 **`input/`**；仅在交互终端提示确认，非交互调用直接删除 |
| `--clean` | gate 全绿后清理 raw/literary/critic 等中间文件；gate 失败时自动跳过；保留 final 与 DB |
| `--clean-only` | 独立清理中间文件，不运行翻译 |
| `--merge-only` | 不翻译，直接收口已有 finals |
| `--no-normalize` | 跳过 finalize 样式规范化 |
| `--skip-deep-profile` | 跳过全书深读，沿用现有/抽样剖面 |
| `--consistency` | 合并后额外运行命名一致性报告 |
| `--golden-gate` | 运行 Golden Gate 质量门禁（失败仅 WARN） |
| `--allow-incomplete` | 调试用，放行 DIRTY/lint/gate 硬闸；不要用于生产 |
| `--auto-merge-on-failure` + `--yes` | API 耗尽时自动合并已完成章节，适合人工值守的恢复场景 |

一键脚本在 `db/.sagatrans.lock` 上持有单实例锁，Python 写库命令和 shell 脚本共用同一把锁，并额外持有 `/tmp` 运行期副锁。Ctrl+C 第一次只警告，第二次才中止；已经完成的 chunk 和 DB 状态会保留，可断点续跑。

### 9.4 手工 CLI

常用命令均从项目根运行。`src.main` 和 `src.translator_agent` 都能作为入口；生产单体通常用后者。

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
python3 -m src.main book-repair --apply
python3 -m src.main recalibrate-critic --all --apply --workers 2
python3 -m src.main book-link-audit --require-sealed
./scripts/translate_book.sh --merge-only

# 诊断与交付
python3 -m src.main lint
python3 -m src.main gate
python3 scripts/editor_review.py --global --review-dirty
python3 scripts/pun_audit.py
python3 scripts/literary_metrics.py --print
python3 scripts/build_epub.py --md "output/<书名>.md" --out output/book.epub
```

### 9.5 命令用途速查

#### 翻译与调度

| 命令 | 用途 |
| --- | --- |
| `pipeline --chapter chXX` | 跑一章的翻译/润色/审计/裁决/回溯闭环 |
| `pipeline --fix-from-review` | 读取 `review_dirty.json`，二次校验锚句后定向重跑 P0 chunk |
| `init --chapter chXX [--force]` | 切 chunk、写 workflow DB、登记韵文/旁路元数据 |
| `split --input ...` | EPUB/TXT/MD 切章，生成 `input/chXX.md` 与转换审计 |
| `rescue [--chapter chXX / --all] [--apply] [--reset-rescue-budget]` | 扫描 PF/FAILED/DIRTY/fallback final；默认只读，`--apply` 执行定向救援 |
| `requeue --chapter chXX [--chunks ...] [--from-ledger] [--force]` | 将指定/账本记录的 PF 块放回 DIRTY；消耗 `pf_requeue_max` 预算 |
| `consolidate --max-rounds N` | 消费 pending backtrack，收敛 DIRTY；未收敛返回非零 |
| `recalibrate-critic --all [--chunks ...] [--force] [--apply]` | 对现有 final 重跑校准裁决并回写 critic 报告；默认只读，`--apply` 才写回 |

#### 专名/术语治理

| 命令 | 用途 |
| --- | --- |
| `name-authority-audit` | 只读检查 Decision DB / entity registry / 锁定目标，输出 quarantine / align / manual-review |
| `name-authority-align --apply` | 执行权威审计允许的隔离/对齐动作，不猜译名 |
| `pending-reconcile [--apply --safe-only / --locked-target-only]` | 对遗留 pending backtrack 分诊，只消费已满足或已定向 DIRTY 的行 |
| `name-only-fastpath --chapter chXX --apply` | 纯专名 DIRTY 的受控快速闭环；只处理其他评分过关的块 |
| `completed-name-repair --all --apply` | 修复 COMPLETED 块中可证明的一字 CJK 漂移并走 calibrated critic 重审；不带 `--apply` 为只读分诊 |
| `confirmed-term-drift-repair --apply` | 只修复已确认、次数可对齐的具体术语漂移 |
| `decision-authority-align` / `decision-target-authority-repair` | Decision 权威对齐与 target 修正的只读/受控路径 |
| `locked-target-authority-audit` / `locked-target-authority-repair` | 锁定 target 的审计与修复 |
| `term-form-audit` / `term-form-contract` | 术语形态审计与合同生成 |
| `consistency [--apply]` | 命名一致性校正；默认只报告 |

#### 判断层与知识图谱

| 命令 | 用途 |
| --- | --- |
| `style-profile [--deep]` | 抽样或全书深读生成/升级 `book_profile.json` |
| `story-graph [--require-answerable]` | 确定性重建实体/场景/叙事图；覆盖率不可答时退出 2 |
| `story-graph-query` | 只读场景检索、注记匹配、DPP 重排、评测与 trace |
| `setup-payoff-audit [--llm --l2 --apply]` | 伏笔候选审计；只有 `--apply` 才受控提升已验证配对 |
| `drift-audit` | 章级语域/句长漂移诊断；`--jev` 是历史参数名 |
| `query-gold-set` | 生成/维护图谱检索评测集 |

#### 成书与交付

| 命令 | 用途 |
| --- | --- |
| `finalize --apply` | 统一成书引号/脚注/策略元语 |
| `book-repair [--apply]` / `--verify` | 全书证据门控修复；默认只报告，`--apply` 才写 final 并生成哈希 manifest |
| `notes-normalize --apply` | 注释去重、孤儿改锚、顺编号 |
| `proofread [--apply]` | 成书机械排错；默认只报告，`--apply` 仅删拼接残留标题/不可见字符 |
| `qa-report` | 从当前 final 重建机器可核验的 QA_REPORT；写命令受单实例锁保护 |
| `lint` / `gate` | 成书格式检查；十三项交付不变量检查 |
| `book-link-audit [--require-sealed]` | 重新推导 source → chunk → final → merged 链接，校验 manifest 与运行日志封存 |
| `book-link-audit --update-merged --stage ...` | 在 notes-normalize / proofread 后更新 merged 哈希与阶段历史 |
| `golden` / `golden-gate [--ab]` | Golden Set 跑分、模型×effort A/B、人工基线动态门禁 |
| `golden --judgment-ab --chapter chXX` | 只比较判断层注记 off/on，不乘入模型矩阵 |
| `memory` / `debug` | 诊断辅助；不是交付门禁 |

### 9.6 测试、镜像与发布前检查

```bash
# 重点契约
python3 -m pytest -q tests/test_round158_book_linker.py
python3 -m pytest -q tests/test_round158_task_logging.py tests/test_round158_task_log_shell.py
python3 -m pytest -q tests/test_round158_name_only_fastpath.py

# 全量回归（3.10）
python3 -m pytest -q tests/

# 生产解释器全量回归（例如 conda/venv）
"$SAGATRANS_PYTHON" -m pytest -q tests/

# 修改散装代码后
python3 scripts/build_integrated.py
python3 scripts/build_integrated.py --check
python3 scripts/sync_book_lint_mirror.py --check

# 配置语法检查
python3 -c "import yaml; yaml.safe_load(open('config.yaml', encoding='utf-8'))"
```

`build_integrated.py` 在生成和 `--check` 时都会做 symtable 自由名扫描：任何“语法合法但函数体引用了未定义全局名”的缺口都会直接失败，不再只比较字节。CI 使用 Python 3.10 和 3.13 两个解释器跑同一套测试；`TestPython310SyntaxCompat` 只检查语法，因此新增代码还应避免依赖仅某个解释器环境里偶然存在的包。真实付费模型 A/B 必须保留报告并区分 mock、legacy、校准 System One 的结果。

### 9.7 生产交付检查表

1. 主锁与 `/tmp` 运行期副锁均空闲（按 3.3 的双锁检查脚本验证；单查主锁不完整）；
2. 当前章节无 `EXTRACTING_TERMS` / `TRANSLATING_RAW` / `REWRITING_LITERARY` / `AUDITING` / `JUDGING` 在途块；
3. `consolidate` 返回 0，`DIRTY=0`；
4. `book-repair --apply` 返回 0，`pending=0 / blocking=0`；
5. `recalibrate-critic --all --apply` 完成，`stale=0`；
6. `book-link-audit --require-sealed` 通过；
7. `gate` ①–⑬ 全绿；
8. `build_integrated.py --check`、`sync_book_lint_mirror.py --check`、双解释器全量测试通过。

任何一项失败都不应通过删除报告、伪造 manifest 或 `--allow-incomplete` 绕过。

---

## 10. 判断层与全书知识图谱

这是项目相对传统 chunk 流水线最有研究价值的部分，也是当前默认最保守的部分。

### 10.1 判断层注记

- 深层阅读产出统一 envelope：`id, type, chapters, claim, evidence, confidence, status, source_model, created_at`；
- evidence 必须能落到 `input/chXX.md`，而不是落到模型看到的拼接文本；
- L1 先 exact，再归一化 exact，最后滑窗相似度；不命中即丢弃；
- L2 用低成本模型小批复核；术语/译名断言直接 rejected；
- 通过验证的 active 注记可用于故事图与全书终审；是否进入生成 prompt 由 `judgment_injection` 单独控制，当前关闭。

这种设计把“理解”变成可检查制品，而不是无法回滚的隐式上下文。注记失败只影响判断层，不回滚正文。

### 10.2 知识图谱的确定性底座

`story-graph` 记录的是**可证明的关系**，不是模型想象中的世界设定：

- 实体来自 Decision DB / entity registry；
- 场景来自章源和标题/分隔线边界；
- 叙事边来自 L1∧L2 通过的 active 注记；
- 解析不了的 alias 写入 `knowledge_gaps`，不静默合并；
- 图带 fingerprint 和生成时间，检索结果可以回到同一批证据。

### 10.3 检索与伏笔回收

`story-graph-query` 提供有界多跳检索、注记种子、DPP 多样性重排和可选 LLM 必要证据复核。它适合研究“某个人物在哪些场景出现”“某个伏笔是否在后文被回应”“某个问题最相关的场景是哪几个”，但不直接改译文。

`setup-payoff-audit` 让模型只从候选 ID 中选回收场景；evidence 由代码按 setup/payoff 章源拼接，再走 L1/L2。模型不能返回自由文本 quote 或新 ID。`--apply` 前会比较图谱基线的 certified coverage 和 unresolved aliases，避免“越审越乱”。

这两条链路的价值是提供可审计的全书叙事检索和知识缺口报告，而不是让 LLM 变成“全书事实数据库”。术语与译名仍由 Decision DB 决定。

---

## 11. 质量语义与恢复边界

### 11.1 fallback 与 PERMANENTLY_FAILED

质量重试耗尽时，pipeline 会把当时的最佳版本写入 final，并标记 `fallback=True`。这保证“已有产出不丢”，但 fallback 不是可交付稿：

- 本章下一次 `pipeline` 运行会自动扫描 PF 和 fallback final，按 `pipeline.rescue` 预算进入定向救援；
- 优先复用 raw/literary 从 `AUDITING` / `REWRITING_LITERARY` 续跑，预算用尽才整块重跑，二者都耗尽才保持 PF；
- 失败分类、续跑阶段、分数、critic backend 和结果写入 `output/rescue/failure_ledger.jsonl`，`rescue_report.md` 汇总根因 cohort；
- 手动入口是 `rescue --chapter chXX --apply` 或 `rescue --all --apply`；`requeue --chunks ... --from-ledger` 可只恢复指定块；
- 终审 `review_dirty` 只对结构 P0 做定向修复；`version_merit` 在重跑失败时恢复旧备份。

因此，一次章节运行中的部分 fallback 不是“最终判决”，但也不能把 fallback 当完成品交付。只要 final 仍带 fallback/quality_waived，gate ③/⑨ 就会让整书失败；lint 或校准检查不会把降级稿“修成”通过，也不会替代 ⑨ 的硬失败。

### 11.2 自动门禁能保证什么

`gate` 全绿证明的是交付不变量，不是文学质量：

- 每个 final 有状态机、source 证据、备份和哈希记录，成书可由 finals 重建；
- 没有 fallback、未校准冒充、阻断性机械错误、待修全书问题或陈旧修复凭证；
- manifest 与运行日志封存证据成立，交付链接可回读；
- 任一硬约束失败都会阻止交付，不能以“队列看起来跑完”或进程退出码代替。

它并不证明文风成熟、脚注安排合理、人物声音统一，也不证明每条 legacy critic 报告都绑定了最终字节。warning 不是失败项时仍可能保留在 `book_lint`、`proofread` 或 `unbound` 证据中，必须按报告判断。

### 11.3 什么时候需要人工介入

以下情况建议人工检查而不是继续自动重跑：转换审计覆盖率异常、实体 registry 硬冲突持续不收敛、gate ⑧ 校准占比低于阈值或出现 `stale` 证据、连续多章 fallback、终审锚句无法唯一映射、故事图 alias 缺口持续扩大，以及 gate ⑩ 报告了无法由安全规则自动修复的编码/结构错误。相关报告都在 `output/` 与 `input/`，且带时间/路径信息。

即使没有上述硬失败，只要目标是文学出版而非内部候选稿，仍需人工双语抽读，重点检查叙述声音、翻译腔、解释性脚注密度、双关与诗性损失、章节标题/编号制式以及跨章人物称谓。机器只能给出证据和告警，不能替代这一层判断。

---

## 12. 当前验收基线（Round 158，2026-11-20）

以下是一次本地参考运行的验收数字，用来说明系统当前的交付强度；它们不是仓库常量，后续书籍和配置会不同。

- 规模：7 章、953 个 chunk；
- 交付：`953/953 COMPLETED`；
- QA fallback：0；
- 双关核销：`8/8`，无 blanket waiver；
- 校准裁决：`100.0%`（`backends={'jev': 953}`；`exact=800`、`repaired=153`、`stale=0`、`unbound=0`）；
- book-repair：`pending=0`、`blocking=0`；
- pending_backtrack：0；
- gate：①–⑬ 全绿；
- 运行证据：`book-link-audit --require-sealed` 为 `passed=30 / failed=0`；
- 解释器回归：Python 3.10 与 3.13 均为 `2537 passed, 11 skipped`；
- 静态检查：`build_integrated.py --check`、`sync_book_lint_mirror.py --check`、`bash -n scripts/translate_book.sh`、`git diff --check` 全绿。

这次交付同时验证了 `run.seal` 的正确语义：manifest 在 merge 阶段先写，后续 gate/审计仍会向同一 JSONL 追加事件；只要已封存字节前缀不变，追加合法；截断或改写前缀仍然 fail-closed。

---

## 13. 给人类研究者与自动化协作者

### 人类研究者

- 先区分四类问题：**生成质量**、**判断可信度**、**一致性治理**、**交付不变量**；不要用一个 BLEU/风格分掩盖系统性问题。
- 研究注记与图谱时，先看 `input/book_profile.json` 的 evidence 是否真的落在章源，再看 L2 是否把术语断言挡掉。
- 研究检索时，先看 `story_graph.json` 的 `knowledge_gaps` 和 fingerprint，再决定是否需要 LLM 复核；不要把模型生成的解释当作图谱事实。
- 真实付费模型 A/B 报告在已跟踪的 `docs/probes/` 中；本地研究计划/审计记录默认不入库，因此 README 不假设它们一定存在于 GitHub。
- 评价“零人工干预”时，请明确是“可交付机械候选稿”还是“可直接出版的文学定稿”；本项目的自动化边界是前者。

### 自动化协作者

- 修改业务代码只改 `src/` 散装模块；不要手改 `src/translator_agent.py` 的生成区。
- 改 `src/book_lint.py` 后必须同步镜像并跑 parity。
- 任何写 `db/`、`output/`、`input/book_profile.json` 的生产命令都要先确认单实例锁。
- 新命令如果会写 DB/制品，必须加入 `_MUTATING_COMMANDS` 或显式 `--apply` 锁判定；不要绕过 flock。
- 不要为通过测试而降低断言；配置清理、schema 兼容、封存语义和 gate 行为都应有回归测试。
- 没有用户明确授权，不要提交、push、清空生产 DB 或启动全量付费翻译。

---

## 14. 边界与声明

- 本项目只应处理**公有领域或已获授权**的作品；译文版权与使用责任由部署者承担。
- 模型端点、代理和配额策略因环境而异；网络受限时可在 shell 层配置代理，但不要在仓库中提交真实代理凭据。
- `.env`、`input/ch*.md`、`input/book_profile.json`、`input/story_graph.json`、`db/`、`output/` 和本地研究文档默认不入库。
- README 描述的是当前代码契约；历史 Round 计划、审计记录和 A/B 报告用于解释演进，不应覆盖当前配置与 gate 语义。

---

**一句话总结：模型提出译文，代码保存事实；判断层提供上下文，交付 gate 决定能否交付。gate 不判断文学是否定稿。**
