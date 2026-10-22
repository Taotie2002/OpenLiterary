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
- **双 LLM 分工路由**：`literal_translator` 承担直译与考据召回；`primary` 承担润色、评分、裁决、考据确认与风格剖面。`thinking` 按角色分档（裁决/考据/风格/困难片段开启；直译/评分默认关闭），高难度片段经难度路由切至主力模型 + thinking。
- **动态风格剖面**：风格提示词由 `StyleProfiler` 从原文抽样动态生成（按作品隔离落盘 `input/book_profile.json`），不硬编码任何作品先验。
- **Jev 仲裁核心**：TypeSafe System One 概率校准裁决；不可用时按 `confidence_source` 分档降级并标记未校准（绝不静默当作 Jev 结论使用）。

## 流水线

```
split → init → pipeline → style-profile → consolidate → finalize
      → notes-normalize → lint → consistency → (golden-gate) → gate → build_epub
```

- **pipeline**：chunk 级「直译 → 文学润色 → 批评评分 → 裁决 → 回溯」闭环，状态机 + SQLite 持久化，可断点续跑。
- **consolidate / finalize**：跨 chunk 合并、章内衔接与终稿整合，产出 `output/ch*/ch*_chunk*_final.json` → `output/translated_full.md`。
- **lint / consistency / gate**：脚注、引号、韵文、双关、术语一致性与交付门禁。

## 质量体系

- **交付门禁 gate（七项）**：状态机完成度 / 产物完整性 / QA_REPORT / lint / EPUB→MD 转换审计（覆盖率） / 韵文校验 / 双关核销。
- **成书级审读** `scripts/editor_review.py`：P0/P1/P2 分级（悬空引语、翻译腔、语域漂移、注密、标点规范等），支持 `--chapters` 定点复审与 `--fail-on-p0` 门禁。
- **文学指标** `scripts/literary_metrics.py`：AI 味密度 / 翻译腔计数 / 重复率。
- **双关核销** `scripts/pun_audit.py`：全书双关点位登记与覆盖门禁。
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
| `HEFEI_API_KEY` | `config.yaml` 中直译角色的 OpenAI 兼容网关（默认 `api_base: token.ai-hf.cn`，模型 Deepseek-V4.1-Flash） |
| `MINIMAX_API_KEY` | 主力模型：润色 / 评分 / 裁决 / 风格剖面 / 考据确认（api.minimax.cn） |
| `TYPESAFE_API_KEY` | TypeSafe Jev 仲裁（可选；缺失时自动降级并标记未校准） |
| `DEEPSEEK_API_KEY` | 直连 DeepSeek 官方 API 时使用（可选） |

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
export SAGATRANS__MAIN__MODELS__PRIMARY__MODEL_NAME=MiniMax-M3.1-Flash
export SAGATRANS__MAIN__MODELS__PRIMARY__API_BASE=https://api.minimax.cn/v1
export SAGATRANS__MAIN__MODELS__PRIMARY__API_KEY=$MINIMAX_API_KEY
./scripts/translate_book.sh book.epub
```

Apple Silicon 本地推理（MLX）见 `config.yaml` 的 `mlx:` 段。

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
