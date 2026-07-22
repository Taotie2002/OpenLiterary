# OpenLiterary: AI 文学语义编译系统

基于多 Agent 协同与增量回溯机制的长篇文学翻译系统。专为解决长篇文学翻译中的术语不一致、风格坍缩及典故缺失问题而设计。

## 核心设计哲学

- **文学编译而非翻译**：将翻译过程解构为"语义提取 → 中间表示 → 目标语言重构"的编译流水线
- **决策驱动**：通过 `Decision DB` 维护全局翻译决策（术语、典故、风格），确保全书一致性
- **双 LLM 路由**：DeepSeek 承担文字量最大的直译与润色任务，MiniMax 承担需要高精度的考据、评分与裁决

## 技术栈

- **推理引擎**：支持 `OpenAI Compatible API`（MiniMax / DeepSeek）及 `MLX`（Apple Silicon 本地推理）
- **存储架构**：SQLite 状态机驱动 + JSON 知识图谱
- **协同架构**：基于原生 Python 状态机的有向无环图 (DAG) 流水线

## 目录结构

```text
.
├── config.yaml           # 全局配置（per-role 后端/模型/参数、任务路由、质量阈值）
├── db/                   # SQLite 任务状态库与 Decision DB
├── docs/                 # 翻译准则与风格定义文档、审计报告
├── input/                # 原文 Markdown 文件（EPUB 自动转换）
├── output/               # 翻译阶段性产物、最终译文、QA 报告
├── scripts/
│   └── translate_book.sh # 一键翻译脚本（全自动流水线）
├── src/
│   ├── translator_agent.py  # 主执行入口（单体聚合版，生产使用）
│   ├── agents/              # Agent Prompt 与逻辑（模块化参考副本）
│   ├── core/                # 状态机调度器与决策引擎
│   ├── utils/               # 文本清洗、切分与 LLM 适配器
│   ├── config.py            # 配置加载器
│   └── pipeline.py          # 模块化流水线（DEPRECATED，历史参考）
└── .gitignore
```

## 快速开始

```bash
# 1. 编辑 config.yaml 填入 API Key
# 2. 一键翻译 EPUB
./scripts/translate_book.sh book.epub --reset

# 3. 完成后查看译文
cat output/translated_full.md
# 查看质量验收报告
cat output/QA_REPORT.md
```

## 环境要求

- Python >= 3.10
- API Key：
  - `DEEPSEEK_API_KEY` — 直译与文学润色（推荐 deepseek-v4-flash）
  - `MINIMAX_API_KEY` — 典故考据、评分与裁决（推荐 MiniMax-M3）

```bash
export DEEPSEEK_API_KEY=sk-xxx
export MINIMAX_API_KEY=xxx
```

## LLM 角色路由

系统将 6 类任务路由至 2 个模型，实现负载与成本的平衡：

| 任务 | 路由 | 模型 | 说明 |
|------|------|------|------|
| `reference_extraction` | MiniMax | MiniMax-M3 | 典故考据（需要高精度） |
| `literal_translation` | DeepSeek | deepseek-v4-flash | 字面直译 |
| `literary_rewrite` | DeepSeek | deepseek-v4-flash | 文学润色（token 最重，分流至低成本模型） |
| `critic_scoring` | MiniMax | MiniMax-M3 | 5 维评分（需要高精度） |
| `judge_decision` | MiniMax | MiniMax-M3 | 最终裁决（需要高精度） |

> 可通过环境变量临时切换任意模型，无需修改 `config.yaml`。

## 流水线阶段

1. **术语提取** (Reference Agent) — 识别文学典故、宗教隐喻、历史名词
2. **字面直译** (Literal Translator) — 字面忠实，不丢失细节
3. **文学润色** (Rewriter) — 依据风格指南重构译文
4. **审辩评分** (Critic) — 5 维评分（流畅度/风格/语义/可读性/一致性）
5. **最终裁决** (Judge) — 判定是否通过，否则退回润色
6. **决策回溯** (Backtrack) — 新决策触发已完 chunk 重译

## 质量保障系统

### 质量重试与安全网

- **重试上限**：`max_quality_retries: 6` — 每 chunk 最多 6 轮润色-裁决循环
- **降级阈值**：实体一致性重试上限为 4 轮（低于通用上限，避免死锁）
- **安全网**：当各维度评分接近阈值（平均分 ≥ 5.5，无严重偏离），即使裁决未通过也可放行并标记为 `🟡 安全网放行`
- **早期止损**：首轮评分低于 4.0 分且超过 3 个维度时，立即停止重试

### 实体一致性

- 全书实体注册表（entity_registry）自动维护译名一致性
- 质量重试 > 0 时启用降级机制：严重不一致项 ≤ 3 时允许放行，> 3 时继续退回润色

### 输出验证

- 每个 chunk 无论通过与否均写入 `*_final.json`（fallback 机制）
- `QA_REPORT.md` 记录所有未通过自动化验收的 chunk 及原因
- 合并阶段校验缺失/不完整章节并报错

## 脚本参数

```bash
./scripts/translate_book.sh <input.epub> [选项]
```

| 参数 | 说明 |
|---|---|
| `-f, --force` | 强制重新初始化章节（覆盖已存在的 chunk 状态） |
| `-n, --dry-run` | 仅跑流程，不调用 LLM（用 mock 后端） |
| `-r, --reset` | **清空所有历史状态**（删除 DB + output/），重新开始 |
| `-c, --clean` | 翻译完成后清理中间产物（保留 final.json 和 DB） |
| `--clean-only` | 独立清理模式，无需输入文件 |
| `--skip-epub` | 跳过 EPUB 转换，直接用 `input/` 下的 `.md` |
| `-h, --help` | 显示帮助 |

## 典型工作流

### 1. 全自动翻译

```bash
./scripts/translate_book.sh book.epub --reset
```

### 2. 按章节翻译

```bash
# 在全自动模式下，失败的章节可以单独重跑：
./scripts/translate_book.sh book.epub --chapters ch03,ch04,ch06
```

### 3. 分步手动执行

```bash
# 切分 EPUB → input/ch*.md
python3 -m src.translator_agent split --input book.epub --input-dir input

# 初始化单章
python3 -m src.translator_agent init --chapter ch01

# 强制定向（覆盖已有状态）
python3 -m src.translator_agent init --chapter ch01 --force

# 跑单章管线
python3 -m src.translator_agent pipeline --chapter ch01
```

## 环境变量覆盖

在不修改 `config.yaml` 的前提下临时切换后端/模型：

```bash
# 切换 DeepSeek 模型
export OPENLITERARY__MAIN__MODELS__LITERAL_TRANSLATOR__MODEL_NAME=deepseek-chat

# 切换 MiniMax 模型
export OPENLITERARY__MAIN__MODELS__PRIMARY__MODEL_NAME=MiniMax-M1

# 使用 mock 后端测试流程（不消耗 token）
OPENLITERARY__LLM_BACKEND=mock ./scripts/translate_book.sh book.md output --dry-run
```

## 决策回溯机制

- **术语一致性**：全书专有名词译法统一
- **典故回溯**：新增典故决策后重译受影响 chunk
- **风格约束**：Critic/Judge 积累的风格规则传递至后续章节
- **质量重试耗尽**：API 级重试耗尽时从中间产物抢救 fallback final

## 配置参考

编辑 `config.yaml` 选择后端和模型：

```yaml
main:
  models:
    literal_translator:
      backend: openai_api
      api_base: https://api.deepseek.com
      api_key: ${DEEPSEEK_API_KEY}
      model_name: deepseek-v4-flash
    primary:
      backend: openai_api
      api_base: https://api.minimaxi.com/v1
      api_key: ${MINIMAX_API_KEY}
      model_name: MiniMax-M3
```

支持 backend: `openai_api`（通用 OpenAI 兼容）、`mlx`（Apple Silicon 本地）。

## 许可证

MIT
