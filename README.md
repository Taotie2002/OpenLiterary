# OpenLiterary: AI 文学语义编译系统

一个基于多 Agent 协同与增量回溯机制的长篇文学翻译操作系统。旨在解决长篇文学翻译中的术语不一致、风格坍缩及典故缺失问题。

## 核心设计哲学
- **文学编译而非翻译**：将翻译过程解构为“语义提取-中间表示-目标语言重构”的编译流水线。
- **决策驱动**：通过 `Decision DB` 维护全局翻译决策，确保全书逻辑一致性。
- **风格三维建模**：基于 `Author Style`, `Translator Style`, `Reference Layer` 的分层建模，规避风格过拟合。

## 技术栈
- **推理引擎**：支持 `MLX` (Apple Silicon 原生) 与 `OpenAI Compatible API` (LM Studio/vLLM)。
- **存储架构**：SQLite 状态机驱动 + JSON 知识图谱 + ChromaDB (可选)。
- **协同架构**：基于原生 Python 状态机的有向无环图 (DAG) 流水线。

## 目录结构
```text
.
├── db/                 # SQLite 任务状态库与 Decision DB
├── docs/               # 翻译准则与风格定义文档
├── input/              # 原文 Markdown 文件 (UTF-8)
├── output/             # 翻译阶段性产物与最终译文
├── src/
│   ├── agents/         # 各类 Agent 的 Prompt 与逻辑
│   ├── core/           # 状态机调度器 (Project Manager)
│   ├── utils/          # 文本清洗、切分与 MLX/API 适配器
│   └── pipeline.py     # 主执行入口
└── plan.md             # MVP 开发路线图