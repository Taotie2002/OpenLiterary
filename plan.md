# MVP 开发路线图

## Phase 1: 基础设施 (Week 1-2)
- [ ] **状态机调度引擎**：实现基于 SQLite 的状态记录，支持断点续传。
- [ ] **MLX/API 双驱动器**：完成 `Adapter` 抽象层，实现模型的按需加载与显存回收 (`clear_cache`)。
- [ ] **语义切分器**：实现基于自然段与场景标识符的智能分块。

## Phase 2: 核心 Agent 与决策系统 (Week 3-4)
- [ ] **Context & Reference Builder**：构建原著与典故提取流水线。
- [ ] **Decision Engine**：建立 SQLite `Decision DB` 并定义三级准入机制。
- [ ] **风格分层模块**：实现 `Author Style` 与 `Translator Style` 的统计特征计算。

## Phase 3: 流水线闭环 (Week 5-6)
- [ ] **Batch Processing**：重构调度器为“阶段批处理模式”，减少热切换开销。
- [ ] **Backtracking Engine**：开发核心回溯逻辑，触发 `DIRTY` 标记后的局部重翻。
- [ ] **Critic & Judge 验证**：实现基于 Fluent/Readability/Style 评分的自动裁决。

## Phase 4: 实验与调优 (Week 7-8)
- [ ] **黄金测试集跑分**：使用《海伯利安》前 5000 字进行测试，记录风格坍缩率。
- [ ] **性能优化**：针对 16GB 内存进行显存压力测试。
- [ ] **最终调优**：调整 `Author_Priority_Ratio` 的动态推断逻辑。

## 待办风险看板
- [ ] **风险 1**：JSON 输出稳定性（已计划通过 Instructor 库引入 Schema 约束）。
- [ ] **风险 2**：Style Expert 的统计特征提取准确性。
- [ ] **风险 3**：回溯重翻引发的“蝴蝶效应”循环。