"""
SagaTrans — AI 文学语义编译系统

包入口。定义公共接口与版本号。

模块组织：
  config                 全局配置加载器（基于 YAML + 环境变量）
  core.scheduler         任务状态机（TaskState, TaskScheduler）
  core.decision_engine   决策引擎（DecisionLevel, DecisionEngine）
  utils.llm_adapter      LLM 后端适配层（4 个适配器 + JSON 解析）
  utils.chunker          文本切分器（SmartChunker）
  agents.*               4 个 Agent（Reference / Rewriter / Critic / Judge）
  pipeline               翻译流水线（TranslationPipeline）
  splitters              输入格式适配器（EPUB / TXT / MD）
  consistency            译后命名一致性校正
  judges                 typesafe-ai 重构层（多原子原语 + verdict 合成）
  main                   CLI 入口（init / pipeline / split / consistency / debug）
  test_*.py              单元测试与质量门禁

历史：
  原 src/translator_agent.py 是上述模块的单体聚合版，
  保留用于外部代码评审。新工作以散装模块为事实源。

T2 重构（typesafe-ai）：
  - 原 critic/judge 双 LLM 调用 → 多 atomic primitives 并行评估 + code-side verdict 合成
  - 三级 fallback：Jev（typesafe 原生）→ OpenAICompat（MiniMax/DeepSeek）→ Mock（启发式离线）
  - 详见 src/judges/
"""
__version__ = "2.1.0"
