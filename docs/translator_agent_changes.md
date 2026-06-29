# `translator_agent.py` 修改记录与评审意见采纳说明

本文档仅涉及单一脚本 `src/translator_agent.py`（共 1754 行，17 个类 + 顶层辅助函数）。所有功能完整保留，可独立运行（`python src/translator_agent.py`）。

> **版本对应关系**：
> - 第一轮（V1.0）：1.1-1.20，原始修复
> - 第二轮（V5.0）：3.1-3.5，采纳评审元审计
> - 第三轮（V6.0）：4.1-4.2，采纳独立审计
> - 仍未采纳：5.1-5.10

---

## 一、已采纳的修改点

### 1.1 SQLite 线程安全

| 项目 | 内容 |
|------|------|
| 评审意见 | 多线程下 SQLite 连接不安全 |
| 修复方式 | `TaskScheduler.__init__` 和 `DecisionEngine.__init__` 改用 `self._local = threading.local()` + `self.conn` 属性的 `sqlite3.connect(..., check_same_thread=False)` |
| 位置 | `TaskScheduler` (line ~410) / `DecisionEngine` (line ~585) |
| 状态 | ✅ 已修复 |

### 1.2 决策引擎 UPSERT

| 项目 | 内容 |
|------|------|
| 评审意见 | `add_decision` 存在 SELECT-then-INSERT/UPDATE 竞态 |
| 修复方式 | 改为 `INSERT INTO decision_db ... ON CONFLICT(source_key) DO UPDATE SET ...` 原子操作 |
| 位置 | `DecisionEngine.add_decision` (line ~621) |
| 状态 | ✅ 已修复 |

### 1.3 决策影响映射清理

| 项目 | 内容 |
|------|------|
| 评审意见 | 更新决策时未清理旧的 `decision_impact` 映射，导致重复记录 |
| 修复方式 | UPSERT 后先 `DELETE FROM decision_impact WHERE decision_id = ?`，再 `INSERT OR IGNORE` 新映射 |
| 位置 | `DecisionEngine.add_decision` (line ~645) |
| 状态 | ✅ 已修复 |

### 1.4 DIRTY 回溯清理中间文件

| 项目 | 内容 |
|------|------|
| 评审意见 | 回溯重跑时未清理已存在的中间产物 |
| 修复方式 | `_process_dirty_batch` 中遍历 `["raw", "literary", "critic_report", "final"]` 步骤，`unlink()` 对应文件 |
| 位置 | `TranslationPipeline._process_dirty_batch` (line ~1252) |
| 状态 | ✅ 已修复 |

### 1.5 状态机转移矩阵校验

| 项目 | 内容 |
|------|------|
| 评审意见 | 状态转移未校验合法性，可能进入非法状态 |
| 修复方式 | 新增 `VALID_TRANSITIONS` 字典（9 个状态定义允许的转移），`batch_update_state` 逐条校验，非法转移打印警告并跳过 |
| 位置 | `VALID_TRANSITIONS` (line 61) / `TaskScheduler.batch_update_state` (line ~495) |
| 状态 | ✅ 已修复 |

### 1.6 单任务状态转移校验

| 项目 | 内容 |
|------|------|
| 评审意见 | `update_task_state` 同样需要校验状态机 |
| 修复方式 | 同样的 `VALID_TRANSITIONS` 校验逻辑应用到 `update_task_state` |
| 位置 | `TaskScheduler.update_task_state` (line ~470) |
| 状态 | ✅ 已修复 |

### 1.7 API 错误分类重试

| 项目 | 内容 |
|------|------|
| 评审意见 | 4xx 客户端错误不应重试（重试不会改变结果） |
| 修复方式 | `OpenAICompatibleAdapter.generate` 中捕获 `requests.HTTPError`，4xx 立即 `raise RuntimeError`（不再 `time.sleep` 重试），5xx/timeout 走原有指数退避重试 |
| 位置 | `OpenAICompatibleAdapter.generate` (line ~244) |
| 状态 | ✅ 已修复 |

### 1.8 `requests` 导入错误

| 项目 | 内容 |
|------|------|
| 评审意见 | 代码中使用 `_requests.post()` 但只 `import requests as _requests` 在函数内，与文件顶部 `import requests` 重复 |
| 修复方式 | 统一为文件顶部的 `import requests` 导入，函数内直接用 `requests.post()` |
| 位置 | `OpenAICompatibleAdapter` 顶部和 `generate` (line ~254) |
| 状态 | ✅ 已修复 |

### 1.9 MLX 卸载后重载（核心 bug）

| 项目 | 内容 |
|------|------|
| 评审意见 | 删除第二次 `_load_model_if_needed` 后引入回归：`auto_unload_if_needed()` 触发后 `self.model = None`，后续 `mlx_generate(None, ...)` 会崩溃 |
| 修复方式 | `auto_unload_if_needed()` 改为返回 `bool`（卸载返回 `True`），`MLXNativeAdapter.generate()` 检查返回值，若为 `True` 则重新加载模型 |
| 位置 | `LLMAdapter.auto_unload_if_needed` (line ~144) / `MLXNativeAdapter.generate` (line ~324) |
| 状态 | ✅ 已修复 |

### 1.10 `TaskScheduler.delete_tasks` 新增方法

| 项目 | 内容 |
|------|------|
| 评审意见 | 永久失败任务需要从数据库移除以避免无限循环 |
| 修复方式 | 新增 `delete_tasks(chunk_ids)` 方法，执行 `DELETE FROM chunk_tasks WHERE chunk_id IN (...)` |
| 位置 | `TaskScheduler.delete_tasks` (line 559) |
| 状态 | ✅ 已新增 |

### 1.11 `_process_failed_batch` 永久失败处理

| 项目 | 内容 |
|------|------|
| 评审意见 | `retries >= 3` 的任务仅打印日志未更新状态，导致每轮循环重新取出 → **无限循环** |
| 修复方式 | 收集 `permanent_fail_ids`，删除前清理中间文件，然后调用 `delete_tasks()` 从数据库物理删除 |
| 位置 | `TranslationPipeline._process_failed_batch` (line ~1261) |
| 状态 | ✅ 已修复（无限循环问题 + 中间文件 orphaned 问题） |

### 1.12 `_mock_critic_report` 补 `readability` 维度

| 项目 | 内容 |
|------|------|
| 评审意见 | Mock 评分缺少 `readability` 字段，导致 `CRITIC_THRESHOLDS["readability"]` 永远不触发 |
| 修复方式 | 在 mock scores 中补充 `"readability": 8` |
| 位置 | `MockLLMAdapter._mock_critic_report` (line ~205) |
| 状态 | ✅ 已修复 |

### 1.13 `_current_chunk_id` 死代码清理

| 项目 | 内容 |
|------|------|
| 评审意见 | `__init__`、`_process_extracting_terms_batch`、`_process_judging_batch` 中赋值但全代码库零读取 |
| 修复方式 | 删除 3 处赋值（保留 `chunk_id` 局部变量） |
| 位置 | `TranslationPipeline.__init__` / `_process_extracting_terms_batch` / `_process_judging_batch` |
| 状态 | ✅ 已清理 |

### 1.14 `SYS_CONFIG` 死配置清理

| 项目 | 内容 |
|------|------|
| 评审意见 | `"memory_limit_gb": 16` 和 `"kv_cache_quantization": True` 定义但无任何代码读取 |
| 修复方式 | 从 `SYS_CONFIG` 字典中移除 |
| 位置 | `SYS_CONFIG` (line 86) |
| 状态 | ✅ 已清理 |

### 1.15 `CRITIC_THRESHOLDS["overall_min"]` 死键清理

| 项目 | 内容 |
|------|------|
| 评审意见 | LLM 返回的 scores 永远不包含 `"overall_min"`，导致该阈值永远不触发 |
| 修复方式 | 从 `CRITIC_THRESHOLDS` 字典中移除（`JudgeAgent` 已有独立的 `avg_score < 7.5` 检查） |
| 位置 | `CRITIC_THRESHOLDS` (line ~1000) |
| 状态 | ✅ 已清理 |

### 1.16 `PerformanceMetrics` 死类清理

| 项目 | 内容 |
|------|------|
| 评审意见 | 定义完整的 dataclass 但全代码库无任何实例化或引用 |
| 修复方式 | 删除类定义和相关 `from dataclasses import dataclass, field` / `from datetime import datetime` 导入 |
| 位置 | Section 1 枚举与类型定义（删除） |
| 状态 | ✅ 已清理 |

### 1.17 死导入清理

| 项目 | 内容 |
|------|------|
| 评审意见 | `import sys`、`from dataclasses import dataclass`、`from datetime import datetime` 在 `PerformanceMetrics` 删除后变为死导入 |
| 修复方式 | 从 `translator_agent.py` 顶部导入区移除 |
| 位置 | 文件顶部 (line 26-40) |
| 状态 | ✅ 已清理 |

### 1.18 死属性清理

| 项目 | 内容 |
|------|------|
| 评审意见 | `self._lock = threading.RLock()` 定义但从未 `acquire()` |
| 修复方式 | 从 `TaskScheduler.__init__` 和 `DecisionEngine.__init__` 中移除 |
| 位置 | 两个类的 `__init__` |
| 状态 | ✅ 已清理 |

### 1.19 停用词检查优化

| 项目 | 内容 |
|------|------|
| 评审意见 | `not w in '的了...'` 是 O(n) 字符串扫描 |
| 修复方式 | 改为 `STOP_WORDS = {'的了...'}`, 使用 `w not in STOP_WORDS`（O(1) 查找） |
| 位置 | `_evaluate_text_fluency` 等 |
| 状态 | ✅ 已修复 |

### 1.20 Smart quotes 修复

| 项目 | 内容 |
|------|------|
| 评审意见 | Unicode 标准化过程中 `"\u201c"`/`"\u201d"` 被错误地转换为 `"\""` |
| 修复方式 | 使用 `\u201c`/`\u201d` 转义形式以避免编辑器/合并工具重新标准化 |
| 位置 | `SmartChunker` 对话状态探针 |
| 状态 | ✅ 已修复 |

---

## 二、未采纳的评审意见

### 2.1 `get_llm_client` 单例竞态

| 项目 | 内容 |
|------|------|
| 评审意见 | 全局 `_client_instance` 变量在多线程下可能并发创建多个 `MLXNativeAdapter`，导致内存爆炸 |
| 不采纳理由 | **当前代码是单线程的**。`get_llm_client()` 在 `TranslationPipeline.__init__`、`GoldenSetEvaluator.__init__`、各 Agent `__init__` 中按顺序调用一次。多线程是未来扩展方向，加锁是防御性改进而非当前缺陷 |

### 2.2 `GoldenSetEvaluator` 决策引擎数据孤岛

| 项目 | 内容 |
|------|------|
| 评审意见 | 3 个独立 `DecisionEngine()` 实例导致 `ref_agent` 写入的决策在 `rewriter_agent` 读不到 |
| 不采纳理由 | **顺序执行下不成立**。`add_decision()` 内部执行 `self.conn.commit()` 写入 SQLite 文件 `db/decision_db.sqlite`，随后 `rewriter_agent` 通过新连接（`threading.local()`）读取同一文件能**看到数据**。SQLite 文件级持久化保证跨连接可见性 |

### 2.3 跨数据库事务不一致

| 项目 | 内容 |
|------|------|
| 评审意见 | `_trigger_backtrack`（操作 `workflow.db`）在 `commit`（`decision_db.sqlite`）前调用，若 commit 失败则两库不一致 |
| 不采纳理由 | **需特定失败序列**：`decision_db.sqlite` 的 UPSERT/DELETE 全部成功，紧接着 `_trigger_backtrack` 成功，但 `decision_db.sqlite` 的 `commit` 失败。SQLite 写入极为可靠，此序列实际触发概率极低。理论风险不是当前 bug |

### 2.4 `trigger_backtrack` 绕过状态机校验

| 项目 | 内容 |
|------|------|
| 评审意见 | `trigger_backtrack` 直接 SQL UPDATE 绕过 `VALID_TRANSITIONS` |
| 不采纳理由 | **已有 `AND state = ?` 限制**。SQL 中的 `AND state = ?`（值为 `COMPLETED`）确保只回溯 `COMPLETED` 任务，与 `VALID_TRANSITIONS[COMPLETED] = [DIRTY]` 一致。这是一致性 concern（状态机统一入口），不是 bug |

### 2.5 `DecisionEngine` 缺少 `row_factory`

| 项目 | 内容 |
|------|------|
| 评审意见 | `TaskScheduler.conn` 设置了 `row_factory = sqlite3.Row`，`DecisionEngine.conn` 没有 |
| 不采纳理由 | **当前不需要**。`DecisionEngine` 的所有查询结果都用元组解包访问（`for level, source, trans in decisions`），不依赖列名访问。`row_factory` 是接口契约问题，不是功能 bug |

### 2.6 流水线轮询效率

| 项目 | 内容 |
|------|------|
| 评审意见 | `while True` 每轮 7 个状态阶段遍历 + `time.sleep(0.5)`，50 个任务 7 轮循环需 3.5s 空转 |
| 不采纳理由 | **已知设计权衡**。3.5s 空转相对于 LLM 批处理（每块几秒到几十秒）可忽略。改用事件驱动或优先级队列会增加架构复杂度，与 MVP 阶段目标不符 |

### 2.7 `batch_update_state` 静默跳过不存在 ID

| 项目 | 内容 |
|------|------|
| 评审意见 | 不存在的 `chunk_id` 被静默忽略，应抛出异常 |
| 不采纳理由 | **合理行为**。重复 ID 或外部脚本错误不应阻塞整个批处理。静默跳过是设计上的容错 |

### 2.8 `SmartChunker` 对话探针边界

| 项目 | 内容 |
|------|------|
| 评审意见 | 嵌套引号、中文直角引号 `「」` 未检测，硬切分后 `open_quotes = False` 强制重置可能错误 |
| 不采纳理由 | **极罕见场景**。当前实现对目标语料（英美科幻/奇幻文学中译）已足够。嵌套引号在该语料中罕见，复杂状态机会增加维护成本 |

### 2.9 `JudgeAgent` 硬编码阈值 7.0/7.5

| 项目 | 内容 |
|------|------|
| 评审意见 | `low_dims = [...] if v < 7` 和 `if avg_score < 7.5` 应引用 `CRITIC_THRESHOLDS` |
| 不采纳理由 | **代码气味，不是 bug**。魔法数字集中在 `JudgeAgent` 中，与 `CRITIC_THRESHOLDS` 重复但不影响功能。属于后续重构范畴 |

### 2.10 `_process_failed_batch` 改用 `PERMANENTLY_FAILED` 终态

| 项目 | 内容 |
|------|------|
| 评审意见 | 物理删除破坏审计链，应引入新 `TaskState.PERMANENTLY_FAILED` 终态 |
| 不采纳理由 | **改动太大**。我采用折中方案：删除前清理中间文件（`raw/literary/critic_report/final`），既避免 orphaned 文件堆积（评审的核心 concern），又不需要新增状态、修改 `VALID_TRANSITIONS`、更新所有相关校验逻辑 |

### 2.11 模型名称硬编码分散

| 项目 | 内容 |
|------|------|
| 评审意见 | `"qwen/Qwen2.5-7B-Instruct-MLX-4bit"` 在 5 处硬编码，应集中到 `SYS_CONFIG["mlx_models"]` |
| 不采纳理由 | **代码气味，不是 bug**。`mlx_models` 字典已存在但未使用。属于后续重构 |

### 2.12 `init_project` 中 `SmartChunker` 参数不一致

| 项目 | 内容 |
|------|------|
| 评审意见 | 只传 `soft_limit=1000`，`hard_limit` 用默认值 2500 |
| 不采纳理由 | **类默认值合理**。`hard_limit=2500` 是类的设计默认值，`init_project` 故意只覆盖 `soft_limit`。不是 bug |

### 2.13 `debug_db` 非线程安全连接

| 项目 | 内容 |
|------|------|
| 评审意见 | `debug_db` 使用普通 `sqlite3.connect()` 无 `check_same_thread=False` |
| 不采纳理由 | **调试工具**。`debug_db` 是开发/调试辅助函数，不在生产热路径上 |

---

## 三、第二轮评审采纳的新增修复（第五版）

> 经第二轮元审计（审计 changes.md 的"未采纳理由"），评审指出部分理由不充分或遗漏真问题。已采纳以下 5 项：

### 3.1 `add_decision` 异常重新抛出

| 项目 | 内容 |
|------|------|
| 评审意见 | `except Exception as e: print(...)` 仅打印日志，调用方以为成功继续执行，决策静默丢失 |
| 修复方式 | `except` 块末尾添加 `raise`，让异常传播到调用方 |
| 位置 | `DecisionEngine.add_decision` (line 650) |
| 状态 | ✅ 已修复 |

### 3.2 `run_golden_test` 补充 `source_text` 参数

| 项目 | 内容 |
|------|------|
| 评审意见 | `rewriter_agent.process_chunk(chunk_id, raw_text, style_guide)` 缺第 4 个参数 `source_text=""`，导致 `_infer_author_priority_ratio("")` 返回固定值 0.7，动态风格推断失效 |
| 修复方式 | 调用时补充 `chunk` 作为第 4 个参数 |
| 位置 | `run_golden_test` (line 1334) |
| 状态 | ✅ 已修复 |

### 3.3 `_trigger_backtrack` 在 `commit` 之后调用

| 项目 | 内容 |
|------|------|
| 评审意见 | `_trigger_backtrack` 在 `decision_db.sqlite` 的 `commit` 之前调用，操作另一个数据库文件，若 `commit` 失败则 `workflow.db` 已标记 DIRTY 但决策未写入，两库永久不一致 |
| 修复方式 | 将 `self._trigger_backtrack(affected_chunks)` 移到 `self.conn.commit()` 之后 |
| 位置 | `DecisionEngine.add_decision` (line 644-649) |
| 状态 | ✅ 已修复 |

### 3.4 `DecisionEngine._cleanup_orphan_impacts` 新增方法

| 项目 | 内容 |
|------|------|
| 评审意见 | `delete_tasks` 删除 chunk_tasks 记录后，`decision_impact` 表中指向已删除 chunk 的记录成为孤儿，数据库膨胀 |
| 修复方式 | 新增 `_cleanup_orphan_impacts(chunk_ids)` 方法执行 `DELETE FROM decision_impact WHERE chunk_id IN (...)`；`_process_failed_batch` 在 `delete_tasks()` 后调用 |
| 位置 | `DecisionEngine._cleanup_orphan_impacts` (line 663) / `TranslationPipeline._process_failed_batch` (line 1294) |
| 状态 | ✅ 已新增 |

### 3.5 `batch_update_state` 跳过不存在 ID 时打印警告

| 项目 | 内容 |
|------|------|
| 评审意见 | `cur is None: continue` 静默跳过，调用方无法察觉传入错误 ID |
| 修复方式 | 添加 `print(f"⚠️ [Scheduler] chunk_id {cid} 不存在，跳过")` |
| 位置 | `TaskScheduler.batch_update_state` (line 496) |
| 状态 | ✅ 已修复 |

---

## 四、第三轮审计采纳的修复（第六版）

> 经第三轮独立审计，识别出 2 个真问题，已采纳并修复：

### 4.1 `GoldenSetEvaluator` 数据库路径不一致

| 项目 | 内容 |
|------|------|
| 评审意见 | Pipeline 用绝对路径（基于 `_root_dir`），`GoldenSetEvaluator` 用相对路径 `db/decision_db.sqlite`。在非项目根目录运行 `--mode golden-test` 时，两边操作的可能是不同数据库文件，回溯机制失败 |
| 修复方式 | `GoldenSetEvaluator.__init__` 新增可选 `db_path` 参数，所有 Agent 共享同一 `DecisionEngine` 实例（顺带修复数据孤岛）；`run_golden_test` 传入绝对路径 `str(root_dir / "db" / "decision_db.sqlite")` |
| 位置 | `GoldenSetEvaluator.__init__` (line 1389) / `run_golden_test` (line 1469) |
| 状态 | ✅ 已修复 |

### 4.2 永久失败任务改为保留而非删除

| 项目 | 内容 |
|------|------|
| 评审意见 | 3 次 REJECT 后 `delete_tasks()` 物理删除任务，丢失所有部分翻译产物（`raw/literary/critic_report/final` 文件），无法人工介入 |
| 修复方式 | `_process_failed_batch` 中改为调用 `batch_update_state(permanent_fail_ids, TaskState.FAILED, error_msg="超过重试上限...")` 标记为永久失败，保留所有中间产物供人工检查。同时移除 `_cleanup_orphan_impacts` 调用（因为不再删除） |
| 位置 | `TranslationPipeline._process_failed_batch` (line 1275) |
| 状态 | ✅ 已修复（**但引入新死循环，见 4.3**） |

---

## 四.3 第四轮审计采纳的关键修复（第七版）

> 第四轮独立审计识别出 4.2 修复引发的回归（主循环死循环）和遗留的 REJECT 死循环。**这两个 bug 在生产环境会导致进程永久卡死**，必须修复。

### 4.3.1 引入 `PERMANENTLY_FAILED` 终态解决主循环死循环

| 项目 | 内容 |
|------|------|
| 评审意见 | 4.2 把 `delete_tasks` 改为 `batch_update_state(FAILED)` 后，失败任务停留在 FAILED 状态，每轮循环被 `get_tasks_by_state(FAILED)` 取出 → `_process_failed_batch` 再次 `batch_update_state(FAILED)` → `any_progress=True` → 永久循环，流水线永不退出 |
| 修复方式 | 引入 `PERMANENTLY_FAILED` 终态：`VALID_TRANSITIONS[PERMANENTLY_FAILED] = []`（不可转移），不在 `pipeline_stages` 中处理。`_process_failed_batch` 中 `retries >= 3` 的任务改为 `batch_update_state(..., TaskState.PERMANENTLY_FAILED, ...)`。`FAILED` 状态现在仅用于"等待重试"（`retries < 3`） |
| 位置 | `TaskState` (line 56) / `VALID_TRANSITIONS` (line 64) / `TranslationPipeline._process_failed_batch` (line 1277) |
| 状态 | ✅ 已修复 |

### 4.3.2 `_process_judging_batch` REJECT 阻断机制

| 项目 | 内容 |
|------|------|
| 评审意见 | REJECT 时调用 `update_task_state(REWRITING_LITERARY, error_msg=...)`，`retries + 1`，但未检查 `retries >= 3`。如果文本始终无法通过，`retries` 达到 100 仍会在 `REWRITING_LITERARY → AUDITING → JUDGING` 之间循环 |
| 修复方式 | REJECT 分支先读 `retries`，如果 `>= 3` 直接 `update_task_state(PERMANENTLY_FAILED, ...)` 转入终态，否则保持原 `REWRITING_LITERARY` |
| 位置 | `TranslationPipeline._process_judging_batch` (line 1362) |
| 状态 | ✅ 已修复 |

### 4.3.3 `TaskState.FAILED` 自转移移除

| 项目 | 内容 |
|------|------|
| 评审意见 | `VALID_TRANSITIONS[FAILED] = [EXTRACTING_TERMS, FAILED]` 允许 `FAILED → FAILED` 自转移，无意义 |
| 修复方式 | `FAILED → FAILED` 移除，改为 `FAILED → PERMANENTLY_FAILED` |
| 位置 | `VALID_TRANSITIONS` (line 64) |
| 状态 | ✅ 已修复 |

---

## 四.4 第五轮审计采纳的高 ROI 修复（第八版）

> 基于 `docs/audit translator agent.md` 全量审计报告（20 项），第一轮采纳 7 项低成本高 ROI 修复：

### 4.4.1 C2：`_trigger_backtrack` 失败不再静默吞掉

| 项目 | 内容 |
|------|------|
| 评审意见 | `_trigger_backtrack` 的 `except Exception` 只 `print`，调用方无法感知跨库不一致事件 |
| 修复方式 | 改为 `raise RuntimeError(f"回溯触发失败，Decision DB 与 Workflow DB 可能不一致: {e}") from e`，让上层能 catch |
| 位置 | `DecisionEngine._trigger_backtrack` (line 656) |
| 状态 | ✅ 已修复 |

### 4.4.2 H2：HTTP 429/503 加入重试白名单

| 项目 | 内容 |
|------|------|
| 评审意见 | `if 400 <= status < 500: raise RuntimeError` 不区分 429/503（限流/服务端临时错误），导致 LM Studio/vLLM 限流时立即失败 |
| 修复方式 | 改为 `if 400 <= status < 500 and status not in {429, 503}: raise RuntimeError`，让 raise_for_status() 抛出后被外层重试逻辑捕获 |
| 位置 | `OpenAICompatibleAdapter.generate` (line 262) |
| 状态 | ✅ 已修复 |

### 4.4.3 H4：内存检测移到模型加载前

| 项目 | 内容 |
|------|------|
| 评审意见 | 当前 `generate()` 先 `_load_model_if_needed` 再 `auto_unload_if_needed`，刚加载的模型立即被检测到内存压力，触发卸载-重载死循环前体 |
| 修复方式 | 改为先 `check_memory_pressure()`（仅当模型名不同且压力大时），卸载旧模型后再加载新模型 |
| 位置 | `MLXNativeAdapter.generate` (line 309) |
| 状态 | ✅ 已修复 |

### 4.4.4 H5：英文原文使用英文 POV/allusion 标记

| 项目 | 内容 |
|------|------|
| 评审意见 | `_infer_author_priority_ratio` 用中文 POV markers (`'我 '`、`'他 '`) 和中文典故 (`'济慈'`、`'莎士比亚'`) 分析英文原文，永远返回固定值 0.7~0.8 |
| 修复方式 | POV markers 改为 `[' I ', ' my ', ' we ', ' he ', ' she ', ' it ']`，allusion markers 改为 `['Keats', 'Shakespeare', 'Milton', 'Dante', 'Homer', 'Shelley', 'Byron']` |
| 位置 | `LiteraryRewriterAgent._infer_author_priority_ratio` (line 928-941) |
| 状态 | ✅ 已修复 |

### 4.4.5 M2：`init_chapter_tasks` 跳过时打印警告

| 项目 | 内容 |
|------|------|
| 评审意见 | `except IntegrityError: pass` 静默跳过已存在 chunk，用户修改源文件后重新 init 不会被告知 |
| 修复方式 | 统计 `skipped` 数，commit 后若 > 0 则打印警告，提示用户用 `--force` 重新初始化 |
| 位置 | `TaskScheduler.init_chapter_tasks` (line 434) |
| 状态 | ✅ 已修复 |

### 4.4.6 M6：`run_golden_test` 单块失败不影响其他块

| 项目 | 内容 |
|------|------|
| 评审意见 | `for i, chunk in enumerate(chunks)` 无 try/except，单块失败导致整个测试中止，已处理的结果丢失 |
| 修复方式 | 在循环体外包 `try/except Exception as e`，捕获后 `print` 警告并 `all_results.append({"chunk_id": ..., "error": ...})` 跳过该块 |
| 位置 | `run_golden_test` (line 1485) |
| 状态 | ✅ 已修复 |

### 4.4.7 H6：Pipeline 在无任务时检查

| 项目 | 内容 |
|------|------|
| 评审意见 | 用户忘记 `init` 直接运行 pipeline，DB 为空时第一轮循环立即退出并打印"全部处理完成"，误导用户 |
| 修复方式 | `run()` 开头查询 `get_all_tasks_by_chapter(self.chapter_id)`，若为空则打印错误并 `return` |
| 位置 | `TranslationPipeline.run` (line 1237) |
| 状态 | ✅ 已修复 |

---

## 四.5 Oracle 元审计补漏（第九版）

> Oracle 元审计发现第一轮修复有 4 处遗留问题，已补漏修复：

### 4.5.1 H5 补漏：emotion_words 改英文

| 项目 | 内容 |
|------|------|
| Oracle 反馈 | H5 修复时只改了 POV 和 allusion markers，遗漏了 `emotion_words` 仍是中文列表（`['痛', '悲', '怒', ...]`），对英文原文永远返回 0 |
| 修复方式 | 改为英文 emotion words（`[' pain', ' sorrow', ' anger', ...]`），与 POV 修复保持一致 |
| 位置 | `LiteraryRewriterAgent._infer_author_priority_ratio` |
| 状态 | ✅ 已修复 |

### 4.5.2 H2 补漏：白名单纠正

| 项目 | 内容 |
|------|------|
| Oracle 反馈 | 原 `{429, 503}` 白名单中 503 是 5xx 不会进入 `400 <= status < 500` 分支（误导），502 也应加入 |
| 修复方式 | 改为单元素白名单 `{429}`，5xx 由 `raise_for_status()` 抛出后被外层重试逻辑捕获 |
| 位置 | `OpenAICompatibleAdapter.generate` |
| 状态 | ✅ 已修复 |

### 4.5.3 M2 补漏：`force` 参数实现

| 项目 | 内容 |
|------|------|
| Oracle 反馈 | M2 警告信息提到 `--force` 但 `init_chapter_tasks` 无 `force` 参数，警告指向不存在的特性 |
| 修复方式 | 新增 `force: bool = False` 参数；`force=True` 时使用 `INSERT ... ON CONFLICT(chunk_id) DO UPDATE SET text_content = excluded.text_content, state = ?, last_error = NULL`，并打印覆盖更新数量 |
| 位置 | `TaskScheduler.init_chapter_tasks` |
| 状态 | ✅ 已修复 |

### 4.5.4 M6 补漏：summary 计算过滤错误条目

| 项目 | 内容 |
|------|------|
| Oracle 反馈 | M6 引入的 `try/except` 让错误块以 `{"chunk_id": ..., "error": str(e)}` 形式追加，但后续 summary 计算 `r["judge_decision"]` 会 KeyError 崩溃 |
| 修复方式 | summary 计算前 `valid_results = [r for r in all_results if "error" not in r]`，聚合指标仅在 valid_results 上计算；分母用 `max(len(valid_results), 1)` 避免零除 |
| 位置 | `run_golden_test` summary 段 |
| 状态 | ✅ 已修复（**但维度汇总段仍 KeyError，见 5.1**） |

---

## 四.6 第二批 A 修复（第十版）

> 基于 `docs/audit translator agent.md` 完整审计报告（验证第一轮修复 + 发现新 bug），本批修复 6 个高 ROI 问题（~21 行）：

### 4.6.1 E1 + H1：M2 `force` 模式完整重置（含 retries）

| 项目 | 内容 |
|------|------|
| 审计报告 | NB2 (retries 未重置) + NB1 (overwritten 计数器永远等于总 chunk 数) |
| 严重级 | 🔴 阻塞 force 重置语义 |
| 问题 | `force=True` 只重置 `text_content/state/last_error`，漏了 `retries` 列。PERMANENTLY_FAILED chunk 改完源文件后跑 pipeline 立即再次 PERMANENTLY_FAILED。`cursor.rowcount` 在 `ON CONFLICT DO UPDATE` 下总是 1，无法区分 INSERT/UPDATE |
| 修复方式 | (1) SQL 增 `retries = 0`；(2) 用执行前 `SELECT COUNT(*)` + 执行后 `COUNT(*)` 差值计算真正被覆盖的 chunk 数 |
| 位置 | `TaskScheduler.init_chapter_tasks` (line 434) |
| 状态 | ✅ 已修复 |

### 4.6.2 E2：M6 dimension summary KeyError

| 项目 | 内容 |
|------|------|
| 审计报告 | NB3 |
| 严重级 | 🔴 单块失败仍崩溃 |
| 问题 | 第一轮 M6 修复只过滤了 `judge_decision` 的 KeyError，同循环下方 `r["preservation_details"]` 仍是直接下标访问 |
| 修复方式 | 改为 `r.get("preservation_details", {}).get(dim, 0)`，分母用 `max(len(valid_results), 1)` |
| 位置 | `run_golden_test` 维度汇总段 (line 1585) |
| 状态 | ✅ 已修复 |

### 4.6.3 E3：H5 emotion_words 去除重复 `' joy'`

| 项目 | 内容 |
|------|------|
| 审计报告 | H5 残留 1 |
| 严重级 | 🟡 简单 typo |
| 问题 | `emotion_words` 中 `' joy'` 出现两次，导致 `emotion_count` 对 "joy" 计双倍 |
| 修复方式 | 删除重复 |
| 位置 | `LiteraryRewriterAgent._infer_author_priority_ratio` (line 970) |
| 状态 | ✅ 已修复 |

### 4.6.4 E4：H5 allusion_markers 移除对话引号

| 项目 | 内容 |
|------|------|
| 审计报告 | H5 残留 2 |
| 严重级 | 🟡 误判风险 |
| 问题 | `'"'` 和 `'"'` 是普通对话引号，对话密集章节误判为典故密集 |
| 修复方式 | 从 `allusion_markers` 移除两个弯引号 |
| 位置 | `LiteraryRewriterAgent._infer_author_priority_ratio` (line 961) |
| 状态 | ✅ 已修复 |

### 4.6.5 H2：CLI `--force` 入口

| 项目 | 内容 |
|------|------|
| 审计报告 | 功能盲区 |
| 严重级 | 🟠 用户无法触发 force 重置 |
| 问题 | `init_chapter_tasks(force=True)` 已实现，但 `init_project` 不传 force，`main()` argparse 没有 `--force` 参数 |
| 修复方式 | (1) `init_project(chapter_id, force=False)` 接受参数；(2) `init_project.py:initialize_translation_project(chapter_id, force=False)` 同步；(3) argparse 增 `--force` flag |
| 位置 | `translator_agent.py:init_project + main` / `init_project.py:initialize_translation_project` |
| 状态 | ✅ 已修复 |

---

## 五、仍未采纳的审计意见

### 5.1 `get_llm_client` 单例竞态

| 项目 | 内容 |
|------|------|
| 评审意见 | 多线程并发创建多个 `MLXNativeAdapter` 导致内存爆炸 |
| 不采纳理由 | ⚠️ **理论风险，当前单线程**。`get_llm_client()` 在 `__init__` 链中顺序调用一次。评审承认这"为未来并行化埋下隐患"但确认当前顺序执行能工作。属于已知技术债 |

### 5.2 `GoldenSetEvaluator` 多个 `DecisionEngine` 实例（**部分采纳，见 4.1**）

| 项目 | 内容 |
|------|------|
| 评审意见 | 3 个独立 `DecisionEngine()` 实例连接冗余，为并行化埋下隐患 |
| 不采纳理由 | ⚠️ **顺序执行能工作**。但通过修复 4.1（统一 db 实例），数据孤岛已部分解决。仍属代码气味 |

### 5.3 流水线轮询效率

| 项目 | 内容 |
|------|------|
| 评审意见 | mock 后端毫秒级响应时 3.5s 空转延迟不可接受 |
| 不采纳理由 | ⚠️ **设计权衡**。真实 LLM 后端每块处理几秒到几十秒，3.5s 相对可忽略。改事件驱动增加架构复杂度 |

### 5.4 `SmartChunker` 对话探针边界（嵌套引号）

| 项目 | 内容 |
|------|------|
| 评审意见 | 单引号 `'` 不检测，中文直角引号 `「」` 未支持 |
| 不采纳理由 | ⚠️ **极罕见场景**。目标语料（英美科幻/奇幻文学中译）嵌套引号罕见，复杂状态机会增加维护成本 |

### 5.5 `JudgeAgent` 硬编码阈值

| 项目 | 内容 |
|------|------|
| 评审意见 | 硬编码 `7.0`/`7.5` 应引用 `CRITIC_THRESHOLDS` |
| 不采纳理由 | 🟢 **代码气味**，不是 bug |

### 5.6 模型名称硬编码分散

| 项目 | 内容 |
|------|------|
| 评审意见 | 5 处硬编码 `"qwen/Qwen2.5-7B-Instruct-MLX-4bit"` |
| 不采纳理由 | 🟢 **代码气味**，属于后续重构 |

### 5.7 `init_project` 中 `SmartChunker` 参数

| 项目 | 内容 |
|------|------|
| 评审意见 | 只传 `soft_limit=1000`，`hard_limit` 用默认值 2500 |
| 不采纳理由 | 🟢 **类默认值合理**，设计意图 |

### 5.8 `debug_db` 非线程安全

| 项目 | 内容 |
|------|------|
| 评审意见 | `debug_db` 使用普通 `sqlite3.connect()` |
| 不采纳理由 | 🟢 **调试工具**，不在生产热路径 |

### 5.9 决策回溯跳过 EXTRACTING_TERMS（第三轮审计）

| 项目 | 内容 |
|------|------|
| 评审意见 | DIRTY → EXTRACTING_TERMS 每次回溯都重跑考据，浪费 LLM 算力 |
| 不采纳理由 | ⚠️ **效率优化建议**，不是 bug。改 state 追踪会增加复杂度 |

### 5.10 MLX 内存管理（第三轮审计）

| 项目 | 内容 |
|------|------|
| 评审意见 | load-unload-load 循环在持续内存压力下无效 |
| 不采纳理由 | ⚠️ **MVP 阶段合理**。修复前会崩溃（None），修复后震荡但不崩。生产化时建议改为先检查内存再加载 |

---

## 六、功能完整性验证

`translator_agent.py` 仍是**完整功能**的脚本：

- 17 个类全部保留（`TaskState`、`DecisionLevel`、`LLMAdapter`、`MockLLMAdapter`、`OpenAICompatibleAdapter`、`MLXNativeAdapter`、`TaskScheduler`、`DecisionEngine`、`SmartChunker`、`ReferenceAgent`、`LiteraryRewriterAgent`、`CriticAgent`、`JudgeAgent`、`TranslationPipeline`、`GoldenSetEvaluator` 等）
- 顶层辅助函数全部保留（`init_project`、`debug_db`、`get_system_memory`、`get_process_memory`、`simulate_memory_load`、`test_memory_pressure`、`run_golden_test`、`print_banner`、`main`）
- 入口 `python src/translator_agent.py` 正常，可执行 `--mode golden-test` / `--mode translate` / `--mode debug` 等子命令
- 通过 `py_compile` 语法检查
- 通过 `import translator_agent` 导入测试
- 所有 class 和 function 在 `dir(translator_agent)` 中可见

---

## 七、修改统计

| 类别 | 数量 |
|------|------|
| 已采纳修复（第一轮） | 20 项 |
| 已采纳修复（第二轮评审） | 5 项 |
| 已采纳修复（第三轮审计） | 2 项 |
| 已采纳修复（第四轮审计） | 3 项 |
| 已采纳修复（第五轮审计） | 7 项 |
| 已采纳修复（Oracle 元审计补漏） | 4 项 |
| 已采纳修复（第二批 A：审计员发现的 6 个 bug） | 6 项 |
| **累计已采纳修复** | **47 项** |
| 仍未采纳评审意见 | 13 项 |
| 死代码/死配置清理 | 6 项 |
| 新增方法 | 2 个（`TaskScheduler.delete_tasks`、`DecisionEngine._cleanup_orphan_impacts`） |
| 删除方法/类 | 1 个（`PerformanceMetrics` dataclass） |
| 新增状态 | 1 个（`TaskState.PERMANENTLY_FAILED`） |
| 新增状态转移 | 1 个（`FAILED → PERMANENTLY_FAILED`） |
| 当前总行数 | 1788 行 |
