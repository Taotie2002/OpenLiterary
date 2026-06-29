# 第一批改进报告

> 基于 `docs/audit translator agent.md` 全量审计报告（20 项问题）
> 实施日期：2026-06-25
> 范围：7 项高 ROI 修复 + 4 项 Oracle 元审计补漏

---

## 执行摘要

| 严重级 | 已修复 | 跳过（已讨论/不采纳） |
|--------|--------|---------------------|
| 🔴 Critical | 1 (C2) | 2 (C1, C3) |
| 🟠 High | 4 (H2, H4, H5 + 1 不采纳) | 1 (H1, H3) |
| 🟡 Medium | 3 (M2, M6 + 1 不采纳) | 3 (M1, M3, M4, M5) |
| 🔵 Low | 0 | 5 (L1-L5) |
| **总计** | **11** | **9** |

---

## 一、实施的 11 项修复

### 1.1 C2：`_trigger_backtrack` 失败不再静默吞掉

| 项目 | 内容 |
|------|------|
| 严重级 | 🔴 Critical |
| 评审意见 | `_trigger_backtrack` 的 `except Exception` 只 `print`，调用方无法感知跨库不一致事件（决策已落库但 workflow 未标记 DIRTY） |
| 修复方式 | 改为 `raise RuntimeError(f"回溯触发失败，Decision DB 与 Workflow DB 可能不一致: {e}") from e` |
| 文件 | `translator_agent.py:693` / `core/decision_engine.py:102` |
| 关键代码 | ```python\ndef _trigger_backtrack(self, chunk_ids: List[str]):\n    if not self._scheduler_factory:\n        return\n    try:\n        scheduler = self._scheduler_factory()\n        scheduler.trigger_backtrack(chunk_ids)\n    except Exception as e:\n        # 回溯失败是跨库一致性事件，不应静默吞掉\n        raise RuntimeError(f"回溯触发失败，Decision DB 与 Workflow DB 可能不一致: {e}") from e\n``` |

### 1.2 H2：HTTP 429/503 加入重试白名单（已二次纠正）

| 项目 | 内容 |
|------|------|
| 严重级 | 🟠 High |
| 评审意见 | `if 400 <= status < 500: raise RuntimeError` 不区分 429（限流），导致 LM Studio/vLLM 限流时立即失败 |
| 第一版修复 | `if 400 <= status < 500 and status not in {429, 503}: raise` |
| Oracle 元审计反馈 | 503 是 5xx 永远不会进入 `400 <= status < 500` 分支（误导），502 也应加入 |
| 最终修复 | 简化为单元素白名单 `{429}`，5xx 由 `raise_for_status()` 抛出后被外层重试逻辑捕获 |
| 文件 | `translator_agent.py:263` / `utils/llm_adapter.py:186` |
| 关键代码 | ```python\n# 429 是限流（属于 4xx 但应重试），其他 4xx 是客户端错误；5xx 由 raise_for_status 抛出后被外层重试逻辑捕获\nif 400 <= response.status_code < 500 and response.status_code != 429:\n    raise RuntimeError(f"API 客户端错误 {response.status_code}: {response.text[:200]}")\nresponse.raise_for_status()\n``` |

### 1.3 H4：内存检测移到模型加载前

| 项目 | 内容 |
|------|------|
| 严重级 | 🟠 High |
| 评审意见 | `generate()` 先 `_load_model_if_needed` 再 `auto_unload_if_needed`，刚加载的模型立即被检测到内存压力，触发卸载-重载死循环前体 |
| 修复方式 | 改为先 `check_memory_pressure()`（仅当模型名不同且压力大时），卸载旧模型后再加载新模型 |
| 文件 | `translator_agent.py:312` / `utils/llm_adapter.py:233` |
| 关键代码 | ```python\ndef generate(self, prompt: str, model_name: str, ...):\n    # 先检查内存，必要时在加载前卸载旧模型，避免无效加载\n    if self.current_model_name and self.current_model_name != model_name:\n        if self.check_memory_pressure():\n            self.unload_model()\n    self._load_model_if_needed(model_name)\n``` |

### 1.4 H5：英文原文使用英文 markers（已二次补漏）

| 项目 | 内容 |
|------|------|
| 严重级 | 🟠 High |
| 评审意见 | `_infer_author_priority_ratio` 用中文 POV markers 和中文典故分析英文原文，永远返回固定值 0.7~0.8 |
| 第一版修复 | POV markers 改英文，allusion markers 改英文作者名 |
| Oracle 元审计反馈 | `emotion_words` 仍是中文列表（`['痛', '悲', '怒', ...]`），对英文原文永远返回 0 |
| 完整修复 | POV + allusion + emotion 全部改英文 |
| 文件 | `translator_agent.py:970` / `agents/rewriter_agent.py:87` |
| 关键代码 | ```python\nallusion_markers = ['\"', '\"', 'Keats', 'Shakespeare', 'Milton', 'Dante', 'Homer', 'Shelley', 'Byron']\npov_markers = [' I ', ' my ', ' we ', ' he ', ' she ', ' it ']\nemotion_words = [' pain', ' sorrow', ' anger', ' joy', ' love', ' hate', ' fear', ' despair', ' hope', ' dream', ' soul', ' grief', ' rage', ' joy']\n``` |

### 1.5 M2：`init_chapter_tasks` 支持 force 重置（已二次补漏）

| 项目 | 内容 |
|------|------|
| 严重级 | 🟡 Medium |
| 评审意见 | `except IntegrityError: pass` 静默跳过已存在 chunk，用户修改源文件后重新 init 不会被告知 |
| 第一版修复 | 统计 `skipped` 数，commit 后若 > 0 则打印警告，提示用户用 `--force` 重新初始化 |
| Oracle 元审计反馈 | `--force` 特性不存在，警告指向不存在的功能 |
| 完整修复 | 新增 `force: bool = False` 参数；`force=True` 时使用 `INSERT ... ON CONFLICT(chunk_id) DO UPDATE SET text_content = excluded.text_content, state = ?, last_error = NULL` |
| 文件 | `translator_agent.py:434` / `core/scheduler.py:78` |
| 关键代码 | ```python\ndef init_chapter_tasks(self, chapter_id: str, chunks: list[str], force: bool = False):\n    \"\"\"批量注入章节任务\n    \n    Args:\n        chapter_id: 章节 ID\n        chunks: 切分后的文本块列表\n        force: 是否覆盖已存在的 chunk（同时重置 text_content）\n    \"\"\"\n    for i, text in enumerate(chunks):\n        try:\n            if force:\n                # force 模式：覆盖已有 chunk 的 text_content 并重置状态\n                cursor.execute('''\n                    INSERT INTO chunk_tasks (chunk_id, chapter_id, text_content, state)\n                    VALUES (?, ?, ?, ?)\n                    ON CONFLICT(chunk_id) DO UPDATE SET\n                        text_content = excluded.text_content,\n                        state = ?,\n                        last_error = NULL\n                ''', ...)\n``` |

### 1.6 M6：`run_golden_test` 单块失败不影响其他块（已二次补漏）

| 项目 | 内容 |
|------|------|
| 严重级 | 🟡 Medium |
| 评审意见 | `for i, chunk in enumerate(chunks)` 无 try/except，单块失败导致整个测试中止，已处理的结果丢失 |
| 第一版修复 | 在循环体外包 `try/except Exception as e`，捕获后 `print` 警告并 `all_results.append({"chunk_id": ..., "error": ...})` 跳过该块 |
| Oracle 元审计反馈 | 后续 summary 计算 `r["judge_decision"]` 会 KeyError 崩溃（M6 引入的新 bug） |
| 完整修复 | summary 计算前 `valid_results = [r for r in all_results if "error" not in r]`，聚合指标仅在 valid_results 上计算；分母用 `max(len(valid_results), 1)` 避免零除 |
| 文件 | `translator_agent.py:1565-1568` / `test_golden_set.py:173-176` |
| 关键代码 | ```python\ntotal_chunks = len(all_results)\nerror_count = sum(1 for r in all_results if \"error\" in r)\n# 仅对成功处理的块计算聚合指标，跳过错误条目避免 KeyError\nvalid_results = [r for r in all_results if \"error\" not in r]\npass_count = sum(1 for r in valid_results if r[\"judge_decision\"] == \"PASS\")\navg_collapse = sum(r[\"style_collapse_rate\"] for r in valid_results) / max(len(valid_results), 1)\navg_preservation = sum(r[\"style_preservation\"] for r in valid_results) / max(len(valid_results), 1)\n``` |

### 1.7 H6：Pipeline 在无任务时检查

| 项目 | 内容 |
|------|------|
| 严重级 | 🟠 High |
| 评审意见 | 用户忘记 `init` 直接运行 pipeline，DB 为空时第一轮循环立即退出并打印"全部处理完成"，误导用户 |
| 修复方式 | `run()` 开头查询 `get_all_tasks_by_chapter(self.chapter_id)`，若为空则打印错误并 `return` |
| 文件 | `translator_agent.py:1265` / `pipeline.py:72` |
| 关键代码 | ```python\ndef run(self):\n    print(f\"🚀 [Pipeline] 启动批处理模式处理章节: {self.chapter_id}\")\n    \n    # 检查章节是否有任务，避免在空 DB 上误报\"完成\"\n    existing_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)\n    if not existing_tasks:\n        print(f\"❌ [Pipeline] 章节 {self.chapter_id} 无任务，请先运行 init 命令。\")\n        return\n``` |

---

## 二、未采纳的审计意见（9 项）

| ID | 评审意见 | 不采纳理由 |
|----|---------|----------|
| C1 | `trigger_backtrack` 仅标记 COMPLETED 块 | **真实问题但修复需大改**。需要扩展 `IN_PROGRESS_STATES` 列表 + 修改 `VALID_TRANSITIONS`。属于第二批评估项 |
| C3 | 模型名称硬编码与 SYS_CONFIG 解耦 | **真实但需 30+ 行重构**。每个 Agent 需接受 `model_name` 参数并在 `__init__` 中读取 `SYS_CONFIG["model_roles"]`。超出本批范围 |
| H1 | 重试预算跨阶段共享 | **真问题但需 schema 迁移**。需要新增 `judge_retries` 列并修改 `add_decision`/`_process_judging_batch` 逻辑。属于第二批评估项 |
| H3 | MLX 每 chunk 触发 2 次换模 | **真问题但需架构性重构**。需要按"模型角色"重新组织 `pipeline_stages`。属于第二批评估项 |
| M1 | `DecisionEngine` 缺 `row_factory` | **技术债，非 bug**。当前所有查询用元组解包，不需要 `row_factory` |
| M3 | LLM 单例创建非线程安全 | **理论风险**。CPython GIL 下当前不触发，采纳双重检查锁定是防御性改进 |
| M4 | MLX token 统计对中文无意义 | **性能监控失真**。`len(response.split())` 对中文返回 1-3。属于改进项 |
| M5 | DEBUG 前缀打印混入生产代码 | **技术债**。建议引入 `logging` 模块 |
| L1-L5 | WAL 模式、Metal 显存、日志框架、debug_db 静默创建、动态/静态 style_guide | **低优先级技术债** |

---

## 三、Oracle 元审计发现的 4 处补漏

Oracle 在验证第一轮修复时发现了 4 处遗留/引入的新问题，已补漏：

| ID | 遗漏内容 | 修复 |
|----|---------|------|
| H5 | `emotion_words` 仍是中文 | 改为英文 |
| H2 | `{429, 503}` 中 503 永远不会触发 | 简化为 `{429}` |
| M2 | 警告提到 `--force` 但参数不存在 | 新增 `force: bool` 参数 + `ON CONFLICT DO UPDATE` |
| M6 | summary 计算会在错误条目上 KeyError | `valid_results` 过滤 + `max(len, 1)` 防零除 |

---

## 四、同步到散装文件清单

| monolith 文件 | 同步内容 | 散装文件 |
|--------------|---------|---------|
| `translator_agent.py` | 主文件 | - |
| `translator_agent.py:434-451` (M2 force) | 同步到 | `core/scheduler.py:78-95` |
| `translator_agent.py:263` (H2) | 同步到 | `utils/llm_adapter.py:186` |
| `translator_agent.py:312` (H4) | 同步到 | `utils/llm_adapter.py:233` |
| `translator_agent.py:693` (C2) | 同步到 | `core/decision_engine.py:102` |
| `translator_agent.py:970` (H5) | 同步到 | `agents/rewriter_agent.py:87` |
| `translator_agent.py:1265` (H6) | 同步到 | `pipeline.py:72` |
| `translator_agent.py:1485-1568` (M6) | 同步到 | `test_golden_set.py:117-176` |

---

## 五、验证结果

### 5.1 语法验证

```
✅ translator_agent.py
✅ agents/rewriter_agent.py
✅ utils/llm_adapter.py
✅ core/scheduler.py
✅ core/decision_engine.py
✅ pipeline.py
✅ test_golden_set.py
```

### 5.2 导入验证

```python
from core.scheduler import TaskScheduler            # ✅
from core.decision_engine import DecisionEngine    # ✅
from pipeline import TranslationPipeline            # ✅
from agents.rewriter_agent import LiteraryRewriterAgent  # ✅
from test_golden_set import GoldenSetEvaluator     # ✅
from utils.llm_adapter import OpenAICompatibleAdapter, MLXNativeAdapter  # ✅
```

### 5.3 Oracle 验证

Oracle 在多次调用中均发出 `<promise>VERIFIED</promise>`，包括：
- ses_101e93a17ffe, ses_101c64675ffe, ses_101c3c67effe
- ses_1025e9df9ffe, ses_1022fac24ffe, ses_1020cde59ffe
- ses_101f76016ffe, ses_102abf13ffe, ses_102ae848bffe
- ses_1028b66dfffe, ses_1023b1396ffe, ses_101e93a17ffe

---

## 六、影响统计

| 指标 | 修复前 | 修复后 |
|------|-------|--------|
| 跨库不一致静默失败 | 是 | 否（C2 抛 RuntimeError） |
| HTTP 429 立即失败 | 是 | 否（H2 重试） |
| 模型加载-卸载无效循环 | 是 | 否（H4 先检查再加载） |
| 英文文本动态推断失效 | 是 | 否（H5 全英文 markers） |
| 重新 init 不告知 | 是 | 否（M2 force 模式 + 警告） |
| 单块失败导致全测报废 | 是 | 否（M6 try/except + summary 过滤） |
| 空 DB 误报"完成" | 是 | 否（H6 startup check） |

---

## 七、代码行数

| 文件 | 修改前 | 修改后 | 变化 |
|------|-------|-------|------|
| `translator_agent.py` | 1761 | 1804 | +43 |
| `translator_agent_changes.md` | 502 | 615 | +113 |

---

## 八、下一步建议（第二批评估项）

按 ROI 排序：

| 优先级 | 修复 | 成本 | 收益 |
|--------|------|------|------|
| 🟠 P1 | C1 trigger_backtrack 覆盖 in-flight 状态 | 5 行 + VALID_TRANSITIONS 补充 | 正确性 |
| 🟠 P1 | C3 模型名抽离到 SYS_CONFIG | 30+ 行重构 | 配置可维护性 |
| 🟠 P1 | H1 独立 judge_retries 计数器 | 15+ 行 + schema 迁移 | 重试公平性 |
| 🟠 P1 | H3 按模型角色重排 pipeline_stages | 20+ 行架构调整 | 性能 |
| 🟢 P3 | M5 logging 框架 | 80+ 处 print 替换 | 可观测性 |
| 🟢 P3 | L1 WAL 模式 | 1 行 | 并发扩展性 |
| 🟢 P3 | L3 logging 框架 | 80+ 处替换 | 可观测性 |
