# 第二批工作计划

> 触发：本工作计划由 `docs/audit translator agent.md` 触发，整合 Oracle 元审计、`first-batch-improvements.md` 自查、用户"下阶段工作方案"三个输入
> 日期：2026-06-25
> 优先级原则：**真 bug > 设计债务 > 代码气味**

---

## 一、紧急级（真 bug，必须立即修）

### 🔴 E1：M2 `force` 模式不重置 `retries`（NB2）

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 NB2 |
| 严重级 | 🔴 阻塞 force 重置语义 |
| 问题 | `force=True` 重置 `text_content/state/last_error` 但漏 `retries`。PERMANENTLY_FAILED chunk 改完源文件后跑 pipeline，立即再次 PERMANENTLY_FAILED |
| 修复 | SQL 增加 `retries = 0` |
| 工作量 | 1 行 |
| 文件 | `translator_agent.py:434` / `core/scheduler.py:78` |
| 状态 | ⏳ 待修复 |

### 🔴 E2：M6 dimension summary KeyError（NB3）

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 NB3 |
| 严重级 | 🔴 单块失败仍崩溃 |
| 问题 | 汇总报告对 `r["preservation_details"]` 直接下标访问，错误条目无此键，KeyError 崩溃。M6 修复只是把崩溃从处理阶段推迟到汇总阶段 |
| 修复 | `r.get("preservation_details", {})` + `max(len(valid_results), 1)` |
| 工作量 | 2 行 |
| 文件 | `translator_agent.py:1556` / `test_golden_set.py:178` |
| 状态 | ⏳ 待修复 |

### 🔴 E3：H5-1 重复 `' joy'`

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 H5 残留 1 |
| 严重级 | 🟡 简单 typo |
| 问题 | `emotion_words = [' pain', ..., ' joy', ..., ' joy']`（手抖） |
| 修复 | 删除重复 |
| 工作量 | 1 行 |
| 文件 | `translator_agent.py:970` / `agents/rewriter_agent.py:87` |
| 状态 | ⏳ 待修复 |

### 🔴 E4：H5-2 allusion_markers 包含对话引号

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 H5 残留 2 |
| 严重级 | 🟡 误判风险 |
| 问题 | 英文小说中 `"`/`"` 是普通对话引号，对话密集章节误判为典故密集，`base_ratio` 虚高 |
| 修复 | 从 `allusion_markers` 移除两个弯引号，改为检测真正的典故信号（如 blockquote `>`） |
| 工作量 | 1-2 行 |
| 文件 | `translator_agent.py:967` / `agents/rewriter_agent.py:70` |
| 状态 | ⏳ 待修复 |

---

## 二、高优先级（设计缺陷 + 功能盲区）

### 🟠 H1：M2 `overwritten` 计数器不准（NB1）

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 NB1 |
| 严重级 | 🟠 计数错（不影响功能但误导用户） |
| 问题 | SQLite `INSERT ... ON CONFLICT DO UPDATE` 的 `cursor.rowcount` 总是 1，无法区分 INSERT vs UPDATE |
| 修复方案 A | 记录执行前 COUNT，按差值计算 |
| 修复方案 B | 简化：`skipped = 新插入数`, `overwritten = len(chunks) - skipped` |
| 工作量 | 5 行 |
| 文件 | `translator_agent.py:434` / `core/scheduler.py:78` |
| 状态 | ⏳ 待修复 |

### 🟠 H2：CLI `--force` 入口缺失

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告功能盲区 |
| 严重级 | 🟠 用户无法触发 force 重置 |
| 问题 | `init_chapter_tasks(force=True)` 已实现，但 `init_project()` 不传 force，`main()` argparse 没有 `--force` |
| 修复 | `init_project(chapter_id, force=False)` + `argparse.add_argument("--force", action="store_true")` + main 分发 |
| 工作量 | 5-10 行 |
| 文件 | `translator_agent.py:init_project` + `main()` / `init_project.py` |
| 状态 | ⏳ 待修复 |

---

## 三、中优先级（核心架构问题，列于第二批评估项）

### 🟠 M1：C1 `trigger_backtrack` 覆盖 in-flight 块

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 + 用户前次方案 |
| 严重级 | 🟠 正确性 |
| 问题 | `WHERE state = 'COMPLETED'` 排除 `REWRITING_LITERARY/AUDITING/JUDGING` 块，新决策无法影响进行中翻译 |
| 修复 | `IN_PROGRESS_STATES = (COMPLETED, REWRITING_LITERARY, AUDITING, JUDGING)` + 同步补充 `VALID_TRANSITIONS` |
| 工作量 | 5 行 |
| 文件 | `translator_agent.py:trigger_backtrack` / `core/scheduler.py` |
| 状态 | ⏳ 待修复 |

### 🟠 M2：H1 独立 `judge_retries` 计数器

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 + 用户前次方案 |
| 严重级 | 🟠 公平性 |
| 问题 | `retries` 全局共享，AUDITING 异常偷走 Judge 重试预算 |
| 修复 | 新增 `judge_retries` 列 + `_process_judging_batch` 单独维护 |
| 工作量 | 15+ 行 + schema 迁移 |
| 文件 | `translator_agent.py` + `core/scheduler.py:_init_db` |
| 状态 | ⏳ 待修复 |

### 🟡 L1：C3 模型名抽离到 SYS_CONFIG

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 + 用户前次方案 |
| 严重级 | 🟡 配置可维护性 |
| 问题 | 5 处硬编码 `"qwen/Qwen2.5-7B-Instruct-MLX-4bit"`，切换 openai_api 后端时名称不兼容 |
| 修复 | `SYS_CONFIG["model_roles"]` 字典 + Agent `__init__` 接受 `model_name` 参数 |
| 工作量 | 30+ 行重构 |
| 文件 | `translator_agent.py` 各 Agent + `agents/*.py` |
| 状态 | ⏳ 待修复 |

---

## 四、低优先级（性能 + 设计债务）

### 🟢 P1：H3 按模型角色重排 pipeline_stages

| 项目 | 内容 |
|------|------|
| 触发 | 审计报告 + 用户前次方案 |
| 严重级 | 🟢 性能（每个 chunk 节省 30-90s） |
| 问题 | Reference(Qwen)→ Raw(Gemma)→ Rewriter(Qwen) 强制换模 2 次 |
| 修复 | 重新组织批处理：先所有 chunk 跑 Qwen 阶段，再所有 chunk 跑 Gemma 阶段，再 Qwen |
| 工作量 | 20+ 行架构调整 |
| 文件 | `translator_agent.py:pipeline_stages` / `pipeline.py` |
| 状态 | ⏳ 待评估成本 |

### 🟢 P2：M5 DEBUG 前缀打印

| 项目 | 内容 |
|------|------|
| 问题 | 3 行 `print(f"DEBUG: ...")` 在生产代码 |
| 修复 | 改用 `logging.debug()` |
| 工作量 | 3 行 |
| 状态 | ⏳ 待修复 |

### 🟢 P3：M4 MLX token 统计

| 项目 | 内容 |
|------|------|
| 问题 | `len(response.split())` 中文返回 1-3 |
| 修复 | `len(response)`（字符数）或用 tokenizer |
| 工作量 | 1 行 |
| 状态 | ⏳ 待修复 |

### 🟢 P4：M3 LLM 单例线程安全

| 项目 | 内容 |
|------|------|
| 问题 | 双重检查锁定（理论风险） |
| 工作量 | 5 行 |
| 状态 | ⏳ 待修复 |

---

## 五、不采纳/不修复项

| ID | 描述 | 不修复理由 |
|----|------|----------|
| M1 | `DecisionEngine` 缺 `row_factory` | 当前元组解包够用，row_factory 是技术债 |
| L1 | SQLite WAL 模式 | 当前单线程不需要 |
| L2 | MLX 内存压力检测使用 RSS | M1 平台特有问题，需换 Metal API |
| L3 | 全局 logging 框架 | 80+ 处 print 替换，scope 大 |
| L4 | `debug_db` 静默创建空库 | 调试工具 |
| L5 | 动态/静态 style_guide 未对齐 | 设计文档 |

---

## 六、执行计划（按 ROI 排序）

### 第二批执行顺序（高 ROI → 低 ROI）

| 步骤 | 任务 | 工作量 | 累计 |
|------|------|--------|------|
| 1 | E1: M2 force 重置 retries | 1 行 | 1 |
| 2 | E2: M6 summary KeyError 修复 | 2 行 | 3 |
| 3 | E3: H5-1 去除重复 ' joy' | 1 行 | 4 |
| 4 | E4: H5-2 移除对话引号 | 1-2 行 | 6 |
| 5 | H1: NB1 overwritten 计数器 | 5 行 | 11 |
| 6 | H2: CLI --force 入口 | 5-10 行 | 21 |
| 7 | M1: C1 in-flight 块回溯 | 5 行 | 26 |
| 8 | M2: H1 judge_retries 独立 | 15+ 行 | 41+ |
| 9 | L1: C3 模型名抽离 | 30+ 行 | 71+ |
| 10 | P1: H3 按模型角色重排 | 20+ 行 | 91+ |
| 11 | P2-P4: M5/M4/M3 | 10 行 | 101+ |

**总工作量**: ~100+ 行

### 拆分批次

**第二批 A (紧急 bug 修复)**：步骤 1-6 (~21 行)
- 解决第一轮修复引入的 5 个新 bug
- 加上 CLI 入口补全

**第二批 B (架构改进)**：步骤 7-9 (~50 行)
- C1 in-flight 回溯
- H1 独立 judge_retries
- C3 模型名抽离

**第二批 C (性能优化)**：步骤 10 (~20 行)
- H3 按模型角色重排

**第二批 D (技术债清理)**：步骤 11 (~10 行)
- M5/M4/M3 修复

---

## 七、关键风险

| 风险 | 缓解 |
|------|------|
| M2 force 重置 retries 可能误重置进行中任务的状态 | 仅在用户主动 `--force` 时重置，明确语义 |
| M1 in-flight 回溯可能打断正在生成的 LLM 调用 | 接受风险；下次 pipeline 轮询会重跑 |
| L1 模型名重构可能破坏现有调用方 | 保持默认值兼容，旧 API 仍工作 |
| C1 状态机扩展可能引入新的非法转移 | 同步更新 `VALID_TRANSITIONS`，保持一致性 |

---

## 八、文档同步

每批修复完成后需更新：
- `docs/translator_agent_changes.md`（追加 5.x 节）
- `docs/first-batch-improvements.md`（如必要）
- 本工作计划末尾追加完成时间

---

## 九、与审计报告的差异分析

### 我之前评估遗漏的关键问题

1. **NB1 (M2 rowcount 错误)**：我自己设计的 `overwritten += 1` 计数器实现就是错的
   - **教训**：使用 `ON CONFLICT DO UPDATE` 时必须用 `COUNT(*)` 差值法
2. **NB2 (retries 未重置)**：我只重置了可见字段，漏了 `retries` 列
   - **教训**：列级重置需要枚举所有"需要清零"的字段
3. **NB3 (KeyError 仍在)**：我只看了 `judge_decision` 漏了同一行下方的 `preservation_details`
   - **教训**：错误条目缺失的字段需要全文搜索
4. **H5-1 (' joy' 重复)**：纯手抖，自己没复检
   - **教训**：对生成的代码必须 `cat` 复检
5. **CLI --force 缺失**：我只改了库函数，没改 CLI 入口
   - **教训**：功能上线必须走完"CLI → 库函数"完整链路

### 审计员未识别但仍需评估的项

- L1 WAL 模式：当前单线程不需要，但**未来扩展性重要**
- L2 Metal 显存：仅 M1 平台相关，**优先级 P3**

---

## 十、立即可执行的下一步

**推荐先执行第二批 A（步骤 1-6）**：
- 工作量小（~21 行）
- 解决第一轮修复引入的全部新 bug
- 完成 CLI 入口闭环
- 总耗时预估：30 分钟
