# 第一批改进代码审计报告

> 审计范围：验证 7 项已声明修复 + 新引入问题扫描  
> 基准文件：`first-batch-improvements.md`  
> 审计日期：2026-06-25

---

## 执行摘要

| 类别 | 数量 |
|------|------|
| ✅ 验证正确 | 5 项 |
| ⚠️ 部分正确（有残留/新 bug） | 2 项 |
| 🔴 修复引入的新 bug | 3 项 |
| 📌 确认未修复（与声明一致） | 9 项 |

---

## 一、逐项验证

---

### ✅ C2 验证通过

**代码（~L693）：**
```python
def _trigger_backtrack(self, chunk_ids: List[str]):
    if not self._scheduler_factory:
        return
    try:
        scheduler = self._scheduler_factory()
        scheduler.trigger_backtrack(chunk_ids)
    except Exception as e:
        raise RuntimeError(f"回溯触发失败，Decision DB 与 Workflow DB 可能不一致: {e}") from e
```

正确抛出，不再静默。外层 `add_decision` 的 `except Exception` 会捕获并 re-raise，调用方可感知。

**遗留注意：** 外层 `except` 的错误信息是 `"❌ 决策写入失败"`，但实际上决策写入已成功（commit 在先），是回溯失败。措辞误导，建议改为 `"❌ 回溯触发失败（决策已落库）"`。这不是本次修复的范围，但需要记录。

---

### ✅ H2 验证通过

**代码（~L263）：**
```python
if 400 <= response.status_code < 500 and response.status_code != 429:
    raise RuntimeError(f"API 客户端错误 {response.status_code}: {response.text[:200]}")
response.raise_for_status()
```

429 现在落入 `raise_for_status()` → 抛出 `requests.HTTPError` → 被 `except Exception as e` 捕获 → 进入重试逻辑。503/502 等 5xx 本就走 `raise_for_status`，不受影响。改进报告中的自我纠错（移除 503）也已正确实施。

---

### ✅ H4 验证通过

**代码（~L312）：**
```python
def generate(self, ...):
    if self.current_model_name and self.current_model_name != model_name:
        if self.check_memory_pressure():
            self.unload_model()
    self._load_model_if_needed(model_name)
```

仅在**切换模型**且**有内存压力**时才先卸载，加载后不再检查。原来的无效加载-卸载-重载循环已消除。旧的 `auto_unload_if_needed` 调用已移除（该方法成为冗余代码，建议后续清理，但不影响正确性）。

---

### ✅ H6 验证通过

**代码（~L1265）：**
```python
existing_tasks = self.scheduler.get_all_tasks_by_chapter(self.chapter_id)
if not existing_tasks:
    print(f"❌ [Pipeline] 章节 {self.chapter_id} 无任务，请先运行 init 命令。")
    return
```

空 DB 时正确提前退出，不再误报"完成"。

---

### ✅ H5 主体验证通过，存在两处残留

**已修复的部分：**
- POV markers → 英文代词 ✅
- emotion_words → 英文情感词 ✅
- allusion_markers → 英文作者名 ✅
- poetry_markers → 英文短语 ✅

**残留问题 1：`emotion_words` 中 `' joy'` 出现两次**

```python
emotion_words = [' pain', ' sorrow', ' anger', ' joy', ' love', ' hate',
                 ' fear', ' despair', ' hope', ' dream', ' soul', ' grief',
                 ' rage', ' joy']  # ← ' joy' 重复
```
重复条目导致 `emotion_count` 对 "joy" 计双倍，使 `emotion_density` 轻微偏高，影响 `author_priority_ratio` 计算准确性。

**残留问题 2：`allusion_markers` 仍包含弯引号 `'"'` 和 `'"'`**

```python
allusion_markers = ['"', '"', 'Keats', 'Shakespeare', ...]
```
英文小说中 `"` 和 `"` 是普通对话引号，在对话密集的章节中，这两个字符的计数会极高，导致 `allusion_density` 虚高，错误地将普通对白段落判定为"典故密集"，进而推高 `base_ratio`。建议移除这两个字符，或改为检测引文块（如 `>` blockquote 语法）。

---

### ⚠️ M2 部分正确，引入 2 个新 bug（见第二节）

---

### ⚠️ M6 部分正确，引入 1 个新 bug（见第二节）

---

## 二、修复引入的新 Bug

---

### NB1：M2 — `force` 模式 `overwritten` 计数器永远等于总 chunk 数

**位置：** `init_chapter_tasks`（~L434）

**问题代码：**
```python
if force:
    cursor.execute('''
        INSERT INTO chunk_tasks ...
        ON CONFLICT(chunk_id) DO UPDATE SET ...
    ''', ...)
    if cursor.rowcount > 0:
        overwritten += 1
```

SQLite 的 `ON CONFLICT DO UPDATE` 无论执行 INSERT（无冲突）还是 UPDATE（有冲突），`cursor.rowcount` 均返回 `1`。因此 `overwritten` 在 force 模式下恒等于 `len(chunks)`，包含新插入的 chunk，计数完全不准确。

**正确实现：**
使用 `cursor.execute("SELECT changes()")` 在 UPDATE 路径上是可靠的，但区分 INSERT/UPDATE 需要记录执行前的行数：

```python
# 执行 UPSERT 前记录已有 chunk 数
cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
count_before = cursor.fetchone()[0]
# ... 批量执行 UPSERT ...
cursor.execute('SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?', (chapter_id,))
count_after = cursor.fetchone()[0]
inserted = count_after - count_before
overwritten = len(chunks) - inserted
```

或者简化：统计 skipped（无冲突的新插入数），`overwritten = len(chunks) - skipped`。

---

### NB2：M2 — `force` 模式不重置 `retries`，PERMANENTLY_FAILED 任务无法复活

**位置：** `init_chapter_tasks`（~L436-L444）

**问题代码：**
```python
ON CONFLICT(chunk_id) DO UPDATE SET
    text_content = excluded.text_content,
    state = ?,
    last_error = NULL
```
`retries` 未被重置。场景：一个 chunk 经历 3 次失败进入 `PERMANENTLY_FAILED`（retries=3），用户修改源文件后以 `force=True` 重新 init，该 chunk 的 state 被重置为 `PENDING`，但 `retries` 仍为 3。

下次运行 pipeline 时：
1. `PENDING → EXTRACTING_TERMS`（正常）
2. 若任何阶段抛异常 → `FAILED`（retries 已经是 3）
3. `_process_failed_batch` 读到 `retries=3 >= 3` → 立即 `PERMANENTLY_FAILED`

force 重置的目的是让任务重新可执行，但 retries 未清零使这个目的形同虚设。

**修复：**
```python
ON CONFLICT(chunk_id) DO UPDATE SET
    text_content = excluded.text_content,
    state = ?,
    last_error = NULL,
    retries = 0   -- ← 必须重置
```

---

### NB3：M6 — dimension summary 仍会在错误条目上 KeyError 崩溃

**位置：** `run_golden_test` 汇总报告段（~L1560）

**问题代码：**
```python
dims = ["avg_sentence_length", "vocabulary_density", "rhetorical_density", "punctuation_density"]
for dim in dims:
    dim_avg = sum(r["preservation_details"].get(dim, 0) for r in all_results) / total_chunks
    #              ^^^^^^^^^^^^^^^^^^^^^^^^
    #  错误条目没有 "preservation_details" 键，KeyError 崩溃
```

改进报告中 Oracle 只指出了 `r["judge_decision"]` 的 KeyError，已用 `valid_results` 修复。但同一循环下方对 `r["preservation_details"]` 的访问使用的仍是 `all_results`（包含错误条目），且直接用下标 `r["preservation_details"]` 而非 `r.get("preservation_details", {})`。一旦任何 chunk 处理失败，汇总报告阶段必定崩溃，之前的 try/except 修复实际上只是把崩溃从处理阶段推迟到了汇总阶段。

**修复：**
```python
for dim in dims:
    dim_avg = (
        sum(r.get("preservation_details", {}).get(dim, 0) for r in all_results)
        / max(len(valid_results), 1)
    )
    print(f"  {dim}: {dim_avg:.2%}")
```

---

## 三、功能盲区：`force` 参数无法从 CLI 触发

**位置：** `init_project`（~L1580）和 `main()`（~L1760）

`init_chapter_tasks` 新增了 `force: bool = False` 参数，但：

1. `init_project()` 调用处：`scheduler.init_chapter_tasks(chapter_id=chapter_id, chunks=chunks)` — 没有传 `force`，固定为 `False`
2. `main()` 的 `argparse`：没有 `--force` 参数

用户完全无法从命令行触发 force 重置，该功能作为死代码存在。

**修复（两处）：**
```python
# init_project 增加参数
def init_project(chapter_id: str = "ch01", force: bool = False):
    ...
    scheduler.init_chapter_tasks(chapter_id=chapter_id, chunks=chunks, force=force)

# argparse 增加参数
parser.add_argument("--force", action="store_true", help="强制重新初始化（覆盖已存在的 chunk）")
...
elif args.command == "init":
    init_project(chapter_id=args.chapter, force=args.force)
```

---

## 四、确认未修复项（与声明一致）

以下 9 项声明为"未采纳"，代码核查确认均未修复，与报告一致：

| ID | 问题 | 状态 |
|----|------|------|
| C1 | `trigger_backtrack` 仅标记 COMPLETED 块 | 未修复，SQL 仍有 `AND state = 'COMPLETED'` |
| C3 | 模型名硬编码 | 未修复，各 Agent 仍有 `"qwen/..."` 硬编码 |
| H1 | 重试预算跨阶段共享 | 未修复，仍用单一 `retries` 列 |
| H3 | MLX 每 chunk 触发 2 次换模 | 未修复，pipeline_stages 顺序不变 |
| M1 | DecisionEngine 缺 row_factory | 未修复 |
| M3 | LLM 单例非线程安全 | 未修复 |
| M4 | MLX token 统计对中文无意义 | 未修复，仍用 `len(response.split())` |
| M5 | DEBUG 前缀打印留在生产代码 | 未修复，`init_project` 中仍有 3 行 `DEBUG:` 打印 |
| L1-L5 | 低优先级技术债 | 全部未修复（声明一致） |

---

## 五、待处理清单（第二批）

综合本次审计发现，建议下一批按以下优先级处理：

| 优先级 | ID | 内容 | 工作量 |
|--------|-----|------|--------|
| 🔴 必须 | NB2 | `force` 模式不重置 `retries` | 1 行 SQL |
| 🔴 必须 | NB3 | dimension summary KeyError | 1 行 |
| 🟠 高 | NB1 | `overwritten` 计数器不准 | 5 行 |
| 🟠 高 | 功能盲区 | `--force` CLI 入口缺失 | 5 行 |
| 🟠 高 | H5残留 | `emotion_words` 重复 `' joy'` | 1 行 |
| 🟠 高 | H5残留 | allusion_markers 包含对话引号 | 1 行 |
| 🟠 高 | C1 | trigger_backtrack 覆盖 in-flight 块 | 5 行 |
| 🟠 高 | C3 | 模型名抽离到 SYS_CONFIG | 30+ 行 |
| 🟠 高 | H1 | 独立 judge_retries 计数器 | 15+ 行 + schema 迁移 |
