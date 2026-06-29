# 第三轮全量审计报告

> 版本：2453行（较上版 +649行，新增 Section 11-12 输入适配器）  
> 审计重点：配置系统迁移正确性、新增 Splitter 模块、残留问题
> 审计日期：2026-06-25

---

## 执行摘要

| 类别 | 数量 |
|------|------|
| 🔴 新增 Critical | 1 |
| 🟠 新增 High | 2 |
| 🟡 新增 Medium | 4 |
| ✅ 本轮已修复确认 | 13 项 |
| 📌 仍未修复（与历次声明一致） | 5 项 |

---

## 一、已修复确认（13项）

本轮对比验证，以下问题确认修复：

| ID | 内容 |
|----|------|
| C2 | `_trigger_backtrack` 不再静默吞异常 |
| H2 | HTTP 429 进入重试逻辑 |
| H4 | 内存检测移至模型加载前 |
| H5 | 英文 markers 修正（去除重复 joy、去除弯引号） |
| H6 | 空 DB 启动检查 |
| M1 | DecisionEngine 补充 `row_factory = sqlite3.Row` |
| M2 | force 模式重置 retries + 正确 overwritten 计数 |
| M4 | Token 估算改为 `tokenizer.encode()`，fallback 字符数 |
| M6/NB3 | dimension summary 使用 `valid_results` + `.get()` 防 KeyError |
| NB4 | `--force` CLI 参数已正确暴露 |
| L1 | WAL 模式已启用（两个 DB 均已添加） |
| C3 | 模型名通过 `_config.resolve_task_model()` 路由，不再硬编码 |
| banner | `print_banner` 输出改为 stderr，不污染 stdout |

---

## 二、新增 Critical

---

### CI1：模块顶层 `from src.config import get_config` 破坏单文件部署

**位置：** 文件第 43 行

**问题：**
```python
from src.config import get_config
_config = get_config()
```

文件头部声明"单文件部署"（monolith 聚合版），但顶层 import 引入了外部模块 `src.config`。在以下场景均会在 **import 阶段** 立即崩溃：

- `python translator_agent.py`（直接运行）
- `python -m translator_agent`（无 src/ 目录时）
- 任何 `import translator_agent` 操作

崩溃信息为 `ModuleNotFoundError: No module named 'src'`，与单文件审计/部署的设计目标完全矛盾。

此外，`_config = get_config()` 在模块载入时即执行，意味着配置文件（YAML/JSON）必须在 import 阶段就已存在且格式正确，否则整个模块不可用。所有 Agent、Scheduler、Pipeline 的实例化都依赖 `_config`，失败无法降级。

**修复建议：**

方案 A（推荐）：在单体聚合版中内联 `get_config()` 的最小实现，使其可独立运行：

```python
def _load_config_safe():
    """单体版配置加载：优先读取 config.yaml，失败时使用内建默认值"""
    try:
        from src.config import get_config
        return get_config()
    except ImportError:
        return _DefaultConfig()  # 内建默认配置对象

_config = _load_config_safe()
```

方案 B：将 `SYS_CONFIG` dict 恢复为内建默认值，`src.config` 作为可选覆盖层。

---

## 三、新增 High

---

### H-New1：`ReferenceAgent` 传 `model_key` 而非 `model_name`，与其他 Agent 行为不一致

**位置：** `ReferenceAgent.process_chunk`（~L640）

**问题：**
```python
# ReferenceAgent — 直接传 model_key
model_key, params = _config.resolve_task_model("reference_extraction")
raw_output = self.llm.generate(
    prompt=prompt,
    model_name=model_key,   # ← model_key 可能是配置键名，非真实模型路径
    **params
)

# LiteraryRewriterAgent — 做额外的 model_cfg 查找
model_key, params = _config.resolve_task_model("literary_rewrite")
model_cfg = _config._get_model_config(model_key)
model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "qwen/...")
final_markdown = self.llm.generate(
    prompt=prompt,
    model_name=model_name,  # ← 真实模型路径
    **params
)
```

`CriticAgent`、`JudgeAgent`、`_run_raw_translator` 均采用后者（额外查找 `_get_model_config`），只有 `ReferenceAgent` 直接用 `model_key`。

若 `resolve_task_model` 返回的 `model_key` 是配置键名（如 `"reference_model"`）而非实际路径（如 `"qwen/Qwen2.5-7B-Instruct-MLX-4bit"`），则 `ReferenceAgent` 会向 LLM API 传入错误的 model_name，导致 404 或模型加载失败。

**修复：** 将 `ReferenceAgent` 改为与其他 Agent 一致的两步查找，或让 `resolve_task_model` 直接返回最终 model_name（同时删除外部对 `_get_model_config` 的调用）。

---

### H-New2：`_process_judging_batch` 重试上限仍硬编码 3，与配置化的 `_process_failed_batch` 不一致

**位置：** `_process_judging_batch`（~L1370）vs `_process_failed_batch`（~L1310）

**问题：**
```python
# _process_failed_batch — 从配置读取
max_retries = _config.pipeline.get("max_retries", 3)
if retries >= max_retries:
    permanent_fail_ids.append(chunk_id)

# _process_judging_batch — 仍然硬编码
elif retries >= 3:
    self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, ...)
```

若 `config.yaml` 中设置 `pipeline.max_retries: 5`，则：
- 普通失败（异常路径）允许 5 次重试后进入 PERMANENTLY_FAILED
- Judge REJECT 路径仍在 3 次后进入 PERMANENTLY_FAILED

同一任务的两条终止路径使用不同的计数阈值，导致行为不一致，难以通过配置统一控制。

**修复：**
```python
max_retries = _config.pipeline.get("max_retries", 3)
elif retries >= max_retries:
    self.scheduler.update_task_state(chunk_id, TaskState.PERMANENTLY_FAILED, ...)
```

---

## 四、新增 Medium（Section 11-12 Splitter 专项）

---

### M-New1：`split_input_to_chapters` 中的 load 条件逻辑颠倒，为无效代码

**位置：** `split_input_to_chapters`（~L2380）

**问题：**
```python
if hasattr(splitter, 'book') or input_format == 'epub':
    splitter.load() if input_format != 'epub' else splitter
```

逐行分析：
- 条件为 True 时（仅 epub 进入），`if input_format != 'epub' else splitter` 求值为 `splitter`（无操作表达式）
- epub 格式永远不会在这里调用 `load()`
- txt/md 格式因 `hasattr` 为 False 且 `input_format == 'epub'` 为 False，完全跳过此块

整个 if 块从不产生任何效果。epub 的实际加载依赖 `EpubSplitter.split()` 内的懒加载：
```python
def split(self) -> list:
    if self.book is None:
        self.load()  ← 真正的加载发生在这里
```

**修复：** 删除整个 if 块（懒加载已覆盖），或修正为：
```python
if input_format == 'epub':
    splitter.load()
```

---

### M-New2：四处业务代码调用 `_config._get_model_config()`（私有方法）

**位置：** `LiteraryRewriterAgent`、`CriticAgent`、`JudgeAgent`、`_run_raw_translator`

**问题：**
```python
model_cfg = _config._get_model_config(model_key)  # 下划线前缀=私有
model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "...")
```

Python 惯例下，`_get_model_config` 是不应被外部调用的私有方法。四处业务代码绕过公共 API 直接调用，表明 `resolve_task_model()` 的返回值设计尚未稳定。

这是 `src/config.py` API 设计不完整的信号——`resolve_task_model` 应直接返回 `(model_name_str, params)` 而非 `(model_key, params)`，避免调用方再查一次。当 `config.py` 的内部结构调整时，四处调用点都要同步修改，维护成本高。

**修复建议：** 让 `resolve_task_model` 直接返回最终可用的 model_name：
```python
# config.py 修改后
def resolve_task_model(self, task_name: str) -> tuple[str, dict]:
    model_key = self.task_model_map[task_name]
    model_cfg = self._get_model_config(model_key)
    model_name = model_cfg.get("model_id") or model_cfg.get("model_name")
    params = {...}
    return model_name, params  # 直接返回可用的 model_name
```

---

### M-New3：`MdSplitter` 以虚构路径实例化 `TextSplitter` 用于代码复用，是架构异味

**位置：** `MdSplitter.split`（~L2320）

**问题：**
```python
fake_path = self.md_path.with_suffix('.txt')  # 可能不存在的路径
fake_text = TextSplitter(
    str(fake_path), str(self.output_dir),
    target_chars=self.target_chars, min_chars=self.min_chars
)
return fake_text._split_by_size(text)  # 还在调用私有方法
```

`TextSplitter` 被以虚构的 `.txt` 路径实例化，目的仅是调用其 `_split_by_size` 私有方法。这有两个问题：

1. 语义混乱：创建一个指向不存在文件的 `TextSplitter` 对象，容易误导维护者
2. 依赖私有方法：与 M-New2 同样问题，`_split_by_size` 属于实现细节

**修复：** 将 `_split_by_size` 提取为模块级函数，供 `TextSplitter` 和 `MdSplitter` 共同调用：
```python
def _split_text_by_size(text: str, target_chars: int, min_chars: int) -> list:
    ...
```

---

### M-New4：`_epub_split_large_doc_by_toc` 仅匹配双引号属性，EPUB 单引号属性失效

**位置：** `_epub_split_large_doc_by_toc`（~L1840）

**问题：**
```python
if f'id="{anchor_id}"' in line or f'name="{anchor_id}"' in line:
```

仅检测双引号格式（`id="anchor"`），不检测单引号格式（`id='anchor'`）或无引号格式（HTML5 允许）。部分 EPUB 生成工具（如 Calibre 导出的旧版 EPUB、基于 EPUB 2.0 标准的文件）使用单引号属性。此时大文档按 TOC 切分静默失败，回退到整章处理（与不切分相同）。

**修复：**
```python
import re as _re_html
anchor_pattern = _re_html.compile(
    r'''(?:id|name)\s*=\s*['"]?''' + _re_html.escape(anchor_id) + r'''['"]?'''
)
if anchor_pattern.search(line):
```

---

## 五、仍未修复项（与历次声明一致）

| ID | 内容 | 说明 |
|----|------|------|
| C1 | `trigger_backtrack` 仅标记 COMPLETED 块，in-flight 状态漏标 | SQL 仍有 `AND state = 'COMPLETED'` |
| H1 | Judge retries 与异常 retries 共用同一计数器 | 无 `judge_retries` 独立列 |
| H3 | MLX 每 chunk 触发 Gemma↔Qwen 换模 | pipeline_stages 顺序不变 |
| M3 | LLM 单例创建非线程安全 | 无 Lock 保护 |
| M5 | `init_project` 中 3 行 `DEBUG:` 打印仍在生产代码 | 未删除 |

---

## 六、待处理优先级清单（第三批）

| 优先级 | ID | 内容 | 工作量 |
|--------|-----|------|--------|
| 🔴 P0 | CI1 | `from src.config import get_config` 顶层 import 破坏单文件运行 | 10行（降级加载策略） |
| 🟠 P1 | H-New1 | ReferenceAgent 传 model_key 而非 model_name | 3行 |
| 🟠 P1 | H-New2 | `_process_judging_batch` max_retries 从配置读取 | 2行 |
| 🟡 P2 | M-New1 | 删除 `split_input_to_chapters` 中的死代码 load 条件 | 2行 |
| 🟡 P2 | M-New2 | `resolve_task_model` 直接返回 model_name，删除外部 `_get_model_config` 调用 | config.py 改动 |
| 🟡 P2 | M-New3 | 提取 `_split_by_size` 为模块级函数 | 5行 |
| 🟡 P2 | M-New4 | EPUB 单引号属性支持 | 5行 |
| 🟡 P2 | M5 | 删除 `init_project` DEBUG 打印 | 3行 |
| 🟠 P1 | C1 | trigger_backtrack 覆盖 in-flight 状态 | 5行（历史遗留） |
