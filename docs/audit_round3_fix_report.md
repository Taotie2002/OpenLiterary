# 第三轮审计修复报告

**日期**: 2026-06-29  
**项目**: OpenLiterary (translator-agent)  
**审计依据**: `docs/audit_round3.md`  
**Git 仓库**: /home/zyu/projects/translator

---

## 📋 提交记录

| Commit Hash | 信息 | 类型 |
|-------------|------|------|
| `b582878` | Initial commit: pre-audit baseline | 基线 |
| `8a995f0` | Fix: 5 audit issues (P0/P1/P2) | 修复 |

---

## ✅ 修复清单（5 项）

| ID | 等级 | 问题 | 修改文件 | 行数变化 |
|----|------|------|----------|----------|
| **CI1** | 🔴 P0 | 单文件部署被破坏 | `src/translator_agent.py:42-100` | +59/-4 |
| **H-New1** | 🟠 P1 | ReferenceAgent 传 model_key | `src/translator_agent.py:935-944` | +3/-1 |
| **H-New2** | 🟠 P1 | judging_batch 硬编码 3 | `src/translator_agent.py:1506-1520` | +2/-1 |
| **M-New1** | 🟡 P2 | split_input_to_chapters 死代码 | `src/translator_agent.py:2481-2493` | -5 |
| **M5** | 🟡 P2 | init_project DEBUG 打印 | `src/translator_agent.py:1846-1848` | -3 |

---

## 📝 详细修改说明

### 1. CI1 (🔴 P0) — 安全配置加载器

**问题**: 文件头部声明"单文件部署"，但 `from src.config import get_config` 在 import 阶段执行，导致单文件模式下 `ModuleNotFoundError: No module named 'src'`。

**修复位置**: `src/translator_agent.py:42-100`

**修复内容**:
```python
# 修复前（破坏单文件模式）
from src.config import get_config
_config = get_config()

# 修复后（双路径降级）
try:
    from src.config import get_config as _get_config_external
    _config = _get_config_external()
except ImportError:
    class _BuiltinConfig:
        # 内建默认配置，覆盖所有访问点
        llm_backend = "mock"
        task_routing = { ... }
        mlx_models = { ... }
        # ... 所有配置项 ...
        def resolve_task_model(self, task_name): ...
        def _get_model_config(self, model_key): ...
    _config = _BuiltinConfig()
    print("[WARN] 未找到 src.config，使用内建默认配置（单文件模式）", file=sys.stderr)
```

**关键点**:
- `try/except ImportError` 双路径降级
- 内建配置 `_BuiltinConfig` 完全复现 `_config` API（属性 + 方法）
- `_config` 对象 API 与外部配置完全兼容（属性访问 + 方法调用）

**顺带修复**: 删除 Section 2 内部残留的 `_config = get_config()`（line 151，早期重构遗留）。

---

### 2. H-New1 (🟠 P1) — ReferenceAgent 使用 model_name

**问题**: `ReferenceAgent` 直接用 `model_key` 作为 `model_name` 传给 LLM，而其他 4 处 Agent 都做两步查找（先拿 key 再查 model_cfg 再取 model_name/id）。导致 LLM API 收到配置键名（如 `reference_model`）而非真实路径（如 `qwen/Qwen2.5-7B-Instruct-MLX-4bit`），触发 404。

**修复位置**: `src/translator_agent.py:935-944`

```python
# 修复前
model_key, params = _config.resolve_task_model("reference_extraction")
raw_output = self.llm.generate(
    prompt=prompt,
    model_name=model_key,   # ← 直接用配置键名
    **params
)

# 修复后（与 LiteraryRewriterAgent/CriticAgent/JudgeAgent 一致）
model_key, params = _config.resolve_task_model("reference_extraction")
model_cfg = _config._get_model_config(model_key)
model_name = model_cfg.get("model_id") or model_cfg.get("model_name", "qwen/Qwen2.5-7B-Instruct-MLX-4bit")
raw_output = self.llm.generate(
    prompt=prompt,
    model_name=model_name,   # ← 真实模型路径
    **params
)
```

**验证**: `llm_backend: mlx` 下 `model_name` 正确解析为 `qwen/Qwen2.5-7B-Instruct-MLX-4bit`。

---

### 3. H-New2 — `_process_judging_batch` 硬编码 `retries >= 3`

**问题**: `_process_failed_batch` 已从配置读 `max_retries`，但 `_process_judging_batch` 仍硬编码 `retries >= 3`，导致 `config.yaml` 中 `pipeline.max_retries: 5` 对 Judge REJECT 路径不生效。

**修复位置**: `src/translator_agent.py:1506-1520`

```python
# 修复前
def _process_judging_batch(self, tasks):
    for task in tasks:
        ...
        elif retries >= 3:
            ...

# 修复后
def _process_judging_batch(self, tasks):
    max_retries = _config.pipeline.get("max_retries", 3)  # 从配置读取
    for task in tasks:
        ...
        elif retries >= max_retries:  # ← 使用配置值
```

**效果**: `config.yaml` 中 `pipeline.max_retries: 5` 现在同时控制异常重试和 Judge REJECT 重试。

---

### 4. M-New1 — `split_input_to_chapters` 死代码

**问题**: `if hasattr(splitter, 'book') or input_format == 'epub': splitter.load() if input_format != 'epub' else splitter` 
- 三元 + `hasattr` + `or` 组合：**永不执行 `load()`**，也从不 `else`。
- epub 实际加载依赖 `EpubSplitter.split()` 内部的懒加载（`self.book is None` 时调用 `self.load()`）。

**修复**: `src/translator_agent.py:2481-2493` 删除整个 if 块。

```python
# 修复前
if hasattr(splitter, 'book') or input_format == 'epub':
    splitter.load() if input_format != 'epub' else splitter
splitter.chapters = splitter.split()
generated = splitter.write_files()

# 修复后（依赖 EpubSplitter.split() 内部懒加载）
splitter.chapters = splitter.split()
generated = splitter.write_files()
```

---

### 5. M5 — `init_project` 3 行 DEBUG 打印

**问题**: 生产代码残留 3 行 `print(f"DEBUG: ...")`。

**修复位置**: `src/translator_agent.py:1846-1848`

```python
# 删除前
print(f"DEBUG: 项目根目录: {root_dir}")
print(f"DEBUG: 检查输入文件: {input_file} -> {'存在' if input_file.exists() else '不存在'}")
print(f"DEBUG: 目标数据库路径: {db_file}")

# 删除后（仅保留必要错误提示）
if not input_file.exists():
    print(f"❌ 找不到文件: {input_file}")
```

---

## 🎁 额外修复

### 重复 `_config = get_config()` 清理

**发现过程**: 修 CI1 时发现 `src/translator_agent.py` 存在两处 `_config = get_config()`：
- Line 44 (import 后)
- Line 151 (Section 2 内部，早期重构遗留)

**处理**: 删除 Line 151 的重复赋值（原代码 `# 全局配置实例` + `_config = get_config()`）。

---

## 🧪 验证结果

### 单文件降级（CI1）
```bash
$ mv src/config.py src/config.py.bak
$ python -c "from src.translator_agent import _config; print(_config.llm_backend)"
[WARN] 未找到 src.config，使用内建默认配置（单文件模式）
mock
$ mv src/config.py.bak src/config.py
```

### model_name 解析正确（H-New1）
```bash
$ python -c "from src.translator_agent import _config; ..."
Backend: mlx → model_name: qwen/Qwen2.5-7B-Instruct-MLX-4bit
```

### 端到端测试通过
```bash
$ ./scripts/translate_book.sh /tmp/test_book.txt output --backend mock --dry-run --force --clean

🎉 翻译全流程完成！

已清理 9 个中间文件 (raw/literary/critic_report)
保留: DB (workflow.db + decision_db.sqlite) + 3 个 final.json + translated_full.md (~45116 bytes)
```

---

## 📊 Git 提交记录

| Commit | Message | Files |
|--------|---------|-------|
| `b582878` | Initial commit: pre-audit baseline | 43 files |
| `8a995f0` | Fix: 5 audit issues (P0/P1/P2) | 1 file (+64/-15) |

```bash
$ git log --oneline -5
8a995f0 Fix: 5 audit issues (P0/P1/P2)
b582878 Initial commit: pre-audit baseline
```

---

## 📋 未修改项（已知取舍）

| ID | 等级 | 原因 |
|----|------|------|
| **H-New2 深化** | 拆分 `exception_retries` / `judge_retries` | 超出最小修复范围，需配置 schema 迁移，单独 RFC |
| **M-New2** | `resolve_task_model` 直接返回 model_name | 涉及 `src/config.py`（外部依赖），用户明令"不要动无关业务代码" |
| **M-New3** | 提取 `_split_by_size` | 可接受，建议不复用/保持简单，P3 |
| **M-New4** | EPUB 单引号属性 | 现代 EPUB 几乎双引号，P3 兜底 |
| **C1/H1/H3** | 历史遗留设计取舍 | 非 bug，已在审计报告标注"设计取舍" |

---

**报告生成时间**: 2026-06-29  
**文件**: `docs/audit_round3_fix_report.md`