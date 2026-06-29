# `input/` — Test Material Catalog

> 用途：覆盖 Phase 1–4 测试方案所需的全部输入素材。
> 基准：Hyperion Cantos 风格（Dan Simmons），文学英译中方向。
> 切分参数：`soft_limit=1000` chars / `hard_limit=2500` chars（来自 `SmartChunker`）。

---

## 一、Smoke Test 主测试集（`input/ch*.md`）

适用于 Phase 1 冷启动验证、Phase 2 状态机防弹、Phase 3 内存压测。

| 文件 | 字数 | 字符数 | 预期 chunks | 文学特征 | 测试场景 |
|------|------|--------|-------------|----------|----------|
| `ch01.md` | 1155 | 6339 | 6–7 | Keats 典故、POV "I"、对话引号、情感词密集 | Phase 1 闭环 / Phase 2 force 覆盖 |
| `ch02.md` | 941 | 5200 | 5–6 | 学者独白、Meta 注释、复合句 | 多章节独立测试 |
| `ch03.md` | 939 | 4951 | 5–6 | 战士 POV、回忆穿插、Moneta 主题 | 情感强度测试 |

**总规模**：3035 词 / 16490 字符 / 约 17–19 chunks。

### 运行方式

```bash
# 初始化单个章节
python3 src/translator_agent.py init --chapter ch01

# 强制覆盖（Phase 2 测试）
python3 src/translator_agent.py init --chapter ch01 --force

# 完整流水线
python3 src/translator_agent.py pipeline --chapter ch01
python3 src/translator_agent.py pipeline --chapter ch02
python3 src/translator_agent.py pipeline --chapter ch03
```

---

## 二、Golden Set 基线集（`input/golden/`）

适用于 Phase 4 文学指标回归测试。

| 文件 | 字数 | 字符数 | 预期 chunks | 文学特征 | 基线指标 |
|------|------|--------|-------------|----------|----------|
| `hyperion_5k.md` | 2747 | 15290 | 13–15 | 5 章节混编、Keats/Shakespeare/Milton/Dante/Homer/Shelley/Byron 全员典故、cruciform 宗教隐喻 | `avg_style_collapse_rate` / `pass_rate` |

**运行方式**：

```bash
python3 src/translator_agent.py golden
# 输出 → output/golden_test_report.json
```

---

## 三、Golden Set 压力测试集（`input/golden/stress/`）

每个文件针对性压测 `_infer_author_priority_ratio()` 中的一个或多个维度。

| 文件 | 字数 | 字符数 | 压测维度 | 预期特征 | 预期比率 |
|------|------|--------|----------|----------|----------|
| `allusion_heavy.md` | 578 | 3326 | `allusion_density` | 7 个典故标记全密集出现 | `author_priority ≈ 0.85–0.90`（clamp） |
| `pov_unstable.md` | 843 | 4229 | `pov_changes` | I/he/she/we POV 频繁切换 | `author_priority ≈ 0.6`（`base_ratio -= 0.1`） |
| `poetry_prose.md` | 480 | 2568 | `poetry_score` | `\n\n`、`——`、`...`、Keats "Beauty is truth" 引用 | `author_priority ≈ 0.8`（`+= 0.1`） |

**运行方式**：将单个 stress 文件替换或追加到 `golden/` 目录后执行 `golden` 命令。

```bash
cp input/golden/stress/allusion_heavy.md input/golden/_allusion_heavy.md
python3 src/translator_agent.py golden
```

---

## 四、Chaos Fixture 边界测试集（`input/chaos/`）

针对性压测 `SmartChunker.split_markdown()` 的边界行为。

| 文件 | 字数 | 字符数 | 压测行为 | 预期输出 |
|------|------|--------|----------|----------|
| `minimal.md` | 42 | 291 | 最小输入 | 1 chunk，无 warning |
| `long_paragraph.md` | 637 | 3946 | 单段超过 `hard_limit=2500` | 触发 `⚠️ 触发硬切分保护 (长度: NNNN)` |
| `unbalanced_quotes.md` | 422 | 2201 | 对话引号穿插 | 软切分被 `open_quotes` 阻塞或放宽 |
| `many_scene_breaks.md` | 80 | 478 | 8 个 `#` 标题 | 每段独立成 chunk（scene-break 强制边界） |
| `unicode_cjk.md` | 57 | 2072 | CJK + 中文标点 + 直引号 | chunker 字符计数应正确处理 CJK |

**运行方式**：替换 `ch01.md` 后执行 init。

```bash
cp input/chaos/long_paragraph.md input/_chaos_test.md
python3 src/translator_agent.py init --chapter _chaos_test
```

或通过 Python 直接调用 chunker：

```python
from src.translator_agent import SmartChunker
chunks = SmartChunker().split_markdown(open("input/chaos/long_paragraph.md").read())
print(f"{len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"[{i}] {len(c)} chars")
```

---

## 五、测试矩阵总览

| 测试类型 | 文件 | Phase | 验证目标 |
|----------|------|-------|----------|
| Smoke | `ch01.md` | Phase 1 | DB 建表 + 切分 + Mock 流水线闭环 |
| Smoke | `ch02.md`, `ch03.md` | Phase 1 | 多章节独立性 |
| Chaos 状态机 | `ch01.md` (modify content) + `--force` | Phase 2 | force 覆盖语义、retries 重置 |
| Chaos 回溯 | 手动 SQL 插入冲突规则 | Phase 2 | DIRTY 自动回退到 EXTRACTING_TERMS |
| Hardware | `ch01.md` + `ch02.md` + `ch03.md` | Phase 3 | MLX 在 16GB 统一内存的驻留极限 |
| Hardware | `golden/hyperion_5k.md` | Phase 3 | 长文本吞吐量（`tok/s`） |
| Golden baseline | `golden/hyperion_5k.md` | Phase 4 | 文学指标基线 |
| Golden stress | `golden/stress/allusion_heavy.md` | Phase 4 | allusion 维度回归 |
| Golden stress | `golden/stress/pov_unstable.md` | Phase 4 | POV 维度回归 |
| Golden stress | `golden/stress/poetry_prose.md` | Phase 4 | 诗歌模式回归 |
| Chunker edge | `chaos/minimal.md` | 全阶段 | 最小输入鲁棒性 |
| Chunker edge | `chaos/long_paragraph.md` | 全阶段 | hard_limit 强制切分 |
| Chunker edge | `chaos/unbalanced_quotes.md` | 全阶段 | 对话引号探针 |
| Chunker edge | `chaos/many_scene_breaks.md` | 全阶段 | scene-break 强制边界 |
| Chunker edge | `chaos/unicode_cjk.md` | 全阶段 | CJK 字符宽度 |

---

## 六、汇总数据

| 指标 | 数值 |
|------|------|
| 测试文件总数 | 14（含 1 个原 `._ch01.md` 隐藏文件） |
| 总字数 | 8931 |
| 总字符数 | 54987 |
| 主测试集 | 3035 词 / 16490 字符 / ~17 chunks |
| Golden baseline | 2747 词 / 15290 字符 / ~14 chunks |
| Golden stress 总计 | 1901 词 / 10123 字符 / ~10 chunks |
| Chaos fixtures | 1238 词 / 8988 字符 |

---

*生成时间：2026-06-26*
*生成方式：手工撰写（Hyperion Cantos 风格）+ 系统性覆盖测试维度*
