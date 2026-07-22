#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# OpenLiterary - 一键翻译脚本
# 用法:
#   ./scripts/translate_book.sh <input.{epub,txt,md}> [output_dir] [--force] [--dry-run] [--consistency] [--golden-gate]
#
# 临时切换后端/模型（不修改 config.yaml，通过环境变量）：
#   export OPENLITERARY__LLM_BACKEND=openai_api
#   export OPENLITERARY__MODELS__PRIMARY__MODEL_NAME=mistral-large
#   export OPENLITERARY__MODELS__PRIMARY__API_BASE=https://api.mistral.ai/v1
#   export OPENLITERARY__MODELS__PRIMARY__API_KEY=$MISTRAL_KEY
#   ./scripts/translate_book.sh book.epub
#
# 示例:
#   ./scripts/translate_book.sh book.epub                          # 默认使用 config.yaml
#   ./scripts/translate_book.sh book.md output --dry-run            # 仅切分+跑流程，不调用 LLM（split 仍执行）
#   OPENLITERARY__LLM_BACKEND=mock ./scripts/translate_book.sh book.md  # mock 后端跑流程

set -euo pipefail

# ===========================
# 默认配置
# ===========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EPUB_FILE=""
OUTPUT_DIR="output"
FORCE_INIT=false
DRY_RUN=false
CLEAN_INTERMEDIATE=false
CLEAN_ONLY=false
RESET=false
SKIP_EPUB_CONVERT=false
RUN_CONSISTENCY=false
RUN_GOLDEN_GATE=false
MERGE_ONLY=false
CHAPTERS_DIR="input"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*" >&2; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $*" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ===========================
# 帮助信息
# ===========================
usage() {
    cat <<EOF
用法: $0 <input.epub> [output_dir] [选项]
       $0 --clean-only                          # 独立清理模式，无需输入文件

选项:
  -f, --force              强制重新初始化章节 (覆盖已存在的 chunk)
  -n, --dry-run            仅跑流程，不实际调用 LLM (用 mock 后端)
  --skip-epub              跳过 EPUB 转换，直接用现有 input/ 下的 .md
  -r, --reset              清空所有历史状态 (删除 DB + output/)，重新开始
            -c, --clean              翻译完成后清理中间产物 (raw/literary/critic_report，保留 final.json 和 DB)
  --consistency            翻译+合并后跑 consistency 子命令，输出命名一致性差异报告
  --golden-gate           翻译+合并后跑 golden-gate 质量门禁（参考人工基线，不通过时仅 WARN）
  --clean-only             独立模式：仅清理中间产物后退出，无需输入文件
  -h, --help               显示帮助

 环境变量 (临时覆盖配置，不修改磁盘 config.yaml；使用 __ 分隔嵌套键):
  OPENLITERARY__LLM_BACKEND
  OPENLITERARY__MODELS__<ROLE>__MODEL_NAME
  OPENLITERARY__MODELS__<ROLE>__API_BASE
  OPENLITERARY__MODELS__<ROLE>__API_KEY
  其中 <ROLE> ∈ literal_translator | primary

示例:
  # 使用 config.yaml 默认配置 (推荐先编辑 config.yaml 选定后端/模型)
  $0 book.epub

  # 临时切到云 API 不改 yaml (Mistral 示例)
  OPENLITERARY__MODELS__PRIMARY__API_BASE=https://api.mistral.ai/v1 \\
  OPENLITERARY__MODELS__PRIMARY__API_KEY=\$MISTRAL_KEY \\
  OPENLITERARY__MODELS__PRIMARY__MODEL_NAME=mistral-large \\
  $0 book.epub

  # 仅测试流程 (不消耗 token)
  OPENLITERARY__LLM_BACKEND=mock $0 book.md output --dry-run

  # 独立清理中间产物（无需输入文件）
  $0 --clean-only
EOF
}

# ===========================
# 参数解析
# ===========================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--force) FORCE_INIT=true; shift ;;
            -n|--dry-run) DRY_RUN=true; shift ;;
            --skip-epub) SKIP_EPUB_CONVERT=true; shift ;;
            -r|--reset) RESET=true; shift ;;
            -c|--clean) CLEAN_INTERMEDIATE=true; shift ;;
            --consistency) RUN_CONSISTENCY=true; shift ;;
            --merge-only) MERGE_ONLY=true; shift ;;
            --golden-gate) RUN_GOLDEN_GATE=true; shift ;;
            --clean-only) CLEAN_ONLY=true; shift ;;
            -h|--help) usage; exit 0 ;;
            -*) log_err "未知选项: $1"; usage; exit 1 ;;
            *)
                if [[ -z "$EPUB_FILE" ]]; then
                    EPUB_FILE="$1"
                elif [[ "$OUTPUT_DIR" == "output" ]]; then
                    OUTPUT_DIR="$1"
                else
                    log_err "过多的位置参数: $1"
                    usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$EPUB_FILE" && "$CLEAN_ONLY" != "true" && "$RESET" != "true" && "$MERGE_ONLY" != "true" ]]; then
        log_err "必须指定输入 EPUB 文件"
        usage
        exit 1
    fi
}

# ===========================
# 依赖检查
# ===========================
check_deps() {
    log_info "检查依赖..."

    # Python
    if ! command -v python3 &>/dev/null; then
        log_err "需要 python3"
        exit 1
    fi

    # 项目依赖
    cd "$PROJECT_ROOT"
    if [[ ! -f "src/translator_agent.py" ]]; then
        log_err "找不到 src/translator_agent.py，请在项目根目录运行"
        exit 1
    fi

    # 核心依赖
    python3 -c "import yaml, requests, psutil, json_repair" 2>/dev/null || {
        log_warn "缺少核心依赖，尝试安装..."
        pip install -q -r requirements-core.txt
    }

    # EPUB 转换依赖
    if [[ "$SKIP_EPUB_CONVERT" != "true" ]]; then
        python3 -c "import ebooklib, bs4, lxml" 2>/dev/null || {
            log_warn "缺少 EPUB 依赖，尝试安装..."
            pip install -q ebooklib beautifulsoup4 lxml
        }
    fi

    # MLX 后端检查：仅在环境变量明确指定 mlx 时安装
    local effective_backend="${OPENLITERARY__LLM_BACKEND:-}"
    if [[ "$effective_backend" == "mlx" && "$DRY_RUN" != "true" ]]; then
        python3 -c "import mlx_lm, mlx" 2>/dev/null || {
            log_warn "MLX 未安装，尝试安装..."
            pip install -q mlx mlx-lm
        }
    fi

    log_ok "依赖检查通过"
}

# ===========================
# 临时覆盖配置（不修改磁盘 config.yaml）
# 通过 OPENLITERARY_* 环境变量实现：
#   OPENLITERARY__LLM_BACKEND
#   OPENLITERARY__MODELS__PRIMARY__MODEL_NAME
#   OPENLITERARY__MODELS__PRIMARY__API_BASE
#   OPENLITERARY__MODELS__PRIMARY__API_KEY
# ===========================

# ===========================
# EPUB -> Markdown
# ===========================
convert_epub() {
    log_info "切分输入 -> 章节 Markdown..."
    mkdir -p "$PROJECT_ROOT/$CHAPTERS_DIR"

    # 使用聚合脚本的 split 子命令（支持 EPUB/TXT/MD 自动识别）
    mapfile -t generated < <(cd "$PROJECT_ROOT" && python3 -m src.translator_agent split \
        --input "$EPUB_FILE" \
        --input-dir "$PROJECT_ROOT/$CHAPTERS_DIR" \
        --input-format auto \
        --chapter-size 5000 \
        --min-chapter-size 1000) || {
        log_err "切分失败"
        exit 1
    }

    if [[ ${#generated[@]} -eq 0 ]]; then
        log_err "未生成任何章节文件"
        exit 1
    fi

    log_ok "生成 ${#generated[@]} 个章节: ${generated[*]}"
    printf '%s\n' "${generated[@]}"
}

# ===========================
# 运行单章翻译
# ===========================
run_chapter() {
    local ch_file="$1"
    local ch_id=$(basename "$ch_file" .md)
    local force_flag=""

    [[ "$FORCE_INIT" == "true" ]] && force_flag="--force"

    log_info "=== 处理章节: $ch_id ==="

    # 1. 初始化（检查退出码）
    log_info "  初始化任务数据库..."
    if ! python3 -m src.translator_agent init --chapter "$ch_id" $force_flag; then
        log_err "  初始化失败: $ch_id"
        return 1
    fi

    # 2. 跑管线（显式捕获退出码，失败即返回非零，不装饰性报完成）
    log_info "  启动翻译管线..."
    local pipe_rc=0
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "  [DRY RUN] 跳过实际 LLM 调用"
        # 可以用 mock 后端快速跑完
        OPENLITERARY__LLM_BACKEND=mock python3 -m src.translator_agent pipeline --chapter "$ch_id"
        pipe_rc=$?
    else
        python3 -m src.translator_agent pipeline --chapter "$ch_id"
        pipe_rc=$?
    fi

    if [[ $pipe_rc -ne 0 ]]; then
        local chapter_output_dir="$PROJECT_ROOT/output/$ch_id"
        local final_count=$(ls -1 "$chapter_output_dir"/*_final.json 2>/dev/null | wc -l)
        if [[ "$final_count" -gt 0 ]]; then
            log_warn "  章节 $ch_id 管线完成但有 chunk 质量未达标(fallback)，译文已写入 $chapter_output_dir/ ($final_count 个 final.json)"
        else
            log_err "  章节 $ch_id 管线失败 (exit=$pipe_rc)，完全无产出"
        fi
        return 1
    fi
    log_ok "  章节 $ch_id 管线通过"
}

# ===========================
# 并行运行所有章节
# ===========================
run_all_chapters() {
    local chapters=("$@")
    local total=${#chapters[@]}

    log_info "开始翻译 $total 个章节 (顺序执行)..."

    local done=0
    local failed_count=0
    local failed_chapters=()
    for ch_file in "${chapters[@]}"; do
        local ch_name=$(basename "$ch_file" .md)
        if run_chapter "$ch_file"; then
            log_ok "  章节完成: $ch_name"
        else
            log_warn "  章节失败: $ch_name，继续下一章"
            failed_count=$((failed_count + 1))
            failed_chapters+=("$ch_name")
        fi
        done=$((done + 1))
        log_info "进度: $done/$total"
    done

    if [[ "$failed_count" -gt 0 ]]; then
        log_err "章节翻译完成，但有 $failed_count/$total 章失败"
        log_err "失败章节: ${failed_chapters[*]}"
        return 1
    fi
    log_ok "所有章节翻译完成"
}

# ===========================
# 合并输出
# ===========================
chapter_title() {
    local ch_id="$1"
    local ch_file="$PROJECT_ROOT/$CHAPTERS_DIR/${ch_id}.md"
    if [[ -f "$ch_file" ]]; then
        # 读取章节 md 首行 H1 标题（去除 # 前缀和首尾空白）
        local title
        title=$(head -10 "$ch_file" 2>/dev/null | grep -m1 '^# ' | sed 's/^# //' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        if [[ -n "$title" ]]; then
            echo "$title"
            return
        fi
    fi
    # 回退：返回章节 ID 本身
    echo "$ch_id"
}

merge_output() {
    local chapters_dir="$1"
    # 注意：$OUTPUT_DIR 仅控制最终合并文件路径，中间产物始终写入 output/<chapter_id>/
    local output_file="$PROJECT_ROOT/$OUTPUT_DIR/translated_full.md"
    local missing_count=0
    local incomplete_count=0
    local total_chapters=0

    log_info "合并译文 -> $output_file"

    mkdir -p "$(dirname "$output_file")"

    # Q1 修复：fallback 告警独立写入 QA_REPORT.md，不污染译文正文
    local qa_report_file="$PROJECT_ROOT/$OUTPUT_DIR/QA_REPORT.md"
    : > "$qa_report_file"
    echo "# 译文质量验收报告 (QA Report)" > "$qa_report_file"
    echo "" >> "$qa_report_file"
    echo "本文件记录未通过自动化验收、以 fallback 最后一版润色稿收录的章节。译文正文 (translated_full.md) 不含本告警。" >> "$qa_report_file"
    echo "" >> "$qa_report_file"
    export QA_REPORT_FILE="$qa_report_file"

    # 按章节顺序合并 final.json
    for ch_file in "$PROJECT_ROOT/$CHAPTERS_DIR"/ch*.md; do
        [[ -f "$ch_file" ]] || continue
        local ch_id=$(basename "$ch_file" .md)
        total_chapters=$((total_chapters + 1))
        # 找出该章节的所有 final.json（每个 chunk 一个）
        local chapter_output_dir="$PROJECT_ROOT/output/$ch_id"
        local chunk_files=($(ls -1 "$chapter_output_dir"/*_final.json 2>/dev/null | sort -V))
        # 与 DB 任务数交叉校验：章节 final 数应等于 chunk_tasks 该章任务数
        local db_count=0
        if [[ -f "$PROJECT_ROOT/db/workflow.db" ]]; then
            db_count=$(python3 -c "
import sqlite3, sys
try:
    c = sqlite3.connect('$PROJECT_ROOT/db/workflow.db')
    print(c.execute(\"SELECT COUNT(*) FROM chunk_tasks WHERE chapter_id = ?\", ('$ch_id',)).fetchone()[0])
except Exception:
    print(0)
" 2>/dev/null)
            [[ -z "$db_count" ]] && db_count=0
        fi
        if [[ ${#chunk_files[@]} -eq 0 ]]; then
            log_warn "  缺少 $chapter_output_dir/*_final.json，跳过"
            missing_count=$((missing_count + 1))
            continue
        fi
        if [[ "$db_count" -gt 0 && "${#chunk_files[@]}" -ne "$db_count" ]]; then
            log_warn "  章节 $ch_id 不完整：DB 任务 $db_count 个，final.json ${#chunk_files[@]} 个"
            incomplete_count=$((incomplete_count + 1))
        fi
        # 规范章节标题：每章统一生成单一 H1，去除 chunk 内嵌的错位/无 # 标题
        # 经循环 stdout 写出（与 python 共用同一重定向 fd，避免 >> 被 > 覆盖）
        echo "# $(chapter_title "$ch_id")"
        echo ""
        local chunk_idx=0
        for final_file in "${chunk_files[@]}"; do
            FINAL_FILE="$final_file" CH_ID="$ch_id" CH_IDX="$chunk_idx" QA_REPORT_FILE="$qa_report_file" python3 -c '
import json, os, re
f = os.environ["FINAL_FILE"]; ch = os.environ["CH_ID"]; idx = os.environ["CH_IDX"]
data = json.load(open(f, encoding="utf-8"))
text = data.get("text", "") if isinstance(data, dict) else str(data)
meta = (data.get("metadata") or {}) if isinstance(data, dict) else {}
if meta.get("fallback"):
    reason = meta.get("judge_reason", "未通过 Critic 阈值审查")
    with open(os.environ.get("QA_REPORT_FILE", "/dev/null"), "a", encoding="utf-8") as qa:
        qa.write(f"## {ch}\n")
        qa.write(f"- 状态: 未通过自动化验收（已 fallback 收录最后一版润色稿）\n")
        qa.write(f"- 原因: {reason}\n\n")
# 剥离古腾堡标题页残留（仅 ch01 首块携带）
guten = re.compile(r"^\s*#\s*古腾堡计划|爱丽丝梦游仙境\s*$|^刘易斯·卡罗尔\s*著|千禧年支点版|^\s*目录\s*$")
text = "\n".join(l for l in text.split("\n") if not guten.match(l.strip()))
# 剥离 chunk 内嵌章节标题行（含错位/无 # 形式），改由合并层统一生成
head = re.compile(r"^\s*#?\s*第[一二三四五六七八九十百千零\d]+\s*[章.。]")
text = "\n".join(l for l in text.split("\n") if not head.match(l.strip()))
# 脚注按 chunk 加命名空间，消除跨 chunk 同名标签碰撞
keys = set(re.findall(r"\[\^([^\]]+)\]", text))
if keys:
    pref = f"{ch}_c{idx}"
    text = re.sub(r"\[\^([^\]]+)\]", lambda m: f"[^{pref}_{m.group(1)}]", text)
print(text)
print()
' 2>/dev/null
            chunk_idx=$((chunk_idx + 1))
        done
    done > "$output_file"

    log_ok "合并完成: $output_file ($(wc -c < "$output_file") bytes)"

    # 残缺校验：存在章节但无 final.json 或 chunk 数不匹配 → 译文不完整，必须非零退出
    if [[ "$missing_count" -gt 0 || "$incomplete_count" -gt 0 ]]; then
        log_err "合并残缺：共 $total_chapters 章，缺失 $missing_count 章，不完整 $incomplete_count 章（译文不完整）"
        return 1
    fi
    return 0
}


# ===========================
# 清理中间产物（保守策略）
# ===========================
clean_intermediate() {
    log_info "清理中间产物（保守策略）..."

    local deleted_count=0
    local saved_count=0
    local saved_size=0

    for pattern in "*_raw.json" "*_literary.json" "*_literary_v*.json" \
                   "*_critic_report.json" "*_rewrite_meta_v*.json"; do
        while IFS= read -r -d '' f; do
            rm -f "$f"
            deleted_count=$((deleted_count + 1))
        done < <(find "$PROJECT_ROOT/output" -name "$pattern" -type f -print0 2>/dev/null)
    done

    for f in "$PROJECT_ROOT/db/workflow.db" "$PROJECT_ROOT/db/decision_db.sqlite" \
             "$PROJECT_ROOT/$OUTPUT_DIR/translated_full.md"; do
        if [[ -f "$f" ]]; then
            saved_count=$((saved_count + 1))
            saved_size=$((saved_size + $(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")))
        fi
    done
    local final_count=$(find "$PROJECT_ROOT/output" -name "*_final.json" -type f 2>/dev/null | wc -l)
    saved_count=$((saved_count + final_count))

    log_ok "已清理 $deleted_count 个中间文件"
    log_ok "保留: DB (workflow.db + decision_db.sqlite) + $final_count 个 final.json + translated_full.md (~${saved_size} bytes)"
}


# ===========================
# 清空历史状态（--reset）
# ===========================
reset_state() {
    local confirm_msg="即将清空所有历史翻译数据：\n  - 任务数据库 (db/workflow.db)\n  - 决策数据库 (db/decision_db.sqlite)\n  - 全部输出文件 (output/)\n  - 章节源文件 (input/ —— 下次运行时会从 EPUB 重新生成)"
    
    if [[ -t 0 ]]; then
        log_warn "$confirm_msg"
        read -r -p "确认清空？[y/N] " reply
        [[ "$reply" != "y" && "$reply" != "Y" ]] && { log_info "已取消"; exit 0; }
    fi

    log_info "清空历史状态..."

    rm -rf "$PROJECT_ROOT/db/workflow.db" "$PROJECT_ROOT/db/workflow.db-shm" "$PROJECT_ROOT/db/workflow.db-wal"
    rm -rf "$PROJECT_ROOT/db/decision_db.sqlite" "$PROJECT_ROOT/db/decision_db.sqlite-shm" "$PROJECT_ROOT/db/decision_db.sqlite-wal"
    rm -rf "$PROJECT_ROOT/output"
    rm -f "$PROJECT_ROOT/$OUTPUT_DIR/translated_full.md"
    rm -rf "$PROJECT_ROOT/$CHAPTERS_DIR"

    log_ok "历史状态已清空（input/ 已删除，下次运行会自动从 EPUB 重新切分）"
}


# ===========================
# 主流程
# ===========================
main() {
    parse_args "$@"

    # --clean-only 独立模式：清理中间产物后退出
    if [[ "$CLEAN_ONLY" == "true" ]]; then
        cd "$PROJECT_ROOT"
        clean_intermediate
        exit 0
    fi

    # --reset 独立模式：清空全部历史后退出；若后续还有翻译流程则顺延执行
    if [[ "$RESET" == "true" ]]; then
        cd "$PROJECT_ROOT"
        reset_state
        # 若只有 --reset 没有输入文件，重置后退出
        if [[ -z "$EPUB_FILE" && "$SKIP_EPUB_CONVERT" != "true" ]]; then
            log_info "已清空全部工作区。下次运行请指定 EPUB 文件："
            log_info "  $0 book.epub"
            exit 0
        fi
    fi

    log_info "========================================"
    log_info "OpenLiterary 一键翻译"
    log_info "输入: $EPUB_FILE"
    log_info "输出目录: $OUTPUT_DIR"
    log_info "后端: ${OPENLITERARY__LLM_BACKEND:-config.yaml}"
    local active_env_overrides=$(env | grep '^OPENLITERARY__' | cut -d= -f1 | tr '\n' ' ' || true)
    [[ -n "$active_env_overrides" ]] && log_info "运行时 env 覆盖: $active_env_overrides"
    log_info "强制重建: $FORCE_INIT"
    log_info "清空历史: $RESET"
    log_info "Dry-run: $DRY_RUN"
    log_info "清理中间产物: $CLEAN_INTERMEDIATE"
    log_info "一致性校验: $RUN_CONSISTENCY"
    log_info "========================================"

    cd "$PROJECT_ROOT"

    check_deps

    if [[ "$MERGE_ONLY" == "true" ]]; then
        log_info "仅合并模式（--merge-only）：跳过切分/翻译，直接合并现有 final.json"
    else
        # 1. EPUB 转换
        local chapter_files=()
        if [[ "$SKIP_EPUB_CONVERT" == "true" ]]; then
            log_info "跳过 EPUB 转换，使用现有 input/"
            chapter_files=($(ls -1 "$CHAPTERS_DIR"/ch*.md 2>/dev/null | sort))
            [[ ${#chapter_files[@]} -eq 0 ]] && { log_err "input/ 下无章节文件"; exit 1; }
        else
            mapfile -t chapter_files < <(convert_epub)
        fi

        # 2. 并行翻译所有章节
        if ! run_all_chapters "${chapter_files[@]}"; then
            log_err "部分章节翻译失败，继续合并（已产出的章节仍会被收录）"
        fi
    fi

    # 3. 合并输出
    if ! merge_output "$CHAPTERS_DIR"; then
        log_err "翻译产出残缺，流程中止（见上方缺失章节提示）。"
        exit 1
    fi

    # 4. 译后命名一致性校验（可选 --consistency）
    if [[ "$RUN_CONSISTENCY" == "true" ]]; then
        log_info "运行译后命名一致性校验 (consistency)..."
        python3 -m src.translator_agent consistency || {
            log_warn "consistency 检出命名不一致（详见 output/consistency_diff.md），未自动修正"
        }
    fi

    # 4.5 译后质量门禁（可选 --golden-gate，Round 18 审计接入）
    # 参考人工评分基线的动态阈值，仅 WARN 不阻断（与 consistency 同模式）
    if [[ "$RUN_GOLDEN_GATE" == "true" ]]; then
        log_info "运行 Golden Gate 质量门禁..."
        if python3 -m src.translator_agent golden-gate; then
            log_ok "golden-gate 通过（pipeline pass_rate 满足人工基线动态阈值）"
        else
            log_warn "golden-gate 未通过（pipeline 质量低于人工基线阈值），详见 output/golden_test_report.json"
        fi
    fi

    # 5. 清理中间产物（仅在 --clean 时）
    if [[ "$CLEAN_INTERMEDIATE" == "true" ]]; then
        clean_intermediate
    fi

    log_info "========================================"
    log_ok "🎉 翻译全流程完成！"
    log_info "最终译文: $PROJECT_ROOT/$OUTPUT_DIR/translated_full.md"
    log_info "Decision DB: $PROJECT_ROOT/db/decision_db.sqlite"
    log_info "========================================"
}

main "$@"