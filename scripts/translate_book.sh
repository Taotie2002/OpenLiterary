#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# OpenLiterary - 一键翻译脚本
# 用法:
#   ./scripts/translate_book.sh <input.{epub,txt,md}> [output_dir] [--backend mlx|openai_api|mock] [--model MODEL_NAME] [--force] [--dry-run]
#
# 示例:
#   ./scripts/translate_book.sh book.epub                          # 默认 MLX 后端
#   ./scripts/translate_book.sh book.txt output --backend openai_api --model qwen2.5-14b-instruct
#   ./scripts/translate_book.sh book.md output --backend mock --dry-run  # 仅跑流程，不调用 LLM

set -euo pipefail

# ===========================
# 默认配置
# ===========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EPUB_FILE=""
OUTPUT_DIR="output"
LLM_BACKEND="mlx"
MODEL_OVERRIDE=""
FORCE_INIT=false
DRY_RUN=false
MAX_PARALLEL=1
CLEAN_INTERMEDIATE=false
SKIP_EPUB_CONVERT=false
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

选项:
  -b, --backend BACKEND    LLM 后端: mlx | openai_api | mock (默认: mlx)
  -m, --model MODEL        覆盖 config.yaml 中的模型名称 (仅 openai_api 有效)
  -f, --force              强制重新初始化章节 (覆盖已存在的 chunk)
  -n, --dry-run            仅跑流程，不实际调用 LLM (用 mock 后端)
  -j, --jobs N             并行章节数 (默认: 1，建议 ≤4)
  --skip-epub              跳过 EPUB 转换，直接用现有 input/ 下的 .md
  -c, --clean              完成后清理中间产物 (raw/literary/critic_report，保留 final.json 和 DB)
  -h, --help               显示帮助

环境变量 (优先级高于参数):
  OPENLITERARY_LLM_BACKEND
  OPENLITERARY_OPENAI_API_BASE
  OPENLITERARY_OPENAI_API_KEY
  OPENLITERARY_MLX_MODELS__REASONING_PRIMARY__MODEL_ID
  等 (见 config.yaml 说明)

示例:
  # 使用 MLX 本地推理 (需 macOS + Apple Silicon)
  $0 book.epub

  # 使用云 API (vLLM / LM Studio / OpenAI 兼容)
  $0 book.epub output --backend openai_api --model qwen2.5-14b-instruct

  # 仅测试流程 (不消耗 token)
  $0 book.epub output --backend mock --dry-run
EOF
}

# ===========================
# 参数解析
# ===========================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--backend) LLM_BACKEND="$2"; shift 2 ;;
            -m|--model) MODEL_OVERRIDE="$2"; shift 2 ;;
            -f|--force) FORCE_INIT=true; shift ;;
            -n|--dry-run) DRY_RUN=true; shift ;;
            -j|--jobs) MAX_PARALLEL="$2"; shift 2 ;;
            --skip-epub) SKIP_EPUB_CONVERT=true; shift ;;
            -c|--clean) CLEAN_INTERMEDIATE=true; shift ;;
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

    if [[ -z "$EPUB_FILE" ]]; then
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
    python3 -c "import yaml, requests, psutil" 2>/dev/null || {
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

    # MLX 后端检查
    if [[ "$LLM_BACKEND" == "mlx" && "$DRY_RUN" != "true" ]]; then
        python3 -c "import mlx_lm, mlx" 2>/dev/null || {
            log_warn "MLX 未安装，尝试安装..."
            pip install -q mlx mlx-lm
        }
    fi

    log_ok "依赖检查通过"
}

# ===========================
# 更新 config.yaml
# ===========================
update_config() {
    local config_file="$PROJECT_ROOT/config.yaml"
    if [[ ! -f "$config_file" ]]; then
        log_err "找不到 config.yaml"
        exit 1
    fi

    log_info "更新配置: llm_backend=$LLM_BACKEND"

    # 使用 yq 或 python 修改 yaml
    python3 <<PYEOF
import yaml, sys
with open("$config_file") as f:
    cfg = yaml.safe_load(f)

cfg['llm_backend'] = "$LLM_BACKEND"

if "$MODEL_OVERRIDE" != "":
    if "$LLM_BACKEND" == "openai_api":
        cfg['openai_api']['models']['reasoning_primary']['model_name'] = "$MODEL_OVERRIDE"
        cfg['openai_api']['models']['reasoning_heavy']['model_name'] = "$MODEL_OVERRIDE"
        cfg['openai_api']['models']['literal_translator']['model_name'] = "$MODEL_OVERRIDE"
    elif "$LLM_BACKEND" == "mlx":
        cfg['mlx']['models']['reasoning_primary']['model_id'] = "$MODEL_OVERRIDE"
        cfg['mlx']['models']['reasoning_heavy']['model_id'] = "$MODEL_OVERRIDE"
        cfg['mlx']['models']['literal_translator']['model_id'] = "$MODEL_OVERRIDE"

with open("$config_file", 'w') as f:
    yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
print("config.yaml 已更新")
PYEOF
}

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

    # 1. 初始化
    log_info "  初始化任务数据库..."
    python3 -m src.translator_agent init --chapter "$ch_id" $force_flag

    # 2. 跑管线
    log_info "  启动翻译管线..."
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "  [DRY RUN] 跳过实际 LLM 调用"
        # 可以用 mock 后端快速跑完
        OPENLITERARY_LLM_BACKEND=mock python3 -m src.translator_agent pipeline --chapter "$ch_id"
    else
        python3 -m src.translator_agent pipeline --chapter "$ch_id"
    fi

    log_ok "  章节 $ch_id 完成"
}

# ===========================
# 并行运行所有章节
# ===========================
run_all_chapters() {
    local chapters=("$@")
    local total=${#chapters[@]}

    log_info "开始翻译 $total 个章节 (顺序执行)..."

    local done=0
    for ch_file in "${chapters[@]}"; do
        run_chapter "$ch_file"
        local rc=$?
        if [[ $rc -ne 0 ]]; then
            log_warn "  run_chapter 返回非零退出码=$rc (继续执行)"
        fi
        done=$((done + 1))
        log_info "进度: $done/$total"
    done

    log_ok "所有章节翻译完成"
}

# ===========================
# 合并输出
# ===========================
merge_output() {
    local chapters_dir="$1"
    local output_file="$PROJECT_ROOT/$OUTPUT_DIR/translated_full.md"

    log_info "合并译文 -> $output_file"

    mkdir -p "$(dirname "$output_file")"

    # 按章节顺序合并 final.json
    for ch_file in "$PROJECT_ROOT/$CHAPTERS_DIR"/ch*.md; do
        [[ -f "$ch_file" ]] || continue
        local ch_id=$(basename "$ch_file" .md)
        # 找出该章节的所有 final.json（每个 chunk 一个）
        local chapter_output_dir="$PROJECT_ROOT/output/$ch_id"
        local chunk_files=($(ls -1 "$chapter_output_dir"/*_final.json 2>/dev/null | sort))
        if [[ ${#chunk_files[@]} -eq 0 ]]; then
            log_warn "  缺少 $chapter_output_dir/*_final.json，跳过"
            continue
        fi
        for final_file in "${chunk_files[@]}"; do
            python3 -c "
import json, sys
with open('$final_file') as f:
    data = json.load(f)
text = data.get('text', '') if isinstance(data, dict) else str(data)
print(text)
print()
" 2>/dev/null
        done
    done > "$output_file"

    log_ok "合并完成: $output_file ($(wc -c < "$output_file") bytes)"
}


# ===========================
# 清理中间产物（保守策略）
# ===========================
clean_intermediate() {
    log_info "清理中间产物（保守策略）..."

    local deleted_count=0
    local saved_count=0
    local saved_size=0

    for pattern in "*_raw.json" "*_literary.json" "*_critic_report.json"; do
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
# 主流程
# ===========================
main() {
    parse_args "$@"

    log_info "========================================"
    log_info "OpenLiterary 一键翻译"
    log_info "输入: $EPUB_FILE"
    log_info "输出目录: $OUTPUT_DIR"
    log_info "后端: $LLM_BACKEND"
    [[ -n "$MODEL_OVERRIDE" ]] && log_info "模型覆盖: $MODEL_OVERRIDE"
    log_info "强制重建: $FORCE_INIT"
    log_info "Dry-run: $DRY_RUN"
    log_info "并行度: $MAX_PARALLEL"
    log_info "清理中间产物: $CLEAN_INTERMEDIATE"
    log_info "========================================"

    cd "$PROJECT_ROOT"

    check_deps
    update_config

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
    run_all_chapters "${chapter_files[@]}"

    # 3. 合并输出
    merge_output "$CHAPTERS_DIR"

    # 4. 清理中间产物（仅在 --clean 时）
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