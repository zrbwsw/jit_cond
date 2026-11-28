#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
EPOCHS="${EPOCHS:-80}"
NOISE_BETA_DINO_PIXEL="${NOISE_BETA_DINO_PIXEL:-0.25}"
NOISED_OUTPUT_DIR="${NOISED_OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/stage1-raev2-dinov3l-k7-stage2-pixeljit-b-noised-e${EPOCHS}}"
PRED_OUTPUT_DIR="${PRED_OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/stage1-raev2-dinov3l-k7-stage2-pixeljit-b-pred-e${EPOCHS}}"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo '用法：bash sh/queue_noised_pred_e80_gpu1.sh [--dry-run]' >&2
    exit 2
fi
if (( ! dry_run )); then
    for output_dir in "$NOISED_OUTPUT_DIR" "$PRED_OUTPUT_DIR"; do
        if [[ -e "$output_dir" ]]; then
            echo "Output directory already exists; refusing to overwrite: $output_dir" >&2
            exit 1
        fi
    done
fi

run_mode() {
    local mode="$1"
    local output_dir="$2"
    local extra_args=()
    if (( dry_run )); then
        extra_args+=(--dry-run)
    fi
    printf 'Starting %s for %s epochs on %s; output=%s\n' "$mode" "$EPOCHS" "$GPU_HOST" "$output_dir"
    CONDITION_MODE="$mode" EPOCHS="$EPOCHS" \
        NOISE_BETA_DINO_PIXEL="$NOISE_BETA_DINO_PIXEL" \
        OUTPUT_DIR="$output_dir" EXP_NAME="$(basename "$output_dir")" \
        GPU_HOST="$GPU_HOST" RESUME= \
        bash "$PROJECT_ROOT/sh/train_pixel_raev2.sh" "${extra_args[@]}"
}

run_mode noised "$NOISED_OUTPUT_DIR"
run_mode pred "$PRED_OUTPUT_DIR"
printf 'Completed noised and pred training on %s.\n' "$GPU_HOST"
