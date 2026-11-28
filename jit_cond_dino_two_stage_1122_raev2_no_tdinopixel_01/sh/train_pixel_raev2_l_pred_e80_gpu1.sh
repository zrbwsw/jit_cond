#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
export MODEL="${MODEL:-JiT-L/16}"
export EPOCHS="${EPOCHS:-80}"
export CONDITION_MODE="${CONDITION_MODE:-pred}"
export NOISE_BETA_DINO_PIXEL="${NOISE_BETA_DINO_PIXEL:-0.25}"
export EXP_NAME="${EXP_NAME:-stage1-raev2-dinov3l-k7-stage2-pixeljit-l-pred-e${EPOCHS}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/$EXP_NAME}"
export RESUME="${RESUME:-}"
if [[ "${1:-}" != "--dry-run" && -z "$RESUME" && -e "$OUTPUT_DIR" ]]; then
    echo "Output directory already exists; refusing to overwrite: $OUTPUT_DIR" >&2
    exit 1
fi
exec "$PROJECT_ROOT/sh/train_pixel_raev2_pred_e80.sh" "$@"
