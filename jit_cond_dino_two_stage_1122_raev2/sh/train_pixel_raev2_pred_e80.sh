#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
export EPOCHS="${EPOCHS:-80}"
export CONDITION_MODE="${CONDITION_MODE:-pred}"
export NOISE_BETA_DINO_PIXEL="${NOISE_BETA_DINO_PIXEL:-0.25}"
export EXP_NAME="${EXP_NAME:-stage1-raev2-dinov3l-k7-stage2-pixeljit-b-pred-e${EPOCHS}}"
exec "$PROJECT_ROOT/sh/train_pixel_raev2.sh" "$@"
