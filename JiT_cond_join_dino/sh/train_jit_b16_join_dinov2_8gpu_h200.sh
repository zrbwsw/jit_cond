#!/usr/bin/env bash
# 通过现有 run_on_gpu.sh 在 GPU1 上运行 dinov2b 训练；不修改连接配置。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/train_jit_b16_join_dinov2_8gpu_h200.sh [--dry-run]' >&2
    exit 2
fi

forwarded_env=(EPOCHS="${EPOCHS:-200}" SAVE_INTERVAL="${SAVE_INTERVAL:-20}" SAVE_LAST_FREQ="${SAVE_LAST_FREQ:-5}")
for name in PYTHON_BIN CUDA_VISIBLE_DEVICES GPUS EXPERIMENT OUTPUT_DIR RESUME \
    IMAGENET_PATH IMAGENET_ROOT PRETRAIN_MODELS_ROOT FID_STATS_ROOT \
    DINOV3_REPO DINO_REPO DINO_WEIGHTS DINO_WEIGHTS_PATH RDZV_PORT DINO_CFG DINOV2_MODEL_PATH; do
    if [[ -v "$name" ]]; then forwarded_env+=("$name=${!name}"); fi
done
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env "${forwarded_env[@]}" bash sh/train_jit_b16_join_dinov2_8gpu_local.sh
