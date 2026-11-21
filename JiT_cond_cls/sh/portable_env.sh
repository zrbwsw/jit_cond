#!/usr/bin/env bash
# Shared path and runtime setup for JiT_cond launchers.
# This file only resolves paths; it does not start a process or create tmux.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(dirname -- "$PROJECT_ROOT")"

# Set PYTHON_BIN to the environment used on the target GPU worker.  Keeping
# this configurable makes the launcher usable from both a shell and tmux.
if [[ -z "${PYTHON_BIN:-}" ]]; then
    # The shared irepa environment is present on the project workers; fall
    # back to PATH for local smoke tests or a user-managed conda environment.
    _default_python="$(dirname -- "$REPOSITORY_ROOT")/envs/irepa-py310-cu126-20261015/bin/python"
    if [[ -x "$_default_python" ]]; then
        PYTHON_BIN="$_default_python"
    else
        PYTHON_BIN="python"
    fi
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
[[ -n "$PYTHON_BIN" ]] || { echo "Python not found; set PYTHON_BIN" >&2; exit 1; }
PYTHON_BIN="$(realpath -- "$PYTHON_BIN")"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-$REPOSITORY_ROOT/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-$REPOSITORY_ROOT/fid_stats}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
export DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}"
export DINO_MODEL_NAME="${DINO_MODEL_NAME:-dinov3_vitb16}"
export DINO_WEIGHTS_PATH="${DINO_WEIGHTS_PATH:-$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"

# Resolve relative paths once, before a launcher changes directory.
export PRETRAIN_MODELS_ROOT="$(realpath -m -- "$PRETRAIN_MODELS_ROOT")"
export FID_STATS_ROOT="$(realpath -m -- "$FID_STATS_ROOT")"
export IMAGENET_ROOT="$(realpath -m -- "$IMAGENET_ROOT")"
export DINOV3_REPO="$(realpath -m -- "$DINOV3_REPO")"
export DINO_WEIGHTS_PATH="$(realpath -m -- "$DINO_WEIGHTS_PATH")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "CUDA_VISIBLE_DEVICES must be comma-separated GPU indices" >&2
    exit 2
}
IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"

cd -- "$PROJECT_ROOT"
