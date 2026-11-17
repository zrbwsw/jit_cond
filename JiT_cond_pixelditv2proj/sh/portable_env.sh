#!/usr/bin/env bash
# Resolve local runtime and assets. Sourcing this file never starts training or SSH.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(dirname -- "$PROJECT_ROOT")"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in \
        "$HOME/anaconda3/envs/irepa/bin/python" \
        "$(dirname -- "$REPOSITORY_ROOT")/envs/irepa-py310-cu126-20261015/bin/python"; do
        if [[ -x "$candidate" ]]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN" 2>/dev/null || true)"
[[ -n "$PYTHON_BIN" ]] || { echo 'Python not found; set PYTHON_BIN.' >&2; return 1; }
export PYTHON_BIN="$(realpath -- "$PYTHON_BIN")"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1

# Prefer assets already installed on this server; explicit environment values win.
if [[ -z "${PRETRAIN_MODELS_ROOT:-}" && -d /data0/pretrain_models ]]; then
    export PRETRAIN_MODELS_ROOT=/data0/pretrain_models
fi
if [[ -n "${IMAGENET_PATH:-}" ]]; then
    export IMAGENET_ROOT="$IMAGENET_PATH"
elif [[ -z "${IMAGENET_ROOT:-}" && -d /data0/datasets/imagenet ]]; then
    export IMAGENET_ROOT=/data0/datasets/imagenet
fi
if [[ -n "${DINO_REPO:-}" ]]; then
    export DINOV3_REPO="$DINO_REPO"
fi
if [[ -n "${DINO_WEIGHTS:-}" ]]; then
    export DINO_WEIGHTS_PATH="$DINO_WEIGHTS"
fi

# The same resolver is used by Python; no dataset or weight is loaded here.
resolved_paths="$("$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from resource_paths import dinov3_repo, fid_stats_root, imagenet_root, pretrain_models_root

print(pretrain_models_root())
print(fid_stats_root())
print(imagenet_root())
print(dinov3_repo())
print(Path(os.environ.get('DINO_WEIGHTS_PATH', str(pretrain_models_root() / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'))).expanduser().resolve())
PY
)"
mapfile -t asset_paths <<< "$resolved_paths"
export PRETRAIN_MODELS_ROOT="${asset_paths[0]}"
export FID_STATS_ROOT="${asset_paths[1]}"
export IMAGENET_ROOT="${asset_paths[2]}"
export DINOV3_REPO="${asset_paths[3]}"
export DINO_WEIGHTS_PATH="${asset_paths[4]}"
export DINO_MODEL_NAME="${DINO_MODEL_NAME:-dinov3_vitb16}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo 'CUDA_VISIBLE_DEVICES must contain comma-separated GPU indices.' >&2
    return 2
}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
cd -- "$PROJECT_ROOT"
