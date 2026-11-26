#!/usr/bin/env bash
# H200: 8 GPUs × 128 images per GPU = 1024 images per optimizer update.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/sh/launch_common.sh"
launch_pixel --model PixelDiT2-H/16 --epochs 160 --batch_size 128 --accum_steps 1 \
    --lr 2e-4 --warmup_steps 6255 --proj_dropout 0.2 "$@"
