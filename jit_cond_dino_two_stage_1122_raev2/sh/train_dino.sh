#!/usr/bin/env bash
# Stage 1：单独训练 DinoDiT，默认 DINOv3-B / 80 epochs。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/sh/launch_common.sh"
run_torch --module stage1.main_dino --dino_encoder "${DINO_ENCODER:-dinov3b}" "$@"
