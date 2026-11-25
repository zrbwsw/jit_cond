#!/usr/bin/env bash
# Stage 2：加载 Stage 1 权重，冻结 DinoDiT，单独训练 Pixel JiT。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/sh/launch_common.sh"
run_torch main_jit.py --dino_encoder "${DINO_ENCODER:-dinov3b}" "$@"
