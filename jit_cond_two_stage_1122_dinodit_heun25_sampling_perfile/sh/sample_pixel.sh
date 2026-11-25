#!/usr/bin/env bash
# 从 Stage 2 checkpoint 采样，先生成 DINO latent，再生成 RGB。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/sh/launch_common.sh"
run_torch main_jit.py --evaluate_gen --dino_encoder "${DINO_ENCODER:-dinov3b}" "$@"
