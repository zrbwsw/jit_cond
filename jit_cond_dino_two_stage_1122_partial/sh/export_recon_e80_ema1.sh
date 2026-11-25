#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DINO_CHECKPOINT="${DINO_CHECKPOINT:-${CHECKPOINT:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-recon-noflip-e200/checkpoint-80.pth}}"
export CACHE_DIR="${CACHE_DIR:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet/stage1-dinov3b-dinodit-b-recon-noflip-e80}"
exec bash "$PROJECT_ROOT/sh/export_dino_latents.sh" "$@"
