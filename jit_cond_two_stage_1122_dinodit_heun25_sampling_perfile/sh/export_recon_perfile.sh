#!/usr/bin/env bash
# 导出 per-file grounding cache (version 4: free sampling)。
# 用法：DINO_CHECKPOINT=<ckpt> CACHE_DIR=<dir> bash sh/export_recon_perfile.sh
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/sh/launch_common.sh"

DINO_CHECKPOINT="${DINO_CHECKPOINT:-}"
CACHE_DIR="${CACHE_DIR:-}"

if [[ -z "$DINO_CHECKPOINT" || -z "$CACHE_DIR" ]]; then
  echo "Usage: DINO_CHECKPOINT=<ckpt> CACHE_DIR=<dir> bash $0"
  exit 1
fi

torchrun \
  --nproc_per_node="${GPUS:-8}" \
  --master_port="${MASTER_PORT:-29502}" \
  -m stage1.export_recon \
  --dino_checkpoint "$DINO_CHECKPOINT" \
  --dino_weights    "${DINO_WEIGHTS:-ema1}" \
  --cache_dir       "$CACHE_DIR" \
  --split           "${SPLIT:-train}" \
  --dino_stat_path  "${STAT}" \
  --steps_dino      "${STEPS_DINO:-25}" \
  --sampling_method_dino heun \
  --cfg_dino        "${CFG_DINO:-2.0}" \
  --seed_base       "${SEED_BASE:-42}" \
  --batch_size      "${BSZ:-8}" \
  "$@"
