#!/usr/bin/env bash
# 两阶段重建评估：DinoDiT 重建 DINO 特征 + Pixel JiT 重建 RGB 像素图。
# 从本目录运行：bash sh/eval_pixel_recon.sh [额外参数]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/sh/launch_common.sh"

# ---- 必填：外部传入或在此覆盖 ----
DINO_CHECKPOINT="${DINO_CHECKPOINT:-}"
PIXEL_CHECKPOINT="${PIXEL_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/eval_out}"

if [[ -z "$DINO_CHECKPOINT" || -z "$PIXEL_CHECKPOINT" ]]; then
  echo "Usage: DINO_CHECKPOINT=<s1.pth> PIXEL_CHECKPOINT=<s2.pth> bash $0"
  exit 1
fi

torchrun \
  --nproc_per_node="${GPUS:-8}" \
  --master_port="${MASTER_PORT:-29501}" \
  "$PROJECT_ROOT/eval_pixel_recon.py" \
  --dino_checkpoint  "$DINO_CHECKPOINT"  --dino_weights  "${DINO_WEIGHTS:-ema1}" \
  --pixel_checkpoint "$PIXEL_CHECKPOINT" --pixel_weights "${PIXEL_WEIGHTS:-ema1}" \
  --data_path        "${IMAGENET_ROOT}"  --split         "${SPLIT:-val}" \
  --dino_stat_path   "${STAT}"           --dinov3_weights "${DINOV3_WEIGHTS:-}" \
  --num_images       "${NUM_IMAGES:-2000}" \
  --eval_batch_size  "${EVAL_BSZ:-8}" \
  --save_images      "${SAVE_IMAGES:-64}" \
  --steps_dino       "${STEPS_DINO:-25}" --sampling_method_dino heun \
  --steps_pixel      "${STEPS_PIXEL:-50}" --sampling_method_pixel heun \
  --cfg_dino         "${CFG_DINO:-1.0}" \
  --cfg_pixel        "${CFG_PIXEL:-1.0}" \
  --output_dir       "$OUTPUT_DIR" \
  "$@"
