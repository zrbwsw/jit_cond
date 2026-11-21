#!/usr/bin/env bash
# JiT-B/16：当前噪声图 DINOv3-B + time-conditioned PixelDiT2 projector。
# 本机 8 卡 RTX 4090 模板：bash sh/train_jit_b16_grounding_8gpu_local.sh [--dry-run]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo 'Usage: bash sh/train_jit_b16_grounding_8gpu_local.sh [--dry-run]' >&2
    exit 2
fi

# ========================== 本机配置 ==========================
# 环境变量可以覆盖以下默认路径；直接使用本机 GPU 启动分布式训练。
export PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export IMAGENET_PATH="${IMAGENET_PATH:-${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}}"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}"
export DINO_REPO="${DINO_REPO:-${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}}"
export DINO_WEIGHTS="${DINO_WEIGHTS:-${DINO_WEIGHTS_PATH:-$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}}"
EXPERIMENT="${EXPERIMENT:-JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT}"
RESUME="${RESUME-$OUTPUT_DIR}"
DIST_PORT="${DIST_PORT:-29401}"
# ==============================================================

# portable_env 仅解析本机解释器和资源，并切换到项目目录。
source "$PROJECT_ROOT/sh/portable_env.sh"
IFS=, read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
GPUS="${GPUS:-${#visible_gpus[@]}}"
[[ "$GPUS" =~ ^[1-9][0-9]*$ && "$GPUS" -le "${#visible_gpus[@]}" ]] || {
    echo 'GPUS must be positive and not exceed CUDA_VISIBLE_DEVICES count.' >&2; exit 2;
}

# 保留原模板的超参数及 B80 实验名；8 卡 × 128 = 全局 batch 1024。
# blr 会在 main_jit.py 中按全局 batch 缩放。
CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --rdzv_endpoint "127.0.0.1:$DIST_PORT" --nproc_per_node="$GPUS"
    main_jit.py
    --model "${MODEL:-JiT-B/16}" --img_size "${IMG_SIZE:-256}"
    --noise_scale "${NOISE_SCALE:-1.0}" --proj_dropout 0.0 --P_mean -0.8 --P_std 0.8
    --batch_size "${BATCH_SIZE:-128}" --blr "${BLR:-5e-5}"
    --epochs "${EPOCHS:-80}" --warmup_epochs 5 --num_workers "${NUM_WORKERS:-12}"
    --gen_bsz "${GEN_BSZ:-256}" --num_images "${NUM_IMAGES:-50000}"
    --cfg 2.9 --interval_min 0.1 --interval_max 1.0
    --sampling_method heun --num_sampling_steps 50
    --eval_freq "${EVAL_FREQ:-20}" --save_last_freq "${SAVE_LAST_FREQ:-5}" --save_interval "${SAVE_INTERVAL:-20}"
    --output_dir "$OUTPUT_DIR" --resume "$RESUME" --data_path "$IMAGENET_ROOT"
    --use_grounding --grounding_model "$DINO_MODEL_NAME" --grounding_dim 768
    --grounding_weights "$DINO_WEIGHTS_PATH" --grounding_repo "$DINOV3_REPO"
    --grounding_proj_blocks "${GROUNDING_PROJ_BLOCKS:-1}"
    --ground_drop_prob "${GROUND_DROP_PROB:-0.0}")
if [[ -n "${GROUNDING_PROJ_HEADS:-}" ]]; then CMD+=(--grounding_proj_heads "$GROUNDING_PROJ_HEADS"); fi
if [[ "${ONLINE_EVAL:-1}" == 1 ]]; then CMD+=(--online_eval); fi
if [[ "${1:-}" == --dry-run ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
[[ -d "$IMAGENET_ROOT/train" ]] || { echo "Training data missing: $IMAGENET_ROOT/train" >&2; exit 1; }
[[ -f "$DINO_WEIGHTS_PATH" ]] || { echo "DINO weights missing: $DINO_WEIGHTS_PATH" >&2; exit 1; }
[[ -f "$DINOV3_REPO/hubconf.py" ]] || { echo "DINO source missing: $DINOV3_REPO" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
