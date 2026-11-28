#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/raev2-dinov3l-k7-pixeldit2-h16-e160/checkpoint-60.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT")/eval_checkpoint-60_keep_grounding_jitcfg2.4_raev2_heun100_pixelheun50_image50000}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/RAEv2/stage1/imagenet/dinov3l-k7/stats.pt}"
DINO_ENCODER_WEIGHTS_PATH="${DINO_ENCODER_WEIGHTS_PATH:-$PRETRAIN_MODELS_ROOT/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
REPA_ENCODER_WEIGHTS_PATH="${REPA_ENCODER_WEIGHTS_PATH:-$PRETRAIN_MODELS_ROOT/dinov2_vitb14_pretrain.pth}"
DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}"
RAEV2_ROOT="${RAEV2_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/RAEv2}"
RAEV2_CHECKPOINT="${RAEV2_CHECKPOINT:-$RAEV2_ROOT/stage2/imagenet/dinov3l-k7/checkpoint.pt}"
RAEV2_CONFIG="${RAEV2_CONFIG:-$RAEV2_ROOT/configs/stage2/sampling/imagenet-dinov3l-k7.yaml}"
FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
GEN_BSZ="${GEN_BSZ:-64}"

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != "--dry-run" ) ]]; then
    echo '用法：bash sh/sample_e60_keep_grounding_gpu1.sh [--dry-run]' >&2
    exit 2
fi
if [[ "${1:-}" != "--dry-run" && ! -f "$CHECKPOINT" ]]; then
    echo "Pixel checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

args=(--checkpoint "$CHECKPOINT" --output_dir "$OUTPUT_DIR" --num_images "$NUM_IMAGES" --gen_bsz "$GEN_BSZ")
printf -v args_string '%q ' "${args[@]}"
printf -v output_dir_q '%q' "$OUTPUT_DIR"
printf -v project_root_q '%q' "$PROJECT_ROOT"
remote_command="set -o pipefail; mkdir -p $output_dir_q; cd -- $project_root_q && bash sh/sample_pixel_keep_grounding.sh $args_string 2>&1 | tee -a $output_dir_q/sampling.log"
ssh_args=()
[[ "${1:-}" != "--dry-run" ]] || ssh_args+=(--dry-run)

export PYTHONPATH="$PROJECT_ROOT:$RAEV2_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    NPROC_PER_NODE="$NPROC_PER_NODE" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    IMAGENET_ROOT="$IMAGENET_ROOT" PRETRAIN_MODELS_ROOT="$PRETRAIN_MODELS_ROOT" \
    DINO_STAT_PATH="$DINO_STAT_PATH" \
    DINO_ENCODER_WEIGHTS_PATH="$DINO_ENCODER_WEIGHTS_PATH" \
    REPA_ENCODER_WEIGHTS_PATH="$REPA_ENCODER_WEIGHTS_PATH" \
    DINOV3_REPO="$DINOV3_REPO" \
    RAEV2_ROOT="$RAEV2_ROOT" RAEV2_CHECKPOINT="$RAEV2_CHECKPOINT" RAEV2_CONFIG="$RAEV2_CONFIG" \
    FID_STATS_ROOT="$FID_STATS_ROOT" \
    HF_HUB_OFFLINE=1 TORCH_NCCL_TRACE_BUFFER_SIZE=4096 \
    bash -c "$remote_command"
