#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-2}"
CONDITION_MODE="${CONDITION_MODE:-clean}"
NOISE_BETA_DINO_PIXEL="${NOISE_BETA_DINO_PIXEL:-0.25}"
CONDITION_SUFFIX=""
if [[ "$CONDITION_MODE" != clean ]]; then
    CONDITION_SUFFIX="-$CONDITION_MODE"
fi
EXP_NAME="${EXP_NAME:-raev2-dinov3l-k7-pixeldit2-h16${CONDITION_SUFFIX}-e160}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/$EXP_NAME}"
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
EVAL_FREQ="${EVAL_FREQ:-20}"
SAVE_LAST_FREQ="${SAVE_LAST_FREQ:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo '用法：bash sh/train_pixel_gpu2.sh [--dry-run] [额外 main_jit.py 参数]'
    echo '默认：agent-gpu-l2-2、8 卡、PixelDiT2-H/16、160 epoch、clean grounding；每 20 epoch 评估并保存编号 ckpt，每 epoch 刷新 checkpoint-last.pth。'
    echo '加噪：CONDITION_MODE=noised NOISE_BETA_DINO_PIXEL=0.25 bash sh/train_pixel_gpu2.sh'
    echo 'DDT 单步预测：CONDITION_MODE=pred NOISE_BETA_DINO_PIXEL=0.25 bash sh/train_pixel_gpu2.sh'
    exit 0
fi

train_args=(--output_dir "$OUTPUT_DIR" --condition_mode_pixel "$CONDITION_MODE" --noise_beta_dino_pixel "$NOISE_BETA_DINO_PIXEL" --online_eval --eval_freq "$EVAL_FREQ" --save_last_freq "$SAVE_LAST_FREQ" --save_interval "$SAVE_INTERVAL" --batch_size "$BATCH_SIZE" --accum_steps "$ACCUM_STEPS" "$@")
printf -v train_args_string '%q ' "${train_args[@]}"
printf -v output_dir_q '%q' "$OUTPUT_DIR"
printf -v project_root_q '%q' "$PROJECT_ROOT"
remote_command="set -o pipefail; mkdir -p $output_dir_q; cd -- $project_root_q && bash sh/train_pixel.sh $train_args_string 2>&1 | tee -a $output_dir_q/train.log"
ssh_args=()
if (( dry_run )); then
    ssh_args+=(--dry-run)
fi

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
