#!/usr/bin/env bash
# JiT-B/16 + dino DiT 联合训练，grounding 编码器 = DINOv2-reg-B/14。本机 8 卡。
# 用法：bash sh/train_jit_b16_join_dinov2_8gpu_local.sh [--dry-run]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/train_jit_b16_join_dinov2_8gpu_local.sh [--dry-run]' >&2
    exit 2
fi

GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EXPERIMENT="${EXPERIMENT:-JiT-B16-JoinDino-dinov2b-B200}"
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
IMAGENET_PATH="${IMAGENET_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
export IMAGENET_ROOT="$IMAGENET_PATH"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}"
export DINOV2_MODEL_PATH="${DINOV2_MODEL_PATH:-$PRETRAIN_MODELS_ROOT/dinov2-with-registers-base}"
export HF_HUB_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
EPOCHS="${EPOCHS:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
SAVE_LAST_FREQ="${SAVE_LAST_FREQ:-5}"
RDZV_PORT="${RDZV_PORT:-29551}"

source "$PROJECT_ROOT/sh/portable_env.sh"

CMD=("$PYTHON_BIN" -m torch.distributed.run
    --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --nnodes=1 --nproc_per_node="$GPUS"
    main_jit.py
    --model JiT-B/16 --proj_dropout 0.0
    --img_size 256 --noise_scale 1.0
    --P_mean -0.8 --P_std 0.8
    --batch_size 128 --blr 5e-5
    --epochs "$EPOCHS" --warmup_epochs 5
    --gen_bsz 256 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0
    --sampling_method heun --num_sampling_steps 50
    --eval_freq 20 --save_last_freq "$SAVE_LAST_FREQ" --save_interval "$SAVE_INTERVAL"
    --output_dir "$OUTPUT_DIR" --resume "$OUTPUT_DIR"
    --data_path "$IMAGENET_PATH" --online_eval
    --use_grounding --grounding_dim 768
    --dino_model DinoDiT-B --dino_encoder dinov2b
    --D_mean -1.2 --D_std 1.0 --dino_weight 0.333 --dino_noise_beta 0.25
    --dino_steps 25 --dino_sampling_method heun
    --dino_cfg "${DINO_CFG:-1.0}" --dino_interval_min 0.0 --dino_interval_max 1.0)

if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 8, "本机训练需要 8 张可见 CUDA GPU"'
"$PYTHON_BIN" -c 'import os; from transformers import Dinov2WithRegistersModel; Dinov2WithRegistersModel.from_pretrained(os.environ["DINOV2_MODEL_PATH"], local_files_only=True)' \
    || { echo "缺少本地 DINOv2-reg-B 权重：$DINOV2_MODEL_PATH" >&2; exit 1; }
for path in "$IMAGENET_ROOT/train" "$IMAGENET_ROOT/val" \
    "$FID_STATS_ROOT/jit_in256_stats.npz" "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth" \
    "$PROJECT_ROOT/stat/dinov2b_rae_imagenet1k_stat.pt"; do
    [[ -e "$path" ]] || { echo "训练资源不存在：$path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
