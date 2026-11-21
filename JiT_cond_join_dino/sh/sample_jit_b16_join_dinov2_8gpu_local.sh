#!/usr/bin/env bash
# JiT-B/16 + dino DiT 联合采样，grounding 编码器 = DINOv2-reg-B/14。本机 8 卡。
# 用法：CHECKPOINT=/abs/ckpt.pth bash sh/sample_jit_b16_join_dinov2_8gpu_local.sh [--dry-run]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：[CHECKPOINT=] bash sh/sample_jit_b16_join_dinov2_8gpu_local.sh [--dry-run]' >&2
    exit 2
fi

GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EXPERIMENT="${EXPERIMENT:-JiT-B16-JoinDino-dinov2b-B200}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/$EXPERIMENT/checkpoint-last.pth}"
IMAGENET_PATH="${IMAGENET_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
export IMAGENET_ROOT="$IMAGENET_PATH"
export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models}"
export FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}"
export HF_HUB_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
RDZV_PORT="${RDZV_PORT:-29552}"
CKPT_CFG="${CKPT_CFG:-2.9}"
DINO_CFG="${DINO_CFG:-1.0}"
DINO_STEPS="${DINO_STEPS:-25}"

source "$PROJECT_ROOT/sh/portable_env.sh"
CKPT_DIR="$(dirname "$CHECKPOINT")"
CKPT_TAG="$(basename "$CHECKPOINT" .pth)"
EVAL_DIR="${EVAL_DIR:-$CKPT_DIR/eval_${CKPT_TAG}_dinov2b_jitcfg${CKPT_CFG}_dinocfg${DINO_CFG}_dino${DINO_STEPS}_heun50_image50000}"

CMD=("$PYTHON_BIN" -m torch.distributed.run
    --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --nnodes=1 --nproc_per_node="$GPUS"
    main_jit.py
    --model JiT-B/16 --img_size 256 --noise_scale 1.0
    --evaluate_gen --checkpoint "$CHECKPOINT" --resume "$CKPT_DIR"
    --gen_bsz 256 --num_images 50000
    --cfg "$CKPT_CFG" --interval_min 0.1 --interval_max 1.0
    --sampling_method heun --num_sampling_steps 50
    --output_dir "$EVAL_DIR" --data_path "$IMAGENET_PATH"
    --use_grounding --grounding_dim 768
    --dino_model DinoDiT-B --dino_encoder dinov2b
    --dino_steps "$DINO_STEPS" --dino_sampling_method heun
    --dino_cfg "$DINO_CFG" --dino_interval_min 0.0 --dino_interval_max 1.0)

if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi

for path in "$CHECKPOINT" "$IMAGENET_PATH/val" \
    "$FID_STATS_ROOT/jit_in256_stats.npz" "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth" \
    "$PROJECT_ROOT/stat/dinov2b_rae_imagenet1k_stat.pt"; do
    [[ -e "$path" ]] || { echo "采样资源不存在：$path" >&2; exit 1; }
done
"$PYTHON_BIN" -c 'from transformers import Dinov2WithRegistersModel; Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-base", local_files_only=True)' \
    || { echo "缺少本地 DINOv2-reg-B 缓存" >&2; exit 1; }
cd "$PROJECT_ROOT"
mkdir -p "$EVAL_DIR"
"${CMD[@]}" 2>&1 | tee "$EVAL_DIR/sampling.log"
