#!/usr/bin/env bash
# JiT-B/16 + DINOv3-B/16 CLS block-level AdaLN sampling.
# 采样配置集中在顶部，--dry-run 只打印最终命令。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"

# ========================== 采样配置 ==========================
GPU_HOST=agent-gpu-l2-2
GPUS=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CHECKPOINT="$PROJECT_ROOT/checkpoints/JiT-B16-DINOv3B-CLS-BlockAdaLN-ValGrounding-B600/checkpoint-last.pth"
IMAGENET_PATH=/mnt/shared-storage-user/datafactory-share/gushishen/imagenet
PRETRAIN_MODELS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models
FID_STATS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats
DINO_REPO="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3"
DINO_WEIGHTS="$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# ===============================================================

source "$PROJECT_ROOT/sh/portable_env.sh"
has_local_gpu() { nvidia-smi -L 2>/dev/null | grep -q '^GPU'; }

CKPT_DIR="$(dirname "$CHECKPOINT")"
CKPT_TAG="$(basename "$CHECKPOINT" .pth)"
EVAL_DIR="$CKPT_DIR/eval_${CKPT_TAG}_cfg2p9_interval0p1_1p0_heun50_image50000_gpus${GPUS}_bsz256"

CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPUS"
    main_jit.py \
    --model JiT-B/16 \
    --img_size 256 --noise_scale 1.0 \
    --evaluate_gen --checkpoint "$CHECKPOINT" --resume "$CKPT_DIR" \
    --gen_bsz 256 --num_images 50000 \
    --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
    --sampling_method heun --num_sampling_steps 50 \
    --output_dir "$EVAL_DIR" --data_path "$IMAGENET_PATH" \
    --use_grounding --grounding_model dinov3_vitb16 --grounding_dim 768 \
    --grounding_weights "$DINO_WEIGHTS" --grounding_repo "$DINO_REPO" \
    --grounding_source "${GROUNDING_SOURCE:-val}")

if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

if [[ "${SAMPLE_ON_GPU:-0}" != 1 && "$GPU_HOST" != local ]] && ! has_local_gpu; then
    exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "$GPU_HOST" env \
        SAMPLE_ON_GPU=1 bash "$SCRIPT_PATH" --worker
fi

export CUDA_VISIBLE_DEVICES
cd "$PROJECT_ROOT"
mkdir -p "$EVAL_DIR"
"${CMD[@]}" 2>&1 | tee "$EVAL_DIR/sampling.log"
