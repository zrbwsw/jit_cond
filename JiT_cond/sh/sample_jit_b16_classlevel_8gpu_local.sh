#!/usr/bin/env bash
# JiT-B/16 patch-level AdaLN，class-level grounding 采样（本机 8 卡）。
# 每个类别用固定种子随机抽一张验证图过 DINOv3，该类 50 个样本共用这份 grounding，
# 只有噪声不同。其余配置与表中 instance-level 的评测一致。
# 用法：bash sh/sample_jit_b16_classlevel_8gpu_local.sh [--dry-run]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/sample_jit_b16_classlevel_8gpu_local.sh [--dry-run]' >&2
    exit 2
fi

# ========================== 采样配置 ==========================
GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/JiT-B16-DINOv3B-PatchAdaLN-ValGrounding-B600/checkpoint-101.pth}"
CLASS_REF_SEED="${CLASS_REF_SEED:-0}"
IMAGENET_PATH=/mnt/shared-storage-user/datafactory-share/gushishen/imagenet
export IMAGENET_ROOT="$IMAGENET_PATH"
export PRETRAIN_MODELS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models
export FID_STATS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats
DINO_REPO=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3
DINO_WEIGHTS="$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
RDZV_PORT="${RDZV_PORT:-29541}"
# ==============================================================

source "$PROJECT_ROOT/sh/portable_env.sh"
CKPT_DIR="$(dirname "$CHECKPOINT")"
CKPT_TAG="$(basename "$CHECKPOINT" .pth)"
EVAL_DIR="$CKPT_DIR/eval_${CKPT_TAG}_classlevel_seed${CLASS_REF_SEED}_cfg2.9_interval0p1_1p0_heun50_image50000_bsz256"

CMD=("$PYTHON_BIN" -m torch.distributed.run
    --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT"
    --nnodes=1 --nproc_per_node="$GPUS"
    main_jit.py
    --model JiT-B/16
    --img_size 256 --noise_scale 1.0
    --evaluate_gen --checkpoint "$CHECKPOINT" --resume "$CKPT_DIR"
    --gen_bsz 256 --num_images 50000
    --cfg 2.9 --interval_min 0.1 --interval_max 1.0
    --sampling_method heun --num_sampling_steps 50
    --output_dir "$EVAL_DIR" --data_path "$IMAGENET_PATH"
    --use_grounding --grounding_model dinov3_vitb16 --grounding_dim 768
    --grounding_weights "$DINO_WEIGHTS" --grounding_repo "$DINO_REPO"
    --grounding_source val
    --val_grounding_level class --class_ref_seed "$CLASS_REF_SEED")

if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

for path in "$CHECKPOINT" "$IMAGENET_PATH/val" "$DINO_REPO/hubconf.py" "$DINO_WEIGHTS" \
    "$FID_STATS_ROOT/jit_in256_stats.npz" "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth"; do
    [[ -e "$path" ]] || { echo "采样资源不存在：$path" >&2; exit 1; }
done

cd "$PROJECT_ROOT"
mkdir -p "$EVAL_DIR"
"${CMD[@]}" 2>&1 | tee "$EVAL_DIR/sampling.log"
