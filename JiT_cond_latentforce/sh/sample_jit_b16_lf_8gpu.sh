#!/usr/bin/env bash
# JiT-B/16 + Latent Forcing 语义条件的推理脚本。
#
# 与 sample_jit_b16_grounding_8gpu.sh 的唯一区别是 grounding 来源：
# 训练好的 JiT 权重不变，采样时把"clean image → DINOv3 patch token"换成
# "Latent Forcing 预训练权重采样 25 步 latent → DINOv2 patch token"。
# 训练路径完全未改动。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/sample_jit_b16_lf_8gpu.sh [--dry-run]' >&2
    exit 2
fi

# ========================== 推理配置 ==========================
GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# patch-level AdaLN 的训练权重（grounding_proj 学的是 DINOv3 patch token 空间）
CHECKPOINT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond/checkpoints/JiT-B16-DINOv3B-PatchAdaLN-ValGrounding-B600/checkpoint-100.pth
IMAGENET_PATH=/mnt/shared-storage-user/datafactory-share/gushishen/imagenet
export IMAGENET_ROOT="$IMAGENET_PATH"
export PRETRAIN_MODELS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models
export FID_STATS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats
DINO_REPO=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3
DINO_WEIGHTS="$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

# Latent Forcing：只用前 25 步推进 latent（50 步级联调度的前半段）。
LF_CKPT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/LatentForcing/lfjit-l-200.pth
LF_EMA=1                  # README 默认使用 model_ema1
LF_STEPS=25
LF_METHOD=heun            # 25 步 = 50 次网络评估
LF_CFG="${LF_CFG:-1.0}"   # LF 内部采样 latent 的 CFG：1.0=不加引导；可用环境变量覆盖
LF_GUIDANCE="${LF_GUIDANCE:-none}"  # none / cfg；与 LF_CFG 配套
LF_SHIFT_DINO=1.0
LF_GEN_BSZ=128            # latent 采样分块，控制显存
CKPT_CFG=2.9              # JiT 采样本身的 CFG，沿用表中 patch-level 的设置
TAG="${TAG:-lf${LF_STEPS}_ema${LF_EMA}_lfcfg${LF_CFG}}"
# ==============================================================

source "$PROJECT_ROOT/sh/portable_env.sh"
CKPT_DIR="$(dirname "$CHECKPOINT")"
CKPT_TAG="$(basename "$CHECKPOINT" .pth)"
EVAL_DIR="$CKPT_DIR/eval_${TAG}_cfg${CKPT_CFG}_interval0p1_1p0_heun50_image50000_bsz256"

CMD=("$PYTHON_BIN" -m torch.distributed.run
    --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:${RDZV_PORT:-29531}"
    --nnodes=1 --nproc_per_node="$GPUS"
    main_jit.py
    --model JiT-B/16
    --img_size 256 --noise_scale 1.0
    --evaluate_gen --checkpoint "$CHECKPOINT" --resume "$CKPT_DIR"
    --gen_bsz 256 --num_images 50000
    --cfg "$CKPT_CFG" --interval_min 0.1 --interval_max 1.0
    --sampling_method heun --num_sampling_steps 50
    --output_dir "$EVAL_DIR" --data_path "$IMAGENET_PATH"
    --use_grounding --grounding_dim 768
    --grounding_weights "$DINO_WEIGHTS" --grounding_repo "$DINO_REPO"
    --grounding_source lf
    --lf_ckpt "$LF_CKPT" --lf_ema "$LF_EMA"
    --lf_latent_steps "$LF_STEPS" --lf_sampling_method "$LF_METHOD"
    --lf_cfg "$LF_CFG" --lf_guidance "$LF_GUIDANCE" --lf_gen_shift_dino "$LF_SHIFT_DINO"
    --lf_gen_bsz "$LF_GEN_BSZ")

if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

for path in "$CHECKPOINT" "$LF_CKPT" "$IMAGENET_PATH/val" "$DINO_REPO/hubconf.py" \
    "$DINO_WEIGHTS" "$FID_STATS_ROOT/jit_in256_stats.npz" \
    "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth"; do
    [[ -e "$path" ]] || { echo "推理资源不存在：$path" >&2; exit 1; }
done
[[ -d "$PROJECT_ROOT/latent_force" ]] || { echo "缺少 $PROJECT_ROOT/latent_force" >&2; exit 1; }

cd "$PROJECT_ROOT"
mkdir -p "$EVAL_DIR"
"${CMD[@]}" 2>&1 | tee "$EVAL_DIR/sampling.log"
