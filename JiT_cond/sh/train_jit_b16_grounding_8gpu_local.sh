#!/usr/bin/env bash
# JiT-B/16 + DINOv3-B/16 patch-level AdaLN：本机 8 张 RTX 4090 训练模板。
# 在本机运行：bash sh/train_jit_b16_grounding_8gpu_local.sh [--dry-run]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/train_jit_b16_grounding_8gpu_local.sh [--dry-run]' >&2
    exit 2
fi

# ========================== 本机配置 ==========================
GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EXPERIMENT=JiT-B16-DINOv3B-PatchAdaLN-ValGrounding-B600
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
IMAGENET_PATH=/data0/datasets/imagenet
export IMAGENET_ROOT="$IMAGENET_PATH"
export PRETRAIN_MODELS_ROOT=/data0/pretrain_models
export FID_STATS_ROOT=/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats
DINO_REPO=/data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_rmsnorm2_Residual_1016_normft_1028/SSL_code/dinov3
DINO_WEIGHTS="$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
export DINOV3_REPO="$DINO_REPO"
export DINO_WEIGHTS_PATH="$DINO_WEIGHTS"
PYTHON_BIN="${PYTHON_BIN:-/data0/zhangruobei/anaconda3/envs/irepa/bin/python}"
# ==============================================================

# 复用环境变量和路径解析，随后直接在本机启动 torchrun。
source "$PROJECT_ROOT/sh/portable_env.sh"

# 沿用原脚本超参数：8 卡 × 128 = 全局 batch 1024，600 epochs。
CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPUS"
    main_jit.py
    --model JiT-B/16
    --proj_dropout 0.0
    --P_mean -0.8 --P_std 0.8
    --img_size 256 --noise_scale 1.0
    --batch_size 128 --blr 5e-5
    --epochs 600 --warmup_epochs 5
    --gen_bsz 256 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0
    --sampling_method heun --num_sampling_steps 50
    --eval_freq 20 --save_last_freq 5 --save_interval 20
    --output_dir "$OUTPUT_DIR" --resume "$OUTPUT_DIR"
    --data_path "$IMAGENET_PATH" --online_eval
    --use_grounding --grounding_model dinov3_vitb16 --grounding_dim 768
    --grounding_weights "$DINO_WEIGHTS" --grounding_repo "$DINO_REPO"
    --grounding_source "${GROUNDING_SOURCE:-val}")

# 只打印命令；不启动训练，也不创建实验输出目录。
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 8, "本机训练需要 8 张可见 CUDA GPU"'
for path in "$IMAGENET_ROOT/train" "$IMAGENET_ROOT/val" "$DINOV3_REPO/hubconf.py" \
    "$DINO_WEIGHTS_PATH" "$FID_STATS_ROOT/jit_in256_stats.npz" \
    "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth"; do
    [[ -e "$path" ]] || { echo "训练资源不存在：$path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"

