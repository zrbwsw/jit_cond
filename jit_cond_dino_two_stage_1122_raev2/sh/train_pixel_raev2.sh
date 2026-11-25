#!/usr/bin/env bash
# 训练 Pixel JiT（使用 RAEv2 DINOv3-L k=7 encoder）
set -euo pipefail

cd "$(dirname "$0")/.."

# 路径配置（需在 H200 上修改）
IMAGENET="/path/to/imagenet"
STAT_PATH="stat/dinov3l_k7_imagenet1k_stats.pt"  # 需从 HuggingFace 下载
DINOV3_WEIGHTS="/data0/pretrain_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINOV3_REPO="/path/to/dinov3"  # DINOv3 源码目录

# 训练参数
EPOCHS=160
BATCH_SIZE=128  # per GPU
LR=1e-4

# Pixel JiT 参数（与原版 JiT 一致）
MODEL="JiT-B/16"
STEPS_PIXEL=50
CFG_PIXEL=2.9

# 分布式训练（8 卡）
torchrun --standalone --nproc_per_node=8 main_jit.py \
    --data_path "$IMAGENET" \
    --dino_stat_path "$STAT_PATH" \
    --dino_encoder_weights_path "$DINOV3_WEIGHTS" \
    --dino_encoder_repo "$DINOV3_REPO" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --steps_pixel "$STEPS_PIXEL" \
    --cfg_pixel "$CFG_PIXEL" \
    --online_eval --eval_freq 20 \
    "$@"
