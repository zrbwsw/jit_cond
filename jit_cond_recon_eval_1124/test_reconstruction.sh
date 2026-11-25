#!/usr/bin/env bash
# 测试重建逻辑：用本机已有的 checkpoint 重建 ImageNet 训练集样本
set -euo pipefail

# DINOv3 源码路径（本机）
export DINOV3_REPO=/data0/zhangruobei/Work/Pixel_Diffusion/Ours_BaseT_rmsnorm2_Residual_1016_dinoycond_consistency/SSL_code/dinov3

# 使用已有的 E160 checkpoint
DINO_CKPT="/data0/zhangruobei/Work/jit_cond_ablation/jit_cond_dino_two_stage_1122/checkpoints/stage1-dinov3b-dinodit-b-e160/checkpoint-160.pth"
PIXEL_CKPT="/data0/zhangruobei/Work/jit_cond_ablation/jit_cond_dino_two_stage_1122/checkpoints/stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-pred-e200/checkpoint-160.pth"
IMAGENET="/data0/datasets/imagenet"
STAT="/data0/zhangruobei/Work/jit_cond_ablation/jit_cond_dino_two_stage_1122/dinov3b_rae_imagenet1k_stats.pt"

cd "$(dirname "$0")"

python test_reconstruction.py \
    --dino_checkpoint "$DINO_CKPT" --dino_weights ema1 \
    --pixel_checkpoint "$PIXEL_CKPT" --pixel_weights ema1 \
    --data_path "$IMAGENET" \
    --dino_stat_path "$STAT" \
    --steps_dino 25 --sampling_method_dino heun \
    --steps_pixel 50 --sampling_method_pixel heun \
    --cfg_dino 1.0 --cfg_pixel 1.0 \
    --num_samples 8 --seed 42 \
    --output_dir test_reconstruction_output
