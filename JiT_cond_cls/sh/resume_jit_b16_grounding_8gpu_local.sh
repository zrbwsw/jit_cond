#!/usr/bin/env bash
# JiT-B/16 + DINOv3-B/16 CLS block-level AdaLN：本机 8 卡，自动恢复 checkpoint 进度并续训至 600 epochs。
# 在本机运行：bash sh/resume_jit_b16_grounding_8gpu_local.sh [--dry-run]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/resume_jit_b16_grounding_8gpu_local.sh [--dry-run]' >&2
    exit 2
fi

# ========================== 本机配置 ==========================
GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EXPERIMENT=JiT-B16-DINOv3B-CLS-BlockAdaLN-ValGrounding-B600
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
# 下载文件实际名为 checkpoint-last.pth；续训时会定期更新同名文件。
# 后续重启自动接着最新进度；CHECKPOINT=/absolute/path/to/checkpoint 可显式指定文件。
RESUME_CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoint-last.pth}"
IMAGENET_PATH=/mnt/shared-storage-user/datafactory-share/gushishen/imagenet
export IMAGENET_ROOT="$IMAGENET_PATH"
export PRETRAIN_MODELS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models
export FID_STATS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats
DINO_REPO=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3
DINO_WEIGHTS="$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
export DINOV3_REPO="$DINO_REPO"
export DINO_WEIGHTS_PATH="$DINO_WEIGHTS"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
# ==============================================================

# 复用环境变量和路径解析，随后直接在本机启动 torchrun。
source "$PROJECT_ROOT/sh/portable_env.sh"

# 沿用原脚本超参数：8 卡 × 128 = 全局 batch 1024，总训练目标仍为 600 epochs。
# start_epoch 由 checkpoint 恢复，不手动指定或重置。
# 本次下载的 epoch=40 是旧格式的零基索引：Adam step=51291=41×1251，
# 实际已完成 41 轮，因此从索引 41（第 42 轮）继续；后续 checkpoint 自动恢复各自进度。
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
    --output_dir "$OUTPUT_DIR" --resume "$OUTPUT_DIR" --checkpoint "$RESUME_CHECKPOINT"
    --data_path "$IMAGENET_PATH" --online_eval
    --use_grounding --grounding_model dinov3_vitb16 --grounding_dim 768
    --grounding_weights "$DINO_WEIGHTS" --grounding_repo "$DINO_REPO"
    --grounding_source "${GROUNDING_SOURCE:-val}")

# 只打印命令；不启动训练，也不创建实验输出目录。
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

# 文件不存在或缺少完整续训状态时立即退出，避免 main_jit 静默从头训练。
[[ -f "$RESUME_CHECKPOINT" ]] || { echo "续训 checkpoint 不存在：$RESUME_CHECKPOINT" >&2; exit 1; }
"$PYTHON_BIN" - "$RESUME_CHECKPOINT" <<'PY_CHECKPOINT'
import sys
import torch

# 本实验由用户保存的可信 checkpoint 含 argparse.Namespace；mmap 避免预检查复制所有 tensor。
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
required = {"model", "model_ema1", "model_ema2", "optimizer", "epoch"}
missing = required.difference(checkpoint)
if missing:
    raise RuntimeError(f"Checkpoint 缺少续训状态：{sorted(missing)}")
if not checkpoint["optimizer"].get("state") or not checkpoint["optimizer"].get("param_groups"):
    raise RuntimeError("Checkpoint 中的优化器状态为空，无法完整续训")
if not checkpoint["model_ema1"] or not checkpoint["model_ema2"]:
    raise RuntimeError("Checkpoint 中的 EMA 状态为空，无法完整续训")
completed = int(checkpoint["epoch"])
if not hasattr(checkpoint.get("args"), "save_interval"):
    completed += 1
if not 0 < completed < 600:
    raise RuntimeError(f"需要已完成 1 至 599 epochs 的 checkpoint，实际已完成 {completed}")
print(f"续训文件：{sys.argv[1]}")
print(f"已完成 {completed} epochs；从索引 {completed}（第 {completed + 1} 轮）继续，目标总计 600 epochs。")
PY_CHECKPOINT

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 8, "本机训练需要 8 张可见 CUDA GPU"'
for path in "$IMAGENET_ROOT/train" "$IMAGENET_ROOT/val" "$DINOV3_REPO/hubconf.py" \
    "$DINO_WEIGHTS_PATH" "$FID_STATS_ROOT/jit_in256_stats.npz" \
    "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth"; do
    [[ -e "$path" ]] || { echo "训练资源不存在：$path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train_resume_local.log"

