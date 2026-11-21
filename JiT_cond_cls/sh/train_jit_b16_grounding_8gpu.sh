#!/usr/bin/env bash
# JiT-B/16 + DINOv3-B/16 CLS block-level AdaLN（验证集真实语义，上限实验）。
# 在开发机 tmux 中运行：bash sh/train_jit_b16_grounding_8gpu.sh
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"

# ========================== 实验配置 ==========================
GPU_HOST=agent-gpu-l2-2
GPUS=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EPOCHS="${EPOCHS:-600}"
EXPERIMENT=JiT-B16-DINOv3B-CLS-BlockAdaLN-ValGrounding-B600
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
IMAGENET_PATH=/mnt/shared-storage-user/datafactory-share/gushishen/imagenet
PRETRAIN_MODELS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models
FID_STATS_ROOT=/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats
DINO_REPO="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3"
DINO_WEIGHTS="$PRETRAIN_MODELS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
# ==============================================================

source "$PROJECT_ROOT/sh/portable_env.sh"

# 本机是否真的有卡。开发机上的 `nvidia-smi -L` 会以 0 退出但输出为空，
# 只看退出码会把开发机误判成 GPU 节点，进而试图在无卡机器上训练。
has_local_gpu() { nvidia-smi -L 2>/dev/null | grep -q '^GPU'; }

# 8 卡 × 128 = 全局 batch 1024；blr 会在 main_jit.py 中按全局 batch 缩放。
CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPUS"
    main_jit.py \
    --model JiT-B/16 \
    --proj_dropout 0.0 \
    --P_mean -0.8 --P_std 0.8 \
    --img_size 256 --noise_scale 1.0 \
    --batch_size 128 --blr 5e-5 \
    --epochs "$EPOCHS" --warmup_epochs 5 \
    --gen_bsz 256 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
    --sampling_method heun --num_sampling_steps 50 \
    --eval_freq 20 --save_last_freq 5 --save_interval 20 \
    --output_dir "$OUTPUT_DIR" --resume "$OUTPUT_DIR" \
    --data_path "$IMAGENET_PATH" --online_eval \
    --use_grounding --grounding_model dinov3_vitb16 --grounding_dim 768 \
    --grounding_weights "$DINO_WEIGHTS" --grounding_repo "$DINO_REPO" \
    --grounding_source "${GROUNDING_SOURCE:-val}")

if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

# 开发机只负责 SSH；GPU 节点上的脚本保持前台运行。
# CUDA_VISIBLE_DEVICES 必须显式传给远端，否则 portable_env.sh 会把它重置成 0，
# 8 卡训练就退化到 1 张卡。
if [[ "${TRAIN_ON_GPU:-0}" != 1 && "$GPU_HOST" != local ]] && ! has_local_gpu; then
    exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "$GPU_HOST" env \
        TRAIN_ON_GPU=1 \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        GPUS="$GPUS" \
        EXPERIMENT="$EXPERIMENT" \
        OUTPUT_DIR="$OUTPUT_DIR" \
        IMAGENET_PATH="$IMAGENET_PATH" \
        PRETRAIN_MODELS_ROOT="$PRETRAIN_MODELS_ROOT" \
        FID_STATS_ROOT="$FID_STATS_ROOT" \
        DINO_REPO="$DINO_REPO" \
        DINO_WEIGHTS="$DINO_WEIGHTS" \
        GROUNDING_SOURCE="${GROUNDING_SOURCE:-val}" \
        bash "$SCRIPT_PATH" --worker
fi

# 走到这里必须确实有卡；宁可报错，也不要在 CPU 上静默开跑。
if ! has_local_gpu; then
    echo "错误：本机没有可见 GPU（nvidia-smi -L 无 GPU 行），也没能切换到 $GPU_HOST。" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
