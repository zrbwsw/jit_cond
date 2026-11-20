#!/usr/bin/env bash
# JiT-B/16 + 独立 dino DiT 联合训练（join-dino）：本机 8 张 GPU。
# JiT 的 grounding 由 dino DiT 生成的 DINOv2 latent 提供，c = t + y + t_dino + W_g·grounding。
# 两个网络参数独立（约 264M 可训练），冻结 DINOv2 只做训练目标编码。
# 用法：bash sh/train_jit_b16_join_dino_8gpu_local.sh [--dry-run]
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/train_jit_b16_join_dino_8gpu_local.sh [--dry-run]' >&2
    exit 2
fi

# ========================== 本机配置 ==========================
GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EXPERIMENT=JiT-B16-JoinDino-DINOv2regB-B600
OUTPUT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
IMAGENET_PATH=/data0/datasets/imagenet
export IMAGENET_ROOT="$IMAGENET_PATH"
export PRETRAIN_MODELS_ROOT=/data0/pretrain_models
export FID_STATS_ROOT=/data0/zhangruobei/Work/Pixel_Diffusion/fid_stats
# DINOv2-reg-B 从 HF 本地缓存加载（已缓存，不联网）。
export HF_HUB_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-/data0/zhangruobei/anaconda3/envs/irepa/bin/python}"
RDZV_PORT="${RDZV_PORT:-29551}"
# ==============================================================

source "$PROJECT_ROOT/sh/portable_env.sh"

# JiT path 沿用 JiT_cond 的超参（P_mean/std=-0.8/0.8）；dino path 的 t 调度 follow LF
# （D_mean/std=-1.2/1.0），dino_weight=0.333、cascaded_noised β=0.25。
CMD=("$PYTHON_BIN" -m torch.distributed.run
    --rdzv-backend=c10d --rdzv-endpoint="127.0.0.1:$RDZV_PORT" --nnodes=1 --nproc_per_node="$GPUS"
    main_jit.py
    --model JiT-B/16 --proj_dropout 0.0
    --img_size 256 --noise_scale 1.0
    --P_mean -0.8 --P_std 0.8
    --batch_size 128 --blr 5e-5
    --epochs 600 --warmup_epochs 5
    --gen_bsz 256 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0
    --sampling_method heun --num_sampling_steps 50
    --eval_freq 20 --save_last_freq 5 --save_interval 20
    --output_dir "$OUTPUT_DIR" --resume "$OUTPUT_DIR"
    --data_path "$IMAGENET_PATH" --online_eval
    --use_grounding --grounding_dim 768
    --dino_model DinoDiT-B --dinov2_path facebook/dinov2-with-registers-base
    --D_mean -1.2 --D_std 1.0 --dino_weight 0.333 --dino_noise_beta 0.25
    --dino_steps 25 --dino_sampling_method heun
    --dino_cfg "${DINO_CFG:-1.0}" --dino_interval_min 0.0 --dino_interval_max 1.0)

if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 8, "本机训练需要 8 张可见 CUDA GPU"'
"$PYTHON_BIN" -c 'from transformers import Dinov2WithRegistersModel; Dinov2WithRegistersModel.from_pretrained("facebook/dinov2-with-registers-base", local_files_only=True)' \
    || { echo "缺少本地 DINOv2-reg-B 缓存，请先下载 facebook/dinov2-with-registers-base" >&2; exit 1; }
for path in "$IMAGENET_ROOT/train" "$IMAGENET_ROOT/val" \
    "$FID_STATS_ROOT/jit_in256_stats.npz" "$FID_STATS_ROOT/weights-inception-2015-12-05-6726825d.pth" \
    "$PROJECT_ROOT/latent_force/dinov2_wreg_base_imagenet1k_stat.pt"; do
    [[ -e "$path" ]] || { echo "训练资源不存在：$path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
