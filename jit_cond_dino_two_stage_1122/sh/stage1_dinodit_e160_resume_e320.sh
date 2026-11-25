#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
EPOCHS="${EPOCHS:-320}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt}"
RESUME="${RESUME:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-e160/checkpoint-160.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-e160}"
DINO_CFG="${DINO_CFG:-1.0}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/stage1_dinodit_e160_resume_e320.sh [--dry-run]' >&2
    exit 2
fi

args=(
    --epochs "$EPOCHS" --resume "$RESUME" --output_dir "$OUTPUT_DIR"
    --dino_model DinoDiT-B --dino_encoder dinov3b --dino_stat_path "$DINO_STAT_PATH"
    --img_size 256 --grounding_dim 768 --class_num 1000
    --batch_size 128 --blr 5e-5 --warmup_epochs 5 --num_workers 12
    --loss_scale_dino 1.0 --t_mean_dino -1.2 --t_std_dino 1.0
    --noise_scale_dino 1.0 --steps_dino 25 --sampling_method_dino heun
    --cfg_dino "$DINO_CFG" --interval_min_dino 0.0 --interval_max_dino 1.0
    --save_last_freq 5 --save_interval 80
)
printf -v args_string '%q ' "${args[@]}"
printf -v output_dir_q '%q' "$OUTPUT_DIR"
printf -v project_root_q '%q' "$PROJECT_ROOT"
remote_command="set -o pipefail; mkdir -p $output_dir_q; cd -- $project_root_q && bash sh/train_dino.sh $args_string 2>&1 | tee -a $output_dir_q/train.log"
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
exec "$PROJECT_ROOT/../JiT_cond_join_dino/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    IMAGENET_ROOT="$IMAGENET_ROOT" \
    PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
    FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
    DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}" \
    HF_HUB_OFFLINE=1 TORCH_NCCL_TRACE_BUFFER_SIZE=4096 \
    bash -c "$remote_command"
