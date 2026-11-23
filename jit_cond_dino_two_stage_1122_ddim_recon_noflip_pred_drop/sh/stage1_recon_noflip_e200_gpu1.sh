#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
EPOCHS="${EPOCHS:-200}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt}"
DINO_CFG="${DINO_CFG:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-recon-noflip-e${EPOCHS}}"
RESUME="${RESUME:-}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/stage1_recon_noflip_e200_gpu1.sh [--dry-run]' >&2
    exit 2
fi

args=(
    --dino_encoder dinov3b --epochs "$EPOCHS"
    --output_dir "$OUTPUT_DIR" --dino_stat_path "$DINO_STAT_PATH"
    --batch_size 128 --blr 5e-5 --warmup_epochs 5 --num_workers 12
    --loss_scale_dino 1.0 --recon_weight 0.1
    --recon_train_steps 4 --recon_train_batch 2
    --recon_eval_freq 20 --recon_eval_batch 8 --recon_val_expected 50000
    --steps_dino 25 --sampling_method_dino heun --cfg_dino "$DINO_CFG"
    --interval_min_dino 0.0 --interval_max_dino 1.0
    --t_mean_dino -1.2 --t_std_dino 1.0 --noise_scale_dino 1.0
    --save_last_freq 5 --save_interval 80 --dist_timeout "$DIST_TIMEOUT_MINUTES"
)
if [[ -n "$RESUME" ]]; then
    args+=(--resume "$RESUME")
fi
printf -v args_string '%q ' "${args[@]}"
printf -v output_dir_q '%q' "$OUTPUT_DIR"
remote_command="set -o pipefail; mkdir -p $output_dir_q; cd -- $(printf '%q' "$PROJECT_ROOT") && bash sh/train_dino.sh $args_string 2>&1 | tee -a $output_dir_q/train.log"
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" \
    IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}" \
    PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
    FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
    DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}" \
    bash -c "$remote_command"
