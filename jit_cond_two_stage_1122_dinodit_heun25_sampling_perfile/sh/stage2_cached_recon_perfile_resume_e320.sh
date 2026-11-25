#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
EPOCHS="${EPOCHS:-320}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt}"
GROUNDING_CACHE="${GROUNDING_CACHE:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet/stage1-dinov3b-dinodit-b-recon-noflip-e80}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-recon-noflip-e80-stage2-pixeljit-b-dinodit-cached-recon-perfile-e320}"
RESUME="${RESUME:-$OUTPUT_DIR/checkpoint-last-perfile.pth}"
DINO_CFG="${DINO_CFG:-2.0}"
CACHE_CFG_DINO="${CACHE_CFG_DINO:-1.0}"
JIT_CFG="${JIT_CFG:-2.9}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/stage2_cached_recon_perfile_resume_e320.sh [--dry-run]' >&2
    exit 2
fi
if [[ "${1:-}" != --dry-run && ! -f "$RESUME" ]]; then
    echo "续训 checkpoint 不存在：$RESUME" >&2
    exit 1
fi

args=(
    --epochs "$EPOCHS" --condition_mode_pixel cached_recon
    --model JiT-B/16 --dino_model DinoDiT-B --dino_encoder dinov3b
    --resume "$RESUME" --dino_weights ema1
    --dino_stat_path "$DINO_STAT_PATH" --grounding_cache "$GROUNDING_CACHE"
    --cache_cfg_dino "$CACHE_CFG_DINO" --data_path "$IMAGENET_ROOT" --output_dir "$OUTPUT_DIR"
    --batch_size 128 --blr 5e-5 --warmup_epochs 5 --num_workers 12
    --noise_beta_dino_pixel 0.25 --t_schedule_dino_pixel uniform
    --t_mean_pixel -0.8 --t_std_pixel 0.8 --t_mean_dino -1.2 --t_std_dino 1.0
    --noise_scale_pixel 1.0 --noise_scale_dino 1.0
    --steps_dino 25 --steps_pixel 50 --sampling_method_dino heun --sampling_method_pixel heun
    --cfg_dino "$DINO_CFG" --cfg_pixel "$JIT_CFG"
    --interval_min_dino 0.0 --interval_max_dino 1.0
    --interval_min_pixel 0.1 --interval_max_pixel 1.0
    --gen_bsz 256 --num_images 50000 --eval_freq 20
    --save_last_freq 5 --save_interval 80 --dist_timeout "$DIST_TIMEOUT_MINUTES"
    --online_eval
)
printf -v args_string '%q ' "${args[@]}"
printf -v output_dir_q '%q' "$OUTPUT_DIR"
printf -v project_root_q '%q' "$PROJECT_ROOT"
remote_command="set -o pipefail; mkdir -p $output_dir_q; cd -- $project_root_q && bash sh/train_pixel.sh $args_string 2>&1 | tee -a $output_dir_q/train.log"
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    IMAGENET_ROOT="$IMAGENET_ROOT" \
    PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
    FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
    DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}" \
    HF_HUB_OFFLINE=1 TORCH_NCCL_TRACE_BUFFER_SIZE=4096 \
    bash -c "$remote_command"
