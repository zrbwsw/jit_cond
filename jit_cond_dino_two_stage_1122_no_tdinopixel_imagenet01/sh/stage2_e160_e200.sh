#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
CONDITION_MODE="${CONDITION_MODE:-pred}"
EPOCHS="${EPOCHS:-200}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-$PROJECT_ROOT/../jit_cond_dino_two_stage_1122/checkpoints/stage1-dinov3b-dinodit-b-e160/checkpoint-160.pth}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-${CONDITION_MODE}-e${EPOCHS}}"
if [[ "$CONDITION_MODE" != clean && "$CONDITION_MODE" != noised && "$CONDITION_MODE" != pred ]]; then
    echo 'CONDITION_MODE 必须为 clean、noised 或 pred' >&2
    exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：CONDITION_MODE=clean|noised|pred GPU_HOST=agent-gpu-l2-1|agent-gpu-l2-2 bash sh/stage2_e160_e200.sh [--dry-run]' >&2
    exit 2
fi

init_args=()
if [[ -n "${RESUME:-}" ]]; then
    init_args+=(--resume "$RESUME")
else
    init_args+=(--dino_checkpoint "$DINO_CHECKPOINT" --dino_weights ema1)
fi

args=(
    --epochs "$EPOCHS" --condition_mode_pixel "$CONDITION_MODE"
    --model JiT-B/16 --dino_model DinoDiT-B --dino_encoder dinov3b
    "${init_args[@]}" --dino_stat_path "$DINO_STAT_PATH"
    --batch_size 128 --blr 5e-5 --warmup_epochs 5
    --noise_beta_dino_pixel 0.25 --t_schedule_dino_pixel uniform
    --t_mean_pixel -0.8 --t_std_pixel 0.8 --t_mean_dino -1.2 --t_std_dino 1.0
    --noise_scale_pixel 1.0 --noise_scale_dino 1.0
    --steps_dino 25 --steps_pixel 50 --sampling_method_dino heun --sampling_method_pixel heun
    --cfg_dino 2.0 --cfg_pixel 2.9
    --interval_min_dino 0.0 --interval_max_dino 1.0
    --interval_min_pixel 0.1 --interval_max_pixel 1.0
    --gen_bsz 256 --num_images 50000 --num_workers 12
    --eval_freq 20 --save_last_freq 5 --save_interval 80 --online_eval
    --output_dir "$OUTPUT_DIR"
)
printf -v args_string '%q ' "${args[@]}"
printf -v output_dir_q '%q' "$OUTPUT_DIR"
remote_command="set -o pipefail; mkdir -p $output_dir_q; cd -- $(printf '%q' "$PROJECT_ROOT") && bash sh/train_pixel.sh $args_string 2>&1 | tee -a $output_dir_q/train.log"
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" \
    IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}" \
    PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
    FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
    DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}" \
    bash -c "$remote_command"
