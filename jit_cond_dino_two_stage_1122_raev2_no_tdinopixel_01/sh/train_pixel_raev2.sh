#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
EPOCHS="${EPOCHS:-160}"
MODEL="${MODEL:-JiT-B/16}"
CONDITION_MODE="${CONDITION_MODE:-clean}"
NOISE_BETA_DINO_PIXEL="${NOISE_BETA_DINO_PIXEL:-0.25}"
CONDITION_SUFFIX=""
if [[ "$CONDITION_MODE" != clean ]]; then
    CONDITION_SUFFIX="-$CONDITION_MODE"
fi
EXP_NAME="${EXP_NAME:-stage1-raev2-dinov3l-k7-stage2-pixeljit-b${CONDITION_SUFFIX}-e${EPOCHS}}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
DINO_STAT_PATH="${DINO_STAT_PATH:-$PROJECT_ROOT/../jit_cond_dino_two_stage_1122_raev2/stat/dinov3l_k7_imagenet1k_stats.pt}"
DINO_ENCODER_WEIGHTS_PATH="${DINO_ENCODER_WEIGHTS_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
DINO_ENCODER_REPO="${DINO_ENCODER_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}"
RAEV2_ROOT="${RAEV2_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/RAEv2}"
RAEV2_CHECKPOINT="${RAEV2_CHECKPOINT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt}"
RAEV2_CONFIG="${RAEV2_CONFIG:-$RAEV2_ROOT/configs/stage2/sampling/imagenet-dinov3l-k7.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/$EXP_NAME}"
RESUME="${RESUME:-}"
JIT_CFG="${JIT_CFG:-2.9}"
DIST_TIMEOUT_MINUTES="${DIST_TIMEOUT_MINUTES:-60}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：bash sh/train_pixel_raev2.sh [--dry-run]' >&2
    exit 2
fi
if [[ "${1:-}" != --dry-run ]]; then
    for path in "$DINO_STAT_PATH" "$DINO_ENCODER_WEIGHTS_PATH" "$RAEV2_CHECKPOINT" "$RAEV2_CONFIG"; do
        if [[ ! -f "$path" ]]; then
            echo "Required resource not found: $path" >&2
            exit 1
        fi
    done
    if [[ ! -f "$DINO_ENCODER_REPO/hubconf.py" ]]; then
        echo "DINOv3 hubconf.py not found: $DINO_ENCODER_REPO" >&2
        exit 1
    fi
    if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
        echo "Resume checkpoint not found: $RESUME" >&2
        exit 1
    fi
fi

args=(
    --data_path "$IMAGENET_ROOT"
    --dino_stat_path "$DINO_STAT_PATH"
    --dino_encoder_weights_path "$DINO_ENCODER_WEIGHTS_PATH"
    --dino_encoder_repo "$DINO_ENCODER_REPO"
    --raev2_checkpoint "$RAEV2_CHECKPOINT"
    --raev2_config "$RAEV2_CONFIG"
    --model "$MODEL"
    --condition_mode_pixel "$CONDITION_MODE" --noise_beta_dino_pixel "$NOISE_BETA_DINO_PIXEL"
    --epochs "$EPOCHS"
    --output_dir "$OUTPUT_DIR"
    --batch_size 128 --blr 5e-5 --warmup_epochs 5 --num_workers 12
    --t_mean_pixel -0.8 --t_std_pixel 0.8 --noise_scale_pixel 1.0
    --steps_pixel 50 --sampling_method_pixel heun --cfg_pixel "$JIT_CFG"
    --interval_min_pixel 0.1 --interval_max_pixel 1.0
    --label_drop_prob 0.1 --gen_bsz 256 --num_images 50000 --eval_freq 20
    --save_last_freq 5 --save_interval 80 --dist_timeout "$DIST_TIMEOUT_MINUTES"
    --online_eval
)
if [[ -n "$RESUME" ]]; then
    args+=(--resume "$RESUME")
fi
printf -v args_string '%q ' "${args[@]}"
printf -v output_dir_q '%q' "$OUTPUT_DIR"
printf -v project_root_q '%q' "$PROJECT_ROOT"
printf -v python_bin_q '%q' "$PYTHON_BIN"
remote_command="set -o pipefail; mkdir -p $output_dir_q; cd -- $project_root_q && $python_bin_q -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 main_jit.py $args_string 2>&1 | tee -a $output_dir_q/train.log"
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
export PYTHONPATH="$PROJECT_ROOT:$RAEV2_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    IMAGENET_ROOT="$IMAGENET_ROOT" \
    PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
    FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
    DINOV3_REPO="$DINO_ENCODER_REPO" \
    HF_HUB_OFFLINE=1 TORCH_NCCL_TRACE_BUFFER_SIZE=4096 \
    bash -c "$remote_command"
