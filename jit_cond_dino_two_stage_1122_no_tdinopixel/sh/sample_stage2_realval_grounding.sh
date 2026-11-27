#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-2}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-pred-e200/checkpoint-160.pth}"
CHECKPOINT_TAG="$(basename "${CHECKPOINT%.pth}")"
JIT_CFG="${JIT_CFG:-2.9}"
CONDITION_MODE="${CONDITION_MODE:-pred}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
GEN_BSZ="${GEN_BSZ:-256}"
STEPS_PIXEL="${STEPS_PIXEL:-50}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEED="${SEED:-0}"
GROUNDING_SPLIT="${GROUNDING_SPLIT:-val}"
GROUNDING_PER_CLASS="${GROUNDING_PER_CLASS:-}"
GROUNDING_SELECTION_SEED="${GROUNDING_SELECTION_SEED:-0}"
GROUNDING_TAG="real${GROUNDING_SPLIT}"
if [[ -n "$GROUNDING_PER_CLASS" ]]; then
    GROUNDING_TAG+="${GROUNDING_PER_CLASS}_selectseed${GROUNDING_SELECTION_SEED}"
fi
EVAL_DIR="${EVAL_DIR:-$(dirname "$CHECKPOINT")/eval_${CHECKPOINT_TAG}_${GROUNDING_TAG}_dinov3b_instance_jitcfg${JIT_CFG}_heun${STEPS_PIXEL}_image${NUM_IMAGES}}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：CHECKPOINT=/abs/checkpoint-160.pth JIT_CFG=2.9 bash sh/sample_stage2_realval_grounding.sh [--dry-run]' >&2
    exit 2
fi

args=(
    --checkpoint "$CHECKPOINT" --condition_mode_pixel "$CONDITION_MODE"
    --dino_encoder dinov3b --dino_stat_path "$DINO_STAT_PATH"
    --data_path "$IMAGENET_ROOT" --grounding_split "$GROUNDING_SPLIT"
    --grounding_selection_seed "$GROUNDING_SELECTION_SEED" --seed "$SEED"
    --steps_pixel "$STEPS_PIXEL" --sampling_method_pixel heun
    --cfg_pixel "$JIT_CFG" --interval_min_pixel 0.1 --interval_max_pixel 1.0
    --gen_bsz "$GEN_BSZ" --num_images "$NUM_IMAGES" --output_dir "$EVAL_DIR"
)
if [[ -n "$GROUNDING_PER_CLASS" ]]; then
    args+=(--grounding_per_class "$GROUNDING_PER_CLASS")
fi
if [[ "${SKIP_METRICS:-0}" == 1 ]]; then
    args+=(--skip_metrics)
fi
if [[ "${MATCH_DINO_NOISE_RNG:-1}" == 0 ]]; then
    args+=(--no-match_dino_noise_rng)
fi
printf -v args_string '%q ' "${args[@]}"
printf -v eval_dir_q '%q' "$EVAL_DIR"
printf -v project_root_q '%q' "$PROJECT_ROOT"
printf -v python_bin_q '%q' "$PYTHON_BIN"
remote_command="set -o pipefail; mkdir -p $eval_dir_q; cd -- $project_root_q && $python_bin_q -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE grounding_evaluation.py $args_string 2>&1 | tee -a $eval_dir_q/sampling.log"
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
exec "$PROJECT_ROOT/../JiT_cond_join_dino/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    IMAGENET_ROOT="$IMAGENET_ROOT" \
    PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
    FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
    DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}" \
    HF_HUB_OFFLINE=1 \
    bash -c "$remote_command"
