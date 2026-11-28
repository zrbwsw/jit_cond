#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-2}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-pred-e200/checkpoint-160.pth}"
CHECKPOINT_TAG="$(basename "${CHECKPOINT%.pth}")"
JIT_CFG="${JIT_CFG:-2.9}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
GEN_BSZ="${GEN_BSZ:-256}"
STEPS_PIXEL="${STEPS_PIXEL:-50}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DRAWS="${DRAWS:-3}"
GROUNDING_PER_CLASS="${GROUNDING_PER_CLASS:-50}"
GROUNDING_SEED_BASE="${GROUNDING_SEED_BASE:-0}"
SEED="${SEED:-0}"
EVAL_ROOT="${EVAL_ROOT:-$(dirname "$CHECKPOINT")/eval_${CHECKPOINT_TAG}_realtrain${GROUNDING_PER_CLASS}_dinov3b_instance_jitcfg${JIT_CFG}_heun${STEPS_PIXEL}_image${NUM_IMAGES}}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：CHECKPOINT=/abs/checkpoint-160.pth JIT_CFG=2.9 DRAWS=3 bash sh/sample_stage2_realtrain_grounding.sh [--dry-run]' >&2
    exit 2
fi
if [[ ! "$DRAWS" =~ ^[1-9][0-9]*$ || ! "$GROUNDING_PER_CLASS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'DRAWS 和 GROUNDING_PER_CLASS 必须是正整数' >&2
    exit 2
fi
if [[ "${1:-}" != --dry-run && ! -f "$CHECKPOINT" ]]; then
    echo "Pixel checkpoint 不存在：$CHECKPOINT" >&2
    exit 1
fi

for ((draw = 1; draw <= DRAWS; draw++)); do
    selection_seed=$((GROUNDING_SEED_BASE + draw - 1))
    eval_dir="$EVAL_ROOT/draw${draw}_selectseed${selection_seed}"
    printf 'Realtrain draw %d/%d: selection_seed=%d, generation_seed=%d, output=%s\n' \
        "$draw" "$DRAWS" "$selection_seed" "$SEED" "$eval_dir"
    CHECKPOINT="$CHECKPOINT" CONDITION_MODE="${CONDITION_MODE:-pred}" JIT_CFG="$JIT_CFG" \
        NUM_IMAGES="$NUM_IMAGES" GEN_BSZ="$GEN_BSZ" STEPS_PIXEL="$STEPS_PIXEL" \
        NPROC_PER_NODE="$NPROC_PER_NODE" SEED="$SEED" GPU_HOST="$GPU_HOST" \
        GROUNDING_SPLIT=train GROUNDING_PER_CLASS="$GROUNDING_PER_CLASS" \
        GROUNDING_SELECTION_SEED="$selection_seed" EVAL_DIR="$eval_dir" \
        bash "$PROJECT_ROOT/sh/sample_stage2_realval_grounding.sh" "$@"
done
