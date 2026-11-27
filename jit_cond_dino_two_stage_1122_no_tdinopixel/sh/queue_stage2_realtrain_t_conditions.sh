#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-2}"
WAIT_SESSION="${WAIT_SESSION:-realtrain_noised_e160}"
POLL_SECONDS="${POLL_SECONDS:-30}"
NOISED_CHECKPOINT="${NOISED_CHECKPOINT:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-noised-e200/checkpoint-160.pth}"
PRED_CHECKPOINT="${PRED_CHECKPOINT:-$PROJECT_ROOT/checkpoints/stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-pred-e200/checkpoint-160.pth}"
JIT_CFG="${JIT_CFG:-2.9}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
GEN_BSZ="${GEN_BSZ:-256}"
STEPS_PIXEL="${STEPS_PIXEL:-50}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SEED="${SEED:-0}"
GROUNDING_SELECTION_SEED="${GROUNDING_SELECTION_SEED:-0}"
GROUNDING_PER_CLASS="${GROUNDING_PER_CLASS:-50}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：GPU_HOST=agent-gpu-l2-2 bash sh/queue_stage2_realtrain_t_conditions.sh [--dry-run]' >&2
    exit 2
fi
if [[ "${1:-}" != --dry-run ]]; then
    for checkpoint in "$NOISED_CHECKPOINT" "$PRED_CHECKPOINT"; do
        if [[ ! -f "$checkpoint" ]]; then
            echo "Pixel checkpoint 不存在：$checkpoint" >&2
            exit 1
        fi
    done
    if [[ -n "$WAIT_SESSION" ]]; then
        printf 'Waiting for tmux session %s to finish\n' "$WAIT_SESSION"
        while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
            sleep "$POLL_SECONDS"
        done
    fi
fi

for mode in noised pred; do
    if [[ "$mode" == noised ]]; then
        checkpoint="$NOISED_CHECKPOINT"
    else
        checkpoint="$PRED_CHECKPOINT"
    fi
    checkpoint_tag="$(basename "${checkpoint%.pth}")"
    for t_mode in uniform fixed; do
        if [[ "$t_mode" == uniform ]]; then
            fixed_t=""
            t_tag="t_dino_pixel"
        else
            fixed_t="0.75"
            t_tag="0.75"
        fi
        eval_dir="$(dirname "$checkpoint")/eval_${checkpoint_tag}_realtrain${GROUNDING_PER_CLASS}_${mode}_${t_tag}_jitcfg${JIT_CFG}_heun${STEPS_PIXEL}_image${NUM_IMAGES}_selectseed${GROUNDING_SELECTION_SEED}"
        printf 'Realtrain noised-condition run: mode=%s, t=%s, selection_seed=%d, generation_seed=%d, output=%s\n' \
            "$mode" "$t_tag" "$GROUNDING_SELECTION_SEED" "$SEED" "$eval_dir"
        CHECKPOINT="$checkpoint" CONDITION_MODE="$mode" JIT_CFG="$JIT_CFG" \
            NUM_IMAGES="$NUM_IMAGES" GEN_BSZ="$GEN_BSZ" STEPS_PIXEL="$STEPS_PIXEL" \
            NPROC_PER_NODE="$NPROC_PER_NODE" SEED="$SEED" GPU_HOST="$GPU_HOST" \
            GROUNDING_SPLIT=train GROUNDING_PER_CLASS="$GROUNDING_PER_CLASS" \
            GROUNDING_SELECTION_SEED="$GROUNDING_SELECTION_SEED" \
            FIXED_T_DINO_PIXEL="$fixed_t" EVAL_DIR="$eval_dir" \
            bash "$PROJECT_ROOT/sh/sample_stage2_realval_noised_grounding.sh" "$@"
    done
done
