#!/usr/bin/env bash
# Class-conditional generation from random noise; no reference image/data_path.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo 'Usage: bash sh/sample_jit_b16_grounding_8gpu.sh [--dry-run]' >&2
    exit 2
fi

if [[ "$GPU_HOST" != local ]]; then
    forwarded_env=(GPU_HOST=local)
    for name in PYTHON_BIN CUDA_VISIBLE_DEVICES GPUS CHECKPOINT EVAL_DIR EXPERIMENT \
        PRETRAIN_MODELS_ROOT FID_STATS_ROOT DINOV3_REPO DINO_REPO DINO_WEIGHTS \
        DINO_WEIGHTS_PATH DINO_MODEL_NAME MODEL IMG_SIZE NOISE_SCALE GEN_BSZ \
        NUM_IMAGES CLASS_NUM CFG INTERVAL_MIN INTERVAL_MAX SAMPLING_METHOD \
        NUM_SAMPLING_STEPS GROUNDING_PROJ_BLOCKS GROUNDING_PROJ_HEADS; do
        if [[ -v "$name" ]]; then forwarded_env+=("$name=${!name}"); fi
    done
    ssh_args=()
    [[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
    exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
        env "${forwarded_env[@]}" bash sh/sample_jit_b16_grounding_8gpu.sh
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
source "$PROJECT_ROOT/sh/portable_env.sh"
IFS=, read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
GPUS="${GPUS:-${#visible_gpus[@]}}"
[[ "$GPUS" =~ ^[1-9][0-9]*$ && "$GPUS" -le "${#visible_gpus[@]}" ]] || {
    echo 'GPUS must be positive and not exceed CUDA_VISIBLE_DEVICES count.' >&2; exit 2;
}
EXPERIMENT="${EXPERIMENT:-JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/$EXPERIMENT/checkpoint-last.pth}"
CKPT_DIR="$(dirname -- "$CHECKPOINT")"
CKPT_TAG="$(basename -- "$CHECKPOINT" .pth)"
CFG="${CFG:-2.9}"
SAMPLING_METHOD="${SAMPLING_METHOD:-heun}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-50}"
EVAL_DIR="${EVAL_DIR:-$CKPT_DIR/eval_${CKPT_TAG}_${SAMPLING_METHOD}${NUM_SAMPLING_STEPS}_cfg${CFG}_noisy_projector}"
CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$GPUS"
    main_jit.py
    --model "${MODEL:-JiT-B/16}" --img_size "${IMG_SIZE:-256}" --noise_scale "${NOISE_SCALE:-1.0}"
    --evaluate_gen --checkpoint "$CHECKPOINT"
    --gen_bsz "${GEN_BSZ:-256}" --num_images "${NUM_IMAGES:-50000}" --class_num "${CLASS_NUM:-1000}"
    --cfg "$CFG" --interval_min "${INTERVAL_MIN:-0.1}" --interval_max "${INTERVAL_MAX:-1.0}"
    --sampling_method "$SAMPLING_METHOD" --num_sampling_steps "$NUM_SAMPLING_STEPS"
    --output_dir "$EVAL_DIR"
    --use_grounding --grounding_model "$DINO_MODEL_NAME" --grounding_dim 768
    --grounding_weights "$DINO_WEIGHTS_PATH" --grounding_repo "$DINOV3_REPO"
    --grounding_proj_blocks "${GROUNDING_PROJ_BLOCKS:-1}")
if [[ -n "${GROUNDING_PROJ_HEADS:-}" ]]; then CMD+=(--grounding_proj_heads "$GROUNDING_PROJ_HEADS"); fi
if [[ "${1:-}" == --dry-run ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
[[ -f "$CHECKPOINT" ]] || { echo "Sampling checkpoint missing: $CHECKPOINT" >&2; exit 1; }
[[ -f "$DINO_WEIGHTS_PATH" ]] || { echo "DINO weights missing: $DINO_WEIGHTS_PATH" >&2; exit 1; }
[[ -f "$DINOV3_REPO/hubconf.py" ]] || { echo "DINO source missing: $DINOV3_REPO" >&2; exit 1; }
mkdir -p "$EVAL_DIR"
"${CMD[@]}" 2>&1 | tee "$EVAL_DIR/sampling.log"
