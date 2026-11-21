#!/usr/bin/env bash
# Usage: bash sh/run.sh dino|pixel|sample [--dry-run] [training arguments...]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?Specify dino, pixel or sample}"
shift
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
    DRY_RUN=1
    shift
fi
case "$MODE" in
    dino) ENTRY=main_dino.py ;;
    pixel|sample) ENTRY=main_jit.py ;;
    *) echo 'Mode must be dino, pixel or sample' >&2; exit 2 ;;
esac

if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$HOME/anaconda3/envs/irepa/bin/python"
    [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export HF_HUB_OFFLINE=1
cd -- "$ROOT"

CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1
    --nproc_per_node="${#GPUS[@]}" "$ENTRY"
    --dino_encoder "${DINO_ENCODER:-dinov3b}"
    --epochs 600 --batch_size 128 --blr 5e-5 --warmup_epochs 5
    --output_dir "${OUTPUT_DIR:-$ROOT/checkpoints/${MODE}_${DINO_ENCODER:-dinov3b}}"
    --save_last_freq 5 --save_interval 20 --eval_freq 20
    --dino_steps 25 --num_sampling_steps 50
    --cfg 2.9 --interval_min 0.1 --interval_max 1.0
    --dino_cfg 1.0 --gen_bsz 64)
if [[ "$MODE" == sample ]]; then
    CMD+=(--evaluate_gen)
fi
CMD+=("$@")
printf '%q ' "${CMD[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi
exec "${CMD[@]}"
