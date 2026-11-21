#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
EXPERIMENT='JiT-B16-JoinDino-dinov3b-B200'
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
CHECKPOINT="$CHECKPOINT_ROOT/checkpoint-80.pth"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
DINO_CFG='2.0'
DINO_STEPS='25'
JI_CFG_LIST=(4.0 4.5 5.0 5.5 6.0)
RDZV_PORT="${RDZV_PORT:-29555}"

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo "用法：bash sh/$(basename "$0") [--dry-run]" >&2
    exit 2
fi

for jit_cfg in "${JI_CFG_LIST[@]}"; do
    eval_dir="$CHECKPOINT_ROOT/eval_checkpoint-80_dinov3b_jitcfg${jit_cfg}_dinocfg${DINO_CFG}_dino${DINO_STEPS}_heun50_image50000"
    sampling_log="$eval_dir/sampling.log"

    if [[ "${1:-}" == --dry-run ]]; then
        CHECKPOINT="$CHECKPOINT" EVAL_DIR="$eval_dir" CKPT_CFG="$jit_cfg" \
            DINO_CFG="$DINO_CFG" DINO_STEPS="$DINO_STEPS" RDZV_PORT="$RDZV_PORT" \
            PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_ROOT/sh/sample_jit_b16_join_dinov3_8gpu_local.sh" --dry-run
        continue
    fi

    if [[ -f "$sampling_log" ]] && "$PYTHON_BIN" -c \
        'import pathlib, sys; sys.exit(0 if "FID:" in pathlib.Path(sys.argv[1]).read_text(errors="replace") else 1)' \
        "$sampling_log"; then
        echo "跳过已完成的 JiT CFG $jit_cfg：$sampling_log"
        continue
    fi

    echo "[$(date '+%F %T')] 开始 JiT CFG $jit_cfg，DINO CFG=$DINO_CFG，checkpoint=$CHECKPOINT"
    env EXPERIMENT="$EXPERIMENT" CHECKPOINT="$CHECKPOINT" EVAL_DIR="$eval_dir" \
        CKPT_CFG="$jit_cfg" DINO_CFG="$DINO_CFG" DINO_STEPS="$DINO_STEPS" \
        RDZV_PORT="$RDZV_PORT" PYTHON_BIN="$PYTHON_BIN" GPU_HOST="$GPU_HOST" \
        bash "$PROJECT_ROOT/sh/sample_jit_b16_join_dinov3_8gpu_h200.sh"
    echo "[$(date '+%F %T')] 完成 JiT CFG $jit_cfg"
done

if [[ "${1:-}" == --dry-run ]]; then
    echo 'Dry-run 完成，未启动采样或修改训练任务。'
else
    echo 'DINO CFG 2.0、JiT CFG 4.0-6.0 五组采样全部完成。'
fi
