#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
EXPERIMENT='JiT-B16-JoinDino-dinov3b-B200'
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/$EXPERIMENT"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
CKPT_CFG='2.9'
DINO_CFG='2.5'
DINO_STEPS='25'
EPOCH_LIST=(20 40 60 80 100)
KEEPALIVE='/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/scripts/zrb.sh'
PREVIOUS_LOG="$CHECKPOINT_ROOT/eval_dinocfg2p0_e20-100_queue.log"
PREVIOUS_MARKER='DINO CFG 2.0 五组采样全部完成。'

if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo "用法：bash sh/$(basename "$0") [--dry-run]" >&2
    exit 2
fi

on_error() {
    status=$?
    echo "采样队列失败（exit $status），尝试恢复 GPU 占卡。" >&2
    bash "$PROJECT_ROOT/sh/run_on_gpu.sh" "$GPU_HOST" bash "$KEEPALIVE" start || true
    exit "$status"
}
trap on_error ERR

if [[ "${1:-}" != --dry-run ]]; then
    echo '等待 DINO CFG 2.0 五组采样完成。'
    while true; do
        if [[ -f "$PREVIOUS_LOG" ]]; then
            previous_contents="$(<"$PREVIOUS_LOG")"
            if [[ "$previous_contents" == *"$PREVIOUS_MARKER"* ]]; then
                echo 'DINO CFG 2.0 已完成，启动 DINO CFG 2.5。'
                break
            fi
            if [[ "$previous_contents" == *'采样队列失败'* ]]; then
                echo 'DINO CFG 2.0 队列失败，取消启动 DINO CFG 2.5。' >&2
                exit 1
            fi
        fi
        sleep 20
    done
    bash "$PROJECT_ROOT/sh/run_on_gpu.sh" "$GPU_HOST" bash "$KEEPALIVE" stop
fi

for epoch in "${EPOCH_LIST[@]}"; do
    checkpoint="$CHECKPOINT_ROOT/checkpoint-$epoch.pth"
    eval_dir="$CHECKPOINT_ROOT/eval_checkpoint-${epoch}_dinov3b_jitcfg${CKPT_CFG}_dinocfg${DINO_CFG}_dino${DINO_STEPS}_heun50_image50000"
    sampling_log="$eval_dir/sampling.log"

    if [[ "${1:-}" == --dry-run ]]; then
        CHECKPOINT="$checkpoint" EVAL_DIR="$eval_dir" CKPT_CFG="$CKPT_CFG" \
            DINO_CFG="$DINO_CFG" DINO_STEPS="$DINO_STEPS" PYTHON_BIN="$PYTHON_BIN" \
            bash "$PROJECT_ROOT/sh/sample_jit_b16_join_dinov3_8gpu_local.sh" --dry-run
        continue
    fi

    if [[ -f "$sampling_log" ]] && "$PYTHON_BIN" -c \
        'import pathlib, sys; sys.exit(0 if "FID:" in pathlib.Path(sys.argv[1]).read_text(errors="replace") else 1)' \
        "$sampling_log"; then
        echo "跳过已完成的 epoch $epoch：$sampling_log"
        continue
    fi

    echo "[$(date '+%F %T')] 开始 epoch $epoch：JiT CFG=$CKPT_CFG，DINO CFG=$DINO_CFG"
    env EXPERIMENT="$EXPERIMENT" CHECKPOINT="$checkpoint" EVAL_DIR="$eval_dir" \
        CKPT_CFG="$CKPT_CFG" DINO_CFG="$DINO_CFG" DINO_STEPS="$DINO_STEPS" \
        PYTHON_BIN="$PYTHON_BIN" GPU_HOST="$GPU_HOST" \
        bash "$PROJECT_ROOT/sh/sample_jit_b16_join_dinov3_8gpu_h200.sh"
    echo "[$(date '+%F %T')] 完成 epoch $epoch"
done

if [[ "${1:-}" == --dry-run ]]; then
    echo 'Dry-run 完成，未等待前序队列、停止占卡或启动采样。'
else
    echo 'DINO CFG 2.5 五组采样全部完成。'
fi
