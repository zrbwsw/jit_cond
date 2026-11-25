#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt}"
DINO_CFG="${DINO_CFG:-2.0}"
JIT_CFG_LIST="${JIT_CFG_LIST:-2.1 2.3 2.5}"
if [[ -z "${CHECKPOINT:-}" ]]; then
    echo '必须设置 CHECKPOINT' >&2
    exit 2
fi
EVAL_ROOT="${EVAL_ROOT:-$(dirname "$CHECKPOINT")}"
CHECKPOINT_TAG="$(basename "${CHECKPOINT%.pth}")"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：CHECKPOINT=/abs/checkpoint-80.pth GPU_HOST=agent-gpu-l2-1 bash sh/sample_stage2_ckpt_scales.sh [--dry-run]' >&2
    exit 2
fi

for jit_cfg in $JIT_CFG_LIST; do
    eval_dir="$EVAL_ROOT/eval_${CHECKPOINT_TAG}_jitcfg${jit_cfg}_dinocfg${DINO_CFG}_dino25_heun50_image50000"
    sampling_log="$eval_dir/sampling.log"
    if [[ "${1:-}" != --dry-run && -f "$sampling_log" ]] && "$PYTHON_BIN" -c \
        'import pathlib, sys; sys.exit(0 if "FID:" in pathlib.Path(sys.argv[1]).read_text(errors="replace") else 1)' \
        "$sampling_log"; then
        echo "跳过已完成的 JiT CFG $jit_cfg：$sampling_log"
        continue
    fi

    args=(
        --checkpoint "$CHECKPOINT" --dino_encoder dinov3b --dino_stat_path "$DINO_STAT_PATH"
        --condition_mode_pixel "${CONDITION_MODE:-pred}"
        --steps_dino 25 --steps_pixel 50 --sampling_method_dino heun --sampling_method_pixel heun
        --cfg_dino "$DINO_CFG" --cfg_pixel "$jit_cfg"
        --interval_min_dino 0.0 --interval_max_dino 1.0
        --interval_min_pixel 0.1 --interval_max_pixel 1.0
        --gen_bsz 256 --num_images 50000 --output_dir "$eval_dir"
    )
    printf -v args_string '%q ' "${args[@]}"
    printf -v eval_dir_q '%q' "$eval_dir"
    remote_command="set -o pipefail; mkdir -p $eval_dir_q; cd -- $(printf '%q' "$PROJECT_ROOT") && bash sh/sample_pixel.sh $args_string 2>&1 | tee -a $eval_dir_q/sampling.log"
    echo "[$(date '+%F %T')] 开始 JiT CFG $jit_cfg，DINO CFG=$DINO_CFG，GPU=$GPU_HOST"
    if [[ "${1:-}" == --dry-run ]]; then
        printf 'DRY-RUN: %s\n' "$remote_command"
    else
        ssh_args=()
        "$PROJECT_ROOT/../JiT_cond_join_dino/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
            env PYTHON_BIN="$PYTHON_BIN" \
            IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}" \
            PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
            FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
            DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}" \
            bash -c "$remote_command"
    fi
    echo "[$(date '+%F %T')] 完成 JiT CFG $jit_cfg"
done
