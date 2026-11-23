#!/usr/bin/env bash
# 设定本机或集群节点上的 Python 与可见 GPU，由阶段脚本选择入口。

run_torch() {
    local dry_run=0 arg
    local -a entry_args=() gpus=()
    for arg in "$@"; do
        if [[ "$arg" == --dry-run ]]; then
            dry_run=1
        else
            entry_args+=("$arg")
        fi
    done

    if [[ -z "${PYTHON_BIN:-}" ]]; then
        PYTHON_BIN="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python"
    fi
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
    export IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
    export PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}"
    export FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}"
    export DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    IFS=, read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
    export HF_HUB_OFFLINE=1
    cd -- "$ROOT"

    local -a cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1
        --nproc_per_node="${#gpus[@]}" "${entry_args[@]}")
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "$dry_run" == 0 ]]; then
        exec "${cmd[@]}"
    fi
}
