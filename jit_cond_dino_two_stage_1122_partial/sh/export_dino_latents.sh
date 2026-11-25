#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
DINO_CHECKPOINT="${DINO_CHECKPOINT:-}"
DINO_ENCODER="${DINO_ENCODER:-dinov3b}"
DINO_WEIGHTS="${DINO_WEIGHTS:-ema1}"
DINO_STAT_PATH="${DINO_STAT_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt}"
DATA_PATH="${DATA_PATH:-$IMAGENET_ROOT}"
SPLIT="${SPLIT:-train}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SHARD_SIZE="${SHARD_SIZE:-128}"
STEPS_DINO="${STEPS_DINO:-25}"
SAMPLING_METHOD_DINO="${SAMPLING_METHOD_DINO:-heun}"
CFG_DINO="${CFG_DINO:-1.0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-12}"

usage() {
    cat <<'EOF'
用法：
  DINO_CHECKPOINT=/path/to/checkpoint-80.pth [CACHE_DIR=/path/to/cache] \
    bash sh/export_dino_latents.sh [--dry-run]

默认使用 GPU1 的 8 卡导出 ImageNet train：EMA1、25 步 Heun、CFG 1.0、
每卡 batch 8 和 12 个 DataLoader worker，每个分片 128 张图。
CACHE_DIR 省略时写入 IMAGENET_ROOT/<实验名>-e<checkpoint epoch>。
可通过 GPU_HOST、DINO_ENCODER、DINO_WEIGHTS、DINO_STAT_PATH、DATA_PATH、
SPLIT、BATCH_SIZE、SHARD_SIZE、STEPS_DINO、CFG_DINO、NPROC_PER_NODE、
NUM_WORKERS 环境变量覆盖默认值。checkpoint-last.pth 需要显式指定 CACHE_DIR。
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    usage >&2
    exit 2
fi
if [[ -z "$DINO_CHECKPOINT" || ! -f "$DINO_CHECKPOINT" ]]; then
    echo '请通过 DINO_CHECKPOINT 指定一个存在的 DINO DiT checkpoint。' >&2
    usage >&2
    exit 2
fi
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo 'NPROC_PER_NODE 必须是正整数。' >&2
    exit 2
fi
if [[ -z "${CACHE_DIR:-}" ]]; then
    checkpoint_name="$(basename -- "$DINO_CHECKPOINT")"
    if [[ ! "$checkpoint_name" =~ ^checkpoint-([0-9]+)\.pth$ ]]; then
        echo '无法从 checkpoint 文件名推断 epoch，请显式指定 CACHE_DIR。' >&2
        exit 2
    fi
    experiment_name="$(basename "$(dirname -- "$DINO_CHECKPOINT")")"
    experiment_name="${experiment_name%-e[0-9]*}"
    CACHE_DIR="$IMAGENET_ROOT/${experiment_name}-e${BASH_REMATCH[1]}"
fi

args=(
    --dino_checkpoint "$DINO_CHECKPOINT" --dino_encoder "$DINO_ENCODER"
    --dino_weights "$DINO_WEIGHTS" --dino_stat_path "$DINO_STAT_PATH"
    --data_path "$DATA_PATH" --cache_dir "$CACHE_DIR" --split "$SPLIT"
    --batch_size "$BATCH_SIZE" --steps_dino "$STEPS_DINO"
    --sampling_method_dino "$SAMPLING_METHOD_DINO" --cfg_dino "$CFG_DINO"
    --shard_size "$SHARD_SIZE" --num_workers "$NUM_WORKERS"
)
printf -v args_string '%q ' "${args[@]}"
printf -v cache_dir_q '%q' "$CACHE_DIR"
printf -v project_root_q '%q' "$PROJECT_ROOT"
printf -v python_bin_q '%q' "$PYTHON_BIN"
remote_command="set -o pipefail; mkdir -p $cache_dir_q; cd -- $project_root_q && $python_bin_q -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE --module stage1.export_recon $args_string 2>&1 | tee -a $cache_dir_q/export.log"
ssh_args=()
[[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
exec "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
    env PYTHON_BIN="$PYTHON_BIN" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    IMAGENET_ROOT="$IMAGENET_ROOT" \
    PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
    FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
    DINOV3_REPO="${DINOV3_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}" \
    HF_HUB_OFFLINE=1 \
    bash -c "$remote_command"
