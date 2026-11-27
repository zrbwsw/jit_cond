#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-2}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/imagenet}"
DINO_STAT_PATH="${DINO_STAT_PATH:-$PROJECT_ROOT/../jit_cond_dino_two_stage_1122_raev2/stat/dinov3l_k7_imagenet1k_stats.pt}"
DINO_ENCODER_WEIGHTS_PATH="${DINO_ENCODER_WEIGHTS_PATH:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/pretrain_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"
DINO_ENCODER_REPO="${DINO_ENCODER_REPO:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3}"
RAEV2_ROOT="${RAEV2_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/RAEv2}"
RAEV2_CHECKPOINT="${RAEV2_CHECKPOINT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt}"
RAEV2_CONFIG="${RAEV2_CONFIG:-$RAEV2_ROOT/configs/stage2/sampling/imagenet-dinov3l-k7.yaml}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/stage1-raev2-dinov3l-k7-stage2-pixeljit-b-e320/checkpoint-80.pth}"
CHECKPOINT_TAG="$(basename "${CHECKPOINT%.pth}")"
NUM_IMAGES="${NUM_IMAGES:-50000}"
GEN_BSZ="${GEN_BSZ:-256}"
STEPS_PIXEL="${STEPS_PIXEL:-50}"
SEED="${SEED:-0}"
read -r -a cfgs <<< "${CFGS:-2.1 2.3}"
if [[ $# -gt 1 || ( $# -eq 1 && "${1:-}" != --dry-run ) ]]; then
    echo '用法：CHECKPOINT=/abs/checkpoint-80.pth CFGS="2.1 2.3" bash sh/sample_pixel_raev2_e80_jit_cfgs.sh [--dry-run]' >&2
    exit 2
fi
if [[ "${1:-}" != --dry-run ]]; then
    for path in "$CHECKPOINT" "$DINO_STAT_PATH" "$DINO_ENCODER_WEIGHTS_PATH" "$RAEV2_CHECKPOINT" "$RAEV2_CONFIG"; do
        if [[ ! -f "$path" ]]; then
            echo "Required resource not found: $path" >&2
            exit 1
        fi
    done
    if [[ ! -f "$DINO_ENCODER_REPO/hubconf.py" ]]; then
        echo "DINOv3 hubconf.py not found: $DINO_ENCODER_REPO" >&2
        exit 1
    fi
fi

export PYTHONPATH="$PROJECT_ROOT:$RAEV2_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
for cfg in "${cfgs[@]}"; do
    eval_dir="$(dirname "$CHECKPOINT")/eval_${CHECKPOINT_TAG}_jitcfg${cfg}_${RAEV2_DIR_TAG:-raev2}_heun100_pixelheun${STEPS_PIXEL}_image${NUM_IMAGES}"
    args=(
        --checkpoint "$CHECKPOINT"
        --dino_stat_path "$DINO_STAT_PATH"
        --dino_encoder_weights_path "$DINO_ENCODER_WEIGHTS_PATH"
        --dino_encoder_repo "$DINO_ENCODER_REPO"
        --raev2_checkpoint "$RAEV2_CHECKPOINT" --raev2_config "$RAEV2_CONFIG"
        --data_path "$IMAGENET_ROOT" --output_dir "$eval_dir" --seed "$SEED"
        --steps_pixel "$STEPS_PIXEL" --sampling_method_pixel heun
        --cfg_pixel "$cfg" --interval_min_pixel 0.1 --interval_max_pixel 1.0
        --gen_bsz "$GEN_BSZ" --num_images "$NUM_IMAGES"
    )
    printf -v args_string '%q ' "${args[@]}"
    printf -v eval_dir_q '%q' "$eval_dir"
    printf -v project_root_q '%q' "$PROJECT_ROOT"
    remote_command="set -o pipefail; mkdir -p $eval_dir_q; cd -- $project_root_q && bash sh/sample_pixel.sh $args_string 2>&1 | tee -a $eval_dir_q/sampling.log"
    ssh_args=()
    [[ "${1:-}" != --dry-run ]] || ssh_args+=(--dry-run)
    "$PROJECT_ROOT/sh/run_on_gpu.sh" "${ssh_args[@]}" "$GPU_HOST" \
        env PYTHON_BIN="$PYTHON_BIN" PYTHONPATH="$PYTHONPATH" \
        PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
        IMAGENET_ROOT="$IMAGENET_ROOT" RAEV2_ROOT="$RAEV2_ROOT" \
        PRETRAIN_MODELS_ROOT="${PRETRAIN_MODELS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models}" \
        FID_STATS_ROOT="${FID_STATS_ROOT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats}" \
        DINOV3_REPO="$DINO_ENCODER_REPO" \
        HF_HUB_OFFLINE=1 TORCH_NCCL_TRACE_BUFFER_SIZE=4096 \
        bash -c "$remote_command"
done
