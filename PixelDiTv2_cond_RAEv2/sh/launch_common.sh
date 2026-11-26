#!/usr/bin/env bash
# Run directly on a GPU node. CUDA_VISIBLE_DEVICES or NPROC_PER_NODE selects the workers.
launch_pixel() {
    local dry_run=0 arg workers
    local -a args=() devices=()
    for arg in "$@"; do
        if [[ "$arg" == --dry-run ]]; then dry_run=1; else args+=("$arg"); fi
    done
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    fi
    workers="${NPROC_PER_NODE:-${#devices[@]}}"
    if [[ "$workers" == 0 ]]; then workers=8; fi
    local -a cmd=("${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nnodes=1
                  --nproc_per_node="$workers" "$PROJECT_ROOT/main_jit.py" "${args[@]}")
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if (( ! dry_run )); then
        exec "${cmd[@]}"
    fi
}
