#!/usr/bin/env bash
# Sample RAEv2 semantics and draw RGB with the Pixel EMA. Trailing CLI flags override defaults.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/sh/launch_common.sh"
launch_pixel --evaluate_gen \
    --checkpoint "${PIXEL_CHECKPOINT:-$PROJECT_ROOT/checkpoints/raev2-dinov3l-k7-pixeldit2-h16-e160/checkpoint-last.pth}" \
    --steps_pixel 50 --cfg_pixel 2.4 --interval_min_pixel 0.125 --interval_max_pixel 0.9 "$@"
