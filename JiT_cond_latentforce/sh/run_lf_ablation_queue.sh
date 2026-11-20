#!/usr/bin/env bash
# 串行跑两个 Latent Forcing grounding 版本，与现有 GPU 任务并行共存。
#   v1: LF 内部不加引导 (lf_cfg=1.0, guidance=none)
#   v2: LF 内部用主网 CFG (lf_cfg=1.5, guidance=cfg)
# 两者 JiT 采样配置相同（cfg 2.9, interval [0.1,1], heun 50），唯一变量是
# LF 生成 latent 时的引导强度。串行执行：每个任务各自占用 8 卡做 DDP。
set -uo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="$PROJECT_ROOT/checkpoints/JiT-B16-DINOv3B-PatchAdaLN-ValGrounding-B600"
mkdir -p "$LOG_ROOT"

run() {
    local tag="$1" lfcfg="$2" guid="$3" port="$4"
    echo "[$(date '+%F %T')] >>> start $tag (lf_cfg=$lfcfg guidance=$guid)"
    TAG="$tag" LF_CFG="$lfcfg" LF_GUIDANCE="$guid" RDZV_PORT="$port" \
        bash "$PROJECT_ROOT/sh/sample_jit_b16_lf_8gpu.sh"
    echo "[$(date '+%F %T')] <<< done $tag (exit $?)"
}

run "lf25_ema1_lfcfgNone" "1.0" "none" 29531
run "lf25_ema1_lfcfg1p5"  "1.5" "cfg"  29532
echo "[$(date '+%F %T')] all Latent Forcing grounding runs finished"
