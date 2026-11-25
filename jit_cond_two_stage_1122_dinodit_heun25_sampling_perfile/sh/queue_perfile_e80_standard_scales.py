import argparse
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "stage1-dinov3b-dinodit-b-recon-noflip-e80-stage2-pixeljit-b-dinodit-cached-recon-perfile-e320"
CURRENT_SESSION = "noised-e160-standard-cfg2p9-2p1-queue-gpu1"
CURRENT_LOG = PROJECT_ROOT.parent / "jit_cond_dino_two_stage_1122" / "checkpoints" / "stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-noised-e200" / "queue_e160_standard_jitcfg2p9_2p1.log"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / EXPERIMENT / "checkpoint-80.pth"
DINO_STAT = "/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/JiT_cond_join_dino/stat/dinov3b_rae_imagenet1k_stats.pt"


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def current_queue_complete():
    if CURRENT_LOG.exists() and "Standard DinoDiT sampling for JiT CFG 2.9 and 2.1 completed" in CURRENT_LOG.read_text(errors="replace"):
        return True
    for scale in ("2.9", "2.1"):
        sampling_log = CURRENT_LOG.parent / f"eval_checkpoint-160_jitcfg{scale}_dinocfg2.0_dino25_heun50_image50000/sampling.log"
        if not sampling_log.exists() or "FID:" not in sampling_log.read_text(errors="replace"):
            return False
    return True


def wait_for_current_queue():
    while True:
        if current_queue_complete():
            return
        text = CURRENT_LOG.read_text(errors="replace") if CURRENT_LOG.exists() else ""
        if "Traceback" in text or "RuntimeError" in text or "ChildFailedError" in text:
            raise RuntimeError(f"Current sampling queue failed; see {CURRENT_LOG}")
        if subprocess.run(["tmux", "has-session", "-t", CURRENT_SESSION], check=False).returncode != 0:
            raise RuntimeError("Current sampling queue stopped before completing both scales")
        time.sleep(20)


def run_scale(scale, env):
    eval_dir = PROJECT_ROOT / "checkpoints" / EXPERIMENT / f"eval_checkpoint-80_jitcfg{scale}_dinocfg2.0_dino25_heun50_image50000"
    sampling_log = eval_dir / "sampling.log"
    if sampling_log.exists() and "FID:" in sampling_log.read_text(errors="replace"):
        log(f"Skipping completed per-file E80 JiT CFG {scale}")
        return
    log(f"Starting per-file cached_recon E80 sampling with JiT CFG {scale}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "--checkpoint", str(CHECKPOINT), "--condition_mode_pixel", "cached_recon",
        "--model", "JiT-B/16", "--dino_model", "DinoDiT-B", "--dino_encoder", "dinov3b",
        "--dino_weights", "ema1", "--dino_stat_path", DINO_STAT,
        "--data_path", "/mnt/shared-storage-user/datafactory-share/gushishen/imagenet",
        "--steps_dino", "25", "--steps_pixel", "50",
        "--sampling_method_dino", "heun", "--sampling_method_pixel", "heun",
        "--cfg_dino", "2.0", "--cache_cfg_dino", "1.0", "--cfg_pixel", scale,
        "--interval_min_dino", "0.0", "--interval_max_dino", "1.0",
        "--interval_min_pixel", "0.1", "--interval_max_pixel", "1.0",
        "--gen_bsz", "256", "--num_images", "50000", "--output_dir", str(eval_dir),
    ]
    with sampling_log.open("a", encoding="utf-8") as stream:
        subprocess.run(["bash", "sh/sample_pixel.sh", *args], cwd=PROJECT_ROOT,
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    log(f"Completed per-file cached_recon E80 sampling with JiT CFG {scale}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        log(f"Would wait for {CURRENT_SESSION}, then sample {CHECKPOINT} with JiT CFG 2.9, 2.1")
        return
    wait_for_current_queue()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               IMAGENET_ROOT="/mnt/shared-storage-user/datafactory-share/gushishen/imagenet",
               PRETRAIN_MODELS_ROOT="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models",
               FID_STATS_ROOT="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats",
               DINOV3_REPO="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3",
               HF_HUB_OFFLINE="1")
    for scale in ("2.9", "2.1"):
        run_scale(scale, env)
    log("Per-file cached_recon E80 sampling for JiT CFG 2.9 and 2.1 completed")


if __name__ == "__main__":
    main()
