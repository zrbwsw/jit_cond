import argparse
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "stage1-dinov3b-dinodit-b-e160-stage2-pixeljit-b-dinodit-noised-e200"
CURRENT_SESSION = "noised-e160-realval-upperbound-jitcfg3p3-gpu1"
CURRENT_LOG = PROJECT_ROOT / "checkpoints" / EXPERIMENT / "eval_checkpoint-160_realval_dinov3b_instance_jitcfg3.3_heun50_image50000/sampling.log"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / EXPERIMENT / "checkpoint-160.pth"


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        log(f"Would wait for {CURRENT_SESSION} to produce FID, then run standard DinoDiT sampling: JiT CFG 2.9, 2.1")
        return
    while True:
        text = CURRENT_LOG.read_text(errors="replace") if CURRENT_LOG.exists() else ""
        if "FID:" in text:
            break
        if "Traceback" in text or "ChildFailedError" in text:
            raise RuntimeError(f"Current real-val sampling failed; see {CURRENT_LOG}")
        if subprocess.run(["tmux", "has-session", "-t", CURRENT_SESSION], check=False).returncode != 0:
            raise RuntimeError("Current real-val sampling stopped without producing FID")
        time.sleep(20)
    log("Current real-val sampling completed; starting standard DinoDiT sampling with JiT CFG 2.9, then 2.1")
    env = os.environ.copy()
    env.update(CHECKPOINT=str(CHECKPOINT), CONDITION_MODE="noised", JIT_CFG_LIST="2.9 2.1",
               DINO_CFG="2.0", GPU_HOST="agent-gpu-l2-1", HF_HUB_OFFLINE="1")
    subprocess.run(["bash", "sh/sample_stage2_ckpt_scales.sh"], cwd=PROJECT_ROOT, env=env, check=True)
    log("Standard DinoDiT sampling for JiT CFG 2.9 and 2.1 completed")


if __name__ == "__main__":
    main()
