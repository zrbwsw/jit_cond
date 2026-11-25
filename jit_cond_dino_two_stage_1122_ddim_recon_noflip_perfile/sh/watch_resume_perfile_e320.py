import argparse
import json
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/mnt/shared-storage-user/datafactory-share/gushishen/imagenet/stage1-dinov3b-dinodit-b-recon-noflip-e80")
PROGRESS = CACHE / "migration_progress.json"
CONVERSION_LOG = CACHE / "migration.log"
OUTPUT = PROJECT_ROOT / "checkpoints/stage1-dinov3b-dinodit-b-recon-noflip-e80-stage2-pixeljit-b-dinodit-cached-recon-perfile-e320"
CHECKPOINT = OUTPUT / "checkpoint-last-perfile.pth"
TRAIN_LOG = OUTPUT / "train.log"
SESSION = "stage2-cached-recon-perfile-e45-resume-e320-gpu1"
PYTHON = "/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python"


def log(message):
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def conversion_complete():
    if not PROGRESS.exists():
        return False
    return json.loads(PROGRESS.read_text()).get("complete", False)


def wait_for_conversion():
    while not conversion_complete():
        if CONVERSION_LOG.exists():
            text = CONVERSION_LOG.read_text(errors="replace")
            if "Traceback" in text:
                raise RuntimeError("Conversion failed; see migration.log")
        time.sleep(20)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing migrated checkpoint: {CHECKPOINT}")
    log("Conversion and migrated checkpoint are complete")


def training_ranks():
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,comm=,args="], text=True)
    launchers = set()
    for line in output.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and fields[2] == "python" and "main_jit.py" in fields[3] and OUTPUT.name in fields[3]:
            launchers.add(fields[1])
    return launchers


def wait_for_training():
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        text = TRAIN_LOG.read_text(errors="replace") if TRAIN_LOG.exists() else ""
        if "Epoch: [45]" in text:
            if training_ranks():
                log("Confirmed first resumed batch at epoch index 45 with active training ranks")
                return
        if "Traceback" in text or "ChildFailedError" in text:
            raise RuntimeError(f"Training failed; see {TRAIN_LOG}")
        if not training_ranks() and TRAIN_LOG.exists():
            raise RuntimeError("Training exited before producing a resumed batch")
        time.sleep(10)
    raise TimeoutError("Timed out waiting for the first resumed training batch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        log(f"conversion_complete={conversion_complete()}; checkpoint_exists={CHECKPOINT.exists()}")
        return
    wait_for_conversion()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               IMAGENET_ROOT="/mnt/shared-storage-user/datafactory-share/gushishen/imagenet",
               PRETRAIN_MODELS_ROOT="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/jit_cond_ablation/pretrain_models",
               FID_STATS_ROOT="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/fid_stats",
               DINOV3_REPO="/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/Pixel_DIffusion/Ours_BaseT_rmsnorm2_Residual_1016/SSL_code/dinov3",
               HF_HUB_OFFLINE="1")
    log("Validating migrated checkpoint and per-file cache")
    subprocess.run([PYTHON, "sh/validate_perfile_resume.py"], cwd=PROJECT_ROOT, env=env, check=True)
    if subprocess.run(["tmux", "has-session", "-t", SESSION], check=False).returncode == 0:
        raise RuntimeError(f"Training tmux session already exists: {SESSION}")
    command = (f"cd -- {PROJECT_ROOT} && GPU_HOST=agent-gpu-l2-1 "
               "bash sh/stage2_cached_recon_perfile_resume_e320.sh")
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, command], check=True)
    log(f"Started resumed training in tmux {SESSION}")
    wait_for_training()
    log("Stopping the GPU keepalive after successful training startup")
    subprocess.run(["bash", "/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/scripts/zrb.sh", "stop"],
                   check=True)
    log("Per-file resumed training is running and keepalive is stopped")


if __name__ == "__main__":
    main()
