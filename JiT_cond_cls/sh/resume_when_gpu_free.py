#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "JiT-B16-DINOv3B-CLS-BlockAdaLN-ValGrounding-B600"
OUTPUT = ROOT / "checkpoints" / EXPERIMENT
LOG_PATH = OUTPUT / "train.log"
CHECKPOINT = OUTPUT / "checkpoint-last.pth"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def gpu_state():
    try:
        result = subprocess.run(
            ["bash", "sh/run_on_gpu.sh", "agent-gpu-l2-2", "nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        )
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "").strip().splitlines()
        log(f"GPU query failed: {output[-1] if output else exc}")
        return None
    state = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, utilization, memory = parts
        state.append((int(index), int(utilization.rstrip(" %")), int(memory.split()[0])))
    return state


def gpus_are_free(state):
    return bool(state) and len(state) == 8 and all(utilization <= 1 and memory <= 100 for _, utilization, memory in state)


def pane_command(session):
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_current_command}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def checkpoint_epoch():
    import torch
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert "optimizer" in checkpoint and "model_ema1" in checkpoint and "model_ema2" in checkpoint
    return checkpoint["epoch"]


def backup_log():
    destination = OUTPUT / "train_before_resume_to80.log"
    if destination.exists():
        log(f"Backup already exists: {destination}")
        return
    shutil.copy2(LOG_PATH, destination)
    log(f"Backed up train.log to {destination}")


def start_training():
    if pane_command(EXPERIMENT) is None:
        subprocess.run(["tmux", "new-session", "-d", "-s", EXPERIMENT, "-c", str(ROOT)], check=True)
    elif pane_command(EXPERIMENT) == "ssh":
        raise RuntimeError(f"{EXPERIMENT} is already running")
    old_mtime = LOG_PATH.stat().st_mtime_ns
    command = (
        "EPOCHS=80 bash sh/train_jit_b16_grounding_8gpu.sh "
        f"2>&1 | tee -a '{OUTPUT / 'launcher_resume80.log'}'"
    )
    subprocess.run(["tmux", "send-keys", "-t", EXPERIMENT, command, "Enter"], check=True)
    log("Started EPOCHS=80 training after all eight GPUs became free")
    return old_mtime


def wait_for_log(needle, old_mtime, timeout=300):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            stat = LOG_PATH.stat()
            if stat.st_mtime_ns > old_mtime and needle in LOG_PATH.read_text(errors="replace"):
                log(f"Confirmed in new train.log: {needle}")
                return
        except FileNotFoundError:
            pass
        if pane_command(EXPERIMENT) != "ssh" and time.monotonic() - started > 30:
            raise RuntimeError(f"Training exited before log contained {needle!r}")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {needle!r}")


def main():
    parser = argparse.ArgumentParser(description="Resume jit_cond_cls to epoch 80 after GPU2 is fully free.")
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--required-checks", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    saved_epoch = checkpoint_epoch()
    if saved_epoch != 40:
        raise RuntimeError(f"Expected old checkpoint epoch 40, got {saved_epoch}")
    log("Checkpoint epoch 40 is readable; resume will start at epoch 41")
    if args.dry_run:
        state = gpu_state()
        log(f"GPU state: {state}")
        log(f"All eight GPUs free: {gpus_are_free(state)}")
        log("Dry-run: no log was backed up and no training was started")
        return

    free_checks = 0
    last_report = 0.0
    while free_checks < args.required_checks:
        state = gpu_state()
        if gpus_are_free(state):
            free_checks += 1
            log(f"All eight GPUs are free ({free_checks}/{args.required_checks})")
        else:
            free_checks = 0
        now = time.monotonic()
        if now - last_report >= 300:
            log(f"Waiting for GPU2: {state}")
            last_report = now
        if free_checks < args.required_checks:
            time.sleep(args.poll)

    backup_log()
    old_mtime = start_training()
    wait_for_log("Resumed checkpoint", old_mtime)
    wait_for_log("Epoch: [41]", old_mtime)
    log("cond_cls resumed successfully at epoch 41 and will stop after epoch 80")


if __name__ == "__main__":
    main()
