import argparse
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch

EXPERIMENT = "stage1-dinov3b-dinodit-b-recon-noflip-e200"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / EXPERIMENT
CHECKPOINT = OUTPUT_DIR / "checkpoint-160.pth"


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def training_processes():
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,comm=,args="], text=True)
    launchers = {}
    for line in output.splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        pid, parent, command, args = fields
        if command == "python" and "stage1.main_dino" in args and EXPERIMENT in args:
            launchers.setdefault(int(parent), []).append(int(pid))
    return launchers


def checkpoint_ready():
    if not CHECKPOINT.is_file():
        return False
    try:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    except (OSError, EOFError, RuntimeError):
        return False
    required = {"model", "optimizer", "model_ema1", "model_ema2", "args"}
    valid = (checkpoint.get("epoch") == 160 and checkpoint.get("training_stage") == "dino"
             and required.issubset(checkpoint))
    del checkpoint
    return valid


def stop_training(launcher, rank_pids):
    log(f"Stopping launcher {launcher} and {len(rank_pids)} ranks after validating {CHECKPOINT}")
    os.kill(launcher, signal.SIGTERM)
    deadline = time.monotonic() + 90
    while training_processes() and time.monotonic() < deadline:
        time.sleep(2)
    remaining = training_processes()
    if remaining:
        pids = {launcher, *(pid for processes in remaining.values() for pid in processes)}
        log(f"Graceful stop timed out; sending SIGKILL to {sorted(pids)}")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(2)
    if training_processes():
        raise RuntimeError("Failed to stop all noflip training processes")
    log("noflip Stage 1 stopped; checkpoint-160.pth and logs were preserved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll", type=float, default=30.0)
    args = parser.parse_args()
    if args.dry_run:
        log(f"Target: {CHECKPOINT}")
        log(f"Ready: {checkpoint_ready()}; training processes: {training_processes()}")
        return
    while not checkpoint_ready():
        if not training_processes():
            raise RuntimeError(f"Training stopped before a valid {CHECKPOINT} was available")
        time.sleep(args.poll)
    launchers = training_processes()
    if len(launchers) != 1:
        raise RuntimeError(f"Expected one noflip launcher, found {launchers}")
    launcher, rank_pids = next(iter(launchers.items()))
    stop_training(launcher, rank_pids)


if __name__ == "__main__":
    main()
