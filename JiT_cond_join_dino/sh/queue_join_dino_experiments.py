#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ("dinov3b", "train_jit_b16_join_dinov3_8gpu_h200.sh", "JiT-B16-JoinDino-dinov3b-B200"),
    ("dinov2b", "train_jit_b16_join_dinov2_8gpu_h200.sh", "JiT-B16-JoinDino-dinov2b-B200"),
]


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def gpu_state():
    result = subprocess.run(
        ["bash", "sh/run_on_gpu.sh", "agent-gpu-l2-1", "nvidia-smi",
         "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    )
    state = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, utilization, memory = parts
        state.append((int(index), int(utilization.rstrip(" %")), int(memory.split()[0])))
    return state


def gpus_are_free(state):
    return bool(state) and len(state) == 8 and all(utilization <= 1 and memory <= 200 for _, utilization, memory in state)


def wait_for_free_gpus(poll, required_checks):
    checks = 0
    while checks < required_checks:
        state = gpu_state()
        if gpus_are_free(state):
            checks += 1
            log(f"GPU1 is free ({checks}/{required_checks})")
        else:
            checks = 0
            log(f"Waiting for GPU1: {state}")
        if checks < required_checks:
            time.sleep(poll)


def pane_command(session):
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_current_command}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def start_experiment(script, session):
    output = ROOT / "checkpoints" / session
    output.mkdir(parents=True, exist_ok=True)
    train_log = output / "train.log"
    if train_log.exists():
        backup = output / "train_before_queue.log"
        if not backup.exists():
            shutil.copy2(train_log, backup)
            log(f"Backed up previous log to {backup}")
    if pane_command(session) is None:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT)], check=True)
    elif pane_command(session) == "ssh":
        raise RuntimeError(f"{session} is already running")
    command = f"bash sh/{script} 2>&1 | tee -a '{output / 'launcher.log'}'"
    subprocess.run(["tmux", "send-keys", "-t", session, command, "Enter"], check=True)
    log(f"Started {session} on GPU1")
    return output


def wait_for_completion(session, output, poll):
    train_log = output / "train.log"
    started = time.monotonic()
    while True:
        text = train_log.read_text(errors="replace") if train_log.exists() else ""
        if "Training time:" in text:
            checkpoint = output / "checkpoint-last.pth"
            import torch
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state["epoch"] != 200 or not hasattr(state.get("args"), "save_interval"):
                raise RuntimeError(f"Unexpected final checkpoint for {session}: epoch={state['epoch']}")
            log(f"Completed {session}: epoch=200")
            return
        command = pane_command(session)
        if command != "ssh" and time.monotonic() - started > 30:
            raise RuntimeError(f"{session} stopped before normal completion (pane={command!r})")
        time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description="Queue dinov3b then dinov2b on GPU1.")
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--required-checks", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log("Queue: dinov3b -> dinov2b, 200 epochs each, save_interval=20")
    if args.dry_run:
        log(f"Current GPU state: {gpu_state()}")
        log("Dry-run: no experiment was started")
        return

    (ROOT / "checkpoints").mkdir(exist_ok=True)
    wait_for_free_gpus(args.poll, args.required_checks)
    for _encoder, script, session in EXPERIMENTS:
        output = start_experiment(script, session)
        wait_for_completion(session, output, args.poll)
    log("Both experiments completed")


if __name__ == "__main__":
    main()
