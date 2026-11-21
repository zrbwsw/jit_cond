#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path

B80_ROOT = Path(__file__).resolve().parents[1]
CLS_ROOT = B80_ROOT.parent / "JiT_cond_cls"
B80_SESSION = "JiT-B16-DINOv3B-Noisy-PixelDiT2Proj-B80"
CLS_SESSION = "JiT-B16-DINOv3B-CLS-BlockAdaLN-ValGrounding-B600"
B80_OUTPUT = B80_ROOT / "checkpoints" / B80_SESSION
CLS_OUTPUT = CLS_ROOT / "checkpoints" / CLS_SESSION
B80_LOG = B80_OUTPUT / "train.log"
B80_CHECKPOINT = B80_OUTPUT / "checkpoint-last.pth"
CLS_LOG = CLS_OUTPUT / "train.log"


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def pane_command(session):
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_current_command}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def send_command(session, command):
    subprocess.run(["tmux", "send-keys", "-t", session, command, "Enter"], check=True)


def epoch_total_completed(log_path, epoch):
    try:
        text = Path(log_path).read_text(errors="replace")
    except FileNotFoundError:
        return False
    return re.search(rf"Epoch: \[{epoch}\] Total time:", text) is not None


def checkpoint_info(path):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint["epoch"]
    has_new_format = hasattr(checkpoint.get("args"), "save_interval")
    assert "optimizer" in checkpoint and "model_ema1" in checkpoint and "model_ema2" in checkpoint
    return epoch, has_new_format


def wait_for_checkpoint(epoch, session, poll):
    log(f"Waiting for checkpoint epoch >= {epoch}")
    while True:
        try:
            saved_epoch, has_new_format = checkpoint_info(B80_CHECKPOINT)
            if saved_epoch >= epoch:
                log(f"Checkpoint ready: epoch={saved_epoch}, new_format={has_new_format}")
                return saved_epoch, has_new_format
        except Exception:
            pass
        if pane_command(session) != "ssh":
            raise RuntimeError(f"{session} stopped before checkpoint epoch {epoch}")
        time.sleep(poll)


def stop_training(session):
    if pane_command(session) == "ssh":
        send_command(session, "C-c")
    for _ in range(45):
        command = pane_command(session)
        if command != "ssh":
            log(f"Training stopped: pane={command!r}")
            time.sleep(3)
            return
        time.sleep(2)
    raise RuntimeError(f"Could not stop {session}")


def backup_log(source, destination):
    if Path(destination).exists():
        log(f"Backup already exists: {destination}")
        return
    shutil.copy2(source, destination)
    log(f"Backed up {source} to {destination}")


def start_training(session, project_root, command):
    if pane_command(session) is None:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(project_root)], check=True)
    send_command(session, command)
    log(f"Started command in {session}: {command}")


def wait_for_log_text(log_path, needle, old_mtime, session, timeout=240):
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        try:
            stat = log_path.stat()
            if stat.st_mtime_ns > old_mtime and needle in log_path.read_text(errors="replace"):
                log(f"Confirmed in new train.log: {needle}")
                return
        except FileNotFoundError:
            pass
        command = pane_command(session)
        if command != "ssh" and time.monotonic() - started > 30:
            raise RuntimeError(f"{session} exited before log contained {needle!r}")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {needle!r} in {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Resume B80 at a checkpoint, then queue jit_cond_cls after normal completion.")
    parser.add_argument("--restart-epoch", type=int, default=20)
    parser.add_argument("--poll", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log(f"B80 output: {B80_OUTPUT}")
    log(f"CLS output: {CLS_OUTPUT}")
    log(f"Planned checkpoint switch at epoch {args.restart_epoch}; final B80 checkpoint must be epoch 80")
    if args.dry_run:
        log(f"B80 pane: {pane_command(B80_SESSION)!r}; CLS pane: {pane_command(CLS_SESSION)!r}")
        return

    while not epoch_total_completed(B80_LOG, args.restart_epoch):
        if pane_command(B80_SESSION) != "ssh":
            raise RuntimeError(f"{B80_SESSION} stopped before epoch {args.restart_epoch}")
        log(f"Waiting for epoch {args.restart_epoch} to finish")
        time.sleep(args.poll)
    saved_epoch, _ = wait_for_checkpoint(args.restart_epoch, B80_SESSION, 5)
    stop_training(B80_SESSION)
    backup_log(B80_LOG, B80_OUTPUT / "train_before_new_checkpoint_logic.log")

    old_mtime = B80_LOG.stat().st_mtime_ns
    restart = (
        "GPU_HOST=agent-gpu-l2-2 DIST_PORT=29401 bash sh/train_jit_b16_grounding_8gpu.sh "
        f"2>&1 | tee -a '{B80_OUTPUT / 'launcher.log'}'"
    )
    start_training(B80_SESSION, B80_ROOT, restart)
    wait_for_log_text(B80_LOG, "Resumed checkpoint", old_mtime, B80_SESSION)
    wait_for_log_text(B80_LOG, f"Epoch: [{saved_epoch + 1}]", old_mtime, B80_SESSION)
    log("B80 resumed with the new checkpoint logic")

    log("Waiting for B80 to finish normally before starting jit_cond_cls")
    while pane_command(B80_SESSION) == "ssh":
        time.sleep(args.poll)
    if "Training time:" not in B80_LOG.read_text(errors="replace"):
        raise RuntimeError("B80 stopped without a normal Training time message; jit_cond_cls will not start")
    final_epoch, new_format = checkpoint_info(B80_CHECKPOINT)
    if final_epoch != 80 or not new_format:
        raise RuntimeError(f"Unexpected final B80 checkpoint: epoch={final_epoch}, new_format={new_format}")
    log("B80 completed with checkpoint epoch 80")

    backup_log(CLS_LOG, CLS_OUTPUT / "train_before_resume_from40.log")
    cls_old_mtime = CLS_LOG.stat().st_mtime_ns
    cls_command = f"bash sh/train_jit_b16_grounding_8gpu.sh 2>&1 | tee -a '{CLS_OUTPUT / 'launcher.log'}'"
    start_training(CLS_SESSION, CLS_ROOT, cls_command)
    wait_for_log_text(CLS_LOG, "Resumed checkpoint", cls_old_mtime, CLS_SESSION)
    wait_for_log_text(CLS_LOG, "Epoch: [41]", cls_old_mtime, CLS_SESSION)
    log("jit_cond_cls resumed successfully at epoch 41")


if __name__ == "__main__":
    main()
