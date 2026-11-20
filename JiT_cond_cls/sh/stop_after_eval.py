#!/usr/bin/env python3
import argparse
import re
import subprocess
import time
from pathlib import Path


def completed_evaluation(log_path, target_epoch):
    latest_epoch = None
    evaluation_epoch = None
    result = None
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        match = re.search(r"Epoch: \[(\d+)\]", line)
        if match:
            latest_epoch = int(match.group(1))
        if "Generation step 0/" in line:
            evaluation_epoch = latest_epoch
        if "FID:" in line and evaluation_epoch == target_epoch:
            result = line.strip()
    return result


def pane_command(session):
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_current_command}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser(description="Stop a tmux training session after an epoch's FID/IS completes.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--poll", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Waiting for epoch {args.epoch} evaluation in {args.log}", flush=True)
    last_report = 0.0
    while True:
        result = completed_evaluation(args.log, args.epoch)
        if result is not None:
            print(f"Evaluation completed: {result}", flush=True)
            break
        if args.dry_run:
            print("Evaluation is still pending; no session was stopped.", flush=True)
            return 0
        command = pane_command(args.session)
        if command != "ssh":
            raise RuntimeError(
                f"Training session {args.session!r} is no longer active (pane command: {command!r})"
            )
        now = time.monotonic()
        if now - last_report >= 300:
            print(f"Still waiting; training pane is {command!r}.", flush=True)
            last_report = now
        time.sleep(args.poll)

    if args.dry_run:
        print("Dry-run: no session was stopped.", flush=True)
        return 0

    subprocess.run(["tmux", "send-keys", "-t", args.session, "C-c"], check=True)
    print(f"Sent Ctrl-C to {args.session}; waiting for training to exit.", flush=True)
    for _ in range(30):
        command = pane_command(args.session)
        if command != "ssh":
            print(f"Training stopped; pane command is {command!r}.", flush=True)
            return 0
        time.sleep(2)
    raise RuntimeError(f"Ctrl-C was sent, but {args.session} is still active")


if __name__ == "__main__":
    main()
