#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_HOST="${GPU_HOST:-agent-gpu-l2-1}"
OLD_SESSION="${OLD_SESSION:-raev2-dinov3l-k7-noised-e160}"
OLD_DIR="${OLD_DIR:-$PROJECT_ROOT/checkpoints/stage1-raev2-dinov3l-k7-stage2-pixeljit-b-noised-e160}"
NEW_DIR="${NEW_DIR:-$PROJECT_ROOT/checkpoints/stage1-raev2-dinov3l-k7-stage2-pixeljit-b-noised-e80}"
KEEPALIVE_SCRIPT="${KEEPALIVE_SCRIPT:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/scripts/zrb.sh}"
if [[ "${1:-}" == --dry-run ]]; then
    printf 'Wait for the current E20 FID in %s\n' "$OLD_DIR/train.log"
    printf 'Stop E160, preserve checkpoint-last.pth as checkpoint-20.pth, rename to %s\n' "$NEW_DIR"
    printf 'Resume noised from E20 to E80, then run pred-e80 and restore keepalive\n'
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo '用法：bash sh/queue_noised_e80_then_pred_e80.sh [--dry-run]' >&2
    exit 2
fi
for path in "$OLD_DIR/checkpoint-last.pth" "$OLD_DIR/train.log" "$PROJECT_ROOT/sh/train_pixel_raev2.sh" "$PROJECT_ROOT/sh/train_pixel_raev2_pred_e80.sh"; do
    if [[ ! -f "$path" ]]; then
        echo "Required path not found: $path" >&2
        exit 1
    fi
done
if [[ -e "$NEW_DIR" ]]; then
    echo "Target experiment directory already exists: $NEW_DIR" >&2
    exit 1
fi

python3 - "$OLD_DIR/train.log" "$OLD_SESSION" <<'PY'
from pathlib import Path
import subprocess
import sys
import time
path = Path(sys.argv[1])
session = sys.argv[2]
existing = path.read_text(errors='replace')
if 'FID:' in existing:
    lines = [line for line in existing.splitlines() if 'FID:' in line]
    print('Current online evaluation already completed: ' + lines[-1], flush=True)
    raise SystemExit(0)
offset = path.stat().st_size
while True:
    with path.open('r', errors='replace') as handle:
        handle.seek(offset)
        appended = handle.read()
    if 'FID:' in appended:
        lines = [line for line in appended.splitlines() if 'FID:' in line]
        print('Current online evaluation completed: ' + lines[-1], flush=True)
        break
    if any(error in appended for error in ('Traceback', 'ChildFailedError', 'CUDA out of memory')):
        print(appended[-4000:], flush=True)
        raise SystemExit('Training failed before the current evaluation completed')
    if subprocess.run(['tmux', 'has-session', '-t', session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise SystemExit('Original training session ended before the current FID was produced')
    time.sleep(10)
PY

sleep 5
bash "$PROJECT_ROOT/sh/run_on_gpu.sh" "$GPU_HOST" python3 - "$OLD_DIR" <<'PY'
import os
from pathlib import Path
import signal
import sys
output = os.fsencode(sys.argv[1])
pids = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        argv = (entry / 'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if b'torch.distributed.run' in argv and any(arg.rsplit(b'/', 1)[-1] == b'main_jit.py' for arg in argv) and output in argv:
        pids.append(int(entry.name))
if len(pids) != 1:
    raise SystemExit(f'Expected one E160 torchrun launcher, found {pids}; refusing to stop anything')
os.kill(pids[0], signal.SIGTERM)
print(f'Stopped E160 torchrun launcher pid={pids[0]}', flush=True)
PY

while tmux has-session -t "$OLD_SESSION" 2>/dev/null; do
    sleep 5
done
bash "$PROJECT_ROOT/sh/run_on_gpu.sh" "$GPU_HOST" bash "$KEEPALIVE_SCRIPT" stop

start_keepalive() {
    bash "$PROJECT_ROOT/sh/run_on_gpu.sh" "$GPU_HOST" bash "$KEEPALIVE_SCRIPT" start || true
}
trap start_keepalive EXIT

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${PYTHON_BIN:-/mnt/shared-storage-user/datafactory-share/gushishen/agent-z/envs/irepa-py310-cu126-20261015/bin/python}" - "$OLD_DIR/checkpoint-last.pth" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)
epoch = checkpoint.get('epoch')
if epoch != 20:
    raise SystemExit(f'Expected checkpoint epoch 20, found {epoch}; refusing to resume')
print('Verified resume checkpoint epoch=20', flush=True)
PY
cp --reflink=auto -- "$OLD_DIR/checkpoint-last.pth" "$OLD_DIR/checkpoint-20.pth"
mv -- "$OLD_DIR" "$NEW_DIR"
printf 'Renamed experiment directory to %s\n' "$NEW_DIR"

set +e
CONDITION_MODE=noised EPOCHS=80 RESUME="$NEW_DIR/checkpoint-20.pth" \
    OUTPUT_DIR="$NEW_DIR" EXP_NAME="$(basename "$NEW_DIR")" GPU_HOST="$GPU_HOST" \
    bash "$PROJECT_ROOT/sh/train_pixel_raev2.sh"
noised_status=$?
set -e
if [[ "$noised_status" -ne 0 ]]; then
    exit "$noised_status"
fi

set +e
GPU_HOST="$GPU_HOST" bash "$PROJECT_ROOT/sh/train_pixel_raev2_pred_e80.sh"
pred_status=$?
set -e
exit "$pred_status"
