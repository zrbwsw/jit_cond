#!/usr/bin/env bash
# Explicit SSH helper. HOST must be configured in the user's SSH config or resolvable.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash sh/run_on_gpu.sh [--dry-run] HOST [COMMAND ARG...]

HOST may be an SSH alias or user@hostname. No hard-coded cluster/key is used.
REMOTE_PROJECT_ROOT overrides the project directory on the remote machine.
GPU_SSH_KEY optionally selects a private key. No command starts without an
explicit invocation; --dry-run only prints the SSH command.
HELP
}

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then dry_run=true; shift; fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
gpu_host="$1"
shift
[[ "$gpu_host" != -* ]] || { echo 'Invalid SSH host.' >&2; exit 2; }
project_dir="${REMOTE_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
tty_option=-T
if [[ $# -eq 0 ]]; then set -- bash -i; tty_option=-t; fi

remote_args=(bash -c 'cd -- "$1" && shift && exec "$@"' bash "$project_dir" "$@")
printf -v remote_command '%q ' "${remote_args[@]}"
ssh_options=("$tty_option" -o StrictHostKeyChecking=yes -o ConnectTimeout=20
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
if [[ -n "${GPU_SSH_KEY:-}" ]]; then
    ssh_options+=(-i "$GPU_SSH_KEY" -o IdentitiesOnly=yes)
fi
if "$dry_run"; then
    printf '%q ' ssh "${ssh_options[@]}" -- "$gpu_host" "$remote_command"
    printf '\n'
    exit 0
fi
exec ssh "${ssh_options[@]}" -- "$gpu_host" "$remote_command"
