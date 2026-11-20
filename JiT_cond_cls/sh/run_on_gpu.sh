#!/usr/bin/env bash
# 只负责连接 GPU 并在共享项目目录执行命令；tmux 和训练由你显式启动。
set -euo pipefail

usage() {
    cat <<'EOF'
用法：bash sh/run_on_gpu.sh [--dry-run] HOST [COMMAND ARG...]

HOST: agent-gpu-l2-1 或 agent-gpu-l2-2
例如：
  bash sh/run_on_gpu.sh agent-gpu-l2-1 bash scripts/train_rmsnorm2_reverse_nomls_B80.sh
  bash sh/run_on_gpu.sh agent-gpu-l2-1 nvidia-smi
  bash sh/run_on_gpu.sh agent-gpu-l2-2 nvidia-smi

不提供 COMMAND 时，进入 GPU 上的项目目录，打开交互式 Bash。
优先使用本机 SSH 配置中的别名；无别名时使用下面的集群连接信息。
GPU_SSH_KEY 可指定私钥路径。--dry-run 仅显示命令，不建立连接。
EOF
}

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=true
    shift
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
gpu_host="$1"
shift

case "$gpu_host" in
    agent-gpu-l2-1)
        ssh_user='agent-gpu-l2-1-2132956-74593-cfcd2.gushishen+root.ailab-safeshared.pod'
        worker_name='gpu-lg-cmc-h-h200-0127'
        aliases=(agent-gpu-l2-1)
        ;;
    agent-gpu-l2-2)
        ssh_user='agent-gpu-l2-2-84043580-cfcd2.gushishen+root.ailab-safeshared.pod'
        worker_name='gpu-lg-cmc-h-h200-1777'
        aliases=(agent-gpu-l2-2)
        ;;
    *) echo "未知 GPU：$gpu_host（使用 agent-gpu-l2-1 或 agent-gpu-l2-2）" >&2; exit 2 ;;
esac

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tty_option=-T
if [[ $# -eq 0 ]]; then
    set -- bash -i
    tty_option=-t
elif [[ -t 0 ]]; then
    tty_option=-t
fi

# 在目标 GPU 上调用时直接执行，不再绕一圈 SSH。
local_name="$(hostname -s)"
if [[ "$local_name" == "$gpu_host" || ( -n "$worker_name" && "$local_name" == "$worker_name" ) ]]; then
    if "$dry_run"; then
        printf 'cd -- %q && ' "$project_dir"
        printf '%q ' "$@"
        printf '\n'
        exit 0
    fi
    cd -- "$project_dir"
    export TZ="${TZ:-Asia/Shanghai}"
    exec "$@"
fi

# ssh -G 只解析本机配置，不建立连接。别名不存在时，hostname 仍等于别名。
target="$ssh_user@h.pjlab.org.cn"
use_alias=false
for alias_name in "${aliases[@]}"; do
    configured_host="$(ssh -G "$alias_name" 2>/dev/null | awk '$1 == "hostname" { print $2 }')" || configured_host=''
    if [[ -n "$configured_host" && "$configured_host" != "$alias_name" ]]; then
        target="$alias_name"
        use_alias=true
        break
    fi
done

ssh_options=("$tty_option" -o StrictHostKeyChecking=yes -o UpdateHostKeys=no
    -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
if ! "$use_alias"; then
    ssh_options+=(-p 22)
fi
if [[ -n "${GPU_SSH_KEY:-}" ]] || ! "$use_alias"; then
    identity="${GPU_SSH_KEY:-/data/.ssh/id_ed25519_pjlab_20261013}"
    ssh_options+=(-i "$identity" -o IdentitiesOnly=yes)
fi

# 两层 Bash 均按 argv 传参，保留路径中的空格、括号及参数中的特殊字符。
remote_args=(bash -c 'cd -- "$1" && shift && export TZ="${TZ:-Asia/Shanghai}" && exec "$@"' bash "$project_dir" "$@")
printf -v remote_command '%q ' "${remote_args[@]}"
if "$dry_run"; then
    printf '%q ' ssh "${ssh_options[@]}" "$target" "$remote_command"
    printf '\n'
    exit 0
fi
exec ssh "${ssh_options[@]}" "$target" "$remote_command"
