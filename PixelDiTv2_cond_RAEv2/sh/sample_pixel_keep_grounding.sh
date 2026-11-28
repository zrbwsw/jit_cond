#!/usr/bin/env bash
# 使用同一份 RAEv2 grounding 比较真实类别与 null 类别。
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/sample_pixel.sh" --cfg_keep_grounding "$@"
