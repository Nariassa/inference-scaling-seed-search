#!/usr/bin/env bash
# Single schedule by default; accepts every psp_replay.py option, including --suite all.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PSP_PYTHON:-python3}" "${SCRIPT_DIR}/psp_replay.py" "$@"
