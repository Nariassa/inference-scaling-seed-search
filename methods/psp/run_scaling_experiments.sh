#!/usr/bin/env bash
# Run both distinct studies and produce their comparison plots and findings.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_psp.sh" --suite scaling "$@"
