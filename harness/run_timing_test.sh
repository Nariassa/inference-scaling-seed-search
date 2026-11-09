#!/usr/bin/env bash
# Times 2 prompts x 32 seeds at the project settings, then estimates the full 120-prompt run.
# Run from the repo root:  bash harness/run_timing_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=cache/timing_geneval
rm -rf "$OUT"
python harness/generate_cache.py \
  --prompts-path data/geneval_subset_120.jsonl \
  --output-dir "$OUT" \
  --num-prompts 2 --num-seeds 32 --time-steps 64 \
  --batch-size "${BATCH_SIZE:-16}" --decode-batch "${DECODE_BATCH:-4}"
python - <<'PY'
import pandas as pd
log = pd.read_csv("cache/timing_geneval/compute_log.csv")
s = log["sec_total"].iloc[-1]          # 2nd prompt: excludes one-time warm-up
print(f"\n{s:.1f} s per prompt (32 seeds, T=64)")
for n in (120, 90, 60):
    print(f"  {n} prompts -> {s * n / 3600:.1f} GPU-hours")
PY
