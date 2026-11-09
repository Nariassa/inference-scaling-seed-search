# Progressive Seed Pruning (owner: B)

Runs PSP offline on the shared cache, using the official replay script from
[rogerioagjr/PSP](https://github.com/rogerioagjr/PSP) (`results_scripts/baselines/compute_mt_scheduled_from_rewards.py`).

Inputs: `cache/sd15_geneval_T64/metrics.csv` (from the harness) and `geneval_per_image.csv` (from `eval/`).

Notes:
- Pass `--guidance-metric ir`. The HPS column in our cache is empty, and the script fails without this flag.
- Prune points must be multiples of 4 (x̂0 is scored every 4 steps).
- Report compute as C = Σ_t k_t trajectory-steps, the same unit as Best-of-N.

Put PSP wrapper scripts, schedule sweeps and result CSVs here.
