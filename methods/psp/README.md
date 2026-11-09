# Progressive Seed Pruning (owner: B)

Replay PSP offline on the shared cache using the selection, validation, scoring,
and report functions from the unchanged official
[rogerioagjr/PSP](https://github.com/rogerioagjr/PSP) script,
`results_scripts/baselines/compute_mt_scheduled_from_rewards.py`.
The wrapper adds input checks, sweep configurations, compute accounting and CSV exports.
No GPU or model downloads are needed. Replay requires Python 3.10+ and pandas;
the combined findings and plots also use NumPy and Matplotlib.

## Inputs

Paths below are relative to the `ai6132` repository root:

- `cache/T64_metrics.csv`: intermediate and final ImageReward trajectories.
- `data/geneval_per_image.csv`: final-image GenEval, with `prompt_id,seed,correct,tag`.
- `../PSP`: the official PSP checkout. Override with `--official-repo` or `PSP_OFFICIAL_REPO`.

Defaults are resolved relative to the scripts, so the commands work from any directory.
The examples below assume you are in `ai6132` with a Python environment containing pandas:

```bash
conda activate geneval-psp

# Baseline: K=8, prune to 4 at step 16, then to 2 at step 32.
bash methods/psp/run_psp.sh

# Seed-count study: K=1,2,4,8,16,32, with a fixed halving rule.
bash methods/psp/run_seed_sweep.sh

# Budget study: hold K=16 fixed; C=128,256,512,1024.
bash methods/psp/run_budget_sweep.sh

# Run both scaling studies and generate their findings, paired CIs and plots.
bash methods/psp/run_scaling_experiments.sh

# Pruning-time sweep: K=8 -> 4 at steps 8,16,24,32,40,48.
bash methods/psp/run_pruning_sweep.sh

# Six schedules with K=8, T=64, C=256 (Nbar=4).
bash methods/psp/run_schedule_sweep.sh

# Run all four sweep suites and generate the scaling findings.
bash methods/psp/run_psp.sh --suite all
```

To use a different cache or a custom schedule:

```bash
bash methods/psp/run_psp.sh \
  --metrics cache/T64_metrics.csv \
  --geneval-csv data/geneval_per_image.csv \
  --official-repo ../PSP \
  --k-init 8 --cutoff-times 12,40 --remaining-particles 4,2 \
  --logical-seeds 0,1,2 --out-dir results/results_psp_custom
```

Set `PSP_PYTHON=/path/to/python` to choose the interpreter without activating an environment.
Run `bash methods/psp/run_psp.sh --help` for all options.

## Selection and compute

- Guidance is always ImageReward (`ir`); the empty `human_preference` column is ignored.
- Cutoffs must be distinct, increasing multiples of 4 with a score for every trajectory.
- GenEval is used only after selecting the final winner, never for pruning or ranking.
- By default, all complete, non-overlapping seed groups are evaluated. With 32 seeds
  per prompt, K=1,2,4,8,16,32 gives 32,16,8,4,2,1 groups respectively. `--logical-seeds 0,1,2`
  requests exactly those three groups and fails if any group is incomplete.
- Compute is `C = K*t1 + keep1*(t2-t1) + ... + keep_last*(T-t_last)`;
  effective multiplier `Nbar = C/T`. The baseline costs `8*16 + 4*16 + 2*32 = 256`.
- The pruning-time sweep varies compute: `C = 256 + 4*t`, from 288 to 448.
  It is a timing/compute trade-off, not a matched-compute comparison.
- The fixed-budget schedule suite compares cutoffs `8,48`, `12,40`, `16,32`,
  `20,24` with survivors `4,2`; survivors `6,1` at `16,32`; and survivors
  `6,5,2` at `8,16,32`. Every configuration costs exactly C=256.
- CSV keys, finite rewards, duplicate rows, checkpoint coverage and final GenEval
  coverage are validated before a sweep starts.

## Two distinct scaling experiments

**Seed count:** hold the halving rule fixed at steps 16 and 32 and vary initial K.
K=4,8,16,32 use K -> K/2 -> K/4; K=2 prunes once to 1 at step 16;
K=1 is a standard-sampling reference. This study measures scaling under one rule;
compute grows with K and is explicitly reported. The two boundary references do
not have two pruning stages.

**Compute budget:** hold K=16 and T=64 fixed and vary C over 128,256,512,1024.
The wrapper enumerates all exact-budget two-stage schedules on the declared cutoff
grid {4,8,16,24,32,40,48,56,60} and descending survivor pairs from {8,4,2,1}.
There are 1,5,5 schedules at the first three budgets; C=1024 is the no-pruning
boundary with 16 completed seeds. All 12 configurations are reported. No schedule
is selected using either evaluation score. The plotted reference schedules are
specified analytically in `BUDGET_REFERENCE`, not chosen by their measured performance.

`run_scaling_experiments.sh` (or `--suite scaling`) produces `findings/findings.md`,
the two study plots in PNG/PDF, a reference budget curve, within-budget metric ranges,
and paired-difference CSVs. CIs resample prompts within category 5,000 times after
averaging seed groups per prompt. Both studies use the same cache, so they answer
distinct questions but are not independent replications. This is exploratory analysis
of an existing cache, not a prospective preregistered study.

The former coupled K/budget results are preserved in `legacy_coupled/` when the new
budget study is first run. They are superseded by the separate `seeds/` and `budget/` suites.

## Outputs

Each run writes under `results/results_psp/<suite>/` (`single`, `seeds`, `budget`, `pruning`,
or `schedule`), or under `--out-dir/<suite>/`:

| File | Contents |
|---|---|
| `summary.csv` | Schedule, C, Nbar, number of groups, ImageReward and GenEval means/std |
| `per_group.csv` | Metrics for each non-overlapping seed group |
| `selections.csv` | Selected prompt/seed pairs with final ImageReward and binary GenEval |
| `per_tag.csv` | Category accuracies for each configuration and seed group |
| `official_reports/*.txt` | Reports produced with the author's report formatter |
| `protocol.json` | Input paths/hashes, author-script hash, schedules and group indices |

Means and sample standard deviations are computed across group means. Standard
deviation is blank when only one group is available. GenEval overall is the mean
of category accuracies. `selections.csv` includes the column `correct`; intermediate
GenEval scores are unnecessary. Repeating a suite replaces its output files.

## Verification

```bash
python -m unittest discover -s methods/psp -p 'test_*.py' -v
```

The checks compare wrapper output against the official CLI on a synthetic cache,
including tie handling, and exercise missing checkpoints, incomplete seed groups,
held-out score coverage, and compute accounting.
