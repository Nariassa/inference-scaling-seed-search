# Two PSP scaling experiments

## 1. Seed-count scaling under a fixed pruning rule

Vary initial K over 1,2,4,8,16,32 at T=64. Halve survivors at steps 16 and 32 when possible; K=1 is a standard-sampling reference and K=2 has only the first prune. K=4..32 share exactly the same fractional pruning rule. Compute increases with K, so this measures the performance/cost of scaling seeds under that rule.

| K | C | Nbar | ImageReward | GenEval |
| --- | --- | --- | --- | --- |
| 1 | 64 | 1.00 | -0.1249 | 0.4370 |
| 2 | 80 | 1.25 | 0.1873 | 0.4760 |
| 4 | 128 | 2.00 | 0.5353 | 0.5146 |
| 8 | 256 | 4.00 | 0.8263 | 0.5563 |
| 16 | 512 | 8.00 | 1.0768 | 0.5958 |
| 32 | 1024 | 16.00 | 1.2466 | 0.6167 |

Observed K=4 to K=32 change: ImageReward difference: +0.7114 (paired 95% CI +0.6226 to +0.8071); GenEval difference: +10.21 percentage points (paired 95% CI +5.10 to +15.42).

## 2. Compute allocation with fixed initial seed count

Hold K=16 and T=64 fixed. Target C=128,256,512,1024 (Nbar=2,4,8,16). Enumerate all exact-budget two-stage schedules from cutoffs {4,8,16,24,32,40,48,56,60} and descending survivor pairs from {8,4,2,1}. There are 1,5,5 feasible schedules at the first three budgets. C=1024 is the no-pruning boundary: all 16 trajectories run to completion.

The following representative schedules are specified analytically in code: (4,8)/(2,1), (8,32)/(4,1), (16,32)/(8,4), and no pruning. No schedule is chosen by ImageReward or GenEval. All 12 configurations remain in budget/summary.csv; their within-budget ranges are in budget_configuration_ranges.csv.

| K | C | Nbar | ImageReward | GenEval |
| --- | --- | --- | --- | --- |
| 16 | 128 | 2.00 | 0.2136 | 0.4750 |
| 16 | 256 | 4.00 | 0.7721 | 0.5667 |
| 16 | 512 | 8.00 | 1.0768 | 0.5958 |
| 16 | 1024 | 16.00 | 1.1478 | 0.6042 |

Observed C=128 to C=1024 change with K=16 fixed: ImageReward difference: +0.9342 (paired 95% CI +0.8056 to +1.0679); GenEval difference: +12.92 percentage points (paired 95% CI +7.92 to +17.92).

## Interpretation and uncertainty

These are distinct experimental questions on the same cached trajectories, not independent datasets or replications. This is exploratory analysis of an existing cache, not a prospective preregistered study. The first scales initial seeds and compute together under a fixed rule; the second fixes initial seeds and changes the amount and allocation of denoising work. Schedule details still matter within a budget. The grid is limited; no global optimum is claimed. GenEval is used only for final evaluation.

Summary CSV standard deviations describe variation across complete, non-overlapping seed groups. Paired difference CIs use 5,000 bootstrap resamples of prompts within category (seed 2026), averaging seed groups per prompt first. They describe prompt-sampling uncertainty conditional on this seed cache; they do not include new-seed or model uncertainty. Compute counts trajectory-denoising steps and excludes reward scoring and VAE decoding.
