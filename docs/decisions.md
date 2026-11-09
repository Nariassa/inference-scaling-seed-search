# Project 1 – Day 1 locked decisions (inference-time scaling: BoN vs PSP)

Source for defaults: Guimarães & Perona, "Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning", arXiv:2607.21591 (code: github.com/rogerioagjr/PSP).

## Model & sampler
- Backbone: Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5` or mirror), 512x512, fp16
- Sampler: DDIM, eta = 0 (deterministic → all randomness comes from the seed)
- Steps: T = 64
- CFG scale: 7.5
- Why SD1.5: cheapest of the paper's 3 backbones and the one with the largest BoN→PSP gap; SDXL/SD3.5 are out of reach for a 1-week, shared-GPU budget.

## Compute accounting (same for BoN and PSP)
- Unit = denoising step of one trajectory. Total cost C = Σ_t k_t (k_t = trajectories alive at step t).
- Report effective multiplier N̄ = C / T (N̄ = 1 is a single standard sample).
- One CFG step (cond + uncond batched) counts as 1 step. Reward-model calls and x̂0 VAE decodes are logged separately (wall-clock + count), not added to C.
- Harness asserts: BoN at N gives C = N·T exactly.

## Reward / evaluation
- Guidance (selection) reward: ImageReward.
- Held-out metrics (never used for selection): GenEval detector score (+ CLIP score / HPSv2 if time).
- Reward-hacking evidence = gap between ImageReward gains and GenEval gains.

## Prompts
- `geneval_subset_120.jsonl`: 120 GenEval prompts, 20 per category (single_object, two_object, counting, colors, position, color_attr), sampled with seed 2026. Keeps original GenEval metadata so the official evaluator works.

## Key design choice: generate once, replay offline
- Like the paper (Supp. S5/S7), run 32 seeds × 120 prompts × 64 DDIM steps ONCE and cache per (prompt_id, seed, step): intermediate ImageReward on x̂0, plus final image.
- BoN (any N ≤ 32) = pick max final reward among a seed subset. PSP (any schedule) = replay pruning decisions on cached intermediate rewards. Adaptive schedules can be searched on the cache with no GPU.
- Intermediate rewards only at every 4th step (16 checkpoints) to cut VAE+reward cost; PSP prune points must lie on these.
- Report mean ± std over non-overlapping seed groups (e.g., 32 seeds → 4 groups of 8 for N̄ ≤ 4).

## Sweeps (planned)
- BoN seed count: N ∈ {1, 2, 4, 8, 16, 32} at T = 64.
- Compute budget: N̄ ∈ {2, 4, 8, 16} for BoN and PSP (PSP default: start 2N̄ seeds, prune to N̄ at 25%, to N̄/2 at 50%).
- Steps-vs-seeds trade-off at fixed C (needs extra runs at T = 32/16; only if time allows).

## Owners
- A (me): harness + compute counter + cache; BoN sweeps.
- B: PSP replay + schedule sweep + adaptive schedule.
- C: prompt subset (done), GenEval/held-out evaluation.
- D: qualitative grids, diversity, intro/conclusion.
