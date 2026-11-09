# Generation harness and Best-of-N: details

Generate once, replay many times. One GPU run caches every (prompt, seed) trajectory's
intermediate and final ImageReward. Best-of-N, PSP, and any pruning schedule are then
computed offline from that cache in seconds, on the exact same seeds.

| File | What it does | GPU? |
|---|---|---|
| `harness/generate_cache.py` | SD v1.5 + DDIM(η=0), T=64, CFG 7.5. Caches x̂0 ImageReward every 4 steps, final image + reward, and a compute log | yes |
| `methods/bon/bon_replay.py` | Best-of-N seed-count and compute-budget sweeps from the cache, with plots | no |
| `harness/run_timing_test.sh` | 2 prompts × 32 seeds, to estimate the full run | yes |
| `eval/geneval_bridge.py` | Exports the cache to GenEval's folder layout, and parses GenEval results into `geneval_per_image.csv` | no |
| `notebooks/colab_pipeline.ipynb` | The whole workflow on Colab, with Drive-backed resume | yes |
| `data/geneval_subset_120.jsonl` | 120 GenEval prompts, 20 per category | — |

## 1. Environment

Either use the PSP repo's environment so both methods share identical library versions:

```bash
git clone https://github.com/rogerioagjr/PSP.git
conda create -n psp python=3.10 -y && conda activate psp
cd PSP && bash setup/setup.sh        # torch 2.4, diffusers, ImageReward, hpsv2
```

or install the tested pins from `requirements.txt` (this is what the Colab notebook does):

```bash
pip install -r requirements.txt && pip uninstall -y peft   # peft breaks diffusers with transformers 4.38
```

The SD v1.5 weights load from `stable-diffusion-v1-5/stable-diffusion-v1-5` (the original
`runwayml/...` repo was removed from Hugging Face; pass `--model-id` to change it).

## 2. Timing test (do this first)

```bash
bash harness/run_timing_test.sh            # writes cache/timing_geneval/ and prints seconds per prompt
```

Full-run estimate = seconds per prompt × 120 prompts. If that's too long for one evening,
cut prompts with `--num-prompts`, not seeds — the BoN sweep needs 32 seeds for N=32.

## 3. Full cache run

```bash
python harness/generate_cache.py \
  --prompts-path data/geneval_subset_120.jsonl \
  --output-dir cache/sd15_geneval_T64 \
  --num-seeds 32 --time-steps 64 --batch-size 16
```

- **Keep `geneval` in the output folder name.** PSP's replay scripts look for `*geneval*/metrics.csv`.
- **Resumable:** rerun the same command after a crash and finished prompts are skipped.
- **Sharding across GPUs:** use `--prompt-start-id` and `--num-prompts`, with one output folder per
  shard under the same parent. Both replay scripts merge shards automatically.
- **Out of memory:** lower `--batch-size` (seeds in parallel) or `--decode-batch`.
- **`--save-x0`:** also saves 256px x̂0 previews, useful for D's qualitative figures.

## 4. Outputs (same format as PSP's `collect_image_reward_signal.py`)

```
cache/sd15_geneval_T64/
  metrics.csv       prompt_id, prompt, seed, step, timestep, image_reward, human_preference
  compute_log.csv   per prompt: traj_steps (counted), expected, UNet calls, seconds
  samples/00000/00000.png ...      final images, one per seed
  args.json
```

- `step` is 1-indexed. Rows exist for steps 4, 8, ..., 60 (x̂0) and step 64 (final image).
- Seed convention matches PSP: `seed = 0 + prompt_id × 32 + i`.
- `human_preference` is left empty (we guide with ImageReward only).

## 5. Compute accounting

- **Unit:** one denoising step of one trajectory. C = Σ_t k_t, and N̄ = C / T.
- **CFG:** the cond+uncond pass for one trajectory at one step counts as 1.
- **Measured, not computed:** the counter wraps the UNet's forward and counts rows actually processed.
  The harness stops with an error if any prompt's count isn't exactly seeds × T.
- **Not counted in C:** reward-model calls and x̂0 VAE decodes. They're logged separately
  (`x0_scoring_calls`, seconds) for the overhead discussion.

## 6. Best-of-N sweeps (no GPU)

```bash
# Seed-count sweep N = 1..32, T = 64
python methods/bon/bon_replay.py --cache cache/sd15_geneval_T64 \
  --prompts-path data/geneval_subset_120.jsonl --out-dir results_bon

# Add held-out GenEval once C has per-image results (prompt_id,seed,correct,tag)
python methods/bon/bon_replay.py --cache cache/sd15_geneval_T64 --geneval-csv geneval_per_image.csv --out-dir results_bon

# Compute-budget sweep, seeds vs steps at equal C (needs extra caches at T=32 / T=16)
python methods/bon/bon_replay.py --cache cache/sd15_geneval_T64 cache/sd15_geneval_T32 cache/sd15_geneval_T16 --out-dir results_bon_budget
```

Results are reported as mean ± std over non-overlapping seed groups: 32 groups at N=1, down to 1 group at N=32.
Use `--bootstrap 200` for confidence intervals at large N, or `--max-groups 3` to match the paper's 3 logical seeds.

The script writes `bon_sweep.csv`, `bon_selections.csv` (chosen seed per prompt, for GenEval and
qualitative figures), `bon_per_tag.csv`, `bon_reward_vs_N.png`, and `bon_reward_vs_C.png`.

## 7. For B: running PSP on this cache

From the PSP repo root, with the cache placed under `REWARDS_ROOT`:

```bash
python results_scripts/baselines/compute_mt_scheduled_from_rewards.py \
  --model-label sdv15 --rewards-root cache --geneval-csv <geneval csv> \
  --k-init 8 --cutoff-times 16,32 --remaining-particles 4,2 --total-steps 64 \
  --logical-seeds 0,1,2 --guidance-metric ir --output-dir results_psp
```

- **Required:** `--guidance-metric ir`. Without it, the script tries to read the empty HPS column and fails.
- **Prune points** must be multiples of 4 (the scoring interval). For finer schedules, rerun the cache with `--score-every 2` or `1`.
- **GenEval CSV** needs columns `prompt_id, seed, correct, tag` (plus `geneval_soft_score` for their BoN script).

## Checks done (CPU, tiny random model; no GPU or real weights available here)
- **Hooks don't change sampling:** final images are pixel-identical to a plain `StableDiffusionPipeline` call with the same seeds.
- **x̂0 is correct:** the captured x̂0 equals the DDIM formula (x_t − √(1−ᾱ)·ε)/√ᾱ exactly, at every step.
- **Compute counter:** with CFG on, it counts seeds × T (e.g. 6 × 8 = 48) while the UNet makes 16 batched calls.
- **Resume:** a rerun skips finished prompts.
- **BoN vs the PSP repo:** on a synthetic cache, `methods/bon/bon_replay.py` at N=4 with 3 groups gives the same
  IR and GenEval as PSP's `compute_regular_vs_bestofn.py`.
- **PSP replay on our cache:** PSP's `compute_mt_scheduled_from_rewards.py` runs on our sparse
  (every-4-steps) cache format.
- **Not yet tested:** real SD v1.5 weights and the real ImageReward model. Your timing test is the first run that loads them.
