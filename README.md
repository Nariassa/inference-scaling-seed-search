# Inference-Time Scaling for Diffusion: Best-of-N vs Progressive Seed Pruning

**AI6132 Generative AI for Visual Synthesis (NTU), Technical Review Project**

Instead of spending more compute on training or on more denoising steps, we spend it at sampling time. We draw many noise seeds and keep the best one under a reward model.
This repo compares two ways of doing that at **matched compute**:

- **Best-of-N (BoN):** run N seeds to completion and keep the highest-reward image.
- **Progressive Seed Pruning (PSP)** ([Guimarães & Perona, arXiv:2607.21591](https://arxiv.org/abs/2607.21591)): score the intermediate denoised estimate x̂0 during sampling, and drop weak seeds early so their compute goes to the strong ones.

The repo also covers sweeps over seed count, compute budget, pruning step and verifier, a discussion of reward hacking and diversity loss, and an adaptive pruning schedule as our improvement idea.

## Key idea: generate once, replay offline

Every method selects from the **same** cached trajectories:

```
                     GPU, once                                CPU, seconds
prompts ──► harness/generate_cache.py ──► cache/ ──┬──► methods/bon/       Best-of-N sweeps
            SD v1.5 · DDIM η=0 · T=64              ├──► methods/psp/       PSP (official replay)
            32 seeds/prompt                        ├──► methods/adaptive/  schedule search
            x̂0 ImageReward every 4 steps           └──► eval/              GenEval (held-out)
```

This makes comparisons exact (same seeds, same images) and turns every ablation into a cheap replay, the same protocol the PSP paper uses.
The cache format matches the PSP repo, so its scripts run on our cache directly.

## Setup

| Setting | Value |
|---|---|
| Model | Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), 512×512, fp16 |
| Sampler | DDIM, η = 0 (deterministic given the seed), T = 64, CFG 7.5 |
| Prompts | 120 GenEval prompts, 20 per category ([`data/geneval_subset_120.jsonl`](data/geneval_subset_120.jsonl)) |
| Seeds | 32 per prompt; seed = prompt_id × 32 + i |
| Selection reward | ImageReward |
| Held-out metric | GenEval object-detector accuracy (never used for selection) |
| Compute unit | One denoising step of one trajectory, counted inside the UNet. C = Σ_t k_t, N̄ = C / T |

Full rationale: [`docs/decisions.md`](docs/decisions.md). Harness internals and all command-line options: [`docs/harness.md`](docs/harness.md).

## Repository layout

```
harness/            generate_cache.py (generation + compute counter), run_timing_test.sh
methods/bon/        bon_replay.py: Best-of-N seed-count and compute-budget sweeps
methods/psp/        PSP runs on the shared cache
methods/adaptive/   adaptive pruning schedule (improvement idea)
eval/               geneval_bridge.py: cache → GenEval layout → per-image CSV
analysis/           qualitative grids, diversity, reward-hacking examples
notebooks/          colab_pipeline.ipynb: end-to-end Colab run with Drive-backed resume
data/               prompt subset
results/            small CSVs and figures (no images)
docs/               decisions.md, harness.md
report/             technical review draft
```

The cache (about 1.5 GB of images plus `metrics.csv`) lives on the shared Google Drive folder, not in git.

## Quick start

**Colab:** open [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb), set `REPO_URL` in cell 4, and run the cells in order.

**Local / server:**

```bash
pip install -r requirements.txt && pip uninstall -y peft

bash harness/run_timing_test.sh                       # seconds per prompt on your GPU

python harness/generate_cache.py \
  --prompts-path data/geneval_subset_120.jsonl \
  --output-dir cache/sd15_geneval_T64 \
  --num-seeds 32 --time-steps 64 --batch-size 16 --decode-batch 4

python methods/bon/bon_replay.py --cache cache/sd15_geneval_T64 \
  --prompts-path data/geneval_subset_120.jsonl --out-dir results/bon
```

On an L4 (24 GB), one prompt × 32 seeds takes about 203 s, so all 120 prompts take about 6.8 GPU-hours. The run is resumable.

## Experiments

| Experiment | Variable | Owner | Status |
|---|---|---|---|
| BoN seed count | N ∈ {1, 2, 4, 8, 16, 32} at T = 64 | A | cache running |
| BoN compute budget | seeds vs steps at equal C (T = 64 / 32 / 16) | A | planned |
| PSP vs BoN at matched compute | N̄ ∈ {2, 4, 8, 16} | B | waiting on cache |
| Pruning step | when to prune (step 8 … 48) | B | waiting on cache |
| Verifier | ImageReward vs CLIP score for selection | B / C | planned |
| Held-out evaluation | GenEval on selected images | C | waiting on cache |
| Reward hacking and diversity | IR gain vs GenEval gain, seed diversity | C / D | planned |
| Adaptive pruning schedule | data-driven prune points | B | planned |

## Results

*To be filled in from `results/`.*

| Method | N̄ | ImageReward ↑ | GenEval ↑ |
|---|---|---|---|
| Standard sampling | 1 | | |
| Best-of-N | 4 | | |
| PSP | 4 | | |

## Team

| Member | Role |
|---|---|
| (A) Ka Lam LI | Coordinator; generation harness, compute counter, Best-of-N sweeps |
| (B) Rui YANG | PSP replay, schedule sweeps, adaptive schedule |
| (C)Canying CHEN | Prompt subset, GenEval and held-out evaluation |
| (D)Kaixin ZHU | Qualitative figures, diversity, introduction and conclusion |

Timeline: Nov 9 setup → Nov 10 cache handed over → Nov 11–13 sweeps and analysis → Nov 14 writing → **Nov 16 deadline**.

## References

- Guimarães & Perona. *Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning.* arXiv:2607.21591, 2026. Code: [rogerioagjr/PSP](https://github.com/rogerioagjr/PSP)
- Ghosh et al. *GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment.* NeurIPS 2023. Code: [djghosh13/geneval](https://github.com/djghosh13/geneval)
- Xu et al. *ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation.* NeurIPS 2023.
- Song et al. *Denoising Diffusion Implicit Models.* ICLR 2021.
