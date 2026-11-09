#!/usr/bin/env python3
"""
Shared generation harness for the BoN vs PSP project.

Runs SD v1.5 + DDIM (eta=0) ONCE for every (prompt, seed) and caches:
  * per-step ImageReward of the denoised estimate x0_hat (every --score-every steps)
  * the final image and its ImageReward (always recorded as step = T)
  * a compute log (trajectory-steps actually executed, counted inside the UNet)

Best-of-N and PSP are then replayed offline from this cache (see bon_replay.py,
and PSP's results_scripts/baselines/compute_mt_scheduled_from_rewards.py).

Output layout and metrics.csv columns match PSP's collect_image_reward_signal.py,
so the PSP repo's replay scripts can read our cache directly:

  <output-dir>/
      args.json
      metrics.csv        prompt_id,prompt,seed,step,timestep,image_reward,human_preference
      compute_log.csv    per-prompt counted trajectory-steps, UNet calls, timings
      samples/<prompt_id:05d>/<seed:05d>.png
      x0/<prompt_id:05d>/<seed:05d>_s<step:02d>.jpg   (only with --save-x0)

Differences from PSP's collector (same numbers, less GPU time):
  * x0_hat is taken from the scheduler's own step (pred_original_sample) instead of
    re-running the UNet, so collection costs 1 UNet pass per step, not 2.
  * Intermediate rewards are scored every --score-every steps instead of every step.
    PSP prune points must be multiples of --score-every (16 and 32 are, for 4).
  * The step-T reward is computed on the final output image.
  * Resumable: prompts whose samples are complete are skipped.

Seed convention (same as PSP): seed = --seed + prompt_id * --num-seeds + i.
"""
import argparse
import csv
import functools
import json
import os
import time
from contextlib import contextmanager
from typing import Dict, List

import torch


# --------------------------------------------------------------------------- #
# Compute counter
# --------------------------------------------------------------------------- #
class ComputeCounter:
    """Counts denoiser work by wrapping the UNet forward.

    Compute unit = one denoising step of one trajectory. With classifier-free
    guidance the cond and uncond halves are batched together, so a UNet call on
    2B rows advances B trajectories by one step (counted as B, per our convention).
    """

    def __init__(self, unet, cfg: bool):
        self.cfg = cfg
        self.unet_calls = 0
        self.unet_rows = 0
        self.traj_steps = 0
        self._orig_forward = unet.forward

        @functools.wraps(self._orig_forward)
        def counted_forward(sample, *args, **kwargs):
            rows = int(sample.shape[0])
            self.unet_calls += 1
            self.unet_rows += rows
            self.traj_steps += rows // 2 if self.cfg else rows
            return self._orig_forward(sample, *args, **kwargs)

        unet.forward = counted_forward

    def snapshot(self) -> Dict[str, int]:
        return {
            "unet_calls": self.unet_calls,
            "unet_rows": self.unet_rows,
            "traj_steps": self.traj_steps,
        }


@contextmanager
def capture_x0(scheduler, store: Dict[str, torch.Tensor]):
    """Temporarily wrap scheduler.step so each call stashes pred_original_sample."""
    orig_step = scheduler.step

    # functools.wraps keeps the signature visible to inspect, which diffusers uses
    # to decide whether to pass eta/generator to scheduler.step.
    @functools.wraps(orig_step)
    def step(*args, **kwargs):
        return_dict = kwargs.pop("return_dict", True)
        out = orig_step(*args, return_dict=True, **kwargs)
        store["x0"] = out.pred_original_sample
        return out if return_dict else (out.prev_sample, out.pred_original_sample)

    scheduler.step = step
    try:
        yield
    finally:
        scheduler.step = orig_step


# --------------------------------------------------------------------------- #
# Model / reward loading
# --------------------------------------------------------------------------- #
def build_pipeline(model_id: str, device: str, dtype: torch.dtype):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class ImageRewardScorer:
    def __init__(self, device: str):
        import ImageReward as RM  # pip install image-reward

        self.model = RM.load("ImageReward-v1.0", device=device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, prompt: str, images) -> List[float]:
        out = self.model.score(prompt, list(images))
        return [float(out)] if not isinstance(out, (list, tuple)) else [float(v) for v in out]


class MockScorer:
    """Test-only: deterministic fake reward (mean brightness). No downloads."""

    def __call__(self, prompt: str, images) -> List[float]:
        import numpy as np

        return [float(np.asarray(im, dtype=np.float32).mean() / 255.0) for im in images]


@torch.no_grad()
def decode_to_pil(pipe, latents: torch.Tensor, chunk: int):
    images = []
    vae = pipe.vae
    for i in range(0, latents.shape[0], chunk):
        z = latents[i : i + chunk].to(vae.dtype) / vae.config.scaling_factor
        x = vae.decode(z, return_dict=False)[0]
        images += pipe.image_processor.postprocess(x, output_type="pil")
    return images


def read_prompts(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts-path", required=True, help="GenEval-style .jsonl (one {'prompt': ...} per line)")
    p.add_argument("--output-dir", required=True,
                   help="Keep 'geneval' in the folder name so PSP's replay scripts find it.")
    p.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp32", action="store_true", help="Run in float32 (default fp16 on cuda)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt-start-id", type=int, default=0)
    p.add_argument("--num-prompts", type=int, default=None)
    p.add_argument("--num-seeds", type=int, default=32)
    p.add_argument("--time-steps", type=int, default=64)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16, help="Seeds denoised in parallel")
    p.add_argument("--decode-batch", type=int, default=8, help="VAE decode chunk for x0_hat")
    p.add_argument("--score-every", type=int, default=4,
                   help="Score x0_hat every k steps (k=1 reproduces PSP's collector exactly)")
    p.add_argument("--save-x0", action="store_true", help="Also save scored x0_hat previews as JPEGs")
    p.add_argument("--reward", choices=["imagereward", "mock"], default="imagereward")
    p.add_argument("--_test_pipe", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    T = args.time_steps
    assert 1 <= args.score_every <= T
    score_steps = sorted(set(range(args.score_every, T, args.score_every)))  # step T scored on final image
    dtype = torch.float16 if (args.device.startswith("cuda") and not args.fp32) else torch.float32

    prompts = read_prompts(args.prompts_path)
    end = len(prompts) if args.num_prompts is None else min(len(prompts), args.prompt_start_id + args.num_prompts)
    ids = list(range(args.prompt_start_id, end))

    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump({**vars(args), "score_steps": score_steps + [T], "resolved_prompt_ids": [ids[0], ids[-1]]}, f, indent=2)

    if args._test_pipe:  # unit tests inject a tiny local pipeline
        import pickle
        with open(args._test_pipe, "rb") as f:
            pipe = pickle.load(f).to(args.device)
        pipe.set_progress_bar_config(disable=True)
    else:
        pipe = build_pipeline(args.model_id, args.device, dtype)
    scorer = ImageRewardScorer(args.device) if args.reward == "imagereward" else MockScorer()
    counter = ComputeCounter(pipe.unet, cfg=args.guidance_scale > 1.0)

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    compute_path = os.path.join(args.output_dir, "compute_log.csv")
    new_metrics = not os.path.exists(metrics_path)
    new_compute = not os.path.exists(compute_path)
    mf = open(metrics_path, "a", newline="", encoding="utf-8")
    cf = open(compute_path, "a", newline="", encoding="utf-8")
    mw = csv.DictWriter(mf, fieldnames=["prompt_id", "prompt", "seed", "step", "timestep", "image_reward", "human_preference"])
    cw = csv.DictWriter(cf, fieldnames=["prompt_id", "num_seeds", "time_steps", "traj_steps", "expected_traj_steps",
                                        "unet_calls", "x0_scoring_calls", "sec_total"])
    if new_metrics:
        mw.writeheader()
    if new_compute:
        cw.writeheader()

    for n_done, pid in enumerate(ids):
        prompt = prompts[pid]["prompt"]
        sample_dir = os.path.join(args.output_dir, "samples", f"{pid:05d}")
        seeds = [args.seed + pid * args.num_seeds + i for i in range(args.num_seeds)]
        if all(os.path.exists(os.path.join(sample_dir, f"{s:05d}.png")) for s in seeds):
            print(f"[skip] prompt {pid} already complete")
            continue
        os.makedirs(sample_dir, exist_ok=True)
        if args.save_x0:
            os.makedirs(os.path.join(args.output_dir, "x0", f"{pid:05d}"), exist_ok=True)

        before = counter.snapshot()
        n_score_calls = 0
        sync(args.device)
        t0 = time.time()
        rows = []
        for b in range(0, len(seeds), args.batch_size):
            batch = seeds[b : b + args.batch_size]
            gens = [torch.Generator(device="cpu").manual_seed(s) for s in batch]
            store: Dict[str, torch.Tensor] = {}
            step_scores: Dict[int, List[float]] = {}
            step_t: Dict[int, int] = {}

            def on_step_end(_pipe, i, t, kw):
                nonlocal n_score_calls
                step = i + 1  # 1-indexed, like PSP
                if step in score_steps:
                    imgs = decode_to_pil(pipe, store["x0"], args.decode_batch)
                    step_scores[step] = scorer(prompt, imgs)
                    step_t[step] = int(t)
                    n_score_calls += 1
                    if args.save_x0:
                        for s, im in zip(batch, imgs):
                            im.resize((256, 256)).save(
                                os.path.join(args.output_dir, "x0", f"{pid:05d}", f"{s:05d}_s{step:02d}.jpg"), quality=85)
                return {}

            with capture_x0(pipe.scheduler, store), torch.no_grad():
                out = pipe(
                    [prompt] * len(batch),
                    height=args.height,
                    width=args.width,
                    num_inference_steps=T,
                    guidance_scale=args.guidance_scale,
                    eta=0.0,
                    generator=gens,
                    callback_on_step_end=on_step_end,
                )
            finals = out.images
            step_scores[T] = scorer(prompt, finals)
            step_t[T] = int(pipe.scheduler.timesteps[-1])
            n_score_calls += 1
            for s, im in zip(batch, finals):
                im.save(os.path.join(sample_dir, f"{s:05d}.png"))
            for step in sorted(step_scores):
                for s, r in zip(batch, step_scores[step]):
                    rows.append({"prompt_id": pid, "prompt": prompt, "seed": s, "step": step,
                                 "timestep": step_t[step], "image_reward": r, "human_preference": ""})
        sync(args.device)
        dt = time.time() - t0

        after = counter.snapshot()
        used = after["traj_steps"] - before["traj_steps"]
        expected = len(seeds) * T
        if used != expected:
            raise RuntimeError(f"Compute counter mismatch on prompt {pid}: counted {used}, expected {expected}")
        mw.writerows(sorted(rows, key=lambda r: (r["seed"], r["step"])))
        cw.writerow({"prompt_id": pid, "num_seeds": len(seeds), "time_steps": T, "traj_steps": used,
                     "expected_traj_steps": expected, "unet_calls": after["unet_calls"] - before["unet_calls"],
                     "x0_scoring_calls": n_score_calls, "sec_total": round(dt, 2)})
        mf.flush(); cf.flush()
        finals = [r["image_reward"] for r in rows if r["step"] == T]
        print(f"[prompt {pid}] {len(seeds)} seeds in {dt:.1f}s | traj-steps {used} (= {len(seeds)}x{T}) | "
              f"final IR mean {sum(finals)/len(finals):.3f} max {max(finals):.3f} | "
              f"{n_done + 1}/{len(ids)} | ETA {dt * (len(ids) - n_done - 1) / 60:.0f} min")

    mf.close(); cf.close()
    print(f"Done. Cache in {args.output_dir}")


if __name__ == "__main__":
    main()
