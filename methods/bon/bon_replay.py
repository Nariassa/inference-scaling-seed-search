#!/usr/bin/env python3
"""
Best-of-N replayed offline from the generation cache (no GPU).

For each N, the cached seeds of every prompt are split into non-overlapping groups
of N (group g = local seeds g*N .. g*N+N-1, same convention as PSP's
compute_regular_vs_bestofn.py). In each group, the seed with the highest final
ImageReward is selected. We report the mean selected reward over prompts, with
mean +/- std across groups. With --bootstrap B, groups are instead B random
N-subsets per prompt (smoother curves, CIs for large N).

Compute: BoN with N seeds at T steps costs C = N*T trajectory-steps per prompt
(effective multiplier N_bar = C/T = N). This is cross-checked against the counts
the harness logged in compute_log.csv.

Several caches with different T (e.g. 64, 32, 16) can be passed at once; the
budget table then compares seeds vs steps at equal C.

Outputs (in --out-dir):
  bon_sweep.csv          one row per (T, N): C, N_bar, IR mean/std, [GenEval], ...
  bon_selections.csv     chosen seed per (T, N, group, prompt) -> for GenEval/qualitative
  bon_per_tag.csv        mean selected IR per GenEval tag
  bon_reward_vs_N.png    selected IR vs N (one line per T)
  bon_reward_vs_C.png    selected IR vs compute C (one line per T)
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def load_cache(root: str):
    """Return (final-reward dataframe, T, compute_log or None) for one cache dir."""
    paths = sorted(glob.glob(os.path.join(root, "metrics.csv")) + glob.glob(os.path.join(root, "*", "metrics.csv")))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv in {root} or its subfolders")
    T = None
    logs = []
    for p in paths:
        a = os.path.join(os.path.dirname(p), "args.json")
        if os.path.exists(a):
            with open(a) as f:
                t = json.load(f).get("time_steps")
            if T is not None and t != T:
                raise ValueError(f"Mixed time_steps inside {root}: {T} vs {t}")
            T = t
        c = os.path.join(os.path.dirname(p), "compute_log.csv")
        if os.path.exists(c):
            logs.append(pd.read_csv(c))
    df = pd.concat([pd.read_csv(p, usecols=["prompt_id", "seed", "step", "image_reward"]) for p in paths])
    df = df.drop_duplicates(["prompt_id", "seed", "step"], keep="last")
    if T is None:
        T = int(df["step"].max())
    final = df[df["step"] == T].copy()
    if final.empty:
        raise ValueError(f"No step-{T} rows in {root}")
    final = final.sort_values(["prompt_id", "seed"]).reset_index(drop=True)
    final["local_seed_idx"] = final.groupby("prompt_id").cumcount()
    log = pd.concat(logs).drop_duplicates("prompt_id", keep="last") if logs else None
    return final, int(T), log


def check_compute(final: pd.DataFrame, T: int, log):
    """Each cached trajectory must have cost exactly T counted steps."""
    if log is None:
        print(f"[warn] T={T}: no compute_log.csv found, skipping compute cross-check")
        return
    seeds_per_prompt = final.groupby("prompt_id").size()
    merged = log.set_index("prompt_id").join(seeds_per_prompt.rename("cached_seeds"), how="inner")
    bad = merged[(merged["traj_steps"] != merged["cached_seeds"] * T)]
    if len(bad):
        raise AssertionError(f"Compute log disagrees with cache for prompts {list(bad.index)[:10]}")
    print(f"[ok] T={T}: counted traj-steps == seeds x T for all {len(merged)} prompts "
          f"(mean {merged['sec_total'].mean():.1f}s/prompt for {int(merged['cached_seeds'].iloc[0])} seeds)")


def select_groups(final: pd.DataFrame, N: int, max_groups, bootstrap: int, rng):
    """Yield (group_id, selected rows) — one selected seed per prompt per group."""
    n_seeds = int(final.groupby("prompt_id").size().min())
    if N > n_seeds:
        return
    if bootstrap:
        r = final["image_reward"].to_numpy()
        pid = final["prompt_id"].to_numpy()
        for g in range(bootstrap):
            picks = []
            for p, idx in final.groupby("prompt_id").indices.items():
                sub = rng.choice(idx, size=N, replace=False)
                picks.append(sub[np.argmax(r[sub])])
            yield g, final.loc[picks]
        return
    n_groups = n_seeds // N
    if max_groups:
        n_groups = min(n_groups, max_groups)
    for g in range(n_groups):
        cur = final[(final["local_seed_idx"] >= g * N) & (final["local_seed_idx"] < (g + 1) * N)]
        yield g, cur.loc[cur.groupby("prompt_id")["image_reward"].idxmax()]


def geneval_overall(sel: pd.DataFrame, gen: pd.DataFrame):
    m = sel.merge(gen, on=["prompt_id", "seed"], how="left")
    if m["correct"].isna().any():
        return np.nan
    return float(m.groupby("tag")["correct"].mean().mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", nargs="+", required=True, help="Cache dir(s) from generate_cache.py (one per T)")
    ap.add_argument("--n-values", default="1,2,4,8,16,32")
    ap.add_argument("--max-groups", type=int, default=None, help="Cap non-overlapping groups per N (PSP used 3)")
    ap.add_argument("--bootstrap", type=int, default=0, help="Use B random N-subsets per prompt instead of fixed groups")
    ap.add_argument("--prompts-path", default=None, help="GenEval jsonl, for per-tag breakdown")
    ap.add_argument("--geneval-csv", default=None,
                    help="Optional per-image GenEval results: prompt_id,seed,correct,tag (held-out metric)")
    ap.add_argument("--out-dir", default="results_bon")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    n_values = [int(x) for x in args.n_values.split(",")]
    gen = pd.read_csv(args.geneval_csv)[["prompt_id", "seed", "correct", "tag"]] if args.geneval_csv else None
    tags = None
    if args.prompts_path:
        with open(args.prompts_path) as f:
            tags = pd.DataFrame([{"prompt_id": i, "tag": json.loads(l).get("tag", "")}
                                 for i, l in enumerate(x for x in f if x.strip())])

    rows, selections, per_tag = [], [], []
    for root in args.cache:
        final, T, log = load_cache(root)
        check_compute(final, T, log)
        n_prompts = final["prompt_id"].nunique()
        for N in n_values:
            g_ir, g_gen, g_mean_all = [], [], []
            for g, sel in select_groups(final, N, args.max_groups, args.bootstrap, rng):
                g_ir.append(sel["image_reward"].mean())
                if gen is not None:
                    g_gen.append(geneval_overall(sel, gen))
                s = sel[["prompt_id", "seed", "image_reward"]].assign(T=T, N=N, group=g)
                selections.append(s)
                if tags is not None:
                    per_tag.append(s.merge(tags, on="prompt_id").groupby("tag")["image_reward"].mean()
                                   .rename("ir").reset_index().assign(T=T, N=N, group=g))
            if not g_ir:
                print(f"[skip] T={T} N={N}: not enough cached seeds")
                continue
            row = {"T": T, "N": N, "N_bar": N, "C_per_prompt": N * T, "n_prompts": n_prompts,
                   "n_groups": len(g_ir), "ir_mean": np.mean(g_ir),
                   "ir_std": np.std(g_ir, ddof=1) if len(g_ir) > 1 else np.nan,
                   "ir_ci95_lo": np.percentile(g_ir, 2.5) if args.bootstrap else np.nan,
                   "ir_ci95_hi": np.percentile(g_ir, 97.5) if args.bootstrap else np.nan}
            if gen is not None:
                row.update({"geneval_mean": np.nanmean(g_gen),
                            "geneval_std": np.nanstd(g_gen, ddof=1) if len(g_gen) > 1 else np.nan})
            rows.append(row)
            extra = f" | GenEval {row['geneval_mean']:.3f}" if gen is not None else ""
            print(f"T={T:3d} N={N:3d} C={N*T:5d}  IR {row['ir_mean']:.3f} ± {row['ir_std']:.3f} "
                  f"({len(g_ir)} groups){extra}")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(args.out_dir, "bon_sweep.csv"), index=False)
    pd.concat(selections).to_csv(os.path.join(args.out_dir, "bon_selections.csv"), index=False)
    if per_tag:
        (pd.concat(per_tag).groupby(["T", "N", "tag"])["ir"].mean().unstack("tag")
         .to_csv(os.path.join(args.out_dir, "bon_per_tag.csv")))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for xcol, fname, xlabel in [("N", "bon_reward_vs_N.png", "Number of seeds N"),
                                ("C_per_prompt", "bon_reward_vs_C.png", "Compute C (trajectory-steps per prompt)")]:
        fig, ax = plt.subplots(figsize=(5, 3.6), dpi=150)
        for T, d in sweep.groupby("T"):
            ax.errorbar(d[xcol], d["ir_mean"], yerr=d["ir_std"].fillna(0), marker="o", capsize=3, label=f"BoN, T={T}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Selected ImageReward")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, fname))
        plt.close(fig)
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
