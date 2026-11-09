"""Summarize two PSP scaling studies without choosing configurations by score."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_differences(selected, baseline, draws=5000):
    """Paired bootstrap over prompts within category; average seed groups first."""
    means = selected.groupby(["configuration", "prompt_id", "tag"]).agg(
        image_reward=("image_reward", "mean"), geneval=("correct", "mean")).reset_index()
    reference = means[means.configuration.eq(baseline)]
    rng = np.random.default_rng(2026)
    records = []
    for name, group in means.groupby("configuration", sort=False):
        paired = group.merge(reference, on=["prompt_id", "tag"], suffixes=("", "_base"),
                             validate="one_to_one")
        if len(paired) != len(reference):
            raise ValueError("Configurations do not share complete prompt coverage.")
        delta = paired[["image_reward", "geneval"]].to_numpy() - paired[
            ["image_reward_base", "geneval_base"]].to_numpy()
        boot = np.zeros((draws, 2))
        # Categories receive equal weight, matching GenEval aggregation.
        for _, positions in paired.groupby("tag").indices.items():
            values = delta[positions]
            sample = rng.integers(len(values), size=(draws, len(values)))
            boot += values[sample].mean(axis=1) / paired["tag"].nunique()
        effect = np.stack([delta[pos].mean(axis=0)
                           for pos in paired.groupby("tag").indices.values()]).mean(axis=0)
        low, high = np.quantile(boot, [0.025, 0.975], axis=0)
        records.append(dict(configuration=name, baseline=baseline, num_prompts=len(paired),
                            image_reward_delta=effect[0], image_reward_ci_low=low[0],
                            image_reward_ci_high=high[0], geneval_delta=effect[1],
                            geneval_ci_low=low[1], geneval_ci_high=high[1]))
    return pd.DataFrame(records)


def table(frame, columns):
    lines = ["| " + " | ".join(title for _, title, _ in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format(row[key], fmt) for key, _, fmt in columns) + " |")
    return "\n".join(lines)


def make_plots(seeds, budget, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    for study, data, x, xlabel in (("seed_count", seeds, "k_init", "Initial seed count K"),
                                   ("compute_budget", budget, "compute_C", "Compute C (trajectory-steps)")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        curve = data if study == "seed_count" else data[data.is_reference]
        curve = curve.sort_values(x)
        for ax, metric, deviation, ylabel in zip(
                axes, ("image_reward_mean", "geneval_mean"),
                ("image_reward_std", "geneval_std"), ("Final ImageReward", "Final GenEval accuracy")):
            if study == "compute_budget":
                ax.scatter(data[x], data[metric], alpha=.55, color="#888888",
                           label="All declared schedules")
            ax.errorbar(curve[x], curve[metric], yerr=curve[deviation].fillna(0),
                        marker="o", capsize=3, label=("Fixed halving rule" if study == "seed_count"
                                                     else "Declared reference schedules"))
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(data[x].unique()))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            if metric == "geneval_mean":
                ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        fig.suptitle("Vary seeds; fixed pruning rule" if study == "seed_count"
                     else "Vary budget; fixed K=16, T=64")
        fig.text(.5, .01, "Error bars: SD across seed-group means; omitted for a single group.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{study}.{suffix}", dpi=180)
        plt.close(fig)


def write_report(root: Path):
    output = root / "findings"
    output.mkdir(parents=True, exist_ok=True)
    summaries, selections, protocols = {}, {}, {}
    for suite in ("seeds", "budget"):
        summaries[suite] = pd.read_csv(root / suite / "summary.csv")
        selections[suite] = pd.read_csv(root / suite / "selections.csv")
        protocols[suite] = json.loads((root / suite / "protocol.json").read_text())
    for key in ("metrics_sha256", "geneval_sha256", "official_script_sha256", "total_steps"):
        if protocols["seeds"][key] != protocols["budget"][key]:
            raise ValueError("Scaling studies use different inputs or evaluator versions.")
    seeds = summaries["seeds"].sort_values("k_init")
    budget = summaries["budget"]
    if not budget.k_init.eq(16).all():
        raise ValueError("Budget experiment must hold K=16 fixed.")
    references = budget[budget.is_reference].sort_values("compute_C")
    if references.compute_C.tolist() != [128, 256, 512, 1024]:
        raise ValueError("Expected one predeclared reference at each budget.")
    differences = []
    for suite, baseline in (("seeds", "k4"), ("budget", "c128_t4_8_r2_1")):
        diff = paired_differences(selections[suite], baseline)
        differences.append(diff.assign(experiment=suite))
    effects = pd.concat(differences, ignore_index=True)
    effects.to_csv(output / "paired_differences.csv", index=False)
    references.to_csv(output / "budget_reference_curve.csv", index=False)
    ranges = budget.groupby(["compute_C", "nbar"]).agg(
        configurations=("configuration", "size"), image_reward_min=("image_reward_mean", "min"),
        image_reward_max=("image_reward_mean", "max"), geneval_min=("geneval_mean", "min"),
        geneval_max=("geneval_mean", "max")).reset_index()
    ranges.to_csv(output / "budget_configuration_ranges.csv", index=False)
    columns = [("k_init", "K", ".0f"), ("compute_C", "C", ".0f"),
               ("nbar", "Nbar", ".2f"), ("image_reward_mean", "ImageReward", ".4f"),
               ("geneval_mean", "GenEval", ".4f")]
    seed_effect = effects[(effects.experiment == "seeds") & (effects.configuration == "k32")].iloc[0]
    budget_effect = effects[(effects.experiment == "budget") &
                            (effects.configuration == "c1024_no_pruning")].iloc[0]

    def finding(effect):
        return (f"ImageReward difference: {effect.image_reward_delta:+.4f} "
                f"(paired 95% CI {effect.image_reward_ci_low:+.4f} to {effect.image_reward_ci_high:+.4f}); "
                f"GenEval difference: {100*effect.geneval_delta:+.2f} percentage points "
                f"(paired 95% CI {100*effect.geneval_ci_low:+.2f} to {100*effect.geneval_ci_high:+.2f}).")

    text = ("# Two PSP scaling experiments\n\n"
            "## 1. Seed-count scaling under a fixed pruning rule\n\n"
            "Vary initial K over 1,2,4,8,16,32 at T=64. Halve survivors at steps 16 and 32 "
            "when possible; K=1 is a standard-sampling reference and K=2 has only the first prune. "
            "K=4..32 share exactly the same fractional pruning rule. Compute increases with K, "
            "so this measures the performance/cost of scaling seeds under that rule.\n\n"
            + table(seeds, columns) + "\n\n"
            "Observed K=4 to K=32 change: " + finding(seed_effect) + "\n\n"
            "## 2. Compute allocation with fixed initial seed count\n\n"
            "Hold K=16 and T=64 fixed. Target C=128,256,512,1024 (Nbar=2,4,8,16). "
            "Enumerate all exact-budget two-stage schedules from cutoffs "
            "{4,8,16,24,32,40,48,56,60} and descending survivor pairs from {8,4,2,1}. "
            "There are 1,5,5 feasible schedules at the first three budgets. C=1024 is the "
            "no-pruning boundary: all 16 trajectories run to completion.\n\n"
            "The following representative schedules are specified analytically in code: "
            "(4,8)/(2,1), (8,32)/(4,1), (16,32)/(8,4), and no pruning. "
            "No schedule is chosen by ImageReward or GenEval. All 12 configurations remain "
            "in budget/summary.csv; their within-budget ranges are in budget_configuration_ranges.csv.\n\n"
            + table(references, columns) + "\n\n"
            "Observed C=128 to C=1024 change with K=16 fixed: " + finding(budget_effect) + "\n\n"
            "## Interpretation and uncertainty\n\n"
            "These are distinct experimental questions on the same cached trajectories, not "
            "independent datasets or replications. This is exploratory analysis of an existing cache, "
            "not a prospective preregistered study. The first scales initial seeds and compute "
            "together under a fixed rule; the second fixes initial seeds and changes the amount "
            "and allocation of denoising work. Schedule details still matter within a budget. "
            "The grid is limited; no global optimum is claimed. GenEval is used only for final evaluation.\n\n"
            "Summary CSV standard deviations describe variation across complete, non-overlapping seed "
            "groups. Paired difference CIs use 5,000 bootstrap resamples of prompts within category "
            "(seed 2026), averaging seed groups per prompt first. They describe prompt-sampling "
            "uncertainty conditional on this seed cache; they do not include new-seed or model uncertainty. "
            "Compute counts trajectory-denoising steps and excludes reward scoring and VAE decoding.\n")
    (output / "findings.md").write_text(text)
    make_plots(seeds, budget, output)
    print(f"Saved experimental findings and plots: {output}", flush=True)
