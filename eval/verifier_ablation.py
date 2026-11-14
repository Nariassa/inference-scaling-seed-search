
import os
import json
import logging
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
BON_DIR = os.path.join(RESULTS_DIR, "results_bon")
DATA_DIR = os.path.join(ROOT, "data")
C_DIR = os.path.join(RESULTS_DIR, "results_C")

os.makedirs(C_DIR, exist_ok=True)

LOG_PATH = os.path.join(C_DIR, "verifier_ablation.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

N_VALUES = [1, 2, 4, 8, 16, 32]
TOTAL_SEEDS = 32


def load_data():
    logger.info("Loading source data...")

    metrics = pd.read_csv(
        os.path.join(CACHE_DIR, "T64_metrics.csv")
    )

    clip = pd.read_csv(
        os.path.join(C_DIR, "clip_scores_quickgelu.csv")
    )

    geneval = pd.read_csv(
        os.path.join(DATA_DIR, "geneval_per_image.csv")
    )

    bon = pd.read_csv(
        os.path.join(BON_DIR, "bon_selections.csv")
    )

    # Only final ImageReward scores.
    metrics = metrics[metrics["step"] == 64].copy()

    metrics = metrics[
        ["prompt_id", "prompt", "seed", "image_reward"]
    ]

    clip = clip[
        ["prompt_id", "seed", "clip_score"]
    ]

    geneval = geneval[
        [
            "prompt_id",
            "seed",
            "correct",
            "geneval_soft_score",
            "tag"
        ]
    ]

    for name, df in [
        ("ImageReward", metrics),
        ("CLIP", clip),
        ("GenEval", geneval)
    ]:
        if df.duplicated(["prompt_id", "seed"]).any():
            raise ValueError(
                f"Duplicate prompt-seed pairs in {name}"
            )

    candidates = metrics.merge(
        clip,
        on=["prompt_id", "seed"],
        how="inner",
        validate="one_to_one"
    )

    candidates = candidates.merge(
        geneval,
        on=["prompt_id", "seed"],
        how="inner",
        validate="one_to_one"
    )

    expected = 120 * TOTAL_SEEDS

    if len(candidates) != expected:
        raise ValueError(
            f"Expected {expected} candidates, found {len(candidates)}"
        )

    candidates["local_seed_idx"] = (
        candidates["seed"]
        - candidates["prompt_id"] * TOTAL_SEEDS
    )

    if not candidates["local_seed_idx"].between(
        0, TOTAL_SEEDS - 1
    ).all():
        raise ValueError("Unexpected seed indexing.")

    logger.info("Candidate images: %d", len(candidates))

    return candidates, bon


def select_winners(candidates, N):
    """
    Partition each prompt's 32 candidates into consecutive groups of N.
    Select the highest-scoring candidate using each verifier.
    """

    if TOTAL_SEEDS % N != 0:
        raise ValueError(f"N={N} does not divide 32.")

    df = candidates.copy()

    df["N"] = N
    df["group"] = df["local_seed_idx"] // N

    grouping = ["prompt_id", "N", "group"]

    ir_idx = (
        df.groupby(grouping)["image_reward"]
        .idxmax()
    )

    clip_idx = (
        df.groupby(grouping)["clip_score"]
        .idxmax()
    )

    ir_selected = df.loc[ir_idx].copy()
    ir_selected["verifier"] = "ImageReward"

    clip_selected = df.loc[clip_idx].copy()
    clip_selected["verifier"] = "CLIP"

    return pd.concat(
        [ir_selected, clip_selected],
        ignore_index=True
    )


def validate_ir_against_bon(selected, bon):
    """
    Confirm that reconstructed ImageReward winners match
    the original BoN selection records exactly.
    """

    ir = selected[
        selected["verifier"] == "ImageReward"
    ][
        ["prompt_id", "N", "group", "seed"]
    ].copy()

    original = bon[
        ["prompt_id", "N", "group", "seed"]
    ].copy()

    comparison = ir.merge(
        original,
        on=["prompt_id", "N", "group"],
        how="outer",
        suffixes=("_reconstructed", "_original"),
        indicator=True,
        validate="one_to_one"
    )

    unmatched = (
        comparison["_merge"] != "both"
    ).sum()

    matched = comparison[
        comparison["_merge"] == "both"
    ]

    different = (
        matched["seed_reconstructed"]
        != matched["seed_original"]
    ).sum()

    logger.info(
        "BoN reconstruction: matched=%d, unmatched=%d, "
        "different selected seeds=%d",
        len(matched), unmatched, different
    )

    if unmatched > 0 or different > 0:
        raise ValueError(
            "Reconstructed ImageReward selections do not match BoN."
        )

    return comparison


def build_outputs(candidates, bon):
    all_selections = []

    for N in N_VALUES:
        logger.info("Selecting winners for N=%d", N)

        selected = select_winners(candidates, N)
        all_selections.append(selected)

    selections = pd.concat(
        all_selections,
        ignore_index=True
    )

    # Validate the reconstructed IR selections.
    validate_ir_against_bon(selections, bon)

    # Save individual selected images.
    selection_columns = [
        "prompt_id",
        "N",
        "group",
        "verifier",
        "seed",
        "image_reward",
        "clip_score",
        "correct",
        "geneval_soft_score",
        "tag"
    ]

    selections[selection_columns].to_csv(
        os.path.join(C_DIR, "verifier_ablation_selections.csv"),
        index=False
    )

    # Per-group metrics: average over 120 prompts.
    group_metrics = (
        selections.groupby(
            ["N", "group", "verifier"],
            as_index=False
        )
        .agg(
            num_prompts=("prompt_id", "nunique"),
            image_reward_mean=("image_reward", "mean"),
            clip_score_mean=("clip_score", "mean"),
            geneval_accuracy=("correct", "mean"),
            geneval_soft_score=("geneval_soft_score", "mean")
        )
    )

    group_metrics.to_csv(
        os.path.join(C_DIR, "verifier_ablation_per_group.csv"),
        index=False
    )

    # Summary: average and standard deviation across groups.
    metrics = [
        "image_reward_mean",
        "clip_score_mean",
        "geneval_accuracy",
        "geneval_soft_score"
    ]

    summary_rows = []

    for (N, verifier), g in group_metrics.groupby(
        ["N", "verifier"]
    ):
        row = {
            "N": int(N),
            "compute_C": int(N * 64),
            "verifier": verifier,
            "num_groups": len(g)
        }

        for metric in metrics:
            row[metric] = g[metric].mean()
            row[metric + "_std"] = (
                g[metric].std(ddof=1)
                if len(g) > 1 else np.nan
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Selection overlap between verifiers.
    overlap_rows = []

    for N in N_VALUES:
        subset = selections[selections["N"] == N]

        ir = subset[
            subset["verifier"] == "ImageReward"
        ][["prompt_id", "group", "seed"]]

        clip = subset[
            subset["verifier"] == "CLIP"
        ][["prompt_id", "group", "seed"]]

        paired = ir.merge(
            clip,
            on=["prompt_id", "group"],
            suffixes=("_ir", "_clip"),
            validate="one_to_one"
        )

        same = (
            paired["seed_ir"] == paired["seed_clip"]
        )

        overlap_rows.append({
            "N": N,
            "compute_C": N * 64,
            "num_prompt_groups": len(paired),
            "same_selected_image_count": int(same.sum()),
            "selection_overlap": same.mean()
        })

    overlap = pd.DataFrame(overlap_rows)

    summary = summary.merge(
        overlap[["N", "selection_overlap"]],
        on="N",
        how="left",
        validate="many_to_one"
    )

    summary = summary.sort_values(
        ["N", "verifier"]
    )

    summary.to_csv(
        os.path.join(C_DIR, "verifier_ablation_summary.csv"),
        index=False
    )

    overlap.to_csv(
        os.path.join(C_DIR, "verifier_ablation_overlap.csv"),
        index=False
    )

    # Paired CLIP-minus-ImageReward differences at group level.
    wide = group_metrics.pivot(
        index=["N", "group"],
        columns="verifier",
        values=[
            "image_reward_mean",
            "clip_score_mean",
            "geneval_accuracy",
            "geneval_soft_score"
        ]
    )

    difference_rows = []

    for N in N_VALUES:
        try:
            current = wide.loc[N]
        except KeyError:
            continue

        row = {
            "N": N,
            "compute_C": N * 64,
            "num_groups": len(current)
        }

        for metric in metrics:
            clip_values = current[(metric, "CLIP")]
            ir_values = current[(metric, "ImageReward")]

            diff = clip_values - ir_values

            row[metric + "_clip_minus_ir_mean"] = diff.mean()
            row[metric + "_clip_minus_ir_std"] = (
                diff.std(ddof=1)
                if len(diff) > 1 else np.nan
            )

        difference_rows.append(row)

    differences = pd.DataFrame(difference_rows)

    differences.to_csv(
        os.path.join(C_DIR, "verifier_ablation_paired_differences.csv"),
        index=False
    )

    # Metadata.
    metadata = {
        "experiment": "Final-candidate verifier ablation",
        "candidate_pool": "32 images per prompt",
        "image_reward_step": 64,
        "N_values": N_VALUES,
        "selection_rule": (
            "Partition 32 seeds into consecutive groups of N; "
            "select the maximum score independently for each verifier."
        ),
        "metrics": [
            "ImageReward",
            "CLIP cosine similarity",
            "GenEval accuracy",
            "GenEval soft score"
        ],
        "overlap_definition": (
            "Fraction of prompt-group pairs where both verifiers "
            "select the same seed."
        ),
        "note": (
            "This is a final-candidate selection ablation. "
            "It does not test CLIP as an intermediate PSP pruning verifier."
        )
    }

    with open(
        os.path.join(C_DIR, "verifier_ablation_metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    return summary, overlap, differences


def main():
    candidates, bon = load_data()

    selections = []
    for N in N_VALUES:
        selections.append(select_winners(candidates, N))

    all_selections = pd.concat(
        selections,
        ignore_index=True
    )

    validate_ir_against_bon(all_selections, bon)

    # Reuse the validated selections for all outputs.
    selection_columns = [
        "prompt_id", "N", "group", "verifier", "seed",
        "image_reward", "clip_score", "correct",
        "geneval_soft_score", "tag"
    ]

    all_selections[selection_columns].to_csv(
        os.path.join(C_DIR, "verifier_ablation_selections.csv"),
        index=False
    )

    group_metrics = (
        all_selections.groupby(
            ["N", "group", "verifier"],
            as_index=False
        )
        .agg(
            num_prompts=("prompt_id", "nunique"),
            image_reward_mean=("image_reward", "mean"),
            clip_score_mean=("clip_score", "mean"),
            geneval_accuracy=("correct", "mean"),
            geneval_soft_score=("geneval_soft_score", "mean")
        )
    )

    group_metrics.to_csv(
        os.path.join(C_DIR, "verifier_ablation_per_group.csv"),
        index=False
    )

    metric_cols = [
        "image_reward_mean",
        "clip_score_mean",
        "geneval_accuracy",
        "geneval_soft_score"
    ]

    summary_rows = []

    for (N, verifier), g in group_metrics.groupby(
        ["N", "verifier"]
    ):
        row = {
            "N": int(N),
            "compute_C": int(N * 64),
            "verifier": verifier,
            "num_groups": len(g)
        }

        for metric in metric_cols:
            row[metric] = g[metric].mean()
            row[metric + "_std"] = (
                g[metric].std(ddof=1)
                if len(g) > 1 else np.nan
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    overlap_rows = []

    for N in N_VALUES:
        subset = all_selections[all_selections["N"] == N]

        ir = subset[
            subset["verifier"] == "ImageReward"
        ][["prompt_id", "group", "seed"]]

        clip = subset[
            subset["verifier"] == "CLIP"
        ][["prompt_id", "group", "seed"]]

        paired = ir.merge(
            clip,
            on=["prompt_id", "group"],
            suffixes=("_ir", "_clip"),
            validate="one_to_one"
        )

        same = paired["seed_ir"] == paired["seed_clip"]

        overlap_rows.append({
            "N": N,
            "compute_C": N * 64,
            "num_prompt_groups": len(paired),
            "same_selected_image_count": int(same.sum()),
            "selection_overlap": same.mean()
        })

    overlap = pd.DataFrame(overlap_rows)

    summary = summary.merge(
        overlap[["N", "selection_overlap"]],
        on="N",
        how="left"
    )

    summary.to_csv(
        os.path.join(C_DIR, "verifier_ablation_summary.csv"),
        index=False
    )

    overlap.to_csv(
        os.path.join(C_DIR, "verifier_ablation_overlap.csv"),
        index=False
    )

    wide = group_metrics.pivot(
        index=["N", "group"],
        columns="verifier",
        values=metric_cols
    )

    diff_rows = []

    for N in N_VALUES:
        current = wide.loc[N]
        row = {
            "N": N,
            "compute_C": N * 64,
            "num_groups": len(current)
        }

        for metric in metric_cols:
            diff = (
                current[(metric, "CLIP")]
                - current[(metric, "ImageReward")]
            )

            row[metric + "_clip_minus_ir_mean"] = diff.mean()
            row[metric + "_clip_minus_ir_std"] = (
                diff.std(ddof=1)
                if len(diff) > 1 else np.nan
            )

        diff_rows.append(row)

    differences = pd.DataFrame(diff_rows)

    differences.to_csv(
        os.path.join(C_DIR, "verifier_ablation_paired_differences.csv"),
        index=False
    )

    metadata = {
        "experiment": "Final-candidate verifier ablation",
        "candidate_pool": "32 images per prompt",
        "image_reward_step": 64,
        "N_values": N_VALUES,
        "selection_rule": (
            "Partition 32 seeds into consecutive groups of N; "
            "select maximum verifier score independently."
        ),
        "note": (
            "Final-candidate selection only; not an intermediate "
            "PSP pruning verifier experiment."
        )
    }

    with open(
        os.path.join(C_DIR, "verifier_ablation_metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    logger.info("Verifier ablation completed.")

    print("\n" + "=" * 75)
    print("VERIFIER ABLATION SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("SELECTION OVERLAP")
    print("=" * 75)
    print(overlap.to_string(index=False))

    print("\n" + "=" * 75)
    print("PAIRED CLIP - IMAGEREWARD DIFFERENCES")
    print("=" * 75)
    print(differences.to_string(index=False))

    print("\n" + "=" * 75)
    print("OUTPUT FILES")
    print("=" * 75)

    for filename in [
        "verifier_ablation_selections.csv",
        "verifier_ablation_per_group.csv",
        "verifier_ablation_summary.csv",
        "verifier_ablation_overlap.csv",
        "verifier_ablation_paired_differences.csv",
        "verifier_ablation_metadata.json",
        "verifier_ablation.log"
    ]:
        print(os.path.join(C_DIR, filename))


if __name__ == "__main__":
    main()
