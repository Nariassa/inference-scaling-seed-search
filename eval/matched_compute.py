
import os
import json
import logging
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BON_DIR = os.path.join(ROOT, "results", "results_bon")
PSP_DIR = os.path.join(ROOT, "results", "results_psp", "budget")
C_DIR = os.path.join(ROOT, "results", "results_C")
DATA_DIR = os.path.join(ROOT, "data")

OUTPUT_DIR = C_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, "matched_compute.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    logger.info("Loading experiment data...")

    bon = pd.read_csv(
        os.path.join(BON_DIR, "bon_selections.csv")
    )

    psp = pd.read_csv(
        os.path.join(PSP_DIR, "selections.csv")
    )

    geneval = pd.read_csv(
        os.path.join(DATA_DIR, "geneval_per_image.csv")
    )

    clip = pd.read_csv(
        os.path.join(C_DIR, "clip_scores_quickgelu.csv")
    )

    psp_protocol = pd.read_csv(
        os.path.join(PSP_DIR, "summary.csv")
    )

    logger.info("BoN rows: %d", len(bon))
    logger.info("PSP rows: %d", len(psp))
    logger.info("GenEval rows: %d", len(geneval))
    logger.info("CLIP rows: %d", len(clip))

    return bon, psp, geneval, clip, psp_protocol


def attach_metrics(df, geneval, clip, method):
    """
    Match selected images to GenEval and CLIP using prompt_id + seed.
    """

    df = df.copy()

    geneval_cols = [
        "prompt_id",
        "seed",
        "correct",
        "geneval_soft_score",
        "tag"
    ]

    geneval_small = geneval[geneval_cols].copy()

    clip_small = clip[
        ["prompt_id", "seed", "clip_score"]
    ].copy()

    # Verify that each source has unique prompt-seed pairs.
    assert not geneval_small.duplicated(
        ["prompt_id", "seed"]
    ).any(), "Duplicate GenEval prompt-seed pairs."

    assert not clip_small.duplicated(
        ["prompt_id", "seed"]
    ).any(), "Duplicate CLIP prompt-seed pairs."

    # Preserve PSP's own correctness values for validation.
    if method == "PSP" and "correct" in df.columns:
        df = df.rename(columns={"correct": "psp_correct"})

    df = df.merge(
        geneval_small,
        on=["prompt_id", "seed"],
        how="left",
        validate="many_to_one",
        indicator="geneval_match"
    )

    unmatched_geneval = (df["geneval_match"] != "both").sum()

    df = df.drop(columns=["geneval_match"])

    df = df.merge(
        clip_small,
        on=["prompt_id", "seed"],
        how="left",
        validate="many_to_one",
        indicator="clip_match"
    )

    unmatched_clip = (df["clip_match"] != "both").sum()

    df = df.drop(columns=["clip_match"])

    logger.info(
        "%s unmatched GenEval rows: %d",
        method, unmatched_geneval
    )
    logger.info(
        "%s unmatched CLIP rows: %d",
        method, unmatched_clip
    )

    if unmatched_geneval > 0 or unmatched_clip > 0:
        raise ValueError(
            f"{method}: some selected images could not be matched."
        )

    if method == "PSP" and "psp_correct" in df.columns:
        mismatch = (
            pd.to_numeric(df["psp_correct"], errors="coerce")
            != pd.to_numeric(df["correct"], errors="coerce")
        ).sum()

        logger.info(
            "PSP correctness mismatch against GenEval source: %d",
            mismatch
        )

    return df


def calculate_group_metrics(df, method):
    """
    Calculate metrics within each independent seed group.
    """

    rows = []

    if method == "BoN":
        grouped = df.groupby(
            ["T", "N", "group"],
            dropna=False
        )

        for (T, N, group_id), g in grouped:
            rows.append({
                "method": "BoN",
                "configuration": f"BoN_N{int(N)}",
                "compute_C": int(T * N),
                "nbar": float(N),
                "group_id": int(group_id),
                "num_prompts": g["prompt_id"].nunique(),
                "selected_images": len(g),
                "image_reward_mean": g["image_reward"].mean(),
                "geneval_accuracy": g["correct"].mean(),
                "geneval_soft_score": g["geneval_soft_score"].mean(),
                "clip_score_mean": g["clip_score"].mean()
            })

    elif method == "PSP":
        grouped = df.groupby(
            ["configuration", "logical_seed"],
            dropna=False
        )

        for (config, logical_seed), g in grouped:
            rows.append({
                "method": "PSP",
                "configuration": config,
                "compute_C": int(g["compute_C"].iloc[0]),
                "nbar": float(g["nbar"].iloc[0]),
                "group_id": int(logical_seed),
                "num_prompts": g["prompt_id"].nunique(),
                "selected_images": len(g),
                "image_reward_mean": g["image_reward"].mean(),
                "geneval_accuracy": g["correct"].mean(),
                "geneval_soft_score": g["geneval_soft_score"].mean(),
                "clip_score_mean": g["clip_score"].mean()
            })

    return pd.DataFrame(rows)



def summarize_groups(group_df):
    """
    Aggregate independent group-level means.
    Means and sample standard deviations are stored in separate columns.
    """

    metrics = {
        "image_reward_mean": "image_reward_std",
        "geneval_accuracy": "geneval_accuracy_std",
        "geneval_soft_score": "geneval_soft_score_std",
        "clip_score_mean": "clip_score_std"
    }

    rows = []

    grouped = group_df.groupby(
        ["method", "configuration", "compute_C", "nbar"],
        dropna=False
    )

    for keys, g in grouped:
        method, config, compute_C, nbar = keys

        row = {
            "method": method,
            "configuration": config,
            "compute_C": int(compute_C),
            "nbar": float(nbar),
            "num_groups": len(g),
            "num_prompts": int(g["num_prompts"].min()),
            "selected_images": int(g["selected_images"].sum())
        }

        for mean_col, std_col in metrics.items():
            row[mean_col] = g[mean_col].mean()
            row[std_col] = (
                g[mean_col].std(ddof=1)
                if len(g) > 1 else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    bon, psp, geneval, clip, psp_protocol = load_data()

    # Attach common metrics to selected images.
    bon = attach_metrics(bon, geneval, clip, "BoN")
    psp = attach_metrics(psp, geneval, clip, "PSP")

    # Add exact compute budget to PSP selections.
    config_budget = psp_protocol[
        ["configuration", "compute_C", "nbar"]
    ].drop_duplicates()

    psp = psp.merge(
        config_budget,
        on="configuration",
        how="left",
        validate="many_to_one"
    )

    if psp["compute_C"].isna().any():
        raise ValueError("Some PSP configurations have no compute budget.")

    # Calculate independent group-level metrics.
    bon_groups = calculate_group_metrics(bon, "BoN")
    psp_groups = calculate_group_metrics(psp, "PSP")

    all_groups = pd.concat(
        [bon_groups, psp_groups],
        ignore_index=True
    )

    # Aggregate group-level results.
    summary = summarize_groups(all_groups)

    # Attach the pre-declared PSP reference flag.
    reference_flags = psp_protocol[
        ["configuration", "is_reference"]
    ].drop_duplicates()

    summary = summary.merge(
        reference_flags,
        on="configuration",
        how="left"
    )

    summary.loc[
        summary["method"] == "BoN",
        "is_reference"
    ] = True

    # Main comparison: BoN + designated PSP reference configurations.
    reference_comparison = summary[
        (summary["method"] == "BoN")
        | (
            (summary["method"] == "PSP")
            & (summary["is_reference"] == True)
        )
    ].copy()

    # Configuration analysis: every PSP configuration, no cherry-picking.
    all_configurations = summary.copy()

    # Sort for readability.
    sort_cols = ["compute_C", "method", "configuration"]

    all_groups = all_groups.sort_values(sort_cols)
    summary = summary.sort_values(sort_cols)
    reference_comparison = reference_comparison.sort_values(sort_cols)
    all_configurations = all_configurations.sort_values(sort_cols)

    # Save outputs.
    outputs = {
        "matched_compute_per_group.csv": all_groups,
        "matched_compute_summary.csv": summary,
        "matched_compute_reference_comparison.csv": reference_comparison,
        "matched_compute_all_configurations.csv": all_configurations
    }

    for filename, df in outputs.items():
        path = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(path, index=False)
        logger.info("Saved: %s (%d rows)", path, len(df))

    # Save metadata.
    metadata = {
        "experiment": "Matched-compute comparison: BoN vs PSP",
        "matching_keys": ["prompt_id", "seed"],
        "metrics": [
            "ImageReward",
            "GenEval accuracy",
            "GenEval soft score",
            "CLIP score"
        ],
        "aggregation": (
            "Mean and sample standard deviation across independent "
            "seed-group means."
        ),
        "psp_reference_rule": (
            "Use the is_reference flag declared in PSP budget summary; "
            "all configurations are retained in the full table."
        ),
        "num_bon_selected_images": int(len(bon)),
        "num_psp_selected_images": int(len(psp)),
        "num_bon_groups": int(len(bon_groups)),
        "num_psp_groups": int(len(psp_groups))
    }

    with open(
        os.path.join(OUTPUT_DIR, "matched_compute_metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    logger.info("Matched-compute analysis completed.")

    print("\n" + "=" * 75)
    print("MAIN REFERENCE COMPARISON")
    print("=" * 75)

    display_cols = [
        "method",
        "configuration",
        "compute_C",
        "nbar",
        "num_groups",
        "image_reward_mean",
        "geneval_accuracy",
        "geneval_soft_score",
        "clip_score_mean"
    ]

    print(
        reference_comparison[display_cols].to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("ALL PSP CONFIGURATIONS")
    print("=" * 75)

    print(
        all_configurations[
            all_configurations["method"] == "PSP"
        ][display_cols + ["is_reference"]].to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("OUTPUT FILES")
    print("=" * 75)

    for filename in list(outputs.keys()) + [
        "matched_compute_metadata.json",
        "matched_compute.log"
    ]:
        print(os.path.join(OUTPUT_DIR, filename))


if __name__ == "__main__":
    main()
