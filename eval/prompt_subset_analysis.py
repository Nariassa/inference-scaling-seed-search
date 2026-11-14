
import os
import json
import logging
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
C_DIR = os.path.join(ROOT, "results", "results_C")

os.makedirs(C_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(C_DIR, "prompt_subset_analysis.log"),
            mode="w"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    path = os.path.join(DATA_DIR, "geneval_per_image.csv")
    df = pd.read_csv(path)

    required = [
        "prompt_id", "seed", "tag",
        "correct", "geneval_soft_score"
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[["prompt_id", "seed"]].duplicated().any():
        raise ValueError("Duplicate prompt-seed pairs found.")

    # Verify one tag per prompt.
    tag_counts = df.groupby("prompt_id")["tag"].nunique()
    if not (tag_counts == 1).all():
        raise ValueError("Some prompts have inconsistent tags.")

    # Prompt-level summary.
    prompt_summary = (
        df.groupby("prompt_id")
        .agg(
            num_images=("seed", "count"),
            num_unique_seeds=("seed", "nunique"),
            tag=("tag", "first"),
            accuracy=("correct", "mean"),
            soft_score=("geneval_soft_score", "mean")
        )
        .reset_index()
    )

    if len(prompt_summary) != 120:
        raise ValueError(
            f"Expected 120 prompts, found {len(prompt_summary)}"
        )

    if not (prompt_summary["num_images"] == 32).all():
        raise ValueError("Each prompt must have exactly 32 images.")

    if not (prompt_summary["num_unique_seeds"] == 32).all():
        raise ValueError("Each prompt must have 32 unique seeds.")

    # Tag-level summary.
    tag_summary = (
        prompt_summary.groupby("tag")
        .agg(
            num_prompts=("prompt_id", "nunique"),
            num_images=("num_images", "sum"),
            mean_prompt_accuracy=("accuracy", "mean"),
            std_prompt_accuracy=("accuracy", "std"),
            mean_prompt_soft_score=("soft_score", "mean"),
            std_prompt_soft_score=("soft_score", "std")
        )
        .reset_index()
        .sort_values("tag")
    )

    if len(tag_summary) != 6:
        raise ValueError(
            f"Expected 6 tags, found {len(tag_summary)}"
        )

    if not (tag_summary["num_prompts"] == 20).all():
        raise ValueError("Expected 20 prompts per tag.")

    prompt_summary.to_csv(
        os.path.join(C_DIR, "prompt_subset_per_prompt.csv"),
        index=False
    )

    tag_summary.to_csv(
        os.path.join(C_DIR, "prompt_subset_by_tag.csv"),
        index=False
    )

    metadata = {
        "dataset": "GenEval-style prompt subset",
        "num_prompts": int(len(prompt_summary)),
        "num_images": int(len(df)),
        "images_per_prompt": 32,
        "num_tags": int(len(tag_summary)),
        "prompts_per_tag": 20,
        "tags": tag_summary["tag"].tolist(),
        "duplicate_prompt_seed_pairs": 0,
        "validation": "passed"
    }

    with open(
        os.path.join(C_DIR, "prompt_subset_metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("PROMPT SUBSET VALIDATION")
    print("=" * 70)
    print(json.dumps(metadata, indent=2))

    print("\n" + "=" * 70)
    print("PROMPT DISTRIBUTION BY TAG")
    print("=" * 70)
    print(tag_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    for filename in [
        "prompt_subset_per_prompt.csv",
        "prompt_subset_by_tag.csv",
        "prompt_subset_metadata.json",
        "prompt_subset_analysis.log"
    ]:
        print(os.path.join(C_DIR, filename))

    logger.info("Prompt subset analysis completed.")


if __name__ == "__main__":
    main()
