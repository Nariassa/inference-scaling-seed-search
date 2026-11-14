
import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = "/content/drive/MyDrive/6132gp"

REPO_DIR = os.path.join(
    PROJECT_ROOT, "inference-scaling-seed-search"
)

RESULTS_DIR = os.path.join(
    REPO_DIR, "results", "results_C"
)

EMBEDDINGS_FILE = os.path.join(
    RESULTS_DIR, "clip_image_embeddings.npy"
)

CLIP_CSV = os.path.join(
    RESULTS_DIR, "clip_scores_quickgelu.csv"
)

OUTPUT_DIR = RESULTS_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

PER_PROMPT_FILE = os.path.join(
    OUTPUT_DIR, "clip_diversity_per_prompt.csv"
)

BY_TAG_FILE = os.path.join(
    OUTPUT_DIR, "clip_diversity_by_tag.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR, "diversity_summary.json"
)

LOG_FILE = os.path.join(
    OUTPUT_DIR, "diversity_scorer.log"
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

EXPECTED_IMAGES = 3840
EXPECTED_PROMPTS = 120
IMAGES_PER_PROMPT = 32

METRIC_NAME = "mean_pairwise_cosine_distance"


# ============================================================
# 4. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)


# ============================================================
# 5. LOAD DATA
# ============================================================

logger.info("=" * 65)
logger.info("DIVERSITY SCORING STARTED")
logger.info("=" * 65)

assert os.path.exists(EMBEDDINGS_FILE), (
    f"Missing embeddings: {EMBEDDINGS_FILE}"
)

assert os.path.exists(CLIP_CSV), (
    f"Missing CLIP CSV: {CLIP_CSV}"
)

logger.info("Loading image embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

logger.info("Embedding shape: %s", embeddings.shape)

logger.info("Loading CLIP records...")

df = pd.read_csv(CLIP_CSV)

required_columns = [
    "prompt_id",
    "tag"
]

for col in required_columns:
    assert col in df.columns, f"Missing column: {col}"

assert len(df) == EXPECTED_IMAGES
assert embeddings.shape[0] == EXPECTED_IMAGES
assert embeddings.shape[1] == 512

assert df["prompt_id"].nunique() == EXPECTED_PROMPTS

counts = df.groupby("prompt_id").size()

assert (counts == IMAGES_PER_PROMPT).all(), (
    "Each prompt must have exactly 32 images."
)

assert np.isfinite(embeddings).all(), (
    "Embeddings contain NaN or Inf."
)

# Verify that image embeddings are normalized.
norms = np.linalg.norm(embeddings, axis=1)

assert np.allclose(norms, 1.0, atol=1e-3), (
    "Embeddings are not normalized."
)

logger.info("Images: %d", len(df))
logger.info("Prompts: %d", df["prompt_id"].nunique())


# ============================================================
# 6. COMPUTE PAIRWISE DIVERSITY
# ============================================================

logger.info("Computing pairwise cosine distances...")

results = []

total_pairs = 0

for prompt_id, group in df.groupby(
    "prompt_id", sort=True
):

    indices = group.index.to_numpy()

    prompt_embeddings = embeddings[indices]

    # Cosine similarity because embeddings are normalized.
    similarity_matrix = (
        prompt_embeddings @ prompt_embeddings.T
    )

    # Keep only unique image pairs.
    upper_indices = np.triu_indices(
        IMAGES_PER_PROMPT, k=1
    )

    similarities = similarity_matrix[upper_indices]

    # Cosine distance = 1 - cosine similarity.
    distances = 1.0 - similarities

    assert len(distances) == 496

    mean_diversity = float(np.mean(distances))
    std_diversity = float(np.std(distances))

    tag_values = group["tag"].dropna().unique()

    assert len(tag_values) == 1, (
        f"Inconsistent tags for prompt {prompt_id}"
    )

    results.append({
        "prompt_id": prompt_id,
        "tag": tag_values[0],
        "num_images": IMAGES_PER_PROMPT,
        "num_pairs": len(distances),
        "diversity_mean": mean_diversity,
        "diversity_std": std_diversity
    })

    total_pairs += len(distances)


# ============================================================
# 7. SAVE PER-PROMPT RESULTS
# ============================================================

per_prompt = pd.DataFrame(results)

assert len(per_prompt) == EXPECTED_PROMPTS
assert total_pairs == 59520

per_prompt.to_csv(
    PER_PROMPT_FILE,
    index=False
)

logger.info("Saved per-prompt diversity results.")


# ============================================================
# 8. AGGREGATE BY TAG
# ============================================================

by_tag = (
    per_prompt
    .groupby("tag")
    .agg(
        num_prompts=("prompt_id", "count"),
        diversity_mean=("diversity_mean", "mean"),
        diversity_std=("diversity_mean", "std"),
        diversity_min=("diversity_mean", "min"),
        diversity_max=("diversity_mean", "max")
    )
    .reset_index()
)

by_tag.to_csv(
    BY_TAG_FILE,
    index=False
)

logger.info("Saved tag-level diversity results.")


# ============================================================
# 9. GLOBAL SUMMARY
# ============================================================

global_stats = per_prompt["diversity_mean"].describe()

summary = {
    "experiment": "CLIP Image Diversity Evaluation",
    "metric": METRIC_NAME,
    "model": "ViT-B-32-quickgelu",
    "pretrained": "openai",
    "num_images": EXPECTED_IMAGES,
    "num_prompts": EXPECTED_PROMPTS,
    "images_per_prompt": IMAGES_PER_PROMPT,
    "pairs_per_prompt": 496,
    "total_pairs": total_pairs,
    "embedding_dimension": 512,
    "global_diversity": {
        "mean": float(global_stats["mean"]),
        "std": float(global_stats["std"]),
        "min": float(global_stats["min"]),
        "25%": float(global_stats["25%"]),
        "median": float(global_stats["50%"]),
        "75%": float(global_stats["75%"]),
        "max": float(global_stats["max"])
    },
    "interpretation": (
        "Mean pairwise cosine distance among the 32 candidate "
        "images for each prompt. This measures candidate-pool "
        "diversity, not selected-output diversity."
    ),
    "completed_at": datetime.now().isoformat()
}

with open(SUMMARY_FILE, "w") as f:
    json.dump(summary, f, indent=4)


# ============================================================
# 10. VERIFY OUTPUTS
# ============================================================

for path in [
    PER_PROMPT_FILE,
    BY_TAG_FILE,
    SUMMARY_FILE,
    LOG_FILE
]:
    assert os.path.exists(path), f"Missing output: {path}"

logger.info("=" * 65)
logger.info("DIVERSITY SCORING COMPLETED")
logger.info("=" * 65)

logger.info("Total prompts: %d", len(per_prompt))
logger.info("Total image pairs: %d", total_pairs)

logger.info(
    "Global diversity statistics:\n%s",
    per_prompt["diversity_mean"].describe()
)

logger.info("\nDiversity by tag:\n%s", by_tag.to_string(index=False))

logger.info("All output files verified successfully.")
