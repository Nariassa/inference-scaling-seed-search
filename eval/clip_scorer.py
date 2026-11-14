
"""
CLIP Scoring for Inference-Time Scaling for Diffusion

Project: AI6132 Group Project
Component: C - Evaluation and Analysis

Model:
    OpenCLIP ViT-B-32-quickgelu
    Pretrained weights: openai

Input:
    120 GenEval prompts
    3840 generated images (32 images per prompt)

Output:
    clip_scores_quickgelu.csv
    clip_image_embeddings.npy
    clip_scoring_summary.json
    clip_scorer.log

This script evaluates existing generated images.
It does NOT train CLIP or generate new images.
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import open_clip

from PIL import Image
from tqdm.auto import tqdm


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = "/content/drive/MyDrive/6132gp"

REPO_DIR = os.path.join(
    PROJECT_ROOT,
    "inference-scaling-seed-search"
)

SAMPLES_DIR = (
    "/content/drive/MyDrive/ai6132_its/"
    "cache/sd15_geneval_T64/samples"
)

DATA_DIR = os.path.join(REPO_DIR, "data")

OUTPUT_DIR = os.path.join(
    REPO_DIR,
    "results",
    "results_C"
)

GENEVAL_CSV = os.path.join(
    DATA_DIR,
    "geneval_per_image.csv"
)

PROMPT_JSONL = os.path.join(
    DATA_DIR,
    "geneval_subset_120.jsonl"
)

SCORE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "clip_scores_quickgelu.csv"
)

EMBEDDING_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "clip_image_embeddings.npy"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "clip_scoring_summary.json"
)

LOG_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "clip_scorer.log"
)

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

EXPECTED_IMAGES = 3840
EXPECTED_PROMPTS = 120
EXPECTED_IMAGES_PER_PROMPT = 32

BATCH_SIZE = 64


# ============================================================
# 2. Logging
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_OUTPUT,
            mode="w",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. Environment checks
# ============================================================

logger.info("=" * 65)
logger.info("CLIP SCORING STARTED")
logger.info("=" * 65)

logger.info("PyTorch version: %s", torch.__version__)
logger.info("OpenCLIP version: %s", open_clip.__version__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Device: %s", DEVICE)

if DEVICE == "cuda":
    logger.info(
        "GPU: %s",
        torch.cuda.get_device_name(0)
    )

assert os.path.isdir(SAMPLES_DIR), (
    f"Samples directory not found: {SAMPLES_DIR}"
)

assert os.path.isfile(GENEVAL_CSV), (
    f"GenEval CSV not found: {GENEVAL_CSV}"
)

assert os.path.isfile(PROMPT_JSONL), (
    f"Prompt JSONL not found: {PROMPT_JSONL}"
)


# ============================================================
# 4. Load GenEval records
# ============================================================

logger.info("Loading GenEval records...")

geneval_df = pd.read_csv(GENEVAL_CSV)

required_columns = {
    "prompt_id",
    "seed",
    "tag"
}

missing_columns = required_columns - set(
    geneval_df.columns
)

assert not missing_columns, (
    f"Missing GenEval columns: {missing_columns}"
)

assert len(geneval_df) == EXPECTED_IMAGES, (
    f"Expected {EXPECTED_IMAGES} images, "
    f"found {len(geneval_df)}"
)

assert not geneval_df.duplicated(
    subset=["prompt_id", "seed"]
).any(), "Duplicate prompt_id and seed records detected."

geneval_df["prompt_id"] = geneval_df[
    "prompt_id"
].astype(int)

geneval_df["seed"] = geneval_df[
    "seed"
].astype(int)

logger.info("GenEval records: %d", len(geneval_df))


# ============================================================
# 5. Load text prompts
# ============================================================

logger.info("Loading text prompts...")

prompts = {}

with open(
    PROMPT_JSONL,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        if not line.strip():
            continue

        item = json.loads(line)

        prompt_id = int(item["prompt_id"])

        prompts[prompt_id] = item["prompt"]

assert len(prompts) == EXPECTED_PROMPTS, (
    f"Expected {EXPECTED_PROMPTS} prompts, "
    f"found {len(prompts)}"
)

assert set(geneval_df["prompt_id"]) == set(prompts), (
    "Prompt IDs in GenEval CSV and JSONL do not match."
)

images_per_prompt = geneval_df.groupby(
    "prompt_id"
).size()

assert (
    images_per_prompt == EXPECTED_IMAGES_PER_PROMPT
).all(), "Some prompts do not have exactly 32 images."

logger.info("Prompts loaded: %d", len(prompts))


# ============================================================
# 6. Load pretrained OpenCLIP model
# ============================================================

logger.info("Loading OpenCLIP model...")

model, _, preprocess = (
    open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED
    )
)

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

model = model.to(DEVICE)
model.eval()

logger.info("Model: %s", MODEL_NAME)
logger.info("Pretrained weights: %s", PRETRAINED)


# ============================================================
# 7. Encode text prompts
# ============================================================

logger.info("Encoding text prompts...")

prompt_ids = sorted(prompts.keys())

text_features = {}

with torch.inference_mode():

    for prompt_id in tqdm(
        prompt_ids,
        desc="Encoding prompts"
    ):

        prompt = prompts[prompt_id]

        tokens = tokenizer([prompt]).to(DEVICE)

        features = model.encode_text(tokens)

        features = features / features.norm(
            dim=-1,
            keepdim=True
        ).clamp(min=1e-12)

        text_features[prompt_id] = features


logger.info("Text encoding completed.")


# ============================================================
# 8. Encode images and calculate CLIP similarity
# ============================================================

logger.info("Starting image encoding and scoring...")

all_embeddings = []
all_scores = []

for start in tqdm(
    range(0, len(geneval_df), BATCH_SIZE),
    desc="Scoring images"
):

    batch_df = geneval_df.iloc[
        start:start + BATCH_SIZE
    ]

    image_tensors = []
    batch_metadata = []

    for _, row in batch_df.iterrows():

        prompt_id = int(row["prompt_id"])
        seed = int(row["seed"])

        image_path = os.path.join(
            SAMPLES_DIR,
            f"{prompt_id:05d}",
            f"{seed:05d}.png"
        )

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            image_tensor = preprocess(image)

        image_tensors.append(image_tensor)

        batch_metadata.append({
            "prompt_id": prompt_id,
            "seed": seed,
            "filename": image_path,
            "tag": row["tag"],
            "prompt": prompts[prompt_id]
        })

    image_batch = torch.stack(
        image_tensors
    ).to(DEVICE)

    with torch.inference_mode():

        image_features = model.encode_image(
            image_batch
        )

        image_features = image_features / (
            image_features.norm(
                dim=-1,
                keepdim=True
            ).clamp(min=1e-12)
        )

        # Calculate cosine similarity between
        # each image and its corresponding prompt.
        batch_prompt_features = torch.cat([
            text_features[item["prompt_id"]]
            for item in batch_metadata
        ], dim=0)

        similarities = torch.sum(
            image_features * batch_prompt_features,
            dim=-1
        )

    image_features_cpu = (
        image_features.cpu().numpy().astype(np.float32)
    )

    similarities_cpu = (
        similarities.cpu().numpy()
    )

    all_embeddings.append(image_features_cpu)

    for item, score in zip(
        batch_metadata,
        similarities_cpu
    ):

        all_scores.append({
            **item,
            "clip_score": float(score)
        })


# ============================================================
# 9. Assemble and validate results
# ============================================================

logger.info("Assembling results...")

embeddings = np.concatenate(
    all_embeddings,
    axis=0
)

scores_df = pd.DataFrame(all_scores)

assert len(scores_df) == EXPECTED_IMAGES

assert embeddings.shape == (
    EXPECTED_IMAGES,
    512
), f"Unexpected embedding shape: {embeddings.shape}"

assert np.isfinite(embeddings).all(), (
    "Non-finite values found in image embeddings."
)

assert np.isfinite(
    scores_df["clip_score"]
).all(), "Non-finite CLIP scores detected."

assert not scores_df.duplicated(
    subset=["prompt_id", "seed"]
).any(), "Duplicate scoring records detected."

# Check that each image embedding is normalized.
embedding_norms = np.linalg.norm(
    embeddings,
    axis=1
)

assert np.allclose(
    embedding_norms,
    1.0,
    atol=1e-4
), "Image embeddings are not normalized."


# ============================================================
# 10. Save results to Google Drive
# ============================================================

logger.info("Saving experiment results...")

scores_df.to_csv(
    SCORE_OUTPUT,
    index=False
)

np.save(
    EMBEDDING_OUTPUT,
    embeddings
)

score_statistics = (
    scores_df["clip_score"]
    .describe()
    .to_dict()
)

summary = {
    "experiment": "CLIP Scoring",
    "project": "Inference-Time Scaling for Diffusion",
    "model": MODEL_NAME,
    "pretrained": PRETRAINED,
    "metric": "Cosine similarity",
    "device": DEVICE,
    "image_count": len(scores_df),
    "prompt_count": len(prompts),
    "images_per_prompt": EXPECTED_IMAGES_PER_PROMPT,
    "embedding_shape": list(embeddings.shape),
    "embedding_dimension": int(embeddings.shape[1]),
    "embedding_normalized": True,
    "score_statistics": score_statistics,
    "completed_at": datetime.now().isoformat(),
    "score_file": SCORE_OUTPUT,
    "embedding_file": EMBEDDING_OUTPUT
}

with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 11. Final verification
# ============================================================

logger.info("=" * 65)
logger.info("CLIP SCORING COMPLETED")
logger.info("=" * 65)

logger.info("Images scored: %d", len(scores_df))
logger.info("Prompts: %d", len(prompts))
logger.info("Embedding shape: %s", embeddings.shape)

logger.info(
    "CLIP score statistics:\n%s",
    scores_df["clip_score"].describe().to_string()
)

for output_path in [
    SCORE_OUTPUT,
    EMBEDDING_OUTPUT,
    SUMMARY_OUTPUT,
    LOG_OUTPUT
]:

    assert os.path.isfile(output_path), (
        f"Output file missing: {output_path}"
    )

    logger.info(
        "Saved: %s (%.2f MB)",
        output_path,
        os.path.getsize(output_path) / 1024 / 1024
    )

logger.info("All output files verified successfully.")
