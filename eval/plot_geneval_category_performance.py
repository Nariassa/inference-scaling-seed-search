
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_DIR = "/content/drive/MyDrive/6132gp/inference-scaling-seed-search"
RESULTS_DIR = os.path.join(REPO_DIR, "results", "results_C")

# Load verified category statistics
csv_path = os.path.join(
    RESULTS_DIR,
    "fig1_geneval_category_statistics.csv"
)

df = pd.read_csv(csv_path)

category_order = [
    "single_object",
    "colors",
    "counting",
    "two_object",
    "color_attr",
    "position"
]

df["tag"] = pd.Categorical(
    df["tag"],
    categories=category_order,
    ordered=True
)

df = df.sort_values("tag")

labels = [
    "Single object",
    "Colors",
    "Counting",
    "Two objects",
    "Color attributes",
    "Position"
]

accuracy = df["mean_accuracy"].to_numpy()
soft_score = df["mean_soft_score"].to_numpy()

accuracy_std = df["std_accuracy"].to_numpy()
soft_std = df["std_soft_score"].to_numpy()

x = np.arange(len(labels))
width = 0.36

# Create figure
fig, ax = plt.subplots(figsize=(7.2, 4.0))

ax.bar(
    x - width / 2,
    accuracy,
    width,
    yerr=accuracy_std,
    capsize=3,
    label="Accuracy",
    edgecolor="black",
    linewidth=0.6,
    error_kw={"elinewidth": 0.9}
)

ax.bar(
    x + width / 2,
    soft_score,
    width,
    yerr=soft_std,
    capsize=3,
    label="Soft score",
    hatch="//",
    edgecolor="black",
    linewidth=0.6,
    error_kw={"elinewidth": 0.9}
)

# Formatting
ax.set_ylabel("Mean score across prompts", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.tick_params(axis="y", labelsize=9)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="upper right"
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.5
)

ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# Save figures
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

pdf_path = os.path.join(
    FIG_DIR,
    "fig1_geneval_category_performance.pdf"
)

png_path = os.path.join(
    FIG_DIR,
    "fig1_geneval_category_performance.png"
)

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

plt.show()
plt.close(fig)

print("Figure generation completed.")
print("PDF:", pdf_path)
print("PNG:", png_path)
print("Number of categories:", len(df))
print("Prompts per category:", df["num_prompts"].tolist())
