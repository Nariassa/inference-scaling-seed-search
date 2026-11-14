
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_DIR = (
    "/content/drive/MyDrive/6132gp/"
    "inference-scaling-seed-search"
)

RESULTS_DIR = os.path.join(
    REPO_DIR, "results", "results_C"
)

# Load verified summary
csv_path = os.path.join(
    RESULTS_DIR,
    "verifier_ablation_summary.csv"
)

df = pd.read_csv(csv_path)

N_VALUES = [1, 2, 4, 8, 16, 32]

# Consistent ordering
df["verifier"] = pd.Categorical(
    df["verifier"],
    categories=["CLIP", "ImageReward"],
    ordered=True
)

df = df.sort_values(["N", "verifier"])

clip = df[df["verifier"] == "CLIP"].sort_values("N")
ir = df[df["verifier"] == "ImageReward"].sort_values("N")

assert clip["N"].tolist() == N_VALUES
assert ir["N"].tolist() == N_VALUES

# Extract means and standard deviations
clip_score = clip["clip_score_mean"].to_numpy()
clip_score_std = clip["clip_score_mean_std"].to_numpy()

ir_clip_score = ir["clip_score_mean"].to_numpy()
ir_clip_score_std = ir["clip_score_mean_std"].to_numpy()

clip_geneval = clip["geneval_accuracy"].to_numpy()
clip_geneval_std = clip["geneval_accuracy_std"].to_numpy()

ir_geneval = ir["geneval_accuracy"].to_numpy()
ir_geneval_std = ir["geneval_accuracy_std"].to_numpy()

# Plotting
fig, axes = plt.subplots(
    1, 2,
    figsize=(7.2, 3.25)
)

x = np.arange(len(N_VALUES))

# Panel A: CLIP similarity
ax = axes[0]

ax.errorbar(
    x,
    clip_score,
    yerr=clip_score_std,
    marker="o",
    markersize=4,
    linewidth=1.6,
    capsize=3,
    label="CLIP selection"
)

ax.errorbar(
    x,
    ir_clip_score,
    yerr=ir_clip_score_std,
    marker="s",
    markersize=4,
    linewidth=1.6,
    capsize=3,
    linestyle="--",
    label="ImageReward selection"
)

ax.set_title("(a) CLIP similarity", fontsize=10)
ax.set_ylabel("Mean CLIP similarity", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(N_VALUES)
ax.set_xlabel("Candidate group size N", fontsize=9)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel B: GenEval accuracy
ax = axes[1]

ax.errorbar(
    x,
    clip_geneval,
    yerr=clip_geneval_std,
    marker="o",
    markersize=4,
    linewidth=1.6,
    capsize=3,
    label="CLIP selection"
)

ax.errorbar(
    x,
    ir_geneval,
    yerr=ir_geneval_std,
    marker="s",
    markersize=4,
    linewidth=1.6,
    capsize=3,
    linestyle="--",
    label="ImageReward selection"
)

ax.set_title("(b) GenEval accuracy", fontsize=10)
ax.set_ylabel("GenEval accuracy", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(N_VALUES)
ax.set_xlabel("Candidate group size N", fontsize=9)
ax.set_ylim(0.35, 0.70)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.08),
    ncol=2,
    frameon=False,
    fontsize=8.5
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

# Save
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

pdf_path = os.path.join(
    FIG_DIR,
    "fig2_verifier_ablation.pdf"
)

png_path = os.path.join(
    FIG_DIR,
    "fig2_verifier_ablation.png"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Figure 2 generation completed.")
print("PDF:", pdf_path)
print("PNG:", png_path)
print("N values:", N_VALUES)
print("Groups per N:", clip["N"].map(
    df.groupby("N")["num_groups"].first()
).tolist())
print("N=32 standard deviations are NaN:", np.isnan(
    clip_geneval_std[-1]
) and np.isnan(ir_geneval_std[-1]))
