import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def main():
    merged_csv = Path("outputs/merged/test_03_04_05_merged.csv")
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)
    print("[INFO] Loaded merged CSV:", df.shape)

    # Labels
    labels = ["MI", "STTC", "HYP", "CD", "NORM"]

    # --------- Figure 18: MI distribution baseline vs multimodal ----------
    plot_mi_distribution(df, out_dir / "figure18_mi_distribution.png")

    # --------- Figure 19: Overall distribution (all labels) ---------------
    plot_overall_distribution(df, labels, out_dir / "figure19_overall_distribution.png")

    # --------- Figure 20: AF prediction distribution ----------------------
    if "y_true_AF" in df.columns:
        plot_af_distribution(df, out_dir / "figure20_af_distribution.png")

    print("[INFO] All distribution figures saved.")


# ----------------------- FIGURE 18 -------------------------------------

def plot_mi_distribution(df, out_path):
    """Baseline vs multimodal — MI prediction probability distribution."""

    y_true = df["y_true_MI"].values
    baseline_prob = df["y_prob_MI"].values
    mm_prob = df["y_prob_MI_mm"].values

    plt.figure(figsize=(8, 5))
    sns.kdeplot(baseline_prob[y_true == 1], label="Baseline MI=1", color="#4C72B0", shade=True)
    sns.kdeplot(baseline_prob[y_true == 0], label="Baseline MI=0", color="#4C72B0", linestyle="--")

    sns.kdeplot(mm_prob[y_true == 1], label="Multimodal MI=1", color="#DD8452", shade=True)
    sns.kdeplot(mm_prob[y_true == 0], label="Multimodal MI=0", color="#DD8452", linestyle="--")

    plt.title("Figure 18. MI prediction probability distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ----------------------- FIGURE 19 -------------------------------------

def plot_overall_distribution(df, labels, out_path):
    """Combine all classes into one big positive/negative set."""
    pos_vals = []
    neg_vals = []
    pos_vals_mm = []
    neg_vals_mm = []

    for lb in labels:
        y_true = df[f"y_true_{lb}"].values
        baseline_prob = df[f"y_prob_{lb}"].values
        mm_prob = df[f"y_prob_{lb}_mm"].values

        pos_vals.extend(baseline_prob[y_true == 1])
        neg_vals.extend(baseline_prob[y_true == 0])

        pos_vals_mm.extend(mm_prob[y_true == 1])
        neg_vals_mm.extend(mm_prob[y_true == 0])

    plt.figure(figsize=(8, 5))
    sns.kdeplot(pos_vals, label="Baseline Positive", color="#4C72B0")
    sns.kdeplot(neg_vals, label="Baseline Negative", color="#4C72B0", linestyle="--")

    sns.kdeplot(pos_vals_mm, label="Multimodal Positive", color="#DD8452")
    sns.kdeplot(neg_vals_mm, label="Multimodal Negative", color="#DD8452", linestyle="--")

    plt.title("Figure 19. Overall positive vs negative prediction distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ----------------------- FIGURE 20 -------------------------------------

def plot_af_distribution(df, out_path):
    """AF prediction probability distribution."""

    y_true = df["y_true_AF"].values
    y_prob = df[[c for c in df.columns if c.startswith("y_prob_AF")][0]].values

    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_prob[y_true == 1], label="AF = 1", color="#55A868", shade=True)
    sns.kdeplot(y_prob[y_true == 0], label="AF = 0", color="#55A868", linestyle="--")

    plt.title("Figure 20. AF prediction probability distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
