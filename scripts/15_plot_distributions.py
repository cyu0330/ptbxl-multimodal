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

    labels = ["MI", "STTC", "HYP", "CD", "NORM"]

    # MI distribution
    plot_mi_distribution(df, out_dir / "mi_distribution.png")

    # All-class positive/negative distribution
    plot_overall_distribution(df, labels, out_dir / "overall_prediction_distribution.png")

    # AF binary model distribution (if available)
    if "y_true_AF" in df.columns:
        plot_af_distribution(df, out_dir / "af_prediction_distribution.png")

    print("[INFO] Distribution figures saved.")


# --------------------------------------------------------------------
# MI
# --------------------------------------------------------------------
def plot_mi_distribution(df, out_path):
    """Probability distribution for MI: baseline vs multimodal."""

    y_true = df["y_true_MI"].values
    p_base = df["y_prob_MI"].values
    p_mm = df["y_prob_MI_mm"].values

    plt.figure(figsize=(8, 5))

    sns.kdeplot(p_base[y_true == 1], label="Baseline (MI=1)", color="#4C72B0", shade=True)
    sns.kdeplot(p_base[y_true == 0], label="Baseline (MI=0)", color="#4C72B0", linestyle="--")

    sns.kdeplot(p_mm[y_true == 1], label="Multimodal (MI=1)", color="#DD8452", shade=True)
    sns.kdeplot(p_mm[y_true == 0], label="Multimodal (MI=0)", color="#DD8452", linestyle="--")

    plt.title("MI prediction probability distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# --------------------------------------------------------------------
# Overall (all 5 classes pooled together)
# --------------------------------------------------------------------
def plot_overall_distribution(df, labels, out_path):
    """Combine all classes together and compare positive vs negative."""

    pos_base, neg_base = [], []
    pos_mm, neg_mm = [], []

    for lb in labels:
        yt = df[f"y_true_{lb}"].values
        pb = df[f"y_prob_{lb}"].values
        pm = df[f"y_prob_{lb}_mm"].values

        pos_base.extend(pb[yt == 1])
        neg_base.extend(pb[yt == 0])
        pos_mm.extend(pm[yt == 1])
        neg_mm.extend(pm[yt == 0])

    plt.figure(figsize=(8, 5))

    sns.kdeplot(pos_base, label="Baseline (Positive)", color="#4C72B0")
    sns.kdeplot(neg_base, label="Baseline (Negative)", color="#4C72B0", linestyle="--")

    sns.kdeplot(pos_mm, label="Multimodal (Positive)", color="#DD8452")
    sns.kdeplot(neg_mm, label="Multimodal (Negative)", color="#DD8452", linestyle="--")

    plt.title("Prediction probability distribution (all classes combined)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# --------------------------------------------------------------------
# AF
# --------------------------------------------------------------------
def plot_af_distribution(df, out_path):
    """Probability distribution for the AF binary classifier."""

    y_true = df["y_true_AF"].values
    prob_col = [c for c in df.columns if c.startswith("y_prob_AF")][0]
    p = df[prob_col].values

    plt.figure(figsize=(8, 5))

    sns.kdeplot(p[y_true == 1], label="AF = 1", color="#55A868", shade=True)
    sns.kdeplot(p[y_true == 0], label="AF = 0", color="#55A868", linestyle="--")

    plt.title("AF prediction probability distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
