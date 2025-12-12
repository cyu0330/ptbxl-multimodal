import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
import seaborn as sns


def main():
    merged_csv = Path("outputs/merged/test_03_04_05_merged.csv")
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_csv)
    print("[INFO] Loaded merged CSV:", df.shape)

    labels = ["MI", "STTC", "HYP", "CD", "NORM"]

    # ROC curves for each baseline class
    plot_baseline_per_class_roc(df, labels, out_dir / "baseline_per_class_roc.png")

    # Precision–Recall curves for each baseline class
    plot_baseline_per_class_pr(df, labels, out_dir / "baseline_per_class_pr.png")

    # MI probability distribution (baseline only)
    plot_baseline_mi_distribution(df, out_dir / "baseline_mi_distribution.png")

    print("[INFO] All baseline figures saved:", out_dir.resolve())


# ----------------------------------------------------------------------
# ROC per class
# ----------------------------------------------------------------------
def plot_baseline_per_class_roc(df, labels, out_path: Path):
    """Plot ROC curves for the baseline model (per diagnostic class)."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))

    for lb in labels:
        y_true = df[f"y_true_{lb}"].values.astype(float)
        y_prob = df[f"y_prob_{lb}"].values.astype(float)

        # skip classes without both positive and negative samples
        if np.unique(y_true).size < 2:
            print(f"[WARN] Skipped ROC for {lb} (y_true has single value).")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)

        ax.plot(fpr, tpr, linewidth=2, label=f"{lb} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Baseline model — ROC curves (per class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# PR per class
# ----------------------------------------------------------------------
def plot_baseline_per_class_pr(df, labels, out_path: Path):
    """Plot Precision–Recall curves for the baseline model (per class)."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))

    for lb in labels:
        y_true = df[f"y_true_{lb}"].values.astype(float)
        y_prob = df[f"y_prob_{lb}"].values.astype(float)

        if np.unique(y_true).size < 2:
            print(f"[WARN] Skipped PR for {lb} (y_true has single value).")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        ax.plot(recall, precision, linewidth=2, label=f"{lb} (AUPRC={auprc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Baseline model — Precision–Recall curves (per class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# MI probability distribution (baseline)
# ----------------------------------------------------------------------
def plot_baseline_mi_distribution(df, out_path: Path):
    """Plot the distribution of MI prediction probabilities for the baseline model."""
    plt.style.use("default")

    y_true = df["y_true_MI"].values.astype(float)
    y_prob = df["y_prob_MI"].values.astype(float)

    plt.figure(figsize=(8, 5))

    sns.kdeplot(
        y_prob[y_true == 1],
        label="MI positive",
        color="#4C72B0",
        shade=True
    )
    sns.kdeplot(
        y_prob[y_true == 0],
        label="MI negative",
        color="#4C72B0",
        linestyle="--"
    )

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Baseline model — MI probability distribution")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
