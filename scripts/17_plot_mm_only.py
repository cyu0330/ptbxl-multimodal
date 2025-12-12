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

    # per-class ROC (multimodal)
    fig_m1 = out_dir / "mm_m1_per_class_roc.png"
    plot_mm_per_class_roc(df, labels, fig_m1)

    # per-class PR (multimodal)
    fig_m2 = out_dir / "mm_m2_per_class_pr.png"
    plot_mm_per_class_pr(df, labels, fig_m2)

    # MI probability distribution (multimodal)
    fig_m3 = out_dir / "mm_m3_mi_distribution.png"
    plot_mm_mi_distribution(df, fig_m3)

    print("[INFO] Multimodal figures saved to:", out_dir.resolve())


def plot_mm_per_class_roc(df, labels, out_path: Path):
    """Plot ROC curves for each class using multimodal probabilities."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))

    for lb in labels:
        y_true = df[f"y_true_{lb}"].values.astype(float)
        y_prob = df[f"y_prob_{lb}_mm"].values.astype(float)

        # skip class if no positive/negative variation
        if np.unique(y_true).size < 2:
            print(f"[WARN] ROC skipped for {lb} (insufficient label variation).")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{lb} (AUROC={auroc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multimodal per-class ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_mm_per_class_pr(df, labels, out_path: Path):
    """Plot Precision–Recall curves for each class using multimodal probabilities."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))

    for lb in labels:
        y_true = df[f"y_true_{lb}"].values.astype(float)
        y_prob = df[f"y_prob_{lb}_mm"].values.astype(float)

        if np.unique(y_true).size < 2:
            print(f"[WARN] PR skipped for {lb} (insufficient label variation).")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{lb} (AUPRC={auprc:.3f})", linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multimodal per-class Precision–Recall curves")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_mm_mi_distribution(df, out_path: Path):
    """Density distribution of MI predictions using multimodal probabilities."""
    plt.style.use("default")

    y_true = df["y_true_MI"].values.astype(float)
    y_prob = df["y_prob_MI_mm"].values.astype(float)

    plt.figure(figsize=(8, 5))

    sns.kdeplot(y_prob[y_true == 1], label="MI = 1", color="#DD8452", fill=True)
    sns.kdeplot(y_prob[y_true == 0], label="MI = 0", color="#DD8452", linestyle="--")

    plt.title("Multimodal MI prediction distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
