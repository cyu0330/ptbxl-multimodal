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

    # --------- Baseline per-class ROC (Figure B1) ----------
    fig_b1_path = out_dir / "baseline_b1_per_class_roc.png"
    plot_baseline_per_class_roc(df, labels, fig_b1_path)

    # --------- Baseline per-class PR (Figure B2) ----------
    fig_b2_path = out_dir / "baseline_b2_per_class_pr.png"
    plot_baseline_per_class_pr(df, labels, fig_b2_path)

    # --------- Baseline MI probability distribution (Figure B3) ----------
    fig_b3_path = out_dir / "baseline_b3_mi_distribution.png"
    plot_baseline_mi_distribution(df, fig_b3_path)

    print("[INFO] Baseline-only figures saved to:", out_dir.resolve())
    print("[INFO] You can treat these as Figure B1, B2, B3 in the thesis.")


def plot_baseline_per_class_roc(df, labels, out_path: Path):
    """Baseline model: per-class ROC curves in a single figure."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))

    for lb in labels:
        y_true = df[f"y_true_{lb}"].values.astype(float)
        y_prob = df[f"y_prob_{lb}"].values.astype(float)

        # skip if no positive/negative variation
        if np.unique(y_true).size < 2:
            print(f"[WARN] ROC skipped for {lb} (y_true has single value).")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{lb} (AUROC={auroc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Baseline per-class ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_baseline_per_class_pr(df, labels, out_path: Path):
    """Baseline model: per-class Precision–Recall curves."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 6))

    for lb in labels:
        y_true = df[f"y_true_{lb}"].values.astype(float)
        y_prob = df[f"y_prob_{lb}"].values.astype(float)

        if np.unique(y_true).size < 2:
            print(f"[WARN] PR skipped for {lb} (y_true has single value).")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{lb} (AUPRC={auprc:.3f})", linewidth=2)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Baseline per-class Precision–Recall curves")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_baseline_mi_distribution(df, out_path: Path):
    """Baseline-only MI prediction probability distribution."""
    plt.style.use("default")

    y_true = df["y_true_MI"].values.astype(float)
    y_prob = df["y_prob_MI"].values.astype(float)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(y_prob[y_true == 1], label="MI=1 (positive)", color="#4C72B0", shade=True)
    sns.kdeplot(y_prob[y_true == 0], label="MI=0 (negative)", color="#4C72B0", linestyle="--")

    plt.title("Baseline MI prediction probability distribution")
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
