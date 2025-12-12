# scripts/15_plot_all_figures.py
#
# Generate summary metrics and publication-style plots for:
#   - ECG-only baseline
#   - ECG + demographics (multimodal)
#   - AF binary classifier
#
# Output figures follow the numbering used in the dissertation:
#   Figure 14: Macro AUROC / AUPRC
#   Figure 15: Per-class AUROC
#   Figure 16: ROC for MI
#   Figure 17: AF ROC & PR curves (if available)

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def compute_multilabel_metrics(y_true, y_prob):
    """Compute AUROC/AUPRC (macro + per-class)."""
    n_cls = y_true.shape[1]
    aurocs, auprcs = [], []

    for k in range(n_cls):
        yt = y_true[:, k]
        yp = y_prob[:, k]

        # skip trivial classes
        if np.unique(yt).size < 2:
            aurocs.append(np.nan)
            auprcs.append(np.nan)
            continue

        aurocs.append(roc_auc_score(yt, yp))
        auprcs.append(average_precision_score(yt, yp))

    return {
        "auroc_macro": float(np.nanmean(aurocs)),
        "auprc_macro": float(np.nanmean(auprcs)),
        "auroc_per_class": aurocs,
        "auprc_per_class": auprcs,
    }


def save_metrics_table(metrics, labels, out_path):
    """Store metrics into a CSV for use in the report."""
    rows = []

    for model_key, m in metrics.items():
        r = {
            "model": model_key,
            "auroc_macro": m["auroc_macro"],
            "auprc_macro": m["auprc_macro"],
        }
        for lb, v in zip(labels, m["auroc_per_class"]):
            r[f"auroc_{lb}"] = v
        for lb, v in zip(labels, m["auprc_per_class"]):
            r[f"auprc_{lb}"] = v
        rows.append(r)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[INFO] Saved metrics table: {out_path}")


# -------------------------------------------------------------
# Figure 14 – Macro AUROC / AUPRC
# -------------------------------------------------------------
def plot_macro_scores(metrics, model_defs, out_path):
    plt.style.use("default")

    model_keys = list(model_defs.keys())
    x = np.arange(len(model_keys))

    auroc_vals = [metrics[k]["auroc_macro"] for k in model_keys]
    auprc_vals = [metrics[k]["auprc_macro"] for k in model_keys]

    width = 0.35
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, auroc_vals, width, label="AUROC", color=colors[0])
    ax.bar(x + width / 2, auprc_vals, width, label="AUPRC", color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels([model_defs[k]["name"] for k in model_keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Macro AUROC / AUPRC on PTB-XL test set")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # annotate
    for i, v in enumerate(auroc_vals):
        ax.text(x[i] - width / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(auprc_vals):
        ax.text(x[i] + width / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------
# Figure 15 – Per-class AUROC
# -------------------------------------------------------------
def plot_per_class_auc(metrics, model_defs, labels, out_path):
    plt.style.use("default")

    model_keys = list(model_defs.keys())
    x = np.arange(len(labels))
    width = 0.35

    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, key in enumerate(model_keys):
        offset = (-0.5 + i) * width
        ax.bar(
            x + offset,
            metrics[key]["auroc_per_class"],
            width,
            label=model_defs[key]["name"],
            color=colors[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-class AUROC comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------
# Figure 16 – ROC curves for one class (e.g., MI)
# -------------------------------------------------------------
def plot_single_roc(y_true, y_prob_dict, auroc_dict, model_defs, class_name, out_path):
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(5, 5))

    colors = {
        "ecg": "#4C72B0",
        "mm": "#DD8452",
    }

    for key, yp in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, yp)
        ax.plot(
            fpr, tpr,
            label=f"{model_defs[key]['name']} (AUROC={auroc_dict[key]:.3f})",
            linewidth=2,
            color=colors[key],
        )

    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves for {class_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------
# Figure 17 – AF ROC & PR curves
# -------------------------------------------------------------
def plot_af_curves(y_true, y_prob, out_path):
    plt.style.use("default")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    color = "#55A868"

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"AUROC={auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
    ax.set_title("AF ROC curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.grid(alpha=0.3)

    # PR
    ax = axes[1]
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f"AUPRC={auprc:.3f}")
    ax.set_title("AF Precision–Recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    merged_path = Path("outputs/merged/test_03_04_05_merged.csv")
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["MI", "STTC", "HYP", "CD", "NORM"]

    print(f"[INFO] Loading merged results: {merged_path}")
    df = pd.read_csv(merged_path)
    print(f"[INFO] Shape: {df.shape}")

    # ground truth
    y_true = df[[f"y_true_{lb}" for lb in labels]].values.astype(float)

    # model definitions
    model_defs = {
        "ecg": {
            "name": "ECG-only",
            "cols": [f"y_prob_{lb}" for lb in labels],
        },
        "mm": {
            "name": "ECG+demographics",
            "cols": [f"y_prob_{lb}_mm" for lb in labels],
        },
    }

    # compute metrics
    metrics = {}
    y_probs = {}

    for key, md in model_defs.items():
        y_prob = df[md["cols"]].values.astype(float)
        y_probs[key] = y_prob
        metrics[key] = compute_multilabel_metrics(y_true, y_prob)

    # save summary table
    save_metrics_table(metrics, labels, out_dir / "metrics_summary.csv")

    # Figure 14
    plot_macro_scores(metrics, model_defs, out_dir / "figure14_macro_scores.png")

    # Figure 15
    plot_per_class_auc(metrics, model_defs, labels, out_dir / "figure15_per_class_auroc.png")

    # Figure 16 – MI = index 0
    plot_single_roc(
        y_true[:, 0],
        {k: y_probs[k][:, 0] for k in y_probs},
        {k: metrics[k]["auroc_per_class"][0] for k in metrics},
        model_defs,
        class_name="MI",
        out_path=out_dir / "figure16_mi_roc.png",
    )

    # Figure 17 – AF if available
    if "y_true_AF" in df.columns and any(c.startswith("y_prob_AF") for c in df.columns):
        y_true_af = df["y_true_AF"].values.astype(float)
        y_prob_af = df[[c for c in df.columns if c.startswith("y_prob_AF")]].values[:, 0]
        plot_af_curves(y_true_af, y_prob_af, out_dir / "figure17_af_curves.png")
        print("[INFO] AF figure saved.")
    else:
        print("[WARN] AF predictions not found; skip AF plots.")

    print("[INFO] Finished. All figures saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
