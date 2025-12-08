import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from pathlib import Path


def main():
    # ===== 1. Paths & basic config =====
    merged_csv = Path("outputs/merged/test_03_04_05_merged.csv")
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # High-level diagnostic classes (consistent with label_maps.py)
    labels = ["MI", "STTC", "HYP", "CD", "NORM"]

    print(f"[INFO] Loading merged CSV: {merged_csv}")
    df = pd.read_csv(merged_csv)
    print(f"[INFO] df shape: {df.shape}")
    print("[INFO] Example columns:", df.columns[:20].tolist())

    # ===== 2. Prepare y_true and y_pred for ECG-only and ECG+Demo =====
    # expected pattern: y_true_MI, y_prob_MI_ecg, y_prob_MI_mm, etc.
    y_true_cols = [f"y_true_{lb}" for lb in labels]
    for c in y_true_cols:
        if c not in df.columns:
            raise KeyError(
                f"Required ground-truth column '{c}' not found in merged CSV. "
                "Please check column names."
            )
    y_true = df[y_true_cols].values.astype(float)

    model_defs = {
        "ecg": {
            "pretty_name": "ECG-only baseline",
            # ECG baseline columns are named like: y_prob_MI, y_prob_STTC, ...
            "prob_cols": [f"y_prob_{lb}" for lb in labels],
        },
        "mm": {
            "pretty_name": "ECG+demographics",
            # Multimodal columns are named like: y_prob_MI_mm, ...
            "prob_cols": [f"y_prob_{lb}_mm" for lb in labels],
        },
    }

    metrics = {}
    y_probs = {}

    for key, cfg in model_defs.items():
        prob_cols = cfg["prob_cols"]
        missing = [c for c in prob_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing probability columns for model '{key}': {missing}. "
                "Please check column names in the merged CSV."
            )
        y_prob = df[prob_cols].values.astype(float)
        y_probs[key] = y_prob
        metrics[key] = compute_multilabel_metrics(y_true, y_prob)

    # ===== 3. Save a simple metrics summary (for tables in thesis) =====
    save_metrics_summary(metrics, labels, out_dir / "metrics_summary.csv")

    # ===== 4. Figure 14: macro AUROC & AUPRC (bar plot) =====
    fig14_path = out_dir / "figure14_macro_auroc_auprc.png"
    plot_macro_bar(metrics, model_defs, fig14_path)

    # ===== 5. Figure 15: per-class AUROC (grouped bar plot) =====
    fig15_path = out_dir / "figure15_per_class_auroc.png"
    plot_per_class_auroc(metrics, model_defs, labels, fig15_path)

    # ===== 6. Figure 16: ROC curves for MI (index 0) =====
    fig16_path = out_dir / "figure16_mi_roc_curves.png"
    plot_single_class_roc(
        y_true[:, 0],  # MI
        {k: y_probs[k][:, 0] for k in y_probs.keys()},
        {k: metrics[k]["auroc_per_class"][0] for k in metrics.keys()},
        model_defs,
        class_name="MI",
        out_path=fig16_path,
    )

    # ===== 7. Figure 17: AF binary model ROC & PR (if available) =====
    af_y_true_col = None
    af_y_prob_col = None
    for col in df.columns:
        if col == "y_true_AF":
            af_y_true_col = col
        if col.startswith("y_prob_AF"):
            af_y_prob_col = col

    if af_y_true_col is not None and af_y_prob_col is not None:
        print("[INFO] Found AF columns, plotting AF ROC & PR curves.")
        y_true_af = df[af_y_true_col].values.astype(float)
        y_prob_af = df[af_y_prob_col].values.astype(float)
        fig17_path = out_dir / "figure17_af_roc_pr.png"
        plot_af_roc_pr(y_true_af, y_prob_af, fig17_path)
    else:
        print("[WARN] AF columns not found, skip Figure 17.")

    print("[INFO] All figures saved to:", out_dir.resolve())


def compute_multilabel_metrics(y_true, y_prob):
    """Compute per-class and macro AUROC/AUPRC for multi-label setting."""
    n_classes = y_true.shape[1]
    aurocs, auprcs = [], []

    for k in range(n_classes):
        y_t = y_true[:, k]
        y_p = y_prob[:, k]

        if np.unique(y_t).size < 2:
            # skip classes that are all-zero or all-one
            aurocs.append(np.nan)
            auprcs.append(np.nan)
            continue

        aurocs.append(roc_auc_score(y_t, y_p))
        auprcs.append(average_precision_score(y_t, y_p))

    auroc_macro = np.nanmean(aurocs)
    auprc_macro = np.nanmean(auprcs)

    return {
        "auroc_macro": float(auroc_macro),
        "auprc_macro": float(auprc_macro),
        "auroc_per_class": aurocs,
        "auprc_per_class": auprcs,
    }


def save_metrics_summary(metrics, labels, out_path: Path):
    """Save macro & per-class metrics into a CSV for reporting."""
    rows = []
    for model_key, m in metrics.items():
        row = {
            "model": model_key,
            "auroc_macro": m["auroc_macro"],
            "auprc_macro": m["auprc_macro"],
        }
        # add per-class AUROC / AUPRC
        for lb, v in zip(labels, m["auroc_per_class"]):
            row[f"auroc_{lb}"] = v
        for lb, v in zip(labels, m["auprc_per_class"]):
            row[f"auprc_{lb}"] = v
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Saved metrics summary to {out_path}")


def plot_macro_bar(metrics, model_defs, out_path: Path):
    """Figure 14: bar plot of macro AUROC and AUPRC for two models."""
    plt.style.use("default")  # clean, colourful
    model_keys = list(model_defs.keys())
    x = np.arange(len(model_keys))
    width = 0.35

    aurocs = [metrics[k]["auroc_macro"] for k in model_keys]
    auprcs = [metrics[k]["auprc_macro"] for k in model_keys]

    colors = ["#4C72B0", "#DD8452"]  # blue, orange
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x - width / 2, aurocs, width, label="AUROC", color=colors[0])
    ax.bar(x + width / 2, auprcs, width, label="AUPRC", color=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels([model_defs[k]["pretty_name"] for k in model_keys])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Macro AUROC and AUPRC on PTB-XL test set")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # annotate bars with values
    for i, v in enumerate(aurocs):
        ax.text(x[i] - width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(auprcs):
        ax.text(x[i] + width / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_per_class_auroc(metrics, model_defs, labels, out_path: Path):
    """Figure 15: grouped bar plot for per-class AUROC."""
    plt.style.use("default")
    model_keys = list(model_defs.keys())
    n_classes = len(labels)
    x = np.arange(n_classes)
    width = 0.35

    colors = ["#4C72B0", "#DD8452"]  # consistent with Figure 14

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, key in enumerate(model_keys):
        offset = (-0.5 + i) * width
        scores = metrics[key]["auroc_per_class"]
        ax.bar(
            x + offset,
            scores,
            width,
            label=model_defs[key]["pretty_name"],
            color=colors[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-class AUROC comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_single_class_roc(y_true, y_prob_dict, auroc_dict, model_defs, class_name, out_path):
    """Figure 16: ROC curves for a single class (e.g., MI)."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5, 5))

    colors = {
        "ecg": "#4C72B0",
        "mm": "#DD8452",
    }

    for key, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = auroc_dict[key]
        ax.plot(
            fpr,
            tpr,
            label=f"{model_defs[key]['pretty_name']} (AUROC={auroc:.3f})",
            linewidth=2,
            color=colors.get(key, None),
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves for {class_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_af_roc_pr(y_true_af, y_prob_af, out_path):
    """Figure 17: AF binary model ROC and PR curves in one figure."""
    plt.style.use("default")

    fpr, tpr, _ = roc_curve(y_true_af, y_prob_af)
    precision, recall, _ = precision_recall_curve(y_true_af, y_prob_af)

    auroc = roc_auc_score(y_true_af, y_prob_af)
    auprc = average_precision_score(y_true_af, y_prob_af)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # colors
    color_main = "#55A868"

    # ROC
    ax_roc = axes[0]
    ax_roc.plot(fpr, tpr, label=f"AF model (AUROC={auroc:.3f})", linewidth=2, color=color_main)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("AF ROC curve")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)

    # PR
    ax_pr = axes[1]
    ax_pr.plot(recall, precision, label=f"AF model (AUPRC={auprc:.3f})", linewidth=2, color=color_main)
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("AF Precision–Recall curve")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
