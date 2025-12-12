import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """
    Compute macro-level AUROC, AUPRC, and F1 for multi-label classification.

    Args:
        y_true: array of shape [N, L], ground-truth binary labels
        y_prob: array of shape [N, L], predicted probabilities
        threshold: threshold for converting probabilities to binary predictions

    Returns:
        dict with keys:
            - auroc_macro
            - auprc_macro
            - f1_macro
    """
    metrics = {}

    # AUROC (macro)
    try:
        metrics["auroc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["auroc_macro"] = float("nan")

    # AUPRC (macro)
    try:
        metrics["auprc_macro"] = average_precision_score(
            y_true, y_prob, average="macro"
        )
    except ValueError:
        metrics["auprc_macro"] = float("nan")

    # F1 (macro)
    y_pred = (y_prob >= threshold).astype(int)
    metrics["f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    return metrics
