import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    y_true, y_prob: [N, L]
    """
    metrics = {}
    # 防止全是一个类时崩溃
    try:
        metrics["auroc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["auroc_macro"] = np.nan

    try:
        metrics["auprc_macro"] = average_precision_score(y_true, y_prob, average="macro")
    except ValueError:
        metrics["auprc_macro"] = np.nan

    y_pred = (y_prob >= threshold).astype(int)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return metrics
