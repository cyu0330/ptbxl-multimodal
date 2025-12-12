# src/training/loop.py

from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from src.training.metrics import compute_metrics


def train_one_epoch(model, loader: DataLoader, optimizer, device) -> float:
    """
    Standard training loop for single-input ECG models.
    Outputs only a BCE loss.
    """
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)
        # Support both logits and (logits, features, ...)
        logits = out[0] if isinstance(out, tuple) else out

        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader: DataLoader, device) -> Dict[str, float]:
    """
    Standard evaluation loop for ECG baseline models.
    Returns BCE + threshold-based metrics.
    """
    model.eval()

    all_targets = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

            loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item() * x.size(0)

            prob = torch.sigmoid(logits)
            all_targets.append(y.cpu().numpy())
            all_probs.append(prob.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics["bce_loss"] = total_loss / len(loader.dataset)

    return metrics
