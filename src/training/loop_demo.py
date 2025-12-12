# src/training/loop_demo.py

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from src.training.metrics import compute_metrics

bce_loss_fn = BCEWithLogitsLoss()


def train_one_epoch_demo(model, loader, optimizer, device):
    """
    Training loop for the ECG + demographic model.
    Computes a single BCE loss on all output labels.
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Train-ECG+Demo", leave=False)

    for x_ecg, x_demo, y in pbar:
        x_ecg = x_ecg.to(device)
        x_demo = x_demo.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x_ecg, x_demo)
        loss = bce_loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix(loss=loss.item())

    return total_loss / max(1, num_batches)


def eval_one_epoch_demo(model, loader, device):
    """
    Evaluation loop for the ECG + demographic model.
    Returns BCE loss and per-label metrics.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val-ECG+Demo", leave=False)

        for x_ecg, x_demo, y in pbar:
            x_ecg = x_ecg.to(device)
            x_demo = x_demo.to(device)
            y = y.to(device)

            logits = model(x_ecg, x_demo)
            loss = bce_loss_fn(logits, y)

            total_loss += loss.item()
            num_batches += 1

            prob = torch.sigmoid(logits)
            all_probs.append(prob.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / max(1, num_batches)

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_prob)
    metrics["bce_loss"] = float(avg_loss)

    return metrics
