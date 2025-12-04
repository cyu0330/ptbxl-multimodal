# src/training/loop_demo.py

from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from src.training.metrics import compute_metrics


def train_one_epoch_demo(model, loader: DataLoader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for x_ecg, x_demo, y in tqdm(loader, desc="Train-ECG+Demo", leave=False):
        x_ecg = x_ecg.to(device)    # [B, 12, T]
        x_demo = x_demo.to(device)  # [B, 5]
        y = y.to(device)            # [B, C]

        optimizer.zero_grad()

        logits = model(x_ecg, x_demo)              # [B, C]
        loss = F.binary_cross_entropy_with_logits(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_ecg.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch_demo(model, loader: DataLoader, device) -> Dict[str, float]:
    model.eval()
    ys = []
    ps = []
    total_loss = 0.0

    with torch.no_grad():
        for x_ecg, x_demo, y in tqdm(loader, desc="Eval-ECG+Demo", leave=False):
            x_ecg = x_ecg.to(device)
            x_demo = x_demo.to(device)
            y = y.to(device)

            logits = model(x_ecg, x_demo)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
            total_loss += loss.item() * x_ecg.size(0)

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics["bce_loss"] = total_loss / len(loader.dataset)
    return metrics
