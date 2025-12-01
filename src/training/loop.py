from typing import Dict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from src.training.metrics import compute_metrics


def train_one_epoch(model, loader: DataLoader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)
        # 兼容：
        # - logits
        # - (logits, features)
        # - (logits, features, ...)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader: DataLoader, device) -> Dict[str, float]:
    model.eval()
    ys = []
    ps = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            loss = F.binary_cross_entropy_with_logits(logits, y)

            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
            total_loss += loss.item() * x.size(0)

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)

    # 你原来的 compute_metrics 调用保持不变
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics["bce_loss"] = total_loss / len(loader.dataset)
    return metrics
