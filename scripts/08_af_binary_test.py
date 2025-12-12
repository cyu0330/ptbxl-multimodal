# scripts/07_af_binary_test.py

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.datasets.ptbxl_af import PTBXLAFDataset
from src.models.ecg_cnn import ECGCNN
from src.training.metrics import compute_metrics


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("[INFO] Running AF test script...")

    # load config and set seed
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {}).get("ecg", {})

    base_dir = data_cfg["base_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # test split
    test_ds = PTBXLAFDataset(
        base_dir=base_dir,
        split="test",
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    print("[AF] Test size:", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=False,
    )

    # AF model (same settings as training)
    model = ECGCNN(
        in_leads=model_cfg.get("in_leads", 12),
        feat_dim=model_cfg.get("feat_dim", 256),
        num_labels=1,  # binary output
    ).to(device)

    # load checkpoint
    ckpt_path = args.ckpt
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # evaluation loop
    import torch.nn.functional as F
    from tqdm import tqdm

    ys = []
    ps = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Eval-AF-Test", leave=False):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

            loss = F.binary_cross_entropy_with_logits(logits, y)

            prob = torch.sigmoid(logits)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())

            total_loss += loss.item() * x.size(0)

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)

    # compute metrics
    metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
    metrics["bce_loss"] = total_loss / len(test_loader.dataset)

    print(f"[AF][TEST] metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # save predictions
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    y_true_flat = y_true.reshape(-1)
    y_prob_flat = y_prob.reshape(-1)
    y_pred_flat = (y_prob_flat >= args.threshold).astype(int)

    df = pd.DataFrame({
        "y_true_AF": y_true_flat.astype(int),
        "y_prob_AF": y_prob_flat,
        "y_pred_AF": y_pred_flat,
    })
    df.to_csv(args.out_csv, index=False)

    print(f"[INFO] Saved AF test predictions to: {args.out_csv}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
