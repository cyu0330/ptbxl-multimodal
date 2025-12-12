import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm

from src.utils.seed import set_seed
from src.datasets.ptbxl_ecg_multimodal import PTBXLECGMultimodalDataset
from src.models.ecg_multimodal import ECGMultimodal
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

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg_all = cfg.get("model", {})
    model_cfg = model_cfg_all.get("ecg_multimodal", model_cfg_all.get("ecg_demo", {}))

    classes = data_cfg.get("labels", ["MI", "STTC", "HYP", "CD", "NORM"])
    base_dir = data_cfg["base_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # test fold = 10
    test_ds = PTBXLECGMultimodalDataset(
        base_dir=base_dir,
        split="test",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    print("[ECG-MM] test size =", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=False,
    )

   
    model = ECGMultimodal(
        num_labels=len(classes),
        ecg_feat_dim=model_cfg.get("ecg_feat_dim", 256),
        demo_hidden_dim=model_cfg.get("demo_hidden_dim", model_cfg.get("demo_feat_dim", 64)),
        in_leads=model_cfg.get("in_leads", 12),
    ).to(device)

    ckpt_path = args.ckpt
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded ECG-MM checkpoint: {ckpt_path}")

    bce_loss_fn = BCEWithLogitsLoss()

    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Test-ECG-MM", leave=False)
        for x_ecg, x_demo, y in pbar:
            x_ecg = x_ecg.to(device)
            x_demo = x_demo.to(device)
            y = y.to(device)

            logits = model(x_ecg, x_demo)
            loss = bce_loss_fn(logits, y)

            total_loss += loss.item()
            n_batches += 1

            prob = torch.sigmoid(logits)
            all_probs.append(prob.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / max(1, n_batches)

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
    metrics["bce_loss"] = float(avg_loss)

    print("[ECG-MM][TEST] metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df_dict = {}
    for i, cls in enumerate(classes):
        df_dict[f"y_true_{cls}"] = y_true[:, i].astype(int)
        df_dict[f"y_prob_{cls}_mm"] = y_prob[:, i]
        df_dict[f"y_pred_{cls}_mm"] = (y_prob[:, i] >= args.threshold).astype(int)

    df = pd.DataFrame(df_dict)
    df.to_csv(args.out_csv, index=False)

    print(f"[INFO] Saved ECG-MM test predictions to: {args.out_csv}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
