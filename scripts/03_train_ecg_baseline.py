# scripts/03_train_ecg_baseline.py
#
# Train a single-modal ECG baseline model on PTB-XL.
# Saving logs, metrics CSV and the best checkpoint.

import argparse
import os
import csv
from datetime import datetime

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.utils.seed import set_seed
from src.datasets.ptbxl import PTBXLDataset
from src.models.ecg_cnn import ECGCNN
from src.training.loop import train_one_epoch, eval_one_epoch

os.environ["TORCH_CUDA_ARCH_LIST"] = "native"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device (script import):", device)


def log_epoch_to_csv(csv_path, run_name, epoch, train_loss, val_metrics, ckpt_path, config_path):
    """Append one line of metrics into a CSV. Create file with header if needed."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "datetime",
                "run_name",
                "epoch",
                "train_bce",
                "val_auroc_macro",
                "val_auprc_macro",
                "val_f1_macro",
                "val_bce_loss",
                "ckpt_path",
                "config_path",
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_name,
            epoch,
            float(train_loss),
            float(val_metrics.get("auroc_macro", -1)),
            float(val_metrics.get("auprc_macro", -1)),
            float(val_metrics.get("f1_macro", -1)),
            float(val_metrics.get("bce_loss", -1)),
            ckpt_path,
            config_path,
        ])


def main(args):
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {}).get("ecg", {})
    log_cfg = cfg["log"]

    classes = data_cfg.get("labels", ["MI", "STTC", "HYP", "CD", "NORM"])
    base_dir = data_cfg["base_dir"]

    # Output structure
    root_out = log_cfg.get("out_dir", "outputs")
    run_name = log_cfg.get("run_name", "ecg_baseline")

    out_dir = os.path.join(root_out, run_name)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    metrics_csv = os.path.join(log_dir, "metrics_ecg_baseline.csv")

    print("[INFO] Using config:", args.config)
    print("[INFO] Output dir:", out_dir)
    print("[INFO] Metrics CSV:", metrics_csv)

    # Dataset & Loader
    train_ds = PTBXLDataset(
        base_dir,
        split="train",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    val_ds = PTBXLDataset(
        base_dir,
        split="val",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    print("[Baseline] train size =", len(train_ds))
    print("[Baseline] val size   =", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device (training):", device)

    # Model
    model = ECGCNN(
        in_leads=model_cfg.get("in_leads", 12),
        feat_dim=model_cfg.get("feat_dim", 256),
        num_labels=len(classes),
    ).to(device)

    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 0.0))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Training
    best_auprc = -1
    ckpt_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ecg_baseline_best.pth")

    print("[INFO] Checkpoints ->", ckpt_path)

    for epoch in range(train_cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train BCE: {train_loss:.4f}")

        val_metrics = eval_one_epoch(model, val_loader, device)
        print("Val metrics:", val_metrics)

        # Save logs
        log_epoch_to_csv(
            metrics_csv,
            run_name,
            epoch + 1,
            train_loss,
            val_metrics,
            ckpt_path,
            args.config,
        )

        # Save best checkpoint
        auprc = float(val_metrics.get("auprc_macro", -1))
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save({"model_state": model.state_dict(), "classes": classes}, ckpt_path)
            print(f"★ New best AUPRC: {best_auprc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ecg_baseline.yaml",
    )
    args = parser.parse_args()
    main(args)
