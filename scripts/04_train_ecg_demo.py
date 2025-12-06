# scripts/04_train_ecg_demo.py

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import yaml

from src.utils.seed import set_seed
from src.datasets.ptbxl_ecg_demo import PTBXLECGDemoDataset
from src.models.ecg_demo import ECGDemoModel
from src.training.loop_demo import train_one_epoch_demo, eval_one_epoch_demo


def main(args):
    print(f"[INFO] Using config: {args.config}")

    # 1. load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. set seed
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {}).get("ecg_demo", {})
    log_cfg = cfg["log"]

    classes = data_cfg.get("labels", ["MI", "STTC", "HYP", "CD", "NORM"])
    base_dir = data_cfg["base_dir"]

    print("[INFO] Classes:", classes)
    print("[INFO] Base dir:", base_dir)

    # 3. datasets
    train_ds = PTBXLECGDemoDataset(
        base_dir=base_dir,
        split="train",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    val_ds = PTBXLECGDemoDataset(
        base_dir=base_dir,
        split="val",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    print("[ECG+Demo] train size =", len(train_ds))
    print("[ECG+Demo] val size   =", len(val_ds))


    # 4. dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 5. model
    model = ECGDemoModel(
        num_labels=len(classes),
        ecg_feat_dim=model_cfg.get("ecg_feat_dim", 256),
        demo_feat_dim=model_cfg.get("demo_feat_dim", 64),
        in_leads=model_cfg.get("in_leads", 12),
    ).to(device)

    # 6. optimizer
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 7. training loop
    best_auprc = -1.0
    out_dir = log_cfg["out_dir"]
    ckpt_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ecg_demo_best.pth")

    print(f"[INFO] Checkpoints will be saved to: {ckpt_path}")

    for epoch in range(train_cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")
        train_loss = train_one_epoch_demo(model, train_loader, optimizer, device)
        print(f"Train-ECG+Demo BCE: {train_loss:.4f}")

        val_metrics = eval_one_epoch_demo(model, val_loader, device)
        print("Val-ECG+Demo metrics:", val_metrics)

        auprc = float(val_metrics.get("auprc_macro", -1))
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save(
                {"model_state": model.state_dict(), "classes": classes},
                ckpt_path,
            )
            print(f"⭐ New best ECG+Demo AUPRC {best_auprc:.4f}, saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/ecg_demo.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args)
