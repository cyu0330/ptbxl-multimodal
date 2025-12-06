import argparse
import yaml
import os
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.utils.seed import set_seed
from src.datasets.ptbxl import PTBXLDataset
from src.models.ecg_cnn import ECGCNN
from src.training.loop import train_one_epoch, eval_one_epoch


def log_epoch_to_csv(csv_path, run_name, epoch, train_loss, val_metrics, ckpt_path, config_path):
    """
    Append one line of metrics into a CSV file.
    If the file doesn't exist, write a header first.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
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
    # 1. 读取配置
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {}).get("ecg", {})
    log_cfg = cfg["log"]

    classes = data_cfg.get("labels", ["MI", "STTC", "HYP", "CD", "NORM"])
    base_dir = data_cfg["base_dir"]

    # out_dir / logs / metrics
    out_dir = log_cfg["out_dir"]
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    metrics_csv = os.path.join(log_dir, "metrics_ecg_baseline.csv")
    run_name = log_cfg.get("run_name", "ecg_baseline")

    # 2. 数据集 & DataLoader
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
        num_workers=4,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # 3. 模型 & 优化器
    model = ECGCNN(
        in_leads=model_cfg.get("in_leads", 12),
        feat_dim=model_cfg.get("feat_dim", 256),
        num_labels=len(classes),
    ).to(device)

    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # 4. 训练循环
    best_auprc = -1.0
    ckpt_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ecg_baseline_best.pth")
    print(f"[INFO] Checkpoints will be saved to: {ckpt_path}")

    for epoch in range(train_cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train BCE: {train_loss:.4f}")

        val_metrics = eval_one_epoch(model, val_loader, device)
        print("Val metrics:", val_metrics)

        # 写入 CSV
        log_epoch_to_csv(
            csv_path=metrics_csv,
            run_name=run_name,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_metrics=val_metrics,
            ckpt_path=ckpt_path,
            config_path=args.config,
        )

        auprc = float(val_metrics.get("auprc_macro", -1))
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save(
                {"model_state": model.state_dict(), "classes": classes},
                ckpt_path,
            )
            print(f"⭐ New best AUPRC {best_auprc:.4f}, saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ecg_baseline.yaml")
    args = parser.parse_args()
    main(args)
