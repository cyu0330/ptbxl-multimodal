# scripts/05_train_af_binary.py

import argparse
import os
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import yaml

from src.utils.seed import set_seed
from src.datasets.ptbxl_af import PTBXLAFDataset
from src.models.ecg_cnn import ECGCNN
from src.training.loop import train_one_epoch, eval_one_epoch

import os
import torch

os.environ["TORCH_CUDA_ARCH_LIST"] = "native"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", device)

def log_epoch_to_csv(csv_path, run_name, epoch, train_loss, val_metrics, ckpt_path, config_path):
    """
    把每个 epoch 的指标写入 CSV。
    首次运行会自动写入表头。
    """
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
    # 1. 加载配置
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {}).get("ecg", {})
    log_cfg = cfg["log"]

    base_dir = data_cfg["base_dir"]

    # === 输出目录 ===
    out_dir = log_cfg["out_dir"]
    log_dir = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    metrics_csv = os.path.join(log_dir, "metrics_af_binary.csv")
    ckpt_path = os.path.join(ckpt_dir, "af_binary_best.pth")
    run_name = log_cfg.get("run_name", "af_binary")

    print(f"[INFO] Metrics CSV: {metrics_csv}")
    print(f"[INFO] Best checkpoint: {ckpt_path}")

    # 2. 数据集
    train_ds = PTBXLAFDataset(
        base_dir=base_dir,
        split="train",
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    val_ds = PTBXLAFDataset(
        base_dir=base_dir,
        split="val",
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    print("[AF] Train size:", len(train_ds))
    print("[AF] Val size:", len(val_ds))

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
    print("[INFO] Device:", device)

    # 3. 模型：num_labels=1
    model = ECGCNN(
        in_leads=model_cfg.get("in_leads", 12),
        feat_dim=model_cfg.get("feat_dim", 256),
        num_labels=1,   # 二分类（AF vs 非 AF）
    ).to(device)

    # 4. 优化器
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. 训练循环
    best_auprc = -1.0

    for epoch in range(train_cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train-AF BCE: {train_loss:.4f}")

        val_metrics = eval_one_epoch(model, val_loader, device)
        print("Val-AF metrics:", val_metrics)

        # === 写入 CSV ===
        log_epoch_to_csv(
            csv_path=metrics_csv,
            run_name=run_name,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_metrics=val_metrics,
            ckpt_path=ckpt_path,
            config_path=args.config,
        )

        # === 保存最佳 ===
        auprc = float(val_metrics.get("auprc_macro", -1))
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            print(f"⭐ New best AF AUPRC: {best_auprc:.4f}, saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/af_binary.yaml",
        help="Path to YAML config file."
    )
    args = parser.parse_args()
    main(args)
