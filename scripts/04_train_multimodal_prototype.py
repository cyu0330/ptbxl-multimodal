# scripts/04_train_multimodal_prototype.py

import argparse
import os
import csv
from datetime import datetime

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.utils.seed import set_seed
from src.datasets.ptbxl_ecg_multimodal import PTBXLECGMultimodalDataset
from src.models.ecg_multimodal import ECGMultimodal
from src.training.loop_demo import train_one_epoch_demo, eval_one_epoch_demo


def log_epoch_to_csv(
    csv_path,
    run_name,
    epoch,
    train_loss,
    val_metrics,
    ckpt_path,
    config_path,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
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
                ]
            )

        writer.writerow(
            [
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
            ]
        )


def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg_all = cfg.get("model", {})
    model_cfg = model_cfg_all.get("ecg_multimodal", model_cfg_all.get("ecg_demo", {}))
    log_cfg = cfg["log"]

    classes = data_cfg.get("labels", ["MI", "STTC", "HYP", "CD", "NORM"])
    base_dir = data_cfg["base_dir"]

    out_dir = log_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    base_run_name = log_cfg.get("run_name", "ecg_multimodal")

    # NOTE: keep run_name stable (no timestamp)
    run_name = base_run_name

    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # NOTE: overwrite metrics file for this run
    metrics_csv = os.path.join(log_dir, f"metrics_{run_name}.csv")

    print("[INFO] Using config:", args.config)
    print("[INFO] Classes:", classes)
    print("[INFO] Base dir:", base_dir)
    print("[INFO] Run name:", run_name)

    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    num_workers = int(train_cfg.get("num_workers", 4))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 1000))

    train_ds = PTBXLECGMultimodalDataset(
        base_dir=base_dir,
        split="train",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    val_ds = PTBXLECGMultimodalDataset(
        base_dir=base_dir,
        split="val",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    print("[ECG-MM] train size =", len(train_ds))
    print("[ECG-MM] val size   =", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = ECGMultimodal(
        num_labels=len(classes),
        ecg_feat_dim=model_cfg.get("ecg_feat_dim", 256),
        demo_hidden_dim=model_cfg.get(
            "demo_hidden_dim", model_cfg.get("demo_feat_dim", 64)
        ),
        in_leads=model_cfg.get("in_leads", 12),
    ).to(device)

    pretrained_ecg_ckpt = model_cfg.get("pretrained_ecg_ckpt", None)
    if pretrained_ecg_ckpt is not None and os.path.exists(pretrained_ecg_ckpt):
        print(f"[INFO] Loading pretrained ECG encoder from: {pretrained_ecg_ckpt}")
        ckpt = torch.load(pretrained_ecg_ckpt, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.ecg_backbone.load_state_dict(state, strict=False)
        print("[INFO] ECG encoder loaded. missing:", missing)
        print("[INFO] unexpected:", unexpected)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_auprc = -1.0
    epochs_no_improve = 0

    ckpt_dir = os.path.join(out_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # NOTE: fixed checkpoint name
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pth")

    print(f"[INFO] Best checkpoint will be saved to: {ckpt_path}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch_demo(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        print(f"Train-ECG-MM BCE: {train_loss:.4f}")

        val_metrics = eval_one_epoch_demo(
            model=model,
            loader=val_loader,
            device=device,
        )
        print("Val-ECG-MM metrics:", val_metrics)

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
            epochs_no_improve = 0
            torch.save(
                {"model_state": model.state_dict(), "classes": classes},
                ckpt_path,
            )
            print(f"[INFO] New best AUPRC {best_auprc:.4f}, saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("[INFO] Early stopping.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ecg_multimodal.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args)
