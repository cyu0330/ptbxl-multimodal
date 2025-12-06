import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.utils.seed import set_seed
from src.datasets.ptbxl import PTBXLDataset
from src.models.ecg_cnn import ECGCNN
from src.training.loop import train_one_epoch, eval_one_epoch

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

    # 2. 数据集 & DataLoader
    train_ds = PTBXLDataset(base_dir, split="train", classes=classes,
                            normalize=data_cfg.get("normalize", "per_lead"))
    val_ds = PTBXLDataset(base_dir, split="val", classes=classes,
                          normalize=data_cfg.get("normalize", "per_lead"))
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

    # 3. 模型 & 优化器
    model = ECGCNN(
        in_leads=model_cfg.get("in_leads", 12),
        feat_dim=model_cfg.get("feat_dim", 256),
        num_labels=len(classes),
    ).to(device)

    # 确保 lr 和 weight_decay 是 float（即使 yaml 里是字符串也没关系）
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


    # 4. 训练循环
    best_auprc = -1
    os.makedirs(os.path.join(log_cfg["out_dir"], "ckpts"), exist_ok=True)
    ckpt_path = os.path.join(log_cfg["out_dir"], "ckpts", "ecg_baseline_best.pth")

    for epoch in range(train_cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Train BCE: {train_loss:.4f}")

        val_metrics = eval_one_epoch(model, val_loader, device)
        print("Val metrics:", val_metrics)

        auprc = val_metrics.get("auprc_macro", -1)
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save({"model_state": model.state_dict(), "classes": classes}, ckpt_path)
            print(f"⭐ New best AUPRC {best_auprc:.4f}, saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args)
