# scripts/06_ecg_baseline_test.py

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd

# 让 Python 能找到 src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.datasets.ptbxl import PTBXLDataset
from src.models.ecg_cnn import ECGCNN
from src.training.metrics import compute_metrics  # 和 loop.py 里的一致


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

    print("[DEBUG] 06_ecg_baseline_test.py is running...")

    # 1. 加载配置 & 设置随机种子（和训练脚本完全同风格）
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {}).get("ecg", {})
    classes = data_cfg.get("labels", ["MI", "STTC", "HYP", "CD", "NORM"])
    base_dir = data_cfg["base_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 2. Test 数据集 & DataLoader（和 train 脚本一致，只是 split='test'）
    test_ds = PTBXLDataset(
        base_dir=base_dir,
        split="test",
        classes=classes,
        normalize=data_cfg.get("normalize", "per_lead"),
    )
    print("[Baseline] test size =", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # 3. 构建模型（参数和 03_train_ecg_baseline.py 完全一致）
    model = ECGCNN(
        in_leads=model_cfg.get("in_leads", 12),
        feat_dim=model_cfg.get("feat_dim", 256),
        num_labels=len(classes),
    ).to(device)

    # 4. 加载 best checkpoint
    ckpt_path = args.ckpt
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    # 你训练时保存的是 {"model_state": ..., "classes": ...}
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded ckpt: {ckpt_path}")

    # 5. 在 TEST 上跑一遍，算 metrics + 收集预测
    import torch.nn.functional as F
    from tqdm import tqdm

    ys = []
    ps = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Eval-Test", leave=False):
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

    y_true = np.concatenate(ys, axis=0)   # (N, C)
    y_prob = np.concatenate(ps, axis=0)   # (N, C)

    # 计算 metrics（调用方式和 loop.eval_one_epoch 一样）
    metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
    metrics["bce_loss"] = total_loss / len(test_loader.dataset)

    print(f"[Baseline][TEST] metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 6. 保存 per-sample 预测到 CSV，方便后面和 04/05 合并
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df_dict = {}
    for i, name in enumerate(classes):
        df_dict[f"y_true_{name}"] = y_true[:, i].astype(int)
        df_dict[f"y_prob_{name}"] = y_prob[:, i]
        df_dict[f"y_pred_{name}"] = (y_prob[:, i] >= args.threshold).astype(int)

    df = pd.DataFrame(df_dict)
    df.to_csv(args.out_csv, index=False)

    print(f"[INFO] Saved baseline TEST preds to: {args.out_csv}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
