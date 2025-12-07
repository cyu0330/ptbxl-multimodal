# scripts/06_af_binary_test.py

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
from src.datasets.ptbxl_af import PTBXLAFDataset
from src.models.ecg_cnn import ECGCNN
from src.training.metrics import compute_metrics  # 和 loop.py 里用的是同一个


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

    print("[DEBUG] 06_af_binary_test.py is running...")

    # 1. 加载配置 & seed（风格和 05_train_af_binary 一致）
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg.get("model", {}).get("ecg", {})

    base_dir = data_cfg["base_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 2. TEST 数据集 & DataLoader：用 split="test"
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

    # 3. 构建 AF 模型：完全照 05_train_af_binary.py
    model = ECGCNN(
        in_leads=model_cfg.get("in_leads", 12),
        feat_dim=model_cfg.get("feat_dim", 256),
        num_labels=1,  # 二分类
    ).to(device)

    # 4. 加载最佳 ckpt
    ckpt_path = args.ckpt
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location=device)
    # 训练时保存的是 {"model_state": ...}
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded AF ckpt: {ckpt_path}")

    # 5. 在 TEST 上评估：算 metrics + 收集预测
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
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out  # (B, 1)

            loss = F.binary_cross_entropy_with_logits(logits, y)

            prob = torch.sigmoid(logits)      # (B, 1)
            ys.append(y.cpu().numpy())
            ps.append(prob.cpu().numpy())
            total_loss += loss.item() * x.size(0)

    y_true = np.concatenate(ys, axis=0)   # (N, 1)
    y_prob = np.concatenate(ps, axis=0)   # (N, 1)

    # 计算 metrics：调用方式和 loop.eval_one_epoch 完全一致
    metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
    metrics["bce_loss"] = total_loss / len(test_loader.dataset)

    print(f"[AF][TEST] metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 6. 保存 per-sample 预测 CSV，后面要和 03/04 合并
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # 展平成一维方便看
    y_true_flat = y_true.reshape(-1)
    y_prob_flat = y_prob.reshape(-1)
    y_pred_flat = (y_prob_flat >= args.threshold).astype(int)

    df = pd.DataFrame({
        "y_true_AF": y_true_flat.astype(int),
        "y_prob_AF": y_prob_flat,
        "y_pred_AF": y_pred_flat,
    })
    df.to_csv(args.out_csv, index=False)

    print(f"[INFO] Saved AF TEST preds to: {args.out_csv}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
