# scripts/00_save_demo_ecg.py
#
# Run once on your own machine (with PTB-XL available).
# It will export a few ECG samples from PTB-XL test split into small .npy files:
#   data/demo/demo_ecg_0.npy, demo_ecg_1.npy, ...

import os
import argparse
import numpy as np
import torch

from src.utils.seed import set_seed
from src.datasets.ptbxl import PTBXLDataset


def main(args):
    set_seed(42)

    classes = args.classes.split(",") if args.classes else ["MI", "STTC", "HYP", "CD", "NORM"]

    ds = PTBXLDataset(
        base_dir=args.base_dir,
        split="test",
        classes=classes,
        normalize="per_lead",
    )
    print(f"[INFO] PTBXLDataset(test) size = {len(ds)}")

    os.makedirs(args.out_dir, exist_ok=True)

    n = min(args.num_samples, len(ds))
    for i in range(n):
        x, y = ds[i]          # x: [12, T], y: [C]
        x_np = x.numpy()
        y_np = y.numpy()

        save_path = os.path.join(args.out_dir, f"demo_ecg_{i}.npy")
        np.save(save_path, x_np)

        print(f"[SAVE] demo ECG #{i} -> {save_path} | y = {y_np}")

    print("[DONE] All demo ECG saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="PTB-XL base dir, e.g. C:/Users/Administrator/Desktop/ptb-xl/1.0.3",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/demo",
        help="Where to save demo npy files.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="How many ECG samples to export.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="MI,STTC,HYP,CD,NORM",
        help="Class list used in PTBXLDataset (comma-separated).",
    )
    args = parser.parse_args()
    main(args)
