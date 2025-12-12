# scripts/00_save_demo_ecg.py
#
# Export a few PTB-XL ECG samples as .npy files.
# These small files are only used for quick demonstrations.

import os
import argparse
import numpy as np
import torch

from src.utils.seed import set_seed
from src.datasets.ptbxl import PTBXLDataset


def main(args):
    set_seed(42)

    classes = args.classes.split(",") if args.classes else ["MI", "STTC", "HYP", "CD", "NORM"]

    # Load test split from PTB-XL
    ds = PTBXLDataset(
        base_dir=args.base_dir,
        split="test",
        classes=classes,
        normalize="per_lead",
    )
    print(f"[INFO] PTBXLDataset(test) size = {len(ds)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Save a few ECG samples
    n = min(args.num_samples, len(ds))
    for i in range(n):
        x, y = ds[i]        # x: [12, T], y: [C]
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
        help="PTB-XL base directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/demo",
        help="Directory to save demo npy files.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of ECG files to export.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="MI,STTC,HYP,CD,NORM",
        help="Class list (comma-separated).",
    )
    args = parser.parse_args()
    main(args)
