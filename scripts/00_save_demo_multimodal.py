# scripts/00_save_demo_multimodal.py
#
# Export a few ECG+Demo samples from PTB-XL test split
# so that we can run ECG+Demo inference without PTB-XL.

import os
import argparse
import numpy as np

from src.utils.seed import set_seed
from src.datasets.ptbxl_ecg_demo import PTBXLECGDemoDataset


def main(args):
    set_seed(42)

    classes = args.classes.split(",") if args.classes else ["MI", "STTC", "HYP", "CD", "NORM"]

    ds = PTBXLECGDemoDataset(
        base_dir=args.base_dir,
        split="test",
        classes=classes,
        normalize="per_lead",
    )
    print(f"[INFO] PTBXLECGDemoDataset(test) size = {len(ds)}")

    os.makedirs(args.out_dir, exist_ok=True)

    n = min(args.num_samples, len(ds))
    for i in range(n):
        x_ecg, x_demo, y = ds[i]   # x_ecg:[12,T], x_demo:[5], y:[C]
        ecg_np = x_ecg.numpy()
        demo_np = x_demo.numpy()
        y_np = y.numpy()

        ecg_path = os.path.join(args.out_dir, f"demo_mm_ecg_{i}.npy")
        demo_path = os.path.join(args.out_dir, f"demo_mm_demo_{i}.npy")

        np.save(ecg_path, ecg_np)
        np.save(demo_path, demo_np)

        print(f"[SAVE] multimodal demo #{i}:")
        print(f"       ECG  -> {ecg_path}  shape={ecg_np.shape}")
        print(f"       DEMO -> {demo_path} shape={demo_np.shape}  y={y_np}")

    print("[DONE] All multimodal demos saved.")


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
        default=1,
        help="How many multimodal samples to export.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="MI,STTC,HYP,CD,NORM",
        help="Class list (comma-separated).",
    )
    args = parser.parse_args()
    main(args)
