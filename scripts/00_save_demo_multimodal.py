# scripts/00_save_demo_multimodal.py
#
# Export a few ECG + demographic samples from the PTB-XL test split.
# These small .npy files allow multimodal inference without requiring PTB-XL.

import os
import argparse
import numpy as np

from src.utils.seed import set_seed
from datasets.ptbxl_ecg_multimodal import PTBXLECGMultimodalDataset


def main(args):
    set_seed(42)

    # class list used by the dataset
    classes = args.classes.split(",") if args.classes else ["MI", "STTC", "HYP", "CD", "NORM"]

    # load PTB-XL test split
    ds = PTBXLECGMultimodalDataset(
        base_dir=args.base_dir,
        split="test",
        classes=classes,
        normalize="per_lead",
    )
    print(f"[INFO] PTBXLECGMultimodalDataset(test) size = {len(ds)}")

    os.makedirs(args.out_dir, exist_ok=True)

    n = min(args.num_samples, len(ds))
    for i in range(n):
        # x_ecg: [12, T], x_demo: [5], y: [C]
        x_ecg, x_demo, y = ds[i]

        ecg_np = x_ecg.numpy()
        demo_np = x_demo.numpy()
        y_np = y.numpy()

        ecg_path = os.path.join(args.out_dir, f"demo_mm_ecg_{i}.npy")
        demo_path = os.path.join(args.out_dir, f"demo_mm_demo_{i}.npy")

        np.save(ecg_path, ecg_np)
        np.save(demo_path, demo_np)

        print(f"[SAVE] multimodal sample #{i}:")
        print(f"       ECG  -> {ecg_path}  shape={ecg_np.shape}")
        print(f"       DEMO -> {demo_path} shape={demo_np.shape}  y={y_np}")

    print("[DONE] Multimodal demo samples exported.")


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
        help="Directory to save demo files.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of multimodal samples to export.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="MI,STTC,HYP,CD,NORM",
        help="Comma-separated class list.",
    )

    args = parser.parse_args()
    main(args)
