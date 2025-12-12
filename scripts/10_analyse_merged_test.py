# scripts/10_analyse_merged_test.py

import os
import sys
import pandas as pd
import numpy as np

# allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.metrics import compute_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged_csv",
        type=str,
        default="outputs/merged/test_03_04_05_merged.csv",
        help="Merged prediction file from baseline, multimodal and AF models.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used for computing F1 and other metrics.",
    )
    args = parser.parse_args()

    print("[INFO] Loading merged CSV:", args.merged_csv)
    df = pd.read_csv(args.merged_csv)
    print("[INFO] merged shape:", df.shape)

    # high-level diagnostic classes
    ecg_labels = ["CD", "HYP", "MI", "NORM", "STTC"]
    print("[INFO] ECG labels:", ecg_labels)

    # ----- baseline ECG metrics -----
    y_true = df[[f"y_true_{lbl}" for lbl in ecg_labels]].values.astype(np.float32)
    y_prob_base = df[[f"y_prob_{lbl}" for lbl in ecg_labels]].values.astype(np.float32)

    print("\n[Baseline ECG][TEST] metrics:")
    metrics_base = compute_metrics(y_true, y_prob_base, threshold=args.threshold)
    for k, v in metrics_base.items():
        print(f"  {k}: {v}")

    # ----- ECG + demographics metrics (multimodal) -----
    if all(f"y_prob_{lbl}_mm" in df.columns for lbl in ecg_labels):
        y_prob_mm = df[[f"y_prob_{lbl}_mm" for lbl in ecg_labels]].values.astype(
            np.float32
        )

        print("\n[ECG + demographics][TEST] metrics:")
        metrics_mm = compute_metrics(y_true, y_prob_mm, threshold=args.threshold)
        for k, v in metrics_mm.items():
            print(f"  {k}: {v}")
    else:
        print("\n[WARN] Multimodal columns not found; skip ECG+demographics metrics.")

    # ----- AF binary metrics -----
    if "y_true_AF" in df.columns and "y_prob_AF" in df.columns:
        y_true_af = df["y_true_AF"].values.astype(np.float32).reshape(-1, 1)
        y_prob_af = df["y_prob_AF"].values.astype(np.float32).reshape(-1, 1)

        print("\n[AF binary][TEST] metrics:")
        metrics_af = compute_metrics(y_true_af, y_prob_af, threshold=args.threshold)
        for k, v in metrics_af.items():
            print(f"  {k}: {v}")
    else:
        print("\n[WARN] AF columns not found in merged CSV.")


if __name__ == "__main__":
    main()
