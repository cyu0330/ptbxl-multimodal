# scripts/09_merge_all_test.py

import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline_csv",
        type=str,
        default="outputs/ecg_baseline/preds/ecg_baseline_test_preds.csv",
    )
    parser.add_argument(
        "--multimodal_csv",
        type=str,
        default="outputs/ecg_multimodal/preds/ecg_multimodal_test_preds.csv",
    )
    parser.add_argument(
        "--af_csv",
        type=str,
        default="outputs/af_binary/preds/af_binary_test_preds.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="outputs/merged/test_03_04_05_merged.csv",
    )

    args = parser.parse_args()

    print("[INFO] Loading baseline:", args.baseline_csv)
    df_base = pd.read_csv(args.baseline_csv)

    print("[INFO] Loading multimodal:", args.multimodal_csv)
    df_mm = pd.read_csv(args.multimodal_csv)

    print("[INFO] Loading AF:", args.af_csv)
    df_af = pd.read_csv(args.af_csv)

    # Check row count
    n = len(df_base)
    if len(df_mm) != n or len(df_af) != n:
        raise ValueError(
            f"Row count mismatch: baseline={len(df_base)}, multimodal={len(df_mm)}, AF={len(df_af)}"
        )

    # Remove duplicated ground-truth columns (keep baseline GT only)
    mm_cols = [c for c in df_mm.columns if not c.startswith("y_true_")]
    df_mm = df_mm[mm_cols]

    # Merge results
    df_merged = pd.concat([df_base, df_mm, df_af], axis=1)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_merged.to_csv(args.out_csv, index=False)

    print("[INFO] Saved merged CSV to:", args.out_csv)
    print("[INFO] merged shape:", df_merged.shape)


if __name__ == "__main__":
    main()
