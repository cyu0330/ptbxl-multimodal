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
        "--demo_csv",
        type=str,
        default="outputs/ecg_demo/preds/ecg_demo_test_preds.csv",
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

    print("[INFO] Loading demo:", args.demo_csv)
    df_demo = pd.read_csv(args.demo_csv)

    print("[INFO] Loading AF:", args.af_csv)
    df_af = pd.read_csv(args.af_csv)

    # 检查行数是否一致
    n = len(df_base)
    if len(df_demo) != n or len(df_af) != n:
        raise ValueError(
            f"Row count mismatch: baseline={len(df_base)}, demo={len(df_demo)}, AF={len(df_af)}"
        )

    # 删除 demo 中的 y_true_* 列（保留 baseline 的 ground truth）
    demo_cols = [c for c in df_demo.columns if not c.startswith("y_true_")]
    df_demo = df_demo[demo_cols]

    # 合并三组预测
    df_merged = pd.concat([df_base, df_demo, df_af], axis=1)

    # 保存
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_merged.to_csv(args.out_csv, index=False)

    print("[INFO] Saved merged CSV to:", args.out_csv)
    print("[INFO] merged shape:", df_merged.shape)


if __name__ == "__main__":
    main()
