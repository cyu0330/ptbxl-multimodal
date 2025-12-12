# scripts/02_prepare_data.py

import argparse
import os
import pandas as pd


def main(base_dir: str) -> None:
    print(f"Base dir: {base_dir}")

    db_path = os.path.join(base_dir, "ptbxl_database.csv")
    scp_path = os.path.join(base_dir, "scp_statements.csv")

    # basic checks
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")

    # load metadata
    df = pd.read_csv(db_path)
    scp = pd.read_csv(scp_path)

    # rename first column (PTB-XL has unnamed first col sometimes)
    first_col = scp.columns[0]
    scp = scp.rename(columns={first_col: "scp_code"})

    print(f"\nLoaded ptbxl_database.csv: {len(df)} rows")
    print("Columns:", list(df.columns))

    print("\nstrat_fold distribution:")
    print(df["strat_fold"].value_counts().sort_index())

    print(f"\nLoaded scp_statements.csv: {len(scp)} rows")
    print("Columns:", list(scp.columns))

    # basic view of diagnostic class distribution
    if "diagnostic_class" in scp.columns:
        print("\nDiagnostic classes:")
        print(scp["diagnostic_class"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to PTB-XL 1.0.3 directory (contains ptbxl_database.csv)",
    )
    args = parser.parse_args()
    main(args.base_dir)
