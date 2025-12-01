# src/utils/label_maps.py

import os
import ast
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_metadata(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ptbxl_database.csv and scp_statements.csv,
    and make sure scp_statements has a 'scp_code' column.
    """
    db_path = os.path.join(base_dir, "ptbxl_database.csv")
    scp_path = os.path.join(base_dir, "scp_statements.csv")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at {db_path}")
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"scp_statements.csv not found at {scp_path}")

    df = pd.read_csv(db_path)
    scp = pd.read_csv(scp_path)

    # Ensure first column is called 'scp_code'
    first_col = scp.columns[0]
    if first_col != "scp_code":
        scp = scp.rename(columns={first_col: "scp_code"})

    return df, scp


def build_label_matrix(
    df: pd.DataFrame,
    scp: pd.DataFrame,
    classes: List[str],
) -> np.ndarray:
    """
    Build a multi-hot label matrix of shape [N, num_classes] using
    diagnostic_class information from scp_statements.

    Args:
        df: ptbxl_database dataframe (must contain 'scp_codes' column)
        scp: scp_statements dataframe (must contain 'scp_code' + 'diagnostic_class')
        classes: list of high-level diagnostic classes to keep, e.g.
                 ["MI", "STTC", "HYP", "CD", "NORM"]

    Returns:
        labels: np.ndarray of shape [len(df), len(classes)], with 0/1 values.
    """
    # Map scp_code -> diagnostic_class
    scp = scp.set_index("scp_code")
    if "diagnostic_class" not in scp.columns:
        raise KeyError("Column 'diagnostic_class' not found in scp_statements.")

    code_to_class = scp["diagnostic_class"].to_dict()

    num_samples = len(df)
    num_classes = len(classes)
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    class_idx = {c: i for i, c in enumerate(classes)}

    for i, row in df.iterrows():
        scp_codes_str = row["scp_codes"]
        try:
            codes_dict = ast.literal_eval(scp_codes_str)
        except Exception:
            # If parsing fails, skip (leave all-zero row)
            continue

        if not isinstance(codes_dict, dict):
            continue

        for code in codes_dict.keys():
            if code not in code_to_class:
                continue
            diag_class = code_to_class[code]
            if diag_class in class_idx:
                labels[i, class_idx[diag_class]] = 1.0

    return labels
