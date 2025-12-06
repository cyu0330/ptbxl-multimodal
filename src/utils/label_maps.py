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


def build_af_binary_labels(
    df: pd.DataFrame,
    scp: pd.DataFrame,
    af_description_keywords: List[str] = None,
) -> np.ndarray:
    """
    Build a binary label vector y ∈ {0,1} indicating presence of atrial fibrillation (AF)
    for each record in df.

    We detect AF-related SCP codes from scp_statements.csv by searching for
    keywords such as "atrial fibrillation" in the description column (case-insensitive).

    Args:
        df:  ptbxl_database dataframe (must contain 'scp_codes' column)
        scp: scp_statements dataframe (must contain 'scp_code' + 'description')
        af_description_keywords: optional list of lowercase substrings to match
                                 in the description column. If None, defaults
                                 to ["atrial fibrillation"].

    Returns:
        labels: np.ndarray of shape [len(df), 1], with 0/1 values:
                1 -> AF present, 0 -> non-AF (or AF not annotated).
    """
    if af_description_keywords is None:
        # 可以以后再加 "afib" 之类的 keyword，现在先用最干净的
        af_description_keywords = ["atrial fibrillation"]

    scp_local = scp.copy()

    if "description" not in scp_local.columns:
        raise KeyError("Column 'description' not found in scp_statements.csv")

    # 统一小写，方便匹配
    scp_local["description_lower"] = scp_local["description"].astype(str).str.lower()

    # 找到所有 description 里包含 AF 关键词的 scp_code
    mask = False
    for kw in af_description_keywords:
        mask = mask | scp_local["description_lower"].str.contains(kw, na=False)

    af_codes = scp_local.loc[mask, "scp_code"].tolist()

    if len(af_codes) == 0:
        print("[WARN] build_af_binary_labels: no AF-related SCP codes found "
              f"with keywords {af_description_keywords}.")
    else:
        print("[INFO] AF-related SCP codes detected:", af_codes)

    num_samples = len(df)
    labels = np.zeros((num_samples, 1), dtype=np.float32)

    # 遍历每条记录，看看它有没有 AF 相关的 scp_code
    for i, row in df.iterrows():
        scp_codes_str = row["scp_codes"]

        try:
            codes_dict = ast.literal_eval(scp_codes_str)
        except Exception:
            # parsing 失败就当作全 0
            continue

        if not isinstance(codes_dict, dict):
            continue

        codes = list(codes_dict.keys())

        # 如果该条 ECG 的任意 scp_code 属于 AF 相关 code，则标记为 1
        if any(code in af_codes for code in codes):
            labels[i, 0] = 1.0

    return labels
