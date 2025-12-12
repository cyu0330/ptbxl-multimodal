import os
import ast
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_metadata(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ptbxl_database.csv and scp_statements.csv.
    Ensures scp_statements contains a 'scp_code' column.
    """
    db_path = os.path.join(base_dir, "ptbxl_database.csv")
    scp_path = os.path.join(base_dir, "scp_statements.csv")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ptbxl_database.csv not found at: {db_path}")
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"scp_statements.csv not found at: {scp_path}")

    df = pd.read_csv(db_path)
    scp = pd.read_csv(scp_path)

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
    Build a multi-hot label matrix [N, C] for high-level diagnostic classes.

    Each sample may map to multiple diagnostic classes depending on the
    SCP codes listed in ptbxl_database.csv.
    """
    scp_map = scp.set_index("scp_code")

    if "diagnostic_class" not in scp_map.columns:
        raise KeyError("Column 'diagnostic_class' missing in scp_statements.csv.")

    code_to_class = scp_map["diagnostic_class"].to_dict()

    num_samples = len(df)
    num_classes = len(classes)
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)

    class_index = {cls: i for i, cls in enumerate(classes)}

    for i, row in df.iterrows():
        try:
            codes = ast.literal_eval(row["scp_codes"])
        except Exception:
            continue
        if not isinstance(codes, dict):
            continue

        for code in codes.keys():
            diag = code_to_class.get(code)
            if diag in class_index:
                labels[i, class_index[diag]] = 1.0

    return labels


def build_af_binary_labels(
    df: pd.DataFrame,
    scp: pd.DataFrame,
    keywords: List[str] = None,
) -> np.ndarray:
    """
    Build binary AF labels [N, 1] based on SCP descriptions.

    A record is labeled as AF if any of its SCP codes corresponds
    to an SCP entry whose description indicates atrial fibrillation.
    """
    if keywords is None:
        keywords = ["atrial fibrillation"]

    scp_local = scp.copy()
    if "description" not in scp_local.columns:
        raise KeyError("Column 'description' missing in scp_statements.csv.")

    scp_local["desc_lower"] = scp_local["description"].astype(str).str.lower()

    mask = False
    for kw in keywords:
        mask = mask | scp_local["desc_lower"].str.contains(kw, na=False)

    af_codes = scp_local.loc[mask, "scp_code"].tolist()

    num_samples = len(df)
    labels = np.zeros((num_samples, 1), dtype=np.float32)

    for i, row in df.iterrows():
        try:
            codes = ast.literal_eval(row["scp_codes"])
        except Exception:
            continue
        if not isinstance(codes, dict):
            continue

        if any(code in af_codes for code in codes.keys()):
            labels[i, 0] = 1.0

    return labels
