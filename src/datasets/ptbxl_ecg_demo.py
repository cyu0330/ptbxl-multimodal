# src/datasets/ptbxl_ecg_demo.py

import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

from src.utils.label_maps import load_metadata, build_label_matrix
from src.datasets.ptbxl import _is_valid_ecg   # reuse baseline ECG validity check


def _load_ecg(record_path: str) -> np.ndarray:
    """
    Load ECG signal from a PTB-XL record and return [12, T].
    """
    try:
        sig, meta = wfdb.rdsamp(record_path)  # sig: [T, n_leads]
    except Exception as e:
        raise RuntimeError(f"Failed to read ECG record {record_path}: {e}")

    sig = np.asarray(sig, dtype=np.float32)

    if sig.ndim != 2:
        raise RuntimeError(
            f"Unexpected ECG ndim for {record_path}: {sig.ndim}, expected 2"
        )

    T, n_leads = sig.shape
    if n_leads != 12:
        raise RuntimeError(
            f"Unexpected number of leads for {record_path}: {n_leads}, expected 12"
        )

    # [T, 12] -> [12, T]
    return sig.T


class PTBXLECGDemoDataset(Dataset):
    """
    PTB-XL dataset returning:
      - ECG waveform [12, T]
      - demographic features [age_norm, sex_id, height_norm, weight_norm, pacemaker]
      - multi-label targets for diagnostic superclasses
    """
    def __init__(
        self,
        base_dir: str,
        split: str,
        classes: List[str],
        normalize: str = "per_lead",
    ):
        super().__init__()
        self.base_dir = base_dir
        self.normalize = normalize
        self.classes = classes

        # 1) Load ptbxl_database + scp_statements
        df, scp = load_metadata(base_dir)

        # 2) Official stratified split
        if split == "test":
            df_split = df[df["strat_fold"] == 10].reset_index(drop=True)
        elif split == "val":
            df_split = df[df["strat_fold"] == 9].reset_index(drop=True)
        else:  # "train"
            df_split = df[df["strat_fold"] <= 8].reset_index(drop=True)

        num_before = len(df_split)

        # 3) Keep records with valid ECG files (same as PTBXLDataset)
        mask_valid = df_split["filename_hr"].apply(
            lambda rel: _is_valid_ecg(base_dir, rel)
        )
        df_split = df_split.loc[mask_valid].reset_index(drop=True)
        num_after_valid = len(df_split)

        # 4) Drop rows with missing age or sex (rare)
        mask_demo = df_split["age"].notna() & df_split["sex"].notna()
        df_split = df_split.loc[mask_demo].reset_index(drop=True)
        num_after_demo = len(df_split)

        print(
            f"[PTBXLECGDemoDataset] split={split} | "
            f"total={num_before} | valid_ecg={num_after_valid} | "
            f"after_drop_missing_age_sex={num_after_demo} | "
            f"dropped={num_before - num_after_demo}"
        )

        self.df = df_split

        # 5) Build multi-label matrix
        self.y = build_label_matrix(self.df, scp, classes)

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_ecg(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "per_lead":
            # x: [12, T]
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            return (x - mean) / std
        return x

    def _build_demo_vector(self, row) -> np.ndarray:
        """
        demo = [age_norm, sex_id, height_norm, weight_norm, pacemaker]
        """
        # age
        age = row.get("age", np.nan)
        try:
            age = float(age)
        except Exception:
            age = 0.0
        if (not np.isfinite(age)) or (age < 0):
            age = 0.0
        if age >= 300:  # anonymized >89
            age = 90.0
        age_norm = age / 100.0

        # sex
        sex = row.get("sex", "UNKNOWN")
        if sex == "M":
            sex_id = 0.0
        elif sex == "F":
            sex_id = 1.0
        else:
            sex_id = 0.5

        # height
        height = row.get("height", np.nan)
        try:
            height = float(height)
        except Exception:
            height = 0.0
        if (not np.isfinite(height)) or (height <= 0):
            height = 0.0
        height_norm = height / 250.0

        # weight
        weight = row.get("weight", np.nan)
        try:
            weight = float(weight)
        except Exception:
            weight = 0.0
        if (not np.isfinite(weight)) or (weight <= 0):
            weight = 0.0
        weight_norm = weight / 200.0

        # pacemaker
        pacemaker = row.get("pacemaker", 0)
        try:
            pacemaker_val = float(pacemaker)
        except Exception:
            pacemaker_val = 0.0
        if not np.isfinite(pacemaker_val):
            pacemaker_val = 0.0

        demo = np.array(
            [age_norm, sex_id, height_norm, weight_norm, pacemaker_val],
            dtype=np.float32,
        )
        return demo

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          x_ecg:  [12, T] float32 ECG waveform
          x_demo: [5]     float32 demographic vector
          y:      [num_labels] float32 multi-hot label vector
        """
        row = self.df.iloc[idx]

        # ECG
        rel_path = row["filename_hr"]
        record_path = os.path.join(self.base_dir, rel_path)
        x_ecg = _load_ecg(record_path)      # [12, T]
        x_ecg = self._normalize_ecg(x_ecg)

        # Demographic vector
        x_demo = self._build_demo_vector(row)   # [5]

        # Labels
        y = self.y[idx]                         # [num_labels]

        x_ecg_t = torch.from_numpy(x_ecg).float()
        x_demo_t = torch.from_numpy(x_demo).float()
        y_t = torch.from_numpy(y).float()

        return x_ecg_t, x_demo_t, y_t
