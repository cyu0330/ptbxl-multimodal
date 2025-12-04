# src/datasets/ptbxl_ecg_demo.py

import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

from src.utils.label_maps import load_metadata, build_label_matrix


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
    PTB-XL dataset that returns:
      - ECG waveform [12, T]
      - demographic features [age_norm, sex_id, height_norm, weight_norm, pacemaker]
      - multi-label targets for diagnostic superclasses (e.g. MI/STTC/HYP/CD/NORM)
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

        # 1) Load metadata (ptbxl_database + scp_statements)
        df, scp = load_metadata(base_dir)

        # 2) Official split by strat_fold
        if split == "test":
            df_split = df[df["strat_fold"] == 10].reset_index(drop=True)
        elif split == "val":
            df_split = df[df["strat_fold"] == 9].reset_index(drop=True)
        else:  # "train"
            df_split = df[df["strat_fold"] <= 8].reset_index(drop=True)

        # 3) Drop rows with missing age or sex (rare, but cleaner)
        num_before = len(df_split)
        mask = df_split["age"].notna() & df_split["sex"].notna()
        df_split = df_split.loc[mask].reset_index(drop=True)
        num_after = len(df_split)

        print(
            f"[PTBXLECGDemoDataset] split={split} | total={num_before} | "
            f"after_drop_missing_age_sex={num_after} | dropped={num_before - num_after}"
        )

        self.df = df_split

        # 4) Build label matrix for the chosen superclasses
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
        Build demo = [age_norm, sex_id, height_norm, weight_norm, pacemaker]

        - age >= 300 (anonymized 90+ years) -> clamp to 90
        - age_norm   = age / 100.0
        - height_norm= height_cm / 250.0  (missing -> 0)
        - weight_norm= weight_kg / 200.0  (missing -> 0)
        - sex_id: M=0, F=1, other/unknown=2
        - pacemaker: 0/1
        """
        # ----- age -----
        age = row.get("age", np.nan)
        try:
            age = float(age)
        except Exception:
            age = 0.0

        if age >= 300:       # PTB-XL: age>89 anonymized as 300
            age = 90.0       # clamp to 90
        elif age < 0:
            age = 0.0

        age_norm = age / 100.0   # roughly [0, 0.9]

        # ----- sex -----
        sex = row.get("sex", "UNKNOWN")
        if sex == "M":
            sex_id = 0.0
        elif sex == "F":
            sex_id = 1.0
        else:
            sex_id = 2.0

        # ----- height -----
        height = row.get("height", np.nan)
        try:
            height = float(height)
        except Exception:
            height = 0.0

        if not np.isfinite(height) or height <= 0:
            height = 0.0

        height_norm = height / 250.0   # assume < 250 cm

        # ----- weight -----
        weight = row.get("weight", np.nan)
        try:
            weight = float(weight)
        except Exception:
            weight = 0.0

        if not np.isfinite(weight) or weight <= 0:
            weight = 0.0

        weight_norm = weight / 200.0   # assume < 200 kg

        # ----- pacemaker -----
        pacemaker = row.get("pacemaker", 0)
        try:
            pacemaker_val = float(pacemaker)
        except Exception:
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
        rel_path = row["filename_hr"]              # e.g. "records500/00000/00001_hr"
        record_path = os.path.join(self.base_dir, rel_path)
        x_ecg = _load_ecg(record_path)             # [12, T]
        x_ecg = self._normalize_ecg(x_ecg)

        # Demographic vector
        x_demo = self._build_demo_vector(row)      # [5]

        # Labels
        y = self.y[idx]                            # [num_labels]

        x_ecg_t = torch.from_numpy(x_ecg).float()
        x_demo_t = torch.from_numpy(x_demo).float()
        y_t = torch.from_numpy(y).float()

        return x_ecg_t, x_demo_t, y_t
