# src/datasets/ptbxl_af.py

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

from src.utils.label_maps import load_metadata, build_af_binary_labels
from src.datasets.ptbxl import _is_valid_ecg


def _load_ecg(record_path: str) -> np.ndarray:
    """
    Load a PTB-XL ECG record and return an array of shape [12, T].
    """
    sig, _ = wfdb.rdsamp(record_path)   # sig: [T, n_leads]
    sig = np.asarray(sig, dtype=np.float32)

    if sig.ndim != 2:
        raise RuntimeError(f"Invalid ECG shape at {record_path}: {sig.ndim}D")
    if sig.shape[1] != 12:
        raise RuntimeError(f"Expected 12 leads at {record_path}, got {sig.shape[1]}")

    return sig.T   # [12, T]


class PTBXLAFDataset(Dataset):
    """
    PTB-XL subset for binary AF detection.

    Each item returns:
        x_ecg: [12, T] float32 ECG signal
        y:     [1]    float32 label (1 = AF, 0 = non-AF)
    """

    def __init__(
        self,
        base_dir: str,
        split: str,
        normalize: str = "per_lead",
    ):
        super().__init__()
        self.base_dir = base_dir
        self.normalize = normalize

        # Load PTB-XL metadata
        df, scp = load_metadata(base_dir)

        # Official stratified patient-wise split
        if split == "test":
            df_split = df[df["strat_fold"] == 10].reset_index(drop=True)
        elif split == "val":
            df_split = df[df["strat_fold"] == 9].reset_index(drop=True)
        else:  # train
            df_split = df[df["strat_fold"] <= 8].reset_index(drop=True)

        n_before = len(df_split)

        # Remove records with invalid ECG files
        mask = df_split["filename_hr"].apply(
            lambda rel: _is_valid_ecg(base_dir, rel)
        )
        df_split = df_split.loc[mask].reset_index(drop=True)
        n_after = len(df_split)

        print(
            f"[PTBXLAFDataset] split={split} | "
            f"total={n_before} | valid_ecg={n_after} | dropped={n_before - n_after}"
        )

        self.df = df_split
        self.y = build_af_binary_labels(self.df, scp)  # [N, 1]

    def __len__(self) -> int:
        return len(self.df)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Per-lead standardization.
        """
        if self.normalize == "per_lead":
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            return (x - mean) / std
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        record_rel = row["filename_hr"]
        record_path = os.path.join(self.base_dir, record_rel)

        x = _load_ecg(record_path)
        x = self._normalize(x)

        y = self.y[idx]   # [1]

        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
