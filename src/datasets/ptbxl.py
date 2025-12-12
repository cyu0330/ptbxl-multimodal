# src/datasets/ptbxl.py

import os
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

from src.utils.label_maps import load_metadata, build_label_matrix


def _load_ecg(record_path: str) -> np.ndarray:
    """
    Read a PTB-XL ECG record and return the signal as [12, T].

    Args:
        record_path: full record path without extension.

    Returns:
        np.ndarray: ECG signal of shape [n_leads, T].
    """
    try:
        sig, meta = wfdb.rdsamp(record_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read record {record_path}: {e}")

    sig = np.asarray(sig, dtype=np.float32)

    if sig.ndim != 2:
        raise RuntimeError(
            f"Unexpected shape for {record_path}: ndim={sig.ndim}, expected 2."
        )

    T, n_leads = sig.shape
    if n_leads != 12:
        raise RuntimeError(
            f"Invalid lead count for {record_path}: {n_leads}, expected 12."
        )

    return sig.T  # [12, T]


def _is_valid_ecg(base_dir: str, rel_path: str) -> bool:
    """
    Check whether the required .hea and .dat files exist and whether
    the signal can be read correctly by wfdb.
    """
    rec_path = os.path.join(base_dir, rel_path)
    hea = rec_path + ".hea"
    dat = rec_path + ".dat"

    if not (os.path.exists(hea) and os.path.exists(dat)):
        return False

    try:
        sig, meta = wfdb.rdsamp(rec_path)
        sig = np.asarray(sig)

        if sig.ndim != 2:
            return False

        T, n_leads = sig.shape
        if n_leads != 12:
            return False

    except Exception:
        return False

    return True


class PTBXLDataset(Dataset):
    """
    PTB-XL 12-lead ECG dataset for multi-label baseline classification.
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
        self.classes = classes
        self.normalize = normalize

        # Load PTB-XL metadata
        df, scp = load_metadata(base_dir)

        # Train/val/test split based on stratified folds
        if split == "test":
            df_split = df[df["strat_fold"] == 10].reset_index(drop=True)
        elif split == "val":
            df_split = df[df["strat_fold"] == 9].reset_index(drop=True)
        else:  # train
            df_split = df[df["strat_fold"] <= 8].reset_index(drop=True)

        num_total = len(df_split)

        # Filter invalid or unreadable ECG files
        valid_mask = df_split["filename_hr"].apply(
            lambda rel: _is_valid_ecg(base_dir, rel)
        )
        df_split = df_split.loc[valid_mask].reset_index(drop=True)
        num_valid = len(df_split)

        print(
            f"[PTBXLDataset] split={split} | total={num_total} | "
            f"valid={num_valid} | dropped={num_total - num_valid}"
        )

        self.df = df_split
        self.y = build_label_matrix(self.df, scp, classes)

    def __len__(self) -> int:
        return len(self.df)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "per_lead":
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            return (x - mean) / std
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: [12, T] ECG signal
            y: [num_classes] multi-hot label vector
        """
        row = self.df.iloc[idx]
        rec_path = os.path.join(self.base_dir, row["filename_hr"])

        x = _load_ecg(rec_path)
        x = self._normalize(x)
        y = self.y[idx]

        return torch.from_numpy(x), torch.from_numpy(y).float()
