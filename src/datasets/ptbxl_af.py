# src/datasets/ptbxl_af.py

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import wfdb

from src.utils.label_maps import load_metadata, build_af_binary_labels
from src.datasets.ptbxl import _is_valid_ecg  # 复用你现有的 ECG 检查逻辑


def _load_ecg(record_path: str) -> np.ndarray:
    """
    Load ECG signal from a PTB-XL record and return [12, T].
    """
    sig, meta = wfdb.rdsamp(record_path)  # sig: [T, n_leads]
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

    return sig.T  # [12, T]


class PTBXLAFDataset(Dataset):
    """
    PTB-XL dataset for binary AF detection (AF vs non-AF).

    Returns:
        x_ecg: [12, T] float32 ECG waveform
        y:     [1]    float32, 1 for AF, 0 for non-AF
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

        # 1) load metadata
        df, scp = load_metadata(base_dir)

        # 2) official split by strat_fold
        if split == "test":
            df_split = df[df["strat_fold"] == 10].reset_index(drop=True)
        elif split == "val":
            df_split = df[df["strat_fold"] == 9].reset_index(drop=True)
        else:  # "train"
            df_split = df[df["strat_fold"] <= 8].reset_index(drop=True)

        num_before = len(df_split)

        # 3) filter by valid ECG
        mask_valid = df_split["filename_hr"].apply(
            lambda rel: _is_valid_ecg(base_dir, rel)
        )
        df_split = df_split.loc[mask_valid].reset_index(drop=True)
        num_after_valid = len(df_split)

        print(
            f"[PTBXLAFDataset] split={split} | "
            f"total={num_before} | valid_ecg={num_after_valid} | "
            f"dropped={num_before - num_after_valid}"
        )

        self.df = df_split

        # 4) build AF labels: [N, 1]
        self.y = build_af_binary_labels(self.df, scp)

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_ecg(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "per_lead":
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            return (x - mean) / std
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        rel_path = row["filename_hr"]  # e.g. "records500/00000/00001_hr"
        record_path = os.path.join(self.base_dir, rel_path)
        x_ecg = _load_ecg(record_path)         # [12, T]
        x_ecg = self._normalize_ecg(x_ecg)

        y = self.y[idx]                       # [1]

        x_ecg_t = torch.from_numpy(x_ecg).float()
        y_t = torch.from_numpy(y).float()

        return x_ecg_t, y_t
