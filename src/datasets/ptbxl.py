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
    Load ECG signal from a PTB-XL record.

    Args:
        record_path: full path without extension, e.g.
                     ".../records500/00000/00001_hr"

    Returns:
        signal: np.ndarray of shape [n_leads, T]
    """
    try:
        sig, meta = wfdb.rdsamp(record_path)
    except Exception as e:
        # Raise a clear error so we know exactly which record failed
        raise RuntimeError(f"Failed to read ECG record {record_path}: {e}")

    # sig is expected to be [T, n_leads]
    sig = np.asarray(sig)
    if sig.ndim != 2:
        raise RuntimeError(
            f"Unexpected ECG ndim for {record_path}: "
            f"{sig.ndim}, expected 2 (T, n_leads)"
        )

    T, n_leads = sig.shape

    # For PTB-XL high-resolution records we expect 12 leads
    if n_leads != 12:
        raise RuntimeError(
            f"Unexpected number of leads for {record_path}: "
            f"{n_leads}, expected 12"
        )

    # Convert to [n_leads, T]
    return sig.T.astype(np.float32)


def _is_valid_ecg(base_dir: str, rel_path: str) -> bool:
    """
    Check whether this PTB-XL record is valid locally:
      1. Both .hea and .dat files exist.
      2. wfdb can fully read the record without errors.
      3. The resulting signal has a reasonable shape (T, 12).

    This performs a full read, so it is more expensive but
    guarantees that invalid records (corrupted .dat, reshape errors, etc.)
    are filtered out before training.
    """
    rec_path = os.path.join(base_dir, rel_path)
    hea_path = rec_path + ".hea"
    dat_path = rec_path + ".dat"

    # Check file existence
    if not (os.path.exists(hea_path) and os.path.exists(dat_path)):
        print(
            f"[PTBXLDataset] missing files for {rel_path} "
            f"(hea_exists={os.path.exists(hea_path)}, "
            f"dat_exists={os.path.exists(dat_path)})"
        )
        return False

    try:
        sig, meta = wfdb.rdsamp(rec_path)  # full read

        sig = np.asarray(sig)
        if sig.ndim != 2:
            print(
                f"[PTBXLDataset] skip invalid record {rel_path}: "
                f"sig.ndim={sig.ndim} (expected 2)"
            )
            return False

        T, n_leads = sig.shape
        if n_leads != 12:
            print(
                f"[PTBXLDataset] skip invalid record {rel_path}: "
                f"n_leads={n_leads} (expected 12)"
            )
            return False

        # Optional extra check: total size must be divisible by 12
        total_points = T * n_leads
        if total_points % n_leads != 0:
            print(
                f"[PTBXLDataset] skip invalid record {rel_path}: "
                f"T={T}, n_leads={n_leads}, total_points={total_points}"
            )
            return False

    except Exception as e:
        # This will catch the reshape-related ValueErrors inside wfdb
        print(f"[PTBXLDataset] skip invalid record {rel_path}: {e}")
        return False

    return True


class PTBXLDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str,
        classes: List[str],
        normalize: str = "per_lead",
    ):
        """
        PTB-XL waveform dataset for baseline ECG classification.

        Args:
            base_dir: root directory of PTB-XL (contains ptbxl_database.csv,
                      records500/, etc.)
            split: "train", "val", or "test"
            classes: list of high-level diagnostic classes,
                     e.g. ["MI", "STTC", "HYP", "CD", "NORM"]
            normalize: "per_lead" or "none"
        """
        super().__init__()

        self.base_dir = base_dir
        self.normalize = normalize
        self.classes = classes

        # Load metadata (ptbxl_database + scp_statements)
        df, scp = load_metadata(base_dir)

        # Official-style split by strat_fold
        if split == "test":
            df_split = df[df["strat_fold"] == 10].reset_index(drop=True)
        elif split == "val":
            df_split = df[df["strat_fold"] == 9].reset_index(drop=True)
        else:  # "train"
            df_split = df[df["strat_fold"] <= 8].reset_index(drop=True)

        num_before = len(df_split)

        # Filter to only records that:
        #   - exist on disk
        #   - can be fully read by wfdb with valid shape
        mask = df_split["filename_hr"].apply(
            lambda rel: _is_valid_ecg(base_dir, rel)
        )
        df_split = df_split.loc[mask].reset_index(drop=True)
        num_after = len(df_split)
        num_dropped = num_before - num_after

        print(
            f"[PTBXLDataset] split={split} | total={num_before} | "
            f"valid_ecg={num_after} | dropped={num_dropped}"
        )

        self.df = df_split
        self.y = build_label_matrix(self.df, scp, classes)

    def __len__(self) -> int:
        return len(self.df)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "per_lead":
            # x: [n_leads, T]
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            return (x - mean) / std
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          x: [12, T] float32 ECG waveform
          y: [num_labels] float32 multi-hot label vector
        """
        row = self.df.iloc[idx]
        rel_path = row["filename_hr"]  # e.g. "records500/00000/00001_hr"
        record_path = os.path.join(self.base_dir, rel_path)

        # At this point, files should have passed _is_valid_ecg,
        # so _load_ecg is expected to succeed. If it still fails,
        # it will raise a clear RuntimeError.
        x = _load_ecg(record_path)     # [n_leads, T]
        x = self._normalize(x)
        y = self.y[idx]                # [num_labels]

        x_t = torch.from_numpy(x)             # [12, T]
        y_t = torch.from_numpy(y).float()     # [C]
        return x_t, y_t
