# src/datasets/ptb_test.py

import os
from torch.utils.data import DataLoader

from src.datasets.ptbxl import PTBXLDataset
from src.datasets.ptbxl_ecg_multimodal import PTBXLECGMultimodalDataset
from src.datasets.ptbxl_af import PTBXLAFDataset


def make_baseline_test_loader(config):
    """
    Create test loader for the ECG baseline model.
    Assumes test split = 'test' (strat_fold = 10 in PTB-XL).
    """
    data_cfg = config["data"]
    train_cfg = config["train"]

    ds = PTBXLDataset(
        base_dir=data_cfg["base_dir"],
        split="test",
        classes=data_cfg["labels"],
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    loader = DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=False,
    )
    return loader


def make_multimodal_test_loader(config):
    """
    Create test loader for the ECG + demographics multimodal model.
    """
    data_cfg = config["data"]
    train_cfg = config["train"]

    ds = PTBXLECGMultimodalDataset(
        base_dir=data_cfg["base_dir"],
        split="test",
        classes=data_cfg["labels"],
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    loader = DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=False,
    )
    return loader


def make_af_test_loader(config):
    """
    Create test loader for the AF binary classifier.
    """
    data_cfg = config["data"]
    train_cfg = config["train"]

    ds = PTBXLAFDataset(
        base_dir=data_cfg["base_dir"],
        split="test",
        normalize=data_cfg.get("normalize", "per_lead"),
    )

    loader = DataLoader(
        ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=False,
    )
    return loader
