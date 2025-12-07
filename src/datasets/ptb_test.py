# src/datasets/ptb_test.py

import os
from torch.utils.data import DataLoader

from src.datasets.ptbxl import PTBXLDataset

from src.datasets.ptbxl_ecg_demo import PTBXLECGDemoDataset
from src.datasets.ptbxl_af import PTBXLAFDataset


def make_baseline_test_loader(config):
    """
    Test loader for 03 ECG baseline on strat_fold=10.
    """
    # 这里我们约定：val fold 仍然来自 config["data"]["split"]（比如 strat_fold_9）
    # 而 test fold = 10，在 PTBXLDataSet 内部处理，或者在 config 里约定好
    ds = PTBXLDataset(split="test", config=config)
    loader = DataLoader(
        ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )
    return loader


def make_demo_test_loader(config):
    """
    Test loader for 04 ECG+Demo multimodal model on strat_fold=10.
    """
    ds = PTBXLECGDemoDataset(split="test", config=config)
    loader = DataLoader(
        ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )
    return loader


def make_af_test_loader(config):
    """
    Test loader for 05 AF binary model on strat_fold=10.
    """
    ds = PTBXLAFDataset(split="test", config=config)
    loader = DataLoader(
        ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
    )
    return loader
