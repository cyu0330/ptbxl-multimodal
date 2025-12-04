# src/models/ecg_demo.py

import torch
import torch.nn as nn

from src.models.ecg_cnn import ECGCNN


class DemoEncoder(nn.Module):
    """
    Simple MLP to embed demographic vector:
    [age_norm, sex_id, height_norm, weight_norm, pacemaker]
    """
    def __init__(self, in_dim: int = 5, hidden_dim: int = 32, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 5]
        return: [B, out_dim]
        """
        return self.net(x)


class ECGDemoModel(nn.Module):
    """
    Multimodal model: ECG + demographics (age, sex, height, weight, pacemaker).
    """
    def __init__(
        self,
        num_labels: int,
        ecg_feat_dim: int = 256,
        demo_feat_dim: int = 64,
        in_leads: int = 12,
    ):
        super().__init__()

        # ECG encoder (reuse your CNN)
        # We will use (logits, z) where z is the ECG feature.
        self.ecg_encoder = ECGCNN(
            in_leads=in_leads,
            feat_dim=ecg_feat_dim,
            num_labels=num_labels,   # ECG head logits can be used as aux if needed
        )

        # Demographic encoder
        self.demo_encoder = DemoEncoder(
            in_dim=5,
            hidden_dim=32,
            out_dim=demo_feat_dim,
        )

        fusion_dim = ecg_feat_dim + demo_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def forward(self, ecg: torch.Tensor, demo: torch.Tensor):
        """
        ecg:  [B, 12, T]
        demo: [B, 5]

        Returns:
            logits: [B, num_labels]
        """
        # ECG encoder returns (logits, features) if return_features=True
        ecg_logits, z_ecg = self.ecg_encoder(ecg, return_features=True)  # [B, ecg_feat_dim]

        # Demographic embedding
        z_demo = self.demo_encoder(demo)                                 # [B, demo_feat_dim]

        # Fusion
        z_fuse = torch.cat([z_ecg, z_demo], dim=-1)                      # [B, fusion_dim]

        logits = self.classifier(z_fuse)                                 # [B, num_labels]
        return logits
