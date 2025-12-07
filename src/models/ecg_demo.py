# src/models/ecg_demo.py

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 15, p: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGBackbone(nn.Module):
    """
    Simple 1D-CNN backbone for ECG.
    Input:  [B, in_leads, T]
    Output: [B, feat_dim]
    """
    def __init__(self, in_leads: int = 12, feat_dim: int = 256):
        super().__init__()
        chs = [32, 64, 128, 256]
        c = in_leads
        blocks = []
        for n in chs:
            blocks.append(ConvBlock(c, n))
            c = n
        self.backbone = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(chs[-1], feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_leads, T]
        return: [B, feat_dim]
        """
        h = self.backbone(x)          # [B, C, T']
        g = self.gap(h).squeeze(-1)   # [B, C]
        z = self.proj(g)              # [B, feat_dim]
        return z


class DemoEncoder(nn.Module):
    """
    Encode demographic features: [age_norm, sex_id, height_norm, weight_norm, pacemaker]
    Input:  [B, demo_dim]
    Output: [B, hidden_dim]
    """
    def __init__(self, demo_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(demo_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_demo: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_demo)


class ECGDemo(nn.Module):
    """
    ECG + Demo with FiLM-style conditioning:

    1) ECG backbone -> z_ecg: [B, feat_dim]
    2) Demo encoder -> h_demo: [B, hidden_dim]
    3) From h_demo produce gamma, beta: [B, feat_dim] each
    4) z_cond = gamma * z_ecg + beta
    5) Classifier head on z_cond.

    This lets the demographic information "modulate" the ECG features.
    """

    def __init__(
        self,
        in_leads: int = 12,
        feat_dim: int = 256,
        demo_dim: int = 5,
        num_labels: int = 5,
        demo_hidden_dim: int = 64,
        # 兼容旧脚本里使用的 ecg_feat_dim 参数名：
        ecg_feat_dim: int = None,
        # 再保险，防止脚本里还有别的多余 kwargs：
        **kwargs,
    ):
        super().__init__()

        # 如果外面传了 ecg_feat_dim，就覆盖 feat_dim
        if ecg_feat_dim is not None:
            feat_dim = ecg_feat_dim

        self.feat_dim = feat_dim
        self.demo_dim = demo_dim
        self.num_labels = num_labels
        self.demo_hidden_dim = demo_hidden_dim

        self.ecg_backbone = ECGBackbone(in_leads=in_leads, feat_dim=feat_dim)
        self.demo_encoder = DemoEncoder(demo_dim=demo_dim, hidden_dim=demo_hidden_dim)

        # demo -> 2 * feat_dim (for gamma and beta)
        self.film_gen = nn.Linear(demo_hidden_dim, 2 * feat_dim)

        # classifier head on modulated ECG features
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x_ecg: torch.Tensor, x_demo: torch.Tensor) -> torch.Tensor:
        """
        x_ecg: [B, in_leads, T]
        x_demo: [B, demo_dim]
        return: logits [B, num_labels]
        """
        # 1) ECG features
        z_ecg = self.ecg_backbone(x_ecg)       # [B, feat_dim]

        # 2) Demo features
        h_demo = self.demo_encoder(x_demo)     # [B, demo_hidden_dim]

        # 3) Generate gamma and beta
        film_params = self.film_gen(h_demo)    # [B, 2 * feat_dim]
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1)  # each [B, feat_dim]

        # Stability: constrain gamma to be around 1, e.g. via tanh
        gamma = 1.0 + torch.tanh(gamma)        # gamma in (0, 2) approximately

        # 4) FiLM conditioning
        z_cond = gamma * z_ecg + beta          # [B, feat_dim]

        # 5) Classifier
        logits = self.head(z_cond)             # [B, num_labels]
        return logits
