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
    1D-CNN encoder for ECG.
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
        h = self.backbone(x)
        g = self.gap(h).squeeze(-1)
        z = self.proj(g)
        return z


class DemoEncoder(nn.Module):
    """
    Encoder for demographic features:
    [age_norm, sex_id, height_norm, weight_norm, pacemaker]
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


class ECGMultimodal(nn.Module):
    """
    Multimodal model with FiLM conditioning:
    ECG features are modulated by demographic features.
    """
    def __init__(
        self,
        in_leads: int = 12,
        feat_dim: int = 256,
        demo_dim: int = 5,
        num_labels: int = 5,
        demo_hidden_dim: int = 64,
        ecg_feat_dim: int = None,
        **kwargs,
    ):
        super().__init__()

        if ecg_feat_dim is not None:
            feat_dim = ecg_feat_dim

        self.ecg_backbone = ECGBackbone(in_leads=in_leads, feat_dim=feat_dim)
        self.demo_encoder = DemoEncoder(demo_dim=demo_dim, hidden_dim=demo_hidden_dim)

        self.film_gen = nn.Linear(demo_hidden_dim, 2 * feat_dim)
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x_ecg: torch.Tensor, x_demo: torch.Tensor) -> torch.Tensor:
        z_ecg = self.ecg_backbone(x_ecg)
        h_demo = self.demo_encoder(x_demo)

        film_params = self.film_gen(h_demo)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        gamma = 1.0 + torch.tanh(gamma)
        z_cond = gamma * z_ecg + beta

        logits = self.head(z_cond)
        return logits
