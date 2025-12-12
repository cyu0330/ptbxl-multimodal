import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic 1D convolution block:
    Conv1d → BatchNorm → ReLU → MaxPool.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 15, p: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGCNN(nn.Module):
    """
    CNN encoder for 12-lead ECG classification.

    Args:
        in_leads: number of input ECG channels (default 12)
        feat_dim: dimension of the latent feature vector
        num_labels: number of output classes
    """
    def __init__(self, in_leads: int = 12, feat_dim: int = 256, num_labels: int = 3):
        super().__init__()

        channels = [32, 64, 128, 256]
        c = in_leads
        blocks = []

        # Convolution backbone
        for n in channels:
            blocks.append(ConvBlock(c, n))
            c = n
        self.backbone = nn.Sequential(*blocks)

        # Global aggregation + projection
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels[-1], feat_dim)

        # Classification head
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: [B, in_leads, T]
            return_features: optionally return latent vector z

        Returns:
            logits or (logits, z)
        """
        h = self.backbone(x)          # [B, C, T']
        g = self.gap(h).squeeze(-1)   # [B, C]
        z = self.proj(g)              # [B, feat_dim]
        logits = self.head(z)         # [B, num_labels]

        if return_features:
            return logits, z
        return logits
