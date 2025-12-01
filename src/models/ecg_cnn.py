import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 15, p: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECGCNN(nn.Module):
    def __init__(self, in_leads: int = 12, feat_dim: int = 256, num_labels: int = 3):
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
        self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: [B, in_leads, T]
            return_features: if True, also return z (feature vector)

        Returns:
            if return_features == False:
                logits  # [B, num_labels]
            else:
                logits, z
        """
        h = self.backbone(x)            # [B, C, T']
        g = self.gap(h).squeeze(-1)     # [B, C]
        z = self.proj(g)                # [B, feat_dim]
        logits = self.head(z)           # [B, num_labels]

        if return_features:
            return logits, z
        return logits
