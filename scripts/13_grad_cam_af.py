# scripts/13_grad_cam_af.py
#
# Grad-CAM for AF binary model
# 输出：红色 ECG 热力图

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.models.ecg_cnn import ECGCNN
from src.datasets.ptbxl_af import PTBXLAFDataset
from src.utils.seed import set_seed

os.environ["TORCH_CUDA_ARCH_LIST"] = "native"


# --------------------------------------------------------
# 1. Grad-CAM for AF model
# --------------------------------------------------------
class GradCAM1D_AF:
    """
    target_layer 必须是最后一层卷积层，例如：
    model.backbone[-1].net[0]
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate_cam(self, x, signal_length):
        """
        x: [1, 12, T]
        signal_length: T
        """
        self.model.zero_grad()

        logits = self.model(x)      # [1, 1]
        score = logits[:, 0].sum()  # AF 概率的 logit
        score.backward()

        acts = self.activations     # [1, C, T']
        grads = self.gradients      # [1, C, T']

        # GAP → 通道权重
        weights = grads.mean(dim=-1, keepdim=True)   # [1, C, 1]

        cam = (weights * acts).sum(dim=1)            # [1, T']
        cam = F.relu(cam)

        # 插值回 T（原始长度）
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=signal_length,
            mode="linear",
            align_corners=False
        ).squeeze(1)

        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)

        return cam[0].cpu()


# --------------------------------------------------------
# 2. 画图：ECG + 红色 GradCAM
# --------------------------------------------------------
def plot_ecg_cam(ecg, cam, lead_idx, title, save_path):
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()

    sig = ecg[lead_idx]
    T = sig.shape[-1]
    t = np.arange(T)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-9)

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)

    cam_2d = np.expand_dims(cam, axis=0)

    ax.imshow(
        cam_2d,
        aspect="auto",
        cmap="Reds",
        alpha=0.7,
        extent=[0, T, sig.min(), sig.max()],
        origin="lower",
        interpolation="bilinear",
    )

    ax.plot(t, sig, color="black", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"ECG Lead {lead_idx}")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[SAVE] AF Grad-CAM saved to: {save_path}")


# --------------------------------------------------------
# 3. 加载 AF 模型
# --------------------------------------------------------
def load_af_model(ckpt_path, device):
    model = ECGCNN(in_leads=12, feat_dim=256, num_labels=1)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------
# 4. 主流程
# --------------------------------------------------------
def main(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # 数据集
    test_ds = PTBXLAFDataset(
        base_dir=args.base_dir,
        split="test",
        normalize="per_lead",
    )
    print("[INFO] AF test size:", len(test_ds))

    # 模型
    model = load_af_model(args.ckpt, device)

    # Grad-CAM target layer（最后一层卷积）
    target_layer = model.backbone[-1].net[0]

    gradcam = GradCAM1D_AF(model, target_layer)

    # 取样本
    x, y = test_ds[args.index]  # [12, T], [1]
    T = x.shape[-1]

    x_t = x.unsqueeze(0).to(device)

    print(f"[INFO] Running AF Grad-CAM on sample {args.index} (y={y.item()})")

    cam = gradcam.generate_cam(x_t, signal_length=T)

    # 保存 CAM 数值
    os.makedirs("outputs/gradcam_af", exist_ok=True)

    npy_path = os.path.join(
        "outputs/gradcam_af",
        f"sample_{args.index}_AF_cam.npy",
    )
    np.save(npy_path, cam.numpy())
    print("[SAVE] CAM saved to:", npy_path)

    # 保存图像
    fig_path = os.path.join(
        "outputs/gradcam_af",
        f"sample_{args.index}_AF_plot.png"
    )
    plot_ecg_cam(
        ecg=x,
        cam=cam,
        lead_idx=args.lead,
        title=f"AF Grad-CAM | sample {args.index} | AF label = {y.item()}",
        save_path=fig_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="C:/Users/Administrator/Desktop/ptb-xl/1.0.3")
    parser.add_argument("--ckpt", type=str, default="outputs/af_binary/ckpts/af_binary_best.pth")
    parser.add_argument("--index", type=int, default=10)
    parser.add_argument("--lead", type=int, default=0)
    args = parser.parse_args()

    main(args)
