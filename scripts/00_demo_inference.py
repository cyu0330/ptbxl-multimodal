# scripts/00_demo_inference.py
#
# Minimal runnable demo for your teachers:
#   - loads a small ECG sample from data/demo/demo_ecg_0.npy
#   - loads pretrained baseline model from outputs/ecg_baseline/ckpts/ecg_baseline_best.pth
#   - runs forward pass + Grad-CAM for one class (default: MI)
#   - saves a red heatmap figure into outputs/demo/

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.models.ecg_cnn import ECGCNN

CLASSES = ["MI", "STTC", "HYP", "CD", "NORM"]


# ---------- Grad-CAM for ECGCNN ----------
class GradCAM1D_ECG:
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
        # full_backward_hook 避免 future warning
        self.hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def generate_cam(self, x, class_idx, signal_length):
        """
        x: [1, 12, T]
        class_idx: index in CLASSES (0..4)
        signal_length: T
        """
        self.model.zero_grad()
        logits = self.model(x)          # [1, num_labels]
        score = logits[:, class_idx].sum()
        score.backward()

        acts = self.activations        # [1, C, T']
        grads = self.gradients         # [1, C, T']

        weights = grads.mean(dim=-1, keepdim=True)   # [1, C, 1]
        cam = (weights * acts).sum(dim=1)            # [1, T']
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=signal_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)  # [1, T]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        return cam[0].cpu()


# ---------- Pretty plot (红色背景 + 黑色 ECG) ----------
def plot_ecg_with_cam(ecg, cam, lead_idx, title, save_path):
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()

    sig = ecg[lead_idx]    # [T]
    T = sig.shape[-1]
    t = np.arange(T)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-9)

    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(111)

    cam_2d = np.expand_dims(cam, axis=0)

    ax.imshow(
        cam_2d,
        aspect="auto",
        cmap="Reds",          # 红色系
        alpha=0.7,
        extent=[0, T, sig.min(), sig.max()],
        origin="lower",
        interpolation="bilinear",
    )
    ax.plot(t, sig, color="black", linewidth=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel(f"ECG (lead {lead_idx})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[SAVE] Demo Grad-CAM figure saved to: {save_path}")


# ---------- Load baseline model ----------
def load_baseline_model(ckpt_path, device):
    model = ECGCNN(in_leads=12, feat_dim=256, num_labels=len(CLASSES))
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] Loaded baseline model. missing:", missing)
    print("[INFO] unexpected:", unexpected)
    model.to(device)
    model.eval()
    return model


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # 1) Load demo ECG
    ecg_np = np.load(args.demo_path)   # [12, T]
    print("[INFO] Loaded demo ECG:", ecg_np.shape)

    ecg_t = torch.from_numpy(ecg_np).float().unsqueeze(0).to(device)  # [1, 12, T]
    T = ecg_np.shape[-1]

    # 2) Load baseline model
    model = load_baseline_model(args.ckpt, device)

    # 3) Forward + probs (顺便打印一下预测概率，老师好看)
    with torch.no_grad():
        logits = model(ecg_t)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    print("[INFO] Predicted probabilities:")
    for i, p in enumerate(probs):
        print(f"  {CLASSES[i]}: {p:.3f}")

    # 4) Grad-CAM on chosen class
    class_idx = args.class_idx
    class_name = CLASSES[class_idx]
    print(f"[INFO] Running Grad-CAM for class: {class_name} (index {class_idx})")

    # target_layer: last ConvBlock's Conv1d
    target_layer = model.backbone[-1].net[0]
    gradcam = GradCAM1D_ECG(model, target_layer)

    cam = gradcam.generate_cam(ecg_t, class_idx=class_idx, signal_length=T)

    # 5) Save figure
    os.makedirs("outputs/demo", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.demo_path))[0]
    fig_path = os.path.join(
        "outputs/demo",
        f"{base_name}_gradcam_{class_name}.png",
    )
    plot_ecg_with_cam(
        ecg=ecg_np,
        cam=cam,
        lead_idx=args.lead,
        title=f"Demo Grad-CAM | {base_name} | class {class_name}",
        save_path=fig_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo_path",
        type=str,
        default="data/demo/demo_ecg_0.npy",
        help="Path to demo ECG npy file [12, T].",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/ecg_baseline/ckpts/ecg_baseline_best.pth",
        help="Path to baseline ECG checkpoint.",
    )
    parser.add_argument(
        "--class_idx",
        type=int,
        default=0,
        help="Class index (0..4) in ['MI','STTC','HYP','CD','NORM'].",
    )
    parser.add_argument(
        "--lead",
        type=int,
        default=0,
        help="Which lead to plot (0..11).",
    )
    args = parser.parse_args()
    main(args)
