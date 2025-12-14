# scripts/00_demo_inference.py
#
# Simple Grad-CAM demo on a single ECG sample using the baseline CNN.
# Supports .npy ([12,T]) and .npz (ecg, y, classes).

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.seed import set_seed
from src.models.ecg_cnn import ECGCNN

CLASSES = ["MI", "STTC", "HYP", "CD", "NORM"]


class GradCAM1D_ECG:
    """Grad-CAM on a selected Conv1d layer."""
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

    def generate_cam(self, x, class_idx, signal_length):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        acts = self.activations
        grads = self.gradients

        weights = grads.mean(dim=-1, keepdim=True)   # [1, C, 1]
        cam = (weights * acts).sum(dim=1)            # [1, T']
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=signal_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-9)
        return cam[0].cpu()


def find_last_conv1d(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            last = m
    if last is None:
        raise RuntimeError("No Conv1d layer found for Grad-CAM.")
    return last


def plot_ecg_with_cam(ecg, cam, lead_idx, title, save_path):
    if isinstance(ecg, torch.Tensor):
        ecg = ecg.cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.cpu().numpy()

    sig = ecg[lead_idx]
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
        cmap="Reds",
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


def load_demo_file(path: str):
    """
    Returns:
      ecg_np: [12,T]
      y_true: [5] or None
      classes: list[str] or default CLASSES
    """
    if path.endswith(".npy"):
        ecg_np = np.load(path)
        return ecg_np, None, CLASSES

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        ecg_np = z["ecg"]
        y_true = z["y"] if "y" in z.files else None
        classes = list(z["classes"]) if "classes" in z.files else CLASSES
        classes = [str(c) for c in classes]
        return ecg_np, y_true, classes

    raise ValueError(f"Unsupported demo file: {path}. Use .npy or .npz")


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    ecg_np, y_true, classes = load_demo_file(args.demo_path)
    print("[INFO] Loaded demo ECG:", ecg_np.shape)

    ecg_t = torch.from_numpy(ecg_np).float().unsqueeze(0).to(device)
    T = ecg_np.shape[-1]

    model = load_baseline_model(args.ckpt, device)

    with torch.no_grad():
        logits = model(ecg_t)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    print("[INFO] Predicted probabilities:")
    for i, p in enumerate(probs):
        name = CLASSES[i] if i < len(CLASSES) else f"cls_{i}"
        print(f"  {name}: {p:.3f}")

    if y_true is not None:
        y_true = np.asarray(y_true).astype(np.float32)
        print("[INFO] Ground-truth labels:")
        for i in range(min(len(CLASSES), len(y_true))):
            print(f"  {CLASSES[i]}: {int(y_true[i])}")

    class_idx = args.class_idx
    class_name = CLASSES[class_idx]
    print(f"[INFO] Running Grad-CAM for class: {class_name} (index {class_idx})")

    target_layer = find_last_conv1d(model)
    print("[INFO] Grad-CAM target layer:", target_layer)

    gradcam = GradCAM1D_ECG(model, target_layer)
    cam = gradcam.generate_cam(ecg_t, class_idx=class_idx, signal_length=T)

    os.makedirs("outputs/demo", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.demo_path))[0]
    fig_path = os.path.join("outputs/demo", f"{base_name}_gradcam_{class_name}.png")

    title = f"Demo Grad-CAM | {base_name} | class {class_name}"
    if y_true is not None:
        title += f" | GT={int(y_true[class_idx])}"

    plot_ecg_with_cam(
        ecg=ecg_np,
        cam=cam,
        lead_idx=args.lead,
        title=title,
        save_path=fig_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_path", type=str, default="data/demo/single/single_sample_00.npz",
                        help="Path to demo ECG file (.npy or .npz).")
    parser.add_argument("--ckpt", type=str, default="outputs/ecg_baseline/ckpts/ecg_baseline_best.pth",
                        help="Path to baseline ECG checkpoint.")
    parser.add_argument("--class_idx", type=int, default=0, help="Class index (0..4).")
    parser.add_argument("--lead", type=int, default=0, help="Lead index to plot (0..11).")
    args = parser.parse_args()
    main(args)
